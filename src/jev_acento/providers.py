"""Provider seam for the Jev System One API.

The same runner must work against two backends without changing a line of calling code:

    gateway   Vercel AI Gateway passthrough (default; no waitlist)
    typesafe  TypeSafe AI directly (requires an approved account)

Both speak the *native* TypeSafe wire format, so there is no AI SDK and no TypeScript here.

Everything in this module that looks like a workaround is one, and each is traceable to a
finding from the Phase 0 smoke test (2026-09-20, recorded in ``docs/smoke-test.md``):

* The Gateway rejects versioned model ids (``typesafe-ai/jev-1.13.0`` -> HTTP 404
  ``model_not_found``) and echoes back the alias ``typesafe-ai/jev``. We therefore cannot pin a
  version. :func:`fetch_model_release_date` retrieves the provider's advertised ``release_date``
  so a run can at least *detect* a model change, and every row records ``generation_id``.
* ``provider_metadata.gateway.cost`` is the string ``"0"`` when the Gateway bills with system
  credentials. The real spend is ``marketCost``. :func:`_extract_cost` reads that instead.
* Noul answers carry **no** ``confidence`` field, and ``provider_metadata.typesafe.confidence``
  is an empty object for them. ``Answer.confidence`` is therefore ``None`` for Noul.
* The Gateway exposes no rate-limit headers, so the pacer is open-loop (see :class:`Pacer`) and
  429s can only be handled reactively.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

QuestionKind = Literal["choice", "score", "noul"]

# TypeSafe hard limits, from the provider docs (verified 2026-09-20).
MAX_REQUEST_TOKENS = 64_000
MAX_STATE_TOKENS = 32_000

# USD per million input tokens. Output tokens are free. Confirmed against ``marketCost`` in the
# Phase 0 smoke test: 403 input tokens billed exactly 0.000016926 USD.
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000


@dataclass(frozen=True)
class Provider:
    """Where to send requests, and under which model id."""

    name: str
    base_url: str
    api_key_env: str
    model: str

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    @property
    def models_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set. "
                f"Put it in .env (which is gitignored) or export it."
            )
        return key


PROVIDERS: dict[str, Provider] = {
    "gateway": Provider(
        name="gateway",
        base_url="https://ai-gateway.vercel.sh/typesafe",
        api_key_env="AI_GATEWAY_API_KEY",
        model="typesafe-ai/jev",
    ),
    "typesafe": Provider(
        name="typesafe",
        base_url="https://api.typesafe.ai",
        api_key_env="TYPESAFE_API_KEY",
        # Direct access accepts a versioned id, and we always use one rather than the alias so
        # the run is reproducible. Bump deliberately, never silently.
        model="jev-1.13.0",
    ),
}


@dataclass
class Answer:
    """One question's answer, normalised across primitives.

    ``pred`` is always derived from ``probs`` by argmax, never read from the provider's own
    ``choice`` field. The two can disagree because probabilities are rounded to 2 decimals and
    can tie; ``choice_disagrees_with_argmax`` counts those cases so the run can report them.
    """

    kind: QuestionKind
    pred: str
    probs: dict[str, float]
    p_max: float
    confidence: float | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    choice_disagrees_with_argmax: bool = False


@dataclass
class JevResponse:
    """A whole API response: one or more answers plus accounting."""

    model: str
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    generation_id: str | None
    latency_ms: float
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _argmax_prob(probs: dict[str, float]) -> tuple[str, float]:
    """Argmax over a probability map, with ties broken by key for determinism.

    Probabilities arrive rounded to 2 decimals, so exact ties are common and *expected*. Sorting
    the key as a tiebreaker makes the choice reproducible across runs and across the varying key
    order the Gateway returns.
    """
    best_key = min(
        probs,
        key=lambda k: (-probs[k], k),  # highest probability, then lowest key
    )
    return best_key, probs[best_key]


def parse_answer(qid: str, payload: dict[str, Any]) -> Answer:
    """Normalise one raw answer object into an :class:`Answer`.

    Choice and Score expose an explicit probability map. Noul returns a single float, which we
    expand into an explicit two-outcome distribution so every primitive shares one code path
    downstream (metrics, thresholds, reliability bins).
    """
    kind = payload.get("type")

    if kind in ("choice", "score"):
        probs_raw = payload.get("probabilities")
        if not isinstance(probs_raw, dict) or not probs_raw:
            raise ValueError(f"question {qid!r}: {kind} answer has no probabilities: {payload!r}")
        probs = {str(k): float(v) for k, v in probs_raw.items()}
        pred, p_max = _argmax_prob(probs)
        stated = payload.get("choice") if kind == "choice" else payload.get("score")
        stated_str = None if stated is None else str(stated)
        return Answer(
            kind=kind,
            pred=pred,
            probs=probs,
            p_max=p_max,
            confidence=_as_float_or_none(payload.get("confidence")),
            raw=payload,
            choice_disagrees_with_argmax=stated_str is not None and stated_str != pred,
        )

    if kind == "noul":
        p_true = float(payload["noul"])
        probs = {"true": p_true, "false": round(1.0 - p_true, 10)}
        # Brief section 9: a tie at exactly 0.50 resolves to `false`, and is counted.
        pred = "true" if p_true > 0.5 else "false"
        return Answer(
            kind="noul",
            pred=pred,
            probs=probs,
            p_max=max(p_true, 1.0 - p_true),
            confidence=_as_float_or_none(payload.get("confidence")),  # absent in practice
            raw=payload,
            choice_disagrees_with_argmax=False,
        )

    raise ValueError(f"question {qid!r}: unknown answer type {kind!r}")


def _as_float_or_none(v: Any) -> float | None:
    return None if v is None else float(v)


def _extract_cost(body: dict[str, Any], input_tokens: int) -> float:
    """Actual USD spend for one call.

    Prefers the provider's own accounting, falling back to the published price. ``marketCost`` is
    checked before ``cost`` because the Gateway reports ``cost: "0"`` when it bills with system
    credentials, which would silently defeat the ``--max-usd`` cap.
    """
    gw = body.get("provider_metadata", {}).get("gateway", {})
    for key in ("marketCost", "cost"):
        raw = gw.get(key)
        if raw not in (None, "", "0"):
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return input_tokens * USD_PER_INPUT_TOKEN


def _extract_generation_id(body: dict[str, Any]) -> str | None:
    return body.get("provider_metadata", {}).get("gateway", {}).get("generationId")


def parse_response(body: dict[str, Any], latency_ms: float) -> JevResponse:
    """Normalise a full API response body."""
    answers_raw = body.get("answers")
    if not isinstance(answers_raw, dict):
        raise ValueError(f"response has no answers object: {body!r}")
    usage = body.get("usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    return JevResponse(
        model=str(body.get("model", "")),
        answers={qid: parse_answer(qid, a) for qid, a in answers_raw.items()},
        input_tokens=input_tokens,
        output_tokens=int(usage.get("output_tokens", 0)),
        cost_usd=_extract_cost(body, input_tokens),
        generation_id=_extract_generation_id(body),
        latency_ms=latency_ms,
        raw=body,
    )


class Pacer:
    """Open-loop rate limiter.

    The Gateway publishes no rate-limit headers, so there is no budget to read back and this can
    only ever be a best-effort throttle spaced evenly in time. Actual overload is handled
    reactively by the 429 retry path in :meth:`JevClient.ask`.
    """

    def __init__(self, rpm: int) -> None:
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
            delay = slot - now
        if delay > 0:
            await asyncio.sleep(delay)


class CostCapExceeded(RuntimeError):
    """Raised when cumulative spend would exceed ``--max-usd``. Hard stop, never a warning."""


class JevClient:
    """Async client with pacing, retries and a hard spend cap.

    Retries 429 and 5xx with exponential backoff plus jitter, honouring ``retry-after`` when the
    server sends one. 4xx other than 429 are *not* retried: a malformed request will stay
    malformed, and retrying it only burns budget.
    """

    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        provider: Provider,
        *,
        rpm: int = 600,
        concurrency: int = 8,
        max_usd: float = 5.0,
        max_retries: int = 6,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider
        self.pacer = Pacer(rpm)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_usd = max_usd
        self.max_retries = max_retries
        self.spent_usd = 0.0
        self.calls = 0
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {provider.api_key()}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    async def __aenter__(self) -> JevClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> JevResponse:
        """Send one request and return the normalised response.

        Raises :class:`CostCapExceeded` *before* spending when the cap is already reached.
        """
        if self.spent_usd >= self.max_usd:
            raise CostCapExceeded(
                f"spend cap reached: {self.spent_usd:.4f} USD >= {self.max_usd:.4f} USD"
            )

        payload = {"model": self.provider.model, "state": state, "questions": questions}
        last_error: Exception | None = None

        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                await self.pacer.wait()
                started = time.perf_counter()
                try:
                    resp = await self._client.post(self.provider.endpoint, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    await self._backoff(attempt, None)
                    continue

                latency_ms = (time.perf_counter() - started) * 1000.0

                if resp.status_code == 200:
                    parsed = parse_response(resp.json(), latency_ms)
                    self.spent_usd += parsed.cost_usd
                    self.calls += 1
                    return parsed

                if resp.status_code in self.RETRY_STATUSES and attempt < self.max_retries:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                    await self._backoff(attempt, resp.headers.get("retry-after"))
                    continue

                # Non-retryable, or out of retries: surface the body, it explains the failure.
                raise RuntimeError(
                    f"{self.provider.name} returned HTTP {resp.status_code}: {resp.text[:500]}"
                )

        raise RuntimeError(f"exhausted retries against {self.provider.name}") from last_error

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        """Sleep before a retry: honour ``retry-after``, else exponential with full jitter."""
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 60.0))
                return
            except (TypeError, ValueError):
                pass
        await asyncio.sleep(random.uniform(0, min(2.0**attempt, 30.0)))


async def fetch_model_release_date(provider: Provider) -> dict[str, Any]:
    """Snapshot what the provider currently advertises about the model.

    This is the closest thing to a version pin available through the Gateway, which returns only
    ``{"name": "jev", "release_date": "..."}``. A run records this before and after; if it moves
    mid-run, the run is not internally comparable and must be discarded.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            provider.models_endpoint,
            headers={"Authorization": f"Bearer {provider.api_key()}"},
        )
        resp.raise_for_status()
        body = resp.json()

    models = body.get("models", body.get("data", []))
    for m in models:
        name = str(m.get("name", m.get("id", "")))
        if "jev" in name.lower():
            return {k: v for k, v in m.items() if k != "description"}
    return {"name": "unknown", "release_date": None}


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing.

    ``sort_keys`` is load-bearing, not cosmetic: the Gateway returns ``probabilities`` keys in a
    different order on every call, so an unsorted dump would hash the same answer differently
    each time.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
