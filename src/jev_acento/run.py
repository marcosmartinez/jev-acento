"""The runner: turns frozen prompts and sampled items into rows in ``runs/``.

Design constraints that shaped this module:

* **It must refuse to run unfrozen.** :func:`execute` calls :func:`~jev_acento.freeze.assert_frozen`
  before opening a single output file. Skipping the freeze is possible only via ``--dry-run``,
  which cannot write to ``runs/`` at all.
* **It must be resumable.** ~19k calls over ~35 minutes will be interrupted. Every cell file is
  append-only JSONL, and a restart reads back which item ids are already present and skips them.
  Nothing is ever rewritten, so a crash mid-write costs at most one row.
* **It must not be able to overspend.** The cap is checked inside the client *before* each call,
  not tallied afterwards.
* **One item, one question, one call.** Batching would let one question's wording contaminate
  another's answer, and would make per-item latency and token accounting meaningless.

The two passes are the stability gate. The API does not cache identical requests (verified in
the Phase 0 smoke test: a byte-identical repeat returns a fresh ``generationId``), so pass 1 is
a genuine re-measurement, and the difference between passes bounds how much of any arm-to-arm
difference is just the model moving on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import freeze
from .data import DEFAULT_CACHE, SPECS, Item, load_aligned, resolve_options, stratified_sample
from .providers import (
    PROVIDERS,
    Answer,
    CostCapExceeded,
    JevClient,
    JevResponse,
    Provider,
    canonical_json,
    fetch_model_release_date,
    parse_answer,
)
from .questions import ARMS, QuestionSpec, load_arm_prompt, validate_prompt_set

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------
# Fake API, for --dry-run and the test suite
# --------------------------------------------------------------------------------------------


class FakeJev:
    """A local stand-in for the API. Never touches the network.

    **The numbers it produces are fabricated and mean nothing.** Its only job is to exercise
    every code path -- parsing, resumption, accounting, metrics, figures -- without spending
    money or waiting on a rate limit.

    It is deterministic: the same state and questions always yield the same answer, derived from
    a hash. When ``gold`` is supplied it steers roughly ``target_accuracy`` of items to the
    correct label, which keeps dry-run outputs realistic enough that the analysis and figures are
    actually tested rather than fed degenerate input.
    """

    def __init__(self, target_accuracy: float = 0.82, seed: int = 0) -> None:
        self.target_accuracy = target_accuracy
        self.seed = seed
        self.calls = 0
        self.spent_usd = 0.0

    @staticmethod
    def _unit(*parts: str) -> float:
        """A stable pseudo-random float in [0, 1) from arbitrary strings."""
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    async def ask(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        *,
        gold: str | None = None,
    ) -> JevResponse:
        self.calls += 1
        key = canonical_json({"s": state, "q": questions, "seed": self.seed})
        answers: dict[str, Answer] = {}

        for qid, q in questions.items():
            keys = list(q.get("criteria") or {}) or ["true", "false"]
            roll = self._unit(key, qid, "pick")
            if gold is not None and gold in keys and roll < self.target_accuracy:
                winner = gold
            else:
                winner = keys[int(self._unit(key, qid, "alt") * len(keys)) % len(keys)]

            # Confidence concentrated on the winner, rounded to 2 decimals like the real API.
            p_win = round(0.50 + 0.49 * self._unit(key, qid, "conf"), 2)
            rest = [k for k in keys if k != winner]
            share = round((1.0 - p_win) / len(rest), 2) if rest else 0.0

            if q.get("type") == "noul":
                p_true = p_win if winner == "true" else round(1.0 - p_win, 2)
                payload: dict[str, Any] = {"type": "noul", "noul": p_true}
            else:
                probs = {k: share for k in rest}
                probs[winner] = p_win
                payload = {
                    "type": q.get("type", "choice"),
                    "choice": winner,
                    "probabilities": probs,
                    "confidence": round((len(keys) * p_win - 1) / max(len(keys) - 1, 1), 2),
                }
            answers[qid] = parse_answer(qid, payload)

        input_tokens = max(1, len(canonical_json({"state": state, "questions": questions})) // 4)
        return JevResponse(
            model="fake/jev",
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=20,
            cost_usd=0.0,
            generation_id=f"fake_{self._unit(key, 'gen'):.12f}",
            latency_ms=1.0,
        )

    async def aclose(self) -> None:  # interface parity with JevClient
        return None


# --------------------------------------------------------------------------------------------
# Configuration and row schema
# --------------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    run_id: str
    provider: str = "gateway"
    datasets: tuple[str, ...] = tuple(SPECS)
    arms: tuple[str, ...] = tuple(ARMS)
    passes: int = 2
    seed: int = 20260920
    rpm: int | None = None
    """Pacer target. ``None`` means use the provider's own documented limit."""
    concurrency: int = 8
    max_usd: float = 5.0
    dry_run: bool = False
    n_override: int | None = None
    """Cap n for every dataset. Used by tests and smoke runs; never in a real run."""

    def n_for(self, dataset: str) -> int:
        target = SPECS[dataset].n_target
        return min(target, self.n_override) if self.n_override else target


ROW_FIELDS = (
    "dataset", "item_id", "arm", "pass", "gold", "pred", "p_max", "confidence", "probs",
    "state_hash", "input_tokens", "latency_ms", "model", "generation_id",
    "choice_disagrees_with_argmax", "ts",
)


def cell_path(runs_dir: Path, run_id: str, dataset: str, arm: str, pass_k: int) -> Path:
    return runs_dir / f"{run_id}-{dataset}-{arm}-pass{pass_k}.jsonl"


def completed_item_ids(path: Path) -> set[str]:
    """Item ids already recorded in a cell file, tolerating a truncated final line."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # a partial row from an interrupted write; it will simply be redone
    return done


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------


@dataclass
class CellPlan:
    dataset: str
    arm: str
    pass_k: int
    items: list[Item]
    spec: QuestionSpec
    state_lang: str


@dataclass
class RunSummary:
    run_id: str
    calls: int = 0
    skipped: int = 0
    spent_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    model_release_before: dict[str, Any] = field(default_factory=dict)
    model_release_after: dict[str, Any] = field(default_factory=dict)
    alignment: list[dict[str, Any]] = field(default_factory=list)
    question_overhead: dict[str, int] = field(default_factory=dict)
    """``"<dataset>/<arm>" -> input tokens for the question with an empty-field state."""
    started: str = ""
    finished: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


async def _run_one(
    client: Any,
    plan: CellPlan,
    item: Item,
    out_lock: asyncio.Lock,
    out_file: Any,
    summary: RunSummary,
) -> None:
    state = item.states[plan.state_lang]
    questions = plan.spec.payload()

    try:
        if isinstance(client, FakeJev):
            resp = await client.ask(state, questions, gold=item.gold)
        else:
            resp = await client.ask(state, questions)
    except CostCapExceeded:
        raise
    except Exception as exc:
        summary.errors.append(f"{plan.dataset}/{plan.arm}/p{plan.pass_k}/{item.item_id}: {exc}")
        return

    answer = resp.answers.get(plan.spec.question_id)
    if answer is None:
        summary.errors.append(
            f"{plan.dataset}/{item.item_id}: response has no answer for "
            f"{plan.spec.question_id!r} (got {list(resp.answers)})"
        )
        return

    row = {
        "dataset": plan.dataset,
        "item_id": item.item_id,
        "arm": plan.arm,
        "pass": plan.pass_k,
        "gold": item.gold,
        "pred": answer.pred,
        "p_max": answer.p_max,
        "confidence": answer.confidence,
        "probs": answer.probs,
        "state_hash": item.state_hash(plan.state_lang),
        "input_tokens": resp.input_tokens,
        "latency_ms": round(resp.latency_ms, 2),
        "model": resp.model,
        "generation_id": resp.generation_id,
        "choice_disagrees_with_argmax": answer.choice_disagrees_with_argmax,
        "ts": _now(),
    }

    async with out_lock:
        out_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        out_file.flush()
        summary.calls += 1


async def _run_cell(client: Any, plan: CellPlan, runs_dir: Path, summary: RunSummary) -> None:
    path = cell_path(runs_dir, summary.run_id, plan.dataset, plan.arm, plan.pass_k)
    already = completed_item_ids(path)
    todo = [it for it in plan.items if it.item_id not in already]
    summary.skipped += len(plan.items) - len(todo)

    if not todo:
        return

    lock = asyncio.Lock()
    with path.open("a", encoding="utf-8") as fh:
        await asyncio.gather(
            *(_run_one(client, plan, item, lock, fh, summary) for item in todo)
        )


def build_plans(
    config: RunConfig,
    prompts_dir: Path,
    cache_dir: Path,
    summary: RunSummary,
) -> list[CellPlan]:
    """Load, align, validate, sample -- once per dataset, reused across arms and passes."""
    plans: list[CellPlan] = []
    for dataset in config.datasets:
        spec = SPECS[dataset]
        items, report = load_aligned(spec, cache_dir)
        summary.alignment.append(asdict(report))

        options = resolve_options(spec, items)
        validate_prompt_set(dataset, options, prompts_dir)

        sample = stratified_sample(items, config.n_for(dataset), config.seed)
        for arm in config.arms:
            state_lang, _ = ARMS[arm]
            qspec = load_arm_prompt(dataset, arm, prompts_dir)
            for pass_k in range(config.passes):
                plans.append(CellPlan(dataset, arm, pass_k, sample, qspec, state_lang))
    return plans


async def execute(
    config: RunConfig,
    *,
    root: Path = REPO_ROOT,
    prompts_dir: Path | None = None,
    runs_dir: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE,
    progress: bool = True,
) -> RunSummary:
    """Run every (dataset, arm, pass) cell and return what happened."""
    prompts_dir = prompts_dir or root / "prompts"
    runs_dir = runs_dir or root / "runs"

    if not config.dry_run:
        # Refuse before touching runs/. A dry run is exempt because it writes nowhere real.
        freeze.assert_frozen(root)

    runs_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(run_id=config.run_id, started=_now())

    plans = build_plans(config, prompts_dir, cache_dir, summary)

    provider: Provider = PROVIDERS[config.provider]
    if config.dry_run:
        client: Any = FakeJev()
    else:
        summary.model_release_before = await fetch_model_release_date(provider)
        client = JevClient(
            provider,
            rpm=config.rpm,
            concurrency=config.concurrency,
            max_usd=config.max_usd,
        )

    started = time.monotonic()
    try:
        seen_overhead: set[tuple[str, str]] = set()
        for i, plan in enumerate(plans, 1):
            key = (plan.dataset, plan.arm)
            if key not in seen_overhead and plan.items:
                seen_overhead.add(key)
                try:
                    summary.question_overhead[f"{plan.dataset}/{plan.arm}"] = (
                        await probe_question_overhead(
                            client, plan.spec, plan.items[0].states[plan.state_lang]
                        )
                    )
                except Exception as exc:
                    summary.errors.append(f"overhead probe {key}: {exc}")
            if progress:
                print(
                    f"[{i}/{len(plans)}] {plan.dataset} arm {plan.arm} pass {plan.pass_k} "
                    f"({len(plan.items)} items)",
                    flush=True,
                )
            await _run_cell(client, plan, runs_dir, summary)
    except CostCapExceeded as exc:
        summary.errors.append(f"HALTED: {exc}")
    finally:
        summary.spent_usd = getattr(client, "spent_usd", 0.0)
        if not config.dry_run:
            try:
                summary.model_release_after = await fetch_model_release_date(provider)
            except Exception as exc:
                summary.errors.append(f"could not re-read model release date: {exc}")
        await client.aclose()

    summary.finished = _now()
    if progress:
        elapsed = time.monotonic() - started
        print(
            f"\n{summary.calls} calls, {summary.skipped} skipped, "
            f"{summary.spent_usd:.4f} USD, {elapsed:.0f}s, {len(summary.errors)} errors"
        )

    _write_manifest(runs_dir, config, summary)
    return summary


def _write_manifest(runs_dir: Path, config: RunConfig, summary: RunSummary) -> None:
    """Record what was run, under what configuration, against which advertised model.

    ``model_release_before`` and ``model_release_after`` are the only version evidence available
    through the Gateway. If they differ, the run spans a model change and is not internally
    comparable -- :mod:`jev_acento.analyse` refuses such a run.
    """
    manifest = runs_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    existing[summary.run_id] = {
        "config": asdict(config),
        "summary": {k: v for k, v in asdict(summary).items() if k != "alignment"},
        "alignment": summary.alignment,
    }
    manifest.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def probe_question_overhead(
    client: Any, spec: QuestionSpec, state_keys: Iterable[str]
) -> int:
    """Input tokens consumed by everything *except* the state text.

    Sends the real frozen question with a state that has the real field names mapped to empty
    strings. Subtracting this from an item's ``input_tokens`` leaves the tokens attributable to
    the item's text alone, which is what the ES/EN "state-only" ratio is supposed to measure.

    Using empty *fields* rather than an empty object is deliberate: the field names are
    structural and identical in every arm, so they belong in the overhead, not in the text. On
    XNLI the difference is 16 tokens (413 for ``{}`` versus 429 for empty fields) -- small, but
    it would bias every ratio in the same direction.

    The overhead is arm-specific, because arm C's Spanish instructions do not tokenise to the
    same length as arm A and B's English ones.
    """
    empty_state = {key: "" for key in state_keys}
    resp = await client.ask(empty_state, spec.payload())
    return resp.input_tokens


def iter_rows(runs_dir: Path, run_id: str) -> Iterable[dict[str, Any]]:
    """Every row of a run, across all cell files, in a stable file order."""
    for path in sorted(runs_dir.glob(f"{run_id}-*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
