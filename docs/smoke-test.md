# Phase 0 — smoke test

Run on **2026-09-20** against the Vercel AI Gateway passthrough, before any harness code existed.
Everything recorded here is a finding that shaped the design, and several are worked around in
`providers.py`.

Total spend: roughly 1,550 input tokens, about USD 0.00007.

## What was confirmed

| Claim | Result |
|---|---|
| `POST https://ai-gateway.vercel.sh/typesafe/v1/systemone` with a Bearer key | HTTP 200 |
| Latency | 0.6–0.8 s per call |
| Price of USD 0.042 per million input tokens | **Exact.** 403 input tokens billed `marketCost` 0.000016926 |
| Choice returns `choice`, `probabilities`, `confidence` | Yes |
| Noul returns `noul` as a float in [0, 1] | Yes |
| Identical requests are not cached | Confirmed: a byte-identical repeat returns a fresh `generationId` and `x-vercel-cache: MISS` |

## Findings that changed the design

**1. Noul answers carry no `confidence`.** Neither in the answer object nor in
`provider_metadata.typesafe.confidence`, which comes back as `{}`. Choice answers carry it in
both places. `Answer.confidence` is therefore nullable, and PAWS-X — the only Noul dataset —
derives all of its calibration signal from `p_max = max(p, 1 − p)`.

**2. The Gateway rejects versioned model ids.** All three spellings return HTTP 404
`model_not_found`:

```
typesafe-ai/jev-1.13.0
jev-1.13.0
typesafe-ai/jev@1.13.0
```

The response's `model` field echoes the alias `typesafe-ai/jev`. **There is no way to pin a
version through the Gateway.** Mitigation: every row records `ts` and `generation_id`, and each
run snapshots `release_date` before and after (see finding 3). This is declared as a limitation
in the README rather than papered over.

**3. `GET /typesafe/v1/models` is the only version signal available.** It returns exactly:

```json
{"name": "jev", "release_date": "2026-09-15"}
```

Not an id, but it moves when the model does. A run that starts and ends on different
`release_date` values spans a model change; `analyse.py` raises a loud warning and the run is
discarded.

**4. `cost` is `"0"`; the real number is `marketCost`.** The Gateway bills with system
credentials and reports `provider_metadata.gateway.cost` as the string `"0"`. Reading that field
would have made `--max-usd` useless — the cap would never trigger. `_extract_cost` prefers
`marketCost`.

**5. `probabilities` key order varies between calls.** Two byte-identical requests returned the
same probabilities in a different key order. Any hash of a response must sort keys;
`canonical_json` does.

**6. No rate-limit headers.** The Gateway returns only `x-vercel-id` and `x-vercel-cache`. There
is no budget to read back, so the pacer is open-loop and 429s can only be handled reactively.

**7. Noul accepts `criteria`.** Not documented clearly, so it was tested: adding `true`/`false`
descriptions to a Noul question returns HTTP 200 and raises input tokens from 337 to 369, so the
criteria are read and billed. PAWS-X uses them, which gives arm C something to translate beyond
the instructions string.

**8. An empty state is accepted, and empty *fields* are the right overhead probe.** Three
variants of the same XNLI question:

| State | Input tokens |
|---|---|
| `{}` | 413 |
| `{"premise": "", "hypothesis": ""}` | 429 |
| `{"premise": "a", "hypothesis": "b"}` | 431 |

The 16-token gap is the field names themselves. Since field names are English and identical in
every arm, they belong to the overhead, not to the text — so the state-only token ratio is
computed against the empty-*fields* baseline. Using `{}` would have biased every ratio in the
same direction.

## The direct path

Verified against the TypeSafe documentation on 2026-09-20, **not yet against a live key**. The
contract is identical where it matters and differs in three ways that the code handles:

| | Gateway | Direct |
|---|---|---|
| Endpoint | `…/typesafe/v1/systemone` | `https://api.typesafe.ai/v1/systemone` |
| `model` echoed in the response | `typesafe-ai/jev` (alias) | `jev-1.13.0` (the version that answered) |
| `provider_metadata` | present | **absent** — so no `marketCost` and no `generationId` |
| Documented rate limit | none published | 1200 rpm / 250k tokens per second |
| Retryable statuses | 429, 5xx | 429, 5xx, **529 Overloaded** |

The docs list `jev-1.13.0` as the versioned id, with `jev-latest` and `jev-preview` as aliases
that currently point to it. Every response example in the API reference shows `jev-1.13.0` even
where the request used `jev-latest`, which is what makes the direct path capable of pinning a
run: the API tells you which version answered rather than repeating the name you asked for.

Context limits, restated precisely from the docs: 64k tokens per request covering state plus all
questions, and 32k for the state plus **the single longest question** — slightly tighter than
"32k for the state" alone.

**Still to verify against a live key:** that `jev-1.13.0` is accepted as sent, that the response
echoes it, whether any rate-limit headers are returned, and whether the account's quota allows
the roughly 19,200 calls the full audit needs.

## Inherited check, first data point

`confidence ≈ (k · p_max − 1) / (k − 1)`: with k = 3 and p_max = 0.99 the formula gives 0.985,
and the API reported **0.98**. Consistent with truncation, not with rounding. `metrics.py`
reports residuals under both interpretations across every cell.
