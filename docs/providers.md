# Two ways in

The same runner reaches Jev through either backend. Pick with `--provider`; everything else —
prompts, sampling, metrics, decision rule — is identical.

```bash
acento run --provider typesafe    # direct, needs TYPESAFE_API_KEY
acento run --provider gateway     # Vercel AI Gateway, needs AI_GATEWAY_API_KEY
```

| | `gateway` (default) | `typesafe` (direct) |
|---|---|---|
| Base URL | `https://ai-gateway.vercel.sh/typesafe` | `https://api.typesafe.ai` |
| Key | `AI_GATEWAY_API_KEY` | `TYPESAFE_API_KEY` |
| Model sent | `typesafe-ai/jev` | `jev-1.13.0` |
| **Model reported back** | `typesafe-ai/jev` — an alias | **`jev-1.13.0` — the version that answered** |
| Versioned ids accepted | **No** (HTTP 404 `model_not_found`) | Yes |
| Rate limit | Undocumented; no headers | 1200 rpm / 250k tokens per second, documented |
| Per-call cost reported | Yes, `marketCost` | No; computed from the list price |
| `generationId` per call | Yes | No |
| Access | No waitlist | Requires an approved account |

## Check a provider before trusting it

```bash
acento smoke --provider typesafe
```

Three cheap calls (well under a cent) that print the facts the audit depends on: the response
shape, whether `confidence` is present per primitive, token counts, cost, and — the line that
matters — `VERSION PINNED: YES/NO`.

## Why the direct path is better for an audit

**It pins the model.** This is the whole difference. An audit compares three arms measured over
about half an hour; if the model changes partway through, the comparison is between two models
rather than two languages, and nothing in the results would reveal it.

Through the Gateway the response's `model` field is the alias `typesafe-ai/jev`, and sending a
versioned id is rejected outright. The best available mitigation is indirect: record a timestamp
and `generationId` per row, and snapshot the provider's advertised `release_date` before and
after the run. That *detects* a coarse change; it does not pin anything.

The direct API resolves whatever name you send and echoes the versioned id that actually
answered. Every row then carries its own proof of which model produced it. `analyse.py` reads
those values back and classifies the run:

- **pinned** — one versioned id across every row. The run is anchored.
- **unpinned** — one alias across every row. A warning is emitted and the limitation stands.
- **mixed** — more than one id. The run spans a model change or mixes providers; it is not
  comparable and gets a loud warning telling you to discard it.

This is checked from the data, not from configuration, so it cannot be satisfied by claiming it.

## Why the Gateway path stays supported

It needs no waitlist, so anyone can reproduce this work without an approved TypeSafe account.
It is also the only path that reports a real per-call cost, which makes `--max-usd` exact rather
than estimated. Keeping both means the audit can be run by people who cannot get direct access.

## What is the same

Both speak the identical native wire format, so there is no AI SDK and no TypeScript anywhere in
this repo. Request and response shapes match; the direct path simply omits the
`provider_metadata` block that the Gateway adds. `parse_response` handles both, falling back to
the published price of USD 0.042 per million input tokens when no cost metadata is present.
