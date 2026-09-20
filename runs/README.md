# `runs/`

One append-only JSONL file per `(run_id, dataset, arm, pass)` cell, plus `manifest.json`.

Each row is a *derived* record — never source text:

```
dataset, item_id, arm, pass, gold, pred, p_max, confidence, probs, state_hash,
input_tokens, latency_ms, model, generation_id, choice_disagrees_with_argmax, ts
```

These files are the evidence behind `results.md`, so they are committed for real runs and
released under MIT along with the code. `make reproduce` regenerates every published number and
figure from them alone, with no network access.

Files from `--dry-run` are named `dryrun*` and are gitignored: their numbers come from the local
fake API and are fabricated.
