# `examples/` — a runnable demo of `acento compare`

A small synthetic dataset for trying the tool without a key and without your own data.

- `tickets.jsonl` — 30 invented Spanish support tickets, each with an `id`, a `state` object and
  a `gold` label per question. **Entirely made up**; no real customer data.
- `q.en.json` / `q.es.json` — the same two questions (`urgency`, a 3-way Choice, and
  `is_complaint`, a Noul) written once in English and once in Spanish. Identical question ids and
  identical option keys, exactly as the audit's arms require.

## Run it with no key at all

```bash
acento compare --data examples/tickets.jsonl \
               --questions examples/q.en.json examples/q.es.json \
               --out /tmp/demo --dry-run
```

`--dry-run` uses the built-in fake API. **The numbers it produces are fabricated** — the point is
to show you the shape of the report and to prove the pipeline runs, not to tell you anything
about Jev.

## Run it for real

```bash
export AI_GATEWAY_API_KEY=...
acento compare --data examples/tickets.jsonl \
               --questions examples/q.en.json examples/q.es.json \
               --out /tmp/demo
```

120 calls, well under a cent. At n = 30 every verdict will be wide and inconclusive; that is
correct behaviour, not a bug. Use a few hundred of your own labelled items before reading
anything into a delta.

## Using your own data

Replace `tickets.jsonl` with your own JSONL:

```json
{"id": "abc", "state": {"any": "json", "you": "like"}, "gold": {"urgency": "high"}}
```

`state` can be any JSON your questions reference with backticks. `gold` maps each question id to
its human label — and it must be *human*: labelling your evaluation set with an LLM and then
measuring an LLM against it tells you only that the two models agree.
