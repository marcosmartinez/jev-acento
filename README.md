# jev-acento

**Does Jev understand your accent?**

An independent, reproducible audit of [Jev](https://typesafe.ai) — TypeSafe AI's "System One"
evaluation model — on **Spanish**, plus a CLI that lets anyone run the same comparison on their
own labelled data.

**Run `20260921-es-v1`** — 19,200 calls, 3,200 paired items, model `jev-1.13.0` (version-pinned),
USD 0.58, 0 errors. Every number below comes from [`results.json`](results.json) via
`make reproduce`; none is typed by hand.

## Findings

**1. Spanish costs accuracy on every dataset.** Holding the instructions in English and swapping
only the `state` from English to Spanish (B − A), Jev is measurably worse on all four:

| Dataset | A (EN state) | B (ES state) | Δ accuracy | Verdict |
|---|---|---|---|---|
| XNLI | 0.850 | 0.786 | −6.4 pp `[−8.6, −4.3]` | **measurably worse** |
| PAWS-X | 0.834 | 0.772 | −6.2 pp `[−8.5, −3.6]` | **measurably worse** |
| MASSIVE | 0.845 | 0.808 | −3.7 pp `[−5.8, −1.5]` | **measurably worse** |
| Belebele | 0.982 | 0.952 | −3.0 pp `[−4.5, −1.7]` | **measurably worse** |

**2. It also costs calibration, on the two hardest tasks.** ECE roughly doubles on XNLI
(0.057 → 0.101) and PAWS-X (0.033 → 0.078) — *less calibrated* under the pre-registered rule.
On MASSIVE and Belebele the change is not detectable. Calibration matters more than accuracy
here: the operational consequence is that automating at `p_max ≥ 0.9` covers **72.2% of XNLI in
English but only 63.4% in Spanish**, and the items you do automate are *less* accurate
(0.938 → 0.904), not more.

**3. Writing the instructions in Spanish does not help.** This is the question nobody had
measured, and the answer is a clean null on three of four datasets (C − B):

| Dataset | Δ accuracy | Verdict |
|---|---|---|
| XNLI | −0.2 pp `[−1.1, +0.7]` | no detectable difference |
| MASSIVE | −0.7 pp `[−1.8, +0.7]` | no detectable difference |
| Belebele | +0.5 pp `[+0.0, +1.2]` | no detectable difference |
| PAWS-X | +1.6 pp `[+0.7, +2.6]` | ambiguous — real but below the 3 pp threshold |

Calibration shows no detectable difference on all four. **Practical advice: keep your
`instructions` and `criteria` in English.** It is never worse, it is what the model is
documented to be best at, and on MASSIVE the Spanish wording costs 6.2% more input tokens for
nothing.

**4. Spanish text costs 17–38% more input tokens** than the same content in English
(state-only, with the fixed question overhead subtracted). Far less than the ~3× the Russian
audit found for Cyrillic.

### Compared with the Russian audit

| | Russian (prior work) | Spanish (this repo) |
|---|---|---|
| XNLI accuracy | 88.3% → 77.3% (−11.0 pp) | 85.0% → 78.6% (−6.4 pp) |
| XNLI ECE | 0.032 → 0.096 | 0.057 → 0.101 |
| Token ratio | ~3× | ~1.23× |

Spanish degrades less than Russian, which is what you would expect from a Latin-script language
closer to the training distribution. The direction is the same; the magnitude is roughly half.

### Was the model stable enough to trust this?

Yes, and it was checked before anything above was interpreted. Pass 0 and pass 1 sent
byte-identical requests: flip rates 0.2–2.1%, Cohen's κ ≥ 0.95, and **accuracy jitter between
passes of 0.000–0.005 against deltas of 0.030–0.064**. The model's own noise is an order of
magnitude smaller than the effects being reported.

Full tables, reliability bins, selective-accuracy curves and per-arm coverage:
[`results.md`](results.md). Deviations from the pre-registration: [`DEVIATIONS.md`](DEVIATIONS.md)
(there were none).

## The questions

Jev does not generate text. It takes a `state` and typed questions (Choice, Score, Noul) and
returns answers **with probabilities**. Its entire value proposition is that those probabilities
are *calibrated*, so software can automate above a threshold and route the rest to a human.

TypeSafe states that English is the primary language and that other languages are supported
"but not as well", without publishing numbers. This repo measures three things:

1. **Cost of the language.** With English instructions, does Jev lose accuracy and calibration
   when the `state` is in Spanish, on the same human-labelled items?
2. **Language of the instructions.** With a Spanish `state`, is it better to write `instructions`
   and `criteria` in English, or in Spanish?
3. **Operational cost.** ES/EN token ratio, latency p50/p95, and automatable coverage at typical
   thresholds (0.5 / 0.9).

**Question 2 is the original contribution.** It is the practical decision every Spanish-speaking
team has to make, and nobody has measured it.

## Design

Paired design — the same items in every arm. For the target language `es`:

| Arm | `state` | `instructions` + `criteria` |
|-----|---------|------------------------------|
| A   | English | English                      |
| B   | Spanish | English                      |
| C   | Spanish | Spanish                      |

Option keys (`entailment`, `alarm_set`, `option_1`…) stay in English and identical across all
three arms. Arm C translates only the `instructions` and the `criteria` descriptions. This
isolates the effect of instruction language from the effect of the label space.

Pre-registered primary comparisons: **B − A** (comparable to the
[Russian audit](https://github.com/AHTOOOXA/jev-cyrillic-audit)) and **C − B** (novel).

### Figures

![Calibration by arm](figures/reliability_paired_es.png)

*Confidence against observed accuracy, all three arms on the same items. Annotations are
bin counts — the sparse low-confidence bins hold single digits and should not be read as
trends. On XNLI and PAWS-X the Spanish arms sit visibly below the diagonal: overconfident.*

![Selective accuracy by arm](figures/selective_accuracy_es.png)

*Accuracy against coverage, most-confident-first. This is the operational view: how much
can you automate, and how accurate is what you automate? Dotted lines mark the coverage
reached at `p_max ≥ 0.5` and `≥ 0.9`.*

## Datasets

Public, parallel, human-labelled only. No source text is redistributed — this repo commits only
item ids, gold labels and state hashes; the text is recovered by re-joining against the
upstream dataset.

| Dataset | HF id | Primitive | n | Notes |
|---------|-------|-----------|---|-------|
| XNLI | `facebook/xnli` | Choice, 3 options | 1000 | Human translation |
| PAWS-X | `google-research-datasets/paws-x` | Noul | 1000 | Only Noul coverage in v0.1 |
| MASSIVE intent | `mteb/amazon_massive_intent` | Choice, ~60 intents | 600 | es-ES, not Rioplatense |
| Belebele | `facebook/belebele` | Choice, 4 options | 600 | Capped by dataset size (900 rows) |

## Why Spanish only

The brief originally scoped Spanish **and** Portuguese. v0.1 ships Spanish only, for two reasons:

1. **Two of the four datasets have no Portuguese at all.** XNLI covers 15 languages and PAWS-X
   covers 7; neither includes `pt`. Portuguese existed only in MASSIVE (as `pt-PT`, not `pt-BR`)
   and Belebele — half the evidence, and the weaker half.
2. **No native reviewer was available.** This repo does not freeze a prompt that a native speaker
   has not reviewed. Running Portuguese unreviewed would have produced a number that cannot be
   defended.

Portuguese is **future work**, and it is cheap to add: arm A (English state) is shared across
target languages, so a v0.2 needs only arms B and C on MASSIVE and Belebele.

## Two ways in

```bash
acento run --provider typesafe    # direct; pins the model version
acento run --provider gateway     # Vercel AI Gateway; no waitlist
```

Both speak the identical native wire format. The direct path is preferred for the audit because
it is the only one that reports which model version answered. The Gateway path is kept so the
work is reproducible without an approved TypeSafe account. Full comparison in
[`docs/providers.md`](docs/providers.md).

## Install

```bash
uv tool install jev-acento     # or: pipx install jev-acento
```

## Use it on your own data

The audit is one use of the harness. The other is comparing **your** question wordings on
**your** labelled data:

```bash
acento compare --data mis_tickets.jsonl \
               --questions q.en.json q.es.json \
               --out reporte/
```

- `--data`: JSONL with `id`, `state` and `gold` per question. **No parallel English data needed.**
- `--questions`: two or more versions of the same question set — same keys, different language or
  wording. Each version runs over the same items.
- Output: `reporte/results.md`, `results.json` and figures, with paired deltas of accuracy, ECE
  against a noise floor, coverage at threshold, and tokens.

Nothing leaves your machine except the calls to Jev.

A synthetic example dataset ships in `examples/`, and works against the built-in fake API with no
key at all:

```bash
acento compare --data examples/tickets.jsonl \
               --questions examples/q.en.json examples/q.es.json \
               --out /tmp/demo --dry-run
```

## Reproduce the audit

The pre-registration is **not frozen yet** — `PREREG.md` is a draft awaiting review of the
Spanish wordings. Until `make freeze` is run and its manifests are committed, `make check-prereg`
correctly reports that nothing is registered, and `make run` refuses to write to `runs/`.

```bash
acento smoke --provider typesafe   # verify a provider before trusting it (3 calls)
make check-prereg   # verify the frozen prompts and pre-registration still hash correctly
make test           # metrics on synthetic data, runner against the fake API, freeze checks
make dry-run        # full pipeline, no network, no spend
make run            # the real thing (~19k calls, ~USD 0.50, ~35 min)
make reproduce      # regenerate results.* and figures/ deterministically from runs/
make verify         # the full pre-publication gate: freeze, tests, secret scan, reproducibility
```

## Limitations

- **One wording per cell.** Each cell uses a single wording, which is a sample of size 1 from
  the space of possible wordings. The C − B null means *these two wordings* performed the same,
  not that Spanish and English instructions are interchangeable in general. This is the single
  largest threat to finding 3, and it is stated here rather than buried.
- **MASSIVE is es-ES**, Belebele and XNLI are translationese, and none of it is Rioplatense or
  any other regional variety. A team writing for Argentine or Mexican users should treat these
  numbers as an upper bound on how well their own text will do.
- **Score is not covered.** No suitable parallel ordinal dataset exists, so only Choice and Noul
  were measured. Nothing here says anything about Score.
- **Four academic benchmarks are not your workload.** XNLI, PAWS-X, MASSIVE and Belebele are
  clean, short and balanced; your tickets are not. Use `acento compare` on your own labelled
  data rather than assuming these deltas transfer.
- **Version pinning depends on the provider.** **This run was pinned:** all 19,200 rows report
  `jev-1.13.0`, verified from the rows rather than assumed. Run through the **direct** TypeSafe API and
  every row records the versioned model that answered it (`jev-1.13.0`), anchoring the run.
  Run through the **Vercel AI Gateway** and it does not: the Gateway rejects versioned model
  ids (`typesafe-ai/jev-1.13.0` → 404) and echoes back the alias `typesafe-ai/jev`, so a silent
  model update mid-run can only be detected, not excluded. See
  [`docs/providers.md`](docs/providers.md).
- **A single run.** Stability was measured *within* this run (two passes, ~20 minutes apart).
  Nothing here bounds how much Jev's Spanish behaviour drifts between model versions.

## Licence

MIT, for both the code and the derived rows in `runs/`. See [`THIRD_PARTY.md`](THIRD_PARTY.md)
for dataset licences and credit to prior work.
