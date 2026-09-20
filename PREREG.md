# Pre-registration — jev-acento v0.1

**Status: DRAFT.** Not yet frozen. Pending review of the Spanish prompt wordings by Marcos
Martinez. Once frozen, this file is hashed into `PREREG.sha256` and the runner refuses to write
to `runs/` if either this file or any prompt changes.

- **Registered on:** *(pending — the date this file is committed together with its hash)*
- **Registered by:** Marcos Martinez
- **Model under test:** Jev (TypeSafe AI System One), via the Vercel AI Gateway passthrough
- **Random seed:** `20260920`, used for sampling, all bootstraps and all noise-floor simulations

---

## 1. Questions

**Q1 — Cost of the language.** With English instructions, does Jev lose accuracy and calibration
when the `state` is in Spanish rather than English, on the same human-labelled items?

**Q2 — Language of the instructions.** With a Spanish `state`, does writing `instructions` and
`criteria` in Spanish rather than English change accuracy or calibration?

**Q3 — Operational cost.** What are the ES/EN token ratios (per call and state-only), the
latency p50/p95, and the automatable coverage at thresholds 0.5 and 0.9?

Q2 is the novel contribution. Q1 exists to be comparable with the prior Russian audit.

## 2. Hypotheses

| # | Hypothesis | Direction |
|---|------------|-----------|
| H1 | Accuracy in arm B is lower than in arm A. | Directional (B < A) |
| H2 | ECE in arm B is higher than in arm A. | Directional (B > A) |
| H3 | Accuracy differs between arms C and B. | **Non-directional** |
| H4 | Spanish states consume more input tokens than English ones. | Directional (B > A) |

H3 is deliberately two-sided. There is a plausible mechanism in each direction — instructions in
the state's own language may help the model stay in one linguistic register, or may hurt if the
model's instruction-following is strongest in English — and we have no basis for choosing one in
advance. Predicting a direction we cannot justify would be a way of taking credit for a coin flip.

## 3. Arms

| Arm | `state` | `instructions` + `criteria` |
|-----|---------|------------------------------|
| A | English | English |
| B | Spanish | English |
| C | Spanish | Spanish |

**Invariants, enforced in code by `questions.validate_prompt_set`:**

- Option keys (`entailment`, `alarm_set`, `option_1`, …) are English and byte-identical in all
  three arms, in the same order.
- State field names (`premise`, `utterance`, `passage`, …) are English in all three arms. Only
  the *values* change language.
- Arm C translates the `instructions` string and the `criteria` descriptions. Nothing else.

Arms A and B share one prompt file, so the B − A contrast cannot be contaminated by a wording
difference.

**Primary comparisons:** **B − A** and **C − B**. A − C is not pre-registered and will not be
reported as a primary result.

## 4. Datasets and sample sizes

Public, parallel, human-labelled. Test split in every case.

| Dataset | HF id | Primitive | n | Strata |
|---------|-------|-----------|---|--------|
| XNLI | `facebook/xnli` | Choice (3) | 1000 | gold label |
| PAWS-X | `google-research-datasets/paws-x` | Noul | 1000 | gold label |
| MASSIVE intent | `mteb/amazon_massive_intent` | Choice (59) | 600 | gold intent |
| Belebele | `facebook/belebele` | Choice (4) | 600 | gold option |

**Why these n.** XNLI and PAWS-X are raised to 1000 because the C − B effect is expected to be
small, and at n = 600 a moderately discordant comparison produces a paired CI whose half-width
exceeds the 3 pp null threshold, which would return AMBIGUOUS — the one outcome that cannot be
interpreted. MASSIVE stays at 600 because its macro-F1 is limited by rare intents rather than by
overall n. Belebele stays at 600 because the dataset contains only 900 rows per language.

**Sampling.** Stratified by gold label, largest-remainder allocation, every non-empty stratum
guaranteed at least one item, seed `20260920`. The resulting item ids are frozen in
`data/items.parquet` together with gold labels and state hashes. No source text is committed.

**Alignment.** Languages are joined on a per-dataset key (row index for XNLI, which ships no id;
`id` for PAWS-X and MASSIVE; `link` + `question_number` for Belebele). Any item whose gold label
differs between languages is **excluded before sampling** and counted in the run manifest. The
verification run of 2026-09-20 found 0 mismatches in all four datasets — notably, the two crossed
XNLI rows reported by the Russian audit for `ru`/`en` do not appear in `es`/`en`.

## 5. Procedure

- One item per call. One question per call.
- Two passes per cell. **Pass 0 is the primary result**; pass 1 exists only to measure stability.
- The API does not cache identical requests (verified 2026-09-20: a byte-identical repeat returns
  a fresh `generationId`), so pass 1 is a genuine re-measurement.
- Concurrency 8, pacer 600 rpm, retries with backoff on 429/5xx honouring `retry-after`.
- Hard spend cap, default USD 5.

## 6. Metrics

- **Accuracy** and **macro-F1**, with percentile bootstrap CIs (1000 resamples). Macro-F1 is
  averaged over the full declared label space, so a never-predicted intent contributes 0.
- **ECE** over `p_max`, 10 equal-width bins, left-closed and right-open, top bin closed. For
  Noul, `p_max = max(p, 1 − p)`.
- **ECE noise floor**: 1000 simulations of `correct_i ~ Bernoulli(p_max_i)` using the arm's own
  confidence vector. Reported alongside every ECE as `ece`, `ece_floor` and their ratio.
- Reliability bins with counts, selective accuracy vs coverage, AURC, and coverage → accuracy at
  thresholds 0.5 and 0.9.
- **Paired deltas**: item indices are resampled once per replicate and *both* arms recomputed on
  that same resample.
- Stability: flip rate, Cohen's κ, mean and max |Δp_max| between passes.
- Tokens: mean input tokens per call, and a state-only figure obtained by subtracting a measured
  per-(dataset, arm) overhead — the token count of the same frozen question sent with a state
  whose fields are present but empty. Latency p50/p95.

## 7. Decision rule

Applied in this order, per comparison. Matched to the Russian audit so results are comparable.

**Gate 1 — stability.** If the accuracy jitter between pass 0 and pass 1 (taken as the larger of
the two arms' jitter) is greater than or equal to |Δ accuracy|, the verdict is **UNSTABLE** and
the comparison is not interpreted further.

**Gate 2 — accuracy.**
- CI excludes 0 **and** |Δ| ≥ 3 pp → *measurably better / worse*
- CI includes 0 **and** half-width ≤ 3 pp → *no detectable difference*
- otherwise → **AMBIGUOUS**

**Gate 3 — calibration.** Assessed only if `ECE / floor ≥ 1.5` in **both** arms; otherwise *not
measurable*.
- CI excludes 0 **and** |Δ ECE| ≥ 0.02 → *more / less calibrated*
- CI includes 0 → *no detectable difference*
- otherwise → **AMBIGUOUS**

**A verdict is never reached by observing that two per-arm confidence intervals overlap.**
Overlapping per-arm intervals routinely coexist with a significant paired difference. Only the
interval on the delta decides.

**AMBIGUOUS is a reportable result.** It will be published as such, not resolved by adding items,
changing the threshold, or picking a different metric after the fact.

## 8. Declared in advance

**One wording per cell.** Each cell uses a single wording, which is a sample of size 1 from the
space of possible wordings. A C − B difference is therefore a difference between *these two
wordings*, not between the Spanish and English languages in general. This is the single largest
threat to the validity of Q2 and it is stated in the README, not buried.

**Spanish register.** The Spanish prompts use neutral, impersonal phrasing (infinitives and
impersonal constructions) and avoid both `tú` and `vos`. This keeps the wording from being
specifically Peninsular or specifically Rioplatense, at the cost of sounding slightly formal.
Regional register is a Phase 3 question, not a v0.1 one.

**Spanish only.** Portuguese was in the original scope and is not in v0.1: two of the four
datasets have no Portuguese at all, and no native reviewer was available to approve a Portuguese
wording. Freezing an unreviewed translation would produce a number nobody could defend.

**Score primitive not covered.** No suitable parallel ordinal dataset exists. Only Choice and
Noul are measured.

**MASSIVE is es-ES.** Not Rioplatense, not Latin American.

**No version pin.** The Gateway rejects versioned model ids and reports only the alias. Each row
records its timestamp and `generationId`, and each run snapshots the provider's advertised
`release_date` before and after. If the two snapshots differ, the run spans a model change, is
not internally comparable, and will be discarded rather than reported.

## 9. Exploratory (not confirmatory)

Anything below may be reported but is explicitly **not** covered by the decision rule above, and
will be labelled exploratory wherever it appears:

- Per-label and per-intent breakdowns.
- Any comparison other than B − A and C − B.
- The relationship between `confidence` and `p_max`, and the count of `choice` ≠ argmax.
- Second-paraphrase robustness on a 200-item subset, if run.
- Anything suggested by looking at the results.

## 10. Deviations

Any departure from this document will be recorded in `results.md` under a "Deviations" heading,
with what changed and why. **This file is not rewritten after the first real run.** If the
pre-registration turns out to be wrong, it stays wrong in the record and the correction is
reported next to it.
