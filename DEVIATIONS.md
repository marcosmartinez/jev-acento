The run of 2026-09-21 (`20260921-es-v1`) followed `PREREG.md` **exactly**. No datasets, sample
sizes, arms, seed, metrics or decision-rule thresholds were changed after registration, and no
comparison was added, dropped or re-specified after seeing the data.

Three things are worth recording anyway, none of which is a departure:

**1. One verdict sits exactly on a boundary.** For Belebele `C − B`, the paired accuracy CI is
`[+0.0000, +0.0117]` — its lower bound is exactly zero, not a rounded value. The rule requires
`lo > 0` to call an interval "excludes 0", so this is classified as *no detectable difference*.
Had the bound fallen one resample the other way it would have read the same way in practice
(|Δ| = 0.005 is far below the 3 pp threshold, so the alternative branch is AMBIGUOUS, not a
finding either way). Recorded because a reader deserves to see that it was close, not because
the outcome is in doubt.

**2. The state-only token ratio is exactly 1.000 for every `C − B` comparison.** This is a
sanity check passing, not a suspicious number: arms B and C send the *identical Spanish state*
and differ only in the language of the instructions. A state-only ratio of anything other than
1.000 there would have indicated a bug in the overhead probe.

**3. `confidence` is not an exact function of `p_max`.** The inherited check from the Russian
audit tested `confidence ≈ (k·p_max − 1)/(k − 1)`. The largest residual under truncation is
0.0200 across all cells, so the relationship is close but does not hold exactly. `confidence`
was not used in any pre-registered metric — everything is computed from `p_max` — so this
changes no result. It is reported as the exploratory observation section 9 says it is.
