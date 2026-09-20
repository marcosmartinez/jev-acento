"""Metrics, implemented from scratch on numpy alone.

Everything here is deliberately dependency-free so the numbers can be audited by reading this
file. ``netcal`` appears only in the dev extra, used by the test suite to cross-check the ECE
implementation against an independent one.

Three choices are load-bearing and easy to get wrong:

**ECE binning.** Ten equal-width bins over ``p_max`` with explicit edges: each bin is closed on
the left and open on the right, except the top bin which is closed on both sides. This matters
because Jev rounds probabilities to 2 decimals, so values land exactly on bin edges constantly.
Leaving the rule implicit would make the number depend on floating-point luck.

**The ECE noise floor.** With finite n, ECE is biased *upward* even for a perfectly calibrated
model: each bin's empirical accuracy is a noisy estimate of its true rate, and ECE sums absolute
deviations, which never cancel. Reporting a raw ECE of 0.03 as "miscalibrated" is therefore
meaningless without knowing what a perfect model would have scored on the same number of items
at the same confidences. :func:`ece_noise_floor` simulates exactly that.

**Paired bootstrap.** Both arms see the same items, so the arms are correlated and an unpaired
interval on the difference is far too wide. :func:`paired_delta_ci` resamples *item indices* once
per replicate and recomputes both arms on that same resample, which cancels the shared item
difficulty and leaves only the effect of the arm.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

N_BOOT = 1000
N_BINS = 10
N_NOISE_SIMS = 1000


# --------------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------------


def accuracy(correct: np.ndarray) -> float:
    """Fraction correct. ``correct`` is a boolean array."""
    return float(np.mean(correct)) if correct.size else float("nan")


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Unweighted mean of per-label F1.

    Averaged over the declared label space, not over the labels that happen to appear. A label
    the model never predicts and never gets right contributes an honest 0, which is the point of
    macro-F1 on a long tail like MASSIVE's 59 intents.
    """
    gold_arr = np.asarray(gold)
    pred_arr = np.asarray(pred)
    if labels is None:
        labels = sorted(set(gold_arr.tolist()) | set(pred_arr.tolist()))

    scores = []
    for label in labels:
        tp = int(np.sum((gold_arr == label) & (pred_arr == label)))
        fp = int(np.sum((gold_arr != label) & (pred_arr == label)))
        fn = int(np.sum((gold_arr == label) & (pred_arr != label)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores)) if scores else float("nan")


def bin_edges(n_bins: int = N_BINS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def assign_bins(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Index of the bin each probability falls in.

    Left-closed / right-open, with the final bin closed so that ``p == 1.0`` lands in the top
    bin rather than falling off the end.
    """
    idx = np.floor(np.asarray(p, dtype=float) * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    p_max: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS
) -> list[ReliabilityBin]:
    """Per-bin counts, mean confidence and empirical accuracy.

    Empty bins are returned too, with ``count == 0`` and NaN statistics. They are kept because a
    reliability diagram that silently omits them misrepresents where the model actually lives --
    for a 3-class problem ``p_max >= 1/3``, so the bottom three bins are empty by construction.
    """
    p_max = np.asarray(p_max, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = bin_edges(n_bins)
    idx = assign_bins(p_max, n_bins)

    out = []
    for b in range(n_bins):
        mask = idx == b
        n = int(np.sum(mask))
        out.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=n,
                mean_confidence=float(np.mean(p_max[mask])) if n else float("nan"),
                accuracy=float(np.mean(correct[mask])) if n else float("nan"),
            )
        )
    return out


def ece(p_max: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: count-weighted mean gap between confidence and accuracy."""
    p_max = np.asarray(p_max, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = p_max.size
    if n == 0:
        return float("nan")

    idx = assign_bins(p_max, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        total += (cnt / n) * abs(float(np.mean(correct[mask])) - float(np.mean(p_max[mask])))
    return total


def ece_noise_floor(
    p_max: np.ndarray, n_sims: int = N_NOISE_SIMS, n_bins: int = N_BINS, seed: int = 0
) -> float:
    """Expected ECE of a *perfectly calibrated* model with this arm's confidence vector.

    Draws ``correct_i ~ Bernoulli(p_max_i)``, which is what perfect calibration means, and
    measures the ECE that finite-sample noise alone produces. An observed ECE below this floor is
    not evidence of good calibration, and one barely above it is not evidence of bad calibration.
    The decision rule in :mod:`jev_acento.analyse` requires a ratio of at least 1.5 before it
    will call calibration measurable at all.
    """
    p_max = np.asarray(p_max, dtype=float)
    if p_max.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.random((n_sims, p_max.size)) < p_max  # (n_sims, n) booleans
    return float(np.mean([ece(p_max, row, n_bins) for row in draws]))


# --------------------------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------------------------


def coverage_at_threshold(p_max: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """What you get if you automate every item at or above ``threshold`` and defer the rest."""
    p_max = np.asarray(p_max, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    mask = p_max >= threshold
    n_cov = int(np.sum(mask))
    return {
        "threshold": float(threshold),
        "coverage": float(n_cov / p_max.size) if p_max.size else float("nan"),
        "n_covered": n_cov,
        "accuracy": float(np.mean(correct[mask])) if n_cov else float("nan"),
    }


def risk_coverage_curve(p_max: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) as a function of coverage, most-confident-first.

    Ties in ``p_max`` are broken by original order. With probabilities rounded to 2 decimals ties
    are everywhere, so the curve is a valid summary but its fine structure is not meaningful.
    """
    p_max = np.asarray(p_max, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-p_max, kind="stable")
    errors = (~correct[order]).astype(float)
    n = p_max.size
    cum_err = np.cumsum(errors)
    k = np.arange(1, n + 1)
    return k / n, cum_err / k


def aurc(p_max: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage_curve(p_max, correct)
    return float(np.mean(risk)) if risk.size else float("nan")


# --------------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------------


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap CI for a statistic computed from resampled item indices.

    ``statistic`` receives an index array, not the data, so the caller controls exactly which
    arrays are resampled together.
    """
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(n))
    reps = np.array([statistic(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point=float(point), lo=float(lo), hi=float(hi))


def paired_delta_ci(
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """CI for ``stat_b - stat_a`` under a *shared* resample of item indices.

    One index draw per replicate feeds both arms. This is the whole point: the arms are measured
    on the same items, so their errors are correlated, and an unpaired interval would be wide
    enough to hide a real effect.
    """
    rng = np.random.default_rng(seed)
    base = np.arange(n)
    point = stat_b(base) - stat_a(base)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[i] = stat_b(idx) - stat_a(idx)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point=float(point), lo=float(lo), hi=float(hi))


# --------------------------------------------------------------------------------------------
# Stability between passes
# --------------------------------------------------------------------------------------------


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Chance-corrected agreement between two label sequences."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.size == 0:
        return float("nan")
    labels = sorted(set(a_arr.tolist()) | set(b_arr.tolist()))
    observed = float(np.mean(a_arr == b_arr))
    expected = sum(
        float(np.mean(a_arr == lab)) * float(np.mean(b_arr == lab)) for lab in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


@dataclass
class StabilityReport:
    """How much the model moves when handed a byte-identical request twice.

    This is the stability gate: if the model's own jitter between passes is as large as the
    difference between two arms, that difference is not evidence of anything.
    """

    n: int
    flip_rate: float
    kappa: float
    mean_abs_dp: float
    max_abs_dp: float
    accuracy_pass0: float
    accuracy_pass1: float

    @property
    def accuracy_jitter(self) -> float:
        return abs(self.accuracy_pass0 - self.accuracy_pass1)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "flip_rate": self.flip_rate,
            "kappa": self.kappa,
            "mean_abs_dp_max": self.mean_abs_dp,
            "max_abs_dp_max": self.max_abs_dp,
            "accuracy_pass0": self.accuracy_pass0,
            "accuracy_pass1": self.accuracy_pass1,
            "accuracy_jitter": self.accuracy_jitter,
        }


def stability(
    pred0: Sequence[str],
    pred1: Sequence[str],
    p0: np.ndarray,
    p1: np.ndarray,
    correct0: np.ndarray,
    correct1: np.ndarray,
) -> StabilityReport:
    pred0_arr, pred1_arr = np.asarray(pred0), np.asarray(pred1)
    dp = np.abs(np.asarray(p0, dtype=float) - np.asarray(p1, dtype=float))
    return StabilityReport(
        n=int(pred0_arr.size),
        flip_rate=float(np.mean(pred0_arr != pred1_arr)) if pred0_arr.size else float("nan"),
        kappa=cohens_kappa(pred0_arr, pred1_arr),
        mean_abs_dp=float(np.mean(dp)) if dp.size else float("nan"),
        max_abs_dp=float(np.max(dp)) if dp.size else float("nan"),
        accuracy_pass0=accuracy(np.asarray(correct0, dtype=bool)),
        accuracy_pass1=accuracy(np.asarray(correct1, dtype=bool)),
    )


# --------------------------------------------------------------------------------------------
# Inherited sanity checks (brief section 9)
# --------------------------------------------------------------------------------------------


def confidence_formula_residuals(
    p_max: np.ndarray, confidence: np.ndarray, k: int
) -> dict[str, float]:
    """Test the hypothesis that ``confidence == (k * p_max - 1) / (k - 1)``.

    Inherited from the Russian audit. If it holds, ``confidence`` carries no information beyond
    ``p_max`` and can be ignored; if it does not, it is a second signal worth reporting. The
    smoke test found k=3, p_max=0.99 -> predicted 0.985, reported 0.98, which is consistent with
    truncation rather than rounding -- hence both residuals below.
    """
    p_max = np.asarray(p_max, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    mask = ~np.isnan(confidence)
    if not np.any(mask) or k < 2:
        return {"n": 0, "max_abs_residual": float("nan"), "max_abs_residual_trunc": float("nan")}

    predicted = (k * p_max[mask] - 1.0) / (k - 1)
    observed = confidence[mask]
    return {
        "n": int(np.sum(mask)),
        "max_abs_residual": float(np.max(np.abs(np.round(predicted, 2) - observed))),
        "max_abs_residual_trunc": float(
            np.max(np.abs(np.floor(predicted * 100) / 100 - observed))
        ),
    }


@dataclass
class ArmMetrics:
    """Every number reported for a single (dataset, arm) cell."""

    dataset: str
    arm: str
    n: int
    accuracy: Interval
    macro_f1: Interval
    ece: float
    ece_floor: float
    ece_ratio: float
    aurc: float
    coverage: list[dict] = field(default_factory=list)
    bins: list[ReliabilityBin] = field(default_factory=list)
    mean_input_tokens: float = float("nan")
    latency_p50: float = float("nan")
    latency_p95: float = float("nan")
    n_choice_disagrees: int = 0

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "arm": self.arm,
            "n": self.n,
            "accuracy": self.accuracy.as_dict(),
            "macro_f1": self.macro_f1.as_dict(),
            "ece": self.ece,
            "ece_floor": self.ece_floor,
            "ece_ratio": self.ece_ratio,
            "aurc": self.aurc,
            "coverage": self.coverage,
            "bins": [vars(b) for b in self.bins],
            "mean_input_tokens": self.mean_input_tokens,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "n_choice_disagrees": self.n_choice_disagrees,
        }
