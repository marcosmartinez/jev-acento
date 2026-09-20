"""Analysis and the pre-registered decision rule.

The rule is deliberately conservative and deliberately mechanical. It is written down here, in
code, so that the verdict for each comparison is a function of the data and nothing else -- not
of how the result looks once it arrives.

It is applied **in order**, and each gate can stop the comparison:

1. **Stability gate.** Pass 0 and pass 1 are byte-identical requests. If the model's own
   movement between them is as large as the difference between two arms, that difference is not
   evidence of anything, and the comparison returns ``UNSTABLE`` without being interpreted.
2. **Accuracy.** A paired bootstrap CI that excludes 0 *and* a point estimate of at least 3
   percentage points gives "measurably better/worse". A CI that includes 0 and is tight enough
   (half-width <= 3pp) gives "no detectable difference" -- a real, reportable finding, not a
   failure. Anything else is ``AMBIGUOUS``: the study could not tell.
3. **Calibration.** Only assessed when ECE is meaningfully above its own noise floor
   (ratio >= 1.5) in *both* arms. Below that, the ECEs are finite-sample noise and comparing
   them is comparing noise to noise.

Two rules that are not conditions but habits:

* Verdicts are never reached by eyeballing whether two per-arm CIs overlap. Overlapping
  intervals routinely hide a significant paired difference. The delta has its own interval.
* ``AMBIGUOUS`` is reported as a result. It is what an underpowered comparison honestly yields.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics as M
from .data import SPECS
from .questions import ARMS

# Pre-registered thresholds. Matched to the Russian audit so the two are comparable.
MIN_ACCURACY_DELTA = 0.03       # 3 percentage points
MAX_NULL_HALF_WIDTH = 0.03      # a null result needs a CI at least this tight
MIN_ECE_DELTA = 0.02
MIN_ECE_RATIO = 1.5             # ECE must be this far above its noise floor to be measurable
COVERAGE_THRESHOLDS = (0.5, 0.9)

PRIMARY_COMPARISONS = (("A", "B"), ("B", "C"))


@dataclass
class Cell:
    """One (dataset, arm, pass) worth of rows, as parallel arrays ordered by item id."""

    dataset: str
    arm: str
    pass_k: int
    item_ids: list[str]
    gold: np.ndarray
    pred: np.ndarray
    p_max: np.ndarray
    confidence: np.ndarray
    input_tokens: np.ndarray
    latency_ms: np.ndarray
    n_choice_disagrees: int

    @property
    def correct(self) -> np.ndarray:
        return self.gold == self.pred

    def subset(self, item_ids: Sequence[str]) -> Cell:
        """Restrict to the given item ids, in the given order. Used to pair arms."""
        pos = {iid: i for i, iid in enumerate(self.item_ids)}
        idx = np.array([pos[i] for i in item_ids], dtype=int)
        return Cell(
            dataset=self.dataset,
            arm=self.arm,
            pass_k=self.pass_k,
            item_ids=list(item_ids),
            gold=self.gold[idx],
            pred=self.pred[idx],
            p_max=self.p_max[idx],
            confidence=self.confidence[idx],
            input_tokens=self.input_tokens[idx],
            latency_ms=self.latency_ms[idx],
            n_choice_disagrees=self.n_choice_disagrees,
        )


def load_cells(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], Cell]:
    """Group raw rows into cells, de-duplicating by item id.

    A resumed run can legitimately contain a repeated item id if it was interrupted between the
    call and the write. The first occurrence wins, so the result does not depend on how many
    times the run was restarted.
    """
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["arm"], int(row["pass"]))
        grouped[key].setdefault(row["item_id"], row)

    cells: dict[tuple[str, str, int], Cell] = {}
    for key, by_id in grouped.items():
        ordered = [by_id[i] for i in sorted(by_id)]
        cells[key] = Cell(
            dataset=key[0],
            arm=key[1],
            pass_k=key[2],
            item_ids=[r["item_id"] for r in ordered],
            gold=np.array([r["gold"] for r in ordered]),
            pred=np.array([r["pred"] for r in ordered]),
            p_max=np.array([float(r["p_max"]) for r in ordered]),
            confidence=np.array(
                [np.nan if r.get("confidence") is None else float(r["confidence"])
                 for r in ordered]
            ),
            input_tokens=np.array([int(r["input_tokens"]) for r in ordered]),
            latency_ms=np.array([float(r["latency_ms"]) for r in ordered]),
            n_choice_disagrees=sum(
                1 for r in ordered if r.get("choice_disagrees_with_argmax")
            ),
        )
    return cells


def arm_metrics(cell: Cell, labels: Sequence[str], seed: int) -> M.ArmMetrics:
    """Every reported number for one cell."""
    n = len(cell.item_ids)
    correct = cell.correct

    acc = M.bootstrap_ci(lambda idx: M.accuracy(correct[idx]), n, seed=seed)
    f1 = M.bootstrap_ci(
        lambda idx: M.macro_f1(cell.gold[idx], cell.pred[idx], labels), n, seed=seed + 1
    )
    observed_ece = M.ece(cell.p_max, correct)
    floor = M.ece_noise_floor(cell.p_max, seed=seed + 2)

    return M.ArmMetrics(
        dataset=cell.dataset,
        arm=cell.arm,
        n=n,
        accuracy=acc,
        macro_f1=f1,
        ece=observed_ece,
        ece_floor=floor,
        ece_ratio=observed_ece / floor if floor and not np.isnan(floor) else float("nan"),
        aurc=M.aurc(cell.p_max, correct),
        coverage=[M.coverage_at_threshold(cell.p_max, correct, t) for t in COVERAGE_THRESHOLDS],
        bins=M.reliability_bins(cell.p_max, correct),
        mean_input_tokens=float(np.mean(cell.input_tokens)),
        latency_p50=float(np.percentile(cell.latency_ms, 50)),
        latency_p95=float(np.percentile(cell.latency_ms, 95)),
        n_choice_disagrees=cell.n_choice_disagrees,
    )


@dataclass
class Comparison:
    """One pre-registered arm-to-arm comparison, with its verdicts."""

    dataset: str
    arm_from: str
    arm_to: str
    n_paired: int
    delta_accuracy: M.Interval
    delta_ece: M.Interval
    accuracy_verdict: str
    accuracy_reason: str
    calibration_verdict: str
    calibration_reason: str
    stability_jitter: float
    token_ratio_per_call: float
    token_ratio_state_only: float
    ece_ratio_from: float
    ece_ratio_to: float

    @property
    def label(self) -> str:
        return f"{self.arm_to} - {self.arm_from}"

    def as_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["delta_accuracy"] = self.delta_accuracy.as_dict()
        d["delta_ece"] = self.delta_ece.as_dict()
        d["label"] = self.label
        return d


def accuracy_verdict(delta: M.Interval, jitter: float) -> tuple[str, str]:
    """Gate 1 then gate 2 of the decision rule."""
    magnitude = abs(delta.point)

    # `jitter > 0` guards the degenerate case: a model that returns byte-identical answers to
    # byte-identical requests has zero jitter, and must never be called unstable -- not even
    # when the delta is also exactly zero, where `0 >= 0` would otherwise fire the gate.
    if jitter > 0 and jitter >= magnitude:
        return (
            "UNSTABLE",
            f"between-pass jitter ({jitter:.4f}) is at least as large as the observed delta "
            f"({delta.point:+.4f}); the difference is within the model's own noise",
        )
    if delta.excludes_zero() and magnitude >= MIN_ACCURACY_DELTA:
        direction = "better" if delta.point > 0 else "worse"
        return (
            f"MEASURABLY {direction.upper()}",
            f"CI [{delta.lo:+.4f}, {delta.hi:+.4f}] excludes 0 and |delta| = {magnitude:.4f} "
            f">= {MIN_ACCURACY_DELTA}",
        )
    if not delta.excludes_zero() and delta.half_width <= MAX_NULL_HALF_WIDTH:
        return (
            "NO DETECTABLE DIFFERENCE",
            f"CI [{delta.lo:+.4f}, {delta.hi:+.4f}] includes 0 and is tight "
            f"(half-width {delta.half_width:.4f} <= {MAX_NULL_HALF_WIDTH})",
        )
    return (
        "AMBIGUOUS",
        f"CI [{delta.lo:+.4f}, {delta.hi:+.4f}] with |delta| = {magnitude:.4f}: neither a "
        f"clear effect nor a tight null. The study cannot tell.",
    )


def calibration_verdict(
    delta: M.Interval, ratio_from: float, ratio_to: float
) -> tuple[str, str]:
    """Gate 3 of the decision rule."""
    if np.isnan(ratio_from) or np.isnan(ratio_to):
        return "NOT MEASURABLE", "an ECE noise floor could not be computed"
    if ratio_from < MIN_ECE_RATIO or ratio_to < MIN_ECE_RATIO:
        return (
            "NOT MEASURABLE",
            f"ECE is not meaningfully above its noise floor in both arms "
            f"(ratios {ratio_from:.2f} and {ratio_to:.2f}, need >= {MIN_ECE_RATIO}); "
            f"the observed ECEs are consistent with finite-sample noise",
        )
    magnitude = abs(delta.point)
    if delta.excludes_zero() and magnitude >= MIN_ECE_DELTA:
        direction = "less" if delta.point > 0 else "more"
        return (
            f"{direction.upper()} CALIBRATED",
            f"CI [{delta.lo:+.4f}, {delta.hi:+.4f}] excludes 0 and |delta ECE| = "
            f"{magnitude:.4f} >= {MIN_ECE_DELTA}",
        )
    if not delta.excludes_zero():
        return (
            "NO DETECTABLE DIFFERENCE",
            f"CI [{delta.lo:+.4f}, {delta.hi:+.4f}] includes 0",
        )
    return (
        "AMBIGUOUS",
        f"CI [{delta.lo:+.4f}, {delta.hi:+.4f}] excludes 0 but |delta ECE| = {magnitude:.4f} "
        f"< {MIN_ECE_DELTA}: statistically detectable, practically negligible",
    )


def compare(
    cell_from: Cell,
    cell_to: Cell,
    *,
    jitter: float,
    overhead: dict[str, int],
    seed: int,
) -> Comparison:
    """One paired comparison between two arms of the same dataset."""
    shared = sorted(set(cell_from.item_ids) & set(cell_to.item_ids))
    a, b = cell_from.subset(shared), cell_to.subset(shared)
    n = len(shared)

    correct_a, correct_b = a.correct, b.correct
    d_acc = M.paired_delta_ci(
        lambda idx: M.accuracy(correct_a[idx]),
        lambda idx: M.accuracy(correct_b[idx]),
        n,
        seed=seed,
    )
    d_ece = M.paired_delta_ci(
        lambda idx: M.ece(a.p_max[idx], correct_a[idx]),
        lambda idx: M.ece(b.p_max[idx], correct_b[idx]),
        n,
        seed=seed + 1,
    )

    ratio_from = M.ece(a.p_max, correct_a) / M.ece_noise_floor(a.p_max, seed=seed + 2)
    ratio_to = M.ece(b.p_max, correct_b) / M.ece_noise_floor(b.p_max, seed=seed + 3)

    # Tokens. The per-call ratio is what you are billed; the state-only ratio is the property of
    # the language itself, with the fixed question wording subtracted out.
    mean_a, mean_b = float(np.mean(a.input_tokens)), float(np.mean(b.input_tokens))
    oh_a = overhead.get(f"{a.dataset}/{a.arm}", 0)
    oh_b = overhead.get(f"{b.dataset}/{b.arm}", 0)
    state_a, state_b = mean_a - oh_a, mean_b - oh_b

    acc_v, acc_r = accuracy_verdict(d_acc, jitter)
    cal_v, cal_r = calibration_verdict(d_ece, ratio_from, ratio_to)

    return Comparison(
        dataset=a.dataset,
        arm_from=a.arm,
        arm_to=b.arm,
        n_paired=n,
        delta_accuracy=d_acc,
        delta_ece=d_ece,
        accuracy_verdict=acc_v,
        accuracy_reason=acc_r,
        calibration_verdict=cal_v,
        calibration_reason=cal_r,
        stability_jitter=jitter,
        token_ratio_per_call=mean_b / mean_a if mean_a else float("nan"),
        token_ratio_state_only=state_b / state_a if state_a > 0 else float("nan"),
        ece_ratio_from=ratio_from,
        ece_ratio_to=ratio_to,
    )


@dataclass
class Results:
    run_id: str
    arms: list[dict] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    stability: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "arms": self.arms,
            "comparisons": self.comparisons,
            "stability": self.stability,
            "checks": self.checks,
            "warnings": self.warnings,
        }


def analyse(
    rows: list[dict[str, Any]],
    run_id: str,
    *,
    manifest: dict[str, Any] | None = None,
    seed: int = 20260920,
) -> Results:
    """Full analysis of a run: per-arm metrics, stability, and the pre-registered comparisons."""
    cells = load_cells(rows)
    results = Results(run_id=run_id)

    manifest = manifest or {}
    overhead = manifest.get("summary", {}).get("question_overhead", {})

    before = manifest.get("summary", {}).get("model_release_before", {})
    after = manifest.get("summary", {}).get("model_release_after", {})
    if before and after and before.get("release_date") != after.get("release_date"):
        results.warnings.append(
            f"MODEL CHANGED MID-RUN: release_date went from {before.get('release_date')!r} to "
            f"{after.get('release_date')!r}. This run is not internally comparable; discard it."
        )

    datasets = sorted({c.dataset for c in cells.values()})

    for dataset in datasets:
        spec = SPECS.get(dataset)
        labels = sorted({g for c in cells.values() if c.dataset == dataset for g in c.gold})
        if spec and spec.options:
            labels = list(spec.options)

        # Per-arm metrics, always from pass 0 (the pre-registered primary pass).
        for arm in ARMS:
            cell = cells.get((dataset, arm, 0))
            if cell is None:
                continue
            results.arms.append(arm_metrics(cell, labels, seed).as_dict())

        # Stability: pass 0 against pass 1, paired on item id.
        jitter_by_arm: dict[str, float] = {}
        for arm in ARMS:
            c0, c1 = cells.get((dataset, arm, 0)), cells.get((dataset, arm, 1))
            if c0 is None or c1 is None:
                continue
            shared = sorted(set(c0.item_ids) & set(c1.item_ids))
            s0, s1 = c0.subset(shared), c1.subset(shared)
            report = M.stability(s0.pred, s1.pred, s0.p_max, s1.p_max, s0.correct, s1.correct)
            jitter_by_arm[arm] = report.accuracy_jitter
            results.stability.append(
                {"dataset": dataset, "arm": arm, **report.as_dict()}
            )

        for arm_from, arm_to in PRIMARY_COMPARISONS:
            c_from, c_to = cells.get((dataset, arm_from, 0)), cells.get((dataset, arm_to, 0))
            if c_from is None or c_to is None:
                continue
            # The gate uses the larger of the two arms' jitter: the comparison is only as
            # trustworthy as its noisier side.
            jitter = max(
                jitter_by_arm.get(arm_from, 0.0), jitter_by_arm.get(arm_to, 0.0)
            )
            results.comparisons.append(
                compare(c_from, c_to, jitter=jitter, overhead=overhead, seed=seed).as_dict()
            )

    results.checks = _inherited_checks(cells)
    return results


def _inherited_checks(cells: dict[tuple[str, str, int], Cell]) -> dict:
    """The sanity checks carried over from the Russian audit (brief section 9)."""
    disagreements = sum(c.n_choice_disagrees for c in cells.values())
    formula: dict[str, Any] = {}
    for (dataset, arm, pass_k), cell in sorted(cells.items()):
        if pass_k != 0 or np.all(np.isnan(cell.confidence)):
            continue
        k = len(set(cell.gold.tolist()))
        formula[f"{dataset}/{arm}"] = M.confidence_formula_residuals(
            cell.p_max, cell.confidence, k
        )
    return {
        "n_choice_disagrees_with_argmax": disagreements,
        "confidence_formula": formula,
    }


def write_results(results: Results, out_dir: Path) -> tuple[Path, Path]:
    """Write ``results.json`` and the human-readable ``results.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(results.as_dict(), indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    md_path = out_dir / "results.md"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    return json_path, md_path


def _fmt_ci(d: dict) -> str:
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def render_markdown(results: Results) -> str:
    """Render the report. Every number here comes from ``results.json``; none is typed by hand."""
    out: list[str] = [
        f"# Results — run `{results.run_id}`",
        "",
        "Generated by `make reproduce`. Do not edit by hand: re-run instead.",
        "",
    ]

    if results.warnings:
        out += ["## ⚠️ Warnings", ""]
        out += [f"- {w}" for w in results.warnings]
        out += [""]

    out += [
        "## Verdicts",
        "",
        "Arms: **A** = English state, English instructions · **B** = Spanish state, English "
        "instructions · **C** = Spanish state, Spanish instructions.",
        "",
        "| Dataset | Comparison | Accuracy | Δ accuracy | Calibration | Δ ECE |",
        "|---|---|---|---|---|---|",
    ]
    for c in results.comparisons:
        out.append(
            f"| {c['dataset']} | {c['label']} | {c['accuracy_verdict']} | "
            f"{_fmt_ci(c['delta_accuracy'])} | {c['calibration_verdict']} | "
            f"{_fmt_ci(c['delta_ece'])} |"
        )

    out += ["", "### Why each verdict", ""]
    for c in results.comparisons:
        out += [
            f"**{c['dataset']} · {c['label']}** (n = {c['n_paired']} paired)",
            "",
            f"- Accuracy — *{c['accuracy_verdict']}*: {c['accuracy_reason']}",
            f"- Calibration — *{c['calibration_verdict']}*: {c['calibration_reason']}",
            f"- Tokens — {c['token_ratio_per_call']:.3f}× per call, "
            f"{c['token_ratio_state_only']:.3f}× state-only",
            "",
        ]

    out += [
        "## Per-arm metrics",
        "",
        "| Dataset | Arm | n | Accuracy | Macro-F1 | ECE | Floor | Ratio | AURC | "
        "Tokens | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in results.arms:
        out.append(
            f"| {a['dataset']} | {a['arm']} | {a['n']} | "
            f"{a['accuracy']['point']:.4f} [{a['accuracy']['lo']:.4f}, {a['accuracy']['hi']:.4f}] | "
            f"{a['macro_f1']['point']:.4f} | {a['ece']:.4f} | {a['ece_floor']:.4f} | "
            f"{a['ece_ratio']:.2f} | {a['aurc']:.4f} | {a['mean_input_tokens']:.0f} | "
            f"{a['latency_p50']:.0f} | {a['latency_p95']:.0f} |"
        )

    out += [
        "",
        "### Coverage at threshold",
        "",
        "What you get if you automate every item at or above the threshold and defer the rest.",
        "",
        "| Dataset | Arm | Threshold | Coverage | Accuracy on covered |",
        "|---|---|---|---|---|",
    ]
    for a in results.arms:
        for cov in a["coverage"]:
            out.append(
                f"| {a['dataset']} | {a['arm']} | {cov['threshold']:.2f} | "
                f"{cov['coverage']:.3f} | {cov['accuracy']:.4f} |"
            )

    out += [
        "",
        "## Stability (pass 0 vs pass 1)",
        "",
        "Byte-identical requests, sent twice. This bounds how much of any arm-to-arm difference "
        "is the model moving on its own.",
        "",
        "| Dataset | Arm | n | Flip rate | κ | mean \\|Δp\\| | max \\|Δp\\| | Accuracy jitter |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in results.stability:
        out.append(
            f"| {s['dataset']} | {s['arm']} | {s['n']} | {s['flip_rate']:.4f} | "
            f"{s['kappa']:.4f} | {s['mean_abs_dp_max']:.4f} | {s['max_abs_dp_max']:.4f} | "
            f"{s['accuracy_jitter']:.4f} |"
        )

    checks = results.checks
    out += [
        "",
        "## Inherited checks",
        "",
        f"- `choice` disagreed with argmax of `probabilities` in "
        f"**{checks.get('n_choice_disagrees_with_argmax', 0)}** answers "
        f"(2-decimal rounding produces ties; argmax is used throughout).",
    ]
    formula = checks.get("confidence_formula", {})
    if formula:
        worst = max(
            (v.get("max_abs_residual_trunc", float("nan")) for v in formula.values()),
            default=float("nan"),
        )
        out.append(
            f"- `confidence ≈ (k·p_max − 1)/(k − 1)`: largest residual under truncation is "
            f"**{worst:.4f}** across all cells."
        )
    out.append("")
    return "\n".join(out)
