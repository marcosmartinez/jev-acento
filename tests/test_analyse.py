"""The decision rule. Each gate is exercised on data constructed to land in exactly one branch."""

from __future__ import annotations

import numpy as np

from jev_acento.analyse import (
    accuracy_verdict,
    analyse,
    calibration_verdict,
    load_cells,
)
from jev_acento.metrics import Interval

# ---------------------------------------------------------------- gate 1: stability


def test_stability_gate_fires_when_jitter_swamps_the_delta():
    delta = Interval(point=0.02, lo=0.01, hi=0.03)  # would otherwise be significant
    verdict, reason = accuracy_verdict(delta, jitter=0.05)
    assert verdict == "UNSTABLE"
    assert "jitter" in reason


def test_stability_gate_runs_before_everything_else():
    """A large, tight, significant delta is still discarded if the model is that noisy."""
    delta = Interval(point=0.20, lo=0.15, hi=0.25)
    assert accuracy_verdict(delta, jitter=0.25)[0] == "UNSTABLE"


def test_stable_comparison_passes_the_gate():
    delta = Interval(point=-0.08, lo=-0.11, hi=-0.05)
    assert accuracy_verdict(delta, jitter=0.004)[0] == "MEASURABLY WORSE"


# ---------------------------------------------------------------- gate 2: accuracy


def test_significant_and_large_enough_is_measurable():
    assert accuracy_verdict(Interval(0.05, 0.02, 0.08), 0.0)[0] == "MEASURABLY BETTER"
    assert accuracy_verdict(Interval(-0.05, -0.08, -0.02), 0.0)[0] == "MEASURABLY WORSE"


def test_significant_but_too_small_is_ambiguous():
    """Statistically detectable, practically irrelevant: below 3pp it is not a finding."""
    verdict, _ = accuracy_verdict(Interval(0.015, 0.005, 0.025), 0.0)
    assert verdict == "AMBIGUOUS"


def test_tight_null_is_a_real_result_not_a_failure():
    verdict, reason = accuracy_verdict(Interval(0.002, -0.015, 0.019), 0.0)
    assert verdict == "NO DETECTABLE DIFFERENCE"
    assert "includes 0" in reason


def test_wide_null_is_ambiguous_not_a_null():
    """The commonest way to overclaim: reading an underpowered study as evidence of no effect."""
    verdict, _ = accuracy_verdict(Interval(0.005, -0.09, 0.10), 0.0)
    assert verdict == "AMBIGUOUS"


def test_the_null_threshold_boundary():
    assert accuracy_verdict(Interval(0.0, -0.03, 0.03), 0.0)[0] == "NO DETECTABLE DIFFERENCE"
    assert accuracy_verdict(Interval(0.0, -0.031, 0.031), 0.0)[0] == "AMBIGUOUS"


# ---------------------------------------------------------------- gate 3: calibration


def test_calibration_not_measurable_near_the_noise_floor():
    verdict, reason = calibration_verdict(Interval(0.05, 0.02, 0.08), 1.2, 3.0)
    assert verdict == "NOT MEASURABLE"
    assert "noise floor" in reason


def test_calibration_requires_both_arms_above_the_floor():
    assert calibration_verdict(Interval(0.05, 0.02, 0.08), 3.0, 1.4)[0] == "NOT MEASURABLE"
    assert calibration_verdict(Interval(0.05, 0.02, 0.08), 1.5, 1.5)[0] == "LESS CALIBRATED"


def test_calibration_direction():
    # A positive delta means arm_to has MORE error, i.e. it is less calibrated.
    assert calibration_verdict(Interval(0.05, 0.02, 0.08), 3.0, 3.0)[0] == "LESS CALIBRATED"
    assert calibration_verdict(Interval(-0.05, -0.08, -0.02), 3.0, 3.0)[0] == "MORE CALIBRATED"


def test_calibration_null_and_ambiguous():
    assert calibration_verdict(Interval(0.001, -0.01, 0.012), 3.0, 3.0)[0] == (
        "NO DETECTABLE DIFFERENCE"
    )
    # Excludes zero but below the 0.02 practical threshold.
    assert calibration_verdict(Interval(0.01, 0.004, 0.016), 3.0, 3.0)[0] == "AMBIGUOUS"


# ---------------------------------------------------------------- plumbing


def _row(item_id, arm, pass_k, gold, pred, p_max, **extra):
    return {
        "dataset": "toy", "item_id": item_id, "arm": arm, "pass": pass_k,
        "gold": gold, "pred": pred, "p_max": p_max, "confidence": None,
        "probs": {}, "input_tokens": 100, "latency_ms": 10.0,
        "model": "fake", "generation_id": "g", "ts": "now",
        "choice_disagrees_with_argmax": False, **extra,
    }


def test_load_cells_deduplicates_a_resumed_item():
    """A restart can legitimately re-record an item; the first write must win."""
    rows = [
        _row("i-1", "A", 0, "x", "x", 0.9),
        _row("i-1", "A", 0, "x", "y", 0.5),   # duplicate from a resumed run
        _row("i-2", "A", 0, "x", "x", 0.8),
    ]
    cell = load_cells(rows)[("toy", "A", 0)]
    assert len(cell.item_ids) == 2
    assert cell.pred[0] == "x"


def test_cells_are_ordered_by_item_id_so_arms_pair_correctly():
    rows = [_row(f"i-{i}", "A", 0, "x", "x", 0.9) for i in (3, 1, 2)]
    cell = load_cells(rows)[("toy", "A", 0)]
    assert cell.item_ids == ["i-1", "i-2", "i-3"]


def test_subset_pairs_arms_on_shared_items_only():
    rows = [_row(f"i-{i}", "A", 0, "x", "x", 0.9) for i in range(5)]
    cell = load_cells(rows)[("toy", "A", 0)]
    sub = cell.subset(["i-1", "i-3"])
    assert sub.item_ids == ["i-1", "i-3"]
    assert len(sub.p_max) == 2


def test_model_change_mid_run_produces_a_loud_warning():
    """The only version evidence the Gateway gives us; a change invalidates the run."""
    rows = [_row("i-1", "A", 0, "x", "x", 0.9)]
    manifest = {"summary": {
        "model_release_before": {"release_date": "2026-09-15"},
        "model_release_after": {"release_date": "2026-09-21"},
    }}
    results = analyse(rows, "r", manifest=manifest)
    assert any("MODEL CHANGED MID-RUN" in w for w in results.warnings)


def test_no_release_date_warning_when_the_model_held_still():
    rows = [_row("i-1", "A", 0, "x", "x", 0.9, model="jev-1.13.0")]
    manifest = {"summary": {
        "model_release_before": {"release_date": "2026-09-15"},
        "model_release_after": {"release_date": "2026-09-15"},
    }}
    warnings = analyse(rows, "r", manifest=manifest).warnings
    assert not any("MODEL CHANGED" in w for w in warnings)
    assert warnings == [], "a pinned model with a stable release date warrants no warning at all"


# ---------------------------------------------------------------- model provenance


def test_a_versioned_model_id_counts_as_pinned():
    """The direct TypeSafe API echoes the version that answered; that anchors the run."""
    rows = [_row(f"i-{i}", "A", 0, "x", "x", 0.9, model="jev-1.13.0") for i in range(4)]
    prov = analyse(rows, "r").checks["model_provenance"]
    assert prov["status"] == "pinned"
    assert prov["pinned"] is True
    assert prov["model"] == "jev-1.13.0"


def test_a_gateway_alias_is_flagged_as_unpinned():
    rows = [_row(f"i-{i}", "A", 0, "x", "x", 0.9, model="typesafe-ai/jev") for i in range(4)]
    results = analyse(rows, "r")
    assert results.checks["model_provenance"]["status"] == "unpinned"
    assert any("MODEL NOT PINNED" in w for w in results.warnings)


def test_jev_latest_is_not_a_pin():
    """An alias can be repointed between two passes of the same run."""
    rows = [_row("i-1", "A", 0, "x", "x", 0.9, model="jev-latest")]
    assert analyse(rows, "r").checks["model_provenance"]["pinned"] is False


def test_mixed_model_ids_invalidate_the_run():
    rows = [
        _row("i-1", "A", 0, "x", "x", 0.9, model="jev-1.13.0"),
        _row("i-2", "A", 0, "x", "x", 0.9, model="jev-1.14.0"),
    ]
    results = analyse(rows, "r")
    assert results.checks["model_provenance"]["status"] == "mixed"
    assert any("MIXED MODEL IDS" in w for w in results.warnings)


def test_pinned_provenance_is_stated_in_the_report():
    from jev_acento.analyse import render_markdown

    rows = [_row(f"i-{i}", "A", 0, "x", "x", 0.9, model="jev-1.13.0") for i in range(4)]
    assert "Model pinned" in render_markdown(analyse(rows, "r"))


def test_analyse_produces_both_primary_comparisons():
    rng = np.random.default_rng(5)
    rows = []
    for arm, acc in (("A", 0.9), ("B", 0.8), ("C", 0.82)):
        for pass_k in (0, 1):
            for i in range(120):
                ok = rng.random() < acc
                rows.append(_row(f"i-{i:03d}", arm, pass_k, "x", "x" if ok else "y",
                                 round(rng.uniform(0.5, 1.0), 2)))
    results = analyse(rows, "r")
    labels = {c["label"] for c in results.comparisons}
    assert labels == {"B - A", "C - B"}
    assert len(results.arms) == 3
    assert len(results.stability) == 3
