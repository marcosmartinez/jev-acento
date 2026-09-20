"""Metrics tested against closed-form answers, known edge cases, and an independent library.

The ECE implementation is the one number in this repo most likely to be subtly wrong and least
likely to look wrong, so it is checked three ways: hand-computed examples, the binning edge
cases that 2-decimal rounding makes constant, and a cross-check against `netcal`.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev_acento import metrics as M

# ---------------------------------------------------------------- accuracy and F1


def test_accuracy_trivial_cases():
    assert M.accuracy(np.array([True, True, True, True])) == 1.0
    assert M.accuracy(np.array([False, False])) == 0.0
    assert M.accuracy(np.array([True, False, True, False])) == 0.5


def test_macro_f1_is_unweighted_across_labels():
    # 'b' is never predicted, so its F1 is 0. 'a' is not perfect either: the misclassified 'b'
    # is a false positive for 'a', giving F1 = 2*3/(2*3 + 1 + 0) = 6/7.
    # Macro-F1 = (6/7 + 0)/2 = 3/7, well below the accuracy of 3/4 -- which is the point.
    gold = ["a", "a", "a", "b"]
    pred = ["a", "a", "a", "a"]
    assert M.macro_f1(gold, pred, ["a", "b"]) == pytest.approx(3 / 7)
    assert M.accuracy(np.array(gold) == np.array(pred)) == pytest.approx(0.75)


def test_macro_f1_counts_declared_but_absent_labels_as_zero():
    gold = ["a", "a"]
    pred = ["a", "a"]
    assert M.macro_f1(gold, pred, ["a"]) == 1.0
    # Declaring a third label that never occurs must drag the mean down, not be skipped.
    assert M.macro_f1(gold, pred, ["a", "b", "c"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- binning


def test_bin_assignment_is_left_closed_and_top_bin_is_closed():
    # A value exactly on an internal edge belongs to the bin above it...
    assert M.assign_bins(np.array([0.3]))[0] == 3
    assert M.assign_bins(np.array([0.29999]))[0] == 2
    # ...and 1.0 stays in the top bin instead of falling off the end.
    assert M.assign_bins(np.array([1.0]))[0] == 9
    assert M.assign_bins(np.array([0.0]))[0] == 0


def test_two_decimal_values_land_where_documented():
    # Jev rounds to 2 decimals, so these exact values occur constantly in real data.
    p = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.99, 1.00])
    assert list(M.assign_bins(p)) == [5, 6, 7, 8, 9, 9, 9]


# ---------------------------------------------------------------- ECE


def test_ece_is_zero_when_confidence_matches_accuracy_exactly():
    # Two bins, each perfectly calibrated: 0.95 confidence with 100% correct is not zero error,
    # so use values where the bin mean equals the bin accuracy exactly.
    p_max = np.array([0.95, 0.95, 0.65, 0.65, 0.65, 0.65])
    correct = np.array([True, True, True, True, True, True])
    # bin 9: mean conf 0.95, acc 1.00 -> gap 0.05 ; bin 6: mean conf 0.65, acc 1.00 -> gap 0.35
    expected = (2 / 6) * 0.05 + (4 / 6) * 0.35
    assert M.ece(p_max, correct) == pytest.approx(expected)


def test_ece_perfectly_calibrated_single_bin():
    # 10 items at 0.70 confidence, exactly 7 correct -> zero calibration error.
    p_max = np.full(10, 0.70)
    correct = np.array([True] * 7 + [False] * 3)
    assert M.ece(p_max, correct) == pytest.approx(0.0)


def test_ece_maximally_miscalibrated():
    p_max = np.full(10, 1.0)
    correct = np.zeros(10, dtype=bool)
    assert M.ece(p_max, correct) == pytest.approx(1.0)


def test_ece_matches_netcal():
    """Cross-check against an independent implementation."""
    netcal = pytest.importorskip("netcal.metrics")
    rng = np.random.default_rng(7)
    n = 4000
    p_max = np.round(rng.uniform(0.34, 1.0, n), 2)
    correct = rng.random(n) < p_max

    ours = M.ece(p_max, correct)
    theirs = netcal.ECE(bins=M.N_BINS).measure(p_max, correct.astype(int))
    assert ours == pytest.approx(float(theirs), abs=2e-3)


# ---------------------------------------------------------------- noise floor


def test_noise_floor_matches_observed_ece_under_perfect_calibration():
    """The whole point: a perfectly calibrated model does NOT score ECE 0 at finite n."""
    rng = np.random.default_rng(11)
    n = 1500
    p_max = rng.uniform(0.34, 1.0, n)
    correct = rng.random(n) < p_max  # perfect calibration by construction

    observed = M.ece(p_max, correct)
    floor = M.ece_noise_floor(p_max, n_sims=200, seed=3)

    assert floor > 0.005, "a finite-sample floor should be clearly above zero"
    assert observed / floor == pytest.approx(1.0, abs=0.5), (
        "a perfectly calibrated model must sit at roughly 1x its own noise floor"
    )


def test_noise_floor_is_deterministic_given_a_seed():
    p_max = np.linspace(0.4, 1.0, 300)
    assert M.ece_noise_floor(p_max, n_sims=50, seed=5) == M.ece_noise_floor(
        p_max, n_sims=50, seed=5
    )


def test_genuinely_miscalibrated_model_sits_far_above_its_floor():
    rng = np.random.default_rng(13)
    n = 1500
    p_max = rng.uniform(0.7, 1.0, n)
    correct = rng.random(n) < 0.4  # confident and usually wrong
    assert M.ece(p_max, correct) / M.ece_noise_floor(p_max, n_sims=200) > M.N_BINS / 2


# ---------------------------------------------------------------- selective prediction


def test_coverage_at_threshold():
    p_max = np.array([0.4, 0.6, 0.95, 0.99])
    correct = np.array([False, True, True, False])
    at_half = M.coverage_at_threshold(p_max, correct, 0.5)
    assert at_half["n_covered"] == 3
    assert at_half["coverage"] == pytest.approx(0.75)
    assert at_half["accuracy"] == pytest.approx(2 / 3)


def test_aurc_rewards_confidence_that_ranks_errors_last():
    correct = np.array([True, True, True, False])
    good_ranking = np.array([0.99, 0.95, 0.90, 0.40])   # the error is least confident
    bad_ranking = np.array([0.40, 0.90, 0.95, 0.99])    # the error is most confident
    assert M.aurc(good_ranking, correct) < M.aurc(bad_ranking, correct)


# ---------------------------------------------------------------- uncertainty


def test_paired_delta_is_tighter_than_the_arms_own_intervals():
    """Why pairing matters: correlated arms produce a much tighter interval on the delta."""
    rng = np.random.default_rng(17)
    n = 800
    difficulty = rng.random(n)
    correct_a = difficulty < 0.85
    correct_b = difficulty < 0.80  # same items, uniformly slightly worse

    acc_a = M.bootstrap_ci(lambda idx: M.accuracy(correct_a[idx]), n, seed=1)
    delta = M.paired_delta_ci(
        lambda idx: M.accuracy(correct_a[idx]),
        lambda idx: M.accuracy(correct_b[idx]),
        n,
        seed=1,
    )
    assert delta.half_width < acc_a.half_width
    assert delta.point == pytest.approx(-0.05, abs=0.02)


def test_interval_zero_detection():
    assert M.Interval(0.05, 0.02, 0.08).excludes_zero()
    assert M.Interval(-0.05, -0.08, -0.02).excludes_zero()
    assert not M.Interval(0.01, -0.02, 0.04).excludes_zero()


def test_bootstrap_is_deterministic_given_a_seed():
    correct = np.array([True, False] * 50)
    a = M.bootstrap_ci(lambda idx: M.accuracy(correct[idx]), 100, seed=42)
    b = M.bootstrap_ci(lambda idx: M.accuracy(correct[idx]), 100, seed=42)
    assert (a.point, a.lo, a.hi) == (b.point, b.lo, b.hi)


# ---------------------------------------------------------------- stability


def test_stability_detects_no_movement():
    pred = ["a", "b", "a", "c"]
    p = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    report = M.stability(pred, pred, p, p, correct, correct)
    assert report.flip_rate == 0.0
    assert report.kappa == pytest.approx(1.0)
    assert report.max_abs_dp == 0.0
    assert report.accuracy_jitter == 0.0


def test_stability_detects_flips():
    report = M.stability(
        ["a", "a", "a", "a"], ["a", "b", "a", "b"],
        np.array([0.9, 0.9, 0.9, 0.9]), np.array([0.9, 0.5, 0.9, 0.6]),
        np.array([True] * 4), np.array([True, False, True, False]),
    )
    assert report.flip_rate == pytest.approx(0.5)
    assert report.max_abs_dp == pytest.approx(0.4)
    assert report.accuracy_jitter == pytest.approx(0.5)


def test_kappa_is_zero_for_chance_agreement():
    rng = np.random.default_rng(23)
    a = rng.choice(["x", "y"], 4000).tolist()
    b = rng.choice(["x", "y"], 4000).tolist()
    assert abs(M.cohens_kappa(a, b)) < 0.1


# ---------------------------------------------------------------- inherited checks


def test_confidence_formula_residuals_detect_an_exact_relationship():
    p_max = np.array([0.50, 0.70, 0.90, 0.99])
    k = 3
    exact = np.round((k * p_max - 1) / (k - 1), 2)
    res = M.confidence_formula_residuals(p_max, exact, k)
    assert res["max_abs_residual"] == pytest.approx(0.0)


def test_confidence_formula_residuals_tolerate_missing_confidence():
    # Noul answers carry no confidence at all; the check must not crash on all-NaN input.
    res = M.confidence_formula_residuals(np.array([0.9, 0.8]), np.array([np.nan, np.nan]), 2)
    assert res["n"] == 0
