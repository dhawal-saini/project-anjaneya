"""Unit tests for ``core/research/significance.py`` (spec 24.9 - is any edge real?).

Spec 24 asks for significance "with correction for multiple testing", which is exactly where a
research tool earns or loses its credibility. These tests pin the four decisions that matter:

* **the arithmetic is the textbook procedure.** Both the two-proportion z-test and the
  Benjamini-Hochberg step-up are checked against hand-computed values, and against SciPy's own
  ``false_discovery_control`` where the two must agree.
* **no p-value without a sample.** A cell or a baseline below the minimum comes back
  ``INSUFFICIENT_SAMPLE`` with ``None`` p-values - never a number.
* **a degenerate test is refused.** All-winners on both sides, or constant P&L on both sides,
  is ``NOT_TESTABLE`` rather than a fudged ``p = 1.0``.
* **the family is the tests that ran.** ``None`` (untested) cells are excluded from the
  correction's denominator, so a matrix full of tiny cells cannot loosen the correction for the
  cells that were testable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from core.research.significance import (
    CORRECTIONS,
    INSUFFICIENT_SAMPLE,
    NOT_SIGNIFICANT,
    NOT_TESTABLE,
    SIGNIFICANT,
    adjust_p_values,
    compare_two_samples,
    mean_pnl_p_value,
    significance_of_matrix,
    win_rate_p_value,
)
from tests.unit.test_research_segments import trade


# ---------------------------------------------------------------------------
# The two tests
# ---------------------------------------------------------------------------
def test_the_two_proportion_test_matches_the_textbook_formula() -> None:
    # 60/100 vs 40/100: pooled p = 0.5, se = sqrt(0.25 * (0.01 + 0.01)) = 0.0707..., z = 2.828
    expected = 2.0 * stats.norm.sf(0.2 / math.sqrt(0.5 * 0.5 * (1 / 100 + 1 / 100)))

    value = win_rate_p_value(60, 40, 40, 60)

    assert value == pytest.approx(expected, rel=1e-12)
    assert value < 0.01


def test_the_two_proportion_test_refuses_an_undefined_comparison() -> None:
    assert win_rate_p_value(0, 0, 5, 5) is None
    assert win_rate_p_value(5, 5, 0, 0) is None
    assert win_rate_p_value(5, 0, 4, 0) is None  # every decided trade won: no variance


def test_welch_on_net_pnl_separates_a_real_difference_from_noise() -> None:
    same = [1.0, -1.0, 1.0, -1.0]
    different = [10.0, 11.0, 9.0, 10.0]

    identical = mean_pnl_p_value(same, same)
    contrast = mean_pnl_p_value(different, [-10.0, -11.0, -9.0, -10.0])

    assert identical == pytest.approx(1.0)
    assert contrast is not None and contrast < 0.001


def test_welch_refuses_a_test_it_cannot_run() -> None:
    assert mean_pnl_p_value([1.0], [2.0, 3.0]) is None  # fewer than two observations
    assert mean_pnl_p_value([1.0, 1.0], [1.0, 1.0]) is None  # no variance at all


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------
def test_benjamini_hochberg_matches_hand_computed_values() -> None:
    # sorted: 0.001, 0.02, 0.03, 0.2 with n = 4 -> 0.004, 0.04, 0.04, 0.2
    adjusted = adjust_p_values([0.001, 0.02, 0.03, 0.2], "benjamini_hochberg")

    assert adjusted == pytest.approx([0.004, 0.04, 0.04, 0.2])


def test_benjamini_hochberg_agrees_with_scipy() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]

    ours = adjust_p_values(p_values, "benjamini_hochberg")
    theirs = stats.false_discovery_control(np.asarray(p_values), method="bh")

    assert ours == pytest.approx(list(theirs))


def test_bonferroni_multiplies_by_the_family_and_caps_at_one() -> None:
    assert adjust_p_values([0.01, 0.4], "bonferroni") == pytest.approx([0.02, 0.8])
    assert adjust_p_values([0.9, 0.9], "bonferroni") == pytest.approx([1.0, 1.0])


def test_no_correction_leaves_the_values_alone() -> None:
    assert adjust_p_values([0.01, None, 0.2], "none") == [0.01, None, 0.2]


def test_untested_cells_do_not_inflate_the_family() -> None:
    """The family size is the number of tests that ran, not the number of cells."""
    adjusted = adjust_p_values([0.01, None, None, 0.03], "benjamini_hochberg")

    # n = 2, so the smallest becomes 0.02 and the largest stays 0.03
    assert adjusted == pytest.approx([0.02, None, None, 0.03])


def test_an_unknown_correction_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown correction"):
        adjust_p_values([0.1], "magic")
    assert "benjamini_hochberg" in CORRECTIONS


# ---------------------------------------------------------------------------
# One cell against the rest
# ---------------------------------------------------------------------------
def test_a_cell_below_the_sample_rule_is_never_tested() -> None:
    verdict = compare_two_samples(
        [trade(index=index) for index in range(3)],
        [trade(index=index + 10) for index in range(4)],
        preliminary_min_samples=5,
    )

    assert verdict.verdict == INSUFFICIENT_SAMPLE
    assert verdict.tested is False
    assert verdict.win_rate_p_value is None
    assert verdict.pnl_p_value is None
    assert "3 trade(s) in the cell and 4 outside it" in verdict.reason


def test_a_degenerate_cell_is_not_testable() -> None:
    # every trade on both sides is a winner: there is no variance to test
    winners = [trade(index=index, net=100.0) for index in range(6)]
    verdict = compare_two_samples(winners, list(winners), preliminary_min_samples=5)

    assert verdict.verdict == NOT_TESTABLE
    assert verdict.tested is False
    assert "constant" in verdict.reason or "no variance" in verdict.reason


def test_a_clearly_different_cell_is_tested() -> None:
    cell = [trade(index=index, net=90.0 + 10 * index) for index in range(10)]
    rest = [trade(index=index + 10, net=-90.0 - 10 * index) for index in range(10)]

    verdict = compare_two_samples(cell, rest, preliminary_min_samples=5)

    assert verdict.tested is True
    assert verdict.win_rate_p_value is not None
    assert verdict.pnl_p_value is not None
    assert verdict.win_rate_p_value < 0.01
    assert verdict.pnl_p_value < 0.01
    assert verdict.verdict == NOT_SIGNIFICANT  # the correction happens in test_matrix


# ---------------------------------------------------------------------------
# A whole matrix as one family
# ---------------------------------------------------------------------------
def test_a_matrix_is_tested_cell_by_cell_and_reported_whole() -> None:
    trades = [
        *[trade(index=index, bucket="CLOSE_1430_1515", net=100.0) for index in range(20)],
        *[trade(index=index + 20, bucket="OPEN_915_930", net=-100.0) for index in range(20)],
    ]
    matrix = build_matrix_for_test(trades)

    summary = significance_of_matrix(matrix, trades, significance_level=0.05, correction="bonferroni")

    assert summary.matrix == "time_of_day_x_option_type"
    assert summary.family_size == 2
    assert {test.label for test in summary.tests} == {
        "CLOSE_1430_1515 | CE",
        "OPEN_915_930 | CE",
    }
    assert len(summary.significant_cells) >= 1
    payload = summary.to_dict()
    assert payload["correction"] == "bonferroni"
    assert payload["family_size"] == 2
    assert all(
        test["verdict"] in {SIGNIFICANT, NOT_SIGNIFICANT, NOT_TESTABLE, INSUFFICIENT_SAMPLE}
        for test in payload["tests"]
    )


def test_a_matrix_of_small_cells_reports_that_nothing_was_tested() -> None:
    trades = [trade(index=0), trade(index=1)]
    matrix = build_matrix_for_test(trades)

    summary = significance_of_matrix(matrix, trades, significance_level=0.05, correction="benjamini_hochberg")

    assert summary.family_size == 0
    assert summary.significant_cells == []
    assert summary.preliminary_cells == ["MORNING_930_1100 | CE"]


def build_matrix_for_test(trades):
    """The bucket x side matrix the significance tests cut (imported lazily to stay local)."""
    from core.research.segments import build_matrix

    return build_matrix("time_of_day_x_option_type", trades, preliminary_min_samples=5)

