"""Spec 24.9: is any apparent edge real? - tested, then corrected for multiple testing.

The matrix cells in :mod:`core.research.segments` are the *hypotheses*: "the close bucket
pays", "the expiry-day phase pays". Tested naively, one of them always looks good, so this
module tests each cell **against everything outside it** - the cell's own trades versus the
rest of the same ledger - on the two things spec 24 asks about:

* the **win rate** (a two-proportion z-test: the cell's wins/losses vs the rest's), and
* the **mean net P&L per trade** (Welch's t-test, which does not assume equal variances - a
  stop-loss leg and a runner leg never have the same spread).

Three rules make the answer usable rather than decorative:

* **No test without a sample.** A cell (or a baseline) below
  ``evaluation.preliminary_min_samples`` is *not tested at all*: its verdict is
  ``INSUFFICIENT_SAMPLE`` and its p-values stay ``None``. A p-value computed from six trades
  is noise dressed as evidence, and printing one is how a small sample becomes a finding.
* **The correction is applied to the tests that were performed.** ``none``,
  ``bonferroni`` and ``benjamini_hochberg`` are implemented here (no SciPy version
  surprises), each matrix is one *family* - it is one question asked several ways - and the
  family size is reported next to every adjusted p-value.
* **A degenerate test is refused, not fudged.** If every trade on both sides won (or all
  lost) the win rate has no variance to test, and if both sides' P&L are constant the
  t-test is undefined: those come back as ``None`` with the reason recorded.

The result is a :class:`CellTest` per cell and a :class:`SignificanceSummary` per matrix,
each carrying the sample sizes, both raw p-values, both adjusted p-values and the verdict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import stats

from core.backtest.metrics import net_pnl_values

__all__ = [
    "CORRECTIONS",
    "INSUFFICIENT_SAMPLE",
    "NOT_SIGNIFICANT",
    "NOT_TESTABLE",
    "SIGNIFICANT",
    "ZERO_CORRECTION",
    "CellTest",
    "SignificanceSummary",
    "SignificanceVerdict",
    "adjust_p_values",
    "compare_two_samples",
    "mean_pnl_p_value",
    "significance_of_matrix",
    "win_rate_p_value",
]

#: The correction names ``config/model.yaml``'s ``evaluation.multiple_testing_correction``
#: may take. ``none`` is a legitimate choice - it just has to be a stated one.
CORRECTIONS: tuple[str, ...] = ("none", "bonferroni", "benjamini_hochberg")

#: The preferred spelling of "no correction at all".
ZERO_CORRECTION = "none"

#: Verdicts a cell can carry.
SIGNIFICANT = "SIGNIFICANT"
NOT_SIGNIFICANT = "NOT_SIGNIFICANT"
NOT_TESTABLE = "NOT_TESTABLE"
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


@dataclass(frozen=True, slots=True)
class SignificanceVerdict:
    """One cell's answer, with the numbers a reader has to see to weigh it."""

    verdict: str
    reason: str
    win_rate_p_value: float | None = None
    win_rate_p_adjusted: float | None = None
    pnl_p_value: float | None = None
    pnl_p_adjusted: float | None = None

    @property
    def tested(self) -> bool:
        """Was at least one test actually performed on this cell?"""
        return self.win_rate_p_value is not None or self.pnl_p_value is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "win_rate_p_value": self.win_rate_p_value,
            "win_rate_p_adjusted": self.win_rate_p_adjusted,
            "pnl_p_value": self.pnl_p_value,
            "pnl_p_adjusted": self.pnl_p_adjusted,
        }


def _wins_losses(trades: Sequence[Any]) -> tuple[int, int, list[float]]:
    """``(wins, losses, net_pnl)`` for a group - a flat trade counts in neither tail."""
    net = net_pnl_values(trades)
    wins = sum(1 for value in net if value > 0)
    losses = sum(1 for value in net if value < 0)
    return wins, losses, net


def win_rate_p_value(wins: int, losses: int, base_wins: int, base_losses: int) -> float | None:
    """Two-sided two-proportion z-test p-value, or ``None`` when it is undefined.

    Undefined means one of the two samples carries no decided outcome at all, or every
    decided outcome on both sides went the same way - in either case there is no difference
    left to measure, so no p-value should be printed for it.
    """
    total = wins + losses
    base_total = base_wins + base_losses
    if total <= 0 or base_total <= 0:
        return None
    pooled = (wins + base_wins) / (total + base_total)
    if pooled <= 0.0 or pooled >= 1.0:
        return None
    standard_error = float(math.sqrt(pooled * (1.0 - pooled) * (1.0 / total + 1.0 / base_total)))
    if standard_error <= 0.0:  # pragma: no cover - impossible while pooled is inside (0, 1)
        return None
    difference = wins / total - base_wins / base_total
    return float(2.0 * stats.norm.sf(abs(difference / standard_error)))


def mean_pnl_p_value(values: Sequence[float], baseline: Sequence[float]) -> float | None:
    """Two-sided Welch t-test p-value on the mean net P&L, or ``None`` when undefined.

    Undefined means either sample has fewer than two observations, or either sample has **zero
    spread**: Welch's test estimates each group's variance from that group's own trades, and a
    group whose P&L never varied provides no such estimate - a p-value from it would be an
    artefact of the fixture (and SciPy warns that the moment calculation lost precision).
    """
    first = np.asarray(list(values), dtype="float64")
    second = np.asarray(list(baseline), dtype="float64")
    if first.size < 2 or second.size < 2:
        return None
    if float(first.std(ddof=1)) == 0.0 or float(second.std(ddof=1)) == 0.0:
        return None
    result = stats.ttest_ind(first, second, equal_var=False)
    p_value = float(result.pvalue)
    if not math.isfinite(p_value):
        return None
    return p_value


def adjust_p_values(p_values: Sequence[float | None], method: str) -> list[float | None]:
    """Correct a family of p-values, leaving the ``None``\\ s (untested cells) untouched.

    ``benjamini_hochberg`` is the step-up procedure: sort ascending, scale the *i*-th
    smallest by ``n / i``, then enforce monotonicity by walking down from the largest, so an
    adjusted p-value can never be smaller than one it stands downstream of. ``bonferroni``
    multiplies by the family size and caps at 1. ``none`` returns the values unchanged.

    The family size is the number of *performed* tests - the honest denominator: a cell that
    was never tested must not inflate the correction for the ones that were.
    """
    if method not in CORRECTIONS:
        raise ValueError(f"unknown correction {method!r}; known corrections: {list(CORRECTIONS)}")
    if method == ZERO_CORRECTION:
        return list(p_values)
    indexes = [index for index, value in enumerate(p_values) if value is not None]
    family = len(indexes)
    if family == 0:
        return list(p_values)
    if method == "bonferroni":
        return [None if value is None else float(min(1.0, value * family)) for value in p_values]
    ordered = sorted(indexes, key=lambda index: float(p_values[index]))
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank, index in enumerate(reversed(ordered), start=1):
        position = family - rank + 1
        running = min(running, float(p_values[index]) * family / position)
        adjusted[index] = float(min(1.0, running))
    return [None if value is None else adjusted[index] for index, value in enumerate(p_values)]


def compare_two_samples(
    trades: Sequence[Any],
    baseline: Sequence[Any],
    *,
    preliminary_min_samples: int,
) -> SignificanceVerdict:
    """Test one cell against everything outside it, *before* any correction is applied."""
    minimum = int(preliminary_min_samples)
    if len(trades) < minimum or len(baseline) < minimum:
        return SignificanceVerdict(
            verdict=INSUFFICIENT_SAMPLE,
            reason=(
                f"{len(trades)} trade(s) in the cell and {len(baseline)} outside it, against a "
                f"minimum of {minimum}: not tested - any number printed for it would be noise"
            ),
        )
    wins, losses, values = _wins_losses(trades)
    base_wins, base_losses, base_values = _wins_losses(baseline)
    win_p = win_rate_p_value(wins, losses, base_wins, base_losses)
    pnl_p = mean_pnl_p_value(values, base_values)
    if win_p is None and pnl_p is None:
        return SignificanceVerdict(
            verdict=NOT_TESTABLE,
            reason=(
                "neither comparison is defined: the win rate has no variance left to test and "
                "the net P&L is constant on at least one side"
            ),
        )
    missing = []
    if win_p is None:
        missing.append("win rate (no variance to test)")
    if pnl_p is None:
        missing.append("net P&L (constant on one side)")
    return SignificanceVerdict(
        verdict=NOT_SIGNIFICANT,
        reason="tested; compare the adjusted p-values"
        + (f"; not testable: {', '.join(missing)}" if missing else ""),
        win_rate_p_value=win_p,
        pnl_p_value=pnl_p,
    )


# ---------------------------------------------------------------------------
# One matrix's family of tests
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CellTest:
    """One cell's test against the rest of the ledger, with both corrections applied."""

    matrix: str
    row_label: str
    column_label: str
    sample: int
    baseline_sample: int
    verdict: SignificanceVerdict

    @property
    def label(self) -> str:
        """``row | column`` - the same spelling a matrix uses for a preliminary cell."""
        return f"{self.row_label} | {self.column_label}"

    @property
    def preliminary(self) -> bool:
        """Is this cell below the sample rule (i.e. was it never tested)?"""
        return self.verdict.verdict == INSUFFICIENT_SAMPLE

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "matrix": self.matrix,
            "row": self.row_label,
            "column": self.column_label,
            "sample": self.sample,
            "baseline_sample": self.baseline_sample,
            "preliminary": self.preliminary,
            **self.verdict.to_dict(),
        }


@dataclass(slots=True)
class SignificanceSummary:
    """Every cell of one matrix, tested as one family and corrected together."""

    matrix: str
    correction: str
    significance_level: float
    preliminary_min_samples: int
    tests: list[CellTest] = field(default_factory=list)

    @property
    def family_size(self) -> int:
        """How many tests were actually performed (the correction's denominator)."""
        return sum(1 for test in self.tests if test.verdict.tested)

    @property
    def significant_cells(self) -> list[str]:
        """Labels whose adjusted win-rate or P&L p-value clears the level."""
        return [test.label for test in self.tests if test.verdict.verdict == SIGNIFICANT]

    @property
    def preliminary_cells(self) -> list[str]:
        """Labels that were never tested because their sample is too small."""
        return [test.label for test in self.tests if test.preliminary]

    @property
    def testable_cells(self) -> int:
        """How many cells could be tested at all - what the matrix's answer rests on."""
        return self.family_size

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "matrix": self.matrix,
            "correction": self.correction,
            "significance_level": self.significance_level,
            "preliminary_min_samples": self.preliminary_min_samples,
            "family_size": self.family_size,
            "significant_cells": self.significant_cells,
            "preliminary_cells": self.preliminary_cells,
            "tests": [test.to_dict() for test in self.tests],
        }




def significance_of_matrix(
    matrix: Any,
    trades: Sequence[Any],
    *,
    significance_level: float,
    correction: str,
    preliminary_min_samples: int | None = None,
) -> SignificanceSummary:
    """Test every cell of ``matrix`` against the rest of ``trades``, then correct the family.

    A cell's own trades come from the same labels the matrix was built with, and the baseline
    is *everything else in the ledger*, so the two samples are disjoint and together they are
    the whole ledger. An untested cell keeps ``None`` p-values rather than a placeholder, and
    the family size counts only the tests that actually ran.
    """
    if correction not in CORRECTIONS:
        raise ValueError(
            f"unknown correction {correction!r}; known corrections: {list(CORRECTIONS)}"
        )
    from core.backtest.metrics import SEGMENT_KEYS

    row_of = SEGMENT_KEYS[matrix.row_key][1]
    column_of = SEGMENT_KEYS[matrix.column_key][1]
    minimum = int(
        preliminary_min_samples
        if preliminary_min_samples is not None
        else matrix.preliminary_min_samples
    )
    tests: list[CellTest] = []
    for (row, column), cell in sorted(matrix.cells.items()):
        cell_trades = [
            trade for trade in trades if row_of(trade) == row and column_of(trade) == column
        ]
        baseline = [
            trade for trade in trades if not (row_of(trade) == row and column_of(trade) == column)
        ]
        tests.append(
            CellTest(
                matrix=matrix.name,
                row_label=cell.row_label,
                column_label=cell.column_label,
                sample=len(cell_trades),
                baseline_sample=len(baseline),
                verdict=compare_two_samples(
                    cell_trades, baseline, preliminary_min_samples=minimum
                ),
            )
        )
    _apply_corrections(tests, correction=correction, significance_level=float(significance_level))
    return SignificanceSummary(
        matrix=matrix.name,
        correction=correction,
        significance_level=float(significance_level),
        preliminary_min_samples=minimum,
        tests=tests,
    )



def _apply_corrections(
    tests: Sequence[CellTest], *, correction: str, significance_level: float
) -> None:
    """Adjust both p-value columns across the family and set the final verdicts.

    The correction runs over the whole family for each metric *separately*: a win-rate test
    and a P&L test answer different questions, and pooling them would silently double the
    family size for both. A cell becomes SIGNIFICANT when either adjusted p-value clears the
    level - the verdict is never weaker than the strongest thing it says - and an untested
    cell is left exactly as it was.
    """
    win_adjusted = adjust_p_values([test.verdict.win_rate_p_value for test in tests], correction)
    pnl_adjusted = adjust_p_values([test.verdict.pnl_p_value for test in tests], correction)
    for index, test in enumerate(tests):
        verdict = test.verdict
        if not verdict.tested:
            continue
        values = (win_adjusted[index], pnl_adjusted[index])
        significant = any(value is not None and value <= significance_level for value in values)
        test.verdict = replace(
            verdict,
            verdict=SIGNIFICANT if significant else NOT_SIGNIFICANT,
            reason=(
                f"adjusted p-value clears {significance_level:g}"
                if significant
                else f"neither adjusted p-value clears {significance_level:g}"
            ),
            win_rate_p_adjusted=win_adjusted[index],
            pnl_p_adjusted=pnl_adjusted[index],
        )
