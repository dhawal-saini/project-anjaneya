"""ANJANEYA-RESEARCH: the analysis that turns a trade ledger into spec 24's answers.

Spec 24 is called the project's *primary research output*, and it is a list of questions, not
a chart gallery: which time-of-day bucket pays, how that changes with the expiry-cycle phase,
what the weekday and days-to-expiry profiles look like, how time-of-day interacts with the
NIFTY regime, how long winners are held, how volatile each bucket is, what the average
intraday path looks like, whether any apparent edge survives a multiple-testing correction,
and which of those cells are too small to be read as anything but an observation.

The package is deliberately **offline and read-only**:

* :mod:`core.research.segments` builds the matrices, the per-segment distributions and the
  holding-time profile (items 1-6) from the *backtest ledger* - the same unfiltered rows
  ``--mode backtest`` writes, losers included.
* :mod:`core.research.significance` tests each cell against everything outside it (win rate
  and mean net P&L) and corrects the family of tests (item 9), refusing to test a cell whose
  sample is below ``evaluation.preliminary_min_samples`` (item 10).
* :mod:`core.research.profile` reads the *stored NIFTY bars* for the volatility profile by
  bucket and cycle phase (item 7) and the average cumulative intraday path per bucket
  (item 8) - a property of the market, not of a strategy.
* :mod:`core.research.report` assembles the ten answers into one :class:`ResearchReport` that
  renders to JSON and Markdown, labels every small cell PRELIMINARY, and says out loud which
  questions the stored data cannot answer yet.

Nothing here writes anything, and nothing here reaches the network. The app mode
(``--mode report``) owns the file I/O, exactly as ``--mode backtest`` owns the ledger's.
"""

from __future__ import annotations

from core.research.profile import (
    BucketProfile,
    BucketProfileTable,
    IntradayPath,
    IntradayPathPoint,
    ProfileError,
    bucket_order,
    build_bucket_profiles,
    build_intraday_paths,
)
from core.research.report import (
    ALL_ITEMS,
    PRELIMINARY_LABEL,
    RESEARCH_ITEMS,
    ReportPolicy,
    ResearchReport,
    build_research_report,
    render_markdown,
)
from core.research.segments import (
    MATRIX_SPECS,
    SIMPLE_SEGMENT_KEYS,
    HoldingTimeProfile,
    MatrixCell,
    PerformanceMatrix,
    ResearchError,
    build_matrices,
    build_matrix,
    holding_time_profile,
    ledger_trades,
    simple_segment_tables,
)
from core.research.significance import (
    CellTest,
    SignificanceSummary,
    SignificanceVerdict,
    adjust_p_values,
    compare_two_samples,
    significance_of_matrix,
)

__all__ = [
    "ALL_ITEMS",
    "MATRIX_SPECS",
    "PRELIMINARY_LABEL",
    "RESEARCH_ITEMS",
    "SIMPLE_SEGMENT_KEYS",
    "BucketProfile",
    "BucketProfileTable",
    "CellTest",
    "HoldingTimeProfile",
    "IntradayPath",
    "IntradayPathPoint",
    "MatrixCell",
    "PerformanceMatrix",
    "ProfileError",
    "ReportPolicy",
    "ResearchError",
    "ResearchReport",
    "SignificanceSummary",
    "SignificanceVerdict",
    "adjust_p_values",
    "bucket_order",
    "build_bucket_profiles",
    "build_intraday_paths",
    "build_matrices",
    "build_matrix",
    "build_research_report",
    "compare_two_samples",
    "holding_time_profile",
    "ledger_trades",
    "render_markdown",
    "significance_of_matrix",
    "simple_segment_tables",
]
