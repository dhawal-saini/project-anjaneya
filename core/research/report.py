"""Spec 24's ten answers, assembled into one report - numbers first, caveats attached.

Spec 24 lists ten things ANJANEYA-AI *must* produce, and this module is where they become one
object rather than ten files. Every answer is already computed by :mod:`core.research.segments`,
:mod:`core.research.significance` and :mod:`core.research.profile`; nothing is recomputed here.
What this module adds is the part a reader cannot get from any single table:

* **the ten items are named and always present.** A section that the stored data cannot
  answer (no index bars, no cell big enough to test) is present with its reason, so "there
  is nothing here" and "we did not look" can never look the same.
* **the sample rules travel with the report.** Every cell, row and point carries its sample
  and its PRELIMINARY flag, computed from one policy
  (``evaluation.preliminary_min_samples`` / ``evaluation.significance_level`` /
  ``evaluation.multiple_testing_correction``, and the ATR period from
  ``config/indicators.yaml``), so no number in the JSON can be quoted without the rule that
  produced it.
* **the caveats are part of the document.** The ledger's signal rule is a threshold, the
  costs are configuration, and a *seasonality* claim about weekdays or cycle phases needs
  ``data_gates.seasonality_min_days`` captured days. Those sentences are fields, not a
  README nobody reads, and ``to_markdown()`` prints them under the numbers.
* **the report never says "edge".** It says a cell's adjusted p-value cleared a level, with
  the family size beside it. Interpretation is the reader's job and stays theirs.

``RESEARCH_ITEMS`` is the spine: item name -> the attribute that answers it. Anything that
builds a report (the CLI, a test, a notebook) can therefore iterate the spec instead of
remembering it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from core.research.profile import (
    CYCLE_PHASE_ORDER,
    BucketProfileTable,
    IntradayPath,
    ProfileError,
    bucket_order,
    build_bucket_profiles,
    build_intraday_paths,
)
from core.research.segments import (
    HoldingTimeProfile,
    PerformanceMatrix,
    build_matrices,
    holding_time_profile,
    ledger_trades,
    simple_segment_tables,
)
from core.research.significance import SignificanceSummary, significance_of_matrix

__all__ = [
    "ALL_ITEMS",
    "PRELIMINARY_LABEL",
    "RESEARCH_ITEMS",
    "ReportPolicy",
    "ResearchReport",
    "build_research_report",
    "render_markdown",
]

#: The tag printed beside anything whose sample is below the configured minimum. It is the
#: same word the backtest report and the trainer use, so one rule has one name.
PRELIMINARY_LABEL = "PRELIMINARY"

#: Spec 24's ten deliverables: item -> (what it answers, the report attribute that carries it).
RESEARCH_ITEMS: dict[str, tuple[str, str]] = {
    "1_time_of_day_x_option_type": (
        "spec 24.1 - performance by time-of-day bucket x option side",
        "matrices",
    ),
    "2_time_of_day_x_cycle_phase": (
        "spec 24.2 - performance by time-of-day bucket x expiry-cycle phase",
        "matrices",
    ),
    "3_day_of_week": ("spec 24.3 - performance by day of week", "segment_tables"),
    "4_days_to_expiry": ("spec 24.4 - performance by days to expiry", "segment_tables"),
    "5_interactions": (
        "spec 24.5 - regime x time-of-day and regime x cycle phase",
        "matrices",
    ),
    "6_holding_time_by_bucket": (
        "spec 24.6 - holding-time distribution by bucket",
        "holding_time",
    ),
    "7_volatility_profile": (
        "spec 24.7 - ATR / range profile by bucket and cycle phase",
        "volatility",
    ),
    "8_intraday_path": (
        "spec 24.8 - average cumulative intraday return curve per bucket",
        "paths",
    ),
    "9_significance": (
        "spec 24.9 - significance of any apparent edge, corrected for multiple testing",
        "significance",
    ),
    "10_preliminary_labels": (
        "spec 24.10 - PRELIMINARY labels wherever the sample is too small",
        "preliminary_cells",
    ),
}

#: The same items in order, for a caller that wants to iterate them.
ALL_ITEMS: tuple[str, ...] = tuple(RESEARCH_ITEMS)


@dataclass(frozen=True, slots=True)
class ReportPolicy:
    """Every rule the report applies, read from the configuration in one place.

    The defaults mirror ``config/model.yaml`` and ``config/indicators.yaml``, so a caller that
    has no configuration (a test, a notebook) still gets the shipped numbers rather than a
    second, silently different set.
    """

    preliminary_min_samples: int = 30
    significance_level: float = 0.05
    multiple_testing_correction: str = "benjamini_hochberg"
    seasonality_min_days: int = 250
    profile_min_sessions: int = 2
    atr_period: int = 14
    atr_method: str = "wilder"

    @classmethod
    def from_config(cls, config: Any) -> ReportPolicy:
        """Read the policy out of a loaded :class:`core.config_loader.AnjaneyaConfig`.

        The sample rule and the correction come from ``model.evaluation``, the seasonality
        horizon from ``model.data_gates`` and the ATR definition from ``indicators.atr`` -
        the same three places the backtest and the trainer read, so one rule has one source.
        """
        evaluation = config.model.evaluation
        return cls(
            preliminary_min_samples=int(evaluation.preliminary_min_samples),
            significance_level=float(evaluation.significance_level),
            multiple_testing_correction=str(evaluation.multiple_testing_correction),
            seasonality_min_days=int(config.model.data_gates.seasonality_min_days),
            atr_period=int(config.indicators.atr.period),
            atr_method=str(config.indicators.atr.method),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every report so a number stays re-derivable."""
        return {
            "preliminary_min_samples": self.preliminary_min_samples,
            "significance_level": self.significance_level,
            "multiple_testing_correction": self.multiple_testing_correction,
            "seasonality_min_days": self.seasonality_min_days,
            "profile_min_sessions": self.profile_min_sessions,
            "atr_period": self.atr_period,
            "atr_method": self.atr_method,
        }


@dataclass(slots=True)
class ResearchReport:
    """Spec 24's ten answers over one ledger, with the samples and caveats attached."""

    policy: ReportPolicy
    ledger: str
    trades: int
    days: int
    first_entry: Any
    last_exit: Any
    matrices: list[PerformanceMatrix] = field(default_factory=list)
    segment_tables: list[Any] = field(default_factory=list)
    holding_time: HoldingTimeProfile | None = None
    significance: list[SignificanceSummary] = field(default_factory=list)
    volatility: BucketProfileTable | None = None
    volatility_unavailable: str | None = None
    paths: list[IntradayPath] = field(default_factory=list)
    paths_unavailable: str | None = None
    gate: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)

    # -- the ten items ------------------------------------------------------
    @property
    def preliminary_cells(self) -> dict[str, list[str]]:
        """Item 10: every PRELIMINARY label, by the table that carries it."""
        tables: dict[str, list[str]] = {}
        for matrix in self.matrices:
            if matrix.preliminary_cells:
                tables[matrix.name] = [
                    f"{row} | {column}" for row, column in matrix.preliminary_cells
                ]
        for table in self.segment_tables:
            if table.preliminary_rows:
                tables[table.name] = list(table.preliminary_rows)
        if self.holding_time is not None and self.holding_time.preliminary_labels:
            tables[self.holding_time.name] = list(self.holding_time.preliminary_labels)
        if self.volatility is not None and self.volatility.preliminary_cells:
            tables["volatility_profile"] = list(self.volatility.preliminary_cells)
        for path in self.paths:
            if path.preliminary:
                tables.setdefault("intraday_path", []).append(path.bucket)
        return tables

    @property
    def significant_cells(self) -> dict[str, list[str]]:
        """Item 9's headline: which cells cleared the level after correction, by matrix."""
        return {
            summary.matrix: summary.significant_cells
            for summary in self.significance
            if summary.significant_cells
        }

    @property
    def testable_cells(self) -> int:
        """How many cells were actually tested across every matrix."""
        return sum(summary.family_size for summary in self.significance)

    def item(self, name: str) -> Any:
        """The payload of one of spec 24's ten items, by the name in :data:`RESEARCH_ITEMS`."""
        if name not in RESEARCH_ITEMS:
            raise KeyError(f"unknown research item {name!r}; known items: {list(RESEARCH_ITEMS)}")
        return getattr(self, RESEARCH_ITEMS[name][1])

    def to_markdown(self) -> str:
        """The report as a Markdown document - the shape a human reads first."""
        return render_markdown(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - the whole report, samples and caveats included."""
        return {
            "report": "spec 24 - time-of-day / expiry-cycle analysis",
            "ledger": self.ledger,
            "policy": self.policy.to_dict(),
            "trades": self.trades,
            "days": self.days,
            "first_entry": _stamp(self.first_entry),
            "last_exit": _stamp(self.last_exit),
            "gate": self.gate,
            "items": {name: description for name, (description, _) in RESEARCH_ITEMS.items()},
            "matrices": [matrix.to_dict() for matrix in self.matrices],
            "segment_tables": [table.to_dict() for table in self.segment_tables],
            "holding_time": self.holding_time.to_dict() if self.holding_time else None,
            "significance": [summary.to_dict() for summary in self.significance],
            "significant_cells": self.significant_cells,
            "testable_cells": self.testable_cells,
            "volatility": self.volatility.to_dict() if self.volatility else None,
            "volatility_unavailable": self.volatility_unavailable,
            "intraday_paths": [path.to_dict() for path in self.paths],
            "intraday_paths_unavailable": self.paths_unavailable,
            "preliminary_cells": self.preliminary_cells,
            "caveats": list(self.caveats),
        }


def _stamp(value: Any) -> str | None:
    """An ISO timestamp (or date) for the JSON, or ``None`` when there is nothing to print."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return None if pd.isna(stamp) else stamp.isoformat()


# ---------------------------------------------------------------------------
# Building the report
# ---------------------------------------------------------------------------
def _entry_days(trades: Sequence[Any]) -> list[date]:
    """The distinct exchange-local dates on which the ledger's trades opened, in order."""
    days = set()
    for trade in trades:
        stamp = trade.entry_timestamp
        if stamp is None:
            continue
        moment = pd.Timestamp(stamp)
        if not pd.isna(moment):
            days.add(moment.date())
    return sorted(days)


def _window(trades: Sequence[Any]) -> tuple[Any, Any]:
    """``(first entry, last exit)`` - the span the report is allowed to talk about."""
    entries = [
        pd.Timestamp(trade.entry_timestamp) for trade in trades if trade.entry_timestamp is not None
    ]
    exits = [pd.Timestamp(trade.exit_timestamp) for trade in trades if trade.exit_timestamp]
    return (min(entries) if entries else None, max(exits) if exits else None)


def build_research_report(
    ledger: Any,
    *,
    policy: ReportPolicy,
    calendar: Any | None = None,
    bars: Any | None = None,
    ledger_name: str = "in-memory",
    gate: Any | None = None,
    captured_days: int | None = None,
) -> ResearchReport:
    """Assemble spec 24's ten answers over one ledger (optionally plus the index bars).

    ``bars`` are the stored index bars that the volatility profile (item 7) and the intraday
    paths (item 8) need; ``bars=None`` does not quietly drop those two items - it records why
    they are absent (``volatility_unavailable`` / ``paths_unavailable``), which is the
    difference between "not measured" and "measured as zero". A profile that cannot be built
    from the bars that were handed over is caught the same way: the reason is carried, not
    raised, because eight good answers beat none.

    ``gate`` is the data-accumulation status (any object with ``to_dict()`` and
    ``describe()``) and ``captured_days`` is how many sessions exist overall. Both decide
    *when* a claim may be made, and both end up in the caveats rather than being left for the
    reader to remember.
    """
    trades = ledger_trades(ledger)
    orders = {
        "expiry_cycle_phase": list(CYCLE_PHASE_ORDER),
        "time_of_day_bucket": bucket_order(calendar) if calendar is not None else None,
    }
    matrices = build_matrices(
        trades,
        preliminary_min_samples=policy.preliminary_min_samples,
        orders=orders,
    )
    report = ResearchReport(
        policy=policy,
        ledger=str(ledger_name),
        trades=len(trades),
        days=len(_entry_days(trades)),
        first_entry=_window(trades)[0],
        last_exit=_window(trades)[1],
        matrices=matrices,
        segment_tables=simple_segment_tables(
            trades, preliminary_min_samples=policy.preliminary_min_samples
        ),
        holding_time=holding_time_profile(
            trades,
            preliminary_min_samples=policy.preliminary_min_samples,
            order=orders["time_of_day_bucket"],
        ),
        significance=[
            significance_of_matrix(
                matrix,
                trades,
                significance_level=policy.significance_level,
                correction=policy.multiple_testing_correction,
                preliminary_min_samples=policy.preliminary_min_samples,
            )
            for matrix in matrices
        ],
    )
    report.gate = gate.to_dict() if gate is not None else None
    if bars is not None and calendar is not None:
        report.volatility, report.volatility_unavailable = _profiles(bars, calendar, policy)
        report.paths, report.paths_unavailable = _paths(bars, calendar, policy)
    else:
        reason = (
            "no stored index bars were supplied to the report, so the market's own profile "
            "(items 7 and 8) was not measured - it is not zero, it is absent"
        )
        report.volatility_unavailable = reason
        report.paths_unavailable = reason
    report.caveats = _caveats(report, captured_days=captured_days)
    return report



def _profiles(
    bars: Any, calendar: Any, policy: ReportPolicy
) -> tuple[BucketProfileTable | None, str | None]:
    """Item 7, with a failure turned into a stated reason rather than an exception."""
    try:
        table = build_bucket_profiles(
            bars,
            calendar,
            atr_period=policy.atr_period,
            atr_method=policy.atr_method,
            min_sessions=policy.profile_min_sessions,
        )
    except ProfileError as exc:
        return None, f"the volatility profile could not be built: {exc}"
    return table, None


def _paths(
    bars: Any, calendar: Any, policy: ReportPolicy
) -> tuple[list[IntradayPath], str | None]:
    """Item 8, with a failure turned into a stated reason rather than an exception."""
    try:
        paths = build_intraday_paths(bars, calendar, min_sessions=policy.profile_min_sessions)
    except ProfileError as exc:
        return [], f"the intraday paths could not be built: {exc}"
    return paths, None


def _caveats(report: ResearchReport, *, captured_days: int | None) -> list[str]:
    """The sentences that must travel with these numbers, in the order they matter."""
    caveats = [
        "the ledger's entries come from a probability threshold, not from spec 24's decision "
        "layer (--mode decide, which is a separate, read-only record): these are costs and "
        "shapes, not a claimed edge",
        "costs are configuration (config/model.yaml's costs: block), not truth - verify the "
        "statutory rates against a current contract note before quoting any P&L",
        f"a cell is read as an observation below {report.policy.preliminary_min_samples} "
        "trades on either side of the comparison, and is never tested",
    ]
    if report.testable_cells == 0:
        caveats.append(
            "no cell of any matrix had enough trades on both sides to be tested, so item 9 has "
            "no p-value to report at all"
        )
    else:
        caveats.append(
            f"{report.testable_cells} cell(s) were tested, each matrix corrected as one "
            f"{report.policy.multiple_testing_correction} family at "
            f"alpha={report.policy.significance_level:g}; a cell that clears the level is still "
            "one cell of one ledger"
        )
    if captured_days is None:
        caveats.append(
            "the number of captured days was not supplied, so the seasonality gate "
            f"({report.policy.seasonality_min_days} days) could not be checked here"
        )
    elif captured_days < report.policy.seasonality_min_days:
        caveats.append(
            f"{captured_days} captured day(s) against {report.policy.seasonality_min_days} for a "
            "seasonality claim: the weekday and cycle-phase tables are PRELIMINARY by the "
            "project's own accumulation rule, however many trades they contain"
        )
    if report.days < 2:
        caveats.append(
            f"the ledger covers {report.days} trading day(s) of entries, so every distribution "
            "here is one session's shape"
        )
    return caveats



# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _number(value: Any, digits: int = 2) -> str:
    """A number for a table cell, or ``n/a`` when there is honestly nothing to print."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """A GitHub-flavoured Markdown table, or a line saying there are no rows."""
    if not rows:
        return ["_nothing to show_"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _matrix_rows(matrix: PerformanceMatrix) -> list[list[str]]:
    """One Markdown row per matrix row, a cell per column: net P&L, sample, PRELIMINARY flag."""
    rows = []
    for row_label in matrix.row_labels:
        cells = []
        for column_label in matrix.column_labels:
            cell = matrix.cell(row_label, column_label)
            if cell is None:
                cells.append("-")
                continue
            flag = f" **{PRELIMINARY_LABEL}**" if cell.preliminary else ""
            cells.append(f"{_number(cell.metrics.total_net_pnl)} ({cell.sample}){flag}")
        rows.append([row_label, *cells])
    return rows


def _segment_rows(table: Any) -> list[list[str]]:
    """One Markdown row per segment of a ``SegmentTable`` (spec 25's cut, spec 24's labels)."""
    rows = []
    for row in table.rows:
        flag = PRELIMINARY_LABEL if row.preliminary else ""
        rows.append(
            [
                row.label,
                row.sample,
                _number(row.metrics.total_net_pnl),
                _number(row.metrics.win_rate, 3),
                _number(row.metrics.average_holding_minutes, 1),
                flag,
            ]
        )
    return rows


def _significance_rows(summary: SignificanceSummary) -> list[list[str]]:
    """One Markdown row per tested cell: both raw p-values, both corrected ones, the verdict."""
    rows = []
    for test in summary.tests:
        verdict = test.verdict
        rows.append(
            [
                test.label,
                test.sample,
                test.baseline_sample,
                _number(verdict.win_rate_p_value, 4),
                _number(verdict.win_rate_p_adjusted, 4),
                _number(verdict.pnl_p_value, 4),
                _number(verdict.pnl_p_adjusted, 4),
                verdict.verdict,
            ]
        )
    return rows



def render_markdown(report: ResearchReport) -> str:
    """The whole report as Markdown: the ten items in spec order, caveats last.

    The layout is deliberate - every table carries its sample in the same row as the number,
    and every table that can contain a small cell says so in the cell itself, so a screenshot
    of one table cannot lose the warning that belongs to it.
    """
    policy = report.policy
    lines: list[str] = [
        "# ANJANEYA research report - time-of-day / expiry-cycle analysis (spec 24)",
        "",
        f"**Ledger:** `{report.ledger}`  ",
        f"**Trades:** {report.trades} closed  ",
        f"**Entry days:** {report.days}  ",
        f"**Window:** {_stamp(report.first_entry) or 'n/a'} .. {_stamp(report.last_exit) or 'n/a'}  ",
        f"**Sample rule:** {policy.preliminary_min_samples} trades (below that a cell is "
        f"{PRELIMINARY_LABEL} and is never tested)  ",
        f"**Significance:** {policy.multiple_testing_correction} at alpha={policy.significance_level:g}  ",
        f"**ATR:** period {policy.atr_period}, method {policy.atr_method}  ",
    ]
    if report.gate:
        lines.append(f"**Data gate:** {report.gate.get('stage')} - {report.gate.get('allowed')}  ")
    lines.append("")

    by_name = {matrix.name: matrix for matrix in report.matrices}
    for name, (description, _) in RESEARCH_ITEMS.items():
        lines.extend([f"## {name}. {description}", ""])
        if name == "1_time_of_day_x_option_type":
            lines.extend(_matrix_section(by_name["time_of_day_x_option_type"]))
        elif name == "2_time_of_day_x_cycle_phase":
            lines.extend(_matrix_section(by_name["time_of_day_x_cycle_phase"]))
        elif name == "3_day_of_week":
            lines.extend(_segment_section(report, "day_of_week"))
        elif name == "4_days_to_expiry":
            lines.extend(_segment_section(report, "days_to_expiry"))
        elif name == "5_interactions":
            lines.extend(_matrix_section(by_name["regime_x_time_of_day"]))
            lines.append("")
            lines.extend(_matrix_section(by_name["regime_x_cycle_phase"]))
        elif name == "6_holding_time_by_bucket":
            lines.extend(_holding_section(report))
        elif name == "7_volatility_profile":
            lines.extend(_volatility_section(report))
        elif name == "8_intraday_path":
            lines.extend(_path_section(report))
        elif name == "9_significance":
            lines.extend(_significance_section(report))
        else:
            lines.extend(_preliminary_section(report))
        lines.append("")

    lines.extend(["## Caveats", ""])
    lines.extend(f"* {caveat}" for caveat in report.caveats)
    lines.append("")
    return "\n".join(lines)



def _matrix_section(matrix: PerformanceMatrix) -> list[str]:
    """One matrix as a Markdown table with ``net P&L (sample)`` cells."""
    headers = [matrix.row_key, *matrix.column_labels]
    rows = _matrix_rows(matrix)
    return [
        f"_{matrix.description}_",
        "",
        *(
            _table(headers, rows)
            if rows
            else ["_no trades in this cut - the cells are absent, not zero_"]
        ),
        "",
        f"Cells below {matrix.preliminary_min_samples} trades are marked "
        f"**{PRELIMINARY_LABEL}**; a dash means nothing traded in that intersection.",
    ]


def _segment_section(report: ResearchReport, name: str) -> list[str]:
    """Spec 24.3 / 24.4: one single-axis cut, with its samples."""
    table = next((item for item in report.segment_tables if item.name == name), None)
    if table is None:
        return ["_this cut was not built_"]
    headers = ["segment", "sample", "net P&L", "win rate", "avg holding (min)", ""]
    return [
        f"_{table.description}_",
        "",
        *_table(headers, _segment_rows(table)),
    ]


def _holding_section(report: ResearchReport) -> list[str]:
    """Spec 24.6: the holding-time distribution per bucket."""
    profile = report.holding_time
    if profile is None or not profile.rows:
        return ["_the holding-time profile is empty - no closed trades in the ledger_"]
    headers = ["bucket", "sample", "mean (min)", "p50", "min", "max", "net P&L", ""]
    rows = [
        [
            row.label,
            row.sample,
            _number(row.distribution.mean, 1),
            _number(row.distribution.percentiles.get("p50"), 1),
            _number(row.distribution.minimum, 1),
            _number(row.distribution.maximum, 1),
            _number(row.metrics.total_net_pnl),
            PRELIMINARY_LABEL if row.preliminary else "",
        ]
        for row in profile.rows
    ]
    return [f"_{profile.description}_", "", *_table(headers, rows)]


def _volatility_section(report: ResearchReport) -> list[str]:
    """Spec 24.7: ATR and range per bucket x cycle phase, from the stored index bars."""
    table = report.volatility
    if table is None:
        return [f"_not measured: {report.volatility_unavailable or 'no index bars'}_"]
    headers = ["bucket", "phase", "sessions", "bars", "mean ATR", "mean range", "range %", ""]
    rows = [
        [
            profile.bucket,
            profile.cycle_phase,
            profile.sessions,
            profile.bars,
            _number(profile.atr_mean),
            _number(profile.range_mean),
            _number(profile.range_mean_pct, 4),
            PRELIMINARY_LABEL if profile.preliminary else "",
        ]
        for profile in table.profiles
    ]
    return [
        f"_{table.symbol}: {table.bars} bar(s) over {table.sessions} session(s); "
        f"{table.dropped_out_of_session} bar(s) fell outside every configured bucket and were "
        "counted, not folded in._",
        "",
        *_table(headers, rows),
    ]


def _path_section(report: ResearchReport) -> list[str]:
    """Spec 24.8: the average cumulative return path that starts at each bucket."""
    if not report.paths:
        return [f"_not measured: {report.paths_unavailable or 'no index bars'}_"]
    lines: list[str] = []
    for path in report.paths:
        flag = f" **{PRELIMINARY_LABEL}**" if path.preliminary else ""
        lines.append(
            f"**{path.bucket}** - filled at the open of the {path.base_minute} bar, "
            f"{path.sessions} session(s){flag}"
        )
        lines.append("")
        rows = [
            [point.minute, _number(point.mean_return_pct, 3), _number(point.stdev_return_pct, 3),
             point.sample, PRELIMINARY_LABEL if point.preliminary else ""]
            for point in path.points
        ]
        lines.extend(_table(["minute", "mean %", "stdev %", "sessions", ""], rows))
        lines.append("")
    return lines


def _significance_section(report: ResearchReport) -> list[str]:
    """Spec 24.9: every test, both p-values, both corrections."""
    if not report.significance:
        return ["_no matrix was tested_"]
    lines: list[str] = []
    for summary in report.significance:
        lines.append(
            f"**{summary.matrix}** - {summary.family_size} test(s) performed, corrected as one "
            f"{summary.correction} family at alpha={summary.significance_level:g}"
        )
        lines.append("")
        if summary.family_size == 0:
            lines.append(
                f"_every cell was below {summary.preliminary_min_samples} trades on one side of "
                "the comparison, so nothing was tested_"
            )
            lines.append("")
            continue
        headers = ["cell", "n", "rest", "p (win)", "p adj", "p (P&L)", "p adj", "verdict"]
        lines.extend(_table(headers, _significance_rows(summary)))
        lines.append("")
    return lines


def _preliminary_section(report: ResearchReport) -> list[str]:
    """Spec 24.10: every label that must not be read as a finding yet."""
    tables = report.preliminary_cells
    if not tables:
        return [
            f"_no cell, row or path fell below the sample rule "
            f"({report.policy.preliminary_min_samples} trades) - every number above has a "
            "sample behind it_"
        ]
    lines = [
        f"Everything below is marked **{PRELIMINARY_LABEL}**: its sample is below the "
        f"configured minimum, so it is an observation and not a finding.",
        "",
    ]
    for name, labels in tables.items():
        lines.append(f"* **{name}** ({len(labels)}): {', '.join(labels)}")
    return lines

