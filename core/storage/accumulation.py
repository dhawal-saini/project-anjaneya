"""Read side of spec 29's accumulation report: the stored facts, measured once.

:mod:`core.accumulation` is the *model* - it decides what a number means and how it is
printed. This module is the *measurement*: it turns the DuckDB tables into the plain
:class:`core.accumulation.AccumulationInputs` container the model consumes, so no section of a
report can re-read the database and quietly disagree with another.

Three deliberate choices:

* **Only the tables a weekly report needs are read.** ``TIMESTAMP_COLUMNS`` fixes which column
  each table's span and day count come from, so "rows over days" always means the same thing:
  the row count, the number of distinct calendar days its timestamp spans, and that span.
* **A table that cannot be read is reported, never skipped.** An old install without a
  ``features`` table is a fact worth printing; :func:`read_table_profile` returns a profile
  carrying the reason instead of raising or pretending the table is empty.
* **One query builds the coverage days.** The ledger is grouped by
  ``(trading_date, quality_flag)`` and assembled in Python, which yields both the per-day
  verdict and the database-wide quality-flag histogram from a single pass.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from core.accumulation import (
    WEEK_DAYS,
    AccumulationInputs,
    CoverageDay,
    TableProfile,
    WeekCapture,
)
from core.storage.writers import quote_identifier

__all__ = [
    "ACCUMULATION_TABLES",
    "TIMESTAMP_COLUMNS",
    "read_accumulation_inputs",
    "read_coverage_days",
    "read_table_profile",
    "read_table_profiles",
    "read_weekly_capture",
]

#: table -> the column its span and day count are measured from (``None`` = count only).
#: The order is the order the report prints them: raw capture first, derived rows next, the
#: ledger and the run history last.
TIMESTAMP_COLUMNS: dict[str, str | None] = {
    "instruments": None,
    "ticks": "timestamp",
    "option_chain_snapshots": "timestamp",
    "index_history": "timestamp",
    "candles": "timestamp",
    "features": "timestamp",
    "labels": "timestamp",
    "trades": "entry_timestamp",
    "data_coverage": "trading_date",
    "run_manifest": "started_at",
}

#: Every table the report profiles, in print order.
ACCUMULATION_TABLES: tuple[str, ...] = tuple(TIMESTAMP_COLUMNS)


def read_table_profile(connection: Any, table: str, column: str | None = None) -> TableProfile:
    """One table's row count, spanned-day count and span - or why it could not be read."""
    identifier = quote_identifier(table)
    try:
        if column is None:
            row = connection.execute(f"SELECT count(*) FROM {identifier}").fetchone()
            return TableProfile(table=table, rows=int(row[0] or 0))
        stamp = quote_identifier(column)
        row = connection.execute(
            f"SELECT count(*), count(DISTINCT CAST({stamp} AS DATE)), min({stamp}), max({stamp}) "
            f"FROM {identifier}"
        ).fetchone()
    except Exception as exc:  # noqa: BLE001 - a missing table is reported, not raised
        return TableProfile(table=table, unavailable=f"{type(exc).__name__}: {exc}")
    return TableProfile(
        table=table,
        rows=int(row[0] or 0),
        days=int(row[1] or 0),
        first=row[2],
        last=row[3],
    )


def read_table_profiles(
    connection: Any, *, tables: Sequence[str] | None = None
) -> list[TableProfile]:
    """Profile every configured table (in :data:`TIMESTAMP_COLUMNS` order)."""
    names = list(tables) if tables is not None else list(ACCUMULATION_TABLES)
    unknown = [name for name in names if name not in TIMESTAMP_COLUMNS]
    if unknown:
        raise ValueError(
            f"unknown table(s) {unknown} for the accumulation report; known: "
            f"{list(TIMESTAMP_COLUMNS)}"
        )
    return [read_table_profile(connection, name, TIMESTAMP_COLUMNS[name]) for name in names]


def read_coverage_days(connection: Any) -> list[CoverageDay]:
    """Collapse the coverage ledger into one :class:`CoverageDay` per trading day.

    Grouping by ``(trading_date, quality_flag)`` in one pass gives two things at once: the
    day's per-flag counts (kept on the day, so a caller can see which flags a day carried) and,
    summed across days, the database-wide quality-flag histogram the report prints.

    A ledger that cannot be read (no table, a locked database) yields ``[]`` - the caller
    already learns the table is unavailable from its :class:`TableProfile`, and a report must
    never fail because one of its inputs is missing.
    """
    try:
        rows = connection.execute(
            "SELECT trading_date, quality_flag, count(*), "
            "sum(CASE WHEN gap_count > 0 THEN 1 ELSE 0 END), "
            "sum(coalesce(gap_count, 0)), max(coalesce(largest_gap_seconds, 0)), "
            "min(coverage_ratio) "
            "FROM data_coverage GROUP BY trading_date, quality_flag ORDER BY trading_date"
        ).fetchall()
    except Exception:  # noqa: BLE001 - the caller reports the missing ledger
        return []

    days: dict[date, dict[str, Any]] = {}
    for trading_date, flag, streams, gap_streams, gap_count, largest_gap, min_ratio in rows:
        entry = days.setdefault(
            trading_date,
            {
                "streams": 0,
                "complete": 0,
                "gap_streams": 0,
                "missing": 0,
                "suspect": 0,
                "gap_count": 0,
                "largest_gap": 0.0,
                "min_ratio": None,
                "flags": {},
            },
        )
        count = int(streams or 0)
        name = str(flag)
        entry["streams"] += count
        entry["flags"][name] = entry["flags"].get(name, 0) + count
        entry["gap_streams"] += int(gap_streams or 0)
        entry["gap_count"] += int(gap_count or 0)
        entry["largest_gap"] = max(entry["largest_gap"], float(largest_gap or 0.0))
        if name == "OK":
            entry["complete"] += count
        elif name == "MISSING_TICK":
            entry["missing"] += count
        elif name == "SUSPECT":
            entry["suspect"] += count
        if min_ratio is not None:
            ratio = float(min_ratio)
            entry["min_ratio"] = ratio if entry["min_ratio"] is None else min(entry["min_ratio"], ratio)

    return [
        CoverageDay(
            trading_date=trading_date,
            streams=entry["streams"],
            complete_streams=entry["complete"],
            gap_streams=entry["gap_streams"],
            missing_streams=entry["missing"],
            suspect_streams=entry["suspect"],
            min_ratio=entry["min_ratio"],
            gap_count=entry["gap_count"],
            largest_gap_seconds=round(entry["largest_gap"], 3),
            quality_flags=entry["flags"],
        )
        for trading_date, entry in sorted(days.items())
    ]


def read_weekly_capture(
    connection: Any, *, as_of: date, weeks: int
) -> list[WeekCapture]:
    """Bucket the captured trading days into the last ``weeks`` weeks ending on ``as_of``.

    Weeks are aligned to ``as_of`` (the report's day) rather than to ISO weeks: bucket ``i``
    covers ``[as_of - 7i - 6, as_of - 7i]``, so "this week" always ends on the day the report
    is read. Reading the distinct dates and bucketing in Python keeps the arithmetic identical
    to the window the report prints.
    """
    weeks = max(1, int(weeks))
    try:
        rows = connection.execute("SELECT DISTINCT trading_date FROM data_coverage").fetchall()
    except Exception:  # noqa: BLE001 - the caller reports the missing ledger
        return []
    captured = {row[0] for row in rows if row[0] is not None}

    buckets: list[WeekCapture] = []
    for index in range(weeks):
        end = as_of - timedelta(days=index * WEEK_DAYS)
        start = end - timedelta(days=WEEK_DAYS - 1)
        days = sum(1 for moment in captured if start <= moment <= end)
        buckets.append(
            WeekCapture(label=f"{start.isoformat()}..{end.isoformat()}", start=start, end=end, days=days)
        )
    buckets.reverse()
    return buckets


def read_accumulation_inputs(
    connection: Any,
    *,
    as_of: date,
    weeks: int,
    tables: Sequence[str] | None = None,
) -> AccumulationInputs:
    """Measure everything spec 29's report needs, in one call.

    The newest trained model is deliberately *not* read here: it lives on the filesystem, not
    in DuckDB, so the app mode attaches it (``inputs.newest_model``) after this returns. That
    keeps the storage layer to SQL and the filesystem knowledge in the app.
    """
    profiles = read_table_profiles(connection, tables=tables)
    coverage_days = read_coverage_days(connection)
    quality_counts: dict[str, int] = {}
    for day in coverage_days:
        for name, count in day.quality_flags.items():
            quality_counts[name] = quality_counts.get(name, 0) + count
    weekly = read_weekly_capture(connection, as_of=as_of, weeks=weeks)
    return AccumulationInputs(
        tables=profiles,
        coverage_days=coverage_days,
        quality_counts=quality_counts,
        weekly=weekly,
    )

