"""Per-stream coverage accounting, shared by ANJANEYA-CAPTURE and -ENGINE.

Why this lives in ``core`` rather than in the capture app
---------------------------------------------------------
Two applications ask the identical question - *"how much of this stream did we
actually record?"* - and an operator must never see two different verdicts for
the same fact. Capture feeds this machinery ticks (whose expected volume is not
predictable), the engine feeds it candles (whose expected count is the session's
bar count). So the vocabulary is shared:

* :class:`StreamStats` accumulates one stream's first/last timestamp, row count,
  gap count and per-quality-flag counts;
* :class:`CoverageTracker` groups those stats per ``(tradingsymbol, timeframe)``
  and detects silences longer than ``gap_seconds`` *while the market is open*;
* :func:`build_coverage_row` turns one stream's stats into the ``data_coverage``
  row that is written to DuckDB (spec 4.3 / 6.5).

A gap is measured in *time*, not in rows, because that is the only definition
that works for both a tick stream and a candle stream: 30 quiet seconds and one
missing 1m bar are the same kind of hole. ``expected_rows`` sharpens the verdict
when the caller knows the bar count (the engine does, capture does not).

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

from core.schemas import CoverageRow, QualityFlag

__all__ = ["CoverageTracker", "StreamStats", "build_coverage_row"]


@dataclass(slots=True)
class StreamStats:
    """Per-stream counters used to build coverage rows (spec 4.3 / 6.5)."""

    tradingsymbol: str
    timeframe: str
    rows: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    gap_count: int = 0
    largest_gap_seconds: float = 0.0
    expected_rows: int | None = None
    quality_flags: dict[str, int] = field(default_factory=dict)

    def record_flag(self, flag: QualityFlag | str) -> None:
        """Count a row's quality flag."""
        name = flag.value if isinstance(flag, QualityFlag) else str(flag)
        self.quality_flags[name] = self.quality_flags.get(name, 0) + 1

    @property
    def coverage_ratio(self) -> float | None:
        """``rows / expected_rows`` when an expectation exists, else ``None``."""
        if not self.expected_rows:
            return None
        return min(1.0, self.rows / self.expected_rows)

    @property
    def quality_flag(self) -> QualityFlag:
        """Verdict for this stream.

        * no rows at all          -> ``MISSING_TICK`` (nothing was captured),
        * gaps beyond tolerance   -> ``GAP_FILLED``,
        * any SUSPECT/STALE row   -> ``SUSPECT``,
        * otherwise outside RTH   -> ``OUT_OF_SESSION``,
        * otherwise               -> ``OK``.
        """
        if self.rows == 0:
            return QualityFlag.MISSING_TICK
        if self.gap_count > 0:
            return QualityFlag.GAP_FILLED
        for name in (QualityFlag.SUSPECT.value, QualityFlag.STALE.value):
            if self.quality_flags.get(name):
                return QualityFlag.SUSPECT
        if self.quality_flags.get(QualityFlag.OUT_OF_SESSION.value):
            return QualityFlag.OUT_OF_SESSION
        return QualityFlag.OK

    def note(self) -> str:
        """Human-readable breakdown appended to the coverage row."""
        if self.expected_rows is not None:
            parts = [f"{self.rows}/{self.expected_rows} expected rows"]
        else:
            parts = [f"{self.rows} rows"]
        if self.gap_count:
            parts.append(f"{self.gap_count} gap(s), largest {self.largest_gap_seconds:.1f}s")
        flagged = {
            name: count for name, count in self.quality_flags.items() if name != QualityFlag.OK.value
        }
        if flagged:
            parts.append(
                "flags: " + ", ".join(f"{name}={count}" for name, count in sorted(flagged.items()))
            )
        return "; ".join(parts)


class CoverageTracker:
    """Accumulates per-stream statistics, counts gaps and builds coverage rows.

    A stream has no fixed row count, so coverage cannot be "expected vs actual
    rows" alone. Every stream therefore records its first/last timestamp and
    counts silences longer than ``gap_seconds`` *while the market is open* -
    exactly what an operator needs to see afterwards.
    """

    def __init__(
        self, *, gap_seconds: float, is_open: Callable[[datetime], bool] | None = None
    ) -> None:
        self.gap_seconds = gap_seconds
        self._is_open = is_open or (lambda moment: True)
        self._streams: dict[tuple[str, str], StreamStats] = {}

    def stream(self, tradingsymbol: str, timeframe: str) -> StreamStats:
        """Fetch (creating if needed) the stats object for one stream."""
        key = (tradingsymbol, timeframe)
        stats = self._streams.get(key)
        if stats is None:
            stats = StreamStats(tradingsymbol=tradingsymbol, timeframe=timeframe)
            self._streams[key] = stats
        return stats

    def record(
        self,
        *,
        tradingsymbol: str,
        timeframe: str,
        timestamp: datetime,
        quality_flag: QualityFlag | str = QualityFlag.OK,
        expected_rows: int | None = None,
    ) -> bool:
        """Record one row; returns ``True`` when this row closed a gap."""
        stats = self.stream(tradingsymbol, timeframe)
        gap = False
        if stats.last_timestamp is not None:
            silence = (timestamp - stats.last_timestamp).total_seconds()
            if silence < 0:
                # Out-of-order arrival: count the row, but never corrupt the
                # first/last bookkeeping or invent a gap.
                stats.rows += 1
                stats.record_flag(quality_flag)
                return False
            if silence > self.gap_seconds and self._is_open(stats.last_timestamp):
                stats.gap_count += 1
                stats.largest_gap_seconds = max(stats.largest_gap_seconds, silence)
                gap = True
            stats.last_timestamp = timestamp
        else:
            stats.first_timestamp = timestamp
            stats.last_timestamp = timestamp
        stats.rows += 1
        stats.record_flag(quality_flag)
        if expected_rows is not None:
            stats.expected_rows = expected_rows
        return gap

    @property
    def streams(self) -> list[StreamStats]:
        """Every tracked stream, ordered by symbol then timeframe."""
        return [self._streams[key] for key in sorted(self._streams)]

    @property
    def row_count(self) -> int:
        """Total rows recorded across every stream."""
        return sum(stats.rows for stats in self._streams.values())

    def coverage_rows(self, *, run_id: str, trading_date: date | None = None) -> list[CoverageRow]:
        """Build one :class:`CoverageRow` per tracked stream."""
        rows: list[CoverageRow] = []
        for stats in self.streams:
            day = trading_date
            if day is None:
                reference = stats.first_timestamp or stats.last_timestamp
                day = reference.date() if reference else None
            if day is None:
                continue
            rows.append(build_coverage_row(trading_date=day, stats=stats, run_id=run_id))
        return rows


def build_coverage_row(*, trading_date: date, stats: StreamStats, run_id: str) -> CoverageRow:
    """Turn one stream's stats into a coverage row."""
    return CoverageRow(
        trading_date=trading_date,
        tradingsymbol=stats.tradingsymbol,
        timeframe=stats.timeframe,
        expected_rows=stats.expected_rows,
        actual_rows=stats.rows,
        first_timestamp=stats.first_timestamp,
        last_timestamp=stats.last_timestamp,
        gap_count=stats.gap_count,
        largest_gap_seconds=round(stats.largest_gap_seconds, 3),
        coverage_ratio=stats.coverage_ratio,
        quality_flag=stats.quality_flag,
        note=stats.note(),
        run_id=run_id,
    )
