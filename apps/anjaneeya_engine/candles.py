"""Tick -> candle construction for ANJANEYA-ENGINE (spec 4.7 / 6.3).

Three rules shape everything in this module
------------------------------------------
1. **A candle's ``timestamp`` is its OPEN time**, floored in IST. The 09:15:00
   bar therefore holds every tick in ``[09:15:00, 09:16:00)`` - no session
   anchor, no offset, no provider-specific convention.
2. **Nothing is fabricated.** A minute with no tick produces no bar. A missing
   minute is reported through ``data_coverage`` (gap count / coverage ratio),
   never invented with a synthetic price or a zero volume: a hole that is
   visible in the ledger is useful, a hole filled by a guess is not.
3. **Higher timeframes are merged from the 1m bars, not rebuilt from ticks.**
   That is far cheaper, and - more importantly - it guarantees the 15m open is
   always the first 1m open, so a feature computed on 15m data can never
   disagree with the same feature computed on 1m data.

Volume, the awkward part
------------------------
Kite publishes *cumulative* day volume per contract, so a bar's volume is the
sum of the increments observed inside it. The first tick of the day has no
predecessor (increment 0, which makes the first bar's volume ``last - first``
within that minute), and a **negative** increment means the exchange counter was
reset mid-session (a re-subscribe, a contract roll). A reset is counted in
:attr:`CandleBuildStats.volume_resets` and contributes 0 rather than a bogus
negative number - the ledger stays auditable instead of the number being quietly
dropped.

Failure policy: an empty frame is not an error (that is simply "no ticks"), but
a frame missing a required column or carrying a non-numeric price raises
:class:`CandleBuildError`. Guessing is never an option.

Everything here is pure: frames in, row objects out. All I/O (reading ticks,
writing candles, recording coverage) belongs to the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from core.constants import DEFAULT_TIMEZONE, get_timezone, timeframe_to_minutes
from core.schemas import CandleRow, QualityFlag, Source

__all__ = [
    "REQUIRED_TICK_COLUMNS",
    "CandleBuildError",
    "CandleBuildStats",
    "MissingBarSummary",
    "aggregate_candles",
    "build_1m_candles",
    "build_timeframes",
    "bucket_start",
    "expected_bar_openings",
    "missing_bar_summary",
]

#: Tick columns a 1m build cannot work without.
REQUIRED_TICK_COLUMNS = ("timestamp", "last_price")


class CandleBuildError(ValueError):
    """Raised when a tick frame cannot honestly be turned into candles."""


@dataclass(slots=True)
class CandleBuildStats:
    """What one :func:`build_1m_candles` call saw and produced."""

    tradingsymbol: str
    ticks: int = 0
    bars: int = 0
    volume_resets: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "tradingsymbol": self.tradingsymbol,
            "ticks": self.ticks,
            "bars": self.bars,
            "volume_resets": self.volume_resets,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
        }


@dataclass(slots=True)
class MissingBarSummary:
    """How much of a session a bar sequence actually covers."""

    expected_bars: int
    missing_bars: int
    gap_count: int
    largest_gap_seconds: float


def bucket_start(moment: datetime, minutes: int) -> datetime:
    """Floor ``moment`` to the open time of its ``minutes``-minute bucket.

    The floor is taken against wall-clock midnight, which is what every NSE
    session (09:15-15:30, the half-day close, the hour-aligned special sessions)
    is compatible with. ``tzinfo`` is preserved, never converted.
    """
    if minutes < 1:
        raise CandleBuildError(f"minutes must be >= 1, got {minutes}")
    floored = moment.replace(second=0, microsecond=0)
    if minutes == 1:
        return floored
    elapsed = floored.hour * 60 + floored.minute
    anchor = elapsed - elapsed % minutes
    return floored.replace(hour=anchor // 60, minute=anchor % 60)


def expected_bar_openings(
    *, session_start: datetime, session_end: datetime, minutes: int
) -> list[datetime]:
    """Every bar open time a complete session would produce (session half-open)."""
    step = timedelta(minutes=minutes)
    openings: list[datetime] = []
    moment = bucket_start(session_start, minutes)
    while moment < session_end:
        openings.append(moment)
        moment += step
    return openings


def missing_bar_summary(
    bars: Sequence[CandleRow],
    *,
    session_start: datetime,
    session_end: datetime,
    minutes: int,
) -> MissingBarSummary:
    """Compare a bar sequence against a complete session.

    A *gap* is a maximal run of consecutive missing bars (the same "count the
    silences" rule the tick coverage tracker uses), and its length is reported in
    seconds so 3 missing 15m bars read as ``2700s`` rather than "3".
    """
    openings = expected_bar_openings(
        session_start=session_start, session_end=session_end, minutes=minutes
    )
    present = {bar.timestamp for bar in bars}
    missing_bars = 0
    gap_count = 0
    largest_run = 0
    current_run = 0
    for opening in openings:
        if opening in present:
            current_run = 0
            continue
        missing_bars += 1
        current_run += 1
        if current_run == 1:
            gap_count += 1
        largest_run = max(largest_run, current_run)
    return MissingBarSummary(
        expected_bars=len(openings),
        missing_bars=missing_bars,
        gap_count=gap_count,
        largest_gap_seconds=largest_run * minutes * 60.0,
    )


# ---------------------------------------------------------------------------
# Value coercion: pandas hands back numpy/float/NaN, the row contracts want
# clean Python types or None. Coercion happens here, once, so no other module
# has to know about pandas dtypes.
# ---------------------------------------------------------------------------
def _present(value: Any) -> bool:
    """True when ``value`` is a real observation (not None/NaN/NaT)."""
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic object dtype
        return True


def _as_float(value: Any) -> float | None:
    return float(value) if _present(value) else None


def _as_int(value: Any) -> int | None:
    return int(value) if _present(value) else None


def _as_date(value: Any) -> Any:
    """A ``datetime.date`` from whatever the storage layer hands back.

    DuckDB converts a ``DATE`` column to pandas ``Timestamp`` on the way out, and
    the row contract wants a plain ``date`` - converting explicitly means an
    expiry can never depend on pydantic's lax-mode coercion.
    """
    if not _present(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


def _first_present(series: Any) -> Any:
    """First non-null value of a column subset (``None`` when absent/empty)."""
    if series is None:
        return None
    values = series.dropna()
    return values.iloc[0] if not values.empty else None


def _last_present(series: Any) -> Any:
    """Last non-null value of a column subset (``None`` when absent/empty)."""
    if series is None:
        return None
    values = series.dropna()
    return values.iloc[-1] if not values.empty else None


def _volume_increments(frame: pd.DataFrame) -> tuple[pd.Series | None, int]:
    """Per-tick cumulative-volume increments, plus the number of counter resets.

    Returns ``(None, 0)`` when the frame has no ``volume_traded`` column or no
    observed volume at all (the NIFTY index carries none), so the caller can
    distinguish "volume unknown" from "volume is genuinely zero".
    """
    if "volume_traded" not in frame.columns:
        return None, 0
    observed = pd.to_numeric(frame["volume_traded"], errors="coerce").dropna()
    if observed.empty:
        return None, 0
    delta = observed.diff()
    resets = int((delta < 0).sum())
    # NaN (the day's first observation) and resets both contribute 0: the bar
    # keeps the volume it can actually account for.
    increments = delta.where(delta > 0, other=0.0).astype("int64")
    return increments, resets


def build_1m_candles(
    frame: pd.DataFrame,
    *,
    tradingsymbol: str,
    run_id: str,
    timezone_name: str = DEFAULT_TIMEZONE,
    source: Source = Source.DERIVED,
    quality_flag: QualityFlag = QualityFlag.OK,
) -> tuple[list[CandleRow], CandleBuildStats]:
    """Build the 1m bars of one contract from its ticks.

    The frame is expected to hold a single ``tradingsymbol`` (that is what the
    reader returns and what keeps one contract's price history from leaking into
    another's candles). Bars come back ordered by open time, each carrying
    ``source=DERIVED``: the *ticks* are the imported data, the candle is our
    artifact, so its provenance says so - the tick's own source stays visible in
    the ``ticks`` table.

    Raises
    ------
    CandleBuildError
        When a required column is missing, a price is not numeric, or a
        timestamp is naive.
    """
    missing = [name for name in REQUIRED_TICK_COLUMNS if name not in frame.columns]
    if missing:
        raise CandleBuildError(
            f"tick frame for {tradingsymbol!r} is missing column(s) {missing}; "
            f"expected at least {REQUIRED_TICK_COLUMNS}"
        )
    stats = CandleBuildStats(tradingsymbol=tradingsymbol)
    if frame.empty:
        return [], stats

    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    prices = pd.to_numeric(ordered["last_price"], errors="coerce")
    bad = int(prices.isna().sum())
    if bad:
        raise CandleBuildError(
            f"{bad} tick(s) for {tradingsymbol!r} have a missing or non-numeric last_price; "
            "refusing to build candles from them"
        )
    ordered["last_price"] = prices

    stamps = pd.to_datetime(ordered["timestamp"], errors="raise")
    if stamps.dt.tz is None:
        raise CandleBuildError(
            f"tick timestamps for {tradingsymbol!r} are naive; ANJANEYA only stores "
            "timezone-aware data (see settings.time.forbid_naive_timestamps)"
        )
    zone = get_timezone(timezone_name)
    minutes = stamps.dt.tz_convert(timezone_name).dt.tz_localize(None).dt.floor("min")

    increments, resets = _volume_increments(ordered)
    has_volume = increments is not None
    if has_volume:
        # Align the increments with their ticks once; the per-minute sum is then
        # a plain column aggregation and cannot drift out of step with the group.
        ordered["volume_increment"] = increments.reindex(ordered.index)

    bars: list[CandleRow] = []
    for minute, chunk in ordered.groupby(minutes, sort=True):
        volume = None
        if has_volume:
            volume = _as_int(chunk["volume_increment"].sum(min_count=1))
        bars.append(
            CandleRow(
                timestamp=minute.to_pydatetime().replace(tzinfo=zone),
                timeframe="1m",
                instrument_token=_as_int(_first_present(chunk.get("instrument_token"))),
                tradingsymbol=tradingsymbol,
                expiry=_as_date(_first_present(chunk.get("expiry"))),
                strike=_as_float(_first_present(chunk.get("strike"))),
                option_type=_first_present(chunk.get("option_type")),
                open=_as_float(chunk["last_price"].iloc[0]),
                high=_as_float(chunk["last_price"].max()),
                low=_as_float(chunk["last_price"].min()),
                close=_as_float(chunk["last_price"].iloc[-1]),
                volume=volume,
                oi=_as_int(_last_present(chunk.get("oi"))),
                source=source,
                fetch_timestamp=_last_present(chunk.get("fetch_timestamp")),
                quality_flag=quality_flag,
                run_id=run_id,
            )
        )

    stats.ticks = int(len(ordered))
    stats.bars = len(bars)
    stats.volume_resets = resets
    stats.first_timestamp = bars[0].timestamp if bars else None
    stats.last_timestamp = bars[-1].timestamp if bars else None
    return bars, stats


def build_timeframes(
    candles_1m: Sequence[CandleRow], *, timeframes: Sequence[str]
) -> dict[str, list[CandleRow]]:
    """Build every requested timeframe from one contract's 1m bars.

    ``"1m"`` returns the input bars unchanged (not a copy of a copy), so a caller
    can iterate over the result without special-casing the base timeframe. An
    unknown label raises ``ValueError`` via :func:`timeframe_to_minutes` - a typo
    must never produce an empty (and therefore silently "complete") timeframe.
    """
    built: dict[str, list[CandleRow]] = {}
    for timeframe in timeframes:
        timeframe_to_minutes(timeframe)
        if timeframe == "1m":
            built[timeframe] = list(candles_1m)
        else:
            built[timeframe] = aggregate_candles(candles_1m, timeframe=timeframe)
    return built


def aggregate_candles(candles: Sequence[CandleRow], *, timeframe: str) -> list[CandleRow]:
    """Merge 1m bars into ``timeframe`` bars.

    Bars are grouped by **open-time bucket**, so a 15m bar starting 09:15 merges
    09:15, 09:16 ... 09:29. Missing 1m bars are simply absent from the merge (the
    higher bar then covers slightly less volume and the coverage ledger reports
    the hole at 1m level, where it actually happened).
    """
    minutes = timeframe_to_minutes(timeframe)
    if minutes == 1:
        return list(candles)
    if not candles:
        return []

    merged: list[CandleRow] = []
    bucket: list[CandleRow] = []
    open_time: datetime | None = None
    for candle in sorted(candles, key=lambda bar: bar.timestamp):
        key = bucket_start(candle.timestamp, minutes)
        if open_time is None:
            open_time, bucket = key, [candle]
        elif key != open_time:
            merged.append(_merge_bars(bucket, timeframe=timeframe, open_time=open_time))
            open_time, bucket = key, [candle]
        else:
            bucket.append(candle)
    if bucket and open_time is not None:
        merged.append(_merge_bars(bucket, timeframe=timeframe, open_time=open_time))
    return merged


def _merge_bars(bars: Sequence[CandleRow], *, timeframe: str, open_time: datetime) -> CandleRow:
    """Merge chronologically ordered 1m bars into one higher-timeframe bar."""
    first = bars[0]
    volumes = [bar.volume for bar in bars if bar.volume is not None]
    stamps = [bar.fetch_timestamp for bar in bars if bar.fetch_timestamp is not None]
    if first.tradingsymbol != bars[-1].tradingsymbol:  # pragma: no cover - defensive
        raise CandleBuildError("refusing to merge bars from different contracts")
    return CandleRow(
        timestamp=open_time,
        timeframe=timeframe,
        instrument_token=first.instrument_token,
        tradingsymbol=first.tradingsymbol,
        expiry=first.expiry,
        strike=first.strike,
        option_type=first.option_type,
        open=first.open,
        high=max(bar.high for bar in bars),
        low=min(bar.low for bar in bars),
        close=bars[-1].close,
        volume=sum(volumes) if volumes else None,
        oi=_last_not_none(bars, "oi"),
        # ``iv`` moves intraday, so the merged bar takes the newest observation;
        # the greeks are per-contract constants and carry over unchanged.
        iv=_last_not_none(bars, "iv"),
        delta=first.delta,
        gamma=first.gamma,
        theta=first.theta,
        vega=first.vega,
        days_to_expiry=first.days_to_expiry,
        distance_from_atm=first.distance_from_atm,
        source=first.source,
        fetch_timestamp=max(stamps) if stamps else None,
        quality_flag=first.quality_flag,
        run_id=first.run_id,
    )


def _last_not_none(bars: Sequence[CandleRow], field: str) -> Any:
    """Newest non-``None`` value of ``field`` across ``bars``."""
    for bar in reversed(bars):
        value = getattr(bar, field)
        if value is not None:
            return value
    return None
