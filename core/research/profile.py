"""Spec 24.7 and 24.8: the market's own intraday profile, from stored NIFTY bars.

The matrices in :mod:`core.research.segments` describe *the strategy*. These two questions
describe *the market*, and they cannot be answered by a trade ledger at all:

* **24.7 - the volatility profile by time-of-day bucket and cycle phase.** An ATR per
  bucket x phase, computed with the same indicator the feature rows use
  (:func:`core.indicators.atr`), so the profile and the columns in ``features`` agree about
  what an ATR is.
* **24.8 - the average cumulative intraday path per bucket.** For each bucket, the mean path
  of ``close / (first bar of the bucket's open) - 1`` from that bar to the end of the
  session, averaged over sessions: "if you were filled at the open of this bucket, what did
  the rest of the day look like, on average". Each point carries its own sample, because a
  minute that only three sessions reached is not a curve.

Both read the stored index bars (``index_history``: REST OHLCV with a ``timeframe`` column)
and bucket them with the **market calendar**, not with a hard-coded table:

* a bar is bucketed by its **close** moment (``timestamp + timeframe``), which is the same
  convention ``--mode build-features`` uses for the ``time_of_day_bucket`` column - the
  stored timestamp is the bar's *open*;
* the cycle phase comes from :meth:`core.market_calendar.NseCalendar.expiry_cycle_phase`,
  so a change to ``config/calendar.yaml`` moves this report with it;
* a bar outside every configured bucket is *counted* (``OUT_OF_SESSION``) and excluded,
  never silently folded into the nearest one.

One symbol per call is enforced: averaging NIFTY's ATR with a futures contract's would be a
number about nothing. Every table is returned even when the answer is "not enough sessions
yet" - it carries the sample and the ``preliminary`` flag rather than an empty promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.constants import DEFAULT_TIMEZONE, timeframe_to_minutes
from core.indicators import atr as atr_series
from core.schemas import ExpiryCyclePhase, TimeOfDayBucket

__all__ = [
    "CYCLE_PHASE_ORDER",
    "OUT_OF_SESSION",
    "BucketProfile",
    "BucketProfileTable",
    "IntradayPath",
    "IntradayPathPoint",
    "ProfileError",
    "bucket_order",
    "build_bucket_profiles",
    "build_intraday_paths",
]

#: The cycle's own order: the expiry day first, then the sessions after it.
CYCLE_PHASE_ORDER: tuple[str, ...] = (
    ExpiryCyclePhase.EXPIRY_DAY.value,
    ExpiryCyclePhase.START_OF_CYCLE.value,
    ExpiryCyclePhase.MID_CYCLE.value,
)

#: What a bar outside every configured bucket is labelled (and counted) as.
OUT_OF_SESSION = TimeOfDayBucket.OUT_OF_SESSION.value

#: The columns a bar frame must carry to be profiled at all.
BAR_COLUMNS: tuple[str, ...] = ("timestamp", "tradingsymbol", "timeframe", "open", "high", "low", "close")


class ProfileError(RuntimeError):
    """Raised when a profile cannot be computed from the bars it was handed."""


def bucket_order(calendar: Any) -> list[str]:
    """The configured bucket names, open to close (``OUT_OF_SESSION`` is never included)."""
    ordered = [name for name, _, _ in calendar.bucket_boundaries()]
    return [name for name in ordered if name != OUT_OF_SESSION]


def _frame(bars: Any, calendar: Any) -> pd.DataFrame:
    """Validate the bar frame and return a calendar-local, time-ordered copy.

    Adds three private columns the builders share: ``_minutes`` (the bar's timeframe),
    ``_closed`` (the instant the bar closed, in the calendar's timezone) and ``_day`` (its
    exchange-local date). Bucketing on the close is deliberate: the stored timestamp is the
    bar's *open*, and ``--mode build-features`` puts a bar in a bucket by when it closed.
    """
    if bars is None or len(bars) == 0:
        raise ProfileError("no bars to profile")
    missing = [name for name in BAR_COLUMNS if name not in bars.columns]
    if missing:
        raise ProfileError(
            f"the bar frame is missing column(s) {missing}; an index bar carries {list(BAR_COLUMNS)}"
        )
    frame = bars.loc[:, list(BAR_COLUMNS)].copy()
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], errors="raise"))
    if stamps.tz is None:
        raise ProfileError(
            "bar timestamps are naive; ANJANEYA only profiles timezone-aware bars "
            f"(they are stored in {DEFAULT_TIMEZONE})"
        )
    minutes = [timeframe_to_minutes(str(value)) for value in frame["timeframe"]]
    closed = (stamps + pd.to_timedelta(minutes, unit="m")).tz_convert(calendar.timezone)
    frame["timestamp"] = stamps
    frame["_minutes"] = minutes
    frame["_closed"] = pd.DatetimeIndex(closed)
    frame["_day"] = frame["_closed"].dt.date
    symbols = sorted({str(value) for value in frame["tradingsymbol"].dropna()})
    if len(symbols) > 1:
        raise ProfileError(
            f"the bar frame holds {len(symbols)} symbols ({symbols}); a profile is one symbol's "
            "- averaging two contracts' volatility describes neither"
        )
    return frame.sort_values("_closed", kind="stable").reset_index(drop=True)



# ---------------------------------------------------------------------------
# 24.7 - the volatility profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BucketProfile:
    """One bucket x cycle-phase cell of the volatility profile."""

    bucket: str
    cycle_phase: str
    sessions: int
    bars: int
    atr_mean: float | None
    range_mean: float | None
    range_mean_pct: float | None
    preliminary: bool

    @property
    def label(self) -> str:
        """``bucket | phase`` - the same spelling the matrices use for a cell."""
        return f"{self.bucket} | {self.cycle_phase}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "bucket": self.bucket,
            "cycle_phase": self.cycle_phase,
            "sessions": self.sessions,
            "bars": self.bars,
            "atr_mean": self.atr_mean,
            "range_mean": self.range_mean,
            "range_mean_pct": self.range_mean_pct,
            "preliminary": self.preliminary,
        }


@dataclass(slots=True)
class BucketProfileTable:
    """Spec 24.7: the ATR/range profile of one symbol, by bucket and cycle phase."""

    symbol: str
    bars: int
    sessions: int
    dropped_out_of_session: int
    min_sessions: int
    profiles: list[BucketProfile]

    @property
    def preliminary_cells(self) -> list[str]:
        """Cells with fewer than ``min_sessions`` sessions behind them."""
        return [profile.label for profile in self.profiles if profile.preliminary]

    def cell(self, bucket: str, cycle_phase: str) -> BucketProfile | None:
        """The profile of one cell, or ``None`` when that court never traded."""
        for profile in self.profiles:
            if profile.bucket == bucket and profile.cycle_phase == cycle_phase:
                return profile
        return None

    def grid(self, value: str = "atr_mean") -> list[list[Any]]:
        """A rectangular view of one column, ``None`` where there is no cell - heatmap ready."""
        buckets = sorted(
            {profile.bucket for profile in self.profiles},
            key=lambda name: [profile.bucket for profile in self.profiles].index(name),
        )
        return [
            [
                getattr(profile, value) if (profile := self.cell(bucket, phase)) else None
                for phase in CYCLE_PHASE_ORDER
            ]
            for bucket in buckets
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "sessions": self.sessions,
            "dropped_out_of_session": self.dropped_out_of_session,
            "min_sessions": self.min_sessions,
            "preliminary_cells": self.preliminary_cells,
            "profiles": [profile.to_dict() for profile in self.profiles],
        }


def build_bucket_profiles(
    bars: Any,
    calendar: Any,
    *,
    atr_period: int = 14,
    atr_method: str = "wilder",
    min_sessions: int = 2,
) -> BucketProfileTable:
    """Spec 24.7: mean ATR and mean bar range per time-of-day bucket and cycle phase.

    ``min_sessions`` is the sample rule for this table: a cell built from one session is
    flagged ``preliminary`` (one session has a shape, not a distribution). A cell whose ATR
    is undefined - a bucket with fewer bars than the ATR period - keeps ``atr_mean=None``
    while still reporting its range, rather than dropping the row.
    """
    frame = _frame(bars, calendar)
    symbol = str(frame["tradingsymbol"].iloc[0])
    ranges = atr_series(
        frame["high"], frame["low"], frame["close"], period=int(atr_period), method=atr_method
    )
    frame["_atr"] = ranges
    frame["_range"] = pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(
        frame["low"], errors="coerce"
    )
    frame["_range_pct"] = frame["_range"] / pd.to_numeric(frame["close"], errors="coerce")
    frame["_bucket"] = [calendar.time_of_day_bucket(moment) for moment in frame["_closed"]]
    frame["_phase"] = [calendar.expiry_cycle_phase(day).value for day in frame["_day"]]
    dropped = int((frame["_bucket"] == OUT_OF_SESSION).sum())
    inside = frame[frame["_bucket"] != OUT_OF_SESSION]
    sessions_total = int(inside["_day"].nunique())
    profiles: list[BucketProfile] = []
    for (bucket, phase), group in inside.groupby(["_bucket", "_phase"], sort=False):
        atr_values = group["_atr"].dropna()
        range_values = group["_range"].dropna()
        pct_values = group["_range_pct"].dropna()
        profiles.append(
            BucketProfile(
                bucket=str(bucket),
                cycle_phase=str(phase),
                sessions=int(group["_day"].nunique()),
                bars=int(len(group)),
                atr_mean=float(atr_values.mean()) if len(atr_values) else None,
                range_mean=float(range_values.mean()) if len(range_values) else None,
                range_mean_pct=float(pct_values.mean()) if len(pct_values) else None,
                preliminary=int(group["_day"].nunique()) < int(min_sessions),
            )
        )
    ordered = {name: index for index, name in enumerate(bucket_order(calendar))}
    phase_rank = {name: index for index, name in enumerate(CYCLE_PHASE_ORDER)}
    profiles.sort(
        key=lambda profile: (
            ordered.get(profile.bucket, len(ordered)),
            phase_rank.get(profile.cycle_phase, len(phase_rank)),
        )
    )
    return BucketProfileTable(
        symbol=symbol,
        bars=int(len(frame)),
        sessions=sessions_total,
        dropped_out_of_session=dropped,
        min_sessions=int(min_sessions),
        profiles=profiles,
    )


# ---------------------------------------------------------------------------
# 24.8 - the average cumulative intraday path
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IntradayPathPoint:
    """One minute of one bucket's average path, with the sessions that reached it."""

    minute: str
    bar_index: int
    mean_return_pct: float | None
    stdev_return_pct: float | None
    sample: int
    preliminary: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "minute": self.minute,
            "bar_index": self.bar_index,
            "mean_return_pct": self.mean_return_pct,
            "stdev_return_pct": self.stdev_return_pct,
            "sample": self.sample,
            "preliminary": self.preliminary,
        }


@dataclass(slots=True)
class IntradayPath:
    """Spec 24.8: the average cumulative return path that starts at one bucket's fill."""

    bucket: str
    base_minute: str
    sessions: int
    preliminary: bool
    points: list[IntradayPathPoint]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "bucket": self.bucket,
            "base_minute": self.base_minute,
            "sessions": self.sessions,
            "preliminary": self.preliminary,
            "points": [point.to_dict() for point in self.points],
        }


def build_intraday_paths(
    bars: Any,
    calendar: Any,
    *,
    min_sessions: int = 2,
    min_point_samples: int = 1,
) -> list[IntradayPath]:
    """Spec 24.8: for every bucket, the average ``close / bucket-open - 1`` path of a session.

    The base is the **open of the first bar whose close lands in that bucket**, and the path
    then runs to the last bar the session actually has - it is not truncated at the bucket's
    end, because the question ("if you were filled here, what did the rest of the day look
    like") is about what follows the fill. Every point is averaged over the sessions that
    reached that minute and carries its own sample, so a shape drawn by three sessions is
    labelled ``preliminary`` rather than averaged in with a hundred.

    ``min_point_samples`` drops minutes that fewer than that many sessions reached (a chart
    point backed by one session is a line through a dot); dropping is per point, and the
    count of dropped points is not hidden - the path simply ends where the data does.
    """
    frame = _frame(bars, calendar)
    frame["_bucket"] = [calendar.time_of_day_bucket(moment) for moment in frame["_closed"]]
    inside = frame[frame["_bucket"] != OUT_OF_SESSION]
    by_session: dict[Any, list[Any]] = {}
    for _, row in inside.iterrows():
        by_session.setdefault(row["_day"], []).append(row)
    paths: list[IntradayPath] = []
    for bucket in bucket_order(calendar):
        bucket_bars = inside[inside["_bucket"] == bucket]
        if bucket_bars.empty:
            continue
        samples: dict[str, list[float]] = {}
        base_minutes: dict[str, int] = {}
        sessions = 0
        for rows in by_session.values():
            rows = sorted(rows, key=lambda row: row["_closed"])
            base_index = next(
                (index for index, row in enumerate(rows) if row["_bucket"] == bucket), None
            )
            if base_index is None:
                continue
            base = float(rows[base_index]["open"])
            if not np.isfinite(base) or base <= 0.0:
                continue
            sessions += 1
            for offset, row in enumerate(rows[base_index:], start=1):
                close = float(row["close"])
                if not np.isfinite(close):
                    continue
                key = pd.Timestamp(row["_closed"]).strftime("%H:%M")
                samples.setdefault(key, []).append(close / base - 1.0)
                base_minutes.setdefault(key, offset)
        if sessions == 0:
            continue
        points = []
        for key in sorted(samples):
            values = samples[key]
            if len(values) < int(min_point_samples):
                continue
            array = np.asarray(values, dtype="float64")
            points.append(
                IntradayPathPoint(
                    minute=key,
                    bar_index=int(base_minutes[key]),
                    mean_return_pct=float(array.mean()) * 100.0,
                    stdev_return_pct=(float(array.std(ddof=1)) * 100.0 if array.size > 1 else None),
                    sample=int(array.size),
                    preliminary=array.size < int(min_sessions),
                )
            )
        if not points:
            continue
        paths.append(
            IntradayPath(
                bucket=bucket,
                base_minute=points[0].minute,
                sessions=sessions,
                preliminary=sessions < int(min_sessions),
                points=points,
            )
        )
    return paths

