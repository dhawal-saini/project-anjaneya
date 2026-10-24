"""Daily and previous-day levels (spec 12).

The rule that makes this engine honest is stated in the spec in capitals: at time
``T``, "*today_high*" means the running high up to ``T``, never the finished
day's high, and previous-day values must come only from sessions that had already
closed before ``T``. Both are non-negotiable, and both have a dedicated test.

How the two are honoured here
-----------------------------
* **Running extremes** are ``groupby(session).cummax()/cummin()`` - a bar can only
  ever see itself and the bars before it in its own session.
* **Previous-day levels** are the *last* bar of the previous session present in
  the frame, mapped onto the current session's bars. A session is only previous
  once all of its bars are behind ``T``, which is automatic because the values
  are attached per session date rather than per bar position.

One caveat is worth stating out loud: "the previous session" means the previous
session **in the frame**. If a day is missing from the candles (a capture gap),
the level comes from the last session that is present, and the coverage ledger is
where that hole is reported - not a made-up value here.

Distances are ATR-normalised, like every other distance in the feature row, so
``distance_to_previous_day_high = -0.4`` reads "40% of an ATR below the previous
day's high". The raw levels sit in the same row, so nothing is lost.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.constants import DEFAULT_TIMEZONE, get_timezone
from core.indicators.series import as_float_series, require_pair

__all__ = [
    "LEVELS_FEATURE_COLUMNS",
    "level_features",
    "session_levels",
    "trade_dates",
]

#: Columns this module contributes to the unified feature row.
LEVELS_FEATURE_COLUMNS = (
    "level_today_open",
    "level_today_high_running",
    "level_today_low_running",
    "level_previous_day_open",
    "level_previous_day_high",
    "level_previous_day_low",
    "level_previous_day_close",
    "distance_to_today_high",
    "distance_to_today_low",
    "distance_to_previous_day_high",
    "distance_to_previous_day_low",
)

#: The raw level columns (without distances), in storage order.
_RAW_LEVEL_COLUMNS = LEVELS_FEATURE_COLUMNS[:7]


def trade_dates(frame: pd.DataFrame, *, timezone_name: str = DEFAULT_TIMEZONE) -> pd.Series:
    """The IST trading date of every bar.

    Sessions are defined by exchange-local midnight, exactly as in the coverage
    ledger. Naive timestamps are rejected - a naive bar cannot be assigned to a
    session, and guessing the timezone is how a session boundary silently moves.
    """
    if "timestamp" not in frame.columns:
        raise ValueError("level computation needs a 'timestamp' column")
    stamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if stamps.dt.tz is None:
        raise ValueError("level computation needs timezone-aware timestamps (IST)")
    return stamps.dt.tz_convert(get_timezone(timezone_name)).dt.date


def session_levels(frame: pd.DataFrame, *, timezone_name: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Per-bar today/previous-day levels, index-aligned to ``frame``.

    ``level_today_open`` is the session's first bar's open, the two running
    columns are the extremes *so far within that bar's own session*, and the four
    ``level_previous_day_*`` values come from the last completed session present
    in the frame.
    """
    if frame.empty:
        return pd.DataFrame(columns=_RAW_LEVEL_COLUMNS, index=frame.index, dtype="float64")

    scoped = pd.DataFrame(
        {
            "date": trade_dates(frame, timezone_name=timezone_name).to_numpy(),
            "open": as_float_series(frame["open"]).to_numpy(),
            "high": as_float_series(frame["high"]).to_numpy(),
            "low": as_float_series(frame["low"]).to_numpy(),
            "close": as_float_series(frame["close"]).to_numpy(),
        },
        index=frame.index,
    )
    grouped = scoped.groupby("date", sort=True)
    previous = grouped.agg(
        previous_day_open=("open", "first"),
        previous_day_high=("high", "max"),
        previous_day_low=("low", "min"),
        previous_day_close=("close", "last"),
    ).shift(1)

    return pd.DataFrame(
        {
            "level_today_open": grouped["open"].transform("first"),
            "level_today_high_running": grouped["high"].cummax(),
            "level_today_low_running": grouped["low"].cummin(),
            "level_previous_day_open": scoped["date"].map(previous["previous_day_open"]),
            "level_previous_day_high": scoped["date"].map(previous["previous_day_high"]),
            "level_previous_day_low": scoped["date"].map(previous["previous_day_low"]),
            "level_previous_day_close": scoped["date"].map(previous["previous_day_close"]),
        },
        index=frame.index,
    )


def level_features(
    frame: pd.DataFrame, *, atr: Any, timezone_name: str = DEFAULT_TIMEZONE
) -> pd.DataFrame:
    """Levels plus their ATR-normalised distances, ready for the feature row.

    ``distance_to_*`` is ``(close - level) / atr``, so negative means price is
    below the level. ``None`` wherever the level or the ATR is not yet known (the
    first session in a frame has no previous day, and an ATR needs its warm-up).
    """
    levels = session_levels(frame, timezone_name=timezone_name)
    close = as_float_series(frame["close"], name="close")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, ranges, names=("close", "atr"))
    scale = ranges.where(ranges > 0)

    return pd.DataFrame(
        {
            **{column: levels[column] for column in _RAW_LEVEL_COLUMNS},
            "distance_to_today_high": (close - levels["level_today_high_running"]) / scale,
            "distance_to_today_low": (close - levels["level_today_low_running"]) / scale,
            "distance_to_previous_day_high": (close - levels["level_previous_day_high"]) / scale,
            "distance_to_previous_day_low": (close - levels["level_previous_day_low"]) / scale,
        },
        index=frame.index,
    )

