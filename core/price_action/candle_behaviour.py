"""Candle behaviour engine (spec 11).

What a candle *is* and what it *says*, expressed as numbers.

Geometry first - body, range and the two wicks are pure measurements:

    body       = |close - open|
    range      = high - low
    upper_wick = high - max(open, close)
    lower_wick = min(open, close) - low

and then the ratios that make those numbers comparable between a 120-point NIFTY
bar and a 4-point option bar. Ratios are ``NaN`` when the candle has no range at
all: ``0/0`` is undefined, not ``0``.

Relationship to the previous candle second - the part spec 11 calls "store the
numeric measurements as ML features, names are secondary": gap, engulf ratio,
inside/outside flags, the change in body and wick ratios, and where this close
landed inside the previous bar's range.

:func:`candle_patterns` then turns those numbers into the named detections the
research reports talk about (small body, long wick, engulfing, inside, outside)
using the thresholds in ``config/indicators.yaml``. The names are derived - they
never feed a feature value, and they never turn into a signal.

Everything here reads backwards only: a candle may use its own fields and the
previous candle's, never the next candle's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config_loader import CandleBehaviourConfig
from core.indicators.series import as_float_series
from core.schemas import CandleDirection

__all__ = [
    "SHAPE_COLUMNS",
    "candle_features",
    "candle_patterns",
    "candle_relationships",
    "candle_shape",
]

#: Columns produced by :func:`candle_shape`.
SHAPE_COLUMNS = (
    "candle_body",
    "candle_range",
    "candle_upper_wick",
    "candle_lower_wick",
    "candle_body_ratio",
    "candle_upper_wick_ratio",
    "candle_lower_wick_ratio",
    "candle_direction",
)

#: Relationship columns produced by :func:`candle_relationships`.
RELATIONSHIP_COLUMNS = (
    "candle_gap",
    "candle_engulf_ratio",
    "candle_inside_flag",
    "candle_outside_flag",
    "candle_body_ratio_change",
    "candle_wick_ratio_change",
    "candle_close_position_within_prev_range",
)

#: Every column :func:`candle_features` produces, in feature-contract order.
FEATURE_ORDER = SHAPE_COLUMNS + RELATIONSHIP_COLUMNS

_OHLC = ("open", "high", "low", "close")


def _require_ohlc(frame: pd.DataFrame) -> None:
    missing = [name for name in _OHLC if name not in frame.columns]
    if missing:
        raise ValueError(f"candle frame is missing column(s) {missing}; expected OHLC")
    if frame.empty:
        return
    invalid = (frame["high"] < frame["low"]).fillna(False)
    if bool(invalid.any()):
        raise ValueError("candle frame contains a bar whose high is below its low")


def candle_shape(frame: pd.DataFrame) -> pd.DataFrame:
    """Body, range, wicks, their ratios and the candle direction.

    Index-aligned to ``frame``. Ratios are ``NaN`` for a zero-range candle
    (``0/0`` is not a flat candle, it is an undefined ratio); ``inf`` is never
    produced.
    """
    _require_ohlc(frame)
    opened = as_float_series(frame["open"], name="open")
    high = as_float_series(frame["high"], name="high")
    low = as_float_series(frame["low"], name="low")
    close = as_float_series(frame["close"], name="close")

    body = (close - opened).abs()
    span = high - low
    top = pd.concat([opened, close], axis=1).max(axis=1)
    bottom = pd.concat([opened, close], axis=1).min(axis=1)
    safe = span.where(span > 0)

    direction = pd.Series(CandleDirection.FLAT.value, index=frame.index, dtype="object")
    direction = direction.mask(close > opened, CandleDirection.BULLISH.value)
    direction = direction.mask(close < opened, CandleDirection.BEARISH.value)
    unknown = opened.isna() | close.isna() | span.isna()
    direction = direction.mask(unknown, None)

    return pd.DataFrame(
        {
            "candle_body": body,
            "candle_range": span,
            "candle_upper_wick": high - top,
            "candle_lower_wick": bottom - low,
            "candle_body_ratio": body / safe,
            "candle_upper_wick_ratio": (high - top) / safe,
            "candle_lower_wick_ratio": (bottom - low) / safe,
            "candle_direction": direction,
        },
        index=frame.index,
    )


def _body_bounds(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Top and bottom of each candle body (``max``/``min`` of open and close)."""
    opened = as_float_series(frame["open"], name="open")
    close = as_float_series(frame["close"], name="close")
    pair = pd.concat([opened, close], axis=1)
    return pair.max(axis=1), pair.min(axis=1)


def candle_relationships(frame: pd.DataFrame) -> pd.DataFrame:
    """How this candle relates to the one before it.

    * ``candle_gap`` - ``open[t] - close[t-1]`` in price points. A gap is a price
      fact that reports quote in points; every *distance* in the feature row is
      ATR-normalised and the ATR is stored beside it.
    * ``candle_engulf_ratio`` - how much of the previous **body** this body
      covers, signed: ``+`` for a bullish body engulfing a bearish one, ``-`` for
      the reverse, ``0`` when both candles point the same way (same-direction
      candles cannot engulf), ``None`` on the first bar.
    * ``candle_inside_flag`` / ``candle_outside_flag`` - this range sits inside
      the previous range / strictly engulfs it.
    * ``candle_body_ratio_change`` / ``candle_wick_ratio_change`` - the bar-on-bar
      change of the shape ratios; the wick change uses upper + lower, i.e. the
      total wick share of the range.
    * ``candle_close_position_within_prev_range`` - ``(close - prev_low) /
      (prev_high - prev_low)``. Deliberately **not** clipped: a close above the
      previous high is a real gap-through and must read as ``> 1``. ``None`` when
      the previous range was zero.

    The first bar of the frame has no predecessor, so every relationship there is
    ``None``.
    """
    shape = candle_shape(frame)
    high = as_float_series(frame["high"], name="high")
    low = as_float_series(frame["low"], name="low")
    close = as_float_series(frame["close"], name="close")
    opened = as_float_series(frame["open"], name="open")
    previous_high, previous_low = high.shift(1), low.shift(1)

    direction = shape["candle_direction"]
    previous_direction = direction.shift(1)
    top, bottom = _body_bounds(frame)
    previous_top, previous_bottom = top.shift(1), bottom.shift(1)
    overlap = pd.concat([top, previous_top], axis=1).min(axis=1) - pd.concat(
        [bottom, previous_bottom], axis=1
    ).max(axis=1)
    previous_body = shape["candle_body"].shift(1)
    coverage = (overlap.clip(lower=0) / previous_body.where(previous_body > 0)).clip(upper=1.0)

    engulfing_bull = (direction == CandleDirection.BULLISH.value) & (
        previous_direction == CandleDirection.BEARISH.value
    )
    engulfing_bear = (direction == CandleDirection.BEARISH.value) & (
        previous_direction == CandleDirection.BULLISH.value
    )
    comparable = direction.notna() & previous_direction.notna()
    engulf = pd.Series(np.nan, index=frame.index, dtype="float64")
    engulf = engulf.mask(comparable & ~engulfing_bull & ~engulfing_bear, 0.0)
    engulf = engulf.mask(engulfing_bull, coverage)
    engulf = engulf.mask(engulfing_bear, -coverage)

    previous_range = previous_high - previous_low
    wick_ratio = shape["candle_upper_wick_ratio"] + shape["candle_lower_wick_ratio"]
    have_previous = previous_high.notna() & previous_low.notna()
    return pd.DataFrame(
        {
            "candle_gap": opened - close.shift(1),
            "candle_engulf_ratio": engulf,
            "candle_inside_flag": ((high <= previous_high) & (low >= previous_low)).where(
                have_previous, other=pd.NA
            ),
            "candle_outside_flag": ((high > previous_high) & (low < previous_low)).where(
                have_previous, other=pd.NA
            ),
            "candle_body_ratio_change": shape["candle_body_ratio"].diff(),
            "candle_wick_ratio_change": wick_ratio - wick_ratio.shift(1),
            "candle_close_position_within_prev_range": (close - previous_low)
            / previous_range.where(previous_range > 0),
        },
        index=frame.index,
    ).astype({"candle_inside_flag": "boolean", "candle_outside_flag": "boolean"})

def candle_patterns(
    frame: pd.DataFrame,
    *,
    config: CandleBehaviourConfig,
    shape: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Named detections derived from the numbers (spec 11), as booleans.

    Thresholds come from ``config/indicators.yaml`` (``candle.small_body_ratio``,
    ``candle.long_wick_ratio``, ``candle.engulf_min_ratio``). These columns exist
    for reports and sanity checks: a *name* is never a feature value, and none of
    them is a trading signal.

    ``NaN``/``pd.NA`` where the underlying measurement is undefined (a zero-range
    candle has no body ratio to compare against a threshold).
    """
    geometry = candle_shape(frame) if shape is None else shape
    relationships = candle_relationships(frame)
    direction = geometry["candle_direction"]
    definable = direction.notna() & (geometry["candle_range"] > config.epsilon)

    engulf = relationships["candle_engulf_ratio"]
    return pd.DataFrame(
        {
            "small_body": (geometry["candle_body_ratio"] < config.small_body_ratio).where(
                definable, other=pd.NA
            ),
            "long_upper_wick": (
                geometry["candle_upper_wick_ratio"] > config.long_wick_ratio
            ).where(definable, other=pd.NA),
            "long_lower_wick": (
                geometry["candle_lower_wick_ratio"] > config.long_wick_ratio
            ).where(definable, other=pd.NA),
            "bullish_engulfing": (engulf >= config.engulf_min_ratio).where(
                engulf.notna(), other=pd.NA
            ),
            "bearish_engulfing": (engulf <= -config.engulf_min_ratio).where(
                engulf.notna(), other=pd.NA
            ),
            "inside": relationships["candle_inside_flag"],
            "outside": relationships["candle_outside_flag"],
        },
        index=frame.index,
    ).astype("boolean")


def candle_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Every candle-behaviour column of the unified feature row (spec 18).

    One call, one frame, columns named exactly as the ``features`` table spells
    them - so the feature builder maps them across without a translation table
    that could drift out of step with the schema.
    """
    combined = pd.concat([candle_shape(frame), candle_relationships(frame)], axis=1)
    return combined[list(FEATURE_ORDER)]


