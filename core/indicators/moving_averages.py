"""Moving averages: SMA and EMA, plus the alignment of an MA stack.

Two moving averages, deliberately only two:

* :func:`sma` - a rolling mean. The value at ``t`` uses exactly the ``period``
  bars ending at ``t``; before that many bars exist there is no value (``NaN``),
  because a "5-bar average" of two bars is not a 5-bar average.
* :func:`ema` - an exponential average with an **explicit warm-up rule**. The
  recursion seeds on the first observation (pandas ``adjust=False``), but the
  value is only *published* once ``period`` observations have been seen. Without
  that rule ``ema(200)`` returns a number after two bars, which would silently
  poison every feature built from the long NIFTY averages (SMA 2056, EMA 254...).
  ANJANEYA would rather have a gap it can see than a number it cannot trust.

:func:`ma_alignment` then turns a stack of averages into one number in
``[-1, +1]``: ``+1`` when the fastest average is above the next, above the next,
and so on (a bullish stack), ``-1`` when the stack is perfectly inverted. That is
the "price above/below, MA alignment" of spec 8/10 expressed numerically rather
than as a label.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from core.indicators.series import as_float_series, require_period

__all__ = [
    "MA_SOURCES",
    "MaType",
    "ema",
    "ma_alignment",
    "ma_source",
    "moving_average",
    "sma",
]

#: Price fields a moving average may be computed on. ``hl2`` and ``hlc3`` are the
#: two classic derived prices (midpoint, typical) built from columns the candle
#: row already carries - no extra storage.
MA_SOURCES = ("open", "high", "low", "close", "hl2", "hlc3")

#: Which moving average to compute, as spelled in ``config/indicators.yaml``.
MaType = Literal["sma", "ema"]

#: Frame columns each source needs.
_SOURCE_INPUTS: dict[str, tuple[str, ...]] = {
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
    "hl2": ("high", "low"),
    "hlc3": ("high", "low", "close"),
}


def ma_source(frame: pd.DataFrame, source: str) -> pd.Series:
    """Extract the price series an MA is computed on.

    Raises for an unknown source or a frame missing the required columns, so a
    typo in ``config/indicators.yaml`` fails loudly at build time instead of
    quietly producing a feature computed on the wrong field.
    """
    if source not in _SOURCE_INPUTS:
        raise ValueError(
            f"Unknown MA source {source!r}; expected one of {', '.join(MA_SOURCES)}"
        )
    missing = [name for name in _SOURCE_INPUTS[source] if name not in frame.columns]
    if missing:
        raise ValueError(f"MA source {source!r} needs column(s) {missing} in the candle frame")

    if source == "hl2":
        values = (frame["high"] + frame["low"]) / 2.0
    elif source == "hlc3":
        values = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    else:
        values = frame[source]
    return as_float_series(values, name=source)


def sma(values: Any, period: int, *, min_periods: int | None = None) -> pd.Series:
    """Simple moving average over ``period`` observations.

    ``min_periods`` defaults to ``period``. Lower it only when a partial window
    is genuinely wanted - the returned Series then mixes warm-up values with real
    ones, which is exactly what the default avoids.
    """
    window = require_period(period)
    minimum = window if min_periods is None else require_period(min_periods, min_period=1)
    series = as_float_series(values)
    return series.rolling(window=window, min_periods=minimum).mean()


def ema(values: Any, period: int) -> pd.Series:
    """Exponential moving average, published only after ``period`` observations.

    ``alpha = 2 / (period + 1)``, seeded on the first observation, which is the
    standard "EMA with ``adjust=False``" recursion. The difference from a naive
    ``ewm`` call is the warm-up mask: positions before the ``period``-th
    observation are ``NaN``, so a long-period EMA cannot appear out of nowhere.
    """
    span = require_period(period)
    series = as_float_series(values)
    seen = series.notna().cumsum()
    averaged = series.ewm(span=span, adjust=False, ignore_na=False).mean()
    return averaged.where(seen >= span)


def moving_average(values: Any, *, ma_type: MaType | str, period: int) -> pd.Series:
    """Dispatch to :func:`sma` or :func:`ema` using the config vocabulary."""
    if ma_type == "sma":
        return sma(values, period)
    if ma_type == "ema":
        return ema(values, period)
    raise ValueError(f"Unknown moving-average type {ma_type!r}; expected 'sma' or 'ema'")


def ma_alignment(*averages: Any) -> pd.Series:
    """Agreement of an MA stack, in ``[-1, +1]``.

    Arguments must be ordered **fastest first**. Each adjacent pair votes ``+1``
    when the faster average is above the slower one and ``-1`` when it is below;
    the result is the mean vote. ``NaN`` while any pair is still undefined (the
    stack is only as ready as its slowest member), and for a single average -
    there is no alignment to measure.
    """
    if len(averages) < 2:
        raise ValueError("ma_alignment needs at least two moving averages")
    series = [
        as_float_series(values, name=f"average_{index}") for index, values in enumerate(averages)
    ]
    votes = [np.sign(faster - slower) for faster, slower in zip(series, series[1:], strict=False)]
    stacked = pd.concat(votes, axis=1)
    return stacked.mean(axis=1, skipna=False)
