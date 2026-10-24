"""Average True Range - the scale that makes every other feature comparable.

ATR is not just another indicator here: it is the **unit** ANJANEYA measures
things in. Slopes are divided by it, option spreads are divided by it, level
proximity is expressed in fractions of it. Getting it wrong rescales half the
feature table, so it is defined explicitly:

* **True range** ``TR[t] = max(high-low, |high - close[t-1]|, |low - close[t-1]|)``.
  The first bar has no previous close, so its true range is simply its range.
* **Wilder's smoothing** (:func:`wilder_rma`) is a recursive average with
  ``alpha = 1/period``, *seeded with the mean of the first ``period`` true
  ranges* - the classic definition (and the one TA-Lib uses). A value exists from
  the ``period``-th bar onwards; before that it is ``NaN``.
* **Simple** smoothing is a plain rolling mean of the true ranges, also requiring
  a full window.

Both refuse to publish a warm-up value: an ATR-14 after three bars is not an
ATR-14, and a normalised slope divided by it would be meaningless.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from core.indicators.moving_averages import sma
from core.indicators.series import (
    as_float_series,
    first_full_window,
    nan_before,
    require_pair,
    require_period,
)

__all__ = [
    "AtrMethod",
    "atr",
    "true_range",
    "wilder_rma",
]

#: Smoothing method for :func:`atr`, as spelled in ``config/indicators.yaml``.
AtrMethod = Literal["wilder", "simple"]


def true_range(high: Any, low: Any, close: Any) -> pd.Series:
    """Bar-by-bar true range.

    ``max(high, previous_close) - min(low, previous_close)`` is algebraically the
    same as the textbook ``max(high-low, |high-pc|, |low-pc|)`` but vectorises
    without a Python loop.
    """
    highs = as_float_series(high, name="high")
    lows = as_float_series(low, name="low")
    closes = as_float_series(close, name="close")
    require_pair(highs, lows, names=("high", "low"))
    require_pair(highs, closes, names=("high", "close"))
    if (highs < lows).any():
        raise ValueError("true_range received a bar whose high is below its low")

    previous_close = closes.shift(1)
    # numpy's maximum/minimum propagate NaN, which matters: a bar with a missing
    # low must produce a missing true range, not a half-computed one.
    upper = np.maximum(highs.to_numpy(), previous_close.to_numpy())
    lower = np.minimum(lows.to_numpy(), previous_close.to_numpy())
    spanned = pd.Series(upper - lower, index=highs.index, dtype="float64")
    return spanned.where(previous_close.notna(), highs - lows)


def wilder_rma(values: Any, *, period: int) -> pd.Series:
    """Wilder's recursive average (the smoothing behind ATR and RSI).

    ``rma[t] = (rma[t-1] * (period - 1) + x[t]) / period``, seeded with the mean
    of the first ``period`` observations. Implemented as an ``alpha = 1/period``
    EWM whose head is *forced* to the seed, so the recursion is exact while the
    computation stays vectorised.
    """
    span = require_period(period)
    series = as_float_series(values)
    seed_index = first_full_window(series, span)
    if seed_index is None:
        return nan_before(series, None)

    window = series.iloc[seed_index - span + 1 : seed_index + 1]
    prepared = series.copy()
    prepared.iloc[: seed_index + 1] = float(window.mean())
    averaged = prepared.ewm(alpha=1.0 / span, adjust=False, ignore_na=False).mean()
    return nan_before(averaged, seed_index)


def atr(
    high: Any,
    low: Any,
    close: Any,
    *,
    period: int = 14,
    method: AtrMethod | str = "wilder",
) -> pd.Series:
    """Average true range over ``period`` bars.

    ``method="wilder"`` is the default because that is the definition the
    configured baselines assume; ``"simple"`` is offered for comparison.
    """
    ranges = true_range(high, low, close)
    if method == "wilder":
        return wilder_rma(ranges, period=period)
    if method == "simple":
        return sma(ranges, period)
    raise ValueError(f"Unknown ATR method {method!r}; expected 'wilder' or 'simple'")
