"""Slope of a moving average, in the two units ANJANEYA needs.

A raw MA difference is in price points, which means the same "flat" NIFTY line
and a weekly option look wildly different. Spec 8/10 therefore define the
*normalised* slope

    normalized_slope[t] = (MA[t] - MA[t-k]) / (ATR[t] * k)

- points travelled per bar, expressed as a fraction of the instrument's own
average true range, so the number is comparable across strikes, expiries and
underlyings. :func:`slope` keeps the raw points-per-bar version as well, because
a research report has to be able to show the actual move, not only its ratio.

A zero ATR (a perfectly flat, zero-range stretch) yields ``NaN`` rather than an
infinity: "not measurable" and "infinitely steep" are different statements and
only one of them is true.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.indicators.series import as_float_series, require_pair, require_period

__all__ = [
    "difference",
    "normalized_slope",
    "slope",
]


def difference(values: Any, *, periods: int = 1) -> pd.Series:
    """``values[t] - values[t-periods]`` - a change over ``periods`` bars."""
    steps = require_period(periods, min_period=1, name="periods")
    series = as_float_series(values)
    return series - series.shift(steps)


def slope(values: Any, *, k: int) -> pd.Series:
    """Average change per bar over ``k`` bars, in the series' own units."""
    lookback = require_period(k, min_period=1, name="k")
    return difference(values, periods=lookback) / lookback


def normalized_slope(values: Any, *, atr: Any, k: int) -> pd.Series:
    """Spec's scale-free slope: ``(MA[t] - MA[t-k]) / (ATR[t] * k)``.

    ``atr`` is the *level* ATR at each bar (see :func:`core.indicators.atr.atr`).
    ``NaN`` wherever either the movement or the ATR is unavailable, and where the
    ATR is zero or negative.
    """
    lookback = require_period(k, min_period=1, name="k")
    series = as_float_series(values)
    ranges = as_float_series(atr, name="atr")
    require_pair(series, ranges, names=("values", "atr"))

    scale = ranges * lookback
    movement = series - series.shift(lookback)
    return (movement / scale).where(scale > 0)
