"""Spread between two series and how fast that spread is changing.

Spec 8/10 define the MA separation as ``(fast_MA - slow_MA) / ATR`` - i.e. the
gap between two averages measured in units of the instrument's own volatility,
which is what makes "EMA 20 is above EMA 200" comparable between a NIFTY future
leg and a 25000-strike option.

Spread velocity is then the k-bar change of that (already normalised) spread, so
it answers the second question a trend engine has to answer: not only *how far
apart* the averages are, but whether the gap is *opening or closing*.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.indicators.series import as_float_series, require_pair, require_period

__all__ = [
    "spread",
    "spread_velocity",
]


def spread(fast: Any, slow: Any, *, atr: Any | None = None) -> pd.Series:
    """``fast - slow``, optionally normalised by ``atr``.

    Pass ``atr`` for the spec's volatility-normalised separation; omit it for the
    raw point gap (useful in reports). ``NaN`` wherever either average is
    undefined, and wherever a supplied ATR is zero or negative.
    """
    faster = as_float_series(fast, name="fast")
    slower = as_float_series(slow, name="slow")
    require_pair(faster, slower, names=("fast", "slow"))
    gap = faster - slower
    if atr is None:
        return gap

    ranges = as_float_series(atr, name="atr")
    require_pair(gap, ranges, names=("spread", "atr"))
    return (gap / ranges).where(ranges > 0)


def spread_velocity(values: Any, *, k: int) -> pd.Series:
    """``spread[t] - spread[t-k]`` - is the gap opening (+), closing (-) or flat?"""
    lookback = require_period(k, min_period=1, name="k")
    series = as_float_series(values, name="spread")
    return series - series.shift(lookback)
