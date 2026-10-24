"""Basic statistics used as features: dispersion, z-scores, momentum, extremes.

These are the "Phase 4 - basic statistics" of the master spec: the small,
boring measurements that make the rest of the feature table interpretable.

* :func:`rolling_mean` / :func:`rolling_std` - the local baseline and its spread;
* :func:`zscore` - how unusual the current value is *relative to the recent past
  only* (a full-series z-score would leak the future into every row);
* :func:`percentage_change` - the one-bar return, guarded against non-positive
  prices;
* :func:`volatility` - the rolling standard deviation of those returns, which is
  the price-level-independent volatility measure;
* :func:`momentum` - ``price[t] - price[t-n]``, optionally divided by ATR so a
  NIFTY move and a 25000-strike option move are the same kind of number;
* :func:`running_max` / :func:`running_min` - the highest/lowest value *so far*.
  These are what "today's high" has to mean at time ``t`` (spec 12): a running
  extreme, never the finished day's extreme.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.indicators.series import as_float_series, require_pair, require_period

__all__ = [
    "momentum",
    "percentage_change",
    "rolling_mean",
    "rolling_std",
    "running_max",
    "running_min",
    "volatility",
    "zscore",
]


def rolling_mean(values: Any, *, period: int) -> pd.Series:
    """Mean of the last ``period`` observations; ``NaN`` until the window is full."""
    window = require_period(period)
    return as_float_series(values).rolling(window=window, min_periods=window).mean()


def rolling_std(values: Any, *, period: int, ddof: int = 1) -> pd.Series:
    """Sample standard deviation of the last ``period`` observations.

    ``ddof=1`` (the pandas default) because these windows are samples of a
    process, not a population of everything that exists.
    """
    window = require_period(period)
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    return as_float_series(values).rolling(window=window, min_periods=window).std(ddof=ddof)


def zscore(values: Any, *, period: int, ddof: int = 1) -> pd.Series:
    """``(x[t] - rolling_mean) / rolling_std`` over ``period`` bars.

    ``NaN`` while the window is incomplete and wherever the window is perfectly
    flat (a zero standard deviation means "no scale", not "infinitely unusual").
    """
    window = require_period(period)
    series = as_float_series(values)
    deviation = rolling_std(series, period=window, ddof=ddof)
    baseline = rolling_mean(series, period=window)
    return ((series - baseline) / deviation).where(deviation > 0)


def percentage_change(values: Any, *, periods: int = 1) -> pd.Series:
    """``x[t] / x[t-periods] - 1``.

    ``NaN`` when the earlier value is missing or non-positive - a return against
    a zero price is undefined, and reporting it as ``inf`` would travel straight
    into a model.
    """
    steps = require_period(periods, min_period=1, name="periods")
    series = as_float_series(values)
    previous = series.shift(steps)
    return (series / previous - 1.0).where(previous > 0)


def volatility(values: Any, *, period: int) -> pd.Series:
    """Rolling standard deviation of one-bar percentage changes.

    Percentage changes (not price differences) so the number does not scale with
    the instrument's price level, which is what makes it usable as a feature
    across strikes and expiries.
    """
    window = require_period(period)
    return rolling_std(percentage_change(values), period=window)


def momentum(values: Any, *, period: int, atr: Any | None = None) -> pd.Series:
    """``x[t] - x[t-period]``, optionally normalised by ``atr[t]``.

    Without ``atr`` this is the raw point move; with it, the move in units of the
    instrument's own average true range (the form the option trend engine uses).
    """
    lookback = require_period(period, min_period=1, name="period")
    series = as_float_series(values)
    change = series - series.shift(lookback)
    if atr is None:
        return change

    ranges = as_float_series(atr, name="atr")
    require_pair(change, ranges, names=("momentum", "atr"))
    return (change / ranges).where(ranges > 0)


def running_max(values: Any) -> pd.Series:
    """Highest value seen so far, including the current one (no look-ahead)."""
    return as_float_series(values).cummax()


def running_min(values: Any) -> pd.Series:
    """Lowest value seen so far, including the current one (no look-ahead)."""
    return as_float_series(values).cummin()
