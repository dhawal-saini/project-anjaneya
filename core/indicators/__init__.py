"""Indicator primitives - the pure-moving-parts layer of ANJANEYA-ENGINE.

Everything in this package is a **pure function of a past-only window**: a
``pandas.Series`` goes in, an index-aligned Series comes out, and the value at
position ``i`` is computable from positions ``<= i`` alone. No I/O, no config
lookups, no state, no exceptions for "not enough data" - just ``NaN``.

That single property is what makes the rest of the engine testable. The feature
builder can compute an indicator over a lookback window and trust the last value
to equal the value a full-history run would have produced, so the same code
serves a single-day rebuild and a long backfill without a second implementation.
The prefix tests in ``tests/unit/test_indicators.py`` assert exactly that
invariant, indicator by indicator.

Two conventions hold everywhere:

``NaN`` means *not yet defined*
    A 200-period average after 12 bars, an ATR before its first full window, a
    slope divided by a zero ATR. ``None``/``NULL`` downstream - never ``0``.

``0`` is a measurement
    A flat slope, a zero spread, a perfectly agreeing MA stack. Zero is a value
    the engines reason about, so it must stay distinguishable from "no data".

Modules
-------
:mod:`~core.indicators.series`
    Input normalisation and the shared warm-up helpers.
:mod:`~core.indicators.moving_averages`
    SMA, EMA and MA-stack alignment.
:mod:`~core.indicators.atr`
    True range and ATR (Wilder and simple) - the unit other features are
    measured in.
:mod:`~core.indicators.slope`
    Raw and ATR-normalised slope.
:mod:`~core.indicators.spread`
    Normalised spread between two averages, and its velocity.
:mod:`~core.indicators.persistence`
    Direction and trend persistence.
:mod:`~core.indicators.statistics`
    Rolling statistics, z-scores, returns, momentum and running extremes.
"""

from __future__ import annotations

from core.indicators.atr import AtrMethod, atr, true_range, wilder_rma
from core.indicators.moving_averages import (
    MA_SOURCES,
    MaType,
    ema,
    ma_alignment,
    ma_source,
    moving_average,
    sma,
)
from core.indicators.persistence import direction, trend_persistence
from core.indicators.series import as_float_series, first_full_window, nan_before, require_period
from core.indicators.slope import difference, normalized_slope, slope
from core.indicators.spread import spread, spread_velocity
from core.indicators.statistics import (
    momentum,
    percentage_change,
    rolling_mean,
    rolling_std,
    running_max,
    running_min,
    volatility,
    zscore,
)

__all__ = [
    "MA_SOURCES",
    "AtrMethod",
    "MaType",
    "as_float_series",
    "atr",
    "difference",
    "direction",
    "ema",
    "first_full_window",
    "ma_alignment",
    "ma_source",
    "momentum",
    "moving_average",
    "nan_before",
    "normalized_slope",
    "percentage_change",
    "require_period",
    "rolling_mean",
    "rolling_std",
    "running_max",
    "running_min",
    "sma",
    "slope",
    "spread",
    "spread_velocity",
    "trend_persistence",
    "true_range",
    "volatility",
    "wilder_rma",
    "zscore",
]
