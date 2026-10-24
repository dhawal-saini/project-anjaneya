"""Synthetic candle frames and indicator configs for the trend-engine tests.

The trend engines (specs 8-10) are pure functions of a candle frame and a
resolved ``IndicatorsConfig``, so their tests need exactly two fixtures - a frame
and a config - and this module is the single place that builds them. Two rules
shape both:

* **Short periods.** The engines only care about slopes, alignment and warm-up
  boundaries, so the test configs use 2-, 3-, 4- and 6-bar averages. The shipped
  ``config/indicators.yaml`` has SMA 2056 in it; a test that used it would need a
  week of bars to see a single composite.
* **Checkable arithmetic.** :func:`ramp_frame` is a straight ``+step``-per-bar
  ramp with symmetric ranges, so a two-bar average of ``low``/``high`` sits a
  known distance from the close and the expected score can be written as a
  formula with ``math.tanh`` in it rather than copied from a previous run.

The configs are built through ``model_validate`` rather than around it: a test
then exercises the same validated object the engine receives in production.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.config_loader import IndicatorsConfig

#: The regime thresholds of ``config/indicators.yaml`` (spec 9).
THRESHOLDS = {
    "strong_bullish": 0.60,
    "bullish": 0.25,
    "bullish_developing": 0.08,
    "bearish_developing": -0.08,
    "bearish": -0.25,
    "strong_bearish": -0.60,
}

#: The role weights of ``config/indicators.yaml`` (spec 9).
WEIGHTS = {"fast": 0.40, "medium": 0.35, "long": 0.25}

#: The S/R weights - only needed so the ``levels`` config section validates.
SR_WEIGHTS = {
    "touches": 0.30,
    "rejections": 0.25,
    "holds": 0.20,
    "recency": 0.15,
    "distance": 0.10,
}

#: Short lookbacks, so the arithmetic in the tests stays checkable by hand.
SLOPE_K = 2
PERSISTENCE_WINDOW = 3
VELOCITY_K = 2
MOMENTUM_PERIOD = 2

#: The default test combination: one short average per role.
DEFAULT_COMBINATION = [
    {"name": "ema_3_low", "type": "ema", "period": 3, "source": "low", "role": "fast"},
    {"name": "sma_6_close", "type": "sma", "period": 6, "source": "close", "role": "medium"},
    {"name": "sma_12_close", "type": "sma", "period": 12, "source": "close", "role": "long"},
]

#: The default option stack: short enough to warm up inside a test frame.
DEFAULT_OPTION_AVERAGES = [
    {"name": "ema_2_low", "type": "ema", "period": 2, "source": "low"},
    {"name": "ema_2_high", "type": "ema", "period": 2, "source": "high"},
    {"name": "ema_4_close", "type": "ema", "period": 4, "source": "close"},
]


def ma(
    name: str,
    period: int,
    *,
    ma_type: str = "ema",
    source: str = "close",
    role: str | None = None,
) -> dict[str, Any]:
    """One moving-average definition, as it appears in ``config/indicators.yaml``."""
    spec: dict[str, Any] = {"name": name, "type": ma_type, "period": period, "source": source}
    if role is not None:
        spec["role"] = role
    return spec


def indicators_config(
    *,
    combination: list[dict[str, Any]] | None = None,
    option_averages: list[dict[str, Any]] | None = None,
    policy: str = "flag",
    min_candles: int = 4,
    spread_pair: dict[str, str] | None = None,
) -> IndicatorsConfig:
    """A small, fully validated indicators config with short periods.

    ``combination`` replaces all three NIFTY combinations at once (the tests
    compare combinations against each other, which needs them to differ only when
    the test says so), ``option_averages`` replaces the option stack, ``policy`` /
    ``min_candles`` drive spec 10's insufficient-history behaviour, and
    ``spread_pair`` names the two option averages the published spread sits between
    (``None`` leaves the engine's fastest-versus-slowest fallback in place).
    """
    nifty = combination if combination is not None else DEFAULT_COMBINATION
    averages = option_averages if option_averages is not None else DEFAULT_OPTION_AVERAGES
    return IndicatorsConfig.model_validate(
        {
            "slope": {"k": SLOPE_K},
            "persistence": {"window": PERSISTENCE_WINDOW},
            "spread_velocity": {"k": VELOCITY_K},
            "momentum": {"period": MOMENTUM_PERIOD},
            "nifty_moving_averages": {
                "combination_1": [dict(spec) for spec in nifty],
                "combination_2": [dict(spec) for spec in nifty],
                "combination_3": [dict(spec) for spec in nifty],
            },
            "nifty_scores": {"weights": WEIGHTS, "regime_thresholds": THRESHOLDS},
            "option_trend": {
                "moving_averages": [dict(spec) for spec in averages],
                "insufficient_history_policy": policy,
                "min_candles_for_full_set": min_candles,
                **(
                    {"spread_pair": dict(spread_pair)}
                    if spread_pair is not None
                    else {}
                ),
            },
            "levels": {"support_resistance": {"weights": SR_WEIGHTS}},
        }
    )


def ramp_frame(
    count: int, *, start: float = 100.0, step: float = 1.0, spread: float = 1.0
) -> pd.DataFrame:
    """A straight ramp: ``count`` bars whose close rises by ``step`` each bar.

    Ranges are symmetric around the close, so a two-bar simple average of ``low``
    or ``high`` is a hand-computable number, and every average on the frame rises
    by exactly ``step`` per bar - which is what makes the slope vote checkable.
    """
    closes = [start + step * index for index in range(count)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + spread / 2 for value in closes],
            "low": [value - spread / 2 for value in closes],
            "close": closes,
        }
    )


def flat_frame(count: int, *, price: float = 50.0) -> pd.DataFrame:
    """A perfectly flat frame: every OHLC value is the same, every range is zero."""
    return pd.DataFrame(
        {
            "open": [price] * count,
            "high": [price] * count,
            "low": [price] * count,
            "close": [price] * count,
        }
    )


def flat_atr(frame: pd.DataFrame, value: float = 1.0) -> pd.Series:
    """A constant ATR, so a normalised number is the raw number divided by ``value``."""
    return pd.Series(value, index=frame.index, dtype="float64")


def two_bar_specs() -> list[Any]:
    """One two-bar SMA spec - the shortest average the config schema allows."""
    config = indicators_config(combination=[ma("sma_2_close", 2, ma_type="sma", role="fast")])
    return list(config.nifty_moving_averages.combination_1)
