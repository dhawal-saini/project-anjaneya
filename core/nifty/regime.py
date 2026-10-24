"""NIFTY regime classification (spec 9).

The regime is the composite trend score with the config's thresholds applied - a
*description* of conditions, never a signal. Spec 9 is explicit that this engine
"does NOT emit BUY/SELL", and nothing in this module argues with that: it maps a
number in ``[-1, +1]`` onto one of seven states and stops.

Where the boundaries fall
-------------------------
Thresholds come from ``nifty_scores.regime_thresholds`` and must increase strictly
from ``strong_bearish`` to ``strong_bullish`` (the config loader enforces it). The
mapping is

    ==============================  ====================
    composite                       state
    ==============================  ====================
    ``>= strong_bullish``           ``STRONG_BULLISH``
    ``>= bullish``                  ``BULLISH``
    ``>= bullish_developing``       ``BULLISH_DEVELOPING``
    ``> bearish_developing``        ``NEUTRAL``
    ``> bearish``                   ``BEARISH_DEVELOPING``
    ``> strong_bearish``            ``BEARISH``
    otherwise                       ``STRONG_BEARISH``
    ==============================  ====================

Two properties of that table are deliberate:

* **The boundaries are symmetric.** A score of ``+0.25`` is ``BULLISH`` and
  ``-0.25`` is ``BEARISH``; ``+0.08`` is ``BULLISH_DEVELOPING`` and ``-0.08`` is
  ``BEARISH_DEVELOPING``, and so on. The bullish side is inclusive (``>=``) and
  the bearish side exclusive (``>``), which is what makes ``±threshold`` land on
  the two matching states instead of one state and one neighbour.
* **``NEUTRAL`` is a band, not a point.** It covers ``|score| <
  bullish_developing``, so it is the config's ``bullish_developing`` that says how
  quiet "no trend" has to be. A score that is *exactly* zero is ``NEUTRAL``, and a
  missing score (``NaN``) is ``None`` - a distinct statement: not "flat" but
  "unknown".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config_loader import RegimeThresholdsConfig
from core.indicators.series import as_float_series
from core.schemas import RegimeState

__all__ = [
    "NIFTY_REGIME_FEATURE_COLUMNS",
    "classify_regime",
    "regime_features",
]

#: The single column the ``features`` table carries for the regime.
NIFTY_REGIME_FEATURE_COLUMNS = ("nifty_regime",)


def classify_regime(composite: Any, *, thresholds: RegimeThresholdsConfig) -> pd.Series:
    """Map composite scores to :class:`RegimeState` values, bar by bar.

    Returns an object column of state names (``RegimeState`` values), with ``None``
    where the composite is ``NaN``. See the module docstring for the boundary
    rules - they are asserted bar by bar in the tests, threshold by threshold, so
    an inclusive/exclusive slip cannot pass unnoticed.
    """
    values = as_float_series(composite, name="composite")
    conditions = [
        values >= thresholds.strong_bullish,
        values >= thresholds.bullish,
        values >= thresholds.bullish_developing,
        values > thresholds.bearish_developing,
        values > thresholds.bearish,
        values > thresholds.strong_bearish,
    ]
    choices = [
        RegimeState.STRONG_BULLISH.value,
        RegimeState.BULLISH.value,
        RegimeState.BULLISH_DEVELOPING.value,
        RegimeState.NEUTRAL.value,
        RegimeState.BEARISH_DEVELOPING.value,
        RegimeState.BEARISH.value,
    ]
    states = np.select(conditions, choices, default=RegimeState.STRONG_BEARISH.value)
    return pd.Series(states, index=values.index, dtype="object").where(values.notna(), None)


def regime_features(
    composite: Any, *, thresholds: RegimeThresholdsConfig
) -> pd.DataFrame:
    """The ``nifty_regime`` column of the unified feature row (spec 18).

    Takes the *composite* rather than a candle frame: the regime is a function of
    that one number, so passing it in keeps the thresholds applied in one place -
    :func:`core.nifty.trend_score.trend_features` produces the score, this produces
    the label, and neither can drift away from the other.
    """
    values = as_float_series(composite, name="composite")
    return pd.DataFrame(
        {"nifty_regime": classify_regime(values, thresholds=thresholds)}, index=values.index
    )

