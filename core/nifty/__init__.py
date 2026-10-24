"""NIFTY index engines: what the index itself is doing (specs 8-9).

Implemented (Stage 3.2d)

* :mod:`~core.nifty.trend_score` - spec 8's per-average measurements and spec 9's
  fast/medium/long role scores, the weighted composite, its acceleration and its
  persistence;
* :mod:`~core.nifty.regime` - the composite mapped onto the seven regime states.

Like the price-action engines, these take a **candle frame for the index** plus the
resolved configuration: nothing here reads a file, opens a connection or looks at a
YAML file, which is what makes the score and the regime testable with a hand-built
frame in three lines. Both are pure functions of past bars - a score at ``t`` never
reads a bar after ``t`` - and both publish ``NaN``/``None`` rather than a number
built from a partial window.
"""

from __future__ import annotations

from core.nifty.regime import (
    NIFTY_REGIME_FEATURE_COLUMNS,
    classify_regime,
    regime_features,
)
from core.nifty.trend_score import (
    MA_METRIC_SUFFIXES,
    NIFTY_SCORE_ROLES,
    NIFTY_TREND_FEATURE_COLUMNS,
    combination_metrics,
    combination_score,
    combination_scores,
    ma_metrics,
    ma_table,
    role_score,
    trend_features,
)

__all__ = [
    "MA_METRIC_SUFFIXES",
    "NIFTY_REGIME_FEATURE_COLUMNS",
    "NIFTY_SCORE_ROLES",
    "NIFTY_TREND_FEATURE_COLUMNS",
    "classify_regime",
    "combination_metrics",
    "combination_score",
    "combination_scores",
    "ma_metrics",
    "ma_table",
    "regime_features",
    "role_score",
    "trend_features",
]

