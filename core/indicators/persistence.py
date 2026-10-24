"""Trend persistence: does the recent past agree with the present?

A single trend score says where the market stands *now*. Persistence asks the
follow-up question every trend engine needs: is that reading new, or has it been
saying the same thing for the last ``window`` bars? Spec 8/10 asks for exactly
that, with a configurable window (``persistence.window``).

Definition used here
--------------------
Take the score's direction ``sign(score[t])`` (``+1`` / ``-1`` / ``0``, with a
configurable dead-band) and let

    persistence[t] = (# of the last `window` directions equal to sign(score[t]))
                     / (# of the last `window` directions that are non-zero)

so the value is in ``[0, 1]``: ``1.0`` when every recent bar agreed, ``0.5`` when
the window is split. Three honest ``NaN`` cases instead of invented numbers:

* the current direction is ``0`` (there is no trend to be persistent),
* the window contains no non-zero direction at all,
* the window is not yet full.

The calculation looks backwards only - the value at ``t`` never uses a bar after
``t`` - so it is safe to append a bar at a time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.indicators.series import as_float_series, require_period

__all__ = [
    "direction",
    "trend_persistence",
]


def direction(score: Any, *, deadband: float = 0.0) -> pd.Series:
    """Direction of a score as ``+1`` / ``-1`` / ``0``.

    ``deadband`` treats ``|score| <= deadband`` as flat, which keeps a score that
    is essentially zero from being counted as a trend. ``NaN`` scores stay
    ``NaN`` - they are not flat, they are unknown.
    """
    if not isinstance(deadband, (int, float)) or isinstance(deadband, bool):
        raise TypeError(f"deadband must be a number, got {type(deadband).__name__}")
    if deadband < 0:
        raise ValueError(f"deadband must be >= 0, got {deadband}")
    series = as_float_series(score, name="score")
    bullish = (series > deadband).astype("float64")
    bearish = (series < -deadband).astype("float64")
    return (bullish - bearish).where(series.notna())


def trend_persistence(score: Any, *, window: int, deadband: float = 0.0) -> pd.Series:
    """Fraction of the last ``window`` bars agreeing with the current direction.

    See the module docstring for the exact definition and the ``NaN`` cases.
    """
    span = require_period(window)
    signs = direction(score, deadband=deadband)
    rolling = signs.rolling(window=span, min_periods=span)
    # ``non_flat`` is 1 where the bar had a direction, 0 where it was flat and
    # NaN where the score itself was unknown, so the two rolling sums below are
    # "how many bars voted" and "the net vote (+1 bullish / -1 bearish)".
    non_flat = signs.abs().clip(upper=1.0).rolling(window=span, min_periods=span).sum()
    net = rolling.sum()
    agreeing = (non_flat + signs * net) / 2.0
    ratio = agreeing / non_flat
    return ratio.where((signs != 0) & (non_flat > 0))
