"""Swing detection with an explicit confirmation delay (spec 15).

A swing high is a bar whose high is the highest in the window
``[i - left_bars, i + right_bars]``. The **right side is the whole problem**:
that bar is only *known* to be a swing once ``right_bars`` further bars exist, so
a feature computed at bar ``i`` may not use it.

This module therefore keeps two things apart:

* :func:`swing_points` - the raw swings, each carrying the bar it was confirmed
  at (``i + right_bars``) as well as the bar it happened at;
* :func:`swing_levels` - the per-bar view the feature row needs, where a swing's
  price is attached to a row **only from its confirmation bar onwards**.

That distinction is what makes "never use a swing that would only be confirmed in
the future" (spec 15, non-negotiable) true by construction rather than by review:
the value simply does not exist in the frame before the bar that could have
observed it.

Ties count as swings at every tied bar - with equal highs there is no unique
extreme, and silently picking the first one would be an invented convention.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config_loader import SwingConfig
from core.indicators.series import as_float_series

__all__ = [
    "SWING_LEVEL_COLUMNS",
    "SwingPoint",
    "recent_swings",
    "swing_levels",
    "swing_points",
]

#: Columns produced by :func:`swing_levels`.
SWING_LEVEL_COLUMNS = (
    "swing_high_1",
    "swing_low_1",
    "swing_high_2",
    "swing_low_2",
    "swing_label",
)

_HIGH = "high"
_LOW = "low"


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """One confirmed swing: where it happened and when it became knowable."""

    kind: str
    #: Position of the swing bar itself (0-based, within the analysed frame).
    position: int
    price: float
    #: Position of the first bar at which this swing could be known.
    confirmed_position: int


def _swing_flags(frame: pd.DataFrame, *, config: SwingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays marking raw swing highs and lows within the frame."""
    height = config.left_bars + config.right_bars + 1
    high = as_float_series(frame["high"], name="high")
    low = as_float_series(frame["low"], name="low")
    is_high = high == high.rolling(height, center=True, min_periods=height).max()
    is_low = low == low.rolling(height, center=True, min_periods=height).min()
    return is_high.fillna(False).to_numpy(), is_low.fillna(False).to_numpy()


def _iter_swings(frame: pd.DataFrame, *, config: SwingConfig) -> Iterator[SwingPoint]:
    """Raw swings in chronological order (highs and lows interleaved)."""
    is_high, is_low = _swing_flags(frame, config=config)
    highs = as_float_series(frame["high"]).to_numpy()
    lows = as_float_series(frame["low"]).to_numpy()
    found = sorted(
        [(int(position), _HIGH) for position in np.flatnonzero(is_high)]
        + [(int(position), _LOW) for position in np.flatnonzero(is_low)]
    )
    for position, kind in found:
        yield SwingPoint(
            kind=kind,
            position=position,
            price=float(highs[position] if kind == _HIGH else lows[position]),
            confirmed_position=position + config.right_bars,
        )

def swing_points(frame: pd.DataFrame, *, config: SwingConfig) -> pd.DataFrame:
    """All confirmed swings in the frame, one row per swing.

    Columns: ``kind`` (``high``/``low``), ``position``, ``price``,
    ``confirmed_position`` and ``confirmed_at`` (the timestamp at the confirmation
    bar). The last ``right_bars`` bars of the frame can never hold a *confirmed*
    swing - there is not enough data after them yet, and that is reported by the
    frame simply not containing it.
    """
    swings = list(_iter_swings(frame, config=config))
    stamps = frame["timestamp"].to_numpy() if "timestamp" in frame.columns else None
    return pd.DataFrame(
        {
            "kind": [swing.kind for swing in swings],
            "position": [swing.position for swing in swings],
            "price": [swing.price for swing in swings],
            "confirmed_position": [swing.confirmed_position for swing in swings],
            "confirmed_at": [
                None if stamps is None else stamps[swing.confirmed_position] for swing in swings
            ],
        }
    )


def _recent(frame: pd.DataFrame, swings: list[SwingPoint], *, count: int) -> list[pd.Series]:
    """Per-bar series of the ``count`` most recent confirmed swings of one kind.

    Index ``j`` (0-based) is the ``j+1``-th most recent swing. Each one is attached
    to a row only from the confirmation bar onwards, so a bar can never see a swing
    that would still need future bars to be recognised.
    """
    blocks = pd.Series(np.nan, index=frame.index, dtype="float64")
    positions: list[int] = []
    prices: list[float] = []
    for swing in swings:
        if swing.confirmed_position < len(frame):
            positions.append(swing.confirmed_position)
            prices.append(swing.price)
    if not positions:
        return [blocks.copy() for _ in range(count)]

    # Within one kind the confirmation positions are strictly increasing (one bar
    # yields at most one swing of each kind), so shifting the event series by
    # ``rank`` ranks the swings by recency without any sorting or searching.
    events = pd.Series(prices, index=positions, dtype="float64")
    return [
        _scatter(template=blocks, positions=positions, events=events, rank=rank)
        for rank in range(count)
    ]


def _scatter(
    *, template: pd.Series, positions: list[int], events: pd.Series, rank: int
) -> pd.Series:
    """The (``rank``+1)-th most recent event of a kind, forward-filled per bar."""
    column = template.copy()
    column.iloc[positions] = events.shift(rank).to_numpy()
    return column.ffill()


def recent_swings(
    frame: pd.DataFrame, *, config: SwingConfig, count: int | None = None
) -> pd.DataFrame:
    """The ``count`` most recent confirmed swings of each kind, per bar.

    Columns ``swing_high_1..swing_high_n`` then ``swing_low_1..swing_low_n``, most
    recent first. ``count`` defaults to ``config.max_swings`` - history the
    support/resistance engine offers as candidate levels (spec 14). Where fewer
    swings have been confirmed the column is simply ``NaN``: there will never be
    that many, and a gap says so honestly.
    """
    if count is None:
        count = config.max_swings
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    swings = list(_iter_swings(frame, config=config))
    columns: dict[str, pd.Series] = {}
    for kind in (_HIGH, _LOW):
        series = _recent(frame, [swing for swing in swings if swing.kind == kind], count=count)
        for offset, values in enumerate(series, start=1):
            columns[f"swing_{kind}_{offset}"] = values
    if frame.empty:
        return pd.DataFrame(
            {name: pd.Series([], dtype="float64") for name in columns}, index=frame.index
        )
    return pd.DataFrame(columns, index=frame.index)


def swing_levels(frame: pd.DataFrame, *, config: SwingConfig) -> pd.DataFrame:
    """The swing columns of the feature row (spec 18), per bar.

    * ``swing_high_1`` / ``swing_low_1`` - the most recent *confirmed* swing;
    * ``swing_high_2`` / ``swing_low_2`` - the one before it, which is what tells
      a higher high from a lower high (and what the structure engine reads);
    * ``swing_label`` - the shape of the two most recent swing pairs, as
      ``HH_HL`` (bullish structure), ``LH_LL`` (bearish), ``LH_HL`` (contracting
      range) or ``HH_LL`` (expanding transition). ``None`` until both a high and
      a low pair are known: an invented label would be worse than a gap.
    """
    swings = list(_iter_swings(frame, config=config))
    highs = _recent(frame, [swing for swing in swings if swing.kind == _HIGH], count=2)
    lows = _recent(frame, [swing for swing in swings if swing.kind == _LOW], count=2)
    high_1, high_2 = highs[0], highs[1]
    low_1, low_2 = lows[0], lows[1]

    higher_high = high_1 > high_2
    higher_low = low_1 > low_2
    comparable = high_2.notna() & low_2.notna()
    label = pd.Series(None, index=frame.index, dtype="object")
    label = label.mask(comparable & higher_high & higher_low, "HH_HL")
    label = label.mask(comparable & ~higher_high & ~higher_low, "LH_LL")
    label = label.mask(comparable & ~higher_high & higher_low, "LH_HL")
    label = label.mask(comparable & higher_high & ~higher_low, "HH_LL")

    return pd.DataFrame(
        {
            "swing_high_1": high_1,
            "swing_low_1": low_1,
            "swing_high_2": high_2,
            "swing_low_2": low_2,
            "swing_label": label,
        },
        index=frame.index,
    )

