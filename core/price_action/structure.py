"""Market structure from confirmed swings (spec 15).

Structure is the only engine in this package that is *built on* another engine:
it reads the labelled swing pairs from :mod:`core.price_action.swings` and turns
them into one state, never re-detecting swings itself. That keeps the
confirmation delay in exactly one place.

The mapping (from the two most recent confirmed swing pairs):

===============  ==========================================
``HH_HL``        ``BULLISH_STRUCTURE`` - highs and lows both rising
``LH_LL``        ``BEARISH_STRUCTURE`` - both falling
``LH_HL``        ``RANGE`` - contracting, the range is getting tighter
``HH_LL``        ``TRANSITION`` - expanding in both directions
not yet known    ``None`` - fewer than two confirmed swings
===============  ==========================================

``None`` when structure cannot be stated is deliberate: on a young contract, or
early in a session, there is genuinely no structure yet, and inventing
``RANGE`` would be a claim the data does not support.

Distances to the latest swings are **ATR-normalised**, like every other distance
in the feature row, so a reading is comparable across contracts and strikes. The
raw swing prices are stored alongside, so nothing is lost.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.config_loader import SwingConfig
from core.indicators.series import as_float_series, require_pair
from core.price_action.swings import swing_levels
from core.schemas import MarketStructureState

__all__ = [
    "STRUCTURE_FEATURE_COLUMNS",
    "market_structure",
    "structure_features",
]

#: Columns this module contributes to the unified feature row.
STRUCTURE_FEATURE_COLUMNS = (
    "swing_high_1",
    "swing_low_1",
    "swing_label",
    "market_structure_state",
    "distance_from_swing_high",
    "distance_from_swing_low",
)

#: Swing label -> market structure state.
_LABEL_STATES: dict[str, MarketStructureState] = {
    "HH_HL": MarketStructureState.BULLISH_STRUCTURE,
    "LH_LL": MarketStructureState.BEARISH_STRUCTURE,
    "LH_HL": MarketStructureState.RANGE,
    "HH_LL": MarketStructureState.TRANSITION,
}


def market_structure(swing_label: Any) -> pd.Series:
    """Map ``swing_label`` values to a :class:`MarketStructureState`, bar by bar.

    Unknown labels raise rather than becoming ``None``: a label this module does
    not understand means the swing engine changed and the mapping was not updated,
    which must not be discovered later from a silent gap in the feature table.
    """
    labels = pd.Series(swing_label)
    unknown = {value for value in labels.dropna().unique() if value not in _LABEL_STATES}
    if unknown:
        raise ValueError(f"Unknown swing label(s) {sorted(unknown)}; expected one of {sorted(_LABEL_STATES)}")
    states = labels.map({key: state.value for key, state in _LABEL_STATES.items()})
    return states.where(labels.notna(), None)


def structure_features(
    frame: pd.DataFrame, *, config: SwingConfig, atr: Any
) -> pd.DataFrame:
    """Swing levels, structure state and ATR-normalised distance to the swings.

    ``atr`` is the instrument's ATR series for the same bars (see
    :func:`core.indicators.atr.atr`); distances are ``(close - swing) / atr``, so
    a distance of ``-0.5`` reads "half an ATR below the latest confirmed swing
    high". ``None`` wherever the swing or the ATR does not exist yet.
    """
    levels = swing_levels(frame, config=config)
    close = as_float_series(frame["close"], name="close")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, ranges, names=("close", "atr"))
    scale = ranges.where(ranges > 0)

    return pd.DataFrame(
        {
            "swing_high_1": levels["swing_high_1"],
            "swing_low_1": levels["swing_low_1"],
            "swing_label": levels["swing_label"],
            "market_structure_state": market_structure(levels["swing_label"]),
            "distance_from_swing_high": (close - levels["swing_high_1"]) / scale,
            "distance_from_swing_low": (close - levels["swing_low_1"]) / scale,
        },
        index=frame.index,
    )
