"""Price-action engines: what the candles themselves are saying.

These engines take a **candle frame** for one contract - already ordered, already
built from stored ticks - and return extra columns for it. Nothing here reads a
file, opens a connection or looks at the config YAML: the caller resolves
configuration and passes it in, which is what makes every engine testable with a
hand-built frame in three lines.

Implemented (Stage 3.2)

* :mod:`~core.price_action.candle_behaviour` - body/wick geometry, the
  relationship to the previous candle, and the named patterns (spec 11);
* :mod:`~core.price_action.swings` - swings with an explicit confirmation delay,
  and the per-bar swing levels (spec 15);
* :mod:`~core.price_action.structure` - ``HH/HL/LH/LL`` -> market-structure state
  and ATR-normalised distance to the latest swings (spec 15);
* :mod:`~core.price_action.levels` - today's running levels and the previous
  session's levels, with ATR-normalised distances (spec 12);
* :mod:`~core.price_action.level_events` - the level interaction state machine,
  its per-bar flags and its timestamped event log (spec 13);
* :mod:`~core.price_action.support_resistance` - candidate levels, their
  interaction statistics and the blended strength scores (spec 14).

Two house rules apply to all of them:

* **no look-ahead** - a bar may read its own fields and earlier bars only. Where
  a measurement is inherently forward-looking (a swing high needs later bars to
  exist) the value is attached to the bar at which it *became knowable*, never to
  an earlier one;
* **distances are in ATR units**, so a distance is comparable across contracts and
  strikes. The raw levels are stored in the same row, so nothing is lost.
"""

from __future__ import annotations

from core.price_action.candle_behaviour import (
    RELATIONSHIP_COLUMNS,
    SHAPE_COLUMNS,
    candle_features,
    candle_patterns,
    candle_relationships,
    candle_shape,
)
from core.price_action.level_events import (
    EVENT_SUFFIXES,
    INTERACTION_LEVEL_NAMES,
    LEVEL_EVENT_FLAG_COLUMNS,
    LEVEL_STATE_COLUMNS,
    interaction_levels,
    level_event_flags,
    level_event_history,
    level_events,
    level_interaction,
)
from core.price_action.levels import (
    LEVELS_FEATURE_COLUMNS,
    level_features,
    session_levels,
    trade_dates,
)
from core.price_action.structure import (
    STRUCTURE_FEATURE_COLUMNS,
    market_structure,
    structure_features,
)
from core.price_action.support_resistance import (
    CANDIDATE_STATISTIC_COLUMNS,
    SUPPORT_RESISTANCE_FEATURE_COLUMNS,
    candidate_levels,
    candidate_statistics,
    support_resistance_detail,
    support_resistance_features,
)
from core.price_action.swings import (
    SWING_LEVEL_COLUMNS,
    SwingPoint,
    recent_swings,
    swing_levels,
    swing_points,
)

__all__ = [
    "CANDIDATE_STATISTIC_COLUMNS",
    "EVENT_SUFFIXES",
    "INTERACTION_LEVEL_NAMES",
    "LEVELS_FEATURE_COLUMNS",
    "LEVEL_EVENT_FLAG_COLUMNS",
    "LEVEL_STATE_COLUMNS",
    "RELATIONSHIP_COLUMNS",
    "SHAPE_COLUMNS",
    "STRUCTURE_FEATURE_COLUMNS",
    "SUPPORT_RESISTANCE_FEATURE_COLUMNS",
    "SWING_LEVEL_COLUMNS",
    "SwingPoint",
    "candle_features",
    "candle_patterns",
    "candle_relationships",
    "candle_shape",
    "candidate_levels",
    "candidate_statistics",
    "interaction_levels",
    "level_event_flags",
    "level_event_history",
    "level_events",
    "level_features",
    "level_interaction",
    "market_structure",
    "recent_swings",
    "session_levels",
    "structure_features",
    "support_resistance_detail",
    "support_resistance_features",
    "swing_levels",
    "swing_points",
    "trade_dates",
]

