"""Price-level interaction state machine (spec 13).

Spec 13 asks for a state engine over "important levels" with thirteen states, and
for the two worked examples in the spec to be detectable:

    149 -> 151 -> 152 -> 151 -> 153   approach, touch, cross above, close above,
                                      hold above, retest, continuation
    149 -> 151 -> 148 -> 147          break above, rejection, failed breakout,
                                      back below

Both fall out of one forward walk per level. The walk is the whole engine, and it
is deliberately *forward only*: at bar ``i`` it may look at bar ``i`` and at state
carried from bars ``< i``. Nothing is centred, nothing is revised later, which is
what "each event is timestamped and non-repainting" (spec 13) means in practice -
and ``tests/unit/test_price_action.py`` proves it by appending bars and checking
that no earlier value moved.

The levels
----------
Four level series are tracked, spelled exactly as the ``features`` table spells
their state columns:

================================  ==========================================
column                            level
================================  ==========================================
``event_prev_day_high_state``     the previous session's high
``event_prev_day_low_state``      the previous session's low
``event_today_high_state``        today's running high **before this bar**
``event_today_low_state``         today's running low **before this bar**
================================  ==========================================

The last two are the reason this is not degenerate. Compared against *its own*
running high a bar could only ever be at or below the level - price cannot be
above a high it is currently setting - so the level is the running extreme as it
stood **before** the bar, and the state then says something real: this bar is
making a new high (``CLOSED_ABOVE``/``HELD_ABOVE``) or it failed to
(``FAILED_BREAKOUT``).

Bands and thresholds
--------------------
Bands come from ``config/indicators.yaml`` (``levels.interaction``) and are ATR
multiples, so they mean the same thing on NIFTY and on a 4-point option:

* ``touch_atr_fraction`` (0.05) - the bar's range came within 5% of an ATR of the
  level: it *touched* it;
* ``approaching_atr_fraction`` (0.25) - within a quarter ATR: *approaching*;
* ``min_closes_to_hold`` (2) - closes beyond the level needed before a break is
  called *held* rather than broken-and-back.

The decision table
------------------
Per bar, first match wins. ``touch``/``cross``/``approach`` are conditions of the
bar itself; the rest depend on carried state. ``up`` and ``down`` below mean
"above the level" and "below the level"; the sides are mirrored.

===  ==========================================================  ====================
#    condition                                                   state
===  ==========================================================  ====================
1    level or ATR not known at this bar                          ``NULL`` (no invention)
2    first bar of the level's life (or a session roll)           ``NOT_TESTED``
3    holding up: close returns below                             ``CLOSED_BELOW``
4    holding up: bar touched the level and closed above          ``RETESTED``
5    holding up: anything else                                   ``HELD_ABOVE``
6    break pending up, close below (hold never reached)          ``FAILED_BREAKOUT``
7    break pending up, closes above reach ``min_closes_to_hold`` ``HELD_ABOVE``
8    break pending up, close above (hold not reached yet)        ``CLOSED_ABOVE``
9    break pending up, wick above only                           ``CROSSED_ABOVE``
10   bar closes beyond the level for the first time              ``CLOSED_ABOVE``
11   bar trades beyond the level but closes back                 ``CROSSED_ABOVE``
12   bar touched the level, close still inside the touch band    ``TOUCHED``
13   bar touched the level, close moved away from it             ``REJECTED``
14   bar came within the approach band                           ``APPROACHING``
15   nothing of the above                                        carry the last state
===  ==========================================================  ====================

Two details of the table are worth stating plainly. On the first bar of a level's
life - the frame's first bar at which the level is defined, or a new session for a
session level - the bar's own **open** stands in for the missing previous close, so
an intrabar cross on that bar is still a real cross. And ``NOT_TESTED`` is the
honest state when price is already beyond the level but never crossed it: the level
genuinely has not been tested, it was gapped over.

A **break is only a failure if the hold was never established**: cross, close
back, done - ``FAILED_BREAKOUT``. If the hold *was* established and price then
closes back through, that is a genuine reversal (``CLOSED_BELOW``), not a failure.
That single rule is the difference between "the level rejected the move" and "the
level broke and turned", and it is why ``min_closes_to_hold`` exists.

The state column is the *most recent* state, so it persists until something
replaces it (price drifting away does not erase the record). The per-bar flags
below are the precise truth; the state is their human-readable summary.

Flags
-----
:func:`level_interaction` also returns eleven boolean event columns per level -
``touched``, ``crossed_above``/``_below``, ``closed_above``/``_below``,
``held_above``/``_below``, ``rejected``, ``retested``,
``failed_breakout``/``_breakdown`` - which is the "touched/crossed/closed/
rejected/broke/held flags" of spec 18 and the evidence
:mod:`core.price_action.support_resistance` counts. ``NULL`` means the level or
the ATR was not known, which is not the same as "it did not happen" - that is
``False``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config_loader import LevelInteractionConfig, LevelsConfig
from core.constants import DEFAULT_TIMEZONE
from core.indicators.series import as_float_series, require_pair
from core.price_action.levels import session_levels, trade_dates
from core.schemas import LevelState

__all__ = [
    "EVENT_SUFFIXES",
    "INTERACTION_LEVEL_NAMES",
    "LEVEL_EVENT_FLAG_COLUMNS",
    "LEVEL_STATE_COLUMNS",
    "interaction_levels",
    "level_event_flags",
    "level_event_history",
    "level_events",
    "level_interaction",
]

#: Level keys, spelled as the ``features`` table spells them.
INTERACTION_LEVEL_NAMES = ("prev_day_high", "prev_day_low", "today_high", "today_low")

#: The four state columns of the unified feature row (spec 18).
LEVEL_STATE_COLUMNS = tuple(f"event_{name}_state" for name in INTERACTION_LEVEL_NAMES)

#: Event flags, ordered as :class:`core.schemas.LevelState` orders the states.
EVENT_SUFFIXES = (
    "touched",
    "crossed_above",
    "crossed_below",
    "closed_above",
    "closed_below",
    "held_above",
    "held_below",
    "rejected",
    "retested",
    "failed_breakout",
    "failed_breakdown",
)

#: Every flag column :func:`level_event_flags` produces, in storage order.
LEVEL_EVENT_FLAG_COLUMNS = tuple(
    f"event_{name}_{suffix}" for name in INTERACTION_LEVEL_NAMES for suffix in EVENT_SUFFIXES
)

#: Levels that are replaced when a new session begins (the tracker restarts).
_SESSION_LEVELS = frozenset({"prev_day_high", "prev_day_low"})


def _prior_session_extremes(frame: pd.DataFrame, dates: pd.Series) -> dict[str, pd.Series]:
    """Today's running high/low *before* each bar, reset at every session.

    ``shift(1)`` is applied **inside** each session group, so a session's first bar
    has no level yet instead of inheriting the previous session's extreme.
    """
    running_high = as_float_series(frame["high"], name="high").groupby(dates).cummax()
    running_low = as_float_series(frame["low"], name="low").groupby(dates).cummin()
    return {
        "today_high": running_high.groupby(dates).shift(1),
        "today_low": running_low.groupby(dates).shift(1),
    }


def interaction_levels(
    frame: pd.DataFrame,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    levels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The four level series the interaction engine tracks, one per bar.

    ``prev_day_high``/``prev_day_low`` are the previous completed session's
    extremes (spec 12) and ``today_high``/``today_low`` are the running extremes
    **as they stood before the bar** - see the module docstring for why the prior
    value is the only one a bar can cross.

    ``levels`` may be passed in when the caller already built the session levels
    (the feature builder does), saving one pass over the frame.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=list(INTERACTION_LEVEL_NAMES), index=frame.index, dtype="float64"
        )
    raw = session_levels(frame, timezone_name=timezone_name) if levels is None else levels
    dates = trade_dates(frame, timezone_name=timezone_name)
    prior = _prior_session_extremes(frame, dates)
    return pd.DataFrame(
        {
            "prev_day_high": raw["level_previous_day_high"],
            "prev_day_low": raw["level_previous_day_low"],
            "today_high": prior["today_high"],
            "today_low": prior["today_low"],
        },
        index=frame.index,
    )

def _walk(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    level: np.ndarray,
    atr: np.ndarray,
    *,
    config: LevelInteractionConfig,
    reset: np.ndarray,
    opened: np.ndarray,
) -> tuple[list[str | None], dict[str, np.ndarray], np.ndarray]:
    """Run the state machine over one level, bar by bar. See the module docstring.

    ``opened`` is the bar's open. It is used only on the first bar of a level's
    life: there is no earlier bar *of this level* to have crossed from, and the
    open is the earliest real price of that life.
    """
    size = len(close)
    states: list[str | None] = [None] * size
    flags = {suffix: np.zeros(size, dtype=bool) for suffix in EVENT_SUFFIXES}
    known = np.zeros(size, dtype=bool)

    state: str | None = None
    side = 0  # side of the last known close: +1 above, -1 below, 0 unknown
    held = 0  # +1 = HELD_ABOVE, -1 = HELD_BELOW
    pending_up = pending_down = False
    closes_above = closes_below = 0
    have_level = False
    previous_level = np.nan
    not_tested = LevelState.NOT_TESTED.value

    for index in range(size):
        bar_level = float(level[index])
        bar_range = float(atr[index])
        available = bool(np.isfinite(bar_level) and np.isfinite(bar_range) and bar_range > 0)
        if not available:
            # The level (or the scale that makes it comparable) is undefined here:
            # the state cannot be claimed, so it is NULL. The tracker is untouched.
            states[index] = None
            previous_level = np.nan
            continue

        fresh = not have_level or bool(reset[index]) or not np.isfinite(previous_level)
        if fresh:
            # First bar of this level's life, or of a new session: a brand new level
            # has not been tested yet and no state may be inherited from the old one.
            side = 0
            held = 0
            pending_up = pending_down = False
            closes_above = closes_below = 0
            state = not_tested
        have_level = True
        previous_level = bar_level

        bar_high = float(high[index])
        bar_low = float(low[index])
        bar_close = float(close[index])
        touch_band = config.touch_atr_fraction * bar_range
        approach_band = config.approaching_atr_fraction * bar_range

        touched = bool(bar_low <= bar_level + touch_band and bar_high >= bar_level - touch_band)
        approaching = bool(
            bar_low <= bar_level + approach_band and bar_high >= bar_level - approach_band
        )
        close_above = bar_close > bar_level
        close_below = bar_close < bar_level
        origin = side
        if fresh:
            # No prior close exists for this level, so the bar's own open stands in
            # for it - an intrabar cross on this bar is still a real cross.
            bar_open = float(opened[index])
            origin = 1 if bar_open > bar_level else (-1 if bar_open < bar_level else 0)
        crossed_up = bool(bar_high > bar_level and origin <= 0)
        crossed_down = bool(bar_low < bar_level and origin >= 0)

        flags["touched"][index] = touched
        flags["crossed_above"][index] = crossed_up
        flags["crossed_below"][index] = crossed_down

        if held > 0:
            if close_below:
                state = LevelState.CLOSED_BELOW.value
                flags["closed_below"][index] = True
                held, pending_up, pending_down = 0, False, True
                closes_below = 1
            elif touched and close_above:
                state = LevelState.RETESTED.value
                flags["retested"][index] = True
            else:
                state = LevelState.HELD_ABOVE.value
        elif held < 0:
            if close_above:
                state = LevelState.CLOSED_ABOVE.value
                flags["closed_above"][index] = True
                held, pending_up, pending_down = 0, True, False
                closes_above = 1
            elif touched and close_below:
                state = LevelState.RETESTED.value
                flags["retested"][index] = True
            else:
                state = LevelState.HELD_BELOW.value
        elif pending_up:
            if close_above:
                closes_above += 1
                if closes_above >= config.min_closes_to_hold:
                    state = LevelState.HELD_ABOVE.value
                    flags["held_above"][index] = True
                    held, pending_up, closes_above = 1, False, 0
                else:
                    state = LevelState.CLOSED_ABOVE.value
                    flags["closed_above"][index] = True
            elif close_below:
                # Crossed, never held, back down: the breakout failed.
                state = LevelState.FAILED_BREAKOUT.value
                flags["failed_breakout"][index] = True
                pending_up, closes_above = False, 0
            elif crossed_up:
                state = LevelState.CROSSED_ABOVE.value
            # else: carry the state - the break is open, price has not decided.
        elif pending_down:
            if close_below:
                closes_below += 1
                if closes_below >= config.min_closes_to_hold:
                    state = LevelState.HELD_BELOW.value
                    flags["held_below"][index] = True
                    held, pending_down, closes_below = -1, False, 0
                else:
                    state = LevelState.CLOSED_BELOW.value
                    flags["closed_below"][index] = True
            elif close_above:
                state = LevelState.FAILED_BREAKDOWN.value
                flags["failed_breakdown"][index] = True
                pending_down, closes_below = False, 0
            elif crossed_down:
                state = LevelState.CROSSED_BELOW.value
        elif crossed_up:
            pending_up = True
            if close_above:
                closes_above = 1
                flags["closed_above"][index] = True
                if closes_above >= config.min_closes_to_hold:
                    state = LevelState.HELD_ABOVE.value
                    flags["held_above"][index] = True
                    held, pending_up, closes_above = 1, False, 0
                else:
                    state = LevelState.CLOSED_ABOVE.value
            else:
                state = LevelState.CROSSED_ABOVE.value
        elif crossed_down:
            pending_down = True
            if close_below:
                closes_below = 1
                flags["closed_below"][index] = True
                if closes_below >= config.min_closes_to_hold:
                    state = LevelState.HELD_BELOW.value
                    flags["held_below"][index] = True
                    held, pending_down, closes_below = -1, False, 0
                else:
                    state = LevelState.CLOSED_BELOW.value
            else:
                state = LevelState.CROSSED_BELOW.value
        elif touched:
            if abs(bar_close - bar_level) <= touch_band:
                state = LevelState.TOUCHED.value
            else:
                state = LevelState.REJECTED.value
                flags["rejected"][index] = True
        elif approaching:
            state = LevelState.APPROACHING.value

        if close_above:
            side = 1
        elif close_below:
            side = -1
        known[index] = True
        states[index] = state

    return states, flags, known


def level_interaction(
    frame: pd.DataFrame,
    *,
    level: Any,
    atr: Any,
    config: LevelInteractionConfig,
    reset: Any = None,
) -> pd.DataFrame:
    """State and event flags for **one** level, one row per bar.

    Columns: ``state`` first (a :class:`core.schemas.LevelState` value, or ``None``
    when the level or the ATR is unknown at that bar), then the eleven flags of
    :data:`EVENT_SUFFIXES` as nullable booleans.

    ``reset`` is an optional boolean series marking bars at which the tracker
    starts over - the first bar of a new session, for levels that belong to a
    session. It exists because the two kinds of level differ: a *previous-day*
    level is a new level every morning (yesterday's hold must not leak into
    today), while today's running extreme moves *because price moved* and the
    machine has to see the cross that caused the move.
    """
    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            raise ValueError(f"level interaction needs a {column!r} column; expected OHLC")
    close = as_float_series(frame["close"], name="close")
    levels = as_float_series(level, name="level")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, levels, names=("close", "level"))
    require_pair(close, ranges, names=("close", "atr"))

    if frame.empty:
        empty = pd.DataFrame(index=frame.index)
        empty["state"] = pd.Series([], dtype="object", index=frame.index)
        for suffix in EVENT_SUFFIXES:
            empty[suffix] = pd.Series([], dtype="boolean", index=frame.index)
        return empty

    if reset is None:
        reset_mask = np.zeros(len(frame), dtype=bool)
    else:
        reset_series = pd.Series(reset, index=frame.index, dtype="boolean")
        reset_mask = reset_series.fillna(True).to_numpy(dtype=bool)

    states, flags, known = _walk(
        as_float_series(frame["high"], name="high").to_numpy(),
        as_float_series(frame["low"], name="low").to_numpy(),
        close.to_numpy(),
        levels.to_numpy(),
        ranges.to_numpy(),
        config=config,
        reset=reset_mask,
        opened=as_float_series(frame["open"], name="open").to_numpy(),
    )
    known_series = pd.Series(known, index=frame.index)
    result = pd.DataFrame(index=frame.index)
    result["state"] = pd.Series(states, index=frame.index, dtype="object")
    for suffix in EVENT_SUFFIXES:
        result[suffix] = pd.Series(flags[suffix], index=frame.index, dtype="boolean").where(
            known_series, other=pd.NA
        )
    return result


def _session_reset_mask(frame: pd.DataFrame, *, timezone_name: str) -> pd.Series:
    """``True`` at the first bar of every session - where a level's life begins."""
    dates = trade_dates(frame, timezone_name=timezone_name)
    return dates.ne(dates.shift(1)).fillna(True)


def _each_level(
    frame: pd.DataFrame,
    *,
    atr: Any,
    config: LevelsConfig,
    levels: pd.DataFrame | None,
    timezone_name: str,
) -> dict[str, pd.DataFrame]:
    """One :func:`level_interaction` result per tracked level."""
    resolved = interaction_levels(frame, timezone_name=timezone_name, levels=levels)
    reset = _session_reset_mask(frame, timezone_name=timezone_name)
    return {
        name: level_interaction(
            frame,
            level=resolved[name],
            atr=atr,
            config=config.interaction,
            reset=reset if name in _SESSION_LEVELS else None,
        )
        for name in INTERACTION_LEVEL_NAMES
    }


def level_events(
    frame: pd.DataFrame,
    *,
    atr: Any,
    config: LevelsConfig,
    levels: pd.DataFrame | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """The four ``event_*_state`` columns of the unified feature row (spec 18).

    One call, columns named exactly as ``core/storage/schema.sql`` spells them, so
    the feature builder maps them across without a translation table that could
    drift out of step with the schema.

    The frame should start no later than the first bar of the session being
    analysed (one session earlier is better): the state at a session's first bar
    is only meaningful if the bars that led to it are present.
    """
    if frame.empty:
        return pd.DataFrame(
            {column: pd.Series([], dtype="object") for column in LEVEL_STATE_COLUMNS},
            index=frame.index,
        )
    interactions = _each_level(
        frame, atr=atr, config=config, levels=levels, timezone_name=timezone_name
    )
    return pd.DataFrame(
        {
            f"event_{name}_state": interactions[name]["state"]
            for name in INTERACTION_LEVEL_NAMES
        },
        index=frame.index,
    )


def level_event_flags(
    frame: pd.DataFrame,
    *,
    atr: Any,
    config: LevelsConfig,
    levels: pd.DataFrame | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Every per-bar event flag, four levels wide (spec 18's "flags").

    These are the columns support/resistance counts are built from and the ones a
    report reads. Only the four ``*_state`` columns go into the stored feature
    row; the flags stay available so a research question can be answered without
    re-deriving the walk.
    """
    if frame.empty:
        return pd.DataFrame(
            {column: pd.Series([], dtype="boolean") for column in LEVEL_EVENT_FLAG_COLUMNS},
            index=frame.index,
        )
    interactions = _each_level(
        frame, atr=atr, config=config, levels=levels, timezone_name=timezone_name
    )
    return pd.DataFrame(
        {
            f"event_{name}_{suffix}": interactions[name][suffix]
            for name in INTERACTION_LEVEL_NAMES
            for suffix in EVENT_SUFFIXES
        },
        index=frame.index,
    )


def level_event_history(
    frame: pd.DataFrame,
    *,
    atr: Any,
    config: LevelsConfig,
    levels: pd.DataFrame | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Timestamped event log: one row per (event, level, bar).

    Columns ``timestamp``, ``level``, ``state`` and ``event``. This is spec 13's
    "each event is timestamped" made literal - the per-bar flag columns say *that*
    something happened, this says *when*, and lets a report list a day's level
    history in order. Sorted by ``(timestamp, level, event)`` so two runs over the
    same data produce identical output.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=["timestamp", "level", "state", "event"], index=frame.index[:0]
        )
    interactions = _each_level(
        frame, atr=atr, config=config, levels=levels, timezone_name=timezone_name
    )
    stamps = frame["timestamp"].to_numpy() if "timestamp" in frame.columns else None
    records: list[tuple[Any, str, str | None, str]] = []
    for name in INTERACTION_LEVEL_NAMES:
        interaction = interactions[name]
        states = interaction["state"].to_numpy()
        for suffix in EVENT_SUFFIXES:
            fired = np.flatnonzero(interaction[suffix].fillna(False).to_numpy(dtype=bool))
            for position in fired:
                records.append(
                    (
                        None if stamps is None else stamps[int(position)],
                        name,
                        states[int(position)],
                        suffix,
                    )
                )
    history = pd.DataFrame(records, columns=["timestamp", "level", "state", "event"])
    if not history.empty:
        history = history.sort_values(
            ["timestamp", "level", "event"], kind="stable", ignore_index=True
        )
    return history

