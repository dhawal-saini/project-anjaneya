"""Support and resistance engine (spec 14).

Spec 14 asks for candidate levels, for each candidate a set of interaction counts,
and then "configurable numerical support_strength and resistance_strength". It also
says, in as many words: "These are analytical features. Do not auto-convert into
trading signals." Nothing here votes, scores a trade, or looks at a label.

Candidates
----------
================================  ============================================
candidate                         source
================================  ============================================
``prev_day_open``/``high``/``low``/``close``
                                  the previous completed session (spec 12)
``today_open``                    this session's first bar's open
``today_high``/``today_low``      today's running extreme **before** the bar
``swing_high_1..n``/``swing_low_1..n``
                                  the ``levels.swing.max_swings`` most recent
                                  *confirmed* swings (spec 15)
================================  ============================================

Two deliberate choices, both to avoid inventing a second, differently-tuned
detector for the same thing:

* "local extremes" of spec 14 are the swing engine's job. A swing high *is* a local
  extreme, with ``left_bars``/``right_bars`` configurable - so the swings are fed
  in rather than re-pivoted here.
* the running high/low are the values *before* the bar, exactly as in
  :mod:`core.price_action.level_events`. A bar cannot touch a level it is
  currently creating; using the running value *including* the bar would make every
  new extreme a "touch", which is how a touch count stops meaning anything.

Statistics per candidate
------------------------
The interaction walk of :func:`core.price_action.level_events.level_interaction`
is reused verbatim, and its event flags are counted cumulatively - so
"touch count" in this module and ``TOUCHED`` in the feature row can never disagree.
Each statistic is defined in :func:`candidate_statistics`; the shortest version is
that ``touches`` count bars whose range reached the level, ``rejections`` count
touches that closed away from it, ``breakouts``/``breakdowns`` count crossings in
each direction, and ``failed_*`` count crossings that came straight back without
ever holding.

``reaction_strength`` is the largest excursion price has made away from the level
since it was first touched, in ATR units and signed by the side it went to. It is
a *running* statistic: at bar ``t`` it is computed from bars ``<= t`` only. How far
price travels *after* a signal is a label, and labels live in
:mod:`core.labels` - never here.

The counts are cumulative over the level's whole life, so a level that stays in
play for weeks accumulates tens of touches; the score saturates through
``n / (n + 1)`` instead of growing without bound. If a later stage wants a
recency-weighted count or a decayed score, that is a new, explicitly configured
column - not a silent change to these, which are meant to be readable in a report
as "how many times has this level been tested".

From statistics to strength
---------------------------
Five evidence terms, each already normalised to ``[0, 1]``, blended with the
weights from ``config/indicators.yaml`` (they must sum to 1.0, so the result is in
``[0, 1]`` as well):

===========  =================================  =============================
term         normalisation                      reads as
===========  =================================  =============================
touches      ``n / (n + 1)``                    0 -> 0, 1 -> 0.5, 3 -> 0.75
rejections   ``n / (n + 1)``                    ditto
holds        ``n / (n + 1)``                    ditto
recency      ``1 / (1 + bars_since_touch)``     touched now -> 1, never -> 0
distance     ``1 / (1 + |distance_atr|)``       1 ATR away -> 0.5
===========  =================================  =============================

``n / (n + 1)`` and ``1 / (1 + x)`` are used on purpose: they are bounded and
monotone without introducing a saturation constant that nobody measured. A level
with no evidence scores only its distance term, which is the honest answer - it is
nearby, and that is all that is known about it.

Which candidate becomes the feature
-----------------------------------
Per bar, every candidate is classified by where the level sits relative to the
close - at or below it is *support*, above it is *resistance* - and the **nearest**
candidate in each role becomes ``support_strength``/``resistance_strength`` (a tie
in distance goes to the better-tested of the two). So the two columns answer "how
well-tested is the support just under price, and the resistance just over it".

Nearest, not strongest, on purpose. Ranking by score alone is dominated by the
oldest, most-touched candidate and produces a near-constant number on every bar
(the long-lived swing levels always win), which carries almost no information
about *this* bar. The level price would reach first is the one being tested now,
and its score is what says how much evidence stands behind it. ``support_touch_count``
and friends come from that same winning candidate, so the strength and its evidence
always describe one level. No candidate in a role means ``NULL``, not a zero: "no
level below price" is an absence of input, and a zero strength would be a claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config_loader import LevelsConfig
from core.constants import DEFAULT_TIMEZONE
from core.indicators.series import as_float_series, require_pair
from core.price_action.level_events import interaction_levels, level_interaction
from core.price_action.levels import session_levels
from core.price_action.swings import recent_swings

__all__ = [
    "CANDIDATE_STATISTIC_COLUMNS",
    "SUPPORT_RESISTANCE_FEATURE_COLUMNS",
    "candidate_levels",
    "candidate_statistics",
    "support_resistance_detail",
    "support_resistance_features",
]

#: The six columns the ``features`` table carries for spec 14.
SUPPORT_RESISTANCE_FEATURE_COLUMNS = (
    "support_strength",
    "resistance_strength",
    "support_touch_count",
    "support_rejection_count",
    "resistance_touch_count",
    "resistance_rejection_count",
)

#: Every per-candidate statistic, in storage order.
CANDIDATE_STATISTIC_COLUMNS = (
    "touch_count",
    "rejection_count",
    "breakout_count",
    "breakdown_count",
    "failed_breakout_count",
    "failed_breakdown_count",
    "retest_count",
    "hold_count",
    "distance_atr",
    "reaction_strength",
    "bars_since_last_touch",
    "strength",
)

#: Event flags that are merely counted, mapped to the statistic they feed.
_COUNTED_EVENTS = {
    "touched": "touch_count",
    "rejected": "rejection_count",
    "crossed_above": "breakout_count",
    "crossed_below": "breakdown_count",
    "failed_breakout": "failed_breakout_count",
    "failed_breakdown": "failed_breakdown_count",
    "retested": "retest_count",
}

#: Weights whose evidence term is a count (``n / (n + 1)``).
_COUNT_WEIGHTS = {
    "touches": "touch_count",
    "rejections": "rejection_count",
    "holds": "hold_count",
}


def candidate_levels(
    frame: pd.DataFrame,
    *,
    config: LevelsConfig,
    levels: pd.DataFrame | None = None,
    swings: pd.DataFrame | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Every candidate level of spec 14, one column per candidate, per bar.

    Raw prices (never ATR-normalised - :func:`candidate_statistics` does that).
    A candidate is ``NaN`` at bars where the level is not known yet, which is the
    honest answer for the first session's previous-day levels and for swings that
    have not been confirmed.

    ``levels``/``swings`` let the caller hand in already-computed frames; the
    feature builder does exactly that so the three engines share one pass.
    """
    session = session_levels(frame, timezone_name=timezone_name) if levels is None else levels
    tracked = interaction_levels(frame, timezone_name=timezone_name, levels=session)
    swing_frame = recent_swings(frame, config=config.swing) if swings is None else swings

    columns: dict[str, pd.Series] = {
        "prev_day_open": session["level_previous_day_open"],
        "prev_day_high": tracked["prev_day_high"],
        "prev_day_low": tracked["prev_day_low"],
        "prev_day_close": session["level_previous_day_close"],
        "today_open": session["level_today_open"],
        "today_high": tracked["today_high"],
        "today_low": tracked["today_low"],
    }
    for name in swing_frame.columns:
        columns[str(name)] = swing_frame[name]
    return pd.DataFrame(columns, index=frame.index)



def _cumulative(flags: pd.Series, *, defined: pd.Series) -> pd.Series:
    """Running count of a boolean event column, ``NULL`` where the level is not known."""
    counts = np.cumsum(flags.fillna(False).to_numpy(dtype=bool))
    return pd.Series(counts, index=flags.index).astype("Int64").where(defined, other=pd.NA)


def _count_terms(counts: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """``n / (n + 1)`` for each count-based evidence term."""
    terms: dict[str, pd.Series] = {}
    for weight_name, statistic in _COUNT_WEIGHTS.items():
        values = counts[statistic].astype("float64")
        terms[weight_name] = values / (values + 1.0)
    return terms


def _strength(
    *,
    counts: dict[str, pd.Series],
    bars_since_last_touch: pd.Series,
    distance_atr: pd.Series,
    config: LevelsConfig,
) -> pd.Series:
    """Blend the five evidence terms with the configured weights (see module docs)."""
    weights = config.support_resistance.weights
    terms = _count_terms(counts)
    # A level nobody has tested yet gets no recency credit - 0, not "unknown":
    # "never touched" is a fact about the level, unlike "level not defined".
    terms["recency"] = (1.0 / (1.0 + bars_since_last_touch)).fillna(0.0)
    terms["distance"] = 1.0 / (1.0 + distance_atr.abs())
    blended = sum(getattr(weights, name) * term for name, term in terms.items())
    return blended.clip(lower=0.0, upper=1.0)


def candidate_statistics(
    frame: pd.DataFrame, *, level: Any, atr: Any, config: LevelsConfig
) -> pd.DataFrame:
    """Per-bar statistics for **one** candidate level (spec 14), index-aligned.

    =======================  ==================================================
    statistic                definition
    =======================  ==================================================
    ``touch_count``          bars whose range reached the level's touch band
    ``rejection_count``      touches that closed away from the level
    ``breakout_count``       bars that traded above the level after being at/below
    ``breakdown_count``      the mirror: traded below after being at/above
    ``failed_breakout_count`` breaks up that closed back down before a hold
    ``failed_breakdown_count`` the mirror
    ``retest_count``         bars that returned to the level while holding it
    ``hold_count``           bars that completed ``min_closes_to_hold`` closes beyond
    ``distance_atr``         ``(close - level) / atr``; positive = price above
    ``reaction_strength``    largest excursion away from the level since the first
                             touch, in ATR units, signed by the side it went to
    ``bars_since_last_touch`` bars since the level was last touched (``NULL`` if never)
    ``strength``             the blended ``[0, 1]`` score (see module docs)
    =======================  ==================================================

    Counts and ``strength`` are ``NULL`` at bars where the level or the ATR is not
    known - the same bars where :func:`core.price_action.level_events.level_interaction`
    reports ``NULL`` flags. Nothing is inferred from the future: every column at bar
    ``t`` uses bars ``<= t`` only.
    """
    close = as_float_series(frame["close"], name="close")
    levels = as_float_series(level, name="level")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, levels, names=("close", "level"))
    require_pair(close, ranges, names=("close", "atr"))

    defined = levels.notna() & ranges.notna() & (ranges > 0)
    interaction = level_interaction(
        frame, level=levels, atr=ranges, config=config.interaction
    )
    counts = {
        statistic: _cumulative(interaction[event], defined=defined)
        for event, statistic in _COUNTED_EVENTS.items()
    }
    held = interaction["held_above"].fillna(False) | interaction["held_below"].fillna(False)
    counts["hold_count"] = _cumulative(held, defined=defined)

    distance_atr = (close - levels) / ranges.where(ranges > 0)

    positions = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    touched = interaction["touched"].fillna(False)
    last_touch = positions.where(touched).ffill()
    bars_since_last_touch = positions - last_touch

    # Peak/trough of the ATR excursion since the level was first touched. Past only:
    # the excursion *after* this bar would be a label, and labels are not features.
    opened = touched.cummax().astype(bool)
    seen = distance_atr.where(opened)
    peaks = seen.cummax()
    troughs = seen.cummin()
    reaction = peaks.where(peaks.abs() >= troughs.abs(), troughs)

    statistics = pd.DataFrame(
        {
            **counts,
            "distance_atr": distance_atr,
            "reaction_strength": reaction.where(defined),
            "bars_since_last_touch": bars_since_last_touch.where(touched.cummax().astype(bool)),
        },
        index=frame.index,
    )
    statistics["strength"] = _strength(
        counts=counts,
        bars_since_last_touch=bars_since_last_touch,
        distance_atr=distance_atr,
        config=config,
    ).where(defined)
    return statistics[list(CANDIDATE_STATISTIC_COLUMNS)]



#: Two strengths closer than this are treated as equal, and proximity decides.
_TIE = 1.0e-12


def _pick_best(
    *,
    proximity: np.ndarray,
    strength: np.ndarray,
    name: str,
    best_proximity: np.ndarray,
    best_strength: np.ndarray,
    best_name: np.ndarray,
    payload: dict[str, np.ndarray],
    best_payload: dict[str, np.ndarray],
) -> None:
    """Keep this candidate where it is nearer (ties broken by strength)."""
    finite = np.isfinite(proximity)
    tied = np.abs(proximity - best_proximity) <= _TIE
    better = finite & ((proximity < best_proximity - _TIE) | (tied & (strength > best_strength)))
    if not better.any():
        return
    best_proximity[better] = proximity[better]
    best_strength[better] = strength[better]
    best_name[better] = name
    for key, values in payload.items():
        best_payload[key][better] = values[better]

def support_resistance_detail(
    frame: pd.DataFrame,
    *,
    atr: Any,
    config: LevelsConfig,
    candidates: pd.DataFrame | None = None,
    levels: pd.DataFrame | None = None,
    swings: pd.DataFrame | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """The six feature columns plus the name of each winning candidate.

    Same numbers as :func:`support_resistance_features`, with
    ``support_candidate``/``resistance_candidate`` added so a report (and a test)
    can say *which* level a strength came from without recomputing it. The extra
    two columns are not part of the stored feature row.
    """
    close = as_float_series(frame["close"], name="close")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, ranges, names=("close", "atr"))
    table = (
        candidate_levels(
            frame,
            config=config,
            levels=levels,
            swings=swings,
            timezone_name=timezone_name,
        )
        if candidates is None
        else candidates
    )

    size = len(frame)
    best = {
        role: {
            "strength": np.full(size, -np.inf),
            "proximity": np.full(size, np.inf),
            "name": np.full(size, None, dtype=object),
            "touch_count": np.zeros(size, dtype=np.float64),
            "rejection_count": np.zeros(size, dtype=np.float64),
        }
        for role in ("support", "resistance")
    }
    defined_atr = (ranges.notna() & (ranges > 0)).to_numpy(dtype=bool)
    close_values = close.to_numpy(dtype=np.float64)

    for column in table.columns:
        level = as_float_series(table[column], name=str(column))
        if not level.notna().any():
            # A candidate that never becomes defined (a swing that never confirms)
            # contributes nothing, and skipping it costs nothing.
            continue
        statistics = candidate_statistics(frame, level=level, atr=ranges, config=config)
        strength = statistics["strength"].to_numpy(dtype=np.float64)
        level_values = level.to_numpy(dtype=np.float64)
        distance = np.abs(level_values - close_values)
        proximity = np.where(np.isfinite(strength), distance, np.nan)
        eligible = defined_atr & np.isfinite(strength)
        # At or below price is support, strictly above is resistance (a level exactly
        # at the close is the floor being stood on, not a ceiling).
        for role, mask in (
            ("support", eligible & (level_values <= close_values)),
            ("resistance", eligible & (level_values > close_values)),
        ):
            _pick_best(
                proximity=np.where(mask, proximity, np.nan),
                strength=strength,
                name=str(column),
                best_proximity=best[role]["proximity"],
                best_strength=best[role]["strength"],
                best_name=best[role]["name"],
                payload={
                    "touch_count": statistics["touch_count"]
                    .astype("float64")
                    .to_numpy(dtype=np.float64),
                    "rejection_count": statistics["rejection_count"]
                    .astype("float64")
                    .to_numpy(dtype=np.float64),
                },
                best_payload=best[role],
            )

    decided = {role: np.isfinite(best[role]["proximity"]) for role in best}

    def _counts(role: str, key: str) -> pd.Series:
        counted = pd.Series(best[role][key], index=frame.index).round().astype("Int64")
        return counted.where(decided[role], other=pd.NA)

    def _strength(role: str) -> pd.Series:
        values = np.where(decided[role], best[role]["strength"], np.nan)
        return pd.Series(values, index=frame.index, dtype="float64")

    return pd.DataFrame(
        {
            "support_strength": _strength("support"),
            "resistance_strength": _strength("resistance"),
            "support_touch_count": _counts("support", "touch_count"),
            "support_rejection_count": _counts("support", "rejection_count"),
            "resistance_touch_count": _counts("resistance", "touch_count"),
            "resistance_rejection_count": _counts("resistance", "rejection_count"),
            "support_candidate": pd.Series(best["support"]["name"], index=frame.index),
            "resistance_candidate": pd.Series(best["resistance"]["name"], index=frame.index),
        },
        index=frame.index,
    )

def support_resistance_features(
    frame: pd.DataFrame,
    *,
    atr: Any,
    config: LevelsConfig,
    candidates: pd.DataFrame | None = None,
    levels: pd.DataFrame | None = None,
    swings: pd.DataFrame | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """The six ``support_*``/``resistance_*`` columns of the feature row (spec 14).

    Columns are spelled exactly as ``core/storage/schema.sql`` spells them. Cost is
    linear in (candidates x bars), so the builder calls it per instrument and
    timeframe rather than over every contract at once.
    """
    detail = support_resistance_detail(
        frame,
        atr=atr,
        config=config,
        candidates=candidates,
        levels=levels,
        swings=swings,
        timezone_name=timezone_name,
    )
    return detail[list(SUPPORT_RESISTANCE_FEATURE_COLUMNS)]

