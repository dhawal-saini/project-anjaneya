"""NIFTY trend engine: the MA measurements, the role scores and the composite (specs 8-9).

Spec 8 asks for a lot of per-average measurements; spec 9 collapses them into one
reading of the market. This module is that two-step in code, with the collapse
written down explicitly, because "how do eight moving averages become one number"
is the part a research report has to be able to defend.

Step 1 - measure every configured average (:func:`ma_metrics`)
    For each average of a combination: its value, which side of it price closed
    on, its raw slope (points per bar), its ATR-normalised slope, and its distance
    from price in ATR units. None of that is a judgement, it is arithmetic.

Step 2 - vote, then average the votes (:func:`role_score`)
    Each combination member carries a role (fast / medium / long). A role scores
    the mean of the evidence families *that role can express*:

    =========  =================================================  =============================
    family     value                                              reads as
    =========  =================================================  =============================
    position   mean of ``tanh((close-MA)/ATR)`` over the role      is price above these averages?
    slope      mean of ``tanh(normalized_slope)`` over the role     are these averages rising?
    stack      ``ma_alignment`` of the role's MAs, fastest first    is the stack in order?
    =========  =================================================  =============================

    ``tanh`` is the only scale choice in this file, and it exists to avoid having
    to make one: it maps any ATR-normalised quantity into ``(-1, +1)``
    monotonically, so the sign is the direction, the magnitude is the conviction,
    and no arbitrary "saturation constant" has to be configured or explained. A
    role with a single average has no stack, so it votes with position and slope
    only - each role score is the mean of the families that role actually has.

Step 3 - weigh the roles (:func:`combination_score`)

    ``composite = sum(weight * role_score) / sum(weight)``

    over the roles defined at that bar. The renormalisation matters: combination 3
    defines no medium average, so a plain weighted sum would leave its composite
    permanently ``NaN``. The weights come from ``nifty_scores.weights`` and must
    sum to 1.0 (the config loader validates that); the renormalisation only ever
    redistributes them across the roles a combination actually has.

``nifty_trend_acceleration`` is then the k-bar change of the composite ("is the
reading strengthening?") and ``nifty_trend_persistence`` is the indicator
library's persistence of the composite's direction - spec 9's two extra columns.

Which combination the feature row carries
    ``combination_1`` is primary (:func:`trend_features`) because it is the one
    the spec calls "full"; combinations 2 and 3 are computed alongside it by
    :func:`combination_scores` so the three can be compared over the same bars.
    Promoting a different combination changes the meaning of six persisted
    columns, so it is a feature-version decision rather than a parameter tweak -
    which is why the default is stated here and not hidden in a config file.

No look-ahead
    Every average comes from the indicator library on a past-only window, and every
    aggregate is a pointwise mean of those columns, so a bar's score reads its own
    bar and earlier bars only. The tests append bars and assert that earlier scores
    do not move.

``NaN`` means "not enough history for this reading yet"
    A 2056-bar SMA needs 2056 bars; until then its vote is undefined and the score
    that depends on it is ``NULL``. Nothing is published from a partial window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config_loader import IndicatorsConfig, MovingAverageSpec
from core.indicators.moving_averages import ma_alignment, ma_source, moving_average
from core.indicators.persistence import trend_persistence
from core.indicators.series import as_float_series, require_pair
from core.indicators.slope import difference, normalized_slope, slope
from core.indicators.spread import spread, spread_velocity

__all__ = [
    "MA_METRIC_SUFFIXES",
    "NIFTY_SCORE_ROLES",
    "NIFTY_TREND_FEATURE_COLUMNS",
    "combination_metrics",
    "combination_score",
    "combination_scores",
    "ma_metrics",
    "ma_table",
    "role_score",
    "trend_features",
]

#: The three score roles of spec 9, in the order the feature row stores them.
NIFTY_SCORE_ROLES = ("fast", "medium", "long")

#: The six columns the ``features`` table carries for spec 9.
NIFTY_TREND_FEATURE_COLUMNS = (
    "nifty_fast_score",
    "nifty_medium_score",
    "nifty_long_score",
    "nifty_composite_trend_score",
    "nifty_trend_acceleration",
    "nifty_trend_persistence",
)

#: Measurements produced per configured average, in the order :func:`ma_metrics`
#: lays them out. ``<name><suffix>`` is the column spelling.
MA_METRIC_SUFFIXES = (
    "_average",
    "_price_above",
    "_slope",
    "_normalized_slope",
    "_distance_atr",
)

#: The three combinations of spec 8, as ``config/indicators.yaml`` names them.
_COMBINATIONS = ("combination_1", "combination_2", "combination_3")


def _combination_specs(config: IndicatorsConfig, combination: str) -> list[MovingAverageSpec]:
    """The configured averages of one combination, in config order."""
    if combination not in _COMBINATIONS:
        raise ValueError(
            f"Unknown MA combination {combination!r}; expected one of {', '.join(_COMBINATIONS)}"
        )
    return list(getattr(config.nifty_moving_averages, combination))


def _fastest_first(specs: list[MovingAverageSpec]) -> list[MovingAverageSpec]:
    """Order a role's averages fastest first.

    Sorting is by period only, and :func:`sorted` is stable, so two averages of the
    same period (``ema_8_low`` and ``ema_8_high``) keep their config order instead
    of an order that depends on the interpreter.
    """
    return sorted(specs, key=lambda spec: spec.period)


def _mean_of(votes: list[pd.Series]) -> pd.Series:
    """Pointwise mean of equal-weight votes, ``NaN`` while any vote is undefined."""
    if not votes:
        return pd.Series(dtype="float64")
    return pd.concat(votes, axis=1).mean(axis=1, skipna=False)


def ma_table(frame: pd.DataFrame, *, specs: list[MovingAverageSpec]) -> pd.DataFrame:
    """One column per configured average, named exactly as the config names it.

    The names are the config's (``ema_8_low``, ``sma_2056_high``, ...), so a
    research table can be read against ``config/indicators.yaml`` line by line.
    """
    columns = {
        spec.name: moving_average(
            ma_source(frame, spec.source), ma_type=spec.type, period=spec.period
        )
        for spec in specs
    }
    return pd.DataFrame(columns, index=frame.index, dtype="float64")


def ma_metrics(
    frame: pd.DataFrame,
    *,
    specs: list[MovingAverageSpec],
    atr: Any,
    slope_k: int,
) -> pd.DataFrame:
    """Spec 8's per-average measurements, for one combination.

    Columns are ``<name>_average``, ``<name>_price_above`` (``+1``/``-1``/``0``),
    ``<name>_slope`` (points per bar), ``<name>_normalized_slope`` and
    ``<name>_distance_atr`` (``(close - MA) / ATR``).

    This table is what a trend report shows and what the scores are built from; it
    is deliberately *not* part of the stored feature row, which keeps one score per
    role rather than thirty columns of moving averages.
    """
    if "close" not in frame.columns:
        raise ValueError("trend computation needs a 'close' column")
    close = as_float_series(frame["close"], name="close")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, ranges, names=("close", "atr"))

    averages = ma_table(frame, specs=specs)
    columns: dict[str, pd.Series] = {}
    for spec in specs:
        average = averages[spec.name]
        columns[f"{spec.name}_average"] = average
        columns[f"{spec.name}_price_above"] = np.sign(close - average)
        columns[f"{spec.name}_slope"] = slope(average, k=slope_k)
        columns[f"{spec.name}_normalized_slope"] = normalized_slope(average, atr=ranges, k=slope_k)
        columns[f"{spec.name}_distance_atr"] = ((close - average) / ranges).where(ranges > 0)
    return pd.DataFrame(columns, index=frame.index, dtype="float64")


def _undefined(index: pd.Index) -> pd.Series:
    """An all-``NaN`` float column - "this reading does not exist for this frame"."""
    return pd.Series(np.nan, index=index, dtype="float64")


def role_score(
    frame: pd.DataFrame,
    *,
    specs: list[MovingAverageSpec],
    role: str,
    atr: Any,
    slope_k: int,
) -> pd.Series:
    """One role's score in ``[-1, +1]``, from the evidence families of spec 8.

    See the module docstring for the three families; a role with a single average
    votes without the stack term. A role the combination does not define (spec 8's
    combination 3 has no medium average) returns an all-``NaN`` column rather than
    raising, because the stored feature row still needs that column to exist.
    """
    if role not in NIFTY_SCORE_ROLES:
        raise ValueError(f"Unknown role {role!r}; expected one of {', '.join(NIFTY_SCORE_ROLES)}")
    members = [spec for spec in specs if spec.role == role]
    if not members:
        return _undefined(frame.index)

    metrics = ma_metrics(frame, specs=members, atr=atr, slope_k=slope_k)
    votes = [
        _mean_of([np.tanh(metrics[f"{spec.name}_distance_atr"]) for spec in members]),
        _mean_of([np.tanh(metrics[f"{spec.name}_normalized_slope"]) for spec in members]),
    ]
    if len(members) > 1:
        ordered = _fastest_first(members)
        votes.append(
            ma_alignment(*[metrics[f"{spec.name}_average"] for spec in ordered]).rename("stack")
        )
    return _mean_of(votes)


def _weighted_composite(scores: dict[str, pd.Series], *, weights: Any) -> pd.Series:
    """Weight-renormalised mean of the role scores defined at each bar.

    ``sum(weight * score) / sum(weight)`` over the roles that have a value: the
    denominator drops the weight of every undefined role, which is what lets
    combination 3 (no medium average) produce a composite at all.

    Both sums skip ``NaN`` deliberately. Plain series addition would propagate the
    ``NaN`` of one undefined role into every bar and the composite would be empty
    forever on any combination that is missing a role - which is exactly the
    combination the renormalisation exists for.
    """
    weighted = pd.DataFrame(
        {role: getattr(weights, role) * score for role, score in scores.items()}
    ).sum(axis=1, skipna=True)
    total = pd.DataFrame(
        {role: getattr(weights, role) * score.notna() for role, score in scores.items()}
    ).sum(axis=1)
    return (weighted / total).where(total > 0)


def _role_scores(
    frame: pd.DataFrame, *, config: IndicatorsConfig, specs: list[MovingAverageSpec], atr: Any
) -> dict[str, pd.Series]:
    """The three role scores of one combination, in :data:`NIFTY_SCORE_ROLES` order."""
    return {
        role: role_score(frame, specs=specs, role=role, atr=atr, slope_k=config.slope.k)
        for role in NIFTY_SCORE_ROLES
    }


def combination_score(
    frame: pd.DataFrame,
    *,
    config: IndicatorsConfig,
    atr: Any,
    combination: str = "combination_1",
) -> pd.Series:
    """The composite trend score of one MA combination (spec 9), in ``[-1, +1]``.

    The composite is only published once at least one role is defined, so on a
    young frame it is ``NaN`` rather than a reading built out of one average.

    ``atr`` must share ``frame``'s index - the caller computes one ATR per
    instrument and timeframe and shares it (see :mod:`core.indicators.atr`).
    """
    specs = _combination_specs(config, combination)
    scores = _role_scores(frame, config=config, specs=specs, atr=atr)
    return _weighted_composite(scores, weights=config.nifty_scores.weights)


def _role_alignment(
    averages: dict[str, pd.Series], specs: list[MovingAverageSpec], index: pd.Index
) -> pd.Series:
    """Stack alignment of a group of averages, fastest first; empty for fewer than two."""
    ordered = _fastest_first(specs)
    if len(ordered) < 2:
        return _undefined(index)
    return ma_alignment(*[averages[spec.name] for spec in ordered])


def combination_metrics(
    frame: pd.DataFrame,
    *,
    config: IndicatorsConfig,
    atr: Any,
    combination: str = "combination_1",
) -> pd.DataFrame:
    """Spec 8's full measurement table for one combination.

    :func:`ma_metrics` per average, plus the combination-level readings:

    * ``alignment`` - ``ma_alignment`` of every average in the combination,
      fastest first (``NaN`` for a single-average combination: there is no stack);
    * ``spread`` / ``spread_velocity`` - the ATR-normalised gap between the
      combination's *fastest* and *slowest* average, and its k-bar change. That is
      spec 8's "MA separation/spread" and "spread velocity". The pair is described
      as fastest/slowest rather than e.g. "20/200" because the configuration, not
      this module, decides which averages exist;
    * ``alignment_<role>`` and ``<role>_score`` - what each role contributed.

    ``combination_3`` has no medium average, so its ``alignment_medium`` and
    ``medium_score`` columns exist and stay empty. A stable column set is worth
    more than a shorter frame: two runs can be compared without first checking
    which columns happened to appear.
    """
    specs = _combination_specs(config, combination)
    metrics = ma_metrics(frame, specs=specs, atr=atr, slope_k=config.slope.k)
    averages = {spec.name: metrics[f"{spec.name}_average"] for spec in specs}
    ranges = as_float_series(atr, name="atr")

    columns: dict[str, pd.Series] = dict(metrics)
    columns["alignment"] = _role_alignment(averages, specs, frame.index)

    ordered = _fastest_first(specs)
    if len(ordered) > 1:
        gap = spread(averages[ordered[0].name], averages[ordered[-1].name], atr=ranges)
        columns["spread"] = gap
        columns["spread_velocity"] = spread_velocity(gap, k=config.spread_velocity.k)
    else:
        columns["spread"] = _undefined(frame.index)
        columns["spread_velocity"] = _undefined(frame.index)

    for role in NIFTY_SCORE_ROLES:
        columns[f"alignment_{role}"] = _role_alignment(
            averages, [spec for spec in specs if spec.role == role], frame.index
        )
        columns[f"{role}_score"] = role_score(
            frame, specs=specs, role=role, atr=atr, slope_k=config.slope.k
        )
    return pd.DataFrame(columns, index=frame.index)


def combination_scores(frame: pd.DataFrame, *, config: IndicatorsConfig, atr: Any) -> pd.DataFrame:
    """All three combinations' composites side by side, over the same bars.

    The comparison table behind "which combination should the feature row carry?".
    Nothing here is stored; :func:`trend_features` publishes exactly one of them.
    """
    return pd.DataFrame(
        {
            name: combination_score(frame, config=config, atr=atr, combination=name)
            for name in _COMBINATIONS
        },
        index=frame.index,
    )


def trend_features(
    frame: pd.DataFrame,
    *,
    config: IndicatorsConfig,
    atr: Any,
    combination: str = "combination_1",
) -> pd.DataFrame:
    """The six ``nifty_*`` score columns of the unified feature row (spec 9).

    Column names are spelled exactly as ``core/storage/schema.sql`` spells them, so
    the feature builder maps them across without a translation table that could
    drift out of step with the schema. ``nifty_trend_acceleration`` uses the same k
    as the slope indicator and ``nifty_trend_persistence`` counts agreement over
    ``persistence.window`` bars - both read from ``config/indicators.yaml``.

    A combination with no member of a role still produces that role's column, all
    ``NaN`` (combination 3 has no medium average), so the frame's column set is the
    same whichever combination is asked for.

    ``nifty_regime`` is deliberately *not* here: it classifies this composite, and
    :func:`core.nifty.regime.regime_features` builds it from the composite so the
    two can never disagree about which thresholds were used.
    """
    if frame.empty:
        return pd.DataFrame(
            {column: pd.Series([], dtype="float64") for column in NIFTY_TREND_FEATURE_COLUMNS},
            index=frame.index,
        )

    specs = _combination_specs(config, combination)
    scores = _role_scores(frame, config=config, specs=specs, atr=atr)
    composite = _weighted_composite(scores, weights=config.nifty_scores.weights)
    return pd.DataFrame(
        {
            "nifty_fast_score": scores["fast"],
            "nifty_medium_score": scores["medium"],
            "nifty_long_score": scores["long"],
            "nifty_composite_trend_score": composite,
            "nifty_trend_acceleration": difference(composite, periods=config.slope.k),
            "nifty_trend_persistence": trend_persistence(
                composite, window=config.persistence.window
            ),
        },
        index=frame.index,
    )






