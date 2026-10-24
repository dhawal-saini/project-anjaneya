"""Option trend engine: a short-window trend reading per option contract (spec 10).

Spec 10 is explicit that the NIFTY combination must *not* be applied to weekly
options: EMA 254, SMA 720 and SMA 2056 need more bars than a weekly contract has
minutes. This engine therefore measures a small stack of averages (by default
EMA 8 low, EMA 8 high, EMA 20 close, EMA 55 close, EMA 200 close) over **one
contract's own candles**, and it never mixes two contracts - spec 6.4 identifies a
contract by ``(underlying, expiry, strike, option_type)``, and an average computed
across a roll is not an average of anything.

What is published (the ``option_*`` columns of spec 18)
------------------------------------------------------
``option_trend_score``
    Mean of the votes that exist: *position* (``mean tanh((close-MA)/ATR)`` over the
    stack), *slope* (``mean tanh(normalized_slope)``) and *stack*
    (``ma_alignment``, fastest first). Same families as the NIFTY role score
    (:mod:`core.nifty.trend_score`), flat instead of role-weighted and allowed to
    publish a partial mean - a weekly contract is judged on the evidence it has.
``option_ma_alignment``
    ``ma_alignment`` of the configured stack, fastest first.
``option_ma_slope`` / ``option_normalized_slope``
    Points per bar of the **fastest** configured average, and the same number in
    ATR units. The fastest average is the one that turns first, and a stack-wide
    slope would need a second aggregation rule the schema has no column for.
``option_ema20_ema200_spread`` / ``option_spread_velocity``
    ``(fast - slow) / ATR`` for the pair the config names in
    ``option_trend.spread_pair``, and its k-bar change. The column name is a schema
    constant (spec 18) and the shipped config names exactly EMA 20 close and
    EMA 200 close, so the stored name and the arithmetic agree. When a config names
    no pair, the fallback is the slowest minus the fastest configured average; a
    config that drops EMA 20/200 must bump the feature version, because the column
    would otherwise silently start meaning something else.
``option_atr``
    The contract's ATR, passed in rather than recomputed, so the option engine and
    the price-action engines measure distance against one shared number.
``option_momentum``
    ATR-normalised momentum over ``momentum.period`` bars.
``option_trend_persistence``
    How long the score has been saying the same thing (``persistence.window``).
``option_insufficient_history``
    See below.

Insufficient history - three honest answers
-------------------------------------------
A weekly option often has fewer bars than a 200-period average needs, so
``option_trend.insufficient_history_policy`` decides what is published. The flag is
the same under all three: ``option_insufficient_history`` is True while the contract
has fewer than ``min_candles_for_full_set`` bars **or** any configured average is
still undefined.

=================  ======================================================================
policy             behaviour
=================  ======================================================================
``flag`` (default)  publish the score from the averages that do exist; the flag marks the gap
``skip``            withhold every average-derived column until the full stack exists
``shorter_window``  replace each too-long average by the longest window that exists, and flag it
=================  ======================================================================

The config comment describes ``skip`` as "drop the row for that MA". A unified
feature row is wide - one row per timestamp, not per moving average - so the strict
reading is used: while the full stack is unavailable the average-derived columns are
withheld (``NULL``) and only the ATR and momentum facts are published. ``flag`` is
the default because a flag a model can learn to discount beats a hole in the
history; ``shorter_window`` is an approximation and is documented as one - a 40-bar
"EMA 200" is not an EMA 200.

Bars seen are counted **from the first bar of the frame**, so the caller must pass a
frame that starts at the contract's first bar for the flag to be meaningful; the
feature builder does exactly that.

This module keeps its own copy of the "ATR-normalised evidence blend" rather than
importing the NIFTY one: the two engines are allowed to drift apart, and a weekly
option should never inherit a NIFTY-specific aggregation rule by accident.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config_loader import IndicatorsConfig, MovingAverageSpec, OptionTrendConfig
from core.indicators.moving_averages import ma_alignment, ma_source, moving_average
from core.indicators.persistence import trend_persistence
from core.indicators.series import as_float_series, require_pair
from core.indicators.slope import normalized_slope, slope
from core.indicators.spread import spread, spread_velocity
from core.indicators.statistics import momentum

__all__ = [
    "OPTION_TREND_FEATURE_COLUMNS",
    "average_stack",
    "option_metrics",
    "option_trend_features",
]

#: The ten columns the ``features`` table carries for spec 10.
OPTION_TREND_FEATURE_COLUMNS = (
    "option_trend_score",
    "option_ma_alignment",
    "option_ma_slope",
    "option_normalized_slope",
    "option_ema20_ema200_spread",
    "option_spread_velocity",
    "option_atr",
    "option_momentum",
    "option_trend_persistence",
    "option_insufficient_history",
)

#: The feature columns that only mean something with a full average stack. Under the
#: ``skip`` policy every one of them is withheld while history is insufficient.
_AVERAGE_DERIVED_COLUMNS = (
    "option_trend_score",
    "option_ma_alignment",
    "option_ma_slope",
    "option_normalized_slope",
    "option_ema20_ema200_spread",
    "option_spread_velocity",
    "option_trend_persistence",
)

#: The three policies, spelled as ``config/indicators.yaml`` spells them.
_POLICIES = ("flag", "skip", "shorter_window")


def _ordered(specs: list[MovingAverageSpec]) -> list[MovingAverageSpec]:
    """Fastest first, by period; ties keep their config order (:func:`sorted` is stable)."""
    return sorted(specs, key=lambda spec: spec.period)


def _fastest_spec(specs: list[MovingAverageSpec]) -> MovingAverageSpec:
    """The fastest configured average - the one ``option_ma_slope`` reports."""
    return _ordered(specs)[0]


def _slowest_spec(specs: list[MovingAverageSpec]) -> MovingAverageSpec:
    """The slowest configured average."""
    return _ordered(specs)[-1]


def _spread_pair(
    option_config: OptionTrendConfig, specs: list[MovingAverageSpec]
) -> tuple[MovingAverageSpec, MovingAverageSpec]:
    """The two configured averages the published spread sits between, fast first.

    The config names them (``option_trend.spread_pair``) because the stored column
    is called ``option_ema20_ema200_spread``; when it names none, the fallback is the
    fastest versus the slowest average of the stack. Both members always come from
    the configured stack, so the spread can never mix a configured average with an
    invented one, and the order is always ``(fast, slow)`` so the sign of the spread
    means "the fast average is above the slow one".
    """
    if option_config.spread_pair is not None:
        by_name = {spec.name: spec for spec in specs}
        return by_name[option_config.spread_pair.fast], by_name[option_config.spread_pair.slow]
    return _fastest_spec(specs), _slowest_spec(specs)


def _bars_seen(index: pd.Index) -> pd.Series:
    """How many bars of this contract each bar of the frame has seen, itself included."""
    return pd.Series(np.arange(1, len(index) + 1), index=index, dtype="int64")


def _undefined(index: pd.Index, *, dtype: str = "float64") -> pd.Series:
    """An all-``NaN`` column - "this reading does not exist for this frame"."""
    return pd.Series(np.nan, index=index, dtype=dtype)


def average_stack(frame: pd.DataFrame, *, specs: list[MovingAverageSpec]) -> pd.DataFrame:
    """One column per configured average, named as the config names it.

    The same shape as the NIFTY engine's ``ma_table``, kept separate on purpose: an
    option contract's stack is a different instrument's history, and neither engine
    should have to change when the other's needs change.
    """
    columns = {
        spec.name: moving_average(
            ma_source(frame, spec.source), ma_type=spec.type, period=spec.period
        )
        for spec in specs
    }
    return pd.DataFrame(columns, index=frame.index, dtype="float64")


def _withheld(stack: pd.DataFrame, insufficient: pd.Series) -> pd.DataFrame:
    """Blank every stacked average on the bars whose history is insufficient."""
    blank = pd.DataFrame(
        np.repeat(insufficient.to_numpy()[:, None], stack.shape[1], axis=1),
        index=stack.index,
        columns=stack.columns,
    )
    return stack.mask(blank)


def _shortened_stack(
    stack: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    specs: list[MovingAverageSpec],
    seen: pd.Series,
) -> pd.DataFrame:
    """Replace each too-long average by the longest window that exists.

    "The longest available period" is every bar seen so far: at bar ``b`` the
    substitute is the average of those ``b`` bars, computed with the same moving
    average type the config asked for. A one-bar window is not an average, so the
    very first bar of a contract stays ``NaN`` - there is nothing to average yet.

    Cost is bounded by the configured periods, not by the length of the frame: only
    the leading bars of a contract are ever short of history.
    """
    capped = stack.copy()
    for spec in specs:
        short = seen < spec.period
        if not bool(short.any()):
            continue
        values = ma_source(frame, spec.source)
        substituted = capped[spec.name].copy()
        for position in np.flatnonzero(short.to_numpy()):
            bars = int(seen.iloc[position])
            if bars < 2:
                substituted.iloc[position] = np.nan
                continue
            window = moving_average(values.iloc[:bars], ma_type=spec.type, period=bars)
            substituted.iloc[position] = float(window.iloc[-1])
        capped[spec.name] = substituted
    return capped


def _distance_atr(close: pd.Series, average: pd.Series, ranges: pd.Series) -> pd.Series:
    """``(close - average) / ATR`` - positive when price is above the average."""
    return ((close - average) / ranges).where(ranges > 0)


def _mean_of(votes: list[pd.Series], *, skipna: bool) -> pd.Series:
    """Pointwise mean of equal-weight votes."""
    return pd.concat(votes, axis=1).mean(axis=1, skipna=skipna)


def _option_score(
    stack: pd.DataFrame,
    *,
    specs: list[MovingAverageSpec],
    close: pd.Series,
    ranges: pd.Series,
    config: IndicatorsConfig,
) -> pd.Series:
    """The option trend score: mean of the position, slope and stack votes.

    ``skipna=True`` here (unlike the NIFTY role score) because of spec 10's
    "handle insufficient history explicitly": a young contract publishes the vote
    it can support, and ``option_insufficient_history`` says so in the same row.
    """
    ordered = _ordered(specs)
    votes = [
        _mean_of(
            [_distance_atr(close, stack[spec.name], ranges).map(np.tanh) for spec in ordered],
            skipna=True,
        ),
        _mean_of(
            [
                normalized_slope(stack[spec.name], atr=ranges, k=config.slope.k).map(np.tanh)
                for spec in ordered
            ],
            skipna=True,
        ),
        ma_alignment(*[stack[spec.name] for spec in ordered]),
    ]
    return _mean_of(votes, skipna=True)


def option_metrics(
    frame: pd.DataFrame, *, config: IndicatorsConfig, atr: Any
) -> pd.DataFrame:
    """Spec 10's measurements for one contract, under the configured policy.

    Returns the configured average stack (one column per average, named as the
    config names it) followed by the ten ``option_*`` feature columns, spelled
    exactly as ``core/storage/schema.sql`` spells them.

    ``frame`` must be **one contract's** candles, starting at that contract's first
    bar: the insufficient-history flag counts bars seen since the start of the frame.
    """
    option_config = config.option_trend
    policy = option_config.insufficient_history_policy
    if policy not in _POLICIES:
        raise ValueError(
            f"Unknown insufficient-history policy {policy!r}; expected one of {', '.join(_POLICIES)}"
        )
    specs = list(option_config.moving_averages)
    if len(specs) < 2:
        raise ValueError(
            "option_trend.moving_averages needs at least two averages: alignment and "
            "spread are undefined with a single one"
        )

    if frame.empty:
        columns: dict[str, pd.Series] = {spec.name: _undefined(frame.index) for spec in specs}
        for column in OPTION_TREND_FEATURE_COLUMNS:
            columns[column] = _undefined(
                frame.index, dtype="boolean" if column == "option_insufficient_history" else "float64"
            )
        return pd.DataFrame(columns, index=frame.index)

    if "close" not in frame.columns:
        raise ValueError("option trend computation needs a 'close' column")
    close = as_float_series(frame["close"], name="close")
    ranges = as_float_series(atr, name="atr")
    require_pair(close, ranges, names=("close", "atr"))

    seen = _bars_seen(frame.index)
    stack = average_stack(frame, specs=specs)
    # The flag always describes the *configured* stack: a substituted window is
    # still a window shorter than the config asked for.
    insufficient = (seen < option_config.min_candles_for_full_set) | stack.isna().any(axis=1)

    if policy == "skip":
        # Everything in _AVERAGE_DERIVED_COLUMNS, and the stack it comes from, is
        # withheld: an average cannot exist without the bars it needs.
        stack = _withheld(stack, insufficient)
    elif policy == "shorter_window":
        stack = _shortened_stack(stack, frame=frame, specs=specs, seen=seen)

    fastest = _fastest_spec(specs).name
    spread_fast, spread_slow = _spread_pair(option_config, specs)
    gap = spread(stack[spread_fast.name], stack[spread_slow.name], atr=ranges)
    score = _option_score(stack, specs=specs, close=close, ranges=ranges, config=config)

    columns = {spec.name: stack[spec.name] for spec in specs}
    columns["option_trend_score"] = score
    columns["option_ma_alignment"] = ma_alignment(
        *[stack[spec.name] for spec in _ordered(specs)]
    )
    columns["option_ma_slope"] = slope(stack[fastest], k=config.slope.k)
    columns["option_normalized_slope"] = normalized_slope(
        stack[fastest], atr=ranges, k=config.slope.k
    )
    columns["option_ema20_ema200_spread"] = gap
    columns["option_spread_velocity"] = spread_velocity(gap, k=config.spread_velocity.k)
    columns["option_atr"] = ranges
    columns["option_momentum"] = momentum(close, period=config.momentum.period, atr=ranges)
    columns["option_trend_persistence"] = trend_persistence(
        score, window=config.persistence.window
    )
    columns["option_insufficient_history"] = insufficient
    return pd.DataFrame(columns, index=frame.index)


def option_trend_features(
    frame: pd.DataFrame, *, config: IndicatorsConfig, atr: Any
) -> pd.DataFrame:
    """The ten ``option_*`` columns of the unified feature row (spec 10).

    Column names come from the schema's own vocabulary, so the feature builder maps
    them across without a translation table that could drift out of step.
    """
    return option_metrics(frame, config=config, atr=atr)[list(OPTION_TREND_FEATURE_COLUMNS)]



