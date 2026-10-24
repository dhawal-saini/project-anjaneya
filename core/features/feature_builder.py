"""Unified feature assembly - the row every engine feeds (spec 18, Stage 3.2e).

What this module is
-------------------
One place that turns *candles* (already built from ticks) into the features table's
row: the NIFTY group, the option group, the candle/levels/structure/S-R groups, the
option context and the relative-behaviour columns. Every number comes from an engine
that is tested on its own; this module only decides *which frame is fed to which
engine* and *how the results are stitched into one row*.

Four rules shape it, and each one exists because the alternative silently produces a
wrong research dataset:

1. **A feature at ``t`` may only read ``<= t``.** The NIFTY block is joined onto the
   contract's bars by **bar close instant**, never by bar *open* time: an index bar
   opening 09:15 on a 15m series is only knowable at 09:30, so a 1m contract bar at
   09:20 can never use it. Joining on the open time (the obvious implementation) is a
   one-bar look-ahead that no test of a single engine would catch.
2. **The lookback window is a contract, not a tuning knob.**
   :func:`required_lookback_bars` is the longest configured average plus the slope
   lookback - on the shipped config ``SMA 2056 + k=5`` -> 2062 bars. A shorter window
   is *not* wrong (it is how a single-session rebuild works), but ``nifty_long_score``
   is then ``NULL`` at the start of the window and the composite is a two-role reading
   there. The ``NULL`` is what says so; the value is never back-filled from a longer
   run.
3. **``NULL`` means "not available", never 0.** Early bars, a contract's first bar, a
   missing previous session, an ``iv`` the capture layer has not filled yet
   (Stage 3.3) - all of them stay ``None``. :func:`feature_rows` therefore converts
   ``NaN`` to ``None`` explicitly instead of letting a sentinel reach storage.
4. **One row per ``(timestamp, contract)``, as the schema's primary key says.** The
   ``features`` table has no timeframe in its key, so a build is authoritative for a
   whole *contract-day*: rebuilding one day at a different timeframe replaces that
   contract's rows for the day rather than storing two rows that share a key. The
   writer owns that scope rule (``FeatureWriter.replace_features``).

What this module deliberately does *not* do
-------------------------------------------
No I/O (frames in, frames out) and no labels (spec 20 keeps them physically separate).
It also does not *price* anything: ``iv``/``delta``/``gamma``/``theta``/``vega`` arrive
with the candle row, filled by ``--mode enrich-greeks`` (spec 16's Black-Scholes model,
:mod:`core.options.greeks`). A candle row without them is not an error - those columns
simply stay ``NULL``, and with no ``delta`` the relative block stays ``NULL`` too rather
than inventing an expectation.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from core.constants import (
    DEFAULT_TIMEZONE,
    FEATURE_COLUMNS,
    PROVENANCE_COLUMNS,
    get_timezone,
    timeframe_to_minutes,
)
from core.indicators import atr as atr_series
from core.nifty import (
    NIFTY_REGIME_FEATURE_COLUMNS,
    NIFTY_TREND_FEATURE_COLUMNS,
    regime_features,
    trend_features,
)
from core.options import OPTION_TREND_FEATURE_COLUMNS, option_trend_features
from core.price_action import (
    LEVEL_STATE_COLUMNS,
    LEVELS_FEATURE_COLUMNS,
    RELATIONSHIP_COLUMNS,
    SHAPE_COLUMNS,
    STRUCTURE_FEATURE_COLUMNS,
    SUPPORT_RESISTANCE_FEATURE_COLUMNS,
    candidate_levels,
    candle_features,
    level_events,
    level_features,
    recent_swings,
    session_levels,
    structure_features,
    support_resistance_features,
)
from core.schemas import FeatureRow, QualityFlag, Source

__all__ = [
    "BOOLEAN_BLOCK_COLUMNS",
    "CATEGORICAL_BLOCK_COLUMNS",
    "CONTRACT_BLOCK_COLUMNS",
    "CONTEXT_FEATURE_COLUMNS",
    "CANDLE_FEATURE_COLUMNS",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_ORDER",
    "IDENTITY_FEATURE_COLUMNS",
    "LEVEL_EVENT_FEATURE_COLUMNS",
    "NIFTY_FEATURE_COLUMNS",
    "OPTION_FEATURE_COLUMNS",
    "RELATIVE_FEATURE_COLUMNS",
    "REQUIRED_CANDLE_COLUMNS",
    "FeatureBuildError",
    "aggregate_index_frame",
    "assert_feature_group_parity",
    "build_feature_frame",
    "close_moments",
    "contract_feature_block",
    "feature_rows",
    "nifty_feature_block",
    "required_lookback_bars",
    "timeframe_of",
]

#: Candle columns a build cannot work without. Everything else (identity, volume,
#: ``oi``, the greeks) is optional: a candle row written by Stage 3.1 carries no IV,
#: and a build must still produce an honest row for it.
REQUIRED_CANDLE_COLUMNS = ("timestamp", "open", "high", "low", "close")


class FeatureBuildError(ValueError):
    """Raised when a candle window cannot honestly be turned into feature rows."""


# ---------------------------------------------------------------------------
# The feature contract, grouped
# ---------------------------------------------------------------------------
#: Identity of the row. These are the only columns that must always be present -
#: everything else in a row may legitimately be ``NULL`` at some point in a
#: contract's life, and saying so is the point.
IDENTITY_FEATURE_COLUMNS = (
    "timestamp",
    "underlying",
    "tradingsymbol",
    "instrument_token",
    "expiry",
    "strike",
    "option_type",
    "timeframe",
)

#: Spec 16's option context. ``iv_change``/``oi_change`` are bar-to-bar changes, so
#: they are ``NULL`` on the first bar of a series; the greeks are carried through from
#: the candle row, where ``--mode enrich-greeks`` wrote them.
CONTEXT_FEATURE_COLUMNS = (
    "distance_from_atm",
    "days_to_expiry",
    "expiry_cycle_phase",
    "time_of_day_bucket",
    "day_of_week",
    "iv",
    "iv_change",
    "delta",
    "gamma",
    "theta",
    "vega",
    "oi",
    "oi_change",
    "volume",
)

#: Spec 16's relative behaviour, computed by ``_relative_block`` against the *previous*
#: bar's delta. All three are ``NULL`` until a candle carries a delta (i.e. until
#: ``--mode enrich-greeks`` has run for that scope) and a NIFTY move is available.
RELATIVE_FEATURE_COLUMNS = (
    "expected_option_move",
    "actual_option_move",
    "option_residual",
)

#: The NIFTY group: spec 9's six scores plus the regime label derived from the
#: composite (the threshold is never applied in two places).
NIFTY_FEATURE_COLUMNS = (*NIFTY_TREND_FEATURE_COLUMNS, *NIFTY_REGIME_FEATURE_COLUMNS)

#: The option group (spec 10).
OPTION_FEATURE_COLUMNS = OPTION_TREND_FEATURE_COLUMNS

#: The candle group (spec 11).
CANDLE_FEATURE_COLUMNS = (*SHAPE_COLUMNS, *RELATIONSHIP_COLUMNS)

#: The level-interaction group (spec 13's four state columns).
LEVEL_EVENT_FEATURE_COLUMNS = LEVEL_STATE_COLUMNS

#: Every group of the unified row, in the order ``core/constants.FEATURE_COLUMNS``
#: declares them. The union is asserted to be *exactly* the stored contract (see
#: :func:`assert_feature_group_parity`), so a column added to the schema but
#: forgotten here - or produced here but absent from the schema - fails at import
#: time rather than at 09:15 on a trading day.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "identity": IDENTITY_FEATURE_COLUMNS,
    "nifty": NIFTY_FEATURE_COLUMNS,
    "option": OPTION_FEATURE_COLUMNS,
    "candle": CANDLE_FEATURE_COLUMNS,
    "levels": LEVELS_FEATURE_COLUMNS,
    "level_events": LEVEL_EVENT_FEATURE_COLUMNS,
    "structure": STRUCTURE_FEATURE_COLUMNS,
    "support_resistance": SUPPORT_RESISTANCE_FEATURE_COLUMNS,
    "context": CONTEXT_FEATURE_COLUMNS,
    "relative": RELATIVE_FEATURE_COLUMNS,
    "provenance": PROVENANCE_COLUMNS,
}

#: Group names in row order (the order the writer sees after assembly).
FEATURE_GROUP_ORDER: tuple[str, ...] = tuple(FEATURE_GROUPS)

#: The contract-side block: everything one contract's own candles produce, in row
#: order. ``build_feature_frame`` adds the identity, context, relative and provenance
#: columns on top of this.
CONTRACT_BLOCK_COLUMNS = (
    *CANDLE_FEATURE_COLUMNS,
    *LEVELS_FEATURE_COLUMNS,
    *LEVEL_EVENT_FEATURE_COLUMNS,
    *STRUCTURE_FEATURE_COLUMNS,
    *SUPPORT_RESISTANCE_FEATURE_COLUMNS,
    *OPTION_FEATURE_COLUMNS,
)

#: Block columns that hold a label rather than a number (``None`` when unknown).
CATEGORICAL_BLOCK_COLUMNS = frozenset(
    {
        "candle_direction",
        "swing_label",
        "market_structure_state",
        *LEVEL_EVENT_FEATURE_COLUMNS,
    }
)

#: Block columns whose ``False`` is a measurement and whose ``None`` is "unknown".
BOOLEAN_BLOCK_COLUMNS = frozenset(
    {"candle_inside_flag", "candle_outside_flag", "option_insufficient_history"}
)




def assert_feature_group_parity() -> None:
    """Raise unless the groups above reconstruct ``FEATURE_COLUMNS`` exactly.

    This is a pure, side-effect-free consistency check (no I/O), run at import time
    for the same reason :func:`core.schemas.assert_contract_parity` is: a feature
    column that is persisted but never computed, or computed but never persisted,
    should fail the moment anything imports the builder - not be discovered by a
    research report months later.
    """
    assembled = tuple(column for group in FEATURE_GROUP_ORDER for column in FEATURE_GROUPS[group])
    duplicates = sorted({column for column in assembled if assembled.count(column) > 1})
    if duplicates:
        raise RuntimeError(f"ANJANEYA feature groups declare duplicate column(s): {duplicates}")
    if assembled != FEATURE_COLUMNS:
        missing = [column for column in FEATURE_COLUMNS if column not in assembled]
        extra = [column for column in assembled if column not in FEATURE_COLUMNS]
        problems: list[str] = []
        if missing:
            problems.append(f"never produced: {missing}")
        if extra:
            problems.append(f"not in the stored contract: {extra}")
        if not problems:  # same columns, different order
            problems.append("column order differs from FEATURE_COLUMNS")
        raise RuntimeError("ANJANEYA feature-group mismatch: " + "; ".join(problems))


assert_feature_group_parity()


def required_lookback_bars(config: Any) -> int:
    """Bars of history a feature row needs *before* the session it describes.

    The number is not a tuning knob: it is the longest configured average (every NIFTY
    combination, the option stack and the ATR) plus the slope lookback ``k`` plus one
    bar for the difference the slope takes. On the shipped configuration that is
    ``SMA 2056 + 5 + 1 = 2062`` 1m bars, i.e. about five and a half sessions.

    A shorter window still produces rows - that is how a single-session rebuild works -
    but each engine reports the shortfall honestly (``NULL`` scores, ``NULL`` swings)
    instead of inventing a value.
    """
    periods = [config.atr.period, config.momentum.period]
    combinations = list(type(config.nifty_moving_averages).model_fields)
    for name in combinations:
        periods.extend(spec.period for spec in getattr(config.nifty_moving_averages, name))
    periods.extend(spec.period for spec in config.option_trend.moving_averages)
    return max(periods) + config.slope.k + 1


def timeframe_of(frame: pd.DataFrame, *, default: str | None = None) -> str:
    """The single timeframe label a frame carries, else ``default``.

    Used to align an index series: the *index* frame's own timeframe decides when its
    bars close, and assuming the feature timeframe there is exactly the bug that hands a
    1m candle a 15m bar's close before it closed.
    """
    if "timeframe" in frame.columns:
        labels = {str(label) for label in frame["timeframe"].dropna().unique()}
        if len(labels) == 1:
            return labels.pop()
        if len(labels) > 1:
            raise FeatureBuildError(f"expected one timeframe in the frame, found {sorted(labels)}")
    if default is None:
        raise FeatureBuildError(
            "the frame carries no timeframe column; pass one explicitly (close_moments / "
            "timeframe_of)"
        )
    return default


def close_moments(frame: pd.DataFrame, *, timeframe: str | None = None) -> pd.Series:
    """Instant each bar *closes*: its open time plus its timeframe, **as a UTC instant**.

    ``timeframe`` may be omitted when the frame carries its own ``timeframe`` column
    (which is what :func:`aggregate_index_frame` returns). The close instant is the
    only honest key for aligning two series of different resolutions - it is the first
    moment at which both numbers are knowable.

    The result is normalised to ``datetime64[ns, UTC]`` because a *close instant* is an
    instant, and pandas refuses to ``merge_asof`` two timezone-aware columns whose
    ``tzinfo`` objects differ. That is not hypothetical: DuckDB hands frames back with a
    ``pytz`` zone, :func:`aggregate_index_frame` localises with ``zoneinfo.ZoneInfo``, and
    the two are neither identical nor equal - the merge dies with "incompatible merge
    keys [0] ... must be the same type" even though both say ``Asia/Kolkata``. Two
    canonical UTC columns always merge, no matter where the rows came from.
    """
    if timeframe is None:
        if "timeframe" not in frame.columns:
            raise FeatureBuildError(
                "close_moments needs a timeframe: pass it explicitly or give the frame a "
                "'timeframe' column"
            )
        timeframe = timeframe_of(frame)
    minutes = timeframe_to_minutes(timeframe)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], errors="raise"))
    if stamps.tz is None:
        raise FeatureBuildError(
            "candle timestamps are naive; ANJANEYA only stores timezone-aware data "
            "(see settings.time.forbid_naive_timestamps)"
        )
    # ``datetime.timedelta`` rather than ``pd.Timedelta``: pandas' own constructor
    # trips a NumPy "generic timedelta unit" deprecation on this stack, and a plain
    # timedelta is exact here (no calendar arithmetic is involved).
    closed = stamps.tz_convert("UTC").as_unit("ns") + timedelta(minutes=minutes)
    return pd.Series(closed, index=frame.index)

# ---------------------------------------------------------------------------
# The NIFTY index series
# ---------------------------------------------------------------------------
#: Index columns :func:`aggregate_index_frame` carries through (when present).
_INDEX_CARRY_COLUMNS = (
    "tradingsymbol",
    "source",
    "quality_flag",
    "run_id",
    "fetch_timestamp",
    "volume",
    "oi",
)


def _empty_block(index: Any, columns: Sequence[str]) -> pd.DataFrame:
    """An empty frame carrying ``columns``, with the categorical ones as objects.

    Used where an engine is handed no bars at all: the *shape* of the block stays
    identical either way, so a caller never has to special-case "no rows".
    """
    frame = pd.DataFrame(index=index)
    for column in columns:
        if column in CATEGORICAL_BLOCK_COLUMNS or column in BOOLEAN_BLOCK_COLUMNS:
            frame[column] = pd.Series([None] * len(index), index=index, dtype="object")
        else:
            frame[column] = pd.Series(np.nan, index=index, dtype="float64")
    return frame


def aggregate_index_frame(
    frame: pd.DataFrame | None, *, timeframe: str, timezone_name: str = DEFAULT_TIMEZONE
) -> pd.DataFrame:
    """Resample an index frame onto ``timeframe`` bars, keyed by *open* time.

    The NIFTY index is captured as daily history plus an intraday top-up, so a build
    at 15m has no 15m index rows to read. Rather than let a feature row silently mix
    resolutions, the 1m series is merged with the same wall-clock bucket rule the
    candle builder uses (``apps.anjaneeya_engine.candles.bucket_start``) - and an
    assertion in ``tests/unit/test_feature_builder.py`` pins the two implementations
    together, because two definitions of "when does a 15m bar start" would be a quiet
    source of disagreement.

    A frame already at ``timeframe`` is returned unchanged.
    """
    minutes = timeframe_to_minutes(timeframe)
    if frame is None:
        return pd.DataFrame(columns=["timestamp", "timeframe", *(_INDEX_CARRY_COLUMNS)])

    missing = [column for column in ("timestamp", "open", "high", "low", "close") if column not in frame.columns]
    if missing:
        raise FeatureBuildError(f"the index frame is missing column(s) {missing}")
    if frame.empty:
        return frame.copy()

    if "timeframe" in frame.columns:
        labels = {str(label) for label in frame["timeframe"].dropna().unique()}
        if labels == {timeframe}:
            return frame.copy()
    if minutes == 1:
        return frame.copy()

    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], errors="raise"))
    if stamps.tz is None:
        raise FeatureBuildError(
            "index timestamps are naive; ANJANEYA only stores timezone-aware data"
        )
    if "timeframe" in frame.columns:
        labels = {str(label) for label in frame["timeframe"].dropna().unique()}
        if labels == {timeframe}:
            return frame.copy()
    if minutes == 1:
        return frame.copy()

    local = pd.DatetimeIndex(stamps.tz_convert(timezone_name).tz_localize(None))
    # Floor the *wall-clock* local time, exactly as ``bucket_start`` does, so a 15m bar
    # starts at 09:15 in IST regardless of the machine's own timezone. A ``ZoneInfo``
    # object is passed deliberately: localising with a bare string lets pandas pick a
    # different tzinfo implementation, and two "Asia/Kolkata" dtypes that are not the
    # same object cannot be merged.
    opened = local.floor(f"{minutes}min").tz_localize(get_timezone(timezone_name))

    aggregations: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    for column in _INDEX_CARRY_COLUMNS:
        if column in frame.columns:
            aggregations[column] = "max" if column == "fetch_timestamp" else "last"
    aggregations["volume"] = lambda values: values.sum(min_count=1)

    working = frame.assign(timestamp=pd.Series(opened, index=frame.index)).sort_values(
        "timestamp", kind="stable"
    )
    aggregated = working.groupby("timestamp", sort=True, as_index=False).agg(aggregations)
    aggregated["timeframe"] = timeframe
    ordered = ["timestamp", "tradingsymbol", "timeframe", "open", "high", "low", "close"]
    ordered += [column for column in _INDEX_CARRY_COLUMNS if column in aggregated.columns and column != "tradingsymbol"]
    return aggregated[[column for column in ordered if column in aggregated.columns]]


def _nifty_columns() -> list[str]:
    return list(NIFTY_FEATURE_COLUMNS)


def nifty_feature_block(
    index_frame: pd.DataFrame | None, *, config: Any, combination: str = "combination_1"
) -> pd.DataFrame:
    """The seven ``nifty_*`` columns of the unified feature row (specs 8/9).

    ``index_frame`` must be the **index's own** candles at the feature timeframe. An
    absent or empty frame produces the seven columns all ``NULL`` - the honest answer
    when ``index_history`` holds nothing for the window - rather than raising, because
    a missing index series is a data problem the coverage ledger and the run summary
    report, and the option-side features are still perfectly computable.
    """
    if index_frame is None or index_frame.empty:
        return _empty_block(
            pd.RangeIndex(0) if index_frame is None else index_frame.index, NIFTY_FEATURE_COLUMNS
        )
    missing = [column for column in ("high", "low", "close") if column not in index_frame.columns]
    if missing:
        raise FeatureBuildError(f"the index frame is missing column(s) {missing} for the NIFTY block")
    ranges = atr_series(
        index_frame["high"],
        index_frame["low"],
        index_frame["close"],
        period=config.atr.period,
        method=config.atr.method,
    )
    trend = trend_features(index_frame, config=config, atr=ranges, combination=combination)
    regime = regime_features(
        trend["nifty_composite_trend_score"], thresholds=config.nifty_scores.regime_thresholds
    )
    return pd.concat([trend, regime], axis=1)[_nifty_columns()]


def contract_feature_block(
    frame: pd.DataFrame,
    *,
    config: Any,
    atr: Any = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    levels: pd.DataFrame | None = None,
    swings: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every :data:`CONTRACT_BLOCK_COLUMNS` column for one contract, one window.

    The frame must be **one contract's** candles, in order, starting at that contract's
    first known bar where possible: the option engine counts bars since the start of
    the frame when it decides whether a weekly contract has enough history.

    ``levels``/``swings``/``candidates`` are the intermediate frames the S/R engines
    would otherwise recompute; passing them in (as the builder does) means the level
    state machine, the candidate list and the strength blend all describe the *same*
    levels.
    """
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureBuildError(
            f"the candle frame is missing column(s) {missing}; expected at least "
            f"{REQUIRED_CANDLE_COLUMNS}"
        )
    if frame.empty:
        return _empty_block(frame.index, CONTRACT_BLOCK_COLUMNS)
    if not frame["timestamp"].is_monotonic_increasing:
        raise FeatureBuildError(
            "the candle frame must be ordered by timestamp: every engine in this block "
            "reads its own and earlier bars only"
        )

    ranges = (
        atr
        if atr is not None
        else atr_series(
            frame["high"],
            frame["low"],
            frame["close"],
            period=config.atr.period,
            method=config.atr.method,
        )
    )
    level_frame = session_levels(frame, timezone_name=timezone_name) if levels is None else levels
    # ``swings`` is the *numeric* recent-swing frame (``recent_swings``), not
    # ``swing_levels``: the S/R engines feed each column to ``as_float_series``, and a
    # ``swing_label`` string in there is a bug they rightly refuse.
    swing_frame = recent_swings(frame, config=config.levels.swing) if swings is None else swings
    candidate_frame = (
        candidate_levels(
            frame,
            config=config.levels,
            levels=level_frame,
            swings=swing_frame,
            timezone_name=timezone_name,
        )
        if candidates is None
        else candidates
    )

    block = pd.concat(
        [
            candle_features(frame),
            level_features(frame, atr=ranges, timezone_name=timezone_name),
            level_events(
                frame,
                atr=ranges,
                config=config.levels,
                levels=level_frame,
                timezone_name=timezone_name,
            ),
            structure_features(frame, config=config.levels.swing, atr=ranges),
            support_resistance_features(
                frame,
                atr=ranges,
                config=config.levels,
                candidates=candidate_frame,
                levels=level_frame,
                swings=swing_frame,
                timezone_name=timezone_name,
            ),
            option_trend_features(frame, config=config, atr=ranges),
        ],
        axis=1,
    )
    return block[list(CONTRACT_BLOCK_COLUMNS)]


def _align_index(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    index_frame: pd.DataFrame,
    nifty: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the NIFTY block to the contract's bars by **bar close instant**.

    ``merge_asof(direction="backward")`` takes the most recent index bar that had
    already closed when the contract's bar closed. Two series at the same resolution
    line up exactly; a coarse index series simply does not reach a finer contract bar
    until it has closed - which is the whole point. Joining on the open time instead
    would hand a 09:20 1m bar the 09:15-09:30 index bar's close.
    """
    right = pd.concat(
        [
            index_frame[["timestamp"]].reset_index(drop=True),
            nifty.reset_index(drop=True),
            index_frame[["close"]].reset_index(drop=True).rename(columns={"close": "nifty_close"}),
        ],
        axis=1,
    )
    right["_moment"] = close_moments(
        right, timeframe=timeframe_of(index_frame, default=timeframe)
    )
    left = frame.assign(_moment=close_moments(frame, timeframe=timeframe))
    merged = pd.merge_asof(
        left,
        right[["_moment", "nifty_close", *_nifty_columns()]].sort_values("_moment", kind="stable"),
        on="_moment",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns=["_moment"])


def _diff(frame: pd.DataFrame, column: str) -> pd.Series:
    """Bar-to-bar change of an optional column (all ``NaN`` when it is absent)."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").diff()


def _column_or_none(frame: pd.DataFrame, column: str) -> pd.Series:
    """A column's values, or all-``None`` when the candle row does not carry it."""
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame[column]


def _relative_block(working: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Spec 16's relative behaviour: expected move, actual move and the residual.

    ``actual_option_move`` is a price difference over the window (so the first bar of a
    session is compared against the previous session's last bar rather than being
    ``NULL``); ``expected_option_move`` is ``delta * NIFTY move``; ``option_residual`` is
    what the model did not explain.

    Two decisions are pinned here because both are invisible in the numbers:

    * **The delta is the one from the previous bar.** The move happens *during*
      ``(t-1, t]``, so the delta that was knowable when it started is ``delta[t-1]``.
      Using ``delta[t]`` would let the move partly define its own expectation and would
      mechanically shrink ``|residual|`` whenever the bar itself moved the delta (a
      strike crossing ATM is the extreme case). Both conventions are computable from the
      stored row if a later stage prefers the other one.
    * **The NIFTY move is the one the row already joined**, i.e. the change in the index
      level aligned to these bars by *close instant* (``_align_index``). Computing it
      from anything else - a different series, or the bar's open - would make
      ``expected_option_move`` describe a move the row does not contain.

    ``NULL`` propagates honestly: no ``delta`` column (a build before
    ``--mode enrich-greeks``), no ``nifty_close`` (no index history), or the first bar of
    the window leave the expected move and the residual ``NULL`` rather than zero.
    """
    actual_move = pd.to_numeric(working["close"], errors="coerce").diff()
    delta = _numeric_column(working, "delta")
    nifty_move = (
        pd.to_numeric(merged["nifty_close"], errors="coerce").diff()
        if "nifty_close" in merged.columns
        else pd.Series(np.nan, index=working.index, dtype="float64")
    )
    expected_move = delta.shift(1) * nifty_move
    return pd.DataFrame(
        {
            "expected_option_move": expected_move,
            "actual_option_move": actual_move,
            "option_residual": actual_move - expected_move,
        },
        index=working.index,
    )


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """A numeric column, or all-``NaN`` when the frame does not carry it."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _as_date(value: Any) -> Any:
    """A ``datetime.date`` from whatever the storage layer handed back."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


def _empty_feature_frame() -> pd.DataFrame:
    """A frame with every feature column and no rows."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in FEATURE_COLUMNS})


def build_feature_frame(
    *,
    candles: pd.DataFrame,
    calendar: Any,
    config: Any,
    timeframe: str,
    underlying: str,
    run_id: str,
    index_candles: pd.DataFrame | None = None,
    session_start: datetime | None = None,
    session_end: datetime | None = None,
    combination: str = "combination_1",
    timezone_name: str | None = None,
    source: Source = Source.DERIVED,
) -> pd.DataFrame:
    """Assemble one contract's feature rows for one window (spec 18).

    ``candles`` is that contract's candle window - the session being built *plus* the
    lookback :func:`required_lookback_bars` asks for - and ``index_candles`` is the
    NIFTY index series for the same window (``None`` when ``index_history`` has
    nothing, in which case the seven ``nifty_*`` columns are ``NULL`` and every
    contract-side number is still produced).

    ``session_start``/``session_end`` restrict the **published** rows (half-open
    ``[start, end)``); pass ``None`` for both to keep the whole window, which is what
    the no-look-ahead probe in :mod:`core.features.feature_validation` does.

    Bar-to-bar columns (``iv_change``, ``oi_change``, ``actual_option_move``,
    ``expected_option_move``, ``option_residual``) are differenced over the *window*, so
    the first published bar of a session is compared against the previous session's last
    bar rather than being ``NULL`` for no reason. The expected move uses the delta of the
    bar *before* the move - see :func:`_relative_block`.
    """
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        raise FeatureBuildError(
            f"the candle frame is missing column(s) {missing}; expected at least "
            f"{REQUIRED_CANDLE_COLUMNS}"
        )
    if candles.empty:
        return _empty_feature_frame()

    working = candles.sort_values("timestamp", kind="stable").reset_index(drop=True)
    stamps = pd.to_datetime(working["timestamp"], errors="raise")
    if stamps.dt.tz is None:
        raise FeatureBuildError(
            "candle timestamps are naive; ANJANEYA only stores timezone-aware data "
            "(see settings.time.forbid_naive_timestamps)"
        )
    working["timestamp"] = stamps
    if working["timestamp"].duplicated().any():
        raise FeatureBuildError(
            "the candle window carries duplicate timestamps; refusing to build rows that "
            "would collide on the (timestamp, tradingsymbol) primary key"
        )
    zone = timezone_name or DEFAULT_TIMEZONE

    index_frame = aggregate_index_frame(index_candles, timeframe=timeframe, timezone_name=zone)
    if index_frame.empty:
        merged = working.assign(nifty_close=np.nan)
        for column in _nifty_columns():
            merged[column] = np.nan
    else:
        merged = _align_index(
            working,
            timeframe=timeframe,
            index_frame=index_frame,
            nifty=nifty_feature_block(index_frame, config=config, combination=combination),
        )
    if "strike" in merged.columns:
        merged["distance_from_atm"] = (
            pd.to_numeric(merged["strike"], errors="coerce") - merged["nifty_close"]
        )
    else:
        merged["distance_from_atm"] = np.nan

    # The engines read the full candle row (volume, oi, iv...), so the block is built
    # from the whole window and joined positionally - every frame here shares one index.
    block = contract_feature_block(working, config=config, timezone_name=zone)
    relative = _relative_block(working, merged)
    changes = pd.DataFrame(
        {
            "iv_change": _diff(working, "iv"),
            "oi_change": _diff(working, "oi"),
        },
        index=working.index,
    )
    merged = pd.concat([merged, block, relative, changes], axis=1)

    published = pd.Series(True, index=merged.index)
    if session_start is not None:
        published &= merged["timestamp"] >= session_start
    if session_end is not None:
        published &= merged["timestamp"] < session_end
    session = merged[published].reset_index(drop=True)
    if session.empty:
        return _empty_feature_frame()
    return _publish(
        session,
        calendar=calendar,
        timeframe=timeframe,
        underlying=underlying,
        run_id=run_id,
        source=source,
    )



def _publish(
    session: pd.DataFrame,
    *,
    calendar: Any,
    timeframe: str,
    underlying: str,
    run_id: str,
    source: Source,
) -> pd.DataFrame:
    """Shape one window's computed columns into exactly ``FEATURE_COLUMNS``.

    The calendar-derived context is resolved here, row by row, over the *published*
    rows only: ``days_to_expiry``/``expiry_cycle_phase``/``time_of_day_bucket`` are
    defined by ``config/calendar.yaml``, and asking the calendar (rather than
    re-deriving a bucket boundary here) is what keeps the feature row and the rest of
    the platform telling the same story about a session.
    """
    assembled = pd.DataFrame(index=session.index)
    assembled["timestamp"] = session["timestamp"]
    assembled["underlying"] = underlying
    assembled["tradingsymbol"] = session["tradingsymbol"]
    assembled["instrument_token"] = _column_or_none(session, "instrument_token")
    # ``expiry`` arrives as a ``datetime.date`` from the row contract but as a
    # ``Timestamp`` from DuckDB, and the calendar compares it with a *date*. Normalise
    # once, here, so both callers behave identically.
    expiries = [
        _as_date(value)
        for value in (
            session["expiry"] if "expiry" in session.columns else [None] * len(session)
        )
    ]
    assembled["expiry"] = expiries
    assembled["strike"] = pd.to_numeric(_column_or_none(session, "strike"), errors="coerce")
    assembled["option_type"] = _column_or_none(session, "option_type")
    assembled["timeframe"] = timeframe
    for column in _nifty_columns():
        assembled[column] = session[column]
    for column in CONTRACT_BLOCK_COLUMNS:
        assembled[column] = session[column]
    assembled["distance_from_atm"] = session["distance_from_atm"]
    assembled["days_to_expiry"] = [
        calendar.days_to_expiry(moment, expiry) if expiry is not None else None
        for moment, expiry in zip(session["timestamp"], expiries, strict=False)
    ]
    assembled["expiry_cycle_phase"] = [
        calendar.expiry_cycle_phase(moment.date(), expiry)
        for moment, expiry in zip(session["timestamp"], expiries, strict=False)
    ]
    assembled["time_of_day_bucket"] = [
        calendar.time_of_day_bucket(moment) for moment in session["timestamp"]
    ]
    assembled["day_of_week"] = [moment.weekday() for moment in session["timestamp"]]
    for column in ("iv", "delta", "gamma", "theta", "vega", "oi", "volume"):
        assembled[column] = _column_or_none(session, column)
    for column in (
        "iv_change",
        "oi_change",
        "expected_option_move",
        "actual_option_move",
        "option_residual",
    ):
        assembled[column] = session[column]
    assembled["source"] = source
    assembled["fetch_timestamp"] = _column_or_none(session, "fetch_timestamp")
    assembled["quality_flag"] = _column_or_none(session, "quality_flag").fillna(
        QualityFlag.OK.value
    )
    assembled["run_id"] = run_id
    return assembled[list(FEATURE_COLUMNS)]


def _clean(value: Any) -> Any:
    """One frame cell as a value the row contract accepts.

    ``NaN``/``NaT`` become ``None`` because pandas uses them for *missing*, and the
    row contract says missing is ``None``. ``inf`` is deliberately **not** sanitised:
    an infinite feature is a bug, and :mod:`core.features.feature_validation` is where
    it must be seen, not quietly dropped on the way to storage.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return value
    if isinstance(value, np.datetime64):  # pragma: no cover - defensive
        return pd.Timestamp(value).to_pydatetime()
    return value


def feature_rows(frame: pd.DataFrame) -> list[FeatureRow]:
    """Turn an assembled frame into validated :class:`core.schemas.FeatureRow` objects.

    Validation happens *here*, on purpose: a feature row that violates the contract
    (an unknown column, a negative count, a label smuggled in) fails before anything is
    written, so a partially-written feature table is impossible.
    """
    if frame.empty:
        return []
    unknown = [column for column in frame.columns if column not in FEATURE_COLUMNS]
    if unknown:
        raise FeatureBuildError(
            f"the frame carries column(s) {unknown} that the feature contract does not "
            "declare; FEATURE_COLUMNS is the contract"
        )
    records = frame[list(FEATURE_COLUMNS)].to_dict("records")
    return [FeatureRow(**{column: _clean(value) for column, value in record.items()}) for record in records]
