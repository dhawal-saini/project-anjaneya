"""Tests for :mod:`core.features.feature_builder` (the unified feature row, spec 18).

Three things are worth protecting here, and each of them is a *silent* failure mode
if it regresses:

* the row's column set is exactly ``FEATURE_COLUMNS`` - a column that is computed but
  never stored, or stored but never computed, is a contract break;
* the NIFTY block is joined by **bar close instant**, so a coarse index series can
  never hand a finer contract bar a value that was not knowable yet;
* the lookback window is a documented number (the longest average plus the slope
  lookback), because a shorter one quietly turns ``nifty_long_score`` into ``NaN``.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from apps.anjaneeya_engine.candles import bucket_start
from core.constants import FEATURE_COLUMNS
from core.features import (
    CONTRACT_BLOCK_COLUMNS,
    FEATURE_GROUPS,
    NIFTY_FEATURE_COLUMNS,
    FeatureBuildError,
    aggregate_index_frame,
    build_feature_frame,
    close_moments,
    contract_feature_block,
    feature_rows,
    nifty_feature_block,
    required_lookback_bars,
)
from core.market_calendar import NseCalendar
from tests.support.engine import (
    CE_SYMBOL,
    EXPIRY,
    SESSION_START,
    feature_window,
    window_candles,
    window_index,
)


@pytest.fixture
def calendar(config) -> NseCalendar:
    return NseCalendar(config.calendar)


def assemble(config, calendar, *, window=None, index=True, **overrides) -> pd.DataFrame:
    """Build a feature frame for the standard synthetic window."""
    window = window or feature_window()
    arguments = {
        "candles": window.candles,
        "index_candles": window.index if index else None,
        "calendar": calendar,
        "config": config.indicators,
        "timeframe": "1m",
        "underlying": "NIFTY",
        "run_id": "run-1",
        "session_start": window.session_start,
        "session_end": window.session_end,
    }
    arguments.update(overrides)
    return build_feature_frame(**arguments)


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_groups_reconstruct_the_stored_contract() -> None:
    assembled = [column for group in FEATURE_GROUPS.values() for column in group]
    assert tuple(assembled) == FEATURE_COLUMNS
    assert len(set(assembled)) == len(assembled)


@pytest.mark.unit
def test_the_contract_block_is_every_engine_group() -> None:
    assert len(CONTRACT_BLOCK_COLUMNS) == 52
    assert "candle_body" in CONTRACT_BLOCK_COLUMNS
    assert "support_strength" in CONTRACT_BLOCK_COLUMNS
    assert "option_trend_score" in CONTRACT_BLOCK_COLUMNS
    assert "nifty_regime" not in CONTRACT_BLOCK_COLUMNS


@pytest.mark.unit
def test_the_lookback_is_the_longest_average_plus_the_slope_window(config) -> None:
    """SMA 2056 + slope k (5) + one bar for the difference the slope takes."""
    assert required_lookback_bars(config.indicators) == 2062


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_contract_block_is_exactly_the_declared_columns(config) -> None:
    block = contract_feature_block(window_candles(), config=config.indicators)
    assert list(block.columns) == list(CONTRACT_BLOCK_COLUMNS)
    assert len(block) == 72


@pytest.mark.unit
def test_the_contract_block_refuses_an_unordered_frame(config) -> None:
    frame = window_candles().iloc[::-1]
    with pytest.raises(FeatureBuildError, match="ordered by timestamp"):
        contract_feature_block(frame, config=config.indicators)


@pytest.mark.unit
def test_the_contract_block_needs_ohlc(config) -> None:
    frame = window_candles().drop(columns=["close"])
    with pytest.raises(FeatureBuildError, match="missing column"):
        contract_feature_block(frame, config=config.indicators)


@pytest.mark.unit
def test_an_empty_frame_still_has_the_whole_block_shape(config) -> None:
    empty = window_candles().iloc[0:0]
    block = contract_feature_block(empty, config=config.indicators)
    assert list(block.columns) == list(CONTRACT_BLOCK_COLUMNS)
    assert block.empty


@pytest.mark.unit
def test_the_nifty_block_has_the_seven_nifty_columns(config) -> None:
    block = nifty_feature_block(window_index(), config=config.indicators)
    assert list(block.columns) == list(NIFTY_FEATURE_COLUMNS)
    assert block["nifty_composite_trend_score"].notna().any()


@pytest.mark.unit
def test_a_missing_index_series_leaves_the_nifty_block_null_without_raising(config) -> None:
    block = nifty_feature_block(None, config=config.indicators)
    assert list(block.columns) == list(NIFTY_FEATURE_COLUMNS)
    assert block.empty


# ---------------------------------------------------------------------------
# Index resampling
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_index_resampling_uses_the_same_bucket_rule_as_the_candle_builder() -> None:
    """Two definitions of "when does a 15m bar start" would be a quiet disagreement."""
    frame = window_index(timeframe="1m", lookback_bars=30, session_bars=30)
    aggregated = aggregate_index_frame(frame, timeframe="15m")

    expected = []
    seen = set()
    for moment in frame["timestamp"]:
        opening = bucket_start(moment.to_pydatetime(), 15)
        if opening not in seen:
            seen.add(opening)
            expected.append(pd.Timestamp(opening))
    assert list(aggregated["timestamp"]) == expected


@pytest.mark.unit
def test_an_already_aggregated_index_frame_is_returned_unchanged() -> None:
    frame = window_index(timeframe="15m", lookback_bars=15, session_bars=15)
    assert aggregate_index_frame(frame, timeframe="15m") is not None
    assert list(aggregate_index_frame(frame, timeframe="15m")["timestamp"]) == list(
        frame["timestamp"]
    )


@pytest.mark.unit
def test_index_resampling_keeps_the_bars_ohlc_honest() -> None:
    frame = window_index(timeframe="1m", lookback_bars=30, session_bars=30)
    aggregated = aggregate_index_frame(frame, timeframe="5m")
    first = frame.iloc[:5]
    assert aggregated.iloc[0]["open"] == first.iloc[0]["open"]
    assert aggregated.iloc[0]["close"] == first.iloc[-1]["close"]
    assert aggregated.iloc[0]["high"] == max(frame["high"].iloc[:5])
    assert aggregated.iloc[0]["low"] == min(frame["low"].iloc[:5])


@pytest.mark.unit
def test_close_moments_are_open_time_plus_the_timeframe() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime([SESSION_START])})
    assert list(close_moments(frame, timeframe="15m")) == [
        pd.Timestamp(SESSION_START + timedelta(minutes=15))
    ]


@pytest.mark.unit
def test_close_moments_reads_the_frames_own_timeframe_column() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime([SESSION_START]), "timeframe": ["1m"]})
    assert list(close_moments(frame)) == [pd.Timestamp(SESSION_START + timedelta(minutes=1))]


@pytest.mark.unit
def test_close_moments_refuses_an_ambiguous_frame() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime([SESSION_START, SESSION_START])})
    with pytest.raises(FeatureBuildError, match="needs a timeframe"):
        close_moments(frame)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def test_the_published_frame_is_exactly_the_stored_contract(config, calendar) -> None:
    frame = assemble(config, calendar)

    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert len(frame) == 12  # only the session's bars are published
    assert frame["timeframe"].unique().tolist() == ["1m"]
    assert frame["underlying"].unique().tolist() == ["NIFTY"]
    assert frame["tradingsymbol"].unique().tolist() == [CE_SYMBOL]
    assert frame["run_id"].unique().tolist() == ["run-1"]
    assert frame["source"].unique().tolist() == ["DERIVED"]


def test_the_window_is_used_but_only_the_session_is_published(config, calendar) -> None:
    window = feature_window()
    published = assemble(config, calendar, window=window)
    everything = build_feature_frame(
        candles=window.candles,
        index_candles=window.index,
        calendar=calendar,
        config=config.indicators,
        timeframe="1m",
        underlying="NIFTY",
        run_id="run-1",
    )

    assert len(everything) == len(window.candles)
    assert published["timestamp"].iloc[0] == pd.Timestamp(window.session_start)
    assert published["timestamp"].iloc[-1] < pd.Timestamp(window.session_end)


def test_bar_to_bar_columns_are_differenced_over_the_window(config, calendar) -> None:
    """The first bar of a session is compared with the previous session's last bar."""
    frame = assemble(config, calendar)

    # A price change exists from the very first published bar...
    assert frame["actual_option_move"].iloc[0] == pytest.approx(0.05)
    # ...while IV was never captured, so its change is NULL rather than 0.
    assert frame["iv_change"].isna().all()
    assert frame["oi_change"].iloc[0] == 1.0


def test_distance_from_atm_is_the_strike_minus_the_index_close(config, calendar) -> None:
    window = feature_window()
    frame = assemble(config, calendar, window=window)

    close = window.index.set_index("timestamp")["close"]
    for row in frame.itertuples():
        assert row.distance_from_atm == pytest.approx(25000.0 - close.loc[row.timestamp])


def test_a_coarse_index_bar_is_never_used_before_it_closes(config, calendar) -> None:
    """A 15m index bar opening 09:15 closes 09:30: a 09:20 candle must not see it."""
    window = feature_window(lookback_bars=60, session_bars=30)
    coarse = aggregate_index_frame(window.index, timeframe="15m")
    closes = {row.timestamp: row.close for row in coarse.itertuples()}
    frame = assemble(config, calendar, window=window, index_candles=coarse)
    # ``distance_from_atm`` = strike - index close, so this inverts it back to the close.
    seen = {row.timestamp: 25000.0 - row.distance_from_atm for row in frame.itertuples()}

    for moment, close in sorted(seen.items()):
        closed_in_time = [
            value
            for opened, value in sorted(closes.items())
            if opened + timedelta(minutes=15) <= moment + timedelta(minutes=1)
        ]
        assert close == pytest.approx(closed_in_time[-1]), (
            f"{moment} used an index bar that had not closed yet"
        )


def test_without_index_history_the_nifty_columns_are_null_and_the_rest_is_not(
    config, calendar
) -> None:
    frame = assemble(config, calendar, index=False)

    for column in NIFTY_FEATURE_COLUMNS:
        assert frame[column].isna().all(), column
    assert frame["option_trend_score"].notna().any()
    assert frame["candle_body"].notna().any()
    assert frame["distance_from_atm"].isna().all()


def test_the_context_columns_come_from_the_calendar(config, calendar) -> None:
    frame = assemble(config, calendar)

    assert frame["days_to_expiry"].iloc[0] > 0
    assert frame["time_of_day_bucket"].iloc[0] == "OPEN_915_930"
    assert frame["day_of_week"].iloc[0] == 1  # Tuesday
    assert frame["expiry_cycle_phase"].iloc[0] in {"START_OF_CYCLE", "MID_CYCLE", "EXPIRY_DAY"}
    assert frame["expiry"].iloc[0] == EXPIRY


def test_the_long_score_is_null_when_the_window_is_shorter_than_the_lookback(
    config, calendar
) -> None:
    """The documented cost of a short window: the long role simply does not vote."""
    frame = assemble(config, calendar)

    assert frame["nifty_long_score"].isna().all()
    assert frame["nifty_fast_score"].notna().any()
    assert frame["nifty_composite_trend_score"].notna().any()


def test_the_relative_columns_wait_for_a_delta(config, calendar) -> None:
    """Spec 16's expected move needs a delta (Stage 3.3): NULL, not a guess."""
    frame = assemble(config, calendar)

    assert frame["expected_option_move"].isna().all()
    assert frame["option_residual"].isna().all()


def test_an_empty_window_produces_no_rows(config, calendar) -> None:
    frame = assemble(config, calendar, candles=window_candles().iloc[0:0])
    assert frame.empty
    assert list(frame.columns) == list(FEATURE_COLUMNS)


def test_duplicate_candle_timestamps_are_refused(config, calendar) -> None:
    window = feature_window()
    doubled = pd.concat([window.candles, window.candles.iloc[[-1]]], ignore_index=True)
    with pytest.raises(FeatureBuildError, match="duplicate timestamps"):
        assemble(config, calendar, candles=doubled)


def test_naive_timestamps_are_refused(config, calendar) -> None:
    window = feature_window()
    naive = window.candles.copy()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(FeatureBuildError, match="naive"):
        assemble(config, calendar, candles=naive)


# ---------------------------------------------------------------------------
# Row objects
# ---------------------------------------------------------------------------
def test_feature_rows_validates_every_published_row(config, calendar) -> None:
    frame = assemble(config, calendar)
    rows = feature_rows(frame)

    assert len(rows) == len(frame)
    assert rows[0].timestamp == pd.Timestamp(SESSION_START).to_pydatetime()
    assert rows[0].strike == 25000.0
    assert rows[0].option_type.value == "CE"
    assert rows[0].nifty_regime is not None
    assert rows[0].nifty_regime.value in {
        "STRONG_BULLISH",
        "BULLISH",
        "BULLISH_DEVELOPING",
        "NEUTRAL",
        "BEARISH_DEVELOPING",
        "BEARISH",
        "STRONG_BEARISH",
    }
    assert any(row.option_trend_score is not None for row in rows)


def test_feature_rows_rejects_a_column_the_contract_does_not_declare(config, calendar) -> None:
    frame = assemble(config, calendar)
    frame["future_return"] = 1.0
    with pytest.raises(FeatureBuildError, match="does not declare"):
        feature_rows(frame)


def test_feature_rows_of_an_empty_frame_is_an_empty_list() -> None:
    assert feature_rows(pd.DataFrame(columns=list(FEATURE_COLUMNS))) == []


# ---------------------------------------------------------------------------
# Spec 16's relative block (Stage 3.3): expected move, actual move, residual
# ---------------------------------------------------------------------------
def with_delta(window, values):
    """The window's candles with an ``iv``/``delta`` column, as enrichment would leave them."""
    candles = window.candles.copy()
    candles["delta"] = values
    candles["iv"] = [0.14 + 0.001 * position for position in range(len(candles))]
    return window.__class__(
        candles=candles,
        index=window.index,
        session_start=window.session_start,
        session_end=window.session_end,
    )


def test_expected_move_is_the_previous_delta_times_the_nifty_move(config, calendar) -> None:
    """``delta[t-1] * (nifty_close[t] - nifty_close[t-1])`` on every published row.

    The index fixture rises exactly one point per minute and the contract's close rises
    0.05, so both the expectation and the residual are predictable numbers rather than
    "whatever the engine produced".
    """
    window = with_delta(feature_window(), 0.5)
    frame = assemble(config, calendar, window=window)

    assert list(frame["expected_option_move"]) == pytest.approx([0.5] * len(frame))
    assert list(frame["actual_option_move"]) == pytest.approx([0.05] * len(frame))
    assert list(frame["option_residual"]) == pytest.approx([0.05 - 0.5] * len(frame))


def test_the_delta_is_the_one_from_the_bar_before_the_move(config, calendar) -> None:
    """A delta that changes *on* the bar must not be the one that explains the move."""
    window = feature_window()
    deltas = [0.0] * (len(window.candles) - 1) + [1.0]  # the last bar flips the delta
    frame = assemble(config, calendar, window=with_delta(window, deltas))

    # The final bar's move is explained by the delta *it started with* (0.0), not by the
    # new one - using delta[t] would have made expected 1.0 and the residual -0.95.
    assert float(frame["expected_option_move"].iloc[-1]) == pytest.approx(0.0)
    assert float(frame["option_residual"].iloc[-1]) == pytest.approx(0.05)


def test_the_relative_block_is_null_until_a_delta_exists(config, calendar) -> None:
    """An unenriched build publishes the actual move and says nothing about the rest."""
    frame = assemble(config, calendar)

    assert frame["expected_option_move"].isna().all()
    assert frame["option_residual"].isna().all()
    assert not frame["actual_option_move"].isna().any()
    assert frame["iv_change"].isna().all()  # no IV either


def test_the_relative_block_is_null_without_a_nifty_move(config, calendar) -> None:
    """No index history means no expectation - the actual move is still published."""
    window = with_delta(feature_window(), 0.5)
    frame = assemble(config, calendar, window=window, index=False)

    assert frame["expected_option_move"].isna().all()
    assert frame["option_residual"].isna().all()
    assert not frame["actual_option_move"].isna().any()


def test_iv_change_is_the_bar_to_bar_difference_of_the_enriched_iv(config, calendar) -> None:
    """``iv_change`` becomes a real number once the candles carry an IV."""
    frame = assemble(config, calendar, window=with_delta(feature_window(), 0.5))

    assert list(frame["iv_change"]) == pytest.approx([0.001] * len(frame))
    assert not frame["iv"].isna().any()


def test_an_unsolved_delta_makes_the_next_rows_expectation_null(config, calendar) -> None:
    """One bar with no IV cannot explain its own move *or* the next bar's."""
    window = feature_window()
    deltas = [0.5] * len(window.candles)
    deltas[-2] = float("nan")
    frame = assemble(config, calendar, window=with_delta(window, deltas))

    # The unsolved bar's own move is still explained by the delta it started with ...
    assert float(frame["expected_option_move"].iloc[-2]) == pytest.approx(0.5)
    # ... and the bar after it has no previous delta to use, so its expectation is NULL.
    assert pd.isna(frame["expected_option_move"].iloc[-1])
    assert pd.isna(frame["option_residual"].iloc[-1])
    assert not pd.isna(frame["actual_option_move"].iloc[-1])
    # The rows before it are untouched.
    assert float(frame["expected_option_move"].iloc[-3]) == pytest.approx(0.5)
