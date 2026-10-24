"""Tests for :mod:`apps.anjaneeya_engine.candles`.

These are the tests that decide whether a candle can be trusted: open-time
keying, volume arithmetic from a cumulative counter, "no tick means no bar", and
the merge rules that make a 15m bar agree with the 1m bars it came from.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from apps.anjaneeya_engine.candles import (
    CandleBuildError,
    aggregate_candles,
    bucket_start,
    build_1m_candles,
    build_timeframes,
    expected_bar_openings,
    missing_bar_summary,
)
from core.schemas import QualityFlag, Source
from tests.support.engine import (
    CE_SYMBOL,
    CE_TOKEN,
    EXPIRY,
    IST,
    RUN_ID,
    SESSION_END,
    SESSION_START,
    at,
    make_candle,
    make_tick,
    tick_frame,
    ticks_for_minutes,
)

TIMEFRAMES = ("1m", "3m", "5m", "15m")


def build(ticks, **overrides):
    """Build 1m bars from tick dicts, via the same frame the reader produces."""
    tradingsymbol = overrides.pop("tradingsymbol", CE_SYMBOL)
    return build_1m_candles(
        tick_frame(ticks), tradingsymbol=tradingsymbol, run_id=RUN_ID, **overrides
    )


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
def test_bucket_start_floors_to_the_open_time_and_keeps_the_timezone() -> None:
    assert bucket_start(at(0, seconds=59), 1) == at(0)
    assert bucket_start(at(4, seconds=59), 5) == at(0)
    assert bucket_start(at(5), 5) == at(5)
    assert bucket_start(at(16), 15) == at(15)
    assert bucket_start(at(0), 15).tzinfo is IST

    floored = bucket_start(datetime(2026, 9, 29, 10, 7, 42, 123456, tzinfo=IST), 5)
    assert (floored.second, floored.microsecond) == (0, 0)
    assert floored == datetime(2026, 9, 29, 10, 5, tzinfo=IST)


def test_bucket_start_rejects_a_non_positive_width() -> None:
    with pytest.raises(CandleBuildError):
        bucket_start(at(0), 0)


# ---------------------------------------------------------------------------
# 1m construction
# ---------------------------------------------------------------------------
def test_a_bar_is_keyed_by_its_open_time_and_holds_that_minute_only() -> None:
    bars, stats = build(ticks_for_minutes([0]))

    assert len(bars) == 1
    bar = bars[0]
    # 09:15:05 and 09:15:35 -> the 09:15 bar (its open time), never 09:16.
    assert bar.timestamp == SESSION_START
    assert bar.open == 120.5
    assert bar.high == 121.0
    assert bar.low == 120.5
    assert bar.close == 121.0
    assert bar.timeframe == "1m"
    assert stats.ticks == 2
    assert stats.bars == 1
    assert stats.first_timestamp == SESSION_START
    assert stats.last_timestamp == SESSION_START


def test_volume_is_the_sum_of_the_cumulative_increments_inside_the_bar() -> None:
    bars, _ = build(ticks_for_minutes([0, 1, 2]))

    # The day's first tick has no predecessor, so the 09:15 bar holds the
    # increments from 09:15:05 onwards (last - first inside its minute = 10);
    # every later bar sees both of its ticks as increments (2 x 10).
    assert [bar.volume for bar in bars] == [10, 20, 20]
    assert [bar.timestamp for bar in bars] == [at(0), at(1), at(2)]


def test_a_minute_without_ticks_produces_no_bar() -> None:
    bars, stats = build(ticks_for_minutes([0, 1, 3]))

    assert [bar.timestamp for bar in bars] == [at(0), at(1), at(3)]
    assert stats.bars == 3


def test_a_volume_counter_reset_is_counted_and_never_goes_negative() -> None:
    ticks = ticks_for_minutes([0, 1])
    # The exchange counter restarts (re-subscribe / contract roll) in minute 2.
    ticks.append(make_tick(timestamp=at(2, seconds=5), volume_traded=5, last_price=130.0))

    bars, stats = build(ticks)

    assert stats.volume_resets == 1
    assert [bar.volume for bar in bars] == [10, 20, 0]
    assert bars[2].close == 130.0


def test_volume_is_none_when_the_contract_publishes_no_volume() -> None:
    without_column = [
        {key: value for key, value in tick.items() if key != "volume_traded"}
        for tick in ticks_for_minutes([0, 1])
    ]
    all_null = [dict(tick, volume_traded=None) for tick in ticks_for_minutes([0, 1])]

    assert [bar.volume for bar in build(without_column)[0]] == [None, None]
    assert [bar.volume for bar in build(all_null)[0]] == [None, None]


def test_contract_fields_come_from_the_bar_and_oi_from_the_newest_tick() -> None:
    ticks = ticks_for_minutes([0, 1])
    ticks[2] = dict(ticks[2], oi=None)  # first tick of minute 1 has no OI yet

    bars, _ = build(ticks)

    bar = bars[1]
    assert bar.oi == 5000  # the last non-null observation wins
    assert bar.instrument_token == CE_TOKEN
    assert bar.option_type.value == "CE"
    assert bar.strike == 25000.0
    assert bar.expiry == EXPIRY
    assert bar.tradingsymbol == CE_SYMBOL
    assert bar.fetch_timestamp == at(1, seconds=35) + timedelta(milliseconds=200)
    # Stage 3.3 fills the greeks and IV; a candle must never invent them.
    assert bar.iv is None
    assert bar.delta is None
    assert bar.days_to_expiry is None
    assert bar.source is Source.DERIVED
    assert bar.quality_flag is QualityFlag.OK


def test_out_of_order_ticks_are_sorted_before_the_bar_is_built() -> None:
    bars, _ = build(list(reversed(ticks_for_minutes([0, 1]))))

    assert [bar.timestamp for bar in bars] == [at(0), at(1)]
    assert bars[0].open == 120.5  # the earliest tick, not the first one in the frame
    assert bars[0].close == 121.0
    assert bars[0].volume == 10


def test_an_empty_frame_is_not_an_error() -> None:
    empty = tick_frame(ticks_for_minutes([0])).iloc[0:0]

    bars, stats = build_1m_candles(empty, tradingsymbol=CE_SYMBOL, run_id=RUN_ID)

    assert bars == []
    assert (stats.ticks, stats.bars, stats.volume_resets) == (0, 0, 0)
    assert stats.first_timestamp is None


def test_a_frame_missing_a_required_column_is_rejected() -> None:
    frame = tick_frame(ticks_for_minutes([0])).drop(columns=["last_price"])

    with pytest.raises(CandleBuildError, match="missing column"):
        build_1m_candles(frame, tradingsymbol=CE_SYMBOL, run_id=RUN_ID)


def test_a_missing_price_is_rejected_rather_than_guessed() -> None:
    frame = tick_frame(ticks_for_minutes([0]))
    frame.loc[0, "last_price"] = float("nan")

    with pytest.raises(CandleBuildError, match="last_price"):
        build_1m_candles(frame, tradingsymbol=CE_SYMBOL, run_id=RUN_ID)


def test_naive_timestamps_are_rejected() -> None:
    frame = tick_frame(ticks_for_minutes([0]))
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)

    with pytest.raises(CandleBuildError, match="naive"):
        build_1m_candles(frame, tradingsymbol=CE_SYMBOL, run_id=RUN_ID)


# ---------------------------------------------------------------------------
# Higher timeframes (merged from the 1m bars, never rebuilt from ticks)
# ---------------------------------------------------------------------------
def test_higher_timeframes_merge_the_1m_bars_they_cover() -> None:
    bars, _ = build(ticks_for_minutes(range(0, 6)))

    merged = aggregate_candles(bars, timeframe="5m")

    assert [bar.timestamp for bar in merged] == [at(0), at(5)]
    first = merged[0]
    assert first.open == bars[0].open
    assert first.close == bars[4].close
    assert first.high == max(bar.high for bar in bars[:5])
    assert first.low == min(bar.low for bar in bars[:5])
    assert first.volume == sum(bar.volume for bar in bars[:5]) == 90
    assert first.timeframe == "5m"
    assert merged[1].volume == 20
    assert merged[1].open == bars[5].open


def test_a_merged_bar_keeps_contract_fields_and_the_newest_metadata() -> None:
    bars = [
        make_candle(timestamp=at(0), iv=0.14, delta=0.51, volume=10, oi=5000),
        make_candle(
            timestamp=at(1),
            iv=0.15,
            delta=0.51,
            volume=None,
            oi=None,
            fetch_timestamp=at(1, seconds=59),
        ),
        make_candle(timestamp=at(2), iv=None, delta=0.51, volume=5, oi=5100),
    ]

    merged = aggregate_candles(bars, timeframe="3m")

    assert len(merged) == 1
    bar = merged[0]
    assert bar.timestamp == at(0)
    assert bar.volume == 15  # a None volume is unknown, never a zero
    assert bar.oi == 5100  # newest non-null
    assert bar.iv == 0.15  # newest non-null IV (the last bar had none yet)
    assert bar.delta == 0.51  # per-contract greeks carry over
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 101.0, 99.0, 100.5)
    assert bar.fetch_timestamp == at(1, seconds=59)
    assert bar.run_id == RUN_ID


def test_a_bucket_with_missing_1m_bars_merges_what_exists() -> None:
    bars, _ = build(ticks_for_minutes([0, 1, 4]))  # 09:19 produced no bar

    merged = aggregate_candles(bars, timeframe="15m")

    assert len(merged) == 1
    assert merged[0].volume == 50  # three observed minutes, not five (10 + 20 + 20)
    assert merged[0].close == bars[-1].close
    assert merged[0].timestamp == bars[0].timestamp


def test_volume_is_none_when_every_input_bar_has_no_volume() -> None:
    bars = [make_candle(timestamp=at(0), volume=None), make_candle(timestamp=at(1), volume=None)]

    merged = aggregate_candles(bars, timeframe="3m")

    assert merged[0].volume is None


def test_aggregating_1m_is_a_no_op_that_returns_a_copy() -> None:
    bars, _ = build(ticks_for_minutes([0]))

    again = aggregate_candles(bars, timeframe="1m")

    assert again == bars
    assert again is not bars


def test_aggregating_nothing_is_nothing() -> None:
    assert aggregate_candles([], timeframe="5m") == []


def test_aggregating_rejects_an_unknown_timeframe() -> None:
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        aggregate_candles([], timeframe="7m")


# ---------------------------------------------------------------------------
# build_timeframes
# ---------------------------------------------------------------------------
def test_build_timeframes_returns_every_requested_label_in_order() -> None:
    bars, _ = build(ticks_for_minutes(range(0, 16)))

    built = build_timeframes(bars, timeframes=TIMEFRAMES)

    assert list(built) == list(TIMEFRAMES)
    assert built["1m"] == bars
    assert built["1m"] is not bars
    assert len(built["1m"]) == 16
    assert len(built["3m"]) == 6  # 09:15, 09:18, 09:21, 09:24, 09:27, 09:30
    assert len(built["5m"]) == 4
    assert len(built["15m"]) == 2
    for timeframe in TIMEFRAMES:
        assert built[timeframe][0].timestamp == at(0)
        assert all(bar.timeframe == timeframe for bar in built[timeframe])


def test_build_timeframes_rejects_a_typo_instead_of_returning_an_empty_list() -> None:
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        build_timeframes([], timeframes=["1m", "7m"])


# ---------------------------------------------------------------------------
# Session expectations and gaps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(1, 375), (3, 125), (5, 75), (15, 25)],
)
def test_a_normal_session_has_the_documented_bar_counts(minutes: int, expected: int) -> None:
    openings = expected_bar_openings(
        session_start=SESSION_START, session_end=SESSION_END, minutes=minutes
    )

    assert len(openings) == expected
    assert openings[0] == SESSION_START
    assert openings[-1] < SESSION_END
    assert all(opening.tzinfo is IST for opening in openings)


def test_a_half_day_session_produces_half_the_bars() -> None:
    end = datetime(2026, 9, 29, 12, 30, tzinfo=IST)

    openings = expected_bar_openings(session_start=SESSION_START, session_end=end, minutes=5)

    assert len(openings) == 39  # 09:15 -> 12:30 is 195 minutes
    assert openings[-1] == datetime(2026, 9, 29, 12, 25, tzinfo=IST)


def test_missing_bars_are_reported_per_run_of_consecutive_holes() -> None:
    bars, _ = build(ticks_for_minutes([0, 1, 4]))

    summary = missing_bar_summary(bars, session_start=at(0), session_end=at(6), minutes=1)

    assert summary.expected_bars == 6
    assert summary.missing_bars == 3  # 09:17, 09:18 and 09:20
    assert summary.gap_count == 2  # two separate silences
    assert summary.largest_gap_seconds == 120.0


def test_a_complete_session_reports_no_gaps() -> None:
    bars, _ = build(ticks_for_minutes(range(0, 6)))

    summary = missing_bar_summary(bars, session_start=at(0), session_end=at(6), minutes=1)

    assert (summary.expected_bars, summary.missing_bars, summary.gap_count) == (6, 0, 0)
    assert summary.largest_gap_seconds == 0.0


def test_a_session_without_bars_is_one_long_gap() -> None:
    summary = missing_bar_summary(
        [], session_start=SESSION_START, session_end=SESSION_END, minutes=5
    )

    assert (summary.expected_bars, summary.missing_bars, summary.gap_count) == (75, 75, 1)
    assert summary.largest_gap_seconds == 75 * 300.0
