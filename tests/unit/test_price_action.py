"""Tests for the price-action engines (spec 11-12, 15).

The engines under test are pure frame -> frame functions, so the tests are
written the same way they are: build a small candle frame by hand, call the
engine, and assert on numbers a reader can verify in their head.

Two invariants get special attention, because breaking either silently corrupts
the feature table:

* **no look-ahead** - a bar's candle, level and swing values must not move when
  later bars are appended (the swing confirmation delay and the running highs are
  the easy places to get this wrong);
* **ATR-normalised distances** - every ``distance_*`` column is in ATR units, and
  its sign says which side of the level price is on.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.config_loader import (
    CandleBehaviourConfig,
    LevelInteractionConfig,
    LevelsConfig,
    SupportResistanceConfig,
    SupportResistanceWeightsConfig,
    SwingConfig,
)
from core.price_action.candle_behaviour import (
    FEATURE_ORDER,
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
from core.price_action.levels import level_features, session_levels, trade_dates
from core.price_action.structure import market_structure, structure_features
from core.price_action.support_resistance import (
    CANDIDATE_STATISTIC_COLUMNS,
    SUPPORT_RESISTANCE_FEATURE_COLUMNS,
    candidate_levels,
    candidate_statistics,
    support_resistance_detail,
    support_resistance_features,
)
from core.price_action.swings import (
    recent_swings,
    swing_levels,
    swing_points,
)
from core.schemas import CandleDirection, FeatureRow, MarketStructureState

IST = ZoneInfo("Asia/Kolkata")
FIRST_BAR = datetime(2026, 9, 29, 9, 15, tzinfo=IST)


def build_frame(rows: list[tuple[float, float, float, float]], *, minutes: int = 1) -> pd.DataFrame:
    """A candle frame from ``(open, high, low, close)`` tuples, 1 minute apart."""
    stamps = [FIRST_BAR + timedelta(minutes=minutes * index) for index in range(len(rows))]
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
        }
    )


def flat_frame(prices: list[float]) -> pd.DataFrame:
    """One bar per price, each with a zero range (the hardest case for ratios)."""
    return build_frame([(price, price, price, price) for price in prices])


CANDLE = CandleBehaviourConfig()


# ---------------------------------------------------------------------------
# Candle shape (spec 11, geometry)
# ---------------------------------------------------------------------------
def test_shape_measures_body_range_and_both_wicks() -> None:
    shape = candle_shape(build_frame([(100.0, 110.0, 95.0, 108.0)]))
    assert shape["candle_body"].iloc[0] == pytest.approx(8.0)
    assert shape["candle_range"].iloc[0] == pytest.approx(15.0)
    assert shape["candle_upper_wick"].iloc[0] == pytest.approx(2.0)
    assert shape["candle_lower_wick"].iloc[0] == pytest.approx(5.0)


def test_shape_ratios_are_fractions_of_the_range() -> None:
    shape = candle_shape(build_frame([(100.0, 110.0, 95.0, 108.0)]))
    assert shape["candle_body_ratio"].iloc[0] == pytest.approx(8 / 15)
    assert shape["candle_upper_wick_ratio"].iloc[0] == pytest.approx(2 / 15)
    assert shape["candle_lower_wick_ratio"].iloc[0] == pytest.approx(5 / 15)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ((100.0, 105.0, 95.0, 104.0), CandleDirection.BULLISH.value),
        ((104.0, 105.0, 95.0, 100.0), CandleDirection.BEARISH.value),
        ((100.0, 105.0, 95.0, 100.0), CandleDirection.FLAT.value),
    ],
)
def test_shape_names_the_direction(row: tuple, expected: str) -> None:
    assert candle_shape(build_frame([row]))["candle_direction"].iloc[0] == expected


def test_shape_ratios_are_undefined_for_a_zero_range_candle() -> None:
    shape = candle_shape(flat_frame([100.0]))
    assert shape["candle_body"].iloc[0] == 0.0
    assert shape["candle_range"].iloc[0] == 0.0
    assert pd.isna(shape["candle_body_ratio"].iloc[0])
    assert pd.isna(shape["candle_upper_wick_ratio"].iloc[0])
    assert shape["candle_direction"].iloc[0] == CandleDirection.FLAT.value


def test_shape_rejects_a_bar_whose_high_is_below_its_low() -> None:
    with pytest.raises(ValueError, match="high is below its low"):
        candle_shape(build_frame([(100.0, 90.0, 95.0, 92.0)]))


def test_shape_rejects_a_frame_without_ohlc() -> None:
    with pytest.raises(ValueError, match="missing column"):
        candle_shape(pd.DataFrame({"close": [1.0]}))


def test_shape_of_an_empty_frame_is_empty_not_an_error() -> None:
    assert candle_shape(pd.DataFrame(columns=["open", "high", "low", "close"])).empty


# ---------------------------------------------------------------------------
# Candle relationships (spec 11, versus the previous candle)
# ---------------------------------------------------------------------------
def test_the_first_candle_has_no_relationship_to_anything() -> None:
    frame = build_frame([(100.0, 105.0, 95.0, 104.0), (104.0, 108.0, 100.0, 107.0)])
    assert candle_relationships(frame).iloc[0].isna().all()


def test_gap_is_the_open_minus_the_previous_close() -> None:
    frame = build_frame([(100.0, 105.0, 95.0, 104.0), (108.0, 112.0, 107.0, 110.0)])
    assert candle_relationships(frame)["candle_gap"].iloc[1] == pytest.approx(4.0)


def test_a_bullish_candle_engulfing_a_bearish_one_is_positive_one() -> None:
    frame = build_frame([(100.0, 101.0, 97.0, 98.0), (97.0, 102.0, 96.0, 101.0)])
    assert candle_relationships(frame)["candle_engulf_ratio"].iloc[1] == pytest.approx(1.0)


def test_a_bearish_candle_engulfing_a_bullish_one_is_negative_one() -> None:
    frame = build_frame([(98.0, 102.0, 97.0, 100.0), (101.0, 102.0, 96.0, 97.0)])
    assert candle_relationships(frame)["candle_engulf_ratio"].iloc[1] == pytest.approx(-1.0)


def test_same_direction_candles_cannot_engulf() -> None:
    frame = build_frame([(98.0, 101.0, 97.0, 100.0), (100.0, 103.0, 99.0, 102.0)])
    assert candle_relationships(frame)["candle_engulf_ratio"].iloc[1] == 0.0


def test_inside_and_outside_flags_compare_the_ranges() -> None:
    inside = build_frame([(100.0, 105.0, 95.0, 102.0), (101.0, 104.0, 96.0, 103.0)])
    outside = build_frame([(100.0, 105.0, 95.0, 102.0), (99.0, 106.0, 94.0, 100.0)])
    assert candle_relationships(inside)["candle_inside_flag"].iloc[1]
    assert not candle_relationships(inside)["candle_outside_flag"].iloc[1]
    assert candle_relationships(outside)["candle_outside_flag"].iloc[1]
    assert not candle_relationships(outside)["candle_inside_flag"].iloc[1]


def test_close_position_within_the_previous_range_is_not_clipped() -> None:
    frame = build_frame([(100.0, 105.0, 95.0, 102.0), (100.0, 112.0, 99.0, 110.0)])
    position = candle_relationships(frame)["candle_close_position_within_prev_range"].iloc[1]
    assert position == pytest.approx(1.5)  # gapped clean through the previous high


def test_close_position_is_undefined_when_the_previous_range_was_zero() -> None:
    frame = build_frame([(100.0, 100.0, 100.0, 100.0), (100.0, 105.0, 99.0, 104.0)])
    assert pd.isna(candle_relationships(frame)["candle_close_position_within_prev_range"].iloc[1])


def test_shape_ratio_changes_are_bar_on_bar_differences() -> None:
    frame = build_frame([(100.0, 110.0, 100.0, 105.0), (105.0, 106.0, 104.0, 105.0)])
    relationships = candle_relationships(frame)
    shape = candle_shape(frame)
    assert relationships["candle_body_ratio_change"].iloc[1] == pytest.approx(
        shape["candle_body_ratio"].iloc[1] - shape["candle_body_ratio"].iloc[0]
    )
    wicks = shape["candle_upper_wick_ratio"] + shape["candle_lower_wick_ratio"]
    assert relationships["candle_wick_ratio_change"].iloc[1] == pytest.approx(
        wicks.iloc[1] - wicks.iloc[0]
    )


# ---------------------------------------------------------------------------
# Candle patterns and the feature callback (spec 11/18)
# ---------------------------------------------------------------------------
def test_patterns_name_a_small_body_and_the_engulfing_candles() -> None:
    # a tiny body inside a long range, after a wide bearish candle
    frame = build_frame([(100.0, 101.0, 97.0, 98.0), (97.0, 102.0, 96.0, 101.0)])
    patterns = candle_patterns(frame, config=CANDLE)
    assert bool(patterns["bullish_engulfing"].iloc[1])
    assert not bool(patterns["bearish_engulfing"].iloc[1])
    expected = candle_shape(frame)["candle_body_ratio"].iloc[0] < CANDLE.small_body_ratio
    assert bool(patterns["small_body"].iloc[0]) == expected


def test_patterns_name_a_long_wick() -> None:
    long_upper = build_frame([(96.0, 110.0, 95.0, 97.0)])  # body 1 of range 15
    long_lower = build_frame([(104.0, 105.0, 90.0, 103.0)])  # body 1 of range 15
    assert bool(candle_patterns(long_upper, config=CANDLE)["long_upper_wick"].iloc[0])
    assert not bool(candle_patterns(long_upper, config=CANDLE)["long_lower_wick"].iloc[0])
    assert bool(candle_patterns(long_lower, config=CANDLE)["long_lower_wick"].iloc[0])


def test_patterns_are_undefined_rather_than_false_on_a_zero_range_candle() -> None:
    patterns = candle_patterns(flat_frame([100.0]), config=CANDLE)
    assert patterns["small_body"].isna().all()
    assert patterns["long_upper_wick"].isna().all()


def test_candle_features_returns_exactly_the_contract_columns_in_order() -> None:
    features = candle_features(build_frame([(100.0, 105.0, 95.0, 104.0)]))
    assert tuple(features.columns) == FEATURE_ORDER
    assert features.index[0] == 0


def test_candle_features_do_not_change_when_later_bars_arrive() -> None:
    rows = [
        (100.0, 105.0, 95.0, 104.0),
        (104.0, 108.0, 100.0, 101.0),
        (101.0, 103.0, 98.0, 102.0),
        (102.0, 112.0, 101.0, 111.0),
    ]
    full = candle_features(build_frame(rows))
    for cut in range(1, len(rows) + 1):
        partial = candle_features(build_frame(rows[:cut]))
        for column in FEATURE_ORDER:
            left = full[column].iloc[cut - 1]
            right = partial[column].iloc[-1]
            if pd.isna(left) and pd.isna(right):
                continue
            assert left == pytest.approx(right), f"{column} moved at bar {cut - 1}"



# ---------------------------------------------------------------------------
# Swings (spec 15) - confirmation delay is the point of the whole engine
# ---------------------------------------------------------------------------
#: A hand-designed shape with exactly two swing highs (24 at bar 4, 25 at bar 12)
#: and two swing lows (8 at bar 3, 9 at bar 10) for ``left_bars=right_bars=3``.
#: Both lows sit at least ``left_bars`` from the start, so they are detectable at
#: all - a swing in the first three bars can never have three bars to its left.
SWING_HIGHS = [20, 21, 22, 23, 24, 23, 22, 21, 20, 21, 22, 23, 25, 24, 23, 22, 21, 20, 21]
SWING_LOWS = [11, 9, 10, 8, 12, 13, 11, 14, 15, 16, 9, 12, 14, 15, 16, 17, 18, 19, 20]
SWINGS = SwingConfig()


def swing_frame() -> pd.DataFrame:
    """The hand-designed swing shape as a candle frame (``high >= low`` holds)."""
    rows = [
        ((high + low) / 2, high, low, (high + low) / 2)
        for high, low in zip(SWING_HIGHS, SWING_LOWS, strict=True)
    ]
    return build_frame(rows)


def test_swing_points_find_the_designed_highs_and_lows() -> None:
    points = swing_points(swing_frame(), config=SWINGS)
    assert list(zip(points["kind"], points["position"], points["price"], strict=True)) == [
        ("low", 3, 8.0),
        ("high", 4, 24.0),
        ("low", 10, 9.0),
        ("high", 12, 25.0),
    ]


def test_a_swing_is_confirmed_right_bars_after_it_happened() -> None:
    points = swing_points(swing_frame(), config=SWINGS)
    assert list(points["confirmed_position"]) == [6, 7, 13, 15]
    assert points["confirmed_at"].iloc[1] == FIRST_BAR + timedelta(minutes=7)


def test_a_swing_needs_right_bars_after_it_before_it_exists() -> None:
    shape = swing_frame()
    assert swing_points(shape.iloc[:6], config=SWINGS).empty  # no bar can be confirmed
    assert not swing_points(shape.iloc[:7], config=SWINGS).empty  # confirmed on the last bar


def test_swing_high_1_appears_only_once_the_swing_is_confirmed() -> None:
    levels = swing_levels(swing_frame(), config=SWINGS)
    assert pd.isna(levels["swing_high_1"].iloc[6])
    assert levels["swing_high_1"].iloc[7] == 24.0
    assert levels["swing_high_1"].iloc[14] == 24.0  # the 25 is not confirmed yet
    assert levels["swing_high_1"].iloc[15] == 25.0


def test_recent_swings_rank_confirmed_swings_by_recency() -> None:
    frame = swing_frame()
    swings = recent_swings(frame, config=SWINGS, count=3)
    assert list(swings.columns) == [
        "swing_high_1",
        "swing_high_2",
        "swing_high_3",
        "swing_low_1",
        "swing_low_2",
        "swing_low_3",
    ]
    # a swing that happened further back is never defined before a more recent one
    assert swings["swing_high_1"].notna().sum() >= swings["swing_high_3"].notna().sum()
    # and the first two columns agree exactly with what the feature row stores
    levels = swing_levels(frame, config=SWINGS)
    assert swings["swing_high_2"].equals(levels["swing_high_2"])
    assert swings["swing_low_2"].equals(levels["swing_low_2"])


def test_recent_swings_reject_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        recent_swings(swing_frame(), config=SWINGS, count=0)


    levels = swing_levels(swing_frame(), config=SWINGS)
    assert pd.isna(levels["swing_label"].iloc[14])  # only one confirmed high pair
    assert levels["swing_label"].iloc[15:].tolist() == ["HH_HL"] * 4
    assert levels["swing_high_2"].iloc[15] == 24.0
    assert levels["swing_low_2"].iloc[15] == 8.0
    assert levels["swing_low_1"].iloc[13] == 9.0


def test_swing_levels_do_not_change_when_later_bars_arrive() -> None:
    frame = swing_frame()
    full = swing_levels(frame, config=SWINGS)
    for cut in range(1, len(frame) + 1):
        partial = swing_levels(frame.iloc[:cut], config=SWINGS)
        for column in ("swing_high_1", "swing_low_1", "swing_high_2", "swing_low_2", "swing_label"):
            left, right = full[column].iloc[cut - 1], partial[column].iloc[-1]
            if pd.isna(left) and pd.isna(right):
                continue
            assert left == right, f"{column} moved at bar {cut - 1}"


# ---------------------------------------------------------------------------
# Structure (spec 15) - reads the swing labels, never re-detects swings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("HH_HL", MarketStructureState.BULLISH_STRUCTURE.value),
        ("LH_LL", MarketStructureState.BEARISH_STRUCTURE.value),
        ("LH_HL", MarketStructureState.RANGE.value),
        ("HH_LL", MarketStructureState.TRANSITION.value),
        (None, None),
    ],
)
def test_market_structure_maps_every_swing_label(label: str | None, expected: str | None) -> None:
    assert market_structure(pd.Series([label])).iloc[0] == expected


def test_market_structure_refuses_an_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unknown swing label"):
        market_structure(pd.Series(["H_H"]))


def test_structure_features_report_state_and_atr_normalised_distances() -> None:
    frame = swing_frame()
    atr = pd.Series([1.0] * len(frame))
    features = structure_features(frame, config=SWINGS, atr=atr)
    expected_state = MarketStructureState.BULLISH_STRUCTURE.value
    assert features["market_structure_state"].iloc[18] == expected_state
    assert features["distance_from_swing_high"].iloc[18] == pytest.approx(
        frame["close"].iloc[18] - 25.0
    )
    assert features["distance_from_swing_low"].iloc[18] == pytest.approx(
        frame["close"].iloc[18] - 9.0
    )
    assert features["distance_from_swing_high"].iloc[:7].isna().all()


def test_structure_distances_are_undefined_without_an_atr_scale() -> None:
    frame = swing_frame()
    features = structure_features(frame, config=SWINGS, atr=pd.Series([0.0] * len(frame)))
    assert pd.isna(features["distance_from_swing_high"].iloc[18])
    assert not pd.isna(features["swing_high_1"].iloc[18])  # the level itself survives



# ---------------------------------------------------------------------------
# Levels (spec 12) - running highs and strictly-closed previous sessions
# ---------------------------------------------------------------------------
DAY_ONE = datetime(2026, 9, 28, 9, 15, tzinfo=IST)
DAY_TWO = datetime(2026, 9, 29, 9, 15, tzinfo=IST)
#: Session one closes at 100 with a 105 high and a 97 low; session two trades
#: 99..109, so every previous-day level is distinguishable from every running one.
SESSION_ONE = [
    (100.0, 102.0, 99.0, 101.0),
    (101.0, 105.0, 100.0, 104.0),
    (104.0, 103.0, 98.0, 99.0),
    (99.0, 101.0, 97.0, 100.0),
]
SESSION_TWO = [
    (100.0, 106.0, 99.0, 105.0),
    (105.0, 108.0, 104.0, 107.0),
    (107.0, 109.0, 106.0, 103.0),
    (103.0, 104.0, 101.0, 102.0),
]


def two_session_frame() -> pd.DataFrame:
    """Four bars on 2026-09-28 followed by four on 2026-09-29 (both 09:15 IST)."""
    moments = [*[DAY_ONE] * 4, *[DAY_TWO] * 4]
    bars = [*SESSION_ONE, *SESSION_TWO]
    return pd.DataFrame(
        [(moment, *bar) for moment, bar in zip(moments, bars, strict=True)],
        columns=["timestamp", "open", "high", "low", "close"],
    )


def test_trade_dates_are_ist_session_dates() -> None:
    dates = trade_dates(two_session_frame())
    assert dates.tolist() == [DAY_ONE.date()] * 4 + [DAY_TWO.date()] * 4


def test_trade_dates_reject_a_naive_timestamp() -> None:
    frame = two_session_frame()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        trade_dates(frame)


def test_trade_dates_need_a_timestamp_column() -> None:
    with pytest.raises(ValueError, match="'timestamp' column"):
        trade_dates(pd.DataFrame({"close": [1.0]}))


def test_today_open_is_the_first_bar_of_each_session() -> None:
    levels = session_levels(two_session_frame())
    assert levels["level_today_open"].tolist() == [100.0] * 8


def test_running_extremes_reset_at_each_session() -> None:
    levels = session_levels(two_session_frame())
    assert levels["level_today_high_running"].tolist() == [102.0, 105.0, 105.0, 105.0, 106.0, 108.0, 109.0, 109.0]
    # session two's low of 99 must not inherit session one's 97
    assert levels["level_today_low_running"].tolist() == [99.0, 99.0, 98.0, 97.0, 99.0, 99.0, 99.0, 99.0]


def test_previous_day_levels_come_from_the_last_closed_session() -> None:
    levels = session_levels(two_session_frame())
    assert levels["level_previous_day_open"].iloc[:4].isna().all()
    assert levels["level_previous_day_open"].iloc[4:].tolist() == [100.0] * 4
    assert levels["level_previous_day_high"].iloc[4:].tolist() == [105.0] * 4
    assert levels["level_previous_day_low"].iloc[4:].tolist() == [97.0] * 4
    assert levels["level_previous_day_close"].iloc[4:].tolist() == [100.0] * 4


def test_distances_are_measured_in_atr_units_with_the_levels_sign() -> None:
    frame = two_session_frame()
    features = level_features(frame, atr=pd.Series([1.0] * len(frame)))
    assert features["distance_to_previous_day_high"].iloc[5] == pytest.approx(2.0)
    assert features["distance_to_previous_day_high"].iloc[7] == pytest.approx(-3.0)
    assert features["distance_to_previous_day_low"].iloc[7] == pytest.approx(5.0)
    assert features["distance_to_today_high"].iloc[7] == pytest.approx(-7.0)
    assert features["distance_to_today_low"].iloc[7] == pytest.approx(3.0)


def test_an_atr_of_two_halves_every_distance() -> None:
    frame = two_session_frame()
    unit = level_features(frame, atr=pd.Series([1.0] * len(frame)))
    doubled = level_features(frame, atr=pd.Series([2.0] * len(frame)))
    assert doubled["distance_to_previous_day_high"].iloc[7] == pytest.approx(
        unit["distance_to_previous_day_high"].iloc[7] / 2
    )


def test_distances_are_undefined_without_an_atr_scale() -> None:
    frame = two_session_frame()
    features = level_features(frame, atr=pd.Series([0.0] * len(frame)))
    assert pd.isna(features["distance_to_previous_day_high"].iloc[7])
    assert features["level_previous_day_high"].iloc[7] == 105.0


def test_first_session_of_a_frame_has_no_previous_day_distances() -> None:
    features = level_features(two_session_frame(), atr=pd.Series([1.0] * 8))
    assert features["distance_to_previous_day_high"].iloc[:4].isna().all()
    assert not pd.isna(features["distance_to_today_high"].iloc[0])


def test_levels_do_not_change_when_later_bars_arrive() -> None:
    frame = two_session_frame()
    atr = pd.Series([1.0] * len(frame))
    full = level_features(frame, atr=atr)
    for cut in range(1, len(frame) + 1):
        partial = level_features(frame.iloc[:cut], atr=atr.iloc[:cut])
        for column in full.columns:
            left, right = full[column].iloc[cut - 1], partial[column].iloc[-1]
            if pd.isna(left) and pd.isna(right):
                continue
            assert left == pytest.approx(right), f"{column} moved at bar {cut - 1}"


def test_levels_of_an_empty_frame_are_empty() -> None:
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    assert session_levels(empty).empty


# ---------------------------------------------------------------------------
# Level interaction (spec 13) - the state machine, one level at a time
# ---------------------------------------------------------------------------
#: The interaction thresholds of ``config/indicators.yaml``: touch within 5% of
#: an ATR, approach within 25%, and two closes to call a break "held".
INTERACTION = LevelInteractionConfig()

#: The S/R weights of ``config/indicators.yaml``.
SR_WEIGHTS = SupportResistanceWeightsConfig(
    touches=0.30, rejections=0.25, holds=0.20, recency=0.15, distance=0.10
)
LEVELS = LevelsConfig(
    swing=SwingConfig(),
    interaction=INTERACTION,
    support_resistance=SupportResistanceConfig(weights=SR_WEIGHTS),
)


def with_wicks(closes: list[float], *, wick: float = 0.2) -> pd.DataFrame:
    """One bar per close, each opening at the previous close and ending at this one."""
    rows = []
    previous = closes[0]
    for close in closes:
        rows.append(
            (previous, max(close, previous) + wick, min(close, previous) - wick, close)
        )
        previous = close
    return build_frame(rows)


def states_of(
    closes: list[float], *, level: float, atr: float, wick: float = 0.2
) -> list[str | None]:
    """Run the interaction walk with one constant level and one constant ATR."""
    frame = with_wicks(closes, wick=wick)
    interaction = level_interaction(
        frame,
        level=pd.Series([level] * len(frame)),
        atr=pd.Series([atr] * len(frame)),
        config=INTERACTION,
    )
    return interaction["state"].tolist()


def level_walk(
    rows: list[tuple[float, float, float, float]], *, level: float, atr: float
) -> pd.DataFrame:
    """Run the interaction walk over explicit ``(open, high, low, close)`` rows."""
    frame = build_frame(rows)
    return level_interaction(
        frame,
        level=pd.Series([level] * len(frame)),
        atr=pd.Series([atr] * len(frame)),
        config=INTERACTION,
    )


def test_the_spec_example_of_a_held_breakout_reads_approach_hold_retest() -> None:
    # spec 13: 149 -> 151 -> 152 -> 151 -> 153 against a previous-day high of 150,
    # with one ATR = 10 (so a touch is within 0.5 of the level). The 150.9 bar is
    # the retest: it comes back into the band and closes above it again.
    interaction = level_walk(
        [
            (149.0, 149.2, 148.8, 149.0),
            (149.0, 151.2, 148.8, 151.0),
            (151.0, 152.2, 150.8, 152.0),
            (152.0, 152.2, 151.0, 151.0),
            (151.0, 151.2, 150.3, 150.9),
            (150.9, 153.2, 150.9, 153.0),
        ],
        level=150.0,
        atr=10.0,
    )
    assert interaction["state"].tolist() == [
        "APPROACHING",
        "CLOSED_ABOVE",
        "HELD_ABOVE",
        "HELD_ABOVE",
        "RETESTED",
        "HELD_ABOVE",
    ]
    assert interaction["touched"].tolist() == [False, True, False, False, True, False]
    assert interaction["retested"].tolist() == [False, False, False, False, True, False]


def test_the_spec_example_of_a_rejected_breakout_reads_failed_breakout() -> None:
    # spec 13: 149 -> 151 -> 148 -> 147 against a previous-day high of 150. The break
    # is not holding when price closes back below, so the third bar fails it - and it
    # fails exactly once, not on every bar that follows.
    interaction = level_walk(
        [
            (149.0, 149.2, 148.8, 149.0),
            (149.0, 151.2, 148.8, 151.0),
            (151.0, 151.2, 147.8, 148.0),
            (148.0, 148.2, 146.8, 147.0),
        ],
        level=150.0,
        atr=10.0,
    )
    assert interaction["state"].tolist()[:3] == ["APPROACHING", "CLOSED_ABOVE", "FAILED_BREAKOUT"]
    assert interaction["failed_breakout"].tolist() == [False, False, True, False]
    assert interaction["crossed_above"].tolist() == [False, True, False, False]


def test_a_wick_through_the_level_is_a_cross_not_a_close_beyond() -> None:
    frame = build_frame([(149.0, 149.4, 148.8, 149.2), (149.2, 150.4, 149.0, 149.6)])
    interaction = level_interaction(
        frame,
        level=pd.Series([150.0, 150.0]),
        atr=pd.Series([1.0, 1.0]),
        config=INTERACTION,
    )
    assert interaction["state"].tolist() == ["NOT_TESTED", "CROSSED_ABOVE"]
    assert interaction["crossed_above"].tolist() == [False, True]
    assert interaction["closed_above"].tolist() == [False, False]


def test_a_touch_that_closes_at_the_level_is_touched() -> None:
    frame = build_frame([(149.0, 149.5, 148.5, 149.3), (149.9, 150.0, 149.7, 149.98)])
    interaction = level_interaction(
        frame,
        level=pd.Series([150.0, 150.0]),
        atr=pd.Series([1.0, 1.0]),
        config=INTERACTION,
    )
    assert interaction["state"].iloc[1] == "TOUCHED"
    assert interaction["touched"].iloc[1]
    assert not interaction["rejected"].iloc[1]


def test_a_touch_that_closes_away_from_the_level_is_rejected() -> None:
    frame = build_frame([(149.0, 149.5, 148.5, 149.3), (149.9, 150.0, 149.2, 149.5)])
    interaction = level_interaction(
        frame,
        level=pd.Series([150.0, 150.0]),
        atr=pd.Series([1.0, 1.0]),
        config=INTERACTION,
    )
    assert interaction["state"].iloc[1] == "REJECTED"
    assert interaction["rejected"].iloc[1]
    assert not interaction["crossed_above"].iloc[1]



def test_losing_a_held_level_is_a_close_below_not_a_failed_breakout() -> None:
    # The distinction the whole machine turns on: the hold *was* established, so
    # closing back below is a reversal - only a break that never held has failed.
    frame = with_wicks([149.0, 151.0, 152.0, 149.0])
    interaction = level_interaction(
        frame,
        level=pd.Series([150.0] * len(frame)),
        atr=pd.Series([10.0] * len(frame)),
        config=INTERACTION,
    )
    assert interaction["state"].tolist() == [
        "APPROACHING",
        "CLOSED_ABOVE",
        "HELD_ABOVE",
        "CLOSED_BELOW",
    ]
    assert interaction["closed_below"].iloc[3]
    assert not interaction["failed_breakout"].any()


def test_the_mirror_states_exist_for_a_broken_support() -> None:
    states = states_of(
        [149.0, 151.0, 152.0, 152.0, 148.0, 147.0, 146.0, 151.0], level=150.0, atr=10.0
    )
    assert states == [
        "APPROACHING",
        "CLOSED_ABOVE",
        "HELD_ABOVE",
        "HELD_ABOVE",
        "CLOSED_BELOW",
        "HELD_BELOW",
        "HELD_BELOW",
        "CLOSED_ABOVE",
    ]


def test_a_cross_back_without_a_hold_is_a_failed_breakdown() -> None:
    states = states_of([149.0, 151.0, 152.0, 152.0, 148.0, 151.0], level=150.0, atr=10.0)
    assert states == [
        "APPROACHING",
        "CLOSED_ABOVE",
        "HELD_ABOVE",
        "HELD_ABOVE",
        "CLOSED_BELOW",
        "FAILED_BREAKDOWN",
    ]


def test_the_state_appears_on_the_first_bar_where_the_level_is_known() -> None:
    frame = with_wicks([100.0, 101.0, 102.0])
    interaction = level_interaction(
        frame,
        level=pd.Series([float("nan"), 101.0, 101.0]),
        atr=pd.Series([1.0] * 3),
        config=INTERACTION,
    )
    assert interaction["state"].iloc[0] is None or pd.isna(interaction["state"].iloc[0])
    assert pd.isna(interaction["touched"].iloc[0])
    # the bar traded through 101 and closed at it: a cross, and not yet a hold
    assert interaction["state"].iloc[1] == "CROSSED_ABOVE"


def test_the_state_is_null_where_the_atr_is_not_known() -> None:
    frame = with_wicks([100.0, 101.0])
    interaction = level_interaction(
        frame,
        level=pd.Series([100.0, 100.0]),
        atr=pd.Series([0.0, 0.0]),
        config=INTERACTION,
    )
    assert interaction["state"].isna().all()
    assert interaction["touched"].isna().all()


def test_level_interaction_rejects_a_frame_without_ohlc() -> None:
    with pytest.raises(ValueError, match="'open' column"):
        level_interaction(
            pd.DataFrame({"close": [1.0]}),
            level=pd.Series([1.0]),
            atr=pd.Series([1.0]),
            config=INTERACTION,
        )


def test_interaction_levels_use_the_running_extreme_before_the_bar() -> None:
    tracked = interaction_levels(two_session_frame())
    # session two's highs are 106, 108, 109, 104: each bar sees the extreme *before* it
    prior = tracked["today_high"].tolist()[4:]
    assert pd.isna(prior[0])  # a session's first bar has no running extreme yet
    assert prior[1:] == [106.0, 108.0, 109.0]
    assert tracked["prev_day_high"].tolist()[4:] == [105.0] * 4


def test_a_session_roll_restarts_the_level_instead_of_inheriting_the_old_state() -> None:
    frame = two_session_frame()
    events = level_events(frame, atr=pd.Series([1.0] * len(frame)), config=LEVELS)
    assert events["event_prev_day_high_state"].iloc[:4].isna().all()
    # session two opens at 100 and trades through yesterday's 105 high intraday
    assert events["event_prev_day_high_state"].iloc[4] == "CROSSED_ABOVE"


def same(left: object, right: object) -> bool:
    """Compare two possibly-missing values without tripping over ``pd.NA``."""
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def test_the_reset_mask_clears_the_tracker_at_the_bar_it_marks() -> None:
    frame = with_wicks([149.0, 151.0, 152.0, 152.0, 150.4])
    level = pd.Series([150.0] * len(frame))
    atr = pd.Series([10.0] * len(frame))
    without = level_interaction(frame, level=level, atr=atr, config=INTERACTION)
    assert without["state"].tolist() == [
        "APPROACHING",
        "CLOSED_ABOVE",
        "HELD_ABOVE",
        "HELD_ABOVE",
        "RETESTED",
    ]
    # the same bars, but the last one starts a new level life: nothing is inherited
    with_reset = level_interaction(
        frame,
        level=level,
        atr=atr,
        config=INTERACTION,
        reset=pd.Series([False, False, False, False, True]),
    )
    assert with_reset["state"].iloc[4] == "TOUCHED"  # the new life starts by touching
    assert not with_reset["retested"].iloc[4]  # and inherits nothing from the old one


def test_the_today_high_state_reacts_to_a_bar_that_makes_a_new_high() -> None:
    frame = two_session_frame()
    events = level_events(frame, atr=pd.Series([1.0] * len(frame)), config=LEVELS)
    assert events["event_today_high_state"].tolist()[4:] == [
        None,
        "CLOSED_ABOVE",
        "FAILED_BREAKOUT",
        "FAILED_BREAKOUT",
    ]


def test_level_events_report_exactly_the_four_contract_columns() -> None:
    events = level_events(two_session_frame(), atr=pd.Series([1.0] * 8), config=LEVELS)
    assert list(events.columns) == list(LEVEL_STATE_COLUMNS)
    for column in LEVEL_STATE_COLUMNS:
        assert column in FeatureRow.model_fields


def test_level_event_flags_cover_every_level_and_event_as_nullable_booleans() -> None:
    flags = level_event_flags(two_session_frame(), atr=pd.Series([1.0] * 8), config=LEVELS)
    assert list(flags.columns) == list(LEVEL_EVENT_FLAG_COLUMNS)
    assert len(LEVEL_EVENT_FLAG_COLUMNS) == 4 * 11
    assert all(str(dtype) == "boolean" for dtype in flags.dtypes)
    # an unknown level is NULL, not False: nothing was observed either way
    assert flags["event_prev_day_high_touched"].iloc[0] is pd.NA


def test_level_event_history_timestamps_every_event() -> None:
    frame = two_session_frame()
    atr = pd.Series([1.0] * len(frame))
    history = level_event_history(frame, atr=atr, config=LEVELS)
    flags = level_event_flags(frame, atr=atr, config=LEVELS)
    assert list(history.columns) == ["timestamp", "level", "state", "event"]
    fired = int(flags.fillna(False).to_numpy(dtype=bool).sum())
    assert len(history) == fired
    assert set(history["level"]) <= set(INTERACTION_LEVEL_NAMES)
    assert set(history["event"]) <= set(EVENT_SUFFIXES)
    assert history["timestamp"].isin(frame["timestamp"]).all()
    assert history["timestamp"].is_monotonic_increasing


def test_level_events_do_not_change_when_later_bars_arrive() -> None:
    frame = two_session_frame()
    atr = pd.Series([1.0] * len(frame))
    full = level_events(frame, atr=atr, config=LEVELS)
    full_flags = level_event_flags(frame, atr=atr, config=LEVELS)
    for cut in range(1, len(frame) + 1):
        partial = level_events(frame.iloc[:cut], atr=atr.iloc[:cut], config=LEVELS)
        partial_flags = level_event_flags(frame.iloc[:cut], atr=atr.iloc[:cut], config=LEVELS)
        for column in full.columns:
            assert same(full[column].iloc[cut - 1], partial[column].iloc[-1]), (
                f"{column} moved at bar {cut - 1}"
            )
        for column in full_flags.columns:
            assert same(full_flags[column].iloc[cut - 1], partial_flags[column].iloc[-1]), (
                f"{column} moved at bar {cut - 1}"
            )


def test_level_events_of_an_empty_frame_are_empty() -> None:
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    events = level_events(empty, atr=pd.Series([], dtype="float64"), config=LEVELS)
    assert events.empty
    assert list(events.columns) == list(LEVEL_STATE_COLUMNS)
    history = level_event_history(empty, atr=pd.Series([], dtype="float64"), config=LEVELS)
    assert history.empty


# ---------------------------------------------------------------------------
# Support and resistance (spec 14) - candidates, counts, strength
# ---------------------------------------------------------------------------
#: A level of 100 that is crossed without holding, rejected, then held and lost.
#: Verified by hand in the tests below, so the numbers are readable, not magic.
SR_ROWS = [
    (98.0, 98.5, 97.5, 98.0),
    (98.0, 100.2, 98.0, 99.5),
    (99.5, 100.1, 99.4, 99.6),
    (99.6, 100.0, 99.5, 99.5),
    (99.5, 101.0, 99.4, 100.5),
    (100.5, 101.0, 100.4, 100.9),
    (100.9, 101.0, 98.5, 98.9),
]


def statistics_of(*, level: float, atr: float = 1.0) -> pd.DataFrame:
    """Per-candidate statistics for :data:`SR_ROWS` against one constant level."""
    frame = build_frame(SR_ROWS)
    return candidate_statistics(
        frame,
        level=pd.Series([level] * len(frame)),
        atr=pd.Series([atr] * len(frame)),
        config=LEVELS,
    )


def test_candidate_statistics_count_every_interaction_type() -> None:
    statistics = statistics_of(level=100.0)
    last = statistics.iloc[-1]
    assert last["touch_count"] == 5  # bars 1, 2, 3, 4, 6 reached the level
    assert last["rejection_count"] == 1  # bar 3 touched and closed 0.5 away
    assert last["breakout_count"] == 3  # bars 1, 2 and 4 traded above 100
    assert last["breakdown_count"] == 1  # bar 6 fell back through the level
    assert last["failed_breakout_count"] == 1  # bar 2 crossed and closed back under
    assert last["failed_breakdown_count"] == 0
    assert last["retest_count"] == 0
    assert last["hold_count"] == 1  # bar 5 was the second consecutive close above
    assert last["distance_atr"] == pytest.approx(-1.1)  # 98.9 is 1.1 ATR under 100
    assert last["bars_since_last_touch"] == 0


def test_reaction_strength_is_the_running_excursion_since_the_first_touch() -> None:
    statistics = statistics_of(level=100.0)
    # from the first touch, the closes ran from -0.5 to +0.9 and back to -1.1 ATR,
    # so the dominant reaction so far is the move to -1.1
    assert statistics["reaction_strength"].iloc[-1] == pytest.approx(-1.1)
    assert pd.isna(statistics["reaction_strength"].iloc[0])  # never touched
    assert statistics["reaction_strength"].iloc[4] == pytest.approx(0.5)


def test_an_untested_level_scores_only_its_distance_term() -> None:
    statistics = statistics_of(level=100.0)
    # nothing has happened yet: 2 ATR away and never touched -> 0.10 * 1/(1+2)
    assert statistics["strength"].iloc[0] == pytest.approx(0.10 / 3.0)


def test_strength_matches_the_configured_weighted_formula() -> None:
    statistics = statistics_of(level=100.0)
    expected = (
        0.30 * (5 / 6)  # five touches
        + 0.25 * (1 / 2)  # one rejection
        + 0.20 * (1 / 2)  # one hold
        + 0.15 * 1.0  # touched on this very bar
        + 0.10 * (1 / 2.1)  # 1.1 ATR away
    )
    assert statistics["strength"].iloc[-1] == pytest.approx(expected)
    assert statistics["strength"].dropna().between(0.0, 1.0).all()


def test_more_evidence_scores_higher() -> None:
    statistics = statistics_of(level=100.0)
    # bars 1 and 2 both touched the level on that bar at a similar distance; bar 2
    # has one more touch and one less rejection behind it
    assert statistics["strength"].iloc[2] > statistics["strength"].iloc[1]


def test_candidate_statistics_are_null_where_the_level_is_not_known() -> None:
    frame = build_frame(SR_ROWS)
    statistics = candidate_statistics(
        frame,
        level=pd.Series([float("nan")] * 2 + [100.0] * 5),
        atr=pd.Series([1.0] * len(frame)),
        config=LEVELS,
    )
    assert statistics["touch_count"].iloc[:2].isna().all()
    assert statistics["strength"].iloc[:2].isna().all()
    assert statistics["touch_count"].iloc[-1] == 4



def test_candidate_levels_offer_previous_day_running_and_swing_levels() -> None:
    frame = two_session_frame()
    candidates = candidate_levels(frame, config=LEVELS)
    expected = {
        "prev_day_open",
        "prev_day_high",
        "prev_day_low",
        "prev_day_close",
        "today_open",
        "today_high",
        "today_low",
    }
    assert expected <= set(candidates.columns)
    swings = [name for name in candidates.columns if name.startswith("swing_")]
    assert len(swings) == 2 * LEVELS.swing.max_swings
    # the previous session's levels only exist from its first completed session on
    assert candidates["prev_day_high"].iloc[:4].isna().all()
    assert candidates["prev_day_high"].iloc[4:].tolist() == [105.0] * 4
    assert candidates["today_open"].tolist() == [100.0] * 8


def test_captured_swings_become_candidate_levels_only_once_confirmed() -> None:
    frame = swing_frame()
    candidates = candidate_levels(frame, config=LEVELS)
    assert not candidates["swing_high_1"].isna().all()
    # the same confirmation delay the swing engine applies: nothing is claimed early
    unconfirmed = candidates["swing_high_1"].iloc[: SWINGS.right_bars + 3]
    assert pd.isna(unconfirmed).all()


def test_the_nearest_support_and_resistance_are_reported_with_their_counts() -> None:
    frame = build_frame(SR_ROWS)
    candidates = pd.DataFrame(
        {"below": [95.0] * len(frame), "above": [105.0] * len(frame)}, index=frame.index
    )
    detail = support_resistance_detail(
        frame,
        atr=pd.Series([1.0] * len(frame)),
        config=LEVELS,
        candidates=candidates,
    )
    assert detail["support_candidate"].tolist() == ["below"] * len(frame)
    assert detail["resistance_candidate"].tolist() == ["above"] * len(frame)
    assert detail["support_strength"].notna().all()
    assert detail["resistance_strength"].notna().all()
    assert (detail["support_strength"] <= 1.0).all()
    assert (detail["resistance_strength"] <= 1.0).all()


def test_the_nearest_level_wins_even_when_a_farther_one_is_better_tested() -> None:
    frame = build_frame(
        [
            (95.0, 95.2, 94.5, 94.8),
            (94.8, 96.0, 94.6, 95.9),
            (95.9, 96.5, 95.8, 96.4),
            (96.4, 99.5, 96.3, 99.4),
            (99.4, 100.2, 99.3, 100.0),
        ]
    )
    candidates = pd.DataFrame(
        {"far": [95.0] * len(frame), "near": [99.0] * len(frame)}, index=frame.index
    )
    detail = support_resistance_detail(
        frame, atr=pd.Series([1.0] * len(frame)), config=LEVELS, candidates=candidates
    )
    assert detail["support_candidate"].iloc[-1] == "near"
    atr = pd.Series([1.0] * len(frame))
    far = candidate_statistics(frame, level=candidates["far"], atr=atr, config=LEVELS)
    near = candidate_statistics(frame, level=candidates["near"], atr=atr, config=LEVELS)
    # the farther level has never been tested more often - and proximity still wins,
    # because the level price would reach first is the one being tested now
    assert far["touch_count"].iloc[-1] > near["touch_count"].iloc[-1]
    assert detail["support_touch_count"].iloc[-1] == near["touch_count"].iloc[-1]
    assert detail["support_strength"].iloc[-1] == pytest.approx(near["strength"].iloc[-1])


def test_no_candidate_above_price_leaves_resistance_null() -> None:
    frame = build_frame(SR_ROWS)
    candidates = pd.DataFrame({"below": [95.0] * len(frame)}, index=frame.index)
    detail = support_resistance_detail(
        frame, atr=pd.Series([1.0] * len(frame)), config=LEVELS, candidates=candidates
    )
    assert detail["resistance_strength"].isna().all()
    assert detail["resistance_touch_count"].isna().all()
    assert detail["support_strength"].notna().all()


def test_support_and_resistance_counts_are_nullable_integers() -> None:
    frame = two_session_frame()
    features = support_resistance_features(
        frame, atr=pd.Series([1.0] * len(frame)), config=LEVELS
    )
    for column in SUPPORT_RESISTANCE_FEATURE_COLUMNS[2:]:
        assert str(features[column].dtype) == "Int64"
    for column in SUPPORT_RESISTANCE_FEATURE_COLUMNS:
        assert column in FeatureRow.model_fields


def test_support_resistance_features_report_exactly_the_contract_columns() -> None:
    frame = two_session_frame()
    features = support_resistance_features(
        frame, atr=pd.Series([1.0] * len(frame)), config=LEVELS
    )
    assert list(features.columns) == list(SUPPORT_RESISTANCE_FEATURE_COLUMNS)
    assert list(CANDIDATE_STATISTIC_COLUMNS)[-1] == "strength"


def test_support_resistance_does_not_change_when_later_bars_arrive() -> None:
    frame = two_session_frame()
    atr = pd.Series([1.0] * len(frame))
    full = support_resistance_features(frame, atr=atr, config=LEVELS)
    for cut in range(1, len(frame) + 1):
        partial = support_resistance_features(
            frame.iloc[:cut], atr=atr.iloc[:cut], config=LEVELS
        )
        for column in full.columns:
            assert same(full[column].iloc[cut - 1], partial[column].iloc[-1]), (
                f"{column} moved at bar {cut - 1}"
            )


def test_support_resistance_of_an_empty_frame_is_empty() -> None:
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    features = support_resistance_features(
        empty, atr=pd.Series([], dtype="float64"), config=LEVELS
    )
    assert features.empty
    assert list(features.columns) == list(SUPPORT_RESISTANCE_FEATURE_COLUMNS)

