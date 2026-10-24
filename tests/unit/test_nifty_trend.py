"""Tests for the NIFTY trend engine and the regime classifier (specs 8-9).

The two engines are pure functions - a candle frame plus a resolved
``IndicatorsConfig`` in, columns out - so the tests are written that way: a
hand-built frame, a config with tiny periods, and assertions a reader can check
with a calculator.

Three properties get the most attention, because each of them would silently
corrupt six persisted feature columns:

* **the spec's MA combinations are what the config actually contains** - the
  shipped ``config/indicators.yaml`` is asserted against spec 8 line by line, so a
  typo in a period is a test failure rather than a research finding;
* **the aggregate is the documented one** - the role score's three families and the
  weight-renormalised composite are recomputed by hand from published numbers;
* **no look-ahead** - appending bars to a frame must not change a single earlier
  score, which is the invariant the feature builder's lookback window depends on.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from core.nifty.regime import (
    NIFTY_REGIME_FEATURE_COLUMNS,
    classify_regime,
    regime_features,
)
from core.nifty.trend_score import (
    MA_METRIC_SUFFIXES,
    NIFTY_SCORE_ROLES,
    NIFTY_TREND_FEATURE_COLUMNS,
    combination_metrics,
    combination_score,
    combination_scores,
    ma_metrics,
    ma_table,
    role_score,
    trend_features,
)
from core.schemas import RegimeState
from tests.support.trend import (
    PERSISTENCE_WINDOW,
    SLOPE_K,
    VELOCITY_K,
    WEIGHTS,
    flat_atr,
    flat_frame,
    indicators_config,
    ma,
    ramp_frame,
    two_bar_specs,
)

#: The shipped regime thresholds, spelled as config/indicators.yaml spells them.
THRESHOLDS = {
    "strong_bullish": 0.60,
    "bullish": 0.25,
    "bullish_developing": 0.08,
    "bearish_developing": -0.08,
    "bearish": -0.25,
    "strong_bearish": -0.60,
}


def two_average_role() -> list[dict[str, Any]]:
    """A fast role with two averages: ``sma_2_low`` and ``sma_2_high``.

    On :func:`ramp_frame` those two averages sit a constant distance apart
    (``low + 2 = high``), so the stack vote is a constant ``-1`` and the whole
    score can be checked with a calculator.
    """
    config = indicators_config(
        combination=[
            ma("sma_2_low", 2, ma_type="sma", source="low", role="fast"),
            ma("sma_2_high", 2, ma_type="sma", source="high", role="fast"),
        ]
    )
    return list(config.nifty_moving_averages.combination_1)


def three_role_combination() -> list[dict[str, Any]]:
    """One short average per role, so a composite is computable within a few bars."""
    config = indicators_config(
        combination=[
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
            ma("sma_4_close", 4, ma_type="sma", role="medium"),
            ma("sma_6_close", 6, ma_type="sma", role="long"),
        ]
    )
    return list(config.nifty_moving_averages.combination_1)


def thresholds():
    """The regime thresholds of the test config (= the shipped ones)."""
    return indicators_config().nifty_scores.regime_thresholds



# ---------------------------------------------------------------------------
# Spec 8 conformance of the shipped configuration
# ---------------------------------------------------------------------------
#: Spec 8's three combinations, as ``(name, type, period, source, role)``.
SPEC_8_COMBINATIONS: dict[str, list[tuple[str, str, int, str, str]]] = {
    "combination_1": [
        ("ema_8_low", "ema", 8, "low", "fast"),
        ("ema_8_high", "ema", 8, "high", "fast"),
        ("sma_108_high", "sma", 108, "high", "medium"),
        ("sma_180_high", "sma", 180, "high", "medium"),
        ("ema_254_close", "ema", 254, "close", "long"),
        ("sma_720_high", "sma", 720, "high", "long"),
        ("sma_1080_close", "sma", 1080, "close", "long"),
        ("sma_2056_high", "sma", 2056, "high", "long"),
    ],
    "combination_2": [
        ("ema_8_low", "ema", 8, "low", "fast"),
        ("ema_8_high", "ema", 8, "high", "fast"),
        ("ema_144_high", "ema", 144, "high", "medium"),
        ("sma_89_high", "sma", 89, "high", "medium"),
        ("ema_55_close", "ema", 55, "close", "medium"),
        ("ema_360_high", "ema", 360, "high", "long"),
    ],
    "combination_3": [
        ("ema_20_close", "ema", 20, "close", "fast"),
        ("ema_200_close", "ema", 200, "close", "long"),
    ],
}


@pytest.mark.parametrize("combination", sorted(SPEC_8_COMBINATIONS))
def test_shipped_config_matches_spec_8(config, combination: str) -> None:
    """The shipped combinations are spec 8's, average for average."""
    specs = getattr(config.indicators.nifty_moving_averages, combination)
    assert [
        (spec.name, spec.type, spec.period, spec.source, spec.role) for spec in specs
    ] == SPEC_8_COMBINATIONS[combination]


def test_shipped_config_weights_sum_to_one(config) -> None:
    weights = config.indicators.nifty_scores.weights
    assert weights.fast + weights.medium + weights.long == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Per-average measurements (spec 8, step 1)
# ---------------------------------------------------------------------------
def test_ma_table_names_columns_after_the_config() -> None:
    frame = ramp_frame(6)
    config = indicators_config(
        combination=[
            ma("ema_3_low", 3, source="low", role="fast"),
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
        ]
    )
    specs = config.nifty_moving_averages.combination_1

    table = ma_table(frame, specs=specs)

    assert list(table.columns) == ["ema_3_low", "sma_2_close"]
    # A two-bar SMA of a +1/bar ramp is the midpoint of two closes: half a step back.
    assert table["sma_2_close"].iloc[3] == pytest.approx(frame["close"].iloc[3] - 0.5)
    assert pd.isna(table["sma_2_close"].iloc[0])  # no window yet
    assert pd.isna(table["ema_3_low"].iloc[1])  # EMA 3 publishes from bar 3


def test_ma_metrics_columns_are_name_plus_suffix() -> None:
    frame = ramp_frame(8)
    config = indicators_config(
        combination=[
            ma("ema_3_low", 3, source="low", role="fast"),
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
        ]
    )
    specs = config.nifty_moving_averages.combination_1

    metrics = ma_metrics(frame, specs=specs, atr=flat_atr(frame), slope_k=SLOPE_K)

    assert list(metrics.columns) == [
        f"{spec.name}{suffix}" for spec in specs for suffix in MA_METRIC_SUFFIXES
    ]


def test_price_above_is_plus_one_above_the_average_and_minus_one_below() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0, 12.0, 10.0, 8.0],
            "high": [10.0, 12.0, 10.0, 8.0],
            "low": [10.0, 12.0, 10.0, 8.0],
            "close": [10.0, 12.0, 10.0, 8.0],
        }
    )
    metrics = ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame), slope_k=SLOPE_K)

    above = metrics["sma_2_close_price_above"]
    assert pd.isna(above.iloc[0])  # the average does not exist yet, so neither does the vote
    assert above.iloc[1] == 1.0  # 12 is above the 11 average
    assert above.iloc[2] == -1.0  # 10 is below the 11 average
    assert above.iloc[3] == -1.0  # 8 is below the 9 average


def test_price_above_is_zero_when_price_sits_on_the_average() -> None:
    frame = flat_frame(4)
    metrics = ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame), slope_k=SLOPE_K)

    assert metrics["sma_2_close_price_above"].iloc[3] == 0.0


def test_distance_is_atr_normalised_and_signed() -> None:
    frame = flat_frame(4, price=50.0)
    metrics = ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame, 4.0), slope_k=SLOPE_K)

    # A flat series: price *is* the average, so both readings are exactly zero.
    assert metrics["sma_2_close_distance_atr"].iloc[3] == pytest.approx(0.0)
    assert metrics["sma_2_close_normalized_slope"].iloc[3] == pytest.approx(0.0)
    assert metrics["sma_2_close_slope"].iloc[3] == pytest.approx(0.0)


def test_distance_is_measured_in_atr_units() -> None:
    frame = ramp_frame(6, start=100.0, step=1.0)
    metrics = ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame, 2.0), slope_k=SLOPE_K)

    # close[5] = 105, the two-bar average is 104.5: half a point, over an ATR of 2.
    assert metrics["sma_2_close_distance_atr"].iloc[5] == pytest.approx(0.25)


def test_normalized_slope_divides_the_move_by_atr_times_k() -> None:
    frame = ramp_frame(10)
    metrics = ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame, 2.0), slope_k=SLOPE_K)

    # The average moves 1 per bar, so over k=2 bars it moves 2: 2 / (2 * 2) = 0.5.
    assert metrics["sma_2_close_slope"].iloc[5] == pytest.approx(1.0)
    assert metrics["sma_2_close_normalized_slope"].iloc[5] == pytest.approx(0.5)


def test_ma_metrics_needs_a_close_column() -> None:
    frame = ramp_frame(4).drop(columns=["close"])

    with pytest.raises(ValueError, match="'close' column"):
        ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame), slope_k=SLOPE_K)


def test_ma_metrics_rejects_an_atr_built_from_another_frame() -> None:
    frame = ramp_frame(6)

    with pytest.raises(ValueError, match="same length"):
        ma_metrics(frame, specs=two_bar_specs(), atr=flat_atr(frame).iloc[:2], slope_k=SLOPE_K)


# ---------------------------------------------------------------------------
# Role scores (spec 9, step 2)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Role scores (spec 9, step 2)
# ---------------------------------------------------------------------------
def test_role_score_is_the_mean_of_the_families_it_has() -> None:
    frame = ramp_frame(8, spread=2.0)
    score = role_score(
        frame, specs=two_average_role(), role="fast", atr=flat_atr(frame), slope_k=SLOPE_K
    )

    # close is 1.5 above the two-bar low average and 0.5 below the two-bar high
    # average; both averages rise 1 per bar, so the k=2 slope is 1.0 / ATR(1);
    # low sits below high, so the stack vote is -1.
    position = (math.tanh(1.5) + math.tanh(-0.5)) / 2
    slope_vote = math.tanh(1.0)
    assert score.iloc[4] == pytest.approx((position + slope_vote - 1.0) / 3)


def test_role_score_with_one_average_has_no_stack_vote() -> None:
    frame = ramp_frame(8, spread=2.0)
    config = indicators_config(
        combination=[ma("sma_2_low", 2, ma_type="sma", source="low", role="fast")]
    )
    score = role_score(
        frame,
        specs=config.nifty_moving_averages.combination_1,
        role="fast",
        atr=flat_atr(frame),
        slope_k=SLOPE_K,
    )

    assert score.iloc[4] == pytest.approx((math.tanh(1.5) + math.tanh(1.0)) / 2)


def test_role_score_waits_for_every_average_of_the_role() -> None:
    frame = ramp_frame(12)
    config = indicators_config(
        combination=[
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
            ma("sma_6_close", 6, ma_type="sma", role="fast"),
        ]
    )
    score = role_score(
        frame,
        specs=config.nifty_moving_averages.combination_1,
        role="fast",
        atr=flat_atr(frame),
        slope_k=SLOPE_K,
    )

    # sma_6 exists from bar 6 (index 5); the k=2 slope then needs two more bars.
    assert score.iloc[:7].isna().all()
    assert score.iloc[7:].notna().all()


def test_role_score_is_empty_when_the_atr_cannot_scale_anything() -> None:
    frame = flat_frame(8)
    score = role_score(
        frame, specs=two_bar_specs(), role="fast", atr=flat_atr(frame, 0.0), slope_k=SLOPE_K
    )

    assert score.isna().all()  # a zero ATR is "not measurable", not "flat"


def test_a_role_the_combination_does_not_define_is_empty_not_an_error() -> None:
    frame = ramp_frame(8)
    score = role_score(
        frame, specs=two_bar_specs(), role="long", atr=flat_atr(frame), slope_k=SLOPE_K
    )

    assert len(score) == len(frame)
    assert score.isna().all()


def test_an_unknown_role_is_rejected() -> None:
    frame = ramp_frame(8)

    with pytest.raises(ValueError, match="Unknown role"):
        role_score(
            frame, specs=two_bar_specs(), role="sideways", atr=flat_atr(frame), slope_k=SLOPE_K
        )


# ---------------------------------------------------------------------------
# Composite score (spec 9, step 3)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Composite score (spec 9, step 3)
# ---------------------------------------------------------------------------
def test_composite_is_the_weighted_mean_of_the_roles() -> None:
    frame = ramp_frame(30)
    ranges = flat_atr(frame)
    config = indicators_config(combination=three_role_combination())
    specs = config.nifty_moving_averages.combination_1
    parts = {
        role: role_score(frame, specs=specs, role=role, atr=ranges, slope_k=SLOPE_K)
        for role in NIFTY_SCORE_ROLES
    }

    composite = combination_score(frame, config=config, atr=ranges)

    expected = sum(WEIGHTS[role] * parts[role].iloc[10] for role in NIFTY_SCORE_ROLES)
    assert composite.iloc[10] == pytest.approx(expected)


def test_composite_renormalises_the_weights_when_a_role_is_missing() -> None:
    """Spec 8's combination 3 has no medium average; it must still score."""
    frame = ramp_frame(30)
    ranges = flat_atr(frame)
    config = indicators_config(
        combination=[
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
            ma("sma_6_close", 6, ma_type="sma", role="long"),
        ]
    )
    specs = config.nifty_moving_averages.combination_1
    fast = role_score(frame, specs=specs, role="fast", atr=ranges, slope_k=SLOPE_K)
    long = role_score(frame, specs=specs, role="long", atr=ranges, slope_k=SLOPE_K)

    composite = combination_score(frame, config=config, atr=ranges)

    total = WEIGHTS["fast"] + WEIGHTS["long"]
    expected = (WEIGHTS["fast"] * fast.iloc[10] + WEIGHTS["long"] * long.iloc[10]) / total
    assert composite.iloc[10] == pytest.approx(expected)


def test_composite_is_empty_until_some_role_is_defined() -> None:
    frame = ramp_frame(4)
    composite = combination_score(frame, config=indicators_config(), atr=flat_atr(frame))

    assert composite.isna().all()


def test_composite_is_bounded_by_one() -> None:
    frame = ramp_frame(60)
    composite = combination_score(frame, config=indicators_config(), atr=flat_atr(frame))

    assert composite.dropna().abs().le(1.0).all()
    assert composite.dropna().any()


def test_all_three_combinations_are_scored_side_by_side() -> None:
    frame = ramp_frame(30)
    scores = combination_scores(frame, config=indicators_config(), atr=flat_atr(frame))

    assert list(scores.columns) == ["combination_1", "combination_2", "combination_3"]
    # The test config gives all three combinations the same averages, so the three
    # columns must agree bar for bar - proof that each is scored independently.
    pd.testing.assert_series_equal(
        scores["combination_1"], scores["combination_3"], check_names=False
    )


def test_an_unknown_combination_name_is_rejected() -> None:
    frame = ramp_frame(6)

    with pytest.raises(ValueError, match="Unknown MA combination"):
        combination_score(
            frame, config=indicators_config(), atr=flat_atr(frame), combination="combination_9"
        )


def test_shipped_combination_1_needs_weeks_of_bars_for_its_long_role(config) -> None:
    """SMA 2056 cannot exist inside 300 bars - and the engine says so rather than guessing."""
    frame = ramp_frame(300)
    features = trend_features(frame, config=config.indicators, atr=flat_atr(frame))
    fast = role_score(
        frame,
        specs=config.indicators.nifty_moving_averages.combination_1,
        role="fast",
        atr=flat_atr(frame),
        slope_k=config.indicators.slope.k,
    )
    medium = role_score(
        frame,
        specs=config.indicators.nifty_moving_averages.combination_1,
        role="medium",
        atr=flat_atr(frame),
        slope_k=config.indicators.slope.k,
    )

    assert features["nifty_long_score"].isna().all()
    assert features["nifty_fast_score"].notna().any()
    # The composite is therefore the renormalised fast+medium reading, not an empty column.
    total = WEIGHTS["fast"] + WEIGHTS["medium"]
    expected = (WEIGHTS["fast"] * fast.iloc[-1] + WEIGHTS["medium"] * medium.iloc[-1]) / total
    assert features["nifty_composite_trend_score"].iloc[-1] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The stored feature columns (spec 18)
# ---------------------------------------------------------------------------
def test_trend_features_columns_are_the_schema_vocabulary() -> None:
    frame = ramp_frame(30)
    features = trend_features(frame, config=indicators_config(), atr=flat_atr(frame))

    assert list(features.columns) == list(NIFTY_TREND_FEATURE_COLUMNS)
    assert len(features) == len(frame)


def test_trend_features_on_an_empty_frame_keeps_the_columns() -> None:
    frame = ramp_frame(0)
    features = trend_features(frame, config=indicators_config(), atr=flat_atr(frame))

    assert features.empty
    assert list(features.columns) == list(NIFTY_TREND_FEATURE_COLUMNS)


def test_acceleration_is_the_k_bar_change_of_the_composite() -> None:
    frame = ramp_frame(40)
    features = trend_features(frame, config=indicators_config(), atr=flat_atr(frame))
    composite = features["nifty_composite_trend_score"]

    expected = (composite - composite.shift(SLOPE_K)).dropna()
    published = features["nifty_trend_acceleration"].dropna()

    assert published.notna().any()
    assert published.to_numpy() == pytest.approx(expected.to_numpy())


def test_persistence_is_one_while_every_recent_bar_agreed() -> None:
    frame = ramp_frame(40)
    features = trend_features(frame, config=indicators_config(), atr=flat_atr(frame))

    # A straight uptrend: every bar of the window votes the same way.
    assert features["nifty_trend_persistence"].iloc[-1] == pytest.approx(1.0)


def test_persistence_waits_for_its_window() -> None:
    frame = ramp_frame(40)
    features = trend_features(frame, config=indicators_config(), atr=flat_atr(frame))
    first = features["nifty_composite_trend_score"].first_valid_index()

    assert pd.isna(features["nifty_trend_persistence"].iloc[first:first + PERSISTENCE_WINDOW - 1]).all()


def test_a_combination_without_a_role_still_publishes_that_column() -> None:
    frame = ramp_frame(30)
    config = indicators_config(
        combination=[
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
            ma("sma_6_close", 6, ma_type="sma", role="long"),
        ]
    )
    features = trend_features(frame, config=config, atr=flat_atr(frame))

    assert features["nifty_medium_score"].isna().all()
    assert features["nifty_composite_trend_score"].notna().any()


def test_appending_bars_does_not_change_earlier_scores() -> None:
    """The no-look-ahead invariant the feature builder's lookback window rests on."""
    frame = ramp_frame(60)
    config = indicators_config()
    short = trend_features(frame.iloc[:40], config=config, atr=flat_atr(frame.iloc[:40]))
    full = trend_features(frame, config=config, atr=flat_atr(frame))

    for column in NIFTY_TREND_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(short[column], full[column].iloc[:40])


# ---------------------------------------------------------------------------
# The research table (spec 8, everything measured)
# ---------------------------------------------------------------------------
def test_combination_metrics_publishes_every_measurement_and_role() -> None:
    frame = ramp_frame(30)
    metrics = combination_metrics(frame, config=indicators_config(), atr=flat_atr(frame))

    for suffix in ("alignment", "spread", "spread_velocity"):
        assert suffix in metrics.columns
    for role in NIFTY_SCORE_ROLES:
        assert f"alignment_{role}" in metrics.columns
        assert f"{role}_score" in metrics.columns
    assert "ema_3_low_average" in metrics.columns


def test_spread_is_the_fastest_slowest_gap_over_atr() -> None:
    frame = ramp_frame(30)
    config = indicators_config(
        combination=[
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
            ma("sma_6_close", 6, ma_type="sma", role="long"),
        ]
    )
    metrics = combination_metrics(frame, config=config, atr=flat_atr(frame))

    # On a +1/bar ramp the two-bar average is 0.5 back and the six-bar average is
    # 2.5 back, so the gap is 2 points; the ATR is 1.
    assert metrics["spread"].iloc[10] == pytest.approx(2.0)
    assert metrics["alignment"].iloc[10] == pytest.approx(1.0)


def test_spread_velocity_is_the_k_bar_change_of_the_spread() -> None:
    closes = [100.0] * 10 + [100.0 + 2 * index for index in range(1, 21)]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
        }
    )
    config = indicators_config(
        combination=[
            ma("sma_2_close", 2, ma_type="sma", role="fast"),
            ma("sma_6_close", 6, ma_type="sma", role="long"),
        ]
    )
    metrics = combination_metrics(frame, config=config, atr=flat_atr(frame))

    gap = metrics["spread"]
    expected = (gap - gap.shift(VELOCITY_K)).dropna()
    published = metrics["spread_velocity"].dropna()

    assert published.notna().any()
    assert published.to_numpy() == pytest.approx(expected.to_numpy())


def test_single_average_combination_has_no_stack_and_no_gap() -> None:
    frame = ramp_frame(30)
    config = indicators_config(combination=[ma("sma_2_close", 2, ma_type="sma", role="fast")])
    metrics = combination_metrics(frame, config=config, atr=flat_atr(frame))

    assert metrics["alignment"].isna().all()
    assert metrics["spread"].isna().all()
    assert metrics["spread_velocity"].isna().all()
    assert metrics["alignment_fast"].isna().all()
    assert metrics["fast_score"].notna().any()


# ---------------------------------------------------------------------------
# Regime (spec 9)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regime (spec 9)
# ---------------------------------------------------------------------------
REGIME_BOUNDARIES = [
    (0.90, RegimeState.STRONG_BULLISH),
    (0.60, RegimeState.STRONG_BULLISH),
    (0.5999, RegimeState.BULLISH),
    (0.25, RegimeState.BULLISH),
    (0.2499, RegimeState.BULLISH_DEVELOPING),
    (0.08, RegimeState.BULLISH_DEVELOPING),
    (0.0799, RegimeState.NEUTRAL),
    (0.0, RegimeState.NEUTRAL),
    (-0.0799, RegimeState.NEUTRAL),
    (-0.08, RegimeState.BEARISH_DEVELOPING),
    (-0.2499, RegimeState.BEARISH_DEVELOPING),
    (-0.25, RegimeState.BEARISH),
    (-0.5999, RegimeState.BEARISH),
    (-0.60, RegimeState.STRONG_BEARISH),
    (-0.90, RegimeState.STRONG_BEARISH),
]


@pytest.mark.parametrize(("score", "expected"), REGIME_BOUNDARIES)
def test_regime_threshold_boundaries(score: float, expected: RegimeState) -> None:
    states = classify_regime(pd.Series([score]), thresholds=thresholds())

    assert states.iloc[0] == expected.value


@pytest.mark.parametrize("magnitude", [0.0, 0.0799, 0.08, 0.25, 0.60, 0.90])
def test_regime_is_symmetric_around_zero(magnitude: float) -> None:
    settings = thresholds()
    up = classify_regime(pd.Series([magnitude]), thresholds=settings).iloc[0]
    down = classify_regime(pd.Series([-magnitude]), thresholds=settings).iloc[0]

    assert down == up.replace("BULLISH", "BEARISH")


def test_a_missing_composite_is_unknown_not_neutral() -> None:
    states = classify_regime(pd.Series([float("nan"), 0.0]), thresholds=thresholds())

    assert pd.isna(states.iloc[0])
    assert states.iloc[0] != RegimeState.NEUTRAL.value
    assert states.iloc[1] == RegimeState.NEUTRAL.value


def test_regime_classification_reads_one_bar_at_a_time() -> None:
    values = pd.Series([0.10, 0.70, -0.30, 0.02])
    settings = thresholds()

    assert (
        classify_regime(values, thresholds=settings).tolist()
        == classify_regime(values.iloc[:3], thresholds=settings).tolist() + [RegimeState.NEUTRAL.value]
    )


def test_scores_beyond_the_configured_range_still_classify() -> None:
    states = classify_regime(pd.Series([5.0, -5.0]), thresholds=thresholds())

    assert states.tolist() == [
        RegimeState.STRONG_BULLISH.value,
        RegimeState.STRONG_BEARISH.value,
    ]


def test_regime_features_names_the_schema_column() -> None:
    features = regime_features(pd.Series([0.5]), thresholds=thresholds())

    assert NIFTY_REGIME_FEATURE_COLUMNS == ("nifty_regime",)
    assert list(features.columns) == ["nifty_regime"]
    assert features["nifty_regime"].iloc[0] == RegimeState.BULLISH.value


def test_regime_features_on_an_empty_series() -> None:
    features = regime_features(pd.Series([], dtype="float64"), thresholds=thresholds())

    assert features.empty
    assert list(features.columns) == ["nifty_regime"]


def test_the_whole_chain_produces_a_known_regime_vocabulary() -> None:
    """Scores -> regime, end to end on a straight uptrend."""
    frame = ramp_frame(60)
    config = indicators_config()
    features = trend_features(frame, config=config, atr=flat_atr(frame))
    regimes = regime_features(
        features["nifty_composite_trend_score"], thresholds=config.nifty_scores.regime_thresholds
    )

    observed = set(regimes["nifty_regime"].dropna().unique())
    assert observed
    assert observed <= {state.value for state in RegimeState}
    assert regimes["nifty_regime"].iloc[-1] == RegimeState.STRONG_BULLISH.value






