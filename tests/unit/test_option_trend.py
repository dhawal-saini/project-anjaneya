"""Tests for the option trend engine (spec 10).

Spec 10's two hard requirements are what these tests are organised around:

* **the NIFTY combination never touches an option** - the engine measures one
  contract's own short stack, and the shipped option stack is asserted against
  spec 10 so a future edit cannot quietly point it at EMA 254;
* **insufficient history is handled explicitly** - all three configured policies
  are exercised on the same frame, and the flag is asserted under each of them, so
  "publish what exists / publish nothing / substitute a shorter window" cannot
  collapse into the same behaviour by accident.

The arithmetic tests use :mod:`tests.support.trend` frames, where every average
moves by an exact amount per bar, so the expected numbers are written as formulas.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from core.config_loader import IndicatorsConfig
from core.indicators.moving_averages import ema
from core.options.option_trend import (
    OPTION_TREND_FEATURE_COLUMNS,
    average_stack,
    option_metrics,
    option_trend_features,
)
from tests.support.trend import (
    MOMENTUM_PERIOD,
    PERSISTENCE_WINDOW,
    VELOCITY_K,
    flat_atr,
    flat_frame,
    indicators_config,
    ma,
    ramp_frame,
)


def two_average_stack() -> list[dict]:
    """``sma_2_low`` and ``sma_2_high``: a stack whose votes are hand-computable."""
    return [
        ma("sma_2_low", 2, ma_type="sma", source="low"),
        ma("sma_2_high", 2, ma_type="sma", source="high"),
    ]


def short_and_long_stack() -> list[dict]:
    """One average that exists quickly and one that cannot exist yet."""
    return [
        ma("sma_2_close", 2, ma_type="sma"),
        ma("sma_10_close", 10, ma_type="sma"),
    ]


# ---------------------------------------------------------------------------
# Spec 10 conformance of the shipped configuration
# ---------------------------------------------------------------------------
def test_shipped_option_stack_matches_spec_10(config) -> None:
    specs = config.indicators.option_trend.moving_averages

    assert [(spec.name, spec.type, spec.period, spec.source) for spec in specs] == [
        ("ema_8_low", "ema", 8, "low"),
        ("ema_8_high", "ema", 8, "high"),
        ("ema_20_close", "ema", 20, "close"),
        ("ema_55_close", "ema", 55, "close"),
        ("ema_200_close", "ema", 200, "close"),
    ]
    assert config.indicators.option_trend.insufficient_history_policy == "flag"


def test_the_published_spread_pair_is_ema20_and_ema200(config) -> None:
    """The stored column is named ``option_ema20_ema200_spread``; it must mean that."""
    option_config = config.indicators.option_trend
    by_name = {spec.name: spec for spec in option_config.moving_averages}

    assert option_config.spread_pair is not None
    assert (option_config.spread_pair.fast, option_config.spread_pair.slow) == (
        "ema_20_close",
        "ema_200_close",
    )
    assert by_name[option_config.spread_pair.fast].period == 20
    assert by_name[option_config.spread_pair.slow].period == 200


def test_the_configured_spread_pair_is_used_not_the_widest_one() -> None:
    frame = ramp_frame(20, step=1.0)
    config = indicators_config(
        option_averages=[
            ma("sma_2_close", 2, ma_type="sma"),
            ma("sma_4_close", 4, ma_type="sma"),
            ma("sma_10_close", 10, ma_type="sma"),
        ],
        spread_pair={"fast": "sma_4_close", "slow": "sma_10_close"},
    )

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    # close[15] = 115: sma_4 = 113.5, sma_10 = 110.5 - a gap of 3 points, not the
    # 4.0 the fastest-versus-slowest pair would have produced.
    assert metrics["option_ema20_ema200_spread"].iloc[15] == pytest.approx(3.0)


def test_the_spread_falls_back_to_the_widest_pair_when_the_config_names_none() -> None:
    frame = ramp_frame(20, step=1.0)
    config = indicators_config(
        option_averages=[
            ma("sma_2_close", 2, ma_type="sma"),
            ma("sma_4_close", 4, ma_type="sma"),
            ma("sma_10_close", 10, ma_type="sma"),
        ]
    )

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    # close[15] = 115: sma_2 = 114.5, sma_10 = 110.5.
    assert metrics["option_ema20_ema200_spread"].iloc[15] == pytest.approx(4.0)


def test_a_spread_pair_that_names_an_unknown_average_is_rejected() -> None:
    payload = indicators_config(option_averages=short_and_long_stack()).model_dump()
    payload["option_trend"]["spread_pair"] = {"fast": "sma_99_close", "slow": "sma_10_close"}

    with pytest.raises(ValidationError, match="not one of"):
        IndicatorsConfig.model_validate(payload)


def test_a_spread_pair_in_the_wrong_order_is_rejected() -> None:
    payload = indicators_config(option_averages=short_and_long_stack()).model_dump()
    payload["option_trend"]["spread_pair"] = {"fast": "sma_10_close", "slow": "sma_2_close"}

    with pytest.raises(ValidationError, match="shorter-period average"):
        IndicatorsConfig.model_validate(payload)



# ---------------------------------------------------------------------------
# The average stack
# ---------------------------------------------------------------------------
def test_average_stack_names_columns_after_the_config() -> None:
    frame = ramp_frame(8)
    specs = indicators_config(option_averages=two_average_stack()).option_trend.moving_averages

    stack = average_stack(frame, specs=specs)

    assert list(stack.columns) == ["sma_2_low", "sma_2_high"]
    # low is half a point under the close; a two-bar average sits half a step back.
    assert stack["sma_2_low"].iloc[3] == pytest.approx(frame["close"].iloc[3] - 1.0)


def test_average_stack_waits_for_the_warm_up() -> None:
    frame = ramp_frame(12)
    specs = indicators_config(option_averages=short_and_long_stack()).option_trend.moving_averages

    stack = average_stack(frame, specs=specs)

    assert stack["sma_10_close"].iloc[:9].isna().all()
    assert stack["sma_10_close"].iloc[9] == pytest.approx(frame["close"].iloc[9] - 4.5, abs=1e-9)
    assert stack["sma_10_close"].iloc[9] == pytest.approx(sum(range(100, 110)) / 10)


def test_option_metrics_columns_are_the_stack_then_the_schema_columns() -> None:
    frame = ramp_frame(12)
    config = indicators_config(option_averages=two_average_stack())

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    assert list(metrics.columns) == [
        "sma_2_low",
        "sma_2_high",
        *OPTION_TREND_FEATURE_COLUMNS,
    ]
    assert len(metrics) == len(frame)


# ---------------------------------------------------------------------------
# The individual readings (spec 10's list)
# ---------------------------------------------------------------------------
def test_option_atr_is_the_series_that_was_passed_in() -> None:
    frame = ramp_frame(12)
    ranges = flat_atr(frame, 3.0)
    metrics = option_metrics(frame, config=indicators_config(), atr=ranges)

    pd.testing.assert_series_equal(metrics["option_atr"], ranges, check_names=False)


def test_option_ma_slope_reports_the_fastest_average() -> None:
    frame = ramp_frame(12, step=1.0)
    metrics = option_metrics(
        frame,
        config=indicators_config(option_averages=short_and_long_stack()),
        atr=flat_atr(frame, 2.0),
    )

    # The two-bar average of a +1/bar ramp rises exactly 1 per bar.
    assert metrics["option_ma_slope"].iloc[5] == pytest.approx(1.0)
    assert metrics["option_normalized_slope"].iloc[5] == pytest.approx(1.0 / 2.0)


def test_option_spread_is_the_fastest_minus_the_slowest_over_atr() -> None:
    frame = ramp_frame(12, step=1.0)
    metrics = option_metrics(
        frame,
        config=indicators_config(option_averages=short_and_long_stack()),
        atr=flat_atr(frame, 2.0),
    )

    # close[10] = 110: sma_2 = 109.5, sma_10 = 105.5, so the gap is 4 points = 2 ATRs.
    assert metrics["option_ema20_ema200_spread"].iloc[10] == pytest.approx(2.0)


def test_option_spread_velocity_is_the_k_bar_change_of_the_spread() -> None:
    closes = [100.0] * 10 + [100.0 + 2 * index for index in range(1, 21)]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
        }
    )
    metrics = option_metrics(
        frame,
        config=indicators_config(option_averages=two_average_stack()),
        atr=flat_atr(frame),
    )

    gap = metrics["option_ema20_ema200_spread"]
    expected = (gap - gap.shift(VELOCITY_K)).dropna()
    published = metrics["option_spread_velocity"].dropna()

    assert published.notna().any()
    assert published.to_numpy() == pytest.approx(expected.to_numpy())


def test_option_momentum_is_atr_normalised() -> None:
    frame = ramp_frame(12, step=1.0)
    metrics = option_metrics(frame, config=indicators_config(), atr=flat_atr(frame, 5.0))

    # Two bars back is 2 points, over an ATR of 5.
    assert metrics["option_momentum"].iloc[5] == pytest.approx(2.0 / 5.0)
    assert metrics["option_momentum"].iloc[:MOMENTUM_PERIOD].isna().all()


def test_option_alignment_is_the_stack_alignment_fastest_first() -> None:
    """On a rising ramp the low/high averages straddle the close; the stack is not in order."""
    frame = ramp_frame(12, spread=2.0)
    metrics = option_metrics(
        frame,
        config=indicators_config(option_averages=two_average_stack()),
        atr=flat_atr(frame),
    )

    # ema_2_low sits below ema_2_high on every bar: one pair, one -1 vote.
    assert metrics["option_ma_alignment"].iloc[-1] == pytest.approx(-1.0)


def test_option_score_is_the_mean_of_the_votes_that_exist() -> None:
    frame = ramp_frame(8, spread=2.0)
    metrics = option_metrics(
        frame,
        config=indicators_config(option_averages=two_average_stack()),
        atr=flat_atr(frame),
    )

    # close is 1.5 above the two-bar low average and 0.5 below the two-bar high
    # average; both rise 1 per bar (slope 1.0 over an ATR of 1); the stack vote is -1.
    position = (math.tanh(1.5) + math.tanh(-0.5)) / 2
    expected = (position + math.tanh(1.0) - 1.0) / 3
    assert metrics["option_trend_score"].iloc[4] == pytest.approx(expected)


def test_option_score_is_bounded_by_one() -> None:
    frame = ramp_frame(40, step=25.0)
    metrics = option_metrics(frame, config=indicators_config(), atr=flat_atr(frame, 0.5))

    assert metrics["option_trend_score"].dropna().abs().le(1.0).all()


def test_option_trend_persistence_counts_agreement() -> None:
    frame = ramp_frame(12)
    metrics = option_metrics(frame, config=indicators_config(), atr=flat_atr(frame))

    # A straight uptrend: the score has been saying the same thing for the whole window.
    assert metrics["option_trend_persistence"].iloc[-1] == pytest.approx(1.0)
    assert metrics["option_trend_persistence"].iloc[:PERSISTENCE_WINDOW].isna().all()


# ---------------------------------------------------------------------------
# Insufficient history: the flag and the three policies
# ---------------------------------------------------------------------------
def test_insufficient_history_is_true_until_the_full_stack_exists() -> None:
    frame = ramp_frame(12)
    config = indicators_config(option_averages=short_and_long_stack(), min_candles=10)

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    flag = metrics["option_insufficient_history"]
    assert flag.iloc[:9].all()  # sma_10 does not exist yet
    assert not flag.iloc[9:].any()


def test_insufficient_history_is_also_true_when_the_bars_are_too_few() -> None:
    """The ``min_candles_for_full_set`` term, with every average already defined."""
    frame = ramp_frame(12)
    config = indicators_config(option_averages=two_average_stack(), min_candles=20)

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    assert metrics["option_insufficient_history"].all()


def test_flag_policy_publishes_the_score_from_the_averages_that_exist() -> None:
    frame = ramp_frame(6)
    config = indicators_config(option_averages=short_and_long_stack(), policy="flag")

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    assert pd.isna(metrics["option_ma_alignment"].iloc[4])  # the stack has a hole
    assert metrics["option_insufficient_history"].iloc[4]
    # The score is the mean of the two votes that do exist: position and slope,
    # both from the two-bar average alone (close is half a point above it).
    expected = (math.tanh(0.5) + math.tanh(1.0)) / 2
    assert metrics["option_trend_score"].iloc[4] == pytest.approx(expected)


def test_skip_policy_withholds_every_average_derived_column() -> None:
    frame = ramp_frame(12)
    config = indicators_config(option_averages=short_and_long_stack(), policy="skip")

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    for column in ("sma_2_close", "sma_10_close", *OPTION_TREND_FEATURE_COLUMNS):
        if column in ("option_atr", "option_momentum", "option_insufficient_history"):
            continue
        assert metrics[column].iloc[:9].isna().all(), column
    # The ATR and momentum facts need no stack, so they are still published.
    assert metrics["option_momentum"].iloc[4] == pytest.approx(2.0)
    assert metrics["option_insufficient_history"].iloc[4]
    # And from bar 10 onwards the full stack exists and everything is published.
    assert metrics["option_trend_score"].iloc[9:].notna().any()


def test_shorter_window_policy_substitutes_the_longest_window_that_exists() -> None:
    frame = ramp_frame(6)
    config = indicators_config(option_averages=short_and_long_stack(), policy="shorter_window")

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    # At bar 3 (three bars of closes 100, 101, 102) the substitute for sma_10 is
    # the mean of everything so far: 101.0.
    assert metrics["sma_10_close"].iloc[2] == pytest.approx(101.0)
    assert pd.isna(metrics["sma_10_close"].iloc[0])  # one bar is not an average
    assert metrics["option_insufficient_history"].iloc[2]
    # position: half a point above sma_2 and one point above the substituted
    # sma_10. The slope vote is still undefined (it needs two bars of the
    # substitute), so the score is the mean of position and the +1 stack vote.
    position = (math.tanh(0.5) + math.tanh(1.0)) / 2
    assert metrics["option_trend_score"].iloc[2] == pytest.approx((position + 1.0) / 2)


def test_shorter_window_is_an_ema_of_the_bars_so_far() -> None:
    frame = ramp_frame(6)
    config = indicators_config(
        option_averages=[
            ma("ema_2_low", 2, source="low"),
            ma("ema_4_close", 4),
        ],
        policy="shorter_window",
    )

    metrics = option_metrics(frame, config=config, atr=flat_atr(frame))

    # ema(3) of closes 100, 101, 102 - alpha 0.5 seeded on the first close.
    assert metrics["ema_4_close"].iloc[2] == pytest.approx(101.25)
    assert pd.isna(metrics["ema_4_close"].iloc[0])
    # From bar 4 the window is long enough and the configured average is used:
    # exactly what the indicator library produces for the first four closes.
    assert metrics["ema_4_close"].iloc[3] == pytest.approx(
        ema(frame["close"].iloc[:4], 4).iloc[-1]
    )


def test_the_three_policies_agree_on_the_flag_and_differ_on_the_values() -> None:
    frame = ramp_frame(12)
    averages = short_and_long_stack()

    results = {
        policy: option_metrics(
            frame,
            config=indicators_config(option_averages=averages, policy=policy),
            atr=flat_atr(frame),
        )
        for policy in ("flag", "skip", "shorter_window")
    }

    flags = {policy: result["option_insufficient_history"] for policy, result in results.items()}
    pd.testing.assert_series_equal(flags["flag"], flags["skip"], check_names=False)
    pd.testing.assert_series_equal(flags["flag"], flags["shorter_window"], check_names=False)

    bar = 4
    assert not pd.isna(results["flag"]["option_trend_score"].iloc[bar])
    assert pd.isna(results["skip"]["option_trend_score"].iloc[bar])
    assert not pd.isna(results["shorter_window"]["option_trend_score"].iloc[bar])
    # The stack itself differs too: only the substituting policy publishes a
    # value for the average that does not exist yet.
    assert pd.isna(results["flag"]["sma_10_close"].iloc[bar])
    assert not pd.isna(results["shorter_window"]["sma_10_close"].iloc[bar])


# ---------------------------------------------------------------------------
# Guards and edge cases
# ---------------------------------------------------------------------------
def test_an_unknown_policy_is_rejected() -> None:
    """The config loader restricts the value; the engine refuses it too if one slips through."""
    config = indicators_config()
    broken = config.model_copy(
        update={
            "option_trend": config.option_trend.model_copy(
                update={"insufficient_history_policy": "guess"}
            )
        }
    )
    frame = ramp_frame(6)

    with pytest.raises(ValueError, match="Unknown insufficient-history policy"):
        option_metrics(frame, config=broken, atr=flat_atr(frame))


def test_a_single_average_stack_is_rejected() -> None:
    config = indicators_config(option_averages=[ma("ema_2_close", 2)])
    frame = ramp_frame(6)

    with pytest.raises(ValueError, match="at least two averages"):
        option_metrics(frame, config=config, atr=flat_atr(frame))


def test_option_trend_features_returns_exactly_the_schema_columns() -> None:
    frame = ramp_frame(12)
    features = option_trend_features(frame, config=indicators_config(), atr=flat_atr(frame))

    assert list(features.columns) == list(OPTION_TREND_FEATURE_COLUMNS)
    assert len(features) == len(frame)


def test_option_trend_features_on_an_empty_frame_keeps_the_columns() -> None:
    frame = ramp_frame(0)
    features = option_trend_features(frame, config=indicators_config(), atr=flat_atr(frame))

    assert features.empty
    assert list(features.columns) == list(OPTION_TREND_FEATURE_COLUMNS)


def test_option_trend_metrics_on_an_empty_frame_keeps_the_stack_columns() -> None:
    frame = ramp_frame(0)
    metrics = option_metrics(frame, config=indicators_config(), atr=flat_atr(frame))

    assert metrics.empty
    assert list(metrics.columns) == [
        "ema_2_low",
        "ema_2_high",
        "ema_4_close",
        *OPTION_TREND_FEATURE_COLUMNS,
    ]


def test_a_missing_close_column_is_rejected() -> None:
    frame = ramp_frame(6).drop(columns=["close"])

    with pytest.raises(ValueError, match="'close' column"):
        option_metrics(frame, config=indicators_config(), atr=flat_atr(frame))


def test_an_atr_from_another_frame_is_rejected() -> None:
    frame = ramp_frame(6)

    with pytest.raises(ValueError, match="same length"):
        option_metrics(frame, config=indicators_config(), atr=flat_atr(frame).iloc[:3])


def test_appending_bars_does_not_change_earlier_readings() -> None:
    """The no-look-ahead invariant: a contract's past is not rewritten by its future."""
    frame = ramp_frame(30)
    config = indicators_config(policy="shorter_window")
    short = option_metrics(frame.iloc[:12], config=config, atr=flat_atr(frame.iloc[:12]))
    full = option_metrics(frame, config=config, atr=flat_atr(frame))

    for column in OPTION_TREND_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(short[column], full[column].iloc[:12])


def test_a_flat_frame_produces_a_flat_score() -> None:
    """Zero range, zero slope, and a perfectly aligned stack: the reading is honest, not empty."""
    frame = flat_frame(12)
    metrics = option_metrics(frame, config=indicators_config(), atr=flat_atr(frame))

    assert metrics["option_trend_score"].iloc[-1] == pytest.approx(0.0)
    assert metrics["option_ma_alignment"].iloc[-1] == pytest.approx(0.0)
    assert metrics["option_ma_slope"].iloc[-1] == pytest.approx(0.0)
    assert metrics["option_ema20_ema200_spread"].iloc[-1] == pytest.approx(0.0)


def test_a_zero_atr_leaves_the_atr_scaled_readings_empty() -> None:
    frame = ramp_frame(12)
    metrics = option_metrics(frame, config=indicators_config(), atr=flat_atr(frame, 0.0))

    assert metrics["option_normalized_slope"].isna().all()
    assert metrics["option_ema20_ema200_spread"].isna().all()
    assert metrics["option_momentum"].isna().all()
    # The stack vote needs no ATR at all, so the score is still published - as the
    # alignment vote alone (0.0 on this ramp: the stack is not in order).
    assert metrics["option_trend_score"].iloc[-1] == pytest.approx(
        metrics["option_ma_alignment"].iloc[-1]
    )



