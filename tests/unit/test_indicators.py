"""Tests for :mod:`core.indicators`.

Three classes of test live here, in increasing order of importance:

1. **Definition tests** - a hand-computed number (SMA of five bars, Wilder's
   recursion written out as a loop, the spec's normalised slope) so that the
   vectorised implementation can never quietly disagree with the definition.
2. **Boundary tests** - the warm-up rules: a value appears exactly when it is
   defined, ``NaN`` before that, and a zero-scale input yields ``NaN`` rather
   than an infinity or a fake ``0``.
3. **Prefix (no-look-ahead) tests** - the invariant the whole engine rests on:
   the value at bar ``i`` computed from the full history equals the value at bar
   ``i`` computed from the first ``i+1`` bars. If an indicator ever used a future
   bar, exactly this test fails.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from core.indicators import (
    as_float_series,
    atr,
    difference,
    direction,
    ema,
    first_full_window,
    ma_alignment,
    ma_source,
    momentum,
    moving_average,
    nan_before,
    normalized_slope,
    percentage_change,
    require_period,
    rolling_mean,
    rolling_std,
    running_max,
    running_min,
    slope,
    sma,
    spread,
    spread_velocity,
    trend_persistence,
    true_range,
    volatility,
    wilder_rma,
    zscore,
)

#: A short, gently trending price series, long enough for 5-bar windows.
PRICES = pd.Series(
    [
        100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 110.0, 109.0, 112.0, 115.0,
        113.0, 116.0, 118.0, 117.0, 120.0, 119.0, 121.0, 124.0, 123.0, 126.0,
    ]
)
HIGHS = PRICES + 1.0
LOWS = PRICES - 1.0


def same(left: float, right: float) -> bool:
    """Equal, treating ``NaN`` as equal to ``NaN`` (that is the point)."""
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == pytest.approx(right, rel=1e-12, abs=1e-12))


def assert_prefix_consistent(compute: Callable[[pd.Series], pd.Series], series: pd.Series) -> None:
    """The value at bar ``i`` must not move when later bars are appended."""
    full = compute(series)
    for cut in range(1, len(series) + 1):
        partial = compute(series.iloc[:cut])
        assert same(full.iloc[cut - 1], partial.iloc[-1]), (
            f"value at bar {cut - 1} changed when computed from {cut} bars"
        )


def wilder_by_hand(values: list[float], period: int) -> list[float | None]:
    """The textbook recursion, written out, to compare against :func:`wilder_rma`."""
    out: list[float | None] = [None] * (period - 1)
    out.append(sum(values[:period]) / period)
    for value in values[period:]:
        out.append((out[-1] * (period - 1) + value) / period)
    return out


# ---------------------------------------------------------------------------
# Input normalisation (core.indicators.series)
# ---------------------------------------------------------------------------
def test_as_float_series_keeps_the_index_and_casts_integers() -> None:
    series = as_float_series(pd.Series([1, 2, 3], index=["a", "b", "c"]))
    assert list(series.index) == ["a", "b", "c"]
    assert series.dtype == np.float64


def test_as_float_series_accepts_lists_and_arrays() -> None:
    assert as_float_series([1, 2]).tolist() == [1.0, 2.0]
    assert as_float_series(np.array([1.5, 2.5])).tolist() == [1.5, 2.5]


def test_as_float_series_refuses_a_string_instead_of_coercing_it() -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        as_float_series(pd.Series(["1", "2"]))


def test_as_float_series_refuses_a_non_series_value() -> None:
    with pytest.raises(TypeError, match="must be a pandas Series"):
        as_float_series("100")


@pytest.mark.parametrize("bad", [1.5, "5", True, None])
def test_require_period_refuses_a_non_integer_period(bad: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        require_period(bad)  # type: ignore[arg-type]


def test_require_period_refuses_a_window_shorter_than_the_minimum() -> None:
    with pytest.raises(ValueError, match="must be >= 2"):
        require_period(1)
    assert require_period(3, min_period=1) == 3


def test_first_full_window_finds_the_first_complete_run() -> None:
    assert first_full_window(pd.Series([1.0, 2.0, 3.0, 4.0]), 3) == 2
    assert first_full_window(pd.Series([1.0, 2.0, 3.0]), 1) == 0
    assert first_full_window(pd.Series([np.nan, 1.0, 2.0]), 2) == 2
    assert first_full_window(pd.Series([1.0, 2.0, 3.0]), 4) is None


def test_nan_before_blanks_only_the_warm_up_part() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    assert pd.isna(nan_before(series, 1).iloc[0])
    assert nan_before(series, 1).tolist()[1:] == [2.0, 3.0]
    assert nan_before(series, 0).tolist() == [1.0, 2.0, 3.0]
    assert nan_before(series, None).isna().all()


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def test_sma_is_undefined_until_the_window_is_full() -> None:
    averaged = sma(PRICES, 5)
    assert averaged.iloc[:4].isna().all()
    assert averaged.iloc[4] == pytest.approx(PRICES.iloc[:5].mean())
    assert averaged.iloc[5] == pytest.approx(PRICES.iloc[1:6].mean())


def test_sma_min_periods_allows_a_partial_window_on_purpose() -> None:
    averaged = sma(PRICES, 5, min_periods=1)
    assert averaged.iloc[0] == pytest.approx(PRICES.iloc[0])
    assert averaged.iloc[2] == pytest.approx(PRICES.iloc[:3].mean())


def test_a_naive_ema_can_be_checked_against_the_recursion() -> None:
    averaged = ema(PRICES, 5)
    alpha = 2 / 6
    manual = PRICES.iloc[0]
    for index in range(1, 5):
        manual = alpha * PRICES.iloc[index] + (1 - alpha) * manual
    assert averaged.iloc[:4].isna().all()
    assert averaged.iloc[4] == pytest.approx(manual)


def test_ema_of_a_constant_series_is_that_constant_after_its_warm_up() -> None:
    flat = ema(pd.Series([7.0] * 8), 3)
    assert flat.iloc[:2].isna().all()
    assert flat.iloc[2:].tolist() == pytest.approx([7.0] * 6)


def test_ema_never_publishes_a_long_average_from_short_history() -> None:
    assert ema(pd.Series([1.0, 2.0, 3.0]), 200).isna().all()


def test_moving_average_dispatches_on_the_configured_type() -> None:
    assert moving_average(PRICES, ma_type="sma", period=5).equals(sma(PRICES, 5))
    assert moving_average(PRICES, ma_type="ema", period=5).equals(ema(PRICES, 5))
    with pytest.raises(ValueError, match="Unknown moving-average type"):
        moving_average(PRICES, ma_type="wma", period=5)


def test_ma_source_builds_the_classic_derived_prices() -> None:
    frame = pd.DataFrame({"open": [1.0], "high": [3.0], "low": [1.0], "close": [2.0]})
    assert ma_source(frame, "close").tolist() == [2.0]
    assert ma_source(frame, "hl2").tolist() == [2.0]
    assert ma_source(frame, "hlc3").tolist() == [2.0]


def test_ma_source_rejects_an_unknown_source_and_a_missing_column() -> None:
    frame = pd.DataFrame({"high": [3.0], "low": [1.0], "close": [2.0]})
    with pytest.raises(ValueError, match="Unknown MA source"):
        ma_source(frame, "vwap")
    with pytest.raises(ValueError, match="needs column"):
        ma_source(frame, "open")


def test_ma_alignment_is_plus_one_for_a_stacked_stack_and_minus_one_inverted() -> None:
    fast = pd.Series([3.0, 1.0])
    medium = pd.Series([2.0, 2.0])
    slow = pd.Series([1.0, 3.0])
    assert ma_alignment(fast, medium, slow).tolist() == [1.0, -1.0]


def test_ma_alignment_needs_at_least_two_averages() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ma_alignment(pd.Series([1.0]))


def test_ma_alignment_is_undefined_until_the_slowest_average_exists() -> None:
    fast = pd.Series([3.0, 3.0, 3.0])
    slow = pd.Series([np.nan, np.nan, 1.0])
    aligned = ma_alignment(fast, slow)
    assert aligned.iloc[:2].isna().all()
    assert aligned.iloc[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "compute",
    [
        lambda series: sma(series, 5),
        lambda series: ema(series, 5),
        lambda series: ma_alignment(series, sma(series, 4)),
    ],
    ids=["sma", "ema", "ma_alignment"],
)
def test_moving_averages_do_not_use_a_future_bar(compute: Callable) -> None:
    assert_prefix_consistent(compute, PRICES)



# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------
def test_the_first_true_range_is_simply_the_first_bar_range() -> None:
    ranges = true_range(HIGHS, LOWS, PRICES)
    assert ranges.iloc[0] == pytest.approx(HIGHS.iloc[0] - LOWS.iloc[0])
    assert ranges.iloc[1] >= HIGHS.iloc[1] - LOWS.iloc[1]


def test_true_range_uses_the_previous_close_for_a_gap() -> None:
    # The bar gapped up from 97 to a 105-110 range: |110 - 97| = 13 beats both the
    # bar's own range (5) and the low-versus-previous-close distance (8).
    ranges = true_range(
        pd.Series([98.0, 110.0]), pd.Series([95.0, 105.0]), pd.Series([97.0, 106.0])
    )
    assert ranges.iloc[1] == pytest.approx(13.0)


def test_true_range_rejects_a_bar_whose_high_is_below_its_low() -> None:
    with pytest.raises(ValueError, match="high is below its low"):
        true_range(pd.Series([1.0]), pd.Series([2.0]), pd.Series([1.5]))


def test_true_range_rejects_misaligned_series() -> None:
    with pytest.raises(ValueError, match="same length"):
        true_range(pd.Series([1.0, 2.0]), pd.Series([0.5]), pd.Series([1.0, 1.5]))


def test_wilder_rma_matches_the_textbook_recursion() -> None:
    values = [2.0, 3.0, 4.0, 3.0, 5.0, 6.0, 4.0, 7.0]
    expected = wilder_by_hand(values, 5)
    actual = wilder_rma(pd.Series(values), period=5).tolist()
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert same(got, np.nan if want is None else want)


def test_wilder_rma_is_undefined_without_a_full_window() -> None:
    assert wilder_rma(pd.Series([1.0, 2.0, 3.0]), period=5).isna().all()


def test_atr_wilder_is_the_wilder_mean_of_the_true_ranges() -> None:
    expected = wilder_rma(true_range(HIGHS, LOWS, PRICES), period=5)
    assert atr(HIGHS, LOWS, PRICES, period=5).equals(expected)


def test_atr_simple_is_the_rolling_mean_of_the_true_ranges() -> None:
    expected = sma(true_range(HIGHS, LOWS, PRICES), 5)
    assert atr(HIGHS, LOWS, PRICES, period=5, method="simple").equals(expected)


def test_atr_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown ATR method"):
        atr(HIGHS, LOWS, PRICES, method="median")


def test_atr_does_not_use_a_future_bar() -> None:
    assert_prefix_consistent(lambda series: atr(series + 1, series - 1, series, period=5), PRICES)


# ---------------------------------------------------------------------------
# Slope and spread
# ---------------------------------------------------------------------------
def test_difference_and_slope_are_the_change_over_k_bars() -> None:
    assert difference(PRICES, periods=2).iloc[2] == pytest.approx(PRICES.iloc[2] - PRICES.iloc[0])
    assert slope(PRICES, k=2).iloc[2] == pytest.approx((PRICES.iloc[2] - PRICES.iloc[0]) / 2)


def test_normalized_slope_is_the_movement_in_units_of_atr() -> None:
    ranges = pd.Series([2.0] * len(PRICES))
    normalized = normalized_slope(PRICES, atr=ranges, k=2)
    manual = (PRICES.iloc[2] - PRICES.iloc[0]) / (2.0 * 2)
    assert normalized.iloc[2] == pytest.approx(manual)
    assert normalized.iloc[:2].isna().all()


def test_normalized_slope_is_undefined_when_atr_has_no_scale() -> None:
    assert normalized_slope(PRICES, atr=pd.Series([0.0] * len(PRICES)), k=2).isna().all()


def test_spread_is_the_point_gap_or_the_atr_normalised_gap() -> None:
    fast = pd.Series([10.0, 12.0])
    slow = pd.Series([8.0, 8.0])
    assert spread(fast, slow).tolist() == [2.0, 4.0]
    normalized = spread(fast, slow, atr=pd.Series([2.0, 0.0]))
    assert normalized.iloc[0] == pytest.approx(1.0)
    assert pd.isna(normalized.iloc[1])


def test_spread_velocity_is_the_change_of_the_spread() -> None:
    velocity = spread_velocity(pd.Series([1.0, 2.0, 3.0, 4.0]), k=2)
    assert velocity.iloc[:2].isna().all()
    assert velocity.iloc[2:].tolist() == [2.0, 2.0]


def test_slope_and_spread_do_not_use_a_future_bar() -> None:
    assert_prefix_consistent(lambda series: normalized_slope(series, atr=series - 1, k=2), PRICES)
    assert_prefix_consistent(lambda series: spread_velocity(series, k=3), PRICES)



# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_direction_marks_the_sign_and_keeps_unknowns_unknown() -> None:
    signed = direction(pd.Series([0.5, -0.5, 0.0, np.nan]))
    assert signed.tolist()[:3] == [1.0, -1.0, 0.0]
    assert pd.isna(signed.iloc[3])


def test_direction_applies_the_dead_band() -> None:
    assert direction(pd.Series([0.05, -0.05, 0.2]), deadband=0.1).tolist() == [0.0, 0.0, 1.0]


def test_direction_validates_the_dead_band() -> None:
    with pytest.raises(ValueError, match="deadband must be >= 0"):
        direction(pd.Series([1.0]), deadband=-1.0)
    with pytest.raises(TypeError, match="deadband must be a number"):
        direction(pd.Series([1.0]), deadband="big")


def test_persistence_is_one_when_every_bar_agrees() -> None:
    persistence = trend_persistence(pd.Series([0.1] * 5), window=3)
    assert persistence.iloc[:2].isna().all()
    assert persistence.iloc[2:].tolist() == [1.0, 1.0, 1.0]


def test_persistence_is_the_share_of_agreeing_bars() -> None:
    persistence = trend_persistence(pd.Series([0.5, -0.2, 0.3, 0.6, 0.4]), window=3)
    assert persistence.iloc[2] == pytest.approx(2 / 3)
    assert persistence.iloc[4] == pytest.approx(1.0)


def test_persistence_is_undefined_without_a_meaningful_direction() -> None:
    # the current bar is flat, so there is no trend to be persistent
    assert pd.isna(trend_persistence(pd.Series([0.5, 0.0, 0.4]), window=2).iloc[1])
    # a window in which nothing moved counts no votes at all
    assert pd.isna(trend_persistence(pd.Series([0.0, 0.0, 0.0]), window=2).iloc[2])
    # the window is not full yet
    assert pd.isna(trend_persistence(pd.Series([0.1, 0.2, 0.3]), window=5).iloc[2])


def test_persistence_does_not_use_a_future_bar() -> None:
    assert_prefix_consistent(lambda series: trend_persistence(series, window=4), PRICES)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_rolling_mean_and_std_require_a_full_window() -> None:
    assert rolling_mean(PRICES, period=3).iloc[:2].isna().all()
    assert rolling_mean(PRICES, period=3).iloc[2] == pytest.approx(PRICES.iloc[:3].mean())
    assert rolling_std(PRICES, period=3).iloc[2] == pytest.approx(PRICES.iloc[:3].std(ddof=1))
    assert rolling_std(PRICES, period=3, ddof=0).iloc[2] == pytest.approx(
        PRICES.iloc[:3].std(ddof=0)
    )


def test_rolling_std_rejects_a_negative_ddof() -> None:
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        rolling_std(PRICES, period=3, ddof=-1)


def test_zscore_measures_the_current_value_against_recent_bars_only() -> None:
    window = PRICES.iloc[:5]
    manual = (window.iloc[-1] - window.mean()) / window.std(ddof=1)
    assert zscore(PRICES, period=5).iloc[4] == pytest.approx(manual)
    assert zscore(PRICES, period=5).iloc[:4].isna().all()


def test_zscore_is_undefined_for_a_perfectly_flat_window() -> None:
    assert zscore(pd.Series([5.0, 5.0, 5.0]), period=3).isna().all()


def test_percentage_change_guards_against_a_non_positive_previous_price() -> None:
    change = percentage_change(pd.Series([100.0, 110.0, 0.0, 5.0]))
    assert change.iloc[1] == pytest.approx(0.10)
    assert change.iloc[2] == pytest.approx(-1.0)  # a genuine -100%, price reached zero
    assert pd.isna(change.iloc[3])  # a return measured against a zero price is undefined


def test_volatility_is_the_rolling_std_of_one_bar_returns() -> None:
    assert volatility(PRICES, period=5).equals(rolling_std(percentage_change(PRICES), period=5))
    assert volatility(PRICES, period=5).iloc[:5].isna().all()


def test_momentum_is_the_point_move_or_the_atr_normalised_move() -> None:
    assert momentum(PRICES, period=2).iloc[2] == pytest.approx(PRICES.iloc[2] - PRICES.iloc[0])
    normalized = momentum(PRICES, period=2, atr=pd.Series([2.0] * len(PRICES)))
    assert normalized.iloc[2] == pytest.approx((PRICES.iloc[2] - PRICES.iloc[0]) / 2)
    assert momentum(PRICES, period=2, atr=pd.Series([0.0] * len(PRICES))).isna().all()


def test_running_extremes_only_look_backwards() -> None:
    values = pd.Series([1.0, 5.0, 3.0, 2.0, 9.0])
    assert running_max(values).tolist() == [1.0, 5.0, 5.0, 5.0, 9.0]
    assert running_min(values).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "compute",
    [
        lambda series: rolling_mean(series, period=4),
        lambda series: zscore(series, period=4),
        lambda series: percentage_change(series),
        lambda series: volatility(series, period=4),
        lambda series: momentum(series, period=3),
        lambda series: running_max(series),
        lambda series: running_min(series),
    ],
    ids=["rolling_mean", "zscore", "percentage_change", "volatility", "momentum", "max", "min"],
)
def test_statistics_do_not_use_a_future_bar(compute: Callable) -> None:
    assert_prefix_consistent(compute, PRICES)

