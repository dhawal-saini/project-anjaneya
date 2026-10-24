"""Tests for :mod:`core.labels.label_builder` - spec 20's forward-looking numbers.

A label is the one kind of row in ANJANEYA that is *allowed* to look at the future,
which is exactly why every way it can look too far - or too little - is pinned here:

* the forward window is the **next** ``h`` bars (never the current one, never
  ``h+1``), and it must be whole: a bar whose horizon runs past the session close or
  across a hole in the frame has **no** label, not a shorter-window one,
* the threshold is formed from data at or before ``t`` (ATR-14 needs its warm-up, and
  an ATR that does not exist yet means *no* class - not a threshold of zero),
* an excursion is floored at zero in its own direction, so ``future_max_up_move`` is
  never a negative "favourite move",
* a class is only published when the prior move is known: continuation and reversal
  are statements *about* a prior trend, and an unknown prior is not a flat one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.constants import LABEL_COLUMNS
from core.labels import (
    REQUIRED_CANDLE_COLUMNS,
    LabelBuildError,
    LabelPolicy,
    build_label_frame,
    label_rows,
)
from core.schemas import LabelClass

IST = ZoneInfo("Asia/Kolkata")
START = datetime(2026, 9, 29, 9, 15, tzinfo=IST)
RUN_ID = "anjaneeya_engine_20260929-153000_deadbeef"


def policy(**overrides) -> LabelPolicy:
    """A small, explicit policy (percent thresholds keep the arithmetic readable)."""
    payload: dict = {
        "horizons": (5,),
        "normalization": "percent",
        "normalization_param": 0.5,
        "percent_threshold": 0.01,
        "fixed_threshold": 2.0,
        "trend_lookback": 3,
        "atr_period": 3,
        "atr_method": "wilder",
        "version": "v1",
    }
    payload.update(overrides)
    return LabelPolicy(**payload)


def candles(rows: list[dict]) -> pd.DataFrame:
    """A candle frame with the columns the builder needs (row-contract dtypes)."""
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["tradingsymbol"] = frame.get("tradingsymbol", "NIFTY01OCT2625000CE")
    return frame


def ramp(*, bars: int = 12, start: int = 1, step: float = 1.0, before: int = 0) -> pd.DataFrame:
    """``bars`` 1m bars from ``before`` bars before :data:`START`, rising by ``step``."""
    rows = []
    for index in range(bars):
        level = start + index * step
        rows.append(
            {
                "timestamp": START + timedelta(minutes=index - before),
                "open": level,
                "high": level + 0.5,
                "low": level - 0.5,
                "close": level,
            }
        )
    return candles(rows)


def session(*, bars: int = 12) -> tuple[datetime, datetime]:
    """``(session_start, session_end)`` for ``bars`` 1m bars from :data:`START`."""
    return START, START + timedelta(minutes=bars)


def build(frame: pd.DataFrame, *, bars: int = 12, **overrides) -> pd.DataFrame:
    """Build labels for :func:`ramp`-shaped data over a session of ``bars`` minutes."""
    start, end = session(bars=bars)
    fields = dict(overrides)
    chosen = fields.pop("policy", None) or policy(**fields)
    timeframe = fields.pop("timeframe", "1m")
    return build_label_frame(
        frame,
        policy=chosen,
        timeframe=timeframe,
        run_id=RUN_ID,
        session_start=start,
        session_end=end,
    )


# ---------------------------------------------------------------------------
# Shape of the output
# ---------------------------------------------------------------------------
def test_one_row_per_session_bar_and_horizon_in_column_order() -> None:
    frame = build(ramp(), bars=12, policy=policy(horizons=(5, 10)))

    assert list(frame.columns) == list(LABEL_COLUMNS)
    assert len(frame) == 12 * 2
    assert sorted(frame["horizon_minutes"].unique().tolist()) == [5, 10]
    assert frame["label_version"].unique().tolist() == ["v1"]
    assert frame["run_id"].unique().tolist() == [RUN_ID]
    assert frame["tradingsymbol"].unique().tolist() == ["NIFTY01OCT2625000CE"]


def test_rows_validate_into_the_label_contract() -> None:
    rows = label_rows(build(ramp(), bars=12, policy=policy(horizons=(5, 10))))
    assert len(rows) == 24
    assert all(row.label_version == "v1" for row in rows)
    assert all(row.horizon_minutes in {5, 10} for row in rows)


def test_the_frame_never_declares_a_column_the_contract_does_not_have() -> None:
    frame = build(ramp())
    frame["future_surprise"] = 1.0
    with pytest.raises(LabelBuildError, match="does not have"):
        label_rows(frame)


# ---------------------------------------------------------------------------
# The forward window
# ---------------------------------------------------------------------------
def test_the_forward_window_is_the_next_bars() -> None:
    """A 5m horizon at 09:15 reads 09:16..09:20 - never 09:15, never 09:21."""
    frame = build(ramp(start=100, step=1.0), bars=12)
    first = frame.iloc[0]
    # close[09:15] = 100; the window is 09:16..09:20 (closes 101..105, lows 100.5..104.5).
    assert first["future_return"] == pytest.approx(0.05)
    assert first["future_max_up_move"] == pytest.approx(5.5)  # high at 09:20: 105 + 0.5
    # A rising ramp never goes against the position, so the adverse excursion is 0 -
    # not a negative "favourite move".
    assert first["future_max_down_move"] == 0.0


def test_a_forward_window_past_the_session_has_no_label_fields() -> None:
    frame = build(ramp(), bars=12)
    tail = frame[frame["timestamp"] >= START + timedelta(minutes=8)]
    assert tail["future_return"].isna().all()
    assert tail["label"].isna().all()
    # ...and the rows still exist, so the hole is visible rather than silent.
    assert len(tail) == 4


def test_a_gap_inside_the_forward_window_removes_the_label() -> None:
    """The next *five bars* are not the next five minutes if one is missing."""
    frame = ramp()
    holed = frame.drop(index=2).reset_index(drop=True)  # drop 09:17
    built = build(holed, bars=12)
    first = built.iloc[0]
    assert pd.isna(first["future_return"])
    assert pd.isna(first["label"])
    # The bar *after* the hole is unaffected for its own window.
    assert built.iloc[4]["future_return"] is not None


def test_rows_before_the_session_are_computed_but_never_published() -> None:
    frame = ramp(bars=16, before=4)
    built = build(frame, bars=12)
    assert built["timestamp"].min() == START
    assert len(built) == 12


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def test_a_percent_threshold_scales_with_the_bar_close() -> None:
    """1% of 100 is 1.0: a 0.5 move is RANGE, a 2.5 move continues."""
    # ``before=10`` gives every session bar its prior move (the class needs one).
    flat = build(ramp(bars=30, start=100, step=0.1, before=10), bars=12, percent_threshold=0.01)
    assert flat.iloc[0]["label"] == LabelClass.RANGE.value  # 0.5 < 1% of ~100
    rising = build(ramp(bars=30, start=100, step=1.0, before=10), bars=12, percent_threshold=0.01)
    assert rising.iloc[0]["label"] == LabelClass.UP_CONTINUATION.value  # 5 > 1


def test_a_fixed_threshold_is_a_raw_price_move() -> None:
    rising = build(
        ramp(bars=30, start=100, step=0.5, before=10),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert rising.iloc[0]["label"] == LabelClass.UP_CONTINUATION.value  # 2.5 > 2.0
    flat = build(
        ramp(bars=30, start=100, step=0.1, before=10),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert flat.iloc[0]["label"] == LabelClass.RANGE.value  # 0.5 < 2.0


def test_an_atr_threshold_needs_its_warm_up_and_says_so() -> None:
    """ATR-14 does not exist after three bars, so no class may be published yet."""
    frame = build(
        ramp(bars=30, start=100, step=1.0),
        bars=24,
        normalization="atr",
        atr_period=14,
        normalization_param=0.5,
    )
    # No ATR yet -> no threshold -> no class (and no normalised trend change either).
    assert frame.iloc[0]["label"] is None
    assert pd.isna(frame.iloc[0]["future_trend_change"])
    # ATR-14 exists from the 14th bar of the frame (frame index 13).
    assert frame.iloc[13]["label"] is not None
    assert pd.notna(frame.iloc[13]["future_trend_change"])


def test_an_unknown_normalization_is_refused() -> None:
    bad = policy()
    object.__setattr__(bad, "normalization", "vibes")
    with pytest.raises(LabelBuildError, match="Unknown normalization"):
        build(ramp(), bars=12, policy=bad)


# ---------------------------------------------------------------------------
# Classification: continuation, reversal, range
# ---------------------------------------------------------------------------
def shaped(prior_step: float, forward_step: float, *, decision: int = 10) -> pd.DataFrame:
    """A frame that moves into the decision bar, then moves ``forward_step`` per bar."""
    rows = []
    for index in range(30):
        if index <= decision:
            level = 100.0 + index * prior_step
        else:
            level = 100.0 + decision * prior_step + (index - decision) * forward_step
        rows.append(
            {
                "timestamp": START + timedelta(minutes=index - decision),
                "open": level,
                "high": level + 0.2,
                "low": level - 0.2,
                "close": level,
            }
        )
    return candles(rows)


def test_a_clearing_move_with_the_prior_trend_is_a_continuation() -> None:
    frame = build(
        shaped(prior_step=1.0, forward_step=1.0),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert frame.iloc[0]["label"] == LabelClass.UP_CONTINUATION.value


def test_a_clearing_move_against_the_prior_trend_is_a_reversal() -> None:
    falling = build(
        shaped(prior_step=1.0, forward_step=-2.0),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert falling.iloc[0]["label"] == LabelClass.REVERSAL.value  # the prior leg was up
    rebounding = build(
        shaped(prior_step=-1.0, forward_step=2.0),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert rebounding.iloc[0]["label"] == LabelClass.REVERSAL.value  # the prior leg was down


def test_a_move_smaller_than_the_threshold_is_range() -> None:
    frame = build(
        shaped(prior_step=1.0, forward_step=0.1),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert frame.iloc[0]["label"] == LabelClass.RANGE.value
    assert frame.iloc[0]["future_return"] is not None  # the numbers still exist


def test_a_flat_prior_with_a_clearing_move_is_a_reversal() -> None:
    """No established trend -> the four-class vocabulary has no "initiation"."""
    frame = build(
        shaped(prior_step=0.0, forward_step=2.0),
        bars=12,
        normalization="fixed",
        fixed_threshold=2.0,
    )
    assert frame.iloc[0]["label"] == LabelClass.REVERSAL.value


def test_a_class_needs_a_prior_move_but_the_numbers_do_not() -> None:
    """Without the lookback bar there is no established trend - so no class."""
    frame = build(ramp(bars=30, start=100, step=1.0), bars=12, policy=policy(trend_lookback=3))
    assert frame.iloc[0]["label"] is None  # no bar three minutes before 09:15 in the frame
    assert pd.isna(frame.iloc[0]["future_trend_change"])
    assert frame.iloc[3]["label"] is not None
    assert pd.notna(frame.iloc[0]["future_return"])


def test_trend_change_is_the_forward_trend_minus_the_prior_trend() -> None:
    """``+1`` means the horizon added one whole threshold of trend."""
    frame = build(
        ramp(bars=30, start=100, step=0.5, before=10),
        bars=12,
        normalization="fixed",
        fixed_threshold=1.0,
        trend_lookback=2,
    )
    # forward move 2.5 (5 bars x 0.5), prior move 1.0 (2 bars) -> (2.5 - 1.0) / 1.0
    assert frame.iloc[0]["future_trend_change"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_a_horizon_that_is_not_whole_bars_is_refused() -> None:
    bad = policy(horizons=(7,))
    with pytest.raises(LabelBuildError, match="not a whole number"):
        build(ramp(bars=30), bars=12, timeframe="5m", policy=bad)


def test_missing_candle_columns_are_named() -> None:
    frame = ramp(bars=30).drop(columns=["high"])
    with pytest.raises(LabelBuildError, match="missing column"):
        build(frame, bars=12)



def test_an_empty_frame_produces_the_contract_and_nothing_else() -> None:
    """A day with no candles still returns the table's shape, not an exception."""
    frame = build_label_frame(
        pd.DataFrame(columns=list(REQUIRED_CANDLE_COLUMNS)),
        policy=policy(),
        timeframe="1m",
        run_id=RUN_ID,
    )
    assert frame.empty
    assert list(frame.columns) == list(LABEL_COLUMNS)
    assert label_rows(frame) == []


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_the_policy_reads_the_shipped_config(config) -> None:
    loaded = LabelPolicy.from_config(config.model, indicators=config.indicators)
    assert loaded.horizons == (5, 10, 15, 30)
    assert loaded.normalization == "atr"
    assert loaded.atr_period == config.indicators.atr.period
    assert loaded.trend_lookback == config.model.labels.trend_lookback
    assert loaded.max_horizon == 30
    assert loaded.bars_for(30, timeframe="1m") == 30
    assert loaded.bars_for(30, timeframe="15m") == 2
    assert loaded.describe()["version"] == config.model.labels.version


def test_the_policy_defaults_without_an_indicators_config() -> None:
    class Bare:
        class labels:
            normalization = "percent"
            normalization_param = 0.5
            percent_threshold = 0.002
            fixed_threshold = 1.0
            version = "v2"

        horizons_minutes = [5]

    loaded = LabelPolicy.from_config(Bare())
    assert loaded.atr_period == 14  # the documented default, not an exception
    assert loaded.atr_method == "wilder"
    assert loaded.trend_lookback == 5
    assert loaded.horizons == (5,)


def test_naive_timestamps_are_refused() -> None:
    frame = ramp(bars=30)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    with pytest.raises(LabelBuildError, match="naive"):
        build(frame, bars=12)


def test_duplicate_timestamps_are_refused() -> None:
    frame = pd.concat([ramp(bars=30), ramp(bars=30).iloc[[0]]], ignore_index=True)
    with pytest.raises(LabelBuildError, match="duplicate timestamps"):
        build(frame, bars=12)

