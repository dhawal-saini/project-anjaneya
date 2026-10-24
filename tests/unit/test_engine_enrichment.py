"""Tests for :mod:`apps.anjaneeya_engine.enrichment` (the alignment and the pass).

The model itself is tested in ``tests/unit/test_option_greeks.py``. What is tested here
is the decision the model cannot make: **which NIFTY level belongs to which bar**, and
what a bar with no answer looks like.

The alignment rule is the whole point. A bar's numbers are knowable at its *close*, so
the spot it is priced against is the index level at that close instant - and a coarse
index series must not reach a fine bar that closed before it. Getting this wrong is a
one-bar look-ahead that no test of the model would ever catch.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from apps.anjaneeya_engine.enrichment import EnrichmentResult, enrich_frame, spot_for_bars
from core.market_calendar import NseCalendar
from core.options import GreeksModel
from tests.support.engine import (
    EXPIRY,
    SESSION_START,
    priced_window,
    window_candles,
    window_index,
)

TIMEFRAME = "1m"


@pytest.fixture
def calendar(config) -> NseCalendar:
    return NseCalendar(config.calendar)


@pytest.fixture
def priced_model(config) -> GreeksModel:
    return GreeksModel.from_config(config.indicators)


def enrich(window, calendar, priced_model):
    return enrich_frame(
        window.candles,
        index_frame=window.index,
        timeframe=TIMEFRAME,
        calendar=calendar,
        model=priced_model,
    )


# ---------------------------------------------------------------------------
# Aligning the spot
# ---------------------------------------------------------------------------
def test_spot_uses_the_index_level_at_the_bars_close() -> None:
    """A 1m bar opening 09:15 closes at 09:16, so it is priced against the 09:16 close."""
    window = priced_window(minutes=(0, 1, 2))
    spot = spot_for_bars(window.candles, index_frame=window.index, timeframe=TIMEFRAME)

    assert list(spot) == pytest.approx([25000.0, 25004.0, 25008.0])
    # ... not the *open* time's level, which would shift the whole column by one bar.
    assert list(spot) != pytest.approx([25004.0, 25008.0, 25012.0])


def test_spot_is_a_backward_join_not_a_forward_one() -> None:
    """A gap in the index series leaves the bars inside it with the last level before it."""
    window = priced_window(minutes=(0, 1, 2, 3, 4, 5))
    sparse = window.index[window.index["timestamp"] != SESSION_START + timedelta(minutes=2)]

    spot = spot_for_bars(window.candles, index_frame=sparse, timeframe=TIMEFRAME)

    assert list(spot) == pytest.approx([25000.0, 25004.0, 25004.0, 25012.0, 25016.0, 25020.0])


def test_a_coarse_index_series_never_reaches_a_finer_bar() -> None:
    """The 15m index bar opening 09:15 closes at 09:30 and cannot price a 09:20 bar."""
    coarse = priced_window(minutes=(0, 5, 10), timeframe="15m")
    fine = window_candles(lookback_bars=0, session_bars=15)

    spot = spot_for_bars(fine, index_frame=coarse.index, timeframe="1m")

    priced = [position for position, value in enumerate(spot) if not np.isnan(value)]
    assert priced == [14]  # only the bar opening 09:29 closes exactly when that one does
    assert float(spot.iloc[14]) == pytest.approx(25000.0)


def test_spot_resamples_a_1m_index_series_onto_the_requested_timeframe() -> None:
    """For a 5m build the index is bucketed first, so the level is the bucket's *close*."""
    index = priced_window(minutes=range(0, 10), timeframe="1m").index  # a 1m NIFTY series
    bars = priced_window(minutes=(0, 5), timeframe="5m").candles  # 5m option bars

    spot = spot_for_bars(bars, index_frame=index, timeframe="5m")

    # The 09:15 5m bucket closes at 09:20 with the 09:19 1m level (25000 + 4*4) - not the
    # 09:15 level (25000), which is what reading the raw 1m series would have produced.
    assert list(spot) == pytest.approx([25016.0, 25036.0])


def test_spot_is_nan_without_an_index_series() -> None:
    """No index history is not an error: it is a NULL column and a reported count."""
    window = window_candles(lookback_bars=0, session_bars=3)
    empty_index = window_index(lookback_bars=1, session_bars=1).iloc[0:0]
    assert spot_for_bars(window, index_frame=None, timeframe=TIMEFRAME).isna().all()
    assert spot_for_bars(window, index_frame=empty_index, timeframe=TIMEFRAME).isna().all()


def test_spot_keeps_the_frames_order_and_index() -> None:
    """Rows arrive in whatever order the caller had; the level follows the row, not the position."""
    window = priced_window(minutes=(0, 1, 2))
    shuffled = window.candles.iloc[::-1].reset_index(drop=True)

    spot = spot_for_bars(shuffled, index_frame=window.index, timeframe=TIMEFRAME)

    assert list(shuffled["timestamp"].dt.minute) == [17, 16, 15]
    assert list(spot) == pytest.approx([25008.0, 25004.0, 25000.0])


# ---------------------------------------------------------------------------
# The frame-level pass
# ---------------------------------------------------------------------------
def test_enrich_frame_recovers_the_volatility_the_prices_were_built_from(calendar, priced_model) -> None:
    """The fixture priced its closes at 18%: the pass must find 18% and say it solved them."""
    window = priced_window(minutes=range(0, 6))
    outcome = enrich(window, calendar, priced_model)

    assert isinstance(outcome, EnrichmentResult)
    assert outcome.bars == 6
    assert outcome.solved_iv == 6
    assert outcome.unsolved_iv == 0
    assert outcome.missing_spot == 0
    assert outcome.solved_ratio == 1.0
    assert list(outcome.frame["iv"]) == pytest.approx([window.sigma] * 6, abs=1e-6)


def test_enrich_frame_reports_the_price_error_of_its_own_solution(calendar, priced_model) -> None:
    """``max |price - BS(iv)|`` is the model's self-check and must be tiny."""
    outcome = enrich(priced_window(minutes=range(0, 6)), calendar, priced_model)

    assert np.isfinite(outcome.max_price_error)
    assert outcome.max_price_error < 0.01
    assert outcome.to_dict()["max_price_error"] == pytest.approx(outcome.max_price_error)


def test_enrich_frame_fills_the_context_columns(calendar, priced_model) -> None:
    """``days_to_expiry`` comes from the calendar and ``distance_from_atm`` from the spot."""
    window = priced_window(minutes=range(0, 4))
    frame = enrich(window, calendar, priced_model).frame

    # calendar.yaml defines days_to_expiry as minutes-to-close (capped), so the value is
    # large - and identical to what the feature row itself will carry.
    expected = calendar.days_to_expiry(window.candles["timestamp"].iloc[0], EXPIRY)
    assert float(frame["days_to_expiry"].iloc[0]) == pytest.approx(expected)
    assert list(frame["distance_from_atm"]) == pytest.approx(
        [25000.0 - level for level in (25000.0, 25004.0, 25008.0, 25012.0)]
    )


def test_enrich_frame_without_a_nifty_level_leaves_every_iv_null(calendar, priced_model) -> None:
    """The honest failure: bars exist, the underlying does not, so no IV is invented."""
    window = priced_window(minutes=range(0, 4))
    outcome = enrich_frame(
        window.candles,
        index_frame=window.index.iloc[0:0],
        timeframe=TIMEFRAME,
        calendar=calendar,
        model=priced_model,
    )

    assert outcome.bars == 4
    assert outcome.solved_iv == 0
    assert outcome.missing_spot == 4
    assert outcome.frame["iv"].isna().all()
    assert not np.isfinite(outcome.max_price_error)


def test_enrich_frame_prices_a_put_as_well_as_a_call(calendar, priced_model) -> None:
    """The whole pass must work for a PE contract, not just for CE."""
    window = priced_window(minutes=range(0, 4), option_type="PE", strike=25100.0)
    outcome = enrich(window, calendar, priced_model)

    assert outcome.solved_iv == 4
    assert list(outcome.frame["iv"]) == pytest.approx([window.sigma] * 4, abs=1e-5)
    assert all(float(value) < 0.0 for value in outcome.frame["delta"])


def test_enrich_frame_an_empty_window_is_not_an_error(calendar, priced_model) -> None:
    """An empty frame short-circuits to an empty result (no model call, no crash)."""
    outcome = enrich_frame(
        pd.DataFrame(),
        index_frame=pd.DataFrame(),
        timeframe=TIMEFRAME,
        calendar=calendar,
        model=priced_model,
    )

    assert outcome.bars == 0
    assert outcome.frame.empty
    assert outcome.solved_ratio == 0.0
