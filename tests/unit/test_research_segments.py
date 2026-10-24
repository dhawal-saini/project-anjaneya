"""Unit tests for the research matrices (``core/research/segments.py``, spec 24.1-24.6).

Spec 24 asks "which bucket pays?", and the three ways that question gets answered dishonestly
are what these tests pin down:

* **a cell quoted without its sample.** Every :class:`MatrixCell` carries ``sample`` and a
  ``preliminary`` flag computed from one threshold, and the flag flips exactly at it.
* **a missing intersection filled with a zero.** A bucket x side that never traded is
  *absent* from the matrix (``cell()`` returns ``None``, ``grid()`` yields ``None``), so a
  heatmap cannot draw a profit where there was no trade.
* **an ordering that lies.** The rows and columns follow the order the caller passes (the
  calendar's buckets, the cycle's phases), and a label outside that order is appended rather
  than dropped.

:func:`ledger_trades` is tested against a Parquet-shaped frame too, because that is the real
input: a round trip turns ``None`` into ``NaN``/``NaT``, and ``nan`` is *truthy* - which is
how a missing cost quietly poisons a sum.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from apps.anjaneeya_ai.backtest_writer import ledger_frame
from core.backtest import MetricsError, Trade
from core.research import (
    MATRIX_SPECS,
    ResearchError,
    build_matrices,
    build_matrix,
    holding_time_profile,
    ledger_trades,
)
from core.research.segments import simple_segment_tables

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 9, 28)
SYMBOL = "NIFTY06OCT2625000CE"


def trade(
    *,
    index: int = 0,
    day: date = DAY,
    hour: int = 10,
    minute: int = 0,
    option_type: str = "CE",
    bucket: str = "MORNING_930_1100",
    phase: str = "MID_CYCLE",
    regime: str = "BULLISH",
    net: float = 100.0,
    holding: float = 10.0,
    days_to_expiry: float = 2.0,
    distance_from_atm: float = 0.0,
) -> Trade:
    """A closed ledger row with every field a matrix can group on."""
    entry = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST) + timedelta(
        minutes=index
    )
    return Trade(
        trade_id=f"run|{SYMBOL}|{index}",
        run_id="run",
        entry_timestamp=entry,
        tradingsymbol=SYMBOL,
        option_type=option_type,
        entry_price=100.0,
        quantity=75,
        exit_timestamp=entry + timedelta(minutes=holding),
        expiry=day + timedelta(days=int(days_to_expiry)),
        strike=25000.0,
        exit_price=101.0,
        entry_reason="SIGNAL:UP_CONTINUATION",
        exit_reason="TARGET" if net > 0 else "STOP_LOSS",
        nifty_state=regime,
        option_state="UP",
        candle_state="GREEN",
        support_resistance_state="NEAR_SUPPORT",
        ai_prediction="UP_CONTINUATION",
        ai_probability=0.5,
        gross_pnl=net + 10.0,
        costs=10.0,
        net_pnl=net,
        mfe=abs(net),
        mae=-abs(net) / 4.0,
        holding_minutes=holding,
        time_of_day_bucket=bucket,
        expiry_cycle_phase=phase,
        days_to_expiry=days_to_expiry,
        distance_from_atm=distance_from_atm,
        day_of_week=day.weekday(),
        bid_ask_spread=0.5,
    )


def ledger(count: int = 12, **overrides) -> list[Trade]:
    """``count`` trades alternating CE/PE so both columns exist."""
    return [
        trade(index=index, **{**overrides, "option_type": "CE" if index % 2 == 0 else "PE"})
        for index in range(count)
    ]



# ---------------------------------------------------------------------------
# The matrices themselves
# ---------------------------------------------------------------------------
def test_a_matrix_groups_on_both_axes_and_counts_its_sample() -> None:
    trades = [*ledger(6), *ledger(4, bucket="CLOSE_1430_1515", phase="EXPIRY_DAY")]

    matrix = build_matrix("time_of_day_x_option_type", trades, preliminary_min_samples=5)

    assert matrix.row_labels == ["CLOSE_1430_1515", "MORNING_930_1100"]
    assert matrix.column_labels == ["CE", "PE"]
    assert matrix.trades == 10
    morning_ce = matrix.cell("MORNING_930_1100", "CE")
    assert morning_ce is not None
    assert morning_ce.sample == 3
    assert morning_ce.preliminary is True
    assert morning_ce.metrics.total_trades == 3
    assert morning_ce.metrics.total_net_pnl == pytest.approx(300.0)


def test_a_missing_intersection_is_absent_not_zero() -> None:
    trades = ledger(4)  # one bucket, both sides

    matrix = build_matrix("time_of_day_x_cycle_phase", trades, preliminary_min_samples=1)

    assert matrix.row_labels == ["MORNING_930_1100"]
    assert matrix.column_labels == ["MID_CYCLE"]
    assert matrix.cell("MORNING_930_1100", "CLOSE_1430_1515") is None
    assert matrix.grid(lambda metrics: metrics.total_net_pnl) == [[400.0]]


def test_the_preliminary_flag_sits_exactly_on_the_threshold() -> None:
    trades = ledger(3)

    at_threshold = build_matrix("regime_x_time_of_day", trades, preliminary_min_samples=3)
    below = build_matrix("regime_x_time_of_day", trades, preliminary_min_samples=4)

    assert at_threshold.cell("BULLISH", "MORNING_930_1100").preliminary is False
    assert below.cell("BULLISH", "MORNING_930_1100").preliminary is True
    assert below.preliminary_cells == [("BULLISH", "MORNING_930_1100")]


def test_the_axes_follow_the_order_they_are_given() -> None:
    trades = [
        trade(index=0, bucket="CLOSE_1430_1515"),
        trade(index=1, bucket="MORNING_930_1100"),
        trade(index=2, bucket="ZZ_UNKNOWN_BUCKET"),
    ]

    matrix = build_matrix(
        "time_of_day_x_option_type",
        trades,
        preliminary_min_samples=1,
        row_order=["MORNING_930_1100", "CLOSE_1430_1515"],
    )

    # the configured order first, anything outside it appended rather than dropped
    assert matrix.row_labels == ["MORNING_930_1100", "CLOSE_1430_1515", "ZZ_UNKNOWN_BUCKET"]


def test_every_configured_matrix_can_be_built_and_describes_spec_24() -> None:
    matrices = build_matrices(ledger(8), preliminary_min_samples=2)

    assert [matrix.name for matrix in matrices] == list(MATRIX_SPECS)
    for matrix in matrices:
        assert "spec 24" in matrix.description
        payload = matrix.to_dict()
        assert payload["trades"] == 8
        assert payload["preliminary_min_samples"] == 2
        assert len(payload["cells"]) == len(matrix.cells)


def test_a_matrix_is_json_serialisable_with_its_samples() -> None:
    matrix = build_matrix("time_of_day_x_option_type", ledger(4), preliminary_min_samples=2)

    payload = json.loads(json.dumps(matrix.to_dict(), default=str))

    assert payload["cells"][0]["sample"] == 2
    assert payload["cells"][0]["preliminary"] is False
    assert payload["preliminary_cells"] == []


def test_an_unknown_matrix_or_segment_is_refused() -> None:
    with pytest.raises(ResearchError, match="unknown matrix"):
        build_matrix("not_a_matrix", ledger(2), preliminary_min_samples=1)
    with pytest.raises(ResearchError, match="unknown segment"):
        holding_time_profile(ledger(2), preliminary_min_samples=1, by="not_a_segment")


# ---------------------------------------------------------------------------
# The ledger round trip
# ---------------------------------------------------------------------------
def test_the_ledger_frame_round_trips_through_trade_objects() -> None:
    """A Parquet frame's NaNs must come back as ``None`` before any metric sees them."""
    frame = ledger_frame(ledger(2))
    frame.loc[0, "distance_from_atm"] = np.nan
    frame.loc[1, "bid_ask_spread"] = np.nan

    rebuilt = ledger_trades(frame)

    assert len(rebuilt) == 2
    assert rebuilt[0].distance_from_atm is None
    assert rebuilt[1].bid_ask_spread is None
    # a NaN that survived as a float would be truthy and poison this sum
    assert sum(trade.costs for trade in rebuilt) == pytest.approx(20.0)


def test_a_ledger_with_an_unknown_column_is_refused() -> None:
    frame = ledger_frame(ledger(1))
    frame["something_else"] = 1

    with pytest.raises(ResearchError, match="does not know"):
        ledger_trades(frame)


def test_an_open_trade_is_refused_rather_than_marked_to_market() -> None:
    rows = ledger(1)
    rows[0].exit_timestamp = None
    rows[0].net_pnl = None

    with pytest.raises(MetricsError, match="closed trades only"):
        build_matrix("time_of_day_x_option_type", rows, preliminary_min_samples=1)


def test_an_empty_ledger_produces_empty_tables() -> None:
    matrices = build_matrices([], preliminary_min_samples=3)

    assert all(not matrix.cells for matrix in matrices)
    assert all(matrix.rows() == [] for matrix in matrices)


# ---------------------------------------------------------------------------
# Holding time and the single-axis cuts
# ---------------------------------------------------------------------------
def test_holding_time_is_a_distribution_per_bucket() -> None:
    trades = [
        trade(index=0, bucket="MORNING_930_1100", holding=5.0),
        trade(index=1, bucket="MORNING_930_1100", holding=15.0),
        trade(index=2, bucket="MIDDAY_1100_1300", holding=30.0),
    ]

    profile = holding_time_profile(
        trades,
        preliminary_min_samples=2,
        order=["MORNING_930_1100", "MIDDAY_1100_1300"],
    )

    assert [row.label for row in profile.rows] == ["MORNING_930_1100", "MIDDAY_1100_1300"]
    first, second = profile.rows
    assert first.sample == 2
    assert first.distribution.mean == pytest.approx(10.0)
    assert first.distribution.maximum == pytest.approx(15.0)
    assert first.preliminary is False
    assert second.preliminary is True
    assert profile.preliminary_labels == ["MIDDAY_1100_1300"]


def test_the_single_axis_cuts_are_the_ones_spec_24_asks_for() -> None:
    trades = [trade(index=0, days_to_expiry=0.0), trade(index=1, days_to_expiry=1.0)]

    tables = simple_segment_tables(trades, preliminary_min_samples=5)

    assert [table.name for table in tables] == ["day_of_week", "days_to_expiry"]
    by_dte = tables[1]
    assert {row.label for row in by_dte.rows} == {"EXPIRY_DAY", "1_DAY"}
    assert all(row.preliminary for row in by_dte.rows)

