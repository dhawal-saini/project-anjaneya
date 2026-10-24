"""Unit tests for the backtest metrics (``core/backtest/metrics.py``, spec 25).

Spec 25's numbers get quoted, so these tests are about the two ways a metric table lies:

* **a rate without its sample.** A segment with two trades is flagged ``PRELIMINARY``, and
  an empty segment set is an empty table rather than a row of zeros.
* **a ratio computed from nothing.** With no losing trade there is no profit factor and no
  payoff ratio: those are ``None``. With no bars there is no exposure. A number that cannot
  be computed is absent, never ``0.0``.

The arithmetic that *can* be checked by hand is checked by hand: win rate, expectancy, profit
factor, max drawdown on a known path, and the cost ratio against the gross edge.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from core.backtest import (
    DistributionSummary,
    MetricsError,
    Trade,
    TradeMetrics,
    build_segment_table,
    build_segment_tables,
    max_drawdown,
    net_pnl_values,
)

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 9, 28)
SYMBOL = "NIFTY01OCT2625000CE"


def trade(
    *,
    index: int = 0,
    net: float = 100.0,
    costs: float = 10.0,
    holding: float = 5.0,
    option_type: str = "CE",
    bucket: str = "OPEN_915_930",
    phase: str = "MID_CYCLE",
    regime: str = "BULLISH",
    reason: str = "TARGET",
    days_to_expiry: float = 2.0,
    distance_from_atm: float = -50.0,
    hour: int = 9,
    minute: int = 20,
) -> Trade:
    """A closed ledger row, with every field a segment might group on."""
    entry = datetime(DAY.year, DAY.month, DAY.day, hour, minute, tzinfo=IST) + timedelta(
        minutes=index
    )
    return Trade(
        trade_id=f"run|{SYMBOL}|{index}",
        run_id="run",
        entry_timestamp=entry,
        exit_timestamp=entry + timedelta(minutes=holding),
        tradingsymbol=SYMBOL,
        expiry=date(2026, 10, 1),
        strike=25000.0,
        option_type=option_type,
        entry_price=100.0,
        exit_price=100.0 + net / 75.0,
        quantity=75,
        entry_reason="SIGNAL:UP_CONTINUATION",
        exit_reason=reason,
        nifty_state=regime,
        candle_state="BULLISH",
        ai_prediction="UP_CONTINUATION",
        ai_probability=0.6,
        gross_pnl=net + costs,
        costs=costs,
        net_pnl=net,
        mfe=max(net / 75.0, 0.0),
        mae=min(net / 75.0, 0.0),
        holding_minutes=holding,
        time_of_day_bucket=bucket,
        expiry_cycle_phase=phase,
        days_to_expiry=days_to_expiry,
        distance_from_atm=distance_from_atm,
        day_of_week=entry.weekday(),
    )


# ---------------------------------------------------------------------------
# The overall metrics
# ---------------------------------------------------------------------------
def test_the_rates_come_from_the_net_pnl_and_can_be_checked_by_hand() -> None:
    """Five trades: two winners, two losers, one scratch - every rate follows from that."""
    trades = [
        trade(index=0, net=200.0),
        trade(index=1, net=-100.0),
        trade(index=2, net=300.0),
        trade(index=3, net=-100.0),
        trade(index=4, net=0.0),
    ]
    metrics = TradeMetrics.from_trades(trades)
    assert metrics.total_trades == 5
    assert (metrics.winners, metrics.losers, metrics.flat) == (2, 2, 1)
    assert metrics.win_rate == pytest.approx(0.4)
    assert metrics.average_win == pytest.approx(250.0)
    assert metrics.average_loss == pytest.approx(-100.0)
    assert metrics.payoff_ratio == pytest.approx(2.5)
    assert metrics.profit_factor == pytest.approx(500 / 200)
    assert metrics.expectancy == pytest.approx(60.0)
    assert metrics.total_net_pnl == pytest.approx(300.0)
    assert metrics.total_costs == pytest.approx(50.0)
    assert metrics.total_gross_pnl == pytest.approx(350.0)
    assert metrics.cost_ratio == pytest.approx(50 / 350)
    assert metrics.average_holding_minutes == pytest.approx(5.0)
    assert metrics.total_holding_minutes == pytest.approx(25.0)
    assert metrics.distribution.count == 5


def test_a_metric_that_cannot_be_computed_is_none_never_zero() -> None:
    """No loser means no profit factor; no trades means no numbers at all."""
    winners_only = TradeMetrics.from_trades([trade(index=0, net=100.0)])
    assert winners_only.profit_factor is None
    assert winners_only.payoff_ratio is None
    assert winners_only.average_loss is None
    assert winners_only.win_rate == pytest.approx(1.0)

    empty = TradeMetrics.from_trades([])
    assert empty.total_trades == 0
    assert empty.win_rate is None
    assert empty.expectancy is None
    assert empty.sharpe_like is None
    assert empty.annualised_sharpe_like is None
    assert empty.max_drawdown == 0.0
    assert empty.max_drawdown_pct is None
    assert empty.distribution.count == 0
    assert empty.to_dict()["total_net_pnl"] == 0.0


def test_max_drawdown_walks_the_cumulative_net_pnl() -> None:
    """Peak 500, trough 250: a 250 drawdown, half of the peak it fell from."""
    assert max_drawdown([500.0, -250.0, 100.0]) == pytest.approx((250.0, 0.5))
    assert max_drawdown([-250.0, 100.0]) == pytest.approx((250.0, None))
    assert max_drawdown([]) == (0.0, None)
    metrics = TradeMetrics.from_trades(
        [trade(index=0, net=500.0), trade(index=1, net=-250.0), trade(index=2, net=100.0)]
    )
    assert metrics.max_drawdown == pytest.approx(250.0)
    assert metrics.max_drawdown_pct == pytest.approx(0.5)


def test_the_distribution_summarises_the_percentiles() -> None:
    """The percentiles a report quotes, with a sample standard deviation."""
    summary = DistributionSummary.from_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary.count == 5
    assert summary.minimum == 1.0
    assert summary.maximum == 5.0
    assert summary.mean == pytest.approx(3.0)
    assert summary.stdev == pytest.approx(1.5811388, rel=1e-6)
    assert summary.percentiles["p50"] == pytest.approx(3.0)
    assert summary.percentiles["p5"] < summary.percentiles["p95"]
    assert set(summary.percentiles) == {"p5", "p25", "p50", "p75", "p95"}
    assert summary.to_dict()["count"] == 5
    assert DistributionSummary.from_values([]).count == 0


def test_exposure_needs_the_bar_counts_and_is_absent_without_them() -> None:
    """A segment does not have its own bars, so its exposure is honestly absent."""
    trades = [trade(index=0, net=100.0)]
    assert TradeMetrics.from_trades(trades).exposure is None
    with_bars = TradeMetrics.from_trades(trades, bars=400, bars_in_position=100)
    assert with_bars.exposure == pytest.approx(0.25)
    assert TradeMetrics.from_trades(trades, bars=0, bars_in_position=0).exposure is None


def test_the_sharpe_like_ratio_is_expectancy_over_volatility() -> None:
    """A per-trade Sharpe, annualised from the span the trades actually covered."""
    trades = [trade(index=0, net=100.0), trade(index=30, net=-50.0, holding=5.0)]
    metrics = TradeMetrics.from_trades(trades)
    values = net_pnl_values(trades)
    mean = sum(values) / len(values)
    assert metrics.expectancy == pytest.approx(mean)
    assert metrics.sharpe_like is not None
    assert metrics.sharpe_like > 0
    assert metrics.annualised_sharpe_like is not None
    assert metrics.annualised_sharpe_like > metrics.sharpe_like


def test_a_trade_without_an_exit_has_no_pnl_and_is_refused() -> None:
    """Marking an open position to market is a decision the caller must make explicitly."""
    open_trade = trade(index=0)
    open_trade.exit_timestamp = None
    open_trade.net_pnl = None
    with pytest.raises(MetricsError, match="no exit"):
        net_pnl_values([open_trade])
    with pytest.raises(MetricsError, match="no exit"):
        TradeMetrics.from_trades([open_trade])


# ---------------------------------------------------------------------------
# The segmented tables
# ---------------------------------------------------------------------------
SEGMENT_TRADES = [
    trade(index=0, net=300.0, option_type="CE", bucket="OPEN_915_930", regime="BULLISH"),
    trade(index=1, net=200.0, option_type="CE", bucket="MIDDAY", regime="BULLISH"),
    trade(index=2, net=-400.0, option_type="PE", bucket="MIDDAY", regime="BEARISH"),
    trade(index=3, net=-100.0, option_type="PE", bucket="OPEN_915_930", regime="BEARISH"),
]


def test_a_segment_groups_by_the_ledger_column_and_keeps_the_sample() -> None:
    """CE vs PE: two trades each, with the numbers of each group - and the sample beside it."""
    table = build_segment_table("option_type", SEGMENT_TRADES, preliminary_min_samples=3)
    assert table.name == "option_type"
    assert table.description == "CE vs PE"
    assert [row.label for row in table.rows] == ["CE", "PE"]
    assert [row.sample for row in table.rows] == [2, 2]
    assert table.rows[0].metrics.total_net_pnl == pytest.approx(500.0)
    assert table.rows[1].metrics.total_net_pnl == pytest.approx(-500.0)
    assert table.rows[0].metrics.win_rate == pytest.approx(1.0)
    assert table.rows[1].metrics.win_rate == pytest.approx(0.0)
    assert table.to_dict()["rows"][0]["label"] == "CE"


def test_a_small_segment_is_flagged_preliminary_on_both_sides_of_the_threshold() -> None:
    """The threshold is the configuration's, and it is reported with the table."""
    small = build_segment_table("option_type", SEGMENT_TRADES, preliminary_min_samples=3)
    assert all(row.preliminary for row in small.rows)
    assert small.preliminary_rows == ["CE", "PE"]
    assert small.preliminary_min_samples == 3

    large = build_segment_table("option_type", SEGMENT_TRADES, preliminary_min_samples=2)
    assert not any(row.preliminary for row in large.rows)
    assert large.preliminary_rows == []


def test_segment_rows_are_ordered_by_net_pnl() -> None:
    """Best first, which is a presentation choice - the sample sits beside every row."""
    table = build_segment_table("time_of_day_bucket", SEGMENT_TRADES, preliminary_min_samples=1)
    nets = [row.metrics.total_net_pnl for row in table.rows]
    assert nets == sorted(nets, reverse=True)
    assert [row.label for row in table.rows] == ["OPEN_915_930", "MIDDAY"]


def test_a_missing_context_is_labelled_unknown_rather_than_dropped() -> None:
    """A trade the ledger cannot place is still counted, under an honest label."""
    bare = trade(index=0, net=100.0)
    bare.time_of_day_bucket = None
    bare.expiry_cycle_phase = None
    bare.nifty_state = None
    bare.distance_from_atm = None
    bare.days_to_expiry = None
    table = build_segment_tables(
        [bare], preliminary_min_samples=1, names=["time_of_day_bucket", "nifty_state"]
    )
    assert [row.label for table_ in table for row in table_.rows] == ["UNKNOWN", "UNKNOWN"]
    buckets = build_segment_table("days_to_expiry", [bare], preliminary_min_samples=1)
    assert buckets.rows[0].label == "UNKNOWN"


def test_the_buckets_are_the_documented_ones() -> None:
    """Days to expiry and strike distance bucket at the edges the module documents."""
    expiry_day = build_segment_table(
        "days_to_expiry", [trade(index=0, days_to_expiry=0.5)], preliminary_min_samples=1
    )
    two_days = build_segment_table(
        "days_to_expiry", [trade(index=0, days_to_expiry=3.0)], preliminary_min_samples=1
    )
    far = build_segment_table(
        "days_to_expiry", [trade(index=0, days_to_expiry=9.0)], preliminary_min_samples=1
    )
    assert expiry_day.rows[0].label == "EXPIRY_DAY"
    assert two_days.rows[0].label == "2_3_DAYS"
    assert far.rows[0].label == "4_PLUS_DAYS"
    near = build_segment_table(
        "strike_distance", [trade(index=0, distance_from_atm=-80.0)], preliminary_min_samples=1
    )
    assert near.rows[0].label == "NEAR"


def test_the_weekday_segment_uses_the_entry_timestamp() -> None:
    """2026-09-28 is a Monday, so that is what the row says."""
    table = build_segment_table("day_of_week", SEGMENT_TRADES, preliminary_min_samples=1)
    assert [row.label for row in table.rows] == ["MONDAY"]


def test_every_documented_segment_can_be_built() -> None:
    """Spec 25's list, plus the exit reason: each one produces a table or an empty one."""
    tables = build_segment_tables(SEGMENT_TRADES, preliminary_min_samples=1)
    names = [table.name for table in tables]
    assert "option_type" in names
    assert "time_of_day_bucket" in names
    assert "expiry_cycle_phase" in names
    assert "day_of_week" in names
    assert "nifty_state" in names
    assert "exit_reason" in names
    assert all(table.rows for table in tables)
    assert build_segment_tables([], preliminary_min_samples=1) != []
    assert all(table.rows == [] for table in build_segment_tables([], preliminary_min_samples=1))


def test_an_unknown_segment_is_refused() -> None:
    """A typo in a segment name is a refusal, not an empty table."""
    with pytest.raises(MetricsError, match="unknown segment"):
        build_segment_table("moon_phase", SEGMENT_TRADES, preliminary_min_samples=1)


