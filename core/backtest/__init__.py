"""Backtesting (specs 23 and 25).

Phase 4.3 lands in three pieces, and they are deliberately separable:

* :mod:`core.backtest.costs` - the Indian F&O contract note. Brokerage, STT (sell side),
  exchange transaction charge, SEBI fee, GST on the charges, stamp duty (buy side),
  slippage, and an optional captured bid/ask spread. Every charge is named and carried in
  an :class:`~core.backtest.costs.OrderCost`, so a reader can take the friction out and see
  the frictionless number.
* :mod:`core.backtest.engine` - the event-driven replay: a signal at candle T's close, a
  fill at T+1's open (plus latency bars), stop / target / time / opposite-signal /
  session-end exits, and one ledger row per closed trade in the shape of spec 23's trade
  record. Losses are kept, and every rejection (no feature row, no prediction, below the
  threshold, no fill bar) is counted.
* :mod:`core.backtest.metrics` - spec 25's overall and segmented numbers, with the sample
  size and a PRELIMINARY flag beside every segmented row.

Nothing in this package predicts anything: the probabilities come from
:mod:`core.ml.predictor`, and the bridge from a probability to a position is
:class:`~core.backtest.engine.SignalPolicy` - a threshold rule that says so out loud, because
spec 24's decision layer is not built yet.
"""

from __future__ import annotations

from core.backtest.costs import (
    COST_MODEL_CAVEAT,
    SIDE_BUY,
    SIDE_SELL,
    SIDES,
    CostError,
    CostModel,
    CostPolicy,
    OrderCost,
    TradeCosts,
)
from core.backtest.engine import (
    ENTRY_SIGNAL,
    EXIT_DATA_END,
    EXIT_REASONS,
    EXIT_SESSION_END,
    EXIT_SIGNAL,
    EXIT_STOP_LOSS,
    EXIT_TARGET,
    EXIT_TIME,
    FILL_BID_ASK,
    FILL_MODES,
    FILL_NEXT_OPEN,
    INTRABAR_PRIORITIES,
    LEDGER_COLUMNS,
    PREDICTION_PREFIX,
    RESEARCH_COLUMNS,
    BacktestCounts,
    Backtester,
    BacktestError,
    BacktestResult,
    ExecutionPolicy,
    SignalDecision,
    SignalPolicy,
    Trade,
    trading_days,
)
from core.backtest.metrics import (
    DEFAULT_ANNUALISATION_DAYS,
    DTE_BUCKETS,
    SEGMENT_KEYS,
    STRIKE_DISTANCE_BUCKETS,
    WEEKDAY_NAMES,
    DistributionSummary,
    MetricsError,
    SegmentRow,
    SegmentTable,
    TradeMetrics,
    build_segment_table,
    build_segment_tables,
    max_drawdown,
    net_pnl_values,
)

__all__ = [
    "COST_MODEL_CAVEAT",
    "DEFAULT_ANNUALISATION_DAYS",
    "DTE_BUCKETS",
    "ENTRY_SIGNAL",
    "EXIT_DATA_END",
    "EXIT_REASONS",
    "EXIT_SESSION_END",
    "EXIT_SIGNAL",
    "EXIT_STOP_LOSS",
    "EXIT_TARGET",
    "EXIT_TIME",
    "FILL_BID_ASK",
    "FILL_MODES",
    "FILL_NEXT_OPEN",
    "INTRABAR_PRIORITIES",
    "LEDGER_COLUMNS",
    "PREDICTION_PREFIX",
    "RESEARCH_COLUMNS",
    "SEGMENT_KEYS",
    "SIDES",
    "SIDE_BUY",
    "SIDE_SELL",
    "STRIKE_DISTANCE_BUCKETS",
    "WEEKDAY_NAMES",
    "BacktestCounts",
    "BacktestError",
    "BacktestResult",
    "Backtester",
    "CostError",
    "CostModel",
    "CostPolicy",
    "DistributionSummary",
    "ExecutionPolicy",
    "MetricsError",
    "OrderCost",
    "SegmentRow",
    "SegmentTable",
    "SignalDecision",
    "SignalPolicy",
    "Trade",
    "TradeCosts",
    "TradeMetrics",
    "build_segment_table",
    "build_segment_tables",
    "max_drawdown",
    "net_pnl_values",
    "trading_days",
]

