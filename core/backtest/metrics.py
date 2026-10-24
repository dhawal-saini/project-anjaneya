"""Backtest metrics and the segmented tables (spec 25).

Spec 25 asks for the obvious summary numbers - trades, winners, losers, win rate, average
win, average loss, profit factor, max drawdown, holding time, total return, a return
distribution, a Sharpe-like ratio and exposure - and then for the same numbers *segmented*
by expiry, strike distance, option side, time-of-day, regime, cycle phase and day of week.

Three things this module refuses to do, because each of them is how a backtest lies:

* **It never divides by zero into a plausible number.** No losing trade means there is no
  profit factor and no payoff ratio: those are ``None``, not ``inf`` and not ``1.0``.
* **It never reports a rate without its sample.** Every segmented row carries its trade
  count, and a row below ``evaluation.preliminary_min_samples`` is flagged PRELIMINARY -
  the same honesty rule the training stage applies to a small fold.
* **It never hides the costs.** ``total_costs``, ``cost_ratio`` (friction as a share of the
  gross edge) and the gross/net pair sit in the same table as the win rate, so a strategy
  that wins often and pays it all away is visible at a glance.

Max drawdown is measured on the trade-ordered cumulative net P&L (equity starts at zero,
because a premium strategy's "capital" is a sizing decision, not a property of the edge).
``max_drawdown_pct`` normalises that rupee figure against the highest equity reached before
the trough, and is ``None`` when the strategy never rose above zero - a percentage of a
non-positive peak is not a number worth printing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DTE_BUCKETS",
    "DISTRIBUTION_PERCENTILES",
    "SEGMENT_KEYS",
    "STRIKE_DISTANCE_BUCKETS",
    "WEEKDAY_NAMES",
    "DistributionSummary",
    "MetricsError",
    "SegmentRow",
    "SegmentTable",
    "TradeMetrics",
    "build_segment_table",
    "build_segment_tables",
    "max_drawdown",
    "net_pnl_values",
]

#: Trading days in a year, used only to annualise the per-trade Sharpe-like ratio.
DEFAULT_ANNUALISATION_DAYS = 252.0

#: The percentiles the return distribution reports.
DISTRIBUTION_PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)

#: Which buckets a days-to-expiry figure falls into (0 = expiry day itself).
DTE_BUCKETS: tuple[float, ...] = (1.0, 2.0, 4.0)
DTE_LABELS: tuple[str, ...] = ("EXPIRY_DAY", "1_DAY", "2_3_DAYS", "4_PLUS_DAYS")

#: How a strike distance is bucketed. ATM is "within half a strike interval".
STRIKE_DISTANCE_BUCKETS: tuple[float, ...] = (50.0, 150.0, 350.0)
STRIKE_DISTANCE_LABELS: tuple[str, ...] = ("ATM", "NEAR", "MID", "FAR")

#: Weekday names, indexed by ``datetime.weekday()``.
WEEKDAY_NAMES: tuple[str, ...] = (
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
)

UNKNOWN_LABEL = "UNKNOWN"


class MetricsError(RuntimeError):
    """Raised when a metric cannot be computed from the trades it was handed."""


# ---------------------------------------------------------------------------
# Distributions and drawdown
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """A return distribution in numbers a reader can compare across segments."""

    count: int
    minimum: float
    maximum: float
    mean: float
    stdev: float
    percentiles: dict[str, float]

    @classmethod
    def from_values(cls, values: Sequence[float]) -> DistributionSummary:
        """Summarise ``values`` (a sample ``stdev``, i.e. ``ddof=1``)."""
        array = np.asarray(values, dtype="float64")
        if array.size == 0:
            return cls(
                count=0,
                minimum=0.0,
                maximum=0.0,
                mean=0.0,
                stdev=0.0,
                percentiles={f"p{int(p)}": 0.0 for p in DISTRIBUTION_PERCENTILES},
            )
        return cls(
            count=int(array.size),
            minimum=float(array.min()),
            maximum=float(array.max()),
            mean=float(array.mean()),
            stdev=float(array.std(ddof=1)) if array.size > 1 else 0.0,
            percentiles={
                f"p{int(p)}": float(np.percentile(array, p)) for p in DISTRIBUTION_PERCENTILES
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "stdev": self.stdev,
            "percentiles": dict(self.percentiles),
        }


def net_pnl_values(trades: Iterable[Any]) -> list[float]:
    """The net P&L of every *closed* trade, in ledger order.

    A trade without an exit has no P&L: rather than inventing a mark-to-market the caller
    never asked for, it is refused, because silently treating an open position as flat is
    the single easiest way to flatter a backtest.
    """
    values: list[float] = []
    for trade in trades:
        if getattr(trade, "exit_timestamp", None) is None or getattr(trade, "net_pnl", None) is None:
            raise MetricsError(
                f"trade {getattr(trade, 'trade_id', '?')!r} has no exit: metrics are computed "
                "on closed trades only"
            )
        values.append(float(trade.net_pnl))
    return values


def max_drawdown(values: Sequence[float]) -> tuple[float, float | None]:
    """``(rupees, fraction)`` maximum peak-to-trough drawdown of a cumulative P&L path.

    The fraction is relative to the highest equity reached *before* the trough; it is
    ``None`` when that peak is not positive, because "a 40% drawdown of a negative number"
    is not a quantity anything should be read from.
    """
    equity = 0.0
    peak = 0.0
    worst = 0.0
    worst_fraction: float | None = None
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        drop = peak - equity
        if drop > worst:
            worst = drop
            worst_fraction = drop / peak if peak > 0 else None
    return worst, worst_fraction




# ---------------------------------------------------------------------------
# The overall metrics (spec 25)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TradeMetrics:
    """Spec 25's overall numbers for a set of closed trades.

    ``exposure`` needs the bar counts of the run that produced the trades (a segment of a
    run does not have its own), so it is ``None`` for segment rows - honestly absent rather
    than copied from the parent.
    """

    total_trades: int
    winners: int
    losers: int
    flat: int
    win_rate: float | None
    average_win: float | None
    average_loss: float | None
    payoff_ratio: float | None
    profit_factor: float | None
    expectancy: float | None
    total_gross_pnl: float
    total_costs: float
    total_net_pnl: float
    cost_ratio: float | None
    max_drawdown: float
    max_drawdown_pct: float | None
    average_holding_minutes: float | None
    total_holding_minutes: float
    exposure: float | None
    sharpe_like: float | None
    annualised_sharpe_like: float | None
    distribution: DistributionSummary

    @classmethod
    def from_trades(
        cls,
        trades: Sequence[Any],
        *,
        bars: int | None = None,
        bars_in_position: int | None = None,
    ) -> TradeMetrics:
        """Every metric, computed from the ledger rows alone (plus the run's bar counts)."""
        net = net_pnl_values(trades)
        count = len(net)
        wins = [value for value in net if value > 0]
        losses = [value for value in net if value < 0]
        flats = [value for value in net if value == 0]
        gross_profit = float(sum(wins))
        gross_loss = float(-sum(losses))
        average_win = float(np.mean(wins)) if wins else None
        average_loss = float(np.mean(losses)) if losses else None
        total_gross = float(sum(float(trade.gross_pnl or 0.0) for trade in trades))
        total_costs = float(sum(float(trade.costs or 0.0) for trade in trades))
        total_net = float(sum(net))
        drawdown, drawdown_pct = max_drawdown(net)
        holding = [
            float(trade.holding_minutes or 0.0)
            for trade in trades
            if trade.holding_minutes is not None
        ]
        exposure = (
            None
            if bars is None or bars_in_position is None or bars <= 0
            else bars_in_position / bars
        )
        array = np.asarray(net, dtype="float64")
        stdev = float(array.std(ddof=1)) if array.size > 1 else 0.0
        sharpe = float(array.mean() / stdev) if stdev > 0 else None
        annualised = None
        if sharpe is not None:
            span = _span_days(trades)
            if span is not None and span > 0 and count:
                trades_per_year = count * DEFAULT_ANNUALISATION_DAYS / span
                annualised = float(sharpe * np.sqrt(trades_per_year))
        return cls(
            total_trades=count,
            winners=len(wins),
            losers=len(losses),
            flat=len(flats),
            win_rate=(len(wins) / count) if count else None,
            average_win=average_win,
            average_loss=average_loss,
            payoff_ratio=(
                abs(average_win / average_loss)
                if average_win is not None and average_loss not in (None, 0.0)
                else None
            ),
            profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else None,
            expectancy=(total_net / count) if count else None,
            total_gross_pnl=total_gross,
            total_costs=total_costs,
            total_net_pnl=total_net,
            cost_ratio=(total_costs / abs(total_gross)) if total_gross != 0 else None,
            max_drawdown=drawdown,
            max_drawdown_pct=drawdown_pct,
            average_holding_minutes=(float(np.mean(holding)) if holding else None),
            total_holding_minutes=float(sum(holding)),
            exposure=exposure,
            sharpe_like=sharpe,
            annualised_sharpe_like=annualised,
            distribution=DistributionSummary.from_values(net),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - the same numbers the report prints."""
        return {
            "total_trades": self.total_trades,
            "winners": self.winners,
            "losers": self.losers,
            "flat": self.flat,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "payoff_ratio": self.payoff_ratio,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "total_gross_pnl": self.total_gross_pnl,
            "total_costs": self.total_costs,
            "total_net_pnl": self.total_net_pnl,
            "cost_ratio": self.cost_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "average_holding_minutes": self.average_holding_minutes,
            "total_holding_minutes": self.total_holding_minutes,
            "exposure": self.exposure,
            "sharpe_like": self.sharpe_like,
            "annualised_sharpe_like": self.annualised_sharpe_like,
            "distribution": self.distribution.to_dict(),
        }


def _span_days(trades: Sequence[Any]) -> float | None:
    """Calendar days from the first entry to the last exit, or ``None`` when unknowable."""
    entries = [t for t in trades if isinstance(getattr(t, "entry_timestamp", None), datetime)]
    exits = [t for t in trades if isinstance(getattr(t, "exit_timestamp", None), datetime)]
    if not entries or not exits:
        return None
    first = min(trade.entry_timestamp for trade in entries)
    last = max(trade.exit_timestamp for trade in exits)
    return (last - first).total_seconds() / 86_400.0


# ---------------------------------------------------------------------------
# Segmenting (spec 25's "segmented" list)
# ---------------------------------------------------------------------------
def _label(value: Any) -> str:
    """A segment label: the value as a string, or UNKNOWN when there is nothing to label."""
    if value is None:
        return UNKNOWN_LABEL
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return UNKNOWN_LABEL
    return text


def _bucket(value: Any, edges: Sequence[float], labels: Sequence[str]) -> str:
    """Which bucket an optional numeric value falls into (edges are exclusive upper bounds)."""
    if value is None:
        return UNKNOWN_LABEL
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return UNKNOWN_LABEL
    if np.isnan(number):
        return UNKNOWN_LABEL
    for edge, label in zip(edges, labels, strict=False):
        if number < edge:
            return label
    return labels[-1]


def _by_option_type(trade: Any) -> str:
    return _label(getattr(trade, "option_type", None))


def _by_time_of_day(trade: Any) -> str:
    return _label(getattr(trade, "time_of_day_bucket", None))


def _by_cycle_phase(trade: Any) -> str:
    return _label(getattr(trade, "expiry_cycle_phase", None))


def _by_day_of_week(trade: Any) -> str:
    stamp = getattr(trade, "entry_timestamp", None)
    if not isinstance(stamp, datetime):
        return UNKNOWN_LABEL
    return WEEKDAY_NAMES[stamp.weekday()]


def _by_days_to_expiry(trade: Any) -> str:
    return _bucket(getattr(trade, "days_to_expiry", None), DTE_BUCKETS, DTE_LABELS)


def _by_strike_distance(trade: Any) -> str:
    return _bucket(
        getattr(trade, "distance_from_atm", None), STRIKE_DISTANCE_BUCKETS, STRIKE_DISTANCE_LABELS
    )


def _by_regime(trade: Any) -> str:
    return _label(getattr(trade, "nifty_state", None))


def _by_exit_reason(trade: Any) -> str:
    return _label(getattr(trade, "exit_reason", None))


def _by_expiry(trade: Any) -> str:
    value = getattr(trade, "expiry", None)
    if value is None:
        return UNKNOWN_LABEL
    stamp = pd.Timestamp(value)
    return UNKNOWN_LABEL if pd.isna(stamp) else stamp.date().isoformat()


#: Segment name -> (what it means, how to label one trade). The order is the report's order.
SEGMENT_KEYS: dict[str, tuple[str, Callable[[Any], str]]] = {
    "option_type": ("CE vs PE", _by_option_type),
    "time_of_day_bucket": ("time-of-day bucket of the entry", _by_time_of_day),
    "expiry_cycle_phase": ("expiry-cycle phase of the entry", _by_cycle_phase),
    "day_of_week": ("weekday of the entry", _by_day_of_week),
    "days_to_expiry": ("days to expiry at entry, bucketed", _by_days_to_expiry),
    "strike_distance": (
        "distance from ATM at entry, bucketed (needs the research column)",
        _by_strike_distance,
    ),
    "nifty_state": ("NIFTY regime at the signal bar", _by_regime),
    "exit_reason": ("why the trade closed", _by_exit_reason),
    "expiry": ("the contract's expiry date", _by_expiry),
}



@dataclass(slots=True)
class SegmentRow:
    """One segment: its label, its sample, its metrics, and whether it can be believed yet."""

    label: str
    sample: int
    preliminary: bool
    metrics: TradeMetrics

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "label": self.label,
            "sample": self.sample,
            "preliminary": self.preliminary,
            **self.metrics.to_dict(),
        }


@dataclass(slots=True)
class SegmentTable:
    """One segmentation of a ledger, ordered by the metric a reader scans first."""

    name: str
    description: str
    rows: list[SegmentRow] = field(default_factory=list)
    preliminary_min_samples: int = 0

    @property
    def preliminary_rows(self) -> list[str]:
        """Labels whose sample is too small to be read as anything but an observation."""
        return [row.label for row in self.rows if row.preliminary]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "name": self.name,
            "description": self.description,
            "preliminary_min_samples": self.preliminary_min_samples,
            "preliminary_labels": self.preliminary_rows,
            "rows": [row.to_dict() for row in self.rows],
        }


def build_segment_table(
    name: str,
    trades: Sequence[Any],
    *,
    preliminary_min_samples: int,
) -> SegmentTable:
    """Group ``trades`` by one segment key and metric each group.

    Rows are ordered by net P&L (best first), which is a presentation choice - the sample
    size and the PRELIMINARY flag sit beside every row so the ordering cannot be mistaken
    for evidence.
    """
    if name not in SEGMENT_KEYS:
        raise MetricsError(f"unknown segment {name!r}; known segments: {sorted(SEGMENT_KEYS)}")
    description, extractor = SEGMENT_KEYS[name]
    groups: dict[str, list[Any]] = {}
    for trade in trades:
        groups.setdefault(extractor(trade), []).append(trade)
    rows = [
        SegmentRow(
            label=label,
            sample=len(group),
            preliminary=len(group) < int(preliminary_min_samples),
            metrics=TradeMetrics.from_trades(group),
        )
        for label, group in groups.items()
    ]
    rows.sort(key=lambda row: (-row.metrics.total_net_pnl, row.label))
    return SegmentTable(
        name=name,
        description=description,
        rows=rows,
        preliminary_min_samples=int(preliminary_min_samples),
    )


def build_segment_tables(
    trades: Sequence[Any],
    *,
    preliminary_min_samples: int,
    names: Sequence[str] | None = None,
) -> list[SegmentTable]:
    """Every (or the requested) segmentation of one ledger, in the report's order."""
    chosen = list(names) if names is not None else list(SEGMENT_KEYS)
    return [
        build_segment_table(name, trades, preliminary_min_samples=preliminary_min_samples)
        for name in chosen
    ]

