"""Event-driven backtest engine with a full trade ledger (spec 23).

The chain the master spec asks for is exactly the chain implemented here:

    signal at the close of candle T  ->  fill at the open of T + 1 (+ latency)
    a position, with stop / target / time / signal / session-end exits
    a ledger row per *closed* trade, costs charged on both legs

Four rules make the result trustworthy rather than decorative:

* **No look-ahead, structurally.** A signal read at bar ``i`` can only be filled at bar
  ``i + 1 + latency``; when that bar does not exist inside the same session the signal is
  dropped and counted. Nothing in the exit logic can reach a bar before the fill bar, and
  every contextual field of a trade (regime, bucket, cycle phase, ATR) is read from the
  *signal* bar's feature row - the information the decision was actually made with.
* **Every trade is a ledger row, including the losers.** A loss is not filtered, rescaled
  or hidden: it is written with its costs, its MFE/MAE and the reason it closed.
* **Every rejection is counted.** A bar with no feature row, a bar with no probability, a
  probability below the threshold, the model naming another direction, a signal with no
  fill bar, a fill mode that wanted a captured spread and had none - each has a counter, so
  "why are there only three trades?" is a number, not a guess.
* **The bridge from prediction to position is a threshold, and it says so.** Spec 24's
  decision layer (`--mode decide`) is separate and this replay does not use it;
  :class:`SignalPolicy` is the deliberately simple stand-in (enter when the model's
  probability for the option side's directional class clears
  ``backtest.signal.min_probability``). Every report prints that sentence. It is a
  way to *measure costs*, not a claimed edge.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from core.backtest.costs import CostModel, CostPolicy, TradeCosts
from core.constants import DEFAULT_TIMEZONE, TRADE_COLUMNS, get_timezone, timeframe_to_minutes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.config_loader import AnjaneyaConfig

__all__ = [
    "ENTRY_SIGNAL",
    "EXIT_DATA_END",
    "EXIT_REASONS",
    "EXIT_SESSION_END",
    "EXIT_SIGNAL",
    "EXIT_STOP_LOSS",
    "EXIT_TARGET",
    "EXIT_TIME",
    "INTRABAR_PRIORITIES",
    "PREDICTION_PREFIX",
    "RESEARCH_COLUMNS",
    "BacktestCounts",
    "BacktestError",
    "BacktestResult",
    "Backtester",
    "ExecutionPolicy",
    "SignalDecision",
    "SignalPolicy",
    "Trade",
    "trading_days",
]

#: Prefix of a predicted class-probability column in the probability frame.
PREDICTION_PREFIX = "p_"

#: Why a trade closed. Each is counted, and each shows up in the ledger's ``exit_reason``.
EXIT_STOP_LOSS = "STOP_LOSS"
EXIT_TARGET = "TARGET"
EXIT_TIME = "TIME_EXIT"
EXIT_SIGNAL = "SIGNAL_EXIT"
EXIT_SESSION_END = "SESSION_END"
EXIT_DATA_END = "DATA_END"
EXIT_REASONS: tuple[str, ...] = (
    EXIT_STOP_LOSS,
    EXIT_TARGET,
    EXIT_TIME,
    EXIT_SIGNAL,
    EXIT_SESSION_END,
    EXIT_DATA_END,
)

#: The entry reason prefix (``SIGNAL:<class>``), so a ledger row says why it opened.
ENTRY_SIGNAL = "SIGNAL"

#: What to do when one bar touches both the stop and the target. There is no way to know
#: the intrabar path from bars, so the choice is explicit and the count is reported.
INTRABAR_PRIORITIES: tuple[str, str] = ("stop_first", "target_first")

#: Research columns a trade carries that the v1 ``trades`` table does not store. They are
#: written to the Parquet ledger and used by the segmented metrics; keeping them *out* of
#: the table is what keeps ``schema_version`` at 1.
RESEARCH_COLUMNS: tuple[str, ...] = ("distance_from_atm", "day_of_week", "bid_ask_spread")


class BacktestError(RuntimeError):
    """Raised when a backtest cannot be run from the inputs it was handed."""

#: The directional class each option side needs the model to be pointing at.
DEFAULT_DIRECTIONS: dict[str, str] = {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}

#: The only signal rule implemented (see the module docstring).
SIGNAL_PROBABILITY_THRESHOLD = "probability_threshold"


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """Why a bar's prediction justified a position, in the terms the model speaks."""

    option_type: str
    directional_class: str
    probability: float
    winning_class: str
    winning_probability: float
    runner_up_probability: float

    @property
    def margin(self) -> float:
        """How far the model's top class is ahead of the second - a confidence proxy."""
        return self.winning_probability - self.runner_up_probability

    @property
    def reason(self) -> str:
        """The ledger's ``entry_reason`` (``SIGNAL:<directional class>``)."""
        return f"{ENTRY_SIGNAL}:{self.directional_class}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "option_type": self.option_type,
            "directional_class": self.directional_class,
            "probability": self.probability,
            "winning_class": self.winning_class,
            "winning_probability": self.winning_probability,
            "runner_up_probability": self.runner_up_probability,
            "margin": self.margin,
        }


@dataclass(frozen=True, slots=True)
class SignalPolicy:
    """The bridge from a predicted distribution to a position (``backtest.signal``).

    This is **not** spec 24's decision layer. It is the smallest honest thing that lets a
    backtest measure costs: enter the option side whose directional class the model gives at
    least ``min_probability``, optionally requiring that class to be the model's top class.
    With ``require_argmax`` off, a strong-but-not-winning directional class still trades -
    which is a different strategy, and the policy records which one ran.
    """

    min_probability: float = 0.40
    directions: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DIRECTIONS)
    )
    require_argmax: bool = True
    mode: str = SIGNAL_PROBABILITY_THRESHOLD

    def __post_init__(self) -> None:
        """Refuse a policy that could never fire or could fire on nothing."""
        if self.mode != SIGNAL_PROBABILITY_THRESHOLD:
            raise BacktestError(
                f"unknown signal mode {self.mode!r}; the only rule this replay implements is "
                f"{SIGNAL_PROBABILITY_THRESHOLD!r} (spec 24's decision layer is the separate "
                "--mode decide, which replays nothing)"
            )
        if not 0.0 < self.min_probability <= 1.0:
            raise BacktestError(
                f"min_probability must be in (0, 1], got {self.min_probability}"
            )
        if not self.directions:
            raise BacktestError("signal.directions must map at least one option type to a class")
        blank = [name for name, value in self.directions.items() if not str(value).strip()]
        if blank:
            raise BacktestError(f"signal.directions has an empty class for {blank}")

    @classmethod
    def from_config(cls, config: AnjaneyaConfig, **overrides: Any) -> SignalPolicy:
        """Build the policy from a loaded configuration (``--min-probability`` overrides)."""
        signal = config.model.backtest.signal
        payload: dict[str, Any] = {
            "min_probability": float(signal.min_probability),
            "directions": {str(k): str(v) for k, v in dict(signal.directions).items()},
            "require_argmax": bool(signal.require_argmax),
            "mode": str(signal.mode),
        }
        payload.update({name: value for name, value in overrides.items() if value is not None})
        return cls(**payload)

    def direction(self, option_type: str | None) -> str | None:
        """The directional class ``option_type`` needs, or ``None`` when unmapped."""
        if not option_type:
            return None
        return self.directions.get(str(option_type).upper())

    def decide(
        self, *, option_type: str | None, probabilities: Mapping[str, float]
    ) -> SignalDecision | None:
        """The decision this distribution justifies, or ``None`` (a pure function)."""
        wanted = self.direction(option_type)
        if wanted is None or not probabilities:
            return None
        ranked = sorted(
            ((str(name), float(value)) for name, value in probabilities.items()),
            key=lambda item: (-item[1], item[0]),
        )
        candidate = next((value for name, value in ranked if name == wanted), None)
        if candidate is None or candidate < self.min_probability:
            return None
        winner, winner_probability = ranked[0]
        if self.require_argmax and winner != wanted:
            return None
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return SignalDecision(
            option_type=str(option_type).upper() if option_type else "",
            directional_class=wanted,
            probability=candidate,
            winning_class=winner,
            winning_probability=winner_probability,
            runner_up_probability=runner_up,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape, including the sentence that keeps it honest."""
        return {
            "mode": self.mode,
            "min_probability": self.min_probability,
            "require_argmax": self.require_argmax,
            "directions": {str(k): str(v) for k, v in self.directions.items()},
            "note": (
                "a threshold rule, not spec 24's decision layer: it exists so costs can be "
                "measured, and every number it produces is conditional on an arbitrary cut"
            ),
        }



#: The fill modes. ``bid_ask`` means "the fill must cross the captured spread"; when no
#: spread is stored for that bar the run falls back to the plain open and *counts* it.
FILL_NEXT_OPEN = "next_open"
FILL_BID_ASK = "bid_ask"
FILL_MODES: tuple[str, str] = (FILL_NEXT_OPEN, FILL_BID_ASK)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """How a signal becomes a fill, and when a position ends (``execution`` + ``backtest``).

    ``latency_ms`` is configuration in milliseconds; bars are what a stored dataset has, so
    it is rounded **up** to whole bars at policy construction and the rounding is recorded
    in :meth:`describe`. A sub-bar latency can never be honoured exactly, and pretending
    otherwise would be a silent look-ahead.
    """

    fill_at: str = FILL_NEXT_OPEN
    signal_at: str = "candle_close"
    latency_bars: int = 0
    latency_ms: int = 0
    timeframe: str = "1m"
    lots: int = 1
    lot_size: int = 1
    quantity: int = 1
    stop_loss_atr: float | None = 1.0
    target_atr: float | None = 1.5
    max_holding_minutes: float | None = 30.0
    flat_at_session_end: bool = True
    intrabar_priority: str = "stop_first"
    exit_on_opposite_signal: bool = True

    def __post_init__(self) -> None:
        """Refuse an execution model that is impossible or unlabelled."""
        if self.signal_at != "candle_close":
            raise BacktestError(
                f"signal_at {self.signal_at!r} is not implemented: a signal is read at the "
                "close of a candle (spec 23), because a bar's close is the first instant its "
                "own high/low/close are known"
            )
        if self.fill_at not in FILL_MODES:
            raise BacktestError(f"unknown fill_at {self.fill_at!r}; expected {list(FILL_MODES)}")
        if self.intrabar_priority not in INTRABAR_PRIORITIES:
            raise BacktestError(
                f"unknown intrabar_priority {self.intrabar_priority!r}; expected "
                f"{list(INTRABAR_PRIORITIES)}"
            )
        if self.quantity <= 0:
            raise BacktestError(f"quantity must be positive, got {self.quantity}")
        if self.latency_bars < 0:
            raise BacktestError("latency (in bars) cannot be negative")
        for name, value in (
            ("stop_loss_atr", self.stop_loss_atr),
            ("target_atr", self.target_atr),
            ("max_holding_minutes", self.max_holding_minutes),
        ):
            if value is not None and value <= 0:
                raise BacktestError(f"{name} must be positive when set, got {value}")

    @classmethod
    def from_config(
        cls, config: AnjaneyaConfig, *, timeframe: str | None = None, **overrides: Any
    ) -> ExecutionPolicy:
        """Build the policy from a loaded configuration (``--timeframe`` selects the bars)."""
        from core.config_loader import ConfigError

        execution = config.model.execution
        backtest = config.model.backtest
        chosen = timeframe or config.model.dataset.timeframe
        if chosen not in config.settings.market.timeframes:
            raise ConfigError(
                f"backtest timeframe {chosen!r} is not enabled in settings.market.timeframes "
                f"({', '.join(config.settings.market.timeframes)})"
            )
        minutes = timeframe_to_minutes(chosen)
        latency_ms = int(overrides.get("latency_ms") or execution.latency_ms)
        bar_ms = minutes * 60_000
        payload: dict[str, Any] = {
            "fill_at": str(execution.fill_at),
            "signal_at": str(execution.signal_at),
            "latency_ms": latency_ms,
            "latency_bars": int(math.ceil(latency_ms / bar_ms)) if latency_ms > 0 else 0,
            "timeframe": chosen,
            "lots": int(backtest.sizing.lots),
            "lot_size": int(backtest.sizing.lot_size),
            "quantity": int(backtest.sizing.lots) * int(backtest.sizing.lot_size),
            "stop_loss_atr": backtest.exit.stop_loss_atr,
            "target_atr": backtest.exit.target_atr,
            "max_holding_minutes": backtest.exit.max_holding_minutes,
            "flat_at_session_end": bool(backtest.exit.flat_at_session_end),
            "intrabar_priority": str(backtest.exit.intrabar_priority),
            "exit_on_opposite_signal": bool(backtest.exit.exit_on_opposite_signal),
        }
        payload.update({name: value for name, value in overrides.items() if value is not None})
        return cls(**payload)

    @property
    def bar_minutes(self) -> int:
        """Minutes per bar (holding time is measured in these)."""
        return timeframe_to_minutes(self.timeframe)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every backtest summary."""
        return {
            "signal_at": self.signal_at,
            "fill_at": self.fill_at,
            "latency_ms": self.latency_ms,
            "latency_bars": self.latency_bars,
            "timeframe": self.timeframe,
            "bar_minutes": self.bar_minutes,
            "lots": self.lots,
            "lot_size": self.lot_size,
            "quantity": self.quantity,
            "stop_loss_atr": self.stop_loss_atr,
            "target_atr": self.target_atr,
            "max_holding_minutes": self.max_holding_minutes,
            "flat_at_session_end": self.flat_at_session_end,
            "intrabar_priority": self.intrabar_priority,
            "exit_on_opposite_signal": self.exit_on_opposite_signal,
        }



#: The in-memory ledger contract: the stored ``trades`` columns, then the research extras.
LEDGER_COLUMNS: tuple[str, ...] = (*TRADE_COLUMNS, *RESEARCH_COLUMNS)


def _plain(value: Any) -> Any:
    """A value DuckDB's ``executemany`` accepts (no numpy/pandas scalar types)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime | date):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


@dataclass(slots=True)
class Trade:
    """One simulated round trip: a long premium position, opened and closed.

    The first 28 fields are exactly the stored ``trades`` columns (spec 23's trade record);
    the last three are research columns the v1 schema does not carry (see
    :data:`RESEARCH_COLUMNS`). A trade is always *closed* when it reaches a ledger: the
    engine forces a flat at the session's last bar and at the end of the data.
    """

    trade_id: str
    run_id: str
    entry_timestamp: datetime
    tradingsymbol: str
    option_type: str | None
    entry_price: float
    quantity: int
    exit_timestamp: datetime | None = None
    expiry: date | None = None
    strike: float | None = None
    exit_price: float | None = None
    entry_reason: str | None = None
    exit_reason: str | None = None
    nifty_state: str | None = None
    option_state: str | None = None
    candle_state: str | None = None
    support_resistance_state: str | None = None
    ai_prediction: str | None = None
    ai_probability: float | None = None
    gross_pnl: float | None = None
    costs: float | None = None
    net_pnl: float | None = None
    mfe: float | None = None
    mae: float | None = None
    holding_minutes: float | None = None
    time_of_day_bucket: str | None = None
    expiry_cycle_phase: str | None = None
    days_to_expiry: float | None = None
    # -- research columns (Parquet ledger + segmented metrics only) ----------
    distance_from_atm: float | None = None
    day_of_week: int | None = None
    bid_ask_spread: float | None = None

    @staticmethod
    def ledger_columns() -> tuple[str, ...]:
        """Every column the Parquet ledger carries, stored columns first."""
        return LEDGER_COLUMNS

    def to_record(self) -> dict[str, Any]:
        """The row as the stored ``trades`` table expects it (spec 23's 28 columns)."""
        return {name: _plain(getattr(self, name)) for name in TRADE_COLUMNS}

    def to_row(self) -> dict[str, Any]:
        """The full research row (stored columns plus the extras)."""
        return {name: _plain(getattr(self, name)) for name in LEDGER_COLUMNS}

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return self.to_row()


def _contract_ok() -> bool:
    """``Trade`` must carry exactly the ledger contract - checked once, at import."""
    fields = tuple(Trade.__dataclass_fields__)
    return set(fields) == set(LEDGER_COLUMNS)


if not _contract_ok():  # pragma: no cover - a development guard, not a runtime path
    raise BacktestError(
        "the Trade ledger does not match TRADE_COLUMNS + RESEARCH_COLUMNS: "
        f"missing {sorted(set(LEDGER_COLUMNS) - set(Trade.__dataclass_fields__))}, "
        f"extra {sorted(set(Trade.__dataclass_fields__) - set(LEDGER_COLUMNS))}"
    )


    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every backtest summary."""

@dataclass(slots=True)
class BacktestCounts:
    """What the engine saw and what it refused - the audit trail behind the ledger.

    Every field exists to answer a question a reader will ask about a suspiciously small
    or suspiciously good ledger: how many bars had context, how many had a prediction, how
    many cleared the threshold, how many filled, and why the rest did not.
    """

    contracts: int = 0
    sessions: int = 0
    bars: int = 0
    decision_bars: int = 0
    bars_without_features: int = 0
    bars_without_probability: int = 0
    in_position_bars: int = 0
    no_signal: int = 0
    decisions: int = 0
    no_fill_bar: int = 0
    entries: int = 0
    last_bar_entries: int = 0
    both_touched: int = 0
    spread_fills: int = 0
    spread_missing: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)

    def count_exit(self, reason: str) -> None:
        """Record one closed trade."""
        self.exit_reasons[reason] = self.exit_reasons.get(reason, 0) + 1

    @property
    def exposure(self) -> float | None:
        """Fraction of bars spent holding a position, or ``None`` when there are no bars."""
        if not self.bars:
            return None
        return self.in_position_bars / self.bars

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "contracts": self.contracts,
            "sessions": self.sessions,
            "bars": self.bars,
            "decision_bars": self.decision_bars,
            "bars_without_features": self.bars_without_features,
            "bars_without_probability": self.bars_without_probability,
            "in_position_bars": self.in_position_bars,
            "exposure": self.exposure,
            "no_signal": self.no_signal,
            "decisions": self.decisions,
            "no_fill_bar": self.no_fill_bar,
            "entries": self.entries,
            "last_bar_entries": self.last_bar_entries,
            "both_touched": self.both_touched,
            "spread_fills": self.spread_fills,
            "spread_missing": self.spread_missing,
            "exit_reasons": dict(self.exit_reasons),
        }


@dataclass(slots=True)
class BacktestResult:
    """One engine run: the ledger, the counters, and the policies it ran under."""

    trades: list[Trade]
    counts: BacktestCounts
    cost_policy: CostPolicy
    execution: ExecutionPolicy
    signal: SignalPolicy
    timeframe: str
    data_start: datetime | None = None
    data_end: datetime | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def gross_pnl(self) -> float:
        """Sum of the ledger's gross P&L (before costs)."""
        return float(sum(trade.gross_pnl or 0.0 for trade in self.trades))

    @property
    def total_costs(self) -> float:
        """Sum of every rupee of friction in the ledger."""
        return float(sum(trade.costs or 0.0 for trade in self.trades))

    @property
    def net_pnl(self) -> float:
        """Sum of the ledger's net P&L (after costs) - losses included by construction."""
        return float(sum(trade.net_pnl or 0.0 for trade in self.trades))

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary of the run (the ledger itself stays in the ledger)."""
        return {
            "timeframe": self.timeframe,
            "trades": len(self.trades),
            "data_start": self.data_start.isoformat() if self.data_start else None,
            "data_end": self.data_end.isoformat() if self.data_end else None,
            "gross_pnl": self.gross_pnl,
            "total_costs": self.total_costs,
            "net_pnl": self.net_pnl,
            "counts": self.counts.to_dict(),
            "exit_reasons": dict(self.counts.exit_reasons),
            "notes": list(self.notes),
        }

        return {
            "signal_at": self.signal_at,
            "fill_at": self.fill_at,
            "latency_ms": self.latency_ms,
            "latency_bars": self.latency_bars,
            "timeframe": self.timeframe,
            "bar_minutes": self.bar_minutes,
            "lots": self.lots,
            "lot_size": self.lot_size,
            "quantity": self.quantity,
            "stop_loss_atr": self.stop_loss_atr,
            "target_atr": self.target_atr,
            "max_holding_minutes": self.max_holding_minutes,
            "flat_at_session_end": self.flat_at_session_end,
            "intrabar_priority": self.intrabar_priority,
            "exit_on_opposite_signal": self.exit_on_opposite_signal,
        }

#: Context columns read from the feature row **at the signal bar** and copied into the
#: ledger. Every one is optional: when the frame does not carry it, the ledger field is
#: None. A numeric feature is never written into a column that holds a *state*.
CONTEXT_MAP: dict[str, str] = {
    "nifty_regime": "nifty_state",
    "candle_direction": "candle_state",
    "time_of_day_bucket": "time_of_day_bucket",
    "expiry_cycle_phase": "expiry_cycle_phase",
    "days_to_expiry": "days_to_expiry",
    "distance_from_atm": "distance_from_atm",
    "expiry": "expiry",
    "strike": "strike",
}

#: Columns the engine requires in each input frame.
CANDLE_REQUIRED: tuple[str, ...] = ("timestamp", "tradingsymbol", "open", "high", "low", "close")
FEATURE_REQUIRED: tuple[str, ...] = ("timestamp", "tradingsymbol")
PROBABILITY_REQUIRED: tuple[str, ...] = ("timestamp", "tradingsymbol")

#: Ledger field -> the context column it is read from (the inverse of ``CONTEXT_MAP``). The
#: two names differ where the ledger and the feature contract call the same thing differently
#: (``nifty_state`` is the feature row's ``nifty_regime``, ``candle_state`` its
#: ``candle_direction``), and stating that here is what stops a silent ``None``.
CONTEXT_SOURCES: dict[str, str] = {ledger: column for column, ledger in CONTEXT_MAP.items()}


def _context_field(context: Mapping[str, Any], field: str) -> Any:
    """A ledger field's value from the signal bar's context (``None`` when absent)."""
    return context.get(CONTEXT_SOURCES.get(field, field))


@dataclass(slots=True)
class _OpenPosition:
    """A position the engine is holding (internal)."""

    trade_id: str
    symbol: str
    signal_index: int
    entry_index: int
    entry_time: datetime
    entry_price: float
    quantity: int
    option_type: str | None
    decision: SignalDecision
    atr: float | None
    context: dict[str, Any]


@dataclass(slots=True)
class _Exit:
    """How and where a position ended (internal).

    ``at_close`` records whether the fill happened at a bar's close (a time/session/data-end
    exit) or at a bar's open (a gap or signal fill). Holding time is measured to the *fill*,
    so a trade that was flattened at a close is one bar longer than its timestamps suggest -
    and a ``TIME_EXIT`` always reports at least the holding limit it was closed by.
    """

    index: int
    price: float
    reason: str
    mfe: float
    mae: float
    at_close: bool = False


def _epoch_ns(value: Any) -> int:
    """A timestamp as nanoseconds since the epoch - an unambiguous join key across frames."""
    return int(pd.Timestamp(value).value)


def _local_days(frame: pd.DataFrame, timezone: Any) -> pd.Series:
    """The exchange-local trading day of every timestamp in ``frame``."""
    return pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(timezone).dt.date


def trading_days(candles: pd.DataFrame, *, timezone_name: str = DEFAULT_TIMEZONE) -> list[date]:
    """The distinct IST trading days a candle frame covers, oldest first.

    A day is defined by exchange-local midnight, exactly as the calendar, the coverage
    ledger and spec 4.3's data gates define it - so "how many days was this backtest run
    over?" has one answer, not one per module.
    """
    days = _local_days(candles, get_timezone(timezone_name))
    return sorted({day for day in days.tolist() if day is not None})


class Backtester:
    """Replay stored bars against a model's predictions and charge spec 23's costs."""

    def __init__(
        self,
        *,
        cost_model: CostModel,
        execution: ExecutionPolicy,
        signal: SignalPolicy,
        run_id: str,
        logger: Any = None,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> None:
        self._costs = cost_model
        self._execution = execution
        self._signal = signal
        self._run_id = run_id
        self._logger = logger
        self._timezone = get_timezone(timezone_name)

    # -- inputs --------------------------------------------------------------
    def run(
        self,
        *,
        candles: pd.DataFrame,
        features: pd.DataFrame,
        probabilities: pd.DataFrame,
    ) -> BacktestResult:
        """Replay ``candles`` with the model's ``probabilities`` and return the ledger.

        All three frames are keyed by ``(timestamp, tradingsymbol)``: the bars are the
        market, the feature rows say what was known at each bar, and the probability frame
        is what the model predicted from those rows. A bar with no feature row or no
        prediction cannot produce a trade - it is counted, never guessed at.
        """
        self._check_frames(candles=candles, features=features, probabilities=probabilities)
        probability_index, _classes = self._index_probabilities(probabilities)
        context_index = self._index_context(features)

        trades: list[Trade] = []
        counts = BacktestCounts()
        first = last = None
        contracts: set[str] = set()
        for (symbol, _day), session in self._sessions(candles):
            if first is None:
                first = session["timestamp"].iloc[0]
            last = session["timestamp"].iloc[-1]
            contracts.add(symbol)
            counts.sessions += 1
            self._run_session(
                symbol=symbol,
                session=session,
                probabilities=probability_index,
                contexts=context_index,
                trades=trades,
                counts=counts,
            )
        counts.contracts = len(contracts)
        counts.entries = len(trades)
        counts.spread_fills = self._costs.spread_orders
        return BacktestResult(
            trades=trades,
            counts=counts,
            cost_policy=self._costs.policy,
            execution=self._execution,
            signal=self._signal,
            timeframe=self._execution.timeframe,
            data_start=first,
            data_end=last,
            notes=self._notes(counts),
        )

    def _notes(self, counts: BacktestCounts) -> list[str]:
        """The sentences a reader should carry away from this run, always printed."""
        notes = [
            "every closed trade is in the ledger, including every loser",
            "prices are the stored bars' own prices; friction is charged as a cost and never "
            "folded into entry_price/exit_price",
        ]
        if self._execution.fill_at == FILL_BID_ASK and self._costs.policy.use_captured_spread:
            notes.append(
                f"bid/ask fills: {counts.spread_fills} order(s) crossed a captured spread, "
                f"{counts.spread_missing} bar(s) had none and fell back to the bar open"
            )
        elif self._execution.fill_at == FILL_BID_ASK:
            notes.append(
                "fill_at is bid_ask but costs.use_captured_spread is false: fills are the bar "
                "open and slippage stands in - which is what the cost model charged"
            )
        return notes

    def _check_frames(
        self, *, candles: pd.DataFrame, features: pd.DataFrame, probabilities: pd.DataFrame
    ) -> None:
        """Refuse inputs that cannot produce an honest ledger, before doing any work."""
        for frame, required, what in (
            (candles, CANDLE_REQUIRED, "candles"),
            (features, FEATURE_REQUIRED, "features"),
            (probabilities, PROBABILITY_REQUIRED, "probabilities"),
        ):
            missing = [name for name in required if name not in frame.columns]
            if missing:
                raise BacktestError(f"the {what} frame is missing column(s) {missing}")
        if candles.empty:
            raise BacktestError("no candles to backtest")
        if not any(str(name).startswith(PREDICTION_PREFIX) for name in probabilities.columns):
            raise BacktestError(
                f"the probabilities frame has no {PREDICTION_PREFIX}<class> column; it must be "
                "the output of core.ml.predictor.ModelPredictor.predict_frame"
            )
        if "option_type" not in features.columns and "option_type" not in candles.columns:
            raise BacktestError(
                "neither the features nor the candles frame carries option_type, so no signal "
                "can be attributed to CE or PE"
            )
        needs_atr = (
            self._execution.stop_loss_atr is not None or self._execution.target_atr is not None
        )
        if needs_atr and "option_atr" not in features.columns:
            raise BacktestError(
                "the exit policy uses an ATR multiple but the features frame has no option_atr "
                "column; build the feature rows first or turn the ATR exits off"
            )
        prices = candles[["open", "high", "low", "close"]].to_numpy(dtype="float64")
        bad = (~np.isfinite(prices).all(axis=1)) | (prices <= 0).any(axis=1)
        if bad.any():
            first_bad = candles.loc[bad, "timestamp"].iloc[0]
            raise BacktestError(
                f"{int(bad.sum())} candle(s) carry a non-positive or non-finite price (first: "
                f"{first_bad}); a premium that is not a positive finite number cannot be filled "
                "and must be fixed in the candle table, not skipped silently"
            )


    # -- indexing ------------------------------------------------------------
    def _index_probabilities(
        self, frame: pd.DataFrame
    ) -> tuple[dict[tuple[str, int], dict[str, float]], tuple[str, ...]]:
        """``(symbol, instant) -> {class: probability}``, refusing anything that is not one."""
        columns = [str(name) for name in frame.columns if str(name).startswith(PREDICTION_PREFIX)]
        classes = tuple(sorted(name[len(PREDICTION_PREFIX) :] for name in columns))
        if not classes:
            raise BacktestError("the probabilities frame carries no class column")
        values = {name: frame[name].to_numpy(dtype="float64") for name in columns}
        index: dict[tuple[str, int], dict[str, float]] = {}
        for position, (stamp, symbol) in enumerate(
            zip(frame["timestamp"].tolist(), frame["tradingsymbol"].tolist(), strict=True)
        ):
            distribution = {
                name[len(PREDICTION_PREFIX) :]: float(values[name][position]) for name in columns
            }
            self._check_distribution(distribution, symbol=symbol, stamp=stamp)
            index[(str(symbol), _epoch_ns(stamp))] = distribution
        return index, classes

    def _check_distribution(
        self, distribution: Mapping[str, float], *, symbol: Any, stamp: Any
    ) -> None:
        """A row that claims to be a probability distribution has to be one."""
        for name, value in distribution.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise BacktestError(
                    f"{symbol} at {stamp}: probability for {name!r} is {value}, which is not in "
                    "[0, 1] - the engine thresholds probabilities, not scores"
                )
        total = sum(distribution.values())
        if abs(total - 1.0) > 1e-3:
            raise BacktestError(
                f"{symbol} at {stamp}: the class probabilities sum to {total:.6f}, not 1; pass "
                "predictions (ModelPredictor), not raw scores - a threshold is meaningless "
                "without a normalised distribution"
            )

    def _index_context(self, frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
        """``(symbol, instant) -> the row's context``, with missing values as ``None``."""
        optional = (
            "option_type",
            "option_atr",
            "bid_ask_spread",
            "event_today_high_state",
            "event_today_low_state",
            *CONTEXT_MAP,
        )
        names = [name for name in dict.fromkeys(optional) if name in frame.columns]
        index: dict[tuple[str, int], dict[str, Any]] = {}
        records = frame[["timestamp", "tradingsymbol", *names]].to_dict("records")
        for record in records:
            key = (str(record["tradingsymbol"]), _epoch_ns(record["timestamp"]))
            index[key] = {name: _missing_to_none(record[name]) for name in names}
        return index

    def _sessions(self, candles: pd.DataFrame) -> Iterable[tuple[tuple[str, date], pd.DataFrame]]:
        """The bars grouped by ``(contract, IST trading day)``, in stored order.

        A session is the unit the engine replays: a position never spans one, because an
        intraday research backtest on weekly options has no business holding overnight.
        """
        frame = candles.assign(_trading_day=_local_days(candles, self._timezone)).sort_values(
            ["tradingsymbol", "_trading_day", "timestamp"], kind="stable"
        )
        for (symbol, day), group in frame.groupby(["tradingsymbol", "_trading_day"], sort=True):
            yield (str(symbol), day), group


    # -- replay --------------------------------------------------------------
    def _run_session(
        self,
        *,
        symbol: str,
        session: pd.DataFrame,
        probabilities: Mapping[tuple[str, int], Mapping[str, float]],
        contexts: Mapping[tuple[str, int], Mapping[str, Any]],
        trades: list[Trade],
        counts: BacktestCounts,
    ) -> None:
        """Walk one contract-day: signal at a bar's close, fill at the next bar's open."""
        stamps = list(pd.to_datetime(session["timestamp"], utc=True).dt.tz_convert(self._timezone))
        opens = session["open"].to_numpy(dtype="float64")
        highs = session["high"].to_numpy(dtype="float64")
        lows = session["low"].to_numpy(dtype="float64")
        closes = session["close"].to_numpy(dtype="float64")
        option_types = (
            session["option_type"].tolist()
            if "option_type" in session.columns
            else [None] * len(stamps)
        )
        keys = [_epoch_ns(stamp) for stamp in stamps]
        bars = len(stamps)
        counts.bars += bars

        index = 0
        while index < bars:
            context = contexts.get((symbol, keys[index]))
            if context is None:
                counts.bars_without_features += 1
                index += 1
                continue
            distribution = probabilities.get((symbol, keys[index]))
            if distribution is None:
                counts.bars_without_probability += 1
                index += 1
                continue
            counts.decision_bars += 1
            option_type = context.get("option_type") or option_types[index]
            decision = self._signal.decide(option_type=option_type, probabilities=distribution)
            if decision is None:
                counts.no_signal += 1
                index += 1
                continue
            counts.decisions += 1
            entry_index = index + 1 + self._execution.latency_bars
            if entry_index >= bars:
                # The signal arrived too late to fill inside its own session: dropping it is
                # the only honest option - a stale fill would be a look-ahead in reverse.
                counts.no_fill_bar += 1
                index += 1
                continue
            if entry_index == bars - 1 and self._execution.flat_at_session_end:
                counts.last_bar_entries += 1
            position = _OpenPosition(
                trade_id=self._trade_id(symbol, stamps[entry_index]),
                symbol=symbol,
                signal_index=index,
                entry_index=entry_index,
                entry_time=stamps[entry_index],
                entry_price=float(opens[entry_index]),
                quantity=self._execution.quantity,
                option_type=str(option_type).upper() if option_type else None,
                decision=decision,
                atr=_positive_or_none(context.get("option_atr")),
                context=dict(context),
            )
            exit_ = self._walk(
                position=position,
                stamps=stamps,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                keys=keys,
                symbol=symbol,
                probabilities=probabilities,
                counts=counts,
            )
            trades.append(
                self._settle(
                    position=position,
                    exit_=exit_,
                    stamps=stamps,
                    exit_context=contexts.get((symbol, keys[exit_.index])),
                    counts=counts,
                )
            )
            counts.count_exit(exit_.reason)
            counts.in_position_bars += exit_.index - entry_index + 1
            # Resume *at* the exit bar: its close is the next legitimate signal instant, and
            # a new position can only fill at a later bar anyway.
            index = exit_.index


    def _walk(
        self,
        *,
        position: _OpenPosition,
        stamps: Sequence[Any],
        opens: Any,
        highs: Any,
        lows: Any,
        closes: Any,
        keys: Sequence[int],
        symbol: str,
        probabilities: Mapping[tuple[str, int], Mapping[str, float]],
        counts: BacktestCounts,
    ) -> _Exit:
        """Follow one position to its exit, honoring the policy's order of priorities.

        Per bar, in order: a gap through the open (filled at the open - the worse price, so
        a gap is *not* a favour), the bar's own stop/target range, the time exit at the
        close, the opposite signal at the close (filled at the next open, like every other
        fill), and finally the session's last bar. The one ambiguity bars cannot resolve -
        a bar that touches both the stop and the target - is resolved by
        :attr:`ExecutionPolicy.intrabar_priority` and counted.
        """
        policy = self._execution
        entry_price = position.entry_price
        stop = (
            entry_price - policy.stop_loss_atr * position.atr
            if policy.stop_loss_atr is not None and position.atr is not None
            else None
        )
        target = (
            entry_price + policy.target_atr * position.atr
            if policy.target_atr is not None and position.atr is not None
            else None
        )
        mfe = 0.0
        mae = 0.0
        bars = len(stamps)
        for bar in range(position.entry_index, bars):
            if bar > position.entry_index:
                open_price = float(opens[bar])
                if stop is not None and open_price <= stop:
                    return _Exit(bar, open_price, EXIT_STOP_LOSS, mfe, mae)
                if target is not None and open_price >= target:
                    return _Exit(bar, open_price, EXIT_TARGET, mfe, mae)
            high, low = float(highs[bar]), float(lows[bar])
            mfe = max(mfe, high - entry_price)
            mae = min(mae, low - entry_price)
            stop_hit = stop is not None and low <= stop
            target_hit = target is not None and high >= target
            if stop_hit and target_hit:
                counts.both_touched += 1
                chosen = "stop" if policy.intrabar_priority == "stop_first" else "target"
            elif stop_hit:
                chosen = "stop"
            elif target_hit:
                chosen = "target"
            else:
                chosen = ""
            if chosen == "stop":
                return _Exit(bar, float(stop), EXIT_STOP_LOSS, mfe, mae)
            if chosen == "target":
                return _Exit(bar, float(target), EXIT_TARGET, mfe, mae)
            held = (
                pd.Timestamp(stamps[bar]) - pd.Timestamp(position.entry_time)
            ).total_seconds() / 60.0 + policy.bar_minutes
            if policy.max_holding_minutes is not None and held >= policy.max_holding_minutes:
                return _Exit(bar, float(closes[bar]), EXIT_TIME, mfe, mae, at_close=True)
            if policy.exit_on_opposite_signal and bar < bars - 1:
                opposite = self._opposite(
                    position.option_type, probabilities.get((symbol, keys[bar]))
                )
                if opposite is not None:
                    return _Exit(bar + 1, float(opens[bar + 1]), EXIT_SIGNAL, mfe, mae)
            if policy.flat_at_session_end and bar == bars - 1:
                return _Exit(bar, float(closes[bar]), EXIT_SESSION_END, mfe, mae, at_close=True)
        # Only reachable with flat_at_session_end off: the data ended while still holding.
        return _Exit(bars - 1, float(closes[bars - 1]), EXIT_DATA_END, mfe, mae, at_close=True)

    def _opposite(
        self, option_type: str | None, distribution: Mapping[str, float] | None
    ) -> SignalDecision | None:
        """A signal for the *other* side of the same contract - the exit rule's trigger."""
        if distribution is None:
            return None
        held = str(option_type or "").upper()
        for other in sorted(str(name) for name in self._signal.directions):
            if other.upper() == held:
                continue
            decision = self._signal.decide(option_type=other, probabilities=distribution)
            if decision is not None:
                return decision
        return None


    def _settle(
        self,
        *,
        position: _OpenPosition,
        exit_: _Exit,
        stamps: Sequence[Any],
        exit_context: Mapping[str, Any] | None,
        counts: BacktestCounts,
    ) -> Trade:
        """Charge both legs, then write the ledger row (losers included, always)."""
        context = position.context
        entry_spread = self._fill_spread(context, counts)
        exit_spread = self._fill_spread(exit_context, counts)
        costs: TradeCosts = self._costs.round_trip(
            entry_price=position.entry_price,
            exit_price=exit_.price,
            quantity=position.quantity,
            entry_spread=entry_spread,
            exit_spread=exit_spread,
        )
        gross = (exit_.price - position.entry_price) * position.quantity
        exit_time = stamps[exit_.index]
        held = (pd.Timestamp(exit_time) - pd.Timestamp(position.entry_time)).total_seconds() / 60.0
        if exit_.at_close:
            # A fill at a bar's close is one bar of exposure longer than the two timestamps
            # suggest (the bar is stamped at its start), which is the same arithmetic the
            # time-exit rule uses.
            held += self._execution.bar_minutes
        entry_reason = position.decision.reason + (":SPREAD" if costs.entry.spread > 0 else "")
        return Trade(
            trade_id=position.trade_id,
            run_id=self._run_id,
            entry_timestamp=_plain(position.entry_time),
            exit_timestamp=_plain(exit_time),
            tradingsymbol=position.symbol,
            expiry=context.get("expiry"),
            strike=_float_or_none(context.get("strike")),
            option_type=position.option_type,
            entry_price=float(position.entry_price),
            exit_price=float(exit_.price),
            quantity=int(position.quantity),
            entry_reason=entry_reason,
            exit_reason=exit_.reason,
            nifty_state=_string_or_none(_context_field(context, "nifty_state")),
            option_state=None,
            candle_state=_string_or_none(_context_field(context, "candle_state")),
            support_resistance_state=_support_resistance_state(context),
            ai_prediction=position.decision.winning_class,
            ai_probability=float(position.decision.winning_probability),
            gross_pnl=float(gross),
            costs=float(costs.total),
            net_pnl=float(gross - costs.total),
            mfe=float(exit_.mfe),
            mae=float(exit_.mae),
            holding_minutes=float(held),
            time_of_day_bucket=_string_or_none(_context_field(context, "time_of_day_bucket")),
            expiry_cycle_phase=_string_or_none(_context_field(context, "expiry_cycle_phase")),
            days_to_expiry=_float_or_none(_context_field(context, "days_to_expiry")),
            distance_from_atm=_float_or_none(_context_field(context, "distance_from_atm")),
            day_of_week=int(pd.Timestamp(position.entry_time).weekday()),
            bid_ask_spread=None if entry_spread is None else float(entry_spread),
        )

    # -- helpers -------------------------------------------------------------
    def _fill_spread(
        self, context: Mapping[str, Any] | None, counts: BacktestCounts
    ) -> float | None:
        """The captured spread a ``bid_ask`` fill must cross, or ``None`` (counted)."""
        if self._execution.fill_at != FILL_BID_ASK:
            return None
        spread = _positive_or_none(None if context is None else context.get("bid_ask_spread"))
        if spread is None:
            counts.spread_missing += 1
        return spread

    def _trade_id(self, symbol: str, entry_time: Any) -> str:
        """Deterministic and unique per run: the same replay yields the same ledger."""
        return f"{self._run_id}|{symbol}|{pd.Timestamp(entry_time).isoformat()}"


def _missing_to_none(value: Any) -> Any:
    """``NaN``/``NaT`` out of a frame become ``None`` - the ledger writes no fake numbers."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - pd.isna on odd objects
        return value
    return value


def _positive_or_none(value: Any) -> float | None:
    """A positive finite float, or ``None`` (a zero/NaN ATR is not an ATR)."""
    number = _float_or_none(value)
    if number is None or number <= 0.0 or not math.isfinite(number):
        return None
    return number


def _float_or_none(value: Any) -> float | None:
    """A float, or ``None`` when the value is missing or not a number."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _string_or_none(value: Any) -> str | None:
    """A non-empty string, or ``None``."""
    if value is None:
        return None
    text = str(value)
    return text if text and text.lower() != "nan" else None


def _support_resistance_state(context: Mapping[str, Any]) -> str | None:
    """What the level-interaction engine said at the signal bar, composed (never invented).

    The feature row keeps four states (previous-day and today, high and low); the ledger has
    one column. Both *today* states are written, names and all, so a reader can see the
    composition rather than trusting a summary the engine made up.
    """
    high = _string_or_none(context.get("event_today_high_state"))
    low = _string_or_none(context.get("event_today_low_state"))
    if high is None and low is None:
        return None
    return f"TODAY_HIGH={high}|TODAY_LOW={low}"

