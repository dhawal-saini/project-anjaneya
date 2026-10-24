"""Unit tests for the event-driven engine (``core/backtest/engine.py``, spec 23).

An engine is judged by four properties, and the tests are grouped by them:

* **no look-ahead** - a signal read at bar *T* can only fill at bar *T+1+latency*; a signal
  with no fill bar inside its session is dropped and counted, never filled at a stale price;
* **the exits are where they claim to be** - target at the target, stop at the stop, a gap
  through the stop at the *open*, the time exit at the close, the opposite signal at the next
  open, and a forced flat at the session's end;
* **the ledger is complete** - losers included, costs charged on both legs, MFE/MAE and the
  context of the *signal* bar carried into a row that matches the stored ``trades`` contract;
* **refusals are counted** - a bar without features, without a prediction, below the
  threshold, or with no fill bar shows up as a number, not as a silent gap.

Every fixture here is a handful of bars built in the test, so every expected price is a
number in the test file rather than a promise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.backtest import (
    ENTRY_SIGNAL,
    EXIT_DATA_END,
    EXIT_SESSION_END,
    EXIT_SIGNAL,
    EXIT_STOP_LOSS,
    EXIT_TARGET,
    EXIT_TIME,
    LEDGER_COLUMNS,
    Backtester,
    BacktestError,
    CostModel,
    CostPolicy,
    ExecutionPolicy,
    SignalPolicy,
    trading_days,
)
from core.constants import TRADE_COLUMNS

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 9, 28)
SYMBOL = "NIFTY01OCT2625000CE"
PE_SYMBOL = "NIFTY01OCT2625000PE"
UP = "UP_CONTINUATION"
DOWN = "DOWN_CONTINUATION"
RANGE = "RANGE"
REVERSAL = "REVERSAL"
CLASSES = (UP, DOWN, RANGE, REVERSAL)


# ---------------------------------------------------------------------------
# Fixtures: a few bars, their context, and their predictions
# ---------------------------------------------------------------------------
def stamps(count: int, *, day: date = DAY) -> list[datetime]:
    """``count`` one-minute bar timestamps from the 09:15 IST session open."""
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    return [start + timedelta(minutes=index) for index in range(count)]


def candle_frame(
    bars: Sequence[tuple[float, float, float, float]],
    *,
    symbol: str = SYMBOL,
    option_type: str = "CE",
    day: date = DAY,
) -> pd.DataFrame:
    """A candle frame from explicit ``(open, high, low, close)`` bars."""
    moments = stamps(len(bars), day=day)
    return pd.DataFrame(
        {
            "timestamp": moments,
            "tradingsymbol": symbol,
            "option_type": option_type,
            "open": [bar[0] for bar in bars],
            "high": [bar[1] for bar in bars],
            "low": [bar[2] for bar in bars],
            "close": [bar[3] for bar in bars],
        }
    )


def feature_frame(candles: pd.DataFrame, *, atr: float = 1.0, regime: str = "BULLISH") -> pd.DataFrame:
    """The context of every bar: what the feature row would have carried at that instant."""
    frame = candles[["timestamp", "tradingsymbol", "option_type"]].copy()
    frame["option_atr"] = atr
    frame["nifty_regime"] = regime
    frame["candle_direction"] = "BULLISH"
    frame["time_of_day_bucket"] = "OPEN_915_930"
    frame["expiry_cycle_phase"] = "MID_CYCLE"
    frame["days_to_expiry"] = 2.0
    frame["distance_from_atm"] = -50.0
    frame["event_today_high_state"] = "APPROACHING"
    frame["event_today_low_state"] = "HELD"
    frame["expiry"] = date(2026, 10, 1)
    frame["strike"] = 25000.0
    return frame


def probability_frame(
    candles: pd.DataFrame,
    per_bar: Mapping[str, float] | Sequence[Mapping[str, float]],
) -> pd.DataFrame:
    """A probability frame: one distribution per bar (or one repeated for every bar)."""
    distributions = (
        [dict(per_bar)] * len(candles)
        if isinstance(per_bar, Mapping)
        else [dict(item) for item in per_bar]
    )
    if len(distributions) != len(candles):
        raise AssertionError("one distribution per bar, or one for all of them")
    payload: dict[str, object] = {
        "timestamp": candles["timestamp"].to_numpy(),
        "tradingsymbol": candles["tradingsymbol"].to_numpy(),
    }
    for name in CLASSES:
        payload[f"p_{name}"] = [item.get(name, 0.0) for item in distributions]
    return pd.DataFrame(payload)


#: Short names for the four classes, so a test can write ``UP=0.1``.
ALIASES = {"UP": UP, "DOWN": DOWN, "RANGE": RANGE, "REVERSAL": REVERSAL}


def distribution(**weights: float) -> dict[str, float]:
    """A distribution over the four classes, with the remainder left to ``RANGE``."""
    payload: dict[str, float] = dict.fromkeys(CLASSES, 0.0)
    for name, value in weights.items():
        key = ALIASES.get(str(name).upper(), str(name).upper())
        if key not in payload:
            raise AssertionError(f"unknown class {name!r}")
        payload[key] = float(value)
    total = sum(payload.values())
    if total > 1.0:
        raise AssertionError(f"weights sum to {total}, not at most 1")
    payload[RANGE] += 1.0 - total
    return payload


def fire(class_name: str = UP, *, probability: float = 0.5) -> dict[str, float]:
    """A distribution in which ``class_name`` is the argmax at ``probability``."""
    others = [name for name in CLASSES if name != class_name]
    remainder = (1.0 - probability) / len(others)
    return {class_name: probability, **dict.fromkeys(others, remainder)}



def run_engine(
    candles: pd.DataFrame,
    *,
    features: pd.DataFrame | None = None,
    probabilities: pd.DataFrame | None = None,
    execution: ExecutionPolicy | None = None,
    signal: SignalPolicy | None = None,
    cost_policy: CostPolicy | None = None,
    run_id: str = "test-run",
):
    """Run the engine with permissive defaults (no stop/target/time unless asked for)."""
    default_execution = ExecutionPolicy(
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=None,
        flat_at_session_end=True,
        quantity=75,
        lots=1,
        lot_size=75,
    )
    engine = Backtester(
        cost_model=CostModel(cost_policy or CostPolicy()),
        execution=execution or default_execution,
        signal=signal or SignalPolicy(min_probability=0.4),
        run_id=run_id,
    )
    return engine.run(
        candles=candles,
        features=feature_frame(candles) if features is None else features,
        probabilities=(
            probability_frame(candles, fire()) if probabilities is None else probabilities
        ),
    )


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------
def test_a_signal_fills_at_the_next_bars_open_not_the_signal_bars_close() -> None:
    """The fill price is the *next* bar's open - the one number a look-ahead would break."""
    bars = [
        (100.0, 101.0, 99.0, 100.5),
        (103.0, 104.0, 102.5, 103.5),
        (105.0, 106.0, 104.0, 105.5),
    ]
    candles = candle_frame(bars)
    probabilities = probability_frame(
        candles, [fire(probability=0.6), distribution(UP=0.1), distribution(UP=0.1)]
    )
    result = run_engine(candles, probabilities=probabilities)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(103.0)
    assert trade.entry_timestamp == stamps(3)[1]
    assert trade.exit_reason == EXIT_SESSION_END
    assert trade.exit_price == pytest.approx(105.5)


def test_latency_pushes_the_fill_further_forward() -> None:
    """``latency_ms`` is rounded up to whole bars, because bars are what a dataset has."""
    bars = [
        (100.0, 101.0, 99.0, 100.5),
        (101.0, 102.0, 100.0, 101.5),
        (102.0, 103.0, 101.0, 102.5),
        (103.0, 104.0, 102.0, 103.5),
    ]
    candles = candle_frame(bars)
    execution = ExecutionPolicy(
        latency_ms=61_000,
        latency_bars=2,
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=None,
        flat_at_session_end=True,
        quantity=75,
    )
    result = run_engine(candles, execution=execution)
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(103.0)
    assert result.trades[0].entry_timestamp == stamps(4)[3]
    assert result.trades[0].exit_reason == EXIT_SESSION_END
    # Only bar 0 and the exit bar were *flat* bars; the last one had no fill bar left.
    assert result.counts.decision_bars == 2
    assert result.counts.no_fill_bar == 1


def test_latency_ms_rounds_up_to_whole_bars(config) -> None:
    """A sub-bar latency can never be honoured exactly, so it is rounded *up* and recorded."""
    for milliseconds, expected in ((0, 0), (1, 1), (30_000, 1), (60_000, 1), (61_000, 2)):
        policy = ExecutionPolicy.from_config(config, timeframe="1m", latency_ms=milliseconds)
        assert policy.latency_bars == expected, (milliseconds, policy.describe())
        assert policy.latency_ms == milliseconds


def test_a_signal_on_the_last_bar_has_no_fill_and_is_counted() -> None:
    """Entering at the open of a bar that does not exist is not an option: it is dropped."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (101.0, 102.0, 100.0, 101.5)])
    probabilities = probability_frame(candles, [distribution(UP=0.1), fire(probability=0.6)])
    result = run_engine(candles, probabilities=probabilities)
    assert result.trades == []
    assert result.counts.no_fill_bar == 1
    assert result.counts.decisions == 1


def test_latency_wider_than_the_session_drops_every_signal() -> None:
    """A three-bar latency in a two-bar session has nothing to fill against."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (101.0, 102.0, 100.0, 101.5)])
    execution = ExecutionPolicy(
        latency_bars=3,
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=None,
        quantity=75,
    )
    result = run_engine(candles, execution=execution)
    assert result.trades == []
    assert result.counts.decisions == 2
    assert result.counts.no_fill_bar == 2


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------
def test_a_probability_below_the_threshold_is_counted_and_not_traded() -> None:
    """The signal rule is a threshold, and a miss is a number the report can show."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (101.0, 102.0, 100.0, 101.5)])
    probabilities = probability_frame(candles, [fire(probability=0.35), fire(probability=0.35)])
    result = run_engine(candles, probabilities=probabilities)
    assert result.trades == []
    assert result.counts.decisions == 0
    assert result.counts.no_signal == 2


def test_require_argmax_blocks_a_strong_but_not_winning_directional_class() -> None:
    """A directional class that is not the model's top pick does not trade when asked not to."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (101.0, 102.0, 100.0, 101.5)])
    probabilities = probability_frame(candles, distribution(UP=0.45, REVERSAL=0.5))
    strict = run_engine(candles, probabilities=probabilities)
    loose = run_engine(
        candles,
        probabilities=probabilities,
        signal=SignalPolicy(min_probability=0.4, require_argmax=False),
    )
    assert strict.trades == []
    assert len(loose.trades) == 1
    assert loose.trades[0].ai_prediction == REVERSAL
    assert loose.trades[0].ai_probability == pytest.approx(0.5)
    assert loose.trades[0].entry_reason == f"{ENTRY_SIGNAL}:{UP}"


# ---------------------------------------------------------------------------
# The exits
# ---------------------------------------------------------------------------
def atr_execution(**overrides) -> ExecutionPolicy:
    """The shipped shape of an ATR stop/target run, with the numbers spelled out."""
    payload = {
        "stop_loss_atr": 1.0,
        "target_atr": 1.5,
        "max_holding_minutes": None,
        "flat_at_session_end": True,
        "quantity": 75,
    }
    payload.update(overrides)
    return ExecutionPolicy(**payload)


def test_the_target_exits_at_the_target_price() -> None:
    """The ledger records the target price, not the bar's close."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (100.5, 103.0, 99.6, 102.0)])
    result = run_engine(candles, execution=atr_execution())
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_TARGET
    assert trade.exit_price == pytest.approx(102.0)  # 100.5 + 1.5 * 1.0
    assert trade.net_pnl > 0
    assert result.counts.exit_reasons == {EXIT_TARGET: 1}


def test_the_stop_exits_at_the_stop_price() -> None:
    """The stop is a fill, not a hope: the ledger says 99.5."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (100.5, 100.9, 98.5, 99.0)])
    result = run_engine(candles, execution=atr_execution())
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_STOP_LOSS
    assert trade.exit_price == pytest.approx(99.5)  # 100.5 - 1.0 * 1.0
    assert trade.net_pnl < 0


def test_a_gap_through_the_stop_fills_at_the_open_not_at_the_stop() -> None:
    """A gap is not a favour: the fill is the worse price the bar actually opened at."""
    candles = candle_frame(
        [
            (100.0, 101.0, 99.0, 100.5),
            (100.5, 100.9, 100.0, 100.4),
            (98.0, 99.0, 97.5, 98.5),
        ]
    )
    result = run_engine(candles, execution=atr_execution())
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.exit_reason == EXIT_STOP_LOSS
    assert trade.exit_price == pytest.approx(98.0)
    assert trade.exit_price < 99.5


def test_a_bar_that_touches_both_takes_the_configured_priority_and_is_counted() -> None:
    """Bars cannot say which came first, so the choice is explicit - and counted."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (100.5, 103.0, 98.5, 100.0)])
    stopped = run_engine(candles, execution=atr_execution(intrabar_priority="stop_first"))
    taken = run_engine(candles, execution=atr_execution(intrabar_priority="target_first"))
    assert stopped.trades[0].exit_reason == EXIT_STOP_LOSS
    assert stopped.trades[0].exit_price == pytest.approx(99.5)
    assert taken.trades[0].exit_reason == EXIT_TARGET
    assert taken.trades[0].exit_price == pytest.approx(102.0)
    assert stopped.counts.both_touched == 1
    assert taken.counts.both_touched == 1


def test_the_time_exit_closes_at_the_close_of_the_bar_that_reached_the_limit() -> None:
    """Holding longer than the policy allows is an exit, taken at a price that exists."""
    flat = [(100.0, 100.2, 99.8, 100.0)] * 4
    candles = candle_frame(flat)
    execution = ExecutionPolicy(
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=2,
        flat_at_session_end=False,
        exit_on_opposite_signal=False,
        quantity=75,
    )
    result = run_engine(candles, execution=execution)
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_TIME
    assert trade.exit_timestamp == stamps(4)[2]
    assert trade.holding_minutes == pytest.approx(2.0)


def test_the_opposite_signal_exits_at_the_next_bars_open() -> None:
    """Holding a CE while the model turns bearish closes the position, at the next open."""
    bars = [(100.0, 101.0, 99.0, 100.5), (100.5, 101.0, 100.0, 100.8), (102.0, 102.5, 101.5, 102.2)]
    candles = candle_frame(bars)
    probabilities = probability_frame(
        candles, [fire(UP), fire(DOWN, probability=0.7), distribution(DOWN=0.1)]
    )
    result = run_engine(candles, probabilities=probabilities)
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_SIGNAL
    assert trade.exit_price == pytest.approx(102.0)
    assert trade.exit_timestamp == stamps(3)[2]


def test_a_position_is_flat_at_the_end_of_its_session() -> None:
    """Weekly options are not held overnight in an intraday research backtest."""
    candles = candle_frame([(100.0, 100.2, 99.8, 100.0)] * 3)
    result = run_engine(candles)
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_SESSION_END
    assert trade.exit_price == pytest.approx(100.0)
    assert trade.exit_timestamp == stamps(3)[-1]


def test_the_data_end_closes_a_position_the_session_rule_did_not() -> None:
    """With session-end flatness off, the ledger still closes: nothing is left open."""
    candles = candle_frame([(100.0, 100.2, 99.8, 100.0)] * 3)
    execution = ExecutionPolicy(
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=None,
        flat_at_session_end=False,
        exit_on_opposite_signal=False,
        quantity=75,
    )
    result = run_engine(candles, execution=execution)
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_DATA_END
    assert trade.exit_timestamp is not None
    assert result.counts.exit_reasons == {EXIT_DATA_END: 1}


def test_a_position_cannot_be_opened_while_another_is_held() -> None:
    """One position at a time per contract; a re-entry waits for a later bar."""
    bars = [
        (100.0, 101.0, 99.0, 100.5),
        (100.5, 101.0, 100.0, 100.8),
        (101.0, 101.5, 100.5, 101.2),
        (102.0, 103.0, 101.5, 102.5),
    ]
    candles = candle_frame(bars)
    probabilities = probability_frame(
        candles, [fire(UP), fire(DOWN, probability=0.7), fire(UP), distribution(UP=0.1)]
    )
    result = run_engine(candles, probabilities=probabilities)
    assert [trade.entry_timestamp for trade in result.trades] == [stamps(4)[1], stamps(4)[3]]
    assert [trade.exit_reason for trade in result.trades] == [EXIT_SIGNAL, EXIT_SESSION_END]
    assert result.counts.entries == 2


# ---------------------------------------------------------------------------
# The ledger (spec 23's trade record)
# ---------------------------------------------------------------------------
LEDGER_CANDLES = [
    (100.0, 101.0, 99.0, 100.5),
    (100.5, 102.5, 100.2, 102.0),
    (102.0, 102.8, 101.5, 102.4),
]


def settle_one_trade(**kwargs):
    """One trade from :data:`LEDGER_CANDLES`, for the ledger-shape assertions."""
    result = run_engine(candle_frame(LEDGER_CANDLES), **kwargs)
    assert len(result.trades) == 1
    return result, result.trades[0]


def test_the_ledger_row_is_exactly_the_stored_trades_contract() -> None:
    """The stored columns *and* the research columns, in the order the schema defines."""
    result, trade = settle_one_trade()
    assert tuple(trade.to_record()) == TRADE_COLUMNS
    assert tuple(trade.to_row()) == LEDGER_COLUMNS
    assert len(LEDGER_COLUMNS) == len(TRADE_COLUMNS) + 3
    assert trade.trade_id.startswith("test-run|")
    assert trade.run_id == "test-run"
    assert trade.tradingsymbol == SYMBOL
    assert trade.option_type == "CE"
    assert trade.quantity == 75
    assert result.trades[0].exit_reason == EXIT_SESSION_END


def test_the_ledger_row_carries_the_signal_bars_context() -> None:
    """No look-ahead in the context either: the *signal* bar's state, not the exit bar's."""
    _, trade = settle_one_trade()
    assert trade.nifty_state == "BULLISH"
    assert trade.candle_state == "BULLISH"
    assert trade.support_resistance_state == "TODAY_HIGH=APPROACHING|TODAY_LOW=HELD"
    assert trade.time_of_day_bucket == "OPEN_915_930"
    assert trade.expiry_cycle_phase == "MID_CYCLE"
    assert trade.days_to_expiry == pytest.approx(2.0)
    assert trade.distance_from_atm == pytest.approx(-50.0)
    assert trade.expiry == date(2026, 10, 1)
    assert trade.strike == pytest.approx(25000.0)
    assert trade.day_of_week == 0  # Monday 2026-09-28
    assert trade.ai_prediction == UP
    assert trade.ai_probability == pytest.approx(0.5)
    assert trade.entry_reason == f"{ENTRY_SIGNAL}:{UP}"


def test_a_losing_trade_is_in_the_ledger_with_its_costs() -> None:
    """The whole point of the ledger: a loss is a row, with the money it actually cost."""
    falling = [
        (100.0, 100.5, 99.0, 99.5),
        (99.5, 99.8, 97.0, 97.5),
        (97.5, 98.0, 95.0, 95.5),
    ]
    candles = candle_frame(falling, symbol=PE_SYMBOL, option_type="PE")
    probabilities = probability_frame(candles, fire(DOWN, probability=0.6))
    result = run_engine(candles, probabilities=probabilities)
    trade = result.trades[0]
    expected_costs = CostModel(CostPolicy()).round_trip(
        entry_price=trade.entry_price, exit_price=trade.exit_price, quantity=75
    )
    assert trade.exit_reason == EXIT_SESSION_END
    assert trade.gross_pnl == pytest.approx((trade.exit_price - trade.entry_price) * 75)
    assert trade.costs == pytest.approx(expected_costs.total)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.costs)
    assert trade.net_pnl < 0
    assert trade.mae < 0
    assert result.net_pnl == pytest.approx(trade.net_pnl)
    assert result.total_costs == pytest.approx(trade.costs)
    assert result.counts.exit_reasons == {EXIT_SESSION_END: 1}


def test_mfe_and_mae_span_the_bars_the_position_was_held() -> None:
    """The excursion is what the bars showed, measured from the entry fill."""
    bars = [
        (100.0, 100.4, 99.6, 100.0),
        (100.0, 103.4, 98.6, 101.0),
        (101.0, 101.5, 100.5, 101.2),
    ]
    candles = candle_frame(bars)
    execution = ExecutionPolicy(
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=1.0,
        flat_at_session_end=False,
        exit_on_opposite_signal=False,
        quantity=75,
    )
    result = run_engine(candles, execution=execution)
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_TIME
    assert trade.mfe == pytest.approx(3.4)
    assert trade.mae == pytest.approx(-1.4)


def test_a_second_run_of_the_same_inputs_produces_the_same_ledger() -> None:
    """The engine is a replay, not a simulation: same bars, same predictions, same ledger."""
    candles = candle_frame(LEDGER_CANDLES)
    first = run_engine(candles)
    second = run_engine(candles)
    assert [trade.to_row() for trade in first.trades] == [
        trade.to_row() for trade in second.trades
    ]
    assert first.counts.to_dict() == second.counts.to_dict()


def test_exposure_is_the_share_of_bars_spent_holding() -> None:
    """Spec 25's exposure, counted from the bars rather than guessed from the trades."""
    candles = candle_frame(LEDGER_CANDLES)
    execution = ExecutionPolicy(
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=1.0,
        flat_at_session_end=False,
        exit_on_opposite_signal=False,
        quantity=75,
    )
    result = run_engine(candles, execution=execution)
    assert result.counts.bars == 3
    # Two one-bar positions (entered at bar 1, then at bar 2), so two of three bars held.
    assert result.counts.entries == 2
    assert result.counts.in_position_bars == 2
    assert result.counts.exposure == pytest.approx(2 / 3)
    assert result.counts.no_fill_bar == 1


def test_trading_days_uses_the_ist_session_not_the_utc_one() -> None:
    """Two IST sessions, one of which would be a different day in UTC."""
    first = candle_frame([(100.0, 101.0, 99.0, 100.5)] * 2)
    second = candle_frame([(100.0, 101.0, 99.0, 100.5)] * 2, day=date(2026, 9, 30))
    combined = pd.concat([first, second], ignore_index=True)
    assert trading_days(combined) == [date(2026, 9, 28), date(2026, 9, 30)]



# ---------------------------------------------------------------------------
# The fill mode and the spread
# ---------------------------------------------------------------------------
def bid_ask_execution() -> ExecutionPolicy:
    """A ``bid_ask`` fill policy with the ATR exits off (one clean trade per session)."""
    return ExecutionPolicy(
        fill_at="bid_ask",
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=None,
        flat_at_session_end=True,
        quantity=75,
    )


def test_bid_ask_mode_charges_the_captured_spread_on_both_legs() -> None:
    """With a spread available and the policy on, crossing it is charged - and visible."""
    candles = candle_frame(LEDGER_CANDLES)
    features = feature_frame(candles)
    features["bid_ask_spread"] = 2.0
    result = run_engine(
        candles,
        features=features,
        execution=bid_ask_execution(),
        cost_policy=CostPolicy(use_captured_spread=True),
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_reason == f"{ENTRY_SIGNAL}:{UP}:SPREAD"
    assert trade.bid_ask_spread == pytest.approx(2.0)
    assert trade.costs > 0
    assert result.counts.spread_fills == 2  # one order in, one out
    assert result.counts.spread_missing == 0


def test_bid_ask_mode_without_a_stored_spread_falls_back_and_counts_it() -> None:
    """A fill mode that wants a spread it cannot have says so, instead of inventing one."""
    candles = candle_frame(LEDGER_CANDLES)
    result = run_engine(
        candles,
        execution=bid_ask_execution(),
        cost_policy=CostPolicy(use_captured_spread=True),
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.bid_ask_spread is None
    assert trade.entry_reason == f"{ENTRY_SIGNAL}:{UP}"
    assert result.counts.spread_fills == 0
    assert result.counts.spread_missing == 2


# ---------------------------------------------------------------------------
# Refusals: inputs that cannot produce an honest ledger
# ---------------------------------------------------------------------------
def engine_for_refusals() -> Backtester:
    """An engine with the cheapest possible policy, for the input-contract tests."""
    return Backtester(
        cost_model=CostModel(CostPolicy()),
        execution=ExecutionPolicy(stop_loss_atr=None, target_atr=None, quantity=1),
        signal=SignalPolicy(),
        run_id="test-run",
    )


def test_a_frame_without_the_columns_the_engine_needs_is_refused() -> None:
    """Each input frame has a contract, and a missing column is a refusal with its name."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (100.5, 101.0, 100.0, 100.8)])
    features = feature_frame(candles)
    probabilities = probability_frame(candles, fire())
    engine = engine_for_refusals()
    with pytest.raises(BacktestError, match="missing column"):
        engine.run(
            candles=candles.drop(columns="high"), features=features, probabilities=probabilities
        )
    with pytest.raises(BacktestError, match="missing column"):
        engine.run(
            candles=candles,
            features=features.drop(columns="timestamp"),
            probabilities=probabilities,
        )
    with pytest.raises(BacktestError, match="no p_"):
        engine.run(
            candles=candles,
            features=features,
            probabilities=probabilities.drop(columns=[f"p_{name}" for name in CLASSES]),
        )


def test_no_option_type_anywhere_is_refused() -> None:
    """Without a CE/PE identity there is no side to take, so the run refuses."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (100.5, 101.0, 100.0, 100.8)])
    features = feature_frame(candles).drop(columns="option_type")
    probabilities = probability_frame(candles, fire())
    with pytest.raises(BacktestError, match="option_type"):
        run_engine(
            candles.drop(columns="option_type"), features=features, probabilities=probabilities
        )


def test_an_atr_exit_without_an_atr_column_is_refused() -> None:
    """A stop that cannot be placed is not a stop: the run says which column is missing."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (100.5, 101.0, 100.0, 100.8)])
    features = feature_frame(candles).drop(columns="option_atr")
    probabilities = probability_frame(candles, fire())
    with pytest.raises(BacktestError, match="option_atr"):
        run_engine(
            candles,
            features=features,
            probabilities=probabilities,
            execution=atr_execution(),
        )


def test_a_non_positive_price_is_refused() -> None:
    """A premium that is not a positive number is a data defect, not a free option."""
    candles = candle_frame([(100.0, 101.0, 99.0, 100.5), (0.0, 1.0, 0.0, 0.5)])
    with pytest.raises(BacktestError, match="non-positive or non-finite"):
        run_engine(candles)


def test_empty_candles_are_refused() -> None:
    """Nothing to replay is a refusal, not a zero-trade success."""
    candles = pd.DataFrame(
        columns=["timestamp", "tradingsymbol", "option_type", "open", "high", "low", "close"]
    )
    with pytest.raises(BacktestError, match="no candles"):
        run_engine(candles)



@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "always_long"},
        {"min_probability": 0.0},
        {"min_probability": 1.5},
        {"directions": {}},
        {"directions": {"CE": "   "}},
    ],
)
def test_a_signal_policy_that_could_never_fire_is_refused(payload: dict[str, object]) -> None:
    """A rule nobody can satisfy must fail at construction, not silently never trade."""
    with pytest.raises(BacktestError):
        SignalPolicy(**payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"signal_at": "candle_open"},
        {"fill_at": "close"},
        {"intrabar_priority": "either"},
        {"quantity": 0},
        {"latency_bars": -1},
        {"stop_loss_atr": 0.0},
        {"target_atr": -1.0},
        {"max_holding_minutes": 0.0},
    ],
)
def test_an_execution_policy_that_cannot_be_honoured_is_refused(
    payload: dict[str, object],
) -> None:
    """Every rule the engine implements has a policy that refuses to be nonsense."""
    with pytest.raises(BacktestError):
        ExecutionPolicy(**payload)


def test_the_signal_policy_decides_only_on_the_directional_class() -> None:
    """``decide`` is a pure function of the distribution, the side and the threshold."""
    policy = SignalPolicy(min_probability=0.4)
    assert policy.direction("CE") == UP
    assert policy.direction("pe") == DOWN
    assert policy.direction("XX") is None
    assert policy.decide(option_type="CE", probabilities=fire(UP, probability=0.6)) is not None
    assert policy.decide(option_type="CE", probabilities=fire(DOWN, probability=0.9)) is None
    assert policy.decide(option_type=None, probabilities=fire(UP, probability=0.9)) is None
    decision = policy.decide(option_type="PE", probabilities=fire(DOWN, probability=0.7))
    assert decision is not None
    assert decision.reason == f"{ENTRY_SIGNAL}:{DOWN}"
    assert decision.margin == pytest.approx(0.7 - 0.1)
    assert decision.to_dict()["option_type"] == "PE"
    assert "threshold rule" in policy.describe()["note"]
    assert policy.describe()["directions"] == {"CE": UP, "PE": DOWN}


def test_the_signal_policy_can_be_built_from_a_loaded_configuration(config) -> None:
    """The shipped configuration is a working policy, and an override wins over it."""
    shipped = SignalPolicy.from_config(config)
    assert shipped.min_probability == pytest.approx(
        config.model.backtest.signal.min_probability
    )
    assert shipped.direction("CE") == config.model.backtest.signal.directions["CE"]
    overridden = SignalPolicy.from_config(config, min_probability=0.55)
    assert overridden.min_probability == pytest.approx(0.55)


def test_the_execution_policy_can_be_built_from_a_loaded_configuration(config) -> None:
    """Sizing, exits and the fill mode all come from ``model.yaml``."""
    policy = ExecutionPolicy.from_config(config, timeframe="1m")
    backtest = config.model.backtest
    assert policy.quantity == backtest.sizing.lots * backtest.sizing.lot_size
    assert policy.stop_loss_atr == pytest.approx(backtest.exit.stop_loss_atr)
    assert policy.target_atr == pytest.approx(backtest.exit.target_atr)
    assert policy.max_holding_minutes == pytest.approx(backtest.exit.max_holding_minutes)
    assert policy.flat_at_session_end is backtest.exit.flat_at_session_end
    assert policy.describe()["timeframe"] == "1m"

