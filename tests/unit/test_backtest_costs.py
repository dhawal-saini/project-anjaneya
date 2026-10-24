"""Unit tests for the Indian F&O cost model (``core/backtest/costs.py``, spec 23).

A cost model is where a backtest quietly stops being a market and starts being a fantasy, so
these tests are arithmetic that can be checked with a pen:

* a **buy** leg pays no STT and does pay stamp duty; a **sell** leg the other way round;
* GST sits on (brokerage + exchange + SEBI) - never on the premium, never on STT or stamp
  duty, which is the difference between a correct contract note and a plausible one;
* slippage is a *cost*, and the fill price stays the market's price (so a reader can subtract
  the friction and see the frictionless number);
* a captured spread, when used, replaces slippage - and when the policy forbids it, it is
  ignored *and counted* rather than silently charged;
* a rate that cannot be a rate (a negative charge) is refused.

Nothing here touches the network, the database or a file.
"""

from __future__ import annotations

import pytest

from core.backtest import (
    COST_MODEL_CAVEAT,
    SIDE_BUY,
    SIDE_SELL,
    CostError,
    CostModel,
    CostPolicy,
)

#: The numbers the expected values below were computed from, by hand.
PRICE = 100.0
EXIT_PRICE = 110.0
QUANTITY = 75


@pytest.fixture()
def policy() -> CostPolicy:
    """The shipped cost policy (every rate is in ``config/model.yaml``)."""
    return CostPolicy()


@pytest.fixture()
def model(policy: CostPolicy) -> CostModel:
    return CostModel(policy)


# ---------------------------------------------------------------------------
# One order, charge by charge
# ---------------------------------------------------------------------------
def test_a_buy_leg_pays_no_stt_and_does_pay_stamp_duty(
    model: CostModel, policy: CostPolicy
) -> None:
    """STT is a sell-side charge on premium; stamp duty is a buy-side one."""
    cost = model.order(side=SIDE_BUY, price=PRICE, quantity=QUANTITY)
    turnover = PRICE * QUANTITY
    assert cost.turnover == pytest.approx(turnover)
    assert cost.stt == 0.0
    assert cost.stamp_duty == pytest.approx(turnover * policy.stamp_duty_rate)
    assert cost.exchange_txn == pytest.approx(turnover * policy.exchange_txn_rate)
    assert cost.sebi == pytest.approx(turnover * policy.sebi_rate)
    assert cost.brokerage == pytest.approx(policy.brokerage_per_order)
    assert cost.slippage == pytest.approx(turnover * policy.slippage_bps / 10_000)
    assert cost.total == pytest.approx(30.684005)
    assert cost.statutory == pytest.approx(cost.total - cost.brokerage - cost.slippage)


def test_a_sell_leg_pays_stt_and_no_stamp_duty(model: CostModel, policy: CostPolicy) -> None:
    """The mirror image of the buy leg, at the exit price's turnover."""
    cost = model.order(side=SIDE_SELL, price=EXIT_PRICE, quantity=QUANTITY)
    turnover = EXIT_PRICE * QUANTITY
    assert cost.stamp_duty == 0.0
    assert cost.stt == pytest.approx(turnover * policy.stt_rate)
    assert cost.total == pytest.approx(36.301156)
    assert cost.effective_price == pytest.approx(109.945)


def test_gst_sits_on_the_charges_and_not_on_the_premium(
    model: CostModel, policy: CostPolicy
) -> None:
    """GST is 18% of (brokerage + exchange + SEBI) - the most commonly mis-stated charge."""
    cost = model.order(side=SIDE_SELL, price=EXIT_PRICE, quantity=QUANTITY)
    expected = (cost.brokerage + cost.exchange_txn + cost.sebi) * policy.gst_rate
    assert cost.gst == pytest.approx(expected)
    assert cost.gst != pytest.approx(cost.turnover * policy.gst_rate)
    assert cost.gst != pytest.approx(cost.statutory * policy.gst_rate)



# ---------------------------------------------------------------------------
# Friction is a cost, not a price
# ---------------------------------------------------------------------------
def test_slippage_is_charged_as_a_cost_and_the_price_stays_the_markets(
    model: CostModel, policy: CostPolicy
) -> None:
    """``price`` is untouched; ``effective_price`` is what the account paid per unit."""
    buy = model.order(side=SIDE_BUY, price=PRICE, quantity=QUANTITY)
    sell = model.order(side=SIDE_SELL, price=PRICE, quantity=QUANTITY)
    per_unit = PRICE * policy.slippage_bps / 10_000
    assert buy.price == PRICE and sell.price == PRICE
    assert buy.effective_price == pytest.approx(PRICE + per_unit)
    assert sell.effective_price == pytest.approx(PRICE - per_unit)


def test_a_round_trip_totals_both_legs(model: CostModel) -> None:
    """The pair is charged on each leg's own turnover."""
    costs = model.round_trip(entry_price=PRICE, exit_price=EXIT_PRICE, quantity=QUANTITY)
    assert costs.total == pytest.approx(costs.entry.total + costs.exit.total)
    assert costs.total == pytest.approx(66.985161)
    assert costs.turnover == pytest.approx((PRICE + EXIT_PRICE) * QUANTITY)
    assert costs.per_unit == pytest.approx(costs.total / QUANTITY)
    assert costs.to_dict()["total"] == pytest.approx(costs.total)


# ---------------------------------------------------------------------------
# The captured spread
# ---------------------------------------------------------------------------
def test_a_captured_spread_replaces_slippage_when_the_policy_allows_it() -> None:
    """Crossing half the spread is charged once, and slippage stands down."""
    model = CostModel(CostPolicy(use_captured_spread=True))
    cost = model.order(side=SIDE_BUY, price=PRICE, quantity=QUANTITY, spread=2.0)
    assert cost.spread == pytest.approx(2.0 / 2 * QUANTITY)
    assert cost.slippage == 0.0
    assert cost.effective_price == pytest.approx(PRICE + 1.0)
    assert model.spread_orders == 1
    assert model.ignored_spreads == 0


def test_a_captured_spread_is_ignored_and_counted_when_the_policy_forbids_it(
    model: CostModel,
) -> None:
    """A spread the run was told not to use is not charged - and not hidden either."""
    cost = model.order(side=SIDE_BUY, price=PRICE, quantity=QUANTITY, spread=2.0)
    assert cost.spread == 0.0
    assert cost.slippage > 0.0
    assert model.spread_orders == 0
    assert model.ignored_spreads == 1


def test_a_spread_that_is_not_positive_is_refused(model: CostModel) -> None:
    """A zero or negative spread is a data error, not a free fill."""
    with pytest.raises(CostError, match="spread must be positive"):
        model.order(side=SIDE_BUY, price=PRICE, quantity=QUANTITY, spread=0.0)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"stt_rate": -0.001},
        {"exchange_txn_rate": -0.1},
        {"sebi_rate": -0.1},
        {"stamp_duty_rate": -0.1},
        {"slippage_bps": -1.0},
        {"brokerage_per_order": -20.0},
        {"gst_rate": 1.5},
    ],
)
def test_a_rate_that_cannot_be_a_rate_is_refused(payload: dict[str, float]) -> None:
    """A negative charge would pay the trader; a 150% GST is not a GST."""
    with pytest.raises(CostError):
        CostPolicy(**payload)


def test_an_order_without_a_side_or_a_quantity_is_refused(model: CostModel) -> None:
    """The numbers an order cannot invent."""
    with pytest.raises(CostError, match="unknown side"):
        model.order(side="HOLD", price=PRICE, quantity=QUANTITY)
    with pytest.raises(CostError, match="quantity must be a positive"):
        model.order(side=SIDE_BUY, price=PRICE, quantity=0)
    with pytest.raises(CostError, match="price cannot be negative"):
        model.order(side=SIDE_BUY, price=-1.0, quantity=QUANTITY)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_describe_states_every_rate_and_the_caveat(model: CostModel) -> None:
    """Every number a report quotes is in the payload, with the sentence that qualifies it."""
    described = model.describe()
    for name in (
        "brokerage_per_order",
        "stt_rate",
        "exchange_txn_rate",
        "sebi_rate",
        "gst_rate",
        "stamp_duty_rate",
        "slippage_bps",
        "use_captured_spread",
        "orders_with_spread",
        "spreads_ignored",
    ):
        assert name in described
    assert described["caveat"] == COST_MODEL_CAVEAT
    assert "verify" in described["caveat"] and "P&L" in described["caveat"]


def test_the_policy_can_be_built_from_a_loaded_configuration(config) -> None:
    """``CostPolicy.from_config`` reads ``model.costs`` and nothing else."""
    policy = CostPolicy.from_config(config)
    assert policy.brokerage_per_order == pytest.approx(config.model.costs.brokerage_per_order)
    assert policy.stt_rate == pytest.approx(config.model.costs.stt_rate)
    assert policy.use_captured_spread is config.model.costs.use_captured_spread
