"""Indian F&O transaction cost model (spec 23).

Every number a backtest reports is *after* this module has taken its cut, so the cuts have
to be visible. The model charges, in the order a broker's contract note prints them:

============================  =========================================================
brokerage                     flat, per *order* (so a round trip pays it twice)
STT                           on the **sell** side only, on premium (options are cash
                              settled; a bought option that expires worthless pays none)
exchange transaction charge   on premium, both sides
SEBI turnover fee             on premium, both sides
GST                           18% on (brokerage + exchange txn + SEBI) - never on STT,
                              stamp duty or the premium itself
stamp duty                    on the **buy** side only
slippage                      an adverse move in the fill price, ``slippage_bps`` of the
                              traded premium, charged on both sides
bid/ask spread                *optionally*, half the captured spread per side, when the
                              run enables ``costs.use_captured_spread``
============================  =========================================================

Three deliberate decisions, each of which a reader can disagree with but not be misled by:

* **The fill price stays the market's own price.** Friction is charged as a *cost*, never
  baked into ``entry_price``/``exit_price``. One place therefore holds every rupee of
  friction, so a reader can subtract it and see the frictionless number;
  ``OrderCost.effective_price`` still shows what the trader's account paid per unit.
* **Slippage and spread never double-count.** Slippage is the always-on model of a market
  order. A captured spread is charged *instead* of slippage on that side (that is what
  crossing the spread means); the counters say which happened, side by side.
* **The rates are configuration, not truth.** Statutory rates and brokerage plans change.
  :data:`COST_MODEL_CAVEAT` is printed by every backtest and stored in its summary, so no
  number in a report can be quoted without the sentence that qualifies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "COST_MODEL_CAVEAT",
    "SIDES",
    "SIDE_BUY",
    "SIDE_SELL",
    "CostError",
    "CostModel",
    "CostPolicy",
    "OrderCost",
    "TradeCosts",
]

#: Printed by every backtest run and stored in its summary. Rates are inputs, not facts.
COST_MODEL_CAVEAT = (
    "cost rates come from model.costs (config, not code): verify brokerage, STT, exchange, "
    "SEBI, GST and stamp-duty rates against the current contract note before quoting any P&L"
)

#: The two sides of an order. ``BUY`` opens a long premium position, ``SELL`` closes it.
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"
SIDES: tuple[str, str] = (SIDE_BUY, SIDE_SELL)


class CostError(RuntimeError):
    """Raised when a cost cannot be computed from the values it was handed."""


@dataclass(frozen=True, slots=True)
class CostPolicy:
    """The Indian F&O cost model's numbers (``config/model.yaml`` ``costs``).

    Every rate is a fraction of *premium turnover* except ``gst_rate`` - a fraction of the
    charges it applies to - and ``slippage_bps``, a fraction of the traded price
    (1 bps = 0.01%).
    """

    brokerage_per_order: float = 20.0
    stt_rate: float = 0.000625
    exchange_txn_rate: float = 0.0003503
    sebi_rate: float = 0.000001
    gst_rate: float = 0.18
    stamp_duty_rate: float = 0.00003
    slippage_bps: float = 5.0
    use_captured_spread: bool = False

    def __post_init__(self) -> None:
        """Refuse a rate that cannot be a rate (a negative charge would pay the trader)."""
        fractions = {
            "stt_rate": self.stt_rate,
            "exchange_txn_rate": self.exchange_txn_rate,
            "sebi_rate": self.sebi_rate,
            "gst_rate": self.gst_rate,
            "stamp_duty_rate": self.stamp_duty_rate,
        }
        negative = [name for name, value in fractions.items() if value < 0]
        if negative:
            raise CostError(f"{negative} cannot be negative: a charge is never a credit")
        if self.gst_rate > 1:
            raise CostError(f"gst_rate {self.gst_rate} is not a fraction of the charges")
        if self.brokerage_per_order < 0:
            raise CostError("brokerage_per_order cannot be negative")
        if self.slippage_bps < 0:
            raise CostError("slippage_bps cannot be negative")

    @classmethod
    def from_config(cls, config: Any) -> CostPolicy:
        """Build the policy from a loaded :class:`core.config_loader.AnjaneyaConfig`."""
        costs = config.model.costs
        return cls(
            brokerage_per_order=float(costs.brokerage_per_order),
            stt_rate=float(costs.stt_rate),
            exchange_txn_rate=float(costs.exchange_txn_rate),
            sebi_rate=float(costs.sebi_rate),
            gst_rate=float(costs.gst_rate),
            stamp_duty_rate=float(costs.stamp_duty_rate),
            slippage_bps=float(costs.slippage_bps),
            use_captured_spread=bool(costs.use_captured_spread),
        )

    @property
    def slippage_fraction(self) -> float:
        """``slippage_bps`` as a fraction of price."""
        return self.slippage_bps / 10_000.0

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every backtest summary."""
        return {
            "brokerage_per_order": self.brokerage_per_order,
            "stt_rate": self.stt_rate,
            "exchange_txn_rate": self.exchange_txn_rate,
            "sebi_rate": self.sebi_rate,
            "gst_rate": self.gst_rate,
            "stamp_duty_rate": self.stamp_duty_rate,
            "slippage_bps": self.slippage_bps,
            "slippage_fraction": self.slippage_fraction,
            "use_captured_spread": self.use_captured_spread,
            "caveat": COST_MODEL_CAVEAT,
        }


@dataclass(frozen=True, slots=True)
class OrderCost:
    """What one order cost, charge by charge, plus the price the account really paid.

    ``total`` is the rupee cost of this side alone. ``effective_price`` is the price a
    trader would see in their fill log: a buy pays ``price + slippage (or half-spread)``
    per unit, a sell receives ``price - slippage (or half-spread)``.
    """

    side: str
    price: float
    quantity: int
    turnover: float
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    gst: float
    stamp_duty: float
    slippage: float
    spread: float
    total: float
    effective_price: float

    @property
    def statutory(self) -> float:
        """STT + exchange + SEBI + GST + stamp duty (everything that is not brokerage/crossing)."""
        return self.stt + self.exchange_txn + self.sebi + self.gst + self.stamp_duty

    @property
    def friction(self) -> float:
        """The price-impact part of the cost (slippage or the captured half-spread)."""
        return self.slippage + self.spread

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape, every charge named."""
        return {
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "turnover": self.turnover,
            "brokerage": self.brokerage,
            "stt": self.stt,
            "exchange_txn": self.exchange_txn,
            "sebi": self.sebi,
            "gst": self.gst,
            "stamp_duty": self.stamp_duty,
            "slippage": self.slippage,
            "spread": self.spread,
            "statutory": self.statutory,
            "friction": self.friction,
            "total": self.total,
            "effective_price": self.effective_price,
        }


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """Both legs of one trade: what it cost to get in, and what it cost to get out."""

    entry: OrderCost
    exit: OrderCost

    @property
    def total(self) -> float:
        """The round trip's total cost in rupees."""
        return self.entry.total + self.exit.total

    @property
    def turnover(self) -> float:
        """Premium turnover of both legs (the base every percentage charge is applied to)."""
        return self.entry.turnover + self.exit.turnover

    @property
    def per_unit(self) -> float:
        """Total cost per unit of premium (the same number, expressed per unit)."""
        quantity = max(self.entry.quantity, 1)
        return self.total / quantity

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "entry": self.entry.to_dict(),
            "exit": self.exit.to_dict(),
            "total": self.total,
            "turnover": self.turnover,
            "per_unit": self.per_unit,
        }


class CostModel:
    """Turns a simulated fill into a contract note (spec 23).

    The model is *counting* as well as charging: it records how often a captured spread was
    used and how often one was present but the policy forbade using it
    (``costs.use_captured_spread: false``). Whether a *fill* wanted a spread it did not get
    is the engine's counter (``BacktestCounts.spread_missing``), not the contract note's -
    the model charges what it is handed and never pretends an absent spread was there.
    """

    def __init__(self, policy: CostPolicy) -> None:
        self.policy = policy
        #: Orders charged through a captured spread.
        self.spread_orders = 0
        #: Orders where a spread was handed over but the policy says not to use it.
        self.ignored_spreads = 0

    # -- one order -----------------------------------------------------------
    def order(
        self,
        *,
        side: str,
        price: float,
        quantity: int,
        spread: float | None = None,
    ) -> OrderCost:
        """The full contract note for one order.

        ``spread`` is the *captured* bid/ask spread (ask minus bid) in price units, when the
        caller has one. It is charged as half the spread per unit - the cost of crossing -
        and only when :attr:`CostPolicy.use_captured_spread` is set.
        """
        if side not in SIDES:
            raise CostError(f"unknown side {side!r}; expected one of {list(SIDES)}")
        if quantity <= 0:
            raise CostError(f"quantity must be a positive number of units, got {quantity}")
        if price < 0:
            raise CostError(f"price cannot be negative, got {price}")

        policy = self.policy
        units = int(quantity)
        turnover = float(price) * units
        crossing = self._crossing_cost(spread=spread) * units

        brokerage = float(policy.brokerage_per_order)
        stt = turnover * policy.stt_rate if side == SIDE_SELL else 0.0
        exchange_txn = turnover * policy.exchange_txn_rate
        sebi = turnover * policy.sebi_rate
        gst = (brokerage + exchange_txn + sebi) * policy.gst_rate
        stamp_duty = turnover * policy.stamp_duty_rate if side == SIDE_BUY else 0.0
        # Slippage and the spread are alternatives on a given side: both model the same
        # crossing, so charging both would bill the trader twice for one inconvenience.
        if crossing > 0.0:
            slippage, spread_cost = 0.0, crossing
        else:
            slippage, spread_cost = turnover * policy.slippage_fraction, 0.0

        friction_per_unit = (slippage + spread_cost) / units
        effective = (
            float(price) + friction_per_unit if side == SIDE_BUY else float(price) - friction_per_unit
        )
        return OrderCost(
            side=side,
            price=float(price),
            quantity=units,
            turnover=turnover,
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn,
            sebi=sebi,
            gst=gst,
            stamp_duty=stamp_duty,
            slippage=slippage,
            spread=spread_cost,
            total=brokerage + stt + exchange_txn + sebi + gst + stamp_duty + slippage + spread_cost,
            effective_price=effective,
        )

    def _crossing_cost(self, *, spread: float | None) -> float:
        """Half the captured spread per unit, or ``0.0`` with the reason counted."""
        if spread is None:
            return 0.0
        if spread <= 0.0:
            raise CostError(f"a captured spread must be positive, got {spread}")
        if not self.policy.use_captured_spread:
            self.ignored_spreads += 1
            return 0.0
        self.spread_orders += 1
        return float(spread) / 2.0

    # -- one trade -----------------------------------------------------------
    def round_trip(
        self,
        *,
        entry_price: float,
        exit_price: float,
        quantity: int,
        entry_spread: float | None = None,
        exit_spread: float | None = None,
    ) -> TradeCosts:
        """Buy to open, sell to close: the two orders a long-premium backtest ever makes."""
        entry = self.order(side=SIDE_BUY, price=entry_price, quantity=quantity, spread=entry_spread)
        exit_ = self.order(side=SIDE_SELL, price=exit_price, quantity=quantity, spread=exit_spread)
        return TradeCosts(entry=entry, exit=exit_)

    def describe(self) -> dict[str, Any]:
        """The policy plus what the model actually did with the orders it was handed."""
        payload = self.policy.describe()
        payload.update(
            {
                "orders_with_spread": self.spread_orders,
                "spreads_ignored": self.ignored_spreads,
            }
        )
        return payload

