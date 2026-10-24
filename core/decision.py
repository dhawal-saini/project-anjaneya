"""Master-prompt phase 24 - the decision layer (spec 24), as a pure policy object.

The master prompt ends its phase list with::

    PHASE 24 - Decision Layer (later, separate)
      Only after features, model and backtest are validated.

and it says the same thing twice more in the configuration it ships: "ANJANEYA never predicts
BUY/SELL. It predicts future behaviour over multiple horizons; **the decision layer (Phase 24)
is separate**". Until now every artefact that could be mistaken for advice has said so out loud -
``core.backtest.engine.SignalPolicy`` is "the smallest honest thing that lets a backtest measure
costs", and :data:`core.ml.explainer.DECISION_LAYER_CAVEAT` travels with every explanation.

This module is that separate piece. It answers one question, per scored bar: *given this
distribution and this policy, what would the policy do?* - and it answers it as a
:class:`Decision` with a **reason code**, never as prose, and never as a position:

* **it is a policy, not a model.** The probabilities come from one stored model artefact; the
  rule that turns them into ENTER or NO_TRADE is configuration (`model.decision`), versioned
  and hashed like every other policy in the project. Changing the rule changes the record, and
  the record says which rule made it.
* **it is a decision, not an edge.** :data:`DECISION_CAVEATS` travels in every artefact, the
  same way :data:`core.accumulation.ACCUMULATION_CAVEATS` does; a policy that fires on 90% of
  bars is a busy rule, not a discovery.
* **it states an exit plan; it does not simulate one.** The stop, the target, the maximum
  holding and the session-end flattening are *echoed* from the policy so a reader knows what
  the decision intended. Replaying that plan against stored prices - charging costs, counting
  winners - is ``--mode backtest``'s job, and this module deliberately does not do it twice,
  because two engines that can disagree about one trade is exactly the failure this project
  keeps designing out.
* **it never places an order.** There is no broker here, no order object and no side effect
  beyond files under ``data/predictions/``; the mode is read-only against DuckDB.

The reason vocabulary is closed and ordered (:data:`DECISION_REASONS`), so "why is this row a
NO_TRADE?" is a value a test can assert rather than a sentence a reader has to parse.

Importing this module performs no I/O. It imports nothing heavier than ``core.gates``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from core.gates import GateStatus

__all__ = [
    "DECISION_CAVEATS",
    "DECISION_COLUMNS",
    "DECISION_REASONS",
    "ENTER_CE",
    "ENTER_PE",
    "NO_TRADE",
    "SPEC",
    "Decision",
    "DecisionError",
    "DecisionPolicy",
    "DecisionReport",
    "DecisionRow",
    "build_decision_report",
    "render_lines",
    "render_markdown",
]

#: The master-spec phase this module implements (printed in every artefact).
SPEC = "24"

#: The two ways a decision can act, and the one way it can decline.
ENTER_CE = "ENTER_CE"
ENTER_PE = "ENTER_PE"
NO_TRADE = "NO_TRADE"

#: ``option_type -> decision`` for the side that was entered.
ENTERED: dict[str, str] = {"CE": ENTER_CE, "PE": ENTER_PE}

#: The closed vocabulary of reasons, in the order they are evaluated. ``ENTERED`` is the only
#: reason that accompanies an entry; every other code explains a refusal, and a test asserts
#: this tuple so a new code cannot be added without saying where it sits in the ladder.
DECISION_REASONS: tuple[str, ...] = (
    "ENTERED",
    "UNMAPPED_OPTION_TYPE",
    "DIRECTIONAL_CLASS_NOT_MODELLED",
    "MISSING_PROBABILITY",
    "BELOW_MIN_PROBABILITY",
    "NOT_ARGMAX",
    "MARGIN_BELOW_FLOOR",
    "DAILY_LIMIT_REACHED",
)

#: The columns one decision row carries - the CSV contract, and the JSON's ``decisions`` rows.
DECISION_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "tradingsymbol",
    "option_type",
    "timeframe",
    "directional_class",
    "top_class",
    "probability",
    "runner_up_probability",
    "margin",
    "decision",
    "reason",
    "lots",
    "lot_size",
    "quantity",
    "stop_loss_atr",
    "target_atr",
    "max_holding_minutes",
    "flat_at_session_end",
)

#: Printed in every artefact. Kept here (not in the writer) so JSON, Markdown and the console
#: block cannot drift apart, and so a test can assert the exact sentences.
DECISION_CAVEATS: tuple[str, ...] = (
    "ANJANEYA is a research system and never places an order: this record is what one "
    "configured rule would have done, not a recommendation and not a trade",
    "a decision is not an edge - the rule is arbitrary configuration, and every row here is "
    "conditional on the thresholds in model.decision, which were not fitted to any outcome",
    "the exit plan is stated, not simulated: stop, target, holding time and the session-end "
    "flatten are echoed from the policy, and replaying them against stored prices (with costs) "
    "is --mode backtest's job, not this mode's",
    "no look-ahead: a decision is read at the close of a bar from features stored for that "
    "instant, so the earliest anything could be filled is the next bar - the same rule spec 30 "
    "imposes on the backtest",
    "the probabilities come from one model artefact, and that model's own results label "
    "(EXPLORATORY, BELOW_GATE, ...) and the spec 4.3 gate travel in this artefact: a decision "
    "inherits the model's standing and adds nothing to it",
)


class DecisionError(RuntimeError):
    """Raised when a decision policy is impossible or a row cannot be decided."""


#: The default option side -> directional class map (``model.labels.classes`` names).
DEFAULT_DIRECTIONS: dict[str, str] = {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}


# ---------------------------------------------------------------------------
# The policy: configuration, turned into one tested pure decision
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """The rule that turns one predicted distribution into one decision (``model.decision``).

    Every field is configuration. The dataclass carries the same defaults the shipped YAML
    does, so a hand-built policy and the loaded one agree - the same discipline
    :class:`core.accumulation.AccumulationPolicy` follows.

    The ladder is evaluated in one order and the first rung that fails names the reason:

    1. the option side has no entry at all (``UNMAPPED_OPTION_TYPE``);
    2. the labels never produce the class that side needs
       (``DIRECTIONAL_CLASS_NOT_MODELLED``) - the model cannot speak about it;
    3. the model gives no probability for it (``MISSING_PROBABILITY``);
    4. the probability is below ``min_probability`` (``BELOW_MIN_PROBABILITY``);
    5. with ``require_argmax``, it is not the model's top class (``NOT_ARGMAX``);
    6. the gap to the runner-up is below ``min_margin`` (``MARGIN_BELOW_FLOOR``);
    7. only then is the day's budget consulted (``DAILY_LIMIT_REACHED``) - a throttle, which is
       why it is applied last and in :func:`build_decision_report` rather than here.
    """

    spec: str = SPEC
    min_probability: float = 0.40
    require_argmax: bool = True
    min_margin: float = 0.0
    directions: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DIRECTIONS)
    )
    lots: int = 1
    lot_size: int = 75
    #: The most entries one trading day may carry. ``None`` means no throttle.
    max_decisions_per_day: int | None = None
    stop_loss_atr: float | None = 1.0
    target_atr: float | None = 1.5
    max_holding_minutes: float | None = 30.0
    flat_at_session_end: bool = True

    def __post_init__(self) -> None:
        """Refuse a policy that could never fire, or that could fire on anything."""
        if not 0.0 < self.min_probability <= 1.0:
            raise DecisionError(
                f"min_probability must be in (0, 1], got {self.min_probability}"
            )
        if not 0.0 <= self.min_margin < 1.0:
            raise DecisionError(f"min_margin must be in [0, 1), got {self.min_margin}")
        if not self.directions:
            raise DecisionError("decision.directions must map at least one option type")
        unknown = sorted({str(name).upper() for name in self.directions} - set(ENTERED))
        if unknown:
            raise DecisionError(
                f"decision.directions has unknown option type(s) {unknown}; expected CE and/or PE"
            )
        blank = sorted(name for name, value in self.directions.items() if not str(value).strip())
        if blank:
            raise DecisionError(f"decision.directions has an empty class for {blank}")
        if self.lots < 1 or self.lot_size < 1:
            raise DecisionError(
                f"lots and lot_size must be at least 1, got {self.lots} and {self.lot_size}"
            )
        if self.max_decisions_per_day is not None and self.max_decisions_per_day < 1:
            raise DecisionError(
                "max_decisions_per_day must be at least 1 when set, got "
                f"{self.max_decisions_per_day}"
            )
        for name, value in (
            ("stop_loss_atr", self.stop_loss_atr),
            ("target_atr", self.target_atr),
            ("max_holding_minutes", self.max_holding_minutes),
        ):
            if value is not None and value <= 0:
                raise DecisionError(f"{name} must be positive when set, got {value}")

    @property
    def quantity(self) -> int:
        """Units one entry would size to (``lots * lot_size``)."""
        return int(self.lots) * int(self.lot_size)


    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> DecisionPolicy:
        """Spec 24's rules, read from ``model.decision`` (with CLI overrides applied).

        ``getattr`` defaults keep the policy buildable from a configuration that predates the
        block; every *value* here is also a dataclass default, so the two can never disagree.
        """
        block = getattr(config.model, "decision", None)
        payload: dict[str, Any] = {
            "spec": str(getattr(block, "spec", SPEC) or SPEC),
            "min_probability": float(getattr(block, "min_probability", 0.40)),
            "require_argmax": bool(getattr(block, "require_argmax", True)),
            "min_margin": float(getattr(block, "min_margin", 0.0)),
            "directions": {
                str(name).upper(): str(value)
                for name, value in dict(
                    getattr(block, "directions", None) or DEFAULT_DIRECTIONS
                ).items()
            },
            "lots": int(getattr(block, "lots", 1)),
            "lot_size": int(getattr(block, "lot_size", 75)),
            "max_decisions_per_day": getattr(block, "max_decisions_per_day", None),
            "stop_loss_atr": getattr(block, "stop_loss_atr", 1.0),
            "target_atr": getattr(block, "target_atr", 1.5),
            "max_holding_minutes": getattr(block, "max_holding_minutes", 30.0),
            "flat_at_session_end": bool(getattr(block, "flat_at_session_end", True)),
        }
        if payload["max_decisions_per_day"] is not None:
            payload["max_decisions_per_day"] = int(payload["max_decisions_per_day"])
        payload.update({name: value for name, value in overrides.items() if value is not None})
        return cls(**payload)

    def direction(self, option_type: str | None) -> str | None:
        """The directional class ``option_type`` needs, or ``None`` when unmapped."""
        if not option_type:
            return None
        return self.directions.get(str(option_type).upper())


    def decide(
        self,
        *,
        option_type: str | None,
        probabilities: Mapping[str, float],
        modelled_classes: Sequence[str] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """``(decision, reason, numbers)`` for one distribution - the whole ladder, purely.

        Returns the decision (``ENTER_CE``/``ENTER_PE``/``NO_TRADE``), the reason code that
        explains it, and the numbers behind it (the directional probability, the model's top
        class and its probability, the runner-up probability and the margin) so the caller
        stores the evidence rather than recomputing it and risking a second answer.

        ``modelled_classes`` are the classes the *model* can emit (its artefact's own class
        list). A side whose directional class is not among them is refused as
        ``DIRECTIONAL_CLASS_NOT_MODELLED`` - a different fact from "this row scored low".
        """
        numbers: dict[str, Any] = {
            "directional_class": None,
            "top_class": None,
            "probability": None,
            "runner_up_probability": None,
            "margin": None,
        }
        wanted = self.direction(option_type)
        if wanted is None:
            return NO_TRADE, "UNMAPPED_OPTION_TYPE", numbers
        numbers["directional_class"] = wanted

        available = {str(name) for name in (modelled_classes or probabilities)}
        if wanted not in available:
            return NO_TRADE, "DIRECTIONAL_CLASS_NOT_MODELLED", numbers

        ordered = sorted(probabilities.items(), key=lambda item: (-float(item[1]), str(item[0])))
        if not ordered:
            return NO_TRADE, "MISSING_PROBABILITY", numbers
        top_class, top_probability = str(ordered[0][0]), float(ordered[0][1])
        runner_probability = float(ordered[1][1]) if len(ordered) > 1 else 0.0
        numbers.update(
            {
                "top_class": top_class,
                "runner_up_probability": runner_probability,
                "margin": top_probability - runner_probability,
            }
        )

        candidate = next(
            (float(value) for name, value in probabilities.items() if str(name) == wanted), None
        )
        if candidate is None:
            return NO_TRADE, "MISSING_PROBABILITY", numbers
        numbers["probability"] = candidate

        if candidate < self.min_probability:
            return NO_TRADE, "BELOW_MIN_PROBABILITY", numbers
        if self.require_argmax and top_class != wanted:
            return NO_TRADE, "NOT_ARGMAX", numbers
        margin = candidate - runner_probability
        if margin < self.min_margin:
            return NO_TRADE, "MARGIN_BELOW_FLOOR", numbers
        side = str(option_type).upper() if option_type else ""
        return ENTERED.get(side, NO_TRADE), "ENTERED", numbers

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every artefact so a decision stays re-derivable."""
        return {
            "spec": self.spec,
            "min_probability": self.min_probability,
            "require_argmax": self.require_argmax,
            "min_margin": self.min_margin,
            "directions": {str(k): str(v) for k, v in self.directions.items()},
            "lots": self.lots,
            "lot_size": self.lot_size,
            "quantity": self.quantity,
            "max_decisions_per_day": self.max_decisions_per_day,
            "stop_loss_atr": self.stop_loss_atr,
            "target_atr": self.target_atr,
            "max_holding_minutes": self.max_holding_minutes,
            "flat_at_session_end": self.flat_at_session_end,
        }

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        argmax = " and the model's top class" if self.require_argmax else ""
        text = (
            f"enter the option side whose directional class reaches p >= "
            f"{self.min_probability:.2f}{argmax}"
        )
        if self.min_margin:
            text += f" with a margin of at least {self.min_margin:.2f}"
        if self.max_decisions_per_day is not None:
            text += f", at most {self.max_decisions_per_day} per day"
        return text



# ---------------------------------------------------------------------------
# Inputs and outputs: one scored bar in, one decision out
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One scored bar: the four fields a decision needs, and nothing else.

    Deliberately not a DataFrame. The model in this module stays free of pandas so that the
    rule can be read, tested and reused without a matrix library - the same reason
    :mod:`core.gates` was extracted from :mod:`core.ml.dataset`. The caller
    (``--mode decide``) owns the frame and hands over what the policy consumes.
    """

    timestamp: datetime
    tradingsymbol: str
    option_type: str | None
    probabilities: Mapping[str, float]
    timeframe: str | None = None
    #: The exchange-local trading day the entry budget is counted against.
    trading_date: date | None = None


@dataclass(frozen=True, slots=True)
class Decision:
    """What the policy did with one bar, and why (spec 24's unit of output)."""

    timestamp: datetime
    tradingsymbol: str
    option_type: str | None
    timeframe: str | None
    decision: str
    reason: str
    directional_class: str | None = None
    top_class: str | None = None
    probability: float | None = None
    runner_up_probability: float | None = None
    margin: float | None = None
    #: The exit plan *echoed* from the policy at the moment of the decision. Stated, not
    #: simulated: see the module docstring and :data:`DECISION_CAVEATS`.
    lots: int = 1
    lot_size: int = 1
    quantity: int = 1
    stop_loss_atr: float | None = None
    target_atr: float | None = None
    max_holding_minutes: float | None = None
    flat_at_session_end: bool = True

    @property
    def entered(self) -> bool:
        """True when the policy acted (``ENTER_CE`` or ``ENTER_PE``)."""
        return self.decision in (ENTER_CE, ENTER_PE)

    @property
    def trading_date(self) -> date:
        """The IST trading day of the row (exchange-local date of ``timestamp``)."""
        return self.timestamp.date()

    def to_record(self) -> dict[str, Any]:
        """One flat row, in :data:`DECISION_COLUMNS` order."""
        return {
            "timestamp": self.timestamp,
            "tradingsymbol": self.tradingsymbol,
            "option_type": self.option_type,
            "timeframe": self.timeframe,
            "directional_class": self.directional_class,
            "top_class": self.top_class,
            "probability": self.probability,
            "runner_up_probability": self.runner_up_probability,
            "margin": self.margin,
            "decision": self.decision,
            "reason": self.reason,
            "lots": self.lots,
            "lot_size": self.lot_size,
            "quantity": self.quantity,
            "stop_loss_atr": self.stop_loss_atr,
            "target_atr": self.target_atr,
            "max_holding_minutes": self.max_holding_minutes,
            "flat_at_session_end": self.flat_at_session_end,
        }

    def describe(self) -> str:
        """One line an operator can read out of a log or a console block."""
        probability = "unknown" if self.probability is None else f"{self.probability:.3f}"
        return (
            f"{self.timestamp.isoformat()} {self.tradingsymbol:<24} "
            f"{self.decision:<8} {self.reason:<30} p={probability}"
        )




# ---------------------------------------------------------------------------
# The report: one in-memory object that JSON, Markdown and the console share
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecisionReport:
    """Every decision the policy made over the rows it was shown, plus how it was reached.

    The report is the *evidence*: the policy that ran, the model that scored, the spec 4.3 gate
    and the model's own label, the counts by reason, and the rows themselves.
    """

    policy: DecisionPolicy
    decisions: tuple[Decision, ...]
    as_of: date
    timeframe: str | None = None
    horizon_minutes: int | None = None
    model: Mapping[str, Any] = field(default_factory=dict)
    gate: GateStatus | None = None
    verdict: Any = None
    #: The read window ``[start, end)`` the rows came from.
    window: tuple[datetime | None, datetime | None] = (None, None)
    symbols: tuple[str, ...] = ()
    rows_in_scope: int = 0
    caveats: tuple[str, ...] = DECISION_CAVEATS

    # -- the counts ----------------------------------------------------------
    @property
    def entries(self) -> tuple[Decision, ...]:
        """Every row the policy acted on (``ENTER_CE``/``ENTER_PE``)."""
        return tuple(decision for decision in self.decisions if decision.entered)

    @property
    def no_trade(self) -> tuple[Decision, ...]:
        """Every row the policy declined."""
        return tuple(decision for decision in self.decisions if not decision.entered)

    @property
    def sessions(self) -> tuple[date, ...]:
        """The distinct IST trading days the decisions span, oldest first."""
        return tuple(sorted({decision.trading_date for decision in self.decisions}))

    @property
    def entry_rate(self) -> float | None:
        """Entries / decisions, or ``None`` when nothing was decided (never ``0``)."""
        if not self.decisions:
            return None
        return len(self.entries) / len(self.decisions)

    @property
    def reason_counts(self) -> dict[str, int]:
        """How many rows each reason code explains (the whole vocabulary, zeros included)."""
        counts = dict.fromkeys(DECISION_REASONS, 0)
        for decision in self.decisions:
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
        return counts

    @property
    def entries_by_option_type(self) -> dict[str, int]:
        """Entries per option side, ``CE``/``PE`` only (the sides the policy can act on)."""
        counts = dict.fromkeys(sorted(self.policy.directions), 0)
        for decision in self.entries:
            side = str(decision.option_type or "").upper()
            counts[side] = counts.get(side, 0) + 1
        return counts

    @property
    def entries_by_day(self) -> dict[date, int]:
        """Entries per IST trading day, oldest first."""
        counts: dict[date, int] = dict.fromkeys(self.sessions, 0)
        for decision in self.entries:
            counts[decision.trading_date] += 1
        return counts

    @property
    def results_label(self) -> str | None:
        """The label spec 4.3's gate gives this run (``EXPLORATORY``, ``BELOW_GATE``, ...)."""
        return getattr(self.verdict, "results_label", None)

    def headline(self) -> str:
        """One sentence an operator can read out of a log."""
        decided = len(self.decisions)
        entered = len(self.entries)
        rate = "n/a" if self.entry_rate is None else f"{self.entry_rate:.1%}"
        return (
            f"{decided} decision(s) over {len(self.sessions)} day(s): {entered} entry(ies) "
            f"({rate}), {decided - entered} no-trade"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape: the run, the counts, then every decision."""
        window = {
            "start": self.window[0].isoformat() if self.window[0] is not None else None,
            "end": self.window[1].isoformat() if self.window[1] is not None else None,
        }
        summary: dict[str, Any] = {
            "decisions": len(self.decisions),
            "entries": len(self.entries),
            "no_trade": len(self.no_trade),
            "entry_rate": self.entry_rate,
            "sessions": len(self.sessions),
            "rows_in_scope": self.rows_in_scope,
            "reasons": self.reason_counts,
            "entries_by_option_type": self.entries_by_option_type,
            "entries_by_day": {
                day.isoformat(): count for day, count in self.entries_by_day.items()
            },
        }
        return {
            "report": f"spec {self.policy.spec} - the decision layer",
            "spec": self.policy.spec,
            "as_of": self.as_of.isoformat(),
            "timeframe": self.timeframe,
            "horizon_minutes": self.horizon_minutes,
            "window": window,
            "symbols": list(self.symbols),
            "policy": self.policy.to_dict(),
            "model": dict(self.model),
            "gate": self.gate.to_dict() if self.gate is not None else None,
            "verdict": self.verdict.to_dict() if self.verdict is not None else None,
            "results_label": self.results_label,
            "summary": summary,
            "decisions": [decision.to_record() for decision in self.decisions],
            "caveats": list(self.caveats),
        }

    def to_markdown(self) -> str:
        """The whole record as Markdown (see :func:`render_markdown`)."""
        return render_markdown(self)


def build_decision_report(
    rows: Sequence[DecisionRow],
    policy: DecisionPolicy,
    *,
    as_of: date,
    modelled_classes: Sequence[str] | None = None,
    timeframe: str | None = None,
    horizon_minutes: int | None = None,
    model: Mapping[str, Any] | None = None,
    gate: GateStatus | None = None,
    verdict: Any = None,
    window: tuple[datetime | None, datetime | None] = (None, None),
    symbols: Sequence[str] = (),
    caveats: Sequence[str] = DECISION_CAVEATS,
) -> DecisionReport:
    """Decide every row, in time order, applying the day's entry budget as it is spent.

    The order is total and deterministic - ``(trading_date, timestamp, tradingsymbol)`` - which
    is what makes ``max_decisions_per_day`` reproducible: the budget goes to the *earliest*
    signals of the day, across every symbol, rather than to whichever contract happened to sort
    first. A row that fails an earlier rung of the ladder keeps its own reason; only a row that
    would otherwise have entered is refused as ``DAILY_LIMIT_REACHED``.
    """
    ordered = sorted(rows, key=lambda row: (row.trading_date, row.timestamp, row.tradingsymbol))
    decided: list[Decision] = []
    spent: dict[date, int] = {}
    for row in ordered:
        decision, reason, numbers = policy.decide(
            option_type=row.option_type,
            probabilities=row.probabilities,
            modelled_classes=modelled_classes,
        )
        day = row.trading_date
        if decision != NO_TRADE and policy.max_decisions_per_day is not None:
            used = spent.get(day, 0)
            if used >= policy.max_decisions_per_day:
                decision, reason = NO_TRADE, "DAILY_LIMIT_REACHED"
            else:
                spent[day] = used + 1
        decided.append(
            Decision(
                timestamp=row.timestamp,
                tradingsymbol=row.tradingsymbol,
                option_type=row.option_type,
                timeframe=row.timeframe if row.timeframe is not None else timeframe,
                decision=decision,
                reason=reason,
                directional_class=numbers["directional_class"],
                top_class=numbers["top_class"],
                probability=numbers["probability"],
                runner_up_probability=numbers["runner_up_probability"],
                margin=numbers["margin"],
                lots=policy.lots,
                lot_size=policy.lot_size,
                quantity=policy.quantity,
                stop_loss_atr=policy.stop_loss_atr,
                target_atr=policy.target_atr,
                max_holding_minutes=policy.max_holding_minutes,
                flat_at_session_end=policy.flat_at_session_end,
            )
        )
    return DecisionReport(
        policy=policy,
        decisions=tuple(decided),
        as_of=as_of,
        timeframe=timeframe,
        horizon_minutes=horizon_minutes,
        model=dict(model or {}),
        gate=gate,
        verdict=verdict,
        window=window,
        symbols=tuple(symbols),
        rows_in_scope=len(rows),
        caveats=tuple(caveats),
    )



# ---------------------------------------------------------------------------
# Rendering: one report, three shapes, no second source of truth
# ---------------------------------------------------------------------------
def _number(value: Any) -> str:
    """A count with thousands separators, or ``-`` when there is no number to print."""
    return "-" if value is None else f"{int(value):,}"


def _stamp(value: Any) -> str:
    """A timestamp as text, or ``unknown`` - never a blank that could read as zero."""
    if value is None:
        return "unknown"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _window_text(report: DecisionReport) -> str:
    """``start .. end`` for the read window, open ends spelled out."""
    start, end = report.window
    return f"{_stamp(start)} .. {_stamp(end)}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """A Markdown table (header, separator, rows) - the report's one table renderer."""
    lines = [
        "| " + " | ".join(str(name) for name in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    lines.append("")
    return lines


def render_lines(report: DecisionReport) -> list[str]:
    """The console block: one ``key : value`` line per fact, greppable and copy-pasteable."""
    lines: list[str] = [
        f"ANJANEYA-AI decision layer (spec {report.policy.spec}) - as of "
        f"{report.as_of.isoformat()}",
        f"  headline   : {report.headline()}",
        f"  policy     : {report.policy.describe()}",
    ]
    if report.results_label:
        lines.append(f"  results    : {report.results_label}")
    if report.gate is not None:
        lines.append(f"  gate       : {report.gate.describe()}")
    lines.append(f"  window     : {_window_text(report)}")
    if report.timeframe:
        lines.append(f"  timeframe  : {report.timeframe}")
    name = report.model.get("name") if report.model else None
    if name:
        horizon = report.horizon_minutes if report.horizon_minutes is not None else "unknown"
        lines.append(f"  model      : {name} (horizon {horizon}m)")

    lines.append("  -- decisions (by reason) --")
    for reason, count in report.reason_counts.items():
        lines.append(f"  {reason:<30} : {_number(count)}")
    lines.append("  -- entries (by option side) --")
    for side, count in report.entries_by_option_type.items():
        lines.append(f"  {side:<30} : {_number(count)}")
    if report.decisions:
        lines.append("  -- entries (by trading day) --")
        for day, count in report.entries_by_day.items():
            lines.append(f"  {day.isoformat():<30} : {_number(count)}")
    return lines


def render_markdown(report: DecisionReport) -> str:
    """The whole record as Markdown: the policy, the model, the counts, the entries, caveats.

    The full decision list is in the JSON and the CSV; the Markdown prints every *entry* (the
    rows a reader would act on if this were advice, which it is not) and the counts that explain
    the rest, so a record of a busy day stays readable without hiding anything - nothing is
    dropped, it is simply counted rather than tabulated.
    """
    lines: list[str] = [
        f"# ANJANEYA decision layer (spec {report.policy.spec}) - what the policy would do",
        "",
        f"**As of:** {report.as_of.isoformat()}  ",
        f"**Window:** {_window_text(report)}  ",
        f"**Timeframe:** {report.timeframe or 'unknown'}  ",
        "**Horizon:** "
        f"{report.horizon_minutes if report.horizon_minutes is not None else 'unknown'}  ",
    ]
    if report.results_label:
        lines.append(f"**Results label:** {report.results_label}  ")
    if report.gate is not None:
        lines.append(f"**Gate:** {report.gate.describe()}  ")
    name = report.model.get("name") if report.model else None
    lines.append(f"**Model:** {name or 'unknown'}  ")
    lines.append("")

    lines.extend(["## 1. The policy", ""])
    lines.extend(
        _table(
            ["setting", "value"],
            [
                ["min_probability", f"{report.policy.min_probability:.4f}"],
                ["require_argmax", "yes" if report.policy.require_argmax else "no"],
                ["min_margin", f"{report.policy.min_margin:.4f}"],
                [
                    "directions",
                    ", ".join(
                        f"{side} <- {value}" for side, value in report.policy.directions.items()
                    ),
                ],
                ["lots x lot_size", f"{report.policy.lots} x {report.policy.lot_size}"],
                ["quantity", report.policy.quantity],
                ["max_decisions_per_day", report.policy.max_decisions_per_day or "unlimited"],
                [
                    "stop / target (ATR)",
                    f"{report.policy.stop_loss_atr} / {report.policy.target_atr}",
                ],
                ["max holding (min)", report.policy.max_holding_minutes],
                ["flat at session end", "yes" if report.policy.flat_at_session_end else "no"],
            ],
        )
    )

    lines.extend(["## 2. What the policy decided", ""])
    lines.extend(
        _table(
            ["decisions", "entries", "no-trade", "entry rate", "days"],
            [
                [
                    _number(len(report.decisions)),
                    _number(len(report.entries)),
                    _number(len(report.no_trade)),
                    "n/a" if report.entry_rate is None else f"{report.entry_rate:.3f}",
                    _number(len(report.sessions)),
                ]
            ],
        )
    )
    if report.reason_counts:
        lines.extend(["### Why (every reason code, including the zeroes)", ""])
        lines.extend(
            _table(
                ["reason", "rows"],
                [[reason, _number(count)] for reason, count in report.reason_counts.items()],
            )
        )

    lines.extend(["## 3. Entries", ""])
    if report.entries:
        lines.extend(
            _table(
                ["timestamp", "tradingsymbol", "side", "decision", "p", "margin", "quantity"],
                [
                    [
                        decision.timestamp.isoformat(),
                        decision.tradingsymbol,
                        decision.option_type,
                        decision.decision,
                        "unknown"
                        if decision.probability is None
                        else f"{decision.probability:.4f}",
                        "unknown" if decision.margin is None else f"{decision.margin:.4f}",
                        decision.quantity,
                    ]
                    for decision in report.entries
                ],
            )
        )
    else:
        lines.extend(["*No entry: the policy declined every row in scope.*", ""])

    lines.extend(["## Caveats", ""])
    lines.extend(f"* {caveat}" for caveat in report.caveats)
    lines.append("")
    return "\n".join(lines)


