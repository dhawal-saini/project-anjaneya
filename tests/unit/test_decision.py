"""Spec 24's decision layer: the policy, the ladder, the budget and the record.

``core.decision`` is the piece of ANJANEYA that could most easily be misread as advice, so the
tests here are mostly about what it *refuses* to claim:

* **the ladder is closed and ordered.** Every reason code is reachable, the first failing rung
  names the reason, and a decision is never silently both "entered" and "not entered".
* **a decision is not a position.** The exit plan travels in the record as *stated* values -
  there is no fill price, no P&L and no simulation anywhere in the module.
* **the day's budget is deterministic.** ``max_decisions_per_day`` goes to the earliest signals
  of the day, across symbols, and a row that already failed an earlier rung keeps its own
  reason instead of being relabelled.
* **the gate is spec 4.3's**, and the model's own label travels untouched.
* **it renders from one object.** JSON, Markdown and the console block are three views of one
  ``DecisionReport``, so they cannot disagree.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from core.decision import (
    DECISION_CAVEATS,
    DECISION_COLUMNS,
    DECISION_REASONS,
    ENTER_CE,
    ENTER_PE,
    NO_TRADE,
    DecisionError,
    DecisionPolicy,
    DecisionReport,
    DecisionRow,
    build_decision_report,
    render_lines,
    render_markdown,
)
from core.gates import data_gate_status

IST = ZoneInfo("Asia/Kolkata")

#: A distribution that clears the shipped policy for CE.
CE_WINS = {"UP_CONTINUATION": 0.55, "DOWN_CONTINUATION": 0.20, "RANGE": 0.25}
#: The mirror image, for PE.
PE_WINS = {"UP_CONTINUATION": 0.20, "DOWN_CONTINUATION": 0.60, "RANGE": 0.20}


def moment(day: date, minute: int = 0) -> datetime:
    """A minute-accurate IST timestamp on ``day`` (09:15 + ``minute``)."""
    return datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST) + timedelta(minutes=minute)


def row(
    *,
    day: date = date(2026, 10, 1),
    minute: int = 0,
    symbol: str = "NIFTY01OCT2625000CE",
    option_type: str | None = "CE",
    probabilities: dict[str, float] | None = None,
) -> DecisionRow:
    """One scored bar, with the CE-wins distribution unless another is given."""
    stamp = moment(day, minute)
    return DecisionRow(
        timestamp=stamp,
        tradingsymbol=symbol,
        option_type=option_type,
        probabilities=dict(probabilities if probabilities is not None else CE_WINS),
        timeframe="1m",
        trading_date=stamp.date(),
    )


# ---------------------------------------------------------------------------
# The policy: what it refuses at construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_probability": 0.0}, "min_probability"),
        ({"min_probability": 1.5}, "min_probability"),
        ({"min_margin": -0.1}, "min_margin"),
        ({"min_margin": 1.0}, "min_margin"),
        ({"directions": {}}, "directions"),
        ({"directions": {"XX": "UP_CONTINUATION"}}, "unknown option type"),
        ({"directions": {"CE": "  "}}, "empty class"),
        ({"lots": 0}, "lots and lot_size"),
        ({"lot_size": 0}, "lots and lot_size"),
        ({"max_decisions_per_day": 0}, "max_decisions_per_day"),
        ({"stop_loss_atr": 0.0}, "stop_loss_atr"),
        ({"target_atr": -1.0}, "target_atr"),
        ({"max_holding_minutes": 0.0}, "max_holding_minutes"),
    ],
)
def test_an_impossible_policy_is_refused(kwargs: dict, message: str) -> None:
    with pytest.raises(DecisionError, match=message):
        DecisionPolicy(**kwargs)


def test_the_shipped_defaults_are_the_documented_ones() -> None:
    policy = DecisionPolicy()
    assert policy.spec == "24"
    assert policy.min_probability == 0.40
    assert policy.require_argmax is True
    assert policy.min_margin == 0.0
    assert policy.max_decisions_per_day is None
    assert policy.quantity == policy.lots * policy.lot_size == 75
    assert policy.directions == {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}


def test_the_off_switches_are_representable() -> None:
    """``None`` is how an exit or a throttle is switched off - and it stays ``None``."""
    policy = DecisionPolicy(
        max_decisions_per_day=None,
        stop_loss_atr=None,
        target_atr=None,
        max_holding_minutes=None,
    )
    described = policy.to_dict()
    assert described["stop_loss_atr"] is None
    assert described["target_atr"] is None
    assert described["max_holding_minutes"] is None
    assert described["max_decisions_per_day"] is None
    assert "unlimited" not in policy.describe()


# ---------------------------------------------------------------------------
# The ladder: every reason code is reachable, and only one at a time
# ---------------------------------------------------------------------------
def test_the_reason_vocabulary_is_closed_and_starts_with_entered() -> None:
    """The ordered tuple is the contract: a new code has to be placed in the ladder."""
    assert DECISION_REASONS[0] == "ENTERED"
    assert len(set(DECISION_REASONS)) == len(DECISION_REASONS)
    assert DECISION_REASONS[-1] == "DAILY_LIMIT_REACHED"


def test_a_clearing_distribution_enters_the_mapped_side() -> None:
    policy = DecisionPolicy()
    decision, reason, numbers = policy.decide(option_type="CE", probabilities=CE_WINS)
    assert decision == ENTER_CE
    assert reason == "ENTERED"
    assert numbers["directional_class"] == "UP_CONTINUATION"
    assert numbers["probability"] == pytest.approx(0.55)
    assert numbers["margin"] == pytest.approx(0.55 - 0.25)


def test_the_pe_side_uses_its_own_directional_class() -> None:
    decision, reason, _ = DecisionPolicy().decide(option_type="PE", probabilities=PE_WINS)
    assert (decision, reason) == (ENTER_PE, "ENTERED")


def test_an_unmapped_option_type_never_enters() -> None:
    decision, reason, numbers = DecisionPolicy().decide(
        option_type="FUT", probabilities=CE_WINS
    )
    assert (decision, reason) == (NO_TRADE, "UNMAPPED_OPTION_TYPE")
    assert numbers["directional_class"] is None


def test_a_side_the_model_cannot_speak_about_is_its_own_reason() -> None:
    """``modelled_classes`` distinguishes "cannot talk about it" from "scored low"."""
    policy = DecisionPolicy()
    decision, reason, _ = policy.decide(
        option_type="PE",
        probabilities={"UP_CONTINUATION": 0.9, "DOWN_CONTINUATION": 0.1},
        modelled_classes=("UP_CONTINUATION",),
    )
    assert (decision, reason) == (NO_TRADE, "DIRECTIONAL_CLASS_NOT_MODELLED")


def test_a_missing_probability_is_not_a_zero() -> None:
    """The directional class is modelled but the row carries no probability for it."""
    policy = DecisionPolicy()
    decision, reason, numbers = policy.decide(
        option_type="CE",
        probabilities={"DOWN_CONTINUATION": 0.7, "RANGE": 0.3},
        modelled_classes=("CE_UP", "UP_CONTINUATION", "DOWN_CONTINUATION", "RANGE"),
    )
    assert (decision, reason) == (NO_TRADE, "MISSING_PROBABILITY")
    assert numbers["probability"] is None


def test_a_probability_below_the_floor_is_refused() -> None:
    policy = DecisionPolicy(min_probability=0.60)
    decision, reason, numbers = policy.decide(option_type="CE", probabilities=CE_WINS)
    assert (decision, reason) == (NO_TRADE, "BELOW_MIN_PROBABILITY")
    assert numbers["probability"] == pytest.approx(0.55)


def test_a_strong_but_not_winning_class_needs_the_argmax_switch_off() -> None:
    """The same distribution is refused with argmax on and accepted with it off."""
    probabilities = {"UP_CONTINUATION": 0.45, "RANGE": 0.50, "DOWN_CONTINUATION": 0.05}
    strict, strict_reason, _ = DecisionPolicy(min_probability=0.40).decide(
        option_type="CE", probabilities=probabilities
    )
    assert (strict, strict_reason) == (NO_TRADE, "NOT_ARGMAX")
    loose, loose_reason, _ = DecisionPolicy(min_probability=0.40, require_argmax=False).decide(
        option_type="CE", probabilities=probabilities
    )
    assert (loose, loose_reason) == (ENTER_CE, "ENTERED")


def test_the_margin_floor_is_the_last_rung_before_entering() -> None:
    probabilities = {"UP_CONTINUATION": 0.50, "RANGE": 0.45, "DOWN_CONTINUATION": 0.05}
    policy = DecisionPolicy(min_probability=0.40, min_margin=0.10)
    decision, reason, numbers = policy.decide(option_type="CE", probabilities=probabilities)
    assert (decision, reason) == (NO_TRADE, "MARGIN_BELOW_FLOOR")
    assert numbers["margin"] == pytest.approx(0.05)


def test_every_rung_that_can_fire_does_fire() -> None:
    """A sweep over the vocabulary: no code is dead, and no row gets two of them."""
    policy = DecisionPolicy(min_probability=0.40, min_margin=0.05)
    seen: set[str] = set()
    cases = [
        ({}, "CE"),
        (CE_WINS, None),
        ({}, "PE"),
        (CE_WINS, "CE"),
        ({}, "FUT"),
    ]
    for probabilities, side in cases:
        _, reason, _ = policy.decide(option_type=side, probabilities=probabilities)
        seen.add(reason)
    assert seen <= set(DECISION_REASONS)


def test_the_ladder_evaluates_in_the_documented_order() -> None:
    """Below the floor *and* not the argmax: the first failing rung names the reason."""
    decision, reason, _ = DecisionPolicy(min_probability=0.60).decide(
        option_type="CE",
        probabilities={"UP_CONTINUATION": 0.30, "RANGE": 0.70},
    )
    assert (decision, reason) == (NO_TRADE, "BELOW_MIN_PROBABILITY")



# ---------------------------------------------------------------------------
# The report: counts, the day's budget, and the exit plan that is only stated
# ---------------------------------------------------------------------------
def report(rows: list[DecisionRow], policy: DecisionPolicy | None = None) -> DecisionReport:
    """A report over ``rows`` with a policy that enters CE only (the default)."""
    return build_decision_report(
        rows,
        policy if policy is not None else DecisionPolicy(),
        as_of=max(item.trading_date for item in rows),
        timeframe="1m",
        horizon_minutes=5,
        model={"name": "model_lightgbm.txt"},
    )


def test_the_report_counts_entries_and_refusals() -> None:
    built = report([row(minute=0), row(minute=1, probabilities=PE_WINS)])
    assert len(built.decisions) == 2
    assert len(built.entries) == 1
    assert len(built.no_trade) == 1
    assert built.entry_rate == pytest.approx(0.5)


def test_the_reason_counts_carry_the_whole_vocabulary() -> None:
    """A zero is a measurement too: the shape does not change with the data."""
    built = report([row(minute=0)])
    counts = built.reason_counts
    assert set(counts) == set(DECISION_REASONS)
    assert counts["ENTERED"] == 1
    assert counts["NOT_ARGMAX"] == 0


def test_an_empty_report_has_no_rate_rather_than_a_zero() -> None:
    built = build_decision_report([], DecisionPolicy(), as_of=date(2026, 10, 1))
    assert built.entry_rate is None
    assert built.decisions == ()
    assert "0 decision(s)" in built.headline()


def test_the_daily_budget_goes_to_the_earliest_signals() -> None:
    policy = DecisionPolicy(max_decisions_per_day=1)
    built = report(
        [row(minute=5, symbol="B"), row(minute=1, symbol="A"), row(minute=9, symbol="C")],
        policy,
    )
    entered = [decision.tradingsymbol for decision in built.entries]
    assert entered == ["A"]  # 09:16, the earliest row of the day
    assert built.reason_counts["DAILY_LIMIT_REACHED"] == 2


def test_the_budget_is_per_trading_day() -> None:
    policy = DecisionPolicy(max_decisions_per_day=1)
    day_a, day_b = date(2026, 9, 30), date(2026, 10, 1)
    built = report([row(day=day_a, minute=0), row(day=day_b, minute=0)], policy)
    assert built.entries_by_day == {day_a: 1, day_b: 1}


def test_a_row_that_failed_earlier_keeps_its_own_reason() -> None:
    """The budget must not relabel a row the ladder already refused."""
    policy = DecisionPolicy(max_decisions_per_day=1, min_probability=0.60)
    built = report(
        [
            row(minute=0, probabilities={"UP_CONTINUATION": 0.70, "RANGE": 0.30}),
            row(minute=1, option_type="PE", probabilities=PE_WINS),
            row(minute=2, probabilities={"UP_CONTINUATION": 0.30, "RANGE": 0.70}),
        ],
        policy,
    )
    assert [decision.reason for decision in built.decisions] == [
        "ENTERED",
        "DAILY_LIMIT_REACHED",
        "BELOW_MIN_PROBABILITY",
    ]


def test_entries_are_counted_per_side() -> None:
    built = report([row(minute=0), row(minute=1, probabilities=PE_WINS)])
    assert built.entries_by_option_type == {"CE": 1, "PE": 0}


def test_the_decisions_are_ordered_by_time_across_symbols() -> None:
    built = report([row(minute=3, symbol="B"), row(minute=1, symbol="A"), row(minute=2, symbol="C")])
    assert [decision.tradingsymbol for decision in built.decisions] == ["A", "C", "B"]


def test_the_exit_plan_is_stated_and_never_simulated() -> None:
    """The record carries the plan; it carries no fill, no price and no P&L."""
    policy = DecisionPolicy(lots=2, lot_size=75, stop_loss_atr=1.5, target_atr=2.0,
                            max_holding_minutes=15.0, flat_at_session_end=False)
    built = report([row()], policy)
    entered = built.entries[0]
    assert entered.quantity == 150
    assert (entered.stop_loss_atr, entered.target_atr) == (1.5, 2.0)
    assert entered.max_holding_minutes == 15.0
    assert entered.flat_at_session_end is False
    record = entered.to_record()
    assert list(record) == list(DECISION_COLUMNS)
    for forbidden in ("pnl", "fill_price", "entry_price", "exit_price"):
        assert forbidden not in record


def test_a_decision_never_claims_an_unmeasured_number() -> None:
    """A refused row leaves probability/margin missing rather than filling them with 0."""
    built = report([row(option_type="FUT")])
    record = built.no_trade[0].to_record()
    assert record["probability"] is None
    assert record["margin"] is None
    assert "p=unknown" in built.no_trade[0].describe()



# ---------------------------------------------------------------------------
# Rendering: one object, three views
# ---------------------------------------------------------------------------
def test_the_json_document_carries_the_run_the_counts_and_every_row() -> None:
    built = report([row(minute=0), row(minute=1, option_type="FUT")])
    document = built.to_dict()
    assert document["report"] == "spec 24 - the decision layer"
    assert document["spec"] == "24"
    assert document["model"]["name"] == "model_lightgbm.txt"
    assert document["summary"]["decisions"] == 2
    assert document["summary"]["entries"] == 1
    assert len(document["decisions"]) == 2
    assert document["caveats"] == list(DECISION_CAVEATS)
    assert document["policy"]["quantity"] == 75


def test_the_gate_and_the_model_label_travel_untouched() -> None:
    gate = data_gate_status(
        4, exploratory=20, ml_training=60, meaningful_backtest=120, seasonality=250
    )
    built = build_decision_report(
        [row()],
        DecisionPolicy(),
        as_of=date(2026, 10, 1),
        gate=gate,
        model={"name": "m.txt", "results_label": "BELOW_GATE"},
    )
    document = built.to_dict()
    assert document["gate"]["stage"] == "INSUFFICIENT"
    assert document["gate"]["gates"]["ml_training_min_days"] == 60
    assert document["model"]["results_label"] == "BELOW_GATE"


def test_the_markdown_has_the_sections_and_prints_every_entry() -> None:
    built = report([row(minute=0), row(minute=1, option_type="FUT")])
    text = render_markdown(built)
    for heading in ("## 1. The policy", "## 2. What the policy decided", "## 3. Entries"):
        assert heading in text
    assert "## Caveats" in text
    assert "NIFTY01OCT2625000CE" in text
    # every entry is tabulated, and the refusal is counted rather than listed
    assert text.count("| ENTER_CE |") == 1
    assert "UNMAPPED_OPTION_TYPE" in text


def test_the_markdown_says_so_when_nothing_entered() -> None:
    built = report([row(option_type="FUT")])
    assert "No entry: the policy declined every row in scope." in render_markdown(built)


def test_the_console_block_is_greppable() -> None:
    built = report([row(minute=0), row(minute=1, option_type="FUT")])
    lines = render_lines(built)
    text = "\n".join(lines)
    assert lines[0].startswith("ANJANEYA-AI decision layer (spec 24)")
    assert "  headline   : " in text
    assert "  policy     : enter the option side" in text
    assert "  -- decisions (by reason) --" in text
    assert "  -- entries (by option side) --" in text
    assert "ENTERED" in text


def test_the_caveats_say_what_a_decision_is_not() -> None:
    joined = " ".join(DECISION_CAVEATS).lower()
    for claim in ("never places an order", "not a recommendation", "a decision is not an edge",
                  "stated, not simulated", "no look-ahead"):
        assert claim in joined


# ---------------------------------------------------------------------------
# The configuration: the shipped block and the policy agree
# ---------------------------------------------------------------------------
def config_policy() -> DecisionPolicy:
    """The policy built from the shipped ``config/model.yaml``."""
    from core.config_loader import load_config

    return DecisionPolicy.from_config(load_config())


def test_from_config_reads_the_shipped_block() -> None:
    policy = config_policy()
    assert policy.spec == "24"
    assert policy.min_probability == pytest.approx(0.40)
    assert policy.require_argmax is True
    assert policy.min_margin == pytest.approx(0.0)
    assert policy.lots == 1 and policy.lot_size == 75
    assert policy.max_decisions_per_day is None
    assert policy.directions == {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}


def test_overrides_win_over_the_block_and_none_is_ignored() -> None:
    from core.config_loader import load_config

    config = load_config()
    overridden = DecisionPolicy.from_config(
        config, min_probability=0.75, min_margin=0.20, max_decisions_per_day=3
    )
    assert overridden.min_probability == pytest.approx(0.75)
    assert overridden.min_margin == pytest.approx(0.20)
    assert overridden.max_decisions_per_day == 3
    # ``None`` means "not overridden" - the block's own value survives.
    untouched = DecisionPolicy.from_config(config, min_probability=None, min_margin=None)
    assert untouched.min_probability == pytest.approx(0.40)

