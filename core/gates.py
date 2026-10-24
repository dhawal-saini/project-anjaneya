"""Spec 4.3's data-accumulation gates, as one pure function (shared, model-free).

Master spec section 4.3 says ANJANEYA "must track and report data coverage" against four
milestones - 20 captured trading days unlock exploratory analysis, 60 unlock a first ML
model, 120 make a backtest statistically meaningful and 250 unlock seasonality and regime
work - and that ANJANEYA "MUST refuse to train a production model below the configured
minimum and must clearly label any preliminary results".

This module is the *rule*; ``config/model.yaml`` holds the *values*. It was extracted from
:mod:`core.ml.dataset` (spec 21) for one reason: a report that has nothing to do with machine
learning - :mod:`core.accumulation`, the continuous data-accumulation report (spec 29) - must
be able to ask "which gate is open?" without importing pandas, LightGBM or a matrix builder.
There is exactly one ladder, so an operator can never see two different verdicts for the same
number of captured days.

Importing this module performs no I/O and imports nothing heavy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "GATE_SENTENCES",
    "GATE_STAGES",
    "GateStatus",
    "data_gate_status",
]

#: Stage names, in the order the gates unlock them.
GATE_STAGES = (
    "INSUFFICIENT",
    "EXPLORATORY",
    "ML_TRAINING",
    "MEANINGFUL_BACKTEST",
    "SEASONALITY",
)

#: What each stage permits, in the spec's own words.
GATE_SENTENCES: dict[str, str] = {
    "INSUFFICIENT": "too little data to conclude anything",
    "EXPLORATORY": "exploratory analysis only",
    "ML_TRAINING": "a first ML model may be trained (results stay PRELIMINARY)",
    "MEANINGFUL_BACKTEST": "backtests become statistically meaningful",
    "SEASONALITY": "seasonality and regime analysis become meaningful",
}


@dataclass(frozen=True, slots=True)
class GateStatus:
    """How many captured trading days a run saw, and what that permits (spec 4.3)."""

    days: int
    stage: str
    allowed: str
    gates: dict[str, int]
    next_stage: str | None = None
    days_to_next: int | None = None

    @property
    def ready_for_training(self) -> bool:
        """True once a production model may be trained (``ml_training_min_days``)."""
        return self.days >= self.gates["ml_training_min_days"]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "days": self.days,
            "stage": self.stage,
            "allowed": self.allowed,
            "ready_for_training": self.ready_for_training,
            "next_stage": self.next_stage,
            "days_to_next": self.days_to_next,
            "gates": dict(self.gates),
        }

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        text = f"{self.days} captured day(s): {self.allowed} [{self.stage}]"
        if self.next_stage:
            text += f"; {self.days_to_next} more day(s) to {self.next_stage}"
        return text


def data_gate_status(
    days: int,
    *,
    exploratory: int,
    ml_training: int,
    meaningful_backtest: int,
    seasonality: int,
) -> GateStatus:
    """Classify ``days`` of captured data against spec 4.3's four gates.

    The thresholds are passed in rather than read from a config object so the rule stays a
    pure function: every *value* in ANJANEYA lives in YAML, every *rule* in a tested
    function. The stage ladder is inclusive from the bottom (``days >= gate``), so a day
    that reaches a gate unlocks it immediately.
    """
    gates = {
        "exploratory_min_days": int(exploratory),
        "ml_training_min_days": int(ml_training),
        "meaningful_backtest_min_days": int(meaningful_backtest),
        "seasonality_min_days": int(seasonality),
    }
    ladder = (
        ("SEASONALITY", gates["seasonality_min_days"]),
        ("MEANINGFUL_BACKTEST", gates["meaningful_backtest_min_days"]),
        ("ML_TRAINING", gates["ml_training_min_days"]),
        ("EXPLORATORY", gates["exploratory_min_days"]),
    )
    stage = "INSUFFICIENT"
    next_stage: str | None = "EXPLORATORY"
    days_to_next: int | None = gates["exploratory_min_days"] - days
    for index, (name, threshold) in enumerate(ladder):
        if days >= threshold:
            stage = name
            if index == 0:
                next_stage, days_to_next = None, None
            else:
                next_name, next_threshold = ladder[index - 1]
                next_stage, days_to_next = next_name, next_threshold - days
            break

    return GateStatus(
        days=int(days),
        stage=stage,
        allowed=GATE_SENTENCES[stage],
        gates=gates,
        next_stage=next_stage,
        days_to_next=days_to_next,
    )
