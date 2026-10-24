"""Spec 29 - the continuous data-accumulation report (master-prompt phase 25).

Master spec section 4.2 explains *why* this report has to exist: NIFTY options history is
"BUILT BY ANJANEYA-CAPTURE, captured live, day by day", so "the options dataset grows over
time" and "ANJANEYA-AI must wait for enough accumulated days before training a serious
model". Section 4.3 then requires ANJANEYA to "track and report data coverage" against four
milestones (20/60/120/250 captured trading days). Phase 25 asks for the recurring shape of
that report: a **weekly summary of data captured, coverage, gaps, sample sizes and readiness
for retraining**.

This module is the *model*; :mod:`core.storage.accumulation` reads the numbers out of DuckDB
and :mod:`apps.anjaneeya_ai.accumulation_writer` turns the result into files. Keeping the
model pure means the same object renders to JSON, to Markdown and to a console block, so the
three can never disagree.

Four rules shape it, in the spirit of the rest of the project:

1. **A count is a fact, never a guess.** Every sample size is ``count(*)`` of what is stored.
   Nothing is extrapolated, interpolated or imputed, and a table that does not exist is
   reported as *unavailable with the reason*, never as zero.
2. **A captured day is not a complete day.** ``coverage_ratio`` is the only thing that says a
   session was recorded whole, so a day with a ratio below the configured floor, or with any
   gap, is called out rather than averaged away.
3. **Readiness is a checklist, not a number.** The gate, the coverage, the feature rows and
   the label rows are tested one by one, each with its ``have`` and its ``need``, so "not
   ready" always comes with the reason and the distance.
4. **The report describes the dataset, not the market.** A green row here means "there is
   enough recorded data to try something", not "something is true" - and every artefact says
   so, because a coverage report is exactly the kind of thing a reader over-reads.

Importing this module performs no I/O. It imports nothing heavier than ``core.gates``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from core.gates import GateStatus, data_gate_status

__all__ = [
    "ACCUMULATION_CAVEATS",
    "READINESS_KEYS",
    "SPEC",
    "AccumulationInputs",
    "AccumulationPolicy",
    "AccumulationReport",
    "CoverageDay",
    "ModelRecord",
    "ReadinessItem",
    "TableProfile",
    "WeekCapture",
    "build_accumulation_report",
    "render_lines",
    "render_markdown",
]

#: The master-spec section this report answers (printed in every artefact).
SPEC = "29"

#: A week, for bucketing captured days.
WEEK_DAYS = 7

#: The readiness checks, in the order they are evaluated and printed.
READINESS_KEYS: tuple[str, ...] = (
    "captured_days",
    "coverage",
    "feature_rows",
    "label_rows",
)

#: Printed in every report. Kept here (not in the writer) so JSON, Markdown and the console
#: block cannot drift apart, and so a test can assert the exact sentences.
ACCUMULATION_CAVEATS: tuple[str, ...] = (
    "this report describes the *dataset*, not the market: a satisfied requirement means "
    "there is enough recorded data to try something, never that something is true",
    "a captured day is not necessarily a complete day - read coverage_ratio and gap_count, "
    "not the day count alone (a day whose ratio is below the configured floor is called out)",
    "every sample size is count(*) of what is stored; nothing is extrapolated, imputed or "
    "smoothed, and a missing table is reported as unavailable rather than as zero",
    "the coverage ledger is written by ANJANEYA-CAPTURE and ANJANEYA-ENGINE; a day can have "
    "ticks and still have no candles until --mode build-candles has run",
    "readiness for retraining is a checklist about the data, not a claim about model quality: "
    "whether a retrain helps is decided by purged walk-forward validation, not here",
)


@dataclass(frozen=True, slots=True)
class AccumulationPolicy:
    """Every rule the report applies, read from the configuration in one place.

    The defaults mirror ``config/model.yaml`` (``accumulation:`` plus the ``data_gates:``
    block), so a caller that has no configuration - a test, a notebook - still gets the
    shipped numbers rather than a second, silently different set.
    """

    spec: str = SPEC
    #: How many weeks back the captured-day series is bucketed (the report's own window).
    window_weeks: int = 4
    #: A capture gap longer than this many calendar days is called out as "capture may have
    #: stopped" - the one failure a weekly report exists to notice.
    stale_days: int = 10
    #: A day counts as *complete* at or above this coverage ratio (spec 4.3's ledger field).
    healthy_coverage_ratio: float = 0.9
    #: The floors a retrain is worth considering against (sample sizes, not conclusions).
    retrain_min_feature_rows: int = 5000
    retrain_min_label_rows: int = 5000
    #: A model older than this many days, on enough data, is flagged as due for a retrain.
    retrain_max_model_age_days: int = 30
    #: Spec 4.3's four gates (the same values ``data_gates`` carries).
    exploratory_min_days: int = 20
    ml_training_min_days: int = 45
    meaningful_backtest_min_days: int = 120
    seasonality_min_days: int = 250

    @classmethod
    def from_config(cls, config: Any) -> AccumulationPolicy:
        """Read the policy out of a loaded :class:`core.config_loader.AnjaneyaConfig`.

        The presentation knobs come from ``model.accumulation`` and the four gate *values*
        from ``model.data_gates`` - the same block ``--mode train`` refuses on, so a weekly
        report and a training run can never disagree about which gate is open.
        """
        block = getattr(config.model, "accumulation", None)
        gates = config.model.data_gates
        return cls(
            spec=str(getattr(block, "spec", SPEC) or SPEC),
            window_weeks=int(getattr(block, "window_weeks", 4)),
            stale_days=int(getattr(block, "stale_days", 10)),
            healthy_coverage_ratio=float(getattr(block, "healthy_coverage_ratio", 0.9)),
            retrain_min_feature_rows=int(getattr(block, "retrain_min_feature_rows", 5000)),
            retrain_min_label_rows=int(getattr(block, "retrain_min_label_rows", 5000)),
            retrain_max_model_age_days=int(getattr(block, "retrain_max_model_age_days", 30)),
            exploratory_min_days=int(gates.exploratory_min_days),
            ml_training_min_days=int(gates.ml_training_min_days),
            meaningful_backtest_min_days=int(gates.meaningful_backtest_min_days),
            seasonality_min_days=int(gates.seasonality_min_days),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every report so a number stays re-derivable."""
        return {
            "spec": self.spec,
            "window_weeks": self.window_weeks,
            "stale_days": self.stale_days,
            "healthy_coverage_ratio": self.healthy_coverage_ratio,
            "retrain_min_feature_rows": self.retrain_min_feature_rows,
            "retrain_min_label_rows": self.retrain_min_label_rows,
            "retrain_max_model_age_days": self.retrain_max_model_age_days,
            "gates": {
                "exploratory_min_days": self.exploratory_min_days,
                "ml_training_min_days": self.ml_training_min_days,
                "meaningful_backtest_min_days": self.meaningful_backtest_min_days,
                "seasonality_min_days": self.seasonality_min_days,
            },
        }

    def gate_status(self, days: int) -> GateStatus:
        """Spec 4.3's verdict for ``days`` captured trading days (one ladder, one function)."""
        return data_gate_status(
            days,
            exploratory=self.exploratory_min_days,
            ml_training=self.ml_training_min_days,
            meaningful_backtest=self.meaningful_backtest_min_days,
            seasonality=self.seasonality_min_days,
        )


@dataclass(frozen=True, slots=True)
class TableProfile:
    """One stored table's sample size and span - or why it could not be read.

    ``unavailable`` is set (and ``rows`` left at the caller's ``0``) when the table does not
    exist in the database: an old install, or a stream that has never run. Reporting it as
    *zero rows* would be a lie - "we did not look" and "we looked and found nothing" are
    different facts, and this project never conflates them.
    """

    table: str
    rows: int = 0
    #: Distinct calendar days the table's timestamp spans. ``None`` for a count-only table
    #: (``instruments``), so the report never prints "over 0 day(s)" for something that has
    #: no timestamp at all.
    days: int | None = None
    first: Any = None
    last: Any = None
    unavailable: str | None = None

    @property
    def present(self) -> bool:
        """True when the table was read (whether or not it holds any rows)."""
        return self.unavailable is None

    @property
    def empty(self) -> bool:
        """True when the table was read and holds nothing."""
        return self.unavailable is None and self.rows == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "table": self.table,
            "rows": self.rows,
            "days": self.days,
            "first": _stamp(self.first),
            "last": _stamp(self.last),
            "unavailable": self.unavailable,
        }


@dataclass(frozen=True, slots=True)
class CoverageDay:
    """One trading day's verdict from the ``data_coverage`` ledger (spec 4.3).

    A day usually has one row per ``(tradingsymbol, timeframe)`` stream. Those rows are
    collapsed here into the four facts a weekly report needs: how many streams exist, how
    many were complete, how many had gaps and how many were missing entirely, plus the worst
    coverage ratio across the day and the day's gap total.
    """

    trading_date: date
    streams: int = 0
    complete_streams: int = 0
    gap_streams: int = 0
    missing_streams: int = 0
    suspect_streams: int = 0
    min_ratio: float | None = None
    gap_count: int = 0
    largest_gap_seconds: float = 0.0
    quality_flags: dict[str, int] = field(default_factory=dict)

    @property
    def has_gap(self) -> bool:
        """True when any stream on this day recorded a gap."""
        return self.gap_count > 0

    def is_complete(self, ratio_floor: float) -> bool:
        """True when no stream gapped, none is missing/suspect, and the worst ratio is fine.

        ``min_ratio`` of ``None`` means no stream on the day carried an expectation (a tick
        stream does not), which is not evidence of completeness - so it is treated as *not*
        complete rather than quietly passing.
        """
        if self.gap_count > 0 or self.missing_streams > 0 or self.suspect_streams > 0:
            return False
        if self.min_ratio is None:
            return False
        return self.min_ratio >= ratio_floor

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "trading_date": self.trading_date.isoformat(),
            "streams": self.streams,
            "complete_streams": self.complete_streams,
            "gap_streams": self.gap_streams,
            "missing_streams": self.missing_streams,
            "suspect_streams": self.suspect_streams,
            "min_ratio": self.min_ratio,
            "gap_count": self.gap_count,
            "largest_gap_seconds": self.largest_gap_seconds,
            "quality_flags": dict(self.quality_flags),
        }


@dataclass(frozen=True, slots=True)
class WeekCapture:
    """One week of the captured-day series (the report's own bucketing)."""

    label: str
    start: date
    end: date
    days: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "label": self.label,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": self.days,
        }


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """The newest trained model artefact, as the filesystem reports it."""

    name: str
    horizon: int | None = None
    created: Any = None
    data_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "name": self.name,
            "horizon_minutes": self.horizon,
            "created": _stamp(self.created),
            "data_days": self.data_days,
        }


@dataclass(slots=True)
class AccumulationInputs:
    """Everything the reader found: the report model's only input.

    Deliberately a plain container of already-measured facts. The storage layer fills it, the
    builder below turns it into an :class:`AccumulationReport`, and nothing in between reads
    the database again - so a number is measured once and cannot drift between sections.
    """

    tables: list[TableProfile] = field(default_factory=list)
    coverage_days: list[CoverageDay] = field(default_factory=list)
    quality_counts: dict[str, int] = field(default_factory=dict)
    weekly: list[WeekCapture] = field(default_factory=list)
    newest_model: ModelRecord | None = None
    unavailable: list[str] = field(default_factory=list)

    def table(self, name: str) -> TableProfile | None:
        """The profile for one table name, or ``None`` when it was not read."""
        for profile in self.tables:
            if profile.table == name:
                return profile
        return None

    def rows_for(self, name: str) -> int:
        """Stored row count for one table (0 when it is absent - check ``unavailable``)."""
        profile = self.table(name)
        return profile.rows if profile is not None else 0


@dataclass(frozen=True, slots=True)
class ReadinessItem:
    """One line of the "is there enough data to retrain?" checklist.

    ``have``/``need`` are numbers the reader can compare at a glance; ``detail`` is the
    sentence that explains *why* the line exists. A requirement that could not be measured at
    all (no coverage ledger, no feature rows read) reports ``met=False`` with the reason in
    ``detail`` rather than silently passing.
    """

    key: str
    requirement: str
    have: Any
    need: Any
    met: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "key": self.key,
            "requirement": self.requirement,
            "have": self.have,
            "need": self.need,
            "met": self.met,
            "detail": self.detail,
        }

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        mark = "OK" if self.met else "MISSING"
        text = f"[{mark}] {self.requirement}: have {self.have}, need {self.need}"
        if self.detail:
            text += f" ({self.detail})"
        return text


@dataclass(slots=True)
class AccumulationReport:
    """Spec 29's weekly report: what was captured, how well, and whether a retrain is due."""

    policy: AccumulationPolicy
    as_of: date
    window_start: date
    window_end: date
    captured_days: int = 0
    trading_days_in_window: int | None = None
    days_since_last_capture: int | None = None
    tables: list[TableProfile] = field(default_factory=list)
    coverage_days: list[CoverageDay] = field(default_factory=list)
    quality_counts: dict[str, int] = field(default_factory=dict)
    weekly: list[WeekCapture] = field(default_factory=list)
    gate: GateStatus | None = None
    readiness: list[ReadinessItem] = field(default_factory=list)
    newest_model: ModelRecord | None = None
    retrain_recommended: bool = False
    retrain_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    # -- derived views -------------------------------------------------------
    @property
    def captured_days_in_window(self) -> int:
        """How many captured days fall inside the report window."""
        return sum(
            1
            for day in self.coverage_days
            if self.window_start <= day.trading_date <= self.window_end
        )

    @property
    def gap_days(self) -> list[CoverageDay]:
        """The days that recorded at least one gap, most gaps first."""
        days = [day for day in self.coverage_days if day.has_gap]
        return sorted(days, key=lambda day: (-day.gap_count, day.trading_date))

    @property
    def incomplete_days(self) -> list[CoverageDay]:
        """Captured days that are not complete by the policy's ratio floor, oldest first."""
        return [
            day
            for day in self.coverage_days
            if not day.is_complete(self.policy.healthy_coverage_ratio)
        ]

    @property
    def latest_capture(self) -> date | None:
        """The newest captured trading day, or ``None`` when nothing was captured."""
        if not self.coverage_days:
            return None
        return max(day.trading_date for day in self.coverage_days)

    @property
    def missing_requirements(self) -> list[ReadinessItem]:
        """The readiness lines that are not satisfied yet, in order."""
        return [item for item in self.readiness if not item.met]

    @property
    def ready_for_retraining(self) -> bool:
        """True when every data-side requirement is satisfied (not a claim about quality)."""
        return bool(self.readiness) and all(item.met for item in self.readiness)

    def headline(self) -> str:
        """One sentence an operator can paste into a standup."""
        text = f"{self.captured_days} captured trading day(s)"
        if self.gate is not None:
            text += f" [{self.gate.stage}]"
        if self.days_since_last_capture is not None:
            text += f", last capture {self.days_since_last_capture} day(s) ago"
        text += "; retrain recommended" if self.retrain_recommended else "; no retrain due"
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - the whole report, samples and caveats included."""
        return {
            "report": "spec 29 - continuous data-accumulation report",
            "spec": self.policy.spec,
            "as_of": self.as_of.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "policy": self.policy.to_dict(),
            "captured_days": self.captured_days,
            "captured_days_in_window": self.captured_days_in_window,
            "trading_days_in_window": self.trading_days_in_window,
            "days_since_last_capture": self.days_since_last_capture,
            "latest_capture": self.latest_capture.isoformat() if self.latest_capture else None,
            "gate": self.gate.to_dict() if self.gate is not None else None,
            "tables": [profile.to_dict() for profile in self.tables],
            "coverage": [day.to_dict() for day in self.coverage_days],
            "gap_days": [day.to_dict() for day in self.gap_days],
            "quality_counts": dict(self.quality_counts),
            "weekly": [week.to_dict() for week in self.weekly],
            "readiness": [item.to_dict() for item in self.readiness],
            "ready_for_retraining": self.ready_for_retraining,
            "missing_requirements": [item.key for item in self.missing_requirements],
            "newest_model": self.newest_model.to_dict() if self.newest_model else None,
            "retrain_recommended": self.retrain_recommended,
            "retrain_reason": self.retrain_reason,
            "warnings": list(self.warnings),
            "unavailable": list(self.unavailable),
            "caveats": list(self.caveats),
        }

    def to_markdown(self) -> str:
        """The report as a Markdown document - the shape a human reads first."""
        return render_markdown(self)


def _as_date(value: Any) -> date | None:
    """Coerce whatever a reader hands back (date, datetime, ISO string, mtime) to a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).date()
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _stamp(value: Any) -> str | None:
    """An ISO timestamp for the JSON, or ``None`` when there is nothing to print."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _readiness(
    inputs: AccumulationInputs,
    policy: AccumulationPolicy,
    *,
    captured_days: int,
    coverage_available: bool,
) -> list[ReadinessItem]:
    """Spec 29's "readiness for retraining" checklist - one line per requirement.

    Every line is a *data* fact with a ``have`` and a ``need``, so "not ready" always comes
    with the number that is missing. The gate line reuses spec 4.3's own ladder; the coverage
    line fails on any recorded gap, because a gap is the one coverage failure an operator can
    still act on (re-capture) before it is baked into every number downstream.
    """
    gap_free = sum(1 for day in inputs.coverage_days if not day.has_gap)
    if not coverage_available:
        coverage_detail = "no coverage ledger could be read (data_coverage)"
        coverage_met = False
    elif captured_days == 0:
        coverage_detail = "nothing has been captured yet"
        coverage_met = False
    else:
        coverage_detail = f"{gap_free} of {captured_days} captured day(s) recorded no gap"
        coverage_met = gap_free == captured_days

    features = inputs.table("features")
    labels = inputs.table("labels")

    def row_item(key: str, table: str, need: int, profile: TableProfile | None) -> ReadinessItem:
        if profile is None or profile.unavailable is not None:
            return ReadinessItem(
                key=key,
                requirement=f"at least {need} stored {table} row(s)",
                have=0,
                need=need,
                met=False,
                detail=f"the {table} table could not be read",
            )
        return ReadinessItem(
            key=key,
            requirement=f"at least {need} stored {table} row(s)",
            have=profile.rows,
            need=need,
            met=profile.rows >= need,
            detail=f"{profile.rows} row(s) stored",
        )

    return [
        ReadinessItem(
            key="captured_days",
            requirement=f"spec 4.3's ML-training gate ({policy.ml_training_min_days} days)",
            have=captured_days,
            need=policy.ml_training_min_days,
            met=captured_days >= policy.ml_training_min_days,
            detail=f"gate stage {policy.gate_status(captured_days).stage}",
        ),
        ReadinessItem(
            key="coverage",
            requirement="captured days recorded no coverage gaps",
            have=gap_free if coverage_available else 0,
            need=captured_days,
            met=coverage_met,
            detail=coverage_detail,
        ),
        row_item("feature_rows", "features", policy.retrain_min_feature_rows, features),
        row_item("label_rows", "labels", policy.retrain_min_label_rows, labels),
    ]


def _warnings(
    inputs: AccumulationInputs,
    policy: AccumulationPolicy,
    *,
    captured_days: int,
    days_since_last_capture: int | None,
    gate: GateStatus,
) -> list[str]:
    """The things a reader must not miss, worst first. Never a claim about the market."""
    messages: list[str] = []
    coverage = inputs.table("data_coverage")

    if coverage is not None and coverage.unavailable is not None:
        messages.append(
            f"the coverage ledger could not be read ({coverage.unavailable}): the day counts "
            "and gap analysis below are incomplete"
        )
    elif captured_days == 0:
        messages.append(
            "no trading day has been captured yet: run `--mode capture` on a trading day, "
            "then re-read this report"
        )

    if days_since_last_capture is not None and days_since_last_capture > policy.stale_days:
        messages.append(
            f"the newest captured day is {days_since_last_capture} day(s) old "
            f"(stale after {policy.stale_days}): capture may have stopped"
        )

    gap_days = [day for day in inputs.coverage_days if day.has_gap]
    if gap_days:
        worst = max(gap_days, key=lambda day: day.gap_count)
        messages.append(
            f"{len(gap_days)} captured day(s) recorded coverage gap(s); worst is "
            f"{worst.trading_date.isoformat()} with {worst.gap_count} gap(s) "
            f"(largest {worst.largest_gap_seconds:.0f}s)"
        )

    below = [
        day
        for day in inputs.coverage_days
        if not day.is_complete(policy.healthy_coverage_ratio) and day.min_ratio is not None
    ]
    if below:
        messages.append(
            f"{len(below)} captured day(s) fall below the "
            f"{policy.healthy_coverage_ratio:g} coverage floor (a day can have ticks and "
            "still have missing bars)"
        )

    if gate.stage == "INSUFFICIENT":
        messages.append(
            "spec 4.3: too little data to conclude anything - keep capturing before "
            "reading anything into a report"
        )

    if not any(profile.present for profile in inputs.tables):
        messages.append("no stored table could be read: is the database present and unlocked?")

    messages.extend(inputs.unavailable)
    return messages


def _recommendation(
    policy: AccumulationPolicy,
    *,
    as_of: date,
    ready: bool,
    missing: list[ReadinessItem],
    newest_model: ModelRecord | None,
    latest_capture: date | None,
) -> tuple[bool, str]:
    """Whether a retrain is *due*, and the one sentence that says why (spec 29)."""
    if not ready:
        names = ", ".join(item.key for item in missing)
        return False, f"not ready: {names}"
    if newest_model is None:
        return True, "no model artefact exists yet"

    reasons: list[str] = []
    model_date = _as_date(newest_model.created)
    if latest_capture is not None and model_date is not None and latest_capture > model_date:
        reasons.append(
            f"captured data ({latest_capture.isoformat()}) is newer than the newest model "
            f"({model_date.isoformat()})"
        )
    if model_date is not None:
        age = (as_of - model_date).days
        if age >= policy.retrain_max_model_age_days:
            reasons.append(
                f"the newest model is {age} day(s) old (limit "
                f"{policy.retrain_max_model_age_days})"
            )
    if not reasons:
        return False, "the newest model is current with the captured data"
    return True, "; ".join(reasons)


def build_accumulation_report(
    inputs: AccumulationInputs,
    policy: AccumulationPolicy,
    *,
    as_of: date,
    window_weeks: int | None = None,
    trading_days_in_window: int | None = None,
) -> AccumulationReport:
    """Turn measured inputs into spec 29's report (pure: no I/O, no clock, no randomness).

    ``as_of`` is passed in rather than read from the clock so a report is reproducible and a
    test can pin it. The window is the last ``window_weeks`` weeks ending on ``as_of`` (today
    inclusive), which is what makes the report *weekly*: run it every Monday and the window is
    the week that just ended plus the three before it.
    """
    weeks = max(1, int(window_weeks if window_weeks is not None else policy.window_weeks))
    window_end = as_of
    window_start = as_of - timedelta(days=weeks * WEEK_DAYS - 1)

    coverage_days = sorted(inputs.coverage_days, key=lambda day: day.trading_date)
    captured_days = len(coverage_days)
    latest_capture = max((day.trading_date for day in coverage_days), default=None)
    days_since_last_capture = (as_of - latest_capture).days if latest_capture is not None else None

    gate = policy.gate_status(captured_days)
    ledger = inputs.table("data_coverage")
    coverage_available = ledger is not None and ledger.unavailable is None
    readiness = _readiness(
        inputs, policy, captured_days=captured_days, coverage_available=coverage_available
    )
    ready = bool(readiness) and all(item.met for item in readiness)
    recommended, reason = _recommendation(
        policy,
        as_of=as_of,
        ready=ready,
        missing=[item for item in readiness if not item.met],
        newest_model=inputs.newest_model,
        latest_capture=latest_capture,
    )

    return AccumulationReport(
        policy=policy,
        as_of=as_of,
        window_start=window_start,
        window_end=window_end,
        captured_days=captured_days,
        trading_days_in_window=trading_days_in_window,
        days_since_last_capture=days_since_last_capture,
        tables=list(inputs.tables),
        coverage_days=coverage_days,
        quality_counts=dict(inputs.quality_counts),
        weekly=list(inputs.weekly),
        gate=gate,
        readiness=readiness,
        newest_model=inputs.newest_model,
        retrain_recommended=recommended,
        retrain_reason=reason,
        warnings=_warnings(
            inputs,
            policy,
            captured_days=captured_days,
            days_since_last_capture=days_since_last_capture,
            gate=gate,
        ),
        unavailable=list(inputs.unavailable),
        caveats=list(ACCUMULATION_CAVEATS),
    )


def _number(value: int | float | None) -> str:
    """A count with thousands separators, or an em dash when there is nothing to print."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return f"{value:,}"


def _window_text(report: AccumulationReport) -> str:
    """The window sentence, shared by the console block and the Markdown header."""
    text = (
        f"{report.window_start.isoformat()} .. {report.window_end.isoformat()} "
        f"({report.policy.window_weeks} week(s), "
        f"{report.captured_days_in_window} captured day(s)"
    )
    if report.trading_days_in_window is not None:
        text += f" of {report.trading_days_in_window} trading day(s)"
    return text + ")"


def render_lines(report: AccumulationReport) -> list[str]:
    """The console block: one ``key : value`` line per fact, greppable and copy-pasteable."""
    lines: list[str] = [
        f"ANJANEYA-AI accumulation report (spec {report.policy.spec}) - "
        f"as of {report.as_of.isoformat()}",
        f"  headline   : {report.headline()}",
        f"  window     : {_window_text(report)}",
    ]
    if report.gate is not None:
        lines.append(f"  gate       : {report.gate.describe()}")

    lines.append("  -- sample sizes (stored rows) --")
    for profile in report.tables:
        if profile.unavailable is not None:
            lines.append(f"  {profile.table:<10} : unavailable ({profile.unavailable})")
            continue
        span = ""
        if profile.first is not None or profile.last is not None:
            span = f" ({_stamp(profile.first)} .. {_stamp(profile.last)})"
        size = (
            f"{_number(profile.rows)} row(s) over {_number(profile.days)} day(s)"
            if profile.days is not None
            else f"{_number(profile.rows)} row(s)"
        )
        lines.append(f"  {profile.table:<10} : {size}{span}")

    lines.append("  -- coverage (data_coverage) --")
    lines.append(f"  days       : {_number(report.captured_days)} captured")
    gap_days = report.gap_days
    if gap_days:
        worst = gap_days[0]
        lines.append(
            f"  gaps       : {len(gap_days)} day(s) with gaps; worst "
            f"{worst.trading_date.isoformat()} ({worst.gap_count} gap(s))"
        )
    else:
        lines.append("  gaps       : none recorded")
    if report.quality_counts:
        lines.append(f"  quality    : {report.quality_counts}")
    below = [day for day in report.incomplete_days if day.min_ratio is not None]
    lines.append(
        f"  below floor: {len(below)} day(s) under "
        f"{report.policy.healthy_coverage_ratio:g}"
    )

    if report.weekly:
        lines.append("  -- captured days by week --")
        for week in report.weekly:
            lines.append(
                f"  {week.start.isoformat()} .. {week.end.isoformat()} : "
                f"{_number(week.days)} day(s)"
            )

    lines.append("  -- readiness for retraining --")
    for item in report.readiness:
        lines.append(f"  {item.describe()}")
    lines.append(f"  ready      : {'yes' if report.ready_for_retraining else 'no'}")

    lines.append("  -- retrain --")
    if report.newest_model is not None:
        lines.append(
            f"  newest     : {report.newest_model.name} "
            f"(horizon {report.newest_model.horizon_minutes}, "
            f"created {_stamp(report.newest_model.created)})"
        )
    else:
        lines.append("  newest     : none")
    lines.append(
        f"  recommended: {'yes' if report.retrain_recommended else 'no'} - "
        f"{report.retrain_reason}"
    )

    if report.warnings:
        lines.append("  -- warnings --")
        for message in report.warnings:
            lines.append(f"  - {message}")

    lines.append("  -- caveats --")
    for caveat in report.caveats:
        lines.append(f"  * {caveat}")
    return lines


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """A Markdown table, or a single italic line when there is nothing to tabulate."""
    if not rows:
        return ["_nothing to report yet_", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    lines.append("")
    return lines


def render_markdown(report: AccumulationReport) -> str:
    """The whole report as Markdown: spec 25's five sections, caveats last.

    Every table carries its own sample - a row count is never printed without the span it
    came from - and the readiness checklist prints the numbers it compared, so a reader cannot
    take "ready" on trust without seeing ``have`` and ``need``.
    """
    lines: list[str] = [
        f"# ANJANEYA accumulation report (spec {report.policy.spec}) - continuous data "
        "accumulation",
        "",
        f"**As of:** {report.as_of.isoformat()}  ",
        f"**Window:** {_window_text(report)}  ",
        f"**Captured days:** {report.captured_days}"
        + (f" [{report.gate.stage}]" if report.gate is not None else "")
        + "  ",
    ]
    if report.days_since_last_capture is not None:
        lines.append(f"**Last capture:** {report.days_since_last_capture} day(s) ago  ")
    if report.gate is not None:
        lines.append(f"**Gate:** {report.gate.describe()}  ")
    lines.append(
        f"**Retrain:** {'recommended' if report.retrain_recommended else 'not due'} - "
        f"{report.retrain_reason}  "
    )
    lines.append("")

    lines.extend(["## 1. Data captured (sample sizes)", ""])
    lines.extend(
        _table(
            ["table", "rows", "days", "first", "last"],
            [
                [
                    profile.table,
                    "unavailable" if profile.unavailable is not None else _number(profile.rows),
                    _number(profile.days) if profile.days is not None else "-",
                    _stamp(profile.first) if profile.unavailable is None else "-",
                    _stamp(profile.last) if profile.unavailable is None else "-",
                ]
                for profile in report.tables
            ],
        )
    )

    lines.extend(["## 2. Coverage", ""])
    gap_days = report.gap_days
    below = [day for day in report.incomplete_days if day.min_ratio is not None]
    lines.extend(
        _table(
            ["captured days", "days with gaps", "days below floor", "quality flags"],
            [
                [
                    _number(report.captured_days),
                    len(gap_days),
                    len(below),
                    ", ".join(f"{name}={count}" for name, count in report.quality_counts.items())
                    or "-",
                ],
            ],
        )
    )
    lines.extend(["### Days with gaps", ""])
    lines.extend(
        _table(
            ["trading date", "streams", "gap count", "largest gap (s)", "min ratio"],
            [
                [
                    day.trading_date.isoformat(),
                    day.streams,
                    day.gap_count,
                    f"{day.largest_gap_seconds:.1f}",
                    "-" if day.min_ratio is None else f"{day.min_ratio:.3f}",
                ]
                for day in gap_days
            ],
        )
    )

    if report.weekly:
        lines.extend(["## 3. Captured days by week", ""])
        lines.extend(
            _table(
                ["week start", "week end", "days"],
                [[week.start.isoformat(), week.end.isoformat(), week.days] for week in report.weekly],
            )
        )

    lines.extend(["## 4. Readiness for retraining", ""])
    lines.extend(
        _table(
            ["requirement", "have", "need", "met", "detail"],
            [
                [item.requirement, item.have, item.need, "yes" if item.met else "no", item.detail]
                for item in report.readiness
            ],
        )
    )
    lines.append(f"**Ready:** {'yes' if report.ready_for_retraining else 'no'}  ")
    if report.newest_model is not None:
        lines.append(
            f"**Newest model:** {report.newest_model.name} "
            f"(horizon {report.newest_model.horizon_minutes}, created "
            f"{_stamp(report.newest_model.created)})  "
        )
    else:
        lines.append("**Newest model:** none  ")
    lines.append("")

    if report.warnings:
        lines.extend(["## 5. Warnings", ""])
        lines.extend(f"* {message}" for message in report.warnings)
        lines.append("")

    lines.extend(["## Caveats", ""])
    lines.extend(f"* {caveat}" for caveat in report.caveats)
    lines.append("")
    return "\n".join(lines)




