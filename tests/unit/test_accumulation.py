"""Unit tests for spec 29's accumulation report (``core.accumulation``).

The report is pure, so every one of its rules is testable without a database: the window, the
gate, the gap analysis, the readiness checklist and the retrain recommendation are all
functions of the inputs a caller hands in. That is the point of the module, and these tests
hold it to it.

The four things worth asserting, over and over:

* the gate is spec 4.3's, re-read rather than re-implemented (one ladder, one verdict);
* a captured day is not a complete day (a gap fails the coverage requirement);
* "ready" and "recommended" are different questions with different reasons;
* nothing is invented - a number is printed only when it was measured.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from core.accumulation import (
    ACCUMULATION_CAVEATS,
    AccumulationInputs,
    AccumulationPolicy,
    CoverageDay,
    ModelRecord,
    TableProfile,
    WeekCapture,
    build_accumulation_report,
    render_lines,
    render_markdown,
)

BASE = date(2026, 9, 1)


def day(offset: int) -> date:
    """A date ``offset`` days after the fixture's base day."""
    return BASE + timedelta(days=offset)


def coverage_day(
    offset: int,
    *,
    streams: int = 4,
    complete: int = 4,
    gap_streams: int = 0,
    missing: int = 0,
    suspect: int = 0,
    gap_count: int = 0,
    largest: float = 0.0,
    min_ratio: float | None = 1.0,
) -> CoverageDay:
    """One ledger day, with sane defaults a single keyword can bend."""
    return CoverageDay(
        trading_date=day(offset),
        streams=streams,
        complete_streams=complete,
        gap_streams=gap_streams,
        missing_streams=missing,
        suspect_streams=suspect,
        min_ratio=min_ratio,
        gap_count=gap_count,
        largest_gap_seconds=largest,
    )


def inputs(
    *,
    days: int = 0,
    tables: list[TableProfile] | None = None,
    coverage: list[CoverageDay] | None = None,
    model: ModelRecord | None = None,
    weeks: int = 4,
) -> AccumulationInputs:
    """An :class:`AccumulationInputs` with ``days`` clean captured days, unless told otherwise."""
    if coverage is None:
        coverage = [coverage_day(index) for index in range(days)]
    profiles = tables
    if profiles is None:
        profiles = [
            TableProfile(table="data_coverage", rows=len(coverage), days=len(coverage)),
            TableProfile(table="features", rows=100_000, days=len(coverage)),
            TableProfile(table="labels", rows=100_000, days=len(coverage)),
        ]
    return AccumulationInputs(
        tables=profiles,
        coverage_days=coverage,
        weekly=[
            WeekCapture(label=f"w{index}", start=day(index * 7), end=day(index * 7 + 6), days=0)
            for index in range(weeks)
        ],
        newest_model=model,
    )


def build(
    data: AccumulationInputs,
    policy: AccumulationPolicy | None = None,
    *,
    as_of: date | None = None,
    model: ModelRecord | None = None,
    **kwargs,
):
    """Build a report with a pinned ``as_of`` so nothing depends on the wall clock."""
    if model is not None:
        data.newest_model = model
    return build_accumulation_report(
        data,
        policy or AccumulationPolicy(),
        as_of=as_of or day(29),
        **kwargs,
    )


def small_gate_policy(**overrides) -> AccumulationPolicy:
    """A policy whose gates are small enough for a ~20-day fixture to satisfy them."""
    values: dict = {
        "exploratory_min_days": 5,
        "ml_training_min_days": 10,
        "meaningful_backtest_min_days": 15,
        "seasonality_min_days": 20,
    }
    values.update(overrides)
    return AccumulationPolicy(**values)


# ---------------------------------------------------------------------------
# The small pieces
# ---------------------------------------------------------------------------
def test_table_profile_present_and_empty() -> None:
    assert TableProfile(table="ticks", rows=5).present is True
    assert TableProfile(table="ticks", rows=5).empty is False
    assert TableProfile(table="ticks", rows=0).empty is True
    assert TableProfile(table="ticks", unavailable="no such table").present is False


def test_coverage_day_is_complete_needs_a_ratio() -> None:
    """A stream with no expectation (``min_ratio is None``) is not evidence of completeness."""
    assert coverage_day(0).is_complete(0.9) is True
    assert coverage_day(0, min_ratio=0.5).is_complete(0.9) is False
    assert coverage_day(0, min_ratio=None).is_complete(0.9) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap_count": 1},
        {"missing": 1},
        {"suspect": 1},
    ],
)
def test_coverage_day_is_incomplete_for_any_problem(kwargs: dict) -> None:
    assert coverage_day(0, **kwargs).is_complete(0.9) is False


def test_policy_defaults_mirror_the_shipped_config(config) -> None:
    """The dataclass defaults and ``config/model.yaml`` describe one policy, not two."""
    from_config = AccumulationPolicy.from_config(config)
    assert from_config == AccumulationPolicy()


def test_from_config_reads_the_gate_values(config) -> None:
    policy = AccumulationPolicy.from_config(config)
    gates = config.model.data_gates
    assert policy.ml_training_min_days == gates.ml_training_min_days
    assert policy.seasonality_min_days == gates.seasonality_min_days


@pytest.mark.parametrize(
    ("days", "stage"),
    [(0, "INSUFFICIENT"), (20, "EXPLORATORY"), (60, "ML_TRAINING"), (120, "MEANINGFUL_BACKTEST")],
)
def test_gate_status_reuses_spec_4_3s_ladder(days: int, stage: str) -> None:
    assert AccumulationPolicy().gate_status(days).stage == stage


# ---------------------------------------------------------------------------
# The window and the day counts
# ---------------------------------------------------------------------------
def test_the_window_is_the_last_n_weeks_ending_on_as_of() -> None:
    report = build(inputs(), AccumulationPolicy(window_weeks=4), as_of=day(29))
    assert report.window_end == day(29)
    assert report.window_start == day(29) - timedelta(days=27)


def test_weeks_override_narrows_the_window() -> None:
    report = build(inputs(), as_of=day(29), window_weeks=1)
    assert report.window_start == day(29) - timedelta(days=6)


def test_captured_days_counts_the_coverage_ledger() -> None:
    assert build(inputs(days=7)).captured_days == 7


def test_captured_days_in_window_filters_old_days() -> None:
    data = inputs(coverage=[coverage_day(-100), coverage_day(28), coverage_day(29)])
    report = build(data, as_of=day(29))
    assert report.captured_days == 3
    assert report.captured_days_in_window == 2


def test_days_since_last_capture_uses_the_newest_day() -> None:
    report = build(inputs(days=3), as_of=day(29))
    assert report.latest_capture == day(2)
    assert report.days_since_last_capture == 27


def test_days_since_last_capture_is_none_without_any_day() -> None:
    report = build(inputs())
    assert report.latest_capture is None
    assert report.days_since_last_capture is None


def test_weekly_buckets_come_from_the_inputs() -> None:
    report = build(inputs(weeks=2))
    assert [week.label for week in report.weekly] == ["w0", "w1"]


def test_gap_days_are_ordered_by_gap_count() -> None:
    data = inputs(
        coverage=[
            coverage_day(0, gap_count=1, gap_streams=1),
            coverage_day(1, gap_count=5, gap_streams=2),
            coverage_day(2),
        ]
    )
    report = build(data)
    assert [item.trading_date for item in report.gap_days] == [day(1), day(0)]


def test_incomplete_days_ignore_ratio_less_streams() -> None:
    """A day with a gap is incomplete; a day whose streams carry no expectation is reported too."""
    data = inputs(coverage=[coverage_day(0, min_ratio=None), coverage_day(1), coverage_day(2)])
    report = build(data)
    assert [item.trading_date for item in report.incomplete_days] == [day(0)]


# ---------------------------------------------------------------------------
# Readiness for retraining
# ---------------------------------------------------------------------------
def test_readiness_has_the_four_requirements() -> None:
    report = build(inputs(days=70))
    assert [item.key for item in report.readiness] == [
        "captured_days",
        "coverage",
        "feature_rows",
        "label_rows",
    ]


def test_readiness_gate_line_uses_the_ml_threshold() -> None:
    report = build(inputs(days=70))
    gate = next(item for item in report.readiness if item.key == "captured_days")
    assert gate.need == AccumulationPolicy().ml_training_min_days
    assert gate.met is True


def test_readiness_coverage_line_fails_on_any_gap() -> None:
    data = inputs(coverage=[coverage_day(0), coverage_day(1, gap_count=2, gap_streams=1)])
    report = build(data)
    coverage = next(item for item in report.readiness if item.key == "coverage")
    assert coverage.met is False
    assert coverage.have == 1
    assert coverage.need == 2


def test_readiness_row_lines_fail_when_the_table_could_not_be_read() -> None:
    data = inputs(
        tables=[TableProfile(table="features", unavailable="no such table: features")],
        days=3,
    )
    report = build(data)
    rows = next(item for item in report.readiness if item.key == "feature_rows")
    assert rows.met is False
    assert "could not be read" in rows.detail


def test_ready_for_retraining_requires_every_line() -> None:
    assert build(inputs(days=70)).ready_for_retraining is True
    assert build(inputs(days=5)).ready_for_retraining is False


def test_ready_for_retraining_is_false_with_no_readiness_at_all() -> None:
    report = build(inputs(weeks=0))
    report.readiness = []
    assert report.ready_for_retraining is False


def test_missing_requirements_lists_the_keys() -> None:
    report = build(inputs(days=5))
    assert report.to_dict()["missing_requirements"] == ["captured_days"]


# ---------------------------------------------------------------------------
# The retrain recommendation
# ---------------------------------------------------------------------------
def test_recommendation_is_not_due_and_names_what_is_missing() -> None:
    report = build(inputs(days=5))
    assert report.retrain_recommended is False
    assert report.retrain_reason.startswith("not ready:")
    assert "captured_days" in report.retrain_reason


def test_recommendation_is_due_once_ready_with_no_model() -> None:
    report = build(inputs(days=70), as_of=day(29))
    assert report.retrain_recommended is True
    assert report.retrain_reason == "no model artefact exists yet"


def test_recommendation_is_due_when_data_is_newer_than_the_model() -> None:
    model = ModelRecord(name="m.txt", horizon=5, created=datetime(2026, 9, 10, 12, 0))
    report = build(inputs(days=70), as_of=day(29), model=model)
    assert report.retrain_recommended is True
    assert "newer than the newest model" in report.retrain_reason


def test_recommendation_is_due_when_the_model_is_old() -> None:
    data = inputs(days=20)  # newest captured day is day(19) = 2026-09-20
    model = ModelRecord(name="m.txt", horizon=5, created=datetime(2026, 9, 20, 12, 0))
    report = build(
        data,
        small_gate_policy(retrain_max_model_age_days=5),
        as_of=day(40),
        model=model,
    )
    assert report.retrain_recommended is True
    assert "day(s) old" in report.retrain_reason


def test_recommendation_is_not_due_when_the_model_is_current() -> None:
    data = inputs(days=20)  # newest captured day is day(19) = 2026-09-20
    model = ModelRecord(name="m.txt", horizon=5, created=datetime(2026, 9, 20, 12, 0))
    report = build(data, small_gate_policy(), as_of=day(19), model=model)
    assert report.retrain_recommended is False
    assert report.retrain_reason == "the newest model is current with the captured data"


# ---------------------------------------------------------------------------
# Warnings and caveats
# ---------------------------------------------------------------------------
def test_warning_when_capture_is_stale() -> None:
    report = build(inputs(days=3), as_of=day(29))
    assert any("may have stopped" in message for message in report.warnings)


def test_warning_when_nothing_was_captured() -> None:
    report = build(inputs())
    assert any("no trading day has been captured yet" in message for message in report.warnings)


def test_warning_when_the_ledger_could_not_be_read() -> None:
    data = inputs(
        tables=[TableProfile(table="data_coverage", unavailable="no such table: data_coverage")]
    )
    report = build(data)
    assert any("coverage ledger could not be read" in message for message in report.warnings)


def test_warning_when_days_have_gaps() -> None:
    data = inputs(coverage=[coverage_day(0, gap_count=3, gap_streams=1, largest=120.0)])
    report = build(data)
    assert any("coverage gap(s)" in message for message in report.warnings)


def test_warning_when_the_gate_is_insufficient() -> None:
    report = build(inputs(days=2))
    assert any("too little data to conclude anything" in message for message in report.warnings)


def test_warning_when_no_table_could_be_read() -> None:
    data = inputs(tables=[TableProfile(table="ticks", unavailable="locked")])
    report = build(data)
    assert any("is the database present and unlocked?" in message for message in report.warnings)


def test_caveats_are_always_present() -> None:
    report = build(inputs())
    assert report.caveats == list(ACCUMULATION_CAVEATS)
    assert report.to_dict()["caveats"] == list(ACCUMULATION_CAVEATS)


# ---------------------------------------------------------------------------
# The three renderings
# ---------------------------------------------------------------------------
def test_to_dict_carries_the_five_sections() -> None:
    payload = build(inputs(days=70)).to_dict()
    for key in ("tables", "coverage", "weekly", "readiness", "warnings", "caveats"):
        assert key in payload
    assert payload["report"] == "spec 29 - continuous data-accumulation report"
    assert payload["spec"] == "29"


def test_headline_names_the_day_count_and_the_gate() -> None:
    headline = build(inputs(days=70)).headline()
    assert "70 captured trading day(s)" in headline
    assert "ML_TRAINING" in headline
    assert "retrain recommended" in headline


def test_render_lines_has_the_sections() -> None:
    text = "\n".join(render_lines(build(inputs(days=70))))
    for heading in ("sample sizes", "coverage (data_coverage)", "readiness for retraining", "caveats"):
        assert heading in text
    assert "ready      : yes" in text


def test_render_markdown_has_the_five_sections() -> None:
    text = render_markdown(build(inputs(days=70)))
    assert text.startswith("# ANJANEYA accumulation report (spec 29)")
    for heading in (
        "## 1. Data captured",
        "## 2. Coverage",
        "## 4. Readiness for retraining",
        "## Caveats",
    ):
        assert heading in text


def test_render_markdown_marks_an_unavailable_table() -> None:
    data = inputs(tables=[TableProfile(table="ticks", unavailable="no such table: ticks")])
    text = render_markdown(build(data))
    assert "| ticks | unavailable |" in text


def test_table_to_dict_keeps_a_nullable_day_count() -> None:
    """A count-only table reports ``days = None`` - "we counted, there is no timestamp"."""
    assert TableProfile(table="instruments", rows=10).to_dict()["days"] is None

