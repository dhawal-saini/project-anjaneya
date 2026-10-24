"""Unit tests for ``core/research/report.py`` (spec 24's ten answers in one document).

The report is the artefact a reader trusts without re-running anything, so these tests check
the four promises it makes:

* **all ten items are present.** ``RESEARCH_ITEMS`` is iterated and each item answers
  *something* - and the two that need index bars answer with a stated reason when there are
  none, because "not measured" and "measured as zero" must never look the same.
* **the policy travels with the numbers.** The sample rule, the correction and the alpha are in
  the JSON, so a quote cannot be separated from the rule that produced it.
* **the caveats are fields, not folklore.** The threshold signal, the seasonality rule and the
  gate are in ``caveats`` and printed under the numbers in the Markdown.
* **the Markdown is renderable from any ledger**, including an empty one.
"""

from __future__ import annotations

import json

import pytest

from core.backtest import MetricsError
from core.research import (
    PRELIMINARY_LABEL,
    RESEARCH_ITEMS,
    ReportPolicy,
    build_research_report,
)
from tests.unit.test_research_profile import OTHER, SESSION, bars, calendar
from tests.unit.test_research_segments import ledger, trade

POLICY = ReportPolicy(preliminary_min_samples=5, seasonality_min_days=3)


def report_with(ledger_rows=None, **kwargs):
    """A report over the fixture ledger (no bars unless the caller brings them)."""
    rows = ledger(4) if ledger_rows is None else ledger_rows
    return build_research_report(rows, policy=POLICY, **kwargs)


def test_every_one_of_spec_24s_ten_items_is_present() -> None:
    report = report_with()

    assert len(RESEARCH_ITEMS) == 10
    for name in RESEARCH_ITEMS:
        payload = report.item(name)
        if name == "7_volatility_profile":
            assert payload is None and report.volatility_unavailable  # absent, and said so
        elif name == "8_intraday_path":
            assert payload == [] and report.paths_unavailable
        else:
            assert payload is not None, name
    assert [matrix.name for matrix in report.matrices] == [
        "time_of_day_x_option_type",
        "time_of_day_x_cycle_phase",
        "regime_x_time_of_day",
        "regime_x_cycle_phase",
    ]
    assert [table.name for table in report.segment_tables] == ["day_of_week", "days_to_expiry"]
    assert report.holding_time is not None
    assert [summary.matrix for summary in report.significance] == [
        matrix.name for matrix in report.matrices
    ]


def test_an_unknown_item_name_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown research item"):
        report_with().item("11_something_else")


def test_without_index_bars_the_market_items_state_why_they_are_absent() -> None:
    report = report_with()

    assert report.volatility is None and report.paths == []
    assert "not measured" in report.volatility_unavailable
    assert report.paths_unavailable == report.volatility_unavailable
    payload = report.to_dict()
    assert payload["volatility"] is None
    assert payload["volatility_unavailable"] is not None
    assert payload["intraday_paths"] == []


def test_the_policy_and_the_samples_travel_with_the_numbers() -> None:
    report = report_with()

    payload = json.loads(json.dumps(report.to_dict(), default=str))

    assert payload["policy"]["preliminary_min_samples"] == 5
    assert payload["policy"]["multiple_testing_correction"] == "benjamini_hochberg"
    assert payload["policy"]["significance_level"] == 0.05
    assert payload["trades"] == 4
    assert payload["testable_cells"] == 0  # four trades, five needed on both sides
    assert payload["preliminary_cells"]["time_of_day_x_option_type"] == [
        "MORNING_930_1100 | CE",
        "MORNING_930_1100 | PE",
    ]


def test_a_cell_below_the_rule_is_never_tested_and_says_so_in_the_caveats() -> None:
    report = report_with()

    summary = report.significance[0]
    assert summary.family_size == 0
    assert all(test.preliminary for test in summary.tests)
    assert any("no p-value to report at all" in caveat for caveat in report.caveats)


def test_the_seasonality_gate_is_a_caveat_when_the_data_is_young() -> None:
    young = report_with(captured_days=1)
    old = report_with(captured_days=500)

    assert any("seasonality claim" in caveat for caveat in young.caveats)
    assert not any("seasonality claim" in caveat for caveat in old.caveats)
    assert any("captured days was not supplied" in caveat for caveat in report_with().caveats)


def test_the_gate_status_is_carried_when_it_is_supplied() -> None:
    class Gate:
        """A stand-in for ``core.ml.GateStatus`` (duck-typed on purpose)."""

        def to_dict(self):
            return {"stage": "INSUFFICIENT", "allowed": "exploration only"}

    report = report_with(gate=Gate())

    assert report.to_dict()["gate"]["stage"] == "INSUFFICIENT"


def test_the_market_items_are_built_when_the_bars_are_there() -> None:
    frame = bars(days=(SESSION, OTHER), bars_per_day=35)

    report = report_with(bars=frame, calendar=calendar(), captured_days=2)

    assert report.volatility is not None
    assert report.volatility.symbol == "NIFTY 50"
    assert report.volatility_unavailable is None
    assert [path.bucket for path in report.paths] == ["OPEN_915_945", "REST_945_1530"]
    assert report.paths_unavailable is None


def test_the_markdown_prints_the_ten_items_and_the_caveats() -> None:
    frame = bars(days=(SESSION, OTHER), bars_per_day=35)
    report = report_with(bars=frame, calendar=calendar(), captured_days=2)

    markdown = report.to_markdown()

    for name in RESEARCH_ITEMS:
        assert f"## {name}." in markdown
    assert "## Caveats" in markdown
    assert PRELIMINARY_LABEL in markdown
    for caveat in report.caveats:
        assert caveat in markdown
    assert "| time_of_day_bucket | CE | PE |" in markdown


def test_an_empty_ledger_still_renders_a_report() -> None:
    report = build_research_report([], policy=POLICY)

    markdown = report.to_markdown()

    assert report.trades == 0
    assert report.days == 0
    assert "## Caveats" in markdown
    assert "_no trades in this cut - the cells are absent, not zero_" in markdown
    assert report.significant_cells == {}


def test_a_malformed_ledger_is_refused_before_any_table_is_built() -> None:
    rows = ledger(1)
    rows[0].exit_timestamp = None
    rows[0].net_pnl = None

    with pytest.raises(MetricsError, match="closed trades only"):
        build_research_report(rows, policy=POLICY)


def test_the_report_counts_the_entry_days_not_the_bars() -> None:
    rows = [trade(index=0), trade(index=1, day=OTHER)]

    report = build_research_report(rows, policy=POLICY, captured_days=2)

    assert report.days == 2
    assert report.first_entry is not None and report.last_exit is not None
    assert not any("one session's shape" in caveat for caveat in report.caveats)

