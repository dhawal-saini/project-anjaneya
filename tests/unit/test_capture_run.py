"""Tests for the calendar advisory in ``apps/anjaneeya_capture/run.py`` and the status report.

The advisory exists because the calendar can silently cancel a *real* trading day: a holiday
date that is still marked ``verified: false`` closes the market in code, the capture session
exits ``NO_SESSION``, and the ticks of that session do not exist anywhere afterwards. The
warning therefore has to appear before a session starts (``--mode capture``, dry or live) and
in every app's ``--mode status`` - which is what these tests pin down.

Nothing here starts a session or touches the network: the advisory is a pure function of the
calendar, and the status report is exercised through the real ``AppContext`` on an isolated
data root.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from types import SimpleNamespace

import pytest

from apps.anjaneeya_capture.run import calendar_advisory
from core.app_bootstrap import status_report
from core.config_loader import CalendarConfig
from core.market_calendar import DEFAULT_HOLIDAY_RISK_DAYS, NseCalendar
from tests.support.engine import make_engine_context

TODAY = date(2026, 10, 1)


class PinnedCalendar(NseCalendar):
    """The real calendar with ``now()`` pinned, so the tests do not depend on today's date."""

    def __init__(self, config: CalendarConfig, *, today: date) -> None:
        super().__init__(config)
        self._today = today

    def now(self) -> datetime:
        return datetime.combine(self._today, time(8, 0), tzinfo=self.timezone)


def calendar_for(*, holidays: list[dict[str, object]], today: date = TODAY) -> PinnedCalendar:
    """A valid calendar whose holiday list is supplied by the test."""
    payload = {
        "timezone": "Asia/Kolkata",
        "sessions": {
            "regular": {"start": "09:15", "end": "15:30"},
            "half_day": {"start": "09:15", "end": "13:00"},
            "half_day_dates": [],
            "special_sessions": [],
        },
        "holidays": holidays,
        "time_of_day_buckets": [{"name": "OPEN", "start": "09:15", "end": "09:30"}],
    }
    return PinnedCalendar(CalendarConfig.model_validate(payload), today=today)


def context_for(calendar: NseCalendar) -> SimpleNamespace:
    """The smallest context :func:`calendar_advisory` reads."""
    return SimpleNamespace(calendar=calendar)


# ---------------------------------------------------------------------------
# The pre-session advisory
# ---------------------------------------------------------------------------
def test_no_provisional_holiday_means_no_advisory() -> None:
    calendar = calendar_for(holidays=[{"date": "2026-10-02", "name": "Confirmed", "verified": True}])
    assert calendar_advisory(context_for(calendar)) == []


def test_a_provisional_holiday_this_week_is_named_before_the_session() -> None:
    calendar = calendar_for(
        holidays=[
            {"date": "2026-10-02", "name": "Gandhi Jayanti", "verified": False},
            {"date": "2026-11-09", "name": "Far Away", "verified": False},
        ]
    )

    body = "\n".join(calendar_advisory(context_for(calendar)))

    assert f"within {DEFAULT_HOLIDAY_RISK_DAYS} day(s)" in body
    assert "2026-10-02" in body and "Gandhi Jayanti" in body and "(in 1 day(s))" in body
    assert "2026-11-09" not in body  # outside the window
    assert "NO_SESSION" in body  # the consequence, spelled out
    assert "scripts/check_calendar.py" in body  # and the way to fix it


def test_a_provisional_holiday_today_is_reported_as_today() -> None:
    calendar = calendar_for(
        holidays=[{"date": TODAY.isoformat(), "name": "Unconfirmed", "verified": False}]
    )

    body = "\n".join(calendar_advisory(context_for(calendar)))
    assert "(today)" in body


# ---------------------------------------------------------------------------
# --mode status carries the same warning
# ---------------------------------------------------------------------------
def test_status_report_carries_the_calendar_risk(config) -> None:
    report = status_report(make_engine_context(config, mode="status"))

    risk = report["calendar_risk"]
    assert risk["window_days"] == DEFAULT_HOLIDAY_RISK_DAYS
    assert risk["count"] == len(risk["unverified_holidays"])
    for entry in risk["unverified_holidays"]:
        assert entry["days_away"] >= 0
        assert entry["name"]


def test_status_report_risk_is_json_friendly(config) -> None:
    json.dumps(status_report(make_engine_context(config, mode="status")), default=str)


def test_the_status_report_does_not_change_the_existing_sections(config) -> None:
    """The advisory is additive: every key an operator already reads is still there."""
    report = status_report(make_engine_context(config, mode="status"))
    for key in ("app", "mode", "config", "calendar", "run_id", "dry_run", "credentials"):
        assert key in report


@pytest.mark.parametrize("mode", ["status", "dataset"])
def test_the_risk_block_does_not_depend_on_the_mode(config, mode: str) -> None:
    assert "calendar_risk" in status_report(make_engine_context(config, mode=mode))
