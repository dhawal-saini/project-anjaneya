"""Tests for ``scripts/refresh_calendar.py`` - the NSE holiday reconciliation.

The network path is stubbed; what matters is the *pure* half: parsing NSE's
``tradingDate`` rows, diffing them against the YAML, and rewriting the YAML's
``holidays:`` block without disturbing a single comment. All three are pinned here,
so a later NSE format change or a sloppy regex is caught offline.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts import refresh_calendar as refresh

PAYLOAD = {
    "FO": [
        {"tradingDate": "26-Jan-2026", "description": "Republic Day"},
        {"tradingDate": "15-Aug-2026", "description": "Independence Day"},
        {"tradingDate": "25-Dec-2026", "description": "Christmas"},
        {"tradingDate": "01-Jan-2027", "description": "New Year"},
    ]
}


def test_holidays_for_segment_normalises_dates_and_filters_the_year() -> None:
    holidays = refresh.holidays_for_segment(PAYLOAD, "FO", year=2026)
    assert [h.day.isoformat() for h in holidays] == ["2026-01-26", "2026-08-15", "2026-12-25"]
    assert holidays[0].name == "Republic Day"


def test_holidays_for_segment_ignores_malformed_rows() -> None:
    payload = {"FO": [{"tradingDate": "", "description": "x"}, {"badly": "shaped"}]}
    assert refresh.holidays_for_segment(payload, "FO", year=2026) == []


def test_diff_reports_added_removed_and_renamed() -> None:
    fetched = [refresh.Holiday(date(2026, 1, 26), "Republic Day"), refresh.Holiday(date(2026, 8, 15), "Independence Day")]
    configured = [refresh.Holiday(date(2026, 1, 26), "Republic Day Holiday"), refresh.Holiday(date(2026, 3, 3), "Holi")]
    diff = refresh.diff_holidays(fetched, configured)
    assert [row["date"] for row in diff["added"]] == ["2026-08-15"]
    assert [row["date"] for row in diff["removed"]] == ["2026-03-03"]
    assert [row["date"] for row in diff["renamed"]] == ["2026-01-26"]


def test_rewrite_calendar_preserves_comments_and_reparses() -> None:
    import yaml

    source = (
        "timezone: Asia/Kolkata\n"
        "holidays_verified: false\n"
        "holidays:\n"
        '  - { date: "2026-01-26", name: "Republic Day", verified: false }\n'
        "\n"
        "# a comment that must survive the rewrite\n"
        "monthly_expiry:\n"
        "  rule: last_weekly_expiry_of_month\n"
    )
    fetched = [refresh.Holiday(date(2026, 1, 26), "Republic Day")]
    out = refresh.rewrite_calendar(source, fetched, year=2026, fetched_on=date(2026, 10, 1))
    body = yaml.safe_load(out)
    assert body["holidays_verified"] is True
    assert body["holidays"] == [{"date": "2026-01-26", "name": "Republic Day", "verified": True}]
    assert "# a comment that must survive the rewrite" in out
    assert body["monthly_expiry"]["rule"] == "last_weekly_expiry_of_month"


def test_main_reports_drift_from_a_saved_payload(tmp_path: Path, config_dir: Path, capsys) -> None:
    saved = tmp_path / "nse.json"
    saved.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    evidence = tmp_path / "evidence"

    exit_code = refresh.main(
        [
            "--config-dir",
            str(config_dir),
            "--input",
            str(saved),
            "--year",
            "2026",
            "--dry-run",
            "--evidence-dir",
            str(evidence),
            "--json",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["year"] == 2026
    assert report["fetched_count"] == 3
    # The shipped calendar has a different 2026 set, so there must be drift.
    assert exit_code == refresh.EXIT_DRIFT
    assert report["drift"] is True
    assert not evidence.exists()  # --dry-run writes nothing
