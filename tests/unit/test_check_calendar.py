"""Tests for ``scripts/check_calendar.py`` - the pre-session calendar guard.

The script exists to stop a silent, permanent loss: a provisional holiday that is really a
trading day makes ``--mode capture`` exit ``NO_SESSION`` and the session's ticks never exist
anywhere. These tests pin the two behaviours that matter:

* a provisional date inside the risk window fails the check (exit 1) and is named, and
* a *confirmed* date, or a provisional one outside the window, does not.

They use a scratch configuration whose holiday list is written by the test, so the assertions
do not depend on the shipped holiday list being provisional.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from scripts.check_calendar import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RISK
from scripts.check_calendar import main as calendar_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def scratch_config(
    tmp_path: Path, *, holidays: list[dict[str, object]], verified: bool = False
) -> Path:
    """A copy of the shipped configuration with a holiday list the test controls."""
    target = tmp_path / "config"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(CONFIG_DIR, target)
    path = target / "calendar.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["holidays_verified"] = verified
    payload["holidays"] = holidays
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def holiday(day: str, name: str, *, verified: bool) -> dict[str, object]:
    """One ``holidays`` entry."""
    return {"date": day, "name": name, "verified": verified}


def run(config_dir: Path, *extra: str) -> int:
    """Run the script's ``main`` against a scratch config."""
    return calendar_main(["--config-dir", str(config_dir), *extra])


# ---------------------------------------------------------------------------
# The risk window
# ---------------------------------------------------------------------------
def test_a_provisional_holiday_inside_the_window_fails_the_check(tmp_path, capsys) -> None:
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-10-02", "Gandhi Jayanti", verified=False)]
    )

    exit_code = run(config_dir, "--as-of", "2026-09-30", "--within", "10")

    output = capsys.readouterr().out
    assert exit_code == EXIT_RISK
    assert "Gandhi Jayanti" in output
    assert "PROVISIONAL" in output
    assert "verdict: RISK" in output


def test_the_risk_window_is_json_reportable(tmp_path, capsys) -> None:
    config_dir = scratch_config(
        tmp_path,
        holidays=[
            holiday("2026-10-02", "Gandhi Jayanti", verified=False),
            holiday("2026-10-05", "Provisional Monday", verified=False),
            holiday("2026-12-25", "Christmas", verified=True),
        ],
    )

    exit_code = run(config_dir, "--as-of", "2026-09-30", "--within", "7", "--json")

    report = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_RISK
    assert report["unverified_total"] == 2
    assert report["holidays_verified"] is False
    assert [item["date"] for item in report["risky_holidays"]] == ["2026-10-02", "2026-10-05"]
    assert [item["days_away"] for item in report["risky_holidays"]] == [2, 5]
    assert report["risk_window"] == {"start": "2026-09-30", "end": "2026-10-07", "days": 7}


def test_a_confirmed_holiday_is_not_a_risk(tmp_path, capsys) -> None:
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-10-02", "Gandhi Jayanti", verified=True)]
    )

    exit_code = run(config_dir, "--as-of", "2026-09-30", "--within", "10")

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "verdict: OK" in output
    assert "holiday: Gandhi Jayanti (verified)" in output


def test_a_provisional_date_outside_the_window_is_not_a_risk(tmp_path, capsys) -> None:
    """The window is the operator's tolerance: 30 days out is news, not an alarm."""
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-10-30", "Provisional Friday", verified=False)]
    )

    assert run(config_dir, "--as-of", "2026-09-30", "--within", "10") == EXIT_OK

    output = capsys.readouterr().out
    assert "verdict: OK" in output
    assert "note: the next provisional holiday is 2026-10-30 (Provisional Friday)" in output

    assert run(config_dir, "--as-of", "2026-09-30", "--within", "10", "--json") == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["risky_holidays"] == []
    assert report["next_unverified"] == {
        "date": "2026-10-30",
        "name": "Provisional Friday",
        "days_away": 30,
    }


def test_within_zero_only_checks_today(tmp_path) -> None:
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-10-02", "Gandhi Jayanti", verified=False)]
    )

    assert run(config_dir, "--as-of", "2026-09-30", "--within", "0") == EXIT_OK
    assert run(config_dir, "--as-of", "2026-10-02", "--within", "0") == EXIT_RISK


# ---------------------------------------------------------------------------
# The calendar listing
# ---------------------------------------------------------------------------
def test_the_listing_separates_trading_days_weekends_and_holidays(tmp_path, capsys) -> None:
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-10-05", "Test holiday", verified=True)]
    )

    assert run(config_dir, "--as-of", "2026-10-02", "--days", "6", "--json") == EXIT_OK

    report = json.loads(capsys.readouterr().out)
    rows = {row["date"]: row for row in report["calendar_days"]}
    assert rows["2026-10-02"]["trading"] is True
    assert rows["2026-10-05"]["trading"] is False
    assert rows["2026-10-05"]["reason"] == "holiday: Test holiday (verified)"
    assert rows["2026-10-03"]["reason"] == "weekend"  # Saturday
    assert rows["2026-10-06"]["session"] == "09:15-15:30"
    assert rows["2026-10-06"]["expiry"] == ["weekly expiry"]
    assert report["trading_days"] == 3  # 02, 06 and 07
    assert report["window_end"] == "2026-10-07"
    assert report["as_of_weekday"] == "Friday"


def test_the_monthly_expiry_is_tagged(tmp_path, capsys) -> None:
    """The last weekly expiry of October 2026 - the monthly - must be visible as such."""
    config_dir = scratch_config(tmp_path, holidays=[])

    assert run(config_dir, "--as-of", "2026-10-26", "--days", "7", "--json") == EXIT_OK

    report = json.loads(capsys.readouterr().out)
    monthly = [row for row in report["calendar_days"] if "monthly expiry" in row["expiry"]]
    assert [row["date"] for row in monthly] == ["2026-10-27"]
    assert report["expiries"]["current_monthly"] == "2026-10-27"


def test_a_broken_configuration_is_reported_not_raised(tmp_path, capsys) -> None:
    exit_code = run(tmp_path / "missing")

    assert exit_code == EXIT_CONFIG_ERROR
    assert "ANJANEYA calendar" in capsys.readouterr().err


def test_the_rendered_report_names_the_session_and_the_expiry_window(tmp_path, capsys) -> None:
    config_dir = scratch_config(tmp_path, holidays=[])

    assert run(config_dir, "--as-of", "2026-09-30", "--days", "2") == EXIT_OK

    output = capsys.readouterr().out
    assert "ANJANEYA calendar check - as of 2026-09-30 (Wednesday)" in output
    assert "regular session : 09:15-15:30" in output
    assert "weekly 2026-10-06" in output
    assert "risk: none" in output


def test_a_bad_date_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        calendar_main(["--as-of", "2026-13-45"])


def test_require_trading_day_is_the_scheduler_guard(tmp_path, capsys) -> None:
    """A cron line can gate itself with `check_calendar.py --require-trading-day`."""
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-10-02", "Gandhi Jayanti", verified=True)]
    )

    assert run(config_dir, "--as-of", "2026-10-01", "--require-trading-day") == EXIT_OK

    capsys.readouterr()
    assert run(config_dir, "--as-of", "2026-10-02", "--require-trading-day") == EXIT_RISK
    assert "verdict: CLOSED" in capsys.readouterr().out
    assert run(config_dir, "--as-of", "2026-10-04", "--require-trading-day") == EXIT_RISK  # Sunday
    # Without the flag a closed day is simply reported, which keeps `--days 30` usable.
    assert run(config_dir, "--as-of", "2026-10-04") == EXIT_OK
    assert "verdict: OK" in capsys.readouterr().out


def test_the_header_says_whether_the_as_of_day_trades(tmp_path, capsys) -> None:
    config_dir = scratch_config(tmp_path, holidays=[])

    assert run(config_dir, "--as-of", "2026-10-03") == EXIT_OK
    assert "as of 2026-10-03 (Saturday): NOT a trading day" in capsys.readouterr().out

    assert run(config_dir, "--as-of", "2026-10-05") == EXIT_OK
    assert "as of 2026-10-05 (Monday): trading day" in capsys.readouterr().out


def test_holidays_that_are_already_past_are_not_the_next_one(tmp_path, capsys) -> None:
    """The YAML holds a whole year: "next" must mean on or after --as-of, never in the past."""
    config_dir = scratch_config(
        tmp_path,
        holidays=[
            holiday("2026-03-04", "Holi", verified=False),
            holiday("2026-11-09", "Provisional November", verified=False),
        ],
    )

    assert run(config_dir, "--as-of", "2026-09-30", "--days", "1", "--json") == EXIT_OK

    report = json.loads(capsys.readouterr().out)
    assert report["unverified_total"] == 2
    assert report["next_unverified"] == {
        "date": "2026-11-09",
        "name": "Provisional November",
        "days_away": 40,
    }


def test_no_upcoming_provisional_holiday_means_no_note(tmp_path, capsys) -> None:
    """A whole-year list can be entirely in the past: then there is nothing to warn about."""
    config_dir = scratch_config(
        tmp_path, holidays=[holiday("2026-03-04", "Holi", verified=False)]
    )

    assert run(config_dir, "--as-of", "2026-09-30", "--days", "2") == EXIT_OK

    output = capsys.readouterr().out
    assert "1 PROVISIONAL" in output
    assert "note: the next provisional holiday" not in output
    assert "verdict: OK" in output


def test_days_zero_prints_no_rows(tmp_path, capsys) -> None:
    config_dir = scratch_config(tmp_path, holidays=[])

    exit_code = run(config_dir, "--as-of", date(2026, 10, 1).isoformat(), "--days", "0", "--json")

    assert exit_code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["calendar_days"] == []
    assert report["trading_days"] == 0
    assert report["window_end"] == "2026-10-01"

