"""Text-level tests for the Windows scheduler scripts.

The Task Scheduler settings this project depends on (no overlapping instances, a
run-whether-logged-on-or-not principal, an 8-hour cap, a calendar gate) are easy to
lose in an edit and impossible to spot until a task misfires. Asserting the wrapper
scripts contain them keeps the intent executable without a Windows CI runner.
"""

from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_schedule_tasks_installs_with_the_required_settings() -> None:
    text = _read("schedule_tasks.ps1")
    assert "IgnoreNew" in text  # never run a second instance over a live capture
    assert "Hours 8" in text  # execution time limit
    assert "S4U" in text  # run whether the user is logged on or not
    assert "Highest" in text  # highest privileges
    assert "Anjaneya-Morning" in text and "Anjaneya-PostClose" in text
    assert "run_scheduled.ps1" in text


def test_run_scheduled_gates_every_stage_on_the_calendar() -> None:
    text = _read("run_scheduled.ps1")
    assert "--require-trading-day" in text
    assert "--within" in text
    assert "run_morning.py" in text and "run_daily_pipeline.py" in text


def test_the_wrapper_scripts_exist() -> None:
    for name in (
        "run_morning.ps1",
        "run_morning.sh",
        "run_morning.py",
        "run_weekly.ps1",
        "run_weekly.sh",
        "run_scheduled.ps1",
        "schedule_tasks.ps1",
    ):
        assert (SCRIPTS / name).is_file(), name
