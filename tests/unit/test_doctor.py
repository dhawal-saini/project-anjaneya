"""Tests for ``scripts/doctor.py`` - the pre-flight readiness check.

The doctor reuses ``health_check``'s sections and adds the four things that stop a
capture. Only the additive half and the report plumbing are pinned here; the reused
checks already have their own tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.storage import bootstrap_database
from scripts import doctor


def _all_ok() -> dict[str, dict[str, object]]:
    return {key: {"ok": True} for key, _title in doctor.SECTION_TITLES}


def test_is_healthy_requires_every_section() -> None:
    assert doctor.is_healthy(_all_ok()) is True
    report = _all_ok()
    report["token"] = {"ok": False, "hint": "expired"}
    assert doctor.is_healthy(report) is False


def test_scheduler_section_never_fails_the_run() -> None:
    payload = doctor.check_scheduler()
    assert payload["ok"] is True


def test_render_lists_every_section_and_a_verdict() -> None:
    rendered = doctor.render(_all_ok())
    for _key, title in doctor.SECTION_TITLES:
        assert title in rendered
    assert "RESULT: READY" in rendered


def test_check_database_writable_is_true_after_a_bootstrap(config) -> None:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_database(config.db_path, schema_version=config.settings.project.schema_version)
    payload = doctor.check_database_writable(config)
    assert payload["ok"] is True
    assert payload["writable"] is True


def test_check_database_writable_flags_a_missing_file(config) -> None:
    payload = doctor.check_database_writable(config)
    assert payload["ok"] is False
    assert payload["writable"] is False


def test_main_emits_a_parseable_report(
    tmp_path: Path, config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = doctor.main(
        [
            "--config-dir",
            str(config_dir),
            "--data-root",
            str(tmp_path / "data"),
            "--db-path",
            str(tmp_path / "db" / "anjaneeya.duckdb"),
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code in {doctor.EXIT_OK, doctor.EXIT_PROBLEMS}
    for key, _title in doctor.SECTION_TITLES:
        assert key in report
