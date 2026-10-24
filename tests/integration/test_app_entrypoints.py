"""Integration tests for the three application entry points.

These run the real ``main()`` functions (and, for one case, a real subprocess)
against an isolated data root, so the whole startup chain is covered:

    argparse -> config load -> directories -> DuckDB bootstrap ->
    logging -> run context -> status report -> manifest

Phase 2 additions exercise the capture modes that can run offline: the
instrument-dump cache makes ``--dry-run`` (and, with a fake provider, a whole
capture session) possible without credentials or a network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_ai.run import main as ai_main
from apps.anjaneeya_capture.instruments import save_dump_file
from apps.anjaneeya_capture.run import main as capture_main
from apps.anjaneeya_engine.run import main as engine_main
from core.app_bootstrap import (
    APP_AI,
    APP_CAPTURE,
    APP_ENGINE,
    DEFAULT_MODES,
    EXIT_CONFIG_ERROR,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
)
from tests.support.capture import DAY, make_dump

APP_MAINS = {
    APP_CAPTURE: capture_main,
    APP_ENGINE: engine_main,
    APP_AI: ai_main,
}

#: Modes that are still genuinely unimplemented (they must fail loudly).
#: ANJANEYA-ENGINE left this list in Stage 3.2e, and ANJANEYA-AI left it with ``report`` in
#: Stage 4.4: every mode either app advertises is now implemented, so the list is empty - and
#: a new *planned* mode belongs here the moment it is declared, which is the point of the
#: guard below.
PLANNED_MODE: dict[str, str] = {}

IST = ZoneInfo("Asia/Kolkata")


def base_args(tmp_path: Path, *, mode: str = "status", extra: list[str] | None = None) -> list[str]:
    args = [
        "--mode",
        mode,
        "--data-root",
        str(tmp_path / "data"),
        "--db-path",
        str(tmp_path / "db" / "anjaneeya.duckdb"),
        "--no-run-artifacts",
    ]
    return args + (extra or [])


def seed_instrument_cache(tmp_path: Path) -> Path:
    """Write a synthetic instrument dump into the configured cache directory.

    This is what makes the capture modes testable offline: the dump is the only
    input the universe resolution needs besides a reference price.
    """
    cache_dir = tmp_path / "data" / "raw" / "instruments"
    moment = datetime(DAY.year, DAY.month, DAY.day, 9, 0, tzinfo=IST)
    for exchange, rows in make_dump().items():
        save_dump_file(
            rows, cache_dir=cache_dir, exchange=exchange, day=DAY, fetched_at=moment
        )
    return cache_dir



@pytest.mark.parametrize("app", sorted(APP_MAINS))
def test_status_mode_reports_and_exits_zero(app: str, tmp_path: Path, capsys) -> None:
    exit_code = APP_MAINS[app](base_args(tmp_path))
    assert exit_code == EXIT_OK

    output = capsys.readouterr().out
    assert f"ANJANEYA {app} - status" in output
    assert "Asia/Kolkata" in output
    assert "tables" in output
    # Credential *presence* is reported, never the values.
    assert "ZERODHA_API_KEY" in output


@pytest.mark.parametrize("app", sorted(APP_MAINS))
def test_status_mode_creates_the_database_and_directories(app: str, tmp_path: Path) -> None:
    assert APP_MAINS[app](base_args(tmp_path)) == EXIT_OK
    assert (tmp_path / "db" / "anjaneeya.duckdb").exists()
    for name in ("raw", "candles", "features", "labels", "logs", "runs"):
        assert (tmp_path / "data" / name).exists(), f"{name} was not created"
    assert (tmp_path / "data" / "logs" / f"{app}.log").exists()


@pytest.mark.parametrize("app", sorted(APP_MAINS))
def test_status_json_output_is_machine_readable(app: str, tmp_path: Path, capsys) -> None:
    assert APP_MAINS[app](base_args(tmp_path, extra=["--json"])) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["app"] == app
    assert report["config"]["timezone"] == "Asia/Kolkata"
    assert report["database"]["exists"] is True
    assert report["calendar"]["expiry_weekday"] == "Tuesday"
    assert set(report["data_gates"]) >= {"exploratory_min_days", "ml_training_min_days"}


@pytest.mark.parametrize("mode", DEFAULT_MODES[APP_AI])
def test_every_ai_mode_actually_runs(mode: str, tmp_path: Path, capsys) -> None:
    """No advertised mode may exit 3 - the guard the Stage 4.4 work needed.

    ``--mode report`` was listed in ``--help`` while it still answered "not implemented in this
    phase" and exited 3. Every AI mode is offline-safe under ``--dry-run``, so this runs each one
    through the real entry point and refuses the answer "planned" for any of them. A new mode
    therefore has to land with its implementation (or with an entry in :data:`PLANNED_MODE`,
    which cannot happen without changing this test's own input).
    """
    exit_code = ai_main(base_args(tmp_path, mode=mode, extra=["--dry-run"]))
    captured = capsys.readouterr()
    assert exit_code != EXIT_NOT_IMPLEMENTED
    assert "not implemented" not in captured.err
    assert mode not in PLANNED_MODE.values()


def test_a_mode_that_is_not_implemented_still_fails_loudly(config, capsys) -> None:
    """The ``exit 3`` contract survives Stage 4.4: it is what a future mode must obey.

    No app advertises a planned mode any more (see :data:`PLANNED_MODE` and the guard below), so
    the path is exercised directly - the mode name never reaches the parser.
    """
    from apps.anjaneeya_ai import run as ai_run
    from tests.support.ai import make_ai_context

    context = make_ai_context(config, mode="future-mode")

    exit_code = ai_run._run_planned_mode(context, "future-mode")

    assert exit_code == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert "not implemented" in captured.err
    assert ai_run.AVAILABLE_MODES in captured.err


@pytest.mark.parametrize("app", sorted(APP_MAINS))
def test_no_app_documents_a_mode_it_cannot_route(app: str) -> None:
    """A mode named in an app's ``MODE_PLAN`` must be one the parser actually accepts.

    ``MODE_PLAN`` is the human-facing "what does this mode do" map that ``--dry-run`` prints, so
    a name in it that the parser refuses is a documentation bug; and for ANJANEYA-AI the map is
    kept exhaustive, so a new mode cannot be added to the CLI without saying what it does.
    """
    from apps.anjaneeya_ai import run as ai_run
    from apps.anjaneeya_capture import run as capture_run
    from apps.anjaneeya_engine import run as engine_run

    plans = {
        APP_AI: ai_run.MODE_PLAN,
        APP_CAPTURE: capture_run.MODE_PLAN,
        APP_ENGINE: engine_run.MODE_PLAN,
    }
    assert set(plans[app]) <= set(DEFAULT_MODES[app])
    if app == APP_AI:
        assert set(plans[app]) == set(DEFAULT_MODES[app])
    assert not set(DEFAULT_MODES[app]) & set(PLANNED_MODE.values())


# ---------------------------------------------------------------------------
# Capture modes (Phase 2)
# ---------------------------------------------------------------------------
def test_capture_dry_run_needs_no_credentials_but_needs_a_cached_master(
    tmp_path: Path, capsys
) -> None:
    """Without a cached dump the app says exactly what to do, and exits 2."""
    exit_code = capture_main(base_args(tmp_path, mode="capture", extra=["--dry-run"]))
    assert exit_code == EXIT_CONFIG_ERROR
    captured = capsys.readouterr()
    assert "--mode instruments" in captured.err


def test_capture_dry_run_prints_the_resolved_universe(tmp_path: Path, capsys) -> None:
    seed_instrument_cache(tmp_path)
    exit_code = capture_main(base_args(tmp_path, mode="capture", extra=["--dry-run"]))
    assert exit_code == EXIT_OK

    output = capsys.readouterr().out
    assert "dry-run: capture" in output
    assert "ATM strike" in output
    assert "option contract(s)" in output
    assert "reference price" in output
    assert "websocket     : enabled" in output


def test_option_chain_dry_run_announces_polling_only(tmp_path: Path, capsys) -> None:
    seed_instrument_cache(tmp_path)
    exit_code = capture_main(base_args(tmp_path, mode="option-chain", extra=["--dry-run"]))
    assert exit_code == EXIT_OK
    output = capsys.readouterr().out
    assert "option-chain polls" in output
    assert "websocket     : disabled" in output


def test_backfill_dry_run_prints_the_request_budget(tmp_path: Path, capsys) -> None:
    seed_instrument_cache(tmp_path)
    exit_code = capture_main(base_args(tmp_path, mode="backfill", extra=["--dry-run"]))
    assert exit_code == EXIT_OK
    output = capsys.readouterr().out
    assert "dry-run: backfill" in output
    assert "NIFTY 50" in output
    assert "requests      :" in output


def test_instruments_dry_run_reports_the_cached_master(tmp_path: Path, capsys) -> None:
    seed_instrument_cache(tmp_path)
    exit_code = capture_main(base_args(tmp_path, mode="instruments", extra=["--dry-run"]))
    assert exit_code == EXIT_OK
    output = capsys.readouterr().out
    assert "source        : cache" in output
    assert "instruments   : 22" in output


@pytest.mark.parametrize("mode", ["instruments", "backfill"])
def test_modes_that_need_the_api_fail_clearly_without_credentials(
    mode: str, tmp_path: Path, capsys, monkeypatch
) -> None:
    """Never pretend to capture: no token means exit 2 and actionable advice."""
    for name in (
        "ZERODHA_API_KEY",
        "ZERODHA_API_SECRET",
        "ZERODHA_USER_ID",
        "ZERODHA_PASSWORD",
        "ZERODHA_TOTP_KEY",
        "ZERODHA_REDIRECT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    seed_instrument_cache(tmp_path)

    exit_code = capture_main(base_args(tmp_path, mode=mode))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "ZERODHA" in capsys.readouterr().err


def test_until_flag_rejects_a_bad_time(tmp_path: Path, capsys) -> None:
    seed_instrument_cache(tmp_path)
    exit_code = capture_main(
        base_args(tmp_path, mode="option-chain", extra=["--until", "25:99"])
    )
    assert exit_code == EXIT_CONFIG_ERROR
    assert "--until" in capsys.readouterr().err



@pytest.mark.parametrize("app", sorted(APP_MAINS))
def test_status_mode_writes_a_run_manifest_when_artifacts_are_enabled(
    app: str, tmp_path: Path
) -> None:
    args = [
        "--mode",
        "status",
        "--data-root",
        str(tmp_path / "data"),
        "--db-path",
        str(tmp_path / "db" / "anjaneeya.duckdb"),
    ]
    assert APP_MAINS[app](args) == EXIT_OK
    run_dirs = list((tmp_path / "data" / "runs").glob(f"{app}_*"))
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["app_name"] == app
    assert manifest["status"] == "SUCCESS"
    assert manifest["config_hash"]


def test_invalid_configuration_exits_with_the_config_error_code(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "settings.yaml").write_text("project: {version: '0.1.0'}\n", encoding="utf-8")
    exit_code = capture_main(
        ["--mode", "status", "--config-dir", str(bad), "--data-root", str(tmp_path / "data")]
    )
    assert exit_code == EXIT_CONFIG_ERROR
    assert "configuration error" in capsys.readouterr().err


def test_naive_timestamp_guard_is_not_needed_for_status(tmp_path: Path) -> None:
    """Status must work with no credentials at all (offline diagnostics)."""
    assert capture_main(base_args(tmp_path)) == EXIT_OK


def test_capture_app_runs_as_a_module_subprocess(tmp_path: Path) -> None:
    """Prove the documented invocation works: ``python -m apps.anjaneeya_capture.run``."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.anjaneeya_capture.run",
            "--mode",
            "status",
            "--data-root",
            str(tmp_path / "data"),
            "--db-path",
            str(tmp_path / "db" / "anjaneeya.duckdb"),
            "--no-run-artifacts",
            "--json",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    report = json.loads(completed.stdout)
    assert report["app"] == APP_CAPTURE
    assert report["calendar"]["timezone"] == "Asia/Kolkata"
