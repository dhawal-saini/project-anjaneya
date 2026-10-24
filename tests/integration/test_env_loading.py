"""End-to-end proof that the entry points load ``.env`` (``core/env.py``).

These tests exist because filling in ``.env`` must be *enough*::

    cp .env.example .env      # put ZERODHA_API_KEY / ZERODHA_API_SECRET in it
    uv run python scripts/daily_login.py --request-token <token>

Before this loader existed the file was ignored, so login failed with "Missing
ZERODHA_API_KEY" even though ``.env`` was correct.

``tests/conftest.py`` disables ``.env`` loading for the whole suite so that a
developer's real credentials can never leak into a test, so each test below
removes that guard explicitly. The load writes into :data:`os.environ`, hence the
autouse fixture that hands the process environment back exactly as it was.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from apps.anjaneeya_capture.run import main as capture_main
from core.env import SKIP_ENV_FILE_VAR
from scripts.daily_login import EXIT_CONFIG_ERROR, EXIT_FAILED
from scripts.daily_login import main as login_main
from scripts.health_check import main as health_main

CREDENTIAL_NAMES = (
    "ZERODHA_API_KEY",
    "ZERODHA_API_SECRET",
    "ZERODHA_USER_ID",
    "ZERODHA_PASSWORD",
    "ZERODHA_TOTP_KEY",
    "ZERODHA_REDIRECT_URL",
)
API_KEY = "test-api-key-123456"
API_SECRET = "test-api-secret-123456"
ENV_BODY = f"ZERODHA_API_KEY={API_KEY}\nZERODHA_API_SECRET={API_SECRET}\n"


@pytest.fixture(autouse=True)
def restore_environ() -> Iterator[None]:
    """Undo the ``os.environ`` writes that loading a real ``.env`` performs."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


@pytest.fixture
def bare_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing relevant in the environment: the ``.env`` file is the only source."""
    for name in (*CREDENTIAL_NAMES, SKIP_ENV_FILE_VAR):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def checkout(tmp_path: Path, config_dir: Path) -> Path:
    """A miniature checkout: a copy of ``config/`` with a ``.env`` beside it.

    A copy is required because the shipped token-store path is repository-relative;
    a test must never point the scripts at the real ``data/`` tree.
    """
    shutil.copytree(config_dir, tmp_path / "config")
    data_sources = tmp_path / "config" / "data_sources.yaml"
    body = yaml.safe_load(data_sources.read_text(encoding="utf-8"))
    body["zerodha"]["token_store"]["path"] = str(tmp_path / "tokens" / "zerodha_token.json")
    data_sources.write_text(yaml.safe_dump(body), encoding="utf-8")
    return tmp_path


def write_env_file(checkout: Path, body: str = ENV_BODY) -> Path:
    """Create the ``.env`` next to the copied ``config/`` directory."""
    path = checkout / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def script_args(checkout: Path, *extra: str, db_path: bool = True) -> list[str]:
    """Arguments pointing both scripts at the scratch checkout."""
    args = [
        "--config-dir",
        str(checkout / "config"),
        "--data-root",
        str(checkout / "data"),
    ]
    if db_path:
        args += ["--db-path", str(checkout / "db" / "anjaneeya.duckdb")]
    return args + list(extra)


# ---------------------------------------------------------------------------
# health_check.py
# ---------------------------------------------------------------------------
def test_health_check_reports_what_came_from_the_env_file(
    checkout: Path, bare_environment, capsys
) -> None:
    env_path = write_env_file(checkout)
    assert health_main(script_args(checkout, "--json")) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["env_file"]["path"] == str(env_path.resolve())
    assert report["env_file"]["loaded"] is True
    assert report["env_file"]["skipped"] is False
    assert report["env_file"]["variables_from_file"] == ["ZERODHA_API_KEY", "ZERODHA_API_SECRET"]
    assert report["credentials"]["variables"]["ZERODHA_API_KEY"] is True
    assert report["credentials"]["variables"]["ZERODHA_PASSWORD"] is False


def test_health_check_explains_a_missing_env_file(checkout: Path, bare_environment, capsys) -> None:
    assert health_main(script_args(checkout, "--json")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["env_file"]["loaded"] is False
    assert "copy .env.example to .env" in report["env_file"]["note"]
    assert report["credentials"]["variables"]["ZERODHA_API_KEY"] is False


def test_health_check_never_prints_a_secret(checkout: Path, bare_environment, capsys) -> None:
    write_env_file(checkout)
    assert health_main(script_args(checkout)) == 0
    output = capsys.readouterr().out
    assert API_KEY not in output
    assert API_SECRET not in output
    assert "ZERODHA_API_KEY" in output  # the name is fine, the value is not


def test_skip_flag_disables_loading_for_the_scripts(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv(SKIP_ENV_FILE_VAR, "1")
    for name in CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)
    write_env_file(checkout)
    assert health_main(script_args(checkout, "--json")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["env_file"]["skipped"] is True
    assert report["env_file"]["variables_from_file"] == []
    assert report["credentials"]["variables"]["ZERODHA_API_KEY"] is False


def test_a_real_environment_variable_beats_the_env_file(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The app path, not just the loader: ``ANJANEYA_DATA_ROOT`` resolves from
    the shell even though ``.env`` names a different directory."""
    monkeypatch.delenv(SKIP_ENV_FILE_VAR, raising=False)
    from_shell = checkout / "from-the-shell"
    write_env_file(checkout, f"ANJANEYA_DATA_ROOT={checkout / 'from-the-file'}\n")
    monkeypatch.setenv("ANJANEYA_DATA_ROOT", str(from_shell))

    # No --data-root here on purpose: the CLI flag would mask the precedence test.
    args = ["--config-dir", str(checkout / "config"), "--json"]
    assert health_main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert Path(report["configuration"]["data_root"]).resolve() == from_shell.resolve()


# ---------------------------------------------------------------------------
# daily_login.py
# ---------------------------------------------------------------------------
def test_daily_login_finds_credentials_in_the_env_file(
    checkout: Path, bare_environment, capsys
) -> None:
    """The regression this loader fixes.

    With ``.env`` ignored, this exited 2 with "Missing ZERODHA_API_KEY". Now the
    credentials are found, and the run stops at the step that genuinely needs the
    operator: passing ``--request-token``.
    """
    write_env_file(checkout)
    exit_code = login_main(script_args(checkout, db_path=False))
    stderr = capsys.readouterr().err
    assert exit_code == EXIT_FAILED, stderr
    assert "--request-token" in stderr


def test_daily_login_without_an_env_file_still_reports_the_missing_variables(
    checkout: Path, bare_environment, capsys
) -> None:
    exit_code = login_main(script_args(checkout, db_path=False))
    assert exit_code == EXIT_CONFIG_ERROR
    assert "ZERODHA_API_KEY" in capsys.readouterr().err


def test_login_says_where_the_credentials_came_from(
    checkout: Path, bare_environment, capsys
) -> None:
    """Even a *failing* login must show whether ``.env`` was picked up."""
    write_env_file(checkout)
    login_main(script_args(checkout, db_path=False))
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert str((checkout / ".env").resolve()) in output
    assert API_SECRET not in output


# ---------------------------------------------------------------------------
# The capture app (bootstrap_app)
# ---------------------------------------------------------------------------
def test_capture_app_status_sees_credentials_from_the_env_file(
    checkout: Path, bare_environment, capsys
) -> None:
    write_env_file(checkout)
    assert capture_main(script_args(checkout, "--mode", "status", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["credentials"]["ZERODHA_API_KEY"] is True
    assert payload["credentials"]["ZERODHA_TOTP_KEY"] is False

