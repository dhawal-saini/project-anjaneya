"""Tests for ``core.env`` - the single place ANJANEYA reads a ``.env`` file.

The rules under test are the ones an operator depends on:

* a real environment variable always wins over the file (CI / one-off overrides);
* a blank value means "unset";
* a missing file is a no-op, because ANJANEYA also runs fully offline;
* ``ANJANEYA_SKIP_DOTENV=1`` disables loading (the test suite relies on this);
* only variable *names* are ever exposed - never a value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.env import (
    ENV_FILE_NAME,
    SKIP_ENV_FILE_VAR,
    EnvFileResult,
    default_env_path,
    env_file_skipped,
    load_env_file,
)
from core.logging_setup import collect_secrets

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SECRET = "kite-secret-value-1234"


def write_env(tmp_path: Path, body: str, name: str = ENV_FILE_NAME) -> Path:
    """Write a dotenv file and return its path."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Missing / skipped files
# ---------------------------------------------------------------------------
def test_missing_file_is_a_no_op(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    result = load_env_file(tmp_path / "absent.env", environ=env)
    assert result.loaded is False
    assert result.skipped is False
    assert bool(result) is False
    assert env == {}


def test_skip_flag_turns_loading_into_a_no_op(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"ZERODHA_API_KEY={SECRET}\n")
    env: dict[str, str] = {SKIP_ENV_FILE_VAR: "1"}
    result = load_env_file(path, environ=env)
    assert result.skipped is True
    assert bool(result) is False
    assert env == {SKIP_ENV_FILE_VAR: "1"}


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On"])
def test_skip_flag_accepts_the_usual_truthy_spellings(value: str) -> None:
    assert env_file_skipped({SKIP_ENV_FILE_VAR: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_skip_flag_is_off_by_default(value: str) -> None:
    assert env_file_skipped({SKIP_ENV_FILE_VAR: value}) is False


def test_the_suite_itself_runs_with_loading_disabled() -> None:
    """Rule 3 of tests/conftest.py, asserted rather than assumed."""
    assert env_file_skipped() is True


# ---------------------------------------------------------------------------
# Applying values
# ---------------------------------------------------------------------------
def test_values_are_applied_to_the_given_mapping(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"ZERODHA_API_KEY={SECRET}\nANJANEYA_LOG_LEVEL=DEBUG\n")
    env: dict[str, str] = {}
    result = load_env_file(path, environ=env)
    assert env["ZERODHA_API_KEY"] == SECRET
    assert env["ANJANEYA_LOG_LEVEL"] == "DEBUG"
    assert result.loaded is True
    assert result.path == path.resolve()
    assert result.names == ("ANJANEYA_LOG_LEVEL", "ZERODHA_API_KEY")


def test_real_environment_variables_win(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"ZERODHA_API_KEY={SECRET}\nANJANEYA_LOG_LEVEL=DEBUG\n")
    env = {"ZERODHA_API_KEY": "from-the-shell"}
    result = load_env_file(path, environ=env)
    assert env["ZERODHA_API_KEY"] == "from-the-shell"
    assert env["ANJANEYA_LOG_LEVEL"] == "DEBUG"
    # ...and the shadowed name is not reported as applied.
    assert result.names == ("ANJANEYA_LOG_LEVEL",)


def test_override_replaces_existing_values(tmp_path: Path) -> None:
    path = write_env(tmp_path, "ZERODHA_API_KEY=from-the-file\n")
    env = {"ZERODHA_API_KEY": "from-the-shell"}
    result = load_env_file(path, environ=env, override=True)
    assert env["ZERODHA_API_KEY"] == "from-the-file"
    assert result.names == ("ZERODHA_API_KEY",)


def test_blank_values_are_treated_as_unset(tmp_path: Path) -> None:
    path = write_env(tmp_path, "ZERODHA_PASSWORD=\nZERODHA_TOTP_KEY='   '\n")
    env: dict[str, str] = {}
    result = load_env_file(path, environ=env)
    assert env == {}
    assert result.loaded is True
    assert result.names == ()


def test_quotes_comments_and_export_prefix_are_handled(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        "# a comment\n"
        'export ZERODHA_API_KEY="quoted value"\n'
        "ZERODHA_API_SECRET=plain # trailing comment\n",
    )
    env: dict[str, str] = {}
    load_env_file(path, environ=env)
    assert env["ZERODHA_API_KEY"] == "quoted value"
    assert env["ZERODHA_API_SECRET"] == "plain"


# ---------------------------------------------------------------------------
# Reporting + downstream wiring
# ---------------------------------------------------------------------------
def test_result_never_exposes_a_value(tmp_path: Path) -> None:
    path = write_env(tmp_path, f"ZERODHA_API_KEY={SECRET}\n")
    env: dict[str, str] = {}
    result = load_env_file(path, environ=env)
    assert "ZERODHA_API_KEY" in result.describe()
    assert SECRET not in result.describe()
    assert SECRET not in repr(result)


def test_describe_states_the_outcome(tmp_path: Path) -> None:
    path = tmp_path / ENV_FILE_NAME
    assert "not found" in EnvFileResult(path=path).describe()
    loaded = EnvFileResult(path=path, loaded=True, names=("A", "B"))
    assert "loaded: A, B" in loaded.describe()
    assert SKIP_ENV_FILE_VAR in EnvFileResult(path=path, skipped=True).describe()


def test_loaded_secrets_reach_the_redaction_list(tmp_path: Path) -> None:
    """Why the load must happen before logging is configured."""
    path = write_env(tmp_path, f"ZERODHA_API_KEY={SECRET}\n")
    env: dict[str, str] = {}
    load_env_file(path, environ=env)
    assert collect_secrets(env, ["ZERODHA_API_KEY"]) == (SECRET,)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def test_default_env_path_follows_the_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "checkout" / "config"
    config_dir.mkdir(parents=True)
    assert default_env_path(config_dir) == (tmp_path / "checkout" / ENV_FILE_NAME)


def test_default_env_path_without_arguments_is_the_repository_root() -> None:
    assert default_env_path() == REPO_ROOT / ENV_FILE_NAME


def test_load_defaults_to_the_repository_style_layout(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_env(tmp_path, "ANJANEYA_LOG_LEVEL=TRACE\n")
    env: dict[str, str] = {}
    result = load_env_file(config_dir=config_dir, environ=env)
    assert result.loaded is True
    assert env["ANJANEYA_LOG_LEVEL"] == "TRACE"

