"""Tests for the operator scripts: ``daily_login.py`` and ``health_check.py``.

Both scripts are the operator's entry points (log in once, then check health), so
they are tested for the behaviour that matters: they never print a secret, a
missing credential is reported as a *configuration* problem, a valid cached token
is reused, and the health check turns a stale capture heartbeat into a non-zero
exit code.
"""

from __future__ import annotations

import io
import json
import shutil
import socket
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import scripts.login_and_start as login_and_start
from core.auth.browser_login import (
    RedirectCatcher,
    login_url,
    obtain_request_token,
    parse_redirect_url,
)
from core.auth.zerodha_auth import AuthenticationError
from scripts.daily_login import EXIT_CONFIG_ERROR, EXIT_OK
from scripts.daily_login import main as login_main
from scripts.health_check import main as health_main
from scripts.login_and_start import EXIT_LOGIN_FAILED
from scripts.login_and_start import main as login_and_start_main

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def scratch_config(tmp_path: Path, config_dir: Path) -> Path:
    """A copy of the shipped config whose token store lives inside ``tmp_path``.

    The default token-store path is repository-relative, so a test must not use
    it: that would write a token file into the working tree.
    """
    target = tmp_path / "config"
    shutil.copytree(config_dir, target)
    data_sources = target / "data_sources.yaml"
    body = yaml.safe_load(data_sources.read_text(encoding="utf-8"))
    body["zerodha"]["token_store"]["path"] = str(tmp_path / "tokens" / "zerodha_token.json")
    data_sources.write_text(yaml.safe_dump(body), encoding="utf-8")
    return target


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credentials anywhere: a test must never read a developer's .env."""
    for name in (
        "ZERODHA_API_KEY",
        "ZERODHA_API_SECRET",
        "ZERODHA_USER_ID",
        "ZERODHA_PASSWORD",
        "ZERODHA_TOTP_KEY",
        "ZERODHA_REDIRECT_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def script_args(scratch_config: Path, tmp_path: Path, *extra: str, db: bool = True) -> list[str]:
    """Common arguments for both scripts (``db`` adds the app-only ``--db-path``)."""
    args = [
        "--config-dir",
        str(scratch_config),
        "--data-root",
        str(tmp_path / "data"),
    ]
    if db:
        args += ["--db-path", str(tmp_path / "db" / "anjaneeya.duckdb")]
    return args + list(extra)


# ---------------------------------------------------------------------------
# daily_login.py
# ---------------------------------------------------------------------------
def test_daily_login_without_credentials_explains_what_is_missing(
    scratch_config: Path, tmp_path: Path, clean_env, capsys
) -> None:
    exit_code = login_main(script_args(scratch_config, tmp_path, db=False))
    assert exit_code == EXIT_CONFIG_ERROR
    assert "ZERODHA_API_KEY" in capsys.readouterr().err


def test_daily_login_reuses_a_valid_cached_token(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    """A token cached by the morning login is reused instead of logging in again."""
    token_path = tmp_path / "tokens" / "zerodha_token.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    token_path.write_text(
        json.dumps(
            {
                "access_token": "abcdefghijklmnop",
                "user_id": "AB1234",
                "api_key": "testkey",
                "login_time": now.isoformat(),
                "expires_at": (now + timedelta(hours=12)).isoformat(),
                "public_token": None,
                "obtained_via": "totp",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")

    exit_code = login_main(script_args(scratch_config, tmp_path, "--no-verify", "--json", db=False))

    assert exit_code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["reused_cached_token"] is True
    assert payload["verified"] is False
    # The secret itself is never printed - only a masked tail.
    assert payload["access_token"] == "...mnop"
    assert payload["status"] == "ok"


def test_daily_login_fails_cleanly_when_the_token_cannot_be_exchanged(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    exit_code = login_main(
        script_args(scratch_config, tmp_path, "--request-token", "not-a-real-token", db=False)
    )
    assert exit_code == 1
    assert "login failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# health_check.py
# ---------------------------------------------------------------------------
def test_health_check_reports_the_capture_section(
    scratch_config: Path, tmp_path: Path, capsys
) -> None:
    assert health_main(script_args(scratch_config, tmp_path, "--json")) == 0
    report = json.loads(capsys.readouterr().out)
    capture = report["capture"]
    assert capture["ok"] is True
    assert capture["policy"]["mode"] == "full"
    assert capture["heartbeat"]["present"] is False
    assert capture["policy"]["mirror_parquet"] is True


def test_health_check_flags_a_stale_running_heartbeat(
    scratch_config: Path, tmp_path: Path, capsys
) -> None:
    """A heartbeat that claims to be RUNNING but has gone cold is a problem."""
    from apps.anjaneeya_capture.heartbeat import HEARTBEAT_FORMAT_VERSION

    heartbeat_path = tmp_path / "data" / "runs" / "heartbeat" / "capture.json"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text(
        json.dumps(
            {
                "format_version": HEARTBEAT_FORMAT_VERSION,
                "app": "anjaneeya_capture",
                "run_id": "run-1",
                "status": "RUNNING",
                "mode": "capture",
                "written_at": (datetime.now(UTC) - timedelta(minutes=30)).isoformat(),
                "poll_failures": 3,
            }
        ),
        encoding="utf-8",
    )
    exit_code = health_main(script_args(scratch_config, tmp_path, "--json"))
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["capture"]["ok"] is False
    assert any("may have died" in problem for problem in report["capture"]["problems"])
    assert any("poll(s) failed" in problem for problem in report["capture"]["problems"])


def test_health_check_accepts_a_stopped_heartbeat(
    scratch_config: Path, tmp_path: Path, capsys
) -> None:
    from apps.anjaneeya_capture.heartbeat import HEARTBEAT_FORMAT_VERSION

    heartbeat_path = tmp_path / "data" / "runs" / "heartbeat" / "capture.json"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text(
        json.dumps(
            {
                "format_version": HEARTBEAT_FORMAT_VERSION,
                "app": "anjaneeya_capture",
                "run_id": "run-1",
                "status": "STOPPED",
                "written_at": (datetime.now(UTC) - timedelta(hours=3)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    assert health_main(script_args(scratch_config, tmp_path, "--json")) == 0


# ---------------------------------------------------------------------------
# The shell wrappers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["start_capture.ps1", "start_capture.sh", "run_backfill.ps1", "run_backfill.sh"]
)
def test_wrappers_exist_and_point_at_the_capture_app(name: str) -> None:
    path = REPO_ROOT / "scripts" / name
    assert path.is_file(), f"{name} is missing"
    body = path.read_text(encoding="utf-8")
    assert "apps.anjaneeya_capture.run" in body
    expected_mode = "capture" if name.startswith("start_capture") else "backfill"
    assert "--mode" in body and expected_mode in body

# ---------------------------------------------------------------------------
# core.auth.browser_login - catching the request_token on the loopback redirect
# ---------------------------------------------------------------------------
def _free_port() -> int:
    """A port nothing is listening on (the catcher is about to bind it)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _FakeBrowser:
    """Stands in for ``webbrowser.open`` by *being* the browser.

    Kite's redirect is simulated exactly as the broker sends it: one GET to the
    registered redirect URL carrying ``request_token`` (or ``status=error``).
    """

    def __init__(self, port: int, query: str = "", *, decline: bool = False) -> None:
        self.port = port
        self.query = query
        self.decline = decline
        self.opened: list[str] = []

    def __call__(self, url: str) -> bool:
        """Record the URL that was opened and hit the catcher once."""
        self.opened.append(url)
        if self.decline:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/?{self.query}") as response:
                response.read()
        except urllib.error.HTTPError:
            pass  # a 4xx page still proves the catcher saw the redirect
        return True


def _cached_token(tmp_path: Path, *, api_key: str = "testkey", hours: float = 12.0) -> Path:
    """Write a token into the scratch token store, as the morning login would."""
    path = tmp_path / "tokens" / "zerodha_token.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    path.write_text(
        json.dumps(
            {
                "access_token": "cachedaccesstoken",
                "user_id": "AB1234",
                "api_key": api_key,
                "login_time": now.isoformat(),
                "expires_at": (now + timedelta(hours=hours)).isoformat(),
                "public_token": None,
                "obtained_via": "browser_login",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_login_url_is_the_kite_connect_v3_endpoint() -> None:
    assert login_url("abc123") == "https://kite.zerodha.com/connect/login?v=3&api_key=abc123"
    with pytest.raises(AuthenticationError, match="without an API key"):
        login_url("")


@pytest.mark.parametrize(
    ("redirect_url", "expected"),
    [
        ("http://192.168.1.20:8444/", "not a loopback address"),
        ("https://127.0.0.1:8444/", "must use plain http"),
        ("https://kite.example.com/redirect", "must use plain http"),
    ],
)
def test_only_a_loopback_http_redirect_url_can_be_served(redirect_url: str, expected: str) -> None:
    """Anything this machine cannot serve must fail loudly, not hang until the timeout."""
    with pytest.raises(AuthenticationError) as excinfo:
        parse_redirect_url(redirect_url)
    assert expected in str(excinfo.value)


def test_redirect_url_needs_an_explicit_port() -> None:
    with pytest.raises(AuthenticationError, match="no explicit port"):
        parse_redirect_url("http://127.0.0.1/")


def test_parse_redirect_url_accepts_every_loopback_spelling() -> None:
    for url in ("http://127.0.0.1:8444/", "http://localhost:8444/", "http://[::1]:8444/"):
        host, port = parse_redirect_url(url)
        assert host in {"127.0.0.1", "localhost", "::1"}
        assert port == 8444


def test_catcher_captures_the_request_token_and_ignores_unrelated_requests() -> None:
    """A favicon-style probe must not end the wait; the real redirect must."""
    with RedirectCatcher("127.0.0.1", 0) as catcher:
        with urllib.request.urlopen(f"http://127.0.0.1:{catcher.port}/") as noise:
            assert b"Nothing to catch here" in noise.read()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{catcher.port}/?request_token=tok-1&action=login&status=success"
        ) as redirect:
            assert b"Login complete" in redirect.read()
        outcome = catcher.wait(timeout=5)
    assert outcome == {"status": "success", "request_token": "tok-1"}


def test_catcher_surfaces_a_kite_error_instead_of_a_token() -> None:
    with RedirectCatcher("127.0.0.1", 0) as catcher:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                f"http://127.0.0.1:{catcher.port}/?status=error&message=Invalid+password"
            )
        assert excinfo.value.code == 400
        outcome = catcher.wait(timeout=5)
    assert outcome["status"] == "error"
    assert outcome["message"] == "Invalid password"


def test_obtain_request_token_returns_what_the_browser_handed_back() -> None:
    """The whole loopback round trip: open the login URL, catch the redirect."""
    port = _free_port()
    browser = _FakeBrowser(port, "request_token=browser-tok&action=login&status=success")
    messages: list[str] = []

    token = obtain_request_token(
        "testkey",
        redirect_url=f"http://127.0.0.1:{port}/",
        timeout=5,
        opener=browser,
        notify=messages.append,
    )

    assert token == "browser-tok"
    assert browser.opened == [login_url("testkey")]
    assert any(f"127.0.0.1:{port}" in message for message in messages)


def test_obtain_request_token_announces_the_url_before_waiting() -> None:
    """``--no-browser`` must still say what to open - and still listen for it."""
    port = _free_port()
    messages: list[str] = []

    with pytest.raises(AuthenticationError, match="Timed out"):
        obtain_request_token(
            "testkey",
            redirect_url=f"http://127.0.0.1:{port}/",
            timeout=0.3,
            open_browser=False,
            notify=messages.append,
        )

    assert any("https://kite.zerodha.com/connect/login" in message for message in messages)


def test_obtain_request_token_times_out_with_an_actionable_message() -> None:
    """A browser that never comes back must say so - and cache nothing."""
    port = _free_port()
    messages: list[str] = []

    with pytest.raises(AuthenticationError) as excinfo:
        obtain_request_token(
            "testkey",
            redirect_url=f"http://127.0.0.1:{port}/",
            timeout=0.3,
            opener=_FakeBrowser(port, decline=True),
            notify=messages.append,
        )

    assert "Timed out" in str(excinfo.value)
    assert "Redirect URL is exactly" in str(excinfo.value)
    assert any("Could not open a browser automatically" in message for message in messages)


def test_obtain_request_token_rejects_an_unservable_url_before_listening() -> None:
    with pytest.raises(AuthenticationError, match="not a loopback address"):
        obtain_request_token("testkey", redirect_url="http://kite.example.com:8444/", timeout=1)

# ---------------------------------------------------------------------------
# scripts/login_and_start.py - login first, capture only on success
# ---------------------------------------------------------------------------
def _record_capture_start(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace the capture child process with a recorder."""
    calls: list[list[str]] = []

    def fake_start(capture_args: list[str]) -> int:
        calls.append(list(capture_args))
        return 0

    monkeypatch.setattr(login_and_start, "start_capture", fake_start)
    return calls


def _record_login(monkeypatch: pytest.MonkeyPatch, *, code: int = EXIT_OK) -> list[list[str]]:
    """Replace the ``daily_login`` child process with a recorder."""
    calls: list[list[str]] = []

    def fake_login(login_args: list[str]) -> tuple[int, str, str]:
        calls.append(list(login_args))
        return code, "ANJANEYA Kite login\n  status          : ok\n", ""

    monkeypatch.setattr(login_and_start, "run_login", fake_login)
    return calls


def test_login_and_start_reuses_a_cached_token_and_starts_the_capture(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    _cached_token(tmp_path)
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    logins = _record_login(monkeypatch)
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(script_args(scratch_config, tmp_path))

    assert exit_code == EXIT_OK
    # A usable cached token means no browser round trip: the login is cache-only.
    assert len(logins) == 1
    assert "--request-token" not in logins[0]
    assert starts == [script_args(scratch_config, tmp_path)]
    assert "cached token" in capsys.readouterr().out


def test_login_and_start_exchanges_a_pasted_token_and_forwards_capture_flags(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    """``--paste`` plus flags after ``--`` is the manual fallback, end to end."""
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO("http://127.0.0.1:8444/?request_token=pasted-tok&status=success\n"),
    )
    logins = _record_login(monkeypatch)
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(
        script_args(scratch_config, tmp_path, "--paste", "--", "--smoke", "--until", "15:30")
    )

    assert exit_code == EXIT_OK
    assert logins[0][:2] == ["--request-token", "pasted-tok"]
    assert "--smoke" in starts[0] and "15:30" in starts[0]
    assert "--" not in starts[0]
    captured = capsys.readouterr()
    # The one-time token is a credential: it must never be echoed.
    assert "pasted-tok" not in captured.out
    assert "pasted-tok" not in captured.err


def test_login_and_start_never_starts_the_capture_when_the_login_failed(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    """A request_token that Kite refuses must stop the run before the capture."""
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO("http://127.0.0.1:8444/?request_token=dead-tok&status=success\n"),
    )
    logins = _record_login(monkeypatch, code=EXIT_LOGIN_FAILED)
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(script_args(scratch_config, tmp_path, "--paste"))

    assert exit_code == EXIT_LOGIN_FAILED
    assert logins[0][:2] == ["--request-token", "dead-tok"]
    assert starts == []
    assert "was NOT started" in capsys.readouterr().err


def test_login_and_start_refuses_skip_login_without_a_usable_token(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    logins = _record_login(monkeypatch)
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(script_args(scratch_config, tmp_path, "--skip-login"))

    assert exit_code == EXIT_LOGIN_FAILED
    assert logins == [] and starts == []
    assert "--skip-login" in capsys.readouterr().err


def test_login_and_start_can_log_in_without_starting_the_capture(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    _cached_token(tmp_path)
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    _record_login(monkeypatch)
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(script_args(scratch_config, tmp_path, "--no-start", "--json"))

    assert exit_code == EXIT_OK
    assert starts == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["capture_started"] is False
    assert payload["login"] == "cached token"
    # The token itself is never part of the report - only where it is stored.
    assert "access_token" not in payload


def test_login_and_start_never_touches_anything_on_a_dry_run(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")

    def forbidden(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("--dry-run must not open a browser")

    monkeypatch.setattr(login_and_start, "obtain_request_token", forbidden)
    logins = _record_login(monkeypatch)
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(script_args(scratch_config, tmp_path, "--dry-run"))

    assert exit_code == EXIT_OK
    assert logins == [] and starts == []
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "127.0.0.1:8444" in out
    assert "apps.anjaneeya_capture.run" in out


def test_login_and_start_reports_a_missing_api_key_as_a_configuration_problem(
    scratch_config: Path, tmp_path: Path, clean_env, capsys
) -> None:
    exit_code = login_and_start_main(script_args(scratch_config, tmp_path))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "ZERODHA_API_KEY" in capsys.readouterr().err


def test_login_and_start_rejects_a_public_redirect_url(
    scratch_config: Path, tmp_path: Path, clean_env, monkeypatch, capsys
) -> None:
    """An app whose redirect URL is not loopback cannot be caught - say so, don't hang."""
    monkeypatch.setenv("ZERODHA_API_KEY", "testkey")
    monkeypatch.setenv("ZERODHA_API_SECRET", "testsecret")
    starts = _record_capture_start(monkeypatch)

    exit_code = login_and_start_main(
        script_args(scratch_config, tmp_path, "--redirect-url", "http://kite.example.com:8444/")
    )

    assert exit_code == EXIT_LOGIN_FAILED
    assert starts == []
    assert "not a loopback address" in capsys.readouterr().err

