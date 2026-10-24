"""Tests for ``scripts/run_morning.py`` - the guarded morning routine.

The whole point of this script is that it *refuses* to start a capture the two ways
that lose a day silently: on a holiday it exits cleanly without touching the broker,
and when the option universe resolves to zero contracts it exits ``4`` instead of
recording an index-only session. Both are pinned here, along with the happy path
(login verified, then the capture started) and the forwarding of ``--`` flags. The
children are all replaced by recorders, so nothing touches the network or a broker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import run_morning
from scripts.run_morning import main as run_morning_main
from scripts.run_morning import parse_args

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

SUBSCRIPTIONS_30 = "  subscriptions       : 30 option contract(s) + 2 reference instrument(s) in 1 batch(es)"
SUBSCRIPTIONS_0 = "  subscriptions       : 0 option contract(s) + 2 reference instrument(s) in 1 batch(es)"


class FakeCalendar:
    """A calendar whose verdict the test controls."""

    trading = True
    weekend = False

    def __init__(self, _config: object) -> None:  # noqa: D401 - test double
        pass

    def now(self) -> datetime:
        return datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    def is_trading_day(self, _day: object) -> bool:
        return FakeCalendar.trading

    def is_weekend(self, _day: object) -> bool:
        return FakeCalendar.weekend

    def holiday_name(self, _day: object) -> None:
        return None

    def unverified_holiday_risk(self, **_kwargs: object) -> list[object]:
        return []


@pytest.fixture
def fake_calendar(monkeypatch: pytest.MonkeyPatch) -> type[FakeCalendar]:
    FakeCalendar.trading = True
    FakeCalendar.weekend = False
    monkeypatch.setattr(run_morning, "NseCalendar", FakeCalendar)
    return FakeCalendar


def cli(tmp_path: Path, *extra: str) -> list[str]:
    """Morning arguments rooted in ``tmp_path`` with the shipped config."""
    return [
        "--config-dir",
        str(CONFIG_DIR),
        "--data-root",
        str(tmp_path / "data"),
        "--db-path",
        str(tmp_path / "db" / "anjaneeya.duckdb"),
        *extra,
    ]


def _strip_remainder(args: object) -> object:
    if getattr(args, "capture_args", None) and args.capture_args[0] == "--":
        args.capture_args = args.capture_args[1:]
    return args


def test_capture_argv_forwards_the_remainder(tmp_path: Path) -> None:
    args = parse_args(cli(tmp_path, "--", "--smoke", "--until", "15:30"))
    # A leading `--` is stripped so the child sees only real capture flags.
    argv = run_morning._capture_argv(_strip_remainder(args), dry_run=False)
    assert argv[-3:] == ["--smoke", "--until", "15:30"]
    assert "--dry-run" not in argv


# ---------------------------------------------------------------------------
# The universe gate
# ---------------------------------------------------------------------------
def test_universe_gate_opens_on_a_non_zero_subscription_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, SUBSCRIPTIONS_30))
    args = parse_args(cli(tmp_path))
    code, subscriptions = run_morning._universe_gate(args, lambda _line: None)
    assert (code, subscriptions) == (run_morning.EXIT_OK, 30)


def test_universe_gate_refuses_zero_subscriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, SUBSCRIPTIONS_0))
    args = parse_args(cli(tmp_path))
    lines: list[str] = []
    code, subscriptions = run_morning._universe_gate(args, lines.append)
    assert code == run_morning.EXIT_GATE_REFUSED
    assert subscriptions == 0
    assert any("resolved to 0 contract" in line for line in lines)


def test_universe_gate_treats_a_failed_plan_as_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (2, "boom"))
    args = parse_args(cli(tmp_path))
    code, subscriptions = run_morning._universe_gate(args, lambda _line: None)
    assert (code, subscriptions) == (run_morning.EXIT_CONFIG_ERROR, None)


def test_universe_gate_without_a_subscription_line_is_a_gate_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, "nothing here"))
    args = parse_args(cli(tmp_path))
    code, subscriptions = run_morning._universe_gate(args, lambda _line: None)
    assert (code, subscriptions) == (run_morning.EXIT_GATE_REFUSED, None)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def test_main_exits_clean_with_no_session_on_a_holiday(
    tmp_path: Path, fake_calendar: type[FakeCalendar], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    fake_calendar.trading = False

    def forbidden(*_a: object, **_k: object) -> tuple[int, str]:
        raise AssertionError("a holiday must not start a capture child")

    monkeypatch.setattr(run_morning, "_run_capture_child", forbidden)
    exit_code = run_morning_main(cli(tmp_path))
    assert exit_code == run_morning.EXIT_OK
    assert "NO_SESSION" in capsys.readouterr().out


def test_main_refuses_when_the_universe_is_empty(
    tmp_path: Path, fake_calendar: type[FakeCalendar], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, SUBSCRIPTIONS_0))
    starts: list[list[str]] = []
    monkeypatch.setattr(run_morning, "start_capture", lambda argv: starts.append(argv) or 0)
    exit_code = run_morning_main(cli(tmp_path))
    captured = capsys.readouterr()
    assert exit_code == run_morning.EXIT_GATE_REFUSED
    assert starts == []
    assert "NOT started" in captured.out
    assert "session not started" in captured.err


def test_main_logs_in_then_starts_capture(
    tmp_path: Path, fake_calendar: type[FakeCalendar], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, SUBSCRIPTIONS_30))
    logins: list[list[str]] = []

    def fake_login(argv: list[str]) -> tuple[int, str, str]:
        logins.append(list(argv))
        return 0, "login ok\n", ""

    starts: list[list[str]] = []
    monkeypatch.setattr(run_morning, "run_login", fake_login)
    monkeypatch.setattr(run_morning, "start_capture", lambda argv: starts.append(argv) or 0)
    exit_code = run_morning_main(cli(tmp_path, "--", "--smoke"))
    assert exit_code == 0
    assert len(logins) == 1 and len(starts) == 1
    assert "--smoke" in starts[0]


def test_main_does_not_start_the_capture_when_login_fails(
    tmp_path: Path, fake_calendar: type[FakeCalendar], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, SUBSCRIPTIONS_30))
    monkeypatch.setattr(run_morning, "run_login", lambda _argv: (2, "", "no token"))
    starts: list[list[str]] = []
    monkeypatch.setattr(run_morning, "start_capture", lambda argv: starts.append(argv) or 0)
    exit_code = run_morning_main(cli(tmp_path))
    assert exit_code == run_morning.EXIT_FAILED
    assert starts == []
    assert "was NOT started" in capsys.readouterr().err


def test_main_dry_run_never_logs_in_or_starts(
    tmp_path: Path, fake_calendar: type[FakeCalendar], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(run_morning, "_run_capture_child", lambda *_a, **_k: (0, SUBSCRIPTIONS_30))

    def forbidden_login(_argv: list[str]) -> tuple[int, str, str]:
        raise AssertionError("--dry-run must not log in")

    monkeypatch.setattr(run_morning, "run_login", forbidden_login)
    monkeypatch.setattr(run_morning, "start_capture", lambda _argv: pytest.fail("no start"))
    exit_code = run_morning_main(cli(tmp_path, "--dry-run", "--json"))
    assert exit_code == run_morning.EXIT_OK
    assert '"status": "DRY-RUN"' in capsys.readouterr().out
