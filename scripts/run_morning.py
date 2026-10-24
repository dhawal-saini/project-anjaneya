"""ANJANEYA morning routine - one command that opens a capture session safely.

The daily operator loop is: confirm the day trades, make sure the option universe
still resolves, mint a fresh Kite token, then start the session. Skipping the
first two steps is exactly how the two silent failures in §1.4 happen - a capture
runs on a holiday and stores nothing, or the universe resolves to *zero*
contracts and the session saves the index with no options while still exiting 0.
Both are discovered hours later, when the ticks are gone for good.

This script performs those steps as one fail-fast chain:

1. **calendar guard** - on a non-trading day it prints ``NO_SESSION`` and exits
   ``0`` without touching the broker (a holiday is not an error);
2. **instrument refresh** (``--refresh-instruments``) - re-downloads Kite's
   instrument master first, so an expired cached dump cannot hide the universe;
3. **dry-run gate** - resolves the option universe offline and refuses to start
   unless it holds at least one contract (exit ``4``);
4. **login** - runs ``scripts/daily_login.py`` (cached token, TOTP, or a pasted
   ``--request-token``), so a dead token can never reach the session as a silent
   no-capture run;
5. **start** - only after the login verified, starts
   ``apps.anjaneeya_capture.run --mode capture`` (exactly what
   ``scripts/start_capture.ps1`` does) and propagates its exit code unchanged.

Usage::

    uv run python scripts/run_morning.py
    uv run python scripts/run_morning.py --refresh-instruments
    uv run python scripts/run_morning.py --dry-run
    uv run python scripts/run_morning.py --json
    uv run python scripts/run_morning.py -- --smoke --until 15:30

Exit codes: 0 = ok (a clean ``NO_SESSION`` on a holiday is a 0), 1 = the login or
the capture failed, 2 = configuration/credentials problem, 4 = the capture plan
resolved to zero subscriptions, so the session was NOT started. Secrets are never
printed - only masked forms; the ``request_token`` never appears in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_morning.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.app_bootstrap import (  # noqa: E402
    EXIT_CONFIG_ERROR,
    EXIT_FAILED,
    EXIT_GATE_REFUSED,
    EXIT_OK,
)
from core.config_loader import AnjaneyaConfig, ConfigError, load_config  # noqa: E402
from core.env import EnvFileResult, load_env_file  # noqa: E402
from core.market_calendar import NseCalendar  # noqa: E402
from scripts.daily_login import EXIT_OK as LOGIN_EXIT_OK  # noqa: E402
from scripts.login_and_start import run_login, start_capture  # noqa: E402

#: How far ahead a provisional holiday is worth shouting about (matches the pipeline).
RISK_WINDOW_DAYS = 7

#: The line ``--mode capture --dry-run`` prints once it has resolved the universe.
SUBSCRIPTIONS_RE = re.compile(r"subscriptions\s*:\s*(\d+)\s+option contract")



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_morning.py",
        description="Guard the calendar, verify the universe, log in to Kite and start the "
        "capture session - one command for the morning routine.",
        epilog=(
            "Arguments after `--` are forwarded to the capture app, e.g. "
            "`-- --smoke --until 15:30`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML configuration.")
    parser.add_argument("--data-root", default=None, help="Override settings.data.root for this run.")
    parser.add_argument("--db-path", default=None, help="Override the DuckDB file for this run.")
    parser.add_argument(
        "--refresh-instruments",
        action="store_true",
        help="Re-download Kite's instrument master before resolving the universe.",
    )
    parser.add_argument(
        "--request-token",
        default=None,
        help="Exchange this Kite request_token during the login instead of a TOTP login.",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Never log in: use the cached token, and fail if there is no usable one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and run the offline checks; never log in or start the session.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a JSON summary instead of the report.",
    )
    parser.add_argument(
        "capture_args",
        nargs=argparse.REMAINDER,
        help="Forwarded to the capture app; prefix them with `--`.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _capture_argv(args: argparse.Namespace, *, dry_run: bool) -> list[str]:
    """Arguments for the capture child (its own flags come after ``--``)."""
    argv: list[str] = []
    if args.config_dir:
        argv += ["--config-dir", str(args.config_dir)]
    if args.data_root:
        argv += ["--data-root", str(args.data_root)]
    if args.db_path:
        argv += ["--db-path", str(args.db_path)]
    if dry_run:
        argv += ["--dry-run"]
    return argv + list(args.capture_args or [])


def _run_capture_child(argv_tail: Sequence[str], *, capture_output: bool) -> tuple[int, str]:
    """Run ``apps.anjaneeya_capture.run`` in a child and return ``(code, output)``.

    A child process keeps the capture's logging, signal handling and crash
    behaviour identical to ``scripts/start_capture.ps1``; ``capture_output`` is on
    only when the plan has to be inspected (the dry-run gate).
    """
    argv = [sys.executable, "-m", "apps.anjaneeya_capture.run", *argv_tail]
    completed = subprocess.run(
        argv, cwd=str(REPO_ROOT), check=False, capture_output=capture_output, text=True
    )
    output = ((completed.stdout or "") + (completed.stderr or "")) if capture_output else ""
    return int(completed.returncode), output


def _login_argv(args: argparse.Namespace, request_token: str | None) -> list[str]:
    """Arguments for :mod:`scripts.daily_login` (the one login implementation)."""
    argv: list[str] = []
    if request_token:
        argv += ["--request-token", request_token]
    if args.config_dir:
        argv += ["--config-dir", str(args.config_dir)]
    if args.data_root:
        argv += ["--data-root", str(args.data_root)]
    return argv


def _calendar_verdict(
    config: AnjaneyaConfig, calendar: NseCalendar
) -> tuple[bool, date, list[str]]:
    """``(trades_today, today, warnings)`` - the offline calendar guard."""
    today = calendar.now().date()
    warnings = [
        f"holiday {entry.date.isoformat()} ({entry.name}) is still PROVISIONAL; if it is really a "
        "trading day the capture will exit NO_SESSION and the day is lost (scripts/check_calendar.py)."
        for entry in calendar.unverified_holiday_risk(today=today, days=RISK_WINDOW_DAYS)
    ]
    return calendar.is_trading_day(today), today, warnings


def _universe_gate(args: argparse.Namespace, emit: Callable[[str], None]) -> tuple[int, int | None]:
    """Resolve the universe offline and gate on ``subscriptions > 0``.

    Returns ``(exit_code, subscriptions)``. ``exit_code`` is ``0`` when the gate is
    open (``subscriptions`` is then the resolved count), ``2`` when the dry-run
    itself failed (a configuration problem) and ``4`` when it resolved to zero.
    """
    code, output = _run_capture_child(
        ["--mode", "capture", "--dry-run", *_capture_argv(args, dry_run=False)],
        capture_output=True,
    )
    for line in output.splitlines():
        emit(line.rstrip())
    if code != 0:
        emit(
            f"the capture plan failed (exit {code}); fix the configuration before starting a session"
        )
        return EXIT_CONFIG_ERROR, None
    match = SUBSCRIPTIONS_RE.search(output)
    if match is None:
        emit(
            "the capture plan printed no subscription line; cannot prove the session would "
            "capture options"
        )
        return EXIT_GATE_REFUSED, None
    subscriptions = int(match.group(1))
    if subscriptions <= 0:
        emit(
            "the option universe resolved to 0 contract(s): the session would capture the index "
            "and no options while still exiting 0. Refresh the instrument master "
            "(--refresh-instruments), reconcile the expiry weekday in config/calendar.yaml, then retry."
        )
        return EXIT_GATE_REFUSED, subscriptions
    return EXIT_OK, subscriptions



def main(argv: Sequence[str] | None = None) -> int:
    """Run the morning chain. Returns 0 ok, 1 failure, 2 configuration, 4 gate refused."""
    args = parse_args(argv)
    if args.capture_args and args.capture_args[0] == "--":
        args.capture_args = args.capture_args[1:]
    if args.data_root:
        os.environ["ANJANEYA_DATA_ROOT"] = str(args.data_root)
    if args.db_path:
        os.environ["ANJANEYA_DB_PATH"] = str(args.db_path)
    env_file: EnvFileResult = load_env_file(config_dir=args.config_dir)
    try:
        config = load_config(Path(args.config_dir) if args.config_dir else None)
    except ConfigError as exc:
        print(f"ANJANEYA morning: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    calendar = NseCalendar(config.calendar)
    report: dict[str, object] = {
        "config_dir": str(config.config_dir),
        "data_root": str(config.data_root),
        "db_path": str(config.db_path),
        "env_file": str(env_file.path),
        "env_file_loaded": env_file.loaded,
        "dry_run": bool(args.dry_run),
    }

    def emit(text: str = "") -> None:
        print(text)

    trades_today, today, warnings = _calendar_verdict(config, calendar)
    report["date"] = today.isoformat()
    report["trading_day"] = trades_today
    report["calendar_warnings"] = warnings

    if not trades_today:
        holiday = calendar.holiday_name(today)
        day_kind = "weekend" if calendar.is_weekend(today) else (holiday or "non-trading day")
        report.update(status="NO_SESSION", subscriptions=None, login=None, capture_exit_code=None)
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            emit("ANJANEYA morning routine")
            emit(f"  date          : {today.isoformat()} ({day_kind})")
            emit(
                f"  session       : NO_SESSION - the calendar says {today.isoformat()} does not trade"
            )
            emit(f"  env file      : {env_file.describe()}")
            emit("  action        : nothing to do; the capture is not started")
        return EXIT_OK

    if not args.as_json:
        emit("ANJANEYA morning routine")
        emit(f"  date          : {today.isoformat()} (trading day)")
        emit(f"  env file      : {env_file.describe()}")
        for warning in warnings:
            emit(f"  warning       : {warning}")

    if args.refresh_instruments:
        if args.dry_run:
            if not args.as_json:
                emit("  instruments   : would re-download the Kite instrument master (dry run)")
        else:
            if not args.as_json:
                emit("  instruments   : re-downloading the Kite instrument master")
            code, _ = _run_capture_child(
                ["--mode", "instruments", "--force-refresh", *_capture_argv(args, dry_run=False)],
                capture_output=False,
            )
            if code != 0:
                report.update(
                    status="FAILED", subscriptions=None, login=None, capture_exit_code=code
                )
                if not args.as_json:
                    emit(f"  instruments   : FAILED (exit {code}); not starting a session")
                return EXIT_FAILED

    gate_code, subscriptions = _universe_gate(
        args, emit if not args.as_json else (lambda _line: None)
    )
    report["subscriptions"] = subscriptions
    if gate_code != EXIT_OK:
        report.update(status="GATE_REFUSED", login=None, capture_exit_code=None)
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            emit("  verdict       : GATE_REFUSED - the capture session was NOT started")
        print("ANJANEYA morning: the option universe is empty; session not started", file=sys.stderr)
        return gate_code
    if not args.as_json:
        emit(f"  universe      : {subscriptions} option contract(s) resolved")

    if args.dry_run:
        report.update(status="DRY-RUN", login=None, capture_exit_code=None)
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            emit("  login         : would log in via scripts/daily_login.py")
            emit(
                f"  capture       : would run mode=capture with "
                f"{args.capture_args or 'no extra flags'}"
            )
            emit("  verdict       : DRY-RUN - nothing was logged in or started")
        return EXIT_OK

    if args.skip_login:
        if not args.as_json:
            emit("  login         : skipped (--skip-login); the session will use the cached token")
        report["login"] = "skipped"
    else:
        if not args.as_json:
            emit("  login         : running scripts/daily_login.py")
        code, out, err = run_login(_login_argv(args, args.request_token))
        if err:
            print(err, end="", file=sys.stderr)
        if code != LOGIN_EXIT_OK:
            report.update(status="LOGIN_FAILED", login="failed", capture_exit_code=None)
            if args.as_json:
                print(json.dumps(report, indent=2))
            if out:
                print(out, end="", file=sys.stderr)
            print(
                f"ANJANEYA morning: login failed (exit {code}); the capture session was NOT started.",
                file=sys.stderr,
            )
            return EXIT_FAILED
        report["login"] = "ok"
        if not args.as_json and out:
            print(out, end="")

    if not args.as_json:
        emit("  capture       : starting mode=capture")
    capture_code = start_capture(_capture_argv(args, dry_run=False))
    report["capture_exit_code"] = capture_code
    report["status"] = "OK" if capture_code == EXIT_OK else "FAILED"
    if args.as_json:
        print(json.dumps(report, indent=2))
    return capture_code


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
