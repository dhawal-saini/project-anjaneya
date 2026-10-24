"""ANJANEYA doctor - the pre-flight an operator runs before trusting a session.

``scripts/health_check.py`` answers *is the installation sound?*. This script
answers the next question - *can the morning routine succeed right now?* - by
reusing those same checks and adding the four things that actually stop a capture:

* is a **usable Kite token** cached, and when does it expire (or is a TOTP/manual
  login needed at all);
* how old is the **instrument dump** the offline universe plan reads;
* is the **database writable** right now (DuckDB allows a single writer, so a live
  capture locks it - that is expected, not an error);
* is the **Windows schedule** installed (``scripts/schedule_tasks.ps1``).

It is offline and read-only apart from the tiny write probe above.

Usage::

    uv run python scripts/doctor.py
    uv run python scripts/doctor.py --json
    uv run python scripts/doctor.py --data-root ./scratch --db-path ./scratch/x.duckdb

Exit codes: 0 = no problems, 1 = problems found, 2 = the configuration could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/doctor.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.auth import TokenStore  # noqa: E402
from core.config_loader import AnjaneyaConfig  # noqa: E402
from core.constants import get_timezone  # noqa: E402
from core.env import load_env_file  # noqa: E402
from core.market_calendar import NseCalendar  # noqa: E402
from core.storage import connect  # noqa: E402
from scripts.health_check import (  # noqa: E402
    check_calendar,
    check_capture,
    check_configuration,
    check_credentials,
    check_data_progress,
    check_database,
    check_env_file,
    check_timezone,
)

EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_CONFIG_ERROR = 2

#: The scheduled tasks ``scripts/schedule_tasks.ps1`` installs.
SCHEDULED_TASKS = (r"\Anjaneya\Anjaneya-Morning", r"\Anjaneya\Anjaneya-PostClose")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/doctor.py",
        description="Pre-flight checks: environment, token, instrument dump, database, schedule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML configuration.")
    parser.add_argument("--data-root", default=None, help="Override settings.data.root for this run.")
    parser.add_argument("--db-path", default=None, help="Override the DuckDB file for this run.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON.")
    return parser.parse_args(argv)



def check_login(config: AnjaneyaConfig) -> dict[str, Any]:
    """Report which login route the credentials allow, and whether the scheduler can use it.

    TOTP is the only route a scheduled, unattended morning run can use; without it
    the operator must paste a ``request_token`` by hand (or run
    ``scripts/login_and_start.py`` in a browser session).
    """
    zerodha = config.data_sources.zerodha
    names = (zerodha.user_id_env, zerodha.password_env, zerodha.totp_key_env)
    present = {name: bool(os.environ.get(name)) for name in names}
    totp = all(present.values())
    return {
        "ok": True,
        "totp_login_available": totp,
        "variables": present,
        "note": (
            "TOTP login is available - an unattended morning run can log in."
            if totp
            else "TOTP login is unavailable; set ZERODHA_USER_ID, ZERODHA_PASSWORD and "
            "ZERODHA_TOTP_KEY for unattended runs, or paste a request_token by hand."
        ),
    }


def check_token(config: AnjaneyaConfig, *, need_token: bool) -> dict[str, Any]:
    """Report the cached access token, its expiry and whether it is still usable.

    Zerodha invalidates the token every day, so a missing/expired token is only a
    problem when ``need_token`` is true (a trading day, before the close).
    """
    zerodha = config.data_sources.zerodha
    now = datetime.now(get_timezone(config.timezone_name))
    store = TokenStore(
        config.token_store_path,
        timezone_name=config.timezone_name,
        validity_cutoff=zerodha.token_store.validity_cutoff,
    )
    token = store.load()
    api_key = (os.environ.get(zerodha.api_key_env) or "").strip() or None
    usable = store.load_usable(
        api_key=api_key, now=now, min_remaining_minutes=zerodha.token_store.min_remaining_minutes
    )
    payload: dict[str, Any] = {
        "ok": True,
        "path": str(config.token_store_path),
        "present": token is not None,
        "usable": usable is not None,
        "need_token_now": need_token,
    }
    if token is not None:
        remaining = token.remaining_minutes(now)
        payload["expires_at"] = token.expires_at.isoformat() if token.expires_at else None
        payload["remaining_minutes"] = round(remaining, 1)
        payload["obtained_via"] = token.obtained_via
    if need_token and usable is None:
        payload["ok"] = False
        payload["hint"] = (
            "a trading day is in progress and no usable token is cached; run "
            "scripts/daily_login.py (TOTP or --request-token) before starting the capture"
        )
    return payload


def check_instrument_dump(config: AnjaneyaConfig) -> dict[str, Any]:
    """Report the age of the cached Kite instrument dump the offline universe plan reads."""
    cache_dir = config.instrument_cache_dir
    refresh_ttl = config.data_sources.zerodha.capture.instruments.refresh_ttl_hours
    dumps = sorted(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    if not dumps:
        return {
            "ok": False,
            "dir": str(cache_dir),
            "present": False,
            "hint": "no instrument dump yet; run `--mode instruments` (needs credentials) so the "
            "offline universe plan can resolve contracts.",
        }
    newest = max(dumps, key=lambda path: path.stat().st_mtime)
    age_hours = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600.0
    fresh = age_hours <= refresh_ttl
    return {
        "ok": fresh,
        "dir": str(cache_dir),
        "present": True,
        "newest": newest.name,
        "files": len(dumps),
        "age_hours": round(age_hours, 1),
        "refresh_ttl_hours": refresh_ttl,
        "hint": (
            None
            if fresh
            else f"the newest dump is {age_hours:.1f}h old (TTL {refresh_ttl}h); refresh it with "
            "`scripts/run_morning.py --refresh-instruments` before the next session"
        ),
    }


def check_database_writable(config: AnjaneyaConfig) -> dict[str, Any]:
    """Probe the database for write access (a live capture holding it is expected, not an error)."""
    if not config.db_path.exists():
        return {"ok": False, "writable": False, "hint": "no database yet; run any mode to create it"}
    try:
        connection = connect(config.db_path)
    except Exception as exc:  # noqa: BLE001 - almost always "a capture is running"
        return {
            "ok": True,
            "writable": False,
            "note": f"not writable right now ({type(exc).__name__}): DuckDB allows a single writer - "
            "a capture is probably running. See the capture section for the heartbeat.",
        }
    try:
        connection.execute("SELECT 1")
    finally:
        connection.close()
    return {"ok": True, "writable": True, "path": str(config.db_path)}


def check_scheduler() -> dict[str, Any]:
    """Report whether the Windows scheduled tasks are installed (Windows-only, informational)."""
    if sys.platform != "win32":
        return {"ok": True, "platform": sys.platform, "note": "Task Scheduler is Windows-only"}
    installed: dict[str, str] = {}
    for name in SCHEDULED_TASKS:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no user text
                ["schtasks", "/query", "/tn", name, "/fo", "LIST"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return {"ok": True, "platform": sys.platform, "note": "schtasks not found"}
        installed[name] = "installed" if completed.returncode == 0 else "not installed"
    return {
        "ok": True,
        "platform": sys.platform,
        "tasks": installed,
        "note": "install the weekday jobs with scripts/schedule_tasks.ps1 install",
    }



#: Section order, and the title each one is rendered under.
SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("timezone", "timezone"),
    ("env_file", "env file"),
    ("configuration", "configuration"),
    ("calendar", "calendar"),
    ("credentials", "credentials"),
    ("login", "login route"),
    ("token", "cached token"),
    ("instrument_dump", "instrument dump"),
    ("database", "database"),
    ("database_writable", "database writable"),
    ("capture", "capture"),
    ("data_progress", "data progress"),
    ("scheduler", "scheduler"),
)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run every check and return the report - the doctor's whole opinion."""
    env_file = load_env_file(config_dir=args.config_dir)
    report: dict[str, Any] = {"timezone": check_timezone(), "env_file": check_env_file(env_file)}
    configuration, config = check_configuration(
        Path(args.config_dir) if args.config_dir else None, args.data_root, args.db_path
    )
    report["configuration"] = configuration

    if config is None:
        skipped: dict[str, Any] = {"ok": False, "error": "skipped: configuration is invalid"}
        for key, _title in SECTION_TITLES:
            report.setdefault(key, dict(skipped))
        return report

    calendar = NseCalendar(config.calendar)
    now = calendar.now()
    today = now.date()
    trading = calendar.is_trading_day(today)
    need_token = trading and now <= calendar.session_end(today)

    calendar_payload = check_calendar(config)
    calendar_payload["today"] = today.isoformat()
    calendar_payload["today_is_trading_day"] = trading
    report["calendar"] = calendar_payload
    report["credentials"] = check_credentials(config)
    report["login"] = check_login(config)
    report["token"] = check_token(config, need_token=need_token)
    report["instrument_dump"] = check_instrument_dump(config)
    report["database"] = check_database(config)
    report["database_writable"] = check_database_writable(config)
    report["capture"] = check_capture(config)
    report["data_progress"] = check_data_progress(config)
    report["scheduler"] = check_scheduler()
    return report


def is_healthy(report: dict[str, Any]) -> bool:
    """True when every section reports ``ok``."""
    return all(bool(report.get(key, {}).get("ok")) for key, _title in SECTION_TITLES)


def render(report: dict[str, Any]) -> str:
    """Render the report as an operator-friendly block."""
    lines: list[str] = ["ANJANEYA doctor", "=" * 62]
    for key, title in SECTION_TITLES:
        payload = report.get(key, {})
        status = "OK" if payload.get("ok") else "PROBLEM"
        lines.append(f"\n[{status}] {title}")
        for field, value in payload.items():
            if field == "ok" or value is None:
                continue
            lines.append(f"    {field:<24}: {value}")
    lines.append("\n" + "=" * 62)
    lines.append("RESULT: READY" if is_healthy(report) else "RESULT: PROBLEMS FOUND (see above)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the doctor. Returns 0 ready, 1 problems, 2 the configuration could not be read."""
    args = parse_args(argv)
    report = build_report(args)
    config_problem = report["configuration"]
    if not config_problem.get("ok") and str(config_problem.get("error", "")).startswith("skipped"):
        print(f"ANJANEYA doctor: {config_problem['error']}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render(report))
    return EXIT_OK if is_healthy(report) else EXIT_PROBLEMS


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
