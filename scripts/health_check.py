"""ANJANEYA health check - one offline command that answers:

* does the Python environment resolve ``Asia/Kolkata`` (the Windows tzdata trap)?
* is the configuration valid, and which files/hash are in force?
* was a ``.env`` file found next to the config directory, and was it applied?
* is the DuckDB file present, at which schema version, with which tables?
* is the NSE calendar self-consistent (expiry weekday, buckets, holidays)?
* are the Zerodha credentials present in the environment (names only, never values)?
* how far along is the data-accumulation timeline (spec 4.3 gates)?

Usage::

    uv run python scripts/health_check.py
    uv run python scripts/health_check.py --json
    uv run python scripts/health_check.py --data-root ./scratch --db-path ./scratch/x.duckdb

Exit codes: 0 = healthy, 1 = problems found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/health_check.py`
    sys.path.insert(0, str(REPO_ROOT))

from apps.anjaneeya_capture.heartbeat import heartbeat_age_seconds, read_heartbeat  # noqa: E402
from core.config_loader import AnjaneyaConfig, ConfigError, load_config  # noqa: E402
from core.constants import DEFAULT_TIMEZONE, get_timezone  # noqa: E402
from core.env import SKIP_ENV_FILE_VAR, EnvFileResult, load_env_file  # noqa: E402
from core.market_calendar import NseCalendar  # noqa: E402
from core.storage import (  # noqa: E402
    bootstrap_database,
    connect,
    current_schema_version,
    table_names,
)

EXPECTED_TABLES = {
    "candles",
    "data_coverage",
    "features",
    "index_history",
    "instruments",
    "labels",
    "option_chain_snapshots",
    "run_manifest",
    "schema_version",
    "ticks",
    "trades",
}


def check_timezone() -> dict[str, object]:
    """Resolve the research timezone - the classic Windows failure mode."""
    try:
        tz = get_timezone(DEFAULT_TIMEZONE)
        offset = datetime(2026, 1, 1, tzinfo=tz).utcoffset()
        return {
            "ok": True,
            "timezone": DEFAULT_TIMEZONE,
            "utc_offset_hours": (offset.total_seconds() / 3600) if offset else None,
        }
    except Exception as exc:  # noqa: BLE001 - reporting failures is the whole point
        return {
            "ok": False,
            "timezone": DEFAULT_TIMEZONE,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Install the tzdata package (`uv sync`); Windows ships no IANA database.",
        }


def check_env_file(env_file: EnvFileResult) -> dict[str, object]:
    """Report whether a ``.env`` file was found and applied - never a value."""
    payload: dict[str, object] = {
        # A missing file or a deliberate skip is not an error: ANJANEYA also runs
        # fully offline, and no secret is ever required to build features.
        "ok": True,
        "path": str(env_file.path),
        "loaded": env_file.loaded,
        "skipped": env_file.skipped,
        "variables_from_file": list(env_file.names),
    }
    if env_file.skipped:
        payload["note"] = f"loading disabled by {SKIP_ENV_FILE_VAR}=1"
    elif env_file.loaded:
        payload["note"] = "a real environment variable always wins over a .env value"
    else:
        payload["note"] = (
            "no .env file: copy .env.example to .env (or export ZERODHA_API_KEY / "
            "ZERODHA_API_SECRET) before running a capture mode"
        )
    return payload


def check_configuration(
    config_dir: Path | None, data_root: str | None, db_path: str | None
) -> tuple[dict[str, object], AnjaneyaConfig | None]:
    """Load and validate the configuration, reporting the resolved summary."""
    if data_root:
        os.environ["ANJANEYA_DATA_ROOT"] = data_root
    if db_path:
        os.environ["ANJANEYA_DB_PATH"] = db_path
    try:
        config = load_config(config_dir)
    except ConfigError as exc:
        return {"ok": False, "error": str(exc)}, None
    return {"ok": True, **config.summary()}, config


def check_calendar(config: AnjaneyaConfig) -> dict[str, object]:
    """Verify the calendar is internally consistent and report its state."""
    calendar = NseCalendar(config.calendar)
    described: dict[str, object] = dict(calendar.describe())
    problems: list[str] = []

    weekday = config.calendar.weekly_expiry.weekday
    if weekday in {"Saturday", "Sunday"}:
        problems.append(f"weekly expiry weekday {weekday} is a weekend day")

    # Every minute of the analysis window must fall inside a configured bucket.
    probe_day = calendar.next_trading_day(date.today())
    probe = calendar.session_start(probe_day)
    end = datetime.combine(probe_day, config.settings.market.analysis_end, tzinfo=calendar.timezone)
    uncovered = 0
    while probe < end:
        if calendar.time_of_day_bucket(probe) == "OUT_OF_SESSION":
            uncovered += 1
        probe += timedelta(minutes=1)
    if uncovered:
        problems.append(f"{uncovered} minute(s) of the analysis window are not covered by a bucket")

    if not config.calendar.holidays_verified:
        described["warning"] = (
            f"{len(config.calendar.unverified_holidays)} holiday date(s) are provisional; "
            "reconcile them with the official NSE circular."
        )
    return {"ok": not problems, "problems": problems, **described}


def check_database(config: AnjaneyaConfig) -> dict[str, object]:
    """Bootstrap (if needed) and inspect the DuckDB file."""
    db_path = config.db_path
    existed = db_path.exists()
    try:
        bootstrap_database(
            db_path,
            schema_version=config.settings.project.schema_version,
            description=f"{config.settings.project.full_title} schema",
        )
    except Exception as exc:  # noqa: BLE001 - report instead of crashing
        return {"ok": False, "path": str(db_path), "error": f"{type(exc).__name__}: {exc}"}

    connection = connect(db_path, read_only=True)
    try:
        version = current_schema_version(connection)
        tables = set(table_names(connection))
        counts = {
            table: connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            for table in sorted(tables)
            if table != "schema_version"
        }
    finally:
        connection.close()

    missing = sorted(EXPECTED_TABLES - tables)
    return {
        "ok": not missing,
        "path": str(db_path),
        "existed_before": existed,
        "schema_version": version,
        "expected_schema_version": config.settings.project.schema_version,
        "missing_tables": missing,
        "row_counts": counts,
    }


def check_credentials(config: AnjaneyaConfig) -> dict[str, object]:
    """Report credential *presence* only - never a value."""
    zerodha = config.data_sources.zerodha
    names = (
        zerodha.api_key_env,
        zerodha.api_secret_env,
        zerodha.user_id_env,
        zerodha.password_env,
        zerodha.totp_key_env,
    )
    presence = {name: bool(os.environ.get(name)) for name in names}
    token_path = config.token_store_path
    return {
        # Missing credentials are not an error for offline work (status/engine).
        "ok": True,
        "variables": presence,
        "automated_totp_login_available": all(
            presence[name]
            for name in (
                zerodha.api_key_env,
                zerodha.api_secret_env,
                zerodha.user_id_env,
                zerodha.password_env,
                zerodha.totp_key_env,
            )
        ),
        "token_store": str(token_path),
        "cached_token_present": token_path.exists(),
        "note": (
            "Capture needs ZERODHA_API_KEY + ZERODHA_API_SECRET plus either a request token "
            "(--request-token) or user id/password/TOTP for automated login."
        ),
    }


def check_data_progress(config: AnjaneyaConfig) -> dict[str, object]:
    """Report progress against the spec 4.3 accumulation gates."""
    trading_days = 0
    if config.db_path.exists():
        try:
            connection = connect(config.db_path, read_only=True)
            try:
                row = connection.execute(
                    "SELECT count(DISTINCT date_trunc('day', timestamp)) FROM ticks"
                ).fetchone()
                trading_days = int(row[0]) if row and row[0] is not None else 0
            finally:
                connection.close()
        except Exception:  # noqa: BLE001 - a fresh/empty database is not an error
            trading_days = 0

    gates = config.model.data_gates
    return {
        "ok": True,
        "captured_trading_days": trading_days,
        "exploratory_ready": trading_days >= gates.exploratory_min_days,
        "ml_training_ready": trading_days >= gates.ml_training_min_days,
        "backtest_ready": trading_days >= gates.meaningful_backtest_min_days,
        "seasonality_ready": trading_days >= gates.seasonality_min_days,
        "thresholds": {
            "exploratory_min_days": gates.exploratory_min_days,
            "ml_training_min_days": gates.ml_training_min_days,
            "meaningful_backtest_min_days": gates.meaningful_backtest_min_days,
            "seasonality_min_days": gates.seasonality_min_days,
        },
        "note": (
            "Do not draw conclusions below these thresholds (spec 4.3). "
            "Captured days are counted from stored ticks."
        ),
    }


def check_capture(config: AnjaneyaConfig) -> dict[str, object]:
    """Report the capture-side state: heartbeat freshness and the coverage ledger.

    This is the only *live* view of a running capture: DuckDB permits a single
    read-write process, so while capture holds the database a second process
    cannot open it. The heartbeat file is written for exactly that situation.
    """
    capture = config.data_sources.zerodha.capture
    heartbeat_path = config.heartbeat_path()
    payload = read_heartbeat(heartbeat_path)
    problems: list[str] = []

    heartbeat: dict[str, object]
    if payload is None:
        heartbeat = {
            "path": str(heartbeat_path),
            "present": False,
            "note": "no capture heartbeat yet (capture has never run here)",
        }
    else:
        age = heartbeat_age_seconds(payload)
        stale = age is None or age > capture.heartbeat.stale_after_seconds
        heartbeat = {
            "path": str(heartbeat_path),
            "present": True,
            "status": payload.get("status"),
            "run_id": payload.get("run_id"),
            "age_seconds": round(age, 1) if age is not None else None,
            "stale": stale,
            "connected": payload.get("connected"),
            "ticks_total": payload.get("ticks_total"),
            "snapshots_total": payload.get("snapshots_total"),
            "gap_count": payload.get("gap_count"),
            "poll_failures": payload.get("poll_failures"),
            "session_date": payload.get("session_date"),
        }
        if stale and payload.get("status") not in {"STOPPED", "FAILED", None}:
            problems.append(
                f"the capture heartbeat is {age:.0f}s old (stale after "
                f"{capture.heartbeat.stale_after_seconds:.0f}s) but reports status "
                f"{payload.get('status')!r}: the capture process may have died"
            )
        if payload.get("poll_failures"):
            problems.append(
                f"{payload['poll_failures']} option-chain poll(s) failed in the last session"
            )

    coverage = _coverage_summary(config)
    return {
        "ok": not problems,
        "problems": problems,
        "heartbeat": heartbeat,
        "policy": {
            "mode": config.data_sources.zerodha.websocket.mode,
            "out_of_session_ticks": capture.out_of_session_ticks,
            "tick_gap_seconds": capture.gaps.tick_gap_seconds,
            "flush_rows": capture.storage.flush_rows,
            "flush_seconds": capture.storage.flush_seconds,
            "mirror_parquet": capture.storage.mirror_parquet,
            "universe_refresh_minutes": capture.universe_refresh_minutes,
        },
        "coverage": coverage,
    }


def _coverage_summary(config: AnjaneyaConfig) -> dict[str, object]:
    """Summarise the coverage ledger (best effort: the DB may be locked)."""
    if not config.db_path.exists():
        return {"available": False, "note": "no database yet"}
    try:
        connection = connect(config.db_path, read_only=True)
    except Exception as exc:  # noqa: BLE001 - almost always "capture is running"
        return {
            "available": False,
            "note": (
                f"the database is not readable right now ({type(exc).__name__}); "
                "capture is probably running - use the heartbeat above"
            ),
        }
    try:
        rows = connection.execute(
            "SELECT trading_date, quality_flag, count(*) FROM data_coverage "
            "GROUP BY trading_date, quality_flag ORDER BY trading_date DESC LIMIT 20"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 - table missing on a very old database
        return {"available": False, "note": f"{type(exc).__name__}: {exc}"}
    finally:
        connection.close()
    return {
        "available": True,
        "recent": [
            {"trading_date": str(row[0]), "quality_flag": row[1], "streams": row[2]}
            for row in rows
        ],
    }


def render(report: dict[str, object]) -> str:
    """Render the report as an operator-friendly block."""
    lines: list[str] = ["ANJANEYA health check", "=" * 62]

    def section(title: str, payload: dict) -> None:
        status = "OK" if payload.get("ok") else "PROBLEM"
        lines.append(f"\n[{status}] {title}")
        for key, value in payload.items():
            if key == "ok":
                continue
            lines.append(f"    {key:<34}: {value}")

    section("timezone", report["timezone"])
    section("env file", report["env_file"])
    section("configuration", report["configuration"])
    section("calendar", report["calendar"])
    section("database", report["database"])
    section("credentials", report["credentials"])
    section("capture", report["capture"])
    section("data progress", report["data_progress"])

    healthy = all(bool(report[key].get("ok")) for key in report)
    lines.append("\n" + "=" * 62)
    lines.append("RESULT: HEALTHY" if healthy else "RESULT: PROBLEMS FOUND (see above)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run every check and return a process exit code (0 healthy, 1 problems)."""
    parser = argparse.ArgumentParser(description="ANJANEYA offline health check")
    parser.add_argument("--config-dir", default=None, help="Configuration directory.")
    parser.add_argument("--data-root", default=None, help="Override settings.data.root.")
    parser.add_argument("--db-path", default=None, help="Override settings.data.db_path.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON.")
    args = parser.parse_args(argv)

    # `.env` is loaded before the configuration so its values can satisfy YAML
    # placeholders and reach the credential check (see core/env.py).
    env_file = load_env_file(config_dir=args.config_dir)

    report: dict[str, object] = {"timezone": check_timezone(), "env_file": check_env_file(env_file)}
    configuration, config = check_configuration(
        Path(args.config_dir) if args.config_dir else None, args.data_root, args.db_path
    )
    report["configuration"] = configuration

    if config is None:
        skipped: dict[str, object] = {"ok": False, "error": "skipped: configuration is invalid"}
        report["calendar"] = dict(skipped)
        report["database"] = dict(skipped)
        report["credentials"] = dict(skipped)
        report["capture"] = dict(skipped)
        report["data_progress"] = dict(skipped)
    else:
        report["calendar"] = check_calendar(config)
        report["database"] = check_database(config)
        report["credentials"] = check_credentials(config)
        report["capture"] = check_capture(config)
        report["data_progress"] = check_data_progress(config)

    if args.as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render(report))

    return 0 if all(bool(report[key].get("ok")) for key in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
