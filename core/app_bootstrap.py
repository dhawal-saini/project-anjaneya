"""Shared CLI bootstrap for the three ANJANEYA applications.

Each app is deliberately thin: it declares its ``--mode`` values, then calls
:func:`bootstrap_app`, which performs the identical startup sequence:

1. load ``.env`` next to ``config/`` (unless ``ANJANEYA_SKIP_DOTENV`` is set),
2. apply CLI overrides (``--data-root``, ``--log-level``),
3. load + validate the YAML configuration (fail fast, never half-configured),
4. create the data directories and the DuckDB file,
5. configure logging with secret redaction,
6. create the run context (``run_id``, config hash, git commit, versions).

Step 1 precedes step 2 on purpose: ``.env`` never overrides a value that is
already in the environment, so an explicit CLI flag always wins over the file.

Keeping this in ``core`` means the three apps cannot drift apart, and Phase 1
can prove the whole startup path is correct before any market logic exists.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config_loader import AnjaneyaConfig, load_config
from core.env import load_env_file
from core.logging_setup import collect_secrets, setup_logging
from core.market_calendar import DEFAULT_HOLIDAY_RISK_DAYS, NseCalendar
from core.run_context import RunContext, create_run_context
from core.storage import bootstrap_database

__all__ = [
    "APP_AI",
    "APP_CAPTURE",
    "APP_ENGINE",
    "DEFAULT_MODES",
    "EXIT_CONFIG_ERROR",
    "EXIT_FAILED",
    "EXIT_GATE_REFUSED",
    "EXIT_NOT_IMPLEMENTED",
    "EXIT_OK",
    "AppContext",
    "bootstrap_app",
    "finalize_run",
    "parse_args",
    "render_status",
    "status_report",
]

APP_CAPTURE = "anjaneeya_capture"
APP_ENGINE = "anjaneeya_engine"
APP_AI = "anjaneeya_ai"

#: ``--mode`` choices per app. ``status`` is always available: it is how the
#: operator checks an installation without touching the network.
DEFAULT_MODES: dict[str, tuple[str, ...]] = {
    APP_CAPTURE: (
        "capture",
        "backfill",
        "backfill-options",
        "option-chain",
        "instruments",
        "status",
    ),
    APP_ENGINE: (
        "build-candles",
        "enrich-greeks",
        "universe",
        "build-features",
        "build-labels",
        "validate",
        "status",
    ),
    APP_AI: (
        "dataset",
        "train",
        "backtest",
        "report",
        "explain",
        "importance",
        "charts",
        "accumulation",
        "decide",
        "status",
    ),
}


@dataclass(slots=True)
class AppContext:
    """Everything an app needs after a successful bootstrap."""

    app_name: str
    mode: str
    config: AnjaneyaConfig
    run_context: RunContext
    calendar: NseCalendar
    logger: Any
    dry_run: bool = False

    @property
    def data_root(self) -> Path:
        """Absolute data root."""
        return self.config.data_root

    @property
    def db_path(self) -> Path:
        """Absolute DuckDB path."""
        return self.config.db_path


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    app_name: str,
    modes: Sequence[str] | None = None,
    description: str = "",
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Parse the common ANJANEYA CLI arguments plus the app's ``--mode`` choices.

    ``configure_parser`` lets an app add its own flags (capture's ``--smoke``,
    engine's ``--from``...) without forking the shared parser - so every app keeps
    the identical ``--config-dir``/``--dry-run``/``--json`` behaviour.
    """
    available = tuple(modes) if modes is not None else DEFAULT_MODES.get(app_name, ("status",))
    parser = argparse.ArgumentParser(
        prog=f"python -m apps.{app_name}.run",
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        default="status",
        choices=available,
        help="What to do. 'status' only reports state and never calls the network.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory containing the six YAML configuration files (default: ./config).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override settings.data.root for this run (useful for tests and scratch runs).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override settings.data.db_path for this run (the local DuckDB file).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging.level for this run (TRACE/DEBUG/INFO/WARNING/ERROR).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and report the plan without writing data or calling APIs.",
    )
    parser.add_argument(
        "--no-run-artifacts",
        action="store_true",
        help="Do not create a run directory / manifest for this execution.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the status report as JSON instead of a human-readable block.",
    )
    if configure_parser is not None:
        configure_parser(parser)
    return parser.parse_args(list(argv) if argv is not None else None)


def bootstrap_app(
    app_name: str,
    args: argparse.Namespace,
    *,
    logger: Any = None,
) -> AppContext:
    """Run the shared startup sequence and return an :class:`AppContext`.

    Raises
    ------
    ConfigError
        When configuration is missing or invalid. The app prints the message and
        exits non-zero; ANJANEYA never starts half-configured.
    """
    # 1. ``.env`` first, so its values can satisfy YAML placeholders - but never
    #    in place of a real environment variable (see core/env.py).
    env_file = load_env_file(config_dir=getattr(args, "config_dir", None))

    # 2. CLI overrides that must be visible to environment placeholder resolution.
    if getattr(args, "data_root", None):
        os.environ["ANJANEYA_DATA_ROOT"] = str(args.data_root)
    if getattr(args, "db_path", None):
        os.environ["ANJANEYA_DB_PATH"] = str(args.db_path)

    mode = getattr(args, "mode", "status")
    dry_run = bool(getattr(args, "dry_run", False))

    # 3. Configuration (validated, hashed, secret-free).
    config: AnjaneyaConfig = load_config(getattr(args, "config_dir", None))

    # 4. Directories + database must exist before the first write.
    if not dry_run:
        config.ensure_directories()
        bootstrap_database(
            config.db_path,
            schema_version=config.settings.project.schema_version,
            description=f"{config.settings.project.full_title} schema",
        )

    # 5. Logging with secret redaction.
    zerodha = config.data_sources.zerodha
    secrets = collect_secrets(
        os.environ,
        [
            zerodha.api_key_env,
            zerodha.api_secret_env,
            zerodha.password_env,
            zerodha.totp_key_env,
        ],
    )
    log_cfg = config.settings.logging
    logger = logger or setup_logging(
        app_name=app_name,
        log_dir=config.log_dir,
        level=(getattr(args, "log_level", None) or log_cfg.level),
        console=log_cfg.console,
        file=log_cfg.file,
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        json_sink=log_cfg.json_sink,
        json_sink_name=log_cfg.json_sink_name,
        intercept_stdlib=log_cfg.intercept_stdlib,
        secret_values=secrets,
    )

    # 6. Run context for reproducibility.
    write_artifacts = config.settings.run_tracking.enabled and not getattr(
        args, "no_run_artifacts", False
    )
    run_context = create_run_context(
        app_name,
        mode=mode,
        config_hash=config.config_hash,
        config_dir=config.config_dir,
        artifacts_root=config.runs_dir if write_artifacts else None,
        random_seed=config.model.random_seed,
        timezone_name=config.timezone_name,
        package_names=config.settings.run_tracking.record_packages,
        base_dir=config.base_dir,
    )

    context = AppContext(
        app_name=app_name,
        mode=mode,
        config=config,
        run_context=run_context,
        calendar=NseCalendar(config.calendar),
        logger=logger,
        dry_run=dry_run,
    )

    logger.info(f"ANJANEYA {app_name} starting: mode={mode} run_id={run_context.run_id}")
    logger.debug(f"environment: {env_file.describe()}")
    logger.info(
        f"config={config.config_hash[:12]} schema_version={config.settings.project.schema_version} "
        f"timezone={config.timezone_name} dry_run={dry_run}"
    )
    if not config.calendar.holidays_verified:
        count = len(config.calendar.unverified_holidays)
        message = (
            f"{count} holiday date(s) in config/calendar.yaml are still PROVISIONAL "
            "(holidays_verified: false). Reconcile them with the official NSE circular before "
            "drawing conclusions from expiry-related analysis."
        )
        logger.warning(message)
        run_context.add_note(message)
    if dry_run:
        logger.info("dry-run: no data will be written and no provider API will be called")
    return context


def status_report(context: AppContext) -> dict[str, Any]:
    """Collect an offline health snapshot for ``--mode status``.

    This deliberately performs **no network access**: it reports what the
    installation looks like right now (configuration, schema, calendar, data
    volume, credential *presence*), which is what an operator needs before
    starting a capture session.
    """
    from core.storage import connect, table_names

    config = context.config
    report: dict[str, Any] = {
        "app": context.app_name,
        "mode": context.mode,
        "config": config.summary(),
        "calendar": context.calendar.describe(),
        "run_id": context.run_context.run_id,
        "dry_run": context.dry_run,
    }

    # The one calendar fact an operator must not learn after the session: a holiday that is
    # still provisional can turn a real trading day into NO_SESSION, and that day's ticks
    # cannot be recovered. Advisory in every app's `--mode status`, never a refusal.
    today = context.calendar.now().date()
    risky = context.calendar.unverified_holiday_risk(today=today)
    report["calendar_risk"] = {
        "window_days": DEFAULT_HOLIDAY_RISK_DAYS,
        "count": len(risky),
        "unverified_holidays": [
            {
                "date": entry.date.isoformat(),
                "name": entry.name,
                "days_away": (entry.date - today).days,
            }
            for entry in risky
        ],
    }

    # Credential presence only - never the values themselves.
    zerodha = config.data_sources.zerodha
    report["credentials"] = {
        name: bool(os.environ.get(name))
        for name in (
            zerodha.api_key_env,
            zerodha.api_secret_env,
            zerodha.user_id_env,
            zerodha.password_env,
            zerodha.totp_key_env,
        )
    }
    report["rate_limits"] = zerodha.rate_limits.model_dump()

    # Database + row counts (read-only, and tolerant of a fresh install).
    report["database"] = {"path": str(config.db_path), "exists": config.db_path.exists()}
    if config.db_path.exists():
        try:
            connection = connect(config.db_path, read_only=True)
            try:
                tables = table_names(connection)
                counts: dict[str, int] = {}
                for table in tables:
                    if table == "schema_version":
                        continue
                    row = connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()
                    counts[table] = int(row[0]) if row else 0
                report["database"].update({"tables": tables, "row_counts": counts})
            finally:
                connection.close()
        except Exception as exc:  # noqa: BLE001 - a status report must never crash
            report["database"]["error"] = f"{type(exc).__name__}: {exc}"

    # Data-accumulation gates (spec 4.3): how far along is the dataset?
    coverage_dir = config.data_dir("candles")
    report["data_gates"] = {
        "exploratory_min_days": config.model.data_gates.exploratory_min_days,
        "ml_training_min_days": config.model.data_gates.ml_training_min_days,
        "meaningful_backtest_min_days": config.model.data_gates.meaningful_backtest_min_days,
        "seasonality_min_days": config.model.data_gates.seasonality_min_days,
        "candles_dir_exists": coverage_dir.exists(),
    }
    return report


def _record_manifest_row(context: AppContext, status: str) -> None:
    """Best-effort insert of the run manifest into DuckDB.

    A failure here must never fail the run: the JSON manifest on disk is the
    authoritative record, the database row is a convenience for SQL queries.
    """
    try:
        from core.storage import connect

        manifest = context.run_context.to_manifest_dict(status=status)
        row = {
            "run_id": manifest["run_id"],
            "app_name": manifest["app_name"],
            "mode": manifest["mode"],
            "status": manifest["status"],
            "started_at": context.run_context.started_at,
            "finished_at": datetime.now(context.calendar.timezone),
            "config_hash": manifest["config_hash"],
            "config_dir": manifest["config_dir"],
            "git_commit": manifest["git_commit"],
            "python_version": manifest["python_version"],
            "platform": manifest["platform"],
            "random_seed": manifest["random_seed"],
            "package_versions": json.dumps(manifest["package_versions"]),
            "data_coverage": json.dumps(manifest["data_coverage"]),
            "artifacts": json.dumps(manifest["artifacts"]),
            "notes": json.dumps(manifest["notes"]),
            "manifest_path": str(
                (context.run_context.artifacts_dir or "") / "manifest.json"
            )
            if context.run_context.artifacts_dir
            else None,
        }
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        connection = connect(context.db_path)
        try:
            connection.execute(
                f"INSERT OR REPLACE INTO run_manifest ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                [row[column] for column in columns],
            )
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001 - never fail a run over metadata
        context.logger.warning(f"Could not record the run manifest in DuckDB: {exc}")


def finalize_run(
    context: AppContext,
    *,
    status: str = "SUCCESS",
    exit_code: int = 0,
) -> int:
    """Write the run manifest, record it in DuckDB and log the outcome."""
    if context.run_context.artifacts_dir is not None:
        try:
            path = context.run_context.write_manifest(status=status)
            context.logger.info(f"run manifest written: {path}")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            context.logger.error(f"Could not write the run manifest: {exc}")
    if context.db_path.exists() and not context.dry_run:
        _record_manifest_row(context, status)
    context.logger.info(f"ANJANEYA {context.app_name} finished: status={status} exit={exit_code}")
    return exit_code


def render_status(report: dict[str, Any], *, as_json: bool = False) -> str:
    """Render a status report either as JSON or as an operator-friendly block.

    The human format intentionally prints one ``key: value`` pair per line so it
    is greppable and copy-pasteable from a terminal.
    """
    if as_json:
        return json.dumps(report, indent=2, default=str)

    lines: list[str] = []
    config = report.get("config", {})
    calendar = report.get("calendar", {})
    database = report.get("database", {})
    credentials = report.get("credentials", {})

    lines.append(f"ANJANEYA {report.get('app')} - status (mode={report.get('mode')})")
    lines.append(f"  run_id            : {report.get('run_id')}")
    lines.append(f"  dry_run           : {report.get('dry_run')}")
    lines.append("  -- configuration --")
    for key in (
        "project",
        "version",
        "schema_version",
        "timezone",
        "config_hash",
        "data_root",
        "db_path",
        "timeframes",
        "horizons_minutes",
        "algorithm",
        "holidays_total",
        "holidays_verified",
        "unverified_holidays",
    ):
        if key in config:
            lines.append(f"  {key:<18}: {config[key]}")
    lines.append("  -- calendar --")
    for key in (
        "today",
        "today_is_trading_day",
        "expiry_weekday",
        "holiday_roll",
        "current_weekly_expiry",
        "next_weekly_expiry",
        "current_monthly_expiry",
        "days_to_expiry_definition",
        "buckets",
    ):
        if key in calendar:
            lines.append(f"  {key:<18}: {calendar[key]}")
    lines.append("  -- credentials present --")
    for name, present in credentials.items():
        lines.append(f"  {name:<18}: {'yes' if present else 'no'}")
    lines.append("  -- database --")
    lines.append(f"  path              : {database.get('path')}")
    lines.append(f"  exists            : {database.get('exists')}")
    if "tables" in database:
        lines.append(f"  tables            : {', '.join(database['tables'])}")
    if "row_counts" in database:
        lines.append(f"  row_counts        : {database['row_counts']}")
    if "error" in database:
        lines.append(f"  error             : {database['error']}")
    gates = report.get("data_gates", {})
    if gates:
        lines.append("  -- data gates (spec 4.3) --")
        for key, value in gates.items():
            lines.append(f"  {key:<18}: {value}")
    heartbeat = report.get("heartbeat")
    if heartbeat:
        lines.append("  -- capture heartbeat --")
        for key, value in heartbeat.items():
            lines.append(f"  {key:<18}: {value}")
    if report.get("capture_policy"):
        lines.append("  -- capture policy --")
        for key, value in report["capture_policy"].items():
            lines.append(f"  {key:<18}: {value}")
    return "\n".join(lines)


#: Exit codes used consistently by all three apps.
EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_FAILED = 1
#: A run refused the work it was asked to do *on purpose* - a data-accumulation gate that is
#: not open yet (spec 4.3), not a crash. Distinct from ``EXIT_FAILED`` so a pipeline can tell
#: "not enough data" from "something broke", and distinct from ``EXIT_OK`` so it can never be
#: mistaken for a completed run.
EXIT_GATE_REFUSED = 4
