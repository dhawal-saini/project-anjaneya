"""ANJANEYA daily post-close pipeline - one command from captured ticks to a backtest.

After the 15:30 close the day's work is always the same chain, and every step needs the
one before it:

    build-candles   ticks            -> 1m/3m/5m/15m bars + the data_coverage ledger
    enrich-greeks   bars             -> IV and greeks on the stored bars
    build-features  bars + greeks    -> the section-5 feature matrix
    build-labels    bars + features  -> the forward-return labels
    validate        features         -> the no-look-ahead validation pass
    dataset         features+labels  -> the training matrix and its time-ordered splits
    train           dataset          -> a LightGBM artefact under data/models/
    backtest        model + bars     -> the spec-23 replay ledger and summary

Run them by hand and it is eight commands, eight log files and eight chances to forget
one. This script runs them in order, fail-fast, into a single dated log file, and reports
one table at the end.

Usage::

    uv run python scripts/run_daily_pipeline.py                  # newest day with ticks
    uv run python scripts/run_daily_pipeline.py --date 2026-10-01
    uv run python scripts/run_daily_pipeline.py --force           # train below the gate
    uv run python scripts/run_daily_pipeline.py --dry-run --json  # plan only, writes nothing
    uv run python scripts/run_daily_pipeline.py --skip validate,backtest

Flags of note:

* ``--date``            which day the *engine* builds (the AI steps always read all history,
                        because a model trained on one day is not a model);
* ``--timeframe``       the single timeframe features/labels/validate/the dataset use
                        (candles and greeks still build every configured timeframe);
* ``--horizon``         repeatable; forwarded to dataset and train. ``--mode backtest``
                        replays one model and therefore one horizon, so a multi-horizon run
                        needs ``--skip backtest`` (the pipeline says so rather than letting
                        the child fail cryptically);
* ``--weekly``          append the cadence steps after the daily chain (report, explain,
                        importance, charts, accumulation) - off by default, so a plain run
                        is byte-for-byte the chain it always was;
* ``--skip``            comma-separated step keys (candles, greeks, features, labels,
                        validate, dataset, train, backtest, report, explain, importance,
                        charts, accumulation);
* ``--force``           forward ``--force`` to ``--mode train`` (below spec 4.3's gate);
* ``--dry-run``         every step gets ``--dry-run`` (+ ``--no-run-artifacts``): the whole
                        chain is planned and validated, nothing is written;
* ``--continue-on-error`` keep going after a failing step (default: stop at the first).

Exit codes, chosen so a scheduler can tell the cases apart:

    0  every step that ran succeeded
    1  a step failed (the failing step is named on stderr)
    2  a configuration problem (bad ``--skip`` name, no ticks captured yet, locked database)
    4  nothing failed, but ``--mode train`` refused to train because the exploratory gate is
       not open yet - pass ``--force`` to train anyway, or wait for the data to accumulate.

The pipeline never touches the network: it reads and writes only the local DuckDB file and
the configured ``data/`` tree. Capturing the session (and the index backfill) stays where it
belongs - in ``apps.anjaneeya_capture.run``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_daily_pipeline.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.app_bootstrap import (  # noqa: E402
    APP_AI,
    APP_ENGINE,
    EXIT_CONFIG_ERROR,
    EXIT_FAILED,
    EXIT_GATE_REFUSED,
    EXIT_OK,
)
from core.config_loader import AnjaneyaConfig, ConfigError, load_config  # noqa: E402
from core.env import load_env_file  # noqa: E402
from core.market_calendar import NseCalendar  # noqa: E402
from core.storage import (  # noqa: E402
    bootstrap_database,
    connect,
    read_index_history,
    trading_days_with_ticks,
)

#: Which module implements each app.
APP_MODULES: dict[str, str] = {
    APP_ENGINE: "apps.anjaneeya_engine.run",
    APP_AI: "apps.anjaneeya_ai.run",
}

#: How far ahead an unverified holiday is worth shouting about.
RISK_WINDOW_DAYS = 7

#: Status vocabulary for one step (kept short so the table stays readable).
STATUS_OK = "OK"
STATUS_FAILED = "FAILED"
STATUS_GATE = "GATE"
STATUS_SKIPPED = "SKIPPED"
STATUS_DRY_RUN = "DRY-RUN"


class PipelineError(RuntimeError):
    """A problem the operator must fix before the chain can run (exit code 2)."""


@dataclass(frozen=True, slots=True)
class Step:
    """One step of the post-close chain."""

    key: str
    app: str
    mode: str
    summary: str
    #: The timeframe the *engine* flavours of the step need exactly one of.
    single_timeframe: bool = False
    #: Steps this one reads, for the operator's benefit (not enforced - ``--skip`` is the
    #: operator's call, and each app reports its own missing input).
    depends_on: tuple[str, ...] = ()
    #: True when the step reads the whole accumulation rather than one day.
    whole_history: bool = False
    #: True when the step cannot do anything without ticks for the day. The four engine
    #: steps that build from ticks need them; ``validate`` reads features, and the AI steps
    #: read all history - so a features-only rerun does not need a capture session.
    needs_ticks: bool = True

    @property
    def module(self) -> str:
        """The importable entry point for this step."""
        return APP_MODULES[self.app]


#: The chain, in the only order that can work.
STEPS: tuple[Step, ...] = (
    Step("candles", APP_ENGINE, "build-candles", "ticks -> candles + the coverage ledger"),
    Step(
        "greeks", APP_ENGINE, "enrich-greeks", "IV and greeks on the stored bars", depends_on=("candles",)
    ),
    Step(
        "features",
        APP_ENGINE,
        "build-features",
        "the section-5 feature matrix",
        single_timeframe=True,
        depends_on=("candles", "greeks"),
    ),
    Step(
        "labels",
        APP_ENGINE,
        "build-labels",
        "the forward-return labels",
        single_timeframe=True,
        depends_on=("candles", "features"),
    ),
    Step(
        "validate",
        APP_ENGINE,
        "validate",
        "the no-look-ahead validation pass",
        single_timeframe=True,
        depends_on=("features",),
        needs_ticks=False,
    ),
    Step(
        "dataset",
        APP_AI,
        "dataset",
        "the training matrix and its splits",
        depends_on=("features", "labels"),
        whole_history=True,
        needs_ticks=False,
    ),
    Step(
        "train",
        APP_AI,
        "train",
        "a LightGBM artefact under data/models/",
        depends_on=("dataset",),
        whole_history=True,
        needs_ticks=False,
    ),
    Step(
        "backtest",
        APP_AI,
        "backtest",
        "the spec-23 replay ledger and summary",
        depends_on=("train",),
        whole_history=True,
        needs_ticks=False,
    ),
)

#: Extra steps an operator runs on a cadence rather than after every close. They read the
#: artefacts the daily chain left behind (a model, a ledger, an importance dump), so they are
#: opt-in via ``--weekly`` and never change what a default ``run_daily_pipeline`` does.
WEEKLY_STEPS: tuple[Step, ...] = (
    Step(
        "report",
        APP_AI,
        "report",
        "the spec-24 research report from the newest ledger",
        depends_on=("backtest",),
        whole_history=True,
        needs_ticks=False,
    ),
    Step(
        "explain",
        APP_AI,
        "explain",
        "plain structured explanations of the newest predictions",
        depends_on=("train",),
        whole_history=True,
        needs_ticks=False,
    ),
    Step(
        "importance",
        APP_AI,
        "importance",
        "where the model's information came from",
        depends_on=("train",),
        whole_history=True,
        needs_ticks=False,
    ),
    Step(
        "charts",
        APP_AI,
        "charts",
        "the figures that make the numbers checkable by eye",
        depends_on=("train",),
        whole_history=True,
        needs_ticks=False,
    ),
    Step(
        "accumulation",
        APP_AI,
        "accumulation",
        "the data-accumulation state and the retraining readiness",
        whole_history=True,
        needs_ticks=False,
    ),
)

STEP_KEYS: tuple[str, ...] = tuple(step.key for step in STEPS)
WEEKLY_STEP_KEYS: tuple[str, ...] = tuple(step.key for step in WEEKLY_STEPS)
STEPS_BY_KEY: dict[str, Step] = {step.key: step for step in (*STEPS, *WEEKLY_STEPS)}


@dataclass(slots=True)
class StepResult:
    """What one step did."""

    key: str
    status: str
    exit_code: int
    seconds: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON form (one row of the report)."""
        return {
            "step": self.key,
            "status": self.status,
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 2),
            "note": self.note,
        }


@dataclass(slots=True)
class DayInspection:
    """The read-only pre-flight verdict for the day this run covers."""

    day: date
    ticks: int = 0
    tick_symbols: int = 0
    index_rows: int = 0
    is_trading_day: bool = True
    holiday_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "day": self.day.isoformat(),
            "ticks": self.ticks,
            "tick_symbols": self.tick_symbols,
            "index_rows": self.index_rows,
            "is_trading_day": self.is_trading_day,
            "holiday_name": self.holiday_name,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# The log tee
# ---------------------------------------------------------------------------
class Tee:
    """Write every child line to the console *and* to one dated log file."""

    def __init__(self, path: Path | None, *, stream: Any = None) -> None:
        self.path = path
        self._stream = stream if stream is not None else sys.stdout
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def line(self, text: str = "", *, echo: bool = True) -> None:
        """One line out (console) and one line down (file).

        ``echo=False`` keeps a line out of the console: with ``--json`` the console is a
        progress log on stderr and stdout must stay a single JSON document.
        """
        if echo:
            print(text, file=self._stream, flush=True)
        if self._handle is not None:
            self._handle.write(text + "\n")
            self._handle.flush()

    def close(self) -> None:
        """Release the log file, if one was opened."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Tee:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Running one step
# ---------------------------------------------------------------------------
def _run_subprocess(argv: Sequence[str], tee: Tee) -> int:
    """Run one step as a child process, teeing its output; returns its exit code.

    ``sys.executable`` is used rather than ``uv``/``python``: the wrapper scripts have
    already resolved the interpreter (venv, uv, or system python), so the child runs in
    exactly the environment the parent is running in - which is the whole point of a
    one-command pipeline.
    """
    tee.line(f"$ {' '.join(argv)}")
    try:
        process = subprocess.Popen(  # noqa: S603 - argv is built here, never from user text
            list(argv),
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:  # pragma: no cover - only if the interpreter vanished
        tee.line(f"could not start the step: {type(exc).__name__}: {exc}")
        return EXIT_FAILED
    assert process.stdout is not None  # PIPE guarantees it
    for line in process.stdout:
        tee.line(line.rstrip("\n"))
    return process.wait()


# ---------------------------------------------------------------------------
# Building the argument list for one step
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RunOptions:
    """Everything the steps need to know about this run."""

    day: date
    timeframe: str | None = None
    horizons: tuple[int, ...] = ()
    dry_run: bool = False
    force: bool = False
    config_dir: Path | None = None
    data_root: Path | None = None
    db_path: Path | None = None


def global_argv(options: RunOptions) -> list[str]:
    """The flags every step accepts (resolved, so parent and child agree exactly)."""
    argv: list[str] = []
    if options.config_dir is not None:
        argv += ["--config-dir", str(options.config_dir)]
    if options.data_root is not None:
        argv += ["--data-root", str(options.data_root)]
    if options.db_path is not None:
        argv += ["--db-path", str(options.db_path)]
    if options.dry_run:
        argv += ["--dry-run", "--no-run-artifacts"]
    return argv


def step_argv(step: Step, options: RunOptions) -> list[str]:
    """The full command line for ``step`` (``[python, -m, module, ...]``)."""
    argv = [sys.executable, "-m", step.module, "--mode", step.mode]
    if step.app == APP_ENGINE:
        # The engine builds one specific day; `--date` is what an operator means after a
        # capture session. The AI steps below deliberately read all history instead.
        argv += ["--date", options.day.isoformat()]
        if step.single_timeframe and options.timeframe:
            argv += ["--timeframes", options.timeframe]
    else:
        if step.key == "dataset" and options.timeframe:
            argv += ["--timeframe", options.timeframe]
        if step.key in {"dataset", "train"}:
            for horizon in options.horizons:
                argv += ["--horizon", str(horizon)]
        if step.key == "backtest" and options.horizons:
            # One model, one horizon - argument validation has already refused a longer list.
            argv += ["--horizon", str(options.horizons[0])]
        if step.key in {"report", "explain", "importance", "charts"} and len(options.horizons) == 1:
            # These read the newest artefact for a horizon; forward a single one when given.
            argv += ["--horizon", str(options.horizons[0])]
        if step.key == "train" and options.force:
            argv += ["--force"]
    return argv + global_argv(options)


# ---------------------------------------------------------------------------
# Pre-flight: which day, and is there anything to build?
# ---------------------------------------------------------------------------
def resolve_day(config: AnjaneyaConfig, *, requested: date | None) -> date:
    """The day this run covers: the requested one, or the newest day with stored ticks."""
    if requested is not None:
        return requested
    bootstrap_database(config.db_path, schema_version=config.settings.project.schema_version)
    connection = connect(config.db_path)
    try:
        days = trading_days_with_ticks(connection, timezone_name=config.timezone_name)
    finally:
        connection.close()
    if not days:
        raise PipelineError(
            "the ticks table is empty - nothing has been captured yet. Run a capture session "
            "first (scripts/start_capture.sh, or start_capture.ps1 on Windows), or pass --date "
            "for a day that is already stored."
        )
    return days[-1]


def inspect_day(config: AnjaneyaConfig, calendar: NseCalendar, *, day: date) -> DayInspection:
    """Count what the day holds and warn about what is missing - reading only."""
    inspection = DayInspection(day=day)
    bootstrap_database(config.db_path, schema_version=config.settings.project.schema_version)
    connection = connect(config.db_path)
    try:
        row = connection.execute(
            "SELECT count(*), count(DISTINCT tradingsymbol) FROM ticks "
            "WHERE CAST(timezone(?, timestamp) AS DATE) = ?",
            [config.timezone_name, day],
        ).fetchone()
        inspection.ticks = int(row[0] or 0)
        inspection.tick_symbols = int(row[1] or 0)
        index_frame = read_index_history(
            connection, start=calendar.session_start(day), end=calendar.session_end(day)
        )
        inspection.index_rows = int(len(index_frame))
    finally:
        connection.close()

    inspection.is_trading_day = calendar.is_trading_day(day)
    inspection.holiday_name = calendar.holiday_name(day)

    if not inspection.is_trading_day:
        detail = f"the holiday {inspection.holiday_name}" if inspection.holiday_name else "a weekend"
        inspection.warnings.append(
            f"{day.isoformat()} is {detail} in config/calendar.yaml but {inspection.ticks} tick(s) "
            "are stored for it: either the calendar is wrong (reconcile the unverified holidays) "
            "or the capture ran outside the session."
        )
    if inspection.index_rows == 0:
        inspection.warnings.append(
            f"index_history holds no NIFTY rows for {day.isoformat()}: enrich-greeks will fall "
            "back to the index candles and the IV columns may stay NULL. Top it up with "
            "scripts/run_backfill.sh (or run_backfill.ps1)."
        )
    for entry in calendar.unverified_holiday_risk(today=day, days=RISK_WINDOW_DAYS):
        inspection.warnings.append(
            f"holiday {entry.date.isoformat()} ({entry.name}) is still PROVISIONAL; if it is "
            "really a trading day, the capture session will exit NO_SESSION and the day is lost. "
            "Reconcile it with the NSE circular (scripts/check_calendar.py)."
        )
    return inspection


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
def run_chain(
    steps: Sequence[Step],
    options: RunOptions,
    tee: Tee,
    *,
    continue_on_error: bool = False,
    blocked: dict[str, str] | None = None,
) -> list[StepResult]:
    """Run ``steps`` in order, stopping at the first failure unless told otherwise."""
    results: list[StepResult] = []
    blocked = dict(blocked or {})
    total = len(steps)
    for position, step in enumerate(steps, start=1):
        if step.key in blocked:
            reason = blocked[step.key]
            results.append(StepResult(step.key, STATUS_SKIPPED, EXIT_OK, 0.0, reason))
            tee.line(f"[{position}/{total}] {step.key}: SKIPPED ({reason})")
            continue
        tee.line(f"[{position}/{total}] {step.key}: {step.summary}")
        started = time.monotonic()
        code = _run_subprocess(step_argv(step, options), tee)
        seconds = time.monotonic() - started
        if code == EXIT_OK:
            status, note = (STATUS_DRY_RUN if options.dry_run else STATUS_OK), ""
        elif step.key == "train" and code == EXIT_GATE_REFUSED:
            status = STATUS_GATE
            note = "train refused: spec 4.3's exploratory gate is not open (use --force)"
            blocked["backtest"] = "no fresh model (train refused by the gate)"
        else:
            status, note = STATUS_FAILED, f"exit {code}"
        results.append(StepResult(step.key, status, code, seconds, note))
        tee.line(f"    -> {status} ({seconds:.1f}s)")
        if status == STATUS_FAILED and not continue_on_error:
            tee.line(f"stopping at {step.key} (exit {code}); --continue-on-error presses on")
            break
    return results


def overall_exit_code(results: Sequence[StepResult]) -> int:
    """``0`` clean, ``1`` a step failed, ``4`` only the training gate stood in the way."""
    if any(result.status == STATUS_FAILED for result in results):
        return EXIT_FAILED
    if any(result.status == STATUS_GATE for result in results):
        return EXIT_GATE_REFUSED
    return EXIT_OK



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_report(
    *,
    options: RunOptions,
    inspection: DayInspection,
    steps: Sequence[Step],
    results: Sequence[StepResult],
    log_path: Path | None,
    exit_code: int,
) -> dict[str, Any]:
    """The machine-readable form of the run (also what ``--json`` prints)."""
    return {
        "day": options.day.isoformat(),
        "timeframe": options.timeframe,
        "horizons": list(options.horizons),
        "dry_run": options.dry_run,
        "force": options.force,
        "db_path": str(options.db_path) if options.db_path else None,
        "preflight": inspection.to_dict(),
        "planned_steps": [step.key for step in steps],
        "results": [result.to_dict() for result in results],
        "log_file": str(log_path) if log_path else None,
        "exit_code": exit_code,
        "verdict": verdict_text(results, exit_code),
    }


def verdict_text(results: Sequence[StepResult], exit_code: int) -> str:
    """One line an operator (or a scheduler's e-mail) can act on."""
    if exit_code == EXIT_CONFIG_ERROR:
        return "CONFIG_ERROR - nothing was built"
    if exit_code == EXIT_GATE_REFUSED:
        return "BELOW_GATE - the chain ran but spec 4.3's exploratory gate refused the model"
    failed = [result.key for result in results if result.status == STATUS_FAILED]
    if exit_code == EXIT_FAILED or failed:
        return f"FAILED at {', '.join(failed) or 'a step'}"
    if any(result.status == STATUS_SKIPPED for result in results):
        return "OK (with skipped steps)"
    if results and all(result.status == STATUS_DRY_RUN for result in results):
        return "DRY-RUN - nothing was written"
    return "OK"


def render_report(report: dict[str, Any]) -> str:
    """The human-readable form of :func:`build_report`."""
    inspection = report["preflight"]
    horizons = report["horizons"]
    lines = [
        f"ANJANEYA daily pipeline - {report['day']}",
        f"  timeframe      : {report['timeframe'] or 'configured default'}",
        f"  horizons       : {', '.join(f'{item}m' for item in horizons) or 'configured default'}",
        f"  dry run        : {report['dry_run']}",
        f"  database       : {report['db_path']}",
        f"  pre-flight     : {inspection['tick_symbols']} contract(s), {inspection['ticks']} tick(s); "
        f"index_history {inspection['index_rows']} row(s); "
        f"{'trading day' if inspection['is_trading_day'] else 'NOT a trading day'}",
    ]
    for warning in inspection["warnings"]:
        lines.append(f"  warning        : {warning}")
    lines.append("")
    lines.append("  step       status    exit   seconds  note")
    for row in report["results"]:
        lines.append(
            f"  {row['step']:<10} {row['status']:<9} {row['exit_code']:>4}  "
            f"{row['seconds']:>8.1f}  {row['note']}"
        )
    lines.append("")
    if report["log_file"]:
        lines.append(f"  log            : {report['log_file']}")
    lines.append(f"  verdict        : {report['verdict']}")
    return "\n".join(lines)



def _iso_date(value: str) -> date:
    """``argparse`` type for ``YYYY-MM-DD``."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_daily_pipeline.py",
        description="Run the whole post-close chain (candles -> ... -> backtest) in order.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML configuration.")
    parser.add_argument("--data-root", default=None, help="Override settings.data.root for this run.")
    parser.add_argument("--db-path", default=None, help="Override the DuckDB file for this run.")
    parser.add_argument(
        "--date",
        type=_iso_date,
        default=None,
        metavar="YYYY-MM-DD",
        help="Trading day the engine steps build. Default: the newest day with stored ticks.",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        metavar="1m",
        help="Timeframe for build-features/build-labels/validate and the dataset. Default: the "
        "engine's configured first timeframe and model.dataset.timeframe respectively.",
    )
    parser.add_argument(
        "--horizon",
        dest="horizons",
        action="append",
        type=int,
        default=None,
        metavar="MINUTES",
        help="Label horizon (repeatable) for dataset and train. Backtest replays one model, so a "
        "multi-horizon run needs --skip backtest.",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Also run the cadence steps (report -> explain -> importance -> charts -> "
        "accumulation) after the daily chain. Off by default, so a plain run is unchanged.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        metavar="STEP,STEP",
        help=f"Comma-separated steps to leave out. Known steps: "
        f"{', '.join((*STEP_KEYS, *WEEKLY_STEP_KEYS))}.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Forward --force to train (below spec 4.3's gate)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Give every step --dry-run and --no-run-artifacts: planned, never written.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", help="Keep going after a failing step."
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the report as JSON.")
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Where the run's log goes. Default: <data root>/logs/daily_pipeline_<date>.log "
        "(no log file at all in --dry-run).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)



# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------
def parse_skip(value: str | None) -> list[str]:
    """``"a, b"`` -> ``["a", "b"]`` (order kept, empties dropped)."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_steps(skipped: Sequence[str] | None = None, *, weekly: bool = False) -> list[Step]:
    """The steps to run, in chain order. An unknown name is a configuration error.

    ``weekly`` appends :data:`WEEKLY_STEPS` after the daily chain; it is the only way
    those steps ever run, so the default behaviour of the script is unchanged.
    """
    skipped = list(skipped or [])
    known = (*STEP_KEYS, *WEEKLY_STEP_KEYS)
    unknown = [name for name in skipped if name not in STEPS_BY_KEY]
    if unknown:
        raise PipelineError(f"unknown --skip step(s) {unknown}; known steps: {', '.join(known)}")
    chain = (*STEPS, *WEEKLY_STEPS) if weekly else STEPS
    return [step for step in chain if step.key not in set(skipped)]


def resolve_options(args: argparse.Namespace, config: AnjaneyaConfig, *, day: date) -> RunOptions:
    """Validate the CLI against the resolved config and build the :class:`RunOptions`."""
    steps = select_steps(parse_skip(args.skip), weekly=bool(getattr(args, "weekly", False)))
    timeframes = tuple(config.settings.market.timeframes)
    if args.timeframe and args.timeframe not in timeframes:
        raise PipelineError(
            f"--timeframe {args.timeframe!r} is not one of the configured timeframes "
            f"({', '.join(timeframes)}); change settings.market.timeframes or drop the flag"
        )
    horizons = tuple(dict.fromkeys(int(value) for value in (args.horizons or ())))
    allowed = tuple(int(value) for value in config.model.horizons_minutes)
    unknown = [horizon for horizon in horizons if horizon not in allowed]
    if unknown:
        raise PipelineError(
            f"--horizon {unknown} is not in model.horizons_minutes ({list(allowed)}); add it to "
            "config/model.yaml and rebuild the labels first"
        )
    if len(horizons) > 1 and any(step.key == "backtest" for step in steps):
        raise PipelineError(
            f"a backtest replays one model and a model is one horizon: got --horizon "
            f"{list(horizons)}. Use a single --horizon, or --skip backtest and run one backtest "
            "per horizon."
        )
    return RunOptions(
        day=day,
        timeframe=args.timeframe,
        horizons=horizons,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        config_dir=config.config_dir,
        data_root=config.data_root,
        db_path=config.db_path,
    )




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 ok, 1 a step failed, 2 configuration, 4 below the training gate."""
    args = parse_args(argv)
    if args.data_root:
        os.environ["ANJANEYA_DATA_ROOT"] = args.data_root
    if args.db_path:
        os.environ["ANJANEYA_DB_PATH"] = args.db_path
    config_dir = Path(args.config_dir) if args.config_dir else None
    load_env_file(config_dir=config_dir)
    try:
        config = load_config(config_dir)
    except ConfigError as exc:
        print(f"ANJANEYA pipeline: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        steps = select_steps(parse_skip(args.skip), weekly=bool(getattr(args, "weekly", False)))
        day = resolve_day(config, requested=args.date)
        options = resolve_options(args, config, day=day)
        inspection = inspect_day(config, NseCalendar(config.calendar), day=day)
        if inspection.ticks == 0 and any(step.needs_ticks for step in steps):
            needs = ", ".join(step.key for step in steps if step.needs_ticks)
            raise PipelineError(
                f"no ticks are stored for {day.isoformat()}: {needs} cannot build anything. "
                "Capture the session first, pass --date for a day that is already stored, or "
                "--skip the steps that need ticks."
            )
    except PipelineError as exc:
        print(f"ANJANEYA pipeline: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    log_path: Path | None = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif not args.dry_run:
        log_path = config.log_dir / f"daily_pipeline_{day.isoformat()}.log"

    with Tee(log_path, stream=sys.stderr if args.as_json else sys.stdout) as tee:
        tee.line(
            f"ANJANEYA daily pipeline: day={day.isoformat()} steps={len(steps)} "
            f"timeframe={options.timeframe or 'default'} dry_run={options.dry_run}"
        )
        tee.line(f"database   : {config.db_path}")
        tee.line(
            f"pre-flight : {inspection.tick_symbols} contract(s), {inspection.ticks} tick(s); "
            f"index_history {inspection.index_rows} row(s)"
        )
        for warning in inspection.warnings:
            tee.line(f"warning    : {warning}")
        tee.line("plan       : " + " -> ".join(step.key for step in steps))
        tee.line("")
        try:
            results = run_chain(steps, options, tee, continue_on_error=bool(args.continue_on_error))
        except KeyboardInterrupt:  # pragma: no cover - operator path
            tee.line("")
            tee.line("interrupted - re-run the same command; each step is idempotent")
            return EXIT_FAILED

        exit_code = overall_exit_code(results)
        report = build_report(
            options=options,
            inspection=inspection,
            steps=steps,
            results=results,
            log_path=log_path,
            exit_code=exit_code,
        )
        tee.line("")
        if args.as_json:
            body = json.dumps(report, indent=2)
            tee.line(body, echo=False)
            print(body)
        else:
            tee.line(render_report(report))

    if exit_code != EXIT_OK:
        failed = [result.key for result in results if result.status == STATUS_FAILED]
        print(f"ANJANEYA pipeline: {report['verdict']} (exit {exit_code})", file=sys.stderr)
        if failed:
            print(f"  first failure: {failed[0]}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
