"""ANJANEYA operator console - the three verbs, a few clicks away.

Why this exists
---------------
``scripts/anj.py`` already gives the operator three verbs, and the CLI stays the
source of truth. What a terminal cannot give is *state at a glance*: is today's
token still alive, how many bars are actually in the database, what did the last
backfill do. This console adds that view, and nothing else.

What it is not
--------------
* It is not a second implementation. Every button shells out to the script that
already owns the behaviour - ``scripts/anj.py``, ``scripts/daily_login.py``,
  ``scripts/run_daily_pipeline.py`` - exactly like the PowerShell wrappers do. If
a button and a command ever disagree, the command is right and the button is a
bug.
* It is not a trading terminal. ANJANEYA places no orders (``pyproject.toml``:
"research and backtesting only"), so there is no trading switch to offer.

Rules this file holds itself to
-------------------------------
1. **One job at a time.** DuckDB allows a single writer, so two buttons pressed
   twice would mean two writers and a locked database. :func:`start_job`
   refuses to start a second job while one is live.
2. **A job outlives the page.** Streamlit re-runs this script on every
   interaction, so module globals are not state. A job is therefore a detached
   child that writes its own status file; the page only ever reads that file.
3. **A preview never holds a connection.** Connect read-only, query, close. A
   cached connection would *be* the lock that blocks the evening pipeline, and
   while a capture session holds the write lock no other process can open the
   database at all - so the console says so instead of hanging.
4. **No secret is ever rendered or persisted.** The token card shows a masked
   token and an expiry. The ``request_token`` box is a password field, is
   exchanged synchronously in this process, and is never written to the status
   file or the log.

Running it::

    uv sync --extra ui
    uv run streamlit run scripts/anj_ui.py

The ``ui`` extra keeps Streamlit out of the runtime dependency set: the capture
session, the engine and the AI app must not need a web framework.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python scripts/anj_ui.py --run-job ...`
    sys.path.insert(0, str(REPO_ROOT))

from core.auth import normalise_request_token  # noqa: E402
from core.config_loader import load_config  # noqa: E402
from core.env import load_env_file  # noqa: E402

# ---------------------------------------------------------------------------
# The owners: every button delegates to one of these, never to its own logic.
# ---------------------------------------------------------------------------
FACADE = "scripts/anj.py"
LOGIN = "scripts/daily_login.py"
LOGIN_AND_START = "scripts/login_and_start.py"
PIPELINE = "scripts/run_daily_pipeline.py"
CAPTURE_APP = "apps.anjaneeya_capture.run"
ENGINE_APP = "apps.anjaneeya_engine.run"

#: ``run_daily_pipeline.STEPS`` + ``WEEKLY_STEPS``, in the order they run.
PIPELINE_ORDER: tuple[str, ...] = (
    "candles",
    "greeks",
    "features",
    "labels",
    "validate",
    "dataset",
    "train",
    "backtest",
    "report",
    "explain",
    "importance",
    "charts",
    "accumulation",
)

#: Steps with no switch. They *are* the dataset; skipping one leaves the rest
#: with nothing to read, so the console never offers to skip them.
ALWAYS_ON: tuple[str, ...] = ("candles", "greeks", "features", "labels", "validate")

#: Console switch -> the pipeline steps it actually controls. Order matters:
#: it is the order the switches are drawn in.
SWITCHED_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ml", ("dataset", "train")),
    ("backtest", ("backtest",)),
    ("research", ("report", "explain", "importance", "charts", "accumulation")),
)

#: Switches a naive UI would draw and this console cannot honour, with the
#: reason. Rendered in the sidebar so the operator is never left guessing.
UNSWITCHABLE: dict[str, str] = {
    "indicator": (
        "not a step - the indicators are computed inside `features` "
        "(core/features/feature_builder.py). Switching them off means dropping "
        "columns, which changes the dataset schema and invalidates every model "
        "already trained against it."
    ),
    "price_action": (
        "not a step - price action (candles, levels, structure, swings) is also "
        "computed inside `features`, with the same schema consequence as "
        "`indicator`."
    ),
    "trading": (
        "not a feature of this project - ANJANEYA places no orders "
        '(pyproject.toml: "research and backtesting only"). There is nothing to '
        "switch on."
    ),
}

#: Tables the data tab may preview, with the column that orders them. A table
#: missing from here is not previewable: this tuple is the format guard as well
#: as the shape of the tab (a table name is never interpolated from a widget
#: value, only looked up).
PREVIEW_TABLES: tuple[tuple[str, str, str], ...] = (
    ("index_history", "timestamp", "NIFTY index history"),
    ("candles", "timestamp", "Candles (after greeks)"),
    ("features", "timestamp", "Feature matrix"),
    ("labels", "timestamp", "Forward-return labels"),
    ("ticks", "timestamp", "Raw ticks"),
    ("trades", "entry_timestamp", "Backtest ledger"),
    ("instruments", "fetched_at", "Instrument master"),
)

#: How many rows a preview shows. Ten is a glance; this is not an export tool.
DEFAULT_LIMIT = 10


def number(value: float) -> str:
    """Render ``2`` rather than ``2.0``, but keep ``0.25``."""
    return f"{float(value):g}"


def steps_for(switches: dict[str, bool]) -> list[str]:
    """The pipeline steps that will run, in pipeline order.

    A switch that is absent counts as *on*: the default must be the full chain,
    so a missing key can never quietly shrink a run.
    """
    kept = set(ALWAYS_ON)
    for name, steps in SWITCHED_STEPS:
        if switches.get(name, True):
            kept.update(steps)
    return [step for step in PIPELINE_ORDER if step in kept]


def skip_argument(switches: dict[str, bool]) -> str:
    """The ``--skip`` value that realises ``switches`` (empty when all are on)."""
    kept = set(steps_for(switches))
    return ",".join(step for step in PIPELINE_ORDER if step not in kept)


def common_flags(*, config_dir: str | None = None) -> list[str]:
    """The forwarded form of the shared options. Blank widget = default config."""
    return ["--config-dir", config_dir] if config_dir else []


def learn_argv(
    switches: dict[str, bool],
    *,
    force: bool = False,
    common: Sequence[str] = (),
) -> list[str]:
    """The evening command for these switches.

    With every switch on this is exactly ``anj.py learn``. With one off the
    facade cannot express it (``learn`` has no ``--skip``), so the console calls
    the same owner the facade delegates to - ``run_daily_pipeline.py --weekly``
    - rather than inventing a third path.
    """
    skip = skip_argument(switches)
    argv = [PIPELINE, "--weekly", "--skip", skip] if skip else [FACADE, "learn"]
    if force:
        argv.append("--force")
    return [*argv, *common]


def day_argv(*, no_capture: bool = False, common: Sequence[str] = ()) -> list[str]:
    """The daily-loop command. ``anj.py day`` owns the fixed pre-close policy.

    The switches deliberately do not reach this verb: ``day`` always skips
    ``train,backtest`` because a model trained before the close is a model
    trained on an incomplete day. That policy belongs to the facade.
    """
    argv = [FACADE, "day"]
    if no_capture:
        argv.append("--no-capture")
    return [*argv, *common]


def history_argv(
    *,
    years: float = 2.0,
    option_years: float = 0.25,
    skip_instruments: bool = False,
    skip_options: bool = False,
    dry_run: bool = False,
    common: Sequence[str] = (),
) -> list[str]:
    """The one-time backfill command."""
    argv = [FACADE, "history", "--years", number(years), "--option-years", number(option_years)]
    if skip_instruments:
        argv.append("--skip-instruments")
    if skip_options:
        argv.append("--skip-options")
    if dry_run:
        argv.append("--dry-run")
    return [*argv, *common]


def login_argv(*, force: bool = False, common: Sequence[str] = ()) -> list[str]:
    """The TOTP login command. It carries no secret - credentials come from env."""
    argv = [LOGIN, "--json"]
    if force:
        argv.append("--force")
    return [*argv, *common]


def morning_argv(*, common: Sequence[str] = (), timeout: int = 240) -> list[str]:
    """The whole morning in one job: browser login, token exchange, capture.

    ``login_and_start.py`` opens the Kite login, catches the ``request_token`` on
    the loopback redirect URL, exchanges and verifies it, and only then starts
    ``--mode capture``. So a no-TOTP operator never pastes a token, and the capture
    can never start on a dead token. A usable cached token skips the browser.
    """
    return [LOGIN_AND_START, "--timeout", str(int(timeout)), *common]


def index_topup_argv(*, common: Sequence[str] = ()) -> list[str]:
    """The after-close index top-up: NIFTY daily history plus ``topup_days`` of 1m.

    ``--mode backfill`` is the capture app's index-history mode. It is *not* the
    overnight ``history`` chain: no instrument dump is re-downloaded and no option
    contracts are fetched, so it is a handful of requests and finishes in seconds.
    It matters because the session tops up at 09:05 (i.e. up to the previous day),
    while the engine builds the day's NIFTY block from ``index_history`` alone.
    """
    return ["-m", CAPTURE_APP, "--mode", "backfill", *common]


def backfill_options_argv(*, years: float = 0.01, common: Sequence[str] = ()) -> list[str]:
    """Heal missing *candles* from REST (option means ``years`` of 1m bars).

    Ticks cannot be recovered after the fact; 1-minute bars can, so a session that
    started late or died mid-day still yields a usable candle series.
    """
    return ["-m", CAPTURE_APP, "--mode", "backfill-options", "--years", number(years), *common]


def index_study_argv(*, common: Sequence[str] = ()) -> list[str]:
    """The P1 study: does the NIFTY signal separate forward returns?

    Read-only over ``index_history`` and offline (no credentials, no network), so it
    is safe to run any evening. It writes a JSON + Markdown pair under
    ``data/reports/index_study`` and never writes to the database.
    """
    return ["-m", ENGINE_APP, "--mode", "index-study", *common]


def login_now_argv(*, common: Sequence[str] = (), timeout: int = 240) -> list[str]:
    """Browser login only (``--no-start``): the session is started separately."""
    return [LOGIN_AND_START, "--timeout", str(int(timeout)), "--no-start", *common]


# ---------------------------------------------------------------------------
# Files the console reads *while* a job runs (the database is locked then).
# ---------------------------------------------------------------------------
def heartbeat_path(data_root: str | Path) -> Path:
    """The capture heartbeat the Live tab renders (rewritten every 10 s)."""
    return Path(data_root) / "runs" / "heartbeat" / "capture.json"


def stop_request_path(data_root: str | Path) -> Path:
    """The flag file a running session polls; it is consumed when honoured."""
    return Path(data_root) / "runs" / "heartbeat" / "stop.request"


def read_heartbeat(data_root: str | Path) -> dict[str, Any] | None:
    """The last heartbeat payload, or ``None`` when absent / unreadable."""
    try:
        payload = json.loads(heartbeat_path(data_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def request_stop(data_root: str | Path) -> Path:
    """Ask a running session to stop, without touching the database."""
    path = stop_request_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    return path


def heartbeat_age(payload: dict[str, Any] | None, *, now: datetime | None = None) -> float | None:
    """Seconds since a heartbeat payload was written (``None`` when unreadable)."""
    if not payload:
        return None
    stamp = payload.get("written_at")
    if not isinstance(stamp, str):
        return None
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    reference = now or datetime.now(moment.tzinfo)
    return max(0.0, (reference - moment).total_seconds())


def stop_requested(data_root: str | Path) -> bool:
    """True while a stop request is still waiting to be honoured."""
    return stop_request_path(data_root).exists()


_STEP_LINE = re.compile(r"^\[(\d+)/(\d+)\]\s+([a-z_]+):\s*(.*)$")
_RESULT_LINE = re.compile(r"^->\s+([A-Z_]+)(?:\s+\(([\d.]+)s\))?")


def parse_pipeline_log(text: str) -> list[dict[str, Any]]:
    """Turn a ``run_daily_pipeline`` log into ``[{step, summary, status, seconds}]``.

    The pipeline prints ``[3/8] candles: ticks -> candles + the coverage ledger`` when
    a step starts and ``-> OK (12.3s)`` when it finishes, so the console can show the
    evening chain live without the pipeline having to publish anything new.
    """
    steps: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        started = _STEP_LINE.match(line)
        if started:
            steps.append(
                {
                    "position": int(started.group(1)),
                    "total": int(started.group(2)),
                    "step": started.group(3),
                    "summary": started.group(4).strip(),
                    "status": "running",
                    "seconds": None,
                }
            )
            continue
        finished = _RESULT_LINE.match(line)
        if finished and steps:
            steps[-1]["status"] = finished.group(1).lower()
            seconds = finished.group(2)
            steps[-1]["seconds"] = float(seconds) if seconds else None
    return steps


def newest_file(root: str | Path, pattern: str) -> Path | None:
    """The newest path matching ``pattern`` under ``root``, or ``None``."""
    matches = sorted(Path(root).glob(pattern), key=_mtime, reverse=True)
    return matches[0] if matches else None


def read_accumulation(data_root: str | Path) -> dict[str, Any] | None:
    """The newest ``--mode accumulation`` JSON, if one has been written."""
    path = newest_file(Path(data_root) / "reports" / "accumulation", "*_accumulation.json")
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def newest_model(data_root: str | Path) -> Path | None:
    """The newest model artefact, if a model has been trained yet."""
    return newest_file(Path(data_root) / "models", "*.txt")


def mask(value: str | None) -> str:
    """Everything but the last four characters, for anything secret-shaped."""
    if not value:
        return "-"
    text = str(value)
    return "*" * max(0, len(text) - 4) + text[-4:]


def preview_query(table: str, *, last: bool = False, limit: int = DEFAULT_LIMIT) -> str:
    """SQL for the first or last rows of a whitelisted table."""
    known = {name: column for name, column, _ in PREVIEW_TABLES}
    if table not in known:
        raise ValueError(f"{table!r} is not a previewable table")
    direction = "DESC" if last else "ASC"
    return f"SELECT * FROM {table} ORDER BY {known[table]} {direction} LIMIT {int(limit)}"


# ---------------------------------------------------------------------------
# Jobs: a detached child plus a status file, because Streamlit keeps no state.
# ---------------------------------------------------------------------------
class JobBusy(RuntimeError):
    """A job is already running. DuckDB has one writer; the console has one job."""


@dataclass(frozen=True)
class JobStatus:
    """What the page shows about the child - read from disk, never from memory."""

    name: str
    state: str = "starting"
    argv: tuple[str, ...] = ()
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    log_path: str | None = None
    pid: int | None = None

    @property
    def active(self) -> bool:
        """True while the recorded child process is actually alive.

        A status file can outlive its process: a crash, a killed console, a reboot
        mid-job. Trusting ``state`` alone then leaves a permanent lock that refuses
        every button - which is exactly what happened on 2026-10-27, when an
        orphaned 'token-force' card blocked the whole morning. Liveness is part of
        the answer, and a status file with no pid is treated as not active.
        """
        return self.state in {"starting", "running"} and pid_alive(self.pid)


def pid_alive(pid: int | None) -> bool:
    """True when ``pid`` is a live process on this machine.

    ``os.kill(pid, 0)`` is deliberately **not** used on Windows: there a zero
    signal is not a probe, it terminates the process. ``OpenProcess`` +
    ``GetExitCodeProcess`` is the read-only equivalent.
    """
    if not pid or int(pid) <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
        except OSError:
            return False
        return True
    import ctypes  # noqa: PLC0415 - Windows-only, imported where it is used

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def now() -> str:
    """A local timestamp a human can read in the log and the card."""
    return datetime.now().isoformat(timespec="seconds")


def ui_dir(data_root: str | Path) -> Path:
    """Where the console keeps its own bookkeeping (never the pipeline's)."""
    return Path(data_root) / "logs" / "ui"


def status_file(data_root: str | Path) -> Path:
    """The single file the page and the child agree on."""
    return ui_dir(data_root) / "job.json"


def write_status(path: Path, status: JobStatus) -> None:
    """Write the status atomically, so a reader never sees half a document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = asdict(status)
    payload["argv"] = list(status.argv)
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_status(path: Path) -> JobStatus | None:
    """The last written status, or ``None`` when no job has ever run."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return JobStatus(
        name=str(payload.get("name", "job")),
        state=str(payload.get("state", "idle")),
        argv=tuple(payload.get("argv") or ()),
        started_at=payload.get("started_at"),
        finished_at=payload.get("finished_at"),
        exit_code=payload.get("exit_code"),
        log_path=payload.get("log_path"),
        pid=payload.get("pid"),
    )


def slug(text: str) -> str:
    """A log-file-safe name for a job."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "job"


def start_job(
    name: str,
    argv: Sequence[str],
    *,
    data_root: str | Path,
    cwd: str | Path = REPO_ROOT,
    spawn=subprocess.Popen,
) -> JobStatus:
    """Start ``argv`` as a detached child and return its initial status.

    Raises :class:`JobBusy` when a job is already live: one writer, one job. The
    child writes the status file itself, so a page reload - Streamlit re-runs
    this script on every click - cannot lose a job that is still working.
    """
    path = status_file(data_root)
    current = read_status(path)
    if current is not None and current.active:
        raise JobBusy(
            f"'{current.name}' is still {current.state}; wait for it to finish, "
            "or press 'Forget job' if the process was killed."
        )
    log_path = ui_dir(data_root) / f"{slug(name)}-{datetime.now():%Y%m%d-%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = JobStatus(
        name=name,
        state="starting",
        argv=tuple(argv),
        started_at=now(),
        log_path=str(log_path),
    )
    write_status(path, status)
    log_path.write_text(
        f"=== {name} | {status.started_at} ===\n$ {' '.join(argv)}\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(Path(__file__).resolve()), "--run-job", str(path), "--", *argv]
    common: dict[str, Any] = {
        "cwd": str(cwd),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        # A *real* console, just invisible. DETACHED_PROCESS leaves the child with no
        # console at all, and on 2026-10-27 every such job died with
        # STATUS_CONTROL_C_EXIT (0xC000013A) seconds after starting - while the
        # status file stayed 'running' and blocked the console. A hidden console
        # keeps console control events behaving the way subprocess expects.
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        process = spawn(
            command,
            **common,
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
            startupinfo=startupinfo,
        )
    else:
        process = spawn(command, **common, start_new_session=True)
    pid = getattr(process, "pid", None)
    if pid:
        status = replace(status, pid=int(pid))
        write_status(path, status)
    return status


def run_job(status_path: Path, argv: Sequence[str]) -> int:
    """Child process: run one command, stream it into the log, record the verdict.

    Everything is wrapped so the status file always reaches a terminal state:
    a page that shows 'running' forever is worse than a page that shows a
    failure, because the operator can read a failure.
    """
    status = read_status(status_path)
    if status is None or not status.log_path:
        print("ANJANEYA ui: no job status file to update.", file=sys.stderr)
        return 2
    log_path = Path(status.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = replace(status, state="running", started_at=status.started_at or now())
    write_status(status_path, status)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n--- running | {now()} ---\n")
        handle.flush()
        try:
            code = subprocess.run(
                [sys.executable, *argv],
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        except OSError as exc:  # the interpreter itself could not start
            handle.write(f"\nANJANEYA ui: could not start the job: {exc}\n")
            code = 1
        handle.write(f"\n--- exit {code} | {now()} ---\n")
    write_status(
        status_path,
        replace(
            status,
            state="ok" if code == 0 else "failed",
            exit_code=code,
            finished_at=now(),
        ),
    )
    return code


def run_job_cli(argv: Sequence[str]) -> int:
    """The child's entry point: ``--run-job <status.json> -- <command...>``."""
    arguments = list(argv)
    try:
        index = arguments.index("--run-job")
    except ValueError:
        return 2
    if index + 1 >= len(arguments):
        return 2
    status_path = Path(arguments[index + 1])
    command = arguments[index + 2 :]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        return 2
    return run_job(status_path, command)


def read_log(path: str | Path | None, *, max_bytes: int = 64_000) -> str:
    """The tail of a job log - bounded, so a long backfill cannot stall the page."""
    if not path:
        return ""
    log_path = Path(path)
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Data previews: connect read-only, query, close. Never keep the handle.
# ---------------------------------------------------------------------------
class DatabaseBusy(RuntimeError):
    """Another process holds the database: a capture session or a pipeline run."""


def preview(
    db_path: str | Path,
    table: str,
    *,
    last: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> tuple[Any, int]:
    """First/last rows plus the row count, with the connection closed again.

    A cached connection would *be* the lock that blocks the evening pipeline: a
    read-only handle still prevents a writer from opening the file. So the
    handle lives for exactly as long as the two queries.
    """
    from core.storage.duckdb_bootstrap import connect

    connection = None
    try:
        connection = connect(db_path, read_only=True)
        frame = connection.execute(preview_query(table, last=last, limit=limit)).df()
        total = int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    except Exception as exc:
        if "lock" in str(exc).lower():
            raise DatabaseBusy(
                "another process holds the database (a capture session or a pipeline run) - try again once it finishes"
            ) from exc
        raise
    finally:
        if connection is not None:
            connection.close()
    return frame, total


def table_names() -> list[str]:
    """The previewable tables, in the order the tab lists them."""
    return [name for name, _, _ in PREVIEW_TABLES]


def table_label(name: str) -> str:
    """The human label for a table key."""
    for table, _, label in PREVIEW_TABLES:
        if table == name:
            return label
    return name


# ---------------------------------------------------------------------------
# Auth and artefacts: read-only views over what the scripts already wrote.
# ---------------------------------------------------------------------------
def data_root_of(config) -> Path:
    """The resolved ``settings.data.root`` - where logs, reports and models go."""
    root = Path(config.settings.data.root)
    return root if root.is_absolute() else Path(config.base_dir) / root


def token_store_path(config) -> Path:
    """The resolved token cache the login script writes and ``day`` reads."""
    store = Path(config.data_sources.zerodha.token_store.path)
    return store if store.is_absolute() else Path(config.base_dir) / store


def token_summary(config) -> dict[str, Any]:
    """The masked state of the cached token. Never the token itself."""
    from core.auth import TokenStore

    token = TokenStore(token_store_path(config)).load()
    if token is None:
        return {"present": False}
    reference = datetime.now(tz=token.expires_at.tzinfo) if token.expires_at else None
    return {
        "present": True,
        "api_key": mask(token.api_key),
        "access_token": mask(token.access_token),
        "expires_at": token.expires_at.isoformat() if token.expires_at else None,
        "remaining_minutes": round(token.remaining_minutes(reference), 1) if reference else None,
        "obtained_via": getattr(token, "obtained_via", None),
        "login_time": token.login_time.isoformat() if token.login_time else None,
        "usable": token.remaining_minutes(reference) > 0 if reference else False,
    }


def exchange_request_token(config, request_token: str, *, common: Sequence[str] = ()) -> tuple[bool, dict[str, Any]]:
    """Exchange a pasted ``request_token`` in this process, and keep nothing.

    Deliberately not a job: a job would have to put the token in the child's
    argv, and both the status file and the log record argv verbatim. Here the
    secret lives only in this call, and only the already-masked fields the login
    script prints are returned.
    """
    completed = subprocess.run(
        [sys.executable, LOGIN, "--json", "--request-token", normalise_request_token(request_token), *common],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except ValueError:
        payload = {}
    if completed.returncode != 0 or payload.get("status") not in {"ok", None}:
        return False, {
            "exit_code": completed.returncode,
            "detail": (completed.stderr or completed.stdout or "").strip()[-2000:],
        }
    return True, {
        "api_key": payload.get("api_key") or mask(None),
        "access_token": payload.get("access_token") or mask(None),
        "expires_at": payload.get("expires_at"),
        "remaining_minutes": payload.get("remaining_minutes"),
        "obtained_via": payload.get("obtained_via"),
    }


def _mtime(path: Path) -> float:
    """Sort key that tolerates a file vanishing between glob and sort."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def artefacts(data_root: str | Path, *, limit: int = 20) -> tuple[list[Path], list[Path]]:
    """The newest reports and charts the pipeline wrote, newest first."""
    root = Path(data_root)
    reports = sorted(root.glob("reports/**/*.md"), key=_mtime, reverse=True)
    images = sorted(root.glob("reports/**/*.png"), key=_mtime, reverse=True)
    return reports[:limit], images[:limit]


# ---------------------------------------------------------------------------
# The page. Every function below only reads state or starts one job.
# ---------------------------------------------------------------------------
def launch(st, name: str, argv: Sequence[str], *, data_root: Path) -> None:
    """Start a job, then re-run the page so the activity card is current.

    The message goes through the session state because ``st.rerun`` discards
    whatever this pass had already drawn.
    """
    try:
        status = start_job(name, argv, data_root=data_root)
    except JobBusy as exc:
        st.warning(str(exc))
        return
    st.session_state["flash"] = f"Started **{status.name}**: `{' '.join(argv)}`"
    st.rerun()


def render_job_panel(st, *, data_root: Path) -> None:
    """The sidebar's one source of truth about the child process."""
    st.sidebar.header("Activity")
    path = status_file(data_root)
    status = read_status(path)
    if status is None:
        st.sidebar.caption("Nothing has run from this console yet.")
        return
    mark = {"starting": "*", "running": "*", "ok": "+", "failed": "!"}.get(status.state, "?")
    st.sidebar.markdown(f"**{mark} {status.name}** - {status.state}")
    st.sidebar.caption(f"started {status.started_at or '-'} / finished {status.finished_at or '-'}")
    if status.exit_code is not None:
        st.sidebar.caption(f"exit code {status.exit_code}")
    with st.sidebar.expander("Command and log tail"):
        st.code(" ".join(status.argv), language="powershell")
        st.code(read_log(status.log_path, max_bytes=8000) or "(no output yet)", language="text")
    columns = st.sidebar.columns(2)
    if columns[0].button("Refresh", key="panel-refresh"):
        st.rerun()
    if columns[1].button("Forget job", key="panel-forget"):
        write_status(path, replace(status, state="abandoned", finished_at=now()))
        st.rerun()


def totp_available() -> bool:
    """True when user id, password and TOTP secret are all set (``.env`` included)."""
    return all(
        os.environ.get(name)
        for name in ("ZERODHA_USER_ID", "ZERODHA_PASSWORD", "ZERODHA_TOTP_KEY")
    )


def token_tab(st, config, *, common: Sequence[str], data_root: Path) -> None:
    """Today's token: is it alive, and the two ways to replace it."""
    summary = token_summary(config)
    if summary["present"]:
        if summary["usable"]:
            st.success(f"Token is usable - {summary['remaining_minutes']} minutes left.")
        else:
            st.error("The cached token has expired (or falls inside the 30-minute cutoff).")
        left, right = st.columns(2)
        left.text_input("API key", summary["api_key"], disabled=True, key="token-api-key")
        left.text_input("Access token", summary["access_token"], disabled=True, key="token-access-token")
        right.text_input("Expires", summary["expires_at"] or "-", disabled=True, key="token-expires")
        right.text_input(
            "Obtained via",
            f"{summary['obtained_via']} at {summary['login_time'] or '-'}",
            disabled=True,
            key="token-origin",
        )
        st.caption(
            "Masked on purpose: the console never renders a secret. "
            f"The token file lives at `{token_store_path(config)}`."
        )
    else:
        st.warning("No cached token yet. Kite expires one every morning, so `day` needs a login first.")

    st.divider()
    st.subheader("Get a fresh token")
    st.caption(
        "Kite expires the access token every morning (~06:00 IST), so this is a daily step. "
        "The console never asks for your password - the login is completed on Kite's own page."
    )
    if st.button("Login with browser (recommended)", type="primary", key="token-browser"):
        launch(st, "token-browser", login_now_argv(common=common), data_root=data_root)
    st.caption(
        "Opens the Kite login on this machine, catches the `request_token` on "
        "`127.0.0.1:5000`, then caches and verifies the token. Finish the password + 2FA "
        "within 4 minutes."
    )

    with st.expander("Kite blocked the browser login? Paste the request_token instead"):
        pasted = st.text_input(
            "request_token or the redirect URL",
            type="password",
            key="request-token",
            help="One use only, and it expires in a couple of minutes. Paste the bare token "
            "or the whole http://127.0.0.1:<port>/?request_token=... URL - both work. "
            "Exchanged here, never written to a log or a status file.",
        )
        if st.button("Exchange request token", disabled=not pasted, key="token-exchange"):
            with st.spinner("Talking to Kite..."):
                ok, detail = exchange_request_token(config, pasted, common=common)
            st.session_state["request-token"] = ""
            if ok:
                st.success(f"Token stored; expires {detail.get('expires_at')}.")
            else:
                st.error(f"The exchange failed (exit {detail.get('exit_code')}): {detail.get('detail')}")

    st.divider()
    if totp_available():
        columns = st.columns(2)
        if columns[0].button("Refresh token (TOTP)", key="token-totp"):
            launch(st, "token", login_argv(common=common), data_root=data_root)
        if columns[1].button("Force re-login (TOTP)", key="token-force"):
            launch(st, "token-force", login_argv(force=True, common=common), data_root=data_root)
        st.caption(
            "TOTP login reads `ZERODHA_USER_ID`, `ZERODHA_PASSWORD` and `ZERODHA_TOTP_KEY` "
            "from `.env`, so it needs no browser."
        )
    else:
        st.caption(
            "TOTP login is unavailable (`ZERODHA_TOTP_KEY` is not set), so those buttons are "
            "hidden: a button that cannot work is worse than no button."
        )


def _render_live_body(st, *, data_root: Path) -> None:
    """The live card: what the heartbeat publishes, re-rendered every 5 s."""
    payload = read_heartbeat(data_root)
    if payload is None:
        st.info("No capture heartbeat yet - no session has run on this machine.")
        return

    status = str(payload.get("status") or "?").upper()
    age = heartbeat_age(payload)
    if status == "RUNNING" and age is not None and age <= 90:
        st.success(f"Capturing - heartbeat {age:.0f}s old")
    elif status == "RUNNING":
        st.error(
            f"Session says RUNNING but the heartbeat is {age:.0f}s old"
            if age is not None
            else "Session says RUNNING but the heartbeat cannot be read"
        )
    else:
        st.info(f"Session {status} (last heartbeat {age:.0f}s ago)" if age is not None else f"Session {status}")

    first, second, third = st.columns(3)
    first.metric("Ticks", f"{int(payload.get('ticks_total') or 0):,}")
    first.metric("Ticks / minute", f"{float(payload.get('ticks_per_minute') or 0):,.0f}")
    second.metric("Snapshots", f"{int(payload.get('snapshots_total') or 0):,}")
    second.metric("Subscribed", f"{int(payload.get('subscribed_instruments') or 0):,}")
    third.metric("ATM strike", f"{payload.get('atm_strike') or '-'}")
    third.metric("Newest price", f"{payload.get('newest_price') or '-'}")

    offset = payload.get("clock_offset_seconds")
    st.table(
        [
            {"field": "connected", "value": str(bool(payload.get("connected")))},
            {"field": "last tick", "value": str(payload.get("last_tick_at") or "-")},
            {"field": "newest tick", "value": str(payload.get("newest_tick_timestamp") or "-")},
            {"field": "clock offset (s)", "value": "-" if offset is None else f"{float(offset):+.2f}"},
            {
                "field": "polls / failures",
                "value": f"{int(payload.get('poll_count') or 0)} / {int(payload.get('poll_failures') or 0)}",
            },
            {"field": "gaps", "value": str(payload.get("gap_count") or 0)},
            {"field": "reconnects", "value": str(payload.get("reconnects") or 0)},
            {"field": "buffered rows", "value": str(payload.get("buffered_rows") or 0)},
            {"field": "expiries", "value": ", ".join(payload.get("expiries") or []) or "-"},
            {"field": "run id", "value": str(payload.get("run_id") or "-")},
        ]
    )
    if offset is not None and abs(float(offset)) >= 1.0:
        st.caption(
            f"Exchange time is {float(offset):+.1f}s from this PC's clock. Bars are stamped "
            "with the **exchange** timestamp; this number is informational only."
        )
    for note in (payload.get("notes") or [])[-4:]:
        st.warning(str(note))


def live_tab(st, *, common: Sequence[str], data_root: Path) -> None:
    """Start or stop a capture, and watch it happen without touching the database."""
    st.subheader("Live capture")
    actions = st.columns(3)
    if actions[0].button("Start capture (browser login)", type="primary", key="live-start"):
        launch(st, "morning", morning_argv(common=common), data_root=data_root)
    if actions[1].button("Stop capture", key="live-stop"):
        path = request_stop(data_root)
        st.warning(
            f"Stop requested (`{path.name}`). The session finishes its current cycle, flushes "
            "and shuts down; everything already captured is kept."
        )
    if actions[2].button("Refresh now", key="live-refresh"):
        st.rerun()
    if stop_requested(data_root):
        st.info("A stop request is waiting for the session to notice it (checked every cycle).")
    st.caption(
        "Start = browser login, then `--mode capture` (waits for 09:15, stops at 15:30). "
        "The card below refreshes itself from `runs/heartbeat/capture.json`, so it keeps "
        "working while the capture holds the database."
    )

    st.divider()
    st.fragment(run_every="5s")(lambda: _render_live_body(st, data_root=data_root))()

    st.divider()
    job = read_status(status_file(data_root))
    st.caption(f"Job log tail - {getattr(job, 'name', 'no job')} ({getattr(job, 'state', '-')})")
    st.code(read_log(getattr(job, "log_path", None), max_bytes=4000) or "(no job log yet)", language="text")


def history_tab(st, *, common: Sequence[str], data_root: Path) -> None:
    """The one-time backfill. Plan first: it is the dry run and writes nothing."""
    st.caption(
        "Run this once on an empty database, then let `day` keep it current. "
        "`Plan` is `--dry-run`: it prints the chain and touches nothing."
    )
    left, middle, right = st.columns(3)
    years = left.number_input("NIFTY index years", 0.25, 10.0, 2.0, 0.25, key="hist-years")
    option_years = middle.number_input("Option years", 0.0, 2.0, 0.25, 0.25, key="hist-option-years")
    skip_options = right.checkbox("Skip the option backfill", value=False, key="hist-skip-options")
    skip_instruments = st.checkbox("Reuse the cached instrument master", value=False, key="hist-skip-instruments")
    st.caption(
        "Kite serves roughly 60 days of 1-minute option bars, so a quarter is plenty; "
        "the index history is daily bars and can go back years."
    )
    arguments = {
        "years": float(years),
        "option_years": float(option_years),
        "skip_instruments": bool(skip_instruments),
        "skip_options": bool(skip_options),
        "common": common,
    }
    columns = st.columns(2)
    if columns[0].button("Plan (no writes)", key="hist-plan"):
        launch(st, "history-plan", history_argv(dry_run=True, **arguments), data_root=data_root)
    if columns[1].button("Fetch history", type="primary", key="hist-fetch"):
        launch(st, "history", history_argv(**arguments), data_root=data_root)
    st.info(
        "A backfill is network-bound and can run for a while; it survives closing this page, "
        "and the sidebar shows the command it is still running."
    )


_PIPELINE_LABELS = {
    "ok": "OK",
    "running": "RUNNING",
    "failed": "FAILED",
    "skipped": "SKIPPED",
    "gate": "GATE",
    "dry_run": "DRY RUN",
}


def _render_pipeline_steps(st, *, data_root: Path) -> None:
    """The evening chain, read from the pipeline's own dated log."""
    st.subheader("Last post-close run")
    path = newest_file(Path(data_root) / "logs", "daily_pipeline_*.log")
    if path is None:
        st.info("No pipeline log yet - press 3 or 4 above.")
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return
    steps = parse_pipeline_log(text)
    st.caption(f"`{path.name}` - {len(steps)} step(s) announced")
    if not steps:
        st.info("The log has no step lines yet.")
        return
    st.table(
        [
            {
                "step": f"{item['position']}/{item['total']} {item['step']}",
                "status": _PIPELINE_LABELS.get(str(item["status"]), str(item["status"]).upper()),
                "seconds": "-" if item["seconds"] is None else f"{item['seconds']:.1f}",
                "what": item["summary"],
            }
            for item in steps
        ]
    )


def _render_readiness(st, config, *, data_root: Path) -> None:
    """How far the accumulation is from a model (files only, no database)."""
    st.subheader("Readiness")
    document = read_accumulation(data_root) or {}
    captured = document.get("captured_days")
    gates = (document.get("gate") or {}).get("gates") or {}
    gate_days = gates.get("ml_training_min_days") or config.model.data_gates.ml_training_min_days
    walk = config.model.walk_forward
    fold_days = walk.holdout_days + walk.train_window_days + walk.validate_window_days
    model = newest_model(data_root)

    columns = st.columns(4)
    columns[0].metric("Captured days", captured if captured is not None else "-")
    columns[1].metric("Train gate", gate_days)
    columns[2].metric("First fold", fold_days)
    columns[3].metric(
        "Days to first fold", max(0, fold_days - captured) if isinstance(captured, int) else "-"
    )
    st.caption(
        f"Newest model: {model.name if model else 'none yet'}. A walk-forward fold needs "
        f"{fold_days} captured days (holdout {walk.holdout_days} + train "
        f"{walk.train_window_days} + validate {walk.validate_window_days}); `--force` skips the "
        "gate but cannot create a fold."
    )
    if not document:
        st.info("No accumulation report yet - step 4 (Learn) writes one.")


def postclose_tab(st, config, *, common: Sequence[str], data_root: Path) -> None:
    """Everything after 15:30, in the order that works."""
    st.subheader("After the close")
    st.caption(
        "Order matters: the index top-up feeds the rebuild, and the rebuild feeds training. "
        "One command per button, one job at a time."
    )
    row = st.columns(2)
    if row[0].button("1 - Top up NIFTY index (network, seconds)", key="post-topup"):
        launch(st, "index-topup", index_topup_argv(common=common), data_root=data_root)
    if row[1].button("2 - Heal candles from REST (network)", key="post-heal"):
        launch(st, "backfill-options", backfill_options_argv(common=common), data_root=data_root)
    row = st.columns(2)
    if row[0].button("3 - Rebuild the day (math only)", key="post-rebuild"):
        launch(st, "day-no-capture", day_argv(no_capture=True, common=common), data_root=data_root)
    if row[1].button("4 - Learn: dataset, train, backtest, reports", type="primary", key="post-learn"):
        launch(st, "learn", learn_argv({}, common=common), data_root=data_root)
    st.caption(
        "`Learn` forwards `--weekly`, so it also writes report / explain / importance / charts / "
        "accumulation. Training exits 4 below the accumulation gate - that is the gate, not a bug."
    )

    st.divider()
    st.subheader("Research: P1 - is the signal worth building a model on?")
    st.caption(
        "Forward-return study over the stored NIFTY index alone: regime, composite band, "
        "distance to the previous session's high/low, and time of day. Read-only and offline. "
        "It answers *before* training whether the signal separates returns at all - no model, "
        "no advice, and every table carries its sample size and corrected p-value."
    )
    if st.button(
        "5 - Index study (P1: forward returns, no model)",
        key="post-index-study",
    ):
        launch(st, "index-study", index_study_argv(common=common), data_root=data_root)

    with st.expander("The exact commands behind these buttons"):
        st.code(
            "\n".join(
                " ".join(argv)
                for argv in (
                    index_topup_argv(common=common),
                    backfill_options_argv(common=common),
                    day_argv(no_capture=True, common=common),
                    learn_argv({}, common=common),
                    index_study_argv(common=common),
                )
            ),
            language="powershell",
        )

    st.divider()
    _render_pipeline_steps(st, data_root=data_root)
    st.divider()
    _render_readiness(st, config, data_root=data_root)
    st.divider()
    with st.expander("First-time setup - full history backfill (run once)"):
        history_tab(st, common=common, data_root=data_root)


def data_tab(st, config, *, data_root: Path) -> None:
    """First or last rows of a table, with the handle closed before the page draws."""
    status = read_status(status_file(data_root))
    if status is not None and status.active:
        st.warning(
            f"`{status.name}` is running and holds the database writer. Previews resume "
            "when it finishes - opening a read-only handle now would only fail with a lock error."
        )
        return
    table = st.selectbox("Table", table_names(), format_func=table_label, key="data-table")
    left, middle, right = st.columns([1, 1, 2])
    edge = left.radio("Rows", ["First", "Last"], horizontal=True, key="data-edge")
    limit = middle.number_input("How many", 1, 100, DEFAULT_LIMIT, 1, key="data-limit")
    if right.button("Reload", key="data-reload"):
        st.rerun()
    try:
        frame, total = preview(config.db_path, table, last=edge == "Last", limit=int(limit))
    except DatabaseBusy as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # a missing table or a schema drift: show it, do not hide it
        st.error(f"{type(exc).__name__}: {exc}")
        return
    st.caption(f"{total:,} rows in `{table}` - showing the {edge.lower()} {len(frame)}.")
    if frame.empty:
        st.info("The table exists but is empty; the step that fills it has not run yet.")
        return
    st.dataframe(frame, width="stretch")


def run_tab(st, *, common: Sequence[str], data_root: Path) -> None:
    """The two chains: the daily loop, and the evening's deliberate run."""
    st.subheader("Switches - these shape `learn` only")
    left, middle, right = st.columns(3)
    switches = {
        "ml": left.checkbox("ML (dataset + train)", value=True, key="switch-ml"),
        "backtest": middle.checkbox("Backtest", value=True, key="switch-backtest"),
        "research": right.checkbox("Research (report, explain, importance, charts)", value=True, key="switch-research"),
    }
    skip = skip_argument(switches)
    st.code(f"--skip {skip}" if skip else "# nothing skipped", language="powershell")
    st.caption("Always on, because they are the dataset: " + ", ".join(ALWAYS_ON) + ".")
    with st.expander("Switches this project cannot offer, and why"):
        for name, reason in UNSWITCHABLE.items():
            st.markdown(f"**{name}** - {reason}")

    st.divider()
    st.subheader("Morning: login + capture (one click at 09:00)")
    st.caption(
        "Opens the Kite login on this machine, catches the `request_token` on "
        "`127.0.0.1:5000`, exchanges and verifies it, then starts the capture (waits for "
        "09:15, stops at 15:30). If today's token is already cached and valid the browser "
        "is skipped and the capture starts immediately. Finish the password + 2FA within "
        "4 minutes, or the job fails cleanly and **nothing is captured**."
    )
    if st.button("Morning - browser login + capture", type="primary", key="run-morning"):
        launch(st, "morning", morning_argv(common=common), data_root=data_root)

    st.divider()
    st.subheader("Daily loop")
    st.caption(
        "`anj.py day` always skips `train,backtest`: a model trained before the close is a "
        "model trained on an incomplete day. The switches above therefore do not reach it."
    )
    columns = st.columns(2)
    if columns[0].button("Day - capture + post-close math", type="primary", key="run-day"):
        launch(st, "day", day_argv(common=common), data_root=data_root)
    if columns[1].button("Day - math only (holiday, or after the fact)", key="run-day-nocap"):
        launch(st, "day-no-capture", day_argv(no_capture=True, common=common), data_root=data_root)

    st.divider()
    st.subheader("Post-close: index top-up")
    st.caption(
        "The session tops the NIFTY index up at 09:05 - i.e. up to the **previous** day - and "
        "the engine builds the day's `nifty_*` block from `index_history` alone. After the "
        "close, run this **first**, then press **Day - math only** above to rebuild the day "
        "with its NIFTY context. Skipping it leaves that day's `nifty_*` columns NULL."
    )
    if st.button("Top up index - NIFTY daily + 1m (network, seconds)", key="run-index-topup"):
        launch(st, "index-topup", index_topup_argv(common=common), data_root=data_root)

    st.divider()
    st.subheader("Evening run")
    force = st.checkbox(
        "Train even if the accumulation gate is not open (--force)",
        value=False,
        key="run-force",
        help="Spec 4.3 refuses to train on fewer days than it wants. This overrides that, knowingly.",
    )
    if st.button("Learn - dataset, train, backtest, reports", type="primary", key="run-learn"):
        launch(st, "learn", learn_argv(switches, force=force, common=common), data_root=data_root)

    with st.expander("The exact commands behind these buttons"):
        st.code(
            " ".join(morning_argv(common=common))
            + "\n"
            + " ".join(day_argv(common=common))
            + "\n"
            + " ".join(day_argv(no_capture=True, common=common))
            + "\n"
            + " ".join(index_topup_argv(common=common))
            + "\n"
            + " ".join(learn_argv(switches, force=force, common=common)),
            language="powershell",
        )


def results_tab(st, *, data_root: Path) -> None:
    """What the pipeline wrote: the newest reports and the charts beside them."""
    reports, images = artefacts(data_root)
    left, right = st.columns(2)
    with left:
        st.subheader("Reports")
        if not reports:
            st.info("No report has been written yet; `learn` produces them.")
        for path in reports:
            with st.expander(path.relative_to(data_root).as_posix(), expanded=False):
                try:
                    st.markdown(path.read_text(encoding="utf-8"))
                except OSError as exc:
                    st.error(f"Could not read {path.name}: {exc}")
    with right:
        st.subheader("Charts")
        if not images:
            st.info("No chart has been written yet; the `charts` step produces them.")
        for path in images:
            st.image(str(path), caption=path.name, width="stretch")


#: The operator routine, rendered in the console. One source: the file below.
PLAYBOOK = REPO_ROOT / "docs" / "daily-playbook.md"


def playbook_tab(st) -> None:
    """The daily routine, straight from ``docs/daily-playbook.md``.

    The document is the source of truth - the console only renders it - so the
    instructions an operator reads here cannot drift from the ones in the repo.
    """
    st.subheader("Daily playbook")
    try:
        text = PLAYBOOK.read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"Could not read {PLAYBOOK}: {exc}")
        return
    st.markdown(text)
    st.caption(f"Source: `{PLAYBOOK.relative_to(REPO_ROOT).as_posix()}` - edit that file, not this page.")


def main(argv: Sequence[str] | None = None) -> int:
    """Either be the child (``--run-job``) or draw the page."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--run-job" in arguments:
        return run_job_cli(arguments)

    import streamlit as st  # the optional `ui` extra; imported only when serving

    st.set_page_config(page_title="ANJANEYA", page_icon="\U0001f9ff", layout="wide")
    st.title("ANJANEYA operator console")
    st.caption(
        "A thin client over `scripts/anj.py`: every button runs a command you could type, "
        "and the CLI stays the source of truth."
    )

    st.sidebar.header("Configuration")
    config_dir = st.sidebar.text_input("Config directory (blank = ./config)", value="", key="config-dir")
    # `.env` is loaded so the page can say whether the TOTP login is even possible -
    # a button that cannot work is worse than no button. Nothing is ever rendered.
    load_env_file(config_dir=Path(config_dir) if config_dir else None)
    try:
        config = load_config(config_dir=Path(config_dir) if config_dir else None)
    except Exception as exc:
        st.error(f"Could not load the configuration: {exc}")
        return 1
    data_root = data_root_of(config)
    common = common_flags(config_dir=config_dir)
    st.sidebar.caption(f"database: {config.db_path}")
    st.sidebar.caption(f"data root: {data_root}")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.info(flash)
    render_job_panel(st, data_root=data_root)

    token_view, live_view, postclose_view, data_view, results_view, playbook_view = st.tabs(
        ["Token", "Live", "Post-close", "Data", "Results", "Playbook"]
    )
    with token_view:
        token_tab(st, config, common=common, data_root=data_root)
    with live_view:
        live_tab(st, common=common, data_root=data_root)
    with postclose_view:
        postclose_tab(st, config, common=common, data_root=data_root)
    with data_view:
        data_tab(st, config, data_root=data_root)
    with results_view:
        results_tab(st, data_root=data_root)
    with playbook_view:
        playbook_tab(st)
    return 0


if __name__ == "__main__":
    # Streamlit imports this file and calls main() itself; it must not see
    # SystemExit(0), which would skip its own error collection. The job child
    # (`--run-job`) still needs a real exit code, so raise only on failure.
    _code = main()
    if _code:
        raise SystemExit(_code)
