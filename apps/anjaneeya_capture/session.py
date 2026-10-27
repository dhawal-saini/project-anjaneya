"""One ANJANEYA-CAPTURE session, start to finish.

The session is where every Phase 2 piece meets: the instrument master decides what
to subscribe to, the WebSocket delivers ticks, the poller adds OI snapshots, the
writer batches them into DuckDB and mirrors them to Parquet, the heartbeat says
whether it is all still alive, and the coverage tracker records what actually
arrived.

Two rules shape the code:

* **Everything expensive is injected** (client, ticker adapter, writer, clock,
  sleep) so a whole session can be exercised offline. A capture loop that only
  works against the real broker is a capture loop nobody can test.
* **Failure is reported, never hidden.** A session that captured half a day must
  end with a summary that says so: exit codes, coverage rows and the run manifest
  all carry the same numbers.

Importing this module performs no I/O.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from apps.anjaneeya_capture.heartbeat import Heartbeat, HeartbeatState
from apps.anjaneeya_capture.instruments import (
    InstrumentDump,
    InstrumentMaster,
    OptionUniverse,
    build_option_universe,
    load_instrument_dump,
    resolve_atm_reference,
    to_instrument_rows,
)
from apps.anjaneeya_capture.option_chain_poller import OptionChainPoller
from apps.anjaneeya_capture.storage_writer import CaptureWriter, CoverageTracker
from apps.anjaneeya_capture.websocket_listener import ContractMeta, TickStream, metadata_for
from core.constants import get_timezone

__all__ = [
    "STATUS_FAILED",
    "STATUS_NO_SESSION",
    "STATUS_PARTIAL",
    "STATUS_SUCCESS",
    "CaptureAborted",
    "CaptureSession",
    "RunSummary",
    "SessionError",
    "SessionPlan",
    "SessionStatus",
]

#: Session outcomes.
SessionStatus = str
STATUS_SUCCESS = "SUCCESS"
STATUS_NO_SESSION = "NO_SESSION"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"


class SessionError(RuntimeError):
    """Raised when a capture session cannot start or must abort."""


class CaptureAborted(SessionError):
    """Raised when a session must stop early without discarding what it captured."""


@dataclass(slots=True)
class SessionPlan:
    """What a session intends to do - printed by ``--dry-run``, logged on start."""

    session_date: date
    is_trading_day: bool
    session_start: datetime | None = None
    session_end: datetime | None = None
    reference_price: float | None = None
    reference_price_source: str | None = None
    universe: OptionUniverse | None = None
    subscription_batches: int = 0
    poll_requests_per_cycle: int = 0
    backfill_requests: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def window_minutes(self) -> float:
        """Minutes between the session bounds (0 when there is no session)."""
        if self.session_start is None or self.session_end is None:
            return 0.0
        return round((self.session_end - self.session_start).total_seconds() / 60, 2)

    def describe(self) -> dict[str, Any]:
        """Log-safe summary (used by ``--dry-run`` and the run manifest)."""
        return {
            "session_date": self.session_date.isoformat(),
            "is_trading_day": self.is_trading_day,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "window_minutes": self.window_minutes,
            "reference_price": self.reference_price,
            "reference_price_source": self.reference_price_source,
            "universe": self.universe.describe() if self.universe is not None else None,
            "subscription_batches": self.subscription_batches,
            "poll_requests_per_cycle": self.poll_requests_per_cycle,
            "backfill_requests": len(self.backfill_requests),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """A short human-readable plan (the ``--dry-run`` printout)."""
        lines = [
            f"session date        : {self.session_date.isoformat()} "
            f"({'trading day' if self.is_trading_day else 'NOT a trading day'})"
        ]
        if self.session_start and self.session_end:
            lines.append(
                f"session window      : {self.session_start:%H:%M} .. {self.session_end:%H:%M} "
                f"({self.window_minutes:.0f} minutes)"
            )
        if self.reference_price is not None:
            lines.append(
                f"reference price     : {self.reference_price} ({self.reference_price_source})"
            )
        if self.universe is not None:
            lines.append(
                f"ATM strike          : {self.universe.atm_strike:g} "
                f"(expiries {', '.join(e.isoformat() for e in self.universe.expiries)})"
            )
            lines.append(
                f"subscriptions       : {len(self.universe.options)} option contract(s) + "
                f"{len(self.universe.reference_tokens)} reference instrument(s) in "
                f"{self.subscription_batches} batch(es)"
            )
            if self.universe.missing:
                lines.append(
                    f"missing contracts   : {len(self.universe.missing)} "
                    f"(e.g. {self.universe.missing[0]})"
                )
        lines.append(f"option-chain polls  : {self.poll_requests_per_cycle} request(s) per cycle")
        lines.append(f"index backfill      : {len(self.backfill_requests)} planned request(s)")
        for note in self.notes:
            lines.append(f"note                : {note}")
        return "\n".join(lines)



@dataclass(slots=True)
class RunSummary:
    """Everything one capture session actually did."""

    status: SessionStatus
    session_date: date
    started_at: datetime | None = None
    finished_at: datetime | None = None
    reason: str | None = None
    universe: dict[str, Any] | None = None
    stream: dict[str, Any] = field(default_factory=dict)
    writer: dict[str, Any] = field(default_factory=dict)
    poller: dict[str, Any] = field(default_factory=dict)
    backfill: dict[str, Any] = field(default_factory=dict)
    coverage_rows: int = 0
    coverage_verdicts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the session ran to completion without data loss."""
        return self.status == STATUS_SUCCESS

    def describe(self) -> dict[str, Any]:
        """Log-safe summary (stored in the run manifest's ``data_coverage``)."""
        return {
            "status": self.status,
            "reason": self.reason,
            "session_date": self.session_date.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "universe": self.universe,
            "stream": self.stream,
            "writer": self.writer,
            "poller": self.poller,
            "backfill": self.backfill,
            "coverage_rows": self.coverage_rows,
            "coverage_verdicts": self.coverage_verdicts,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """An operator-friendly block printed at the end of a session."""
        lines = [f"ANJANEYA-CAPTURE session result: {self.status}"]
        if self.reason:
            lines.append(f"  reason       : {self.reason}")
        lines.append(f"  session date : {self.session_date.isoformat()}")
        if self.universe:
            lines.append(
                f"  universe     : ATM {self.universe.get('atm_strike')} / "
                f"{self.universe.get('options')} option contract(s), "
                f"{self.universe.get('tokens')} instrument(s)"
            )
        stream = self.stream or {}
        lines.append(
            f"  ticks        : {stream.get('ticks', 0)} "
            f"(rejected {stream.get('rejected', 0)}, suspect {stream.get('suspect', 0)}, "
            f"out-of-session {stream.get('out_of_session', 0)})"
        )
        lines.append(
            f"  websocket    : reconnects {stream.get('reconnects', 0)}, "
            f"restarts {stream.get('restarts', 0)}, stalls {stream.get('stalls', 0)}"
        )
        poller = self.poller or {}
        lines.append(
            f"  snapshots    : {poller.get('rows', 0)} over {poller.get('polls', 0)} poll(s), "
            f"{poller.get('failures', 0)} failure(s)"
        )
        writer = self.writer or {}
        lines.append(
            f"  stored       : ticks {writer.get('ticks', 0)}, "
            f"snapshots {writer.get('snapshots', 0)}, "
            f"index history {writer.get('index_history', 0)}, "
            f"coverage {writer.get('coverage', 0)}"
        )
        if writer.get("dropped_rows"):
            lines.append(f"  LOST ROWS    : {writer['dropped_rows']} (see the log!)")
        if writer.get("spilled_rows"):
            lines.append(f"  spilled      : {writer['spilled_rows']} row(s) written to Parquet only")
        if self.coverage_verdicts:
            lines.append(f"  coverage     : {self.coverage_verdicts}")
        for note in self.notes:
            lines.append(f"  note         : {note}")
        return "\n".join(lines)



# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------
class CaptureSession:
    """Runs one capture session end to end.

    Nothing here talks to the broker directly: the REST client, the ticker
    adapter, the writer and the clock are all injectable, which is what lets the
    integration test drive a *complete* session (ticks, snapshots, coverage,
    heartbeat, files on disk) without a network or a real credential.
    """

    def __init__(
        self,
        context: Any,
        *,
        client: Any = None,
        adapter_factory: Callable[[], Any] | None = None,
        writer: Any = None,
        poller_thread: bool = True,
        websocket_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        stop_event: Any = None,
        machine_tz: Any = None,
        heartbeat: Any = None,
        request_token: str | None = None,
        monitor_interval_seconds: float = 1.0,
    ) -> None:
        import time as _time

        self.context = context
        self.config = context.config
        self.calendar = context.calendar
        self.logger = context.logger
        self.run_id = context.run_context.run_id
        self.capture_config = self.config.data_sources.zerodha.capture
        self.websocket_config = self.config.data_sources.zerodha.websocket
        self.tz = get_timezone(self.config.timezone_name)

        self._client = client
        self._adapter_factory = adapter_factory
        self._writer = writer
        self._poller_thread = poller_thread
        self._websocket_enabled = websocket_enabled
        self._clock = clock or (lambda: datetime.now(self.tz))
        self._sleep = sleep or _time.sleep
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._machine_tz = machine_tz
        self._heartbeat = heartbeat
        self._request_token = request_token
        self._monitor_interval = monitor_interval_seconds

        self.plan_result: SessionPlan | None = None
        self.writer: Any = None
        self.stream: Any = None
        self.poller: Any = None
        self._master: InstrumentMaster | None = None

    # -- helpers ------------------------------------------------------------
    def _now(self) -> datetime:
        """The current session-local time (injectable for tests)."""
        return self._clock()

    def _note(self, summary: RunSummary, message: str) -> None:
        """Record a note on the summary and in the run manifest."""
        summary.notes.append(message)
        self.context.run_context.add_note(message)
        if self.logger is not None:
            self.logger.warning(message)

    def client(self) -> Any:
        """Return the REST client, creating one from the stored token if needed.

        Autentication failures surface as :class:`SessionError` with an
        actionable message: a capture session that cannot authenticate must fail
        loudly *before* it claims to be capturing anything.
        """
        if self._client is not None:
            return self._client
        try:
            from core.auth.zerodha_auth import resolve_access_token
            from core.auth.zerodha_client import ZerodhaClient

            zerodha = self.config.data_sources.zerodha
            token = resolve_access_token(
                zerodha,
                request_token=self._request_token,
                timezone_name=self.config.timezone_name,
                validity_cutoff=zerodha.token_store.validity_cutoff,
                refresh_on_startup=zerodha.token_store.refresh_on_startup,
            )
            self._client = ZerodhaClient.from_config(
                self.config, token.access_token, logger=self.logger
            )
        except Exception as exc:  # noqa: BLE001 - normalised into a session error
            raise SessionError(
                f"could not obtain a Kite session: {type(exc).__name__}: {exc}. "
                "Run `uv run python scripts/daily_login.py` (or capture with --request-token) "
                "and check ZERODHA_* in .env."
            ) from exc
        return self._client

    # -- instrument master --------------------------------------------------
    def load_master(
        self, *, client: Any = None, force_refresh: bool = False, now: datetime | None = None
    ) -> InstrumentDump:
        """Load (cache-first) the instrument dump and cache it on disk."""
        instruments_cfg = self.capture_config.instruments
        return load_instrument_dump(
            client,
            cache_dir=self.config.instrument_cache_dir,
            exchanges=instruments_cfg.exchanges,
            ttl_hours=instruments_cfg.refresh_ttl_hours,
            now=now or self._now(),
            logger=self.logger,
            force_download=force_refresh,
        )

    def persist_instruments(self, dump: InstrumentDump, *, now: datetime | None = None) -> int:
        """Upsert the instrument master into DuckDB (keyed by token)."""
        rows = to_instrument_rows(
            dump.master.instruments, run_id=self.run_id, fetched_at=now or self._now()
        )
        if not rows:
            return 0
        from core.storage import connect, insert_rows

        connection = connect(self.config.db_path)
        try:
            written = insert_rows(connection, "instruments", rows, mode="replace")
        finally:
            connection.close()
        dropped = len(dump.master.instruments) - len(rows)
        if dropped and self.logger is not None:
            self.logger.warning(
                f"{dropped} instrument row(s) were not storable (missing contract identity)"
            )
        return written


    # -- reference price + universe ----------------------------------------
    def reference_price(
        self, master: InstrumentMaster, *, client: Any = None, now: datetime | None = None
    ) -> tuple[float, str]:
        """Price used to place the ATM strike - the ``atm.reference`` instrument's.

        ``spot`` means the NIFTY index, ``futures`` the futures contract; the
        instrument is chosen by :func:`apps.anjaneeya_capture.instruments.resolve_atm_reference`,
        the same helper the universe itself uses, so the price and the contract it
        describes can never drift apart.

        A live ``ltp`` when a client is available; otherwise the instrument
        dump's ``last_price`` (the previous close), which is enough to *plan* a
        session offline - and is labelled as such wherever it appears.
        """
        universe_cfg = self.config.option_universe
        instrument = resolve_atm_reference(
            master, universe_config=universe_cfg, day=(now or self._now()).date()
        )
        if instrument is None:
            raise SessionError(
                f"the reference instrument for ATM ({universe_cfg.atm.reference!r}) is not in the "
                "instrument master; check option_universe.yaml and refresh the dump"
            )

        if client is not None:
            try:
                payload = client.ltp(instrument.quote_symbol) or {}
                quote = payload.get(instrument.quote_symbol) or {}
                price = quote.get("last_price")
                if price:
                    return float(price), f"rest-ltp:{instrument.tradingsymbol}"
            except Exception as exc:  # noqa: BLE001 - fall back to the dump price
                if self.logger is not None:
                    self.logger.warning(
                        f"could not fetch the live reference price ({type(exc).__name__}: {exc}); "
                        "falling back to the instrument dump's last price"
                    )
        if instrument.last_price:
            return float(instrument.last_price), (
                f"instrument-dump:{instrument.tradingsymbol} (previous close)"
            )
        stored = self._last_stored_index_close(instrument.tradingsymbol, now=now)
        if stored is not None:
            return stored, f"index-history:{instrument.tradingsymbol} (last stored close)"
        raise SessionError(
            f"no reference price available for {instrument.tradingsymbol}: the live quote failed, "
            "the instrument dump carries no price (Kite reports 0 for index rows) and "
            "index_history has no stored close; run `--mode backfill` once, or plan while the "
            "market is open so the live quote answers"
        )

    def _last_stored_index_close(self, symbol: str, *, now: datetime | None = None) -> float | None:
        """The newest stored ``index_history`` close for ``symbol``, or ``None``.

        The third - and fully offline - way to anchor the ATM. Kite's instrument
        dump reports ``last_price: 0`` for index rows, so a *pre-open* plan has
        neither a live quote nor a dump price; the history ``--mode backfill``
        already stored is enough to show the strikes and expiries a session is
        about to spend its request budget on. Only rows at or before ``now`` are
        read, so a plan can never anchor on a bar that has not happened yet.
        """
        from core.storage import connect, read_index_history

        db_path = Path(self.config.db_path)
        if not db_path.exists():
            return None
        moment = now or self._now()
        try:
            connection = connect(db_path)
        except Exception:  # noqa: BLE001 - a database that will not open is not a price
            return None
        try:
            frame = read_index_history(connection, end=moment, symbols=[symbol])
        except Exception:  # noqa: BLE001 - a missing table is not a price either
            return None
        finally:
            connection.close()
        if frame is None or frame.empty:
            return None
        try:
            return float(frame.sort_values("timestamp").iloc[-1]["close"])
        except (KeyError, TypeError, ValueError):
            return None

    # -- plan ---------------------------------------------------------------
    def plan(
        self,
        *,
        client: Any = None,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> SessionPlan:
        """Resolve everything a session would do, without capturing anything.

        With ``client=None`` this is fully offline: the cached instrument dump
        and its previous-close price are enough to show the strikes, expiries and
        request budget an operator is about to spend.
        """
        moment = now or self._now()
        day = self.calendar.local_date(moment)
        is_trading_day = self.calendar.is_trading_day(day)
        plan = SessionPlan(session_date=day, is_trading_day=is_trading_day)
        if is_trading_day:
            plan.session_start = self.calendar.session_start(day)
            plan.session_end = self.calendar.session_end(day)
        else:
            plan.notes.append("not a trading day: a session would exit immediately")

        dump = self.load_master(client=client, force_refresh=force_refresh, now=moment)
        if dump.source != "download":
            plan.notes.append("instrument master taken from the on-disk cache")
        if dump.master.rejected_rows:
            plan.notes.append(f"{dump.master.rejected_rows} unparseable dump row(s)")
        price, source = self.reference_price(dump.master, client=client, now=moment)
        plan.reference_price = price
        plan.reference_price_source = source
        plan.universe = build_option_universe(
            universe_config=self.config.option_universe,
            master=dump.master,
            reference_price=price,
            day=day,
            calendar=self.calendar,
            resolved_at=moment,
            logger=None,
        )
        tokens = plan.universe.tokens
        plan.subscription_batches = (
            -(-len(tokens) // self.websocket_config.max_instruments) if tokens else 0
        )
        options = plan.universe.options
        plan.poll_requests_per_cycle = (
            -(-len(options) // self.config.data_sources.zerodha.option_chain.quotes_per_request)
            if options
            else 0
        )
        plan.backfill_requests = self._plan_index_requests(dump.master, now=moment)
        return plan


    def index_targets(self, master: InstrumentMaster) -> list[Any]:
        """Map ``index_history.instruments`` names onto real instruments."""
        from apps.anjaneeya_capture.nifty_index_fetcher import FetchTarget

        universe_cfg = self.config.option_universe
        reference_cfg = universe_cfg.reference_instruments
        day = self.calendar.local_date(self._now())
        targets: list[Any] = []
        for name in universe_cfg.index_history.instruments:
            spec = getattr(reference_cfg, name, None)
            if spec is None:
                continue
            if name == "nifty_futures":
                futures = master.futures(
                    exchange=spec.exchange, name=universe_cfg.underlying, on_or_after=day
                )
                if futures:
                    targets.append(FetchTarget.from_instrument(futures[0]))
                continue
            instrument = master.reference(
                exchange=spec.exchange,
                name=universe_cfg.underlying,
                tradingsymbol=spec.tradingsymbol,
            )
            if instrument is not None:
                targets.append(FetchTarget.from_instrument(instrument))
        return targets

    def _plan_index_requests(self, master: InstrumentMaster, *, now: datetime) -> list[str]:
        """Describe the top-up requests a session would make (fully offline)."""
        index_history = self.config.option_universe.index_history
        targets = self.index_targets(master)
        if not targets:
            return []
        from apps.anjaneeya_capture.nifty_index_fetcher import IndexHistoryFetcher

        fetcher = IndexHistoryFetcher(
            client=None,
            historical_config=self.capture_config.historical,
            run_id=self.run_id,
            logger=None,
            now_fn=lambda: now,
            timezone_name=self.config.timezone_name,
        )
        return fetcher.plan(
            targets, days=index_history.topup_days, interval=index_history.topup_interval, now=now
        )

    # -- index top-up -------------------------------------------------------
    def top_up_index(self, master: InstrumentMaster, *, now: datetime | None = None) -> dict[str, Any]:
        """Re-fetch the last ``topup_days`` of index history before the session.

        This is a *healing* step: the backfill establishes the long history, and
        a daily top-up repairs whatever yesterday's run missed. It is deliberately
        non-fatal - a failed top-up is reported, never allowed to prevent live
        capture.
        """
        from apps.anjaneeya_capture.nifty_index_fetcher import IndexHistoryFetcher

        index_history = self.config.option_universe.index_history
        targets = self.index_targets(master)
        if not targets:
            return {"rows": 0, "failures": [], "requests": 0, "planned": 0}
        fetcher = IndexHistoryFetcher(
            client=self.client(),
            historical_config=self.capture_config.historical,
            run_id=self.run_id,
            logger=self.logger,
            now_fn=self._now,
            timezone_name=self.config.timezone_name,
        )
        result = fetcher.top_up(
            targets,
            days=index_history.topup_days,
            interval=index_history.topup_interval,
            now=now or self._now(),
        )
        if result.rows:
            self.writer.write_index_history(result.rows)
        summary = {
            "rows": len(result.rows),
            "requests": result.requests,
            "splits": result.splits,
            "windows_planned": result.windows_planned,
            "rejected": result.rejected,
            "failures": result.failures,
            "planned": len(result.rows) + result.rejected,
            "interval": index_history.topup_interval,
            "symbols": result.symbols,
        }
        if self.logger is not None:
            self.logger.info(
                f"index top-up: {len(result.rows)} row(s) from {result.requests} request(s)"
                + (f", {len(result.failures)} failure(s)" if result.failures else "")
            )
        return summary


    # -- the session window -------------------------------------------------
    def _session_deadline(
        self, session_end: datetime, *, smoke_minutes: int | None, until: datetime | None
    ) -> datetime:
        """The earliest of: session close, ``--until`` and the smoke-test budget."""
        deadline = session_end
        if until is not None:
            deadline = min(deadline, until)
        if smoke_minutes:
            deadline = min(deadline, self._now() + timedelta(minutes=smoke_minutes))
        return deadline

    def _wait_for_open(self, session_start: datetime) -> bool:
        """Wait for the session open; ``False`` means the session was abandoned."""
        guard = timedelta(minutes=self.capture_config.session_start_guard_minutes)
        now = self._now()
        if session_start - now > guard:
            if self.logger is not None:
                self.logger.warning(
                    f"started {(session_start - now).total_seconds() / 60:.0f} minutes before the "
                    f"open; the guard is {self.capture_config.session_start_guard_minutes} minutes. "
                    "Refusing to wait that long - start capture closer to 09:15 or use --smoke."
                )
            return False
        if self.logger is not None:
            self.logger.info(
                f"waiting for the session open ({session_start:%H:%M}); "
                f"{(session_start - now).total_seconds() / 60:.1f} minutes to go"
            )
        while self._now() < session_start:
            if self.stop_event.is_set():
                return False
            self._sleep(min(5.0, (session_start - self._now()).total_seconds()))
        return True

    # -- run ----------------------------------------------------------------
    def run(
        self,
        *,
        smoke_minutes: int | None = None,
        until: datetime | None = None,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> RunSummary:
        """Capture one session: subscribe, stream, poll, store, report.

        Returns a :class:`RunSummary` instead of raising whenever the session
        ended *cleanly but not profitably* (a holiday, a closed session); genuine
        configuration/authentication problems raise :class:`SessionError`.
        """
        started = now or self._now()
        day = self.calendar.local_date(started)
        summary = RunSummary(status=STATUS_SUCCESS, session_date=day, started_at=started)
        self.context.run_context.data_coverage = summary.describe()

        if not self.calendar.is_trading_day(day):
            summary.status = STATUS_NO_SESSION
            summary.reason = f"{day.isoformat()} is not an NSE trading day"
            summary.finished_at = self._now()
            return summary

        session_start = self.calendar.session_start(day)
        session_end = self.calendar.session_end(day)
        if started >= session_end:
            if self.capture_config.auto_stop_at_session_end:
                summary.status = STATUS_NO_SESSION
                summary.reason = f"the session closed at {session_end:%H:%M}"
                summary.finished_at = self._now()
                return summary
            if self.logger is not None:
                self.logger.warning(
                    "capture started after the session close; auto_stop_at_session_end is disabled, "
                    "so ticks (if any) will be stored with quality_flag=OUT_OF_SESSION"
                )
        elif started < session_start and self.capture_config.wait_for_session_open:
            if not self._wait_for_open(session_start):
                summary.status = STATUS_NO_SESSION
                summary.reason = "abandoned while waiting for the session open"
                summary.finished_at = self._now()
                return summary

        return self._capture(summary, session_start=session_start, session_end=session_end,
                             smoke_minutes=smoke_minutes, until=until, force_refresh=force_refresh)


    # -- the capture body ---------------------------------------------------
    def _capture(
        self,
        summary: RunSummary,
        *,
        session_start: datetime,
        session_end: datetime,
        smoke_minutes: int | None,
        until: datetime | None,
        force_refresh: bool,
    ) -> RunSummary:
        """Everything between "the session is open" and "the session is stored"."""
        client = self.client()
        dump = self.load_master(client=client, force_refresh=force_refresh, now=self._now())
        stored = self.persist_instruments(dump)
        self._master = dump.master
        if self.logger is not None:
            self.logger.info(
                f"instrument master ready: {len(dump.master)} instrument(s) ({dump.source}), "
                f"{stored} row(s) stored in DuckDB"
            )
        price, price_source = self.reference_price(dump.master, client=client)
        universe = build_option_universe(
            universe_config=self.config.option_universe,
            master=dump.master,
            reference_price=price,
            day=summary.session_date,
            calendar=self.calendar,
            resolved_at=self._now(),
            logger=self.logger,
        )
        summary.universe = universe.describe()
        if self.logger is not None:
            self.logger.info(f"universe resolved: {universe.describe()} ({price_source})")
        if universe.is_empty:
            summary.status = STATUS_FAILED
            summary.reason = "no option contract could be resolved for the configured strikes"
            for missing in universe.missing[:5]:
                self._note(summary, f"unresolved: {missing}")
            summary.finished_at = self._now()
            return summary
        if universe.missing:
            self._note(
                summary,
                f"{len(universe.missing)} configured contract(s) are not listed (see the log)",
            )

        metadata = metadata_for(dump.master, universe.tokens, symbols=universe.symbols)
        symbols_by_token = {
            token: meta.quote_key for token, meta in metadata.items() if meta.option_type
        }
        tracker = CoverageTracker(
            gap_seconds=self.capture_config.gaps.tick_gap_seconds,
            is_open=self.calendar.is_market_open,
        )
        self.writer = self._writer or CaptureWriter(
            db_path=self.config.db_path,
            parquet_root=self.config.raw_dir,
            storage_config=self.capture_config.storage,
            run_id=self.run_id,
            logger=self.logger,
        )
        heartbeat = self._heartbeat or Heartbeat(
            self.config.heartbeat_path(),
            interval_seconds=self.capture_config.heartbeat.interval_seconds,
            stale_after_seconds=self.capture_config.heartbeat.stale_after_seconds,
            state=HeartbeatState(
                run_id=self.run_id,
                mode=self.context.mode,
                session_date=summary.session_date.isoformat(),
                atm_strike=universe.atm_strike,
                expiries=[expiry.isoformat() for expiry in universe.expiries],
            ),
            logger=self.logger,
        )
        self._heartbeat = heartbeat
        heartbeat.publish(force=True, status="STARTING")

        if self.websocket_config.mode != "full":
            self._note(
                summary,
                f"websocket mode is {self.websocket_config.mode!r}: bid/ask and OI will be NULL",
            )

        summary.backfill = self.top_up_index(dump.master)
        if summary.backfill.get("failures"):
            self._note(
                summary,
                f"index top-up had {len(summary.backfill['failures'])} failure(s); "
                "index history may have a gap",
            )
        return self._start_stream_and_loop(
            summary,
            universe=universe,
            metadata=metadata,
            symbols_by_token=symbols_by_token,
            tracker=tracker,
            heartbeat=heartbeat,
            client=client,
            session_end=session_end,
            smoke_minutes=smoke_minutes,
            until=until,
        )


    def _start_stream_and_loop(
        self,
        summary: RunSummary,
        *,
        universe: OptionUniverse,
        metadata: dict[int, ContractMeta],
        symbols_by_token: dict[int, str],
        tracker: CoverageTracker,
        heartbeat: Heartbeat,
        client: Any,
        session_end: datetime,
        smoke_minutes: int | None,
        until: datetime | None,
    ) -> RunSummary:
        """Subscribe, start polling and run the monitor loop until the deadline."""
        if self._websocket_enabled:
            self.stream = TickStream(
                adapter_factory=self._adapter_factory or self._default_adapter_factory(),
                metadata=metadata,
                run_id=self.run_id,
                mode=self.websocket_config.mode,
                websocket_config=self.websocket_config,
                capture_config=self.capture_config,
                writer=self.writer,
                tracker=tracker,
                heartbeat=heartbeat,
                logger=self.logger,
                clock=self._clock,
                is_open=self.calendar.is_market_open,
                timezone_name=self.config.timezone_name,
                machine_tz=self._machine_tz,
                reference_token=universe.atm_anchor_token,
            )
            self.stream.start(universe.tokens)
            heartbeat.publish(force=True, status="RUNNING")
        elif self.logger is not None:
            self.logger.warning(
                "websocket disabled: this session only polls the option chain (no ticks)"
            )

        poller_stop = threading.Event()
        poller_thread = None
        if symbols_by_token:
            self.poller = OptionChainPoller(
                client=client,
                symbols_by_token=symbols_by_token,
                metadata=metadata,
                run_id=self.run_id,
                poll_config=self.config.data_sources.zerodha.option_chain,
                capture_config=self.capture_config,
                writer=self.writer,
                tracker=tracker,
                heartbeat=heartbeat,
                logger=self.logger,
                now_fn=self._clock,
                is_open=self.calendar.is_market_open,
                timezone_name=self.config.timezone_name,
                machine_tz=self._machine_tz,
            )
            # One synchronous poll before the loop: an OI baseline at the open is
            # worth more than the single request it costs.
            self.poller.poll_once(force=True)
            if self._poller_thread:
                poller_thread = self.poller.to_thread(poller_stop)
                poller_thread.start()

        deadline = self._session_deadline(session_end, smoke_minutes=smoke_minutes, until=until)
        try:
            self._run_loop(
                deadline=deadline, heartbeat=heartbeat, summary=summary, universe=universe
            )
        finally:
            poller_stop.set()
            if poller_thread is not None:
                poller_thread.join(timeout=10)
            self._shutdown(summary, heartbeat=heartbeat, tracker=tracker)
        return summary

    def _default_adapter_factory(self) -> Callable[[], Any]:
        """Build real ``KiteTicker`` adapters (credentials resolved lazily)."""
        from apps.anjaneeya_capture.websocket_listener import KiteTickerAdapter

        def factory() -> Any:
            from core.auth.zerodha_auth import resolve_access_token

            zerodha = self.config.data_sources.zerodha
            token = resolve_access_token(
                zerodha,
                request_token=self._request_token,
                timezone_name=self.config.timezone_name,
                validity_cutoff=zerodha.token_store.validity_cutoff,
                refresh_on_startup=zerodha.token_store.refresh_on_startup,
            )
            return KiteTickerAdapter(
                api_key=token.api_key,
                access_token=token.access_token,
                mode=self.websocket_config.mode,
                websocket_config=self.websocket_config,
                logger=self.logger,
            )

        return factory


    # -- the monitor loop ---------------------------------------------------
    #: Written by an operator (or the console) to stop a running session cleanly.
    STOP_REQUEST_FILENAME = "stop.request"

    @property
    def stop_request_path(self) -> Path:
        """Where a stop request is looked for (beside the heartbeat file)."""
        return self.config.heartbeat_path().with_name(self.STOP_REQUEST_FILENAME)

    def _stop_requested(self) -> bool:
        """True once, when a stop-request file exists (it is removed when honoured).

        The console runs a capture in a different process, so it cannot set
        ``stop_event`` directly; a file is the smallest thing both sides can agree on.
        Removing it on read means the next session starts from a clean slate.
        """
        path = self.stop_request_path
        if not path.exists():
            return False
        with contextlib.suppress(OSError):  # vanished between exists() and unlink()
            path.unlink()
        if self.logger is not None:
            self.logger.info("stop request accepted; finishing the session")
        return True

    def _run_loop(
        self,
        *,
        deadline: datetime,
        heartbeat: Heartbeat,
        summary: RunSummary,
        universe: OptionUniverse,
    ) -> None:
        """Monitor the stream until the deadline (or a stop request)."""
        refresh_minutes = self.capture_config.universe_refresh_minutes
        next_refresh = self._now() + timedelta(minutes=refresh_minutes) if refresh_minutes else None
        next_status = self._now() + timedelta(
            seconds=self.capture_config.status_interval_seconds
        )
        if self.logger is not None:
            self.logger.info(
                f"capturing until {deadline:%Y-%m-%d %H:%M:%S} "
                f"({(deadline - self._now()).total_seconds() / 60:.1f} minutes)"
            )
        while self._now() < deadline:
            if not self.stop_event.is_set() and self._stop_requested():
                self.stop_event.set()
            if self.stop_event.is_set():
                self._note(summary, "stop requested; shutting down cleanly")
                break
            if self.stream is not None:
                self.stream.monitor(now=self._now())
            heartbeat.publish()
            now = self._now()
            if now >= next_status:
                self._log_status(summary)
                next_status = now + timedelta(seconds=self.capture_config.status_interval_seconds)
            if next_refresh is not None and now >= next_refresh:
                self._refresh_universe(universe)
                next_refresh = now + timedelta(minutes=refresh_minutes)
            remaining = (deadline - self._now()).total_seconds()
            self._sleep(min(self._monitor_interval, max(remaining, 0.0)))

    def _log_status(self, summary: RunSummary) -> None:
        """One periodic status line (ticks, buffer, gaps, connection state)."""
        if self.logger is None:
            return
        stream = self.stream.describe() if self.stream is not None else {}
        self.logger.info(
            f"status: ticks={stream.get('ticks', 0)} buffered={self.writer.buffered_rows} "
            f"connected={stream.get('connected', False)} gaps={stream.get('gaps', 0)} "
            f"snapshots={self.writer.stats.snapshots} date={summary.session_date.isoformat()}"
        )

    def _refresh_universe(self, universe: OptionUniverse) -> None:
        """Re-centre the strike window on the ATM implied by the live anchor price.

        NIFTY moves: a window frozen at 09:15 drifts away from the money. The
        refresh is a *diff* - only genuinely new strikes are subscribed and only
        dead ones dropped - so it costs a couple of protocol messages rather than
        a resubscription of the whole universe.

        The price is the *ATM anchor's* - the index for ``atm.reference: spot``,
        the futures contract for ``futures`` (:attr:`OptionUniverse.atm_anchor`)
        - so a re-centre always uses the same reference the universe was built
        from.
        """
        if self.stream is None or self._master is None:
            return
        price = self.stream.last_reference_price
        if price is None:
            if self.logger is not None:
                self.logger.info(
                    "universe refresh skipped: no tick from the ATM anchor "
                    f"({universe.atm_anchor.tradingsymbol}) has arrived yet"
                )
            return
        refreshed = build_option_universe(
            universe_config=self.config.option_universe,
            master=self._master,
            reference_price=price,
            day=self.calendar.local_date(self._now()),
            calendar=self.calendar,
            resolved_at=self._now(),
        )
        if refreshed.atm_strike == universe.atm_strike:
            if self.logger is not None:
                self.logger.info(
                    f"universe refresh: ATM is still {refreshed.atm_strike:g}; no change"
                )
            return
        if refreshed.is_empty or len(refreshed.options) < len(universe.options):
            # Never *shrink* coverage mid-session: a stale or incomplete dump must
            # not be allowed to drop strikes we are already capturing.
            if self.logger is not None:
                self.logger.warning(
                    f"universe refresh skipped: the re-centred window at "
                    f"{refreshed.atm_strike:g} resolved {len(refreshed.options)} contract(s) "
                    f"vs {len(universe.options)} currently subscribed (missing: "
                    f"{len(refreshed.missing)})"
                )
            return
        tokens = refreshed.tokens
        new_metadata = metadata_for(self._master, tokens, symbols=refreshed.symbols)
        summary = self.stream.refresh_universe(
            tokens,
            metadata=new_metadata,
            reference_token=refreshed.atm_anchor_token,
        )
        if self.poller is not None:
            self.poller.update_universe(
                symbols_by_token={
                    token: meta.quote_key for token, meta in new_metadata.items() if meta.option_type
                },
                metadata=new_metadata,
            )
        if self.logger is not None:
            self.logger.info(
                f"universe refreshed after ATM moved {universe.atm_strike:g} -> "
                f"{refreshed.atm_strike:g}: +{len(summary['added'])}/-{len(summary['removed'])}"
            )


    # -- shutdown -----------------------------------------------------------
    def _shutdown(self, summary: RunSummary, *, heartbeat: Heartbeat, tracker: CoverageTracker) -> None:
        """Stop capturing, flush everything and write the coverage ledger.

        Order matters: stop the socket first (no new rows), then stop flushing
        into DuckDB, then *derive* coverage from what was actually recorded, then
        close the Parquet mirror.
        """
        if self.stream is not None:
            self.stream.stop()
        coverage_rows = tracker.coverage_rows(run_id=self.run_id, trading_date=summary.session_date)
        if coverage_rows and self.writer is not None:
            try:
                self.writer.write_coverage(coverage_rows)
            except Exception as exc:  # noqa: BLE001 - a ledger failure must not lose the data
                if self.logger is not None:
                    self.logger.error(f"could not write coverage rows: {type(exc).__name__}: {exc}")
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as exc:  # noqa: BLE001 - reported, never fatal at shutdown
                if self.logger is not None:
                    self.logger.error(f"error while closing the storage writer: {exc}")

        summary.coverage_rows = len(coverage_rows)
        verdicts: dict[str, int] = {}
        for row in coverage_rows:
            name = row.quality_flag.value if hasattr(row.quality_flag, "value") else str(
                row.quality_flag
            )
            verdicts[name] = verdicts.get(name, 0) + 1
        summary.coverage_verdicts = dict(sorted(verdicts.items()))
        summary.stream = self.stream.describe() if self.stream is not None else {}
        summary.writer = self.writer.describe() if self.writer is not None else {}
        summary.poller = self.poller.describe() if self.poller is not None else {}
        summary.finished_at = self._now()

        writer_stats = summary.writer or {}
        if writer_stats.get("dropped_rows"):
            summary.status = STATUS_PARTIAL
            summary.reason = summary.reason or (
                f"{writer_stats['dropped_rows']} row(s) could not be persisted"
            )
        if summary.status == STATUS_SUCCESS and summary.stream.get("stalls"):
            summary.status = STATUS_PARTIAL
            summary.reason = (
                f"{summary.stream['stalls']} websocket stall(s) were recovered by the watchdog"
            )
        if summary.status == STATUS_SUCCESS and summary.stream.get("restarts"):
            summary.reason = (
                f"{summary.stream['restarts']} websocket restart(s); data resumed after each"
            )
        if (
            summary.status == STATUS_SUCCESS
            and self._websocket_enabled
            and not summary.stream.get("ticks")
        ):
            summary.status = STATUS_PARTIAL
            summary.reason = "the session produced no ticks at all"

        if heartbeat is not None:
            heartbeat.update(
                status="STOPPED",
                connected=False,
                gap_count=self.stream.total_gaps if self.stream is not None else 0,
                largest_gap_seconds=(
                    self.stream.largest_gap_seconds if self.stream is not None else 0.0
                ),
            )
            heartbeat.publish(force=True, status="STOPPED")
        self.context.run_context.data_coverage = summary.describe()
        if self.logger is not None:
            self.logger.info(f"session finished: {summary.describe()}")

