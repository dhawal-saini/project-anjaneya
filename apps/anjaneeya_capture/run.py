"""ANJANEYA-CAPTURE - the data acquisition application.

Responsibilities (master spec sections 4-6)

* authenticate against Zerodha Kite Connect,
* subscribe to the option universe over the WebSocket (the live feed),
* append every tick to storage (ticks are never overwritten),
* poll the option chain periodically for OI/IV,
* backfill NIFTY index history from the REST API,
* report data coverage and quality.

Modes
-----
``status``        offline report: configuration, calendar, database, heartbeat.
``instruments``   download/cache Kite's instrument master (needs credentials).
``backfill``      NIFTY index history: full backfill + recent top-up (REST).
``backfill-options`` candles for the option contracts Kite currently lists (REST).
``capture``       one live session: WebSocket ticks + option-chain polling.
``option-chain``  polling only, no ticks (a second, independent OI source).

Every mode supports ``--dry-run``: it resolves and prints exactly what it would
do (universe, strikes, expiries, request budget) without calling the broker - the
instrument-dump cache makes that possible offline.

This app never places an order: ANJANEYA is a research and backtesting system.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from core.app_bootstrap import (
    APP_CAPTURE,
    EXIT_CONFIG_ERROR,
    EXIT_FAILED,
    EXIT_OK,
    AppContext,
    bootstrap_app,
    finalize_run,
    parse_args,
    render_status,
    status_report,
)
from core.config_loader import ConfigError
from core.market_calendar import DEFAULT_HOLIDAY_RISK_DAYS
from core.storage import SchemaVersionError

DESCRIPTION = (
    "ANJANEYA-CAPTURE: fetch and store raw NIFTY50 options market data "
    "(research only - this app never places orders)."
)

#: What each mode does. Phase numbers follow the delivered roadmap.
MODE_PLAN: dict[str, str] = {
    "capture": "WebSocket tick capture + option-chain polling for one session (Phase 2)",
    "option-chain": "periodic option-chain snapshots only, without ticks (Phase 2)",
    "backfill": "NIFTY index history backfill and daily top-up via REST (Phase 2)",
    "backfill-options": "candles for the option contracts Kite currently lists, via REST (Phase 2)",
    "instruments": "download/cache the daily Kite instrument master (Phase 2)",
    "status": "report configuration, calendar, database, heartbeat and coverage state",
}


def _configure_capture_parser(parser: argparse.ArgumentParser) -> None:
    """Capture-specific CLI flags (added on top of the shared parser)."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Capture for `capture.smoke_minutes` and then shut down cleanly.",
    )
    parser.add_argument(
        "--smoke-minutes",
        type=int,
        default=None,
        help="Override capture.smoke_minutes for this run.",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="HH:MM",
        help="Stop the session at this IST time today (e.g. 15:30).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download the instrument master even when the cache is fresh.",
    )
    parser.add_argument(
        "--request-token",
        default=None,
        help="Kite request_token to exchange for an access token (instead of the cached one).",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=None,
        metavar="N",
        help="backfill/backfill-options: window length in years (N*365 calendar days). "
        "Default: option_universe.index_history.backfill_days.",
    )


def _parse_until(value: str | None, *, context: AppContext) -> datetime | None:
    """Parse ``--until HH:MM`` into an IST datetime on the current date."""
    if not value:
        return None
    try:
        hour, minute = (int(part) for part in value.split(":", 1))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (ValueError, TypeError):
        raise ConfigError(
            f"--until {value!r} is not a valid IST time; expected HH:MM (e.g. 15:30)"
        ) from None
    now = datetime.now(context.calendar.timezone)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _install_stop_handler(context: AppContext, stop_event: threading.Event) -> None:
    """Turn Ctrl-C into a clean shutdown request (a second one forces exit)."""
    state = {"count": 0}

    def handler(signum: int, frame: object) -> None:  # pragma: no cover - signal path
        state["count"] += 1
        if state["count"] >= 2:
            print("\nANJANEYA-CAPTURE: forcing exit (second interrupt).", file=sys.stderr)
            raise SystemExit(EXIT_FAILED)
        context.logger.warning("interrupt received: finishing the current flush and shutting down")
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover - only the main thread may install handlers
        context.logger.debug("signal handlers can only be installed in the main thread")


def _run_status(context: AppContext, *, as_json: bool) -> int:
    """Print the offline status report, including the capture heartbeat."""
    from apps.anjaneeya_capture.heartbeat import heartbeat_age_seconds, read_heartbeat

    report = status_report(context)
    capture = context.config.data_sources.zerodha.capture
    path = context.config.heartbeat_path()
    payload = read_heartbeat(path)
    if payload is None:
        report["heartbeat"] = {"path": str(path), "present": False}
    else:
        age = heartbeat_age_seconds(payload)
        stale = age is None or age > capture.heartbeat.stale_after_seconds
        report["heartbeat"] = {
            "path": str(path),
            "present": True,
            "status": payload.get("status"),
            "age_seconds": round(age, 1) if age is not None else None,
            "stale": stale,
            "connected": payload.get("connected"),
            "ticks_total": payload.get("ticks_total"),
            "snapshots_total": payload.get("snapshots_total"),
            "run_id": payload.get("run_id"),
        }
    report["capture_policy"] = {
        "websocket_mode": context.config.data_sources.zerodha.websocket.mode,
        "out_of_session_ticks": capture.out_of_session_ticks,
        "tick_gap_seconds": capture.gaps.tick_gap_seconds,
        "flush_rows": capture.storage.flush_rows,
        "flush_seconds": capture.storage.flush_seconds,
        "mirror_parquet": capture.storage.mirror_parquet,
        "universe_refresh_minutes": capture.universe_refresh_minutes,
        "smoke_minutes": capture.smoke_minutes,
    }
    print(render_status(report, as_json=as_json))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------
def _load_master(
    context: AppContext, session: Any, *, force_refresh: bool, client: Any = None
) -> Any | None:
    """Load the instrument master, reporting the problem instead of raising.

    Returns ``None`` after printing an actionable message when the dump is
    unavailable (no cache and no credentials, or the cache is of an incompatible
    layout) - a CLI must never dump a traceback at an operator.
    """
    try:
        return session.load_master(client=client, force_refresh=force_refresh)
    except Exception as exc:  # noqa: BLE001 - normalised into an operator message
        print(f"ANJANEYA-CAPTURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _run_instruments(context: AppContext, args: argparse.Namespace) -> int:
    """Download (or read) the instrument master and store it in DuckDB."""
    from apps.anjaneeya_capture.session import CaptureSession

    session = CaptureSession(context, request_token=args.request_token)
    if context.dry_run:
        dump = _load_master(context, session, force_refresh=args.force_refresh)
        if dump is None:
            return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
        print("ANJANEYA-CAPTURE dry-run: instruments")
        print(f"  source        : {dump.source}")
        print(f"  fetched at    : {dump.fetched_at}")
        print(f"  instruments   : {len(dump.master)}")
        print(f"  per exchange  : {dump.describe()['rows']}")
        print(f"  cache dir     : {context.config.instrument_cache_dir}")
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    try:
        client = session.client()
    except Exception as exc:  # noqa: BLE001 - credentials are a configuration problem
        print(f"ANJANEYA-CAPTURE: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
    dump = _load_master(context, session, force_refresh=args.force_refresh, client=client)
    if dump is None:
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)

    stored = session.persist_instruments(dump)
    print("ANJANEYA-CAPTURE instrument master")
    print(f"  source        : {dump.source}")
    print(f"  fetched at    : {dump.fetched_at}")
    print(f"  instruments   : {len(dump.master)} ({dump.master.rejected_rows} unparseable row(s))")
    print(f"  stored rows   : {stored} (keyed by instrument_token)")
    print(f"  cache files   : {[str(path) for path in dump.paths.values()]}")
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------
def _history_days(args: argparse.Namespace, index_history: Any) -> int:
    """The history window in calendar days: ``--years`` when given, else the config."""
    if getattr(args, "years", None):
        return int(round(float(args.years) * 365))
    return int(index_history.backfill_days)


def _run_backfill(context: AppContext, args: argparse.Namespace) -> int:
    """Backfill and top up the NIFTY index history."""
    from apps.anjaneeya_capture.nifty_index_fetcher import IndexHistoryFetcher
    from apps.anjaneeya_capture.session import CaptureSession
    from apps.anjaneeya_capture.storage_writer import CaptureWriter

    index_history = context.config.option_universe.index_history
    capture_config = context.config.data_sources.zerodha.capture
    session = CaptureSession(context, request_token=args.request_token)
    days = _history_days(args, index_history)

    def now() -> datetime:
        return datetime.now(context.calendar.timezone)

    if context.dry_run:
        dump = _load_master(context, session, force_refresh=args.force_refresh)
        if dump is None:
            return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
        targets = session.index_targets(dump.master)
        planner = IndexHistoryFetcher(
            client=None,
            historical_config=capture_config.historical,
            run_id=context.run_context.run_id,
            now_fn=now,
            timezone_name=context.config.timezone_name,
        )
        plan = planner.plan(
            targets, days=days, interval=index_history.backfill_interval
        )
        print("ANJANEYA-CAPTURE dry-run: backfill")
        print(f"  symbols       : {[target.tradingsymbol for target in targets]}")
        print(f"  interval      : {index_history.backfill_interval}")
        print(f"  window        : {days} calendar days")
        print(f"  requests      : {len(plan)}")
        for line in plan:
            print(f"    - {line}")
        print(
            f"  top-up        : {index_history.topup_days} day(s) of "
            f"{index_history.topup_interval} afterwards"
        )
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    try:
        client = session.client()
    except Exception as exc:  # noqa: BLE001 - credentials are a configuration problem
        print(f"ANJANEYA-CAPTURE: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
    dump = _load_master(context, session, force_refresh=args.force_refresh, client=client)
    if dump is None:
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
    targets = session.index_targets(dump.master)

    fetcher = IndexHistoryFetcher(
        client=client,
        historical_config=capture_config.historical,
        run_id=context.run_context.run_id,
        logger=context.logger,
        now_fn=now,
        timezone_name=context.config.timezone_name,
    )
    writer = CaptureWriter(
        db_path=context.config.db_path,
        parquet_root=context.config.raw_dir,
        storage_config=capture_config.storage,
        run_id=context.run_context.run_id,
        logger=context.logger,
    )
    try:
        result = fetcher.backfill(
            targets, days=days, interval=index_history.backfill_interval
        )
        written = writer.write_index_history(result.rows)
        top_up = fetcher.top_up(
            targets, days=index_history.topup_days, interval=index_history.topup_interval
        )
        written += writer.write_index_history(top_up.rows)
    finally:
        writer.close(compact=False)

    failures = result.failures + top_up.failures
    print("ANJANEYA-CAPTURE index history")
    print(f"  symbols       : {[target.tradingsymbol for target in targets]}")
    print(f"  fetched rows  : {len(result.rows)} (+{len(top_up.rows)} from the top-up)")
    print(f"  stored rows   : {written} (idempotent upsert on (timestamp, symbol, timeframe))")
    print(f"  requests      : {result.requests + top_up.requests}")
    print(f"  window splits : {result.splits + top_up.splits}")
    if failures:
        print(f"  FAILURES      : {len(failures)}")
        for failure in failures[:10]:
            print(f"    - {failure}")
    return finalize_run(
        context,
        status="SUCCESS" if not failures else "PARTIAL",
        exit_code=EXIT_OK if not failures else EXIT_FAILED,
    )


# ---------------------------------------------------------------------------
# backfill-options
# ---------------------------------------------------------------------------
def _candle_from_index_row(row: Any, instrument: Any) -> Any:
    """Turn a fetched REST bar into an option :class:`CandleRow`.

    The fetcher is generic (it serves the index, futures and active options), so it
    returns :class:`IndexHistoryRow`. An option candle additionally carries its
    contract identity - expiry, strike and option_type *together*, which
    :class:`CandleRow` insists on - taken from the resolved instrument.
    """
    from core.schemas import CandleRow

    return CandleRow(
        timestamp=row.timestamp,
        timeframe=row.timeframe,
        instrument_token=instrument.instrument_token,
        tradingsymbol=row.tradingsymbol,
        expiry=instrument.expiry,
        strike=instrument.strike,
        option_type=instrument.option_type,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        oi=row.oi,
        source=row.source,
        fetch_timestamp=row.fetch_timestamp,
        quality_flag=row.quality_flag,
        run_id=row.run_id,
    )


def _run_backfill_options(context: AppContext, args: argparse.Namespace) -> int:
    """Bootstrap candles for the option contracts Kite currently lists.

    Hard provider constraint: ``client.historical`` serves the index, NIFTY futures
    and **active** options, and *never expired* ones. So this mode cannot reconstruct
    a past option dataset - it can only seed the contracts listed *today*, which the
    live capture then keeps current. Bars are written to the keyed ``candles`` table
    with ``source=zerodha_rest`` so a later tick-built candle for the same minute
    simply overwrites it. A universe that resolves to zero contracts is a hard error.
    """
    from apps.anjaneeya_capture.instruments import build_option_universe
    from apps.anjaneeya_capture.nifty_index_fetcher import FetchTarget, IndexHistoryFetcher
    from apps.anjaneeya_capture.session import CaptureSession
    from apps.anjaneeya_capture.storage_writer import CaptureWriter

    index_history = context.config.option_universe.index_history
    capture_config = context.config.data_sources.zerodha.capture
    session = CaptureSession(context, request_token=args.request_token)
    interval = index_history.topup_interval
    days = _history_days(args, index_history)

    def now() -> datetime:
        return datetime.now(context.calendar.timezone)

    def resolve_universe(master: Any) -> Any:
        price, _source = session.reference_price(master, now=now())
        return build_option_universe(
            universe_config=context.config.option_universe,
            master=master,
            reference_price=price,
            day=context.calendar.local_date(now()),
            calendar=context.calendar,
            resolved_at=now(),
            logger=None,
        )

    if context.dry_run:
        dump = _load_master(context, session, force_refresh=args.force_refresh)
        if dump is None:
            return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
        universe = resolve_universe(dump.master)
        targets = [FetchTarget.from_instrument(item) for item in universe.options]
        planner = IndexHistoryFetcher(
            client=None,
            historical_config=capture_config.historical,
            run_id=context.run_context.run_id,
            now_fn=now,
            timezone_name=context.config.timezone_name,
        )
        plan = planner.plan(targets, days=days, interval=interval)
        print("ANJANEYA-CAPTURE dry-run: backfill-options")
        print(f"  contracts     : {len(targets)} resolved ({len(universe.missing)} missing)")
        print(f"  interval      : {interval}")
        print(f"  window        : {days} calendar days")
        print(f"  requests      : {len(plan)}")
        for line in plan[:10]:
            print(f"    - {line}")
        if len(plan) > 10:
            print(f"    ... and {len(plan) - 10} more")
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    try:
        client = session.client()
    except Exception as exc:  # noqa: BLE001 - credentials are a configuration problem
        print(f"ANJANEYA-CAPTURE: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
    dump = _load_master(context, session, force_refresh=args.force_refresh, client=client)
    if dump is None:
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)

    universe = resolve_universe(dump.master)
    by_symbol = {item.tradingsymbol: item for item in universe.options}
    if not by_symbol:
        print(
            "ANJANEYA-CAPTURE: the option universe resolved to 0 contract(s); nothing to "
            "backfill (refresh the instrument master or reconcile the expiry weekday).",
            file=sys.stderr,
        )
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)

    targets = [FetchTarget.from_instrument(item) for item in universe.options]
    fetcher = IndexHistoryFetcher(
        client=client,
        historical_config=capture_config.historical,
        run_id=context.run_context.run_id,
        logger=context.logger,
        now_fn=now,
        timezone_name=context.config.timezone_name,
    )
    writer = CaptureWriter(
        db_path=context.config.db_path,
        parquet_root=context.config.raw_dir,
        storage_config=capture_config.storage,
        run_id=context.run_context.run_id,
        logger=context.logger,
    )
    try:
        result = fetcher.backfill(targets, days=days, interval=interval)
        rows = [
            _candle_from_index_row(row, by_symbol[row.tradingsymbol])
            for row in result.rows
            if row.tradingsymbol in by_symbol
        ]
        written = writer.write_candles(rows)
    finally:
        writer.close(compact=False)

    failures = result.failures
    print("ANJANEYA-CAPTURE option history")
    print(f"  contracts     : {len(targets)} ({len(universe.missing)} missing)")
    print(f"  interval      : {interval} over {days} calendar day(s)")
    print(f"  fetched rows  : {len(result.rows)}")
    print(f"  stored rows   : {written} (idempotent upsert on (timestamp, tradingsymbol, timeframe))")
    print(f"  requests      : {result.requests}")
    if failures:
        print(f"  FAILURES      : {len(failures)}")
        for failure in failures[:10]:
            print(f"    - {failure}")
    return finalize_run(
        context,
        status="SUCCESS" if not failures else "PARTIAL",
        exit_code=EXIT_OK if not failures else EXIT_FAILED,
    )


# ---------------------------------------------------------------------------
# capture / option-chain
# ---------------------------------------------------------------------------
def calendar_advisory(context: AppContext) -> list[str]:
    """Lines naming any provisional holiday close enough to cost this session.

    The calendar decides whether a session runs at all, so a provisional holiday that is
    really a trading day turns the run into ``NO_SESSION`` - and the ticks of that session
    would not exist anywhere afterwards. This is advisory, never a refusal: the operator may
    know something the YAML does not.
    """
    today = context.calendar.now().date()
    upcoming = context.calendar.unverified_holiday_risk(today=today)
    if not upcoming:
        return []
    lines = [
        f"{len(upcoming)} holiday date(s) within {DEFAULT_HOLIDAY_RISK_DAYS} day(s) are still "
        "PROVISIONAL in config/calendar.yaml:"
    ]
    for entry in upcoming:
        days = (entry.date - today).days
        when = "today" if days == 0 else f"in {days} day(s)"
        lines.append(f"  {entry.date.isoformat()}  {entry.name}  ({when})")
    lines.append(
        "confirm them against the NSE circular (python scripts/check_calendar.py): a provisional "
        "date that is really a trading day makes this session exit NO_SESSION and the ticks of "
        "that day cannot be recovered."
    )
    return lines


def _run_session(context: AppContext, args: argparse.Namespace) -> int:
    """Run a live capture (or option-chain-only) session."""
    from apps.anjaneeya_capture.session import (
        STATUS_NO_SESSION,
        STATUS_SUCCESS,
        CaptureSession,
        SessionError,
    )

    smoke_minutes = args.smoke_minutes
    if smoke_minutes is None and args.smoke:
        smoke_minutes = context.config.data_sources.zerodha.capture.smoke_minutes
    until = _parse_until(args.until, context=context)
    mode = context.mode
    websocket_enabled = mode == "capture"

    for line in calendar_advisory(context):
        print(f"ANJANEYA-CAPTURE: {line}")
        context.logger.warning(line)

    session = CaptureSession(
        context,
        websocket_enabled=websocket_enabled,
        request_token=args.request_token,
    )
    if context.dry_run:
        try:
            plan = session.plan(force_refresh=args.force_refresh)
        except Exception as exc:  # noqa: BLE001 - a dry run must still explain itself
            print(
                f"ANJANEYA-CAPTURE: {mode} could not be planned: {type(exc).__name__}: {exc}\n"
                "  hint: run `--mode instruments` once (with credentials) to populate the "
                "instrument-dump cache, then dry-runs work offline.",
                file=sys.stderr,
            )
            return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
        print(f"ANJANEYA-CAPTURE dry-run: {mode}")
        print(plan.render())
        print(f"  websocket     : {'enabled' if websocket_enabled else 'disabled (polls only)'}")
        if smoke_minutes:
            print(f"  smoke budget  : {smoke_minutes} minute(s)")
        if until:
            print(f"  stop at       : {until:%H:%M} IST")
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    stop_event = threading.Event()
    _install_stop_handler(context, stop_event)
    session.stop_event = stop_event
    try:
        summary = session.run(
            smoke_minutes=smoke_minutes, until=until, force_refresh=args.force_refresh
        )
    except SessionError as exc:
        print(f"ANJANEYA-CAPTURE: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the CLI
        context.logger.exception(f"capture session failed: {exc}")
        print(f"ANJANEYA-CAPTURE: the session failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_FAILED)

    print(summary.render())
    # A session that never happened (holiday, after the close) is not an error;
    # a partial session is: it means data is missing and must be noticed.
    if summary.status in {STATUS_SUCCESS, STATUS_NO_SESSION}:
        return finalize_run(context, status=summary.status, exit_code=EXIT_OK)
    return finalize_run(context, status=summary.status, exit_code=EXIT_FAILED)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 1 failed, 2 config)."""
    args = parse_args(
        argv,
        app_name=APP_CAPTURE,
        description=DESCRIPTION,
        configure_parser=_configure_capture_parser,
    )
    try:
        context = bootstrap_app(APP_CAPTURE, args)
    except (ConfigError, SchemaVersionError) as exc:
        print(f"ANJANEYA-CAPTURE configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the CLI
        print(f"ANJANEYA-CAPTURE failed during startup: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    try:
        if context.mode == "status":
            return _run_status(context, as_json=args.as_json)
        if context.mode == "instruments":
            return _run_instruments(context, args)
        if context.mode == "backfill":
            return _run_backfill(context, args)
        if context.mode == "backfill-options":
            return _run_backfill_options(context, args)
        if context.mode in {"capture", "option-chain"}:
            return _run_session(context, args)
    except ConfigError as exc:
        print(f"ANJANEYA-CAPTURE configuration error: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_CONFIG_ERROR)
    except KeyboardInterrupt:  # pragma: no cover - signal path
        print("ANJANEYA-CAPTURE interrupted.", file=sys.stderr)
        return finalize_run(context, status="ABORTED", exit_code=EXIT_FAILED)
    print(f"ANJANEYA-CAPTURE: unknown mode {context.mode!r}", file=sys.stderr)
    return finalize_run(context, status="FAILED", exit_code=EXIT_FAILED)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

