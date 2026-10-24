"""End-to-end capture pipeline test.

This runs a *complete* ANJANEYA-CAPTURE session against a fake provider: fake
instrument dumps, a fake ``KiteTicker`` and a fake REST client. Nothing on our
side is stubbed - the real session, listener, poller, writer, coverage tracker and
heartbeat all run, and the assertions are made against what landed on disk:

    DuckDB rows  ->  ticks / option_chain_snapshots / instruments / data_coverage
    Parquet      ->  one compact file per stream per day (plus no leftover parts)
    heartbeat    ->  a STOPPED payload that names the run
    manifest     ->  the run context carrying the same coverage numbers

The scenario deliberately includes a **websocket reconnect in the middle of the
session**: no tick may be lost across it (spec 4.4), and the coverage ledger must
still account for every row.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.heartbeat import read_heartbeat
from apps.anjaneeya_capture.instruments import save_dump_file
from apps.anjaneeya_capture.session import STATUS_SUCCESS, CaptureSession
from core.config_loader import load_config
from core.storage import connect
from tests.support.capture import (
    DAY,
    NIFTY_SPOT_TOKEN,
    WEEK_EXPIRY,
    FakeClock,
    FakeKite,
    FakeTicker,
    RecordingLogger,
    make_app_context,
    make_dump,
    make_quote,
    make_tick,
    option_token,
)

IST = ZoneInfo("Asia/Kolkata")
OPEN = datetime(2026, 9, 29, 9, 16, 0, tzinfo=IST)
CE_TOKEN = option_token(25000, WEEK_EXPIRY, "CE")
PE_TOKEN = option_token(25000, WEEK_EXPIRY, "PE")
QUOTE_KEYS = {"NFO:NIFTY29SEP2625000CE": CE_TOKEN, "NFO:NIFTY29SEP2625000PE": PE_TOKEN}


@pytest.fixture
def config(isolated_paths):
    return load_config()


@pytest.fixture
def ticker() -> FakeTicker:
    return FakeTicker()


@pytest.fixture
def client() -> FakeKite:
    return FakeKite(
        quotes={
            symbol: make_quote(token=token, last_price=120.0, exchange_timestamp=OPEN)
            for symbol, token in QUOTE_KEYS.items()
        },
        ltp={"NSE:NIFTY 50": {"instrument_token": NIFTY_SPOT_TOKEN, "last_price": 25012.4}},
    )


def seed_cache(config) -> None:
    """Write the synthetic instrument dumps the session resolves against."""
    for exchange, rows in make_dump().items():
        save_dump_file(
            rows,
            cache_dir=config.instrument_cache_dir,
            exchange=exchange,
            day=DAY,
            fetched_at=OPEN,
        )


def adapter_factory(config, ticker: FakeTicker):
    """A real adapter around a fake socket (so subscribe/set_mode are exercised)."""
    from apps.anjaneeya_capture.websocket_listener import KiteTickerAdapter

    def factory():
        ticker.connected = True
        return KiteTickerAdapter(
            api_key="k",
            access_token="t",
            mode=config.data_sources.zerodha.websocket.mode,
            websocket_config=config.data_sources.zerodha.websocket,
            ticker=ticker,
        )

    return factory


def _emit_one_tick(ticker: FakeTicker, clock: FakeClock):
    """A per-sleep hook that delivers exactly one option tick."""

    def emit(seconds: float) -> None:
        ticker.simulate_ticks(
            [make_tick(token=CE_TOKEN, last_price=100.0, exchange_timestamp=clock.now())]
        )

    return emit


def make_session(config, client, ticker, clock, logger=None) -> CaptureSession:
    """Build a session wired to the fakes, with the fake clock driving the loop."""
    context = make_app_context(config, logger=logger)
    return CaptureSession(
        context,
        client=client,
        adapter_factory=adapter_factory(config, ticker),
        poller_thread=False,
        clock=clock.now,
        sleep=clock.sleep,
        machine_tz=IST,
    )


def test_a_whole_session_lands_on_disk(config, client, ticker) -> None:
    seed_cache(config)
    logger = RecordingLogger()
    clock = FakeClock(OPEN)
    state = {"ticks": 0}

    def emit(seconds: float) -> None:
        """One tick message per loop pass, with a mid-session reconnect."""
        state["ticks"] += 1
        if state["ticks"] == 3:
            ticker.simulate_reconnect()
        ticker.simulate_ticks(
            [
                make_tick(
                    token=CE_TOKEN,
                    last_price=100.0 + state["ticks"],
                    exchange_timestamp=clock.now(),
                    oi=5000 + state["ticks"],
                ),
                make_tick(
                    token=NIFTY_SPOT_TOKEN,
                    last_price=25012.0 + state["ticks"],
                    tradable=False,
                    exchange_timestamp=clock.now(),
                ),
            ]
        )

    clock.on_sleep.append(emit)
    session = make_session(config, client, ticker, clock, logger=logger)

    summary = session.run(smoke_minutes=5)

    # -- the session summary -------------------------------------------------
    assert summary.status == STATUS_SUCCESS, summary.describe()
    assert summary.stream["ticks"] == summary.writer["ticks"] == state["ticks"] * 2
    assert summary.stream["reconnects"] == 1  # the SDK reconnected once
    assert summary.poller["rows"] == 2  # two option contracts, one snapshot each

    # -- DuckDB --------------------------------------------------------------
    connection = connect(config.db_path, read_only=True)
    try:
        ticks = connection.execute("SELECT count(*) FROM ticks").fetchone()[0]
        snapshots = connection.execute("SELECT count(*) FROM option_chain_snapshots").fetchone()[0]
        instruments = connection.execute("SELECT count(*) FROM instruments").fetchone()[0]
        distinct_symbols = connection.execute(
            "SELECT count(DISTINCT tradingsymbol) FROM ticks"
        ).fetchone()[0]
        cursor = connection.execute("SELECT timeframe, coverage_ratio FROM data_coverage")
        coverage = cursor.fetchall()
    finally:
        connection.close()

    assert ticks == summary.stream["ticks"]
    assert snapshots == 2
    assert instruments == 22
    # Both the option and the index stream were captured (the reconnect between
    # them lost nothing).
    assert distinct_symbols == 2
    assert {row[0] for row in coverage} == {"TICK", "SNAPSHOT"}
    assert all(row[1] is None for row in coverage)  # tick streams have no row expectation

    # -- Parquet mirror ------------------------------------------------------
    parquet_root = config.raw_dir
    tick_files = list((parquet_root / "ticks").rglob("*.parquet"))
    snapshot_files = list((parquet_root / "option_chain_snapshots").rglob("*.parquet"))
    assert [path.name for path in tick_files] == ["2026-09-29.parquet"]
    assert [path.name for path in snapshot_files] == ["2026-09-29.parquet"]
    # Part files are compacted away: one file per stream per day.
    assert list(parquet_root.rglob("part-*.parquet")) == []

    import duckdb

    mirror = duckdb.connect()
    try:
        mirrored = mirror.execute(
            f"SELECT count(*) FROM read_parquet('{tick_files[0].as_posix()}')"
        ).fetchone()[0]
    finally:
        mirror.close()
    assert mirrored == ticks

    # -- heartbeat -----------------------------------------------------------
    payload = read_heartbeat(config.heartbeat_path())
    assert payload is not None
    assert payload["status"] == "STOPPED"
    assert payload["session_date"] == DAY.isoformat()
    assert payload["ticks_total"] == ticks
    assert payload["poll_count"] >= 1

    # -- run manifest data coverage -----------------------------------------
    assert session.context.run_context.data_coverage["status"] == STATUS_SUCCESS
    assert session.context.run_context.data_coverage["stream"]["ticks"] == ticks
    assert logger.contains("instrument master ready")


def test_the_instrument_master_never_duplicates_across_runs(config, client, ticker) -> None:
    """Keyed tables are idempotent; raw streams are append-only."""
    seed_cache(config)
    tick_counts: list[int] = []
    for _ in range(2):
        clock = FakeClock(OPEN)
        clock.on_sleep.append(_emit_one_tick(ticker, clock))
        summary = make_session(config, client, ticker, clock).run(smoke_minutes=2)
        tick_counts.append(summary.stream["ticks"])

    connection = connect(config.db_path, read_only=True)
    try:
        ticks = connection.execute("SELECT count(*) FROM ticks").fetchone()[0]
        instruments = connection.execute("SELECT count(*) FROM instruments").fetchone()[0]
    finally:
        connection.close()

    # Ticks accumulate (they are the immutable record); the master does not.
    assert ticks == sum(tick_counts)
    assert instruments == 22

