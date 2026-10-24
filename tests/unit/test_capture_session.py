"""Tests for a whole capture session (``apps.anjaneeya_capture.session``)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.session import (
    STATUS_NO_SESSION,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
    CaptureSession,
    SessionError,
    SessionPlan,
)
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
    capture_config,
    make_app_context,
    make_dump,
    make_quote,
    make_tick,
    option_token,
)
from tests.support.engine import make_index_row, seed_index_history

IST = ZoneInfo("Asia/Kolkata")
OPEN = datetime(2026, 9, 29, 9, 16, 0, tzinfo=IST)
CE_TOKEN = option_token(25000, WEEK_EXPIRY, "CE")
PE_TOKEN = option_token(25000, WEEK_EXPIRY, "PE")


@pytest.fixture
def config(isolated_paths):
    return load_config()


def seed_cache(config, *, dump=None) -> None:
    """Write a synthetic instrument dump into the configured cache directory."""
    from apps.anjaneeya_capture.instruments import save_dump_file

    frames = dump if dump is not None else make_dump()
    for exchange, rows in frames.items():
        save_dump_file(
            rows,
            cache_dir=config.instrument_cache_dir,
            exchange=exchange,
            day=DAY,
            fetched_at=OPEN,
        )


def make_session(config, *, clock=None, poller_thread=False, **kwargs) -> tuple[CaptureSession, RecordingLogger]:
    logger = RecordingLogger()
    context = make_app_context(config, logger=logger, mode=kwargs.pop("mode", "capture"))
    session = CaptureSession(
        context,
        poller_thread=poller_thread,
        clock=(clock.now if clock is not None else (lambda: OPEN)),
        sleep=(clock.sleep if clock is not None else (lambda seconds: None)),
        machine_tz=IST,
        **kwargs,
    )
    return session, logger


def live_client(*, quotes: dict | None = None) -> FakeKite:
    """A fake Kite client with a live index price and option quotes."""
    return FakeKite(
        quotes=quotes
        or {
            f"NFO:{'NIFTY29SEP2625000CE'}": make_quote(
                token=CE_TOKEN, last_price=120.0, exchange_timestamp=OPEN
            ),
            f"NFO:{'NIFTY29SEP2625000PE'}": make_quote(
                token=PE_TOKEN, last_price=95.0, exchange_timestamp=OPEN
            ),
        },
        ltp={"NSE:NIFTY 50": {"instrument_token": NIFTY_SPOT_TOKEN, "last_price": 25012.4}},
    )


def run_ticks(ticker: FakeTicker, clock: FakeClock) -> None:
    """Deliver one tick per sleep, so a session loop actually captures data."""

    def on_sleep(seconds: float) -> None:
        ticker.simulate_ticks(
            [
                make_tick(
                    token=CE_TOKEN,
                    last_price=120.0 + len(clock.sleeps),
                    exchange_timestamp=clock.now(),
                ),
                make_tick(
                    token=NIFTY_SPOT_TOKEN,
                    last_price=25012.0 + len(clock.sleeps),
                    tradable=False,
                    exchange_timestamp=clock.now(),
                ),
            ]
        )

    clock.on_sleep.append(on_sleep)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def test_plan_resolves_the_universe_offline(config) -> None:
    seed_cache(config)
    session, _ = make_session(config)
    plan = session.plan()
    assert isinstance(plan, SessionPlan)
    assert plan.is_trading_day is True
    assert plan.session_start is not None and plan.session_end is not None
    assert plan.reference_price == 25012.0
    assert "instrument-dump" in plan.reference_price_source
    assert plan.universe is not None and plan.universe.atm_strike == 25000.0
    assert plan.subscription_batches == 1
    assert plan.poll_requests_per_cycle == 1
    assert len(plan.backfill_requests) >= 1
    assert "session date" in plan.render()


def test_plan_reports_a_non_trading_day(config) -> None:
    seed_cache(config)
    # Sunday 2026-10-04 is never a trading day.
    sunday = datetime(2026, 10, 4, 9, 16, tzinfo=IST)
    session, _ = make_session(config, clock=FakeClock(sunday))
    plan = session.plan()
    assert plan.is_trading_day is False
    assert any("not a trading day" in note for note in plan.notes)


def test_plan_uses_the_live_price_when_a_client_is_available(config) -> None:
    seed_cache(config)
    session, _ = make_session(config)
    plan = session.plan(client=live_client())
    assert plan.reference_price_source.startswith("rest-ltp")


def test_plan_falls_back_when_the_live_quote_fails(config) -> None:
    class BrokenLtp(FakeKite):
        def ltp(self, *symbols):
            raise RuntimeError("429 too many requests")

    seed_cache(config)
    session, logger = make_session(config)
    plan = session.plan(client=BrokenLtp(dump=make_dump()))
    assert plan.reference_price == 25012.0
    assert logger.contains("falling back", "WARNING")


def test_a_pre_open_plan_anchors_on_the_stored_index_close(config, isolated_paths) -> None:
    """Kite reports ``last_price: 0`` for index rows, so before the open the plan
    has no quote at all - the backfilled NIFTY history is the anchor."""
    seed_cache(config, dump=make_dump(spot_price=0.0))
    seed_index_history(isolated_paths["db_path"], [make_index_row(close=24995.0)])
    session, _ = make_session(config)
    plan = session.plan()
    assert plan.reference_price == 24995.0
    assert plan.reference_price_source.startswith("index-history")
    assert plan.universe is not None and plan.universe.atm_strike == 25000.0


def test_a_plan_without_any_price_still_fails_loudly(config) -> None:
    seed_cache(config, dump=make_dump(spot_price=0.0))
    session, _ = make_session(config)
    with pytest.raises(SessionError, match="no reference price available"):
        session.plan()


def test_a_plan_never_anchors_on_a_future_index_bar(config, isolated_paths) -> None:
    """Only rows at or before the plan's moment count: a session plans on what is
    already stored, never on a bar that has not happened yet."""
    seed_cache(config, dump=make_dump(spot_price=0.0))
    seed_index_history(
        isolated_paths["db_path"],
        [
            make_index_row(close=24995.0),
            make_index_row(
                timestamp=OPEN + timedelta(days=1),
                open=26000.0,
                high=26010.0,
                low=25990.0,
                close=26005.0,
            ),
        ],
    )
    session, _ = make_session(config)
    plan = session.plan()
    assert plan.reference_price == 24995.0


# ---------------------------------------------------------------------------
# Session window
# ---------------------------------------------------------------------------
def test_a_holiday_session_stops_immediately(config) -> None:
    seed_cache(config)
    sunday = datetime(2026, 10, 4, 9, 16, tzinfo=IST)
    session, _ = make_session(config, clock=FakeClock(sunday), client=live_client())
    summary = session.run()
    assert summary.status == STATUS_NO_SESSION
    assert "not an NSE trading day" in summary.reason
    assert session.stream is None  # nothing was ever subscribed


def test_a_session_started_after_the_close_does_not_wait(config) -> None:
    seed_cache(config)
    late = datetime(2026, 9, 29, 16, 30, tzinfo=IST)
    session, _ = make_session(config, clock=FakeClock(late), client=live_client())
    summary = session.run()
    assert summary.status == STATUS_NO_SESSION
    assert "closed" in summary.reason


def test_a_session_started_too_early_is_refused(config) -> None:
    """The guard stops an accidental all-morning wait (not a silent one)."""
    seed_cache(config)
    early = datetime(2026, 9, 29, 8, 0, tzinfo=IST)  # 75 minutes before the open
    session, logger = make_session(config, clock=FakeClock(early), client=live_client())
    summary = session.run()
    assert summary.status == STATUS_NO_SESSION
    assert "waiting for the session open" in summary.reason
    assert logger.contains("guard is 15 minutes", "WARNING")


def test_a_session_just_before_the_open_waits_for_it(config) -> None:
    seed_cache(config)
    clock = FakeClock(datetime(2026, 9, 29, 9, 6, tzinfo=IST))
    session, _ = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _noop_adapter(config)
    summary = session.run(smoke_minutes=1)
    assert summary.status == STATUS_PARTIAL  # no ticks: the fake ticker never emits
    assert clock.sleeps  # it actually waited
    assert session.stream is not None
    assert any("waiting for the session open" in message for message in session.stream.logger.messages("INFO"))


def test_a_stop_request_during_the_wait_abandons_the_session(config) -> None:
    import threading

    seed_cache(config)
    stop = threading.Event()
    stop.set()
    clock = FakeClock(datetime(2026, 9, 29, 9, 6, tzinfo=IST))
    session, _ = make_session(
        config, clock=clock, client=live_client(), stop_event=stop
    )
    summary = session.run()
    assert summary.status == STATUS_NO_SESSION
    assert "abandoned" in summary.reason


# ---------------------------------------------------------------------------
# A full session
# ---------------------------------------------------------------------------
def _noop_adapter(config):
    """A real `KiteTickerAdapter` around a fake socket that never ticks."""
    from apps.anjaneeya_capture.websocket_listener import KiteTickerAdapter

    ticker = FakeTicker()
    ticker.connected = True
    return KiteTickerAdapter(
        api_key="k",
        access_token="t",
        mode=config.data_sources.zerodha.websocket.mode,
        websocket_config=config.data_sources.zerodha.websocket,
        ticker=ticker,
    )


def test_a_full_session_captures_ticks_snapshots_and_coverage(config) -> None:
    seed_cache(config)
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, logger = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)
    run_ticks(ticker, clock)

    summary = session.run(smoke_minutes=3)

    assert summary.status == STATUS_SUCCESS, summary.describe()
    assert summary.stream["ticks"] == len(clock.sleeps) * 2
    assert summary.poller["polls"] >= 1
    assert summary.writer["ticks"] == summary.stream["ticks"]
    assert summary.coverage_rows > 0
    assert summary.universe is not None and summary.universe["atm_strike"] == 25000.0
    # The index top-up is attempted (one request for the last 10 days of minutes)
    # but the fake provider returns no candles.
    assert summary.backfill["requests"] == 1
    assert summary.backfill["rows"] == 0
    assert summary.backfill["failures"] == []
    assert "session result: SUCCESS" in summary.render()

    # The data really landed in DuckDB, not just in the summary.
    connection = connect(config.db_path, read_only=True)
    try:
        ticks = connection.execute("SELECT count(*) FROM ticks").fetchone()[0]
        snapshots = connection.execute(
            "SELECT count(*) FROM option_chain_snapshots"
        ).fetchone()[0]
        coverage = connection.execute("SELECT count(*) FROM data_coverage").fetchone()[0]
        instruments = connection.execute("SELECT count(*) FROM instruments").fetchone()[0]
    finally:
        connection.close()
    assert ticks == summary.stream["ticks"]
    assert snapshots == summary.poller["rows"] == 2
    assert coverage == summary.coverage_rows
    assert instruments == 22
    assert logger.contains("instrument master ready")


def _adapter_for(config, ticker: FakeTicker):
    from apps.anjaneeya_capture.websocket_listener import KiteTickerAdapter

    ticker.connected = True
    return KiteTickerAdapter(
        api_key="k",
        access_token="t",
        mode=config.data_sources.zerodha.websocket.mode,
        websocket_config=config.data_sources.zerodha.websocket,
        ticker=ticker,
    )


def test_a_session_writes_snapshot_rows_for_every_option(config) -> None:
    seed_cache(config)
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, _ = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)

    summary = session.run(smoke_minutes=1)

    assert summary.poller["rows"] == 2  # one poll of the two 25000-strike contracts
    assert summary.coverage_verdicts.get("OK") == 2


def test_the_session_writes_a_heartbeat_file(config) -> None:
    seed_cache(config)
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, _ = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)

    session.run(smoke_minutes=1, until=OPEN + timedelta(minutes=1))

    from apps.anjaneeya_capture.heartbeat import heartbeat_age_seconds, read_heartbeat

    payload = read_heartbeat(config.heartbeat_path())
    assert payload is not None
    assert payload["status"] == "STOPPED"
    assert payload["connected"] is False
    assert payload["run_id"] == session.run_id
    assert heartbeat_age_seconds(payload) is not None


def test_option_chain_only_mode_never_subscribes(config) -> None:
    seed_cache(config)
    clock = FakeClock(OPEN)
    session, logger = make_session(config, clock=clock, client=live_client())
    session._websocket_enabled = False

    summary = session.run(smoke_minutes=1)

    assert session.stream is None
    assert summary.status == STATUS_SUCCESS  # no ticks expected in this mode
    assert summary.stream == {}
    assert summary.poller["polls"] >= 1
    assert logger.contains("only polls the option chain", "WARNING")


def test_a_stalled_socket_makes_the_session_partial(config) -> None:
    """A recovered stall means an unknown amount of tick data is missing."""
    seed_cache(config)
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, _ = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)

    # Every loop pass jumps two minutes: far beyond the 60-second stall timeout,
    # so the watchdog restarts the socket and the session is flagged PARTIAL.
    clock.on_sleep.append(lambda seconds: clock.advance(120))
    summary = session.run(smoke_minutes=5)

    assert summary.stream["stalls"] >= 1
    assert summary.status == STATUS_PARTIAL
    assert "stall" in summary.reason


def test_a_session_with_no_ticks_is_reported_partial(config) -> None:
    seed_cache(config)
    session, _ = make_session(config, clock=FakeClock(OPEN), client=live_client())
    session._adapter_factory = lambda: _noop_adapter(config)
    summary = session.run(smoke_minutes=1)
    assert summary.stream["ticks"] == 0
    assert summary.status == STATUS_PARTIAL
    assert summary.reason == "the session produced no ticks at all"


def test_the_universe_is_re_centred_when_the_atm_moves(config) -> None:
    """NIFTY moves: the strike window must follow the money."""
    config = capture_config(config, universe_refresh_minutes=1)
    # A dump that lists enough strikes for both the original and the moved window.
    wide = (24900, 24950, 25000, 25050, 25100, 25150, 25200, 25250)
    seed_cache(config, dump=make_dump(strikes=wide))
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, logger = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)
    run_ticks(ticker, clock)

    def on_sleep(seconds: float) -> None:
        ticker.simulate_ticks(
            [
                make_tick(
                    token=NIFTY_SPOT_TOKEN,
                    last_price=25162.0,
                    tradable=False,
                    exchange_timestamp=clock.now(),
                )
            ]
        )

    clock.on_sleep.append(on_sleep)
    summary = session.run(smoke_minutes=3)

    assert summary.status in {STATUS_SUCCESS, STATUS_PARTIAL}
    assert NIFTY_SPOT_TOKEN in session.stream.subscribed
    assert any("universe refresh" in message for message in logger.messages("INFO"))
    # 25150 is listed only in the widened dump, so it can only be subscribed
    # after the window re-centred on the new ATM - evidence the refresh happened.
    from apps.anjaneeya_capture.instruments import InstrumentMaster

    master = InstrumentMaster.from_dump(
        row for rows in make_dump(strikes=wide).values() for row in rows
    )
    moved = master.option(
        exchange="NFO", name="NIFTY", expiry=WEEK_EXPIRY, strike=25150, option_type="CE"
    )
    assert moved is not None and moved.instrument_token in session.stream.subscribed


def test_a_refresh_never_shrinks_coverage(config) -> None:
    """A dump that does not list the new strikes must not cost us the old ones."""
    config = capture_config(config, universe_refresh_minutes=1)
    seed_cache(config)  # lists 24900..25100 only
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, logger = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)
    before = set(session.plan().universe.tokens)  # type: ignore[union-attr]

    clock.on_sleep.append(
        lambda seconds: ticker.simulate_ticks(
            [
                make_tick(
                    token=NIFTY_SPOT_TOKEN,
                    last_price=25162.0,
                    tradable=False,
                    exchange_timestamp=clock.now(),
                )
            ]
        )
    )
    session.run(smoke_minutes=2)

    assert set(session.stream.subscribed) == before
    assert any("refresh skipped" in message for message in logger.warnings())


def test_the_until_deadline_stops_the_session_early(config) -> None:
    seed_cache(config)
    clock = FakeClock(OPEN)
    ticker = FakeTicker()
    session, _ = make_session(config, clock=clock, client=live_client())
    session._adapter_factory = lambda: _adapter_for(config, ticker)
    run_ticks(ticker, clock)

    summary = session.run(until=OPEN.replace(hour=9, minute=20))

    assert summary.finished_at is not None
    assert summary.finished_at <= OPEN.replace(hour=9, minute=21)
    assert clock.sleeps  # the loop actually ran

