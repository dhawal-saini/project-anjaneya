"""Tests for the option-chain poller (OI snapshots via REST quotes)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.option_chain_poller import (
    OptionChainPoller,
    quote_symbols,
    quotes_to_snapshot_rows,
)
from apps.anjaneeya_capture.storage_writer import SNAPSHOT_TIMEFRAME, CoverageTracker
from apps.anjaneeya_capture.websocket_listener import metadata_for
from core.config_loader import load_config
from core.market_calendar import NseCalendar
from core.schemas import QualityFlag
from tests.support.capture import (
    NIFTY_SPOT_TOKEN,
    WEEK_EXPIRY,
    FakeKite,
    RecordingLogger,
    empty_depth,
    make_master,
    make_quote,
    option_token,
)

IST = ZoneInfo("Asia/Kolkata")
TS = datetime(2026, 9, 29, 9, 17, 0, tzinfo=IST)
RUN_ID = "anjaneeya_capture_20260929-091500_abcdef12"
CE_TOKEN = option_token(25000, WEEK_EXPIRY, "CE")
PE_TOKEN = option_token(25000, WEEK_EXPIRY, "PE")


@pytest.fixture
def config(isolated_paths):
    return load_config()


@pytest.fixture
def metadata():
    return metadata_for(make_master(), [CE_TOKEN, PE_TOKEN, NIFTY_SPOT_TOKEN])


@pytest.fixture
def symbols_by_token(metadata):
    # The poller is fed the *option* universe: the index has no OI and no option
    # identity, so polling it would only produce "no contract identity" noise.
    return {
        token: meta.quote_key for token, meta in metadata.items() if meta.option_type is not None
    }


class FakeWriter:
    """Writer double that records snapshot batches."""

    def __init__(self) -> None:
        self.batches: list[list] = []
        self.buffered_rows = 0

    def write_snapshots(self, rows) -> int:
        batch = list(rows)
        self.batches.append(batch)
        self.buffered_rows += len(batch)
        return len(batch)

    def maybe_flush(self) -> bool:
        self.buffered_rows = 0
        return True

    @property
    def rows(self) -> list:
        return [row for batch in self.batches for row in batch]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def test_quote_symbols_use_the_exchange_prefix(metadata) -> None:
    master = make_master()
    instruments = [master.get(CE_TOKEN), master.get(PE_TOKEN)]
    assert quote_symbols(instruments) == [
        "NFO:NIFTY29SEP2625000CE",
        "NFO:NIFTY29SEP2625000PE",
    ]


def test_a_quote_becomes_a_complete_snapshot_row(metadata, symbols_by_token) -> None:
    symbol = symbols_by_token[CE_TOKEN]
    result = quotes_to_snapshot_rows(
        {symbol: make_quote(token=CE_TOKEN, last_price=123.45, exchange_timestamp=TS, oi=54321)},
        symbols_by_token=symbols_by_token,
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert result.rejected == 0
    row = result.rows[0]
    assert row.tradingsymbol is not None and row.tradingsymbol.endswith("CE")
    assert row.expiry == WEEK_EXPIRY
    assert row.strike == 25000.0
    assert row.oi == 54321
    assert row.volume == 1000
    assert row.bid == pytest.approx(123.40)
    assert row.ask == pytest.approx(123.50)
    assert row.timestamp == TS
    assert row.source.value == "ZERODHA_REST"
    # Kite has no IV field anywhere: it stays NULL rather than being invented.
    assert row.iv is None
    assert result.newest_timestamp == TS


def test_a_quote_without_a_timestamp_falls_back_and_is_flagged(metadata, symbols_by_token) -> None:
    symbol = symbols_by_token[CE_TOKEN]
    received = TS + timedelta(milliseconds=100)
    result = quotes_to_snapshot_rows(
        {symbol: make_quote(token=CE_TOKEN, last_price=100.0, exchange_timestamp=None)},
        symbols_by_token=symbols_by_token,
        metadata=metadata,
        run_id=RUN_ID,
        received_at=received,
        machine_tz=IST,
    )
    row = result.rows[0]
    assert row.timestamp == received
    assert row.quality_flag is QualityFlag.SUSPECT


def test_a_quote_for_an_unidentified_contract_is_reported(metadata, symbols_by_token) -> None:
    result = quotes_to_snapshot_rows(
        {
            "NFO:MYSTERY": make_quote(token=999999, last_price=1.0, exchange_timestamp=TS),
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=10.0, exchange_timestamp=TS, with_depth=False
            ),
        },
        symbols_by_token=symbols_by_token,
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert len(result.rows) == 1
    assert result.rejected == 1
    assert any("no contract identity" in item for item in result.missing)
    assert result.rows[0].bid is None and result.rows[0].ask is None


def test_a_crossed_quote_is_not_stored(metadata, symbols_by_token) -> None:
    symbol = symbols_by_token[CE_TOKEN]
    quote = make_quote(token=CE_TOKEN, last_price=100.0, exchange_timestamp=TS)
    quote["depth"] = {
        "buy": [{"price": 110.0, "quantity": 75, "orders": 1}],
        "sell": [{"price": 100.0, "quantity": 75, "orders": 1}],
    }
    result = quotes_to_snapshot_rows(
        {symbol: quote},
        symbols_by_token=symbols_by_token,
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert (result.rows[0].bid, result.rows[0].ask) == (None, None)


def test_an_empty_depth_block_yields_no_prices(metadata, symbols_by_token) -> None:
    quote = make_quote(token=CE_TOKEN, last_price=100.0, exchange_timestamp=TS)
    quote["depth"] = empty_depth()
    result = quotes_to_snapshot_rows(
        {symbols_by_token[CE_TOKEN]: quote},
        symbols_by_token=symbols_by_token,
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert (result.rows[0].bid, result.rows[0].ask) == (None, None)


# ---------------------------------------------------------------------------
# The poller
# ---------------------------------------------------------------------------
def make_poller(config, metadata, symbols_by_token, *, client, writer=None, clock=None, is_open=None):
    return OptionChainPoller(
        client=client,
        symbols_by_token=symbols_by_token,
        metadata=metadata,
        run_id=RUN_ID,
        poll_config=config.data_sources.zerodha.option_chain,
        capture_config=config.data_sources.zerodha.capture,
        writer=writer or FakeWriter(),
        tracker=CoverageTracker(gap_seconds=60),
        logger=RecordingLogger(),
        now_fn=clock or (lambda: TS),
        is_open=is_open or (lambda moment: True),
        sleep=lambda seconds: None,
        machine_tz=IST,
    )


def test_poll_once_requests_every_symbol_and_stores_snapshots(
    config, metadata, symbols_by_token
) -> None:
    client = FakeKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=123.0, exchange_timestamp=TS
            ),
            symbols_by_token[PE_TOKEN]: make_quote(
                token=PE_TOKEN, last_price=99.0, exchange_timestamp=TS
            ),
        }
    )
    writer = FakeWriter()
    poller = make_poller(config, metadata, symbols_by_token, client=client, writer=writer)

    result = poller.poll_once()

    assert result.requests == 1
    assert len(result.rows) == 2
    assert len(writer.rows) == 2
    assert poller.poll_count == 1
    assert sorted(client.quote_calls[0]) == poller.symbols
    assert len(poller.totals.rows) == len(result.rows)


def test_the_poller_calls_the_client_the_way_zerodha_client_defines_it(
    config, metadata, symbols_by_token
) -> None:
    """``ZerodhaClient.quote(instruments)`` takes ONE sequence - never varargs.

    The poller called ``quote(*batch)`` against that wrapper, so every poll in
    production failed with "quote() takes 2 positional arguments but 21 were given"
    while this file stayed green - its shared double was written against the
    *SDK's* varargs form instead. This double is deliberately strict, so the
    calling convention itself is pinned.
    """

    class StrictKite:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def quote(self, instruments):
            self.calls.append(list(instruments))
            return {}

    client = StrictKite()
    poller = make_poller(config, metadata, symbols_by_token, client=client)

    result = poller.poll_once()

    assert result.failures == []
    assert len(client.calls) == 1
    assert sorted(client.calls[0]) == sorted(poller.symbols)


def test_a_missing_symbol_is_reported_but_the_rest_are_stored(
    config, metadata, symbols_by_token
) -> None:
    client = FakeKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=1.0, exchange_timestamp=TS
            )
        }
    )
    poller = make_poller(config, metadata, symbols_by_token, client=client)
    result = poller.poll_once()

    assert len(result.rows) == 1
    assert result.missing == [symbols_by_token[PE_TOKEN]]
    assert any("not quoted" in message for message in poller.logger.warnings())


def test_a_failed_request_does_not_abort_the_poll(config, metadata, symbols_by_token) -> None:
    class FlakyKite(FakeKite):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.calls = 0

        def quote(self, *symbols):
            self.calls += 1
            # The second of the two one-symbol requests is throttled.
            if self.calls == 2:
                raise RuntimeError("429 Too Many Requests")
            return super().quote(*symbols)

    client = FlakyKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=5.0, exchange_timestamp=TS
            )
        }
    )
    poller = make_poller(config, metadata, symbols_by_token, client=client)
    # Force two requests so the first failure can be survived.
    poller.poll_config = poller.poll_config.model_copy(update={"quotes_per_request": 1})

    result = poller.poll_once()
    assert len(result.failures) == 1
    assert result.requests == 2
    assert poller.failures == 1
    assert len(result.rows) == 1
    assert any("poll failed" in message for message in poller.logger.messages("ERROR"))


def test_symbols_are_batched_by_the_configured_request_size(
    config, metadata, symbols_by_token
) -> None:
    poller = make_poller(config, metadata, symbols_by_token, client=FakeKite())
    poller.poll_config = poller.poll_config.model_copy(update={"quotes_per_request": 1})
    assert len(poller.chunks()) == len(poller.symbols)
    assert poller.describe()["requests_per_poll"] == len(poller.symbols)


def test_polling_is_skipped_outside_the_session(config, metadata, symbols_by_token) -> None:
    client = FakeKite()
    poller = make_poller(
        config, metadata, symbols_by_token, client=client, is_open=lambda moment: False
    )
    result = poller.poll_once()
    assert result.requests == 0
    assert client.quote_calls == []
    # ... unless the caller explicitly forces it.
    poller.poll_once(force=True)
    assert len(client.quote_calls) == 1


def test_run_loop_polls_and_stops_on_the_event(config, metadata, symbols_by_token) -> None:
    import threading

    stop = threading.Event()
    client = FakeKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=1.0, exchange_timestamp=TS
            )
        }
    )
    poller = make_poller(config, metadata, symbols_by_token, client=client)

    accumulated = poller.run(stop, max_iterations=3)
    assert poller.poll_count == 3
    assert len(accumulated.rows) == 3

    stop.set()
    poller.run(stop)
    assert poller.poll_count == 3  # an already-set event exits immediately


def test_update_universe_tracks_a_refreshed_set(config, metadata, symbols_by_token) -> None:
    poller = make_poller(config, metadata, symbols_by_token, client=FakeKite())
    new_token = option_token(25100, WEEK_EXPIRY, "CE")
    poller.update_universe(
        symbols_by_token={new_token: "NIFTY29SEP2625100CE"},
        metadata=metadata_for(make_master(), [new_token]),
    )
    assert poller.symbols == ["NIFTY29SEP2625100CE"]
    # Metadata accumulates: the previously subscribed contracts stay resolvable.
    assert CE_TOKEN in poller.metadata


def test_snapshots_are_recorded_in_the_coverage_ledger(config, metadata, symbols_by_token) -> None:
    client = FakeKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=1.0, exchange_timestamp=TS
            )
        }
    )
    poller = make_poller(config, metadata, symbols_by_token, client=client)
    poller.poll_once()

    snapshot_streams = [
        stats for stats in poller.tracker.streams if stats.timeframe == SNAPSHOT_TIMEFRAME
    ]
    assert len(snapshot_streams) == 1
    assert snapshot_streams[0].rows == 1
    assert snapshot_streams[0].quality_flag is QualityFlag.OK


def test_the_heartbeat_tracks_polls(config, metadata, symbols_by_token) -> None:
    from apps.anjaneeya_capture.heartbeat import Heartbeat, HeartbeatState

    heartbeat = Heartbeat(
        config.heartbeat_dir / "capture.json",
        interval_seconds=0.0,
        stale_after_seconds=60,
        state=HeartbeatState(run_id=RUN_ID),
    )
    client = FakeKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=1.0, exchange_timestamp=TS
            )
        }
    )
    poller = make_poller(config, metadata, symbols_by_token, client=client)
    poller.heartbeat = heartbeat
    poller.poll_once()

    assert heartbeat.state.poll_count == 1
    assert heartbeat.state.snapshots_total == 1
    assert heartbeat.state.poll_failures == 0


def test_to_thread_returns_a_named_daemon_thread(config, metadata, symbols_by_token) -> None:
    import threading

    stop = threading.Event()
    stop.set()
    poller = make_poller(config, metadata, symbols_by_token, client=FakeKite())
    thread = poller.to_thread(stop)
    assert thread.daemon is True
    assert thread.name == "option-chain-poller"


def test_the_poller_respects_the_real_calendar(config, metadata, symbols_by_token) -> None:
    """End-to-end sanity: the calendar decides whether polling happens at all."""
    calendar = NseCalendar(config.calendar)
    client = FakeKite(
        quotes={
            symbols_by_token[CE_TOKEN]: make_quote(
                token=CE_TOKEN, last_price=1.0, exchange_timestamp=TS
            )
        }
    )
    poller = make_poller(
        config,
        metadata,
        symbols_by_token,
        client=client,
        clock=lambda: TS,
        is_open=calendar.is_market_open,
    )
    assert len(poller.poll_once().rows) == 1  # 09:17 on a trading day

    after_close = datetime(2026, 9, 29, 16, 0, tzinfo=IST)
    assert poller.poll_once(now=after_close).requests == 0

