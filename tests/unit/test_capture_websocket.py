"""Tests for the WebSocket listener: conversion traps, subscription, watchdog."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.heartbeat import Heartbeat, HeartbeatState
from apps.anjaneeya_capture.storage_writer import CoverageTracker
from apps.anjaneeya_capture.websocket_listener import (
    ContractMeta,
    KiteTickerAdapter,
    TickStream,
    batch_tokens,
    localize_exchange_timestamp,
    metadata_for,
    ticks_to_rows,
)
from core.config_loader import load_config
from core.schemas import OptionType, QualityFlag
from tests.support.capture import (
    NIFTY_FUT_TOKEN,
    NIFTY_SPOT_TOKEN,
    WEEK_EXPIRY,
    FakeTicker,
    RecordingLogger,
    empty_depth,
    make_master,
    make_tick,
    option_token,
)

IST = ZoneInfo("Asia/Kolkata")
EASTERN = ZoneInfo("US/Eastern")
TS = datetime(2026, 9, 29, 9, 16, 0, tzinfo=IST)
RUN_ID = "anjaneeya_capture_20260929-091500_abcdef12"
OPTION_TOKEN = option_token(25000, WEEK_EXPIRY, "CE")


@pytest.fixture
def config(isolated_paths):
    return load_config()


@pytest.fixture
def metadata() -> dict[int, ContractMeta]:
    master = make_master()
    return metadata_for(master, [NIFTY_SPOT_TOKEN, NIFTY_FUT_TOKEN, OPTION_TOKEN])


# ---------------------------------------------------------------------------
# Timestamps: the SDK's naive machine-local value
# ---------------------------------------------------------------------------
def test_a_naive_sdk_timestamp_is_read_in_the_machines_zone() -> None:
    """`datetime.fromtimestamp` yields machine-local time; IST must be recovered."""
    # 2026-09-29 09:16 IST == 2026-09-29 03:46 UTC == 2026-09-28 23:46 Eastern.
    naive_ist = datetime(2026, 9, 29, 9, 16, 0)
    assert localize_exchange_timestamp(naive_ist, machine_tz=IST) == TS

    naive_eastern = datetime(2026, 9, 28, 23, 46, 0)
    assert localize_exchange_timestamp(naive_eastern, machine_tz=EASTERN) == TS

    # The same wall-clock value means different instants in different zones:
    # reading an Eastern machine's value as IST would shift every tick by 9.5h.
    wrong = localize_exchange_timestamp(naive_eastern, machine_tz=IST)
    assert wrong is not None and wrong != TS


def test_aware_timestamps_are_converted_and_other_types_ignored() -> None:
    aware = datetime(2026, 9, 29, 3, 46, tzinfo=ZoneInfo("UTC"))
    assert localize_exchange_timestamp(aware, machine_tz=IST) == TS
    assert localize_exchange_timestamp("2026-09-29T09:16:00") is None
    assert localize_exchange_timestamp(None) is None


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def test_full_mode_option_tick_is_converted_completely(metadata) -> None:
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=123.45, exchange_timestamp=TS, oi=5000)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert result.rejected == 0
    row = result.rows[0]
    assert row.tradingsymbol == "NIFTY29SEP2625000CE"
    assert row.option_type is OptionType.CE
    assert row.expiry == WEEK_EXPIRY
    assert row.strike == 25000.0
    assert row.oi == 5000
    assert row.volume_traded == 1000
    assert row.bid == pytest.approx(123.40)
    assert row.ask == pytest.approx(123.50)
    assert row.bid_qty == 75
    assert row.quality_flag is QualityFlag.OK
    assert row.timestamp == TS
    assert row.source.value == "ZERODHA_WEBSOCKET"


def test_bid_and_ask_are_derived_from_depth(metadata) -> None:
    """The payload has no bid/ask fields: the top of the book is the source."""
    depth = {
        "buy": [{"quantity": 150, "price": 120.0, "orders": 2}],
        "sell": [{"quantity": 300, "price": 121.5, "orders": 4}],
    }
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=120.75, exchange_timestamp=TS, depth=depth)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    row = result.rows[0]
    assert (row.bid, row.bid_qty, row.ask, row.ask_qty) == (120.0, 150, 121.5, 300)


def test_a_one_sided_book_yields_no_fabricated_price(metadata) -> None:
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=5.0, exchange_timestamp=TS, depth=empty_depth())],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    row = result.rows[0]
    assert (row.bid, row.ask, row.bid_qty, row.ask_qty) == (None, None, None, None)
    assert row.mid_price is None


def test_an_index_tick_has_no_depth_oi_or_option_identity(metadata) -> None:
    result = ticks_to_rows(
        [
            make_tick(
                token=NIFTY_SPOT_TOKEN, last_price=25012.4, tradable=False, exchange_timestamp=TS
            )
        ],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
        reference_token=NIFTY_SPOT_TOKEN,
    )
    row = result.rows[0]
    assert row.tradingsymbol == "NIFTY 50"
    assert row.option_type is None and row.expiry is None and row.strike is None
    assert row.oi is None
    assert row.bid is None and row.ask is None
    assert result.newest_reference_price == 25012.4


def test_a_crossed_book_is_not_stored(metadata) -> None:
    depth = {
        "buy": [{"quantity": 75, "price": 130.0, "orders": 1}],
        "sell": [{"quantity": 75, "price": 120.0, "orders": 1}],
    }
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=125.0, exchange_timestamp=TS, depth=depth)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    row = result.rows[0]
    assert (row.bid, row.ask) == (None, None)


def test_a_tick_without_an_exchange_timestamp_falls_back_to_receive_time(metadata) -> None:
    received = TS + timedelta(milliseconds=250)
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=123.0, with_timestamp=False)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=received,
        machine_tz=IST,
    )
    row = result.rows[0]
    assert row.timestamp == received
    assert row.quality_flag is QualityFlag.SUSPECT
    assert result.suspect == 1


def test_a_future_exchange_timestamp_is_recorded_not_dropped(metadata) -> None:
    """A machine clock behind the exchange must not cost us data.

    The default policy (``trust="exchange"``) keeps the tick ``OK`` and stamps it with
    the exchange time - a slow PC clock is not bad data. The strict policy still
    flags it, which is what ``trust="machine"`` is for; both keep the row.
    """
    tick = make_tick(
        token=OPTION_TOKEN, last_price=123.0, exchange_timestamp=TS + timedelta(seconds=30)
    )
    common = {
        "metadata": metadata,
        "run_id": RUN_ID,
        "received_at": TS,
        "machine_tz": IST,
    }

    trusted = ticks_to_rows([tick], **common).rows[0]
    assert trusted.timestamp == TS + timedelta(seconds=30)
    assert trusted.fetch_timestamp >= trusted.timestamp
    assert trusted.quality_flag is QualityFlag.OK

    strict = ticks_to_rows([tick], **common, trust="machine", tolerance_seconds=2.0).rows[0]
    assert strict.timestamp == TS + timedelta(seconds=30)
    assert strict.quality_flag is QualityFlag.SUSPECT


def test_out_of_session_ticks_are_recorded_with_a_flag(metadata) -> None:
    pre_open = datetime(2026, 9, 29, 9, 5, tzinfo=IST)
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=123.0, exchange_timestamp=pre_open)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=pre_open,
        machine_tz=IST,
        in_session=lambda moment: moment >= TS,
    )
    assert len(result.rows) == 1
    assert result.out_of_session == 1
    assert result.rows[0].quality_flag is QualityFlag.OUT_OF_SESSION


def test_out_of_session_ticks_can_be_dropped_explicitly(metadata) -> None:
    pre_open = datetime(2026, 9, 29, 9, 5, tzinfo=IST)
    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=123.0, exchange_timestamp=pre_open)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=pre_open,
        machine_tz=IST,
        in_session=lambda moment: moment >= TS,
        drop_out_of_session=True,
    )
    assert result.rows == []
    assert result.out_of_session == 1


def test_unusable_ticks_are_counted_with_a_reason(metadata) -> None:
    result = ticks_to_rows(
        [
            "not-a-dict",
            {"last_price": 1.0},
            make_tick(token=OPTION_TOKEN, last_price=-5.0, exchange_timestamp=TS),
        ],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert result.rows == []
    assert result.rejected == 3
    assert result.rejections[0].startswith("not a mapping")


def test_an_unknown_token_is_reported_but_still_stored(metadata) -> None:
    result = ticks_to_rows(
        [make_tick(token=424242, last_price=10.0, exchange_timestamp=TS)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=TS,
        machine_tz=IST,
    )
    assert result.unknown_tokens == 1
    assert result.rows[0].tradingsymbol == "TOKEN424242"
    assert result.rows[0].option_type is None


def test_ltp_mode_ticks_carry_no_oi_or_volume(metadata) -> None:
    """The ltp packet is 8 bytes: only a price exists, and we must not invent more."""
    tick = {"tradable": True, "mode": "ltp", "instrument_token": OPTION_TOKEN, "last_price": 99.0}
    result = ticks_to_rows([tick], metadata=metadata, run_id=RUN_ID, received_at=TS, machine_tz=IST)
    row = result.rows[0]
    assert row.last_price == 99.0
    assert row.oi is None and row.volume_traded is None
    assert row.quality_flag is QualityFlag.SUSPECT  # ltp mode carries no exchange time


def test_a_lagging_machine_clock_no_longer_flags_a_good_tick(metadata) -> None:
    """2026-10-27: the PC clock sat ~18 s behind, so ~93% of good ticks were flagged.

    The tick carries its own exchange timestamp, and with the default
    ``trust="exchange"`` that is what defines the bar - the machine clock is only a
    fallback for ticks that carry no exchange time at all.
    """
    behind = TS - timedelta(seconds=18)  # the local receive clock, 18 s slow

    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=123.0, exchange_timestamp=TS)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=behind,
        machine_tz=IST,
    )

    row = result.rows[0]
    assert row.quality_flag is QualityFlag.OK
    assert row.timestamp == TS  # the exchange time, not the receive time
    assert result.clock_offset_seconds == pytest.approx(18.0)


def test_the_strict_machine_rule_is_still_available(metadata) -> None:
    """``trust="machine"`` keeps the old behaviour for anyone who wants it."""
    behind = TS - timedelta(seconds=18)

    result = ticks_to_rows(
        [make_tick(token=OPTION_TOKEN, last_price=123.0, exchange_timestamp=TS)],
        metadata=metadata,
        run_id=RUN_ID,
        received_at=behind,
        machine_tz=IST,
        trust="machine",
        tolerance_seconds=2.0,
    )

    assert result.rows[0].quality_flag is QualityFlag.SUSPECT


def test_a_tick_with_no_exchange_time_is_suspect_under_either_trust(metadata) -> None:
    tick = make_tick(token=OPTION_TOKEN, last_price=123.0)
    tick.pop("exchange_timestamp", None)
    tick.pop("last_trade_time", None)

    for trust in ("exchange", "machine"):
        result = ticks_to_rows(
            [tick],
            metadata=metadata,
            run_id=RUN_ID,
            received_at=TS,
            machine_tz=IST,
            trust=trust,
        )
        assert result.rows[0].quality_flag is QualityFlag.SUSPECT, trust
        assert result.rows[0].timestamp == TS


# ---------------------------------------------------------------------------
# Subscription batching + metadata
# ---------------------------------------------------------------------------
def test_batch_tokens_deduplicates_and_splits() -> None:
    assert batch_tokens([1, 2, 3, 4, 5], size=2) == [[1, 2], [3, 4], [5]]
    assert batch_tokens([1, 1, 2], size=10) == [[1, 2]]
    assert batch_tokens([], size=10) == []
    with pytest.raises(ValueError, match="batch size"):
        batch_tokens([1], size=0)


def test_metadata_falls_back_to_a_placeholder_for_unknown_tokens() -> None:
    master = make_master()
    resolved = metadata_for(master, [NIFTY_SPOT_TOKEN, 999999], symbols={999999: "SOMETHING"})
    assert resolved[NIFTY_SPOT_TOKEN].tradingsymbol == "NIFTY 50"
    assert resolved[NIFTY_SPOT_TOKEN].is_index is True
    assert resolved[999999].tradingsymbol == "SOMETHING"
    assert resolved[999999].option_type is None


def test_adapter_subscribes_then_sets_the_mode_per_batch(config) -> None:
    """`subscribe()` alone marks tokens QUOTE: set_mode must follow it."""
    ticker = FakeTicker()
    ticker.connected = True
    adapter = KiteTickerAdapter(
        api_key="k",
        access_token="t",
        mode="full",
        websocket_config=config.data_sources.zerodha.websocket,
        ticker=ticker,
    )
    adapter.set_callbacks(
        on_ticks=None,
        on_message=None,
        on_connect=None,
        on_close=None,
        on_error=None,
        on_reconnect=None,
        on_noreconnect=None,
    )
    assert adapter.subscribe([1, 2, 3], batch_size=2) == 3
    assert ticker.events == [
        ("subscribe", [1, 2]),
        ("set_mode", ("full", [1, 2])),
        ("subscribe", [3]),
        ("set_mode", ("full", [3])),
    ]
    # Without set_mode the SDK's own bookkeeping would say QUOTE, not FULL.
    assert ticker.subscribed_tokens == {1: "full", 2: "full", 3: "full"}
    assert adapter.unsubscribe([1, 2], batch_size=10) == 2
    assert ticker.subscribed_tokens == {3: "full"}
    assert adapter.describe() == {"mode": "full", "subscribed": 1}




# ---------------------------------------------------------------------------
# TickStream: subscription, persistence, watchdog
# ---------------------------------------------------------------------------
class FakeWriter:
    """A writer double recording every batch it is handed."""

    def __init__(self, *, flush_on_maybe: bool = False) -> None:
        self.batches: list[list] = []
        self.maybe_flush_calls = 0
        self.flush_on_maybe = flush_on_maybe
        self.buffered_rows = 0

    def write_ticks(self, rows) -> int:
        batch = list(rows)
        self.batches.append(batch)
        self.buffered_rows += len(batch)
        return len(batch)

    def maybe_flush(self) -> bool:
        self.maybe_flush_calls += 1
        if self.flush_on_maybe:
            self.buffered_rows = 0
            return True
        return False

    @property
    def rows(self) -> list:
        return [row for batch in self.batches for row in batch]


def make_stream(
    tmp_path,
    config,
    metadata,
    *,
    ticker: FakeTicker | None = None,
    clock=None,
    is_open=None,
    writer: FakeWriter | None = None,
    heartbeat=None,
) -> tuple[TickStream, FakeTicker, FakeWriter]:
    ticker = ticker or FakeTicker()
    writer = writer or FakeWriter()
    stream = TickStream(
        adapter_factory=lambda: _adapter(config, ticker),
        metadata=metadata,
        run_id=RUN_ID,
        mode="full",
        websocket_config=config.data_sources.zerodha.websocket,
        capture_config=config.data_sources.zerodha.capture,
        writer=writer,
        tracker=CoverageTracker(gap_seconds=30, is_open=is_open or (lambda moment: True)),
        heartbeat=heartbeat,
        logger=RecordingLogger(),
        clock=clock or (lambda: TS),
        is_open=is_open or (lambda moment: True),
        machine_tz=IST,
        reference_token=NIFTY_SPOT_TOKEN,
    )
    return stream, ticker, writer


def _adapter(config, ticker: FakeTicker) -> KiteTickerAdapter:
    adapter = KiteTickerAdapter(
        api_key="k",
        access_token="t",
        mode="full",
        websocket_config=config.data_sources.zerodha.websocket,
        ticker=ticker,
    )
    # The fake's connect() fires on_connect, which flips stream.connected.
    ticker.connected = True
    return adapter


def test_start_connects_and_subscribes(config, metadata, tmp_path) -> None:
    stream, ticker, _ = make_stream(tmp_path, config, metadata)
    stream.start([NIFTY_SPOT_TOKEN, OPTION_TOKEN])

    assert stream.connected is True
    assert stream.subscribed == {NIFTY_SPOT_TOKEN, OPTION_TOKEN}
    assert ticker.subscribed_tokens == {NIFTY_SPOT_TOKEN: "full", OPTION_TOKEN: "full"}
    assert ("connect", True) in ticker.events


def test_a_tick_message_is_converted_and_written(config, metadata, tmp_path) -> None:
    stream, ticker, writer = make_stream(tmp_path, config, metadata)
    stream.start([OPTION_TOKEN])
    ticker.simulate_ticks(
        [make_tick(token=OPTION_TOKEN, last_price=123.45, exchange_timestamp=TS)]
    )

    assert stream.ticks_total == 1
    assert len(writer.rows) == 1
    assert writer.rows[0].tradingsymbol == "NIFTY29SEP2625000CE"
    assert stream.largest_gap_seconds == 0.0


def test_the_keep_alive_frame_marks_liveness_without_a_tick(config, metadata, tmp_path) -> None:
    moments = [TS]

    def clock():
        return moments[0]

    stream, ticker, writer = make_stream(tmp_path, config, metadata, clock=clock)
    stream.start([OPTION_TOKEN])
    ticker.simulate_heartbeat()

    assert stream.messages == 1
    assert stream.ticks_total == 0
    # A quiet market: 55s later the socket is still considered alive.
    moments[0] = TS + timedelta(seconds=55)
    assert stream.monitor() == []
    assert stream.stalls == 0


def test_a_silent_socket_is_restarted_by_the_watchdog(config, metadata, tmp_path) -> None:
    moments = [TS]
    ticker = FakeTicker()
    holder: dict[str, FakeTicker] = {"current": ticker}

    def factory():
        return _adapter(config, holder["current"])

    stream = TickStream(
        adapter_factory=factory,
        metadata=metadata,
        run_id=RUN_ID,
        mode="full",
        websocket_config=config.data_sources.zerodha.websocket,
        capture_config=config.data_sources.zerodha.capture,
        writer=FakeWriter(),
        tracker=CoverageTracker(gap_seconds=30),
        logger=RecordingLogger(),
        clock=lambda: moments[0],
        machine_tz=IST,
        reference_token=NIFTY_SPOT_TOKEN,
    )
    stream.start([OPTION_TOKEN])
    holder["current"] = FakeTicker()  # the socket the restart will create
    moments[0] = TS + timedelta(seconds=120)  # beyond stall_timeout_seconds (60)

    actions = stream.monitor()
    assert "stall_restart" in actions
    assert stream.stalls == 1
    assert stream.restarts == 1
    assert holder["current"].subscribed_tokens == {OPTION_TOKEN: "full"}


def test_the_watchdog_reconnects_when_the_socket_closed(config, metadata, tmp_path) -> None:
    moments = [TS]
    ticker = FakeTicker()
    holder = {"current": ticker}
    stream = TickStream(
        adapter_factory=lambda: _adapter(config, holder["current"]),
        metadata=metadata,
        run_id=RUN_ID,
        mode="full",
        websocket_config=config.data_sources.zerodha.websocket,
        capture_config=config.data_sources.zerodha.capture,
        writer=FakeWriter(),
        tracker=CoverageTracker(gap_seconds=30),
        logger=RecordingLogger(),
        clock=lambda: moments[0],
        machine_tz=IST,
    )
    stream.start([OPTION_TOKEN])
    ticker.simulate_close("server closed")
    holder["current"] = FakeTicker()
    moments[0] = TS + timedelta(seconds=10)

    actions = stream.monitor()
    assert "reconnect" in actions
    assert stream.restarts == 1
    assert stream.connected is True


def test_gaps_are_logged_but_only_up_to_the_configured_cap(config, metadata, tmp_path) -> None:
    config = config.model_copy(
        update={
            "data_sources": config.data_sources.model_copy(
                update={
                    "zerodha": config.data_sources.zerodha.model_copy(
                        update={
                            "capture": config.data_sources.zerodha.capture.model_copy(
                                update={
                                    "gaps": config.data_sources.zerodha.capture.gaps.model_copy(
                                        update={"log_limit_per_session": 2}
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )
    moments = [TS]
    # 45s apart, so every tick after the first is a gap.
    stream, ticker, writer = make_stream(
        tmp_path,
        config,
        metadata,
        clock=lambda: moments[0],
        is_open=lambda moment: True,
    )
    stream.start([OPTION_TOKEN])
    for index in range(5):
        moments[0] = TS + timedelta(seconds=45 * index)
        ticker.simulate_ticks(
            [make_tick(token=OPTION_TOKEN, last_price=100.0 + index, exchange_timestamp=moments[0])]
        )

    assert stream.total_gaps == 4
    warnings = stream.logger.warnings()
    assert sum("gap detected" in message for message in warnings) == 2
    assert any("gap logging cap (2)" in message for message in warnings)


def test_the_stream_updates_the_heartbeat(config, metadata, tmp_path) -> None:
    heartbeat = Heartbeat(
        tmp_path / "hb" / "capture.json",
        interval_seconds=0.0,
        stale_after_seconds=60,
        state=HeartbeatState(run_id=RUN_ID),
    )
    stream, ticker, writer = make_stream(tmp_path, config, metadata, heartbeat=heartbeat)
    stream.start([NIFTY_SPOT_TOKEN, OPTION_TOKEN])
    ticker.simulate_ticks(
        [
            make_tick(token=NIFTY_SPOT_TOKEN, last_price=25012.4, tradable=False, exchange_timestamp=TS),
            make_tick(token=OPTION_TOKEN, last_price=123.45, exchange_timestamp=TS),
        ]
    )

    state = heartbeat.state
    assert state.connected is True
    assert state.ticks_total == 2
    assert state.newest_price == 25012.4  # the reference instrument wins
    assert state.subscribed_instruments == 2
    assert state.gap_count == 0
    assert heartbeat.publish() is not None


def test_reconnect_and_error_callbacks_are_counted(config, metadata, tmp_path) -> None:
    stream, ticker, _ = make_stream(tmp_path, config, metadata)
    stream.start([OPTION_TOKEN])
    ticker.simulate_error("boom")
    ticker.simulate_reconnect()
    ticker.simulate_noreconnect()

    assert stream.reconnects == 1
    assert stream.describe()["restarts"] == 0
    description = stream.describe()
    assert description["subscribed_instruments"] == 1
    assert description["ticks"] == 0


def test_refresh_universe_subscribes_new_and_drops_old_strikes(config, metadata, tmp_path) -> None:
    master = make_master()
    other = option_token(25100, WEEK_EXPIRY, "CE")
    stream, ticker, _ = make_stream(tmp_path, config, metadata)
    stream.start([OPTION_TOKEN])

    summary = stream.refresh_universe(
        [other, NIFTY_SPOT_TOKEN],
        metadata=metadata_for(master, [other, NIFTY_SPOT_TOKEN]),
        reference_token=NIFTY_SPOT_TOKEN,
    )
    assert summary["added"] == sorted([other, NIFTY_SPOT_TOKEN])
    assert summary["removed"] == [OPTION_TOKEN]
    assert stream.subscribed == {other, NIFTY_SPOT_TOKEN}

    # A late tick for the dropped strike is still stored under its real symbol.
    ticker.simulate_ticks([make_tick(token=OPTION_TOKEN, last_price=50.0, exchange_timestamp=TS)])
    assert stream.ticks_total == 1
    assert stream.rejected == 0


def test_stop_closes_the_socket(config, metadata, tmp_path) -> None:
    stream, ticker, _ = make_stream(tmp_path, config, metadata)
    stream.start([OPTION_TOKEN])
    stream.stop()
    assert stream.connected is False
    assert ("close", None) in ticker.events
    # Idempotent, and the monitor stops restarting.
    stream.stop()
    assert stream.monitor() == []


def test_subscribing_before_start_is_an_error(config, metadata, tmp_path) -> None:
    stream, _, _ = make_stream(tmp_path, config, metadata)
    with pytest.raises(RuntimeError, match="start"):
        stream.subscribe([OPTION_TOKEN])


def test_no_tick_is_dropped_across_a_reconnect(config, metadata, tmp_path) -> None:
    """The SDK replays the subscription; in the meantime our buffer stays intact."""
    stream, ticker, writer = make_stream(tmp_path, config, metadata)
    stream.start([OPTION_TOKEN])
    ticker.simulate_ticks([make_tick(token=OPTION_TOKEN, last_price=100.0, exchange_timestamp=TS)])
    ticker.simulate_reconnect()
    ticker.simulate_ticks(
        [make_tick(token=OPTION_TOKEN, last_price=101.0, exchange_timestamp=TS + timedelta(seconds=1))]
    )

    assert [row.last_price for row in writer.rows] == [100.0, 101.0]
    assert ticker.subscribed_tokens == {OPTION_TOKEN: "full"}

