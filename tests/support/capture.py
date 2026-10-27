"""Synthetic market data and provider fakes used across the capture tests.

Everything here is deliberately *shape-accurate* rather than real:

* :func:`make_dump` produces the rows Kite's ``instruments()`` returns, with the
  same quirks (NaN/``""`` for an index's expiry/strike, a ``name`` field, ...).
* :func:`make_tick` produces the dicts ``KiteTicker`` hands to ``on_ticks`` for
  a 184-byte FULL packet - including a **naive** ``exchange_timestamp``, which is
  exactly the trap the listener must survive.
* :func:`make_quote` produces the REST ``quote()`` payload shape.
* :class:`FakeKite` / :class:`FakeTicker` record every call, so a test can assert
  on request budgeting and on subscribe/set_mode ordering rather than on logs.

No test may reach the network (see ``tests/conftest.py``), which is precisely why
these fakes exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from apps.anjaneeya_capture.instruments import InstrumentMaster

#: A stable, realistic-enough NIFTY 50 index token (its low byte encodes segment 9).
NIFTY_SPOT_TOKEN = 256265
NIFTY_FUT_TOKEN = 51001
OPTION_BASE_TOKEN = 900001

#: Trading day used across the capture tests (Tuesday 2026-09-29, IST). Tuesday is
#: the expiry weekday the shipped ``config/calendar.yaml`` configures and the NFO
#: instrument dump the exchange publishes agrees with, so this day *is* an expiry
#: day - the classic session to research.
DAY = date(2026, 9, 29)
#: Weekly expiry of the ISO week containing :data:`DAY` (the expiry day itself).
WEEK_EXPIRY = DAY
#: The following week's expiry.
NEXT_WEEK_EXPIRY = date(2026, 10, 6)
#: Monthly expiry of October 2026 - the last Tuesday of the month.
MONTH_EXPIRY = date(2026, 10, 27)

#: Strikes :func:`make_dump` lists - the shipped +/-200 ATM window around 25 000.
STRIKES: tuple[int, ...] = (24800, 24900, 25000, 25100, 25200)
#: Expiries :func:`make_dump` lists.
DUMP_EXPIRIES: tuple[date, ...] = (WEEK_EXPIRY, NEXT_WEEK_EXPIRY)


def nan() -> float:
    """``NaN`` - the value Kite's dump uses for "not applicable"."""
    return float("nan")


def make_dump(
    *,
    spot_price: float = 25012.0,
    strikes: tuple[int, ...] = STRIKES,
    expiries: tuple[date, ...] = DUMP_EXPIRIES,
    exchanges: tuple[str, ...] = ("NSE", "NFO"),
) -> dict[str, list[dict[str, Any]]]:
    """Build a synthetic Kite instrument dump, keyed by exchange."""
    frames: dict[str, list[dict[str, Any]]] = {"NSE": [], "NFO": []}

    if "NSE" in exchanges:
        frames["NSE"].append(
            {
                "instrument_token": NIFTY_SPOT_TOKEN,
                "exchange_token": "256265",
                "tradingsymbol": "NIFTY 50",
                "name": "NIFTY 50",
                "last_price": spot_price,
                "expiry": "",
                "strike": nan(),
                "tick_size": 0.05,
                "lot_size": nan(),
                "instrument_type": nan(),
                "segment": "INDICES",
                "exchange": "NSE",
            }
        )

    if "NFO" in exchanges:
        frames["NFO"].append(
            {
                "instrument_token": NIFTY_FUT_TOKEN,
                "exchange_token": "51001",
                "tradingsymbol": "NIFTY26OCTFUT",
                "name": "NIFTY",
                "last_price": spot_price + 40,
                "expiry": MONTH_EXPIRY.isoformat(),
                "strike": 0,
                "tick_size": 0.05,
                "lot_size": 75,
                "instrument_type": "FUT",
                "segment": "NFO-FUT",
                "exchange": "NFO",
            }
        )
        token = OPTION_BASE_TOKEN
        for expiry in expiries:
            for strike in strikes:
                for side in ("CE", "PE"):
                    token += 1
                    frames["NFO"].append(
                        {
                            "instrument_token": token,
                            "exchange_token": str(token),
                            "tradingsymbol": f"NIFTY{expiry:%d%b%y}{strike}{side}".upper(),
                            "name": "NIFTY",
                            "last_price": max(5.0, spot_price - strike + 120.0),
                            "expiry": expiry.isoformat(),
                            "strike": strike,
                            "tick_size": 0.05,
                            "lot_size": 75,
                            "instrument_type": side,
                            "segment": "NFO-OPT",
                            "exchange": "NFO",
                        }
                    )
    return {name: rows for name, rows in frames.items() if name in exchanges}


def make_master(**kwargs: Any) -> InstrumentMaster:
    """An :class:`InstrumentMaster` built from the synthetic dump."""
    frames = make_dump(**kwargs)
    return InstrumentMaster.from_dump(row for rows in frames.values() for row in rows)


def option_token(strike: int, expiry: date, side: str) -> int:
    """The token :func:`make_dump` assigns to one synthetic option."""
    index = DUMP_EXPIRIES.index(expiry) * len(STRIKES) * 2 + STRIKES.index(strike) * 2
    return OPTION_BASE_TOKEN + index + (0 if side == "CE" else 1) + 1


def make_tick(
    *,
    token: int,
    last_price: float,
    exchange_timestamp: datetime | None = None,
    with_timestamp: bool = True,
    with_depth: bool = True,
    volume: int = 1000,
    oi: int = 5000,
    tradable: bool = True,
    depth: dict[str, Any] | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    """Build one KiteTicker ``on_ticks`` payload entry (FULL-mode shaped)."""
    tick: dict[str, Any] = {
        "tradable": tradable,
        "mode": mode,
        "instrument_token": token,
        "last_price": last_price,
        "last_traded_quantity": 75,
        "average_traded_price": last_price,
        "volume_traded": volume,
        "total_buy_quantity": 1000,
        "total_sell_quantity": 1200,
        "ohlc": {"open": last_price, "high": last_price, "low": last_price, "close": last_price},
        "change": 0.0,
    }
    if with_timestamp and exchange_timestamp is not None:
        tick["exchange_timestamp"] = exchange_timestamp
    if tradable:
        tick["last_trade_time"] = exchange_timestamp
        tick["oi"] = oi
        tick["oi_day_high"] = oi
        tick["oi_day_low"] = oi
        if with_depth:
            tick["depth"] = depth if depth is not None else default_depth(last_price)
    return tick


def default_depth(last_price: float, *, quantity: int = 75) -> dict[str, list[dict[str, Any]]]:
    """A two-sided book: best bid one tick below, best ask one tick above."""
    return {
        "buy": [
            {"quantity": quantity, "price": round(last_price - 0.05, 2), "orders": 3},
            {"quantity": quantity * 2, "price": round(last_price - 0.10, 2), "orders": 5},
        ],
        "sell": [
            {"quantity": quantity, "price": round(last_price + 0.05, 2), "orders": 4},
            {"quantity": quantity * 2, "price": round(last_price + 0.10, 2), "orders": 6},
        ],
    }


def empty_depth() -> dict[str, list[dict[str, Any]]]:
    """A book with no resting orders (perfectly possible for a far strike)."""
    return {"buy": [], "sell": []}


def make_quote(
    *,
    token: int,
    last_price: float,
    exchange_timestamp: datetime | None = None,
    volume: int = 1000,
    oi: int = 5000,
    with_depth: bool = True,
) -> dict[str, Any]:
    """Build one entry of a REST ``quote()`` response."""
    quote: dict[str, Any] = {
        "instrument_token": token,
        "timestamp": exchange_timestamp,
        "last_price": last_price,
        "last_quantity": 75,
        "last_trade_time": exchange_timestamp,
        "average_price": last_price,
        "volume": volume,
        "buy_quantity": 1000,
        "sell_quantity": 1200,
        "ohlc": {"open": last_price, "high": last_price, "low": last_price, "close": last_price},
        "oi": oi,
    }
    if with_depth:
        quote["depth"] = default_depth(last_price)
    return quote



# ---------------------------------------------------------------------------
# Provider fakes
# ---------------------------------------------------------------------------
class FakeKite:
    """A ``KiteConnect`` stand-in exposing only what the capture app calls."""

    def __init__(
        self,
        *,
        dump: dict[str, list[dict[str, Any]]] | None = None,
        quotes: dict[str, dict[str, Any]] | None = None,
        ltp: dict[str, dict[str, Any]] | None = None,
        history: dict[tuple[int, str], list[dict[str, Any]]] | None = None,
        history_error: Exception | None = None,
    ) -> None:
        self.dump = dump if dump is not None else make_dump()
        self.quotes = quotes or {}
        self.ltp_payload = ltp or {}
        self.history = history or {}
        self.history_error = history_error
        self.instrument_calls: list[str | None] = []
        self.quote_calls: list[list[str]] = []
        self.ltp_calls: list[list[str]] = []
        self.history_calls: list[tuple[int, Any, Any, str, bool, bool]] = []
        self.access_token: str | None = None
        self.session_expiry_hook: Any = None

    # -- session ------------------------------------------------------------
    def set_access_token(self, token: str) -> None:
        self.access_token = token

    def set_session_expiry_hook(self, hook: Any) -> None:
        self.session_expiry_hook = hook

    def profile(self) -> dict[str, Any]:
        return {"user_id": "AB1234", "user_name": "TEST"}

    # -- data ---------------------------------------------------------------
    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        self.instrument_calls.append(exchange)
        if exchange is None:
            return [row for rows in self.dump.values() for row in rows]
        return list(self.dump.get(exchange, []))

    def quote(self, *symbols: Any) -> dict[str, dict[str, Any]]:
        """Quotes for ``symbols``.

        Accepts **both** calling conventions, because two different real objects are
        stood in for here:

        * ``KiteConnect.quote(*instruments)`` - the SDK, used by the session/capture
          paths;
        * ``ZerodhaClient.quote(instruments)`` - the project's wrapper (a single
          sequence), which is what ``OptionChainPoller`` is given.

        *Tolerant on purpose*: the poller used to call ``quote(*batch)`` against the
        wrapper and every poll failed in production while these tests passed, because
        the double only matched the SDK form. A double that accepts both cannot hide
        that class of mismatch again - the strict regression test lives next to the
        poller tests.
        """
        requested = (
            list(symbols[0]) if len(symbols) == 1 and not isinstance(symbols[0], str) else list(symbols)
        )
        self.quote_calls.append(requested)
        return {symbol: self.quotes[symbol] for symbol in requested if symbol in self.quotes}

    def ltp(self, *symbols: str) -> dict[str, dict[str, Any]]:
        self.ltp_calls.append(list(symbols))
        return {symbol: self.ltp_payload[symbol] for symbol in symbols if symbol in self.ltp_payload}

    def historical_data(
        self,
        instrument_token: int,
        from_date: Any,
        to_date: Any,
        interval: str,
        *,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]:
        self.history_calls.append((instrument_token, from_date, to_date, interval, continuous, oi))
        if self.history_error is not None:
            raise self.history_error
        return list(self.history.get((instrument_token, interval), []))



def make_app_context(
    config,
    *,
    mode: str = "capture",
    dry_run: bool = False,
    logger=None,
    bootstrap: bool = True,
):
    """Build a real :class:`AppContext` for the capture app.

    The apps are driven through the identical object the CLI builds, so the tests
    exercise production wiring (paths, calendar, run context) rather than a
    bespoke stand-in. Logging is replaced with :class:`RecordingLogger` so tests
    can assert on warnings without touching the log files.
    """
    from core.app_bootstrap import APP_CAPTURE, AppContext
    from core.market_calendar import NseCalendar
    from core.run_context import create_run_context
    from core.storage import bootstrap_database

    if bootstrap:
        bootstrap_database(config.db_path, schema_version=1)
    run_context = create_run_context(
        APP_CAPTURE,
        mode=mode,
        config_hash=config.config_hash,
        config_dir=config.config_dir,
        timezone_name=config.timezone_name,
        package_names=[],
    )
    return AppContext(
        app_name=APP_CAPTURE,
        mode=mode,
        config=config,
        run_context=run_context,
        calendar=NseCalendar(config.calendar),
        logger=logger if logger is not None else RecordingLogger(),
        dry_run=dry_run,
    )


def capture_config(config, **overrides):
    """Return the config with ``capture.*`` overrides applied.

    Tests need small intervals (a one-minute universe refresh, a tiny flush
    batch) without editing the shipped YAML, so the nested models are copied
    explicitly - the same immutability the apps rely on.
    """
    capture = config.data_sources.zerodha.capture.model_copy(update=overrides)
    zerodha = config.data_sources.zerodha.model_copy(update={"capture": capture})
    data_sources = config.data_sources.model_copy(update={"zerodha": zerodha})
    return config.model_copy(update={"data_sources": data_sources})


class FakeClock:
    """A deterministic clock whose ``sleep`` advances it by the requested amount.

    The capture loop sleeps in small slices and re-reads the clock every pass, so
    advancing on sleep lets a test run through a whole session - including the
    deadline logic - with no real waiting.
    """

    def __init__(self, start: datetime, *, max_step_seconds: float | None = None) -> None:
        self.current = start
        self.max_step_seconds = max_step_seconds
        self.sleeps: list[float] = []
        self.on_sleep: list[Any] = []

    def now(self) -> datetime:
        """The current fake time."""
        return self.current

    def sleep(self, seconds: float) -> None:
        """Record the sleep, advance the clock and run any per-sleep hooks."""
        step = float(seconds)
        if self.max_step_seconds is not None:
            step = min(step, self.max_step_seconds)
        self.sleeps.append(step)
        for hook in self.on_sleep:
            hook(step)
        self.current += timedelta(seconds=step)

    def advance(self, seconds: float) -> None:
        """Move the clock forward without sleeping."""
        self.current += timedelta(seconds=seconds)


@dataclass
class RecordingLogger:
    """A logger double that records the level and message of every call.

    Tests then assert on *what the operator would have seen* (a warning about a
    stale dump, a capped gap log line) rather than on internals.
    """

    records: list[tuple[str, str]] = field(default_factory=list)

    def _add(self, level: str, message: Any) -> None:
        self.records.append((level, str(message)))

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("DEBUG", message)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("INFO", message)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("WARNING", message)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("ERROR", message)

    def exception(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("ERROR", message)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("CRITICAL", message)

    # -- queries ------------------------------------------------------------
    def messages(self, level: str | None = None) -> list[str]:
        """All messages, optionally filtered by level name."""
        return [message for name, message in self.records if level is None or name == level]

    def warnings(self) -> list[str]:
        """Every WARNING message."""
        return self.messages("WARNING")

    def contains(self, needle: str, level: str | None = None) -> bool:
        """True when any recorded message contains ``needle``."""
        return any(needle in message for message in self.messages(level))


@dataclass
class FakeTicker:
    """A ``KiteTicker`` stand-in that records the subscribe/set_mode protocol.

    The real SDK requires ``subscribe()`` *then* ``set_mode()`` per token batch
    and replays both automatically after a reconnect (``_on_open`` ->
    ``resubscribe``). The fake mirrors that contract, so listener tests assert on
    ordering and on reconnect replay instead of trusting a comment.
    """

    api_key: str = "test-key"
    access_token: str = "test-token"
    reconnect_max_tries: int = 50
    reconnect_max_delay: float = 60.0
    connect_timeout: float = 30.0

    on_ticks: Any = None
    on_connect: Any = None
    on_close: Any = None
    on_error: Any = None
    on_reconnect: Any = None
    on_noreconnect: Any = None
    on_message: Any = None

    subscribed_tokens: dict[int, str] = field(default_factory=dict)
    events: list[tuple[str, Any]] = field(default_factory=list)
    connected: bool = False

    # -- provider API -------------------------------------------------------
    def connect(self, threaded: bool = False, **kwargs: Any) -> None:
        self.connected = True
        self.events.append(("connect", threaded))
        self._fire_connect()

    def close(self) -> None:
        self.connected = False
        self.events.append(("close", None))

    def subscribe(self, tokens: list[int]) -> bool:
        self.events.append(("subscribe", list(tokens)))
        if not self.connected:
            raise RuntimeError("cannot subscribe on a closed socket")
        for token in tokens:
            self.subscribed_tokens[token] = "quote"
        return True

    def unsubscribe(self, tokens: list[int]) -> bool:
        self.events.append(("unsubscribe", list(tokens)))
        for token in tokens:
            self.subscribed_tokens.pop(token, None)
        return True

    def set_mode(self, mode: str, tokens: list[int]) -> bool:
        self.events.append(("set_mode", (mode, list(tokens))))
        for token in tokens:
            self.subscribed_tokens[token] = mode
        return True

    # -- test controls ------------------------------------------------------
    def _fire_connect(self) -> None:
        if self.on_connect is not None:
            self.on_connect(self, {"status": "connected"})

    def resubscribe(self) -> None:
        """What the SDK's ``_on_open`` does after a reconnect."""
        self.events.append(("resubscribe", len(self.subscribed_tokens)))
        for token, mode in list(self.subscribed_tokens.items()):
            self.subscribe([token])
            self.set_mode(mode, [token])

    def simulate_reconnect(self) -> None:
        """Close and reopen the socket the way the SDK's factory does."""
        if self.on_reconnect is not None:
            self.on_reconnect(self, 1)
        self.close()
        self.connected = True
        self.events.append(("connect", True))
        self.resubscribe()
        self._fire_connect()

    def simulate_close(self, reason: str = "closed") -> None:
        """Deliver an on_close callback."""
        self.connected = False
        self.events.append(("on_close", reason))
        if self.on_close is not None:
            self.on_close(self, reason)

    def simulate_error(self, message: str = "boom") -> None:
        """Deliver an on_error callback."""
        if self.on_error is not None:
            self.on_error(self, message)

    def simulate_noreconnect(self) -> None:
        """Deliver ``on_noreconnect`` (the SDK has exhausted its retry budget)."""
        if self.on_noreconnect is not None:
            self.on_noreconnect(self)

    def simulate_ticks(self, ticks: list[dict[str, Any]]) -> None:
        """Deliver a binary tick message (on_message, then on_ticks)."""
        self.events.append(("ticks", len(ticks)))
        if self.on_message is not None:
            self.on_message(self, b"x" * 184, True)
        if self.on_ticks is not None:
            self.on_ticks(self, ticks)

    def simulate_heartbeat(self) -> None:
        """Deliver the 1-byte keep-alive frame (never reaches ``on_ticks``)."""
        if self.on_message is not None:
            self.on_message(self, b"\x01", True)

