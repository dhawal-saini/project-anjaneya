"""WebSocket tick capture: the live feed (spec 4.4 / 4.6).

Provider facts this module is built around (all verified against the installed
``kiteconnect`` SDK, not assumed):

* ``KiteTicker`` receives packets of different lengths per mode - ltp 8 bytes,
  quote 44, full 184, and **index** packets 28/32 bytes. Only the full packet
  carries OI and market depth.
* The tick payload has **no ``bid``/``ask`` fields**: the two-sided quote must be
  derived from ``depth["buy"][0]`` / ``depth["sell"][0]``. Index packets have no
  depth at all, so an index tick legitimately has no bid/ask.
* ``exchange_timestamp`` is built by the SDK with ``datetime.fromtimestamp()``,
  i.e. it is a **naive datetime in the *machine's* local timezone**, not in IST.
  Localising it directly to IST would silently shift every tick by the machine's
  offset. :func:`localize_exchange_timestamp` re-interprets the value through the
  machine's own zone first, and the round trip is then exact.
* The SDK sends a 1-byte keep-alive frame which never reaches ``on_ticks``
  (it is filtered by ``len(payload) > 4``). Liveness is therefore tracked on the
  ``on_message`` callback - connectivity, not market activity - so a *quiet
  market* is never mistaken for a dead socket.
* The SDK re-subscribes automatically after a reconnect (``_on_open`` ->
  ``resubscribe``), replaying token *and* mode. We do not duplicate that, but we
  do own the watchdog that restarts the socket when the SDK has given up.

Importing this module performs no I/O and does not import ``kiteconnect``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apps.anjaneeya_capture.instruments import Instrument, InstrumentMaster
from core.constants import DEFAULT_TIMEZONE, get_timezone
from core.schemas import OptionType, QualityFlag, Source, TickRow

__all__ = [
    "CLOCK_SKEW_TOLERANCE_SECONDS",
    "ContractMeta",
    "ConversionResult",
    "KiteTickerAdapter",
    "TickStream",
    "batch_tokens",
    "localize_exchange_timestamp",
    "metadata_for",
    "ticks_to_rows",
]

#: Machine clock skew beyond which a tick is marked SUSPECT (seconds).
CLOCK_SKEW_TOLERANCE_SECONDS = 2.0

#: How many conversion failures are logged in full before we switch to counting.
REJECTION_LOG_SAMPLE = 5


def _machine_timezone() -> Any:
    """The machine's current local timezone.

    ``datetime.fromtimestamp()`` (which the SDK uses) produces a naive datetime
    in exactly this zone, so this is the zone such a value must be read in.
    """
    return datetime.now().astimezone().tzinfo


def localize_exchange_timestamp(
    value: Any,
    *,
    machine_tz: Any = None,
    tz: Any = None,
) -> datetime | None:
    """Convert an SDK timestamp to an aware IST datetime.

    The SDK builds ``exchange_timestamp`` with ``datetime.fromtimestamp(epoch)``,
    which yields a naive datetime in the machine's local zone. Re-interpreting it
    through that same zone and converting to IST is lossless; on a machine already
    running IST it is a no-op. An already-aware value is simply converted.
    """
    if not isinstance(value, datetime):
        return None
    target = tz or get_timezone(DEFAULT_TIMEZONE)
    if value.tzinfo is not None:
        return value.astimezone(target)
    source = machine_tz or _machine_timezone()
    return value.replace(tzinfo=source).astimezone(target)


@dataclass(frozen=True, slots=True)
class ContractMeta:
    """What a tick needs to know about the instrument it belongs to.

    Kite's tick payload carries only the token, so the contract identity
    (expiry, strike, option type) must come from the instrument master - joined
    here rather than guessed later from the symbol string.
    """

    instrument_token: int
    tradingsymbol: str
    exchange: str | None = None
    expiry: Any = None
    strike: float | None = None
    option_type: OptionType | None = None
    lot_size: int | None = None
    tick_size: float | None = None
    is_index: bool = False

    @property
    def quote_key(self) -> str:
        """``"NFO:NIFTY26OCT25000CE"`` - the key REST quotes are returned under."""
        if self.exchange:
            return f"{self.exchange}:{self.tradingsymbol}"
        return self.tradingsymbol

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> ContractMeta:
        """Build metadata from a resolved instrument.

        Kite's dump reports ``strike = 0`` for index and futures rows, and a futures
        row carries an expiry while having no option type. Copying those through
        verbatim makes the row contract see a *partial* option identity
        ("must carry expiry, strike and option_type together") and **reject every
        index and futures tick**. A non-option is therefore normalised here to
        "no expiry, no strike, no option type", which is what it actually is.
        """
        is_option = instrument.option_type is not None
        return cls(
            instrument_token=instrument.instrument_token,
            tradingsymbol=instrument.tradingsymbol,
            exchange=instrument.exchange,
            expiry=instrument.expiry if is_option else None,
            strike=instrument.strike if is_option else None,
            option_type=instrument.option_type,
            lot_size=instrument.lot_size,
            tick_size=instrument.tick_size,
            is_index=instrument.is_index,
        )


def metadata_for(
    master: InstrumentMaster,
    tokens: Iterable[int],
    *,
    symbols: dict[int, str] | None = None,
) -> dict[int, ContractMeta]:
    """Build ``token -> ContractMeta`` for every token we subscribe to.

    A token missing from the master is still included (named from ``symbols``) so
    its ticks can be stored rather than dropped - just without an option
    identity, which the row contract then treats as index/other.
    """
    symbols = symbols or {}
    metadata: dict[int, ContractMeta] = {}
    for token in tokens:
        instrument = master.get(token)
        if instrument is not None:
            metadata[token] = ContractMeta.from_instrument(instrument)
            continue
        metadata[token] = ContractMeta(
            instrument_token=token,
            tradingsymbol=symbols.get(token, f"TOKEN{token}"),
        )
    return metadata


def batch_tokens(tokens: Sequence[int], *, size: int) -> list[list[int]]:
    """Split tokens into subscribe batches of at most ``size``.

    The protocol accepts a list per message; batching keeps a several-thousand
    instrument subscription from arriving as one giant frame.
    """
    if size <= 0:
        raise ValueError(f"batch size must be > 0, got {size}")
    unique = list(dict.fromkeys(int(token) for token in tokens))
    return [unique[index : index + size] for index in range(0, len(unique), size)]



# ---------------------------------------------------------------------------
# Tick -> row conversion
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ConversionResult:
    """Outcome of converting one binary tick message."""

    rows: list[TickRow] = field(default_factory=list)
    rejected: int = 0
    out_of_session: int = 0
    suspect: int = 0
    unknown_tokens: int = 0
    #: ``exchange - received`` for the last tick that carried an exchange stamp;
    #: positive means the local clock is behind. Reported, never used to alter a bar.
    clock_offset_seconds: float | None = None
    newest_timestamp: datetime | None = None
    newest_price: float | None = None
    newest_reference_price: float | None = None
    rejections: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Rows produced plus rows rejected."""
        return len(self.rows) + self.rejected


def _best_quote(depth: Any) -> tuple[float | None, int | None, float | None, int | None]:
    """Extract ``(bid, bid_qty, ask, ask_qty)`` from a market-depth block.

    The tick payload has no explicit bid/ask, so these come from the top of the
    book. An empty side (possible for an illiquid strike) yields ``None`` rather
    than a fabricated price.
    """
    if not isinstance(depth, dict):
        return (None, None, None, None)

    def top(side: str) -> tuple[float | None, int | None]:
        levels = depth.get(side) or []
        if not levels or not isinstance(levels[0], dict):
            return (None, None)
        price = levels[0].get("price")
        quantity = levels[0].get("quantity")
        return (
            float(price) if isinstance(price, (int, float)) else None,
            int(quantity) if isinstance(quantity, (int, float)) else None,
        )

    bid, bid_qty = top("buy")
    ask, ask_qty = top("sell")
    return (bid, bid_qty, ask, ask_qty)


def _number(value: Any, *, default: float | None = None) -> float | None:
    """Coerce a tick field to ``float``."""
    if isinstance(value, bool) or value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any) -> int | None:
    """Coerce a tick field to ``int`` (``None``/malformed -> ``None``)."""
    number = _number(value)
    return None if number is None else int(number)


def ticks_to_rows(
    ticks: Iterable[Any],
    *,
    metadata: dict[int, ContractMeta],
    run_id: str,
    received_at: datetime,
    tz: Any = None,
    machine_tz: Any = None,
    in_session: Callable[[datetime], bool] | None = None,
    drop_out_of_session: bool = False,
    reference_token: int | None = None,
    trust: str = "exchange",
    tolerance_seconds: float = CLOCK_SKEW_TOLERANCE_SECONDS,
) -> ConversionResult:
    """Convert one tick message into storable rows.

    Everything the row contract requires is either in the payload or joined from
    ``metadata``; anything missing is *reported* rather than invented:

    * ``timestamp``    -> ``exchange_timestamp`` (IST-localised), else
      ``last_trade_time``, else the local receive time (flagged ``SUSPECT``);
    * ``bid``/``ask``  -> top of ``depth`` (``None`` when a side is empty);
    * ``oi``           -> full-mode fields only (``None`` in ltp/quote modes);
    * ``quality_flag`` -> ``OK``, ``SUSPECT`` (no exchange time, or - only when
      ``clock_skew.trust`` is ``machine`` - an exchange stamp more than
      ``tolerance_seconds`` ahead of the local clock) or ``OUT_OF_SESSION``.
    """
    target = tz or get_timezone(DEFAULT_TIMEZONE)
    session_check = in_session or (lambda moment: True)
    result = ConversionResult()

    for tick in ticks:
        if not isinstance(tick, dict):
            result.rejected += 1
            result.rejections.append(f"not a mapping: {type(tick).__name__}")
            continue
        token = _integer(tick.get("instrument_token"))
        if token is None:
            result.rejected += 1
            result.rejections.append("tick without instrument_token")
            continue
        meta = metadata.get(token)
        if meta is None:
            # An unexpected token must be visible: it usually means the universe
            # changed under us, or two sessions are sharing one API key.
            result.unknown_tokens += 1
            meta = ContractMeta(instrument_token=token, tradingsymbol=f"TOKEN{token}")

        last_price = _number(tick.get("last_price"))
        if last_price is None or last_price < 0:
            result.rejected += 1
            result.rejections.append(f"{meta.tradingsymbol}: missing/invalid last_price")
            continue

        exchange_time = localize_exchange_timestamp(
            tick.get("exchange_timestamp"), machine_tz=machine_tz, tz=target
        ) or localize_exchange_timestamp(
            tick.get("last_trade_time"), machine_tz=machine_tz, tz=target
        )
        quality = QualityFlag.OK
        if exchange_time is None:
            # No exchange timestamp at all: the receive clock is the only evidence,
            # and it is exactly the clock we do not trust - so the row is flagged.
            quality = QualityFlag.SUSPECT
            timestamp = received_at
        else:
            timestamp = exchange_time
            result.clock_offset_seconds = (exchange_time - received_at).total_seconds()
            # `trust="exchange"` (the default) only flags the *machine* clock: a tick
            # carrying an exchange stamp is good regardless of how far the PC clock
            # has drifted. `trust="machine"` keeps the old strict rule.
            if trust == "machine" and result.clock_offset_seconds > tolerance_seconds:
                quality = QualityFlag.SUSPECT

        if not session_check(timestamp):
            result.out_of_session += 1
            if drop_out_of_session:
                continue
            quality = QualityFlag.OUT_OF_SESSION
        if quality is QualityFlag.SUSPECT:
            result.suspect += 1

        bid, bid_qty, ask, ask_qty = _best_quote(tick.get("depth"))
        if bid is not None and ask is not None and bid > ask:
            # A crossed book is a provider artefact: drop the prices rather than
            # store an impossible quote.
            bid, bid_qty, ask, ask_qty = (None, None, None, None)
        try:
            row = TickRow(
                timestamp=timestamp,
                instrument_token=token,
                tradingsymbol=meta.tradingsymbol,
                expiry=meta.expiry,
                strike=meta.strike,
                option_type=meta.option_type,
                last_price=last_price,
                volume_traded=_integer(tick.get("volume_traded")),
                oi=_integer(tick.get("oi")),
                bid=bid,
                ask=ask,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                source=Source.ZERODHA_WEBSOCKET,
                fetch_timestamp=max(received_at, timestamp),
                quality_flag=quality,
                run_id=run_id,
            )
        except ValueError as exc:
            result.rejected += 1
            result.rejections.append(f"{meta.tradingsymbol}: {exc}")
            continue

        result.rows.append(row)
        if result.newest_timestamp is None or timestamp >= result.newest_timestamp:
            result.newest_timestamp = timestamp
            result.newest_price = last_price
        if reference_token is not None and token == reference_token:
            result.newest_reference_price = last_price
    return result



# ---------------------------------------------------------------------------
# The SDK adapter
# ---------------------------------------------------------------------------
class KiteTickerAdapter:
    """A thin, replaceable wrapper around ``kiteconnect.KiteTicker``.

    Everything the SDK owns (socket, reconnect loop, resubscribe replay) stays
    with the SDK; this class exists so the rest of the capture code never touches
    the SDK directly - which is what makes the listener testable offline.
    """

    def __init__(
        self,
        *,
        api_key: str,
        access_token: str,
        mode: str,
        websocket_config: Any,
        logger: Any = None,
        ticker: Any = None,
    ) -> None:
        self.mode = mode
        self.logger = logger
        self.config = websocket_config
        if ticker is None:
            from kiteconnect import KiteTicker  # noqa: PLC0415 - keep the import lazy

            ticker = KiteTicker(
                api_key,
                access_token,
                reconnect=True,
                reconnect_max_tries=websocket_config.max_reconnect_attempts,
                reconnect_max_delay=websocket_config.reconnect_max_seconds,
                connect_timeout=websocket_config.connect_timeout_seconds,
            )
        self._ticker = ticker

    @property
    def ticker(self) -> Any:
        """The underlying SDK object (for diagnostics and tests)."""
        return self._ticker

    @property
    def subscribed_tokens(self) -> dict[int, str]:
        """Tokens the SDK believes it is streaming, with their modes."""
        return dict(getattr(self._ticker, "subscribed_tokens", {}) or {})

    def set_callbacks(
        self,
        *,
        on_ticks: Callable[..., Any],
        on_message: Callable[..., Any],
        on_connect: Callable[..., Any],
        on_close: Callable[..., Any],
        on_error: Callable[..., Any],
        on_reconnect: Callable[..., Any],
        on_noreconnect: Callable[..., Any],
    ) -> None:
        """Register every callback the SDK supports for a streaming session."""
        self._ticker.on_ticks = on_ticks
        self._ticker.on_message = on_message
        self._ticker.on_connect = on_connect
        self._ticker.on_close = on_close
        self._ticker.on_error = on_error
        self._ticker.on_reconnect = on_reconnect
        self._ticker.on_noreconnect = on_noreconnect

    def connect(self) -> None:
        """Start the socket in the SDK's own thread (it blocks otherwise)."""
        self._ticker.connect(threaded=True)

    def close(self) -> None:
        """Close the socket; the SDK's factory stops retrying afterwards."""
        close = getattr(self._ticker, "close", None)
        if callable(close):
            close()

    def subscribe(self, tokens: Sequence[int], *, batch_size: int) -> int:
        """Subscribe in batches and apply the configured mode to each batch.

        ``subscribe()`` alone marks tokens as QUOTE in the SDK's bookkeeping, so
        ``set_mode`` must follow it - otherwise a "full" session would silently
        receive 44-byte packets and every tick would lack OI and depth.
        """
        subscribed = 0
        for batch in batch_tokens(list(tokens), size=batch_size):
            self._ticker.subscribe(batch)
            self._ticker.set_mode(self.mode, batch)
            subscribed += len(batch)
        return subscribed

    def unsubscribe(self, tokens: Sequence[int], *, batch_size: int) -> int:
        """Unsubscribe in batches."""
        removed = 0
        for batch in batch_tokens(list(tokens), size=batch_size):
            self._ticker.unsubscribe(batch)
            removed += len(batch)
        return removed

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {"mode": self.mode, "subscribed": len(self.subscribed_tokens)}



# ---------------------------------------------------------------------------
# The stream orchestrator
# ---------------------------------------------------------------------------
class TickStream:
    """Owns one WebSocket session: subscription, conversion, watchdog, bookkeeping.

    The class is deliberately thread-free: it reacts to callbacks (which arrive on
    the SDK's own thread) and exposes a :meth:`monitor` step that the session's
    loop calls on a timer. That makes the reconnect/stall logic testable by
    driving a fake adapter synchronously - no sleeping, no flakiness.
    """

    def __init__(
        self,
        *,
        adapter_factory: Callable[[], Any],
        metadata: dict[int, ContractMeta],
        run_id: str,
        mode: str,
        websocket_config: Any,
        capture_config: Any,
        writer: Any,
        tracker: Any,
        heartbeat: Any = None,
        logger: Any = None,
        clock: Callable[[], datetime] | None = None,
        is_open: Callable[[datetime], bool] | None = None,
        timezone_name: str = DEFAULT_TIMEZONE,
        machine_tz: Any = None,
        reference_token: int | None = None,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._metadata = dict(metadata)
        self.run_id = run_id
        self.mode = mode
        self.config = websocket_config
        self.capture = capture_config
        skew = getattr(websocket_config, "clock_skew", None)
        self._clock_trust = str(getattr(skew, "trust", "exchange"))
        self._clock_tolerance = float(
            getattr(skew, "tolerance_seconds", CLOCK_SKEW_TOLERANCE_SECONDS)
        )
        self._report_clock_offset = bool(getattr(skew, "report_offset", True))
        #: Smoothed ``exchange - received`` in seconds (positive = local clock behind).
        self.clock_offset_seconds: float | None = None
        self.writer = writer
        self.tracker = tracker
        self.heartbeat = heartbeat
        self.logger = logger
        self.tz = get_timezone(timezone_name)
        self._clock = clock or (lambda: datetime.now(self.tz))
        self._is_open = is_open or (lambda moment: True)
        self._machine_tz = machine_tz
        self.reference_token = reference_token

        self.adapter: Any = None
        self.subscribed: set[int] = set()
        self.connected = False
        self.ticks_total = 0
        self.messages = 0
        self.reconnects = 0
        self.restarts = 0
        self.stalls = 0
        self.rejected = 0
        self.suspect = 0
        self.unknown_tokens = 0
        self.out_of_session = 0
        self.gap_logs = 0
        self.last_reference_price: float | None = None
        self._last_message_at: datetime | None = None
        self._last_connect_at: datetime | None = None
        self._backoff_seconds = websocket_config.reconnect_initial_seconds
        self._stopped = False
        #: Set by :meth:`handle_connect`; :meth:`start` waits on it before returning,
        #: because ``KiteTicker.connect(threaded=True)`` returns before the socket is
        #: open and ``subscribe()`` dereferences it.
        self._connected_event = threading.Event()
        self._pending_tokens: tuple[int, ...] = ()
        self._subscribe_error: Exception | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self, tokens: Sequence[int]) -> None:
        """Create the adapter, register callbacks, connect, and subscribe once live.

        ``KiteTicker.connect(threaded=True)`` returns **before** the socket is open,
        and subscribing then dereferences ``self.ws`` while it is still ``None``
        (``AttributeError: 'NoneType' object has no attribute 'sendMessage'``), so no
        token is ever requested and the session streams nothing. The subscription is
        therefore issued from :meth:`handle_connect` - the SDK's own "connection is
        live" callback - and this method only waits for that moment, bounded by
        ``websocket.connect_timeout_seconds``.
        """
        self._stopped = False
        self._pending_tokens = tuple(int(token) for token in tokens)
        self._subscribe_error = None
        self._connected_event.clear()
        self.adapter = self._adapter_factory()
        self.adapter.set_callbacks(
            on_ticks=self.handle_ticks,
            on_message=self.handle_message,
            on_connect=self.handle_connect,
            on_close=self.handle_close,
            on_error=self.handle_error,
            on_reconnect=self.handle_reconnect,
            on_noreconnect=self.handle_noreconnect,
        )
        self.adapter.connect()
        self._last_connect_at = self._clock()
        timeout = float(getattr(self.config, "connect_timeout_seconds", 30.0) or 30.0)
        if not self._connected_event.wait(timeout=timeout):
            raise RuntimeError(
                f"the websocket did not connect within {timeout:g}s; nothing was subscribed"
            )
        if self._subscribe_error is not None:
            raise RuntimeError(
                f"the websocket connected but the subscription failed: {self._subscribe_error}"
            ) from self._subscribe_error

    def stop(self) -> None:
        """Close the socket (idempotent)."""
        self._stopped = True
        self._pending_tokens = ()
        self._connected_event.clear()
        if self.adapter is not None:
            self.adapter.close()
        self.connected = False

    # -- subscription -------------------------------------------------------
    def subscribe(self, tokens: Sequence[int]) -> int:
        """Subscribe to ``tokens`` (batched), recording them for bookkeeping."""
        if self.adapter is None:
            raise RuntimeError("TickStream.start() must be called before subscribing")
        subscribed = self.adapter.subscribe(list(tokens), batch_size=self.config.max_instruments)
        self.subscribed.update(int(token) for token in tokens)
        if self.heartbeat is not None:
            self.heartbeat.update(subscribed_instruments=len(self.subscribed))
        if self.logger is not None:
            self.logger.info(
                f"subscribed to {subscribed} instrument(s) in mode {self.mode!r} "
                f"(total {len(self.subscribed)})"
            )
        return subscribed

    def unsubscribe(self, tokens: Sequence[int]) -> int:
        """Unsubscribe from ``tokens`` (batched)."""
        if self.adapter is None:
            return 0
        removed = self.adapter.unsubscribe(list(tokens), batch_size=self.config.max_instruments)
        self.subscribed.difference_update(int(token) for token in tokens)
        if self.heartbeat is not None:
            self.heartbeat.update(subscribed_instruments=len(self.subscribed))
        return removed

    def refresh_universe(
        self,
        tokens: Sequence[int],
        *,
        metadata: dict[int, ContractMeta] | None = None,
        reference_token: int | None = None,
    ) -> dict[str, Any]:
        """Re-centre the subscription on a newly resolved universe.

        Ticks for tokens that have just been dropped are still storable: the
        metadata map is *merged*, never pruned, so an in-flight packet for an old
        strike is stored under its real symbol instead of an invented one.
        """
        if metadata:
            self._metadata.update(metadata)
        if reference_token is not None:
            self.reference_token = reference_token
        wanted = {int(token) for token in tokens}
        added = sorted(wanted - self.subscribed)
        removed = sorted(self.subscribed - wanted)
        if removed:
            self.unsubscribe(removed)
        if added:
            self.subscribe(added)
        summary = {"added": added, "removed": removed, "total": len(self.subscribed)}
        if self.logger is not None and (added or removed):
            self.logger.info(
                f"universe refreshed: +{len(added)}/-{len(removed)} instruments "
                f"(now {len(self.subscribed)})"
            )
        return summary


    # -- callbacks ----------------------------------------------------------
    def handle_message(self, ws: Any = None, payload: Any = None, is_binary: bool = True) -> None:
        """Socket liveness only: any binary frame, including the 1-byte keep-alive.

        ``on_ticks`` never sees the keep-alive frame (the SDK filters short
        payloads), so this callback - not tick arrival - is what tells us the
        socket is alive. A far out-of-the-money option can legitimately produce no
        ticks for minutes; that must not look like a dead socket.
        """
        if not is_binary:
            return
        self.messages += 1
        self._last_message_at = self._clock()
        if self.heartbeat is not None:
            self.heartbeat.update(last_message_at=self._last_message_at, connected=self.connected)

    def handle_ticks(self, ws: Any = None, ticks: Any = None) -> None:
        """Convert a tick message, persist it and update every counter."""
        received_at = self._clock()
        result = ticks_to_rows(
            ticks or [],
            metadata=self._metadata,
            run_id=self.run_id,
            received_at=received_at,
            tz=self.tz,
            machine_tz=self._machine_tz,
            in_session=self._is_open,
            drop_out_of_session=self.capture.out_of_session_ticks == "drop",
            reference_token=self.reference_token,
            trust=self._clock_trust,
            tolerance_seconds=self._clock_tolerance,
        )
        if result.clock_offset_seconds is not None:
            # Smooth it: a per-message value jitters by a second; the drift itself is
            # slow, and the heartbeat is read by a human.
            previous = self.clock_offset_seconds
            self.clock_offset_seconds = (
                result.clock_offset_seconds
                if previous is None
                else 0.9 * previous + 0.1 * result.clock_offset_seconds
            )
        self.rejected += result.rejected
        self.suspect += result.suspect
        self.unknown_tokens += result.unknown_tokens
        self.out_of_session += result.out_of_session

        if result.rejections and self.logger is not None:
            for message in result.rejections[:REJECTION_LOG_SAMPLE]:
                self.logger.warning(f"rejected tick: {message}")
            if len(result.rejections) > REJECTION_LOG_SAMPLE:
                self.logger.warning(
                    f"... {len(result.rejections) - REJECTION_LOG_SAMPLE} further tick "
                    "rejection(s) in this message"
                )

        if not result.rows:
            return
        self.writer.write_ticks(result.rows)
        self.ticks_total += len(result.rows)

        for row in result.rows:
            gapped = self.tracker.record(
                tradingsymbol=row.tradingsymbol,
                timeframe="TICK",
                timestamp=row.timestamp,
                quality_flag=row.quality_flag,
            )
            if gapped:
                self._log_gap(row)
        self._update_heartbeat(result)
        self._log_progress()

    def _log_gap(self, row: TickRow) -> None:
        """Log a detected gap, respecting the per-session log cap."""
        self.gap_logs += 1
        if self.logger is None:
            return
        limit = self.capture.gaps.log_limit_per_session
        if self.gap_logs <= limit:
            self.logger.warning(
                f"tick gap detected on {row.tradingsymbol}: silence longer than "
                f"{self.capture.gaps.tick_gap_seconds}s before {row.timestamp.isoformat()}"
            )
        elif self.gap_logs == limit + 1:
            self.logger.warning(
                f"gap logging cap ({limit}) reached; further gaps are counted in the "
                "coverage ledger only"
            )

    def _update_heartbeat(self, result: ConversionResult) -> None:
        """Publish capture progress into the heartbeat state."""
        if result.newest_reference_price is not None:
            # The anchor price is what places the ATM strike: keep the latest one
            # so the session can re-centre the strike window without a REST call.
            self.last_reference_price = result.newest_reference_price
        if self.heartbeat is None:
            return
        price = result.newest_reference_price
        self.heartbeat.update(
            connected=self.connected,
            ticks_total=self.ticks_total,
            reconnects=self.reconnects,
            buffered_rows=self.writer.buffered_rows,
            last_tick_at=self._clock(),
            newest_tick_timestamp=result.newest_timestamp,
            newest_price=price if price is not None else result.newest_price,
            gap_count=self.total_gaps,
            largest_gap_seconds=self.largest_gap_seconds,
            clock_offset_seconds=(
                round(self.clock_offset_seconds, 3)
                if self._report_clock_offset and self.clock_offset_seconds is not None
                else None
            ),
        )

    def _log_progress(self) -> None:
        """Log a bounded progress line (roughly once per status interval)."""
        if self.logger is None or self.ticks_total == 0:
            return
        interval = max(1, int(self.capture.status_interval_seconds * 10))
        if self.ticks_total % interval == 0:
            self.logger.info(
                f"capture: {self.ticks_total} ticks, {len(self.subscribed)} instruments, "
                f"buffered {self.writer.buffered_rows}, gaps {self.total_gaps}"
            )

    @property
    def total_gaps(self) -> int:
        """Total documented gaps across every tracked stream."""
        return sum(stats.gap_count for stats in self.tracker.streams)

    @property
    def largest_gap_seconds(self) -> float:
        """Largest documented silence across every tracked stream."""
        return max((stats.largest_gap_seconds for stats in self.tracker.streams), default=0.0)


    # -- connection events --------------------------------------------------
    def handle_connect(self, ws: Any = None, response: Any = None) -> None:
        """The SDK reports a live connection (also after its own reconnects)."""
        self.connected = True
        self._last_connect_at = self._clock()
        self._backoff_seconds = self.config.reconnect_initial_seconds
        if self.heartbeat is not None:
            self.heartbeat.update(connected=True)
        if self.logger is not None:
            self.logger.info(f"websocket connected (mode {self.mode!r})")
        # Subscribe here - and only once - because this callback is the first moment
        # `subscribe()` is safe. After a reconnect the SDK replays `subscribed_tokens`
        # itself, so a second call would duplicate every message; `self.subscribed` is
        # what tells the two cases apart.
        if self._pending_tokens and not self.subscribed:
            try:
                self.subscribe(self._pending_tokens)
            except Exception as exc:  # noqa: BLE001 - reported to start(), never swallowed
                self._subscribe_error = exc
                if self.logger is not None:
                    self.logger.error(f"initial subscribe failed: {exc}")
        self._connected_event.set()

    def handle_close(self, ws: Any = None, reason: Any = None) -> None:
        """The socket closed; the SDK will reconnect on its own."""
        self.connected = False
        if self.heartbeat is not None:
            self.heartbeat.update(connected=False)
        if self.logger is not None:
            self.logger.warning(f"websocket closed: {reason!r} (the SDK will reconnect)")

    def handle_error(self, ws: Any = None, error: Any = None) -> None:
        """An SDK-level error: counted and logged, never fatal for the session."""
        if self.heartbeat is not None:
            self.heartbeat.update(errors=self.heartbeat.state.errors + 1)
        if self.logger is not None:
            self.logger.error(f"websocket error: {error!r}")

    def handle_reconnect(self, ws: Any = None, attempts: Any = None) -> None:
        """The SDK is reconnecting (its own backoff, its own resubscribe replay)."""
        self.reconnects += 1
        self.connected = False
        if self.heartbeat is not None:
            self.heartbeat.update(reconnects=self.reconnects, connected=False)
        if self.logger is not None:
            self.logger.warning(f"websocket reconnecting (attempt {attempts})")

    def handle_noreconnect(self, ws: Any = None) -> None:
        """The SDK gave up: the watchdog's next pass restarts the socket."""
        self.connected = False
        if self.heartbeat is not None:
            self.heartbeat.update(connected=False)
            self.heartbeat.note("the SDK exhausted its reconnect attempts")
        if self.logger is not None:
            self.logger.error(
                "websocket reconnect attempts exhausted; the watchdog will restart the socket"
            )

    # -- watchdog -----------------------------------------------------------
    def monitor(self, *, now: datetime | None = None) -> list[str]:
        """One watchdog pass: flush, detect a stalled socket, restart if needed.

        Returns the actions taken, so both the caller and the tests can see what
        happened without parsing log lines.
        """
        moment = now or self._clock()
        actions: list[str] = []

        if self.writer.maybe_flush():
            actions.append("flush")
            if self.heartbeat is not None:
                self.heartbeat.update(last_flush_at=moment, buffered_rows=self.writer.buffered_rows)

        if self.heartbeat is not None:
            self.heartbeat.update(
                buffered_rows=self.writer.buffered_rows,
                connected=self.connected,
                ticks_per_minute=self._ticks_per_minute(moment),
            )

        if self._stopped:
            return actions

        if self._stalled(moment):
            self.stalls += 1
            if self.logger is not None:
                silence = (
                    (moment - self._last_message_at).total_seconds()
                    if self._last_message_at
                    else float("inf")
                )
                self.logger.error(
                    f"websocket stalled: no frame for {silence:.0f}s "
                    f"(timeout {self.config.stall_timeout_seconds}s); restarting the socket"
                )
            if self.heartbeat is not None:
                self.heartbeat.note("restarted the socket after a silent connection")
            self.restart()
            actions.append("stall_restart")
        elif not self.connected and self._backoff_elapsed(moment):
            self.restart()
            actions.append("reconnect")
        return actions

    def _stalled(self, moment: datetime) -> bool:
        """True when the socket has gone silent during session hours.

        ``_last_message_at`` is ``None`` only before the first frame; that counts
        as a stall only once the socket has been up for the full timeout.
        """
        if not self._is_open(moment):
            return False
        reference = self._last_message_at or self._last_connect_at
        if reference is None:
            return False
        return (moment - reference).total_seconds() > self.config.stall_timeout_seconds

    def _backoff_elapsed(self, moment: datetime) -> bool:
        """True when the reconnect backoff has elapsed (exponential, capped)."""
        if self._last_connect_at is None:
            return True
        return (moment - self._last_connect_at).total_seconds() >= self._backoff_seconds


    def restart(self) -> None:
        """Replace the adapter and resubscribe (after a stall, or a give-up).

        Recreating the adapter also resets the SDK's internal attempt counter -
        which is exactly what a long outage needs, since a session must survive
        more than ``reconnect_max_attempts`` consecutive failures.
        """
        tokens = sorted(self.subscribed)
        current_backoff = self._backoff_seconds
        try:
            if self.adapter is not None:
                self.adapter.close()
        except Exception as exc:  # noqa: BLE001 - a broken socket must not block the restart
            if self.logger is not None:
                self.logger.warning(f"error closing the old websocket: {exc}")
        self.adapter = self._adapter_factory()
        self.adapter.set_callbacks(
            on_ticks=self.handle_ticks,
            on_message=self.handle_message,
            on_connect=self.handle_connect,
            on_close=self.handle_close,
            on_error=self.handle_error,
            on_reconnect=self.handle_reconnect,
            on_noreconnect=self.handle_noreconnect,
        )
        self.adapter.connect()
        self._last_connect_at = self._clock()
        self._last_message_at = None
        self.restarts += 1
        self._backoff_seconds = min(
            current_backoff * self.config.reconnect_multiplier, self.config.reconnect_max_seconds
        )
        self.subscribed.clear()
        self.subscribe(tokens)
        if self.logger is not None:
            self.logger.info(
                f"websocket restarted (#{self.restarts}) with {len(tokens)} instrument(s); "
                f"next backoff {self._backoff_seconds:.1f}s"
            )

    def _ticks_per_minute(self, moment: datetime) -> float:
        """Average ticks/minute since the stream connected (0 before the first tick)."""
        started = self._last_connect_at
        if started is None or self.ticks_total == 0:
            return 0.0
        elapsed = max((moment - started).total_seconds(), 1.0)
        return self.ticks_total / (elapsed / 60.0)

    # -- reporting ----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Log-safe summary of the whole stream."""
        return {
            "mode": self.mode,
            "connected": self.connected,
            "subscribed_instruments": len(self.subscribed),
            "messages": self.messages,
            "ticks": self.ticks_total,
            "rejected": self.rejected,
            "suspect": self.suspect,
            "out_of_session": self.out_of_session,
            "unknown_tokens": self.unknown_tokens,
            "reconnects": self.reconnects,
            "restarts": self.restarts,
            "stalls": self.stalls,
            "gaps": self.total_gaps,
            "largest_gap_seconds": round(self.largest_gap_seconds, 3),
        }

