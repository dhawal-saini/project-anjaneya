"""Periodic option-chain polling: the independent OI/quote source.

Why poll when the WebSocket is already streaming?
------------------------------------------------
The spec calls for a second, independent source for OI and two-sided quotes.
Polling gives exactly that: a snapshot per interval that is robust to a WebSocket
outage, and that can *validate* the tick stream instead of trusting it.

Provider facts baked in:

* Kite's REST ``quote()`` is rate limited to roughly one request per second, so
  symbols are batched (``option_chain.quotes_per_request``) and paced by the
  client's own token bucket. A poll must never become a burst.
* **Kite exposes no implied volatility anywhere** (not in the tick payload, not
  in the REST quote). ``iv`` therefore stays ``NULL`` until ANJANEYA derives it
  from price in a later phase - it is never fabricated here.
* A quote is keyed ``"NFO:NIFTY26OCT25000CE"``; contract identity comes from the
  instrument master, so a symbol without identity is reported rather than stored
  half-identified.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apps.anjaneeya_capture.instruments import Instrument
from apps.anjaneeya_capture.storage_writer import SNAPSHOT_TIMEFRAME
from apps.anjaneeya_capture.websocket_listener import ContractMeta, localize_exchange_timestamp
from core.constants import DEFAULT_TIMEZONE, get_timezone
from core.schemas import OptionChainSnapshotRow, QualityFlag, Source

__all__ = [
    "MISSING_LOG_SAMPLE",
    "OptionChainPoller",
    "PollResult",
    "quote_symbols",
    "quotes_to_snapshot_rows",
]

#: How many missing symbols are named in the log before we switch to counting.
MISSING_LOG_SAMPLE = 5

#: Clock skew beyond which a quoted timestamp is marked SUSPECT (seconds).
QUOTE_SKEW_TOLERANCE_SECONDS = 2.0


def quote_symbols(instruments: Iterable[Instrument]) -> list[str]:
    """``"NFO:NIFTY26OCT25000CE"`` keys for the REST quote endpoint."""
    return [instrument.quote_symbol for instrument in instruments]


@dataclass(slots=True)
class PollResult:
    """Outcome of one poll (or of a whole polling loop, when accumulated)."""

    rows: list[OptionChainSnapshotRow] = field(default_factory=list)
    requests: int = 0
    missing: list[str] = field(default_factory=list)
    rejected: int = 0
    failures: list[str] = field(default_factory=list)
    newest_timestamp: datetime | None = None

    @property
    def ok(self) -> bool:
        """True when no request failed."""
        return not self.failures

    def extend(self, other: PollResult) -> None:
        """Accumulate another result into this one."""
        self.rows.extend(other.rows)
        self.requests += other.requests
        self.missing.extend(other.missing)
        self.rejected += other.rejected
        self.failures.extend(other.failures)
        if other.newest_timestamp and (
            self.newest_timestamp is None or other.newest_timestamp > self.newest_timestamp
        ):
            self.newest_timestamp = other.newest_timestamp

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "rows": len(self.rows),
            "requests": self.requests,
            "missing": len(self.missing),
            "rejected": self.rejected,
            "failures": len(self.failures),
        }


def _number(value: Any) -> float | None:
    """Coerce a quote field to ``float``."""
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    """Coerce a quote field to ``int``."""
    number = _number(value)
    return None if number is None else int(number)


def _best_quote(depth: Any) -> tuple[float | None, float | None]:
    """Top-of-book ``(bid, ask)`` from a REST depth block."""
    if not isinstance(depth, dict):
        return (None, None)

    def top(side: str) -> float | None:
        levels = depth.get(side) or []
        if not levels or not isinstance(levels[0], dict):
            return None
        return _number(levels[0].get("price"))

    return (top("buy"), top("sell"))


def quotes_to_snapshot_rows(
    quotes: Mapping[str, Mapping[str, Any]],
    *,
    symbols_by_token: Mapping[int, str],
    metadata: Mapping[int, ContractMeta],
    run_id: str,
    received_at: datetime,
    tz: Any = None,
    machine_tz: Any = None,
) -> PollResult:
    """Convert one REST ``quote()`` response into snapshot rows.

    Only contracts with a full identity (expiry + strike + option type) can be
    stored: the snapshot row contract forbids half-identified options, so a quote
    for an unknown token is counted as rejected with its symbol named.
    """
    target = tz or get_timezone(DEFAULT_TIMEZONE)
    tokens_by_symbol = {symbol: token for token, symbol in symbols_by_token.items()}
    result = PollResult()

    for symbol, quote in quotes.items():
        if not isinstance(quote, Mapping):
            result.rejected += 1
            continue
        token = _integer(quote.get("instrument_token"))
        if token is None:
            token = tokens_by_symbol.get(symbol)
        if token is None:
            result.rejected += 1
            result.missing.append(f"unidentified symbol {symbol}")
            continue
        meta = metadata.get(token)
        if meta is None or meta.expiry is None or meta.strike is None or meta.option_type is None:
            result.rejected += 1
            result.missing.append(f"{symbol}: no contract identity in the instrument master")
            continue

        last_price = _number(quote.get("last_price"))
        if last_price is None or last_price < 0:
            result.rejected += 1
            result.missing.append(f"{symbol}: missing last_price")
            continue

        timestamp = (
            localize_exchange_timestamp(quote.get("timestamp"), machine_tz=machine_tz, tz=target)
            or localize_exchange_timestamp(
                quote.get("last_trade_time"), machine_tz=machine_tz, tz=target
            )
            or received_at
        )
        quality = QualityFlag.OK if quote.get("timestamp") is not None else QualityFlag.SUSPECT
        if (timestamp - received_at).total_seconds() > QUOTE_SKEW_TOLERANCE_SECONDS:
            quality = QualityFlag.SUSPECT

        bid, ask = _best_quote(quote.get("depth"))
        if bid is not None and ask is not None and bid > ask:
            bid, ask = (None, None)
        try:
            row = OptionChainSnapshotRow(
                timestamp=timestamp,
                instrument_token=token,
                tradingsymbol=meta.tradingsymbol,
                expiry=meta.expiry,
                strike=float(meta.strike),
                option_type=meta.option_type,
                last_price=last_price,
                bid=bid,
                ask=ask,
                volume=_integer(quote.get("volume")),
                oi=_integer(quote.get("oi")),
                # Kite publishes no implied volatility on any endpoint: deriving
                # it from price is a later phase's job, never this one's.
                iv=None,
                source=Source.ZERODHA_REST,
                fetch_timestamp=max(received_at, timestamp),
                quality_flag=quality,
                run_id=run_id,
            )
        except ValueError as exc:
            result.rejected += 1
            result.missing.append(f"{symbol}: {exc}")
            continue

        result.rows.append(row)
        if result.newest_timestamp is None or timestamp > result.newest_timestamp:
            result.newest_timestamp = timestamp
    return result



# ---------------------------------------------------------------------------
# The poller
# ---------------------------------------------------------------------------
class OptionChainPoller:
    """Polls the option chain on a fixed interval and stores OI/quote snapshots.

    The poller is synchronous (:meth:`poll_once`) plus a blocking :meth:`run` for
    a worker thread. Both are exposed so tests can drive a single poll
    deterministically while the session uses the loop.
    """

    def __init__(
        self,
        *,
        client: Any,
        symbols_by_token: Mapping[int, str],
        metadata: Mapping[int, ContractMeta],
        run_id: str,
        poll_config: Any,
        capture_config: Any,
        writer: Any,
        tracker: Any,
        heartbeat: Any = None,
        logger: Any = None,
        now_fn: Callable[[], datetime] | None = None,
        is_open: Callable[[datetime], bool] | None = None,
        sleep: Callable[[float], None] | None = None,
        timezone_name: str = DEFAULT_TIMEZONE,
        machine_tz: Any = None,
    ) -> None:
        import time as _time

        self.client = client
        self.symbols_by_token = dict(symbols_by_token)
        self.metadata = dict(metadata)
        self.run_id = run_id
        self.poll_config = poll_config
        self.capture = capture_config
        self.writer = writer
        self.tracker = tracker
        self.heartbeat = heartbeat
        self.logger = logger
        self.tz = get_timezone(timezone_name)
        self._now = now_fn or (lambda: datetime.now(self.tz))
        self._is_open = is_open or (lambda moment: True)
        self._sleep = sleep or _time.sleep
        self._machine_tz = machine_tz

        self.poll_count = 0
        self.failures = 0
        self.rows_written = 0
        self.totals = PollResult()

    # -- introspection ------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        """Every symbol this poller requests, sorted."""
        return sorted(self.symbols_by_token.values())

    def chunks(self) -> list[list[str]]:
        """The request batches implied by ``option_chain.quotes_per_request``."""
        size = self.poll_config.quotes_per_request
        symbols = self.symbols
        return [symbols[index : index + size] for index in range(0, len(symbols), size)]

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "symbols": len(self.symbols_by_token),
            "interval_seconds": self.poll_config.interval_seconds,
            "requests_per_poll": len(self.chunks()),
            "polls": self.poll_count,
            "failures": self.failures,
            "rows": self.rows_written,
        }

    # -- live universe changes ---------------------------------------------
    def update_universe(
        self,
        *,
        symbols_by_token: Mapping[int, str],
        metadata: Mapping[int, ContractMeta],
    ) -> None:
        """Adopt a refreshed universe (kept in sync with the stream's)."""
        self.symbols_by_token = dict(symbols_by_token)
        self.metadata.update(metadata)


    # -- one poll -----------------------------------------------------------
    def poll_once(self, *, now: datetime | None = None, force: bool = False) -> PollResult:
        """Fetch one snapshot of every subscribed contract and store it.

        Partial failure is tolerated by design: one failed request must not cost
        the rest of the snapshots. Every failure is logged and counted.
        """
        moment = now or self._now()
        if not force and not self._is_open(moment):
            return PollResult()

        result = PollResult()
        batches = self.chunks()
        if not batches:
            return result

        for batch in batches:
            result.requests += 1
            try:
                quotes = self.client.quote(*batch) or {}
            except Exception as exc:  # noqa: BLE001 - one bad request must not end the poll
                self.failures += 1
                failure = f"quote({len(batch)} symbols): {type(exc).__name__}: {exc}"
                result.failures.append(failure)
                if self.logger is not None:
                    self.logger.error(f"option-chain poll failed: {failure}")
                continue

            missing = [symbol for symbol in batch if symbol not in quotes]
            if missing and self.logger is not None:
                result.missing.extend(missing)
                sample = ", ".join(missing[:MISSING_LOG_SAMPLE])
                suffix = (
                    f" (+{len(missing) - MISSING_LOG_SAMPLE} more)"
                    if len(missing) > MISSING_LOG_SAMPLE
                    else ""
                )
                self.logger.warning(
                    f"option-chain poll: {len(missing)} of {len(batch)} symbol(s) not quoted, "
                    f"e.g. {sample}{suffix}"
                )
            elif missing:
                result.missing.extend(missing)

            converted = quotes_to_snapshot_rows(
                quotes,
                symbols_by_token=self.symbols_by_token,
                metadata=self.metadata,
                run_id=self.run_id,
                received_at=moment,
                tz=self.tz,
                machine_tz=self._machine_tz,
            )
            result.rows.extend(converted.rows)
            result.rejected += converted.rejected
            result.missing.extend(converted.missing)
            if converted.newest_timestamp and (
                result.newest_timestamp is None
                or converted.newest_timestamp > result.newest_timestamp
            ):
                result.newest_timestamp = converted.newest_timestamp

        self.poll_count += 1
        if result.rows:
            self.writer.write_snapshots(result.rows)
            self.rows_written += len(result.rows)
            for row in result.rows:
                self.tracker.record(
                    tradingsymbol=row.tradingsymbol,
                    timeframe=SNAPSHOT_TIMEFRAME,
                    timestamp=row.timestamp,
                    quality_flag=row.quality_flag,
                )
        self.totals.extend(result)
        if self.heartbeat is not None:
            self.heartbeat.update(
                poll_count=self.poll_count,
                poll_failures=self.failures,
                snapshots_total=self.rows_written,
                buffered_rows=self.writer.buffered_rows,
            )
        if self.logger is not None:
            summary = result.describe()
            self.logger.info(
                f"option-chain poll #{self.poll_count}: {summary['rows']} snapshot(s), "
                f"{summary['requests']} request(s), {summary['missing']} missing, "
                f"{summary['failures']} failure(s)"
            )
        return result

    # -- the loop -----------------------------------------------------------
    def run(self, stop_event: Any, *, max_iterations: int | None = None) -> PollResult:
        """Poll every ``interval_seconds`` until ``stop_event`` is set.

        Sleeping happens in one-second slices so a shutdown request is honoured
        promptly instead of after a full interval.
        """
        accumulated = PollResult()
        iterations = 0
        while not (stop_event is not None and stop_event.is_set()):
            if self._is_open(self._now()):
                accumulated.extend(self.poll_once())
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            self._sleep_until_next_interval(stop_event)
        return accumulated

    def _sleep_until_next_interval(self, stop_event: Any) -> None:
        """Sleep one interval in small slices, waking early when asked to stop."""
        remaining = float(self.poll_config.interval_seconds)
        while remaining > 0:
            if stop_event is not None and stop_event.is_set():
                return
            slice_seconds = min(1.0, remaining)
            self._sleep(slice_seconds)
            remaining -= slice_seconds

    # -- reporting ----------------------------------------------------------
    def to_thread(self, stop_event: Any) -> Any:
        """Run :meth:`run` in a daemon thread and return it (session convenience)."""
        import threading

        thread = threading.Thread(
            target=self.run, args=(stop_event,), name="option-chain-poller", daemon=True
        )
        return thread

