"""NIFTY index history via Kite's REST historical-candle endpoint.

Design constraints taken from the provider, not from optimism
------------------------------------------------------------
* The historical endpoint applies **undocumented per-interval range limits** and
  answers with an error rather than truncating. ANJANEYA therefore plans its own
  windows from ``capture.historical.max_days_per_request`` - and when the server
  still objects, it **halves the window and retries** (``adaptive_split_on_error``)
  instead of losing the range.
* The same endpoint serves the index, NIFTY futures and active options, but
  **never expired options** - which is why the option dataset is accumulated
  live by the WebSocket listener, and only the index is backfilled here.
* Kite's candle ``date`` is exchange-local (IST). Every timestamp is normalised
  to an aware IST datetime and stored under its ANJANEYA ``timeframe`` label, so
  REST bars and WebSocket candles never end up in different vocabularies.
* ``(timestamp, tradingsymbol, timeframe)`` is the primary key, so re-running a
  backfill is an idempotent upsert rather than a duplicate-insert generator.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from typing import Any

from apps.anjaneeya_capture.instruments import Instrument
from core.constants import DEFAULT_TIMEZONE, get_timezone, interval_to_timeframe
from core.schemas import IndexHistoryRow, QualityFlag, Source

__all__ = [
    "FetchResult",
    "FetchTarget",
    "FetchWindow",
    "IndexHistoryFetcher",
    "parse_candles",
    "plan_windows",
    "split_window",
]


@dataclass(frozen=True, slots=True)
class FetchTarget:
    """One instrument to backfill, with the symbol its rows carry."""

    instrument_token: int
    tradingsymbol: str

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> FetchTarget:
        """Build a target from a resolved instrument."""
        return cls(
            instrument_token=instrument.instrument_token,
            tradingsymbol=instrument.tradingsymbol,
        )


@dataclass(frozen=True, slots=True)
class FetchWindow:
    """An inclusive ``[start, end]`` window for one historical request."""

    start: datetime
    end: datetime

    @property
    def days(self) -> float:
        """Window length in days (rounded, for logging)."""
        return round((self.end - self.start).total_seconds() / 86400, 3)

    def describe(self) -> dict[str, str | float]:
        """Log-safe summary."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


@dataclass(slots=True)
class FetchResult:
    """Outcome of a backfill/top-up: the rows plus enough detail to audit them."""

    rows: list[IndexHistoryRow] = field(default_factory=list)
    windows_planned: int = 0
    requests: int = 0
    splits: int = 0
    failures: list[str] = field(default_factory=list)
    rejected: int = 0
    intervals: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every planned request eventually succeeded."""
        return not self.failures

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "rows": len(self.rows),
            "windows_planned": self.windows_planned,
            "requests": self.requests,
            "splits": self.splits,
            "rejected_candles": self.rejected,
            "failures": len(self.failures),
            "symbols": list(self.symbols),
            "intervals": list(self.intervals),
        }


def plan_windows(start: datetime, end: datetime, *, max_days: int) -> list[FetchWindow]:
    """Split ``[start, end]`` into consecutive windows of at most ``max_days`` days.

    Windows are contiguous and non-overlapping; a boundary candle may be fetched
    twice, which is harmless because the write is an idempotent upsert.
    """
    if max_days <= 0:
        raise ValueError(f"max_days must be > 0, got {max_days}")
    if end < start:
        return []
    windows: list[FetchWindow] = []
    cursor = start
    step = timedelta(days=max_days)
    while cursor <= end:
        window_end = min(cursor + step, end)
        windows.append(FetchWindow(start=cursor, end=window_end))
        if window_end >= end:
            break
        cursor = window_end + timedelta(seconds=1)
    return windows


def split_window(window: FetchWindow) -> tuple[FetchWindow, FetchWindow] | None:
    """Halve a window for the adaptive retry path.

    Returns ``None`` when the window is a single day or shorter: halving further
    could not change the server's mind and would loop forever.
    """
    span = window.end - window.start
    if span < timedelta(days=2):
        return None
    middle = window.start + span / 2
    left = FetchWindow(start=window.start, end=middle)
    right = FetchWindow(start=middle + timedelta(seconds=1), end=window.end)
    if right.start > right.end:
        return None
    return left, right


def _to_ist(value: Any, tz: Any) -> datetime | None:
    """Normalise a candle timestamp to an aware IST datetime.

    Kite returns exchange-local (IST) values, so a *naive* value is interpreted
    as IST - the opposite of the WebSocket case, where the SDK converted an epoch
    through the machine's own timezone.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime.combine(value, clock_time(0, 0))
    elif isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment.replace(tzinfo=tz) if moment.tzinfo is None else moment.astimezone(tz)


def _number(value: Any) -> float | None:
    """Coerce a candle field to ``float`` (``None`` when absent/malformed)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    """Coerce a candle field to ``int`` (``None`` when absent/malformed)."""
    number = _number(value)
    return None if number is None else int(number)


def parse_candles(
    raw: Iterable[Any],
    *,
    interval: str,
    tradingsymbol: str,
    run_id: str,
    fetched_at: datetime,
    tz: Any = None,
    window: FetchWindow | None = None,
) -> tuple[list[IndexHistoryRow], int]:
    """Convert raw Kite candles into :class:`IndexHistoryRow` objects.

    Returns ``(rows, rejected)``. A malformed or impossible candle is **counted
    and skipped**, never allowed to abort a 400-day backfill - but the count is
    surfaced, so a silently wrong dataset is impossible.
    """
    tz = tz or get_timezone(DEFAULT_TIMEZONE)
    timeframe = interval_to_timeframe(interval)
    rows: list[IndexHistoryRow] = []
    rejected = 0

    for candle in raw:
        if not isinstance(candle, dict):
            rejected += 1
            continue
        moment = _to_ist(candle.get("date"), tz)
        if moment is None:
            rejected += 1
            continue
        if window is not None and not (window.start <= moment <= window.end):
            # The endpoint can return a candle just outside the requested range
            # (inclusive-boundary effects). Keeping it would make the planned
            # windows disagree with the stored data.
            rejected += 1
            continue
        open_ = _number(candle.get("open"))
        high = _number(candle.get("high"))
        low = _number(candle.get("low"))
        close = _number(candle.get("close"))
        if None in (open_, high, low, close) or not all(
            value is not None and value >= 0 for value in (open_, high, low, close)
        ):
            rejected += 1
            continue
        try:
            rows.append(
                IndexHistoryRow(
                    timestamp=moment,
                    tradingsymbol=tradingsymbol,
                    timeframe=timeframe,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=_integer(candle.get("volume")),
                    oi=_integer(candle.get("oi")),
                    source=Source.ZERODHA_REST,
                    fetch_timestamp=max(fetched_at, moment),
                    quality_flag=QualityFlag.OK,
                    run_id=run_id,
                )
            )
        except ValueError:
            rejected += 1
    return rows, rejected



# ---------------------------------------------------------------------------
# The fetcher
# ---------------------------------------------------------------------------
class IndexHistoryFetcher:
    """Fetch NIFTY index candles, window by window, with adaptive retry.

    The fetcher *produces* rows; persisting them is the writer's job. That split
    keeps the retry/rate-limit logic testable without a database, and lets
    ``--dry-run`` show exactly which requests would be made.
    """

    def __init__(
        self,
        client: Any,
        *,
        historical_config: Any,
        run_id: str,
        logger: Any = None,
        now_fn: Callable[[], datetime] | None = None,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> None:
        self.client = client
        self.config = historical_config
        self.run_id = run_id
        self.logger = logger
        self._now = now_fn or (lambda: datetime.now(get_timezone(timezone_name)))
        self.timezone = get_timezone(timezone_name)

    # -- internals ----------------------------------------------------------
    def _fetch_window(
        self,
        target: FetchTarget,
        interval: str,
        window: FetchWindow,
        result: FetchResult,
        *,
        depth: int,
    ) -> None:
        """Fetch one window, splitting it on failure when that is allowed."""
        fetched_at = self._now()
        result.requests += 1
        try:
            raw = self.client.historical_data(
                target.instrument_token,
                window.start,
                window.end,
                interval,
                continuous=False,
                oi=False,
            )
        except Exception as exc:  # noqa: BLE001 - the retry policy decides what is fatal
            split = split_window(window) if self.config.adaptive_split_on_error else None
            if split is not None and depth < self.config.max_split_depth:
                result.splits += 1
                if self.logger is not None:
                    self.logger.warning(
                        f"historical fetch failed for {target.tradingsymbol} {interval} "
                        f"{window.start:%Y-%m-%d}..{window.end:%Y-%m-%d} "
                        f"({type(exc).__name__}: {exc}); halving the window "
                        f"(split {depth + 1}/{self.config.max_split_depth})"
                    )
                left, right = split
                self._fetch_window(target, interval, left, result, depth=depth + 1)
                self._fetch_window(target, interval, right, result, depth=depth + 1)
                return
            failure = (
                f"{target.tradingsymbol} {interval} {window.start:%Y-%m-%d}..{window.end:%Y-%m-%d}: "
                f"{type(exc).__name__}: {exc}"
            )
            result.failures.append(failure)
            if self.logger is not None:
                self.logger.error(f"historical fetch failed permanently: {failure}")
            return

        rows, rejected = parse_candles(
            raw,
            interval=interval,
            tradingsymbol=target.tradingsymbol,
            run_id=self.run_id,
            fetched_at=fetched_at,
            tz=self.timezone,
            window=window,
        )
        result.rows.extend(rows)
        result.rejected += rejected
        if rejected and self.logger is not None:
            self.logger.warning(
                f"{target.tradingsymbol} {interval}: rejected {rejected} candle(s) in "
                f"{window.start:%Y-%m-%d}..{window.end:%Y-%m-%d} (malformed or out of range)"
            )


    def fetch_range(
        self,
        target: FetchTarget,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        """Fetch ``[start, end]`` for one target/interval, window by window."""
        if interval not in self.config.max_days_per_request:
            raise ValueError(
                f"interval {interval!r} has no configured window size in "
                f"capture.historical.max_days_per_request "
                f"({sorted(self.config.max_days_per_request)})"
            )
        result = FetchResult()
        result.symbols.append(target.tradingsymbol)
        result.intervals.append(interval)

        windows = plan_windows(start, end, max_days=self.config.window_days(interval))
        result.windows_planned += len(windows)
        if not windows:
            if self.logger is not None:
                self.logger.info(
                    f"nothing to fetch for {target.tradingsymbol} {interval}: empty range"
                )
            return result
        if self.logger is not None:
            self.logger.info(
                f"fetching {target.tradingsymbol} {interval}: {len(windows)} window(s) "
                f"{start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} "
                f"(<= {self.config.window_days(interval)} days each)"
            )
        for window in windows:
            self._fetch_window(target, interval, window, result, depth=0)
        return result

    # -- public API ---------------------------------------------------------
    def backfill(
        self,
        targets: Iterable[FetchTarget],
        *,
        days: int,
        interval: str,
        now: datetime | None = None,
    ) -> FetchResult:
        """Fetch the full research history (``index_history.backfill_days``)."""
        moment = now or self._now()
        return self._fetch_all(
            targets, interval=interval, start=moment - timedelta(days=days), end=moment
        )

    def top_up(
        self,
        targets: Iterable[FetchTarget],
        *,
        days: int,
        interval: str,
        now: datetime | None = None,
    ) -> FetchResult:
        """Re-fetch the last ``days`` days to heal any gap before a session."""
        moment = now or self._now()
        return self._fetch_all(
            targets, interval=interval, start=moment - timedelta(days=days), end=moment
        )

    def _fetch_all(
        self,
        targets: Iterable[FetchTarget],
        *,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> FetchResult:
        """Run :meth:`fetch_range` for several targets and merge the results."""
        merged = FetchResult()
        for target in targets:
            part = self.fetch_range(target, interval, start, end)
            merged.rows.extend(part.rows)
            merged.windows_planned += part.windows_planned
            merged.requests += part.requests
            merged.splits += part.splits
            merged.rejected += part.rejected
            merged.failures.extend(part.failures)
            for symbol in part.symbols:
                if symbol not in merged.symbols:
                    merged.symbols.append(symbol)
            for name in part.intervals:
                if name not in merged.intervals:
                    merged.intervals.append(name)
        return merged

    def plan(
        self,
        targets: Iterable[FetchTarget],
        *,
        days: int,
        interval: str,
        now: datetime | None = None,
    ) -> list[str]:
        """Describe the requests a backfill would make, without calling the API.

        This is what ``--dry-run`` prints: the request budget (and therefore the
        rate-limit cost) is visible before anything is fetched.
        """
        moment = now or self._now()
        planned: list[str] = []
        for target in targets:
            for window in plan_windows(
                moment - timedelta(days=days), moment, max_days=self.config.window_days(interval)
            ):
                planned.append(
                    f"{target.tradingsymbol} [{target.instrument_token}] {interval} "
                    f"{window.start:%Y-%m-%d %H:%M} .. {window.end:%Y-%m-%d %H:%M}"
                )
        return planned

