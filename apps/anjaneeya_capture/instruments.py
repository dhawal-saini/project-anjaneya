"""Instrument master, the on-disk dump cache, and option-universe resolution.

Why this module is longer than "download the instruments dump"
-------------------------------------------------------------
Two provider facts shape it:

1. **Symbol strings are not a contract.** NSE periodically changes how NIFTY
   option symbols are built. So ANJANEYA identifies a contract by its
   *attributes* - exchange, underlying name, expiry, strike, option type - and
   looks the ``instrument_token`` up in Kite's daily dump. The configured
   ``symbol_templates`` are used only as a *cross-check*: when a rendered symbol
   disagrees with the dump we log a warning that NSE's format changed, instead
   of silently subscribing to nothing.
2. **The dump changes daily and is rate-limited.** It is cached as JSON
   (``data/raw/instruments/<EXCHANGE>_<YYYY-MM-DD>.json``) and re-downloaded only
   when the cache is older than ``capture.instruments.refresh_ttl_hours``. That
   keeps ``--mode instruments``, ``--mode backfill`` and offline dry-runs usable
   without credentials, and it makes the resolved universe reproducible.

Importing this module performs no I/O.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.constants import DEFAULT_TIMEZONE, get_timezone
from core.market_calendar import NseCalendar
from core.options.universe import (
    EXPIRY_SELECTORS,
    OPTION_SIDES,
    OptionSelection,
    UniverseResolutionError,
    atm_strike_for,
    build_selection,
    current_monthly_expiry,
    next_monthly_expiry,
    render_symbol,
    select_expiries,
    template_pattern,
)
from core.schemas import InstrumentRow, OptionType

__all__ = [
    "DUMP_FORMAT_VERSION",
    "EXPIRY_SELECTORS",
    "Instrument",
    "InstrumentDump",
    "InstrumentDumpUnavailable",
    "InstrumentMaster",
    "OptionUniverse",
    "UniverseResolutionError",
    "atm_strike_for",
    "build_option_universe",
    "dump_path",
    "load_dump_file",
    "load_instrument_dump",
    "current_monthly_expiry",
    "next_monthly_expiry",
    "parse_dump",
    "render_symbol",
    "save_dump_file",
    "select_expiries",
    "template_pattern",
    "to_instrument_rows",
]

#: Bumped when the on-disk cache layout changes; older files are then ignored.
DUMP_FORMAT_VERSION = 1


class InstrumentDumpUnavailable(RuntimeError):
    """Raised when the instrument dump is needed but neither cached nor fetchable."""


# ---------------------------------------------------------------------------
# Field normalisation (the dump is full of "", NaN and mixed types)
# ---------------------------------------------------------------------------
def _clean_text(value: Any) -> str | None:
    """Normalise the dump's empty/NaN text fields to ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _clean_float(value: Any) -> float | None:
    """Normalise the dump's empty/NaN numeric fields to ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _clean_int(value: Any) -> int | None:
    """Normalise the dump's empty/NaN integer fields to ``None``."""
    number = _clean_float(value)
    return None if number is None else int(number)


def _clean_date(value: Any) -> date | None:
    """Normalise the dump's ``expiry`` field (``""``, NaN, date or ISO string)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Instrument:
    """One tradable (or index) instrument from Kite's dump."""

    instrument_token: int
    exchange: str
    tradingsymbol: str
    name: str | None = None
    segment: str | None = None
    instrument_type: str | None = None
    expiry: date | None = None
    strike: float | None = None
    lot_size: int | None = None
    tick_size: float | None = None
    #: The dump's own ``last_price``. It is the *previous close*, not a live
    #: quote - useful for an offline sanity check or a backfill plan, never as a
    #: market price.
    last_price: float | None = None

    @property
    def option_type(self) -> OptionType | None:
        """``OptionType.CE``/``PE`` for an option, else ``None``."""
        kind = (self.instrument_type or "").upper()
        return OptionType(kind) if kind in {"CE", "PE"} else None

    @property
    def is_option(self) -> bool:
        """True for a CE/PE contract."""
        return self.option_type is not None

    @property
    def is_index(self) -> bool:
        """True for an index (no expiry, no strike, not an option)."""
        return self.expiry is None and self.strike is None and not self.is_option

    @property
    def quote_symbol(self) -> str:
        """``"NFO:NIFTY26OCT25000CE"`` - the key Kite's quote/ltp APIs use."""
        return f"{self.exchange}:{self.tradingsymbol}"

    def key(self) -> tuple[str, str | None, date | None, int | None, str | None]:
        """Hashable contract identity: ``(exchange, name, expiry, strike, type)``."""
        strike = None if self.strike is None else int(round(self.strike))
        kind = (self.instrument_type or "").upper() or None
        return (self.exchange, self.name, self.expiry, strike, kind)

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "instrument_token": self.instrument_token,
            "exchange": self.exchange,
            "tradingsymbol": self.tradingsymbol,
            "instrument_type": self.instrument_type,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "strike": self.strike,
        }


def parse_instrument(raw: Mapping[str, Any]) -> Instrument | None:
    """Convert one dump row into an :class:`Instrument`.

    Returns ``None`` (rather than raising) when the row has no usable identity:
    the caller counts rejections, so a mass failure is visible instead of silent.
    """
    token = _clean_int(raw.get("instrument_token"))
    symbol = _clean_text(raw.get("tradingsymbol"))
    exchange = _clean_text(raw.get("exchange"))
    if not token or token <= 0 or symbol is None or exchange is None:
        return None
    return Instrument(
        instrument_token=token,
        exchange=exchange.upper(),
        tradingsymbol=symbol,
        name=_clean_text(raw.get("name")),
        segment=_clean_text(raw.get("segment")),
        instrument_type=(_clean_text(raw.get("instrument_type")) or "").upper() or None,
        expiry=_clean_date(raw.get("expiry")),
        strike=_clean_float(raw.get("strike")),
        lot_size=_clean_int(raw.get("lot_size")),
        tick_size=_clean_float(raw.get("tick_size")),
        last_price=_clean_float(raw.get("last_price")),
    )


def parse_dump(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Instrument], int]:
    """Parse a dump, returning ``(instruments, rejected_count)``."""
    instruments: list[Instrument] = []
    rejected = 0
    for row in rows:
        instrument = parse_instrument(row)
        if instrument is None:
            rejected += 1
            continue
        instruments.append(instrument)
    return instruments, rejected


def to_instrument_rows(
    instruments: Iterable[Instrument],
    *,
    run_id: str,
    fetched_at: datetime,
) -> list[InstrumentRow]:
    """Render instruments as storable :class:`~core.schemas.InstrumentRow` rows.

    Rows violating the contract (an option without expiry/strike) are skipped
    rather than exploding mid-write; the caller sees the difference in length.
    """
    rows: list[InstrumentRow] = []
    for instrument in instruments:
        try:
            rows.append(
                InstrumentRow(
                    instrument_token=instrument.instrument_token,
                    exchange=instrument.exchange,
                    tradingsymbol=instrument.tradingsymbol,
                    name=instrument.name,
                    segment=instrument.segment,
                    instrument_type=instrument.instrument_type,
                    expiry=instrument.expiry,
                    strike=instrument.strike,
                    lot_size=instrument.lot_size,
                    tick_size=instrument.tick_size,
                    fetched_at=fetched_at,
                    run_id=run_id,
                )
            )
        except ValueError:
            continue
    return rows


# ---------------------------------------------------------------------------
# The master
# ---------------------------------------------------------------------------
class InstrumentMaster:
    """Lookups over a parsed instrument dump.

    Every lookup is index-based (O(1)); the option index is keyed by
    ``(exchange, name, expiry, strike, option_type)`` - the same contract
    identity ANJANEYA uses everywhere else - so a contract is found by its
    attributes and never by a hand-built symbol string.
    """

    __slots__ = ("instruments", "_by_token", "_by_symbol", "_options", "_rejected")

    def __init__(self, instruments: Iterable[Instrument], *, rejected: int = 0) -> None:
        self.instruments: tuple[Instrument, ...] = tuple(instruments)
        self._rejected = rejected
        self._by_token: dict[int, Instrument] = {i.instrument_token: i for i in self.instruments}
        self._by_symbol: dict[tuple[str, str], Instrument] = {
            (i.exchange, i.tradingsymbol): i for i in self.instruments
        }
        options: dict[tuple[str, str | None, date | None, int | None, str | None], Instrument] = {}
        for instrument in self.instruments:
            if instrument.is_option:
                options.setdefault(instrument.key(), instrument)
        self._options = options

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dump(cls, rows: Iterable[Mapping[str, Any]]) -> InstrumentMaster:
        """Build a master from raw dump rows (rejections are counted)."""
        instruments, rejected = parse_dump(rows)
        return cls(instruments, rejected=rejected)

    @classmethod
    def from_instruments(cls, instruments: Iterable[Instrument]) -> InstrumentMaster:
        """Build a master from already-parsed instruments."""
        return cls(instruments)

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.instruments)

    @property
    def rejected_rows(self) -> int:
        """How many dump rows could not be parsed (a data-quality signal)."""
        return self._rejected

    @property
    def tokens(self) -> list[int]:
        """Every instrument token, ascending."""
        return sorted(self._by_token)

    def get(self, token: int) -> Instrument | None:
        """Look an instrument up by token."""
        return self._by_token.get(token)

    def get_by_symbol(self, exchange: str, tradingsymbol: str) -> Instrument | None:
        """Look an instrument up by ``(exchange, tradingsymbol)``."""
        return self._by_symbol.get((exchange.upper(), tradingsymbol))

    def option(
        self,
        *,
        exchange: str,
        name: str,
        expiry: date,
        strike: float,
        option_type: OptionType | str,
    ) -> Instrument | None:
        """Look an option contract up by identity (never by symbol string)."""
        kind = option_type.value if isinstance(option_type, OptionType) else str(option_type).upper()
        return self._options.get((exchange.upper(), name, expiry, int(round(strike)), kind))

    def option_expiries(self, *, exchange: str, name: str) -> list[date]:
        """Sorted distinct option expiries available for an underlying."""
        expiries = {
            instrument.expiry
            for instrument in self._options.values()
            if instrument.exchange == exchange.upper()
            and instrument.name == name
            and instrument.expiry is not None
        }
        return sorted(expiries)

    def option_strikes(self, *, exchange: str, name: str, expiry: date) -> list[float]:
        """Sorted distinct strikes listed for one expiry."""
        strikes = {
            instrument.strike
            for instrument in self._options.values()
            if instrument.exchange == exchange.upper()
            and instrument.name == name
            and instrument.expiry == expiry
            and instrument.strike is not None
        }
        return sorted(strikes)

    def futures(
        self, *, exchange: str, name: str, on_or_after: date | None = None
    ) -> list[Instrument]:
        """Futures contracts for an underlying, nearest expiry first."""
        found = [
            instrument
            for instrument in self.instruments
            if instrument.exchange == exchange.upper()
            and instrument.name == name
            and (instrument.instrument_type or "").upper() == "FUT"
            and instrument.expiry is not None
            and (on_or_after is None or instrument.expiry >= on_or_after)
        ]
        return sorted(found, key=lambda item: item.expiry or date.max)

    def reference(
        self, *, exchange: str, name: str, tradingsymbol: str | None = None
    ) -> Instrument | None:
        """Find a reference instrument by exact symbol, else by exchange+name."""
        if tradingsymbol is not None:
            exact = self.get_by_symbol(exchange, tradingsymbol)
            if exact is not None:
                return exact
        for instrument in self.instruments:
            if instrument.exchange == exchange.upper() and instrument.name == name:
                return instrument
        return None

    def describe(self) -> dict[str, Any]:
        """Log-safe summary of the master."""
        by_type: dict[str, int] = {}
        for instrument in self.instruments:
            kind = (instrument.instrument_type or "INDEX/OTHER").upper()
            by_type[kind] = by_type.get(kind, 0) + 1
        return {
            "instruments": len(self.instruments),
            "rejected_rows": self._rejected,
            "by_instrument_type": dict(sorted(by_type.items())),
        }




# ---------------------------------------------------------------------------
# The on-disk dump cache
# ---------------------------------------------------------------------------
def dump_path(cache_dir: Path, exchange: str, day: date) -> Path:
    """``<cache_dir>/<EXCHANGE>_<YYYY-MM-DD>.json``."""
    return Path(cache_dir) / f"{exchange.upper()}_{day.isoformat()}.json"


def save_dump_file(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Path,
    exchange: str,
    day: date,
    fetched_at: datetime,
) -> Path:
    """Write a dump to the cache atomically (temp file + ``os.replace``).

    An interrupted write must never leave a half-written dump that a later run
    would happily parse as a truncated instrument master.
    """
    import os
    import tempfile

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = dump_path(cache_dir, exchange, day)
    payload = {
        "format_version": DUMP_FORMAT_VERSION,
        "exchange": exchange.upper(),
        "fetched_at": fetched_at.isoformat(),
        "instrument_count": len(rows),
        "rows": list(rows),
    }
    handle, temp_name = tempfile.mkstemp(dir=str(cache_dir), prefix=".dump-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, default=str)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


def _parse_fetched_at(value: Any, path: Path) -> datetime:
    """Parse the dump's ``fetched_at`` (falling back to the file's mtime)."""
    tz = get_timezone(DEFAULT_TIMEZONE)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=tz)
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=tz)
    except OSError:  # pragma: no cover - the file existed a moment ago
        return datetime.now(tz)


def load_dump_file(path: Path) -> tuple[list[dict[str, Any]], datetime]:
    """Read a cached dump, returning ``(rows, fetched_at)``.

    Raises :class:`InstrumentDumpUnavailable` for a missing, unreadable or
    layout-incompatible file, so callers can fall back to a download.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstrumentDumpUnavailable(f"cannot read cached instrument dump {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != DUMP_FORMAT_VERSION:
        version = payload.get("format_version") if isinstance(payload, dict) else "?"
        raise InstrumentDumpUnavailable(
            f"cached instrument dump {path} has format_version {version}, expected "
            f"{DUMP_FORMAT_VERSION}; delete it and re-run --mode instruments"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise InstrumentDumpUnavailable(f"cached instrument dump {path} has no 'rows' list")
    return rows, _parse_fetched_at(payload.get("fetched_at"), path)


def latest_cached_dump(cache_dir: Path, exchange: str) -> Path | None:
    """Most recent cached dump for an exchange, or ``None``."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return None
    candidates = sorted(cache_dir.glob(f"{exchange.upper()}_*.json"))
    return candidates[-1] if candidates else None


@dataclass(slots=True)
class InstrumentDump:
    """A parsed instrument dump plus where it came from."""

    master: InstrumentMaster
    frames: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    source: str = "cache"
    fetched_at: datetime | None = None
    paths: dict[str, Path] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "exchanges": sorted(self.frames),
            "rows": {name: len(rows) for name, rows in self.frames.items()},
            **self.master.describe(),
        }


def load_instrument_dump(
    client: Any,
    *,
    cache_dir: Path,
    exchanges: Sequence[str],
    ttl_hours: float,
    now: datetime,
    logger: Any = None,
    force_download: bool = False,
) -> InstrumentDump:
    """Return the freshest usable instrument dump.

    Order of preference (first success wins):

    1. today's cached file, when it is younger than ``ttl_hours``;
    2. a fresh download via ``client.instruments(exchange)``, cached for next time;
    3. the newest cached file of any day, with a loud warning - a stale master is
       far better than no master, but it must never be used silently.

    ``client`` may be ``None`` (offline/dry-run), in which case only the cache is
    consulted, and :class:`InstrumentDumpUnavailable` is raised when it is empty.
    """
    exchanges = [name.upper() for name in exchanges]
    frames: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, Path] = {}
    newest_fetched: datetime | None = None
    stale_days: list[str] = []
    downloaded_any = False

    for exchange in exchanges:
        cached_path = latest_cached_dump(cache_dir, exchange)
        cached_rows: list[dict[str, Any]] | None = None
        cached_at: datetime | None = None
        if cached_path is not None:
            try:
                cached_rows, cached_at = load_dump_file(cached_path)
            except InstrumentDumpUnavailable as exc:
                if logger is not None:
                    logger.warning(f"{exc}")
                cached_rows = None

        fresh = (
            cached_rows is not None
            and cached_at is not None
            and (now - cached_at) <= timedelta(hours=ttl_hours)
        )
        if cached_rows is not None and fresh and not force_download:
            frames[exchange] = cached_rows
            paths[exchange] = cached_path  # type: ignore[assignment]
            newest_fetched = _newer(newest_fetched, cached_at)
            if logger is not None:
                logger.info(
                    f"instrument dump: using cached {cached_path.name} "
                    f"({len(cached_rows)} rows, age {(now - cached_at).total_seconds() / 3600:.1f}h)"
                )
            continue

        if client is None:
            if cached_rows is not None and cached_at is not None:
                age_days = (now - cached_at).days
                stale_days.append(f"{exchange} ({age_days}d old)")
                frames[exchange] = cached_rows
                paths[exchange] = cached_path  # type: ignore[assignment]
                newest_fetched = _newer(newest_fetched, cached_at)
                continue
            raise InstrumentDumpUnavailable(
                f"no cached instrument dump for {exchange} in {Path(cache_dir)} and no API "
                "client is available (offline/dry-run). Run `--mode instruments` once with "
                "credentials to populate the cache."
            )

        rows = list(client.instruments(exchange) or [])
        if not rows:
            raise InstrumentDumpUnavailable(
                f"Kite returned an empty instrument dump for {exchange}; refusing to replace a "
                "usable master with nothing"
            )
        saved = save_dump_file(rows, cache_dir=cache_dir, exchange=exchange, day=now.date(), fetched_at=now)
        frames[exchange] = rows
        paths[exchange] = saved
        newest_fetched = _newer(newest_fetched, now)
        downloaded_any = True
        if logger is not None:
            logger.info(f"instrument dump: downloaded {exchange} ({len(rows)} rows) -> {saved.name}")

    if stale_days and logger is not None:
        logger.warning(
            "using a STALE instrument dump for "
            + ", ".join(stale_days)
            + " - expiries/strikes listed after that date are invisible until a download succeeds"
        )

    master = InstrumentMaster.from_dump(row for exchange in exchanges for row in frames[exchange])
    return InstrumentDump(
        master=master,
        frames=frames,
        source="download" if downloaded_any else "cache",
        fetched_at=newest_fetched,
        paths=paths,
    )


def _newer(current: datetime | None, candidate: datetime | None) -> datetime | None:
    """Return whichever timestamp is later (``None``-safe)."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


# ---------------------------------------------------------------------------
# Expiry selection, ATM and the option universe
# ---------------------------------------------------------------------------
# The *policy* - ATM snapping, the strike ladder, expiry selection and the symbol
# templates - lives in :mod:`core.options.universe` (spec 17, Stage 3.4) and is
# re-exported here so capture's public surface is unchanged. What stays in this
# module is what needs the instrument dump: resolving those strikes to tokens.


def futures_contract(
    master: InstrumentMaster,
    *,
    universe_config: Any,
    day: date,
) -> Instrument | None:
    """The futures contract named by ``reference_instruments.nifty_futures``.

    ``current_month``/``current_week`` mean the nearest live contract,
    ``next_month`` the one after it. ``None`` when the dump holds no such contract
    (or the config names no selector at all).
    """
    spec = universe_config.reference_instruments.nifty_futures
    if spec.expiry_selector is None:
        return None
    candidates = master.futures(
        exchange=spec.exchange, name=universe_config.underlying, on_or_after=day
    )
    if not candidates:
        return None
    if spec.expiry_selector == "next_month":
        return candidates[1] if len(candidates) > 1 else None
    return candidates[0]


def resolve_atm_reference(
    master: InstrumentMaster,
    *,
    universe_config: Any,
    day: date,
    label: str | None = None,
) -> Instrument | None:
    """The instrument whose price places the ATM strike, for one reference label.

    ``spot`` is the NIFTY index; ``futures`` is the futures contract
    :func:`futures_contract` resolves. Single-sourced so the price the session
    anchors on and the contract the universe records as ``futures`` can never be
    two different months.
    """
    reference = label or universe_config.atm.reference
    if reference == "futures":
        return futures_contract(master, universe_config=universe_config, day=day)
    spot_spec = universe_config.reference_instruments.nifty_spot
    return master.reference(
        exchange=spot_spec.exchange,
        name=universe_config.underlying,
        tradingsymbol=spot_spec.tradingsymbol,
    )


@dataclass(frozen=True, slots=True)
class OptionUniverse:
    """The concrete set of instruments one capture session subscribes to."""

    underlying: str
    reference: Instrument
    futures: Instrument | None
    reference_price: float
    atm_strike: float
    expiries: tuple[date, ...]
    options: tuple[Instrument, ...]
    missing: tuple[str, ...]
    resolved_at: datetime
    reference_label: str = "spot"

    @property
    def tokens(self) -> list[int]:
        """Every instrument token in the universe (deduplicated, ascending)."""
        tokens: set[int] = {self.reference.instrument_token}
        if self.futures is not None:
            tokens.add(self.futures.instrument_token)
        tokens.update(instrument.instrument_token for instrument in self.options)
        return sorted(tokens)

    @property
    def option_tokens(self) -> list[int]:
        """Option instrument tokens only."""
        return [instrument.instrument_token for instrument in self.options]

    @property
    def reference_tokens(self) -> list[int]:
        """Reference (index and futures) tokens only."""
        tokens = [self.reference.instrument_token]
        if self.futures is not None:
            tokens.append(self.futures.instrument_token)
        return tokens

    @property
    def atm_anchor(self) -> Instrument:
        """The instrument whose price placed the ATM strike.

        ``atm.reference: spot`` -> the NIFTY index; ``futures`` -> the futures
        contract. The live stream keeps the *latest tick of the anchor* so an
        intraday re-centre uses the same price source the universe was built from.
        """
        if self.reference_label == "futures" and self.futures is not None:
            return self.futures
        return self.reference

    @property
    def atm_anchor_token(self) -> int:
        """``instrument_token`` of :attr:`atm_anchor` (what the stream watches)."""
        return self.atm_anchor.instrument_token

    @property
    def is_empty(self) -> bool:
        """True when no option contract could be resolved."""
        return not self.options

    @property
    def symbols(self) -> dict[int, str]:
        """``token -> tradingsymbol`` for every instrument in the universe."""
        mapping: dict[int, str] = {self.reference.instrument_token: self.reference.tradingsymbol}
        if self.futures is not None:
            mapping[self.futures.instrument_token] = self.futures.tradingsymbol
        for instrument in self.options:
            mapping[instrument.instrument_token] = instrument.tradingsymbol
        return mapping

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {
            "underlying": self.underlying,
            "reference": self.reference.tradingsymbol,
            "reference_label": self.reference_label,
            "reference_price": self.reference_price,
            "atm_strike": self.atm_strike,
            "atm_anchor": self.atm_anchor.tradingsymbol,
            "futures": self.futures.tradingsymbol if self.futures else None,
            "expiries": [expiry.isoformat() for expiry in self.expiries],
            "options": len(self.options),
            "tokens": len(self.tokens),
            "missing": list(self.missing),
        }


def build_option_universe(
    *,
    universe_config: Any,
    master: InstrumentMaster,
    reference_price: float,
    day: date,
    calendar: NseCalendar,
    resolved_at: datetime,
    reference_label: str | None = None,
    logger: Any = None,
) -> OptionUniverse:
    """Resolve the configured option universe against the instrument master.

    The *policy* (which expiries, which strikes, how the price snaps) is applied by
    :func:`core.options.universe.build_selection`; this function is the half that
    needs the dump - turning each ``(expiry, strike, CE/PE)`` specification into a
    real ``instrument_token`` **by attributes**. Anything that cannot be found is
    reported in ``missing`` (never silently skipped): a half-empty universe is the
    classic way to end up with a dataset that looks complete but has holes.

    ``reference_price`` must be the price of the instrument ``atm.reference``
    names - the index for ``spot``, the futures contract for ``futures``. Capture
    fetches it accordingly (:meth:`…capture.session.CaptureSession.reference_price`).

    Raises
    ------
    UniverseResolutionError
        When the reference instrument itself is absent from the master (there is
        nothing to capture then), or when ``atm.reference`` is ``futures`` but no
        futures contract was resolved - snapping a *spot* price to a strike while
        the config says "futures" would silently mis-place the whole universe.
    """
    exchange = universe_config.exchange.upper()
    underlying = universe_config.underlying
    spot_spec = universe_config.reference_instruments.nifty_spot
    label = reference_label or universe_config.atm.reference

    reference = master.reference(
        exchange=spot_spec.exchange, name=underlying, tradingsymbol=spot_spec.tradingsymbol
    )
    if reference is None:
        raise UniverseResolutionError(
            f"reference instrument {spot_spec.exchange}:{spot_spec.tradingsymbol} is absent from "
            "the instrument master; refresh it with `--mode instruments --force-refresh`"
        )

    anchor = resolve_atm_reference(master, universe_config=universe_config, day=day, label=label)
    if label == "futures" and anchor is None:
        raise UniverseResolutionError(
            "atm.reference is 'futures' but the instrument master holds no NIFTY futures contract "
            f"on/after {day.isoformat()}; refresh the dump or switch atm.reference to 'spot'"
        )
    futures = (
        anchor
        if label == "futures"
        else futures_contract(master, universe_config=universe_config, day=day)
    )

    selection: OptionSelection = build_selection(
        universe_config=universe_config,
        reference_price=reference_price,
        day=day,
        calendar=calendar,
        reference_label=label,
        created_at=resolved_at,
    )

    listed_expiries = master.option_expiries(exchange=exchange, name=underlying)
    missing: list[str] = []
    for expiry in selection.expiries:
        if expiry not in listed_expiries:
            missing.append(f"no listed options for expiry {expiry.isoformat()}")
    usable_expiries = [expiry for expiry in selection.expiries if expiry in listed_expiries]

    options: list[Instrument] = []
    for expiry in usable_expiries:
        listed_strikes = set(
            master.option_strikes(exchange=exchange, name=underlying, expiry=expiry)
        )
        for strike in selection.strikes:
            if strike not in listed_strikes:
                missing.append(f"strike {strike:g} not listed for {expiry.isoformat()}")
                continue
            for side in OPTION_SIDES:
                instrument = master.option(
                    exchange=exchange, name=underlying, expiry=expiry, strike=strike, option_type=side
                )
                if instrument is None:
                    missing.append(f"{side.value} missing at {strike:g} exp {expiry.isoformat()}")
                    continue
                options.append(instrument)

    universe = OptionUniverse(
        underlying=underlying,
        reference=reference,
        futures=futures,
        reference_price=selection.reference_price,
        atm_strike=selection.atm_strike,
        expiries=tuple(sorted(set(usable_expiries))),
        options=tuple(options),
        missing=tuple(missing),
        resolved_at=resolved_at,
        reference_label=label,
    )
    _warn_on_symbol_template_drift(universe, universe_config=universe_config, logger=logger)
    return universe


def _warn_on_symbol_template_drift(
    universe: OptionUniverse,
    *,
    universe_config: Any,
    logger: Any = None,
) -> None:
    """Warn once when no resolved symbol matches the configured template shape.

    The dump is authoritative (lookups are attribute-based), so this never
    changes the universe - it is an early warning that ``symbol_templates`` has
    drifted from reality, which matters for any downstream code (or script) that
    parses symbols. A single warning is emitted, never one per contract.
    """
    if logger is None or not universe.options:
        return
    patterns = [
        template_pattern(universe_config.symbol_templates.weekly),
        template_pattern(universe_config.symbol_templates.monthly),
    ]
    symbols = [instrument.tradingsymbol for instrument in universe.options]
    matched = sum(1 for symbol in symbols if any(p.match(symbol) for p in patterns))
    if matched == 0:
        logger.warning(
            "no resolved option symbol matches the configured symbol_templates "
            f"(weekly={universe_config.symbol_templates.weekly!r}); the dump is authoritative and "
            f"capture continues, but examples are {symbols[:3]} - review the templates."
        )

