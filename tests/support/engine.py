"""Synthetic ticks, candles and contexts used by the ANJANEYA-ENGINE tests.

The engine reads a session's ticks and writes candles, so the tests need three
kinds of fixture - and this module is the single place that defines them:

* **ticks**: :func:`ticks_for_minutes` builds a contract's tick stream minute by
  minute (two ticks per minute, cumulative volume, matching what the WebSocket
  would hand over), and :func:`tick_frame` turns that into the pandas frame the
  builder consumes - tz-aware timestamps, ``datetime.date`` expiries, the same
  dtypes DuckDB returns.
* **a seeded database**: :func:`seed_ticks` bootstraps a DuckDB file and appends
  tick rows, so an integration test starts from *stored* ticks rather than from a
  frame it built itself.
* **a context**: :func:`make_engine_context` returns the real
  :class:`core.app_bootstrap.AppContext` the CLI would build, and
  :func:`cli_args` produces the identical argument list.

Nothing here writes into the repository's ``data/`` or ``db/`` - callers pass
``tmp_path`` paths (see the ``isolated_paths`` fixture in ``tests/conftest.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from core.constants import (
    CANDLE_COLUMNS,
    FEATURE_COLUMNS,
    INDEX_HISTORY_COLUMNS,
    INSTRUMENT_COLUMNS,
    TICK_COLUMNS,
    timeframe_to_minutes,
)
from core.schemas import (
    CandleRow,
    FeatureRow,
    IndexHistoryRow,
    InstrumentRow,
    QualityFlag,
    Source,
    TickRow,
)
from core.storage import bootstrap_database, connect, insert_rows

#: Exchange timezone for every synthetic timestamp.
IST = ZoneInfo("Asia/Kolkata")

#: Trading day used across the engine tests (Tuesday 2026-09-29, IST).
DAY = date(2026, 9, 29)
#: A weekend day, for the "not a trading day" paths.
WEEKEND = date(2026, 9, 27)
#: The weekly expiry the fixtures trade: the next Tuesday after :data:`DAY`, which is
#: what the shipped configuration resolves (``current_week`` for 2026-09-29 is the
#: 29th itself, so the contract the tests hold is the following week's, 2026-10-06).
EXPIRY = date(2026, 10, 6)
#: A later expiry (the monthly, the last Tuesday of October), used to prove one
#: contract never bleeds into another.
FAR_EXPIRY = date(2026, 10, 27)

#: Session bounds of :data:`DAY` (the NSE regular session).
SESSION_START = datetime(2026, 9, 29, 9, 15, tzinfo=IST)
SESSION_END = datetime(2026, 9, 29, 15, 30, tzinfo=IST)

#: Two option contracts and one run id - shared so assertions can name them.
CE_SYMBOL = "NIFTY06OCT2625000CE"
PE_SYMBOL = "NIFTY06OCT2625000PE"
CE_TOKEN = 900_001
PE_TOKEN = 900_002
RUN_ID = "anjaneeya_engine_20260929-153000_deadbeef"


def at(minutes: int, *, seconds: int = 0) -> datetime:
    """``minutes`` after the session open on :data:`DAY` (IST)."""
    return SESSION_START + timedelta(minutes=minutes, seconds=seconds)


def make_tick(**overrides: Any) -> dict[str, Any]:
    """One option tick as a plain dict (the shape :func:`tick_frame` expects)."""
    moment = overrides.pop("timestamp", at(0, seconds=5))
    payload: dict[str, Any] = {
        "timestamp": moment,
        "instrument_token": CE_TOKEN,
        "tradingsymbol": CE_SYMBOL,
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "last_price": 120.0,
        "volume_traded": 1000,
        "oi": 5000,
        "fetch_timestamp": moment + timedelta(milliseconds=200),
        "quality_flag": QualityFlag.OK.value,
        "source": Source.ZERODHA_WEBSOCKET.value,
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return payload


def ticks_for_minutes(
    minutes: Sequence[int],
    *,
    symbol: str = CE_SYMBOL,
    token: int = CE_TOKEN,
    option_type: str = "CE",
    strike: float = 25000.0,
    expiry: date = EXPIRY,
    start_price: float = 120.0,
    price_step: float = 0.5,
    volume: int = 1000,
    volume_step: int = 10,
    oi: int = 5000,
) -> list[dict[str, Any]]:
    """Two ticks per requested minute, with a rising price and volume.

    Each minute therefore has a real in-bar high/low (the two ticks differ), and
    the cumulative volume grows by ``volume_step`` per tick - exactly the setup a
    bar's ``volume`` assertion needs.
    """
    rows: list[dict[str, Any]] = []
    price = start_price
    observed = volume
    for minute in minutes:
        for second in (5, 35):
            price += price_step
            observed += volume_step
            rows.append(
                make_tick(
                    timestamp=at(minute, seconds=second),
                    instrument_token=token,
                    tradingsymbol=symbol,
                    option_type=option_type,
                    strike=strike,
                    expiry=expiry,
                    last_price=round(price, 2),
                    volume_traded=observed,
                    oi=oi,
                )
            )
    return rows


def tick_frame(ticks: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Turn tick dicts into the frame :func:`build_1m_candles` expects.

    Timestamps are forced to a tz-aware dtype, so the frame matches what
    :func:`core.storage.readers.read_ticks` returns from DuckDB.
    """
    frame = pd.DataFrame(list(ticks))
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["fetch_timestamp"] = pd.to_datetime(frame["fetch_timestamp"])
    return frame


def make_tick_row(**overrides: Any) -> TickRow:
    """One :class:`TickRow` model (for seeding the database)."""
    return TickRow(**make_tick(**overrides))


def make_candle(**overrides: Any) -> CandleRow:
    """One :class:`CandleRow` with sane defaults (for merge/aggregation tests)."""
    payload: dict[str, Any] = {
        "timestamp": at(0),
        "timeframe": "1m",
        "instrument_token": CE_TOKEN,
        "tradingsymbol": CE_SYMBOL,
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 10,
        "oi": 5000,
        "source": Source.DERIVED,
        "fetch_timestamp": at(0, seconds=59),
        "quality_flag": QualityFlag.OK,
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return CandleRow(**payload)


def seed_ticks(db_path: Any, ticks: Sequence[TickRow], *, schema_version: int = 1) -> Any:
    """Bootstrap ``db_path`` (once) and append ``ticks``; returns the path."""
    bootstrap_database(db_path, schema_version=schema_version)
    connection = connect(db_path)
    try:
        insert_rows(connection, "ticks", ticks, mode="append", columns=TICK_COLUMNS)
    finally:
        connection.close()
    return db_path


def session_ticks(*, minutes: Sequence[int] = range(0, 6)) -> list[TickRow]:
    """A small two-contract session, ready to seed into the ticks table."""
    ce_rows = ticks_for_minutes(minutes)
    pe_rows = ticks_for_minutes(
        minutes, symbol=PE_SYMBOL, token=PE_TOKEN, option_type="PE", start_price=90.0
    )
    return [make_tick_row(**tick) for tick in (*ce_rows, *pe_rows)]


def make_engine_context(
    config: Any,
    *,
    mode: str = "build-candles",
    dry_run: bool = False,
    logger: Any = None,
    bootstrap: bool = True,
) -> Any:
    """Build the real :class:`AppContext` the engine CLI would build."""
    from core.app_bootstrap import APP_ENGINE, AppContext
    from core.market_calendar import NseCalendar
    from core.run_context import create_run_context

    if bootstrap:
        bootstrap_database(config.db_path, schema_version=1)
    run_context = create_run_context(
        APP_ENGINE,
        mode=mode,
        config_hash=config.config_hash,
        config_dir=config.config_dir,
        timezone_name=config.timezone_name,
        package_names=[],
    )
    return AppContext(
        app_name=APP_ENGINE,
        mode=mode,
        config=config,
        run_context=run_context,
        calendar=NseCalendar(config.calendar),
        logger=logger if logger is not None else RecordingLogger(),
        dry_run=dry_run,
    )


def cli_args(
    tmp_path: Any,
    *,
    mode: str = "build-candles",
    extra: Sequence[str] | None = None,
    run_artifacts: bool = False,
) -> list[str]:
    """Arguments for ``apps.anjaneeya_engine.run.main`` against ``tmp_path``."""
    args = [
        "--mode",
        mode,
        "--data-root",
        str(tmp_path / "data"),
        "--db-path",
        str(tmp_path / "db" / "anjaneeya.duckdb"),
    ]
    if not run_artifacts:
        args.append("--no-run-artifacts")
    return args + list(extra or [])


class RecordingLogger:
    """A logger double that records the level and message of every call.

    Engine tests assert on *what the operator would have seen* (a contract that
    could not be built, a mirror that failed) rather than on log files.
    """

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

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

    def messages(self, level: str | None = None) -> list[str]:
        """Recorded messages, optionally filtered by level."""
        return [text for recorded, text in self.records if level is None or recorded == level]

# ---------------------------------------------------------------------------
# Feature-side fixtures (Stage 3.2e)
# ---------------------------------------------------------------------------
#: The NIFTY index ``tradingsymbol`` the shipped configuration fetches.
INDEX_SYMBOL = "NIFTY 50"


@dataclass(slots=True)
class FeatureWindow:
    """A synthetic contract window plus the NIFTY index series beside it."""

    candles: pd.DataFrame
    index: pd.DataFrame
    session_start: datetime
    session_end: datetime

    @property
    def session_bars(self) -> int:
        """Bars published for the session (the feature rows a build should produce)."""
        return len(self.candles[self.candles["timestamp"] >= self.session_start])

    def truncated(self, bars: int) -> FeatureWindow:
        """The same window, cut off after ``bars`` candles (for look-ahead probes)."""
        return FeatureWindow(
            candles=self.candles.iloc[:bars].reset_index(drop=True),
            index=self.index,
            session_start=self.session_start,
            session_end=self.session_end,
        )


def window_candles(
    *,
    symbol: str = CE_SYMBOL,
    token: int = CE_TOKEN,
    option_type: str = "CE",
    timeframe: str = "1m",
    lookback_bars: int = 60,
    session_bars: int = 12,
    price: float = 120.0,
    step: float = 0.05,
) -> pd.DataFrame:
    """One contract's candle window: ``lookback_bars`` before the session, then more.

    Rows are :class:`CandleRow` records (so the dtypes match what
    ``core.storage.readers.read_candles`` returns), one bar per minute, with a gentle
    uptrend so every level and slope is exercised.
    """
    start = SESSION_START - timedelta(minutes=lookback_bars)
    rows = []
    for index in range(lookback_bars + session_bars):
        moment = start + timedelta(minutes=index)
        level = price + index * step
        rows.append(
            make_candle(
                timestamp=moment,
                timeframe=timeframe,
                tradingsymbol=symbol,
                instrument_token=token,
                option_type=option_type,
                close=level + step / 2,
                open=level - step / 2,
                high=level + step,
                low=level - step,
                volume=100 + index,
                oi=5000 + index,
            )
        )
    frame = pd.DataFrame([row.to_record(exclude_none=False) for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["fetch_timestamp"] = pd.to_datetime(frame["fetch_timestamp"])
    return frame


def window_index(
    *,
    symbol: str = INDEX_SYMBOL,
    timeframe: str = "1m",
    lookback_bars: int = 60,
    session_bars: int = 12,
    level: float = 25000.0,
    step: float = 1.0,
) -> pd.DataFrame:
    """The NIFTY index series for the same window (REST history, no ticks involved)."""
    start = SESSION_START - timedelta(minutes=lookback_bars)
    rows = []
    for index in range(lookback_bars + session_bars):
        moment = start + timedelta(minutes=index)
        price = level + index * step
        rows.append(
            make_index_row(
                timestamp=moment,
                tradingsymbol=symbol,
                timeframe=timeframe,
                open=price - step,
                high=price + step,
                low=price - step,
                close=price,
                volume=0,
            )
        )
    frame = pd.DataFrame([row.to_record(exclude_none=False) for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["fetch_timestamp"] = pd.to_datetime(frame["fetch_timestamp"])
    return frame


def feature_window(
    *, lookback_bars: int = 60, session_bars: int = 12, timeframe: str = "1m"
) -> FeatureWindow:
    """A ready-to-publish site: one contract's candles plus the index beside them."""
    return FeatureWindow(
        candles=window_candles(
            timeframe=timeframe, lookback_bars=lookback_bars, session_bars=session_bars
        ),
        index=window_index(
            timeframe=timeframe, lookback_bars=lookback_bars, session_bars=session_bars
        ),
        session_start=SESSION_START,
        session_end=SESSION_START + timedelta(minutes=session_bars),
    )


def make_index_row(**overrides: Any) -> IndexHistoryRow:
    """One :class:`IndexHistoryRow` (a NIFTY index bar from Kite's REST history)."""
    moment = overrides.pop("timestamp", at(0))
    payload: dict[str, Any] = {
        "timestamp": moment,
        "tradingsymbol": INDEX_SYMBOL,
        "timeframe": "1m",
        "open": 25000.0,
        "high": 25010.0,
        "low": 24990.0,
        "close": 25005.0,
        "volume": 0,
        "oi": None,
        "source": Source.ZERODHA_REST,
        "fetch_timestamp": moment + timedelta(seconds=5),
        "quality_flag": QualityFlag.OK,
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return IndexHistoryRow(**payload)


def seed_candles(db_path: Any, rows: Sequence[Any], *, schema_version: int = 1) -> Any:
    """Bootstrap ``db_path`` (once) and append candle rows.

    The greeks-enrichment tests need *stored bars* without building them from ticks, so
    this is the candle-side twin of :func:`seed_ticks`.
    """
    bootstrap_database(db_path, schema_version=schema_version)
    connection = connect(db_path)
    try:
        insert_rows(connection, "candles", rows, mode="append", columns=CANDLE_COLUMNS)
    finally:
        connection.close()
    return db_path


# ---------------------------------------------------------------------------
# Greeks-side fixtures (Stage 3.3)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PricedWindow:
    """A contract's bars whose closes are Black-Scholes prices at a known volatility.

    The point of pricing them *from* a model is that the enrichment has an answer to
    find: solve the IV of these closes and ``sigma`` comes back out. That is the only way
    an end-to-end test can tell "the model ran" apart from "the model ran and was right".
    """

    candles: pd.DataFrame
    index: pd.DataFrame
    sigma: float
    expiry: date
    strike: float

    def rows(self) -> list[CandleRow]:
        """The candles as row models (for seeding the database)."""
        return [make_candle(**record) for record in self.candles.to_dict("records")]


def priced_window(
    *,
    symbol: str = CE_SYMBOL,
    token: int = CE_TOKEN,
    option_type: str = "CE",
    strike: float = 25000.0,
    expiry: date = EXPIRY,
    minutes: Sequence[int] = (0, 1, 2, 3, 4, 5),
    sigma: float = 0.18,
    level: float = 25000.0,
    step: float = 4.0,
    timeframe: str = "1m",
) -> PricedWindow:
    """Candles priced at ``sigma`` off a rising NIFTY index series, minute by minute.

    Each bar's close is ``black_scholes_price(spot at its close, strike, T, sigma)``, so
    the index row for that minute *is* the spot the price was built from. The two series
    are exactly consistent, and any error in the enrichment is the enrichment's - not the
    fixture's.
    """
    from core.config_loader import load_config
    from core.market_calendar import NseCalendar
    from core.options import GreeksModel, black_scholes_price

    config = load_config()
    model = GreeksModel.from_config(config.indicators)
    calendar = NseCalendar(config.calendar)
    bars: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for minute in minutes:
        moment = at(minute)
        spot = level + minute * step
        close_moment = moment + timedelta(minutes=timeframe_to_minutes(timeframe))
        years = (
            calendar.session_end(expiry) - close_moment
        ).total_seconds() / model.year_seconds
        value = round(
            float(
                black_scholes_price(
                    spot, strike, years, sigma, option_type=option_type.upper(), model=model
                )
            ),
            4,
        )
        bars.append(
            {
                "timestamp": moment,
                "timeframe": timeframe,
                "instrument_token": token,
                "tradingsymbol": symbol,
                "expiry": expiry,
                "strike": strike,
                "option_type": option_type.upper(),
                "open": value,
                "high": value + 0.5,
                "low": value - 0.5,
                "close": value,
                "volume": 100,
                "oi": 5000,
                "source": Source.DERIVED,
                "fetch_timestamp": moment + timedelta(seconds=59),
                "quality_flag": QualityFlag.OK,
                "run_id": RUN_ID,
            }
        )
        index_rows.append(
            {
                "timestamp": moment,
                "tradingsymbol": INDEX_SYMBOL,
                "timeframe": timeframe,
                "open": spot,
                "high": spot + 1.0,
                "low": spot - 1.0,
                "close": spot,
                "volume": 0,
                "oi": None,
                "source": Source.ZERODHA_REST,
                "fetch_timestamp": moment + timedelta(seconds=5),
                "quality_flag": QualityFlag.OK,
                "run_id": RUN_ID,
            }
        )
    candles = pd.DataFrame(bars)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"])
    candles["fetch_timestamp"] = pd.to_datetime(candles["fetch_timestamp"])
    index = pd.DataFrame(index_rows)
    index["timestamp"] = pd.to_datetime(index["timestamp"])
    index["fetch_timestamp"] = pd.to_datetime(index["fetch_timestamp"])
    return PricedWindow(candles=candles, index=index, sigma=sigma, expiry=expiry, strike=strike)


def seed_index_history(db_path: Any, rows: Sequence[IndexHistoryRow], *, schema_version: int = 1) -> Any:
    """Bootstrap ``db_path`` (once) and append index-history rows."""
    bootstrap_database(db_path, schema_version=schema_version)
    connection = connect(db_path)
    try:
        insert_rows(connection, "index_history", rows, mode="append", columns=INDEX_HISTORY_COLUMNS)
    finally:
        connection.close()
    return db_path


def make_feature(**overrides: Any) -> FeatureRow:
    """One :class:`FeatureRow` with sane defaults (for writer/validation tests)."""
    payload: dict[str, Any] = {
        "timestamp": at(0),
        "underlying": "NIFTY",
        "tradingsymbol": CE_SYMBOL,
        "instrument_token": CE_TOKEN,
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "timeframe": "1m",
        "nifty_composite_trend_score": 0.25,
        "nifty_regime": "BULLISH",
        "option_trend_score": 0.1,
        "option_atr": 1.5,
        "candle_body": 0.5,
        "candle_direction": "BULLISH",
        "candle_inside_flag": False,
        "support_touch_count": 1,
        "event_today_high_state": "APPROACHING",
        "swing_label": "HH_HL",
        "market_structure_state": "BULLISH_STRUCTURE",
        "distance_from_atm": -750.0,
        "days_to_expiry": 2.0,
        "expiry_cycle_phase": "MID_CYCLE",
        "time_of_day_bucket": "OPEN_915_930",
        "day_of_week": 1,
        "volume": 100,
        "oi": 5000,
        "source": Source.DERIVED,
        "fetch_timestamp": at(0, seconds=59),
        "quality_flag": QualityFlag.OK,
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return FeatureRow(**payload)


def feature_frame(rows: Sequence[FeatureRow]) -> pd.DataFrame:
    """Rows as the frame the validation and writer code consumes."""
    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    return pd.DataFrame([row.to_record(exclude_none=False) for row in rows])


# ---------------------------------------------------------------------------
# Universe-side fixtures (Stage 3.4a)
# ---------------------------------------------------------------------------
#: The five strikes the shipped config selects around a 25 000 ATM (+/-200 in 50s).
UNIVERSE_STRIKES: tuple[int, ...] = (24800, 24900, 25000, 25100, 25200)


def weekly_option_symbol(*, expiry: date, strike: float, option_type: str) -> str:
    """The NSE weekly symbol shape the shipped ``symbol_templates`` expects.

    ``NIFTY{expiry_short}{strike}{option_type}`` with ``expiry_short`` = ``06OCT26``
    - i.e. ``NIFTY06OCT2625000CE``, which is also :data:`CE_SYMBOL` for the default
    contract. Rendering it here (rather than reusing the config's own renderer) keeps
    the test honest: the *stored* symbol is independent of the configured template.
    """
    return f"NIFTY{expiry.strftime('%d%b%y').upper()}{int(round(strike))}{option_type}"


def make_instrument_row(**overrides: Any) -> InstrumentRow:
    """One :class:`InstrumentRow` (a row of the ``instruments`` table)."""
    payload: dict[str, Any] = {
        "instrument_token": CE_TOKEN,
        "exchange": "NFO",
        "tradingsymbol": CE_SYMBOL,
        "name": "NIFTY",
        "segment": "NFO-OPT",
        "instrument_type": "CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "lot_size": 75,
        "tick_size": 0.05,
        "fetched_at": at(0),
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return InstrumentRow(**payload)


def option_instrument_rows(
    *,
    expiries: Sequence[date] = (EXPIRY,),
    strikes: Sequence[float] = UNIVERSE_STRIKES,
    token_base: int = 900_100,
    name: str = "NIFTY",
    exchange: str = "NFO",
    symbol_of: Any = None,
) -> list[InstrumentRow]:
    """The ``instruments`` rows a captured dump would hold for the ATM window.

    ``symbol_of`` lets a test break the symbol shape on purpose (to exercise the
    template-drift reporting) without touching the rest of the row.
    """
    rows: list[InstrumentRow] = []
    token = token_base
    for expiry in expiries:
        for strike in strikes:
            for side in ("CE", "PE"):
                token += 1
                symbol = (
                    symbol_of(expiry=expiry, strike=strike, option_type=side)
                    if symbol_of is not None
                    else weekly_option_symbol(expiry=expiry, strike=strike, option_type=side)
                )
                rows.append(
                    make_instrument_row(
                        instrument_token=token,
                        exchange=exchange,
                        name=name,
                        tradingsymbol=symbol,
                        instrument_type=side,
                        expiry=expiry,
                        strike=float(strike),
                    )
                )
    return rows


def futures_instrument_row(
    *, expiry: date = FAR_EXPIRY, token: int = 510_001, name: str = "NIFTY"
) -> InstrumentRow:
    """One NIFTY futures row (what ``atm.reference: futures`` needs in the master)."""
    return make_instrument_row(
        instrument_token=token,
        exchange="NFO",
        name=name,
        tradingsymbol=f"NIFTY{expiry.strftime('%d%b%y').upper()}FUT",
        segment="NFO-FUT",
        instrument_type="FUT",
        expiry=expiry,
        strike=None,
    )


def seed_instruments(db_path: Any, rows: Sequence[Any], *, schema_version: int = 1) -> Any:
    """Bootstrap ``db_path`` (once) and append instrument rows."""
    bootstrap_database(db_path, schema_version=schema_version)
    connection = connect(db_path)
    try:
        insert_rows(connection, "instruments", rows, mode="append", columns=INSTRUMENT_COLUMNS)
    finally:
        connection.close()
    return db_path

def seed_features(db_path: Any, rows: Sequence[Any], *, schema_version: int = 1) -> Any:
    """Bootstrap ``db_path`` (once) and append feature rows.

    ``--mode universe`` reports how much *feature* data a selected contract has,
    so a test needs feature rows without running a feature build.
    """
    bootstrap_database(db_path, schema_version=schema_version)
    connection = connect(db_path)
    try:
        insert_rows(connection, "features", rows, mode="append", columns=FEATURE_COLUMNS)
    finally:
        connection.close()
    return db_path

