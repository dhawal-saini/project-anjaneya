"""Read side of the storage layer: DuckDB tables -> pandas frames.

Why this module exists
----------------------
ANJANEYA-CAPTURE only ever *writes*; ANJANEYA-ENGINE only ever *derives*. The
engine therefore needs the one thing capture never did: a way to read a trading
session back out of DuckDB as a pandas frame. Everything here is a thin,
parameterised query - **no business logic lives in SQL**, so a candle is always
built by the same Python code no matter how it was fetched.

* :func:`read_ticks` - one session's ticks, ordered ``(tradingsymbol, timestamp)``
  so a builder can group by contract without re-sorting;
* :func:`read_candles` - the same for the derived candle table (used by the
  higher-timeframe checks and by later stages);
* :func:`trading_days_with_ticks` / :func:`symbols_with_ticks` /
  :func:`quality_counts` - the small "what is actually in there?" queries the CLI
  needs to pick a default day and to report *honestly* which rows it left out, with
  :func:`trading_days_with_candles` / :func:`symbols_with_candles` as the bar-side
  twins the greeks-enrichment pass selects its work from.

Day boundaries are **never** computed here: the caller passes the window it took
from :class:`core.market_calendar.NseCalendar`, so a half-day or a special
session cannot be silently replaced by a hard-coded 09:15-15:30.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from core.constants import (
    CANDLE_COLUMNS,
    DEFAULT_TIMEZONE,
    FEATURE_COLUMNS,
    INDEX_HISTORY_COLUMNS,
    LABEL_COLUMNS,
    TICK_COLUMNS,
)
from core.schemas import QualityFlag
from core.storage.writers import quote_identifier

__all__ = [
    "CANDLE_READ_COLUMNS",
    "FEATURE_READ_COLUMNS",
    "INDEX_HISTORY_READ_COLUMNS",
    "LABEL_READ_COLUMNS",
    "TICK_READ_COLUMNS",
    "candle_counts",
    "feature_counts",
    "read_option_instruments",
    "first_candle_timestamp",
    "quality_counts",
    "read_candles",
    "read_features",
    "read_index_history",
    "read_instruments",
    "read_labels",
    "read_ticks",
    "symbols_with_candles",
    "symbols_with_ticks",
    "trading_days_with_candles",
    "trading_days_with_ticks",
]

#: The tick columns a candle build actually needs - deliberately *not* the whole
#: tick row. A day of ticks for a broad option universe is millions of rows, and
#: bid/ask quantities never influence an OHLCV bar.
TICK_READ_COLUMNS = (
    "timestamp",
    "instrument_token",
    "tradingsymbol",
    "expiry",
    "strike",
    "option_type",
    "last_price",
    "volume_traded",
    "oi",
    "fetch_timestamp",
    "quality_flag",
    "run_id",
)

#: Candle columns served by :func:`read_candles` - the whole stored contract. It used to
#: stop short of the option-greeks columns (they were ``NULL`` until Stage 3.3); now
#: that ``--mode enrich-greeks`` fills them, a read that hid half of a row would be the
#: kind of quiet difference between "stored" and "served" this project refuses.
CANDLE_READ_COLUMNS = CANDLE_COLUMNS


def _scope(
    *, start: datetime, end: datetime, symbols: Sequence[str] | None = None
) -> tuple[str, list[Any]]:
    """Half-open window ``[start, end)`` plus an optional symbol filter."""
    clauses = ["timestamp >= ?", "timestamp < ?"]
    params: list[Any] = [start, end]
    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        clauses.append(f"tradingsymbol IN ({placeholders})")
        params.extend(symbols)
    return " AND ".join(clauses), params


def read_ticks(
    connection: Any,
    *,
    start: datetime,
    end: datetime,
    symbols: Sequence[str] | None = None,
    quality_flags: Sequence[QualityFlag | str] | None = (QualityFlag.OK,),
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read the ticks of one session window for candle building.

    ``quality_flags=`` defaults to ``(OK,)``: a tick the capture layer already
    judged untrustworthy (``SUSPECT`` / ``STALE`` / ``OUT_OF_SESSION``) must not
    be allowed to shape an OHLCV bar. Pass ``None`` to read every flag.

    The frame keeps the table's own dtypes, so timestamps come back tz-aware
    (IST) and ``expiry`` as ``datetime.date`` - exactly the inputs
    :func:`apps.anjaneeya_engine.candles.build_1m_candles` expects.
    """
    selected = tuple(columns) if columns is not None else TICK_READ_COLUMNS
    unknown = [name for name in selected if name not in TICK_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown tick columns {unknown}; expected a subset of {TICK_COLUMNS}")

    where, params = _scope(start=start, end=end, symbols=symbols)
    if quality_flags is not None:
        flags = [
            flag.value if isinstance(flag, QualityFlag) else str(flag) for flag in quality_flags
        ]
        if flags:
            where += f" AND quality_flag IN ({', '.join('?' for _ in flags)})"
            params.extend(flags)

    sql = (
        f"SELECT {', '.join(quote_identifier(name) for name in selected)} FROM ticks "
        f"WHERE {where} ORDER BY tradingsymbol, timestamp"
    )
    return connection.execute(sql, params).fetch_df()


def read_candles(
    connection: Any,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read stored candles, optionally scoped by window, symbol and timeframe.

    Unlike :func:`read_ticks` both bounds are optional here: ``build-candles``
    always knows its session, while an ad-hoc query ("every 15m bar for this
    contract") usually does not.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp < ?")
        params.append(end)
    if symbols:
        clauses.append(f"tradingsymbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(symbols)
    if timeframes:
        clauses.append(f"timeframe IN ({', '.join('?' for _ in timeframes)})")
        params.extend(timeframes)

    sql = f"SELECT {', '.join(CANDLE_READ_COLUMNS)} FROM candles"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY tradingsymbol, timeframe, timestamp"
    return connection.execute(sql, params).fetch_df()


def trading_days_with_ticks(
    connection: Any, *, timezone_name: str = DEFAULT_TIMEZONE
) -> list[date]:
    """Distinct IST trading dates present in the ``ticks`` table, oldest first.

    The conversion happens *in the query* (not in pandas) so a day is defined by
    exchange-local midnight - the same definition the calendar and the coverage
    ledger use.
    """
    sql = (
        "SELECT DISTINCT CAST(timezone(?, timestamp) AS DATE) AS trading_date "
        "FROM ticks ORDER BY trading_date"
    )
    rows = connection.execute(sql, [timezone_name]).fetchall()
    return [row[0] for row in rows if row and row[0] is not None]


def symbols_with_ticks(connection: Any, *, start: datetime, end: datetime) -> list[str]:
    """Distinct tradingsymbols holding ticks in the window, ordered by name."""
    where, params = _scope(start=start, end=end)
    sql = f"SELECT DISTINCT tradingsymbol FROM ticks WHERE {where} ORDER BY tradingsymbol"
    return [row[0] for row in connection.execute(sql, params).fetchall()]


def trading_days_with_candles(
    connection: Any,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    timeframes: Sequence[str] | None = None,
) -> list[date]:
    """Distinct IST trading dates present in the ``candles`` table, oldest first.

    ``--mode enrich-greeks`` picks its default day from *candles*, not from ticks: it
    reads and rewrites bars, so a day whose ticks were pruned long ago is still a day
    it can work on. The day boundary is computed in the query, exactly as
    :func:`trading_days_with_ticks` does it.
    """
    sql = "SELECT DISTINCT CAST(timezone(?, timestamp) AS DATE) AS trading_date FROM candles"
    params: list[Any] = [timezone_name]
    if timeframes:
        sql += f" WHERE timeframe IN ({', '.join('?' for _ in timeframes)})"
        params.extend(timeframes)
    sql += " ORDER BY trading_date"
    rows = connection.execute(sql, params).fetchall()
    return [row[0] for row in rows if row and row[0] is not None]


def symbols_with_candles(
    connection: Any,
    *,
    start: datetime,
    end: datetime,
    timeframes: Sequence[str] | None = None,
) -> list[str]:
    """Distinct tradingsymbols holding candles in the window, ordered by name.

    The candle-side twin of :func:`symbols_with_ticks`, used by the enrichment pass
    (and by anything else that works on bars rather than on the raw feed).
    """
    where, params = _scope(start=start, end=end)
    if timeframes:
        where += f" AND timeframe IN ({', '.join('?' for _ in timeframes)})"
        params.extend(timeframes)
    sql = f"SELECT DISTINCT tradingsymbol FROM candles WHERE {where} ORDER BY tradingsymbol"
    return [row[0] for row in connection.execute(sql, params).fetchall()]


def quality_counts(
    connection: Any,
    *,
    start: datetime,
    end: datetime,
    symbols: Sequence[str] | None = None,
) -> dict[str, int]:
    """Rows per ``quality_flag`` in the window - i.e. what the builder left out."""
    where, params = _scope(start=start, end=end, symbols=symbols)
    sql = f"SELECT quality_flag, count(*) FROM ticks WHERE {where} GROUP BY 1 ORDER BY 1"
    return {str(flag): int(count) for flag, count in connection.execute(sql, params).fetchall()}


#: Feature columns served by :func:`read_features` - the whole stored contract, in the
#: schema's own order. Nothing is hidden any more: Stage 3.3 fills the greeks, so
#: they are served like every other stored column.
FEATURE_READ_COLUMNS = FEATURE_COLUMNS

#: Label columns served by :func:`read_labels` - the whole stored contract.
LABEL_READ_COLUMNS = LABEL_COLUMNS


#: Index columns served by :func:`read_index_history`. The NIFTY index is captured
#: from Kite's REST history endpoint, so its rows carry the same provenance block.
INDEX_HISTORY_READ_COLUMNS = INDEX_HISTORY_COLUMNS


def read_features(
    connection: Any,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read stored feature rows, optionally scoped by window, contract and timeframe.

    Ordering is ``(tradingsymbol, timestamp)``: one contract's rows arrive together,
    in time order - the shape every validation check and every later consumer (labels,
    the ML dataset) expects. The frame keeps the table's own dtypes, so timestamps come
    back tz-aware (IST) and ``expiry`` as a date.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp < ?")
        params.append(end)
    if symbols:
        clauses.append(f"tradingsymbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(symbols)
    if timeframes:
        clauses.append(f"timeframe IN ({', '.join('?' for _ in timeframes)})")
        params.extend(timeframes)

    sql = f"SELECT {', '.join(quote_identifier(name) for name in FEATURE_READ_COLUMNS)} FROM features"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY tradingsymbol, timestamp"
    return connection.execute(sql, params).fetch_df()


def read_index_history(
    connection: Any,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read NIFTY index OHLCV, optionally scoped by window, symbol and timeframe.

    This is the *underlying* series the NIFTY trend/regime block is computed on. It
    lives in its own table because it is fetched (REST history), not built from ticks -
    a distinction the storage layer has kept since Phase 1, and the reason a missing
    index series is a data problem rather than a build failure.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp < ?")
        params.append(end)
    if symbols:
        clauses.append(f"tradingsymbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(symbols)
    if timeframes:
        clauses.append(f"timeframe IN ({', '.join('?' for _ in timeframes)})")
        params.extend(timeframes)

    sql = (
        f"SELECT {', '.join(quote_identifier(name) for name in INDEX_HISTORY_READ_COLUMNS)} "
        "FROM index_history"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY tradingsymbol, timeframe, timestamp"
    return connection.execute(sql, params).fetch_df()


def read_instruments(
    connection: Any,
    *,
    exchange: str | None = None,
    name: str | None = None,
    expiries: Sequence[date] | None = None,
    instrument_types: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Instrument-master rows, optionally filtered by exchange/underlying/expiry/type.

    The generic view of the ``instruments`` table: pass ``instrument_types`` to
    narrow it (``("CE", "PE")`` for options, ``("FUT",)`` for the futures contract
    an ATM anchor may need) or leave it out to see everything the dump held. No
    null-filters are applied implicitly - an index row legitimately has neither an
    expiry nor a strike, and hiding it would make "the master holds no futures row"
    indistinguishable from "the master was never captured".

    ``expiry`` comes back as the table's own dtype (a pandas ``Timestamp``, i.e.
    ``datetime64[us]``), so a caller matching it against the ``date`` a market
    calendar produced must normalise first.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if exchange:
        clauses.append("exchange = ?")
        params.append(exchange.upper())
    if name:
        clauses.append("name = ?")
        params.append(name)
    if expiries:
        clauses.append(f"expiry IN ({', '.join('?' for _ in expiries)})")
        params.extend(list(expiries))
    if instrument_types:
        kinds = [str(kind).upper() for kind in instrument_types]
        clauses.append(f"instrument_type IN ({', '.join('?' for _ in kinds)})")
        params.extend(kinds)
    sql = (
        "SELECT instrument_token, exchange, tradingsymbol, name, expiry, strike, "
        "instrument_type, fetched_at FROM instruments"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY expiry NULLS LAST, strike NULLS LAST, instrument_type"
    return connection.execute(sql, params).fetch_df()


def read_option_instruments(
    connection: Any,
    *,
    exchange: str | None = None,
    name: str | None = None,
    expiries: Sequence[date] | None = None,
) -> pd.DataFrame:
    """The *option* contracts the instrument master holds, for one underlying.

    Keyed by the same attributes every other table uses - ``(expiry, strike,
    option_type)`` - so a caller can ask "is this contract actually listed?" (spec
    17's question) without ever building a symbol string. Reference rows (index,
    futures) are excluded: only real options carry both an expiry and a strike.
    """
    return read_instruments(
        connection, exchange=exchange, name=name, expiries=expiries, instrument_types=("CE", "PE")
    )


def candle_counts(
    connection: Any,
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    symbols: Sequence[str] | None = None,
) -> dict[str, int]:
    """``tradingsymbol -> stored bar count`` for one window and timeframe.

    A count query rather than a frame read: "how much data do we have for the
    contracts this universe names?" is asked for dozens of contracts at once, and
    the answer must not depend on materialising their bars.
    """
    where, params = _scope(start=start, end=end, symbols=symbols)
    where += " AND timeframe = ?"
    params.append(timeframe)
    sql = (
        f"SELECT tradingsymbol, count(*) FROM {quote_identifier('candles')} "
        f"WHERE {where} GROUP BY 1 ORDER BY 1"
    )
    return {str(symbol): int(count) for symbol, count in connection.execute(sql, params).fetchall()}


def feature_counts(
    connection: Any,
    *,
    start: datetime,
    end: datetime,
    timeframe: str | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, int]:
    """``tradingsymbol -> stored feature row count`` for one window.

    The feature-side twin of :func:`candle_counts`: it is what separates "we
    captured this contract" from "we have feature rows for it".
    """
    clauses = ["timestamp >= ?", "timestamp < ?"]
    params: list[Any] = [start, end]
    if timeframe:
        clauses.append("timeframe = ?")
        params.append(timeframe)
    if symbols:
        clauses.append(f"tradingsymbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(list(symbols))
    sql = (
        f"SELECT tradingsymbol, count(*) FROM {quote_identifier('features')} "
        f"WHERE {' AND '.join(clauses)} GROUP BY 1 ORDER BY 1"
    )
    return {str(symbol): int(count) for symbol, count in connection.execute(sql, params).fetchall()}



def read_labels(
    connection: Any,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    symbols: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Read stored label rows (spec 20), optionally scoped by window/contract/horizon.

    Ordering is ``(tradingsymbol, timestamp, horizon_minutes)``: one contract's rows
    arrive together, in time order, with its horizons adjacent - the shape the ML
    dataset builder wants when it pivots horizons into columns.

    Labelled and unlabelled rows both come back: a row whose forward window did not
    fit inside its session is stored (so the hole is a fact, not a silence) with
    ``label IS NULL``, and the caller decides what to do with it.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp < ?")
        params.append(end)
    if symbols:
        clauses.append(f"tradingsymbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(symbols)
    if horizons:
        clauses.append(f"horizon_minutes IN ({', '.join('?' for _ in horizons)})")
        params.extend(int(horizon) for horizon in horizons)

    sql = f"SELECT {', '.join(quote_identifier(name) for name in LABEL_READ_COLUMNS)} FROM labels"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY tradingsymbol, timestamp, horizon_minutes"
    return connection.execute(sql, params).fetch_df()


def first_candle_timestamp(
    connection: Any, *, tradingsymbol: str, timeframe: str
) -> datetime | None:
    """The first bar a contract has, ever - or ``None`` when it has none.

    ``build-features`` uses this to start a contract's window at its own first bar
    rather than at the generic lookback boundary. That is what makes
    ``option_insufficient_history`` an honest statement about the *contract*: the
    option engine counts bars since the start of the frame it is given.
    """
    row = connection.execute(
        f"SELECT min(timestamp) FROM {quote_identifier('candles')} "
        "WHERE tradingsymbol = ? AND timeframe = ?",
        [tradingsymbol, timeframe],
    ).fetchone()
    return row[0] if row and row[0] is not None else None
