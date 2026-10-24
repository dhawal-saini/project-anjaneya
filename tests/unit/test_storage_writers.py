"""Tests for :mod:`core.storage.writers` - the single place that writes rows."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.schemas import TickRow
from core.storage import bootstrap_database, connect, insert_rows
from core.storage.writers import count_rows, quote_identifier, row_columns

IST = ZoneInfo("Asia/Kolkata")
TS = datetime(2026, 9, 29, 9, 16, 0, tzinfo=IST)
RUN_ID = "anjaneeya_capture_20260929-091500_abcdef12"


def make_tick(**overrides) -> dict:
    payload = {
        "timestamp": TS,
        "instrument_token": 12345,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": date(2026, 10, 1),
        "strike": 25000.0,
        "option_type": "CE",
        "last_price": 123.45,
        "volume_traded": 1000,
        "oi": 5000,
        "bid": 123.0,
        "ask": 124.0,
        "bid_qty": 75,
        "ask_qty": 150,
        "source": "ZERODHA_WEBSOCKET",
        "fetch_timestamp": TS + timedelta(milliseconds=200),
        "quality_flag": "OK",
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def connection(tmp_path: Path):
    """A freshly bootstrapped DuckDB database."""
    path = tmp_path / "db" / "anjaneeya.duckdb"
    bootstrap_database(path, schema_version=1)
    conn = connect(path)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ticks", "_private", "col_1", "A1"])
def test_valid_identifiers_are_quoted(name: str) -> None:
    assert quote_identifier(name) == f'"{name}"'


@pytest.mark.parametrize("name", ["ticks; DROP TABLE x", "1leading", "with space", "", 'q"q', None, 3])
def test_unsafe_identifiers_are_rejected(name) -> None:
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        quote_identifier(name)


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------
def test_row_columns_use_first_seen_order() -> None:
    assert row_columns([{"b": 1, "a": 2}, {"c": 3, "a": 4}]) == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# insert_rows
# ---------------------------------------------------------------------------
def test_append_writes_rows_and_reports_the_count(connection) -> None:
    written = insert_rows(connection, "ticks", [make_tick(), make_tick(strike=25050.0)])
    assert written == 2
    assert count_rows(connection, "ticks") == 2


def test_models_and_dicts_are_interchangeable(connection) -> None:
    written = insert_rows(
        connection, "ticks", [TickRow(**make_tick()), make_tick(tradingsymbol="NIFTY26OCT25100CE")]
    )
    assert written == 2
    symbols = [row[0] for row in connection.execute("SELECT tradingsymbol FROM ticks").fetchall()]
    assert symbols == ["NIFTY26OCT25000CE", "NIFTY26OCT25100CE"]


def test_sparse_rows_are_padded_with_null_not_shifted(connection) -> None:
    """A row missing a key must not shift its later values into that column."""
    insert_rows(
        connection,
        "ticks",
        [
            make_tick(oi=999),
            {**make_tick(), "oi": None},
        ],
    )
    rows = connection.execute("SELECT oi, tradingsymbol FROM ticks ORDER BY rowid").fetchall()
    assert rows[0][0] == 999
    assert rows[1] == (None, "NIFTY26OCT25000CE")


def test_explicit_columns_survive_an_empty_batch(connection) -> None:
    assert insert_rows(connection, "ticks", [], columns=["timestamp", "run_id"]) == 0
    assert count_rows(connection, "ticks") == 0


def test_unknown_mode_is_rejected(connection) -> None:
    with pytest.raises(ValueError, match="Unknown write mode"):
        insert_rows(connection, "ticks", [make_tick()], mode="upsert")  # type: ignore[arg-type]


def test_unknown_table_is_reported_by_duckdb(connection) -> None:
    import duckdb

    with pytest.raises(duckdb.Error):
        insert_rows(connection, "not_a_table", [make_tick()])


def test_append_mode_does_not_deduplicate_raw_ticks(connection) -> None:
    """Ticks are append-only: identical rows are two ticks, not one."""
    insert_rows(connection, "ticks", [make_tick(), make_tick()])
    assert count_rows(connection, "ticks") == 2


def test_replace_mode_is_idempotent_on_the_primary_key(connection) -> None:
    payload = {
        "timestamp": TS,
        "tradingsymbol": "NIFTY 50",
        "timeframe": "1d",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 0,
        "oi": None,
        "source": "ZERODHA_REST",
        "fetch_timestamp": TS,
        "quality_flag": "OK",
        "run_id": RUN_ID,
    }
    assert insert_rows(connection, "index_history", [payload], mode="replace") == 1
    improved = {**payload, "close": 106.0}
    assert insert_rows(connection, "index_history", [improved], mode="replace") == 1
    assert count_rows(connection, "index_history") == 1
    close = connection.execute("SELECT close FROM index_history").fetchone()[0]
    assert close == pytest.approx(106.0)


def test_count_rows_supports_a_filter(connection) -> None:
    insert_rows(connection, "ticks", [make_tick(), make_tick(strike=25050.0)])
    assert count_rows(connection, "ticks", where="strike = 25050.0") == 1
