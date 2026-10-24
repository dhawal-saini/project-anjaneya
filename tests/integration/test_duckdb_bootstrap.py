"""Integration tests for the DuckDB bootstrap and the on-disk schema.

The most valuable test here compares the **physical** table columns with the
pydantic row models. If somebody adds a field to ``FeatureRow`` but forgets the
DDL (or vice versa), capture would silently drop data - this test makes that
impossible.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.schemas import CandleRow, FeatureRow, LabelRow, TickRow, TradeRecord
from core.storage import (
    DatabaseLockedError,
    SchemaVersionError,
    bootstrap_database,
    connect,
    current_schema_version,
    read_schema_sql,
)

EXPECTED_TABLES = {
    "candles",
    "data_coverage",
    "features",
    "index_history",
    "instruments",
    "labels",
    "option_chain_snapshots",
    "run_manifest",
    "schema_version",
    "ticks",
    "trades",
}

#: Physical table -> the pydantic model that describes its rows.
MODEL_TABLES = {
    "ticks": TickRow,
    "candles": CandleRow,
    "features": FeatureRow,
    "labels": LabelRow,
    "trades": TradeRecord,
}


def columns_of(connection, table: str) -> list[str]:
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [row[0] for row in rows]


def test_bootstrap_creates_every_table(tmp_path: Path) -> None:
    result = bootstrap_database(tmp_path / "db" / "anjaneeya.duckdb")
    assert result.created_file is True
    assert result.applied_schema_now is True
    assert result.schema_version == 1
    assert set(result.tables) >= EXPECTED_TABLES
    assert result.db_path.exists()


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "anjaneeya.duckdb"
    first = bootstrap_database(db_path)
    second = bootstrap_database(db_path)

    assert first.created_file is True
    assert second.created_file is False
    assert second.applied_schema_now is False
    assert second.schema_version == 1
    assert set(second.tables) == set(first.tables)

    connection = connect(db_path, read_only=True)
    try:
        versions = connection.execute("SELECT version FROM schema_version").fetchall()
    finally:
        connection.close()
    # Exactly one bookkeeping row, no duplicated application of the schema.
    assert versions == [(1,)]


def test_bootstrap_without_applying_the_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.duckdb"
    result = bootstrap_database(db_path, apply_schema=False)
    assert result.applied_schema_now is False
    assert result.tables == []


_HOLD_DB_SCRIPT = """
import sys, time
import duckdb

connection = duckdb.connect(sys.argv[1])
connection.execute("SELECT 1").fetchone()
print("locked", flush=True)
time.sleep(60)
"""


def test_a_database_locked_by_another_process_is_reported_clearly(tmp_path: Path) -> None:
    """DuckDB allows one writing process; the error must be actionable.

    This is a real operational constraint: ANJANEYA apps are run one at a time
    against a single database file. Reproducing it requires a second *process*,
    because DuckDB permits multiple connections within one process.
    """
    db_path = tmp_path / "locked.duckdb"
    bootstrap_database(db_path)

    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_DB_SCRIPT, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        deadline = time.monotonic() + 30
        line = ""
        while time.monotonic() < deadline:
            line = holder.stdout.readline().strip()
            if line:
                break
        assert line == "locked", f"holder process did not start: {holder.stderr.read()}"

        with pytest.raises(DatabaseLockedError, match="single writer"):
            bootstrap_database(db_path)
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_newer_schema_version_is_refused(tmp_path: Path) -> None:
    """Downgrading the code must not silently mis-read a newer database."""
    db_path = tmp_path / "future.duckdb"
    bootstrap_database(db_path, schema_version=2)
    connection = connect(db_path)
    try:
        assert current_schema_version(connection) == 2
    finally:
        connection.close()

    with pytest.raises(SchemaVersionError, match="schema_version"):
        bootstrap_database(db_path, schema_version=1)


def test_session_timezone_is_ist(tmp_path: Path) -> None:
    db_path = tmp_path / "tz.duckdb"
    bootstrap_database(db_path)
    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute("SELECT current_setting('TimeZone')").fetchone()
    finally:
        connection.close()
    assert row[0] == "Asia/Kolkata"


def test_schema_sql_is_shipped_as_a_resource() -> None:
    ddl = read_schema_sql()
    assert "CREATE TABLE IF NOT EXISTS ticks" in ddl
    # Every statement must be idempotent so bootstrap can run on every startup.
    assert ddl.count("CREATE TABLE IF NOT EXISTS") == len(EXPECTED_TABLES)


@pytest.mark.parametrize("table", sorted(MODEL_TABLES))
def test_physical_columns_match_the_row_model(tmp_path: Path, table: str) -> None:
    """The DDL and the pydantic contract must describe the same columns."""
    db_path = tmp_path / "parity.duckdb"
    bootstrap_database(db_path)
    connection = connect(db_path, read_only=True)
    try:
        physical = columns_of(connection, table)
    finally:
        connection.close()

    model = MODEL_TABLES[table]
    assert set(physical) == set(model.model_fields), (
        f"{table}: DDL and {model.__name__} disagree "
        f"(only in DDL: {sorted(set(physical) - set(model.model_fields))}, "
        f"only in model: {sorted(set(model.model_fields) - set(physical))})"
    )


def test_run_manifest_table_accepts_a_real_manifest_row(tmp_path: Path) -> None:
    """The manifest insert path used by finalize_run must work end to end."""
    import json

    from core.app_bootstrap import bootstrap_app, finalize_run, parse_args

    db_path = tmp_path / "runs.duckdb"
    data_root = tmp_path / "data"
    args = parse_args(
        ["--mode", "status", "--data-root", str(data_root), "--db-path", str(db_path)],
        app_name="anjaneeya_capture",
    )
    context = bootstrap_app("anjaneeya_capture", args)
    exit_code = finalize_run(context, status="SUCCESS")
    assert exit_code == 0

    manifest_path = context.run_context.artifacts_dir / "manifest.json"
    assert manifest_path.exists()

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT run_id, app_name, status, config_hash FROM run_manifest"
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 1
    assert rows[0][0] == context.run_context.run_id
    assert rows[0][1] == "anjaneeya_capture"
    assert rows[0][2] == "SUCCESS"
    assert rows[0][3] == context.config.config_hash

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
