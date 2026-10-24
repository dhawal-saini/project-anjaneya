"""``core/storage/schema.sql`` and the pydantic row models must never drift.

Phase 1 tested that a *duckdb database built from the DDL* can hold the models'
rows. This test closes the gap one level earlier: it parses the DDL text and
asserts that, for every table a model writes to, the column list is **identical
in name and order** to the model's field order.

Why order matters: the capture writer inserts rows positionally
(``INSERT INTO t VALUES (?, ?, ...)`` built from ``model_fields``), so a column
reordered in one place and not the other silently writes prices into the volume
column. This is a pure text/AST check - no database, no I/O beyond reading the
shipped SQL file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.schemas import (
    CandleRow,
    CoverageRow,
    FeatureRow,
    IndexHistoryRow,
    InstrumentRow,
    LabelRow,
    OptionChainSnapshotRow,
    TickRow,
    TradeRecord,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "core" / "storage" / "schema.sql"

CREATE_TABLE_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS (?P<table>\w+)\s*\((?P<body>.*?)\n\)\s*;",
    re.DOTALL,
)

#: Table name -> model whose field order must match the DDL.
TABLE_MODELS: dict[str, type] = {
    "ticks": TickRow,
    "option_chain_snapshots": OptionChainSnapshotRow,
    "index_history": IndexHistoryRow,
    "instruments": InstrumentRow,
    "data_coverage": CoverageRow,
    "candles": CandleRow,
    "features": FeatureRow,
    "labels": LabelRow,
    "trades": TradeRecord,
}

#: Columns declared inside the DDL body that are not part of the row contract.
TABLE_CONSTRAINTS = {"PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "CONSTRAINT"}


@pytest.fixture(scope="module")
def ddl_columns() -> dict[str, list[str]]:
    """Parse ``column name`` out of every ``CREATE TABLE`` in the shipped DDL."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    tables: dict[str, list[str]] = {}
    for match in CREATE_TABLE_RE.finditer(text):
        columns: list[str] = []
        for raw_line in match.group("body").strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.split()[0].upper() in TABLE_CONSTRAINTS:
                continue
            columns.append(line.split()[0])
        tables[match.group("table")] = columns
    return tables


def test_every_write_table_is_covered_by_a_model(ddl_columns: dict[str, list[str]]) -> None:
    """No table may be created in SQL without a Python contract for it."""
    bookkeeping = {"schema_version", "run_manifest"}
    missing = sorted(set(ddl_columns) - set(TABLE_MODELS) - bookkeeping)
    assert not missing, (
        f"tables {missing} exist in schema.sql without a row model; add one (and to "
        "core.constants) or add it to the bookkeeping allow-list with a reason"
    )


@pytest.mark.parametrize("table", sorted(TABLE_MODELS))
def test_ddl_columns_match_model_field_order(table: str, ddl_columns: dict[str, list[str]]) -> None:
    """The DDL column list must equal the model's field order, exactly."""
    assert table in ddl_columns, f"table {table} is missing from schema.sql"
    model = TABLE_MODELS[table]
    assert ddl_columns[table] == list(model.model_fields), (
        f"{table}: schema.sql columns {ddl_columns[table]} != "
        f"{model.__name__} fields {list(model.model_fields)}"
    )


def test_ddl_is_idempotent(ddl_columns: dict[str, list[str]]) -> None:
    """Every table is created with IF NOT EXISTS (re-running the DDL is a no-op)."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert text.count("CREATE TABLE ") == len(re.findall(r"CREATE TABLE IF NOT EXISTS ", text))
