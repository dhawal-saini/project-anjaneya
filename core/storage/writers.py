"""Low-level row writing helpers shared by every ANJANEYA app.

Everything in ANJANEYA ultimately writes *rows* to DuckDB: ticks and snapshots
from the capture app, index bars from the REST backfill, candles/features from
the engine. Rather than let each app grow its own hand-rolled ``INSERT`` string,
the SQL lives here once, with three guarantees that matter:

1. **Positional order is taken from the contract, never guessed.** Columns are
   derived from the row dictionaries (first-seen order) or from the caller, and
   every row is expanded to the same column list, so two rows with different
   keys can never shift values between columns.
2. **Identifiers are validated and quoted.** A table or column name must match
   ``[A-Za-z_][A-Za-z0-9_]*``; anything else raises instead of reaching DuckDB.
3. **The write mode is explicit.** ``append`` (raw tables - ticks, snapshots),
   ``replace`` (idempotent re-fetch: index history, coverage) or ``ignore``.

A ``replace`` batch is written as a **staged merge**, not as one ``INSERT OR
REPLACE`` per row: the rows land in an unconstrained temporary table (a bulk
insert), the target's matching primary keys are deleted with a single join, and
the stage is inserted once. DuckDB resolves every ``INSERT OR REPLACE`` against
the primary-key index separately, so the per-row form costs ~10 ms per row - a
quarter of option history (hundreds of thousands of 1-minute bars) took over an
hour, while the staged form does the same work in well under a minute. The
per-row form is still used when the table has no discoverable primary key, so
behaviour never silently changes.

Importing this module performs no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

__all__ = [
    "WriteMode",
    "count_rows",
    "insert_rows",
    "quote_identifier",
    "row_columns",
]

WriteMode = Literal["append", "replace", "ignore"]

#: Conservative identifier pattern: anything else is a bug (or an injection attempt).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: The staging table used by the bulk ``replace`` path (per connection, dropped after use).
_STAGE_TABLE = "_anj_upsert_stage"

#: How many rows go into one multi-row ``VALUES`` statement. DuckDB re-plans every
#: parametrised statement, so ~500 rows per statement measured 16x faster than
#: ``executemany`` (3.9k vs 240 rows/s) with no measurable difference from 100 or 2000.
_CHUNK_ROWS = 500

#: Batches at or below this size keep the simple one-statement-per-row path (it is
#: clearer, and test doubles inject failures through ``executemany`` there).
_BULK_THRESHOLD = 25

_MODE_KEYWORDS: dict[str, str] = {
    "append": "INSERT INTO",
    "replace": "INSERT OR REPLACE INTO",
    "ignore": "INSERT OR IGNORE INTO",
}


def _primary_key_columns(connection: Any, table: str) -> list[str]:
    """The ``PRIMARY KEY`` columns of ``table``, or ``[]`` when they are unknown.

    DuckDB exposes its constraints as a table function, so the key is read from
    the database instead of being duplicated here. A non-DuckDB connection (a
    test double, a future backend) simply returns ``[]`` and the caller keeps the
    per-row behaviour.
    """
    try:
        row = connection.execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY' "
            "ORDER BY constraint_index LIMIT 1",
            [table],
        ).fetchone()
    except Exception:  # noqa: BLE001 - an unsupported backend takes the per-row path
        return []
    if not row or not row[0]:
        return []
    return [str(name) for name in row[0]]


def _insert_chunked(connection: Any, header: str, values: Sequence[Sequence[Any]]) -> None:
    """Run ``header`` + multi-row ``VALUES`` tuples, ``_CHUNK_ROWS`` rows per statement.

    ``header`` must end with ``VALUES `` (e.g. ``INSERT INTO "t" ("a", "b") VALUES ``);
    every row in ``values`` must have the same width.
    """
    width = len(values[0])
    for start in range(0, len(values), _CHUNK_ROWS):
        chunk = values[start : start + _CHUNK_ROWS]
        rows_sql = ", ".join("(" + ", ".join("?" for _ in range(width)) + ")" for _ in chunk)
        connection.execute(f"{header}{rows_sql}", [value for one in chunk for value in one])


def _bulk_replace(
    connection: Any,
    table: str,
    columns: Sequence[str],
    values: Sequence[Sequence[Any]],
    key: Sequence[str],
) -> int:
    """Replace the conflicting target rows in one pass and insert the batch once.

    ``values`` is written to a temporary clone of the target that carries no
    constraints, so the insert is a plain bulk write. The target's rows whose
    key appears in the stage are then deleted with a single correlated
    ``EXISTS``, and the stage is copied in once. A key repeated inside the batch
    keeps its *last* occurrence, which is what the per-row ``INSERT OR REPLACE``
    did.
    """
    target = quote_identifier(table)
    stage = quote_identifier(_STAGE_TABLE)
    sequence = quote_identifier("_anj_upsert_seq")
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    key_sql = ", ".join(quote_identifier(column) for column in key)

    connection.execute(f"DROP TABLE IF EXISTS temp.{stage}")
    connection.execute(
        f"CREATE TEMP TABLE {stage} AS "
        f"SELECT *, 0::BIGINT AS {sequence} FROM {target} LIMIT 0"
    )
    _insert_chunked(
        connection,
        f"INSERT INTO {stage} ({column_sql}, {sequence}) VALUES ",
        [[*value, position] for position, value in enumerate(values)],
    )
    match = " AND ".join(
        f"{target}.{quote_identifier(column)} = {stage}.{quote_identifier(column)}"
        for column in key
    )
    connection.execute(f"DELETE FROM {target} WHERE EXISTS (SELECT 1 FROM {stage} WHERE {match})")
    connection.execute(
        f"INSERT INTO {target} ({column_sql}) SELECT {column_sql} FROM {stage} "
        f"QUALIFY row_number() OVER (PARTITION BY {key_sql} ORDER BY {sequence} DESC) = 1"
    )
    connection.execute(f"DROP TABLE IF EXISTS temp.{stage}")
    return len(values)



def quote_identifier(name: str) -> str:
    """Validate and double-quote a SQL identifier.

    Raises
    ------
    ValueError
        If ``name`` is not a plain identifier. Table/column names in ANJANEYA
        are compile-time constants, so this is a guard rail, not escaping.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier {name!r}; expected [A-Za-z_][A-Za-z0-9_]*")
    return f'"{name}"'


def _as_mapping(row: Any) -> Mapping[str, Any]:
    """Coerce one row into a plain mapping.

    Accepts an :class:`~core.storage.…` row model (anything exposing
    ``to_record``) or a mapping. ``to_record(exclude_none=False)`` is used for
    models so a sparse row still supplies every column explicitly - otherwise a
    ``NULL`` could be mistaken for "column omitted".
    """
    to_record = getattr(row, "to_record", None)
    if callable(to_record):
        return to_record(exclude_none=False)
    if isinstance(row, Mapping):
        return row
    raise TypeError(f"Cannot store a row of type {type(row).__name__}; expected a mapping or model")


def row_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Union of the rows' keys, in first-seen order."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def insert_rows(
    connection: Any,
    table: str,
    rows: Iterable[Any],
    *,
    mode: WriteMode = "append",
    columns: Sequence[str] | None = None,
) -> int:
    """Insert ``rows`` into ``table`` and return the number of rows written.

    Parameters
    ----------
    connection:
        A DuckDB connection (duckdb or a test double exposing ``execute``).
    table:
        Target table name; validated with :func:`quote_identifier`.
    rows:
        Mappings or pydantic row models. An empty iterable is a no-op.
    mode:
        ``"append"`` (default), ``"replace"`` or ``"ignore"``.
    columns:
        Explicit column list/order. When omitted the columns are taken from the
        rows themselves (first-seen order) - pass it explicitly when writing into
        a table with NOT NULL columns and a possibly-empty ``rows`` batch.
    """
    if mode not in _MODE_KEYWORDS:
        raise ValueError(f"Unknown write mode {mode!r}; expected one of {sorted(_MODE_KEYWORDS)}")

    records = [_as_mapping(row) for row in rows]
    if not records:
        return 0

    resolved = list(columns) if columns is not None else row_columns(records)
    for identifier in (table, *resolved):
        quote_identifier(identifier)

    values = [[record.get(column) for column in resolved] for record in records]

    if len(values) > _BULK_THRESHOLD:
        if mode == "replace":
            key = _primary_key_columns(connection, table)
            if key:
                return _bulk_replace(connection, table, resolved, values, key)
        column_sql = ", ".join(quote_identifier(column) for column in resolved)
        _insert_chunked(
            connection,
            f"{_MODE_KEYWORDS[mode]} {quote_identifier(table)} ({column_sql}) VALUES ",
            values,
        )
        return len(records)

    placeholders = ", ".join("?" for _ in resolved)
    column_sql = ", ".join(quote_identifier(column) for column in resolved)
    sql = f"{_MODE_KEYWORDS[mode]} {quote_identifier(table)} ({column_sql}) VALUES ({placeholders})"
    connection.executemany(sql, values)
    return len(records)


def count_rows(connection: Any, table: str, *, where: str | None = None) -> int:
    """Return ``count(*)`` for ``table`` (optionally filtered).

    ``where`` is deliberately a raw SQL fragment: callers pass literals they
    built themselves (e.g. ``"run_id = 'abc'"``). It is only ever appended by
    internal code, never by data.
    """
    sql = f"SELECT count(*) FROM {quote_identifier(table)}"
    if where:
        sql = f"{sql} WHERE {where}"
    row = connection.execute(sql).fetchone()
    return int(row[0]) if row else 0
