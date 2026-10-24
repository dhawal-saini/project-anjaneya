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

_MODE_KEYWORDS: dict[str, str] = {
    "append": "INSERT INTO",
    "replace": "INSERT OR REPLACE INTO",
    "ignore": "INSERT OR IGNORE INTO",
}


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
