"""DuckDB bootstrap: create/verify the ANJANEYA analytical database.

The database is the project's query engine over the captured data
(``db/anjaneeya.duckdb`` by default). This module owns

* creating the file and applying ``schema.sql`` (idempotent),
* recording which ``schema_version`` is live,
* refusing to open a database that is **newer** than the running code
  (a downgrade would silently mis-read columns),
* exposing the physical table list for ``scripts/health_check.py``.

Importing this module performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import duckdb

__all__ = [
    "BootstrapResult",
    "DatabaseLockedError",
    "SCHEMA_RESOURCE",
    "bootstrap_database",
    "connect",
    "current_schema_version",
    "read_schema_sql",
    "table_names",
]

#: Resource (inside this package) holding the DDL.
SCHEMA_RESOURCE = "schema.sql"

#: Session timezone set on every connection so TIMESTAMPTZ renders IST.
SESSION_TIMEZONE = "Asia/Kolkata"


class SchemaVersionError(RuntimeError):
    """Raised when the on-disk schema version is incompatible with the code."""


class DatabaseLockedError(RuntimeError):
    """Raised when another process holds the DuckDB file open for writing.

    DuckDB allows a single read-write process per file. ANJANEYA's apps are
    therefore run one at a time (capture during the session, engine and AI
    afterwards); this error turns the raw IO failure into an actionable message.
    """


@dataclass(slots=True)
class BootstrapResult:
    """Outcome of :func:`bootstrap_database`."""

    db_path: Path
    created_file: bool
    schema_version: int
    applied_schema_now: bool
    tables: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Log-safe summary."""
        return {
            "db_path": str(self.db_path),
            "created_file": self.created_file,
            "schema_version": self.schema_version,
            "applied_schema_now": self.applied_schema_now,
            "tables": list(self.tables),
        }


def read_schema_sql() -> str:
    """Return the DDL text shipped inside :mod:`core.storage`."""
    return resources.files("core.storage").joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")


def connect(db_path: str | Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the ANJANEYA session settings applied.

    The session timezone is pinned to IST so that ``TIMESTAMPTZ`` values are
    rendered in exchange-local time for interactive queries, regardless of the
    machine's own timezone.
    """
    connection = duckdb.connect(str(db_path), read_only=read_only)
    connection.execute(f"SET TimeZone='{SESSION_TIMEZONE}'")
    return connection


def table_names(connection: duckdb.DuckDBPyConnection) -> list[str]:
    """Physical table names in the ``main`` schema, sorted."""
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [row[0] for row in rows]


def current_schema_version(connection: duckdb.DuckDBPyConnection) -> int | None:
    """Highest recorded schema version, or ``None`` when unrecorded."""
    exists = connection.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = 'schema_version'"
    ).fetchone()
    if not exists or exists[0] == 0:
        return None
    row = connection.execute("SELECT max(version) FROM schema_version").fetchone()
    return int(row[0]) if row and row[0] is not None else None


def bootstrap_database(
    db_path: str | Path,
    *,
    schema_version: int = 1,
    description: str | None = None,
    create_dirs: bool = True,
    apply_schema: bool = True,
) -> BootstrapResult:
    """Create/upgrade the ANJANEYA DuckDB file and return a summary.

    Parameters
    ----------
    db_path:
        Target ``.duckdb`` file.
    schema_version:
        Version this code expects (``settings.project.schema_version``).
    description:
        Free-text description recorded in the ``schema_version`` table.
    create_dirs:
        Create the parent directory when missing.
    apply_schema:
        When False only the version compatibility check is performed.

    Raises
    ------
    SchemaVersionError
        If the database was written by a newer ANJANEYA version.
    DatabaseLockedError
        If another process currently holds the file open for writing.
    """
    path = Path(db_path)
    created_file = not path.exists()
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        connection = connect(path)
    except duckdb.Error as exc:
        # DuckDB permits one read-write process at a time; anything else is a
        # genuine failure and is re-raised untouched.
        message = str(exc)
        if "being used by another process" in message or "lock" in message.lower():
            raise DatabaseLockedError(
                f"The DuckDB file {path} is already open for writing by another process "
                "(DuckDB allows a single writer). Stop that run first, or point --db-path at "
                "a different file."
            ) from exc
        raise

    try:
        existing_version = current_schema_version(connection)
        if existing_version is not None and existing_version > schema_version:
            raise SchemaVersionError(
                f"DuckDB file {path} uses schema_version {existing_version}, but this code "
                f"expects {schema_version}. Refusing to continue: upgrade ANJANEYA instead of "
                "downgrading the data."
            )

        applied_now = False
        if apply_schema and (created_file or existing_version is None):
            connection.execute(read_schema_sql())
            applied_now = True

        if apply_schema and (applied_now or existing_version is None):
            connection.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                [schema_version, description or f"ANJANEYA schema v{schema_version}"],
            )

        tables = table_names(connection)
        live_version = current_schema_version(connection) or schema_version
    finally:
        connection.close()

    return BootstrapResult(
        db_path=path,
        created_file=created_file,
        schema_version=live_version,
        applied_schema_now=applied_now,
        tables=tables,
    )
