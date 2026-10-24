"""DuckDB + Parquet writer for captured data (batching, mirroring, coverage).

Why a writer *object* rather than a function
--------------------------------------------
A capture session appends tens of thousands of ticks. Two things must hold:

1. **Nothing is lost when the disk is slow.** Rows are buffered in memory and
   flushed in batches; if the database flush fails, the batch is still written to
   a Parquet part file instead of being dropped, and the failure is counted in
   the run summary.
2. **The Parquet mirror is layout-stable.** Every flush writes one immutable
   part file under ``<stream>/<partition>/``; :meth:`CaptureWriter.compact_parquet`
   merges the parts into one file per partition using DuckDB itself (so the
   Parquet schema is exactly the DuckDB schema - no pandas dtype guessing in the
   middle).

Raw streams are append-only (never updated); index history and coverage rows use
``INSERT OR REPLACE`` because they are keyed and therefore idempotent to re-fetch.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.constants import (
    CANDLE_COLUMNS,
    COVERAGE_COLUMNS,
    INDEX_HISTORY_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    TICK_COLUMNS,
)
from core.storage import connect, insert_rows
from core.storage.coverage import CoverageTracker, StreamStats, build_coverage_row

__all__ = [
    "SNAPSHOT_STREAM",
    "SNAPSHOT_TIMEFRAME",
    "TICK_STREAM",
    "TICK_TIMEFRAME",
    "CaptureWriter",
    "CoverageTracker",
    "StreamStats",
    "WriteStats",
    "build_coverage_row",
]

TICK_STREAM = "ticks"
SNAPSHOT_STREAM = "option_chain_snapshots"

#: Parquet directory name -> (table, column contract).
STREAM_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    TICK_STREAM: ("ticks", TICK_COLUMNS),
    SNAPSHOT_STREAM: ("option_chain_snapshots", OPTION_CHAIN_COLUMNS),
}

#: The tick stream's coverage verdict is keyed under this pseudo-timeframe.
TICK_TIMEFRAME = "TICK"
SNAPSHOT_TIMEFRAME = "SNAPSHOT"


@dataclass(slots=True)
class WriteStats:
    """Cumulative counters for one writer's lifetime."""

    ticks: int = 0
    snapshots: int = 0
    index_history: int = 0
    candles: int = 0
    coverage: int = 0
    flushes: int = 0
    parquet_files: int = 0
    spilled_rows: int = 0
    dropped_rows: int = 0
    failures: int = 0

    def describe(self) -> dict[str, int]:
        """Log-safe summary."""
        return {
            "ticks": self.ticks,
            "snapshots": self.snapshots,
            "index_history": self.index_history,
            "candles": self.candles,
            "coverage": self.coverage,
            "flushes": self.flushes,
            "parquet_files": self.parquet_files,
            "spilled_rows": self.spilled_rows,
            "dropped_rows": self.dropped_rows,
            "failures": self.failures,
        }


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------
class CaptureWriter:
    """Batch-append captured rows to DuckDB and mirror them to Parquet."""

    def __init__(
        self,
        *,
        db_path: Any,
        parquet_root: Any,
        storage_config: Any,
        run_id: str,
        logger: Any = None,
        clock: Callable[[], float] | None = None,
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        import time as _time

        self.db_path = db_path
        self.parquet_root = parquet_root
        self.config = storage_config
        self.run_id = run_id
        self.logger = logger
        self._clock = clock or _time.monotonic
        self._connect = connection_factory or connect

        self.stats = WriteStats()
        self._buffers: dict[str, list[Any]] = {TICK_STREAM: [], SNAPSHOT_STREAM: []}
        self._last_flush = self._clock()
        self._connection: Any = None
        self._part_counter = 0

    # -- introspection ------------------------------------------------------
    @property
    def connection(self) -> Any:
        """The open DuckDB writer connection, created on demand.

        Exposed for status/query use during a session - and important to know
        about: DuckDB allows a single read-write process per file, so *no other
        process* (not even ``--mode status``) can open this database while the
        writer is alive.
        """
        return self._ensure_connection()

    @property
    def buffered_rows(self) -> int:
        """Rows currently waiting to be flushed."""
        return sum(len(buffer) for buffer in self._buffers.values())

    def describe(self) -> dict[str, Any]:
        """Log-safe summary."""
        return {"buffered_rows": self.buffered_rows, **self.stats.describe()}

    def _now(self) -> datetime:
        """Wall-clock timestamp used for file naming (a monotonic clock has no date)."""
        from datetime import UTC

        return datetime.now(UTC)

    # -- connection ---------------------------------------------------------
    def _ensure_connection(self) -> Any:
        """Open (once) the DuckDB writer connection."""
        if self._connection is None:
            self._connection = self._connect(self.db_path)
        return self._connection

    # -- buffering ----------------------------------------------------------
    def write_ticks(self, rows: Iterable[Any]) -> int:
        """Buffer tick rows, flushing when a batch is full."""
        return self._buffer(TICK_STREAM, rows)

    def write_snapshots(self, rows: Iterable[Any]) -> int:
        """Buffer option-chain snapshot rows, flushing when a batch is full."""
        return self._buffer(SNAPSHOT_STREAM, rows)

    def _buffer(self, stream: str, rows: Iterable[Any]) -> int:
        batch = list(rows)
        if not batch:
            return 0
        self._buffers[stream].extend(batch)
        rows_threshold = len(self._buffers[stream]) >= self.config.flush_rows
        time_threshold = (self._clock() - self._last_flush) >= self.config.flush_seconds
        if rows_threshold or time_threshold:
            self.flush(force=True)
        return len(batch)

    def maybe_flush(self) -> bool:
        """Flush when the time threshold has elapsed; True when it flushed."""
        if self.buffered_rows == 0:
            return False
        if self._clock() - self._last_flush >= self.config.flush_seconds:
            self.flush(force=True)
            return True
        return False

    # -- flushing -----------------------------------------------------------
    def flush(self, *, force: bool = False) -> WriteStats:
        """Write every buffered stream to DuckDB (and mirror it to Parquet)."""
        pending = {stream: list(rows) for stream, rows in self._buffers.items() if rows}
        if not pending:
            self._last_flush = self._clock()
            return self.stats

        for stream, rows in pending.items():
            self._flush_stream(stream, rows)
            self._buffers[stream].clear()
        self._last_flush = self._clock()
        self.stats.flushes += 1
        return self.stats

    def _flush_stream(self, stream: str, rows: list[Any]) -> None:
        """Insert one stream's batch, then mirror it - never silently lose a batch."""
        table, columns = STREAM_TABLES[stream]
        database_ok = False
        try:
            connection = self._ensure_connection()
            insert_rows(connection, table, rows, mode="append", columns=columns)
            database_ok = True
            if stream == TICK_STREAM:
                self.stats.ticks += len(rows)
            else:
                self.stats.snapshots += len(rows)
        except Exception as exc:  # noqa: BLE001 - the Parquet mirror is the fallback
            self.stats.failures += 1
            self.stats.spilled_rows += len(rows)
            if self.logger is not None:
                self.logger.error(
                    f"DuckDB insert into {table} failed ({type(exc).__name__}: {exc}); "
                    f"spilling {len(rows)} row(s) to the Parquet mirror so nothing is lost"
                )

        if not self.config.mirror_parquet:
            if not database_ok:
                self.stats.dropped_rows += len(rows)
            return
        try:
            self._write_parquet_part(stream, rows)
        except Exception as exc:  # noqa: BLE001 - mirroring must never kill capture
            self.stats.failures += 1
            if not database_ok:
                self.stats.dropped_rows += len(rows)
            message = f"Parquet mirror for {stream} failed ({type(exc).__name__}: {exc})"
            if not database_ok:
                message += f"; {len(rows)} row(s) could not be persisted anywhere"
            if self.logger is not None:
                self.logger.error(message)


    # -- Parquet mirror -----------------------------------------------------
    def _partition_key(self, row: Any) -> str:
        """Directory name for a batch partition (by day, local time)."""
        if self.config.parquet_partition == "none":
            return "all"
        moment = getattr(row, "timestamp", None)
        if isinstance(moment, datetime):
            return moment.date().isoformat()
        return self._now().date().isoformat()

    def _write_parquet_part(self, stream: str, rows: list[Any]) -> Any:
        """Write one immutable Parquet part file for a batch.

        The part file is built through DuckDB: a temporary table with the *same*
        schema as the target table, then ``COPY ... TO``. That way the Parquet
        types are exactly the DuckDB types - no pandas dtype inference in between.
        """
        table, columns = STREAM_TABLES[stream]
        partition = self._partition_key(rows[0])
        directory = self.parquet_root / stream / partition
        directory.mkdir(parents=True, exist_ok=True)
        self._part_counter += 1
        name = f"part-{self._now():%Y%m%dT%H%M%S}-{self._part_counter:05d}.parquet"
        target = directory / name

        connection = self._ensure_connection()
        connection.execute(
            f'CREATE OR REPLACE TEMP TABLE "_anj_mirror" AS SELECT * FROM "{table}" LIMIT 0'
        )
        insert_rows(connection, "_anj_mirror", rows, mode="append", columns=columns)
        connection.execute(
            f'COPY "_anj_mirror" TO \'{target.as_posix()}\' ({self._parquet_options()})'
        )
        self.stats.parquet_files += 1
        return target

    def _parquet_options(self) -> str:
        """``COPY`` options for the configured compression."""
        if self.config.parquet_compression == "uncompressed":
            return "FORMAT PARQUET, COMPRESSION UNCOMPRESSED"
        return f"FORMAT PARQUET, COMPRESSION {self.config.parquet_compression.upper()}"

    def parquet_parts(self, stream: str | None = None) -> list[Any]:
        """Every Parquet part file written so far (sorted)."""
        streams = [stream] if stream else list(STREAM_TABLES)
        parts: list[Any] = []
        for name in streams:
            directory = self.parquet_root / name
            if directory.is_dir():
                parts.extend(sorted(directory.rglob("part-*.parquet")))
        return parts


    def compact_parquet(self, *, delete_parts: bool = True) -> list[Any]:
        """Merge each partition's parts into ``<partition>.parquet``.

        Compact files keep the mirror cheap to scan (one file per stream per day)
        while part files stay immutable and crash-safe during the session.
        """
        import duckdb

        written: list[Any] = []
        for stream in STREAM_TABLES:
            directory = self.parquet_root / stream
            if not directory.is_dir():
                continue
            partitions: dict[str, list[Any]] = {}
            for part in sorted(directory.rglob("part-*.parquet")):
                partitions.setdefault(part.parent.name, []).append(part)
            compacted_here = 0
            for partition, parts in sorted(partitions.items()):
                target = directory / partition / f"{partition}.parquet"
                files = ", ".join(f"'{part.as_posix()}'" for part in parts)
                try:
                    connection = duckdb.connect()
                    try:
                        connection.execute(
                            f"COPY (SELECT * FROM read_parquet([{files}])) TO "
                            f"'{target.as_posix()}' ({self._parquet_options()})"
                        )
                    finally:
                        connection.close()
                except Exception as exc:  # noqa: BLE001 - compaction is an optimisation
                    if self.logger is not None:
                        self.logger.warning(
                            f"Parquet compaction for {stream}/{partition} failed "
                            f"({type(exc).__name__}: {exc}); part files are kept as-is"
                        )
                    continue
                written.append(target)
                compacted_here += 1
                if delete_parts:
                    for part in parts:
                        part.unlink(missing_ok=True)
            if compacted_here and self.logger is not None:
                self.logger.info(
                    f"Parquet mirror compacted for {stream}: {compacted_here} file(s) "
                    f"under {directory}"
                )
        return written

    # -- immediate (keyed) writes ------------------------------------------
    def write_index_history(self, rows: Iterable[Any]) -> int:
        """Upsert index-history rows immediately (small volume, keyed table)."""
        batch = list(rows)
        if not batch:
            return 0
        connection = self._ensure_connection()
        written = insert_rows(
            connection, "index_history", batch, mode="replace", columns=INDEX_HISTORY_COLUMNS
        )
        self.stats.index_history += written
        return written

    def write_candles(self, rows: Iterable[Any]) -> int:
        """Upsert option candles immediately (used by ``--mode backfill-options``).

        Keyed on ``(timestamp, tradingsymbol, timeframe)`` exactly like the engine's
        tick-built candles, so a REST-bootstrapped bar and a later tick-built bar for
        the same minute are the same row - the tick build wins by re-writing it.
        """
        batch = list(rows)
        if not batch:
            return 0
        connection = self._ensure_connection()
        written = insert_rows(
            connection, "candles", batch, mode="replace", columns=CANDLE_COLUMNS
        )
        self.stats.candles += written
        return written

    def write_coverage(self, rows: Iterable[Any]) -> int:
        """Upsert coverage rows immediately (one row per stream per day)."""
        batch = list(rows)
        if not batch:
            return 0
        connection = self._ensure_connection()
        written = insert_rows(
            connection, "data_coverage", batch, mode="replace", columns=COVERAGE_COLUMNS
        )
        self.stats.coverage += written
        return written

    # -- shutdown -----------------------------------------------------------
    def close(self, *, compact: bool = True) -> WriteStats:
        """Flush everything, optionally compact the mirror, and close DuckDB."""
        self.flush(force=True)
        if compact and self.config.mirror_parquet:
            self.compact_parquet()
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None
        if self.logger is not None:
            self.logger.info(f"storage writer closed: {self.describe()}")
        return self.stats

