"""DuckDB + Parquet persistence for derived rows: candles and feature rows.

Why delete-then-append instead of ``INSERT OR REPLACE``
------------------------------------------------------
Coverage and index history use ``INSERT OR REPLACE`` because a re-fetch replaces
exactly the rows it is about. A derived-table rebuild is different: its scope is a
*session* (candles) or a *contract-day* (features), and the correct number of rows in
that scope is only known after the build. If yesterday's build produced 09:15-15:29
and today's tick set ends at 12:00, an upsert would leave 12:01-15:29 from the
previous attempt silently in place - the worst possible outcome for a research
dataset (stale rows that look real). So a rebuild deletes the scope first and then
appends the fresh rows.

Why the feature scope is a contract-day and not ``(day, timeframe, contract)``
-----------------------------------------------------------------------------
``features`` is keyed ``(timestamp, tradingsymbol)`` - no timeframe (spec 18: "one
feature row per (timestamp, contract)"). Two timeframes of the same contract share
the 09:15 bar, so they cannot both live in that table: the second insert would break
the primary key. A feature build is therefore authoritative for the whole
contract-day, and the ``timeframe`` column records which candle series produced the
rows. Switching timeframe for a day is a rebuild of that day, which is exactly what
delete-then-append gives you.

Why the greeks UPDATE instead of another delete-then-append
----------------------------------------------------------
A candle is only ever *replaced* by the pass that builds it from ticks.
``--mode enrich-greeks`` does not build bars: it fills ``iv``/``delta``/``gamma``/
``theta``/``vega``/``days_to_expiry``/``distance_from_atm`` on rows that already exist.
Deleting and re-inserting those rows would let an enrichment run rewrite OHLCV it never
computed (and would silently repair a bar whose ticks were re-captured in between), so
:class:`GreeksWriter` does the opposite: it stages the new greeks in a temp table and
``UPDATE``s exactly those seven columns, keyed by ``(timestamp, timeframe,
tradingsymbol)``. The bar's own ``run_id`` is deliberately left alone - it still names
the build that produced the OHLCV; the enrichment run is recorded in the manifest.

Why one connection, owned by the caller
--------------------------------------
DuckDB allows a single read-write process per database file, and a second connection
to the same file inside one process conflicts just the same. The engine therefore
opens **one** connection for the whole run and hands it here; these classes never open
or close it. That also means the caller must close the connection before
:func:`core.app_bootstrap.finalize_run` writes its manifest row.

Importing this module performs no I/O.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.constants import CANDLE_COLUMNS, COVERAGE_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMNS
from core.options import GREEKS_COLUMNS
from core.storage import insert_rows, quote_identifier

__all__ = [
    "CandleWriteStats",
    "CandleWriter",
    "FeatureWriteStats",
    "FeatureWriter",
    "GreeksWriteStats",
    "GreeksWriter",
    "LabelWriteStats",
    "LabelWriter",
]

#: DuckDB ``COPY`` spelling for the "no compression" configuration values.
_UNCOMPRESSED = {"none", "uncompressed", ""}

#: Temp table the greeks UPDATE reads its new values from: the row key plus the seven
#: greeks columns, in the order the INSERT below writes them.
_GREEKS_TEMP_TABLE = "_anjaneeya_greeks_update"
_GREEKS_UPDATE_COLUMNS = ("tradingsymbol", "timeframe", "timestamp", *GREEKS_COLUMNS)

#: DDL of that temp table. Spelled out rather than inferred from the frame so a row of
#: NULLs still arrives as ``NULL`` (DuckDB would otherwise infer ``VARCHAR``/``INTEGER``
#: from the first batch) and a wrong Python type fails here, loudly, instead of being
#: coerced into the candle table.
_GREEKS_TEMP_DDL = (
    f"CREATE OR REPLACE TEMP TABLE {_GREEKS_TEMP_TABLE} ("
    "tradingsymbol VARCHAR, timeframe VARCHAR, timestamp TIMESTAMPTZ, "
    "iv DOUBLE, delta DOUBLE, gamma DOUBLE, theta DOUBLE, vega DOUBLE, "
    "days_to_expiry DOUBLE, distance_from_atm DOUBLE)"
)


@dataclass(slots=True)
class CandleWriteStats:
    """Counters for one engine run: what was replaced, written and mirrored."""

    candles: int = 0
    deleted: int = 0
    coverage: int = 0
    parquet_files: int = 0
    failures: int = 0

    def describe(self) -> dict[str, int]:
        """Log-safe summary."""
        return {
            "candles": self.candles,
            "deleted": self.deleted,
            "coverage": self.coverage,
            "parquet_files": self.parquet_files,
            "failures": self.failures,
        }


@dataclass(slots=True)
class FeatureWriteStats:
    """Counters for one feature build: what was replaced, written and mirrored."""

    rows: int = 0
    deleted: int = 0
    coverage: int = 0
    parquet_files: int = 0
    failures: int = 0

    def describe(self) -> dict[str, int]:
        """Log-safe summary."""
        return {
            "rows": self.rows,
            "deleted": self.deleted,
            "coverage": self.coverage,
            "parquet_files": self.parquet_files,
            "failures": self.failures,
        }


@dataclass(slots=True)
class GreeksWriteStats:
    """Counters for one IV/greeks enrichment run.

    ``bars`` is everything the model was asked about, ``updated`` the rows whose greek
    columns were actually rewritten. ``unmatched`` should be zero: it counts bars the
    enrichment read but could no longer find in the candle table (a build running
    against the same day while this pass worked). ``unsolved_iv`` counts bars where no
    IV exists under the configured bracket or an input was missing - those are the rows
    whose greeks stay ``NULL``, and the number is reported rather than hidden.
    """

    bars: int = 0
    updated: int = 0
    unmatched: int = 0
    unsolved_iv: int = 0
    parquet_files: int = 0
    failures: int = 0

    def describe(self) -> dict[str, int]:
        """Log-safe summary."""
        return {
            "bars": self.bars,
            "updated": self.updated,
            "unmatched": self.unmatched,
            "unsolved_iv": self.unsolved_iv,
            "parquet_files": self.parquet_files,
            "failures": self.failures,
        }


@dataclass(slots=True)
class LabelWriteStats:
    """Counters for one label build: rows replaced, written and (un)labelled.

    ``labelled`` counts rows that carry a class - the number an operator compares with
    ``rows`` to see how much of a session could be labelled at all. The difference is
    not a failure: the last ``max(horizon)`` bars of every session have no forward
    window, and every row without a threshold (ATR warm-up) has no class either. The
    counts are reported, never hidden.
    """

    rows: int = 0
    labelled: int = 0
    deleted: int = 0
    coverage: int = 0
    parquet_files: int = 0
    failures: int = 0

    def describe(self) -> dict[str, int]:
        """Log-safe summary."""
        return {
            "rows": self.rows,
            "labelled": self.labelled,
            "deleted": self.deleted,
            "coverage": self.coverage,
            "parquet_files": self.parquet_files,
            "failures": self.failures,
        }


class _DerivedWriter:
    """Shared plumbing for the engine's derived tables: scope delete, coverage, mirror.

    ``table``/``order_by`` are class attributes so a subclass declares *what* it writes
    once and inherits the (identical) coverage upsert and Parquet-mirror behaviour.
    """

    #: Target table (validated as an identifier by :func:`quote_identifier`).
    table: str = ""
    #: Column order of the Parquet mirror (deterministic files, diffable between runs).
    order_by: str = "tradingsymbol, timestamp"
    #: True when the table carries a ``timeframe`` column (the mirror filters on it).
    #: ``labels`` does not: spec 20 keys a label row ``(timestamp, contract, horizon)``.
    mirrored_by_timeframe: bool = True

    def __init__(
        self,
        *,
        connection: Any,
        parquet_root: Any,
        run_id: str,
        mirror_parquet: bool = True,
        parquet_compression: str = "zstd",
        logger: Any = None,
    ) -> None:
        self._connection = connection
        self.parquet_root = Path(parquet_root)
        self.run_id = run_id
        self.mirror_parquet = mirror_parquet
        self.compression = _compression_keyword(parquet_compression)
        self.logger = logger
        #: Counters, owned by the subclass (each has its own stats type).
        self.stats: Any = None
        quote_identifier(self.table)

    @property
    def connection(self) -> Any:
        """The borrowed DuckDB connection."""
        return self._connection

    @property
    def parquet_enabled(self) -> bool:
        """True when the Parquet mirror will actually be written."""
        return bool(self.mirror_parquet)

    def parquet_path(self, *, day: date, timeframe: str) -> Path:
        """Mirror file for one ``(day, timeframe)``: ``<tf>/<YYYY-MM-DD>.parquet``."""
        return self.parquet_root / timeframe / f"{day.isoformat()}.parquet"

    def _delete_scope(self, *, start: datetime, end: datetime, where: str, params: list[Any]) -> int:
        """Count then delete the rows of one rebuild scope; returns the deleted count."""
        table = quote_identifier(self.table)
        row = self._connection.execute(
            f"SELECT count(*) FROM {table} WHERE timestamp >= ? AND timestamp < ? AND {where}",
            [start, end, *params],
        ).fetchone()
        deleted = int(row[0]) if row else 0
        self._connection.execute(
            f"DELETE FROM {table} WHERE timestamp >= ? AND timestamp < ? AND {where}",
            [start, end, *params],
        )
        return deleted

    # -- coverage -----------------------------------------------------------
    def write_coverage(self, rows: Iterable[Any]) -> int:
        """Upsert ``data_coverage`` verdicts (keyed, so re-running is idempotent)."""
        written = insert_rows(
            self._connection, "data_coverage", rows, mode="replace", columns=COVERAGE_COLUMNS
        )
        self.stats.coverage += written
        return written

    # -- Parquet mirror -----------------------------------------------------
    def mirror_day(
        self, *, day: date, start: datetime, end: datetime, timeframes: Sequence[str]
    ) -> list[Path]:
        """Mirror one day's rows to ``<timeframe>/<day>.parquet``.

        A failure here is logged and counted, never raised: DuckDB already holds the
        authoritative rows, and a mirror that aborts a whole day's build would turn a
        convenience into a liability.
        """
        if not self.mirror_parquet:
            return []
        written: list[Path] = []
        for timeframe in timeframes:
            path = self._mirror_one(day=day, start=start, end=end, timeframe=timeframe)
            if path is not None:
                written.append(path)
        return written

    def _mirror_one(
        self, *, day: date, start: datetime, end: datetime, timeframe: str
    ) -> Path | None:
        """Write one (day, timeframe) mirror atomically: temp file, then rename.

        The Parquet schema is whatever DuckDB's own ``CREATE TABLE ... AS SELECT``
        produced, so the mirror can never drift from the stored contract the way a
        hand-mapped pandas frame would.
        """
        target = self.parquet_path(day=day, timeframe=timeframe)
        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            where = "timestamp >= ? AND timestamp < ?"
            params: list[Any] = [start, end]
            if self.mirrored_by_timeframe:
                where = "timeframe = ? AND " + where
                params = [timeframe, *params]
            self._connection.execute(
                "CREATE OR REPLACE TEMP TABLE _anjaneeya_mirror AS "
                f"SELECT * FROM {quote_identifier(self.table)} "
                f"WHERE {where} "
                f"ORDER BY {self.order_by}",
                params,
            )
            self._connection.execute(
                "COPY _anjaneeya_mirror TO "
                f"'{_sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION {self.compression})"
            )
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - the database remains the truth
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error",
                f"Parquet mirror for {self.table} {timeframe} {day.isoformat()} failed "
                f"({type(exc).__name__}: {exc}); the DuckDB rows are unaffected",
            )
            return None
        self.stats.parquet_files += 1
        return target

    # -- internals ----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        """Log through the injected logger, tolerating a minimal double."""
        if self.logger is None:
            return
        method = getattr(self.logger, level, None) or self.logger.info
        method(message)


class CandleWriter(_DerivedWriter):
    """Write derived candles to DuckDB, the coverage ledger and Parquet.

    The connection is *borrowed*: see the module docstring for why it must be.
    """

    table = "candles"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stats = CandleWriteStats()

    def replace_candles(
        self,
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        tradingsymbol: str,
        rows: Iterable[Any],
    ) -> int:
        """Replace one ``(session, timeframe, contract)`` scope; returns rows written.

        An empty ``rows`` is meaningful, not a no-op: it is how a rebuild records
        that the scope now holds *no* candles (the ticks were re-read and there
        were none), clearing whatever a previous run left behind.
        """
        deleted = self._delete_scope(
            start=start,
            end=end,
            where="timeframe = ? AND tradingsymbol = ?",
            params=[timeframe, tradingsymbol],
        )
        written = insert_rows(
            self._connection, "candles", rows, mode="append", columns=CANDLE_COLUMNS
        )
        self.stats.deleted += deleted
        self.stats.candles += written
        return written

    def count_candles(
        self, *, start: datetime, end: datetime, timeframe: str, tradingsymbol: str
    ) -> int:
        """Rows currently stored for one scope (read back from DuckDB)."""
        row = self._connection.execute(
            f"SELECT count(*) FROM {quote_identifier('candles')} "
            "WHERE timeframe = ? AND tradingsymbol = ? AND timestamp >= ? AND timestamp < ?",
            [timeframe, tradingsymbol, start, end],
        ).fetchone()
        return int(row[0]) if row else 0


class FeatureWriter(_DerivedWriter):
    """Write unified feature rows to DuckDB, the coverage ledger and Parquet.

    Same contract as :class:`CandleWriter` (borrowed connection, delete-then-append),
    with one deliberate difference: the scope is the **contract-day**, because
    ``features`` is keyed ``(timestamp, tradingsymbol)`` and therefore cannot hold two
    timeframes of the same contract (see the module docstring).
    """

    table = "features"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stats = FeatureWriteStats()

    def replace_features(
        self,
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        tradingsymbol: str,
        rows: Iterable[Any],
    ) -> int:
        """Replace one contract-day's feature rows; returns the number written.

        The ``timeframe`` argument is recorded on the rows by the builder and is
        *not* part of the delete scope: rebuilding the same contract-day from a
        different candle series replaces the day rather than colliding with it.
        """
        deleted = self._delete_scope(
            start=start,
            end=end,
            where="tradingsymbol = ?",
            params=[tradingsymbol],
        )
        written = insert_rows(
            self._connection, "features", rows, mode="append", columns=FEATURE_COLUMNS
        )
        self.stats.deleted += deleted
        self.stats.rows += written
        return written

    def count_features(self, *, start: datetime, end: datetime, tradingsymbol: str) -> int:
        """Rows currently stored for one contract-day (read back from DuckDB)."""
        row = self._connection.execute(
            f"SELECT count(*) FROM {quote_identifier('features')} "
            "WHERE tradingsymbol = ? AND timestamp >= ? AND timestamp < ?",
            [tradingsymbol, start, end],
        ).fetchone()
        return int(row[0]) if row else 0

    def stored_timeframes(self, *, start: datetime, end: datetime) -> dict[str, int]:
        """Timeframe -> row count currently stored in the window (scope reporting)."""
        rows = self._connection.execute(
            f"SELECT timeframe, count(*) FROM {quote_identifier('features')} "
            "WHERE timestamp >= ? AND timestamp < ? GROUP BY 1 ORDER BY 1",
            [start, end],
        ).fetchall()
        return {str(timeframe): int(count) for timeframe, count in rows}


class LabelWriter(_DerivedWriter):
    """Write label rows to DuckDB, the coverage ledger and Parquet (spec 20).

    Same contract as :class:`FeatureWriter` - borrowed connection, delete-then-append -
    with the scope again being the **contract-day**: ``labels`` is keyed
    ``(timestamp, tradingsymbol, horizon_minutes)`` and a day's horizons are only
    meaningful together, so rebuilding one contract's day replaces *every* horizon it
    had. Two tables (``features`` and ``labels``) that never share a row is exactly what
    spec 20's physical separation asks for.

    The Parquet mirror of a table with no ``timeframe`` column lands in
    ``data/labels/<YYYY-MM-DD>.parquet`` - one file per day, whatever timeframe was
    labelled, because the table itself cannot hold two timeframes of the same contract
    and a per-timeframe directory would leave a stale file behind on a switch.
    """

    table = "labels"
    order_by = "tradingsymbol, timestamp, horizon_minutes"
    mirrored_by_timeframe = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stats = LabelWriteStats()

    def parquet_path(self, *, day: date, timeframe: str) -> Path:
        """Mirror file for one day: ``<YYYY-MM-DD>.parquet`` (no timeframe directory)."""
        return self.parquet_root / f"{day.isoformat()}.parquet"

    def replace_labels(
        self,
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        tradingsymbol: str,
        rows: Iterable[Any],
    ) -> int:
        """Replace one contract-day's label rows (all horizons); returns rows written.

        ``timeframe`` is not part of the delete scope for the same reason it is not in
        :meth:`FeatureWriter.replace_features`: it is recorded *on* the rows, and a
        rebuild from another candle series replaces the day rather than colliding.
        """
        deleted = self._delete_scope(
            start=start,
            end=end,
            where="tradingsymbol = ?",
            params=[tradingsymbol],
        )
        written = insert_rows(
            self._connection, "labels", rows, mode="append", columns=LABEL_COLUMNS
        )
        self.stats.deleted += deleted
        self.stats.rows += written
        labelled = sum(
            1
            for row in rows
            if (row.get("label") if isinstance(row, dict) else getattr(row, "label", None))
            is not None
        )
        self.stats.labelled += labelled
        return written

    def count_labels(self, *, start: datetime, end: datetime, tradingsymbol: str) -> int:
        """Rows currently stored for one contract-day (read back from DuckDB)."""
        row = self._connection.execute(
            f"SELECT count(*) FROM {quote_identifier('labels')} "
            "WHERE tradingsymbol = ? AND timestamp >= ? AND timestamp < ?",
            [tradingsymbol, start, end],
        ).fetchone()
        return int(row[0]) if row else 0


class GreeksWriter(_DerivedWriter):
    """Fill the IV/greeks columns of *existing* candle rows, then refresh their mirror.

    The third writer of the engine, and the only one that never adds or removes a row:
    see the module docstring for why an enrichment pass must not rebuild bars. It shares
    the Parquet mirror with :class:`CandleWriter` (``table = "candles"``), so the mirror
    cannot disagree with DuckDB about which greeks were stored.

    It writes no coverage row either: ``data_coverage`` describes *how much* was
    recorded, and enrichment changes no row count.
    """

    table = "candles"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stats = GreeksWriteStats()

    def apply_greeks(
        self,
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        tradingsymbol: str,
        frame: Any,
    ) -> int:
        """``UPDATE`` one contract's ``(day, timeframe)`` bars; returns rows updated.

        ``frame`` carries ``timestamp`` plus :data:`core.options.GREEKS_COLUMNS` (the
        enrichment's output). Rows whose key is not in the candle table are *not*
        inserted - they are counted in ``stats.unmatched`` and logged, because a bar
        that vanished mid-run is a fact the operator has to see.
        """
        columns = ("timestamp", *GREEKS_COLUMNS)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"the enrichment frame is missing column(s) {missing}; expected {list(columns)}"
            )
        # One vectorised pass turns every NaN/NaT into ``None``: DuckDB must receive
        # NULL for "no IV", and a stored NaN would make ``iv IS NULL`` a lie.
        staged = frame[list(columns)].astype(object)
        staged = staged.where(staged.notna(), None)
        records = [
            {"tradingsymbol": tradingsymbol, "timeframe": timeframe, **row}
            for row in staged.to_dict("records")
        ]
        if not records:
            return 0

        table = quote_identifier(_GREEKS_TEMP_TABLE)
        candles = quote_identifier("candles")
        assignments = ", ".join(f"{column} = staged.{column}" for column in GREEKS_COLUMNS)
        keys = (
            "candles.timestamp = staged.timestamp AND candles.timeframe = staged.timeframe "
            "AND candles.tradingsymbol = staged.tradingsymbol"
        )
        self._connection.execute(_GREEKS_TEMP_DDL)
        try:
            insert_rows(
                self._connection,
                _GREEKS_TEMP_TABLE,
                records,
                mode="append",
                columns=list(_GREEKS_UPDATE_COLUMNS),
            )
            matched = self._connection.execute(
                f"SELECT count(*) FROM {candles} JOIN {table} AS staged ON {keys}"
            ).fetchone()
            updated = int(matched[0]) if matched else 0
            self._connection.execute(
                f"UPDATE {candles} SET {assignments} FROM {table} AS staged WHERE {keys}"
            )
        finally:
            self._connection.execute(f"DROP TABLE IF EXISTS {table}")

        self.stats.bars += len(records)
        self.stats.updated += updated
        self.stats.unmatched += len(records) - updated
        self.stats.unsolved_iv += int(frame["iv"].isna().sum())
        if self.stats.unmatched:
            self._log(
                "warning",
                f"{tradingsymbol} {timeframe}: {len(records) - updated} enriched bar(s) no "
                "longer exist in the candles table (a rebuild ran during this pass?); "
                "nothing was inserted for them",
            )
        return updated


def _compression_keyword(parquet_compression: str) -> str:
    """Map a configured compression name to DuckDB's ``COPY`` keyword."""
    value = (parquet_compression or "").strip().lower()
    if value in _UNCOMPRESSED:
        return "UNCOMPRESSED"
    return value.upper()


def _sql_path(path: Path) -> str:
    """Render a path for a DuckDB string literal (forward slashes, escaped)."""
    return path.as_posix().replace("'", "''")


def _safe_unlink(path: Path) -> None:
    """Best-effort removal of a temporary file."""
    try:
        if path.exists():
            path.unlink()
    except OSError:  # pragma: no cover - filesystem dependent
        pass
