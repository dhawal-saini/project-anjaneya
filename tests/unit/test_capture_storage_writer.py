"""Tests for :mod:`apps.anjaneeya_capture.storage_writer`."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.storage_writer import (
    SNAPSHOT_STREAM,
    TICK_STREAM,
    CaptureWriter,
    CoverageTracker,
    StreamStats,
    build_coverage_row,
)
from core.config_loader import load_config
from core.schemas import CoverageRow, QualityFlag, TickRow
from core.storage import bootstrap_database, connect
from tests.support.capture import RecordingLogger

IST = ZoneInfo("Asia/Kolkata")
TS = datetime(2026, 9, 29, 9, 16, 0, tzinfo=IST)
EXPIRY = date(2026, 10, 1)
RUN_ID = "anjaneeya_capture_20260929-091500_abcdef12"


def make_tick(**overrides) -> TickRow:
    moment = overrides.pop("timestamp", TS)
    payload = {
        "timestamp": moment,
        "instrument_token": 12345,
        "tradingsymbol": "NIFTY29SEP2625000CE",
        "expiry": EXPIRY,
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
        "fetch_timestamp": moment + timedelta(milliseconds=200),
        "quality_flag": "OK",
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return TickRow(**payload)


@pytest.fixture
def storage_config(isolated_paths):
    return load_config().data_sources.zerodha.capture.storage


@pytest.fixture
def writer(isolated_paths, storage_config, tmp_path):
    db_path = isolated_paths["db_path"]
    bootstrap_database(db_path, schema_version=1)
    instance = CaptureWriter(
        db_path=db_path,
        parquet_root=tmp_path / "parquet",
        storage_config=storage_config,
        run_id=RUN_ID,
        logger=RecordingLogger(),
    )
    yield instance
    instance.close()


def count(db_path: Path, table: str, *, writer: CaptureWriter | None = None) -> int:
    """Row count, read through the writer's own connection when one is open.

    DuckDB permits a single read-write process per file and refuses a second
    (even read-only) connection to the same file *within the same process*, so
    the query must go through the writer while it is alive.
    """
    connection = writer.connection if writer is not None else connect(db_path, read_only=True)
    try:
        return int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
    finally:
        if writer is None:
            connection.close()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def test_rows_are_buffered_until_a_flush_threshold_is_hit(
    isolated_paths, storage_config, tmp_path
) -> None:
    config = storage_config.model_copy(update={"flush_rows": 3, "flush_seconds": 3600})
    db_path = isolated_paths["db_path"]
    bootstrap_database(db_path, schema_version=1)
    writer = CaptureWriter(
        db_path=db_path,
        parquet_root=tmp_path / "parquet",
        storage_config=config,
        run_id=RUN_ID,
    )
    try:
        writer.write_ticks([make_tick()])
        assert writer.buffered_rows == 1
        assert count(db_path, "ticks", writer=writer) == 0

        writer.write_ticks([make_tick(strike=25050.0), make_tick(strike=25100.0)])
        # The third buffered row trips the batch size and flushes all three.
        assert writer.buffered_rows == 0
        assert count(db_path, "ticks", writer=writer) == 3
    finally:
        writer.close()


def test_time_based_flush_and_maybe_flush(isolated_paths, storage_config, tmp_path) -> None:
    config = storage_config.model_copy(update={"flush_rows": 10_000, "flush_seconds": 5.0})
    db_path = isolated_paths["db_path"]
    bootstrap_database(db_path, schema_version=1)
    ticks = [0.0]

    def clock() -> float:
        return ticks[0]

    writer = CaptureWriter(
        db_path=db_path,
        parquet_root=tmp_path / "parquet",
        storage_config=config,
        run_id=RUN_ID,
        clock=clock,
    )
    try:
        writer.write_ticks([make_tick()])
        assert writer.maybe_flush() is False
        ticks[0] = 6.0
        assert writer.maybe_flush() is True
        assert count(db_path, "ticks", writer=writer) == 1
        assert writer.maybe_flush() is False
    finally:
        writer.close()


def test_close_flushes_whatever_is_left(isolated_paths, storage_config, tmp_path) -> None:
    config = storage_config.model_copy(update={"flush_rows": 10_000, "flush_seconds": 3600})
    db_path = isolated_paths["db_path"]
    bootstrap_database(db_path, schema_version=1)
    writer = CaptureWriter(
        db_path=db_path,
        parquet_root=tmp_path / "parquet",
        storage_config=config,
        run_id=RUN_ID,
        logger=RecordingLogger(),
    )
    writer.write_ticks([make_tick()])
    stats = writer.close()
    assert stats.ticks == 1
    assert count(db_path, "ticks") == 1


def test_ticks_are_never_deduplicated(writer, isolated_paths) -> None:
    writer.write_ticks([make_tick(), make_tick()])
    writer.flush(force=True)
    assert count(isolated_paths["db_path"], "ticks", writer=writer) == 2

# ---------------------------------------------------------------------------
# Parquet mirror
# ---------------------------------------------------------------------------
def read_parquet_rows(paths: list[Path]) -> int:
    """Count the rows across a set of Parquet files using DuckDB."""
    import duckdb

    files = ", ".join(f"'{path.as_posix()}'" for path in paths)
    connection = duckdb.connect()
    try:
        row = connection.execute(f"SELECT count(*) FROM read_parquet([{files}])").fetchone()
        return int(row[0])
    finally:
        connection.close()


def test_flush_mirrors_the_batch_to_parquet(writer, tmp_path) -> None:
    writer.write_ticks([make_tick(), make_tick(strike=25050.0)])
    writer.flush(force=True)

    parts = writer.parquet_parts(TICK_STREAM)
    assert len(parts) == 1
    assert parts[0].parent.name == "2026-09-29"  # partitioned by IST day
    assert read_parquet_rows(parts) == 2


def test_compact_parquet_merges_parts_and_removes_them(writer, tmp_path) -> None:
    writer.write_ticks([make_tick()])
    writer.flush(force=True)
    writer.write_ticks([make_tick(strike=25050.0, timestamp=TS + timedelta(minutes=1))])
    writer.flush(force=True)
    assert len(writer.parquet_parts(TICK_STREAM)) == 2

    written = writer.compact_parquet()
    assert [path.name for path in written] == ["2026-09-29.parquet"]
    assert writer.parquet_parts(TICK_STREAM) == []
    assert read_parquet_rows(written) == 2


def test_batches_from_different_days_land_in_different_partitions(writer) -> None:
    writer.write_ticks([make_tick()])
    writer.flush(force=True)
    writer.write_ticks([make_tick(timestamp=TS + timedelta(days=1))])
    writer.flush(force=True)

    partitions = {part.parent.name for part in writer.parquet_parts(TICK_STREAM)}
    assert partitions == {"2026-09-29", "2026-09-30"}


def test_snapshots_are_mirrored_to_their_own_stream(writer) -> None:
    from core.schemas import OptionChainSnapshotRow

    snapshot = OptionChainSnapshotRow(
        timestamp=TS,
        instrument_token=12345,
        tradingsymbol="NIFTY29SEP2625000CE",
        expiry=EXPIRY,
        strike=25000.0,
        option_type="CE",
        last_price=123.0,
        oi=5000,
        run_id=RUN_ID,
        fetch_timestamp=TS,
    )
    writer.write_snapshots([snapshot])
    writer.flush(force=True)
    assert len(writer.parquet_parts(SNAPSHOT_STREAM)) == 1
    assert writer.parquet_parts(TICK_STREAM) == []
    assert writer.stats.snapshots == 1


def test_mirroring_can_be_disabled(isolated_paths, storage_config, tmp_path) -> None:
    config = storage_config.model_copy(update={"mirror_parquet": False})
    db_path = isolated_paths["db_path"]
    bootstrap_database(db_path, schema_version=1)
    writer = CaptureWriter(
        db_path=db_path,
        parquet_root=tmp_path / "parquet",
        storage_config=config,
        run_id=RUN_ID,
    )
    try:
        writer.write_ticks([make_tick()])
        writer.flush(force=True)
        assert writer.parquet_parts() == []
        assert count(db_path, "ticks", writer=writer) == 1
    finally:
        writer.close()


class FlakyConnection:
    """A real (in-memory) DuckDB with an injectable *insert* failure.

    ``execute`` always works - the schema and the Parquet mirror's temporary
    table are created for real - while ``executemany`` (the actual row insert)
    can be made to fail. That is exactly the failure mode being tested: "the
    database refused the rows, but the disk could still take the mirror".
    """

    def __init__(self, *, fail_on) -> None:
        import duckdb

        from core.storage import read_schema_sql

        self._real = duckdb.connect()
        self._real.execute(read_schema_sql())
        self._fail_on = fail_on
        self.insert_attempts: list[str] = []

    def execute(self, sql: str, *args, **kwargs):
        return self._real.execute(sql, *args, **kwargs)

    def executemany(self, sql: str, values):
        self.insert_attempts.append(sql)
        if self._fail_on(sql):
            raise RuntimeError("simulated insert failure")
        return self._real.executemany(sql, values)

    def close(self) -> None:
        self._real.close()


def sql_targets(sql: str, table: str) -> bool:
    """True when an INSERT statement targets ``table``."""
    return f'"{table}"' in sql


def test_a_database_failure_spills_to_parquet_instead_of_dropping_rows(
    isolated_paths, storage_config, tmp_path
) -> None:
    logger = RecordingLogger()
    connection = FlakyConnection(fail_on=lambda sql: sql_targets(sql, "ticks"))
    writer = CaptureWriter(
        db_path=isolated_paths["db_path"],
        parquet_root=tmp_path / "parquet",
        storage_config=storage_config,
        run_id=RUN_ID,
        logger=logger,
        connection_factory=lambda path: connection,
    )
    writer.write_ticks([make_tick()])
    stats = writer.flush(force=True)

    assert stats.ticks == 0
    assert stats.spilled_rows == 1
    assert stats.dropped_rows == 0  # the Parquet mirror still holds the row
    assert read_parquet_rows(writer.parquet_parts(TICK_STREAM)) == 1
    assert logger.contains("spilling", "ERROR")
    # The mirror writes into its own temporary table, not the raw stream.
    assert any(sql_targets(sql, "_anj_mirror") for sql in connection.insert_attempts)


def test_rows_are_reported_as_dropped_only_when_nothing_could_be_written(
    isolated_paths, storage_config, tmp_path
) -> None:
    logger = RecordingLogger()
    writer = CaptureWriter(
        db_path=isolated_paths["db_path"],
        parquet_root=tmp_path / "parquet",
        storage_config=storage_config,
        run_id=RUN_ID,
        logger=logger,
        connection_factory=lambda path: FlakyConnection(fail_on=lambda sql: True),
    )
    writer.write_ticks([make_tick()])
    stats = writer.flush(force=True)

    assert stats.spilled_rows == 1
    assert stats.dropped_rows == 1
    assert stats.failures == 2  # database + mirror
    assert writer.parquet_parts(TICK_STREAM) == []
    assert logger.contains("could not be persisted anywhere", "ERROR")


# ---------------------------------------------------------------------------
# Keyed writes (index history, coverage)
# ---------------------------------------------------------------------------
def make_index_row(**overrides):
    from core.schemas import IndexHistoryRow

    payload = {
        "timestamp": TS,
        "tradingsymbol": "NIFTY 50",
        "timeframe": "1d",
        "open": 25000.0,
        "high": 25010.0,
        "low": 24995.0,
        "close": 25005.0,
        "volume": 0,
        "oi": None,
        "source": "ZERODHA_REST",
        "fetch_timestamp": TS,
        "quality_flag": "OK",
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return IndexHistoryRow(**payload)


def test_index_history_writes_are_idempotent(writer, isolated_paths) -> None:
    writer.write_index_history([make_index_row()])
    writer.write_index_history([make_index_row(close=25007.0)])

    assert count(isolated_paths["db_path"], "index_history", writer=writer) == 1
    close = writer.connection.execute("SELECT close FROM index_history").fetchone()[0]
    assert close == pytest.approx(25007.0)
    assert writer.stats.index_history == 2


def test_coverage_rows_are_keyed_and_idempotent(writer, isolated_paths) -> None:
    row = CoverageRow(
        trading_date=date(2026, 9, 29),
        tradingsymbol="NIFTY29SEP2625000CE",
        timeframe="TICK",
        actual_rows=100,
        quality_flag=QualityFlag.OK,
        run_id=RUN_ID,
    )
    writer.write_coverage([row])
    writer.write_coverage([row.model_copy(update={"actual_rows": 120})])

    assert count(isolated_paths["db_path"], "data_coverage", writer=writer) == 1
    actual = writer.connection.execute("SELECT actual_rows FROM data_coverage").fetchone()[0]
    assert actual == 120


def test_empty_batches_do_not_open_a_connection(isolated_paths, storage_config, tmp_path) -> None:
    writer = CaptureWriter(
        db_path=isolated_paths["db_path"],
        parquet_root=tmp_path / "parquet",
        storage_config=storage_config,
        run_id=RUN_ID,
        connection_factory=lambda path: pytest.fail("the writer must stay lazy for empty batches"),
    )
    assert writer.write_ticks([]) == 0
    assert writer.write_index_history([]) == 0
    assert writer.write_coverage([]) == 0
    assert writer.flush(force=True).flushes == 0


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------
def test_coverage_tracker_counts_rows_and_flags() -> None:
    tracker = CoverageTracker(gap_seconds=30)
    for index in range(5):
        tracker.record(
            tradingsymbol="NIFTY CE",
            timeframe="TICK",
            timestamp=TS + timedelta(seconds=index),
            quality_flag=QualityFlag.OK,
        )
    stats = tracker.stream("NIFTY CE", "TICK")
    assert stats.rows == 5
    assert stats.first_timestamp == TS
    assert stats.last_timestamp == TS + timedelta(seconds=4)
    assert stats.gap_count == 0
    assert stats.quality_flag is QualityFlag.OK
    assert "5 rows" in stats.note()
    assert tracker.row_count == 5


def test_coverage_tracker_detects_gaps_and_remembers_the_largest() -> None:
    tracker = CoverageTracker(gap_seconds=30)
    tracker.record(tradingsymbol="NIFTY CE", timeframe="TICK", timestamp=TS)
    gap = tracker.record(
        tradingsymbol="NIFTY CE", timeframe="TICK", timestamp=TS + timedelta(seconds=45)
    )
    tracker.record(tradingsymbol="NIFTY CE", timeframe="TICK", timestamp=TS + timedelta(seconds=120))
    stats = tracker.stream("NIFTY CE", "TICK")
    assert gap is True
    assert stats.gap_count == 2
    assert stats.largest_gap_seconds == 75.0
    assert stats.quality_flag is QualityFlag.GAP_FILLED
    assert "largest 75.0s" in stats.note()


def test_gaps_are_only_counted_while_the_market_is_open() -> None:
    tracker = CoverageTracker(gap_seconds=1, is_open=lambda moment: False)
    tracker.record(tradingsymbol="X", timeframe="TICK", timestamp=TS)
    tracker.record(tradingsymbol="X", timeframe="TICK", timestamp=TS + timedelta(hours=2))
    assert tracker.stream("X", "TICK").gap_count == 0


def test_out_of_order_rows_never_corrupt_the_bookkeeping() -> None:
    tracker = CoverageTracker(gap_seconds=30)
    tracker.record(tradingsymbol="X", timeframe="TICK", timestamp=TS + timedelta(minutes=5))
    tracker.record(tradingsymbol="X", timeframe="TICK", timestamp=TS)
    stats = tracker.stream("X", "TICK")
    assert stats.rows == 2
    assert stats.first_timestamp == TS + timedelta(minutes=5)
    assert stats.gap_count == 0
    assert stats.largest_gap_seconds == 0.0


def test_coverage_verdicts_and_ratios() -> None:
    empty = StreamStats(tradingsymbol="X", timeframe="TICK")
    assert empty.quality_flag is QualityFlag.MISSING_TICK
    assert empty.coverage_ratio is None

    flagged = StreamStats(tradingsymbol="X", timeframe="1m", rows=10, expected_rows=20)
    flagged.record_flag(QualityFlag.OK)
    flagged.record_flag(QualityFlag.SUSPECT)
    assert flagged.coverage_ratio == pytest.approx(0.5)
    assert flagged.quality_flag is QualityFlag.SUSPECT

    out_of_session = StreamStats(tradingsymbol="X", timeframe="TICK", rows=3)
    out_of_session.record_flag(QualityFlag.OUT_OF_SESSION)
    assert out_of_session.quality_flag is QualityFlag.OUT_OF_SESSION
    assert "OUT_OF_SESSION=1" in out_of_session.note()


def test_coverage_ratio_is_capped_at_one() -> None:
    stats = StreamStats(tradingsymbol="X", timeframe="1m", rows=25, expected_rows=20)
    assert stats.coverage_ratio == 1.0


def test_build_coverage_row_maps_every_field() -> None:
    stats = StreamStats(
        tradingsymbol="NIFTY 50",
        timeframe="SNAPSHOT",
        rows=3,
        first_timestamp=TS,
        last_timestamp=TS + timedelta(minutes=2),
        gap_count=1,
        largest_gap_seconds=61.25,
        expected_rows=6,
    )
    row = build_coverage_row(trading_date=date(2026, 9, 29), stats=stats, run_id=RUN_ID)
    assert row.trading_date == date(2026, 9, 29)
    assert row.actual_rows == 3
    assert row.coverage_ratio == pytest.approx(0.5)
    assert row.largest_gap_seconds == pytest.approx(61.25)
    assert row.quality_flag is QualityFlag.GAP_FILLED
    assert row.run_id == RUN_ID


def test_tracker_builds_one_row_per_stream_and_day() -> None:
    tracker = CoverageTracker(gap_seconds=30)
    tracker.record(tradingsymbol="A", timeframe="TICK", timestamp=TS)
    tracker.record(tradingsymbol="B", timeframe="TICK", timestamp=TS)
    tracker.record(tradingsymbol="B", timeframe="SNAPSHOT", timestamp=TS)
    rows = tracker.coverage_rows(run_id=RUN_ID)
    assert [(row.tradingsymbol, row.timeframe) for row in rows] == [
        ("A", "TICK"),
        ("B", "SNAPSHOT"),
        ("B", "TICK"),
    ]
    assert all(row.trading_date == date(2026, 9, 29) for row in rows)


def test_tracker_coverage_rows_can_use_an_explicit_trading_date() -> None:
    tracker = CoverageTracker(gap_seconds=30)
    tracker.record(tradingsymbol="A", timeframe="TICK", timestamp=TS)
    rows = tracker.coverage_rows(run_id=RUN_ID, trading_date=date(2026, 9, 25))
    assert rows[0].trading_date == date(2026, 9, 25)

