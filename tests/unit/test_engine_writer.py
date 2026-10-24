"""Tests for :mod:`apps.anjaneeya_engine.storage_writer`.

The writer is where a rebuild can quietly corrupt a research dataset, so these
tests are about *replacement* semantics (delete-then-append, never an upsert that
leaves stale bars behind) and about the Parquet mirror never being able to take
the database down with it.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from apps.anjaneeya_engine.storage_writer import (
    CandleWriter,
    CandleWriteStats,
    FeatureWriter,
    FeatureWriteStats,
    GreeksWriter,
    GreeksWriteStats,
)
from core.schemas import CoverageRow, QualityFlag
from core.storage import bootstrap_database, connect
from tests.support.engine import (
    CE_SYMBOL,
    DAY,
    IST,
    PE_SYMBOL,
    RUN_ID,
    SESSION_END,
    SESSION_START,
    RecordingLogger,
    at,
    make_candle,
    make_feature,
)

TIMEFRAMES = ("1m", "3m", "5m", "15m")


@pytest.fixture
def logger() -> RecordingLogger:
    return RecordingLogger()


@pytest.fixture
def writer(isolated_paths, tmp_path, logger):
    """A writer over a bootstrapped database (the caller owns the connection)."""
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    instance = CandleWriter(
        connection=connection,
        parquet_root=tmp_path / "data" / "candles",
        run_id=RUN_ID,
        logger=logger,
    )
    yield instance
    connection.close()


def put(writer, *, timeframe="1m", symbol=CE_SYMBOL, minutes=(0,)):
    """Store ``minutes`` as bars for one contract, replacing that scope."""
    rows = [
        make_candle(timestamp=at(minute), timeframe=timeframe, tradingsymbol=symbol)
        for minute in minutes
    ]
    return writer.replace_candles(
        start=SESSION_START,
        end=SESSION_END,
        timeframe=timeframe,
        tradingsymbol=symbol,
        rows=rows,
    )


def stored(writer, *, timeframe="1m", symbol=CE_SYMBOL):
    return writer.connection.execute(
        "SELECT timestamp FROM candles WHERE timeframe = ? AND tradingsymbol = ? ORDER BY timestamp",
        [timeframe, symbol],
    ).fetchall()


# ---------------------------------------------------------------------------
# Replacement semantics
# ---------------------------------------------------------------------------
def test_replacing_a_scope_deletes_what_was_there_before(writer) -> None:
    put(writer, minutes=(0, 1))
    assert writer.count_candles(
        start=SESSION_START, end=SESSION_END, timeframe="1m", tradingsymbol=CE_SYMBOL
    ) == 2

    written = put(writer, minutes=(30,))

    assert written == 1
    assert stored(writer) == [(at(30),)]
    assert writer.stats.deleted == 2
    assert writer.stats.candles == 3


def test_replacing_with_no_rows_clears_the_scope(writer) -> None:
    put(writer, minutes=(0, 1, 2))

    written = writer.replace_candles(
        start=SESSION_START,
        end=SESSION_END,
        timeframe="1m",
        tradingsymbol=CE_SYMBOL,
        rows=[],
    )

    assert written == 0
    assert stored(writer) == []
    assert writer.stats.deleted == 3


def test_replacing_one_scope_leaves_every_other_scope_alone(writer) -> None:
    put(writer, timeframe="1m", symbol=CE_SYMBOL, minutes=(0,))
    put(writer, timeframe="5m", symbol=CE_SYMBOL, minutes=(0,))
    put(writer, timeframe="1m", symbol=PE_SYMBOL, minutes=(0,))
    put(writer, timeframe="1m", symbol=CE_SYMBOL, minutes=(30,))

    writer.replace_candles(
        start=SESSION_START,
        end=at(5),
        timeframe="1m",
        tradingsymbol=CE_SYMBOL,
        rows=[make_candle(timestamp=at(1), timeframe="1m")],
    )

    assert stored(writer, timeframe="1m", symbol=CE_SYMBOL) == [(at(1),), (at(30),)]
    assert stored(writer, timeframe="5m", symbol=CE_SYMBOL) == [(at(0),)]
    assert stored(writer, timeframe="1m", symbol=PE_SYMBOL) == [(at(0),)]


def test_count_candles_is_scoped_by_symbol_timeframe_and_window(writer) -> None:
    put(writer, timeframe="1m", minutes=(0, 1))
    put(writer, timeframe="5m", minutes=(0,))
    put(writer, timeframe="1m", symbol=PE_SYMBOL, minutes=(0,))

    assert writer.count_candles(
        start=SESSION_START, end=SESSION_END, timeframe="1m", tradingsymbol=CE_SYMBOL
    ) == 2
    assert writer.count_candles(
        start=SESSION_START, end=at(1), timeframe="1m", tradingsymbol=CE_SYMBOL
    ) == 1
    assert writer.count_candles(
        start=SESSION_START, end=SESSION_END, timeframe="5m", tradingsymbol=CE_SYMBOL
    ) == 1
    assert writer.count_candles(
        start=SESSION_START, end=SESSION_END, timeframe="15m", tradingsymbol=CE_SYMBOL
    ) == 0


# ---------------------------------------------------------------------------
# Coverage ledger
# ---------------------------------------------------------------------------
def coverage_row(*, timeframe="1m", actual_rows=20, symbol=CE_SYMBOL):
    return CoverageRow(
        trading_date=DAY,
        tradingsymbol=symbol,
        timeframe=timeframe,
        expected_rows=375,
        actual_rows=actual_rows,
        first_timestamp=SESSION_START,
        last_timestamp=at(19),
        gap_count=1,
        largest_gap_seconds=21300.0,
        coverage_ratio=min(1.0, actual_rows / 375),
        quality_flag=QualityFlag.GAP_FILLED,
        note="test",
        run_id=RUN_ID,
    )


def test_coverage_rows_are_upserted_by_primary_key(writer) -> None:
    writer.write_coverage([coverage_row()])
    writer.write_coverage([coverage_row(actual_rows=30)])

    rows = writer.connection.execute(
        "SELECT actual_rows FROM data_coverage WHERE timeframe = '1m'"
    ).fetchall()

    assert rows == [(30,)]
    assert writer.stats.coverage == 2


def test_writing_no_coverage_rows_is_a_no_op(writer) -> None:
    assert writer.write_coverage([]) == 0
    assert writer.stats.coverage == 0


# ---------------------------------------------------------------------------
# Parquet mirror
# ---------------------------------------------------------------------------
def test_the_mirror_target_is_one_file_per_timeframe_and_day(writer) -> None:
    assert writer.parquet_path(day=DAY, timeframe="5m") == (
        writer.parquet_root / "5m" / "2026-09-29.parquet"
    )
    assert writer.parquet_enabled is True


def test_mirroring_writes_the_days_candles_per_timeframe(writer) -> None:
    put(writer, timeframe="1m", minutes=(0, 1))
    put(writer, timeframe="5m", minutes=(0,))
    put(writer, timeframe="1m", symbol=PE_SYMBOL, minutes=(0,))
    # A bar of another day must never leak into the 29th's mirror file.
    writer.replace_candles(
        start=datetime(2026, 9, 30, 9, 15, tzinfo=IST),
        end=datetime(2026, 9, 30, 15, 30, tzinfo=IST),
        timeframe="1m",
        tradingsymbol=CE_SYMBOL,
        rows=[make_candle(timestamp=datetime(2026, 9, 30, 9, 15, tzinfo=IST))],
    )

    paths = writer.mirror_day(
        day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m", "5m"]
    )

    assert [path.name for path in paths] == ["2026-09-29.parquet", "2026-09-29.parquet"]
    assert [path.parent.name for path in paths] == ["1m", "5m"]
    assert writer.stats.parquet_files == 2
    assert writer.stats.failures == 0
    assert list(writer.parquet_root.rglob("*.tmp")) == []

    one_minute = writer.parquet_path(day=DAY, timeframe="1m")
    row_count, symbols = writer.connection.execute(
        f"SELECT count(*), count(DISTINCT tradingsymbol) FROM read_parquet('{one_minute.as_posix()}')"
    ).fetchone()
    assert (row_count, symbols) == (3, 2)

    schema = {
        name: dtype
        for name, dtype, *_ in writer.connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{one_minute.as_posix()}')"
        ).fetchall()
    }
    assert schema["timestamp"] == "TIMESTAMP WITH TIME ZONE"
    assert schema["timeframe"] == "VARCHAR"
    assert schema["volume"] == "BIGINT"


def test_mirroring_twice_overwrites_instead_of_appending(writer) -> None:
    put(writer, minutes=(0, 1, 2))
    writer.mirror_day(day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"])
    put(writer, minutes=(0,))  # a rebuild that keeps only one bar

    writer.mirror_day(day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"])

    target = writer.parquet_path(day=DAY, timeframe="1m")
    assert writer.connection.execute(
        f"SELECT count(*) FROM read_parquet('{target.as_posix()}')"
    ).fetchone() == (1,)
    assert list(writer.parquet_root.rglob("*.tmp")) == []


def test_the_mirror_can_be_switched_off(isolated_paths, tmp_path) -> None:
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    try:
        writer = CandleWriter(
            connection=connection,
            parquet_root=tmp_path / "data" / "candles",
            run_id=RUN_ID,
            mirror_parquet=False,
        )
        put(writer, minutes=(0,))

        assert writer.parquet_enabled is False
        assert writer.mirror_day(day=DAY, start=SESSION_START, end=SESSION_END, timeframes=TIMEFRAMES) == []
        assert not writer.parquet_root.exists()
        assert writer.stats.failures == 0
    finally:
        connection.close()


def test_a_mirror_failure_is_counted_logged_and_never_raised(isolated_paths, tmp_path, logger) -> None:
    """The database holds the truth: a broken mirror must not abort a build."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    try:
        writer = CandleWriter(
            connection=connection,
            parquet_root=blocked,
            run_id=RUN_ID,
            logger=logger,
        )
        put(writer, minutes=(0,))

        paths = writer.mirror_day(day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"])

        assert paths == []
        assert writer.stats.failures == 1
        assert writer.stats.parquet_files == 0
        assert any("Parquet mirror" in message for message in logger.messages("ERROR"))
        # The rows are still there - DuckDB was never at risk.
        assert writer.count_candles(
            start=SESSION_START, end=SESSION_END, timeframe="1m", tradingsymbol=CE_SYMBOL
        ) == 1
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("zstd", "ZSTD"), ("gzip", "GZIP"), ("none", "UNCOMPRESSED"), ("uncompressed", "UNCOMPRESSED")],
)
def test_the_configured_compression_maps_to_a_duckdb_keyword(
    isolated_paths, tmp_path, configured, expected
) -> None:
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    try:
        writer = CandleWriter(
            connection=connection,
            parquet_root=tmp_path,
            run_id=RUN_ID,
            parquet_compression=configured,
        )
        assert writer.compression == expected
    finally:
        connection.close()


def test_write_stats_describe_themselves() -> None:
    stats = CandleWriteStats(candles=3, deleted=2, coverage=1, parquet_files=4, failures=5)

    assert stats.describe() == {
        "candles": 3,
        "deleted": 2,
        "coverage": 1,
        "parquet_files": 4,
        "failures": 5,
    }



# ---------------------------------------------------------------------------
# FeatureWriter (Stage 3.2e)
#
# The feature table is keyed ``(timestamp, tradingsymbol)`` - no timeframe - so its
# rebuild scope is a whole *contract-day*. These tests pin that down, because the
# obvious alternative (a scope that includes the timeframe) would collide on the
# primary key the second time a day is built at another resolution.
# ---------------------------------------------------------------------------
@pytest.fixture
def feature_writer(isolated_paths, tmp_path, logger):
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    instance = FeatureWriter(
        connection=connection,
        parquet_root=tmp_path / "data" / "features",
        run_id=RUN_ID,
        logger=logger,
    )
    yield instance
    connection.close()


def put_features(writer, *, symbol=CE_SYMBOL, minutes=(0,), timeframe="1m"):
    rows = [
        make_feature(timestamp=at(minute), tradingsymbol=symbol, timeframe=timeframe)
        for minute in minutes
    ]
    return writer.replace_features(
        start=SESSION_START,
        end=SESSION_END,
        timeframe=timeframe,
        tradingsymbol=symbol,
        rows=rows,
    )


def stored_features(writer, *, symbol=CE_SYMBOL):
    return writer.connection.execute(
        "SELECT timestamp, timeframe FROM features WHERE tradingsymbol = ? ORDER BY timestamp",
        [symbol],
    ).fetchall()


def test_replacing_a_contract_day_deletes_what_was_there_before(feature_writer) -> None:
    put_features(feature_writer, minutes=(0, 1))
    assert feature_writer.count_features(
        start=SESSION_START, end=SESSION_END, tradingsymbol=CE_SYMBOL
    ) == 2

    written = put_features(feature_writer, minutes=(30,))

    assert written == 1
    assert [row[0] for row in stored_features(feature_writer)] == [at(30)]
    assert feature_writer.stats.deleted == 2
    assert feature_writer.stats.rows == 3


def test_the_scope_is_the_contract_day_not_the_timeframe(feature_writer) -> None:
    put_features(feature_writer, minutes=(0,), timeframe="1m")

    put_features(feature_writer, minutes=(0,), timeframe="5m")

    # One row per (timestamp, contract): the 5m build replaced the 1m row instead of
    # trying to store a second row for the same key.
    assert stored_features(feature_writer) == [(at(0), "5m")]


def test_replacing_one_contract_leaves_the_others_alone(feature_writer) -> None:
    put_features(feature_writer, symbol=CE_SYMBOL, minutes=(0,))
    put_features(feature_writer, symbol=PE_SYMBOL, minutes=(0,))

    put_features(feature_writer, symbol=CE_SYMBOL, minutes=(1,))

    assert [row[0] for row in stored_features(feature_writer, symbol=CE_SYMBOL)] == [at(1)]
    assert [row[0] for row in stored_features(feature_writer, symbol=PE_SYMBOL)] == [at(0)]


def test_replacing_features_with_no_rows_clears_the_feature_scope(feature_writer) -> None:
    put_features(feature_writer, minutes=(0, 1, 2))

    written = feature_writer.replace_features(
        start=SESSION_START,
        end=SESSION_END,
        timeframe="1m",
        tradingsymbol=CE_SYMBOL,
        rows=[],
    )

    assert written == 0
    assert stored_features(feature_writer) == []
    assert feature_writer.stats.deleted == 3


def test_stored_timeframes_reports_what_the_window_holds(feature_writer) -> None:
    put_features(feature_writer, symbol=CE_SYMBOL, minutes=(0, 1), timeframe="1m")
    put_features(feature_writer, symbol=PE_SYMBOL, minutes=(0,), timeframe="1m")

    assert feature_writer.stored_timeframes(start=SESSION_START, end=SESSION_END) == {"1m": 3}


def test_feature_coverage_is_upserted_and_counted(feature_writer) -> None:
    row = CoverageRow(
        trading_date=DAY,
        tradingsymbol="NIFTY",
        timeframe="1m",
        expected_rows=2,
        actual_rows=2,
        coverage_ratio=1.0,
        quality_flag=QualityFlag.OK,
        run_id=RUN_ID,
    )

    assert feature_writer.write_coverage([row]) == 1
    assert feature_writer.write_coverage([row]) == 1  # keyed: a re-run replaces it

    assert feature_writer.stats.coverage == 2
    assert feature_writer.connection.execute(
        "SELECT count(*) FROM data_coverage WHERE tradingsymbol = 'NIFTY'"
    ).fetchone() == (1,)


def test_the_feature_mirror_writes_one_file_per_day_and_timeframe(feature_writer) -> None:
    put_features(feature_writer, minutes=(0, 1))

    paths = feature_writer.mirror_day(
        day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"]
    )

    assert [path.name for path in paths] == ["2026-09-29.parquet"]
    assert paths[0].parent.name == "1m"
    assert list(feature_writer.parquet_root.rglob("*.tmp")) == []
    assert feature_writer.connection.execute(
        f"SELECT count(*) FROM read_parquet('{paths[0].as_posix()}')"
    ).fetchone() == (2,)


def test_the_feature_mirror_can_be_switched_off(isolated_paths, tmp_path) -> None:
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    try:
        writer = FeatureWriter(
            connection=connection,
            parquet_root=tmp_path / "data" / "features",
            run_id=RUN_ID,
            mirror_parquet=False,
        )
        put_features(writer, minutes=(0,))

        assert writer.parquet_enabled is False
        assert writer.mirror_day(
            day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"]
        ) == []
        assert not writer.parquet_root.exists()
        assert writer.stats.failures == 0
    finally:
        connection.close()


def test_a_feature_mirror_failure_is_counted_and_never_raised(
    isolated_paths, tmp_path, logger
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    try:
        writer = FeatureWriter(
            connection=connection,
            parquet_root=blocked,
            run_id=RUN_ID,
            logger=logger,
        )
        put_features(writer, minutes=(0,))

        assert writer.mirror_day(
            day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"]
        ) == []
        assert writer.stats.failures == 1
        assert any("features" in message for message in logger.messages("ERROR"))
        assert writer.count_features(
            start=SESSION_START, end=SESSION_END, tradingsymbol=CE_SYMBOL
        ) == 1
    finally:
        connection.close()


def test_feature_write_stats_describe_themselves() -> None:
    stats = FeatureWriteStats(rows=3, deleted=2, coverage=1, parquet_files=4, failures=5)

    assert stats.describe() == {
        "rows": 3,
        "deleted": 2,
        "coverage": 1,
        "parquet_files": 4,
        "failures": 5,
    }


# ---------------------------------------------------------------------------
# GreeksWriter (Stage 3.3): an UPDATE, never a rebuild
# ---------------------------------------------------------------------------
@pytest.fixture
def greeks(isolated_paths, tmp_path, logger):
    """A greeks writer over a bootstrapped database with two 1m bars already stored."""
    bootstrap_database(isolated_paths["db_path"], schema_version=1)
    connection = connect(isolated_paths["db_path"])
    candles = CandleWriter(
        connection=connection, parquet_root=tmp_path / "data" / "candles", run_id=RUN_ID
    )
    candles.replace_candles(
        start=SESSION_START,
        end=SESSION_END,
        timeframe="1m",
        tradingsymbol=CE_SYMBOL,
        rows=[make_candle(timestamp=at(0)), make_candle(timestamp=at(1))],
    )
    candles.replace_candles(
        start=SESSION_START,
        end=SESSION_END,
        timeframe="5m",
        tradingsymbol=CE_SYMBOL,
        rows=[make_candle(timestamp=at(0), timeframe="5m")],
    )
    instance = GreeksWriter(
        connection=connection,
        parquet_root=tmp_path / "data" / "candles",
        run_id=RUN_ID,
        logger=logger,
    )
    yield instance
    connection.close()


def greek_frame(*, minutes=(0, 1), iv=0.18, timeframe="1m", with_values=True):
    """An enrichment-shaped frame: the row key plus the seven greek columns."""
    records = []
    for minute in minutes:
        records.append(
            {
                "timestamp": at(minute),
                "timeframe": timeframe,
                "iv": iv if with_values else float("nan"),
                "delta": 0.5 if with_values else float("nan"),
                "gamma": 0.0003 if with_values else float("nan"),
                "theta": -12.0 if with_values else float("nan"),
                "vega": 8.5 if with_values else float("nan"),
                "days_to_expiry": 3255.0,
                "distance_from_atm": 0.0,
            }
        )
    return pd.DataFrame(records)


def stored_greeks(writer, *, timeframe="1m", minute=0):
    return writer.connection.execute(
        "SELECT iv, delta, gamma, theta, vega, days_to_expiry, distance_from_atm, open, run_id "
        "FROM candles WHERE timeframe = ? AND tradingsymbol = ? AND timestamp = ?",
        [timeframe, CE_SYMBOL, at(minute)],
    ).fetchone()


def apply(writer, *, timeframe="1m", frame=None, symbol=CE_SYMBOL):
    """Run one enrichment update for the standard scope."""
    return writer.apply_greeks(
        start=SESSION_START,
        end=SESSION_END,
        timeframe=timeframe,
        tradingsymbol=symbol,
        frame=greek_frame(timeframe=timeframe) if frame is None else frame,
    )


def test_apply_greeks_updates_only_the_greek_columns(greeks) -> None:
    """The bar's OHLCV and its run_id are the build's, and the update must not touch them."""
    apply(greeks)

    iv, delta, gamma, theta, vega, days, distance, open_, run_id = stored_greeks(greeks)
    assert (iv, delta, gamma, theta, vega) == (0.18, 0.5, 0.0003, -12.0, 8.5)
    assert days == pytest.approx(3255.0)
    assert distance == pytest.approx(0.0)
    assert open_ == pytest.approx(100.0)  # untouched
    assert run_id == RUN_ID  # still the build's run, not the enrichment's
    assert (greeks.stats.bars, greeks.stats.updated) == (2, 2)
    assert greeks.stats.unmatched == 0
    assert greeks.stats.unsolved_iv == 0


def test_apply_greeks_stores_null_for_an_unsolved_bar(greeks) -> None:
    """A bar with no IV must leave SQL ``NULL``, not a NaN that looks like a number."""
    apply(greeks, frame=greek_frame(with_values=False))

    iv, delta, *_ = stored_greeks(greeks)
    assert iv is None and delta is None
    assert greeks.connection.execute("SELECT count(*) FROM candles WHERE iv IS NULL").fetchone() == (
        3,
    )  # the two updated bars plus the untouched 5m bar
    assert greeks.stats.unsolved_iv == 2


def test_apply_greeks_ignores_a_bar_that_no_longer_exists(greeks) -> None:
    """A vanished bar is reported and counted, never inserted."""
    updated = apply(greeks, frame=greek_frame(minutes=(0, 7)))  # 09:22 was never built

    assert updated == 1
    assert greeks.stats.unmatched == 1
    assert greeks.connection.execute("SELECT count(*) FROM candles").fetchone() == (3,)


def test_apply_greeks_is_scoped_to_one_contract_and_timeframe(greeks) -> None:
    """Enriching 5m must not touch the 1m bars, and vice versa."""
    apply(greeks, timeframe="5m", frame=greek_frame(minutes=(0,), iv=0.25, timeframe="5m"))

    assert float(stored_greeks(greeks, timeframe="5m")[0]) == pytest.approx(0.25)
    assert stored_greeks(greeks, timeframe="1m")[0] is None


def test_apply_greeks_is_idempotent_and_clears_a_previous_result(greeks) -> None:
    """Re-running replaces the greeks exactly: no duplicates, no stale values."""
    apply(greeks)
    apply(greeks, frame=greek_frame(with_values=False))

    assert greeks.connection.execute("SELECT count(*) FROM candles").fetchone() == (3,)
    assert stored_greeks(greeks)[0] is None


def test_apply_greeks_leaves_no_temporary_table_behind(greeks) -> None:
    """The staging table is an implementation detail of one UPDATE."""
    apply(greeks)

    names = {
        row[0]
        for row in greeks.connection.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert "_anjaneeya_greeks_update" not in names


def test_apply_greeks_rejects_a_frame_that_is_not_the_enrichment_output(greeks) -> None:
    """A frame without the greek columns is a bug in the caller, not an empty update."""
    with pytest.raises(ValueError, match="missing column"):
        apply(greeks, frame=greek_frame().drop(columns=["vega"]))


def test_apply_greeks_refreshes_the_parquet_mirror(greeks) -> None:
    """The mirror must never keep serving pre-enrichment candles."""
    apply(greeks, frame=greek_frame(iv=0.21))
    files = greeks.mirror_day(day=DAY, start=SESSION_START, end=SESSION_END, timeframes=["1m"])

    assert len(files) == 1
    mirrored = pd.read_parquet(files[0])
    assert list(mirrored["iv"]) == pytest.approx([0.21, 0.21])
    assert greeks.stats.parquet_files == 1


def test_greeks_write_stats_describe_themselves() -> None:
    stats = GreeksWriteStats(bars=10, updated=9, unmatched=1, unsolved_iv=2, parquet_files=4)

    assert stats.describe() == {
        "bars": 10,
        "updated": 9,
        "unmatched": 1,
        "unsolved_iv": 2,
        "parquet_files": 4,
        "failures": 0,
    }
