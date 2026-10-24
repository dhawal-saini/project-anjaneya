"""Storage layer: DuckDB bootstrap, schema, readers, writers and coverage.

Four small modules, each with exactly one job:

* :mod:`core.storage.duckdb_bootstrap` owns the database file and the DDL,
* :mod:`core.storage.writers` is the one place that turns row objects into SQL,
* :mod:`core.storage.readers` is the one place that turns tables into frames,
* :mod:`core.storage.coverage` is the one place that decides whether a stream was
  captured completely.

The capture/engine apps build on top of all four.
"""

from __future__ import annotations

from core.storage.accumulation import (
    ACCUMULATION_TABLES,
    TIMESTAMP_COLUMNS,
    read_accumulation_inputs,
    read_coverage_days,
    read_table_profile,
    read_table_profiles,
    read_weekly_capture,
)
from core.storage.coverage import CoverageTracker, StreamStats, build_coverage_row
from core.storage.duckdb_bootstrap import (
    BootstrapResult,
    DatabaseLockedError,
    SchemaVersionError,
    bootstrap_database,
    connect,
    current_schema_version,
    read_schema_sql,
    table_names,
)
from core.storage.readers import (
    CANDLE_READ_COLUMNS,
    FEATURE_READ_COLUMNS,
    INDEX_HISTORY_READ_COLUMNS,
    LABEL_READ_COLUMNS,
    TICK_READ_COLUMNS,
    candle_counts,
    feature_counts,
    first_candle_timestamp,
    quality_counts,
    read_candles,
    read_features,
    read_index_history,
    read_instruments,
    read_labels,
    read_option_instruments,
    read_ticks,
    symbols_with_candles,
    symbols_with_ticks,
    trading_days_with_candles,
    trading_days_with_ticks,
)
from core.storage.writers import count_rows, insert_rows, quote_identifier, row_columns

__all__ = [
    "ACCUMULATION_TABLES",
    "BootstrapResult",
    "CANDLE_READ_COLUMNS",
    "CoverageTracker",
    "DatabaseLockedError",
    "FEATURE_READ_COLUMNS",
    "INDEX_HISTORY_READ_COLUMNS",
    "LABEL_READ_COLUMNS",
    "SchemaVersionError",
    "StreamStats",
    "TICK_READ_COLUMNS",
    "TIMESTAMP_COLUMNS",
    "bootstrap_database",
    "build_coverage_row",
    "candle_counts",
    "connect",
    "count_rows",
    "current_schema_version",
    "feature_counts",
    "first_candle_timestamp",
    "insert_rows",
    "quality_counts",
    "quote_identifier",
    "read_accumulation_inputs",
    "read_candles",
    "read_coverage_days",
    "read_features",
    "read_index_history",
    "read_instruments",
    "read_labels",
    "read_option_instruments",
    "read_schema_sql",
    "read_table_profile",
    "read_table_profiles",
    "read_ticks",
    "read_weekly_capture",
    "row_columns",
    "symbols_with_candles",
    "symbols_with_ticks",
    "table_names",
    "trading_days_with_candles",
    "trading_days_with_ticks",
]
