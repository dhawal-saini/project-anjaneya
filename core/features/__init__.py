"""Feature assembly (spec 18) and feature validation (spec 19).

Implemented (Stage 3.2e)

* :mod:`~core.features.feature_builder` - the unified row: one place that feeds the
  NIFTY, option, candle, level, structure and S/R engines and stitches their output
  into exactly ``core.constants.FEATURE_COLUMNS``. Pure: frames in, frames/row objects
  out, no I/O, no labels.
* :mod:`~core.features.feature_validation` - spec 19's checks (missing/infinite values,
  duplicate keys, OHLC sanity, contract identity, ordering, gaps and a no-look-ahead
  probe) as a report that never silently drops anything.

The lookback window, the close-instant index join and the "one row per
``(timestamp, contract)``" scope rule are documented in
:mod:`core.features.feature_builder` - they are the three decisions a later reader
would otherwise have to reverse-engineer.
"""

from __future__ import annotations

from core.features.feature_builder import (
    BOOLEAN_BLOCK_COLUMNS,
    CANDLE_FEATURE_COLUMNS,
    CATEGORICAL_BLOCK_COLUMNS,
    CONTEXT_FEATURE_COLUMNS,
    CONTRACT_BLOCK_COLUMNS,
    FEATURE_GROUP_ORDER,
    FEATURE_GROUPS,
    IDENTITY_FEATURE_COLUMNS,
    LEVEL_EVENT_FEATURE_COLUMNS,
    NIFTY_FEATURE_COLUMNS,
    OPTION_FEATURE_COLUMNS,
    RELATIVE_FEATURE_COLUMNS,
    REQUIRED_CANDLE_COLUMNS,
    FeatureBuildError,
    aggregate_index_frame,
    assert_feature_group_parity,
    build_feature_frame,
    close_moments,
    contract_feature_block,
    feature_rows,
    nifty_feature_block,
    required_lookback_bars,
    timeframe_of,
)
from core.features.feature_validation import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    STATUS_WARN,
    CheckResult,
    FeatureValidationReport,
    merge_reports,
    validate_feature_frame,
)

__all__ = [
    "BOOLEAN_BLOCK_COLUMNS",
    "CANDLE_FEATURE_COLUMNS",
    "CATEGORICAL_BLOCK_COLUMNS",
    "CONTRACT_BLOCK_COLUMNS",
    "CONTEXT_FEATURE_COLUMNS",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_ORDER",
    "IDENTITY_FEATURE_COLUMNS",
    "LEVEL_EVENT_FEATURE_COLUMNS",
    "NIFTY_FEATURE_COLUMNS",
    "OPTION_FEATURE_COLUMNS",
    "RELATIVE_FEATURE_COLUMNS",
    "REQUIRED_CANDLE_COLUMNS",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_SKIPPED",
    "STATUS_WARN",
    "CheckResult",
    "FeatureBuildError",
    "FeatureValidationReport",
    "aggregate_index_frame",
    "assert_feature_group_parity",
    "build_feature_frame",
    "close_moments",
    "contract_feature_block",
    "feature_rows",
    "merge_reports",
    "nifty_feature_block",
    "required_lookback_bars",
    "timeframe_of",
    "validate_feature_frame",
]
