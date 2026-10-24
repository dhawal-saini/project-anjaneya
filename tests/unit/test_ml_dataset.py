"""Unit tests for the training matrix (``core/ml/dataset.py``, spec 21).

The dataset builder is where a feature row and a forward label meet, so these tests are
mostly about *refusals*: an inexact join, a shifted label, a repeated key, a label column
offered as a feature, two label versions in one matrix. Each one would make a model look
better than it is, and none of them would announce itself.

The rest pin the honest bookkeeping: how many rows entered the join, how many carried a
class, how many did not, and which columns were constant or missing - all of which a
research report has to be able to quote.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pandas as pd
import pytest

from core.constants import FEATURE_COLUMNS
from core.ml import (
    AUXILIARY_TARGET_COLUMNS,
    CODE_COLUMN,
    TARGET_COLUMN,
    Dataset,
    DatasetError,
    DatasetPolicy,
    assert_no_label_leakage,
    build_dataset,
    data_gate_status,
)
from tests.support.ai import (
    BARS,
    CLASSES,
    DAYS,
    HORIZONS,
    at,
    feature_frame,
    label_frame,
)

#: Feature rows in the fixture: 4 days x 2 contracts x 60 bars.
FEATURE_ROWS = len(DAYS) * 2 * BARS
#: Labelled rows per horizon: minutes 0..30 have a 30-bar window inside the session.
LABELLED_PER_HORIZON = len(DAYS) * 2 * (BARS - max(HORIZONS) + 1)


@pytest.fixture
def policy(config) -> DatasetPolicy:
    """The policy the shipped configuration produces."""
    return DatasetPolicy.from_config(config)


def dataset(config, *, horizon: int = 5, **policy_overrides) -> Dataset:
    """Build a dataset from the synthetic multi-day fixture."""
    base = DatasetPolicy.from_config(config)
    policy_ = replace(base, **policy_overrides) if policy_overrides else base
    return build_dataset(
        feature_frame(), label_frame(), policy=policy_, horizon=horizon
    )


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_policy_comes_from_the_shipped_config(policy: DatasetPolicy) -> None:
    assert policy.horizons == HORIZONS
    assert policy.classes == CLASSES
    assert policy.timeframe == "1m"
    assert policy.label_version == "v1"
    assert "nifty_regime" in policy.categorical_columns
    assert policy.timezone_name == "Asia/Kolkata"


def test_policy_timeframe_can_be_overridden(config) -> None:
    assert DatasetPolicy.from_config(config, timeframe="5m").timeframe == "5m"


def test_policy_describe_lists_every_choice(policy: DatasetPolicy) -> None:
    described = policy.describe()
    assert set(described) == {
        "timeframe",
        "horizons_minutes",
        "classes",
        "categorical_columns",
        "label_version",
        "timezone",
    }
    assert described["horizons_minutes"] == list(HORIZONS)


def test_every_configured_categorical_column_is_a_real_feature(config) -> None:
    for name in config.model.dataset.categorical_columns:
        assert name in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# The label-leakage guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "column",
    [
        "label",
        "label_code",
        "label_version",
        "horizon_minutes",
        *AUXILIARY_TARGET_COLUMNS,
        "future_something_new",
    ],
)
def test_label_derived_columns_are_refused_as_features(column: str) -> None:
    with pytest.raises(DatasetError, match="label-derived"):
        assert_no_label_leakage(["option_atr", column])


def test_ordinary_feature_columns_pass_the_guard() -> None:
    assert_no_label_leakage([name for name in FEATURE_COLUMNS if not name.startswith("future_")])


def test_the_feature_matrix_never_contains_a_label_column(config) -> None:
    matrix = dataset(config).feature_columns
    assert TARGET_COLUMN not in matrix
    assert CODE_COLUMN not in matrix
    assert not [name for name in matrix if name.startswith("future_")]


# ---------------------------------------------------------------------------
# The join: exact, counted, class-encoded
# ---------------------------------------------------------------------------
def test_rows_are_only_the_labelled_ones(config) -> None:
    built = dataset(config)

    assert built.rows == LABELLED_PER_HORIZON
    assert built.counts.feature_rows == FEATURE_ROWS
    assert built.counts.label_rows == FEATURE_ROWS
    assert built.counts.joined_rows == FEATURE_ROWS
    assert built.counts.unlabelled_rows == FEATURE_ROWS - LABELLED_PER_HORIZON
    assert built.counts.features_without_label == 0
    assert built.counts.labels_without_feature == 0


def test_counts_explain_every_dropped_row(config) -> None:
    built = dataset(config)
    counts = built.counts
    # Everything that entered is either in the matrix, unlabelled, unmatched, or filtered
    # out by timeframe/contract - and every one of those is its own published number.
    assert (
        counts.unlabelled_rows
        + counts.labelled_rows
        + counts.features_without_label
        + counts.other_timeframe_rows
        + counts.other_symbol_rows
        == counts.feature_rows
    )


def test_the_join_is_exact_and_never_borrows_a_neighbour(config) -> None:
    """A label one minute away belongs to a different moment: it must not match."""
    day = DAYS[0]
    features = feature_frame(days=[day])
    features = features.loc[features["timestamp"] < at(day, 2)]  # minutes 0 and 1
    labels = label_frame(days=[day])
    labels = labels.loc[(labels["horizon_minutes"] == 5) & (labels["timestamp"] < at(day, 2))]
    labels["timestamp"] = labels["timestamp"] + timedelta(minutes=2)  # minutes 2 and 3

    built = build_dataset(
        features, labels, policy=DatasetPolicy.from_config(config), horizon=5
    )

    assert built.rows == 0
    assert built.counts.joined_rows == 0
    assert built.counts.features_without_label == len(features) == 4


def test_a_feature_without_a_label_row_is_counted(config) -> None:
    labels = label_frame()
    first = labels.index[labels["horizon_minutes"] == 5][0]
    labels = labels.drop(index=first)

    built = build_dataset(feature_frame(), labels, policy=DatasetPolicy.from_config(config), horizon=5)

    assert built.counts.features_without_label == 1
    assert built.rows == LABELLED_PER_HORIZON - 1


def test_other_timeframes_are_excluded_and_counted(config) -> None:
    features = pd.concat([feature_frame(), feature_frame(timeframe="5m")], ignore_index=True)

    built = build_dataset(
        features, label_frame(), policy=DatasetPolicy.from_config(config), horizon=5
    )

    assert built.counts.other_timeframe_rows == FEATURE_ROWS
    assert built.rows == LABELLED_PER_HORIZON


def test_class_codes_follow_the_configured_order(config) -> None:
    built = dataset(config)

    codes = built.target_codes()
    mapping = dict(zip(list(built.classes), range(len(built.classes)), strict=True))
    assert [mapping[str(value)] for value in built.frame[TARGET_COLUMN]] == list(codes)
    assert codes.dtype == "int64"


def test_class_counts_cover_every_configured_class(config) -> None:
    counts = dataset(config).class_counts()
    assert set(counts) == set(CLASSES)
    assert sum(counts.values()) == LABELLED_PER_HORIZON
    # The fixture cycles through the four classes minute by minute.
    assert counts["UP_CONTINUATION"] == counts["DOWN_CONTINUATION"] == counts["RANGE"]
    assert counts["REVERSAL"] == counts["RANGE"] - len(DAYS) * 2


def test_an_unknown_stored_class_is_refused(config) -> None:
    labels = label_frame()
    labels.loc[labels[TARGET_COLUMN].notna(), TARGET_COLUMN] = "SIDEWAYS"

    with pytest.raises(DatasetError, match="SIDEWAYS"):
        build_dataset(feature_frame(), labels, policy=DatasetPolicy.from_config(config), horizon=5)


def test_two_label_versions_in_one_matrix_are_refused(config) -> None:
    labels = label_frame()
    labels.loc[labels.index[0], "label_version"] = "v2"

    with pytest.raises(DatasetError, match="label_version"):
        build_dataset(feature_frame(), labels, policy=DatasetPolicy.from_config(config), horizon=5)


# ---------------------------------------------------------------------------
# Refusals: a bad join must be loud, not approximate
# ---------------------------------------------------------------------------
def test_a_repeated_feature_key_is_refused(config) -> None:
    features = pd.concat([feature_frame(), feature_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(DatasetError, match="repeats the key"):
        build_dataset(features, label_frame(), policy=DatasetPolicy.from_config(config), horizon=5)


def test_a_repeated_label_key_is_refused_and_explains_timeframe_collisions(config) -> None:
    labels = pd.concat([label_frame(), label_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(DatasetError, match="timeframe"):
        build_dataset(feature_frame(), labels, policy=DatasetPolicy.from_config(config), horizon=5)


def test_a_feature_column_absent_from_the_frame_is_reported_not_invented(config) -> None:
    """A column the frame does not carry is reported and left out - never filled."""
    features = feature_frame().drop(columns=["option_atr"])

    built = build_dataset(features, label_frame(), policy=DatasetPolicy.from_config(config), horizon=5)

    assert "option_atr" in built.missing_feature_columns
    assert "option_atr" not in built.feature_columns
    assert built.rows == LABELLED_PER_HORIZON


def test_a_missing_key_column_is_named(config) -> None:
    labels = label_frame().drop(columns=["tradingsymbol"])

    with pytest.raises(DatasetError, match="tradingsymbol"):
        build_dataset(feature_frame(), labels, policy=DatasetPolicy.from_config(config), horizon=5)


@pytest.mark.parametrize("side", ["features", "labels"])
def test_naive_timestamps_are_refused(config, side: str) -> None:
    features, labels = feature_frame(), label_frame()
    target = features if side == "features" else labels
    target["timestamp"] = target["timestamp"].dt.tz_localize(None)

    with pytest.raises(DatasetError, match="timezone-aware"):
        build_dataset(features, labels, policy=DatasetPolicy.from_config(config), horizon=5)


def test_a_horizon_nobody_configured_is_refused(config) -> None:
    with pytest.raises(DatasetError, match="not one of the configured horizons"):
        build_dataset(feature_frame(), label_frame(), policy=DatasetPolicy.from_config(config), horizon=7)


def test_a_categorical_column_that_is_not_in_the_frame_is_refused(config) -> None:
    features = feature_frame().drop(columns=["nifty_regime"])

    with pytest.raises(DatasetError, match="nifty_regime"):
        build_dataset(features, label_frame(), policy=DatasetPolicy.from_config(config), horizon=5)


# ---------------------------------------------------------------------------
# The matrix: what a trainer receives
# ---------------------------------------------------------------------------
def test_feature_columns_are_the_contract_minus_meta_minus_labels(config) -> None:
    built = dataset(config)

    assert set(built.feature_columns) <= set(FEATURE_COLUMNS)
    for name in ("timestamp", "tradingsymbol", "expiry", "strike", "timeframe", "run_id"):
        assert name not in built.feature_columns
    # ``option_type`` stays a feature: CE vs PE is part of what is being predicted.
    assert "option_type" in built.feature_columns
    # The order is the stored contract's, so two runs produce identical columns.
    ordered = [name for name in FEATURE_COLUMNS if name in set(built.feature_columns)]
    assert list(built.feature_columns) == ordered


def test_categoricals_ship_as_category_dtype(config) -> None:
    built = dataset(config)

    for name in built.categorical_columns:
        assert isinstance(built.frame[name].dtype, pd.CategoricalDtype)
    assert not set(built.numeric_columns) & set(built.categorical_columns)
    assert set(built.numeric_columns) | set(built.categorical_columns) == set(built.feature_columns)


def test_the_matrix_is_features_only(config) -> None:
    built = dataset(config)

    matrix = built.matrix()
    assert list(matrix.columns) == list(built.feature_columns)
    assert len(matrix) == built.rows
    # The frame keeps the identity and target columns a split and a report need.
    for name in (TARGET_COLUMN, CODE_COLUMN, "timestamp", "tradingsymbol", *AUXILIARY_TARGET_COLUMNS):
        assert name in built.frame.columns


def test_rows_are_ordered_by_timestamp_then_contract(config) -> None:
    built = dataset(config)

    ordered = built.frame[["timestamp", "tradingsymbol"]].apply(tuple, axis=1).tolist()
    assert ordered == sorted(ordered)


def test_constant_and_missing_columns_are_reported_not_removed(config) -> None:
    features = feature_frame()
    features["option_momentum"] = 0.0  # constant across the whole scope

    built = build_dataset(features, label_frame(), policy=DatasetPolicy.from_config(config), horizon=5)

    assert "option_momentum" in built.feature_columns
    assert "option_momentum" in built.constant_columns()
    assert built.null_shares()["option_momentum"] == 0.0
    assert built.missing_feature_columns == ()


def test_a_present_but_empty_column_is_reported_as_missing_values(config) -> None:
    features = feature_frame()
    features["iv"] = None

    built = build_dataset(features, label_frame(), policy=DatasetPolicy.from_config(config), horizon=5)

    assert built.null_shares()["iv"] == 1.0
    # Entirely missing is constant here: it carries no information either way.
    assert "iv" in built.constant_columns()
    assert "iv" in built.feature_columns


def test_describe_reports_the_scope_of_the_matrix(config) -> None:
    built = dataset(config)
    described = built.describe()

    assert described["horizon_minutes"] == 5
    assert described["rows"] == LABELLED_PER_HORIZON
    assert described["days"] == len(DAYS)
    assert described["symbols"] == 2
    assert described["label_version"] == "v1"
    assert described["first_timestamp"].startswith(DAYS[0].isoformat())
    assert described["counts"]["labelled_rows"] == LABELLED_PER_HORIZON


def test_the_symbols_filter_narrows_the_matrix(config) -> None:
    from tests.support.engine import CE_SYMBOL

    built = build_dataset(
        feature_frame(),
        label_frame(),
        policy=DatasetPolicy.from_config(config),
        horizon=5,
        symbols=[CE_SYMBOL],
    )

    assert built.symbols == (CE_SYMBOL,)
    assert built.rows == LABELLED_PER_HORIZON // 2
    assert built.counts.other_symbol_rows == FEATURE_ROWS // 2
    assert built.counts.features_without_label == 0


# ---------------------------------------------------------------------------
# Empty is a legitimate answer
# ---------------------------------------------------------------------------
def test_no_feature_rows_gives_an_empty_dataset(config) -> None:
    built = build_dataset(
        feature_frame(days=[]), label_frame(), policy=DatasetPolicy.from_config(config), horizon=5
    )

    assert built.rows == 0
    assert built.days == ()
    assert built.feature_columns == ()
    assert built.class_counts() == dict.fromkeys(CLASSES, 0)
    assert built.describe()["first_timestamp"] is None


def test_a_horizon_that_was_never_labeled_gives_an_empty_dataset(config) -> None:
    built = build_dataset(
        feature_frame(), label_frame(horizons=(5,)), policy=DatasetPolicy.from_config(config), horizon=30
    )

    assert built.rows == 0
    assert built.counts.label_rows == 0


def test_rows_whose_label_is_null_leave_an_empty_dataset_with_counts(config) -> None:
    labels = label_frame()
    labels[TARGET_COLUMN] = None

    built = build_dataset(feature_frame(), labels, policy=DatasetPolicy.from_config(config), horizon=5)

    assert built.rows == 0
    assert built.counts.unlabelled_rows == FEATURE_ROWS
    assert built.counts.joined_rows == FEATURE_ROWS


# ---------------------------------------------------------------------------
# The spec 4.3 gates
# ---------------------------------------------------------------------------
GATES = {"exploratory": 20, "ml_training": 60, "meaningful_backtest": 120, "seasonality": 250}


@pytest.mark.parametrize(
    ("days", "stage", "next_stage", "days_to_next", "ready"),
    [
        (0, "INSUFFICIENT", "EXPLORATORY", 20, False),
        (19, "INSUFFICIENT", "EXPLORATORY", 1, False),
        (20, "EXPLORATORY", "ML_TRAINING", 40, False),
        (59, "EXPLORATORY", "ML_TRAINING", 1, False),
        (60, "ML_TRAINING", "MEANINGFUL_BACKTEST", 60, True),
        (119, "ML_TRAINING", "MEANINGFUL_BACKTEST", 1, True),
        (120, "MEANINGFUL_BACKTEST", "SEASONALITY", 130, True),
        (249, "MEANINGFUL_BACKTEST", "SEASONALITY", 1, True),
        (250, "SEASONALITY", None, None, True),
        (500, "SEASONALITY", None, None, True),
    ],
)
def test_gate_status_ladder(days, stage, next_stage, days_to_next, ready) -> None:
    status = data_gate_status(days, **GATES)

    assert status.stage == stage
    assert status.next_stage == next_stage
    assert status.days_to_next == days_to_next
    assert status.ready_for_training is ready
    assert status.days == days


def test_gate_status_reports_the_thresholds_it_used() -> None:
    status = data_gate_status(3, **GATES)

    assert status.gates == {
        "exploratory_min_days": 20,
        "ml_training_min_days": 60,
        "meaningful_backtest_min_days": 120,
        "seasonality_min_days": 250,
    }
    assert "too little data" in status.describe()
    assert status.to_dict()["stage"] == "INSUFFICIENT"


def test_gate_status_matches_the_shipped_gates(config) -> None:
    gates = config.model.data_gates
    status = data_gate_status(
        config.model.data_gates.ml_training_min_days,
        exploratory=gates.exploratory_min_days,
        ml_training=gates.ml_training_min_days,
        meaningful_backtest=gates.meaningful_backtest_min_days,
        seasonality=gates.seasonality_min_days,
    )

    assert status.stage == "ML_TRAINING"
    assert status.ready_for_training




