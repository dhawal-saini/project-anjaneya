"""The training matrix: stored feature rows joined to their forward labels (spec 21).

Spec 20 split ANJANEYA in two: a *feature* describes what the market was doing up to an
instant, a *label* describes what it did after it. One is causal, the other is not, and
the only place they meet is here - in a research artefact, never in a live row.

Four rules shape this module, and each is enforced rather than documented:

1. **The join is exact.** A feature at ``t`` meets the label at ``t`` and nothing else.
   No ``merge_asof``, no tolerance: a label 30 seconds away describes a different
   decision moment, and quietly borrowing it would make every number in a report
   slightly wrong in a way no test would notice.
2. **Labels are physically absent from the feature matrix.** ``future_*``, ``label``,
   ``label_code`` and ``label_version`` are refused as features by
   :func:`assert_no_label_leakage`, so no config knob, no rename and no extra stored
   column can put a forward number into ``X``.
3. **A row without a class is not a training row.** It stays in the *count* (a label that
   could not be computed is a fact about the data, spec 20) but it is dropped from the
   matrix - and the number dropped is published beside the number kept.
4. **Nothing is imputed and nothing is dropped for being awkward.** Missing values are
   left missing (a tree model has its own opinion about them), and constant or mostly-NULL
   columns are *reported* rather than removed: feature selection belongs to the training
   folds (spec 22), not to the dataset builder.

The splitter that consumes this lives in :mod:`core.ml.walk_forward`; spec 4.3's
data-accumulation gates live in :mod:`core.gates` and are re-exported here
(:func:`data_gate_status`) so every caller keeps importing them from one place.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from core.constants import DEFAULT_TIMEZONE, FEATURE_COLUMNS, get_timezone
from core.gates import GATE_STAGES, GateStatus, data_gate_status

__all__ = [
    "AUXILIARY_TARGET_COLUMNS",
    "CODE_COLUMN",
    "DATASET_KEYS",
    "GATE_STAGES",
    "HORIZON_COLUMN",
    "LABEL_DERIVED_COLUMNS",
    "META_COLUMNS",
    "TARGET_COLUMN",
    "Dataset",
    "DatasetCounts",
    "DatasetError",
    "DatasetPolicy",
    "GateStatus",
    "assert_no_label_leakage",
    "build_dataset",
    "data_gate_status",
]

#: The join key. Both tables are keyed by ``(timestamp, tradingsymbol)``; the label table
#: adds the horizon, which this module pins to one value per dataset.
DATASET_KEYS = ("timestamp", "tradingsymbol")

#: The label table's primary-key component that is *not* part of the join.
HORIZON_COLUMN = "horizon_minutes"

#: The class column, and the integer encoding of it that a model consumes.
TARGET_COLUMN = "label"
CODE_COLUMN = "label_code"

#: Spec 20's numeric forward measurements. They are companions of the class (regression
#: targets, and the way a class is explained after the fact) - never features.
AUXILIARY_TARGET_COLUMNS = (
    "future_return",
    "future_max_up_move",
    "future_max_down_move",
    "future_trend_change",
)

#: Every label-side name. Refused as a feature, by name and by the ``future_`` prefix.
LABEL_DERIVED_COLUMNS = (
    *AUXILIARY_TARGET_COLUMNS,
    TARGET_COLUMN,
    CODE_COLUMN,
    "label_version",
    HORIZON_COLUMN,
)

#: Columns that say *which* row this is (contract, expiry, provenance) rather than what
#: the market was doing. They travel with the dataset - splitting, reporting and auditing
#: need them - but they are never handed to a model.
#:
#: ``option_type`` is deliberately *not* here: whether a contract is a CE or a PE is a
#: categorical property of the thing being predicted (RQ1/RQ2 ask exactly that question),
#: so it stays a feature and ships as a categorical column.
META_COLUMNS = (
    "timestamp",
    "underlying",
    "tradingsymbol",
    "instrument_token",
    "expiry",
    "strike",
    "timeframe",
    "source",
    "fetch_timestamp",
    "quality_flag",
    "run_id",
)


#: The label-side columns a dataset keeps. The label frame's own identity columns
#: (expiry, strike, option_type, instrument_token, provenance) are deliberately *not*
#: carried across: the feature row already holds them, the join is keyed on
#: ``(timestamp, tradingsymbol)`` alone, and duplicating them would only produce
#: ``expiry_label``-style noise in the artefact.
LABEL_JOIN_COLUMNS = (
    HORIZON_COLUMN,
    TARGET_COLUMN,
    "label_version",
    *AUXILIARY_TARGET_COLUMNS,
)


class DatasetError(ValueError):
    """Raised when the stored rows cannot form a dataset without guessing."""


def _is_label_derived(name: str) -> bool:
    """True for any column that could carry a forward-looking number."""
    return name in LABEL_DERIVED_COLUMNS or name.startswith("future_")


def assert_no_label_leakage(columns: Iterable[str]) -> None:
    """Raise unless no column in ``columns`` can carry a label (spec 20/30).

    The check is by name *and* by the ``future_`` prefix, so a label column added to a
    future schema is refused the moment somebody tries to make it a feature - the failure
    mode this guards against (a leak that makes every metric look excellent) is silent
    otherwise.
    """
    offenders = sorted({str(name) for name in columns if _is_label_derived(str(name))})
    if offenders:
        raise DatasetError(
            f"Refusing to treat label-derived column(s) {offenders} as features: labels "
            "must never appear in the feature matrix (spec 20/30)."
        )


# ---------------------------------------------------------------------------
# Policy: what a dataset is built from
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DatasetPolicy:
    """The modelling choices a dataset is built under (``config/model.yaml``)."""

    horizons: tuple[int, ...]
    classes: tuple[str, ...]
    timeframe: str
    categorical_columns: tuple[str, ...]
    label_version: str
    timezone_name: str = DEFAULT_TIMEZONE

    @classmethod
    def from_config(cls, config: Any, *, timeframe: str | None = None) -> DatasetPolicy:
        """Build the policy from a loaded :class:`core.config_loader.AnjaneyaConfig`.

        ``timeframe`` overrides ``model.dataset.timeframe`` (``--timeframe`` on the CLI).
        A timeframe nobody enabled simply selects no rows, which the run reports as an
        empty dataset - it is not an error to *ask* for one, only to pretend it had data.
        """
        model = config.model
        return cls(
            horizons=tuple(int(value) for value in model.horizons_minutes),
            classes=tuple(str(name) for name in model.labels.classes),
            timeframe=str(timeframe or model.dataset.timeframe),
            categorical_columns=tuple(str(name) for name in model.dataset.categorical_columns),
            label_version=str(model.labels.version),
            timezone_name=str(getattr(config, "timezone_name", DEFAULT_TIMEZONE)),
        )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary of the policy this dataset was built under."""
        return {
            "timeframe": self.timeframe,
            "horizons_minutes": list(self.horizons),
            "classes": list(self.classes),
            "categorical_columns": list(self.categorical_columns),
            "label_version": self.label_version,
            "timezone": self.timezone_name,
        }


# ---------------------------------------------------------------------------
# Counts: what was joined, what was kept, what was dropped
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DatasetCounts:
    """How many rows entered the join and how many survived it.

    Every drop is a *published* number, never a silent one: ``labelled_rows`` is the
    matrix, and the gap between ``feature_rows`` and it is the whole reason a dataset is
    smaller than the feature table.
    """

    feature_rows: int = 0
    label_rows: int = 0
    joined_rows: int = 0
    labelled_rows: int = 0
    unlabelled_rows: int = 0
    features_without_label: int = 0
    labels_without_feature: int = 0
    other_timeframe_rows: int = 0
    other_symbol_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "feature_rows": self.feature_rows,
            "label_rows": self.label_rows,
            "joined_rows": self.joined_rows,
            "labelled_rows": self.labelled_rows,
            "unlabelled_rows": self.unlabelled_rows,
            "features_without_label": self.features_without_label,
            "labels_without_feature": self.labels_without_feature,
            "other_timeframe_rows": self.other_timeframe_rows,
            "other_symbol_rows": self.other_symbol_rows,
        }


# ---------------------------------------------------------------------------
# The dataset itself
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Dataset:
    """One horizon's training matrix, plus the identity columns a split needs."""

    horizon: int
    timeframe: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    classes: tuple[str, ...]
    counts: DatasetCounts
    label_version: str | None = None
    timezone_name: str = DEFAULT_TIMEZONE
    missing_feature_columns: tuple[str, ...] = field(default_factory=tuple)

    # -- shape ---------------------------------------------------------------
    @property
    def target_column(self) -> str:
        """Name of the class column (``"label"``)."""
        return TARGET_COLUMN

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        """Feature columns that are not categorical."""
        categorical = set(self.categorical_columns)
        return tuple(name for name in self.feature_columns if name not in categorical)

    @property
    def rows(self) -> int:
        """Number of training rows (rows carrying a class)."""
        return int(len(self.frame))

    @property
    def timestamps(self) -> pd.Series:
        """The row timestamps, in stored (IST) order."""
        return self.frame["timestamp"]

    @property
    def days(self) -> tuple[date, ...]:
        """Trading days the matrix covers, oldest first."""
        if self.frame.empty:
            return ()
        tz = get_timezone(self.timezone_name)
        return tuple(sorted({moment.date() for moment in self.timestamps.dt.tz_convert(tz)}))

    @property
    def symbols(self) -> tuple[str, ...]:
        """Contracts the matrix covers."""
        if self.frame.empty:
            return ()
        return tuple(sorted({str(name) for name in self.frame["tradingsymbol"]}))

    # -- target --------------------------------------------------------------
    def class_counts(self) -> dict[str, int]:
        """Rows per class, in the configured class order (missing classes as zero)."""
        counts = dict.fromkeys(self.classes, 0)
        if not self.rows:
            return counts
        for name, value in self.frame[TARGET_COLUMN].value_counts().items():
            counts[str(name)] = int(value)
        return counts

    def target_codes(self) -> pd.Series:
        """The class as an integer code (``classes[i]`` <-> ``i``), as a model wants it."""
        if self.frame.empty:
            return pd.Series(dtype="int64")
        return self.frame[CODE_COLUMN].astype("int64")

    def matrix(self) -> pd.DataFrame:
        """The feature matrix ``X``: feature columns only, in the stored contract's order."""
        return self.frame.loc[:, list(self.feature_columns)].copy()

    # -- honesty about the matrix -------------------------------------------
    def constant_columns(self) -> tuple[str, ...]:
        """Feature columns carrying a single value in this dataset.

        A column that is entirely missing counts as constant - it carries no information
        here either way. Reported, never removed: a column that is constant *here* may vary
        inside a training fold, and feature selection must be fit on training folds only.
        """
        constant = [
            name for name in self.feature_columns if self.frame[name].nunique(dropna=True) <= 1
        ]
        return tuple(constant)

    def null_shares(self) -> dict[str, float]:
        """Share of missing values per feature column (``0.0`` for an empty dataset)."""
        if self.frame.empty:
            return dict.fromkeys(self.feature_columns, 0.0)
        return {name: float(self.frame[name].isna().mean()) for name in self.feature_columns}

    # -- reporting -----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """JSON-friendly description - what a report and the run manifest carry."""
        return {
            "horizon_minutes": self.horizon,
            "timeframe": self.timeframe,
            "label_version": self.label_version,
            "rows": self.rows,
            "features": len(self.feature_columns),
            "numeric_features": len(self.numeric_columns),
            "categorical_features": len(self.categorical_columns),
            "first_timestamp": self.timestamps.iloc[0].isoformat() if self.rows else None,
            "last_timestamp": self.timestamps.iloc[-1].isoformat() if self.rows else None,
            "days": len(self.days),
            "symbols": len(self.symbols),
            "class_counts": self.class_counts(),
            "counts": self.counts.to_dict(),
            "constant_columns": list(self.constant_columns()),
            "missing_feature_columns": list(self.missing_feature_columns),
        }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, source: str) -> None:
    """Raise if ``frame`` is missing a required column, naming every missing one."""
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise DatasetError(
            f"The {source} frame is missing column(s) {missing}; both sides of the join are "
            "described by the stored contracts in core/constants.py"
        )


def _require_aware_timestamps(frame: pd.DataFrame, *, source: str) -> None:
    """Raise unless ``timestamp`` is tz-aware (a naive one would silently mis-join)."""
    if frame.empty:
        return
    if not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        raise DatasetError(
            f"The {source} frame's timestamp column is not timezone-aware; ANJANEYA stores "
            "every timestamp in Asia/Kolkata and a naive one cannot be joined safely"
        )


def _require_unique_keys(frame: pd.DataFrame, *, source: str, note: str = "") -> None:
    """Raise if a key repeats - the join would duplicate rows instead of failing."""
    if frame.empty:
        return
    duplicated = frame.duplicated(subset=list(DATASET_KEYS))
    if bool(duplicated.any()):
        first = frame.loc[duplicated, list(DATASET_KEYS)].iloc[0]
        raise DatasetError(
            f"The {source} frame repeats the key (timestamp, tradingsymbol) "
            f"(first: {first['timestamp']} / {first['tradingsymbol']}); a dataset row must be "
            "one moment of one contract, so a duplicate is a data problem, not something to "
            f"absorb.{(' ' + note) if note else ''}"
        )


def build_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    policy: DatasetPolicy,
    horizon: int,
    symbols: Sequence[str] | None = None,
) -> Dataset:
    """Join one horizon's labels to the features at the same instant (spec 21).

    Only rows carrying a class survive; everything else is counted. An empty result is a
    legitimate answer (a timeframe nobody captured, a horizon nothing has labelled yet)
    and comes back as an empty :class:`Dataset` rather than an exception.
    """
    if horizon not in policy.horizons:
        raise DatasetError(
            f"horizon {horizon} is not one of the configured horizons {list(policy.horizons)}; "
            "add it to model.horizons_minutes rather than labelling it in passing"
        )

    counts = DatasetCounts()
    _require_columns(features, [*DATASET_KEYS, "timeframe"], source="feature")
    _require_columns(labels, [*DATASET_KEYS, HORIZON_COLUMN, TARGET_COLUMN], source="label")
    _require_aware_timestamps(features, source="feature")
    _require_aware_timestamps(labels, source="label")

    counts.feature_rows = int(len(features))
    in_timeframe = features["timeframe"] == policy.timeframe
    counts.other_timeframe_rows = int((~in_timeframe).sum())
    features = features.loc[in_timeframe]
    if symbols:
        wanted = set(symbols)
        counts.other_symbol_rows = int((~features["tradingsymbol"].isin(wanted)).sum())
        features = features.loc[features["tradingsymbol"].isin(wanted)]
    # The key check runs on the *selected* rows: the stored frame legitimately holds the
    # same (timestamp, contract) at several timeframes, and only one of them is ours.
    _require_unique_keys(features, source="feature")

    selected = labels.loc[labels[HORIZON_COLUMN] == horizon].copy()
    counts.label_rows = int(len(selected))
    _require_unique_keys(
        selected,
        source=f"label (horizon {horizon})",
        note=(
            "The labels table is keyed by (timestamp, contract, horizon) and carries no "
            "timeframe column, so label builds of the same contract at different timeframes "
            "collide here: rebuild the labels for the one timeframe this dataset trains on."
        ),
    )

    # One label version per dataset: mixing versions would put two definitions of the same
    # class beside each other in one matrix, which is a research bug, not data to absorb.
    versions = sorted(
        {
            str(value)
            for value in selected.get("label_version", pd.Series(dtype=object)).dropna()
        }
    )
    if len(versions) > 1:
        raise DatasetError(
            f"label rows for horizon {horizon} carry more than one label_version {versions}; "
            "rebuild the labels under one version rather than training on a mixture"
        )
    label_version = versions[0] if versions else None

    if features.empty or selected.empty:
        return _empty_dataset(
            policy=policy, horizon=horizon, counts=counts, label_version=label_version
        )

    label_columns = [name for name in LABEL_JOIN_COLUMNS if name in selected.columns]
    merged = features.merge(
        selected.loc[:, [*DATASET_KEYS, *label_columns]],
        on=list(DATASET_KEYS),
        how="inner",
        validate="one_to_one",
        suffixes=("", "_label"),
    )
    counts.joined_rows = int(len(merged))
    counts.features_without_label = (
        counts.feature_rows
        - counts.other_timeframe_rows
        - counts.other_symbol_rows
        - counts.joined_rows
    )
    counts.labels_without_feature = counts.label_rows - counts.joined_rows

    unlabelled = merged[TARGET_COLUMN].isna()
    counts.unlabelled_rows = int(unlabelled.sum())
    labelled = merged.loc[~unlabelled].copy()
    counts.labelled_rows = int(len(labelled))

    if labelled.empty:
        return _empty_dataset(
            policy=policy, horizon=horizon, counts=counts, label_version=label_version
        )

    unknown = sorted(
        {str(value) for value in labelled[TARGET_COLUMN] if str(value) not in policy.classes}
    )
    if unknown:
        raise DatasetError(
            f"stored label(s) {unknown} are not among the configured classes "
            f"{list(policy.classes)}; the label policy and the model disagree"
        )

    labelled[CODE_COLUMN] = pd.Categorical(
        labelled[TARGET_COLUMN].astype(str), categories=list(policy.classes)
    ).codes.astype("int64")
    labelled = labelled.sort_values(list(DATASET_KEYS), kind="stable").reset_index(drop=True)

    # Features are *the stored contract* minus identity/provenance minus every
    # label-derived name - so a column that appears in the frame by accident (or a label
    # column smuggled in by a future schema) cannot become a feature silently.
    available = set(labelled.columns)
    missing_categorical = [name for name in policy.categorical_columns if name not in available]
    if missing_categorical:
        raise DatasetError(
            f"Configured categorical column(s) {missing_categorical} are not in the joined "
            "frame; the feature contract has drifted from config/model.yaml"
        )
    categorical = tuple(name for name in policy.categorical_columns if name in available)
    feature_columns = tuple(
        name
        for name in FEATURE_COLUMNS
        if name in available and name not in META_COLUMNS and not _is_label_derived(name)
    )
    assert_no_label_leakage(feature_columns)

    # Categoricals ship as pandas ``category`` dtype: the Parquet artefact then records
    # their vocabulary, and the trainer can hand a model the codes it wants.
    for name in categorical:
        labelled[name] = labelled[name].astype("category")

    return Dataset(
        horizon=horizon,
        timeframe=policy.timeframe,
        frame=labelled,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        classes=policy.classes,
        counts=counts,
        label_version=label_version,
        timezone_name=policy.timezone_name,
        missing_feature_columns=tuple(
            name for name in FEATURE_COLUMNS if name not in available
        ),
    )


def _empty_dataset(
    *,
    policy: DatasetPolicy,
    horizon: int,
    counts: DatasetCounts,
    label_version: str | None,
) -> Dataset:
    """An empty (but valid) dataset - the honest answer when nothing can be joined."""
    return Dataset(
        horizon=horizon,
        timeframe=policy.timeframe,
        frame=pd.DataFrame(
            columns=["timestamp", "tradingsymbol", HORIZON_COLUMN, TARGET_COLUMN, CODE_COLUMN]
        ),
        feature_columns=(),
        categorical_columns=(),
        classes=policy.classes,
        counts=counts,
        label_version=label_version,
        timezone_name=policy.timezone_name,
        missing_feature_columns=FEATURE_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Spec 4.3 data-accumulation gates
# ---------------------------------------------------------------------------
# The ladder (:data:`GATE_STAGES`), the stage verdict (:class:`GateStatus`) and the pure rule
# (:func:`data_gate_status`) now live in :mod:`core.gates` and are imported above, so a
# model-free report (spec 29's ``core.accumulation``) can reach them without importing pandas
# or LightGBM. They stay re-exported here because every existing caller imports them from
# ``core.ml`` / ``core.ml.dataset``.







