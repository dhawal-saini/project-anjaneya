"""Purged walk-forward validation: the only splitter ANJANEYA is allowed to use (spec 22).

Financial time series is not a bag of rows. Shuffling it - or splitting it on a random
fold - puts tomorrow's bar next to today's label and produces a model that looks excellent
and is worthless. Spec 22 therefore mandates **purged walk-forward** with an **embargo
equal to the maximum label horizon**, and this module is that mandate in code, with three
properties that make it checkable rather than aspirational:

* **Whole days, in order.** Folds are built from the *trading days the dataset actually
  has*, oldest first: train window, then the validation window that follows it, then the
  window slides by ``step_days``. No fold ever validates on a day it trained on, and the
  final ``holdout_days`` are removed from *every* window before the walk starts - the
  untouched period spec 22 demands.
* **The boundary is guarded twice.** Every training row within ``embargo_minutes`` of the
  validation window is dropped (the embargo), and every training row whose label window
  ``[t, t + horizon)`` would reach into the validation window is counted as purged. With a
  session-bound label and an embargo that equals the longest horizon the second set is a
  subset of the first - that is the *point*: the splitter refuses
  ``embargo < horizon`` outright, so the two guards cannot disagree.
* **The split is printed, not implied.** Every fold reports its exact first/last train and
  validation timestamps, the two cut-offs, and how many rows each guard removed, so a
  research report can quote the split without re-deriving it.

Nothing here reads a file or a database: it takes a :class:`core.ml.dataset.Dataset` and
returns a description of how to cut it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from core.constants import DEFAULT_TIMEZONE, get_timezone
from core.ml.dataset import TARGET_COLUMN, Dataset

__all__ = [
    "DatasetSplit",
    "Fold",
    "HoldoutWindow",
    "SplitError",
    "SplitPolicy",
    "fold_frames",
    "holdout_frame",
    "purged_walk_forward_splits",
    "usable_frame",
]


class SplitError(ValueError):
    """Raised when a split policy would not be a leak-free walk-forward."""


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Spec 22's split policy (``config/model.yaml`` ``walk_forward`` + the horizon)."""

    train_window_days: int
    validate_window_days: int
    step_days: int
    embargo_minutes: int
    holdout_days: int
    horizon_minutes: int
    purge: bool = True
    timezone_name: str = DEFAULT_TIMEZONE

    def __post_init__(self) -> None:
        """Refuse a policy that cannot be leak-free (spec 22)."""
        windows = {
            "train_window_days": self.train_window_days,
            "validate_window_days": self.validate_window_days,
            "step_days": self.step_days,
            "holdout_days": self.holdout_days,
            "horizon_minutes": self.horizon_minutes,
        }
        bad = [name for name, value in windows.items() if value <= 0]
        if bad:
            raise SplitError(f"{bad} must be positive numbers of days/minutes")
        if self.step_days > self.train_window_days:
            raise SplitError("step_days must not exceed train_window_days")
        if self.embargo_minutes < self.horizon_minutes:
            raise SplitError(
                f"embargo_minutes ({self.embargo_minutes}) must be >= the label horizon "
                f"({self.horizon_minutes} minutes): the embargo is what stops a training row's "
                "label from being adjacent to the window it is validated on (spec 22)"
            )

    @property
    def days_required(self) -> int:
        """Captured days needed before a single fold exists (holdout + train + validate)."""
        return self.holdout_days + self.train_window_days + self.validate_window_days

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        horizon_minutes: int,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> SplitPolicy:
        """Build the policy from a loaded :class:`core.config_loader.AnjaneyaConfig`."""
        walk_forward = config.model.walk_forward
        return cls(
            train_window_days=int(walk_forward.train_window_days),
            validate_window_days=int(walk_forward.validate_window_days),
            step_days=int(walk_forward.step_days),
            embargo_minutes=int(walk_forward.embargo_minutes),
            holdout_days=int(walk_forward.holdout_days),
            horizon_minutes=int(horizon_minutes),
            purge=bool(walk_forward.purge),
            timezone_name=str(timezone_name),
        )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary of the policy a split was made under."""
        return {
            "train_window_days": self.train_window_days,
            "validate_window_days": self.validate_window_days,
            "step_days": self.step_days,
            "embargo_minutes": self.embargo_minutes,
            "holdout_days": self.holdout_days,
            "horizon_minutes": self.horizon_minutes,
            "purge": self.purge,
            "days_required": self.days_required,
        }


# ---------------------------------------------------------------------------
# What a fold is
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Fold:
    """One train/validate pair, described exactly enough to be reproducible.

    ``train_rows`` counts the rows that **survive** the guards; ``embargoed_rows`` and
    ``purged_rows`` say what the guards removed and why, and ``removed_rows`` is the size
    of their union - the number that actually left the training set. ``purge`` records
    whether the purge was applied or only *counted* (``walk_forward.purge: false``), because
    a fold that reports ``purged_rows`` without removing them must say so.
    """

    index: int
    train_days: tuple[date, ...]
    validate_days: tuple[date, ...]
    train_rows: int
    validate_rows: int
    removed_rows: int
    embargoed_rows: int
    purged_rows: int
    train_start: datetime | None
    train_end: datetime | None
    validate_start: datetime | None
    validate_end: datetime | None
    embargo_cutoff: datetime | None
    purge_cutoff: datetime | None
    purge: bool = True
    train_class_counts: dict[str, int] = field(default_factory=dict)
    validate_class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (timestamps as ISO strings - spec 22's 'document every split')."""
        return {
            "fold": self.index,
            "train_days": [day.isoformat() for day in self.train_days],
            "validate_days": [day.isoformat() for day in self.validate_days],
            "train_rows": self.train_rows,
            "validate_rows": self.validate_rows,
            "removed_rows": self.removed_rows,
            "embargoed_rows": self.embargoed_rows,
            "purged_rows": self.purged_rows,
            "train_first_timestamp": _iso(self.train_start),
            "train_last_timestamp": _iso(self.train_end),
            "validate_first_timestamp": _iso(self.validate_start),
            "validate_last_timestamp": _iso(self.validate_end),
            "embargo_cutoff": _iso(self.embargo_cutoff),
            "purge_cutoff": _iso(self.purge_cutoff),
            "purge": self.purge,
            "train_class_counts": dict(self.train_class_counts),
            "validate_class_counts": dict(self.validate_class_counts),
        }


@dataclass(slots=True)
class HoldoutWindow:
    """The final untouched period (spec 22). Never part of any fold."""

    days: tuple[date, ...]
    rows: int
    start: datetime | None
    end: datetime | None
    class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "days": [day.isoformat() for day in self.days],
            "rows": self.rows,
            "first_timestamp": _iso(self.start),
            "last_timestamp": _iso(self.end),
            "class_counts": dict(self.class_counts),
        }


@dataclass(slots=True)
class DatasetSplit:
    """Every fold of one dataset, plus the holdout and the reason when there are none."""

    horizon: int
    policy: SplitPolicy
    folds: list[Fold] = field(default_factory=list)
    holdout: HoldoutWindow | None = None
    days: int = 0
    rows: int = 0
    train_rows: int = 0
    validate_rows: int = 0
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when at least one fold exists."""
        return bool(self.folds)

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        if not self.ok:
            return f"horizon {self.horizon}m: no folds - {self.reason}"
        return (
            f"horizon {self.horizon}m: {len(self.folds)} fold(s) over {self.days} day(s), "
            f"holdout {len(self.holdout.days) if self.holdout else 0} day(s) untouched"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - the split manifest written beside the dataset artefact."""
        return {
            "horizon_minutes": self.horizon,
            "policy": self.policy.describe(),
            "days": self.days,
            "rows": self.rows,
            "folds": len(self.folds),
            "train_rows": self.train_rows,
            "validate_rows": self.validate_rows,
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "reason": self.reason,
            "splits": [fold.to_dict() for fold in self.folds],
        }


def _iso(moment: datetime | None) -> str | None:
    """ISO string for a timestamp, or ``None`` when the fold had no such row."""
    return moment.isoformat() if moment is not None else None


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------
def _class_counts(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict[str, int]:
    """Rows per class in a slice of a dataset (missing classes as zero)."""
    counts: dict[str, int] = dict.fromkeys(classes, 0)
    if not frame.empty:
        for name, value in frame[TARGET_COLUMN].value_counts().items():
            counts[str(name)] = int(value)
    return counts


def _days_of(dataset: Dataset) -> pd.Series:
    """The IST calendar day of every row, as a date series (the split's own clock)."""
    tz = get_timezone(dataset.timezone_name)
    return dataset.frame["timestamp"].dt.tz_convert(tz).dt.date


def _select(frame: pd.DataFrame, days: Sequence[date], day_series: pd.Series) -> pd.DataFrame:
    """The rows of ``frame`` that fall on ``days`` - the *only* way a window is materialised."""
    return frame.loc[day_series.isin(days)]


def _boundary_masks(
    train: pd.DataFrame, *, embargo_cutoff: datetime, purge_cutoff: datetime
) -> tuple[pd.Series, pd.Series]:
    """The two boundary guards as row masks over the training rows.

    ``purge_cutoff = validate_start - horizon`` is stored in the fold, so both masks can be
    rebuilt from the fold alone - which is what lets :func:`fold_frames` hand out exactly the
    rows the fold counted instead of a second, possibly different, cut.
    """
    return train["timestamp"] > embargo_cutoff, train["timestamp"] > purge_cutoff


def _kept_rows(
    train: pd.DataFrame, *, embargo_mask: pd.Series, purge_mask: pd.Series, purge: bool
) -> pd.DataFrame:
    """The training rows that survive the guards."""
    removed = (embargo_mask | purge_mask) if purge else embargo_mask
    return train.loc[~removed]


def _build_fold(
    dataset: Dataset,
    *,
    index: int,
    train_days: tuple[date, ...],
    validate_days: tuple[date, ...],
    policy: SplitPolicy,
    day_series: pd.Series,
) -> Fold:
    """Cut one fold out of the dataset and apply the two boundary guards."""
    frame = dataset.frame
    train = _select(frame, train_days, day_series)
    validate = _select(frame, validate_days, day_series)

    validate_start = validate["timestamp"].min() if not validate.empty else None
    validate_end = validate["timestamp"].max() if not validate.empty else None
    embargo_cutoff = purge_cutoff = None
    kept = train
    embargoed = purged = 0
    if validate_start is not None:
        # The embargo: keep nothing that sits within `embargo_minutes` of the validation
        # window - adjacent bars share almost all of their information, so training on them
        # and validating on the next one is a measured, not a predicted, result.
        embargo_cutoff = validate_start - timedelta(minutes=policy.embargo_minutes)
        purge_cutoff = validate_start - timedelta(minutes=policy.horizon_minutes)
        embargo_mask, purge_mask = _boundary_masks(
            train, embargo_cutoff=embargo_cutoff, purge_cutoff=purge_cutoff
        )
        embargoed = int(embargo_mask.sum())
        purged = int(purge_mask.sum())
        kept = _kept_rows(
            train, embargo_mask=embargo_mask, purge_mask=purge_mask, purge=policy.purge
        )

    return Fold(
        index=index,
        train_days=train_days,
        validate_days=validate_days,
        train_rows=int(len(kept)),
        validate_rows=int(len(validate)),
        removed_rows=int(len(train) - len(kept)),
        embargoed_rows=embargoed,
        purged_rows=purged,
        train_start=kept["timestamp"].min() if not kept.empty else None,
        train_end=kept["timestamp"].max() if not kept.empty else None,
        validate_start=validate_start,
        validate_end=validate_end,
        embargo_cutoff=embargo_cutoff,
        purge_cutoff=purge_cutoff,
        purge=policy.purge,
        train_class_counts=_class_counts(kept, dataset.classes),
        validate_class_counts=_class_counts(validate, dataset.classes),
    )


def purged_walk_forward_splits(dataset: Dataset, *, policy: SplitPolicy) -> DatasetSplit:
    """Split a dataset into spec 22's purged walk-forward folds plus a holdout.

    The walk runs over the days the dataset *has*, so a dataset that skips a holiday or a
    day with no capture is split on its own calendar - no phantom days, and no fold whose
    window is mostly empty. When the dataset is too short for even one fold, the reason is
    returned instead of an exception: "not enough data yet" is a normal research state
    (spec 4.3's gates), and the operator needs the number of days still missing.
    """
    split = DatasetSplit(horizon=dataset.horizon, policy=policy, days=len(dataset.days), rows=dataset.rows)
    if not dataset.rows:
        split.reason = "the dataset has no labelled rows to split"
        return split

    days = list(dataset.days)
    holdout_days = tuple(days[-policy.holdout_days :])
    usable = days[: -policy.holdout_days]
    day_series = _days_of(dataset)

    holdout_rows = dataset.frame.loc[day_series.isin(holdout_days)]
    split.holdout = HoldoutWindow(
        days=holdout_days,
        rows=int(len(holdout_rows)),
        start=holdout_rows["timestamp"].min() if not holdout_rows.empty else None,
        end=holdout_rows["timestamp"].max() if not holdout_rows.empty else None,
        class_counts=_class_counts(holdout_rows, dataset.classes),
    )

    needed = policy.train_window_days + policy.validate_window_days
    if len(usable) < needed:
        split.reason = (
            f"needs {policy.days_required} captured day(s) (holdout {policy.holdout_days} + "
            f"train {policy.train_window_days} + validate {policy.validate_window_days}); this "
            f"dataset has {len(days)}"
        )
        return split

    index = 0
    start = 0
    while start + needed <= len(usable):
        train_days = tuple(usable[start : start + policy.train_window_days])
        validate_days = tuple(
            usable[start + policy.train_window_days : start + policy.train_window_days + policy.validate_window_days]
        )
        split.folds.append(
            _build_fold(
                dataset,
                index=index,
                train_days=train_days,
                validate_days=validate_days,
                policy=policy,
                day_series=day_series,
            )
        )
        index += 1
        start += policy.step_days

    split.train_rows = sum(fold.train_rows for fold in split.folds)
    split.validate_rows = sum(fold.validate_rows for fold in split.folds)
    return split




# ---------------------------------------------------------------------------
# Materialising a split: the rows the trainer is allowed to see
# ---------------------------------------------------------------------------
def fold_frames(dataset: Dataset, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The exact ``(train, validate)`` rows of one fold, guards applied.

    The masks are rebuilt from the fold's own stored cut-offs, so the frames handed to the
    trainer are the rows the fold *counted* - ``len(train) == fold.train_rows``. That matters
    more than it sounds: a trainer that re-derived the embargo could quietly train on rows the
    report says were removed, and every metric would look better than the documented split
    allows.
    """
    day_series = _days_of(dataset)
    train = _select(dataset.frame, fold.train_days, day_series)
    validate = _select(dataset.frame, fold.validate_days, day_series)
    if (
        fold.validate_start is None
        or fold.embargo_cutoff is None
        or fold.purge_cutoff is None
    ):
        return train, validate
    embargo_mask, purge_mask = _boundary_masks(
        train, embargo_cutoff=fold.embargo_cutoff, purge_cutoff=fold.purge_cutoff
    )
    kept = _kept_rows(
        train, embargo_mask=embargo_mask, purge_mask=purge_mask, purge=fold.purge
    )
    return kept, validate


def holdout_frame(dataset: Dataset, split: DatasetSplit) -> pd.DataFrame:
    """The rows of the untouched holdout (empty when the split has none)."""
    if split.holdout is None:
        return dataset.frame.iloc[0:0]
    return _select(dataset.frame, split.holdout.days, _days_of(dataset))


def usable_frame(dataset: Dataset, split: DatasetSplit) -> pd.DataFrame:
    """Every row outside the holdout - what a final model may legitimately be fitted on."""
    if split.holdout is None or not split.holdout.days:
        return dataset.frame
    return dataset.frame.loc[~_days_of(dataset).isin(set(split.holdout.days))]


