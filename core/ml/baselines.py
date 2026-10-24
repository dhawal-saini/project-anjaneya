"""Baseline models: the bar a trained model has to clear (spec 21).

A log loss means nothing on its own. Four classes at roughly equal frequencies and a model
that has learned nothing still posts ``log(4) ~= 1.386``; a dataset with a dominant class
posts far less. Spec 21 therefore names the comparisons - **always up**, **always range**,
**random** and **regime only** - and a trained model's report is only readable next to them.

Two properties make these baselines honest rather than decorative:

* **They are fit on the training fold only.** ``nifty_regime_only`` learns its class
  distribution per NIFTY regime from the training rows, so scoring it on the validation
  window is the same kind of out-of-sample measurement as scoring the model. A baseline
  fitted on everything would beat the model for free.
* **They are deterministic given a seed.** ``random`` re-seeds on every call, so the same
  rows and the same seed produce the same probabilities no matter how calls are ordered - a
  baseline that changes between two runs of the same command is not a reference point.

Constant baselines are deliberately *un*smoothed (one class gets probability 1). Their log
loss is then dominated by the clipping at
:data:`core.ml.evaluation.PROBABILITY_EPSILON`, which is the honest way to say "this rule is
wrong whenever it is wrong".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.ml.dataset import TARGET_COLUMN

__all__ = [
    "ALWAYS_RANGE_CLASS",
    "ALWAYS_UP_CLASS",
    "BASELINE_DESCRIPTIONS",
    "BASELINE_NAMES",
    "DEFAULT_REGIME_COLUMN",
    "Baseline",
    "BaselineError",
    "build_baseline",
    "build_baselines",
]

#: The class ``always_up`` predicts. The name is the label builder's (spec 20); a
#: configuration whose class list lacks it cannot express this baseline at all.
ALWAYS_UP_CLASS = "UP_CONTINUATION"

#: The class ``always_range`` predicts.
ALWAYS_RANGE_CLASS = "RANGE"

#: The feature ``nifty_regime_only`` conditions on.
DEFAULT_REGIME_COLUMN = "nifty_regime"

#: Every baseline name ``model.evaluation.baselines`` may contain, in report order.
BASELINE_NAMES: tuple[str, ...] = (
    "always_up",
    "always_range",
    "random",
    "nifty_regime_only",
)

#: What each baseline is, in one sentence - printed in the report and stored in the model.
BASELINE_DESCRIPTIONS: dict[str, str] = {
    "always_up": f"always predicts {ALWAYS_UP_CLASS} with probability 1",
    "always_range": f"always predicts {ALWAYS_RANGE_CLASS} with probability 1",
    "random": "uniform random probabilities over the classes",
    "nifty_regime_only": (
        f"the training fold's class distribution per {DEFAULT_REGIME_COLUMN} level, smoothed; "
        "unseen levels fall back to the training fold's prior"
    ),
}

#: Add-one smoothing for the regime tables. Without it a level seen once would produce a
#: one-hot baseline whose score is clipping, not calibration.
DEFAULT_SMOOTHING = 1.0


class BaselineError(ValueError):
    """Raised when a baseline cannot be built from what it was given."""


@dataclass(frozen=True, slots=True)
class Baseline:
    """One fitted baseline: a rule, and everything it learned from the training fold."""

    name: str
    kind: str
    classes: tuple[str, ...]
    description: str
    seed: int = 0
    level: str | None = None
    prior: tuple[float, ...] = ()
    table: dict[str, tuple[float, ...]] = field(default_factory=dict)
    regime_column: str = DEFAULT_REGIME_COLUMN

    @property
    def levels(self) -> tuple[str, ...]:
        """The regime levels this baseline learned (empty for the other kinds)."""
        return tuple(sorted(self.table))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """A probability per class for every row of ``frame`` (shape ``(rows, n_classes)``)."""
        rows = len(frame)
        width = len(self.classes)
        if rows == 0:
            return np.zeros((0, width), dtype="float64")
        if self.kind == "constant":
            probabilities = np.zeros((rows, width), dtype="float64")
            probabilities[:, self.classes.index(str(self.level))] = 1.0
            return probabilities
        if self.kind == "random":
            generator = np.random.default_rng(self.seed)
            draws = generator.random((rows, width))
            return draws / draws.sum(axis=1, keepdims=True)
        if self.kind == "regime":
            return self._regime_probabilities(frame, rows=rows, width=width)
        raise BaselineError(f"unknown baseline kind {self.kind!r} for {self.name!r}")

    def _regime_probabilities(self, frame: pd.DataFrame, *, rows: int, width: int) -> np.ndarray:
        """Look the regime up row by row, falling back to the training fold's prior."""
        if self.regime_column not in frame.columns:
            raise BaselineError(
                f"baseline {self.name!r} conditions on {self.regime_column!r}, which is not a "
                "column of this frame; check model.dataset.categorical_columns and the stored "
                "feature contract"
            )
        fallback = np.asarray(
            self.prior if self.prior else (1.0 / width,) * width, dtype="float64"
        )
        probabilities = np.empty((rows, width), dtype="float64")
        for position, value in enumerate(frame[self.regime_column]):
            level = _level_of(value)
            probabilities[position] = fallback if level is None else self.table.get(level, fallback)
        return probabilities

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - what the model metadata stores beside the metrics."""
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "classes": list(self.classes),
        }
        if self.kind == "random":
            payload["seed"] = self.seed
        if self.kind == "constant":
            payload["always"] = self.level
        if self.kind == "regime":
            payload["regime_column"] = self.regime_column
            payload["prior"] = list(self.prior)
            payload["levels"] = list(self.levels)
            payload["table"] = {
                name: list(values) for name, values in sorted(self.table.items())
            }
        return payload

    """Raised when a baseline cannot be built from what it was given."""



# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _validated_classes(classes: Sequence[str]) -> tuple[str, ...]:
    """The class list, non-empty and unique, or a ``BaselineError``."""
    names = tuple(str(name) for name in classes)
    if not names:
        raise BaselineError("a baseline needs the configured class list")
    if len(set(names)) != len(names):
        raise BaselineError(f"class names must be unique, got {list(names)}")
    return names


def _level_of(value: Any) -> str | None:
    """A regime level as a string, or ``None`` when the value is missing.

    ``None``/NaN is *not* a level: a row whose regime could not be computed must fall back
    to the training prior rather than create a "nan" regime the training fold happens to
    have seen once.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value)
    return None if text in ("", "nan", "None") else text



def _distribution(counts: np.ndarray, *, smoothing: float) -> tuple[float, ...]:
    """Laplace-smoothed distribution from class counts."""
    smoothed = counts.astype("float64") + smoothing
    return tuple(float(value) for value in smoothed / smoothed.sum())


def _class_counts(frame: pd.DataFrame, *, classes: tuple[str, ...]) -> np.ndarray:
    """Class counts of a frame, ordered by ``classes``."""
    counts = np.zeros(len(classes), dtype="float64")
    if frame.empty or TARGET_COLUMN not in frame.columns:
        return counts
    observed = frame[TARGET_COLUMN].astype(str).value_counts()
    for position, name in enumerate(classes):
        counts[position] = float(observed.get(name, 0))
    return counts


def build_baseline(
    name: str,
    *,
    train_frame: pd.DataFrame,
    classes: Sequence[str],
    seed: int = 0,
    regime_column: str = DEFAULT_REGIME_COLUMN,
    smoothing: float = DEFAULT_SMOOTHING,
) -> Baseline:
    """Fit one baseline on ``train_frame`` (the training fold - never the validation rows)."""
    names = _validated_classes(classes)
    key = str(name)
    if key not in BASELINE_NAMES:
        raise BaselineError(
            f"unknown baseline {key!r}; configured baselines must be one of {list(BASELINE_NAMES)}"
        )
    if smoothing <= 0:
        raise BaselineError("smoothing must be > 0 so a level seen once is not a one-hot")
    if train_frame.empty:
        raise BaselineError(f"baseline {key!r} cannot be fitted: the training frame has no rows")

    if key == "random":
        return Baseline(
            name=key,
            kind="random",
            classes=names,
            description=BASELINE_DESCRIPTIONS[key],
            seed=int(seed),
        )

    if key in ("always_up", "always_range"):
        target = ALWAYS_UP_CLASS if key == "always_up" else ALWAYS_RANGE_CLASS
        if target not in names:
            raise BaselineError(
                f"baseline {key!r} needs a {target!r} class, which model.labels.classes does not "
                f"contain ({list(names)}); add the class or drop the baseline"
            )
        return Baseline(
            name=key,
            kind="constant",
            classes=names,
            description=BASELINE_DESCRIPTIONS[key],
            level=target,
        )

    if regime_column not in train_frame.columns:
        raise BaselineError(
            f"baseline {key!r} conditions on {regime_column!r}, which is not a column of the "
            "training frame; check model.dataset.categorical_columns and the stored feature "
            "contract"
        )
    prior = _distribution(_class_counts(train_frame, classes=names), smoothing=smoothing)
    values = train_frame[regime_column]
    levels = {level for level in (_level_of(value) for value in values) if level is not None}
    table: dict[str, tuple[float, ...]] = {}
    for level in sorted(levels):
        subset = train_frame.loc[
            [str(_level_of(value) or "") == level for value in values]
        ]
        table[level] = _distribution(_class_counts(subset, classes=names), smoothing=smoothing)
    return Baseline(
        name=key,
        kind="regime",
        classes=names,
        description=BASELINE_DESCRIPTIONS[key],
        prior=prior,
        table=table,
        regime_column=str(regime_column),
    )


def build_baselines(
    names: Sequence[str],
    *,
    train_frame: pd.DataFrame,
    classes: Sequence[str],
    seed: int = 0,
    regime_column: str = DEFAULT_REGIME_COLUMN,
) -> list[Baseline]:
    """Fit every configured baseline on the same training fold, in the configured order."""
    wanted = tuple(str(name) for name in names)
    if not wanted:
        raise BaselineError(
            "model.evaluation.baselines is empty; a model with nothing to beat is not a result"
        )
    return [
        build_baseline(
            name,
            train_frame=train_frame,
            classes=classes,
            seed=seed,
            regime_column=regime_column,
        )
        for name in wanted
    ]
