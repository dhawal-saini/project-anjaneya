"""Training a model on a leak-free matrix (spec 21).

The trainer does one thing: given a *training* frame and the metadata a dataset carries
(features, categoricals, classes), it fits a LightGBM multiclass model that outputs a
probability per label class, and it refuses - loudly - anything that would make the
resulting numbers uninterpretable:

* **A single-class training frame is refused.** LightGBM would fit it and every metric
  would look perfect; a fold that happens to contain one class is a *reported* skip, never
  a silent 100%.
* **Label-derived columns are refused as features**
  (:func:`core.ml.dataset.assert_no_label_leakage`), so the last step before ``X`` reaches
  the model runs the same guard the dataset builder does.
* **Only implemented algorithms run.** Spec 21 names LightGBM first; a configuration that
  asks for something this stage cannot honour says so instead of quietly falling back.

The same module owns the walk-forward *run* (:func:`train_and_evaluate`): train on each
fold's surviving rows, score the fold's validation window, score the baselines on exactly
the same rows (spec 21's always-up / always-range / random / regime-only), aggregate the
metrics across folds, and finally refit one model on every row outside the holdout. That
last model is the artefact; the holdout is only scored when the caller explicitly asks for
it, because a holdout you look at twice is not a holdout.

Nothing here reads a file, a database or a configuration object: the caller passes the
matrix, the split and the policies. LightGBM itself is imported inside the training
function, so the dataset stage never pays for the trainer's dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.constants import DEFAULT_TIMEZONE
from core.ml.baselines import BaselineError, build_baselines
from core.ml.dataset import (
    CODE_COLUMN,
    Dataset,
    assert_no_label_leakage,
)
from core.ml.evaluation import (
    EvaluationPolicy,
    FoldComparison,
    GateVerdict,
    MetricAggregate,
    aggregate_metric,
    compare_to_baselines,
    comparison_metrics,
)
from core.ml.walk_forward import DatasetSplit, Fold, fold_frames, holdout_frame, usable_frame

__all__ = [
    "AGGREGATABLE_METRICS",
    "SUPPORTED_ALGORITHMS",
    "FoldResult",
    "TrainedModel",
    "TrainingError",
    "TrainingPolicy",
    "TrainingReport",
    "class_weights",
    "train_and_evaluate",
    "train_model",
]

#: Algorithms this stage can actually train. Spec 21 names LightGBM first; the fallback
#: and the others are configuration *intent* until their implementation lands.
SUPPORTED_ALGORITHMS: tuple[str, ...] = ("lightgbm",)

#: Feature-importance type handed to LightGBM's own reporting.
IMPORTANCE_TYPE = "gain"


class TrainingError(ValueError):
    """Raised when a model cannot be trained without guessing or leaking."""


@dataclass(frozen=True, slots=True)
class TrainingPolicy:
    """Every number the trainer uses (``config/model.yaml`` ``training`` + seed/algorithm)."""

    algorithm: str = "lightgbm"
    random_seed: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 40
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    class_weight: str = "none"
    early_stopping_rounds: int = 0
    n_jobs: int = 0
    timezone_name: str = DEFAULT_TIMEZONE

    def __post_init__(self) -> None:
        """Refuse a policy that would not train what it claims to train."""
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise TrainingError(
                f"algorithm {self.algorithm!r} is not implemented in this stage; supported: "
                f"{list(SUPPORTED_ALGORITHMS)} (model.algorithm and fallback_algorithm are "
                "configuration intent, never a silent downgrade)"
            )
        if self.n_estimators <= 0:
            raise TrainingError("n_estimators must be > 0")
        if not 0 < self.learning_rate <= 1:
            raise TrainingError("learning_rate must be in (0, 1]")
        if self.num_leaves < 2:
            raise TrainingError("num_leaves must be >= 2")
        if self.max_depth == 0 or self.max_depth < -1:
            raise TrainingError("max_depth must be -1 (unbounded) or a positive depth")
        if self.min_child_samples < 1:
            raise TrainingError("min_child_samples must be >= 1")
        for name in ("feature_fraction", "bagging_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise TrainingError(f"{name} must be in (0, 1]")
        if self.bagging_fraction < 1 and self.bagging_freq <= 0:
            raise TrainingError(
                "bagging_fraction < 1 with bagging_freq = 0 is silently ignored by LightGBM; "
                "set bagging_freq >= 1 or bagging_fraction = 1"
            )
        if self.lambda_l1 < 0 or self.lambda_l2 < 0:
            raise TrainingError("lambda_l1/lambda_l2 must be >= 0")
        if self.class_weight not in ("none", "balanced"):
            raise TrainingError(
                f"class_weight must be 'none' or 'balanced', got {self.class_weight!r}"
            )
        if self.early_stopping_rounds < 0:
            raise TrainingError("early_stopping_rounds must be >= 0 (0 disables it)")
        if self.n_jobs < 0:
            raise TrainingError("n_jobs must be >= 0 (0 uses every core)")
        if self.random_seed < 0:
            raise TrainingError("random_seed must be >= 0")

    @property
    def early_stopping(self) -> bool:
        """True when the trainer stops on the fold's validation window (and says so)."""
        return self.early_stopping_rounds > 0

    @classmethod
    def from_config(cls, config: Any, *, seed: int | None = None) -> TrainingPolicy:
        """Build the policy from a loaded :class:`core.config_loader.AnjaneyaConfig`.

        ``seed`` (the CLI's ``--seed``) overrides ``model.random_seed`` for one run;
        everything else comes from ``model.training``.
        """
        model = config.model
        training = model.training
        return cls(
            algorithm=str(model.algorithm),
            random_seed=int(seed if seed is not None else model.random_seed),
            n_estimators=int(training.n_estimators),
            learning_rate=float(training.learning_rate),
            num_leaves=int(training.num_leaves),
            max_depth=int(training.max_depth),
            min_child_samples=int(training.min_child_samples),
            feature_fraction=float(training.feature_fraction),
            bagging_fraction=float(training.bagging_fraction),
            bagging_freq=int(training.bagging_freq),
            lambda_l1=float(training.lambda_l1),
            lambda_l2=float(training.lambda_l2),
            class_weight=str(training.class_weight),
            early_stopping_rounds=int(training.early_stopping_rounds),
            n_jobs=int(training.n_jobs),
        )

    def to_params(self, *, n_classes: int) -> dict[str, Any]:
        """The dictionary handed to LightGBM.

        Determinism is not a nicety here: two runs of the same command over the same matrix
        have to produce the same model, or a reported number cannot be re-derived. Every
        LightGBM sub-seed is pinned to the policy's single seed and histogram construction is
        forced to be reproducible.
        """
        if n_classes < 2:
            raise TrainingError(f"a multiclass model needs at least two classes, got {n_classes}")
        return {
            "objective": "multiclass",
            "num_class": int(n_classes),
            "metric": "multi_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "seed": self.random_seed,
            "bagging_seed": self.random_seed + 1,
            "feature_fraction_seed": self.random_seed + 2,
            "data_random_seed": self.random_seed + 3,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": self.n_jobs,
            "verbosity": -1,
        }

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every model artefact."""
        return {
            "algorithm": self.algorithm,
            "random_seed": self.random_seed,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "class_weight": self.class_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
            "early_stopping": self.early_stopping,
            "n_jobs": self.n_jobs,
        }


def class_weights(counts: Sequence[float], *, kind: str = "balanced") -> tuple[float, ...]:
    """Per-class weights for an imbalanced label distribution.

    ``balanced`` is scikit-learn's convention: ``n_samples / (n_classes * count)``, so a
    class with half the rows of another carries twice the weight, and the *sum* of the
    weights over the data is unchanged. A class with no rows at all gets no weight - the
    model cannot be told anything about it.
    """
    if kind == "none":
        return tuple(1.0 for _ in counts)
    if kind != "balanced":
        raise TrainingError(f"unknown class-weight kind {kind!r}; expected 'none' or 'balanced'")
    total = float(sum(counts))
    present = [float(count) for count in counts]
    if total <= 0:
        raise TrainingError("class weights need at least one training row")
    width = len(present)
    return tuple(
        0.0 if count <= 0 else total / (width * count) for count in present
    )


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrainedModel:
    """A fitted multiclass model plus everything needed to score it honestly later."""

    algorithm: str
    policy: TrainingPolicy
    classes: tuple[str, ...]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    booster: Any
    train_rows: int
    train_class_counts: dict[str, int]
    categorical_vocabulary: dict[str, tuple[str, ...]] = field(default_factory=dict)
    horizon: int | None = None
    best_iteration: int = 0
    importance: dict[str, float] = field(default_factory=dict)

    # -- scoring -------------------------------------------------------------
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """The class distribution per row (shape ``(rows, n_classes)``, columns = classes)."""
        matrix = self.matrix(frame)
        if matrix.empty:
            return np.zeros((0, len(self.classes)), dtype="float64")
        probabilities = np.asarray(self.booster.predict(matrix), dtype="float64")
        if probabilities.ndim == 1:  # pragma: no cover - LightGBM returns (n, k) for multiclass
            probabilities = probabilities.reshape(-1, len(self.classes))
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The most likely class *index* per row (``classes[argmax]``, spec 21's output)."""
        probabilities = self.predict_proba(frame)
        if not probabilities.size:
            return np.zeros(0, dtype="int64")
        return probabilities.argmax(axis=1).astype("int64")

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The feature matrix as the booster must see it: training order, training vocabulary.

        Categorical columns are re-cast with the **training vocabulary**, so a level the model
        never saw becomes missing rather than a different integer code - the failure that
        silently turns a prediction into noise.
        """
        matrix, _ = feature_matrix(
            frame,
            feature_columns=self.feature_columns,
            categorical_columns=self.categorical_columns,
            vocabulary=self.categorical_vocabulary,
        )
        return matrix

    # -- reporting -----------------------------------------------------------
    def feature_importance(self) -> dict[str, float]:
        """Gain importance, in the trainer's feature order."""
        return dict(self.importance)

    def top_features(self, limit: int = 10) -> list[tuple[str, float]]:
        """The ``limit`` most important features, largest gain first."""
        ranked = sorted(self.importance.items(), key=lambda item: (-item[1], item[0]))
        return ranked[: max(int(limit), 0)]

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary - what the run prints and the metadata stores."""
        return {
            "algorithm": self.algorithm,
            "horizon_minutes": self.horizon,
            "classes": list(self.classes),
            "features": len(self.feature_columns),
            "categorical_features": len(self.categorical_columns),
            "train_rows": self.train_rows,
            "train_class_counts": dict(self.train_class_counts),
            "best_iteration": self.best_iteration,
            "early_stopping": self.policy.early_stopping,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Everything needed to reproduce this model, minus the booster's own bytes."""
        payload = self.describe()
        payload.update(
            {
                "feature_columns": list(self.feature_columns),
                "categorical_columns": list(self.categorical_columns),
                "categorical_vocabulary": {
                    name: list(levels) for name, levels in sorted(self.categorical_vocabulary.items())
                },
                "policy": self.policy.describe(),
                "importance": {
                    name: float(value)
                    for name, value in sorted(
                        self.importance.items(), key=lambda item: (-item[1], item[0])
                    )
                },
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def feature_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    vocabulary: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """The model's feature matrix: training column order, categories cast to the vocabulary.

    ``vocabulary`` is what makes scoring a *new* frame safe: a categorical level the model
    never saw becomes missing instead of a different integer code. When training, the
    vocabulary is derived from the frame itself and returned so the model can keep it.
    """
    categorical = tuple(str(name) for name in categorical_columns)
    if frame.empty:
        empty = pd.DataFrame({name: pd.Series(dtype="float64") for name in feature_columns})
        derived = {}
        if vocabulary:
            derived = {
                name: tuple(str(level) for level in vocabulary[name])
                for name in categorical
                if name in vocabulary
            }
        return empty, derived
    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise TrainingError(
            f"the frame is missing feature column(s) {missing}; a training row must carry every "
            "feature the model is told about"
        )
    wanted = set(categorical)
    matrix = pd.DataFrame(index=frame.index)
    derived: dict[str, tuple[str, ...]] = {}
    for name in feature_columns:
        if name in wanted:
            levels = (
                tuple(str(level) for level in vocabulary[name])
                if vocabulary and name in vocabulary
                else tuple(str(level) for level in pd.unique(frame[name].dropna()))
            )
            matrix[name] = pd.Categorical(frame[name].astype(object), categories=list(levels))
            derived[name] = levels
        else:
            try:
                matrix[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
            except (TypeError, ValueError) as exc:
                raise TrainingError(
                    f"feature {name!r} is not numeric ({exc}); a non-numeric feature must be "
                    "listed in model.dataset.categorical_columns"
                ) from exc
    return matrix, derived


def _import_lightgbm() -> Any:
    """Import LightGBM here rather than at module import, with a message that says why."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - the dependency is declared
        raise TrainingError(
            f"lightgbm is not importable ({exc}); it is a declared dependency of ANJANEYA "
            "('lightgbm>=4.3') - reinstall the environment rather than training without it"
        ) from exc
    return lgb


def train_model(
    frame: pd.DataFrame,
    *,
    policy: TrainingPolicy,
    classes: Sequence[str],
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
    horizon: int | None = None,
    valid_frame: pd.DataFrame | None = None,
) -> TrainedModel:
    """Fit one model on ``frame`` (a training fold, or every row outside the holdout).

    ``valid_frame`` is only used when ``policy.early_stopping_rounds`` is set; the caller is
    the one that knows whether stopping on those rows is acceptable for the claim it is about
    to make.
    """
    names = tuple(str(name) for name in classes)
    features = tuple(str(name) for name in feature_columns)
    categorical = tuple(str(name) for name in categorical_columns)
    if len(names) < 2:
        raise TrainingError(f"a multiclass model needs at least two classes, got {list(names)}")
    if not features:
        raise TrainingError("the model needs at least one feature column")
    if len(set(features)) != len(features):
        raise TrainingError(f"feature columns must be unique, got {list(features)}")
    unknown = sorted(set(categorical) - set(features))
    if unknown:
        raise TrainingError(
            f"categorical column(s) {unknown} are not among the feature columns; the dataset "
            "policy and the trainer disagree"
        )
    assert_no_label_leakage((*features, *categorical))
    if frame.empty:
        raise TrainingError("the training frame has no rows")
    if CODE_COLUMN not in frame.columns:
        raise TrainingError(
            f"the training frame has no {CODE_COLUMN!r} column; build it with "
            "core.ml.dataset.build_dataset rather than from raw label rows"
        )
    codes = frame[CODE_COLUMN]
    if not np.issubdtype(np.asarray(codes).dtype, np.integer):
        raise TrainingError(f"{CODE_COLUMN!r} must hold the integer class codes")
    codes = codes.astype("int64")
    if codes.min() < 0 or codes.max() >= len(names):
        raise TrainingError(
            f"{CODE_COLUMN!r} holds code(s) outside [0, {len(names) - 1}]; the label policy "
            "and the model disagree"
        )
    counts = {name: int((codes == position).sum()) for position, name in enumerate(names)}
    present = [name for name, count in counts.items() if count > 0]
    if len(present) < 2:
        raise TrainingError(
            f"the training frame carries {len(present)} class(es) ({counts}); a classifier cannot "
            "be fitted on one class, and the 'perfect' score from one would be meaningless"
        )

    matrix, vocabulary = feature_matrix(
        frame, feature_columns=features, categorical_columns=categorical
    )
    weights: np.ndarray | None = None
    if policy.class_weight != "none":
        per_class = np.asarray(
            class_weights([counts[name] for name in names], kind=policy.class_weight),
            dtype="float64",
        )
        weights = per_class[codes.to_numpy()]

    lgb = _import_lightgbm()
    train_set = lgb.Dataset(
        matrix,
        label=codes.to_numpy(),
        weight=weights,
        categorical_feature=list(categorical),
        free_raw_data=True,
    )
    valid_sets: list[Any] = []
    callbacks: list[Any] = []
    if policy.early_stopping and valid_frame is not None and not valid_frame.empty:
        valid_matrix, _ = feature_matrix(
            valid_frame,
            feature_columns=features,
            categorical_columns=categorical,
            vocabulary=vocabulary,
        )
        valid_sets = [
            lgb.Dataset(
                valid_matrix,
                label=valid_frame[CODE_COLUMN].astype("int64").to_numpy(),
                reference=train_set,
                free_raw_data=True,
            )
        ]
        callbacks = [lgb.early_stopping(policy.early_stopping_rounds, verbose=False)]
    try:
        booster = lgb.train(
            policy.to_params(n_classes=len(names)),
            train_set,
            num_boost_round=policy.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
    except Exception as exc:  # noqa: BLE001 - a library failure is a reported skip, not a crash
        raise TrainingError(f"training failed ({type(exc).__name__}: {exc})") from exc

    trained_features = tuple(str(name) for name in booster.feature_name())
    if trained_features != features:
        raise TrainingError(
            f"the booster reports feature order {list(trained_features)}, which is not the order "
            f"it was given ({list(features)}); predictions would be silently mis-assigned"
        )
    importance = {
        name: float(value)
        for name, value in zip(
            features, booster.feature_importance(importance_type=IMPORTANCE_TYPE), strict=True
        )
    }
    return TrainedModel(
        algorithm=policy.algorithm,
        policy=policy,
        classes=names,
        feature_columns=features,
        categorical_columns=categorical,
        booster=booster,
        train_rows=int(len(matrix)),
        train_class_counts=counts,
        categorical_vocabulary=vocabulary,
        horizon=horizon,
        best_iteration=int(getattr(booster, "best_iteration", 0) or 0),
        importance=importance,
    )

# ---------------------------------------------------------------------------
# The walk-forward run
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FoldResult:
    """One fold's model, metrics and baselines - or the reason it was skipped."""

    fold: Fold
    status: str
    reason: str | None = None
    comparison: FoldComparison | None = None

    @property
    def trained(self) -> bool:
        """True when this fold produced a model and a scored validation window."""
        return self.status == "TRAINED" and self.comparison is not None

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        if not self.trained or self.comparison is None:
            return f"fold {self.fold.index}: {self.status} - {self.reason}"
        return self.comparison.describe()

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "fold": self.fold.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


@dataclass(slots=True)
class TrainingReport:
    """Everything one horizon's walk-forward run produced, in the order it happened."""

    horizon: int
    timeframe: str
    classes: tuple[str, ...]
    policy: TrainingPolicy
    evaluation: EvaluationPolicy
    split: DatasetSplit
    folds: list[FoldResult] = field(default_factory=list)
    aggregates: dict[str, MetricAggregate] = field(default_factory=dict)
    baseline_aggregates: dict[str, dict[str, MetricAggregate]] = field(default_factory=dict)
    final_model: TrainedModel | None = None
    final_train_rows: int = 0
    holdout: FoldComparison | None = None
    verdict: GateVerdict | None = None
    notes: list[str] = field(default_factory=list)

    # -- shape ---------------------------------------------------------------
    @property
    def trained_folds(self) -> list[FoldResult]:
        """The folds that produced a model."""
        return [result for result in self.folds if result.trained]

    @property
    def skipped_folds(self) -> list[FoldResult]:
        """The folds that could not be trained (each one carries its reason)."""
        return [result for result in self.folds if not result.trained]

    @property
    def ok(self) -> bool:
        """True when at least one fold was trained and a final model exists."""
        return bool(self.trained_folds)

    @property
    def results_label(self) -> str:
        """What the data gate allows these results to be called."""
        if self.verdict is not None:
            return self.verdict.results_label
        return "PRELIMINARY" if any(r.comparison for r in self.folds) else "BELOW_GATE"

    def model_log_loss(self) -> float | None:
        """Mean validation log loss across folds (``None`` when nothing was trained)."""
        aggregate = self.aggregates.get("log_loss")
        if aggregate is None or not aggregate.count:
            return None
        return float(aggregate.mean)

    def best_baseline(self) -> tuple[str, float] | None:
        """The baseline with the lowest mean log loss, and that loss."""
        best: tuple[str, float] | None = None
        for name, metrics in self.baseline_aggregates.items():
            aggregate = metrics.get("log_loss")
            if aggregate is None or not aggregate.count:
                continue
            if best is None or aggregate.mean < best[1]:
                best = (name, float(aggregate.mean))
        return best

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        if not self.folds:
            return f"horizon {self.horizon}m: no folds - {self.split.reason}"
        model = self.model_log_loss()
        text = f"horizon {self.horizon}m: {len(self.trained_folds)}/{len(self.folds)} fold(s) trained"
        if model is None:
            return text + " - every fold was skipped"
        aggregate = self.aggregates["log_loss"]
        text += f", log loss {model:.4f} (sd {aggregate.std:.4f})"
        best = self.best_baseline()
        if best is not None:
            text += f" vs best baseline {best[0]} {best[1]:.4f}"
        if self.holdout is not None:
            text += f", holdout {self.holdout.rows} row(s) scored once"
        return text + f" [{self.results_label}]"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - what the run prints under ``--json`` and stores in metadata."""
        return {
            "horizon_minutes": self.horizon,
            "timeframe": self.timeframe,
            "classes": list(self.classes),
            "results_label": self.results_label,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "training": self.policy.describe(),
            "evaluation": self.evaluation.describe(),
            "split": self.split.to_dict(),
            "folds": [result.to_dict() for result in self.folds],
            "trained_folds": len(self.trained_folds),
            "skipped_folds": len(self.skipped_folds),
            "aggregates": {name: value.to_dict() for name, value in self.aggregates.items()},
            "baseline_aggregates": {
                name: {metric: value.to_dict() for metric, value in metrics.items()}
                for name, metrics in self.baseline_aggregates.items()
            },
            "best_baseline": (
                {"name": self.best_baseline()[0], "log_loss": self.best_baseline()[1]}
                if self.best_baseline()
                else None
            ),
            "final_model": self.final_model.to_manifest() if self.final_model else None,
            "final_train_rows": self.final_train_rows,
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "notes": list(self.notes),
        }



def _score_fold(
    dataset: Dataset,
    fold: Fold,
    *,
    policy: TrainingPolicy,
    evaluation: EvaluationPolicy,
    baseline_names: Sequence[str],
) -> FoldResult:
    """Train on one fold's surviving rows and score its validation window.

    Every refusal (a single-class training frame, a fold with no rows, a baseline that
    cannot be built) becomes a ``SKIPPED`` result carrying its reason: one awkward fold must
    not abort a run that is otherwise measuring something, and a skip that is *counted* is
    never mistaken for a perfect score.
    """
    train_frame, validate_frame = fold_frames(dataset, fold)
    if train_frame.empty or validate_frame.empty:
        return FoldResult(
            fold=fold,
            status="SKIPPED",
            reason=(
                f"the fold has {len(train_frame)} training row(s) and {len(validate_frame)} "
                "validation row(s)"
            ),
        )
    try:
        model = train_model(
            train_frame,
            policy=policy,
            classes=dataset.classes,
            feature_columns=dataset.feature_columns,
            categorical_columns=dataset.categorical_columns,
            horizon=dataset.horizon,
            valid_frame=validate_frame,
        )
        baselines = build_baselines(
            baseline_names,
            train_frame=train_frame,
            classes=dataset.classes,
            seed=policy.random_seed,
        )
        probabilities = model.predict_proba(validate_frame)
    except (TrainingError, BaselineError) as exc:
        return FoldResult(fold=fold, status="SKIPPED", reason=str(exc))

    return FoldResult(
        fold=fold,
        status="TRAINED",
        comparison=compare_to_baselines(
            validate_frame[CODE_COLUMN],
            probabilities,
            {baseline.name: baseline.predict_proba(validate_frame) for baseline in baselines},
            classes=dataset.classes,
            policy=evaluation,
            index=fold.index,
            counts=fold.validate_class_counts,
        ),
    )


#: Metric names the report aggregates across folds. Every one of them is a *scalar*: a
#: calibration curve is per-fold evidence, and its summary numbers (the two calibration
#: errors) are what travel into the across-fold table.
AGGREGATABLE_METRICS: tuple[str, ...] = (
    "log_loss",
    "brier_score",
    "accuracy",
    "expected_calibration_error",
    "confidence_ece",
)


def _aggregate(
    folds: Sequence[FoldResult], *, metrics: Sequence[str], baseline_names: Sequence[str]
) -> tuple[dict[str, MetricAggregate], dict[str, dict[str, MetricAggregate]]]:
    """Aggregate the model's and every baseline's metrics over the folds that trained.

    A metric nobody measured is left out of the table rather than reported as ``0.0`` or
    "not measured": a row that says nothing is a row an operator will misread, and the
    per-fold ``ProbabilityMetrics`` already states exactly what was computed.
    """
    scalar = tuple(name for name in metrics if name in AGGREGATABLE_METRICS)
    comparisons = [result.comparison for result in folds if result.comparison is not None]
    model_aggregates = {
        metric: aggregate_metric(
            metric, [getattr(comparison.model, metric, None) for comparison in comparisons]
        )
        for metric in scalar
    }
    baseline_aggregates: dict[str, dict[str, MetricAggregate]] = {}
    for name in baseline_names:
        key = str(name)
        baseline_aggregates[key] = {
            metric: aggregate_metric(
                metric,
                [
                    getattr(comparison.baselines[key], metric, None)
                    for comparison in comparisons
                    if key in comparison.baselines
                ],
            )
            for metric in scalar
        }
    return (
        {name: value for name, value in model_aggregates.items() if value.count},
        {
            key: {name: value for name, value in metrics_.items() if value.count}
            for key, metrics_ in baseline_aggregates.items()
        },
    )


def train_and_evaluate(
    dataset: Dataset,
    *,
    split: DatasetSplit,
    policy: TrainingPolicy,
    evaluation: EvaluationPolicy,
    baseline_names: Sequence[str] = (),
    include_holdout: bool = False,
    verdict: GateVerdict | None = None,
) -> TrainingReport:
    """Run the whole walk-forward: every fold, its baselines, and one final model.

    The final model is fitted on **every row outside the holdout** - not on the last fold's
    training window - because it is the artefact a later stage will use, and the holdout is
    what remains to check it against. That check only happens when ``include_holdout`` is
    set, and the report then says so: a holdout that has been looked at once is no longer a
    holdout, and nothing here pretends otherwise.
    """
    names = tuple(str(name) for name in baseline_names)
    report = TrainingReport(
        horizon=dataset.horizon,
        timeframe=dataset.timeframe,
        classes=dataset.classes,
        policy=policy,
        evaluation=evaluation,
        split=split,
        verdict=verdict,
    )
    if not split.ok:
        report.notes.append(f"no folds were built: {split.reason}")
        return report

    report.folds = [
        _score_fold(
            dataset,
            fold,
            policy=policy,
            evaluation=evaluation,
            baseline_names=names,
        )
        for fold in split.folds
    ]
    metrics = comparison_metrics(evaluation)
    report.aggregates, report.baseline_aggregates = _aggregate(
        report.folds, metrics=metrics, baseline_names=names
    )

    usable = usable_frame(dataset, split)
    report.final_train_rows = int(len(usable))
    try:
        report.final_model = train_model(
            usable,
            policy=policy,
            classes=dataset.classes,
            feature_columns=dataset.feature_columns,
            categorical_columns=dataset.categorical_columns,
            horizon=dataset.horizon,
        )
    except TrainingError as exc:
        report.notes.append(
            f"the final model could not be trained on the {report.final_train_rows} row(s) "
            f"outside the holdout: {exc}"
        )

    report.holdout = _score_holdout(
        dataset,
        split=split,
        report=report,
        policy=policy,
        evaluation=evaluation,
        baseline_names=names,
        usable=usable,
        include_holdout=include_holdout,
    )

    if policy.early_stopping:
        report.notes.append(
            f"early stopping is on ({policy.early_stopping_rounds} round(s)): the number of "
            "trees was chosen on each fold's own validation window, so those metrics are "
            "optimistic by an unknown amount"
        )
    skipped = report.skipped_folds
    if skipped:
        report.notes.append(
            f"{len(skipped)} fold(s) were skipped: "
            + "; ".join(f"fold {result.fold.index}: {result.reason}" for result in skipped)
        )
    if len(report.trained_folds) < 3:
        report.notes.append(
            f"{len(report.trained_folds)} trained fold(s) - too few for a spread; treat every "
            "across-fold number here as descriptive, not as an estimate of future performance"
        )
    if verdict is not None and verdict.note:
        report.notes.append(verdict.note)
    return report


def _score_holdout(
    dataset: Dataset,
    *,
    split: DatasetSplit,
    report: TrainingReport,
    policy: TrainingPolicy,
    evaluation: EvaluationPolicy,
    baseline_names: Sequence[str],
    usable: pd.DataFrame,
    include_holdout: bool,
) -> FoldComparison | None:
    """Score the final model on the untouched holdout - once, and only when asked."""
    if not include_holdout or report.final_model is None:
        return None
    holdout = holdout_frame(dataset, split)
    if holdout.empty:
        report.notes.append("the holdout has no rows to score")
        return None
    try:
        baselines = build_baselines(
            baseline_names,
            train_frame=usable,
            classes=dataset.classes,
            seed=policy.random_seed,
        )
        probabilities = report.final_model.predict_proba(holdout)
    except (TrainingError, BaselineError) as exc:
        report.notes.append(f"the holdout could not be scored: {exc}")
        return None
    report.notes.append(
        "the holdout was scored once: it is no longer untouched for this dataset, and a "
        "second look at it is no longer a holdout"
    )
    return compare_to_baselines(
        holdout[CODE_COLUMN],
        probabilities,
        {baseline.name: baseline.predict_proba(holdout) for baseline in baselines},
        classes=dataset.classes,
        policy=evaluation,
        index=len(split.folds),
        counts=split.holdout.class_counts if split.holdout else None,
        label="HOLDOUT",
    )

