"""Probability metrics for a model that predicts a distribution (spec 21).

ANJANEYA never predicts BUY/SELL: the model's output is a probability per label class, and
the only way to say whether it is any good is to score those probabilities - against the
outcome, and against baselines that use no features at all.

Three decisions shape this module:

1. **Probabilities are clipped, then renormalised, and the clipping is counted.** A
   one-hot prediction that is wrong costs infinite log loss; clipping to
   :data:`PROBABILITY_EPSILON` makes it finite, and the number of rows that needed it is
   reported beside the score - a log loss of 20 that is really "the model said never" must
   not read like a confident, finely-measured number.
2. **A metric that was not asked for is ``None``, not ``0.0``.** "Not measured" and
   "measured zero" are different claims, and a report that confuses them is lying.
3. **The baseline comparison is a delta, not a verdict.** Each fold reports its model
   metrics, the same metrics for every baseline, and the log-loss difference; whether that
   difference means anything is limited by how many folds and how many rows there are,
   which is why every comparison carries a ``PRELIMINARY``/``MEASURED`` label and the fold
   count travels with it.

Spec 4.3's data gate is translated into a label here as well (:func:`gate_verdict`), so
"this is a first ML model" and "these results may be called production" are decided by the
same rule everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "COMPARISON_METRIC",
    "PROBABILITY_EPSILON",
    "SUPPORTED_PROBABILITY_METRICS",
    "CalibrationBin",
    "CalibrationCurve",
    "EvaluationError",
    "EvaluationPolicy",
    "FoldComparison",
    "GateVerdict",
    "MetricAggregate",
    "ProbabilityMetrics",
    "accuracy",
    "aggregate_metric",
    "brier_score",
    "calibration_curve",
    "clip_probabilities",
    "compare_to_baselines",
    "comparison_metrics",
    "evaluate_predictions",
    "gate_verdict",
    "log_loss",
]

#: Smallest probability a metric will ever see. ``-log(1e-9) ~= 20.7``, which is exactly
#: the point: a wrong one-hot is *bad*, not *undefined*.
PROBABILITY_EPSILON = 1e-9

#: Metric names ``model.evaluation.probability_metrics`` may contain.
SUPPORTED_PROBABILITY_METRICS = ("log_loss", "brier_score", "calibration_curve", "accuracy")

#: Default number of equal-width bins in a reliability curve.
DEFAULT_CALIBRATION_BINS = 10


class EvaluationError(ValueError):
    """Raised when predictions and outcomes cannot be scored against each other."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """How predictions are scored (``config/model.yaml`` ``evaluation``)."""

    metrics: tuple[str, ...]
    preliminary_min_samples: int
    bins: int = DEFAULT_CALIBRATION_BINS

    def __post_init__(self) -> None:
        """Refuse a metric name that has no implementation."""
        unknown = sorted({str(name) for name in self.metrics} - set(SUPPORTED_PROBABILITY_METRICS))
        if unknown:
            raise EvaluationError(
                f"unsupported probability metric(s) {unknown}; supported: "
                f"{list(SUPPORTED_PROBABILITY_METRICS)}"
            )
        if self.preliminary_min_samples <= 0:
            raise EvaluationError("preliminary_min_samples must be a positive number of samples")
        if self.bins < 2:
            raise EvaluationError("a calibration curve needs at least two bins")

    @classmethod
    def from_config(cls, config: Any) -> EvaluationPolicy:
        """Build the policy from a loaded :class:`core.config_loader.AnjaneyaConfig`."""
        evaluation = config.model.evaluation
        return cls(
            metrics=tuple(str(name) for name in evaluation.probability_metrics),
            preliminary_min_samples=int(evaluation.preliminary_min_samples),
        )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "probability_metrics": list(self.metrics),
            "preliminary_min_samples": self.preliminary_min_samples,
            "calibration_bins": self.bins,
        }


# ---------------------------------------------------------------------------
# Arrays in, arrays out
# ---------------------------------------------------------------------------
def _codes(outcomes: Any, *, classes: Sequence[str]) -> np.ndarray:
    """The observed classes as integer codes, validated against ``classes``."""
    values = np.asarray(outcomes)
    if values.ndim != 1:
        raise EvaluationError(f"outcome codes must be one-dimensional, got shape {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise EvaluationError(
            "outcomes must be the integer class codes of a dataset (``label_code``), not strings"
        )
    codes = values.astype("int64", copy=False)
    if codes.size and (codes.min() < 0 or codes.max() >= len(classes)):
        raise EvaluationError(
            f"outcome code(s) outside [0, {len(classes) - 1}] for classes {list(classes)}"
        )
    return codes


def _probabilities(probabilities: Any, *, rows: int, classes: Sequence[str]) -> np.ndarray:
    """The predicted distribution, validated against the row count and the class count."""
    array = np.asarray(probabilities, dtype="float64")
    if array.ndim != 2:
        raise EvaluationError(f"probabilities must be two-dimensional, got shape {array.shape}")
    if array.shape != (rows, len(classes)):
        raise EvaluationError(
            f"probabilities have shape {array.shape}; expected {(rows, len(classes))} "
            f"({rows} row(s) x {len(classes)} class(es))"
        )
    if array.size and not np.isfinite(array).all():
        raise EvaluationError("probabilities must be finite; NaN/inf cannot be scored")
    return array


def _one_hot(codes: np.ndarray, *, n_classes: int) -> np.ndarray:
    """The outcome as a one-hot matrix (``codes`` must already be validated)."""
    hot = np.zeros((codes.size, n_classes), dtype="float64")
    if codes.size:
        hot[np.arange(codes.size), codes] = 1.0
    return hot


def clip_probabilities(probabilities: Any, *, epsilon: float = PROBABILITY_EPSILON) -> np.ndarray:
    """Clamp every probability into ``[epsilon, 1 - epsilon]`` and renormalise each row.

    Reapplying the sum-to-one constraint keeps the result a distribution, so one clipped
    vector can feed *every* metric without a second convention. A one-dimensional array is
    refused rather than treated as a single row: guessing which axis a caller meant is how
    a metric ends up computed over the wrong thing.
    """
    array = np.asarray(probabilities, dtype="float64")
    if array.ndim != 2:
        raise EvaluationError(
            f"probabilities must be two-dimensional (rows x classes), got shape {array.shape}"
        )
    clipped = np.clip(array, epsilon, 1.0 - epsilon)
    totals = clipped.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(totals > 0, clipped / totals, 1.0 / clipped.shape[1])
    return scaled


def _clipped_rows(probabilities: np.ndarray, *, epsilon: float = PROBABILITY_EPSILON) -> int:
    """How many rows contain a probability outside ``[epsilon, 1 - epsilon]``."""
    array = np.asarray(probabilities, dtype="float64")
    if not array.size:
        return 0
    return int(((array < epsilon) | (array > 1.0 - epsilon)).any(axis=1).sum())


def log_loss(
    outcomes: Any,
    probabilities: Any,
    *,
    classes: Sequence[str],
    epsilon: float = PROBABILITY_EPSILON,
) -> float:
    """Mean negative log-likelihood of the observed class (an empty set scores ``0.0``)."""
    codes = _codes(outcomes, classes=classes)
    array = _probabilities(probabilities, rows=codes.size, classes=classes)
    if not codes.size:
        return 0.0
    clipped = clip_probabilities(array, epsilon=epsilon)
    picked = clipped[np.arange(codes.size), codes]
    return float(-np.log(picked).mean())


def brier_score(outcomes: Any, probabilities: Any, *, classes: Sequence[str]) -> float:
    """Mean squared error over every (row, class) cell - the multiclass Brier score.

    Taking the mean over cells (rather than the sum) keeps the value comparable between
    datasets with different numbers of classes; the per-class breakdown lives in
    :func:`calibration_curve`.
    """
    codes = _codes(outcomes, classes=classes)
    array = _probabilities(probabilities, rows=codes.size, classes=classes)
    if not codes.size:
        return 0.0
    hot = _one_hot(codes, n_classes=len(classes))
    return float(np.square(array - hot).mean())


def accuracy(outcomes: Any, probabilities: Any, *, classes: Sequence[str]) -> float:
    """Share of rows whose most likely class is the observed one.

    Not a probability metric, but a model whose distribution is well calibrated and whose
    argmax never matches the outcome is still useless to a decision layer, so the number
    travels with the others.
    """
    codes = _codes(outcomes, classes=classes)
    array = _probabilities(probabilities, rows=codes.size, classes=classes)
    if not codes.size:
        return 0.0
    return float((array.argmax(axis=1) == codes).mean())

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One bin of a reliability curve: what was predicted, and what happened."""

    index: int
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_frequency: float | None

    @property
    def gap(self) -> float | None:
        """Signed difference ``predicted - observed`` (``None`` for an empty bin)."""
        if self.mean_probability is None or self.observed_frequency is None:
            return None
        return self.mean_probability - self.observed_frequency

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "bin": self.index,
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_probability": self.mean_probability,
            "observed_frequency": self.observed_frequency,
            "gap": self.gap,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCurve:
    """Reliability curves for one set of predictions: per class, and by confidence."""

    classes: tuple[str, ...]
    bins: int
    per_class: dict[str, tuple[CalibrationBin, ...]]
    confidence: tuple[CalibrationBin, ...]
    class_ece: dict[str, float]
    confidence_ece: float

    @property
    def macro_ece(self) -> float:
        """Mean expected calibration error over the classes."""
        if not self.class_ece:
            return 0.0
        return float(np.mean(list(self.class_ece.values())))

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        text = (
            f"{self.bins}-bin calibration: macro ECE {self.macro_ece:.4f}, "
            f"confidence ECE {self.confidence_ece:.4f}"
        )
        if self.class_ece:
            worst = max(self.class_ece.items(), key=lambda item: item[1])
            text += f" (worst class {worst[0]} {worst[1]:.4f})"
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (every bin, so a report can plot it)."""
        return {
            "bins": self.bins,
            "macro_ece": self.macro_ece,
            "confidence_ece": self.confidence_ece,
            "class_ece": dict(self.class_ece),
            "confidence": [item.to_dict() for item in self.confidence],
            "per_class": {
                name: [item.to_dict() for item in class_bins]
                for name, class_bins in self.per_class.items()
            },
        }


def _curve(
    values: np.ndarray, hits: np.ndarray, *, edges: np.ndarray
) -> tuple[tuple[CalibrationBin, ...], float]:
    """Bin ``values`` against 0/1 ``hits``, returning the bins and the expected error."""
    index = np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)
    total = values.size
    bins: list[CalibrationBin] = []
    ece = 0.0
    for position in range(len(edges) - 1):
        mask = index == position
        count = int(mask.sum())
        mean_probability = float(values[mask].mean()) if count else None
        observed = float(hits[mask].mean()) if count else None
        if count and mean_probability is not None and observed is not None:
            ece += (count / total) * abs(mean_probability - observed)
        bins.append(
            CalibrationBin(
                index=position,
                lower=float(edges[position]),
                upper=float(edges[position + 1]),
                count=count,
                mean_probability=mean_probability,
                observed_frequency=observed,
            )
        )
    return tuple(bins), float(ece)


def calibration_curve(
    outcomes: Any,
    probabilities: Any,
    *,
    classes: Sequence[str],
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationCurve:
    """Reliability curves: one per class (one-vs-rest), plus one by predicted confidence.

    The per-class curve answers "when the model says 0.7 for REVERSAL, does REVERSAL
    happen 70% of the time?"; the confidence curve asks the same question about the
    model's *own* certainty (max probability vs hit rate), which is the curve the decision
    layer (Phase 24) will actually consume.
    """
    if bins < 2:
        raise EvaluationError("a calibration curve needs at least two bins")
    codes = _codes(outcomes, classes=classes)
    array = _probabilities(probabilities, rows=codes.size, classes=classes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    per_class: dict[str, tuple[CalibrationBin, ...]] = {}
    class_ece: dict[str, float] = {}
    for position, name in enumerate(classes):
        curve, ece = _curve(array[:, position], (codes == position).astype("float64"), edges=edges)
        per_class[str(name)] = curve
        class_ece[str(name)] = ece

    if codes.size:
        confidence = array.max(axis=1)
        hit = (array.argmax(axis=1) == codes).astype("float64")
    else:
        confidence, hit = np.zeros(0, dtype="float64"), np.zeros(0, dtype="float64")
    confidence_bins, confidence_ece = _curve(np.clip(confidence, 0.0, 1.0), hit, edges=edges)
    return CalibrationCurve(
        classes=tuple(str(name) for name in classes),
        bins=int(bins),
        per_class=per_class,
        confidence=confidence_bins,
        class_ece=class_ece,
        confidence_ece=confidence_ece,
    )


# ---------------------------------------------------------------------------
# One scored prediction set
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Every metric measured for one set of predictions (unmeasured ones stay ``None``)."""

    rows: int
    classes: tuple[str, ...]
    metrics: tuple[str, ...]
    log_loss: float | None = None
    brier_score: float | None = None
    accuracy: float | None = None
    expected_calibration_error: float | None = None
    confidence_ece: float | None = None
    clipped_predictions: int = 0
    class_counts: dict[str, int] | None = None
    calibration: CalibrationCurve | None = None

    @property
    def measured(self) -> tuple[str, ...]:
        """The metric names that actually carry a number (``None`` means "not measured")."""
        return tuple(name for name in self.metrics if getattr(self, name, None) is not None)

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        parts: list[str] = [f"{self.rows} row(s)"]
        for name in ("log_loss", "brier_score", "accuracy", "expected_calibration_error"):
            value = getattr(self, name, None)
            if value is not None:
                parts.append(f"{name.replace('_', ' ')} {value:.4f}")
        if self.clipped_predictions:
            parts.append(f"{self.clipped_predictions} row(s) clipped at {PROBABILITY_EPSILON:g}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (``null`` means "not measured", never "zero")."""
        return {
            "rows": self.rows,
            "classes": list(self.classes),
            "metrics": list(self.metrics),
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "confidence_ece": self.confidence_ece,
            "clipped_predictions": self.clipped_predictions,
            "class_counts": dict(self.class_counts or {}),
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }




def evaluate_predictions(
    outcomes: Any,
    probabilities: Any,
    *,
    classes: Sequence[str],
    policy: EvaluationPolicy,
    metrics: Sequence[str] | None = None,
    counts: Mapping[str, int] | None = None,
) -> ProbabilityMetrics:
    """Score one set of predictions exactly as the policy asks (and no more)."""
    requested = tuple(str(name) for name in (metrics if metrics is not None else policy.metrics))
    unknown = sorted(set(requested) - set(SUPPORTED_PROBABILITY_METRICS))
    if unknown:
        raise EvaluationError(
            f"unsupported probability metric(s) {unknown}; supported: "
            f"{list(SUPPORTED_PROBABILITY_METRICS)}"
        )
    codes = _codes(outcomes, classes=classes)
    array = _probabilities(probabilities, rows=codes.size, classes=classes)
    curve = (
        calibration_curve(codes, array, classes=classes, bins=policy.bins)
        if "calibration_curve" in requested
        else None
    )
    if counts is not None:
        observed = dict(counts)
    elif codes.size:
        observed = {
            str(name): int((codes == position).sum()) for position, name in enumerate(classes)
        }
    else:
        observed = {}
    return ProbabilityMetrics(
        rows=int(codes.size),
        classes=tuple(str(name) for name in classes),
        metrics=requested,
        log_loss=(log_loss(codes, array, classes=classes) if "log_loss" in requested else None),
        brier_score=(
            brier_score(codes, array, classes=classes) if "brier_score" in requested else None
        ),
        accuracy=accuracy(codes, array, classes=classes) if "accuracy" in requested else None,
        expected_calibration_error=curve.macro_ece if curve else None,
        confidence_ece=curve.confidence_ece if curve else None,
        clipped_predictions=_clipped_rows(array),
        class_counts=observed,
        calibration=curve,
    )


# ---------------------------------------------------------------------------
# Across folds: an aggregate, never a single number
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """One metric across folds: the spread matters more than the mean."""

    metric: str
    values: tuple[float, ...]

    @property
    def count(self) -> int:
        """How many folds contributed a number."""
        return len(self.values)

    @property
    def mean(self) -> float:
        """Mean across folds (``0.0`` when there are none)."""
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def minimum(self) -> float:
        """Best fold - every metric here is "lower is better"."""
        return float(min(self.values)) if self.values else 0.0

    @property
    def maximum(self) -> float:
        """Worst fold."""
        return float(max(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        """Sample standard deviation across folds (``0.0`` for a single fold)."""
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        if not self.values:
            return f"{self.metric}: not measured"
        return (
            f"{self.metric}: mean {self.mean:.4f} over {self.count} fold(s), "
            f"best {self.minimum:.4f}, worst {self.maximum:.4f}, sd {self.std:.4f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (raw folds kept, so an average hides nothing)."""
        return {
            "metric": self.metric,
            "count": self.count,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "std": self.std,
            "values": list(self.values),
        }


def aggregate_metric(metric: str, values: Sequence[float | None]) -> MetricAggregate:
    """Aggregate one metric over the folds that measured it, ignoring the ones that did not."""
    return MetricAggregate(
        metric=str(metric), values=tuple(float(value) for value in values if value is not None)
    )

# ---------------------------------------------------------------------------
# A model against the baselines, in the same fold
# ---------------------------------------------------------------------------
#: The metric the model-vs-baseline comparison is made on. Log loss is the one sensitive to
#: the *whole* distribution (a confident wrong answer is punished hard), which is exactly
#: the failure mode a probability model must not have.
COMPARISON_METRIC = "log_loss"


@dataclass(frozen=True, slots=True)
class FoldComparison:
    """The model and every baseline, scored on the same rows of the same fold."""

    index: int
    rows: int
    label: str
    model: ProbabilityMetrics
    baselines: dict[str, ProbabilityMetrics]
    deltas: dict[str, float | None]
    best_baseline: str | None
    beats: tuple[str, ...]

    @property
    def beats_all_baselines(self) -> bool:
        """True when the model's log loss is lower than *every* baseline's."""
        return bool(self.baselines) and len(self.beats) == len(self.baselines)

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        text = f"fold {self.index}: {self.rows} row(s) [{self.label}] - {self.model.describe()}"
        if self.baselines:
            best = f", best baseline {self.best_baseline}" if self.best_baseline else ""
            text += f"; beat {len(self.beats)}/{len(self.baselines)} baseline(s){best}"
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "fold": self.index,
            "rows": self.rows,
            "label": self.label,
            "model": self.model.to_dict(),
            "baselines": {name: value.to_dict() for name, value in self.baselines.items()},
            "log_loss_delta_vs_baseline": dict(self.deltas),
            "best_baseline": self.best_baseline,
            "beats_baselines": list(self.beats),
            "beats_all_baselines": self.beats_all_baselines,
        }


def comparison_metrics(policy: EvaluationPolicy) -> tuple[str, ...]:
    """The metrics a model-vs-baseline comparison always measures, plus the policy's own.

    A delta on a single metric is not a comparison: a model can win on log loss while its
    argmax never matches anything. Log loss, Brier score and accuracy are therefore always
    measured on both sides, and ``model.evaluation.probability_metrics`` adds to that set.
    """
    return tuple(dict.fromkeys(("log_loss", "brier_score", "accuracy", *policy.metrics)))


def compare_to_baselines(
    outcomes: Any,
    model_probabilities: Any,
    baseline_probabilities: Mapping[str, Any],
    *,
    classes: Sequence[str],
    policy: EvaluationPolicy,
    index: int = 0,
    counts: Mapping[str, int] | None = None,
    label: str | None = None,
) -> FoldComparison:
    """Score the model and every baseline on one fold, and compare log losses.

    A positive delta means the *baseline* was worse, i.e. the model added something. The
    label says how much that is worth: below ``preliminary_min_samples`` rows the
    comparison is ``PRELIMINARY`` no matter how large the delta looks, and a caller scoring
    a holdout passes ``label`` so the row cannot be mistaken for a fold.
    """
    comparison_metrics_ = comparison_metrics(policy)
    model_metrics = evaluate_predictions(
        outcomes,
        model_probabilities,
        classes=classes,
        policy=policy,
        metrics=comparison_metrics_,
        counts=counts,
    )
    baselines = {
        str(name): evaluate_predictions(
            outcomes,
            values,
            classes=classes,
            policy=policy,
            metrics=comparison_metrics_,
            counts=counts,
        )
        for name, values in baseline_probabilities.items()
    }
    deltas: dict[str, float | None] = {}
    beats: list[str] = []
    for name, metrics in baselines.items():
        if metrics.log_loss is None or model_metrics.log_loss is None:
            deltas[name] = None
            continue
        delta = metrics.log_loss - model_metrics.log_loss
        deltas[name] = float(delta)
        if delta > 0:
            beats.append(name)
    measured = {
        name: metrics.log_loss
        for name, metrics in baselines.items()
        if metrics.log_loss is not None
    }
    best_baseline = min(measured, key=lambda name: measured[name]) if measured else None
    return FoldComparison(
        index=int(index),
        rows=model_metrics.rows,
        label=(
            label
            if label is not None
            else ("PRELIMINARY" if model_metrics.rows < policy.preliminary_min_samples else "MEASURED")
        ),
        model=model_metrics,
        baselines=baselines,
        deltas=deltas,
        best_baseline=best_baseline,
        beats=tuple(beats),
    )




# ---------------------------------------------------------------------------
# Spec 4.3's gate, as a label rather than a vibe
# ---------------------------------------------------------------------------
#: What the gate permits a *model* run to call its own results.
_STAGE_LABELS: dict[str, str] = {
    "INSUFFICIENT": "BELOW_GATE",
    "EXPLORATORY": "EXPLORATORY",
    "ML_TRAINING": "PRELIMINARY",
    "MEANINGFUL_BACKTEST": "PRELIMINARY",
    "SEASONALITY": "PRELIMINARY",
}

#: Printed with every verdict: a fold metric is not a backtest.
_GATE_NOTE = (
    "a model metric is not a backtest - spec 4.3 makes results statistically meaningful "
    "once a meaningful backtest can be run (model.data_gates.meaningful_backtest_min_days)"
)


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """What spec 4.3's stage allows this run to claim."""

    stage: str
    results_label: str
    production_model: bool
    refused: bool
    reason: str
    note: str | None = None

    def describe(self) -> str:
        """One line an operator can read out of a log."""
        text = f"{self.stage}: results are {self.results_label}"
        if self.refused:
            text += " - refused"
        return f"{text} ({self.reason})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "stage": self.stage,
            "results_label": self.results_label,
            "production_model": self.production_model,
            "refused": self.refused,
            "reason": self.reason,
            "note": self.note,
        }


def gate_verdict(stage: str, *, days: int | None = None, forced: bool = False) -> GateVerdict:
    """Translate a spec 4.3 stage into what this run may claim (and whether it may run).

    Below ``ML_TRAINING`` a *production* model may not be trained. Below ``EXPLORATORY``
    there is too little data to conclude anything at all, so the run refuses unless the
    operator explicitly forces it - and a forced run never loses the ``BELOW_GATE`` label.
    """
    if stage not in _STAGE_LABELS:
        raise EvaluationError(
            f"unknown data-gate stage {stage!r}; expected one of {sorted(_STAGE_LABELS)}"
        )
    where = f"{days} captured day(s)" if days is not None else "the captured data"
    if stage == "INSUFFICIENT":
        return GateVerdict(
            stage=stage,
            results_label=_STAGE_LABELS[stage],
            production_model=False,
            refused=not forced,
            reason=(
                f"{where} is below the exploratory gate: there is too little data to "
                "conclude anything"
            ),
            note=_GATE_NOTE,
        )
    if stage == "EXPLORATORY":
        return GateVerdict(
            stage=stage,
            results_label=_STAGE_LABELS[stage],
            production_model=False,
            refused=False,
            reason=f"{where}: exploratory analysis only - this is not a production model",
            note=_GATE_NOTE,
        )
    return GateVerdict(
        stage=stage,
        results_label=_STAGE_LABELS[stage],
        production_model=True,
        refused=False,
        reason=f"{where}: a production model may be trained; results stay PRELIMINARY",
        note=_GATE_NOTE,
    )
