"""Unit tests for the probability metrics (``core/ml/evaluation.py``, spec 21).

Metrics are the part of a research engine that gets quoted, so these tests are about
*arithmetic that can be checked by hand*: a perfect prediction scores zero, a uniform
prediction scores ``log(k)``, a confidently wrong one-hot scores the clipping floor and is
counted as clipped, a calibration curve's empty bins are empty rather than zero, and an
aggregate keeps the raw values it averaged.

The last group pins the two honesty rules of the module: a metric nobody asked for is
``None`` (never ``0.0``), and a comparison below ``preliminary_min_samples`` is labelled
``PRELIMINARY`` no matter how large the delta looks.
"""

from __future__ import annotations

from math import log

import numpy as np
import pytest

from core.ml import (
    PROBABILITY_EPSILON,
    EvaluationError,
    EvaluationPolicy,
    accuracy,
    aggregate_metric,
    brier_score,
    calibration_curve,
    clip_probabilities,
    compare_to_baselines,
    comparison_metrics,
    evaluate_predictions,
    gate_verdict,
    log_loss,
)
from tests.support.ai import CLASSES

CLASS_COUNT = len(CLASSES)
CODES = np.arange(CLASS_COUNT, dtype="int64")


def one_hot(codes: np.ndarray) -> np.ndarray:
    """A perfect distribution for ``codes``."""
    probabilities = np.zeros((len(codes), CLASS_COUNT))
    probabilities[np.arange(len(codes)), codes] = 1.0
    return probabilities


def policy(**overrides) -> EvaluationPolicy:
    """The shipped evaluation policy, optionally narrowed."""
    values = {
        "metrics": ("log_loss", "brier_score", "calibration_curve", "accuracy"),
        "preliminary_min_samples": 30,
    }
    values.update(overrides)
    return EvaluationPolicy(**values)


# ---------------------------------------------------------------------------
# The three scalar metrics
# ---------------------------------------------------------------------------
def test_a_perfect_prediction_scores_zero() -> None:
    assert log_loss(CODES, one_hot(CODES), classes=CLASSES) == pytest.approx(1e-9, abs=1e-6)
    assert brier_score(CODES, one_hot(CODES), classes=CLASSES) == 0.0
    assert accuracy(CODES, one_hot(CODES), classes=CLASSES) == 1.0


def test_a_uniform_prediction_scores_log_of_the_class_count() -> None:
    uniform = np.full((CLASS_COUNT, CLASS_COUNT), 1.0 / CLASS_COUNT)
    assert log_loss(CODES, uniform, classes=CLASSES) == pytest.approx(log(CLASS_COUNT))
    # Every row ties, so argmax picks class 0: the observed code 0 is the only hit.
    assert accuracy(CODES, uniform, classes=CLASSES) == pytest.approx(1.0 / CLASS_COUNT)


def test_a_confidently_wrong_prediction_is_clipped_and_finite() -> None:
    wrong = one_hot(np.roll(CODES, 1))
    probabilities = clip_probabilities(wrong[:1])[0]
    assert probabilities.sum() == pytest.approx(1.0)
    # The observed class of row 0 is code 0, and the prediction put ~0 on it.
    assert probabilities[0] == pytest.approx(PROBABILITY_EPSILON / (1 + 2 * PROBABILITY_EPSILON))
    assert log_loss(CODES, wrong, classes=CLASSES) == pytest.approx(
        -log(PROBABILITY_EPSILON / (1 + 2 * PROBABILITY_EPSILON))
    )


def test_clipping_refuses_a_one_dimensional_array() -> None:
    with pytest.raises(EvaluationError, match="two-dimensional"):
        clip_probabilities(np.array([0.5, 0.5]))


def test_clipping_does_not_move_a_reasonable_prediction() -> None:
    probabilities = np.full((2, CLASS_COUNT), 0.25)
    assert clip_probabilities(probabilities) == pytest.approx(probabilities)


def test_an_empty_prediction_set_scores_zero() -> None:
    empty = np.zeros((0, CLASS_COUNT))
    assert log_loss(np.zeros(0, dtype="int64"), empty, classes=CLASSES) == 0.0
    assert brier_score(np.zeros(0, dtype="int64"), empty, classes=CLASSES) == 0.0
    assert accuracy(np.zeros(0, dtype="int64"), empty, classes=CLASSES) == 0.0


def test_the_wrong_shape_is_refused() -> None:
    with pytest.raises(EvaluationError, match="expected"):
        log_loss(CODES, np.zeros((3, CLASS_COUNT)), classes=CLASSES)


def test_one_dimensional_probabilities_are_refused() -> None:
    with pytest.raises(EvaluationError, match="two-dimensional"):
        log_loss(CODES, np.zeros(CLASS_COUNT), classes=CLASSES)


def test_non_finite_probabilities_are_refused() -> None:
    probabilities = np.full((CLASS_COUNT, CLASS_COUNT), np.nan)
    with pytest.raises(EvaluationError, match="finite"):
        log_loss(CODES, probabilities, classes=CLASSES)


def test_class_labels_instead_of_codes_are_refused() -> None:
    with pytest.raises(EvaluationError, match="``label_code``"):
        log_loss(np.array(list(CLASSES)), one_hot(CODES), classes=CLASSES)


def test_a_code_outside_the_class_list_is_refused() -> None:
    with pytest.raises(EvaluationError, match="outside"):
        log_loss(np.array([9]), np.full((1, CLASS_COUNT), 0.25), classes=CLASSES)


def test_two_dimensional_codes_are_refused() -> None:
    with pytest.raises(EvaluationError, match="one-dimensional"):
        log_loss(np.zeros((2, 2), dtype="int64"), np.full((2, CLASS_COUNT), 0.25), classes=CLASSES)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_a_uniform_prediction_is_perfectly_calibrated() -> None:
    uniform = np.full((40, CLASS_COUNT), 1.0 / CLASS_COUNT)
    curve = calibration_curve(CODES.repeat(10), uniform, classes=CLASSES, bins=10)
    assert curve.macro_ece == pytest.approx(0.0, abs=1e-9)
    assert curve.confidence_ece == pytest.approx(0.0, abs=1e-9)
    assert curve.bins == 10
    assert set(curve.per_class) == set(CLASSES)
    assert len(curve.confidence) == 10


def test_a_miscalibrated_confidence_curve_reports_the_gap() -> None:
    # Every row says "90% sure of class 0" and every row really is class 0: the
    # per-confidence bin is under-confident, and the error is the gap itself.
    probabilities = np.zeros((20, CLASS_COUNT))
    probabilities[:, 0] = 0.9
    probabilities[:, 1] = 0.1
    codes = np.zeros(20, dtype="int64")
    curve = calibration_curve(codes, probabilities, classes=CLASSES, bins=10)
    top = curve.confidence[-1]
    assert top.count == 20
    assert top.mean_probability == pytest.approx(0.9)
    assert top.observed_frequency == pytest.approx(1.0)
    assert top.gap == pytest.approx(-0.1)
    assert curve.confidence_ece == pytest.approx(0.1)
    assert curve.describe().startswith("10-bin calibration")


def test_empty_calibration_bins_are_empty_not_zero() -> None:
    uniform = np.full((8, CLASS_COUNT), 1.0 / CLASS_COUNT)
    curve = calibration_curve(CODES.repeat(2), uniform, classes=CLASSES, bins=10)
    empty = [item for item in curve.confidence if item.count == 0]
    assert empty, "the fixture cannot fill every confidence bin"
    for item in empty:
        assert item.mean_probability is None
        assert item.observed_frequency is None
        assert item.gap is None
        assert item.to_dict()["mean_probability"] is None


def test_a_curve_needs_at_least_two_bins() -> None:
    with pytest.raises(EvaluationError, match="two bins"):
        calibration_curve(CODES, one_hot(CODES), classes=CLASSES, bins=1)


def test_the_curve_serialises_every_bin() -> None:
    curve = calibration_curve(CODES, one_hot(CODES), classes=CLASSES, bins=5)
    payload = curve.to_dict()
    assert payload["bins"] == 5
    assert len(payload["confidence"]) == 5
    assert len(payload["per_class"][CLASSES[0]]) == 5


# ---------------------------------------------------------------------------
# One scored prediction set
# ---------------------------------------------------------------------------
def test_only_the_requested_metrics_are_computed() -> None:
    metrics = evaluate_predictions(
        CODES, one_hot(CODES), classes=CLASSES, policy=policy(metrics=("log_loss",))
    )
    assert metrics.log_loss is not None
    assert metrics.brier_score is None
    assert metrics.accuracy is None
    assert metrics.expected_calibration_error is None
    assert metrics.calibration is None
    assert metrics.measured == ("log_loss",)


def test_the_calibration_curve_is_measured_when_asked() -> None:
    metrics = evaluate_predictions(CODES, one_hot(CODES), classes=CLASSES, policy=policy())
    assert metrics.calibration is not None
    assert metrics.expected_calibration_error == pytest.approx(metrics.calibration.macro_ece)
    assert metrics.confidence_ece is not None


def test_an_unsupported_metric_is_refused() -> None:
    with pytest.raises(EvaluationError, match="unsupported probability metric"):
        evaluate_predictions(
            CODES, one_hot(CODES), classes=CLASSES, policy=policy(metrics=("roc_auc",))
        )


def test_the_policy_refuses_a_metric_that_does_not_exist() -> None:
    with pytest.raises(EvaluationError, match="unsupported"):
        EvaluationPolicy(metrics=("log_loss", "sharpe"), preliminary_min_samples=10)


def test_the_policy_refuses_unusable_numbers() -> None:
    with pytest.raises(EvaluationError, match="preliminary_min_samples"):
        EvaluationPolicy(metrics=("log_loss",), preliminary_min_samples=0)
    with pytest.raises(EvaluationError, match="two bins"):
        EvaluationPolicy(metrics=("log_loss",), preliminary_min_samples=10, bins=1)


def test_the_policy_comes_from_the_shipped_config(config) -> None:
    policy_ = EvaluationPolicy.from_config(config)
    assert policy_.metrics == tuple(config.model.evaluation.probability_metrics)
    assert policy_.preliminary_min_samples == config.model.evaluation.preliminary_min_samples
    assert policy_.describe()["calibration_bins"] == 10


def test_clipping_is_counted_beside_the_score() -> None:
    wrong = one_hot(np.roll(CODES, 1))
    metrics = evaluate_predictions(CODES, wrong, classes=CLASSES, policy=policy())
    assert metrics.clipped_predictions == len(CODES)
    assert str(metrics.clipped_predictions) in metrics.describe()



# ---------------------------------------------------------------------------
# Across folds
# ---------------------------------------------------------------------------
def test_an_aggregate_keeps_the_values_it_averaged() -> None:
    aggregate = aggregate_metric("log_loss", [1.0, 2.0, 3.0])
    assert aggregate.count == 3
    assert aggregate.mean == pytest.approx(2.0)
    assert aggregate.minimum == 1.0
    assert aggregate.maximum == 3.0
    assert aggregate.std == pytest.approx(1.0)
    assert "sd" in aggregate.describe()
    assert aggregate.to_dict()["values"] == [1.0, 2.0, 3.0]


def test_an_aggregate_ignores_metrics_that_were_not_measured() -> None:
    aggregate = aggregate_metric("brier_score", [0.1, None, 0.3])
    assert aggregate.count == 2
    assert aggregate.values == (0.1, 0.3)


def test_an_aggregate_of_nothing_says_so() -> None:
    aggregate = aggregate_metric("log_loss", [])
    assert aggregate.count == 0
    assert aggregate.describe() == "log_loss: not measured"
    assert aggregate.std == 0.0


def test_a_comparison_names_the_best_baseline_and_the_deltas() -> None:
    outcomes = CODES.repeat(10)
    model = one_hot(outcomes)
    worse = np.full(model.shape, 1.0 / CLASS_COUNT)
    comparison = compare_to_baselines(
        outcomes, model, {"random": worse}, classes=CLASSES, policy=policy()
    )
    assert comparison.rows == len(outcomes)
    assert comparison.best_baseline == "random"
    assert comparison.beats == ("random",)
    assert comparison.beats_all_baselines is True
    assert comparison.deltas["random"] == pytest.approx(comparison.baselines["random"].log_loss)
    assert "beat 1/1" in comparison.describe()


def test_a_comparison_records_when_the_baseline_wins() -> None:
    outcomes = CODES.repeat(10)
    worse_model = np.full((len(outcomes), CLASS_COUNT), 1.0 / CLASS_COUNT)
    comparison = compare_to_baselines(
        outcomes, worse_model, {"perfect": one_hot(outcomes)}, classes=CLASSES, policy=policy()
    )
    assert comparison.beats == ()
    assert comparison.beats_all_baselines is False
    assert comparison.deltas["perfect"] < 0


def test_a_comparison_carries_three_metrics_whatever_the_policy_asks() -> None:
    outcomes = CODES.repeat(5)
    comparison = compare_to_baselines(
        outcomes,
        one_hot(outcomes),
        {},
        classes=CLASSES,
        policy=policy(metrics=("log_loss",)),
    )
    assert set(comparison.model.metrics) == {"log_loss", "brier_score", "accuracy"}
    assert comparison.best_baseline is None
    assert comparison.beats_all_baselines is False


def test_a_small_comparison_is_labelled_preliminary() -> None:
    outcomes = CODES
    comparison = compare_to_baselines(
        outcomes, one_hot(outcomes), {}, classes=CLASSES, policy=policy(preliminary_min_samples=30)
    )
    assert comparison.label == "PRELIMINARY"


def test_a_large_comparison_is_labelled_measured() -> None:
    outcomes = np.tile(CODES, 10)
    comparison = compare_to_baselines(
        outcomes, one_hot(outcomes), {}, classes=CLASSES, policy=policy(preliminary_min_samples=30)
    )
    assert comparison.label == "MEASURED"


def test_a_caller_can_label_a_holdout() -> None:
    outcomes = np.tile(CODES, 10)
    comparison = compare_to_baselines(
        outcomes,
        one_hot(outcomes),
        {},
        classes=CLASSES,
        policy=policy(),
        index=3,
        label="HOLDOUT",
    )
    assert comparison.label == "HOLDOUT"
    assert comparison.to_dict()["fold"] == 3


def test_the_comparison_metric_set_includes_the_configured_ones() -> None:
    assert comparison_metrics(policy(metrics=("log_loss",))) == (
        "log_loss",
        "brier_score",
        "accuracy",
    )
    assert "calibration_curve" in comparison_metrics(policy())


# ---------------------------------------------------------------------------
# Spec 4.3's gate, as a label
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("stage", "label", "production", "refused"),
    [
        ("INSUFFICIENT", "BELOW_GATE", False, True),
        ("EXPLORATORY", "EXPLORATORY", False, False),
        ("ML_TRAINING", "PRELIMINARY", True, False),
        ("MEANINGFUL_BACKTEST", "PRELIMINARY", True, False),
        ("SEASONALITY", "PRELIMINARY", True, False),
    ],
)
def test_each_gate_stage_gets_the_label_it_earns(
    stage: str, label: str, production: bool, refused: bool
) -> None:
    verdict = gate_verdict(stage, days=90)
    assert verdict.results_label == label
    assert verdict.production_model is production
    assert verdict.refused is refused
    assert verdict.note and "not a backtest" in verdict.note


def test_a_forced_run_is_allowed_but_stays_below_the_gate() -> None:
    verdict = gate_verdict("INSUFFICIENT", days=4, forced=True)
    assert verdict.refused is False
    assert verdict.results_label == "BELOW_GATE"
    assert verdict.production_model is False


def test_the_verdict_says_how_many_days_it_saw() -> None:
    assert "4 captured day(s)" in gate_verdict("EXPLORATORY", days=4).reason
    assert "the captured data" in gate_verdict("EXPLORATORY").reason


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(EvaluationError, match="unknown data-gate stage"):
        gate_verdict("PLENTY")


def test_the_verdict_serialises() -> None:
    payload = gate_verdict("ML_TRAINING", days=90).to_dict()
    assert payload["stage"] == "ML_TRAINING"
    assert payload["production_model"] is True
    assert payload["results_label"] == "PRELIMINARY"

