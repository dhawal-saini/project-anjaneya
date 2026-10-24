"""Unit tests for the trainer (``core/ml/trainer.py``, spec 21).

A trainer is judged by what it *refuses* as much as by what it fits, so these tests are
split in three:

* the policy - a configuration that asks for something the stage cannot honour must fail
  loudly, and the LightGBM parameters must pin every seed the library has;
* the refusals - an empty frame, a single-class frame, a frame without class codes, a
  label-derived column offered as a feature, a categorical column that is not a feature;
* the learning - on a frame whose class is genuinely determined by two features, the model
  has to beat the base rate, be reproducible run to run, and score a *new* frame without
  tripping over an unseen categorical level.

Nothing here touches the network, the database or a file.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.ml import (
    CODE_COLUMN,
    TARGET_COLUMN,
    DatasetError,
    DatasetPolicy,
    EvaluationPolicy,
    SplitPolicy,
    TrainingError,
    TrainingPolicy,
    build_dataset,
    gate_verdict,
    purged_walk_forward_splits,
    train_and_evaluate,
    train_model,
)
from core.ml.trainer import (
    SUPPORTED_ALGORITHMS,
    class_weights,
)
from tests.support.ai import (
    CLASSES,
    DAYS,
    IST,
    ONE_DAY_FOLDS,
    SIGNAL_CATEGORICAL,
    SIGNAL_FEATURES,
    feature_frame,
    label_frame,
    signal_frame,
)

#: A small, fast policy: enough trees to learn the synthetic signal, few enough to be quick.
FAST = TrainingPolicy(n_estimators=30, num_leaves=8, min_child_samples=5, n_jobs=1)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_policy_comes_from_the_shipped_config(config) -> None:
    policy = TrainingPolicy.from_config(config)
    training = config.model.training
    assert policy.algorithm == config.model.algorithm == "lightgbm"
    assert policy.algorithm in SUPPORTED_ALGORITHMS
    assert policy.random_seed == config.model.random_seed
    assert policy.n_estimators == training.n_estimators
    assert policy.num_leaves == training.num_leaves
    assert policy.class_weight == training.class_weight == "none"
    assert policy.early_stopping is False
    assert policy.timezone_name == "Asia/Kolkata"


def test_seed_can_be_overridden_for_one_run(config) -> None:
    assert TrainingPolicy.from_config(config, seed=7).random_seed == 7


def test_policy_describe_records_every_number(config) -> None:
    description = TrainingPolicy.from_config(config).describe()
    assert description["n_estimators"] == config.model.training.n_estimators
    assert description["early_stopping"] is False
    assert description["random_seed"] == config.model.random_seed


def test_params_pin_every_lightgbm_seed() -> None:
    params = replace(FAST, random_seed=11).to_params(n_classes=4)
    assert params["objective"] == "multiclass"
    assert params["num_class"] == 4
    assert params["deterministic"] is True
    assert params["seed"] == 11
    assert {params["bagging_seed"], params["feature_fraction_seed"], params["data_random_seed"]} == {
        12,
        13,
        14,
    }
    assert params["verbosity"] == -1


def test_params_refuse_a_single_class_problem() -> None:
    with pytest.raises(TrainingError, match="at least two classes"):
        FAST.to_params(n_classes=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"algorithm": "xgboost"}, "not implemented"),
        ({"n_estimators": 0}, "n_estimators"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": 2.0}, "learning_rate"),
        ({"num_leaves": 1}, "num_leaves"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_child_samples": 0}, "min_child_samples"),
        ({"feature_fraction": 0.0}, "feature_fraction"),
        ({"bagging_fraction": 0.5, "bagging_freq": 0}, "bagging_freq"),
        ({"lambda_l1": -1.0}, "lambda_l1"),
        ({"class_weight": "sqrt"}, "class_weight"),
        ({"early_stopping_rounds": -5}, "early_stopping_rounds"),
        ({"n_jobs": -1}, "n_jobs"),
        ({"random_seed": -1}, "random_seed"),
    ],
)
def test_policy_refuses_what_it_cannot_honour(overrides: dict, message: str) -> None:
    with pytest.raises(TrainingError, match=message):
        replace(FAST, **overrides)


def test_early_stopping_flag_follows_the_rounds() -> None:
    assert replace(FAST, early_stopping_rounds=5).early_stopping is True
    assert replace(FAST, early_stopping_rounds=0).early_stopping is False


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def test_class_weights_none_are_uniform() -> None:
    assert class_weights([1, 2, 3, 4], kind="none") == (1.0, 1.0, 1.0, 1.0)


def test_balanced_class_weights_favour_the_rare_class() -> None:
    counts = [10, 10, 20, 60]
    weights = class_weights(counts)
    # The rarer the class, the larger its weight: 0/1 (10 rows) > 2 (20) > 3 (60).
    assert weights[3] < weights[2] < weights[0] == weights[1]
    # scikit-learn's convention: the weights summed over the data are the sample count.
    assert sum(weight * count for weight, count in zip(weights, counts, strict=True)) == pytest.approx(
        100.0
    )


def test_balanced_class_weights_skip_an_absent_class() -> None:
    assert class_weights([10, 0])[1] == 0.0


def test_class_weights_refuse_an_unknown_kind() -> None:
    with pytest.raises(TrainingError, match="unknown class-weight kind"):
        class_weights([1, 1], kind="inverse")


def test_class_weights_need_rows() -> None:
    with pytest.raises(TrainingError, match="at least one training row"):
        class_weights([0, 0])


# ---------------------------------------------------------------------------
# Refusals: nothing that would make a metric look better than the model is
# ---------------------------------------------------------------------------
def train(frame: pd.DataFrame, **overrides) -> object:
    """``train_model`` with the fixture's metadata, overridable per test."""
    kwargs: dict = {
        "policy": FAST,
        "classes": CLASSES,
        "feature_columns": SIGNAL_FEATURES,
        "categorical_columns": SIGNAL_CATEGORICAL,
        "horizon": 5,
    }
    kwargs.update(overrides)
    return train_model(frame, **kwargs)


def test_an_empty_training_frame_is_refused() -> None:
    with pytest.raises(TrainingError, match="no rows"):
        train(signal_frame().iloc[:0])


def test_a_single_class_training_frame_is_refused() -> None:
    frame = signal_frame().assign(**{CODE_COLUMN: 0, TARGET_COLUMN: CLASSES[0]})
    with pytest.raises(TrainingError, match="cannot be fitted on one class"):
        train(frame)


def test_a_frame_without_class_codes_is_refused() -> None:
    with pytest.raises(TrainingError, match=CODE_COLUMN):
        train(signal_frame().drop(columns=[CODE_COLUMN]))


def test_class_codes_outside_the_configured_classes_are_refused() -> None:
    frame = signal_frame().assign(**{CODE_COLUMN: 9})
    with pytest.raises(TrainingError, match="outside"):
        train(frame)


def test_non_integer_class_codes_are_refused() -> None:
    frame = signal_frame()
    frame[CODE_COLUMN] = frame[CODE_COLUMN].astype(str)
    with pytest.raises(TrainingError, match="integer class codes"):
        train(frame)


def test_no_feature_columns_is_refused() -> None:
    with pytest.raises(TrainingError, match="at least one feature"):
        train(signal_frame(), feature_columns=())


def test_duplicate_feature_columns_are_refused() -> None:
    with pytest.raises(TrainingError, match="unique"):
        train(signal_frame(), feature_columns=("signal_a", "signal_a"))


def test_a_missing_feature_column_is_refused() -> None:
    with pytest.raises(TrainingError, match="missing feature column"):
        train(signal_frame(), feature_columns=(*SIGNAL_FEATURES, "not_a_column"))


def test_a_categorical_column_that_is_not_a_feature_is_refused() -> None:
    with pytest.raises(TrainingError, match="not among the feature columns"):
        train(signal_frame().assign(spare="x"), categorical_columns=("spare",))


def test_a_label_derived_feature_is_refused() -> None:
    frame = signal_frame().assign(future_return=1.0)
    with pytest.raises(DatasetError, match="label-derived"):
        train(frame, feature_columns=(*SIGNAL_FEATURES, "future_return"))


def test_a_non_numeric_feature_is_refused_unless_it_is_categorical() -> None:
    frame = signal_frame()
    frame["nifty_regime"] = frame["nifty_regime"].astype(str)
    with pytest.raises(TrainingError, match="not numeric"):
        train(frame, categorical_columns=())


def test_fewer_than_two_configured_classes_is_refused() -> None:
    with pytest.raises(TrainingError, match="at least two classes"):
        train(signal_frame(), classes=(CLASSES[0],))


# ---------------------------------------------------------------------------
# Learning: the model has to actually beat the base rate on a learnable frame
# ---------------------------------------------------------------------------
def test_the_model_learns_a_signal_it_can_see() -> None:
    frame = signal_frame()
    model = train(frame)
    probabilities = model.predict_proba(frame)
    hits = (probabilities.argmax(axis=1) == frame[CODE_COLUMN].to_numpy()).mean()
    # The fixture's class is a deterministic function of two features, so a model that
    # learned anything at all must be far above the 25% base rate.
    assert hits > 0.75, hits


def test_probabilities_are_a_distribution_per_row() -> None:
    frame = signal_frame()
    model = train(frame)
    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (len(frame), len(CLASSES))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    predicted = model.predict(frame)
    assert predicted.shape == (len(frame),)
    assert predicted.min() >= 0 and predicted.max() < len(CLASSES)


def test_training_is_reproducible_run_to_run() -> None:
    frame = signal_frame()
    first = train(frame).predict_proba(frame)
    second = train(frame).predict_proba(frame)
    np.testing.assert_allclose(first, second)


def test_a_different_seed_gives_a_different_fit() -> None:
    frame = signal_frame()
    first = train(frame, policy=replace(FAST, random_seed=1)).predict_proba(frame)
    second = train(frame, policy=replace(FAST, random_seed=7)).predict_proba(frame)
    assert not np.allclose(first, second)


def test_missing_values_and_unseen_levels_do_not_break_scoring() -> None:
    frame = signal_frame()
    model = train(frame)
    unseen = frame.iloc[:5].copy()
    unseen.loc[unseen.index[0], "signal_a"] = np.nan
    unseen["nifty_regime"] = "SIDEWAYS"  # a level the training fold never saw
    probabilities = model.predict_proba(unseen)
    assert probabilities.shape == (5, len(CLASSES))
    assert np.isfinite(probabilities).all()


def test_the_model_records_the_categorical_vocabulary() -> None:
    model = train(signal_frame())
    assert model.categorical_vocabulary["nifty_regime"] == ("BEARISH", "BULLISH")
    assert model.categorical_columns == SIGNAL_CATEGORICAL


def test_feature_importance_covers_every_feature() -> None:
    model = train(signal_frame())
    importance = model.feature_importance()
    assert set(importance) == set(SIGNAL_FEATURES)
    ranked = model.top_features(2)
    assert len(ranked) == 2
    # The two signal columns must out-rank the unrelated ones (gain, so this is not subtle).
    assert ranked[0][1] >= ranked[-1][1]
    assert {name for name, _ in ranked} <= {"signal_a", "signal_b"}


def test_manifest_records_how_the_model_was_trained() -> None:
    model = train(signal_frame())
    description = model.describe()
    assert description["algorithm"] == "lightgbm"
    assert description["horizon_minutes"] == 5
    assert description["classes"] == list(CLASSES)
    assert description["features"] == len(SIGNAL_FEATURES)
    assert description["early_stopping"] is False
    manifest = model.to_manifest()
    assert list(manifest["feature_columns"]) == list(SIGNAL_FEATURES)
    assert manifest["policy"]["random_seed"] == FAST.random_seed
    assert set(manifest["importance"]) == set(SIGNAL_FEATURES)


def test_class_weighting_is_recorded_and_changes_the_fit() -> None:
    frame = signal_frame()
    plain = train(frame).predict_proba(frame)
    weighted = train(frame, policy=replace(FAST, class_weight="balanced")).predict_proba(frame)
    assert not np.allclose(plain, weighted)


def test_an_empty_frame_scores_an_empty_result() -> None:
    frame = signal_frame()
    model = train(frame)
    assert model.predict_proba(frame.iloc[:0]).shape == (0, len(CLASSES))
    assert model.predict(frame.iloc[:0]).shape == (0,)


def test_scoring_a_frame_without_the_features_is_refused() -> None:
    model = train(signal_frame())
    with pytest.raises(TrainingError, match="missing feature column"):
        model.predict_proba(signal_frame().drop(columns=["signal_a"]))





# ---------------------------------------------------------------------------
# The walk-forward run
# ---------------------------------------------------------------------------
def per_day_dataset(
    config, *, days: int | None = None, uniform_class: int | None = None, first_day_class: int | None = None
):
    """A dataset whose sessions carry few classes - so a fold boundary changes the mix.

    The shipped fixture cycles the class *within* a minute, which makes every fold look
    identical. Relabelling sessions is what lets a test ask the question that matters: what
    happens to a training fold that happens to contain one class? ``first_day_class``
    restricts it to the opening session (later folds keep a mixed training window) and
    ``uniform_class`` relabels every session.
    """
    window = DAYS if days is None else DAYS[:days]
    labels = label_frame(window)
    days_series = labels["timestamp"].dt.tz_convert(IST).dt.date
    overrides: dict = {}
    if uniform_class is not None:
        overrides = dict.fromkeys(window, uniform_class)
    elif first_day_class is not None:
        overrides = dict.fromkeys(window[:1], first_day_class)
    labels["label"] = [
        None if value is None else CLASSES[overrides.get(day, CLASSES.index(value))]
        for day, value in zip(days_series, labels["label"], strict=True)
    ]
    return build_dataset(
        feature_frame(window), labels, policy=DatasetPolicy.from_config(config), horizon=5
    )


def fixture_dataset(config, *, horizon: int = 5):
    """The shipped fixture's matrix: four contract-days carrying all four classes."""
    return build_dataset(
        feature_frame(), label_frame(), policy=DatasetPolicy.from_config(config), horizon=horizon
    )


def split_of(config, dataset, **overrides):
    """The fixture's split policy widened to one-day folds, so a four-day fixture folds."""
    policy = replace(
        SplitPolicy.from_config(config, horizon_minutes=dataset.horizon),
        **{**ONE_DAY_FOLDS, **overrides},
    )
    return purged_walk_forward_splits(dataset, policy=policy)


def run(config, dataset, split, **overrides):
    """``train_and_evaluate`` with the fast policy and the configured baselines."""
    kwargs: dict = {
        "policy": replace(FAST, random_seed=config.model.random_seed),
        "evaluation": EvaluationPolicy.from_config(config),
        "baseline_names": tuple(config.model.evaluation.baselines),
    }
    kwargs.update(overrides)
    return train_and_evaluate(dataset, split=split, **kwargs)


def test_the_run_trains_every_fold_and_scores_its_validation_window(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    report = run(config, dataset, split)
    assert report.ok
    assert len(report.folds) == len(split.folds) == 2
    assert len(report.trained_folds) == 2
    for result in report.trained_folds:
        assert result.comparison is not None
        # The comparison scored exactly the rows the split says the validation window has.
        assert result.comparison.rows == result.fold.validate_rows
        assert result.comparison.model.rows == result.comparison.rows
    assert set(report.aggregates) >= {"log_loss", "brier_score", "accuracy"}
    assert report.aggregates["log_loss"].count == 2


def test_the_fold_frames_are_the_rows_the_fold_counted(config) -> None:
    from core.ml import fold_frames

    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    for fold in split.folds:
        train_frame, validate_frame = fold_frames(dataset, fold)
        assert len(train_frame) == fold.train_rows
        assert len(validate_frame) == fold.validate_rows
        # And the trainer's score is measured on exactly those validation rows.
    report = run(config, dataset, split)
    for result in report.trained_folds:
        assert result.comparison is not None
        assert result.comparison.rows == result.fold.validate_rows


def test_the_final_model_is_fitted_on_everything_outside_the_holdout(config) -> None:
    from core.ml import usable_frame

    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    report = run(config, dataset, split)
    outside = usable_frame(dataset, split)
    assert report.final_train_rows == len(outside)
    assert report.final_model is not None
    assert report.final_model.train_rows == len(outside)
    assert report.final_train_rows < dataset.rows  # the holdout stayed out of the fit


def test_the_holdout_is_left_alone_unless_it_is_asked_for(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    untouched = run(config, dataset, split)
    assert untouched.holdout is None
    assert not any("no longer untouched" in note for note in untouched.notes)

    scored = run(config, dataset, split, include_holdout=True)
    assert scored.holdout is not None
    assert scored.holdout.label == "HOLDOUT"
    assert scored.holdout.rows == split.holdout.rows
    assert any("no longer untouched" in note for note in scored.notes)


def test_every_baseline_is_scored_on_the_same_rows_as_the_model(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    report = run(config, dataset, split)
    names = set(config.model.evaluation.baselines)
    for result in report.trained_folds:
        comparison = result.comparison
        assert set(comparison.baselines) == names
        for name, metrics in comparison.baselines.items():
            assert metrics.rows == comparison.rows
            delta = comparison.deltas[name]
            assert delta == pytest.approx(metrics.log_loss - comparison.model.log_loss)
        assert comparison.best_baseline in names


def test_a_single_class_training_fold_is_skipped_and_said_so(config) -> None:
    # Day 0 carries one class; the sessions after it keep the fixture's mix.
    dataset = per_day_dataset(config, days=4, first_day_class=0)
    split = split_of(config, dataset)
    report = run(config, dataset, split)
    assert len(report.folds) == 2
    skipped, trained = report.folds[0], report.folds[1]
    assert skipped.status == "SKIPPED"
    assert "one class" in (skipped.reason or "")
    assert skipped.comparison is None
    assert trained.status == "TRAINED"
    assert report.ok is True
    assert any("skipped" in note for note in report.notes)
    assert report.aggregates["log_loss"].count == 1


def test_a_run_with_no_trainable_fold_is_not_an_error(config) -> None:
    dataset = per_day_dataset(config, days=4, uniform_class=0)
    split = split_of(config, dataset)
    report = run(config, dataset, split)
    assert report.ok is False
    assert report.aggregates == {}
    assert report.final_model is None
    assert report.model_log_loss() is None
    assert report.folds and all(result.status == "SKIPPED" for result in report.folds)
    assert any("could not be trained" in note for note in report.notes)


def test_a_split_that_cannot_fold_reports_why(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset, train_window_days=3, validate_window_days=3, holdout_days=1)
    report = run(config, dataset, split)
    assert report.folds == []
    assert report.ok is False
    assert report.notes and "no folds" in report.notes[0]
    assert "no folds" in report.describe()


def test_the_verdict_and_its_caveat_travel_into_the_report(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    verdict = gate_verdict("EXPLORATORY", days=len(dataset.days))
    report = run(config, dataset, split, verdict=verdict)
    assert report.verdict is verdict
    assert report.results_label == "EXPLORATORY"
    assert verdict.note in report.notes


def test_early_stopping_is_flagged_as_optimistic(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    report = run(config, dataset, split, policy=replace(FAST, early_stopping_rounds=5))
    assert any("early stopping is on" in note for note in report.notes)
    assert report.final_model is not None
    assert report.final_model.policy.early_stopping is True


def test_the_report_is_json_serialisable_and_complete(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    report = run(config, dataset, split, include_holdout=True)
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["horizon_minutes"] == dataset.horizon
    assert payload["trained_folds"] == 2
    assert payload["aggregates"]["log_loss"]["count"] == 2
    assert payload["final_model"]["feature_columns"]
    assert payload["holdout"]["label"] == "HOLDOUT"
    assert payload["split"]["folds"] == 2
    assert len(payload["folds"]) == 2


def test_a_short_run_says_the_spread_is_descriptive(config) -> None:
    dataset = fixture_dataset(config)
    split = split_of(config, dataset)
    report = run(config, dataset, split)
    assert any("too few for a spread" in note for note in report.notes)
    assert "log loss" in report.describe()
