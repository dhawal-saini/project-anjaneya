"""Tests for :mod:`core.ml.importance` - spec 27's feature importance.

Four rules carry most of the weight, and each has its own section below:

* **the sources are configuration.** A policy built from the shipped ``config/model.yaml`` and
  the module's own defaults resolve to the same topics, every trainable feature is claimed by
  exactly one of them, and a hand-built policy that breaks an invariant (a column claimed
  twice, one repeat, a reserved group) is refused - so "do not assume" is a test, not a
  promise.
* **the measurement is measured, not asserted.** Permutation importance finds the feature a
  class is actually a function of, ahead of the noise beside it, and two runs at one seed
  produce the same deltas to the last bit.
* **unknown is not zero.** A column the sample never carries, or one holding a single value,
  is reported as *not measured* with the reason - never as a delta of 0, which would read as
  "this feature carries no information".
* **SHAP is optional in the strongest sense.** Missing, off, stubbed and broken are four
  different statuses, and only ``on`` turns a missing package into an error.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from core.constants import FEATURE_COLUMNS
from core.ml.dataset import CODE_COLUMN, TARGET_COLUMN, DatasetPolicy, build_dataset
from core.ml.importance import (
    CORRELATED_FEATURES_CAVEAT,
    DEFAULT_TOPICS,
    IMPORTANCE_CAVEAT,
    RESERVED_IMPORTANCE_GROUPS,
    SHAP_FAILED,
    SHAP_MEASURED,
    SHAP_SKIPPED,
    SHAP_UNAVAILABLE,
    UNMAPPED_TOPIC_KEY,
    ImportanceError,
    ImportancePolicy,
    ImportanceSample,
    ImportanceTopic,
    build_importance_report,
    measurement_caveat,
    metric_score,
    native_importance,
    permutation_importance,
    select_sample,
    shap_importance,
)
from core.ml.trainer import TrainingPolicy, train_model
from core.ml.walk_forward import SplitPolicy, purged_walk_forward_splits
from tests.support.ai import (
    CLASSES,
    feature_frame,
    label_frame,
    write_signal_model,
)

#: A policy small enough for a test: three shuffles, everything in scope.
FAST = {"repeats": 3, "max_rows": 100_000}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------
@dataclass
class StubArtefact:
    """The two attributes the measurement reads off an artefact: its booster and classes."""

    booster: object
    classes: tuple[str, ...]


class StubPredictor:
    """A :class:`core.ml.predictor.ModelPredictor`-shaped double around a fitted model.

    The real predictor is exercised by the integration tests; this one exists so a unit test
    can permute a frame whose class is a known function of one column and ask whether the
    measurement noticed.
    """

    def __init__(self, model, *, feature_columns, classes=CLASSES) -> None:
        self.model = model
        self.artefact = StubArtefact(booster=model.booster, classes=tuple(classes))
        self.feature_columns = tuple(feature_columns)
        self.classes = tuple(classes)

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.model.matrix(frame)


def learnable_frame(rows: int = 400, *, seed: int = 0) -> pd.DataFrame:
    """A frame whose class is a fixed function of one column (and nothing of the column beside it).

    The columns are real contract columns (``option_atr``, ``nifty_trend_persistence``,
    ``nifty_regime``, ``candle_inside_flag``, ``iv``), because a topic may only claim a stored
    feature: the frame is synthetic, the contract is not.
    """
    generator = np.random.default_rng(seed)
    signal = generator.normal(size=rows)
    noise = generator.normal(size=rows)
    code = np.digitize(signal, (-0.5, 0.5, 1.5)).astype("int64")
    frame = pd.DataFrame(
        {
            "option_atr": signal,
            "nifty_trend_persistence": noise,
            "nifty_regime": np.where(signal > 0, "BULLISH", "BEARISH"),
            "candle_inside_flag": np.zeros(rows, dtype=bool),
            "iv": np.full(rows, np.nan),
        }
    )
    frame[CODE_COLUMN] = code
    frame[TARGET_COLUMN] = [CLASSES[int(value)] for value in code]
    return frame


def learnable_predictor(frame: pd.DataFrame, *, seed: int = 0) -> StubPredictor:
    """A small fitted model over :func:`learnable_frame`'s columns."""
    columns = (
        "option_atr",
        "nifty_trend_persistence",
        "nifty_regime",
        "candle_inside_flag",
        "iv",
    )
    model = train_model(
        frame,
        policy=TrainingPolicy(
            algorithm="lightgbm",
            random_seed=seed,
            n_estimators=20,
            num_leaves=4,
            min_child_samples=5,
            n_jobs=1,
        ),
        classes=CLASSES,
        feature_columns=columns,
        categorical_columns=("nifty_regime",),
    )
    return StubPredictor(model, feature_columns=columns)


def sample_of(frame: pd.DataFrame, *, name: str = "matrix", reason: str | None = None) -> ImportanceSample:
    """A hand-built sample (the selection rules are tested separately)."""
    return ImportanceSample(
        name=name,
        frame=frame.reset_index(drop=True),
        rows_available=len(frame),
        rows_used=len(frame),
        capped=False,
        days=1,
        reason=reason,
    )

# ---------------------------------------------------------------------------
# The policy: spec 27's sources, from configuration
# ---------------------------------------------------------------------------
def test_the_default_policy_mirrors_the_shipped_sources(config) -> None:
    """The module's defaults and ``config/model.yaml`` describe one structure, not two."""
    from core.ml import LABEL_DERIVED_COLUMNS, META_COLUMNS

    policy = ImportancePolicy.from_config(config)
    assert policy.spec == "27"
    assert policy.repeats == 5
    assert policy.metric == "log_loss"
    assert policy.sample == "holdout"
    assert [topic.key for topic in policy.topics] == [topic.key for topic in DEFAULT_TOPICS]
    assert policy.columns == tuple(column for topic in DEFAULT_TOPICS for column in topic.columns)
    trainable = [
        name
        for name in FEATURE_COLUMNS
        if name not in META_COLUMNS and name not in LABEL_DERIVED_COLUMNS
    ]
    # Every model input is claimed exactly once, by a group or by a named column.
    assert sorted(policy.columns) == sorted(trainable)
    assert len(set(policy.columns)) == len(policy.columns)
    assert policy.unmapped(trainable) == ()
    # The shipped configuration claims whole groups as well as individual columns.
    assert {source.group for topic in policy.topics for source in topic.sources if source.group}


def test_a_column_claimed_twice_is_refused() -> None:
    """One column, one source: a feature counted twice would be credited twice."""
    with pytest.raises(ImportanceError, match="claimed twice"):
        ImportancePolicy(
            topics=(
                DEFAULT_TOPICS[0],
                DEFAULT_TOPICS[6],
                ImportanceTopic("vol", "volatility again", DEFAULT_TOPICS[6].sources),
            )
        )


def test_a_reserved_group_cannot_be_a_source() -> None:
    from core.ml.importance import _sources

    for name in sorted(RESERVED_IMPORTANCE_GROUPS):
        with pytest.raises(ImportanceError, match="reserved feature group"):
            _sources((name,))


def test_an_unknown_group_or_column_is_refused() -> None:
    from core.ml.importance import _sources

    with pytest.raises(ImportanceError, match="unknown feature group"):
        _sources(("niftee",))
    with pytest.raises(ImportanceError, match="not a column of the stored feature row"):
        _sources((), ("implied_vol",))


def test_a_policy_with_no_topics_is_refused() -> None:
    with pytest.raises(ImportanceError, match="no topics"):
        ImportancePolicy(topics=())


def test_the_unmapped_key_is_reserved() -> None:
    with pytest.raises(ImportanceError, match="reserved"):
        ImportancePolicy(
            topics=(ImportanceTopic(UNMAPPED_TOPIC_KEY, "everything else", DEFAULT_TOPICS[6].sources),)
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repeats": 1}, "at least 2"),
        ({"repeats": 0}, "at least 2"),
        ({"metric": "f1"}, "unknown importance metric"),
        ({"confidence": 1.5}, "confidence"),
        ({"sample": "everything"}, "unknown importance sample"),
        ({"max_rows": 0}, "max_rows"),
        ({"shap": "maybe"}, "unknown importance.shap"),
        ({"shap_rows": 0}, "shap_rows"),
        ({"seed": -1}, "seed"),
        ({"digits": -1}, "digits"),
    ],
)
def test_a_policy_that_cannot_be_honoured_is_refused(kwargs, message) -> None:
    with pytest.raises(ImportanceError, match=message):
        ImportancePolicy(**kwargs)


def test_the_seed_falls_back_to_the_models_own() -> None:
    """A run is reproducible by quoting one number: the model's seed, unless one is configured."""
    assert ImportancePolicy().random_seed(7) == 7
    assert ImportancePolicy(seed=3).random_seed(7) == 3


def test_a_configuration_without_the_block_is_the_default() -> None:
    """An older configuration must not change what an importance run means."""

    class _Model:
        pass

    class _Config:
        model = _Model()

    assert ImportancePolicy.from_config(_Config()) == ImportancePolicy()


def test_the_policy_reports_the_columns_it_claims_and_the_ones_it_does_not() -> None:
    policy = ImportancePolicy()
    assert policy.topic_for("iv") == "iv"
    assert policy.topic_for("not_a_feature") is None
    assert policy.unmapped(("iv", "not_a_feature")) == ("not_a_feature",)
    assert policy.topic("greeks") is not None and policy.topic("nope") is None




# ---------------------------------------------------------------------------
# Native importance: the model's own report, off the booster
# ---------------------------------------------------------------------------
def test_native_importance_comes_off_the_booster_in_feature_order(tmp_path) -> None:
    """The number and the model come from one place: the artefact's own booster."""
    from core.ml import ModelArtefact

    artefact = ModelArtefact.load(write_signal_model(tmp_path / "models"))
    values = native_importance(artefact)
    assert list(values) == list(artefact.feature_columns)
    assert sum(values.values()) > 0
    # The metadata's copy of the same numbers must agree with the booster's, or a reader would
    # have two "native importances" and no reason to prefer either.
    manifest = artefact.metadata["model"]["importance"]
    assert all(values[name] == pytest.approx(manifest[name]) for name in values)


def test_native_importance_refuses_an_artefact_without_a_feature_list() -> None:
    from types import SimpleNamespace

    with pytest.raises(ImportanceError, match="no booster or no feature list"):
        native_importance(SimpleNamespace(booster=object(), feature_columns=()))


def test_native_importance_refuses_a_booster_that_disagrees_with_its_metadata() -> None:
    """A mismatch here would rank columns against numbers that belong to other columns."""
    from types import SimpleNamespace

    class _Booster:
        def feature_importance(self, importance_type: str) -> np.ndarray:
            assert importance_type == "gain"
            return np.array([1.0, 2.0])

    with pytest.raises(ImportanceError, match="disagree"):
        native_importance(SimpleNamespace(booster=_Booster(), feature_columns=("a", "b", "c")))


# ---------------------------------------------------------------------------
# The sample: spec 22's reserved days, or the matrix with a reason
# ---------------------------------------------------------------------------
def fixture_dataset(config):
    """The four-day fixture as a real ``Dataset`` plus a one-day-fold split."""
    dataset_policy = DatasetPolicy.from_config(config, timeframe="1m")
    dataset = build_dataset(
        feature_frame(), label_frame(), policy=dataset_policy, horizon=5, symbols=None
    )
    split = purged_walk_forward_splits(
        dataset,
        policy=SplitPolicy(
            train_window_days=1,
            validate_window_days=1,
            step_days=1,
            embargo_minutes=30,
            holdout_days=1,
            horizon_minutes=5,
            timezone_name=config.timezone_name,
        ),
    )
    return dataset, split


def test_the_sample_is_the_splits_reserved_days_when_it_has_any(config) -> None:
    """Spec 22's holdout is the only set of rows the *split* promises were kept back."""
    dataset, split = fixture_dataset(config)
    sample = select_sample(dataset, split, policy=ImportancePolicy())
    assert sample.name == "holdout"
    assert split.holdout is not None and split.holdout.rows > 0
    assert sample.rows_used == split.holdout.rows
    assert sample.days == len(split.holdout.days)
    assert sample.rows_available == sample.rows_used
    assert sample.capped is False
    assert sample.reason is None
    assert "out-of-sample" in measurement_caveat(sample)
    assert sample.describe()["first_timestamp"] == sample.start.isoformat()


def test_without_a_holdout_the_sample_is_the_matrix_and_says_why(config) -> None:
    """A fallback is allowed; a *silent* fallback is not."""
    dataset, _ = fixture_dataset(config)
    fallback = select_sample(dataset, None, policy=ImportancePolicy())
    assert fallback.name == "matrix"
    assert fallback.rows_used == dataset.rows
    assert fallback.reason and "holdout" in fallback.reason
    caveat = measurement_caveat(fallback)
    assert "in-sample" in caveat and "optimistic" in caveat

    asked = select_sample(dataset, None, policy=ImportancePolicy(sample="matrix"))
    assert asked.name == "matrix" and asked.reason is None
    assert "the configuration asked for the stored matrix" in measurement_caveat(asked)


def test_the_sample_keeps_the_newest_rows_when_it_is_capped(config) -> None:
    """The cap is a count, not a silence: the drop is published beside the measurement."""
    dataset, split = fixture_dataset(config)
    full = select_sample(dataset, split, policy=ImportancePolicy())
    capped = select_sample(dataset, split, policy=ImportancePolicy(max_rows=50))
    assert capped.capped is True
    assert capped.rows_used == 50
    assert capped.rows_available == full.rows_available
    assert capped.days <= full.days
    assert capped.end == full.end


# ---------------------------------------------------------------------------
# Permutation importance: measured, not asserted
# ---------------------------------------------------------------------------
def test_permutation_importance_finds_the_feature_the_class_is_a_function_of() -> None:
    """The whole point: the measured ranking must put the real signal above the noise."""
    frame = learnable_frame()
    predictor = learnable_predictor(frame)
    result = permutation_importance(
        predictor, frame, classes=CLASSES, policy=ImportancePolicy(**FAST)
    )
    ranked = [name for name, _ in result.ranking()]
    assert ranked[0] == "option_atr"
    assert ranked.index("option_atr") < ranked.index("nifty_trend_persistence")
    assert "nifty_regime" in ranked
    assert result.rows == len(frame)
    assert result.deltas.shape == (len(predictor.feature_columns), FAST["repeats"])
    assert all(np.isfinite(result.values("option_atr")))


def test_two_runs_at_one_seed_produce_the_same_deltas() -> None:
    """Reproducibility is a property of the run, not of the operator's memory."""
    frame = learnable_frame()
    predictor = learnable_predictor(frame)
    policy = ImportancePolicy(**FAST)
    first = permutation_importance(predictor, frame, classes=CLASSES, policy=policy)
    second = permutation_importance(predictor, frame, classes=CLASSES, policy=policy)
    np.testing.assert_array_equal(first.deltas, second.deltas)
    assert first.seed == second.seed == policy.random_seed()
    other = permutation_importance(
        predictor, frame, classes=CLASSES, policy=ImportancePolicy(seed=99, **FAST)
    )
    assert other.seed == 99
    assert not np.array_equal(first.deltas, other.deltas)


def test_a_column_the_sample_never_carries_is_not_measured() -> None:
    """``NULL`` everywhere is "these rows cannot say", never a delta of zero."""
    frame = learnable_frame()
    predictor = learnable_predictor(frame)
    result = permutation_importance(
        predictor, frame, classes=CLASSES, policy=ImportancePolicy(**FAST)
    )
    assert result.measured("iv") is False
    assert "NULL" in (result.blocked["iv"] or "")
    assert result.summary("iv")["measured"] is False
    assert result.summary("iv")["mean"] is None
    assert np.isnan(result.values("iv")).all()
    assert "iv" not in [name for name, _ in result.ranking()]


def test_a_constant_column_is_not_measured() -> None:
    frame = learnable_frame()
    predictor = learnable_predictor(frame)
    result = permutation_importance(
        predictor, frame, classes=CLASSES, policy=ImportancePolicy(**FAST)
    )
    assert result.measured("candle_inside_flag") is False
    assert "single value" in (result.blocked["candle_inside_flag"] or "")


def test_permuting_a_categorical_column_keeps_the_models_vocabulary() -> None:
    """The regression that cost a run: a shuffled categorical must stay a categorical.

    An object array assigned into the frame silently downgrades the column, and LightGBM then
    refuses the matrix ("categorical_feature do not match") - or reads the levels as strings in
    an order nobody chose.
    """
    from core.ml.importance import _permuted_column

    frame = learnable_frame()
    predictor = learnable_predictor(frame)
    matrix = predictor.matrix(frame)
    column = matrix["nifty_regime"]
    assert str(column.dtype) == "category"
    order = np.arange(len(column))[::-1]
    permuted = _permuted_column(column, column.to_numpy(), order)
    assert str(permuted.dtype) == "category"
    assert list(permuted.cat.categories) == list(column.cat.categories)
    assert permuted.to_numpy()[0] == column.to_numpy()[-1]
    # and the measurement itself survives it: the categorical is measured, not blocked
    result = permutation_importance(
        predictor, frame, classes=CLASSES, policy=ImportancePolicy(**FAST)
    )
    assert result.measured("nifty_regime") is True
    assert all(np.isfinite(result.values("nifty_regime")))


def test_accuracy_is_a_documented_alternative_metric() -> None:
    """Whatever the rule, a larger delta must keep meaning "more information lost"."""
    frame = learnable_frame()
    predictor = learnable_predictor(frame)
    result = permutation_importance(
        predictor, frame, classes=CLASSES, policy=ImportancePolicy(metric="accuracy", **FAST)
    )
    assert result.metric == "accuracy"
    assert result.direction == "higher_is_better"
    assert 0.0 <= result.baseline <= 1.0
    ranked = [name for name, _ in result.ranking()]
    assert ranked.index("option_atr") < ranked.index("nifty_trend_persistence")
    assert result.summary("option_atr")["mean"] > 0


def test_an_empty_frame_is_refused() -> None:
    frame = learnable_frame()
    with pytest.raises(ImportanceError, match="no rows to permute"):
        permutation_importance(
            learnable_predictor(frame),
            frame.iloc[0:0],
            classes=CLASSES,
            policy=ImportancePolicy(**FAST),
        )


def test_a_sample_without_class_codes_is_refused() -> None:
    frame = learnable_frame().drop(columns=[CODE_COLUMN])
    with pytest.raises(ImportanceError, match="label_code"):
        permutation_importance(
            learnable_predictor(learnable_frame()),
            frame,
            classes=CLASSES,
            policy=ImportancePolicy(**FAST),
        )


def test_an_unknown_metric_is_refused_by_the_scorer_too() -> None:
    with pytest.raises(ImportanceError, match="unknown importance metric"):
        metric_score("f1", np.array([0]), np.array([[1.0]]), classes=("only",))


def test_a_feature_the_measurement_does_not_carry_cannot_be_asked_about() -> None:
    frame = learnable_frame()
    result = permutation_importance(
        learnable_predictor(frame), frame, classes=CLASSES, policy=ImportancePolicy(**FAST)
    )
    with pytest.raises(ImportanceError, match="not a feature of this measurement"):
        result.values("nope")


# ---------------------------------------------------------------------------
# SHAP: optional in the strongest sense
# ---------------------------------------------------------------------------
class _StubExplainer:
    """A stand-in for ``shap.TreeExplainer`` that returns whatever the test asks for."""

    def __init__(self, values, *, explode: str | None = None) -> None:
        self.values = values
        self.explode = explode

    def shap_values(self, matrix: pd.DataFrame) -> object:
        if self.explode:
            raise RuntimeError(self.explode)
        return self.values


def install_shap(monkeypatch, *, values=None, explode: str | None = None) -> list[int]:
    """Install a stub ``shap`` module and return the list of ``TreeExplainer`` calls."""
    calls: list[int] = []
    module = types.ModuleType("shap")

    def TreeExplainer(model) -> _StubExplainer:  # noqa: N802 - the library's own name
        calls.append(1)
        return _StubExplainer(values, explode=explode)

    module.TreeExplainer = TreeExplainer
    monkeypatch.setitem(sys.modules, "shap", module)
    return calls


def test_shap_off_never_imports_the_package(monkeypatch) -> None:
    """``off`` is a promise about the environment, not a hint."""
    calls = install_shap(monkeypatch, values=np.zeros((1, 1)))
    frame = learnable_frame()
    result = shap_importance(
        learnable_predictor(frame), frame, policy=ImportancePolicy(shap="off")
    )
    assert result.status == SHAP_SKIPPED
    assert result.measured is False
    assert calls == []
    assert "off" in (result.reason or "")


def test_shap_auto_without_the_package_records_the_reason(monkeypatch) -> None:
    """An absent optional measurement is reported with the way to get it, never faked."""
    monkeypatch.delitem(sys.modules, "shap", raising=False)
    frame = learnable_frame()
    result = shap_importance(
        learnable_predictor(frame), frame, policy=ImportancePolicy(shap="auto")
    )
    assert result.status == SHAP_UNAVAILABLE
    assert ".[explain]" in (result.reason or "")
    assert result.mean_absolute == {}


def test_shap_on_without_the_package_is_a_configuration_error(monkeypatch) -> None:
    """``on`` means the operator asked for it, so a missing package must stop the run."""
    monkeypatch.delitem(sys.modules, "shap", raising=False)
    frame = learnable_frame()
    with pytest.raises(ImportanceError, match="optional extra"):
        shap_importance(learnable_predictor(frame), frame, policy=ImportancePolicy(shap="on"))


def test_shap_measured_with_a_stub_explainer(monkeypatch) -> None:
    """When it does run, the attribution is aggregated explicitly and per class."""
    frame = learnable_frame(rows=100)
    predictor = learnable_predictor(frame)
    columns = predictor.feature_columns
    values = np.zeros((len(frame), len(columns)))
    values[:, 0] = 2.0
    values[:, 1] = 0.5
    install_shap(monkeypatch, values=values)
    result = shap_importance(predictor, frame, policy=ImportancePolicy(shap="auto"))
    assert result.status == SHAP_MEASURED
    assert result.rows == len(frame)
    assert result.classes == len(CLASSES)
    assert result.mean_absolute["option_atr"] == pytest.approx(2.0)
    assert result.mean_absolute["nifty_trend_persistence"] == pytest.approx(0.5)
    assert set(result.per_class) == set(CLASSES)
    payload = result.to_dict()
    assert payload["status"] == "measured"
    assert list(payload["mean_absolute_shap"])[0] == "option_atr"


def test_shap_understands_a_three_dimensional_attribution(monkeypatch) -> None:
    """``shap``'s layout has moved between releases, so the shape is inspected, not assumed."""
    frame = learnable_frame(rows=60)
    predictor = learnable_predictor(frame)
    columns = predictor.feature_columns
    per_class = np.zeros((len(CLASSES), len(frame), len(columns)))
    per_class[0, :, 0] = 1.0
    per_class[1, :, 1] = 4.0
    install_shap(monkeypatch, values=per_class)
    result = shap_importance(predictor, frame, policy=ImportancePolicy(shap="auto"))
    assert result.status == SHAP_MEASURED
    assert result.per_class[CLASSES[1]]["nifty_trend_persistence"] == pytest.approx(4.0)
    assert result.mean_absolute["option_atr"] == pytest.approx(1.0 / len(CLASSES))


def test_a_shap_shape_that_cannot_be_interpreted_is_a_failure_not_a_guess(monkeypatch) -> None:
    frame = learnable_frame(rows=40)
    predictor = learnable_predictor(frame)
    install_shap(monkeypatch, values=np.zeros((3, len(frame), 3)))
    result = shap_importance(predictor, frame, policy=ImportancePolicy(shap="auto"))
    assert result.status == SHAP_FAILED
    assert "shape" in (result.reason or "")
    assert result.mean_absolute == {}


def test_a_shap_failure_never_fails_the_run(monkeypatch) -> None:
    """The optional measurement cannot take the required ones down with it."""
    frame = learnable_frame(rows=40)
    install_shap(monkeypatch, explode="no GPU today")
    result = shap_importance(
        learnable_predictor(frame), frame, policy=ImportancePolicy(shap="on")
    )
    assert result.status == SHAP_FAILED
    assert "no GPU today" in (result.reason or "")


# ---------------------------------------------------------------------------
# The report: ranks, sources and the caveats that belong to the answer
# ---------------------------------------------------------------------------
def report_for(*, policy=None, sample=None, shap=None):
    """A report measured over the learnable frame (all five columns of it)."""
    frame = learnable_frame(rows=200)
    predictor = learnable_predictor(frame)
    chosen = policy or ImportancePolicy(**FAST)
    result = permutation_importance(predictor, frame, classes=CLASSES, policy=chosen)
    native = {
        name: float(len(predictor.feature_columns) - index)
        for index, name in enumerate(predictor.feature_columns)
    }
    report = build_importance_report(
        policy=chosen,
        model={
            "path": "model.txt",
            "algorithm": "lightgbm",
            "horizon_minutes": 5,
            "results_label": "EXPLORATORY",
        },
        dataset={"rows": len(frame), "days": 1},
        classes=CLASSES,
        sample=sample or sample_of(frame),
        native=native,
        permutation=result,
        shap=shap,
        notes=("a note",),
    )
    return report, result, frame


def test_the_unmapped_bucket_holds_every_feature_no_source_claims() -> None:
    """A feature no configured source claims must be *visible* as exactly that."""
    from core.ml.importance import _sources

    policy = ImportancePolicy(
        repeats=3,
        max_rows=100_000,
        topics=(ImportanceTopic("iv_only", "IV only", _sources((), ("iv",))),),
    )
    report, result, _ = report_for(policy=policy)
    assert report.counts["features"] == len(result.columns)
    assert report.counts["unmapped_features"] == len(result.columns) - 1
    bucket = report.topic(UNMAPPED_TOPIC_KEY)
    assert bucket is not None
    assert bucket.measured is True
    assert bucket.features == len(result.columns) - 1
    assert "iv" not in [feature.column for feature in report.features if feature.topic == UNMAPPED_TOPIC_KEY]
    assert report.topic("iv_only") is not None
    assert report.topic("iv_only").features == 1


def test_a_source_with_no_feature_in_the_model_says_exactly_that() -> None:
    """Spec 27 asks about the greeks and OI; if a model carries none, the report must say so."""
    report, _, _ = report_for()
    for key in ("greeks", "oi", "time_to_expiry"):
        topic = report.topic(key)
        assert topic is not None
        assert topic.measured is False
        assert topic.features == 0
        assert "not here to measure" in (topic.reason or "")
        assert topic.permutation_mean is None and topic.rank is None


def test_a_source_whose_features_are_all_unmeasurable_says_why() -> None:
    from core.ml.importance import _sources

    policy = ImportancePolicy(
        repeats=3,
        max_rows=100_000,
        topics=(
            ImportanceTopic("always_null", "always NULL", _sources((), ("iv",))),
            ImportanceTopic("never_moves", "constant", _sources((), ("candle_inside_flag",))),
        ),
    )
    report, _, _ = report_for(policy=policy)
    for key in ("always_null", "never_moves"):
        topic = report.topic(key)
        assert topic is not None
        assert topic.measured is False
        assert topic.measured_features == 0
        assert topic.features == 1
        assert "could be measured" in (topic.reason or "")


def test_a_source_total_is_the_sum_of_its_features_with_its_own_interval() -> None:
    """Summing the repeats rather than the means is what makes a source's interval honest."""
    from core.ml.importance import _sources

    policy = ImportancePolicy(
        repeats=3,
        max_rows=100_000,
        topics=(
            ImportanceTopic("pair", "signal and noise", _sources((), ("option_atr", "nifty_trend_persistence"))),
            ImportanceTopic("regime_only", "nifty_regime", _sources((), ("nifty_regime",))),
        ),
    )
    report, result, _ = report_for(policy=policy)
    pair = report.topic("pair")
    assert pair is not None and pair.measured is True
    assert pair.measured_features == 2
    expected = result.summary("option_atr")["mean"] + result.summary("nifty_trend_persistence")["mean"]
    assert pair.permutation_mean == pytest.approx(expected)
    assert pair.ci_low <= pair.permutation_mean <= pair.ci_high
    assert pair.rank == 1  # the pair carries more than the regime alone
    assert report.features[0].native_rank == 1


def test_the_three_structural_caveats_are_always_there() -> None:
    """They are part of the answer, not decoration a caller may forget."""
    report, _, _ = report_for()
    assert list(report.caveats[:3]) == [
        IMPORTANCE_CAVEAT,
        CORRELATED_FEATURES_CAVEAT,
        measurement_caveat(sample_of(learnable_frame(rows=10))),
    ]
    assert report.notes == ("a note",)


def test_a_feature_the_sample_could_not_measure_keeps_its_reason_and_no_rank() -> None:
    report, result, frame = report_for()
    by_column = {feature.column: feature for feature in report.features}
    assert by_column["iv"].measured is False
    assert by_column["iv"].rank is None
    assert by_column["iv"].null_share == pytest.approx(1.0)
    assert by_column["iv"].reason == result.blocked["iv"]
    assert by_column["candle_inside_flag"].constant is True
    assert "iv" not in [feature.column for feature in report.ranked()]
    assert report.counts["unmeasured"] == 2
    assert len(frame) == report.counts["rows"]


def test_the_markdown_prints_the_sources_the_intervals_and_the_caveats() -> None:
    """One artefact a human reads: the ranked sources, the features, then what it is not."""
    report, result, _ = report_for()
    text = report.to_markdown()
    assert f"# ANJANEYA feature importance (spec {report.policy.spec})" in text
    assert "## Sources (spec 27's question)" in text
    assert "## Every feature" in text
    assert "## SHAP (optional, spec 27)" in text
    assert "## Caveats" in text
    assert IMPORTANCE_CAVEAT in text
    assert "Sources this run could not measure:" in text  # the shipped topics, all empty here
    assert "Features the sample could not measure (never silently zero):" in text
    assert result.blocked["iv"] in text
    assert "status: not requested" in text
    top = report.ranked(1)[0]
    assert f"`{top.column}`" in text
    assert f"| {top.rank} |" in text


def test_the_report_serialises_to_json() -> None:
    """The JSON artefact is what a later stage (spec 28's charts) reads."""
    import json

    report, _, _ = report_for()
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert set(payload) >= {
        "spec",
        "policy",
        "model",
        "dataset",
        "sample",
        "classes",
        "measurement",
        "shap",
        "counts",
        "topics",
        "features",
        "caveats",
    }
    assert payload["spec"] == "27"
    assert payload["features"][0]["permutation"]["measured"] in {True, False}




