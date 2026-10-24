"""Unit tests for the model predictor (``core/ml/predictor.py``, spec 21).

Scoring a *stored* model is where a backtest's probabilities come from, and there are exactly
two ways to get them wrong - both silent:

* **the wrong matrix.** A frame whose columns are in a different order, or whose categorical
  levels are cast to different integer codes, produces confident noise. The predictor
  rebuilds the training order and the training vocabulary, and refuses a frame that is
  missing a feature rather than substituting a zero.
* **half an artefact.** A booster with no metadata has no feature order and no vocabulary, so
  loading one is refused; so is a pair whose feature counts disagree.

The last group proves the fixture model actually learned its (``option_type`` -> class)
signal, because the backtest integration tests depend on that being true.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.ml import ModelArtefact, ModelError, ModelPredictor, metadata_path_for, predictor_columns
from tests.support.ai import CLASSES, categorical_columns, feature_frame, write_signal_model


@pytest.fixture()
def model_path(tmp_path):
    """A real artefact pair: a fitted booster plus the metadata ``--mode train`` writes."""
    return write_signal_model(tmp_path / "models")


@pytest.fixture()
def predictor(model_path) -> ModelPredictor:
    return ModelPredictor.from_path(model_path)


# ---------------------------------------------------------------------------
# Loading the pair
# ---------------------------------------------------------------------------
def test_loading_a_written_artefact_brings_its_order_and_vocabulary(predictor, model_path) -> None:
    """The metadata is what makes the booster interpretable, so it is loaded with it."""
    artefact = predictor.artefact
    assert artefact.path == model_path
    assert artefact.metadata_path == metadata_path_for(model_path)
    assert artefact.classes == CLASSES
    assert artefact.horizon == 5
    assert artefact.algorithm == "lightgbm"
    assert artefact.train_rows and artefact.train_rows > 0
    assert len(predictor.feature_columns) > 0
    assert set(artefact.vocabulary) <= set(artefact.categorical_columns)
    assert predictor.describe()["probability_columns"] == list(predictor_columns(CLASSES))


def test_a_booster_without_its_metadata_is_refused(tmp_path, model_path) -> None:
    """No metadata means no feature order, so scoring would be guesswork."""
    lonely = tmp_path / "lonely.txt"
    lonely.write_bytes(model_path.read_bytes())
    with pytest.raises(ModelError, match="metadata is missing"):
        ModelArtefact.load(lonely)


def test_a_missing_booster_is_refused(tmp_path) -> None:
    """A metadata file alone is not a model."""
    with pytest.raises(ModelError, match="not found"):
        ModelArtefact.load(tmp_path / "nowhere.txt")


def test_a_metadata_file_that_is_not_an_artefact_is_refused(tmp_path, model_path) -> None:
    """A JSON file with no ``model`` block cannot say how to feed the booster."""
    metadata_path = metadata_path_for(model_path)
    metadata_path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    with pytest.raises(ModelError, match="not an ANJANEYA model artefact"):
        ModelArtefact.load(model_path)


def test_a_pair_that_is_out_of_sync_is_refused(tmp_path, model_path) -> None:
    """If the metadata names a feature the booster has never seen, the pair is broken."""
    metadata_path = metadata_path_for(model_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["model"]["feature_columns"] = [*payload["model"]["feature_columns"], "extra_feature"]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelError, match="out of sync"):
        ModelArtefact.load(model_path)


# ---------------------------------------------------------------------------
# Scoring a frame
# ---------------------------------------------------------------------------
def test_predict_frame_keeps_the_keys_and_adds_one_column_per_class(predictor) -> None:
    """The frame the backtest consumes: keys, the distribution, and the model's top pick."""
    stored = feature_frame()
    scored = predictor.predict_frame(stored)
    assert len(scored) == len(stored)
    assert list(scored["timestamp"]) == list(stored["timestamp"])
    assert list(scored["tradingsymbol"]) == list(stored["tradingsymbol"])
    for name in CLASSES:
        assert f"p_{name}" in scored.columns
    for name in ("predicted_class", "predicted_probability", "runner_up_class", "margin"):
        assert name in scored.columns
    totals = scored[[f"p_{name}" for name in CLASSES]].to_numpy(dtype="float64").sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)
    assert scored["margin"].min() >= 0.0


def test_a_frame_missing_a_feature_column_is_refused(predictor) -> None:
    """A zero is not a feature value: the run says which column is missing."""
    stored = feature_frame()
    column = predictor.feature_columns[0]
    with pytest.raises(ModelError, match="missing"):
        predictor.predict_frame(stored.drop(columns=column))


def test_a_frame_without_the_join_keys_is_refused(predictor) -> None:
    """A prediction that cannot be attributed to a bar is useless to a backtest."""
    stored = feature_frame()
    with pytest.raises(ModelError, match="timestamp"):
        predictor.predict_frame(stored.drop(columns="timestamp"))
    with pytest.raises(ModelError, match="tradingsymbol"):
        predictor.predict_frame(stored.drop(columns="tradingsymbol"))


def test_unseen_categorical_levels_become_missing_not_a_new_code(predictor) -> None:
    """The training vocabulary decides the codes; a level it never saw is simply absent."""
    stored = feature_frame()
    categorical = categorical_columns()[0]
    stored[categorical] = "A_LEVEL_NO_MODEL_HAS_SEEN"
    matrix = predictor.matrix(stored)
    assert matrix[categorical].isna().all()
    scored = predictor.predict_frame(stored)
    assert np.isfinite(scored[[f"p_{name}" for name in CLASSES]].to_numpy()).all()


def test_scoring_is_deterministic_and_row_aligned(predictor) -> None:
    """Same frame, same probabilities - in the frame's own row order."""
    stored = feature_frame()
    first = predictor.predict_frame(stored)
    second = predictor.predict_frame(stored)
    assert first.equals(second)


def test_the_fixture_model_actually_learned_its_signal(predictor) -> None:
    """``option_type`` decides the class in the fixture, and the scored frame shows it.

    This is the property the backtest integration tests lean on: a CE contract is predicted
    ``UP_CONTINUATION`` and a PE contract ``DOWN_CONTINUATION``, with the same probability
    every time - so a replayed premium path produces a winner and a loser, not a coin toss.
    """
    scored = predictor.predict_frame(feature_frame())
    ce = scored[scored["tradingsymbol"].str.endswith("CE")]
    pe = scored[scored["tradingsymbol"].str.endswith("PE")]
    assert len(ce) > 0 and len(pe) > 0
    assert set(ce["predicted_class"]) == {"UP_CONTINUATION"}
    assert set(pe["predicted_class"]) == {"DOWN_CONTINUATION"}
    assert ce["predicted_probability"].min() > 0.5
    assert pe["predicted_probability"].min() > 0.5

