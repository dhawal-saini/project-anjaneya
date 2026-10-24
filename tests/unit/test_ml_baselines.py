"""Unit tests for the spec 21 baselines (``core/ml/baselines.py``).

A baseline is the bar a model has to clear, so the properties that matter are the ones
that keep it honest: it must be fitted on the *training* fold, it must be reproducible for
a given seed, and its probabilities must be a distribution (otherwise log loss is not
comparing like with like). The rest of the file pins the refusals - a baseline whose class
is not configured, a regime column that is not there, a frame with no rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.ml import BASELINE_NAMES, BaselineError, build_baseline, build_baselines
from core.ml.baselines import (
    ALWAYS_RANGE_CLASS,
    ALWAYS_UP_CLASS,
    BASELINE_DESCRIPTIONS,
    DEFAULT_REGIME_COLUMN,
)
from tests.support.ai import CLASSES, signal_frame


def frame() -> pd.DataFrame:
    """The synthetic frame: four classes, a regime column, and a learnable signal."""
    return signal_frame(rows=240)


def test_every_configured_baseline_is_implemented(config) -> None:
    configured = list(config.model.evaluation.baselines)
    assert configured == list(BASELINE_NAMES)
    assert set(BASELINE_DESCRIPTIONS) == set(BASELINE_NAMES)


def test_each_baseline_builds_and_returns_a_distribution() -> None:
    rows = frame()
    for baseline in build_baselines(BASELINE_NAMES, train_frame=rows, classes=CLASSES, seed=3):
        probabilities = baseline.predict_proba(rows)
        assert probabilities.shape == (len(rows), len(CLASSES))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        assert baseline.classes == CLASSES
        assert baseline.description


def test_an_unknown_baseline_is_refused() -> None:
    with pytest.raises(BaselineError, match="unknown baseline"):
        build_baseline("majority", train_frame=frame(), classes=CLASSES)


def test_an_empty_baseline_list_is_refused() -> None:
    with pytest.raises(BaselineError, match="nothing to beat"):
        build_baselines([], train_frame=frame(), classes=CLASSES)


def test_an_empty_training_frame_is_refused() -> None:
    with pytest.raises(BaselineError, match="no rows"):
        build_baseline("random", train_frame=frame().iloc[:0], classes=CLASSES)


def test_classes_must_be_unique() -> None:
    with pytest.raises(BaselineError, match="unique"):
        build_baseline("random", train_frame=frame(), classes=("RANGE", "RANGE"))


# ---------------------------------------------------------------------------
# The constant baselines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "target"),
    [("always_up", ALWAYS_UP_CLASS), ("always_range", ALWAYS_RANGE_CLASS)],
)
def test_a_constant_baseline_gives_one_class_probability_one(name: str, target: str) -> None:
    rows = frame()
    baseline = build_baseline(name, train_frame=rows, classes=CLASSES)
    probabilities = baseline.predict_proba(rows)
    index = CLASSES.index(target)
    np.testing.assert_allclose(probabilities[:, index], 1.0)
    others = [position for position in range(len(CLASSES)) if position != index]
    np.testing.assert_allclose(probabilities[:, others], 0.0)
    assert baseline.describe()["always"] == target


def test_a_constant_baseline_needs_its_class_to_be_configured() -> None:
    with pytest.raises(BaselineError, match="UP_CONTINUATION"):
        build_baseline("always_up", train_frame=frame(), classes=("RANGE", "REVERSAL"))


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------
def test_random_is_reproducible_for_a_seed() -> None:
    rows = frame()
    first = build_baseline("random", train_frame=rows, classes=CLASSES, seed=5).predict_proba(rows)
    second = build_baseline("random", train_frame=rows, classes=CLASSES, seed=5).predict_proba(rows)
    np.testing.assert_allclose(first, second)
    other = build_baseline("random", train_frame=rows, classes=CLASSES, seed=6).predict_proba(rows)
    assert not np.allclose(first, other)


def test_random_does_not_look_at_the_frame_content() -> None:
    rows = frame()
    baseline = build_baseline("random", train_frame=rows, classes=CLASSES, seed=1)
    shuffled = rows.sample(frac=1.0, random_state=9).reset_index(drop=True)
    np.testing.assert_allclose(baseline.predict_proba(rows), baseline.predict_proba(shuffled))
    assert baseline.describe()["seed"] == 1

# ---------------------------------------------------------------------------
# Regime only
# ---------------------------------------------------------------------------
def test_regime_only_learns_the_training_folds_distribution() -> None:
    rows = frame()
    baseline = build_baseline("nifty_regime_only", train_frame=rows, classes=CLASSES)
    assert baseline.regime_column == DEFAULT_REGIME_COLUMN
    assert baseline.levels == ("BEARISH", "BULLISH")
    bullish = rows.loc[rows[DEFAULT_REGIME_COLUMN] == "BULLISH"]
    observed = bullish["label"].value_counts()
    probabilities = baseline.predict_proba(bullish)
    for position, name in enumerate(CLASSES):
        # Add-one smoothing: (count + 1) / (rows + classes), i.e. a level seen once is not
        # a one-hot prediction whose log loss is clipping rather than calibration.
        expected = (float(observed.get(name, 0)) + 1.0) / (len(bullish) + len(CLASSES))
        assert baseline.table["BULLISH"][position] == pytest.approx(expected)
        assert probabilities[:, position] == pytest.approx(expected)


def test_regime_only_falls_back_to_the_prior_for_an_unseen_level() -> None:
    rows = frame()
    baseline = build_baseline("nifty_regime_only", train_frame=rows, classes=CLASSES)
    unseen = rows.iloc[:4].assign(**{DEFAULT_REGIME_COLUMN: "SIDEWAYS"})
    probabilities = baseline.predict_proba(unseen)
    for row in probabilities:
        assert list(row) == pytest.approx(list(baseline.prior))


def test_regime_only_treats_a_missing_regime_as_missing() -> None:
    rows = frame()
    baseline = build_baseline("nifty_regime_only", train_frame=rows, classes=CLASSES)
    missing = rows.iloc[:3].assign(**{DEFAULT_REGIME_COLUMN: np.nan})
    np.testing.assert_allclose(baseline.predict_proba(missing), np.tile(baseline.prior, (3, 1)))
    assert "nan" not in baseline.levels


def test_regime_only_needs_the_regime_column() -> None:
    rows = frame().drop(columns=[DEFAULT_REGIME_COLUMN])
    with pytest.raises(BaselineError, match=DEFAULT_REGIME_COLUMN):
        build_baseline("nifty_regime_only", train_frame=rows, classes=CLASSES)


def test_regime_only_smooths_a_level_seen_once() -> None:
    rows = pd.DataFrame({"label": ["RANGE"], DEFAULT_REGIME_COLUMN: ["ONLY_ONCE"]})
    baseline = build_baseline("nifty_regime_only", train_frame=rows, classes=CLASSES)
    table = baseline.table["ONLY_ONCE"]
    assert max(table) < 1.0
    assert table[CLASSES.index("RANGE")] > min(table)


def test_smoothing_must_be_positive() -> None:
    with pytest.raises(BaselineError, match="smoothing"):
        build_baseline("nifty_regime_only", train_frame=frame(), classes=CLASSES, smoothing=0.0)


def test_regime_describe_records_what_it_learned() -> None:
    baseline = build_baseline("nifty_regime_only", train_frame=frame(), classes=CLASSES)
    description = baseline.describe()
    assert description["kind"] == "regime"
    assert set(description["table"]) == set(baseline.levels)
    assert len(description["prior"]) == len(CLASSES)


def test_an_empty_frame_scores_an_empty_result() -> None:
    rows = frame()
    baseline = build_baseline("always_up", train_frame=rows, classes=CLASSES)
    assert baseline.predict_proba(rows.iloc[:0]).shape == (0, len(CLASSES))
    assert baseline.levels == ()

