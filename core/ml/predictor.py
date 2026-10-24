"""Scoring a stored model artefact (spec 21's ``predictor``).

Spec 21 lists a ``predictor`` beside the ``dataset_builder``, the ``splitter``, the
``trainer`` and the ``evaluator``. This is that component: load a model that was written to
disk (LightGBM's own model text plus the metadata JSON beside it), rebuild the feature
matrix exactly as the trainer built it, and return one probability distribution per row.

A model artefact is a **pair**, and this module refuses to score half of one:

* the ``.txt`` is LightGBM's own bytes - loadable without ANJANEYA, which is why a number in
  a report can still be audited years later;
* the ``_metadata.json`` carries the feature *order* and the categorical *vocabulary*. A
  matrix built in a different column order, or a categorical level cast to a different
  integer code, silently turns a prediction into noise - the failure mode this module
  exists to make impossible.

Two rules follow from that, and both are tested:

* a feature column the metadata names but the frame does not carry is a refusal, not a
  zero;
* a categorical level the model never saw becomes **missing**, never a new code (the same
  rule the training stage's baselines use).

Scoring is deterministic: the class order comes from the metadata, the vocabulary from the
metadata, and LightGBM's prediction for a fixed matrix has no randomness in it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.ml.trainer import TrainingError, feature_matrix

__all__ = [
    "ARTEFACT_METADATA_SUFFIX",
    "PREDICTION_COLUMNS",
    "ModelArtefact",
    "ModelError",
    "ModelPredictor",
    "metadata_path_for",
    "predictor_columns",
]

#: A model's metadata sits beside its booster text under this suffix.
ARTEFACT_METADATA_SUFFIX = "_metadata.json"

#: The columns :meth:`ModelPredictor.predict_frame` adds to the class probabilities.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "predicted_class",
    "predicted_probability",
    "runner_up_class",
    "runner_up_probability",
    "margin",
)


class ModelError(RuntimeError):
    """Raised when an artefact cannot be loaded, or a frame cannot be scored against it."""


def predictor_columns(classes: Sequence[str]) -> tuple[str, ...]:
    """``('p_UP_CONTINUATION', ...)`` - the probability columns for ``classes``."""
    return tuple(f"p_{name}" for name in classes)


def metadata_path_for(model_path: str | Path) -> Path:
    """Where the metadata of ``model_path`` lives (``<stem>_metadata.json``)."""
    path = Path(model_path)
    return path.with_name(path.stem + ARTEFACT_METADATA_SUFFIX)


def _import_lightgbm() -> Any:
    """Import LightGBM here rather than at module import, with a message that says why."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - the dependency is declared
        raise ModelError(
            f"lightgbm is not importable ({exc}); it is a declared dependency of ANJANEYA "
            "('lightgbm>=4.3') - reinstall the environment rather than scoring without it"
        ) from exc
    return lgb



@dataclass(slots=True)
class ModelArtefact:
    """A booster plus the metadata that says how it must be fed."""

    path: Path
    metadata_path: Path | None
    metadata: dict[str, Any]
    booster: Any
    algorithm: str
    classes: tuple[str, ...]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    vocabulary: dict[str, tuple[str, ...]] = field(default_factory=dict)
    horizon: int | None = None
    timeframe: str | None = None
    dataset_name: str | None = None
    results_label: str | None = None
    written_at: str | None = None
    run_id: str | None = None
    train_rows: int | None = None

    @classmethod
    def load(
        cls, model_path: str | Path, *, metadata_path: str | Path | None = None
    ) -> ModelArtefact:
        """Load a model artefact, refusing anything that is not a complete pair."""
        path = Path(model_path)
        if not path.is_file():
            raise ModelError(f"model artefact not found: {path}")
        meta_path = Path(metadata_path) if metadata_path is not None else metadata_path_for(path)
        if not meta_path.is_file():
            raise ModelError(
                f"the model artefact's metadata is missing: {meta_path} (expected beside "
                f"{path.name}). A booster without its metadata has no feature order and no "
                "categorical vocabulary, so scoring it would be guesswork - retrain with "
                "--mode train to write the pair"
            )
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"cannot read {meta_path.name}: {exc}") from None
        block = metadata.get("model")
        if not isinstance(block, dict):
            raise ModelError(
                f"{meta_path.name} has no 'model' block; it is not an ANJANEYA model artefact"
            )
        feature_columns = tuple(str(name) for name in block.get("feature_columns") or ())
        classes = tuple(str(name) for name in block.get("classes") or ())
        if not feature_columns or not classes:
            raise ModelError(
                f"{meta_path.name} records no feature columns or no classes; the artefact is "
                "incomplete and cannot be scored"
            )
        booster = _import_lightgbm().Booster(model_file=str(path))
        vocabulary = {
            str(name): tuple(str(level) for level in levels)
            for name, levels in (block.get("categorical_vocabulary") or {}).items()
        }
        if hasattr(booster, "num_feature") and int(booster.num_feature()) != len(feature_columns):
            raise ModelError(
                f"{path.name} expects {int(booster.num_feature())} feature(s) but "
                f"{meta_path.name} names {len(feature_columns)}; the pair is out of sync and "
                "scoring it would produce noise"
            )
        dataset = metadata.get("dataset") or {}
        return cls(
            path=path,
            metadata_path=meta_path,
            metadata=metadata,
            booster=booster,
            algorithm=str(block.get("algorithm") or "unknown"),
            classes=classes,
            feature_columns=feature_columns,
            categorical_columns=tuple(
                str(name) for name in block.get("categorical_columns") or ()
            ),
            vocabulary=vocabulary,
            horizon=block.get("horizon_minutes"),
            timeframe=dataset.get("timeframe"),
            dataset_name=metadata.get("dataset_name"),
            results_label=metadata.get("results_label"),
            written_at=metadata.get("written_at"),
            run_id=metadata.get("run_id"),
            train_rows=block.get("train_rows"),
        )

    @property
    def probability_columns(self) -> tuple[str, ...]:
        """The ``p_<class>`` columns :meth:`ModelPredictor.predict_frame` emits."""
        return predictor_columns(self.classes)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - copied into every backtest summary that used the model."""
        return {
            "path": str(self.path),
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "algorithm": self.algorithm,
            "classes": list(self.classes),
            "features": len(self.feature_columns),
            "categorical_features": len(self.categorical_columns),
            "horizon_minutes": self.horizon,
            "timeframe": self.timeframe,
            "dataset_name": self.dataset_name,
            "results_label": self.results_label,
            "run_id": self.run_id,
            "written_at": self.written_at,
            "train_rows": self.train_rows,
        }


class ModelPredictor:
    """Score frames with a loaded :class:`ModelArtefact` (spec 21's ``predictor``)."""

    def __init__(self, artefact: ModelArtefact) -> None:
        self.artefact = artefact

    @classmethod
    def from_path(
        cls, model_path: str | Path, *, metadata_path: str | Path | None = None
    ) -> ModelPredictor:
        """Load the artefact at ``model_path`` and wrap it."""
        return cls(ModelArtefact.load(model_path, metadata_path=metadata_path))

    # -- the matrix ----------------------------------------------------------
    @property
    def classes(self) -> tuple[str, ...]:
        """The class order the model outputs (from its metadata, never guessed)."""
        return self.artefact.classes

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """The feature columns, in the order the model was trained on."""
        return self.artefact.feature_columns

    def missing_columns(self, frame: pd.DataFrame) -> list[str]:
        """Feature columns the metadata names that ``frame`` does not carry."""
        return [name for name in self.feature_columns if name not in frame.columns]

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The matrix as the booster must see it: training order, training vocabulary."""
        missing = self.missing_columns(frame)
        if missing:
            raise ModelError(
                f"the frame is missing {len(missing)} feature column(s) the model was trained "
                f"on: {missing[:8]}{' ...' if len(missing) > 8 else ''} - build the feature "
                "rows with the same configuration rather than substituting zeros"
            )
        try:
            matrix, _ = feature_matrix(
                frame,
                feature_columns=self.feature_columns,
                categorical_columns=self.artefact.categorical_columns,
                vocabulary=self.artefact.vocabulary,
            )
        except TrainingError as exc:
            raise ModelError(str(exc)) from None
        return matrix

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The class distribution per row, shape ``(rows, len(classes))``."""
        matrix = self.matrix(frame)
        if matrix.empty:
            return np.zeros((0, len(self.classes)), dtype="float64")
        probabilities = np.asarray(self.artefact.booster.predict(matrix), dtype="float64")
        if probabilities.ndim == 1:  # pragma: no cover - multiclass returns (n, k)
            probabilities = probabilities.reshape(-1, len(self.classes))
        if probabilities.shape[1] != len(self.classes):
            raise ModelError(
                f"the booster returned {probabilities.shape[1]} column(s) for "
                f"{len(self.classes)} class(es); the artefact and its metadata disagree"
            )
        return probabilities

    # -- the frame the backtest consumes -------------------------------------
    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame``'s keys plus one ``p_<class>`` column per class and the top pick.

        The returned frame carries ``timestamp`` and ``tradingsymbol`` so it can be joined
        back to the bars, one row per input row, in the input's order.
        """
        for key in ("timestamp", "tradingsymbol"):
            if key not in frame.columns:
                raise ModelError(
                    f"the frame has no {key!r} column, so a prediction cannot be attributed "
                    "to a bar; pass the stored feature rows"
                )
        probabilities = self.predict(frame)
        payload: dict[str, Any] = {
            "timestamp": frame["timestamp"].to_numpy(),
            "tradingsymbol": frame["tradingsymbol"].to_numpy(),
        }
        for position, name in enumerate(self.classes):
            payload[f"p_{name}"] = probabilities[:, position]
        ranked = np.argsort(-probabilities, axis=1, kind="stable") if probabilities.size else None
        if ranked is None:
            payload.update({name: np.zeros(0) for name in PREDICTION_COLUMNS[:4]})
            payload["margin"] = np.zeros(0)
            payload["predicted_class"] = np.zeros(0, dtype=object)
            payload["runner_up_class"] = np.zeros(0, dtype=object)
            return pd.DataFrame(payload)[
                ["timestamp", "tradingsymbol", *predictor_columns(self.classes), *PREDICTION_COLUMNS]
            ]
        best = ranked[:, 0]
        runner = ranked[:, 1] if len(self.classes) > 1 else ranked[:, 0]
        payload["predicted_class"] = [self.classes[int(index)] for index in best]
        payload["predicted_probability"] = probabilities[np.arange(len(best)), best]
        payload["runner_up_class"] = [self.classes[int(index)] for index in runner]
        payload["runner_up_probability"] = probabilities[np.arange(len(runner)), runner]
        payload["margin"] = (
            payload["predicted_probability"] - payload["runner_up_probability"]
        )
        return pd.DataFrame(payload)[
            ["timestamp", "tradingsymbol", *predictor_columns(self.classes), *PREDICTION_COLUMNS]
        ]

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape: the artefact, plus the columns this predictor emits."""
        payload = self.artefact.describe()
        payload["probability_columns"] = list(predictor_columns(self.classes))
        return payload

