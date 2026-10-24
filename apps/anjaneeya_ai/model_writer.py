"""Model artefacts: one booster text file per (timeframe, horizon) plus its metadata.

A model is a *frozen research artefact* in exactly the sense a dataset is: the thing a
report's numbers came from. So it is paired with the dataset it was trained on, by name:

``data/models/1m_h5_2026-09-28_2026-10-01_lightgbm.txt``
``data/models/1m_h5_2026-09-28_2026-10-01_lightgbm_metadata.json``

The ``.txt`` is LightGBM's own model text (loadable without ANJANEYA, which matters when a
number is questioned a year later). The metadata carries *everything* needed to read the
result honestly: the dataset it came from, the exact training policy and seed, the feature
and categorical vocabulary, the per-fold metrics and baselines, the aggregates, the holdout
result if one was scored, the data-gate verdict, and the caveats the run printed. Nothing
here needs the database or the network to be interpretable.

Both files are written atomically (temp file, then rename), and a failure is counted and
logged rather than raised: the in-memory model is still valid, and a convenience file must
never invalidate a run that already did its work.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.ml.trainer import TrainingReport

__all__ = ["ModelWriteStats", "ModelWriter"]


@dataclass(slots=True)
class ModelWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    models: int = 0
    folds: int = 0
    rows: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "models": self.models,
            "folds": self.folds,
            "rows": self.rows,
            "files": self.files,
            "failures": self.failures,
        }


class ModelWriter:
    """Write model artefacts under ``root`` (``data/models`` by default)."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        config_hash: str | None = None,
        mirror: bool = True,
        logger: Any = None,
    ) -> None:
        self._root = Path(root)
        self._run_id = run_id
        self._config_hash = config_hash
        self._mirror = mirror
        self._logger = logger
        self.stats = ModelWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every artefact is written into."""
        return self._root

    def model_name(self, *, dataset_name: str, algorithm: str) -> str:
        """``<dataset name>_<algorithm>`` - the model stays paired with its matrix."""
        return f"{dataset_name}_{algorithm}"

    def model_path(self, *, dataset_name: str, algorithm: str) -> Path:
        """Where the booster text goes."""
        return self._root / f"{self.model_name(dataset_name=dataset_name, algorithm=algorithm)}.txt"

    def metadata_path(self, *, dataset_name: str, algorithm: str) -> Path:
        """Where the metadata JSON goes."""
        return self._root / (
            f"{self.model_name(dataset_name=dataset_name, algorithm=algorithm)}_metadata.json"
        )

    # -- writing -------------------------------------------------------------
    def write(
        self,
        *,
        dataset_name: str,
        report: TrainingReport,
        dataset_description: Mapping[str, Any],
        dataset_artefact: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> list[Path]:
        """Write the booster and its metadata; returns the files that landed."""
        if not self._mirror or report.final_model is None:
            return []
        algorithm = report.final_model.algorithm
        written: list[Path] = []
        model_path = self._write_model(
            dataset_name=dataset_name, algorithm=algorithm, report=report
        )
        if model_path is not None:
            written.append(model_path)
        metadata_path = self._write_metadata(
            dataset_name=dataset_name,
            algorithm=algorithm,
            report=report,
            dataset_description=dataset_description,
            dataset_artefact=dataset_artefact,
            extra=extra,
        )
        if metadata_path is not None:
            written.append(metadata_path)
        if len(written) == 2:
            self.stats.models += 1
            self.stats.folds += len(report.folds)
            # The rows the artefact was fitted on (everything outside the holdout) - the
            # number a reader would otherwise have to reconstruct from the metadata.
            self.stats.rows += report.final_train_rows
        return written

    def _write_model(
        self, *, dataset_name: str, algorithm: str, report: TrainingReport
    ) -> Path | None:
        """Write LightGBM's own model text, atomically."""
        target = self.model_path(dataset_name=dataset_name, algorithm=algorithm)
        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _remove(temporary)
            report.final_model.booster.save_model(str(temporary))
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _remove(temporary)
            self._log(
                "error",
                f"writing the model artefact {target.name} failed "
                f"({type(exc).__name__}: {exc}); the run's numbers are unaffected",
            )
            return None
        self.stats.files += 1
        return target

    def _write_metadata(
        self,
        *,
        dataset_name: str,
        algorithm: str,
        report: TrainingReport,
        dataset_description: Mapping[str, Any],
        dataset_artefact: str | None,
        extra: Mapping[str, Any] | None,
    ) -> Path | None:
        """Write the JSON a reader needs to interpret (or question) the model."""
        target = self.metadata_path(dataset_name=dataset_name, algorithm=algorithm)
        temporary = target.with_name(target.name + ".tmp")
        payload = self._metadata_payload(
            dataset_name=dataset_name,
            report=report,
            dataset_description=dataset_description,
            dataset_artefact=dataset_artefact,
            extra=extra,
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _remove(temporary)
            temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _remove(temporary)
            self._log(
                "error",
                f"writing the model metadata {target.name} failed "
                f"({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _metadata_payload(
        self,
        *,
        dataset_name: str,
        report: TrainingReport,
        dataset_description: Mapping[str, Any],
        dataset_artefact: str | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Everything a reader needs, in the order they would ask for it."""
        payload: dict[str, Any] = {
            "run_id": self._run_id,
            "config_hash": self._config_hash,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_name": dataset_name,
            "dataset_artefact": dataset_artefact,
            "dataset": dict(dataset_description),
            "model": report.final_model.to_manifest(),
            "results_label": report.results_label,
            "verdict": report.verdict.to_dict() if report.verdict else None,
            "evaluation": report.evaluation.describe(),
            "split": report.split.to_dict(),
            "folds": [result.to_dict() for result in report.folds],
            "trained_folds": len(report.trained_folds),
            "skipped_folds": len(report.skipped_folds),
            "aggregates": {name: value.to_dict() for name, value in report.aggregates.items()},
            "baseline_aggregates": {
                name: {metric: value.to_dict() for metric, value in metrics.items()}
                for name, metrics in report.baseline_aggregates.items()
            },
            "final_train_rows": report.final_train_rows,
            "holdout": report.holdout.to_dict() if report.holdout else None,
            "notes": list(report.notes),
        }
        if extra:
            payload.update(dict(extra))
        return payload

    # -- internals -----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        """Log through the injected logger, tolerating a minimal double."""
        if self._logger is None:
            return
        method = getattr(self._logger, level, None) or self._logger.info
        method(message)


def _remove(paths: Iterable[Path] | Path) -> None:
    """Remove temporary files, ignoring anything that is already gone."""
    items = [paths] if isinstance(paths, Path) else list(paths)
    for path in items:
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue
