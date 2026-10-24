"""Dataset artefacts: one Parquet file per (timeframe, horizon) plus its split manifest.

A dataset is not a database table - it is a *frozen research artefact*: the exact matrix a
model version was trained on, with the exact split that model version was evaluated under.
So the two are written together and named after the scope they cover:

``data/datasets/1m_h5_2026-09-01_2026-09-30.parquet``
``data/datasets/1m_h5_2026-09-01_2026-09-30_splits.json``

Both are written atomically (temp file, then rename), so an interrupted run leaves the
previous artefact intact rather than a half-written one, and a failure is *counted* and
logged rather than raised: the in-memory dataset is still valid, and a convenience file
must never invalidate a run that already did its work.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ml.dataset import Dataset
from core.ml.walk_forward import DatasetSplit

__all__ = ["DatasetWriteStats", "DatasetWriter"]


@dataclass(slots=True)
class DatasetWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    datasets: int = 0
    rows: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "datasets": self.datasets,
            "rows": self.rows,
            "files": self.files,
            "failures": self.failures,
        }


class DatasetWriter:
    """Write dataset artefacts under ``root`` (``data/datasets`` by default)."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        config_hash: str | None = None,
        compression: str = "zstd",
        mirror: bool = True,
        logger: Any = None,
    ) -> None:
        self._root = Path(root)
        self._run_id = run_id
        self._config_hash = config_hash
        self._compression = compression
        self._mirror = mirror
        self._logger = logger
        self.stats = DatasetWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every artefact is written into."""
        return self._root

    def dataset_name(self, *, dataset: Dataset) -> str:
        """``<timeframe>_h<horizon>_<first day>_<last day>`` for a built dataset."""
        days = dataset.days
        first = days[0].isoformat() if days else "empty"
        last = days[-1].isoformat() if days else "empty"
        return f"{dataset.timeframe}_h{dataset.horizon}_{first}_{last}"

    def dataset_path(self, *, dataset: Dataset) -> Path:
        """Where a dataset's Parquet artefact goes."""
        return self._root / f"{self.dataset_name(dataset=dataset)}.parquet"

    def splits_path(self, *, dataset: Dataset) -> Path:
        """Where a dataset's split manifest goes."""
        return self._root / f"{self.dataset_name(dataset=dataset)}_splits.json"


    # -- writing -------------------------------------------------------------
    def write(self, *, dataset: Dataset, split: DatasetSplit) -> list[Path]:
        """Write the dataset and its split manifest; returns the files that landed."""
        if not self._mirror:
            return []
        written: list[Path] = []
        data_path = self._write_parquet(dataset=dataset)
        if data_path is not None:
            written.append(data_path)
        split_path = self._write_splits(dataset=dataset, split=split)
        if split_path is not None:
            written.append(split_path)
        if len(written) == 2:
            self.stats.datasets += 1
            self.stats.rows += dataset.rows
        return written

    def _write_parquet(self, *, dataset: Dataset) -> Path | None:
        """Write the matrix atomically (temp file, then rename)."""
        target = self.dataset_path(dataset=dataset)
        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            dataset.frame.to_parquet(temporary, compression=self._compression, index=False)
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error",
                f"writing the dataset artefact {target.name} failed "
                f"({type(exc).__name__}: {exc}); the in-memory dataset is unaffected",
            )
            return None
        self.stats.files += 1
        return target

    def _write_splits(self, *, dataset: Dataset, split: DatasetSplit) -> Path | None:
        """Write the split manifest beside the dataset (spec 22: document every split)."""
        target = self.splits_path(dataset=dataset)
        temporary = target.with_name(target.name + ".tmp")
        payload = {
            "run_id": self._run_id,
            "config_hash": self._config_hash,
            "dataset": dataset.describe(),
            "feature_columns": list(dataset.feature_columns),
            "categorical_columns": list(dataset.categorical_columns),
            "numeric_columns": list(dataset.numeric_columns),
            "split": split.to_dict(),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error",
                f"writing the split manifest {target.name} failed "
                f"({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    # -- internals -----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        """Log through the injected logger, tolerating a minimal double."""
        if self._logger is None:
            return
        method = getattr(self._logger, level, None) or self._logger.info
        method(message)


def _safe_unlink(paths: Iterable[Path] | Path) -> None:
    """Remove temporary files, ignoring anything that is already gone."""
    items = [paths] if isinstance(paths, Path) else list(paths)
    for path in items:
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue

