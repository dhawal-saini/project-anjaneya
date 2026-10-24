"""Importance artefacts: spec 27's measurement as JSON, as Markdown, and as a flat CSV.

One :class:`~core.ml.importance.ImportanceReport` is written in the three shapes its three
readers want, from one in-memory object (so the three files can never disagree):

* ``data/reports/importance/<stem>_importance.json`` - the whole measurement: the policy, the
  sample it was measured on, the model's own gain, every feature's permutation delta with its
  confidence interval, the per-source totals, the optional attribution and the caveats. This is
  what a later stage (spec 28's charts, a notebook) loads.
* ``data/reports/importance/<stem>_importance.md`` - the same measurement as two tables
  (sources first, then every feature) with the caveats printed under them rather than kept in a
  commit message.
* ``data/reports/importance/<stem>_features.csv`` - one row per feature, so a chart or a
  spreadsheet can rank them without parsing Markdown.

The stem mirrors the model artefact the measurement is about
(``1m_h5_2026-09-28_2026-10-01_lightgbm.txt`` -> ``1m_h5_2026-09-28_2026-10-01_lightgbm``), so a
ranking can always be traced back to the model it describes. Every write is atomic (temp file,
then rename) and a failure is counted rather than raised: a convenience file must never
invalidate a run that already did its work.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from apps.anjaneeya_ai.explain_writer import model_stem
from core.ml.importance import ImportanceReport

__all__ = [
    "FEATURE_COLUMNS",
    "ImportanceWriteStats",
    "ImportanceWriter",
    "features_frame",
    "model_stem",
]

#: The CSV's column order: the ranking first, then the feature, then its numbers.
FEATURE_COLUMNS: tuple[str, ...] = (
    "rank",
    "column",
    "topic",
    "group",
    "measured",
    "native",
    "native_share",
    "native_rank",
    "permutation_mean",
    "permutation_std",
    "ci_low",
    "ci_high",
    "null_share",
    "constant",
    "shap",
    "reason",
)


@dataclass(slots=True)
class ImportanceWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    features: int = 0
    sources: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "features": self.features,
            "sources": self.sources,
            "files": self.files,
            "failures": self.failures,
        }


def features_frame(report: ImportanceReport) -> pd.DataFrame:
    """Every feature of the measurement as one flat row (measured ones ranked first).

    A blocked feature keeps its reason and an empty permutation cell - never a ``0``, which
    would read as "this feature carries no information" when the truth is "this sample cannot
    say".
    """
    rows: list[dict[str, Any]] = []
    ordered = list(report.ranked()) + [item for item in report.features if not item.measured]
    for item in ordered:
        rows.append(
            {
                "rank": item.rank,
                "column": item.column,
                "topic": item.topic,
                "group": item.group,
                "measured": item.measured,
                "native": item.native,
                "native_share": item.native_share,
                "native_rank": item.native_rank,
                "permutation_mean": item.permutation_mean,
                "permutation_std": item.permutation_std,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "null_share": item.null_share,
                "constant": item.constant,
                "shap": item.shap,
                "reason": item.reason,
            }
        )
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
class ImportanceWriter:
    """Write one report's JSON, Markdown and CSV under ``root`` (``data/reports/importance``)."""

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
        self.stats = ImportanceWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every artefact is written into."""
        return self._root

    def stem(self, *, model: str | None, fallback: str | None = None) -> str:
        """The artefact stem: the model's own stem, or ``importance_<run id>``."""
        return model_stem(model) or fallback or f"importance_{self._run_id}"

    def json_path(self, stem: str) -> Path:
        """Where the machine-readable measurement goes."""
        return self._root / f"{stem}_importance.json"

    def markdown_path(self, stem: str) -> Path:
        """Where the human-readable measurement goes."""
        return self._root / f"{stem}_importance.md"

    def csv_path(self, stem: str) -> Path:
        """Where the flat per-feature table goes."""
        return self._root / f"{stem}_features.csv"

    def paths(self, stem: str) -> tuple[Path, Path, Path]:
        """All three artefact paths, in the order :meth:`write` writes them."""
        return self.json_path(stem), self.markdown_path(stem), self.csv_path(stem)

    def write(self, report: ImportanceReport, *, stem: str) -> list[Path]:
        """Write the three artefacts; returns the files that actually landed.

        Each artefact is independent: a failure to write the CSV still leaves the JSON and the
        Markdown in place, and the failure count says so. The measurement was made before any of
        this, so nothing here can change what it says.
        """
        if not self._mirror:
            return []
        frame = features_frame(report)
        self.stats.features = int(report.counts.get("features", len(report.features)))
        self.stats.sources = int(report.counts.get("topics", len(report.topics)))
        written: list[Path] = []
        for path in (
            self._write_text(
                self.json_path(stem),
                _json_text(report, run_id=self._run_id, config_hash=self._config_hash),
            ),
            self._write_text(self.markdown_path(stem), report.to_markdown()),
        ):
            if path is not None:
                written.append(path)
        csv_path = self._write_csv(frame, stem)
        if csv_path is not None:
            written.append(csv_path)
        return written

    # -- atomic writes -------------------------------------------------------
    def _write_text(self, target: Path, text: str) -> Path | None:
        """Write one text artefact atomically (temp file, then rename)."""
        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error",
                f"writing the importance {target.name} failed ({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _write_csv(self, frame: pd.DataFrame, stem: str) -> Path | None:
        """Write the flat per-feature table atomically (temp file, then rename)."""
        target = self.csv_path(stem)
        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            frame.to_csv(temporary, index=False)
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error",
                f"writing the importance features {target.name} failed "
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


def _json_text(report: ImportanceReport, *, run_id: str, config_hash: str | None) -> str:
    """The JSON artefact's text: the run's identity, then the report's own document."""
    payload = {"run_id": run_id, "config_hash": config_hash, **report.to_dict()}
    return json.dumps(payload, indent=2, default=str)


def _safe_unlink(paths: Iterable[Path] | Path) -> None:
    """Remove temporary files, ignoring anything that is already gone."""
    items = [paths] if isinstance(paths, Path) else list(paths)
    for path in items:
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue
