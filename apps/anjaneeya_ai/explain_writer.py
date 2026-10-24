"""Explanation artefacts: spec 26's explanations as JSON, as Markdown, and their lines as CSV.

One :class:`~core.ml.explainer.ExplanationBatch` is written in the three shapes its three
readers want, from one in-memory object (so the three files can never disagree):

* ``data/reports/explanations/<stem>_explanations.json`` - the whole thing, every block of
  every explained prediction, the model's full class distribution, the row's audit block and
  the caveats. This is what a later stage (or a notebook) loads.
* ``data/reports/explanations/<stem>_explanations.md`` - the same explanations as headings and
  bullet lists, with the caveats printed under them rather than in a separate commit message.
* ``data/reports/explanations/<stem>_entries.csv`` - one row per *line* of an explanation
  (which prediction, which block, which feature column, which value), so "which feature said
  what across the last twenty predictions" is a pivot table rather than a scroll.

The stem mirrors the model artefact the predictions came from
(``1m_h5_2026-09-28_2026-10-01_lightgbm.txt`` -> ``1m_h5_2026-09-28_2026-10-01_lightgbm``), so
an explanation can always be traced back to the model that produced it. Every write is atomic
(temp file, then rename) and a failure is counted rather than raised: a convenience file must
never invalidate a run that already did its work.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.ml.explainer import ExplanationBatch

__all__ = [
    "ENTRY_COLUMNS",
    "ExplanationWriteStats",
    "ExplanationWriter",
    "entries_frame",
    "model_stem",
]

#: What a model artefact's name ends with, so an explanation can borrow its stem.
MODEL_SUFFIX = ".txt"

#: The CSV's column order: which prediction, then which line of it, then the value.
ENTRY_COLUMNS: tuple[str, ...] = (
    "tradingsymbol",
    "timestamp",
    "predicted_class",
    "predicted_probability",
    "section",
    "column",
    "group",
    "kind",
    "known",
    "text",
    "value",
)


@dataclass(slots=True)
class ExplanationWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    explanations: int = 0
    entries: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "explanations": self.explanations,
            "entries": self.entries,
            "files": self.files,
            "failures": self.failures,
        }


def model_stem(name: str | None) -> str | None:
    """The stem an explanation borrows from its model path/name, or ``None`` when unknown."""
    if not name:
        return None
    text = Path(str(name)).name
    if text.endswith(MODEL_SUFFIX):
        text = text[: -len(MODEL_SUFFIX)]
    return Path(text).stem or None


def entries_frame(batch: ExplanationBatch) -> pd.DataFrame:
    """Every line of every explanation as one flat row.

    The AI block has no feature columns, so its rows are its class probabilities: one row per
    class, with ``column`` holding the class name and ``value`` its probability. A line whose
    stored value is ``NULL`` keeps ``known=False`` and its rendered ``missing_text`` - never a
    zero.
    """
    rows: list[dict[str, Any]] = []
    for explanation in batch.explanations:
        identity = explanation.identity
        for section in explanation.sections:
            for entry in section.entries:
                rows.append(
                    {
                        "tradingsymbol": identity.get("tradingsymbol"),
                        "timestamp": identity.get("timestamp"),
                        "predicted_class": explanation.predicted_class,
                        "predicted_probability": explanation.predicted_probability,
                        "section": section.key,
                        "column": entry.column,
                        "group": entry.group,
                        "kind": entry.kind,
                        "known": entry.known,
                        "text": entry.text,
                        "value": entry.value,
                    }
                )
            for item in section.probabilities:
                rows.append(
                    {
                        "tradingsymbol": identity.get("tradingsymbol"),
                        "timestamp": identity.get("timestamp"),
                        "predicted_class": explanation.predicted_class,
                        "predicted_probability": explanation.predicted_probability,
                        "section": section.key,
                        "column": item.name,
                        "group": None,
                        "kind": "probability",
                        "known": True,
                        "text": f"{item.probability:.{batch.policy.digits}f}",
                        "value": item.probability,
                    }
                )
    return pd.DataFrame(rows, columns=list(ENTRY_COLUMNS))


class ExplanationWriter:
    """Write one batch's JSON, Markdown and CSV under ``root`` (``data/reports/explanations``)."""

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
        self.stats = ExplanationWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every artefact is written into."""
        return self._root

    def stem(self, *, model: str | None, fallback: str | None = None) -> str:
        """The artefact stem: the model's own stem, or ``explanations_<run id>``."""
        return model_stem(model) or fallback or f"explanations_{self._run_id}"

    def json_path(self, stem: str) -> Path:
        """Where the machine-readable explanations go."""
        return self._root / f"{stem}_explanations.json"

    def markdown_path(self, stem: str) -> Path:
        """Where the human-readable explanations go."""
        return self._root / f"{stem}_explanations.md"

    def csv_path(self, stem: str) -> Path:
        """Where the flat line table goes."""
        return self._root / f"{stem}_entries.csv"

    def paths(self, stem: str) -> tuple[Path, Path, Path]:
        """All three artefact paths, in the order :meth:`write` writes them."""
        return self.json_path(stem), self.markdown_path(stem), self.csv_path(stem)

    def write(self, batch: ExplanationBatch, *, stem: str) -> list[Path]:
        """Write the three artefacts; returns the files that actually landed.

        Each artefact is independent: a failure to write the CSV still leaves the JSON and the
        Markdown in place, and the failure count says so. The explanations were built before
        any of this, so nothing here can change what they say.
        """
        if not self._mirror:
            return []
        frame = entries_frame(batch)
        self.stats.explanations = batch.rows
        self.stats.entries = int(len(frame))
        written: list[Path] = []
        for path in (
            self._write_text(
                self.json_path(stem),
                _json_text(batch, run_id=self._run_id, config_hash=self._config_hash),
            ),
            self._write_text(self.markdown_path(stem), batch.to_markdown()),
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
        import os

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
                f"writing the explanations {target.name} failed ({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _write_csv(self, frame: pd.DataFrame, stem: str) -> Path | None:
        """Write the flat line table atomically (temp file, then rename)."""
        import os

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
                f"writing the explanation lines {target.name} failed "
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


def _json_text(batch: ExplanationBatch, *, run_id: str, config_hash: str | None) -> str:
    """The JSON artefact's text: the run's identity, then the batch's own document."""
    import json

    payload = {"run_id": run_id, "config_hash": config_hash, **batch.to_dict()}
    return json.dumps(payload, indent=2, default=str)


def _safe_unlink(paths: Iterable[Path] | Path) -> None:
    """Remove temporary files, ignoring anything that is already gone."""
    items = [paths] if isinstance(paths, Path) else list(paths)
    for path in items:
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue

