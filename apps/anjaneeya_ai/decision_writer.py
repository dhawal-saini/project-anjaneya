"""Spec 24's decision record as files: the decisions sheet, its Markdown and its JSON.

One :class:`~core.decision.DecisionReport` is written in the three shapes its three readers
want, from one in-memory object (so the three files can never disagree):

* ``data/predictions/<stem>_decisions.json`` - the whole record: the policy, the model and its
  label, the spec 4.3 gate, the counts by reason, every decision row and the caveats. This is
  what a later stage (or a notebook) loads.
* ``data/predictions/<stem>_decisions.md`` - the same record as the three sections that make it
  readable: the policy, the counts, the entries - with the caveats printed under them rather
  than kept in a commit message.
* ``data/predictions/<stem>_decisions.csv`` - **every** decision, one row per scored bar, in
  :data:`core.decision.DECISION_COLUMNS` order, so a spreadsheet can filter on ``reason``
  without parsing Markdown. The Markdown prints the entries; the CSV keeps the whole ladder.

The stem is the model artefact's own (``1m_h5_2026-09-28_2026-10-01``), so a decision record
names the policy *and* the model that produced it, and re-running the same model with the same
policy is one file rather than an ever-growing pile. Every write is atomic (temp file, then
rename) and a failure is counted rather than raised: a convenience file must never invalidate a
run that already did its work.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.decision import DECISION_COLUMNS, DecisionReport

__all__ = [
    "DECISIONS_STEM_FALLBACK",
    "DecisionWriteStats",
    "DecisionWriter",
    "decisions_frame",
    "decisions_stem",
]

#: Used when the model artefact's name carries no stem to borrow.
DECISIONS_STEM_FALLBACK = "decisions"


def decisions_stem(model_stem: str | None, *, fallback: str = DECISIONS_STEM_FALLBACK) -> str:
    """The artefact stem for one decision record: the model's own stem, else a fallback."""
    text = (model_stem or "").strip()
    return text or fallback


@dataclass(slots=True)
class DecisionWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    decisions: int = 0
    entries: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "decisions": self.decisions,
            "entries": self.entries,
            "files": self.files,
            "failures": self.failures,
        }


def decisions_frame(report: DecisionReport) -> pd.DataFrame:
    """Every decision, one row per scored bar, in :data:`DECISION_COLUMNS` order.

    Nothing is filtered: a ``NO_TRADE`` row is as much a part of the record as an entry (it is
    the evidence for the entry rate), and its ``reason`` says which rung of the ladder stopped
    it. ``probability``/``margin`` stay blank where the ladder never got that far - a missing
    number is left missing rather than filled with a zero the model never produced.
    """
    rows: Sequence[dict[str, Any]] = [decision.to_record() for decision in report.decisions]
    frame = pd.DataFrame(list(rows), columns=list(DECISION_COLUMNS))
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


class DecisionWriter:
    """Writes spec 24's decision record under ``data/predictions/`` (atomic, counted)."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        config_hash: str | None = None,
        logger: Any = None,
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.config_hash = config_hash
        self._logger = logger
        self.stats = DecisionWriteStats()

    # -- paths ---------------------------------------------------------------
    def json_path(self, stem: str) -> Path:
        """The JSON artefact's path."""
        return self.root / f"{stem}_decisions.json"

    def markdown_path(self, stem: str) -> Path:
        """The Markdown artefact's path."""
        return self.root / f"{stem}_decisions.md"

    def csv_path(self, stem: str) -> Path:
        """The decisions CSV's path."""
        return self.root / f"{stem}_decisions.csv"

    # -- the write -----------------------------------------------------------
    def write(self, report: DecisionReport, *, stem: str) -> list[Path]:
        """Write all three artefacts; return the paths that actually landed."""
        self.stats.decisions = len(report.decisions)
        self.stats.entries = len(report.entries)
        written: list[Path] = []
        for path in (
            self._write_text(
                self.json_path(stem), _json_text(report, self.run_id, self.config_hash)
            ),
            self._write_text(self.markdown_path(stem), report.to_markdown()),
            self._write_csv(decisions_frame(report), stem),
        ):
            if path is not None:
                written.append(path)
        return written

    # -- internals -----------------------------------------------------------
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
                f"writing the decision record {target.name} failed "
                f"({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _write_csv(self, frame: pd.DataFrame, stem: str) -> Path | None:
        """Write the decisions CSV atomically (temp file, then rename)."""
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
                f"writing the decision rows {target.name} failed ({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _log(self, level: str, message: str) -> None:
        """Log through the injected logger, tolerating a minimal double."""
        if self._logger is None:
            return
        method = getattr(self._logger, level, None) or self._logger.info
        method(message)


def _json_text(report: DecisionReport, run_id: str, config_hash: str | None) -> str:
    """The JSON artefact's text: the run's identity, then the record's own document."""
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

