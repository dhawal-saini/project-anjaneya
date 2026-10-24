"""Accumulation-report artefacts: spec 29's report as JSON, as Markdown, and as a flat CSV.

One :class:`~core.accumulation.AccumulationReport` is written in the three shapes its three
readers want, from one in-memory object (so the three files can never disagree):

* ``data/reports/accumulation/<stem>_accumulation.json`` - the whole report: the policy, the
  gate, every table's sample size, the coverage ledger by day, the weekly captured-day series,
  the readiness checklist and the caveats. This is what a later stage (or a notebook) loads.
* ``data/reports/accumulation/<stem>_accumulation.md`` - the same report as the five sections
  phase 25 names, with the caveats printed under them rather than kept in a commit message.
* ``data/reports/accumulation/<stem>_tables.csv`` - one row per stored table, so a spreadsheet
  or a small script can chart the accumulation without parsing Markdown.

The stem is the day the report describes (``accumulation_2026-10-01``), so a weekly cadence
leaves one dated file per run and the series is the history. Every write is atomic (temp file,
then rename) and a failure is counted rather than raised: a convenience file must never
invalidate a run that already did its work.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from core.accumulation import AccumulationReport

__all__ = [
    "TABLE_COLUMNS",
    "AccumulationWriteStats",
    "AccumulationWriter",
    "accumulation_stem",
    "tables_frame",
]

#: The CSV's column order: the table, its size and its span.
TABLE_COLUMNS: tuple[str, ...] = ("table", "rows", "days", "first", "last", "unavailable")


def accumulation_stem(as_of: date) -> str:
    """The artefact stem for one report: ``accumulation_<YYYY-MM-DD>``."""
    return f"accumulation_{as_of.isoformat()}"


@dataclass(slots=True)
class AccumulationWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    tables: int = 0
    coverage_days: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "tables": self.tables,
            "coverage_days": self.coverage_days,
            "files": self.files,
            "failures": self.failures,
        }


def tables_frame(report: AccumulationReport) -> pd.DataFrame:
    """One flat row per stored table (an unavailable table keeps its reason, never a 0 count)."""
    rows = [
        {
            "table": profile.table,
            "rows": profile.rows,
            "days": profile.days,
            "first": profile.first,
            "last": profile.last,
            "unavailable": profile.unavailable,
        }
        for profile in report.tables
    ]
    frame = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    # ``days`` is nullable (a count-only table has no timestamp), so keep it an integer column
    # with real blanks rather than promoting every row to a float.
    frame["days"] = frame["days"].astype("Int64")
    return frame


class AccumulationWriter:
    """Writes spec 29's report under ``data/reports/accumulation/`` (atomic, counted)."""

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
        self.stats = AccumulationWriteStats()

    # -- paths ---------------------------------------------------------------
    def json_path(self, stem: str) -> Path:
        """The JSON artefact's path."""
        return self.root / f"{stem}_accumulation.json"

    def markdown_path(self, stem: str) -> Path:
        """The Markdown artefact's path."""
        return self.root / f"{stem}_accumulation.md"

    def tables_path(self, stem: str) -> Path:
        """The per-table CSV's path."""
        return self.root / f"{stem}_tables.csv"

    # -- the write -----------------------------------------------------------
    def write(self, report: AccumulationReport, *, stem: str) -> list[Path]:
        """Write all three artefacts; return the paths that actually landed."""
        frame = tables_frame(report)
        self.stats.tables = len(report.tables)
        self.stats.coverage_days = len(report.coverage_days)

        written: list[Path] = []
        for path in (
            self._write_text(self.json_path(stem), _json_text(report, self.run_id, self.config_hash)),
            self._write_text(self.markdown_path(stem), report.to_markdown()),
            self._write_csv(frame, stem),
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
                f"writing the accumulation report {target.name} failed "
                f"({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _write_csv(self, frame: pd.DataFrame, stem: str) -> Path | None:
        """Write the per-table CSV atomically (temp file, then rename)."""
        target = self.tables_path(stem)
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
                f"writing the accumulation tables {target.name} failed "
                f"({type(exc).__name__}: {exc})",
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


def _json_text(report: AccumulationReport, run_id: str, config_hash: str | None) -> str:
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
