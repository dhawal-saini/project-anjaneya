"""Research-report artefacts: the report as JSON, as Markdown, and its cells as one CSV.

A spec 24 report is evidence, so it is written in the three shapes its three readers want,
from one in-memory object (so the three files can never disagree):

* ``data/reports/research/<stem>_report.json`` - the whole thing, machine-readable, samples
  and caveats included. This is what a later stage (or a notebook) loads.
* ``data/reports/research/<stem>_report.md`` - the ten items as tables a human reads first,
  with the caveats printed under them rather than in a separate commit message.
* ``data/reports/research/<stem>_cells.csv`` - one row per matrix cell, joined to its
  significance verdict, for a spreadsheet or a pivot table. A cell that was never tested
  carries empty p-value columns and its PRELIMINARY flag, never a zero.

The stem mirrors the ledger it came from (``1m_h5_2026-09-28_2026-10-01_ledger.parquet`` ->
``1m_h5_2026-09-28_2026-10-01_report.md``), so a report can always be traced back to the
trades it describes. Every write is atomic (temp file, then rename) and a failure is counted
rather than raised: a convenience file must never invalidate a run that already did its work.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.research.report import ResearchReport

__all__ = ["CELL_COLUMNS", "ResearchWriteStats", "ResearchWriter", "cells_frame", "ledger_stem"]

#: What a ledger artefact's name ends with, so a report can borrow its stem.
LEDGER_SUFFIX = "_ledger.parquet"

#: The CSV's column order: identification, the cell's own numbers, then its verdict.
CELL_COLUMNS: tuple[str, ...] = (
    "matrix",
    "row",
    "column",
    "sample",
    "preliminary",
    "total_trades",
    "winners",
    "losers",
    "win_rate",
    "total_gross_pnl",
    "total_costs",
    "total_net_pnl",
    "profit_factor",
    "expectancy",
    "average_holding_minutes",
    "max_drawdown",
    "baseline_sample",
    "win_rate_p_value",
    "win_rate_p_adjusted",
    "pnl_p_value",
    "pnl_p_adjusted",
    "verdict",
)


@dataclass(slots=True)
class ResearchWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    cells: int = 0
    files: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {"cells": self.cells, "files": self.files, "failures": self.failures}


def ledger_stem(name: str | None) -> str:
    """The stem a report borrows from its ledger path/name, or ``research`` when unknown."""
    if not name:
        return "research"
    text = Path(str(name)).name
    if text.endswith(LEDGER_SUFFIX):
        return text[: -len(LEDGER_SUFFIX)]
    return Path(text).stem or "research"


def cells_frame(report: ResearchReport) -> pd.DataFrame:
    """Every matrix cell as one flat row, with its significance verdict joined on.

    The join is by ``(matrix, row, column)`` - the same labels the matrix and the test used -
    so a cell cannot pick up another cell's p-value. The verdict columns are ``None`` (empty
    in CSV) for a cell that was never tested, which is the honest representation of "we did
    not look" as opposed to ``p = 1.0``.
    """
    verdicts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for summary in report.significance:
        for test in summary.tests:
            verdicts[(test.matrix, test.row_label, test.column_label)] = test.to_dict()
    rows: list[dict[str, Any]] = []
    for matrix in report.matrices:
        for cell in matrix.rows():
            key = (matrix.name, str(cell["row"]), str(cell["column"]))
            verdict = verdicts.get(key, {})
            rows.append(
                {
                    "matrix": matrix.name,
                    "row": cell["row"],
                    "column": cell["column"],
                    "sample": cell["sample"],
                    "preliminary": cell["preliminary"],
                    "total_trades": cell["total_trades"],
                    "winners": cell["winners"],
                    "losers": cell["losers"],
                    "win_rate": cell["win_rate"],
                    "total_gross_pnl": cell["total_gross_pnl"],
                    "total_costs": cell["total_costs"],
                    "total_net_pnl": cell["total_net_pnl"],
                    "profit_factor": cell["profit_factor"],
                    "expectancy": cell["expectancy"],
                    "average_holding_minutes": cell["average_holding_minutes"],
                    "max_drawdown": cell["max_drawdown"],
                    "baseline_sample": verdict.get("baseline_sample"),
                    "win_rate_p_value": verdict.get("win_rate_p_value"),
                    "win_rate_p_adjusted": verdict.get("win_rate_p_adjusted"),
                    "pnl_p_value": verdict.get("pnl_p_value"),
                    "pnl_p_adjusted": verdict.get("pnl_p_adjusted"),
                    "verdict": verdict.get("verdict"),
                }
            )
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


class ResearchWriter:
    """Write one report's JSON, Markdown and CSV under ``root`` (``data/reports/research``)."""

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
        self.stats = ResearchWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every artefact is written into."""
        return self._root

    def stem(self, *, ledger: str | None, fallback: str | None = None) -> str:
        """The artefact stem: the ledger's own stem, or ``research_<run id>`` when unknown."""
        if ledger:
            return ledger_stem(ledger)
        return fallback or f"research_{self._run_id}"

    def json_path(self, stem: str) -> Path:
        """Where the machine-readable report goes."""
        return self._root / f"{stem}_report.json"

    def markdown_path(self, stem: str) -> Path:
        """Where the human-readable report goes."""
        return self._root / f"{stem}_report.md"

    def csv_path(self, stem: str) -> Path:
        """Where the flat cell table goes."""
        return self._root / f"{stem}_cells.csv"

    def paths(self, stem: str) -> tuple[Path, Path, Path]:
        """All three artefact paths, in the order :meth:`write` writes them."""
        return self.json_path(stem), self.markdown_path(stem), self.csv_path(stem)

    def write(self, report: ResearchReport, *, stem: str) -> list[Path]:
        """Write the three artefacts; returns the files that actually landed.

        Each artefact is independent: a failure to write the CSV still leaves the JSON and the
        Markdown in place, and the failure count says so. The report's numbers were computed
        before any of this, so nothing here can change an answer.
        """
        if not self._mirror:
            return []
        frame = cells_frame(report)
        self.stats.cells = int(len(frame))
        written: list[Path] = []
        for path in (
            self._write_text(self.json_path(stem), _json_text(report, run_id=self._run_id, config_hash=self._config_hash)),
            self._write_text(self.markdown_path(stem), report.to_markdown()),
        ):
            if path is not None:
                written.append(path)
        csv_path = self._write_csv(frame, stem)
        if csv_path is not None:
            written.append(csv_path)
        return written

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
                f"writing the research report {target.name} failed "
                f"({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target

    def _write_csv(self, frame: pd.DataFrame, stem: str) -> Path | None:
        """Write the flat cell table atomically (temp file, then rename)."""
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
                f"writing the research cells {target.name} failed ({type(exc).__name__}: {exc})",
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


def _json_text(report: ResearchReport, *, run_id: str, config_hash: str | None) -> str:
    """The JSON artefact's text: the run's identity, then the report's own document."""
    import json

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



#: The CSV's column order: identification, the cell's own numbers, then its verdict.
CELL_COLUMNS: tuple[str, ...] = (
    "matrix",
    "row",
    "column",
    "sample",
    "preliminary",
    "total_trades",
    "winners",
    "losers",
    "win_rate",
    "total_gross_pnl",
    "total_costs",
    "total_net_pnl",
    "profit_factor",
    "expectancy",
    "average_holding_minutes",
    "max_drawdown",
    "baseline_sample",
    "win_rate_p_value",
    "win_rate_p_adjusted",
    "pnl_p_value",
    "pnl_p_adjusted",
    "verdict",
)
