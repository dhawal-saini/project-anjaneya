"""Chart artefacts: spec 28's figures as image files, plus one index that names them.

Spec 28's charts are validation convenience, not evidence of record - the JSON and Markdown the
other modes write are. So this writer is deliberately light: it takes the figures a run already
built, saves each one atomically under ``data/reports/charts/``, and writes one
``<stem>_charts.md`` index that lists every figure, its file and the note the run attached to
it. The index is the thing a reader opens first.

Two choices worth recording:

* **the stem is the model's or the ledger's.** The charts describe either a model's predictions
  (the importance bars, the CE/PE curves) or a ledger's behaviour (the heatmaps). The stem
  mirrors whichever artefact drove the run, re-using
  :func:`apps.anjaneeya_ai.explain_writer.model_stem` and
  :func:`apps.anjaneeya_ai.report_writer.ledger_stem`, so a chart is always traceable to the
  thing it drew.
* **a failure is counted, never raised.** A figure that cannot be saved must not invalidate a run
  that already produced every other figure, exactly as the other writers behave.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.anjaneeya_ai.explain_writer import model_stem
from apps.anjaneeya_ai.report_writer import ledger_stem

__all__ = ["ChartWriteStats", "ChartWriter", "ledger_stem", "model_stem"]


@dataclass(slots=True)
class ChartWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    figures: int = 0
    files: int = 0
    failures: int = 0
    #: One ``(name, path)`` per written figure - what the index prints.
    written: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "figures": self.figures,
            "files": self.files,
            "failures": self.failures,
            "written": list(self.written),
        }


class ChartWriter:
    """Write a run's figures and their index under ``root`` (``data/reports/charts``)."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        config_hash: str | None = None,
        mirror: bool = True,
        logger: Any = None,
        fmt: str = "png",
        dpi: int = 120,
    ) -> None:
        self._root = Path(root)
        self._run_id = run_id
        self._config_hash = config_hash
        self._mirror = mirror
        self._logger = logger
        self._format = fmt.lstrip(".") or "png"
        self._dpi = int(dpi)
        self.stats = ChartWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every figure is written into."""
        return self._root

    @property
    def format(self) -> str:
        """The file extension every figure is written with (no dot)."""
        return self._format

    def stem(self, *, model: str | None = None, ledger: str | None = None) -> str:
        """The chart stem: the model's own, else the ledger's, else ``charts_<run id>``."""
        return (
            model_stem(model)
            or (ledger_stem(ledger) if ledger else "")
            or f"charts_{self._run_id}"
        )

    def figure_path(self, stem: str, name: str) -> Path:
        """Where one named figure goes."""
        return self._root / f"{stem}_{name}.{self._format}"

    def index_path(self, stem: str) -> Path:
        """Where the index Markdown goes."""
        return self._root / f"{stem}_charts.md"

    # -- writing -------------------------------------------------------------
    def write(
        self,
        figures: Mapping[str, Any],
        *,
        stem: str,
        notes: Mapping[str, str] | None = None,
    ) -> list[Path]:
        """Save every figure and write the index; returns the files that actually landed.

        A figure that fails to save is counted and skipped, and the index is still written with
        the figures that did land. Each figure is closed after it is saved, so a long run does
        not leak matplotlib's global figure state.
        """
        if not self._mirror:
            return []
        written: list[Path] = []
        for name, figure in figures.items():
            path = self._save(figure, self.figure_path(stem, name))
            if path is not None:
                written.append(path)
                self.stats.written.append((name, str(path)))
        index = self._write_text(self.index_path(stem), self._index_text(stem, notes or {}))
        if index is not None:
            written.append(index)
        return written

    # -- internals -----------------------------------------------------------
    def _save(self, figure: Any, target: Path) -> Path | None:
        """Save one matplotlib figure atomically (temp file, then rename)."""
        import os

        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            figure.savefig(temporary, dpi=self._dpi, bbox_inches="tight", format=self._format)
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error", f"saving the chart {target.name} failed ({type(exc).__name__}: {exc})"
            )
            _close(figure)
            return None
        self.stats.figures += 1
        self.stats.files += 1
        _close(figure)
        return target

    def _index_text(self, stem: str, notes: Mapping[str, str]) -> str:
        """The index Markdown: one line per figure, then the run's own notes."""
        lines = [f"# ANJANEYA charts (spec 28) - {stem}", "", f"run id: `{self._run_id}`"]
        if self._config_hash:
            lines.append(f"config hash: `{self._config_hash}`")
        lines.append("")
        if self.stats.written:
            lines.append("| chart | file | note |")
            lines.append("| --- | --- | --- |")
            for name, path in self.stats.written:
                note = notes.get(name, "").replace("|", "\\|")
                lines.append(f"| {name} | `{Path(path).name}` | {note} |")
        else:
            lines.append("_No figure was written on this run._")
        lines.append("")
        if self.stats.failures:
            lines.append(
                f"**{self.stats.failures} figure(s) could not be written.** The files above are "
                "the ones that landed."
            )
            lines.append("")
        return "\n".join(lines)

    def _write_text(self, target: Path, text: str) -> Path | None:
        """Write the index atomically (temp file, then rename)."""
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
                f"writing the chart index {target.name} failed ({type(exc).__name__}: {exc})",
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


def _close(figure: Any) -> None:
    """Close a figure, ignoring a double that has no pyplot behind it."""
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        return


def _safe_unlink(paths: Iterable[Path] | Path) -> None:
    """Remove temporary files, ignoring anything that is already gone."""
    items = [paths] if isinstance(paths, Path) else list(paths)
    for path in items:
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue
