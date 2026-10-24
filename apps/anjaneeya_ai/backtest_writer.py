"""Backtest artefacts: the trade ledger (Parquet) plus the run summary (JSON).

A backtest's output is evidence, so it is written twice and in two shapes:

* ``data/backtests/1m_h5_2026-09-28_2026-10-01_ledger.parquet`` - one row per closed trade,
  in spec 23's column order plus the research columns the v1 ``trades`` table cannot hold
  (distance from ATM, weekday, the captured spread). It is what a spreadsheet or a notebook
  opens when a number is questioned.
* ``data/backtests/1m_h5_2026-09-28_2026-10-01_summary.json`` - the policies, the model
  artefact, the counters, the metrics, the segments and the caveats. It is the file that
  makes the Parquet interpretable a year later.

The same rows are also validated through :class:`core.schemas.TradeRecord` and appended to
the stored ``trades`` table (spec 23's ledger contract, which already existed in schema
version 1 - so **no schema change**). Validation happens at that boundary on purpose: a
trade that the row contract refuses is counted and logged, never written as a malformed row.

Every write is atomic (temp file, then rename) and a failure is counted rather than raised -
a convenience file must never invalidate a run that already did its work.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.backtest.engine import LEDGER_COLUMNS, BacktestResult, Trade
from core.constants import TRADE_COLUMNS
from core.schemas import TradeRecord
from core.storage import insert_rows

__all__ = ["BacktestWriteStats", "BacktestWriter"]


@dataclass(slots=True)
class BacktestWriteStats:
    """What the writer did (counted, so a partial failure is visible, never silent)."""

    trades: int = 0
    stored_rows: int = 0
    files: int = 0
    invalid_rows: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly shape."""
        return {
            "trades": self.trades,
            "stored_rows": self.stored_rows,
            "files": self.files,
            "invalid_rows": self.invalid_rows,
            "failures": self.failures,
        }


class BacktestWriter:
    """Write a backtest's ledger and summary under ``root`` (``data/backtests``)."""

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
        self.stats = BacktestWriteStats()

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Directory every artefact is written into."""
        return self._root

    def ledger_name(
        self, *, timeframe: str, horizon: int | None, first_day: Any, last_day: Any
    ) -> str:
        """``<timeframe>_h<horizon>_<first day>_<last day>`` - what the run covered."""
        first = _day_text(first_day)
        last = _day_text(last_day)
        return f"{timeframe}_h{horizon if horizon is not None else 'all'}_{first}_{last}"

    def ledger_path(self, *, timeframe: str, horizon: int | None, first_day: Any, last_day: Any) -> Path:
        """Where this run's trade ledger goes."""
        name = self.ledger_name(
            timeframe=timeframe, horizon=horizon, first_day=first_day, last_day=last_day
        )
        return self._root / f"{name}_ledger.parquet"

    def summary_path(
        self, *, timeframe: str, horizon: int | None, first_day: Any, last_day: Any
    ) -> Path:
        """Where this run's summary goes."""
        name = self.ledger_name(
            timeframe=timeframe, horizon=horizon, first_day=first_day, last_day=last_day
        )
        return self._root / f"{name}_summary.json"

    # -- writing -------------------------------------------------------------
    def write(
        self,
        *,
        result: BacktestResult,
        summary: Mapping[str, Any],
        horizon: int | None = None,
        connection: Any = None,
        store_ledger: bool = False,
    ) -> list[Path]:
        """Write the ledger and the summary; returns the files that landed.

        ``connection`` (a writable DuckDB connection) plus ``store_ledger=True`` also appends
        the validated rows to the stored ``trades`` table - spec 23's ledger contract. The
        Parquet artefact is written either way, because the research columns the table cannot
        hold are part of the evidence.
        """
        if not self._mirror:
            return []
        first_day = _day_text(result.data_start)
        last_day = _day_text(result.data_end)
        written: list[Path] = []
        ledger = self._write_ledger(
            result=result, horizon=horizon, first_day=first_day, last_day=last_day
        )
        if ledger is not None:
            written.append(ledger)
        summary_path = self._write_summary(
            result=result,
            summary=summary,
            horizon=horizon,
            first_day=first_day,
            last_day=last_day,
            ledger_name=ledger.name if ledger is not None else None,
        )
        if summary_path is not None:
            written.append(summary_path)
        if len(written) == 2:
            self.stats.trades += len(result.trades)
        if store_ledger and connection is not None:
            self._store(result=result, connection=connection)
        return written

    def _write_ledger(
        self, *, result: BacktestResult, horizon: int | None, first_day: str, last_day: str
    ) -> Path | None:
        """Write the ledger atomically (temp file, then rename)."""
        target = self.ledger_path(
            timeframe=result.timeframe, horizon=horizon, first_day=first_day, last_day=last_day
        )
        temporary = target.with_name(target.name + ".tmp")
        frame = ledger_frame(result.trades)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            frame.to_parquet(temporary, compression=self._compression, index=False)
            os.replace(temporary, target)
        except Exception as exc:  # noqa: BLE001 - report it, then keep going
            self.stats.failures += 1
            _safe_unlink(temporary)
            self._log(
                "error",
                f"writing the trade ledger {target.name} failed "
                f"({type(exc).__name__}: {exc}); the in-memory result is unaffected",
            )
            return None
        self.stats.files += 1
        return target

    def _write_summary(
        self,
        *,
        result: BacktestResult,
        summary: Mapping[str, Any],
        horizon: int | None,
        first_day: str,
        last_day: str,
        ledger_name: str | None,
    ) -> Path | None:
        """Write the summary JSON beside the ledger."""
        target = self.summary_path(
            timeframe=result.timeframe, horizon=horizon, first_day=first_day, last_day=last_day
        )
        temporary = target.with_name(target.name + ".tmp")
        payload = {
            "run_id": self._run_id,
            "config_hash": self._config_hash,
            "ledger": ledger_name,
            "ledger_columns": list(LEDGER_COLUMNS),
            "stored_ledger_columns": list(TRADE_COLUMNS),
            "result": result.describe(),
            **dict(summary),
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
                f"writing the backtest summary {target.name} failed "
                f"({type(exc).__name__}: {exc})",
            )
            return None
        self.stats.files += 1
        return target


    # -- the stored ledger (spec 23's contract) ------------------------------
    def _store(self, *, result: BacktestResult, connection: Any) -> None:
        """Validate every trade through ``TradeRecord`` and append it to ``trades``."""
        records: list[TradeRecord] = []
        for trade in result.trades:
            try:
                records.append(TradeRecord(**trade.to_record()))
            except Exception as exc:  # noqa: BLE001 - a bad row is counted, never written
                self.stats.invalid_rows += 1
                self._log(
                    "error",
                    f"trade {trade.trade_id!r} does not satisfy the stored ledger contract "
                    f"({type(exc).__name__}: {exc}); it stays in the Parquet ledger only",
                )
        if not records:
            return
        try:
            self.stats.stored_rows += insert_rows(
                connection, "trades", records, mode="append", columns=TRADE_COLUMNS
            )
        except Exception as exc:  # noqa: BLE001 - the artefacts are already written
            self.stats.failures += 1
            self._log(
                "error",
                f"storing the ledger in the trades table failed ({type(exc).__name__}: {exc}); "
                "the Parquet ledger and the summary are unaffected",
            )

    # -- internals -----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        """Log through the injected logger, tolerating a minimal double."""
        if self._logger is None:
            return
        method = getattr(self._logger, level, None) or self._logger.info
        method(message)


def ledger_frame(trades: Sequence[Trade]) -> pd.DataFrame:
    """The ledger as a frame in a fixed column order (an empty run included)."""
    rows = [trade.to_row() for trade in trades]
    return pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))


def _day_text(value: Any) -> str:
    """A date or timestamp as ``YYYY-MM-DD``, or ``unknown`` when there is none."""
    if value is None:
        return "unknown"
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return "unknown"
    return stamp.date().isoformat()


def _safe_unlink(paths: Iterable[Path] | Path) -> None:
    """Remove temporary files, ignoring anything that is already gone."""
    items = [paths] if isinstance(paths, Path) else list(paths)
    for path in items:
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue

