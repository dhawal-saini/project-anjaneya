"""The research matrices: spec 24's items 1-6, built from one trade ledger.

Spec 24 asks for the same numbers twice - once overall and once cut by the thing being
studied - and every cut here is made from the rows ``--mode backtest`` wrote, losers
included. Four properties keep the tables honest:

* **A cell carries its sample next to its numbers.** ``sample`` and ``preliminary`` travel
  with every metric, so a spectacular looking cell with four trades cannot be read as a
  finding (spec 24.10).
* **A missing bucket is not a zero.** A bucket with no trades is *absent* from the matrix
  rather than present with a P&L of 0.0, and the matrix records how many trades it was cut
  from. Nothing here invents a row to fill a grid.
* **The two axes are ordered the way the analysis reads them.** Time-of-day runs open to
  close and the expiry cycle runs expiry-day first, so a heatmap's rows mean what a reader
  assumes they mean (callers pass the order, this module never guesses it).
* **Only closed trades are metric-able.** :func:`ledger_trades` hands every row through the
  trade contract, so an open position is refused rather than marked to market.

The metrics themselves are spec 25's, computed by :class:`core.backtest.metrics.TradeMetrics`
- this module never re-derives a win rate, so a matrix cell and the overall summary can never
disagree about arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.backtest.engine import LEDGER_COLUMNS, RESEARCH_COLUMNS, Trade
from core.backtest.metrics import (
    DistributionSummary,
    SegmentTable,
    TradeMetrics,
    build_segment_table,
)

__all__ = [
    "MATRIX_SPECS",
    "SIMPLE_SEGMENT_KEYS",
    "HoldingTimeProfile",
    "HoldingTimeRow",
    "MatrixCell",
    "PerformanceMatrix",
    "ResearchError",
    "build_matrices",
    "build_matrix",
    "holding_time_profile",
    "ledger_trades",
    "simple_segment_tables",
]


class ResearchError(RuntimeError):
    """Raised when a research table cannot be built from the ledger it was handed."""


#: Spec 24's two-axis matrices: name -> (row key, column key, what it answers).
#: The keys are :data:`core.backtest.metrics.SEGMENT_KEYS` names, so a cell is labelled by
#: exactly the same rule as the per-segment table in a backtest report.
MATRIX_SPECS: dict[str, tuple[str, str, str]] = {
    "time_of_day_x_option_type": (
        "time_of_day_bucket",
        "option_type",
        "spec 24.1 - performance by time-of-day bucket x option side (CE / PE)",
    ),
    "time_of_day_x_cycle_phase": (
        "time_of_day_bucket",
        "expiry_cycle_phase",
        "spec 24.2 - performance by time-of-day bucket x expiry-cycle phase",
    ),
    "regime_x_time_of_day": (
        "nifty_state",
        "time_of_day_bucket",
        "spec 24.5 - interaction: NIFTY regime x time-of-day",
    ),
    "regime_x_cycle_phase": (
        "nifty_state",
        "expiry_cycle_phase",
        "spec 24.5 - interaction: NIFTY regime x cycle phase",
    ),
}

#: Spec 24's single-axis cuts (items 3 and 4), which the backtest report already defines.
SIMPLE_SEGMENT_KEYS: tuple[str, ...] = ("day_of_week", "days_to_expiry")


def _missing(value: Any) -> bool:
    """Is this Parquet-round-tripped value really absent (``None``, ``NaN`` or ``NaT``)?"""
    if value is None:
        return True
    if isinstance(value, float):
        return bool(np.isnan(value))
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic scalar, treat as present
        return False


def _clean(value: Any) -> Any:
    """``None`` for an absent value, the value itself otherwise."""
    return None if _missing(value) else value


def ledger_trades(ledger: Any) -> list[Trade]:
    """The ledger as :class:`core.backtest.engine.Trade` rows.

    ``ledger`` may be the Parquet frame read back from ``data/backtests/`` or an iterable of
    :class:`Trade` objects (the in-process case). A Parquet round trip turns ``None`` into
    ``NaN``/``NaT``, and those must become ``None`` again before any metric sees them:
    ``nan`` is *truthy*, which is how a missing cost silently poisons a sum.
    """
    if isinstance(ledger, pd.DataFrame):
        missing = [name for name in LEDGER_COLUMNS if name not in ledger.columns]
        if missing:
            raise ResearchError(
                f"the ledger frame is missing column(s) {missing}; a report analyses the "
                "artefact --mode backtest wrote, whose columns are the trade contract plus "
                f"{list(RESEARCH_COLUMNS)}"
            )
        unknown_columns = [name for name in ledger.columns if name not in LEDGER_COLUMNS]
        if unknown_columns:
            raise ResearchError(
                f"the ledger carries column(s) the trade contract does not know: "
                f"{sorted(unknown_columns)} - this is not a backtest ledger"
            )
        rows: Iterable[Any] = (
            {name: row.get(name) for name in LEDGER_COLUMNS} for _, row in ledger.iterrows()
        )
    else:
        rows = list(ledger)
    trades: list[Trade] = []
    for row in rows:
        if isinstance(row, Trade):
            trades.append(row)
            continue
        if not isinstance(row, Mapping):
            raise ResearchError(
                "a ledger row must be a Trade or a mapping of its columns, got "
                f"{type(row).__name__}"
            )
        unknown = set(row) - set(LEDGER_COLUMNS)
        if unknown:
            raise ResearchError(
                f"the ledger carries column(s) the trade contract does not know: {sorted(unknown)}"
            )
        trades.append(Trade(**{name: _clean(row.get(name)) for name in LEDGER_COLUMNS}))
    return trades


def _ordered(labels: Iterable[str], order: Sequence[str] | None) -> list[str]:
    """``labels`` sorted by ``order`` first (when given), then alphabetically."""
    unique = sorted(set(labels))
    if not order:
        return unique
    ranked = {label: index for index, label in enumerate(order)}
    return sorted(unique, key=lambda label: (ranked.get(label, len(order)), label))


def _extractor(key: str) -> Any:
    """The label extractor of one segment key, refusing an unknown one loudly."""
    from core.backtest.metrics import SEGMENT_KEYS

    if key not in SEGMENT_KEYS:
        raise ResearchError(f"unknown segment {key!r}; known segments: {sorted(SEGMENT_KEYS)}")
    return SEGMENT_KEYS[key][1]


# ---------------------------------------------------------------------------
# Two-axis matrices (items 1, 2, 5)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MatrixCell:
    """One intersection of a matrix: its labels, its sample, its metrics."""

    row_label: str
    column_label: str
    sample: int
    preliminary: bool
    metrics: TradeMetrics

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (the sample is part of the row, never a footnote)."""
        return {
            "row": self.row_label,
            "column": self.column_label,
            "sample": self.sample,
            "preliminary": self.preliminary,
            **self.metrics.to_dict(),
        }


@dataclass(slots=True)
class PerformanceMatrix:
    """One cut of the ledger along two axes, with the samples that stand behind it."""

    name: str
    description: str
    row_key: str
    column_key: str
    row_labels: list[str]
    column_labels: list[str]
    cells: dict[tuple[str, str], MatrixCell]
    preliminary_min_samples: int
    trades: int

    @property
    def preliminary_cells(self) -> list[tuple[str, str]]:
        """Cells whose sample is too small to be read as anything but an observation."""
        return [
            (row, column) for (row, column), cell in self.cells.items() if cell.preliminary
        ]

    def cell(self, row: str, column: str) -> MatrixCell | None:
        """The cell at one intersection, or ``None`` when nothing traded there."""
        return self.cells.get((row, column))

    def rows(self) -> list[dict[str, Any]]:
        """Every cell as a flat row (the CSV/JSON shape), in matrix order."""
        return [
            self.cells[(row, column)].to_dict()
            for row in self.row_labels
            for column in self.column_labels
            if (row, column) in self.cells
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "name": self.name,
            "description": self.description,
            "row_key": self.row_key,
            "column_key": self.column_key,
            "row_labels": list(self.row_labels),
            "column_labels": list(self.column_labels),
            "trades": self.trades,
            "preliminary_min_samples": self.preliminary_min_samples,
            "preliminary_cells": [f"{row} | {column}" for row, column in self.preliminary_cells],
            "cells": self.rows(),
        }

    def grid(self, value: Any) -> list[list[Any]]:
        """A rectangular view of one metric (``None`` where nothing traded) - heatmap ready.

        ``value`` is called with a :class:`TradeMetrics`; a caller wanting the net P&L grid
        passes ``lambda metrics: metrics.total_net_pnl``. Keeping this here means a table and
        any later chart are built from one ordering and cannot drift apart.
        """
        return [
            [
                value(self.cells[(row, column)].metrics) if (row, column) in self.cells else None
                for column in self.column_labels
            ]
            for row in self.row_labels
        ]


def build_matrix(
    name: str,
    trades: Sequence[Any],
    *,
    preliminary_min_samples: int,
    row_order: Sequence[str] | None = None,
    column_order: Sequence[str] | None = None,
) -> PerformanceMatrix:
    """Group ``trades`` on two axes and metric every intersection.

    ``row_order`` / ``column_order`` are the *presentation* orders (bucket start times, the
    expiry cycle's own sequence); a label outside them is appended alphabetically rather than
    dropped, so a stale order can never hide a cell.
    """
    if name not in MATRIX_SPECS:
        raise ResearchError(f"unknown matrix {name!r}; known matrices: {sorted(MATRIX_SPECS)}")
    row_key, column_key, description = MATRIX_SPECS[name]
    if row_key == column_key:
        raise ResearchError(f"matrix {name!r} would cut the same key twice ({row_key!r})")
    row_of = _extractor(row_key)
    column_of = _extractor(column_key)
    groups: dict[tuple[str, str], list[Any]] = {}
    for trade in trades:
        groups.setdefault((row_of(trade), column_of(trade)), []).append(trade)
    cells = {
        key: MatrixCell(
            row_label=key[0],
            column_label=key[1],
            sample=len(group),
            preliminary=len(group) < int(preliminary_min_samples),
            metrics=TradeMetrics.from_trades(group),
        )
        for key, group in groups.items()
    }
    return PerformanceMatrix(
        name=name,
        description=description,
        row_key=row_key,
        column_key=column_key,
        row_labels=_ordered((key[0] for key in groups), row_order),
        column_labels=_ordered((key[1] for key in groups), column_order),
        cells=cells,
        preliminary_min_samples=int(preliminary_min_samples),
        trades=len(trades),
    )


def build_matrices(
    trades: Sequence[Any],
    *,
    preliminary_min_samples: int,
    names: Sequence[str] | None = None,
    orders: Mapping[str, Sequence[str]] | None = None,
) -> list[PerformanceMatrix]:
    """Every (or the requested) spec 24 matrix, in :data:`MATRIX_SPECS` order.

    ``orders`` maps a *segment key* (not a matrix name) to its presentation order, so the
    calendar's bucket order and the cycle's own order apply to every matrix that uses them.
    """
    chosen = list(names) if names is not None else list(MATRIX_SPECS)
    order_map = dict(orders or {})
    return [
        build_matrix(
            name,
            trades,
            preliminary_min_samples=preliminary_min_samples,
            row_order=order_map.get(MATRIX_SPECS[name][0]),
            column_order=order_map.get(MATRIX_SPECS[name][1]),
        )
        for name in chosen
    ]


# ---------------------------------------------------------------------------
# Holding time per bucket (item 6)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class HoldingTimeRow:
    """How long the trades of one segment were held, and what they earned while held."""

    label: str
    sample: int
    preliminary: bool
    distribution: DistributionSummary
    metrics: TradeMetrics

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "label": self.label,
            "sample": self.sample,
            "preliminary": self.preliminary,
            "holding_minutes": self.distribution.to_dict(),
            "total_net_pnl": self.metrics.total_net_pnl,
            "win_rate": self.metrics.win_rate,
            "total_holding_minutes": self.metrics.total_holding_minutes,
        }


@dataclass(slots=True)
class HoldingTimeProfile:
    """Spec 24.6: the holding-time distribution of every segment of the ledger."""

    name: str
    description: str
    label_key: str
    rows: list[HoldingTimeRow] = field(default_factory=list)
    preliminary_min_samples: int = 0

    @property
    def preliminary_labels(self) -> list[str]:
        """Labels whose sample is too small to be read as anything but an observation."""
        return [row.label for row in self.rows if row.preliminary]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "name": self.name,
            "description": self.description,
            "label_key": self.label_key,
            "preliminary_min_samples": self.preliminary_min_samples,
            "preliminary_labels": self.preliminary_labels,
            "rows": [row.to_dict() for row in self.rows],
        }


def holding_time_profile(
    trades: Sequence[Any],
    *,
    preliminary_min_samples: int,
    by: str = "time_of_day_bucket",
    order: Sequence[str] | None = None,
) -> HoldingTimeProfile:
    """Spec 24.6: holding-time distribution (and its sample) per ``by`` segment."""
    extractor = _extractor(by)
    groups: dict[str, list[Any]] = {}
    for trade in trades:
        groups.setdefault(extractor(trade), []).append(trade)
    rows = []
    for label, group in groups.items():
        held = [
            float(getattr(trade, "holding_minutes", 0.0) or 0.0)
            for trade in group
            if getattr(trade, "holding_minutes", None) is not None
        ]
        rows.append(
            HoldingTimeRow(
                label=label,
                sample=len(group),
                preliminary=len(group) < int(preliminary_min_samples),
                distribution=DistributionSummary.from_values(held),
                metrics=TradeMetrics.from_trades(group),
            )
        )
    ranked = {label: index for index, label in enumerate(order or ())}
    rows.sort(key=lambda row: (ranked.get(row.label, len(ranked)), row.label))
    from core.backtest.metrics import SEGMENT_KEYS

    return HoldingTimeProfile(
        name=f"holding_time_by_{by}",
        description=f"spec 24.6 - holding-time distribution by {SEGMENT_KEYS[by][0]}",
        label_key=by,
        rows=rows,
        preliminary_min_samples=int(preliminary_min_samples),
    )


def simple_segment_tables(
    trades: Sequence[Any],
    *,
    preliminary_min_samples: int,
    names: Sequence[str] | None = None,
) -> list[SegmentTable]:
    """Spec 24.3 / 24.4: the single-axis cuts (weekday, days-to-expiry bucket)."""
    return [
        build_segment_table(name, trades, preliminary_min_samples=preliminary_min_samples)
        for name in (list(names) if names is not None else list(SIMPLE_SEGMENT_KEYS))
    ]

