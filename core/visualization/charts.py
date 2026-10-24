"""Spec 28's chart builders: figures from data, and nothing else.

"Charts must be produced for validation" and "visualization is not optional", so spec 28 asks
for two families of pictures:

* **price charts** - the NIFTY series (price, MA lines, trend score, regime) and one option
  contract (price with ``EMA8 high``, ``EMA8 low``, ``EMA20``, ``EMA55`` and ``EMA200``),
  both with the same overlay vocabulary: today's running high/low, the previous day's
  high/low, support/resistance lines, breakout/rejection markers, structure labels and the
  model's own prediction.
* **research charts** - the time-of-day x cycle-phase P&L heatmap, the time-of-day x
  day-of-week heatmap, the CE-vs-PE P&L curves, regime-conditional performance and the
  feature-importance bars.

This module is the pure half: every builder takes plain data (lists, mappings and the small
dataclasses below) and returns a :class:`matplotlib.figure.Figure`. It performs no file I/O,
reads no database and imports no ANJANEYA module - the loading half lives in
:mod:`core.visualization.sources` and the writing half in
:mod:`apps.anjaneeya_ai.chart_writer`, so a builder can be tested with a handful of numbers.

Two rules run through every builder:

* **a value that is not there is left blank, not drawn as zero.** A heatmap cell with no
  trades is not ``0`` and a feature the sample could not measure has no bar. A missing number
  and a real ``0`` look identical once plotted, so the data structures carry ``None`` and the
  builder simply does not draw it.
* **the sample travels with the picture.** Heatmap cells print their trade count and are
  hatched when their sample is below the configured minimum, and the importance bars carry
  their confidence interval. Spec 28 asks for charts "for validation": a chart that hides how
  little stands behind a cell would be decoration, not validation.

The ``Agg`` backend is selected lazily inside :func:`_pyplot`, so importing this module has no
side effect and a headless run (CI, a scheduled task) renders without a display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CHART_KINDS",
    "ChartError",
    "ChartPolicy",
    "EventMarker",
    "FeatureBar",
    "Heatmap",
    "LevelLine",
    "PriceChartData",
    "bar_chart",
    "curve_chart",
    "heatmap",
    "importance_bars",
    "price_chart",
]

#: The chart families this module knows how to draw (used by the CLI's plan and ``--only``).
CHART_KINDS: tuple[str, ...] = ("heatmap", "curves", "regime", "importance", "price")


class ChartError(ValueError):
    """Raised when a chart is asked to plot something that cannot honestly be plotted."""


@dataclass(frozen=True, slots=True)
class ChartPolicy:
    """Spec 28's presentation rules (``model.charts``), never hard-coded in a builder.

    ``from_config`` reads the shipped ``charts:`` block, so the figure size, the resolution,
    the file format, the diverging colormap and how many features the importance chart shows
    are all configuration. The dataclass also carries its own defaults, so a caller building a
    policy by hand (a notebook, a test) gets exactly the shipped behaviour.
    """

    #: The master-spec section the charts come from (printed beside every figure's title).
    spec: str = "28"
    #: Decimals for a printed value (a P&L in rupees, a return in percent).
    digits: int = 2
    #: Raster resolution of a written figure.
    dpi: int = 120
    #: Figure size in inches for a wide chart (price, heatmap, curves).
    width: float = 14.0
    height: float = 7.0
    #: Figure size in inches for a tall, single-column chart (importance bars).
    tall_height: float = 9.0
    #: File extension of a written figure (no dot).
    format: str = "png"
    #: Diverging colormap for a value that can be positive or negative (a P&L heatmap).
    colormap: str = "RdYlGn"
    #: Sequential colormap for a magnitude that is never negative.
    sequential_colormap: str = "viridis"
    #: How many features the importance bars show (the most important ones).
    top_features: int = 20
    #: The most points any single line is drawn from (a dense 1m series is decimated to fit).
    max_points: int = 30_000
    #: The most event markers drawn on one price chart (the newest are kept).
    max_markers: int = 60

    @classmethod
    def from_config(cls, config: Any) -> ChartPolicy:
        """The policy the shipped ``config/model.yaml`` describes.

        Falls back to the dataclass defaults when the section is absent, so a hand-built
        configuration (a test, a Phase-5 era ``model.yaml``) still resolves a usable policy.
        """
        section = getattr(getattr(config, "model", None), "charts", None)
        if section is None:
            return cls()
        return cls(
            spec=str(getattr(section, "spec", cls.spec)),
            digits=int(getattr(section, "digits", cls.digits)),
            dpi=int(getattr(section, "dpi", cls.dpi)),
            width=float(getattr(section, "width", cls.width)),
            height=float(getattr(section, "height", cls.height)),
            tall_height=float(getattr(section, "tall_height", cls.tall_height)),
            format=str(getattr(section, "format", cls.format)).lstrip("."),
            colormap=str(getattr(section, "colormap", cls.colormap)),
            sequential_colormap=str(
                getattr(section, "sequential_colormap", cls.sequential_colormap)
            ),
            top_features=int(getattr(section, "top_features", cls.top_features)),
            max_points=int(getattr(section, "max_points", cls.max_points)),
            max_markers=int(getattr(section, "max_markers", cls.max_markers)),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (stored beside the figures on the run manifest)."""
        return {
            "spec": self.spec,
            "digits": self.digits,
            "dpi": self.dpi,
            "width": self.width,
            "height": self.height,
            "tall_height": self.tall_height,
            "format": self.format,
            "colormap": self.colormap,
            "sequential_colormap": self.sequential_colormap,
            "top_features": self.top_features,
            "max_points": self.max_points,
            "max_markers": self.max_markers,
        }


# ---------------------------------------------------------------------------
# The data contracts the builders consume
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LevelLine:
    """One horizontal overlay: a named price the chart draws across the axis.

    ``kind`` says which *role* the line plays (``previous_day``, ``support``, ``resistance``,
    ``level``), because two lines can sit at the same price for different reasons and the
    legend should say so rather than merge them.
    """

    label: str
    value: float
    kind: str = "level"


@dataclass(frozen=True, slots=True)
class EventMarker:
    """One annotation on a price chart: a breakout, a rejection or a structure label.

    ``kind`` is one of ``breakout``, ``rejection``, ``structure`` or ``prediction``; the
    builder colours and marks them differently so a reader can tell an event from a model
    output at a glance.
    """

    timestamp: Any
    value: float
    label: str
    kind: str = "event"


@dataclass(slots=True)
class PriceChartData:
    """Everything one price chart draws, gathered by :mod:`core.visualization.sources`.

    A builder never queries anything; it is handed the series, the overlays and the markers
    that already exist and draws exactly those. Any list may be empty (a contract with no
    ``EMA200`` yet), and an empty list is drawn as "not there", never as a flat zero line.
    """

    title: str
    symbol: str
    timeframe: str
    timestamps: list[Any]
    close: list[float]
    moving_averages: dict[str, list[float | None]] = field(default_factory=dict)
    levels: list[LevelLine] = field(default_factory=list)
    running: dict[str, list[float | None]] = field(default_factory=dict)
    markers: list[EventMarker] = field(default_factory=list)
    predictions: list[EventMarker] = field(default_factory=list)
    trend: list[float | None] = field(default_factory=list)
    regimes: list[str | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (for the run's notes, not for the figure itself)."""
        return {
            "title": self.title,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "points": len(self.timestamps),
            "moving_averages": sorted(self.moving_averages),
            "levels": [level.label for level in self.levels],
            "running": sorted(self.running),
            "markers": len(self.markers),
            "predictions": len(self.predictions),
            "has_trend": bool(self.trend),
            "has_regime": bool(self.regimes),
        }


@dataclass(slots=True)
class Heatmap:
    """A labelled grid plus the sample behind every cell - the honest shape for a heatmap.

    ``values`` may hold ``None`` (nothing traded in that cell), ``samples`` holds the trade
    count, and ``preliminary`` flags a cell whose sample is below the configured minimum. The
    builder hatches a preliminary cell and prints its count, so a reader never mistakes a
    three-trade cell for a measured one.
    """

    title: str
    row_label: str
    column_label: str
    value_label: str
    rows: list[str]
    columns: list[str]
    values: list[list[float | None]]
    samples: list[list[int]]
    preliminary: list[list[bool]]
    center: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "title": self.title,
            "row_label": self.row_label,
            "column_label": self.column_label,
            "value_label": self.value_label,
            "rows": list(self.rows),
            "columns": list(self.columns),
            "cells": sum(1 for row in self.values for value in row if value is not None),
            "preliminary_cells": sum(1 for row in self.preliminary for flag in row if flag),
            "center": self.center,
        }


@dataclass(frozen=True, slots=True)
class FeatureBar:
    """One bar of the importance chart: the feature, its value and its interval.

    ``low``/``high`` are ``None`` when the measurement has no interval (the model's own native
    gain does not), and the builder then draws the bar without whiskers rather than inventing
    one.
    """

    column: str
    value: float
    low: float | None = None
    high: float | None = None
    topic: str | None = None


# ---------------------------------------------------------------------------
# The matplotlib access point (lazy: importing this module has no side effect)
# ---------------------------------------------------------------------------
def _pyplot() -> Any:
    """The pyplot module, with the headless ``Agg`` backend selected.

    Imported inside a builder, not at module import, so ``import core.visualization.charts``
    costs nothing and never opens a window. ``Agg`` is chosen for every build: a chart is
    always *written to a file*, never shown interactively, so a scheduled run works with no
    display attached.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _figure(policy: ChartPolicy, *, height: float | None = None) -> tuple[Any, Any]:
    """A ``(figure, axes)`` pair sized by the policy."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(policy.width, height or policy.height), dpi=policy.dpi)
    axes.set_facecolor("#f7f7f7")
    figure.patch.set_facecolor("white")
    return figure, axes


def _title(policy: ChartPolicy, title: str) -> str:
    """The full title: the caller's words plus the spec section, so a picture is traceable."""
    return f"{title}  (spec {policy.spec})"


def _decimate(values: list[Any], limit: int) -> list[Any]:
    """Thin a series to at most ``limit`` points, keeping the first and the last.

    A 1-minute contract over several days is tens of thousands of points; the axis cannot show
    them and the file need not carry them. The reduction is uniform and keeps both endpoints,
    so the drawn range is never truncated.
    """
    count = len(values)
    if count <= limit or limit <= 2:
        return list(values)
    step = count / (limit - 1)
    indices = sorted({int(round(index * step)) for index in range(limit - 1)} | {count - 1})
    return [values[index] for index in indices]


def _aligned(x: list[Any], values: list[Any], limit: int) -> tuple[list[Any], list[Any]]:
    """Decimate an overlay and return only the points where the value is a real number.

    A ``None`` overlay value (an EMA200 that has not warmed up) is dropped rather than plotted
    as a gap-to-zero, and the x is taken at the same indices so the line stays attached to the
    right timestamp. An overlay whose length does not match the axis is refused rather than
    truncated: a shifted line is a wrong picture, not a partial one.
    """
    if len(values) != len(x):
        return [], []
    drawn_x = _decimate(list(x), limit)
    drawn_y = _decimate(list(values), limit)
    points = [
        (x_value, y_value)
        for x_value, y_value in zip(drawn_x, drawn_y, strict=False)
        if y_value is not None and _is_number(y_value)
    ]
    if not points:
        return [], []
    return [point[0] for point in points], [point[1] for point in points]


def _is_number(value: Any) -> bool:
    """Whether a value is a real number (not ``None`` and not ``NaN``)."""
    try:
        return bool(value == value)  # NaN is the only value that is not equal to itself
    except Exception:  # noqa: BLE001 - an exotic scalar is treated as not-a-number
        return False


# ---------------------------------------------------------------------------
# Price charts (NIFTY and one option contract share this builder)
# ---------------------------------------------------------------------------
def price_chart(data: PriceChartData, *, policy: ChartPolicy) -> Any:
    """Draw spec 28's price chart: price, MA lines, level/running overlays and markers.

    The NIFTY chart and the option chart are the *same* picture with different series and
    overlays, so there is one builder. What the spec calls out by name - the NIFTY trend score
    and regime, the option's ``EMA8 high``/``EMA8 low``/``EMA20``/``EMA55``/``EMA200``, and the
    overlay vocabulary - is exactly the set of optional arguments here; a chart simply omits
    what the data does not carry.

    The trend score (when supplied) is drawn on a secondary axis because a score in ``[-1, +1]``
    and a price in thousands would otherwise share one scale and flatten each other; the regime
    is drawn as a shaded band behind the price, one faint colour per distinct regime.
    """
    if not data.timestamps or not data.close:
        raise ChartError(f"price chart {data.symbol!r} has no points to draw")

    figure, axes = _figure(policy)
    x = _decimate(list(data.timestamps), policy.max_points)
    y = _decimate(list(data.close), policy.max_points)

    _shade_regimes(axes, data, policy)
    axes.plot(x, y, color="#1f2d3d", linewidth=1.6, label=data.symbol, zorder=3)

    palette = _line_palette()
    for index, (label, series) in enumerate(sorted(data.moving_averages.items())):
        drawn_x, drawn_y = _aligned(x, list(series), policy.max_points)
        if not drawn_y:
            continue
        axes.plot(
            drawn_x,
            drawn_y,
            linewidth=1.1,
            color=palette[index % len(palette)],
            label=label,
            zorder=2,
        )

    for level in data.levels:
        axes.axhline(
            level.value, linestyle=_level_style(level.kind), linewidth=1.0, color="#8a6d3b", zorder=1
        )
        axes.annotate(
            f"{level.label} {level.value:.{policy.digits}f}",
            xy=(1.0, level.value),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7,
            color="#6b4f1d",
        )

    for label, series in sorted(data.running.items()):
        drawn_x, drawn_y = _aligned(x, list(series), policy.max_points)
        if not drawn_y:
            continue
        axes.plot(drawn_x, drawn_y, linewidth=0.9, color="#c0392b", alpha=0.7, label=label)

    _plot_markers(axes, _decimate(list(data.markers), policy.max_markers), policy)
    _plot_markers(axes, _decimate(list(data.predictions), policy.max_markers), policy)

    if data.trend and any(value is not None for value in data.trend):
        _trend_axis(axes, x, list(data.trend), policy)

    axes.set_title(_title(policy, f"{data.title} - {data.symbol} ({data.timeframe})"), fontsize=12)
    axes.set_ylabel("price")
    axes.set_xlabel("time")
    axes.grid(True, alpha=0.25)
    _legend(axes)
    _rotate_ticks(axes, x)

    figure.tight_layout()
    return figure


def _shade_regimes(axes: Any, data: PriceChartData, policy: ChartPolicy) -> None:
    """Shade the background by regime: one faint colour per distinct run of labels."""
    if not data.regimes:
        return
    x = _decimate(list(data.timestamps), policy.max_points)
    regimes = _decimate(list(data.regimes), policy.max_points)
    palette = _regime_palette()
    start = 0
    for index in range(1, len(regimes) + 1):
        current = regimes[index] if index < len(regimes) else None
        if current != regimes[start] or index == len(regimes):
            label = regimes[start]
            if label:
                axes.axvspan(
                    x[start],
                    x[index - 1],
                    color=palette.get(label, "#dddddd"),
                    alpha=0.18,
                    zorder=0,
                )
            start = index


def _trend_axis(axes: Any, x: list[Any], trend: list[Any], policy: ChartPolicy) -> None:
    """Overlay the NIFTY trend score on a secondary axis (a score and a price share no scale)."""
    drawn_x, drawn_y = _aligned(x, trend, policy.max_points)
    if not drawn_y:
        return
    secondary = axes.twinx()
    secondary.plot(
        drawn_x, drawn_y, color="#2e86c1", linewidth=1.0, linestyle="--", label="trend score"
    )
    secondary.set_ylabel("trend score", color="#2e86c1")
    secondary.tick_params(axis="y", colors="#2e86c1")
    secondary.set_ylim(-1.05, 1.05)
    secondary.axhline(0.0, color="#2e86c1", linewidth=0.6, alpha=0.5)


def _plot_markers(axes: Any, markers: list[EventMarker], policy: ChartPolicy) -> None:
    """Draw event/prediction markers, dropping any marker whose value is not a number."""
    style = {
        "breakout": ("^", "#1e8449", "breakout"),
        "rejection": ("v", "#b03a2e", "rejection"),
        "structure": ("s", "#7d3c98", "structure"),
        "prediction": ("*", "#e67e22", "AI prediction"),
        "event": ("o", "#616a6b", "event"),
    }
    seen: set[str] = set()
    for marker in markers:
        if marker.value is None or not _is_number(marker.value):
            continue
        shape, colour, legend = style.get(marker.kind, style["event"])
        axes.scatter(
            [marker.timestamp],
            [marker.value],
            marker=shape,
            color=colour,
            s=42,
            zorder=5,
            label=legend if legend not in seen else None,
        )
        seen.add(legend)


def _legend(axes: Any) -> None:
    """A legend that stays out of the plot's way when a chart carries many overlays."""
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        return
    axes.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=7,
        framealpha=0.85,
        ncol=2 if len(handles) > 6 else 1,
    )


# ---------------------------------------------------------------------------
# Heatmaps (time-of-day x cycle phase, time-of-day x day-of-week)
# ---------------------------------------------------------------------------
def heatmap(cells: Heatmap, *, policy: ChartPolicy) -> Any:
    """Draw a labelled heatmap with each cell's value and sample printed inside it.

    The colormap is diverging around zero when ``center`` is set (a P&L), so positive and
    negative cells are both readable; otherwise it is sequential (a magnitude). A cell with no
    value is left blank; a cell whose sample is below the minimum is hatched *and* prints its
    count, which is the whole point of drawing the sample on the picture.
    """
    if not cells.rows or not cells.columns:
        raise ChartError(f"heatmap {cells.title!r} has no rows or columns to draw")

    import numpy as np

    figure, axes = _figure(policy, height=max(policy.height, 0.5 * len(cells.rows) + 3.0))
    numeric = np.array(
        [[value if value is not None else np.nan for value in row] for row in cells.values],
        dtype="float64",
    )
    finite = numeric[np.isfinite(numeric)] if numeric.size else numeric
    if cells.center:
        bound = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmin, vmax, cmap = -bound, bound, policy.colormap
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        cmap = policy.sequential_colormap
    if vmin == vmax:  # a single value: give the colormap something to stretch over
        vmin, vmax = vmin - 1.0, vmax + 1.0

    image = axes.imshow(numeric, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label(cells.value_label)

    axes.set_xticks(range(len(cells.columns)))
    axes.set_xticklabels(cells.columns, rotation=30, ha="right", fontsize=8)
    axes.set_yticks(range(len(cells.rows)))
    axes.set_yticklabels(cells.rows, fontsize=8)
    axes.set_xlabel(cells.column_label)
    axes.set_ylabel(cells.row_label)
    axes.set_title(_title(policy, cells.title), fontsize=12)

    for row_index, row in enumerate(cells.values):
        for column_index, value in enumerate(row):
            sample = cells.samples[row_index][column_index]
            preliminary = cells.preliminary[row_index][column_index]
            if preliminary:
                axes.add_patch(_hatch_patch(column_index - 0.5, row_index - 0.5))
            if value is None:
                text, colour = (f"n={sample}" if sample else "-"), "#333333"
            else:
                text = f"{value:.{policy.digits}f}\nn={sample}"
                colour = "white" if _is_dark(numeric[row_index][column_index], vmin, vmax) else "#111111"
            axes.text(column_index, row_index, text, ha="center", va="center", fontsize=7, color=colour)
    axes.set_xticks(np.arange(-0.5, len(cells.columns), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(cells.rows), 1), minor=True)
    axes.grid(which="minor", color="white", linewidth=1.0)
    axes.tick_params(which="minor", length=0)

    figure.tight_layout()
    return figure


def _hatch_patch(x: float, y: float) -> Any:
    """A hatched cell overlay marking a PRELIMINARY sample."""
    import matplotlib.patches as patches

    return patches.Rectangle(
        (x, y),
        1.0,
        1.0,
        facecolor="none",
        edgecolor="#111111",
        hatch="//",
        linewidth=0.0,
        alpha=0.35,
    )


def _is_dark(value: float, vmin: float, vmax: float) -> bool:
    """Whether a colormapped cell is dark enough for white text to be readable."""
    if vmax == vmin or not _is_number(value):
        return False
    fraction = (float(value) - vmin) / (vmax - vmin)
    return fraction < 0.15 or fraction > 0.85


# ---------------------------------------------------------------------------
# Line curves (CE vs PE P&L curves)
# ---------------------------------------------------------------------------
def curve_chart(
    curves: dict[str, list[tuple[Any, float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    policy: ChartPolicy,
) -> Any:
    """Draw one line per named series, from ``(x, y)`` pairs.

    Used for the CE-vs-PE P&L curves and any cumulative series. A series with fewer than two
    points is skipped (a line through one dot is not a curve); the caller learns which series
    were drawn from the figure's own legend.
    """
    drawn = {name: points for name, points in curves.items() if len(points) >= 2}
    if not drawn:
        raise ChartError(f"curve chart {title!r} has no series with at least two points")

    figure, axes = _figure(policy)
    palette = _line_palette()
    for index, (name, points) in enumerate(sorted(drawn.items())):
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        axes.plot(x, y, linewidth=1.4, color=palette[index % len(palette)], label=name)
    axes.axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
    axes.set_title(_title(policy, title), fontsize=12)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.25)
    axes.legend(loc="best", fontsize=8)
    _rotate_ticks(axes, [point[0] for points in drawn.values() for point in points])

    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Bar charts (regime-conditional performance, feature importance)
# ---------------------------------------------------------------------------
def bar_chart(
    labels: list[str],
    values: list[float | None],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    policy: ChartPolicy,
    samples: list[int] | None = None,
) -> Any:
    """Draw a vertical bar chart coloured by sign, optionally labelled with each sample.

    Used for regime-conditional performance (spec 28's fifth research chart). A bar with no
    value is simply absent - never a zero-height bar that reads as "measured, and flat".
    """
    paired = [
        (label, value)
        for label, value in zip(labels, values, strict=False)
        if value is not None
    ]
    if not paired:
        raise ChartError(f"bar chart {title!r} has no values to draw")

    figure, axes = _figure(policy, height=max(4.0, 0.5 * len(paired) + 2.0))
    positions = list(range(len(paired)))
    axes.bar(positions, [value for _, value in paired], color=_sign_colours(paired))
    axes.set_xticks(positions)
    axes.set_xticklabels([label for label, _ in paired], rotation=30, ha="right", fontsize=8)
    axes.axhline(0.0, color="#333333", linewidth=0.8)
    if samples is not None:
        counts = dict(zip(labels, samples, strict=False))
        for position, (label, value) in zip(positions, paired, strict=False):
            if label not in counts:
                continue
            axes.annotate(
                f"n={counts[label]}",
                (position, value),
                textcoords="offset points",
                xytext=(0, 3 if value >= 0 else -9),
                ha="center",
                fontsize=7,
                color="#333333",
            )
    axes.set_title(_title(policy, title), fontsize=12)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, axis="y", alpha=0.25)

    figure.tight_layout()
    return figure


def importance_bars(
    bars: list[FeatureBar],
    *,
    title: str,
    value_label: str,
    policy: ChartPolicy,
) -> Any:
    """Draw the feature-importance chart: horizontal bars with confidence intervals.

    The most important features are shown (``policy.top_features``), ranked by absolute value
    so a large negative permutation delta is as visible as a large positive one - the question
    is "how much does this feature matter", not "which way". A bar without an interval simply
    has no whiskers; a feature with no measurement never becomes a zero bar.
    """
    ranked = sorted(bars, key=lambda bar: abs(bar.value), reverse=True)[: policy.top_features]
    if not ranked:
        raise ChartError(f"importance chart {title!r} has no measured features to draw")
    ranked = list(reversed(ranked))

    figure, axes = _figure(policy, height=policy.tall_height)
    labels = [bar.column for bar in ranked]
    values = [bar.value for bar in ranked]
    errors = [_interval(bar) for bar in ranked]
    axes.barh(
        labels,
        values,
        color=_sign_colours([(bar.column, bar.value) for bar in ranked]),
        xerr=errors,
        error_kw={"ecolor": "#444444", "lw": 0.8},
    )
    axes.axvline(0.0, color="#333333", linewidth=0.8)
    axes.set_title(_title(policy, title), fontsize=12)
    axes.set_xlabel(value_label)
    axes.grid(True, axis="x", alpha=0.25)
    axes.tick_params(axis="y", labelsize=8)

    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def _sign_colours(pairs: list[tuple[str, float]]) -> list[str]:
    """Green for a non-negative value, red for a negative one (a P&L, an importance delta)."""
    return ["#1e8449" if value >= 0 else "#b03a2e" for _, value in pairs]


def _interval(bar: FeatureBar) -> list[float]:
    """A symmetric matplotlib ``xerr`` for a bar's confidence interval.

    ``xerr`` is drawn on both sides of the bar's own value, so the two asymmetric distances are
    reduced to the larger one - a bar with an interval narrower on one side than the other is
    drawn with the wider whisker, never clipped.
    """
    if bar.low is None or bar.high is None:
        return [0.0]
    return [max(abs(bar.value - bar.low), abs(bar.high - bar.value))]


def _rotate_ticks(axes: Any, x: list[Any]) -> None:
    """Rotate the x tick labels when the axis is a timestamp (long labels overlap otherwise)."""
    if not x:
        return
    import datetime as _datetime

    if isinstance(x[0], (_datetime.date, _datetime.datetime)) or hasattr(x[0], "isoformat"):
        for label in axes.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment("right")


def _level_style(kind: str) -> Any:
    """Line style for a horizontal overlay, by role."""
    return {
        "previous_day": (0, (4, 3)),
        "today": (0, (1, 1)),
        "support": (0, (6, 2, 1, 2)),
        "resistance": (0, (6, 2, 1, 2)),
        "level": (0, (2, 2)),
    }.get(kind, (0, (2, 2)))


def _line_palette() -> list[str]:
    """A stable colour cycle for MA lines and curve series."""
    return [
        "#e67e22",
        "#2e86c1",
        "#8e44ad",
        "#16a085",
        "#c0392b",
        "#7f8c8d",
        "#d35400",
        "#2471a3",
        "#1e8449",
    ]


def _regime_palette() -> dict[str, str]:
    """One faint colour per NIFTY regime, keyed by the stored regime label."""
    return {
        "STRONG_BULLISH": "#1e8449",
        "BULLISH": "#7dcea0",
        "BULLISH_DEVELOPING": "#d4efdf",
        "NEUTRAL": "#f2f4f4",
        "BEARISH_DEVELOPING": "#fadbd8",
        "BEARISH": "#ec7063",
        "STRONG_BEARISH": "#b03a2e",
    }
