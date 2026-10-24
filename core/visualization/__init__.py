"""Spec 28's charts - the figures that make a result checkable by eye.

Split into two halves that never need each other's globals:

* :mod:`core.visualization.charts` - the pure builders. Data in, a matplotlib figure out; no
  file I/O, no database, no app.
* :mod:`core.visualization.sources` - the loaders that turn stored evidence (a trade ledger, a
  feature frame, a candle frame, an importance JSON) into the builders' inputs.

The writing half is :mod:`apps.anjaneeya_ai.chart_writer` (atomic PNGs plus an index
Markdown), driven by ``--mode charts``.
"""

from __future__ import annotations

from core.visualization.charts import (
    CHART_KINDS,
    ChartError,
    ChartPolicy,
    EventMarker,
    FeatureBar,
    Heatmap,
    LevelLine,
    PriceChartData,
    bar_chart,
    curve_chart,
    heatmap,
    importance_bars,
    price_chart,
)
from core.visualization.sources import (
    WEEKDAY_NAMES,
    feature_bars,
    nifty_price_data,
    option_price_data,
    pnl_heatmap,
    regime_bars,
    side_curves,
    time_of_day_cycle_heatmap,
    time_of_day_weekday_heatmap,
)

__all__ = [
    "CHART_KINDS",
    "WEEKDAY_NAMES",
    "ChartError",
    "ChartPolicy",
    "EventMarker",
    "FeatureBar",
    "Heatmap",
    "LevelLine",
    "PriceChartData",
    "bar_chart",
    "curve_chart",
    "feature_bars",
    "heatmap",
    "importance_bars",
    "nifty_price_data",
    "option_price_data",
    "pnl_heatmap",
    "price_chart",
    "regime_bars",
    "side_curves",
    "time_of_day_cycle_heatmap",
    "time_of_day_weekday_heatmap",
]
