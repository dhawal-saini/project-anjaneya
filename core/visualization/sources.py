"""The loading half of spec 28: stored evidence in, chart-ready data out.

:mod:`core.visualization.charts` draws figures from plain data; this module is what turns the
things ANJANEYA already stores into that data. It is still pure - it reads the objects it is
handed (a list of
:class:`~core.backtest.engine.Trade` rows, a feature frame, a candle frame, an importance
JSON payload) and returns the dataclasses the builders consume. It opens no database, writes
no file and imports no app, so a frame and a handful of trades are enough to test it.

Three questions are answered here:

* **the P&L heatmaps** - the trade ledger grouped on two axes. Cells carry their sample and a
  PRELIMINARY flag computed from the configured minimum, because "three trades happened to
  land here" and "this court is measured" must never look the same on a chart. A cell nothing
  traded in is ``None``, never ``0``.
* **the CE-vs-PE curves and the regime bars** - the same ledger, ordered by entry time (the
  curves) or grouped by the NIFTY regime the entry ran in (the bars).
* **the price charts** - a stored candle series plus the overlays the feature row already
  carries: the contract's own moving averages recomputed from its candles (spec 10's stack,
  ``EMA8 low``/``EMA8 high``/``EMA20``/``EMA55``/``EMA200``), today's running high/low, the
  previous day's high/low, the level-event markers (breakout/rejection), the structure swings
  and the model's own predictions. The NIFTY chart is the same builder fed the index bars and
  the trend/regime columns.

Every ``*_heatmap`` / ``*_bars`` / ``*_curves`` function accepts duck-typed records (anything
with the trade attributes), which is what lets a test pass a tiny stub instead of a full
``Trade``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from core.indicators import moving_average
from core.visualization.charts import (
    ChartError,
    EventMarker,
    FeatureBar,
    Heatmap,
    LevelLine,
    PriceChartData,
)

__all__ = [
    "WEEKDAY_NAMES",
    "feature_bars",
    "nifty_price_data",
    "option_price_data",
    "pnl_heatmap",
    "regime_bars",
    "side_curves",
    "time_of_day_cycle_heatmap",
    "time_of_day_weekday_heatmap",
]

#: Weekday names, indexed by :meth:`datetime.date.weekday` (Monday is 0) - pandas and the
#: ledger both use that convention, so one order covers both.
WEEKDAY_NAMES: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: The state substrings that mean "price broke a level" and "price was turned back".
#: The sustained ``HELD_*`` states are deliberately *not* here: a marker is an event, and a level
#: a price is still holding would otherwise be re-marked on every bar.
_BREAKOUT_MARKS: tuple[str, ...] = ("CLOSED_ABOVE", "CROSSED_ABOVE", "BREAKOUT")
_REJECTION_MARKS: tuple[str, ...] = (
    "FAILED_BREAKOUT",
    "REJECTED",
    "CLOSED_BELOW",
    "CROSSED_BELOW",
)


# ---------------------------------------------------------------------------
# Small, defensive accessors (a record is anything with the trade attributes)
# ---------------------------------------------------------------------------
def _net(record: Any) -> float:
    """A record's net P&L as a float, treating an absent value as ``0`` for *summing* only.

    A missing ``net_pnl`` contributes nothing to a group total; it is never reported as a
    measured zero, because the cell's *value* is a sum (a real number) while its *sample* is the
    count of records that landed in it.
    """
    value = getattr(record, "net_pnl", None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - a non-numeric P&L is not a P&L
        return 0.0


def _stamp(record: Any) -> Any:
    """The moment a record entered (its entry timestamp), or ``None``."""
    return getattr(record, "entry_timestamp", None)


def _segment_value(record: Any, key: str) -> str | None:
    """The label a record contributes to one heatmap axis.

    ``day_of_week`` is special: the ledger stores it as an integer (Monday is 0) and it may be
    absent, so it is either mapped through :data:`WEEKDAY_NAMES` or derived from the entry
    timestamp. Every other key is read straight off the record. An absent label stays ``None``
    and the record is dropped from the grid rather than placed in a fabricated ``unknown`` cell.
    """
    if key == "day_of_week":
        value = getattr(record, "day_of_week", None)
        if value is None:
            moment = _stamp(record)
            value = pd.Timestamp(moment).weekday() if moment is not None else None
        if value is None:
            return None
        try:
            return WEEKDAY_NAMES[int(value) % 7]
        except (TypeError, ValueError):
            return None
    value = getattr(record, key, None)
    if value is None:
        return None
    text = str(value)
    return text or None


def _ordered(present: list[str], order: Sequence[str] | None) -> list[str]:
    """The labels to draw, in the configured order, with anything outside it appended sorted.

    A stale order can therefore never hide a label (the same rule the research matrices use): a
    bucket the calendar no longer lists still appears, at the end, rather than vanishing.
    """
    seen = list(dict.fromkeys(present))
    if not order:
        return sorted(seen)
    ranked = [label for label in order if label in seen]
    extra = sorted(label for label in seen if label not in order)
    return ranked + extra


# ---------------------------------------------------------------------------
# The P&L heatmaps
# ---------------------------------------------------------------------------
def pnl_heatmap(
    records: Sequence[Any],
    *,
    row_key: str,
    column_key: str,
    title: str,
    row_label: str,
    column_label: str,
    value_label: str = "net P&L",
    row_order: Sequence[str] | None = None,
    column_order: Sequence[str] | None = None,
    preliminary_min_samples: int = 1,
    center: bool = True,
) -> Heatmap:
    """Group ``records`` on two axes and sum their net P&L into a labelled grid.

    The grid is built from the labels actually present, ordered by ``row_order`` / ``column_order``
    where they apply and sorted otherwise. Every cell carries its sample and a PRELIMINARY flag; a
    cell nothing landed in holds ``None`` and a sample of ``0``, so the builder leaves it blank
    instead of drawing a confident ``0``.
    """
    totals: dict[tuple[str, str], float] = {}
    samples: dict[tuple[str, str], int] = {}
    for record in records:
        row = _segment_value(record, row_key)
        column = _segment_value(record, column_key)
        if row is None or column is None:
            continue
        key = (row, column)
        totals[key] = totals.get(key, 0.0) + _net(record)
        samples[key] = samples.get(key, 0) + 1

    rows = _ordered([key[0] for key in totals], row_order)
    columns = _ordered([key[1] for key in totals], column_order)
    if not rows or not columns:
        raise ChartError(
            f"heatmap {title!r} has nothing to plot: no record carried both "
            f"{row_key!r} and {column_key!r}"
        )

    values: list[list[float | None]] = []
    counts: list[list[int]] = []
    preliminary: list[list[bool]] = []
    for row in rows:
        values.append([totals.get((row, column)) for column in columns])
        counts.append([samples.get((row, column), 0) for column in columns])
        preliminary.append(
            [samples.get((row, column), 0) < preliminary_min_samples for column in columns]
        )
    return Heatmap(
        title=title,
        row_label=row_label,
        column_label=column_label,
        value_label=value_label,
        rows=rows,
        columns=columns,
        values=values,
        samples=counts,
        preliminary=preliminary,
        center=center,
    )


def time_of_day_cycle_heatmap(
    records: Sequence[Any],
    *,
    bucket_order: Sequence[str] | None = None,
    cycle_order: Sequence[str] | None = None,
    preliminary_min_samples: int = 1,
) -> Heatmap:
    """Spec 28's time-of-day x expiry-cycle-phase P&L heatmap (spec 24 item 2, drawn)."""
    return pnl_heatmap(
        records,
        row_key="time_of_day_bucket",
        column_key="expiry_cycle_phase",
        title="net P&L by time-of-day bucket x expiry-cycle phase",
        row_label="time-of-day bucket",
        column_label="expiry-cycle phase",
        row_order=bucket_order,
        column_order=cycle_order,
        preliminary_min_samples=preliminary_min_samples,
    )


def time_of_day_weekday_heatmap(
    records: Sequence[Any],
    *,
    bucket_order: Sequence[str] | None = None,
    preliminary_min_samples: int = 1,
) -> Heatmap:
    """Spec 28's time-of-day x day-of-week P&L heatmap.

    The weekday is taken from the ledger's own ``day_of_week`` when it is stored and derived from
    the entry timestamp otherwise, so a ledger written before that research column existed still
    charts.
    """
    return pnl_heatmap(
        records,
        row_key="time_of_day_bucket",
        column_key="day_of_week",
        title="net P&L by time-of-day bucket x day of week",
        row_label="time-of-day bucket",
        column_label="day of week",
        row_order=bucket_order,
        column_order=WEEKDAY_NAMES,
        preliminary_min_samples=preliminary_min_samples,
    )


# ---------------------------------------------------------------------------
# The CE-vs-PE curves and the regime-conditional bars
# ---------------------------------------------------------------------------
def side_curves(
    records: Sequence[Any],
    *,
    value: str = "net_pnl",
) -> dict[str, list[tuple[Any, float]]]:
    """One cumulative-P&L curve per option side, ordered by entry time.

    Spec 28 asks for "CE vs PE intraday P&L curves". A ledger can span many sessions, so the
    curve is cumulative net P&L against entry time - the honest "what did this side earn, trade
    by trade" picture. A side with fewer than two trades is still returned (the builder decides
    whether a single point is a curve), and a record whose ``option_type`` is absent is skipped
    rather than pooled into a nameless series.
    """
    grouped: dict[str, list[Any]] = {}
    for record in records:
        side = getattr(record, "option_type", None)
        if not side:
            continue
        grouped.setdefault(str(side), []).append(record)

    curves: dict[str, list[tuple[Any, float]]] = {}
    for side, rows in grouped.items():
        ordered = sorted(rows, key=lambda record: _stamp(record) or 0)
        running = 0.0
        points: list[tuple[Any, float]] = []
        for record in ordered:
            running += float(getattr(record, value, 0.0) or 0.0)
            points.append((_stamp(record), running))
        curves[side] = points
    return curves


def regime_bars(
    records: Sequence[Any],
    *,
    order: Sequence[str] | None = None,
) -> tuple[list[str], list[float], list[int]]:
    """Net P&L per NIFTY regime: ``(labels, values, samples)`` for a bar chart.

    The regime axis is the ledger's own ``nifty_state`` - the state the entry actually ran in -
    so the chart answers "does the regime the trade was entered in change what it earned". A
    record with no regime is dropped rather than gathered into a fabricated ``unknown`` bar.
    """
    totals: dict[str, float] = {}
    samples: dict[str, int] = {}
    for record in records:
        regime = getattr(record, "nifty_state", None)
        if not regime:
            continue
        label = str(regime)
        totals[label] = totals.get(label, 0.0) + _net(record)
        samples[label] = samples.get(label, 0) + 1
    labels = _ordered(list(totals), order)
    if not labels:
        raise ChartError("regime chart has nothing to plot: no record carried a nifty_state")
    return labels, [totals[label] for label in labels], [samples[label] for label in labels]


def feature_bars(
    payload: Mapping[str, Any],
    *,
    metric: str = "permutation_mean",
    limit: int | None = None,
) -> list[FeatureBar]:
    """Turn a stored importance JSON into the bars spec 28's importance chart draws.

    Only *measured* features with a real value for ``metric`` become bars: an unmeasured feature
    (all-``NULL`` or constant in the sample) carries a reason and no number, and is deliberately
    left out rather than drawn as a zero. ``metric`` selects which column of the importance
    artefact is charted (``permutation_mean`` with its interval by default, or ``native``).
    """
    features = payload.get("features")
    if not isinstance(features, list):
        raise ChartError(
            "the importance payload carries no 'features' list - is this a spec 27 artefact?"
        )
    bars: list[FeatureBar] = []
    for entry in features:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("measured") is False:
            continue
        value = entry.get(metric)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        # The interval columns belong to the permutation metric, so they are attached only when
        # the chart is showing that metric - a native-importance bar must not wear a permutation
        # whisker.
        if metric == "permutation_mean":
            low = entry.get("ci_low")
            high = entry.get("ci_high")
            low_value = float(low) if low is not None else None
            high_value = float(high) if high is not None else None
        else:
            low_value = high_value = None
        bars.append(
            FeatureBar(
                column=str(entry.get("column", "?")),
                value=number,
                low=low_value,
                high=high_value,
                topic=str(entry["topic"]) if entry.get("topic") else None,
            )
        )
    bars.sort(key=lambda bar: abs(bar.value), reverse=True)
    return bars[:limit] if limit else bars


# ---------------------------------------------------------------------------
# The price charts (one option contract, and the NIFTY index)
# ---------------------------------------------------------------------------
def _numbers(values: Any) -> list[float | None]:
    """A pandas Series as a plain list, with ``NaN``/``NaT``/``None`` mapped to ``None``."""
    return [None if _missing(value) else float(value) for value in values]


def _missing(value: Any) -> bool:
    """Whether a scalar is absent (``None``, ``NaN`` or ``NaT``) - never ``0``."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic scalar, treat as present
        return False


def _column(frame: Any, name: str) -> list[Any] | None:
    """One column of a frame as a list, or ``None`` when the frame does not carry it."""
    if frame is None or getattr(frame, "empty", True) or name not in frame.columns:
        return None
    return list(frame[name])


def _by_timestamp(frame: Any, column: str) -> dict[Any, Any]:
    """A ``timestamp -> value`` map from a frame, for aligning overlays to another series."""
    if frame is None or getattr(frame, "empty", True) or column not in frame.columns:
        return {}
    return {
        row["timestamp"]: row[column]
        for _, row in frame.iterrows()
        if not _missing(row[column])
    }


def _contract_moving_averages(
    candles: Any,
    moving_averages: Sequence[Any] | None,
) -> dict[str, list[float | None]]:
    """Recompute spec 10's option MA stack from the contract's own candles.

    The stored feature row keeps the *derived* option-trend columns (alignment, slope, spread),
    not the MA lines themselves, so the chart rebuilds them here with the very same
    :func:`core.indicators.moving_average` the engine used - one definition, one result. A
    source the candle row does not carry (a greeks column, say) is skipped rather than guessed.
    """
    result: dict[str, list[float | None]] = {}
    for spec in moving_averages or ():
        source = getattr(spec, "source", "close")
        if source not in candles.columns:
            continue
        values = moving_average(
            candles[source], ma_type=getattr(spec, "type", "ema"), period=int(spec.period)
        )
        result[str(spec.name)] = _numbers(values)
    return result


def _level_overlays(features: Any) -> list[LevelLine]:
    """The horizontal overlays a feature frame carries: previous-day levels and the swings.

    Only the *last* non-null value of each column is used for a constant line, but the previous
    day's high/low and the newest swing are what a reader expects to see named across the
    window. A column that is entirely ``NULL`` produces no line (not a line at zero).
    """
    overlays: list[LevelLine] = []
    for column, label, kind in (
        ("level_previous_day_high", "previous-day high", "previous_day"),
        ("level_previous_day_low", "previous-day low", "previous_day"),
        ("swing_high_1", "swing high", "resistance"),
        ("swing_low_1", "swing low", "support"),
    ):
        values = _column(features, column)
        value = next((float(v) for v in reversed(values) if not _missing(v)), None) if values else None
        if value is not None:
            overlays.append(LevelLine(label=label, value=value, kind=kind))
    return overlays


def _event_markers(features: Any, closes: dict[Any, Any]) -> list[EventMarker]:
    """Breakout/rejection markers from the feature row's four per-level state columns.

    A state is turned into a marker only when it names an event (a break or a rejection); the
    quiet states (``APPROACHING``, ``HELD_*``) are not events. The marker is placed at the
    candle's own close for that timestamp, so it sits on the bar the feature row describes; a
    timestamp with no matching candle is skipped rather than placed at an invented price.
    """
    markers: list[EventMarker] = []
    for column in ("event_prev_day_high_state", "event_prev_day_low_state",
                   "event_today_high_state", "event_today_low_state"):
        if features is None or column not in features.columns:
            continue
        for _, row in features.iterrows():
            state = row[column]
            if _missing(state):
                continue
            text = str(state)
            kind = _event_kind(text)
            if kind is None:
                continue
            price = closes.get(row["timestamp"])
            if _missing(price):
                continue
            markers.append(
                EventMarker(
                    timestamp=row["timestamp"],
                    value=float(price),
                    label=f"{column.replace('event_', '').replace('_state', '')}: {text}",
                    kind=kind,
                )
            )
    return markers


def _event_kind(state: str) -> str | None:
    """Classify a level state as ``breakout``, ``rejection`` or not an event."""
    if state in _BREAKOUT_MARKS:
        return "breakout"
    if state in _REJECTION_MARKS:
        return "rejection"
    return None


def _session_running(bars: Any) -> tuple[list[float | None], list[float | None]]:
    """Today's running high and low at every bar (cumulative within the bar's own session)."""
    frame = bars[["timestamp", "high", "low"]].copy()
    frame["_day"] = [pd.Timestamp(moment).date() for moment in frame["timestamp"]]
    highs = frame.groupby("_day")["high"].cummax()
    lows = frame.groupby("_day")["low"].cummin()
    return _numbers(highs), _numbers(lows)


def _previous_session_levels(bars: Any) -> tuple[list[float | None], list[float | None]]:
    """The previous session's high and low, carried forward across each later session."""
    frame = bars[["timestamp", "high", "low"]].copy()
    frame["_day"] = [pd.Timestamp(moment).date() for moment in frame["timestamp"]]
    daily = frame.groupby("_day").agg(day_high=("high", "max"), day_low=("low", "min"))
    previous_high = daily["day_high"].shift(1)
    previous_low = daily["day_low"].shift(1)
    highs = frame["_day"].map(previous_high)
    lows = frame["_day"].map(previous_low)
    return _numbers(highs), _numbers(lows)


def option_price_data(
    candles: Any,
    features: Any = None,
    *,
    symbol: str,
    timeframe: str,
    moving_averages: Sequence[Any] | None = None,
    predictions: Sequence[EventMarker] | None = None,
) -> PriceChartData:
    """Build one option contract's price chart data from its stored candles and feature rows.

    The candles are the price axis and the source of the MA stack (spec 10's
    ``EMA8 low``/``EMA8 high``/``EMA20``/``EMA55``/``EMA200``); the feature rows supply the
    overlays (previous-day high/low, the newest swings), the running today high/low and the
    level-event markers. Both frames are expected to be scoped to one contract and timeframe by
    the caller - the function does not filter, it *uses what it is given*, so a mismatched frame
    is a caller's bug rather than a silently empty chart.
    """
    if candles is None or getattr(candles, "empty", True):
        raise ChartError(f"no stored candles for {symbol!r} ({timeframe}) - build them first")
    frame = candles.sort_values("timestamp").reset_index(drop=True)

    closes = {row["timestamp"]: row["close"] for _, row in frame.iterrows()}
    running_high = _by_timestamp(features, "level_today_high_running")
    running_low = _by_timestamp(features, "level_today_low_running")
    trend_values = _by_timestamp(features, "nifty_composite_trend_score")
    regime_values = _by_timestamp(features, "nifty_regime")
    timestamps = list(frame["timestamp"])

    return PriceChartData(
        title="option",
        symbol=symbol,
        timeframe=timeframe,
        timestamps=timestamps,
        close=_numbers(frame["close"]),
        moving_averages=_contract_moving_averages(frame, moving_averages),
        levels=_level_overlays(features),
        running={
            "today high (running)": [running_high.get(moment) for moment in timestamps],
            "today low (running)": [running_low.get(moment) for moment in timestamps],
        },
        markers=_event_markers(features, closes),
        predictions=list(predictions or []),
        trend=[trend_values.get(moment) for moment in timestamps],
        regimes=[regime_values.get(moment) for moment in timestamps],
    )


def nifty_price_data(
    bars: Any,
    features: Any = None,
    *,
    symbol: str,
    timeframe: str,
    moving_averages: Sequence[Any] | None = None,
) -> PriceChartData:
    """Build the NIFTY index price chart data from the stored index bars.

    The index bars are the price axis and the source of the MA lines and of the running
    today/previous-day high-low overlays (the index's own session extremes). The NIFTY trend
    score and regime are read off the scoped feature rows - the NIFTY block is identical across
    the contracts captured at one instant, so any one contract's row answers it - and shaded
    behind the price.
    """
    if bars is None or getattr(bars, "empty", True):
        raise ChartError(f"no stored index bars for {symbol!r} ({timeframe}) - backfill them first")
    frame = bars.sort_values("timestamp").reset_index(drop=True)

    previous_high, previous_low = _previous_session_levels(frame)
    running_high, running_low = _session_running(frame)
    trend_values = _by_timestamp(features, "nifty_composite_trend_score")
    regime_values = _by_timestamp(features, "nifty_regime")
    timestamps = list(frame["timestamp"])

    overlays: list[LevelLine] = []
    last_previous_high = next((v for v in reversed(previous_high) if v is not None), None)
    last_previous_low = next((v for v in reversed(previous_low) if v is not None), None)
    if last_previous_high is not None:
        overlays.append(LevelLine("previous-day high", last_previous_high, "previous_day"))
    if last_previous_low is not None:
        overlays.append(LevelLine("previous-day low", last_previous_low, "previous_day"))

    return PriceChartData(
        title="NIFTY",
        symbol=symbol,
        timeframe=timeframe,
        timestamps=timestamps,
        close=_numbers(frame["close"]),
        moving_averages=_contract_moving_averages(frame, moving_averages),
        levels=overlays,
        running={
            "today high (running)": running_high,
            "today low (running)": running_low,
        },
        trend=[trend_values.get(moment) for moment in timestamps],
        regimes=[regime_values.get(moment) for moment in timestamps],
    )
