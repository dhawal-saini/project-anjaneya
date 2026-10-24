"""Tests for :mod:`core.visualization` - spec 28's chart builders and loaders.

The builders are the only place in ANJANEYA that draws anything, so the tests are about the two
rules the module promises plus the shapes that feed them:

* **a figure is produced from data, and an impossible one is refused.** Every builder renders
  from a handful of numbers and raises :class:`ChartError` (never a half-empty figure) when it is
  handed nothing to plot - an empty price series, a grid with no rows, a series with one point.
* **a value that is not there is left blank, not drawn as zero.** A heatmap cell nothing traded
  in holds ``None`` and a zero sample; an unmeasured feature never becomes a bar.
* **the sample travels with the picture.** A cell below the configured minimum is flagged
  preliminary, and a heatmap's grid carries each cell's trade count.
* **the loaders read what is stored.** The moving-average stack is recomputed from the
  contract's own candles with the engine's own function, the breakouts/rejections come from the
  feature row's state columns, and the CE/PE curves and regime bars come from the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.config_loader import load_config
from core.visualization import (
    ChartError,
    ChartPolicy,
    EventMarker,
    FeatureBar,
    Heatmap,
    LevelLine,
    PriceChartData,
    bar_chart,
    curve_chart,
    feature_bars,
    heatmap,
    importance_bars,
    nifty_price_data,
    option_price_data,
    pnl_heatmap,
    price_chart,
    regime_bars,
    side_curves,
    time_of_day_cycle_heatmap,
    time_of_day_weekday_heatmap,
)

IST = ZoneInfo("Asia/Kolkata")

#: A tiny figure (fast to render, no display) used by every builder test.
POLICY = ChartPolicy(dpi=40, width=4.0, height=3.0, tall_height=3.0)


def moment(day: int, hour: int, minute: int) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=IST)


@dataclass
class Record:
    """A duck-typed stand-in for a ledger :class:`~core.backtest.engine.Trade`."""

    entry_timestamp: datetime | None = None
    time_of_day_bucket: str | None = None
    expiry_cycle_phase: str | None = None
    day_of_week: int | None = None
    option_type: str | None = None
    nifty_state: str | None = None
    net_pnl: float | None = None


@dataclass
class MaSpec:
    """A stand-in for the configured :class:`~core.config_loader.MovingAverageSpec`."""

    name: str
    type: str
    period: int
    source: str


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_the_default_policy_is_the_shipped_presentation() -> None:
    """The dataclass defaults and the shipped ``charts:`` block must not drift apart."""
    policy = ChartPolicy()
    assert policy.spec == "28"
    assert policy.format == "png"
    assert policy.colormap == "RdYlGn"
    assert policy.top_features == 20


def test_from_config_reads_the_shipped_block() -> None:
    """``ChartPolicy.from_config`` resolves the six YAML files' own numbers."""
    policy = ChartPolicy.from_config(load_config("config"))
    assert policy.spec == "28"
    assert policy.format == "png"
    assert policy.dpi == 120
    assert policy.colormap == "RdYlGn"
    assert policy.top_features == 20
    assert policy.max_points == 30_000
    assert policy.max_markers == 60


def test_to_dict_round_trips_every_knob() -> None:
    """The policy serialises to the same keys the JSON payload carries."""
    data = ChartPolicy().to_dict()
    assert data["spec"] == "28"
    assert data["top_features"] == 20
    assert set(data) == {
        "spec",
        "digits",
        "dpi",
        "width",
        "height",
        "tall_height",
        "format",
        "colormap",
        "sequential_colormap",
        "top_features",
        "max_points",
        "max_markers",
    }


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    """Close every figure after a test, so many renders never pile up."""
    yield None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")


def sample_price_data() -> PriceChartData:
    """A small contract series with one of every overlay the builder understands."""
    stamps = [moment(1, 9, 20) + timedelta(minutes=index) for index in range(8)]
    return PriceChartData(
        title="option",
        symbol="NIFTY01OCT2625000CE",
        timeframe="1m",
        timestamps=stamps,
        close=[100.0 + index for index in range(8)],
        moving_averages={"ema_8_high": [100.0 + 1.1 * index for index in range(8)]},
        levels=[LevelLine("previous-day high", 108.0, "previous_day")],
        running={"today high (running)": [100.0 + index for index in range(8)]},
        markers=[EventMarker(stamps[3], 103.0, "breakout", "breakout")],
        predictions=[EventMarker(stamps[5], 105.0, "UP", "prediction")],
        trend=[0.1 * index for index in range(8)],
        regimes=["BULLISH"] * 4 + ["BEARISH"] * 4,
    )


# ---------------------------------------------------------------------------
# The price chart
# ---------------------------------------------------------------------------
def test_price_chart_draws_a_figure() -> None:
    """A series with overlays renders into a figure with axes."""
    figure = price_chart(sample_price_data(), policy=POLICY)
    assert figure.axes
    assert "spec 28" in figure.axes[0].get_title()


def test_price_chart_survives_missing_overlays() -> None:
    """An empty MA map, no trend and no regime still draw (they are simply absent)."""
    data = PriceChartData(
        title="NIFTY",
        symbol="NIFTY",
        timeframe="1m",
        timestamps=[moment(1, 9, 20) + timedelta(minutes=i) for i in range(4)],
        close=[10.0, 11.0, 10.5, 12.0],
    )
    figure = price_chart(data, policy=POLICY)
    assert figure.axes


def test_price_chart_refuses_a_series_with_no_points() -> None:
    """An empty series is a refusal, not an empty picture."""
    data = PriceChartData(title="option", symbol="X", timeframe="1m", timestamps=[], close=[])
    with pytest.raises(ChartError):
        price_chart(data, policy=POLICY)


# ---------------------------------------------------------------------------
# The heatmap
# ---------------------------------------------------------------------------
def sample_heatmap() -> Heatmap:
    return Heatmap(
        title="net P&L by bucket x phase",
        row_label="bucket",
        column_label="phase",
        value_label="net P&L",
        rows=["OPEN_915_930", "MORNING_930_1100"],
        columns=["START_OF_CYCLE", "EXPIRY_DAY"],
        values=[[10.0, None], [-5.0, 3.0]],
        samples=[[2, 0], [1, 4]],
        preliminary=[[False, False], [True, False]],
        center=True,
    )


def test_heatmap_draws_every_cell_and_hatches_a_preliminary_one() -> None:
    """A value cell prints its value and sample; a blank cell prints only its count."""
    figure = heatmap(sample_heatmap(), policy=POLICY)
    texts = [text.get_text() for text in figure.axes[0].texts]
    assert any("10" in text and "n=2" in text for text in texts)
    assert any(text == "-" for text in texts)
    # one hatched patch for the preliminary cell
    assert len(figure.axes[0].patches) == 1


def test_heatmap_refuses_an_empty_grid() -> None:
    """A grid with no rows or columns is a refusal."""
    empty = Heatmap(
        title="empty",
        row_label="x",
        column_label="y",
        value_label="z",
        rows=[],
        columns=[],
        values=[],
        samples=[],
        preliminary=[],
    )
    with pytest.raises(ChartError):
        heatmap(empty, policy=POLICY)


def test_heatmap_to_dict_counts_cells_and_preliminary() -> None:
    """The note the index prints is derived from the grid, not recomputed by the caller."""
    summary = sample_heatmap().to_dict()
    assert summary["cells"] == 3
    assert summary["preliminary_cells"] == 1


# ---------------------------------------------------------------------------
# The curve chart
# ---------------------------------------------------------------------------
def test_curve_chart_draws_two_series() -> None:
    curves = {
        "CE": [(moment(1, 9, 20), 0.0), (moment(1, 9, 30), 10.0), (moment(1, 9, 40), 5.0)],
        "PE": [(moment(1, 9, 20), 0.0), (moment(1, 9, 30), -3.0)],
    }
    figure = curve_chart(curves, title="CE vs PE", xlabel="t", ylabel="pnl", policy=POLICY)
    assert len(figure.axes[0].lines) >= 2


def test_curve_chart_skips_a_single_point_series() -> None:
    """One point is not a curve; if every series is one point the chart refuses."""
    with pytest.raises(ChartError):
        curve_chart(
            {"CE": [(moment(1, 9, 20), 0.0)]},
            title="lonely",
            xlabel="t",
            ylabel="pnl",
            policy=POLICY,
        )


# ---------------------------------------------------------------------------
# The bar charts
# ---------------------------------------------------------------------------
def test_bar_chart_draws_and_labels_samples() -> None:
    labels, values, samples = regime_bars(
        [
            Record(nifty_state="BULLISH", net_pnl=10.0),
            Record(nifty_state="BULLISH", net_pnl=5.0),
            Record(nifty_state="BEARISH", net_pnl=-8.0),
        ]
    )
    figure = bar_chart(
        labels, values, title="t", xlabel="x", ylabel="y", samples=samples, policy=POLICY
    )
    assert figure.axes[0].patches


def test_bar_chart_refuses_when_every_value_is_missing() -> None:
    with pytest.raises(ChartError):
        bar_chart(["A"], [None], title="t", xlabel="x", ylabel="y", policy=POLICY)


def test_importance_bars_ranks_by_absolute_value_and_caps() -> None:
    """A large negative delta is as important as a large positive one."""
    bars = [
        FeatureBar("a", 0.1),
        FeatureBar("b", -0.9),
        FeatureBar("c", 0.5),
    ]
    figure = importance_bars(
        bars, title="t", value_label="delta", policy=ChartPolicy(dpi=40, top_features=2)
    )
    labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
    assert labels == ["c", "b"]  # reversed for a horizontal chart: largest at the top


def test_importance_bars_refuse_an_empty_list() -> None:
    with pytest.raises(ChartError):
        importance_bars([], title="t", value_label="delta", policy=POLICY)


# ---------------------------------------------------------------------------
# The heatmap loader (core.visualization.sources)
# ---------------------------------------------------------------------------
def test_pnl_heatmap_totals_samples_and_flags_preliminary() -> None:
    """A cell's value is the summed net P&L; its sample is the count; small cells are flagged."""
    records = [
        Record(time_of_day_bucket="OPEN_915_930", expiry_cycle_phase="MID_CYCLE", net_pnl=10.0),
        Record(time_of_day_bucket="OPEN_915_930", expiry_cycle_phase="MID_CYCLE", net_pnl=-4.0),
        Record(time_of_day_bucket="MORNING_930_1100", expiry_cycle_phase="EXPIRY_DAY", net_pnl=7.0),
    ]
    cells = time_of_day_cycle_heatmap(records, preliminary_min_samples=2)
    assert cells.rows == ["MORNING_930_1100", "OPEN_915_930"]
    assert cells.columns == ["EXPIRY_DAY", "MID_CYCLE"]
    index = cells.rows.index("OPEN_915_930")
    column = cells.columns.index("MID_CYCLE")
    assert cells.values[index][column] == 6.0
    assert cells.samples[index][column] == 2
    assert cells.preliminary[index][column] is False


def test_pnl_heatmap_orders_known_labels_and_appends_the_rest() -> None:
    """The configured order leads; a label outside it is appended, never dropped."""
    records = [
        Record(time_of_day_bucket="LATE", expiry_cycle_phase="X", net_pnl=1.0),
        Record(time_of_day_bucket="EARLY", expiry_cycle_phase="X", net_pnl=1.0),
    ]
    cells = time_of_day_cycle_heatmap(records, bucket_order=["EARLY", "MIDDAY"])
    assert cells.rows == ["EARLY", "LATE"]


def test_pnl_heatmap_refuses_when_nothing_carries_both_axes() -> None:
    with pytest.raises(ChartError):
        pnl_heatmap(
            [Record(time_of_day_bucket="X")],
            row_key="time_of_day_bucket",
            column_key="expiry_cycle_phase",
            title="t",
            row_label="r",
            column_label="c",
        )


def test_the_weekday_axis_is_derived_from_the_entry_when_the_column_is_null() -> None:
    """A ledger written before ``day_of_week`` existed still charts its weekday."""
    records = [
        Record(entry_timestamp=moment(1, 9, 20), time_of_day_bucket="OPEN_915_930", net_pnl=1.0)
    ]
    cells = time_of_day_weekday_heatmap(records)
    assert cells.columns == ["Thu"]  # 2026-10-01 is a Thursday


def test_the_weekday_axis_uses_the_stored_column_when_present() -> None:
    records = [
        Record(
            entry_timestamp=moment(1, 9, 20),
            time_of_day_bucket="OPEN_915_930",
            day_of_week=0,
            net_pnl=1.0,
        )
    ]
    cells = time_of_day_weekday_heatmap(records)
    assert cells.columns == ["Mon"]


# ---------------------------------------------------------------------------
# The curves and the regime bars
# ---------------------------------------------------------------------------
def test_side_curves_are_cumulative_and_ordered_by_entry_time() -> None:
    """The CE curve is the running sum of CE net P&L in entry order."""
    records = [
        Record(entry_timestamp=moment(1, 10, 0), option_type="CE", net_pnl=10.0),
        Record(entry_timestamp=moment(1, 9, 30), option_type="CE", net_pnl=5.0),
        Record(entry_timestamp=moment(1, 9, 45), option_type="PE", net_pnl=-3.0),
    ]
    curves = side_curves(records)
    assert curves["CE"] == [(moment(1, 9, 30), 5.0), (moment(1, 10, 0), 15.0)]
    assert curves["PE"] == [(moment(1, 9, 45), -3.0)]


def test_side_curves_skip_a_record_without_a_side() -> None:
    curves = side_curves([Record(entry_timestamp=moment(1, 9, 30), net_pnl=1.0)])
    assert curves == {}


def test_regime_bars_total_and_order_by_the_configured_sequence() -> None:
    labels, values, samples = regime_bars(
        [
            Record(nifty_state="BULLISH", net_pnl=10.0),
            Record(nifty_state="BULLISH", net_pnl=2.0),
            Record(nifty_state="BEARISH", net_pnl=-5.0),
        ],
        order=("STRONG_BEARISH", "BEARISH", "NEUTRAL", "BULLISH", "STRONG_BULLISH"),
    )
    assert labels == ["BEARISH", "BULLISH"]
    assert values == [-5.0, 12.0]
    assert samples == [1, 2]


def test_regime_bars_refuse_without_a_regime() -> None:
    with pytest.raises(ChartError):
        regime_bars([Record(net_pnl=1.0)])


# ---------------------------------------------------------------------------
# The importance bars loader
# ---------------------------------------------------------------------------
def importance_payload() -> dict:
    return {
        "features": [
            {
                "column": "a",
                "topic": "t",
                "measured": True,
                "permutation_mean": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
            },
            {
                "column": "b",
                "topic": "t",
                "measured": True,
                "permutation_mean": -0.9,
                "ci_low": -1.1,
                "ci_high": -0.7,
            },
            {
                "column": "c",
                "topic": "t",
                "measured": False,
                "permutation_mean": None,
                "reason": "constant in the sample",
            },
        ]
    }


def test_feature_bars_skip_unmeasured_features() -> None:
    bars = feature_bars(importance_payload())
    assert [bar.column for bar in bars] == ["b", "a"]  # ranked by |value|
    assert all(bar.column != "c" for bar in bars)


def test_feature_bars_attach_intervals_only_for_the_permutation_metric() -> None:
    """A native-importance bar must not wear a permutation whisker."""
    permutation = feature_bars(importance_payload(), metric="permutation_mean")
    assert permutation[0].low == -1.1 and permutation[0].high == -0.7
    native_payload = {
        "features": [
            {"column": "a", "measured": True, "native": 12.0, "ci_low": 0.4, "ci_high": 0.6}
        ]
    }
    native = feature_bars(native_payload, metric="native")
    assert native[0].value == 12.0
    assert native[0].low is None and native[0].high is None


def test_feature_bars_refuse_a_payload_without_features() -> None:
    with pytest.raises(ChartError):
        feature_bars({"nope": 1})


# ---------------------------------------------------------------------------
# The price-chart loaders
# ---------------------------------------------------------------------------
def candles_frame() -> pd.DataFrame:
    stamps = [moment(1, 9, 20) + timedelta(minutes=index) for index in range(5)]
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        }
    )


def features_frame() -> pd.DataFrame:
    stamps = [moment(1, 9, 20) + timedelta(minutes=index) for index in range(5)]
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "tradingsymbol": ["NIFTY01OCT2625000CE"] * 5,
            "level_today_high_running": [101.0, 102.0, 103.0, 104.0, 105.0],
            "level_today_low_running": [99.0, 99.0, 99.0, 99.0, 99.0],
            "level_previous_day_high": [None, None, None, None, 108.0],
            "level_previous_day_low": [None, None, None, None, None],
            "swing_high_1": [None, None, None, None, None],
            "swing_low_1": [None, None, None, None, 95.0],
            "event_prev_day_high_state": [None, "APPROACHING", "CLOSED_ABOVE", "HELD_ABOVE", None],
            "event_prev_day_low_state": [None, None, None, "REJECTED", None],
            "event_today_high_state": [None, None, None, None, None],
            "event_today_low_state": [None, None, None, None, None],
        }
    )


def test_option_price_data_recomputes_the_configured_move_average_stack() -> None:
    """The MA lines come from the contract's own candles, via the engine's own function."""
    data = option_price_data(
        candles_frame(),
        features_frame(),
        symbol="NIFTY01OCT2625000CE",
        timeframe="1m",
        moving_averages=[MaSpec("ema_2_high", "ema", 2, "high")],
    )
    assert data.moving_averages["ema_2_high"][:1] == [None]  # EMA is not published early
    assert data.moving_averages["ema_2_high"][-1] == pytest.approx(104.5, abs=0.5)


def test_option_price_data_aligns_running_overlays_to_the_candle_axis() -> None:
    data = option_price_data(
        candles_frame(), features_frame(), symbol="NIFTY01OCT2625000CE", timeframe="1m"
    )
    assert data.running["today high (running)"] == [101.0, 102.0, 103.0, 104.0, 105.0]


def test_option_price_data_marks_only_breakouts_and_rejections() -> None:
    """``APPROACHING``/``HELD_ABOVE`` are states, not events; only the breaks and rejections mark."""
    data = option_price_data(
        candles_frame(), features_frame(), symbol="NIFTY01OCT2625000CE", timeframe="1m"
    )
    kinds = sorted(marker.kind for marker in data.markers)
    assert kinds == ["breakout", "rejection"]
    assert all(marker.value in (102.5, 103.5) for marker in data.markers)


def test_option_price_data_carries_the_previous_day_level_when_present() -> None:
    data = option_price_data(
        candles_frame(), features_frame(), symbol="NIFTY01OCT2625000CE", timeframe="1m"
    )
    labels = {level.label: level.value for level in data.levels}
    assert labels["previous-day high"] == 108.0
    assert labels["swing low"] == 95.0
    assert "previous-day low" not in labels  # the column is entirely NULL -> no line


def test_option_price_data_allows_the_model_s_own_predictions() -> None:
    data = option_price_data(
        candles_frame(),
        features_frame(),
        symbol="NIFTY01OCT2625000CE",
        timeframe="1m",
        predictions=[EventMarker(moment(1, 9, 20), 100.5, "UP", "prediction")],
    )
    assert len(data.predictions) == 1


def test_option_price_data_refuses_without_candles() -> None:
    with pytest.raises(ChartError):
        option_price_data(
            pd.DataFrame(), None, symbol="X", timeframe="1m"
        )


def index_frame() -> pd.DataFrame:
    bars = []
    for day, base in ((30, 25_000.0), (31, 25_100.0)):
        for index in range(3):
            bars.append(
                {
                    "timestamp": moment(day, 9, 20) + timedelta(minutes=index),
                    "open": base + index,
                    "high": base + 5.0,
                    "low": base - 5.0,
                    "close": base + 1.0 + index,
                }
            )
    return pd.DataFrame(bars)


def test_nifty_price_data_computes_running_and_previous_session_levels() -> None:
    data = nifty_price_data(index_frame(), None, symbol="NIFTY", timeframe="1m")
    # the running high never falls inside a session
    assert data.running["today high (running)"][:3] == [25_005.0, 25_005.0, 25_005.0]
    # the second session's previous-day high is the first session's high
    labels = {level.label: level.value for level in data.levels}
    assert labels["previous-day high"] == pytest.approx(25_005.0)


def test_nifty_price_data_refuses_without_bars() -> None:
    with pytest.raises(ChartError):
        nifty_price_data(pd.DataFrame(), None, symbol="NIFTY", timeframe="1m")
