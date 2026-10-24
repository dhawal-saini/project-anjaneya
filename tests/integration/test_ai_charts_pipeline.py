"""End-to-end ANJANEYA-AI charts (Stage 5.3, spec 28).

``--mode charts`` is exercised through the real ``main()`` against a seeded database, a real
ledger artefact and a stored importance artefact, so every assertion is about what actually
happened:

    the inputs    ->  the ledger Parquet, the importance JSON and the stored candles / features /
                      index bars, read back by the mode itself rather than handed in
    the figures   ->  one image per spec-28 chart under ``data/reports/charts/``, plus the index
                      Markdown that names them, all registered on the run manifest
    the controls  ->  ``--only`` narrows to a family, ``--metric`` picks the importance column,
                      ``--dry-run`` and ``--no-write`` write nothing
    the honesty   ->  a missing ledger skips the P&L charts with a reason (it does not fail the
                      run), no chart at all is ``NO_DATA``, and nothing is written to DuckDB

Nothing here touches the network: the only inputs are the stored database and the artefacts on
disk.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from apps.anjaneeya_ai.backtest_writer import ledger_frame
from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_OK
from core.backtest import Trade
from core.storage import connect
from tests.support.ai import cli_args, seed_source, shrink_config
from tests.support.engine import make_index_row, seed_index_history

IST = ZoneInfo("Asia/Kolkata")

#: The four fixture sessions (Mon 2026-09-28 .. Thu 2026-10-01, the weekly expiry).
DAYS: tuple[date, ...] = (
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
)

#: ``(hour, minute, time-of-day bucket)`` of the three entries the fixture trades.
SLOTS: tuple[tuple[int, int, str], ...] = (
    (9, 20, "OPEN_915_930"),
    (10, 30, "MORNING_930_1100"),
    (14, 45, "CLOSE_1430_1515"),
)

#: The seven charts the shipped configuration produces (see ``CHART_NAMES`` in the app).
ALL_CHARTS: tuple[str, ...] = (
    "heatmap_time_of_day_cycle_phase",
    "heatmap_time_of_day_day_of_week",
    "curves_ce_vs_pe",
    "regime_performance",
    "importance_bars",
    "price_nifty",
    "price_option",
)

#: Spec 27's artefact the bars chart reads.
IMPORTANCE_NAME = "1m_h5_2026-09-28_2026-10-01_lightgbm_importance.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def charts_dir(tmp_path) -> Path:
    return tmp_path / "data" / "reports" / "charts"


def backtests_dir(tmp_path) -> Path:
    return tmp_path / "data" / "backtests"


def importance_dir(tmp_path) -> Path:
    return tmp_path / "data" / "reports" / "importance"


def entry(day: date, hour: int, minute: int, *, number: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST) + timedelta(
        minutes=number
    )


def ledger_rows() -> list[Trade]:
    """A ledger with three buckets, both option sides and a winner/loser split on each side."""
    rows: list[Trade] = []
    index = 0
    for day in DAYS:
        phase = "EXPIRY_DAY" if day == DAYS[-1] else "MID_CYCLE"
        for hour, minute, bucket in SLOTS:
            for option_type, sign in (("CE", 1.0), ("PE", -1.0)):
                for number in range(2):
                    moment = entry(day, hour, minute, number=number)
                    net = sign * (100.0 + 10.0 * number)
                    rows.append(
                        Trade(
                            trade_id=f"run|{option_type}|{index}",
                            run_id="run",
                            entry_timestamp=moment,
                            tradingsymbol=f"NIFTY01OCT2625000{option_type}",
                            option_type=option_type,
                            entry_price=120.0,
                            quantity=75,
                            exit_timestamp=moment + timedelta(minutes=20),
                            expiry=DAYS[-1],
                            strike=25000.0,
                            exit_price=121.0,
                            exit_reason="TARGET" if net > 0 else "STOP_LOSS",
                            nifty_state="BULLISH" if option_type == "CE" else "BEARISH",
                            ai_prediction="UP_CONTINUATION" if net > 0 else "DOWN_CONTINUATION",
                            ai_probability=0.55,
                            gross_pnl=net + 40.0,
                            costs=40.0,
                            net_pnl=net,
                            mfe=abs(net) * 1.5,
                            mae=-abs(net) / 3.0,
                            holding_minutes=20.0,
                            time_of_day_bucket=bucket,
                            expiry_cycle_phase=phase,
                            days_to_expiry=float((DAYS[-1] - day).days),
                            distance_from_atm=0.0,
                            day_of_week=day.weekday(),
                            bid_ask_spread=0.5,
                        )
                    )
                    index += 1
    return rows


def index_bars(*, bars_per_day: int = 240) -> list:
    """One-minute NIFTY index bars for every fixture session, in the stored row shape."""
    rows = []
    for day in DAYS:
        for index in range(bars_per_day):
            base = 25_000.0 + index
            rows.append(
                make_index_row(
                    timestamp=entry(day, 9, 15, number=index),
                    open=base,
                    high=base + 3.0,
                    low=base - 3.0,
                    close=base + 1.0,
                )
            )
    return rows


def seed(tmp_path) -> Path:
    """A seeded database (features, candles, labels, index bars) plus the fixture config."""
    seed_source(db_file(tmp_path))
    seed_index_history(db_file(tmp_path), index_bars())
    return shrink_config(tmp_path, model={"evaluation": {"preliminary_min_samples": 2}})


def write_ledger(tmp_path, rows=None) -> Path:
    """Write a fixture ledger where ``--mode backtest`` would have put it."""
    target = backtests_dir(tmp_path) / "1m_h5_2026-09-28_2026-10-01_ledger.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    ledger_frame(rows if rows is not None else ledger_rows()).to_parquet(target, index=False)
    return target


def write_importance(tmp_path) -> Path:
    """Write a small spec-27 importance artefact where ``--mode importance`` would have put it."""
    target = importance_dir(tmp_path) / IMPORTANCE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": [
            {
                "column": "nifty_composite_trend_score",
                "topic": "nifty_regime",
                "measured": True,
                "native": 120.0,
                "permutation_mean": 0.42,
                "ci_low": 0.30,
                "ci_high": 0.54,
            },
            {
                "column": "option_trend_score",
                "topic": "option_ma",
                "measured": True,
                "native": 80.0,
                "permutation_mean": -0.18,
                "ci_low": -0.25,
                "ci_high": -0.11,
            },
            {
                "column": "iv",
                "topic": "iv",
                "measured": False,
                "permutation_mean": None,
                "reason": "constant in the sample",
            },
        ]
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def run_charts(tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode charts`` with the fixture's database and configuration."""
    return ai_main(
        [
            *cli_args(tmp_path, mode="charts", run_artifacts=run_artifacts),
            "--config-dir",
            str(config_dir),
            *extra,
        ]
    )


def figures(tmp_path) -> dict[str, Path]:
    """Every written figure, keyed by the chart name its filename carries."""
    root = charts_dir(tmp_path)
    found: dict[str, Path] = {}
    if not root.exists():
        return found
    for path in root.glob("*.png"):
        for chart in ALL_CHARTS:
            if path.stem.endswith(f"_{chart}"):
                found[chart] = path
    return found


def index_file(tmp_path) -> Path | None:
    root = charts_dir(tmp_path)
    if not root.exists():
        return None
    matches = sorted(root.glob("*_charts.md"))
    return matches[0] if matches else None


def counts(tmp_path) -> dict[str, int]:
    """Row counts of the tables the mode must never write to."""
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return {
            table: int(connection.execute(f"select count(*) from {table}").fetchone()[0])
            for table in ("features", "labels", "trades")
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_the_mode_writes_a_figure_per_chart_and_an_index(tmp_path, capsys) -> None:
    """With a ledger, an importance artefact and stored bars, every spec-28 chart is drawn."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    written = figures(tmp_path)
    assert set(written) == set(ALL_CHARTS)
    assert all(path.stat().st_size > 0 for path in written.values())
    assert index_file(tmp_path) is not None
    assert "7 figure(s)" in capsys.readouterr().out


def test_the_index_names_every_written_figure(tmp_path) -> None:
    """The index Markdown lists one row per figure, so a reader knows what is there."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    run_charts(tmp_path, config_dir)
    text = index_file(tmp_path).read_text(encoding="utf-8")
    for chart in ALL_CHARTS:
        assert chart in text
        assert f"{chart}.png" in text


def test_only_narrows_the_run_to_one_family(tmp_path) -> None:
    """``--only heatmap`` draws the two heatmaps and nothing else."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    run_charts(tmp_path, config_dir, "--only", "heatmap")
    assert set(figures(tmp_path)) == {
        "heatmap_time_of_day_cycle_phase",
        "heatmap_time_of_day_day_of_week",
    }


def test_only_accepts_an_exact_chart_name(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    run_charts(tmp_path, config_dir, "--only", "curves_ce_vs_pe")
    assert set(figures(tmp_path)) == {"curves_ce_vs_pe"}


def test_metric_native_draws_the_importance_bars(tmp_path) -> None:
    """``--metric native`` charts the native column of the artefact."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    run_charts(tmp_path, config_dir, "--metric", "native", "--only", "importance")
    assert set(figures(tmp_path)) == {"importance_bars"}


# ---------------------------------------------------------------------------
# The honest skips and the controls
# ---------------------------------------------------------------------------
def test_without_a_ledger_the_p_and_l_charts_are_skipped_with_a_reason(tmp_path, capsys) -> None:
    """A missing ledger skips the four ledger charts; the run still exits 0."""
    config_dir = seed(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    written = set(figures(tmp_path))
    assert written == {"importance_bars", "price_nifty", "price_option"}
    assert "trade ledger" in capsys.readouterr().out


def test_without_an_importance_artefact_the_bars_are_skipped(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    exit_code = run_charts(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    assert "importance_bars" not in figures(tmp_path)
    assert "--mode importance first" in capsys.readouterr().out


def test_a_ledger_path_that_does_not_exist_skips_the_p_and_l_charts(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir, "--ledger", str(tmp_path / "nope.parquet"))
    assert exit_code == EXIT_OK
    assert "importance_bars" in figures(tmp_path)
    assert "trade ledger" in capsys.readouterr().out


def test_dry_run_writes_nothing_and_lists_the_charts(tmp_path, capsys) -> None:
    """``--dry-run`` plans the figures without importing matplotlib or writing a file."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir, "--dry-run")
    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert "chart plan (spec 28)" in out
    assert "heatmap_time_of_day_cycle_phase" in out
    assert not charts_dir(tmp_path).exists()


def test_no_write_draws_and_reports_without_leaving_a_figure(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir, "--no-write")
    assert exit_code == EXIT_OK
    assert "nothing (--no-write)" in capsys.readouterr().out
    assert not charts_dir(tmp_path).exists()


def test_json_prints_the_payload(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir, "--json", "--only", "heatmap")
    assert exit_code == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["stem"] == "1m_h5_2026-09-28_2026-10-01"
    assert "heatmap_time_of_day_cycle_phase" in document["charts"]
    assert document["policy"]["spec"] == "28"


def test_an_unknown_only_selector_builds_nothing_and_says_so(tmp_path, capsys) -> None:
    """A selector that matches no chart is NO_DATA (exit 0), never a crash."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    exit_code = run_charts(tmp_path, config_dir, "--only", "not-a-chart")
    assert exit_code == EXIT_OK
    assert "no chart could be built" in capsys.readouterr().err


def test_the_mode_never_writes_to_the_database(tmp_path) -> None:
    """Drawing a chart is read-only against every stored table."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    before = counts(tmp_path)
    run_charts(tmp_path, config_dir)
    assert counts(tmp_path) == before


def test_the_figures_are_registered_on_the_run_manifest(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    write_importance(tmp_path)
    run_charts(tmp_path, config_dir, "--only", "regime", run_artifacts=True)
    manifests = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert manifests
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert any("regime_performance.png" in Path(str(item)).name for item in manifest["artifacts"])
