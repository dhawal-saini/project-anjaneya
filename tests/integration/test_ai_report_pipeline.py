"""End-to-end ANJANEYA-AI research report (Stage 4.4, spec 24).

``--mode report`` is exercised through the real ``main()`` against a stored ledger and a
seeded database, so every assertion is about what actually happened:

    the ledger    ->  a real Parquet artefact (the shape ``--mode backtest`` writes), read back
                      by the mode itself rather than handed in as an object
    the tables    ->  spec 24's ten items over those trades: buckets x side, x cycle phase, the
                      weekday and days-to-expiry cuts, the interactions, holding time, the ATR
                      profile and the intraday path (from the *stored index bars*), the
                      corrected significance, and the PRELIMINARY labels
    the artefacts ->  one JSON, one Markdown and one CSV under ``data/reports/research/``,
                      registered on the run manifest
    the honesty   ->  a missing ledger, a missing database and ``--dry-run``/``--no-write``
                      each behave exactly as documented, and nothing is written to DuckDB

Nothing here touches the network: the only inputs are the stored database and the ledger
artefact on disk.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from apps.anjaneeya_ai.backtest_writer import ledger_frame
from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_FAILED, EXIT_OK
from core.backtest import Trade
from core.storage import connect
from tests.support.ai import cli_args, seed_source, shrink_config
from tests.support.engine import INDEX_SYMBOL, make_index_row, seed_index_history

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

#: Entries per (day, slot, side) - enough to clear the sample rule the fixture configures.
PER_CELL = 2

SYMBOL = "NIFTY01OCT2625000CE"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def backtests_dir(tmp_path) -> Path:
    return tmp_path / "data" / "backtests"


def reports_dir(tmp_path) -> Path:
    return tmp_path / "data" / "reports" / "research"


def entry(day: date, hour: int, minute: int, *, number: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST) + timedelta(
        minutes=number
    )


def ledger_rows() -> list[Trade]:
    """A ledger with three buckets, both option sides and a clear winner/loser split.

    Every (bucket, side) cell holds ``PER_CELL`` entries on each of the four sessions, so the
    significance tests have enough trades on both sides of the comparison to run while the
    weekday cells stay deliberately small.
    """
    rows: list[Trade] = []
    index = 0
    for day in DAYS:
        phase = "EXPIRY_DAY" if day == DAYS[-1] else "MID_CYCLE"
        for hour, minute, bucket in SLOTS:
            for option_type, sign in (("CE", 1.0), ("PE", -1.0)):
                for number in range(PER_CELL):
                    # the P&L varies within a cell on purpose: a cell whose trades all made
                    # exactly the same money has no spread to test, and the report says so
                    net = sign * (100.0 + 10.0 * number)
                    moment = entry(day, hour, minute, number=number)
                    rows.append(
                        Trade(
                            trade_id=f"run|{SYMBOL}|{index}",
                            run_id="run",
                            entry_timestamp=moment,
                            tradingsymbol=f"NIFTY01OCT26{option_type}",
                            option_type=option_type,
                            entry_price=120.0,
                            quantity=75,
                            exit_timestamp=moment + timedelta(minutes=20),
                            expiry=DAYS[-1],
                            strike=25000.0,
                            exit_price=121.0,
                            entry_reason="SIGNAL:UP_CONTINUATION",
                            exit_reason="TARGET" if net > 0 else "STOP_LOSS",
                            nifty_state="BULLISH" if option_type == "CE" else "BEARISH",
                            option_state="UP",
                            candle_state="GREEN",
                            support_resistance_state="NEAR_SUPPORT",
                            ai_prediction="UP_CONTINUATION",
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



def index_bars(*, bars_per_day: int = 60) -> list:
    """One-minute NIFTY index bars for every fixture session, in the stored row shape."""
    rows = []
    for day in DAYS:
        for index in range(bars_per_day):
            moment = entry(day, 9, 15, number=index)
            base = 25_000.0 + index
            rows.append(
                make_index_row(
                    timestamp=moment,
                    open=base,
                    high=base + 3.0,
                    low=base - 3.0,
                    close=base + 1.0,
                )
            )
    return rows


def seed(tmp_path) -> Path:
    """A seeded database (features, labels, bars and index history) plus its configuration.

    ``evaluation.preliminary_min_samples`` is set to 5 so the fixture's cells (8 trades each)
    are testable at all; the shipped 30 is exercised by the unit tests instead, where it can
    be asserted directly.
    """
    seed_source(db_file(tmp_path))
    seed_index_history(db_file(tmp_path), index_bars())
    return shrink_config(tmp_path, model={"evaluation": {"preliminary_min_samples": 5}})


def write_ledger(tmp_path, rows=None, *, name: str = "1m_h5_2026-09-28_2026-10-01_ledger.parquet") -> Path:
    """Write a fixture ledger where ``--mode backtest`` would have put it."""
    target = backtests_dir(tmp_path) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    ledger_frame(rows if rows is not None else ledger_rows()).to_parquet(target, index=False)
    return target


def run_report(tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode report`` with the fixture's database and configuration."""
    arguments = [
        *cli_args(tmp_path, mode="report", run_artifacts=run_artifacts),
        "--config-dir",
        str(config_dir),
    ]
    return ai_main([*arguments, *extra])


def artifacts(tmp_path) -> dict[str, Path]:
    """The three report artefacts, keyed by extension."""
    root = reports_dir(tmp_path)
    found = {}
    for suffix in (".json", ".md", ".csv"):
        for path in sorted(root.glob(f"*{suffix}")):
            found[suffix] = path
    return found


def payload(tmp_path) -> dict:
    """The JSON report, read back from disk."""
    return json.loads(artifacts(tmp_path)[".json"].read_text(encoding="utf-8"))


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Nothing to analyse
# ---------------------------------------------------------------------------
def test_without_a_ledger_the_run_says_so_and_writes_nothing(tmp_path, capsys) -> None:
    """A report needs a ledger; the exit code is 0 because nothing went wrong."""
    config_dir = seed(tmp_path)

    exit_code = run_report(tmp_path, config_dir)

    assert exit_code == EXIT_OK
    captured = capsys.readouterr()
    assert "no trade ledger to analyse" in captured.err
    assert "--mode backtest" in captured.err
    assert not reports_dir(tmp_path).exists()


def test_a_ledger_path_that_does_not_exist_is_the_same_message(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)

    exit_code = run_report(tmp_path, config_dir, "--ledger", str(tmp_path / "nope.parquet"))

    assert exit_code == EXIT_OK
    assert "no trade ledger to analyse" in capsys.readouterr().err



# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------
def test_the_report_writes_three_artefacts_spec_24_can_be_read_from(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)

    exit_code = run_report(tmp_path, config_dir)

    assert exit_code == EXIT_OK
    found = artifacts(tmp_path)
    assert set(found) == {".json", ".md", ".csv"}
    # the stem mirrors the ledger it came from
    assert found[".json"].name == "1m_h5_2026-09-28_2026-10-01_report.json"
    document = payload(tmp_path)
    assert document["ledger"] == "1m_h5_2026-09-28_2026-10-01_ledger.parquet"
    assert document["trades"] == 4 * 3 * 2 * PER_CELL
    assert document["days"] == 4
    printed = capsys.readouterr().out
    assert "ANJANEYA-AI research report (spec 24)" in printed
    assert "time_of_day_x_option_type" in printed


def test_the_ten_items_are_in_the_json_with_their_samples(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    run_report(tmp_path, config_dir)

    document = payload(tmp_path)

    assert list(document["items"]) == [
        "1_time_of_day_x_option_type",
        "2_time_of_day_x_cycle_phase",
        "3_day_of_week",
        "4_days_to_expiry",
        "5_interactions",
        "6_holding_time_by_bucket",
        "7_volatility_profile",
        "8_intraday_path",
        "9_significance",
        "10_preliminary_labels",
    ]
    # item 1: bucket x CE/PE, every cell carrying its sample
    matrices = {matrix["name"]: matrix for matrix in document["matrices"]}
    side = matrices["time_of_day_x_option_type"]
    assert side["preliminary_min_samples"] == 5
    cell = next(
        row for row in side["cells"] if row["row"] == "OPEN_915_930" and row["column"] == "CE"
    )
    assert cell["sample"] == 8
    assert cell["preliminary"] is False
    assert cell["total_net_pnl"] == pytest.approx(840.0)
    assert side["trades"] == document["trades"]
    # item 2 and item 5 exist as matrices too
    assert set(matrices) == {
        "time_of_day_x_option_type",
        "time_of_day_x_cycle_phase",
        "regime_x_time_of_day",
        "regime_x_cycle_phase",
    }
    # item 3 and item 4 are the single-axis cuts
    assert {table["name"] for table in document["segment_tables"]} == {
        "day_of_week",
        "days_to_expiry",
    }
    # item 6: holding time per bucket
    assert document["holding_time"]["label_key"] == "time_of_day_bucket"
    assert {row["label"] for row in document["holding_time"]["rows"]} == {
        "OPEN_915_930",
        "MORNING_930_1100",
        "CLOSE_1430_1515",
    }


def test_the_market_items_are_built_from_the_stored_index_bars(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    run_report(tmp_path, config_dir)

    document = payload(tmp_path)

    assert document["volatility_unavailable"] is None
    volatility = document["volatility"]
    assert volatility["symbol"] == INDEX_SYMBOL
    assert volatility["sessions"] == 4
    assert volatility["bars"] == 4 * 60
    assert volatility["profiles"]
    assert all(profile["atr_mean"] is not None for profile in volatility["profiles"])
    assert document["intraday_paths_unavailable"] is None
    # the fixture holds 60 bars (09:15..10:14), so only the first two buckets are reached:
    # the path starts at the fill (the 09:15 bar's open) and runs to the last stored bar
    paths = {path["bucket"]: path for path in document["intraday_paths"]}
    assert set(paths) == {"OPEN_915_930", "MORNING_930_1100"}
    assert paths["OPEN_915_930"]["base_minute"] == "09:16"
    assert paths["MORNING_930_1100"]["base_minute"] == "09:30"
    assert paths["OPEN_915_930"]["sessions"] == 4
    assert len(paths["OPEN_915_930"]["points"]) == 60



def test_significance_is_tested_per_matrix_and_corrected(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    run_report(tmp_path, config_dir)

    document = payload(tmp_path)

    families = {item["matrix"]: item for item in document["significance"]}
    assert set(families) == {
        "time_of_day_x_option_type",
        "time_of_day_x_cycle_phase",
        "regime_x_time_of_day",
        "regime_x_cycle_phase",
    }
    # the CE/PE split is stark in the fixture, and the correction still names its family
    side = families["time_of_day_x_option_type"]
    assert side["correction"] == "benjamini_hochberg"
    assert side["significance_level"] == 0.05
    assert side["family_size"] == 6
    # every CE cell beat the rest and every PE cell fell behind it, so all six clear the level
    assert set(side["significant_cells"]) == {
        "OPEN_915_930 | CE",
        "MORNING_930_1100 | CE",
        "CLOSE_1430_1515 | CE",
        "OPEN_915_930 | PE",
        "MORNING_930_1100 | PE",
        "CLOSE_1430_1515 | PE",
    }
    assert len(side["significant_cells"]) == side["family_size"] == 6
    tested = next(row for row in side["tests"] if row["row"] == "OPEN_915_930")
    assert tested["sample"] + tested["baseline_sample"] == document["trades"]
    assert tested["win_rate_p_value"] is not None
    assert tested["win_rate_p_adjusted"] is not None
    assert tested["verdict"] == "SIGNIFICANT"


def test_the_expiry_day_cells_are_labelled_preliminary_not_dropped(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    run_report(tmp_path, config_dir)

    document = payload(tmp_path)

    # four sessions -> one EXPIRY_DAY session -> 4 trades per bucket cell, below the rule
    assert document["preliminary_cells"]["time_of_day_x_cycle_phase"] == [
        "OPEN_915_930 | EXPIRY_DAY",
        "MORNING_930_1100 | EXPIRY_DAY",
        "CLOSE_1430_1515 | EXPIRY_DAY",
    ]
    # the cell is still *reported* - with its sample and its label
    matrix = next(
        item for item in document["matrices"] if item["name"] == "time_of_day_x_cycle_phase"
    )
    cell = next(
        row
        for row in matrix["cells"]
        if row["row"] == "OPEN_915_930" and row["column"] == "EXPIRY_DAY"
    )
    assert cell["sample"] == 4
    assert cell["preliminary"] is True


def test_the_markdown_and_csv_are_the_same_report_in_other_shapes(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    run_report(tmp_path, config_dir)

    found = artifacts(tmp_path)
    markdown = found[".md"].read_text(encoding="utf-8")
    csv_text = found[".csv"].read_text(encoding="utf-8")

    for name in (
        "1_time_of_day_x_option_type",
        "2_time_of_day_x_cycle_phase",
        "5_interactions",
        "6_holding_time_by_bucket",
        "7_volatility_profile",
        "8_intraday_path",
        "9_significance",
        "10_preliminary_labels",
    ):
        assert f"## {name}." in markdown
    assert "| time_of_day_bucket | CE | PE |" in markdown
    assert "PRELIMINARY" in markdown
    assert "## Caveats" in markdown
    header = csv_text.splitlines()[0].split(",")
    assert header[:3] == ["matrix", "row", "column"]
    assert "win_rate_p_adjusted" in header
    assert "verdict" in header
    # one row per matrix cell, plus the header
    assert len(csv_text.splitlines()) - 1 == len(document_cells(payload(tmp_path)))


def document_cells(document) -> list:
    """Every cell of every matrix in a JSON report (the CSV's own row set)."""
    return [cell for matrix in document["matrices"] for cell in matrix["cells"]]



# ---------------------------------------------------------------------------
# Dry runs, no-write and the database
# ---------------------------------------------------------------------------
def test_dry_run_reports_without_writing_anything(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)

    exit_code = run_report(tmp_path, config_dir, "--dry-run")

    assert exit_code == EXIT_OK
    assert not reports_dir(tmp_path).exists()
    printed = capsys.readouterr().out
    assert "wrote         : nothing" in printed
    assert "ANJANEYA-AI research report (spec 24)" in printed


def test_no_write_still_prints_the_report(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)

    exit_code = run_report(tmp_path, config_dir, "--no-write")

    assert exit_code == EXIT_OK
    assert not reports_dir(tmp_path).exists()
    assert "wrote         : nothing" in capsys.readouterr().out


def test_json_prints_the_whole_document_to_stdout(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)

    exit_code = run_report(tmp_path, config_dir, "--json", "--no-write")

    assert exit_code == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["ledger"] == "1m_h5_2026-09-28_2026-10-01_ledger.parquet"
    assert len(document["items"]) == 10
    assert document["written"] == []


def test_the_report_never_writes_to_the_database(tmp_path) -> None:
    """A report is read-only evidence: the stored tables must be untouched."""
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)
    before = {
        table: query(tmp_path, f"SELECT count(*) FROM {table}")[0][0]
        for table in ("candles", "features", "labels", "index_history", "trades")
    }

    exit_code = run_report(tmp_path, config_dir)

    assert exit_code == EXIT_OK
    after = {
        table: query(tmp_path, f"SELECT count(*) FROM {table}")[0][0]
        for table in ("candles", "features", "labels", "index_history", "trades")
    }
    assert after == before


def test_run_artifacts_are_registered_on_the_manifest(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path)

    exit_code = run_report(tmp_path, config_dir, run_artifacts=True)

    assert exit_code == EXIT_OK
    runs = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert runs
    manifest = json.loads(runs[-1].read_text(encoding="utf-8"))
    written = [str(path) for path in artifacts(tmp_path).values()]
    registered = [str(item) for item in manifest.get("artifacts", [])]
    assert sorted(written) == sorted(registered)
    assert any("research report" in note for note in manifest.get("notes", []))


def test_a_ledger_that_is_not_a_trade_ledger_is_a_failure(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_ledger(tmp_path, name="not_a_ledger_ledger.parquet")  # a real one first
    broken = backtests_dir(tmp_path) / "not_a_ledger_ledger.parquet"
    frame = pd.read_parquet(broken)
    frame["something_else"] = 1.0
    frame.to_parquet(broken, index=False)

    exit_code = run_report(tmp_path, config_dir, "--ledger", str(broken))

    assert exit_code == EXIT_FAILED
    captured = capsys.readouterr().err
    assert "is not a trade ledger" in captured
    assert "does not know" in captured
    assert not reports_dir(tmp_path).exists()

