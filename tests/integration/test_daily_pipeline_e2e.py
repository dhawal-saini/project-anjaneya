"""End-to-end test of ``scripts/run_daily_pipeline.py`` - the real child processes.

``tests/unit/test_daily_pipeline.py`` replaces the children with a recorder and checks the
argv. This file does the opposite: it runs ``main()`` for real, so every step is a separate
interpreter that opens the same DuckDB file, and every assertion is made against what the run
left on disk:

    the engine half   ticks -> candles -> greeks -> features -> labels -> validate
    the AI half       dataset -> train -> backtest
    the guard         a day with no ticks is refused before anything runs
    the report        one table, one log file, one exit code

Two runs, because they prove two different things:

* the engine-half test uses the *shipped* configuration, so it proves what an operator sees
  on day one: six steps succeed and ``--mode train`` exits 4 because spec 4.3's exploratory
  gate is not open yet;
* the whole-chain test uses a walk-forward configuration whose windows fit the fixture, so a
  model artefact and a backtest ledger are really produced - the only way to prove the last
  two steps are wired correctly.

Nothing here touches the network: the only inputs are stored ticks, stored features and the
configuration on disk.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from core.constants import INDEX_HISTORY_COLUMNS
from core.storage import connect, insert_rows
from scripts import run_daily_pipeline as pipeline
from scripts.run_daily_pipeline import main as pipeline_main
from tests.support.ai import FAST_TRAINING, ONE_DAY_FOLDS, seed_source, shrink_config
from tests.support.engine import (
    DAY,
    SESSION_START,
    make_index_row,
    seed_index_history,
    seed_ticks,
    session_ticks,
)

#: Minutes of synthetic session - long enough for every timeframe and every label horizon
#: to have bars, short enough to keep the eight child processes quick.
MINUTES = tuple(range(0, 40))

#: The day the AI-side fixture stores its four days around.
AI_DAY = date(2026, 9, 30)


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def cli(tmp_path, *extra: str) -> list[str]:
    """Pipeline arguments rooted inside ``tmp_path``."""
    return ["--data-root", str(tmp_path / "data"), "--db-path", str(db_file(tmp_path)), *extra]


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def seed_ticks_and_index(tmp_path, *, day: date = DAY) -> None:
    """One captured session: ticks for the day, plus the NIFTY index series the build needs."""
    delta = timedelta(days=(day - DAY).days)
    ticks = [
        row.model_copy(
            update={
                "timestamp": row.timestamp + delta,
                "fetch_timestamp": row.fetch_timestamp + delta,
            }
        )
        for row in session_ticks(minutes=MINUTES)
    ]
    seed_ticks(db_file(tmp_path), ticks)
    seed_index_history(db_file(tmp_path), [])
    connection = connect(db_file(tmp_path))
    try:
        rows = [
            make_index_row(
                timestamp=SESSION_START + delta + timedelta(minutes=minute),
                timeframe="1m",
                open=25000.0 + minute,
                high=25002.0 + minute,
                low=24998.0 + minute,
                close=25001.0 + minute,
            )
            for minute in MINUTES
        ]
        insert_rows(connection, "index_history", rows, mode="append", columns=INDEX_HISTORY_COLUMNS)
    finally:
        connection.close()


def statuses(report: dict) -> dict[str, str]:
    """``{step: status}`` from a run report."""
    return {row["step"]: row["status"] for row in report["results"]}


# ---------------------------------------------------------------------------
# The engine half, on the shipped configuration
# ---------------------------------------------------------------------------
def test_the_engine_half_builds_everything_and_the_gate_stops_the_model(tmp_path, capsys) -> None:
    seed_ticks_and_index(tmp_path)

    exit_code = pipeline_main(cli(tmp_path, "--date", DAY.isoformat(), "--timeframe", "1m", "--json"))

    report = json.loads(capsys.readouterr().out)
    assert exit_code == pipeline.EXIT_GATE_REFUSED
    assert report["exit_code"] == pipeline.EXIT_GATE_REFUSED
    assert statuses(report) == {
        "candles": pipeline.STATUS_OK,
        "greeks": pipeline.STATUS_OK,
        "features": pipeline.STATUS_OK,
        "labels": pipeline.STATUS_OK,
        "validate": pipeline.STATUS_OK,
        "dataset": pipeline.STATUS_OK,
        "train": pipeline.STATUS_GATE,
        "backtest": pipeline.STATUS_SKIPPED,
    }
    assert "exploratory gate" in report["results"][6]["note"]
    # 2 ticks x 2 contracts x 40 minutes.
    assert report["preflight"]["ticks"] == len(MINUTES) * 4
    assert report["preflight"]["is_trading_day"] is True
    assert report["preflight"]["index_rows"] == len(MINUTES)

    # What the run left on disk: bars, the coverage verdict, the mirrors and the features.
    assert query(tmp_path, "SELECT count(*) FROM candles")[0][0] > 0
    assert query(tmp_path, "SELECT count(*) FROM data_coverage")[0][0] > 0
    assert query(tmp_path, "SELECT count(*) FROM features")[0][0] > 0
    assert query(tmp_path, "SELECT count(*) FROM labels")[0][0] > 0
    # enrich-greeks writes the model columns on the stored bars, so a non-NULL IV proves
    # the second step ran against the index history seeded above.
    assert query(tmp_path, "SELECT count(*) FROM candles WHERE iv IS NOT NULL")[0][0] > 0
    days = query(
        tmp_path,
        "SELECT DISTINCT CAST(timezone('Asia/Kolkata', timestamp) AS DATE) FROM candles",
    )
    assert days == [(DAY,)]
    assert list((tmp_path / "data" / "candles" / "1m").glob("*.parquet"))
    assert list((tmp_path / "data" / "reports" / "feature_validation").glob("*.json"))
    assert not list((tmp_path / "data" / "models").glob("*.txt"))


def test_the_run_leaves_one_dated_log_that_explains_the_gate(tmp_path, capsys) -> None:
    seed_ticks_and_index(tmp_path)

    pipeline_main(cli(tmp_path, "--date", DAY.isoformat(), "--timeframe", "1m"))

    capsys.readouterr()
    log_path = tmp_path / "data" / "logs" / f"daily_pipeline_{DAY.isoformat()}.log"
    assert log_path.is_file()
    body = log_path.read_text(encoding="utf-8")
    assert body.startswith(f"ANJANEYA daily pipeline: day={DAY.isoformat()} steps=8")
    assert "[7/8] train:" in body
    assert "[8/8] backtest: SKIPPED (no fresh model (train refused by the gate))" in body
    assert "BELOW_GATE" in body


def test_a_day_without_ticks_is_refused_before_any_step_runs(tmp_path, capsys) -> None:
    seed_ticks_and_index(tmp_path)

    exit_code = pipeline_main(cli(tmp_path, "--date", "2026-08-03"))

    captured = capsys.readouterr()
    assert exit_code == pipeline.EXIT_CONFIG_ERROR
    assert "no ticks are stored for 2026-08-03" in captured.err
    assert captured.out == ""



# ---------------------------------------------------------------------------
# The whole chain, with a walk-forward window that fits the fixture
# ---------------------------------------------------------------------------
def test_the_whole_chain_trains_and_backtests_on_a_shrunk_config(tmp_path, capsys) -> None:
    """Features and labels are seeded, the walk-forward fits, so a model really is trained.

    Every engine step is skipped because the storage layer already holds what they would have
    produced (that is what ``tests/support/ai.py`` seeds) - including ``validate``, whose whole
    job is to rebuild the feature matrix and compare, and which therefore has nothing to say
    about hand-written fixture rows. That keeps this test about the *runner* wiring the AI half
    together, which is the part no other test covers.
    """
    seed_source(db_file(tmp_path))
    config_dir = shrink_config(tmp_path, walk_forward=ONE_DAY_FOLDS, training=FAST_TRAINING)

    exit_code = pipeline_main(
        cli(
            tmp_path,
            "--config-dir",
            str(config_dir),
            "--date",
            AI_DAY.isoformat(),
            "--timeframe",
            "1m",
            "--horizon",
            "5",
            "--force",
            "--skip",
            "candles,greeks,features,labels,validate",
            "--json",
        )
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == pipeline.EXIT_OK
    assert statuses(report) == {
        "dataset": pipeline.STATUS_OK,
        "train": pipeline.STATUS_OK,
        "backtest": pipeline.STATUS_OK,
    }
    assert report["planned_steps"] == ["dataset", "train", "backtest"]
    assert report["force"] is True

    # The trainer's own artefacts, the dataset's splits and the backtest's ledger.
    models = sorted((tmp_path / "data" / "models").glob("*_h5_*.txt"))
    metadata = sorted((tmp_path / "data" / "models").glob("*_h5_*_metadata.json"))
    splits = sorted((tmp_path / "data" / "datasets").glob("*_h5_*_splits.json"))
    ledgers = sorted((tmp_path / "data" / "backtests").glob("*_h5_*_ledger.parquet"))
    summaries = sorted((tmp_path / "data" / "backtests").glob("*_h5_*_summary.json"))
    assert len(models) == 1 and len(metadata) == 1
    assert splits and ledgers and summaries

    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["mode"] == "backtest"
    assert payload["horizon_minutes"] == 5
    assert payload["model"]["horizon_minutes"] == 5
    assert payload["costs"]["caveat"]
    assert payload["counts"]["bars"] > 0

    log_path = tmp_path / "data" / "logs" / f"daily_pipeline_{AI_DAY.isoformat()}.log"
    body = log_path.read_text(encoding="utf-8")
    assert "[3/3] backtest: the spec-23 replay ledger and summary" in body
    assert '"verdict": "OK"' in body  # --json logs the JSON document instead of the table


