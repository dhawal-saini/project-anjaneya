"""End-to-end ANJANEYA-AI backtest pipeline (Stage 4.3, spec 23).

``--mode backtest`` is exercised through the real ``main()`` against a seeded database and a
model artefact, so every assertion is about what actually happened:

    the model     ->  a real artefact pair (booster text + metadata), fitted on a frame whose
                      class follows ``option_type``, so a CE contract is predicted bullish
                      and a PE contract bearish
    the replay    ->  stored bars in, a ledger out: entry at the next open, exits at prices
                      that exist, every loser kept
    the costs     ->  charged on both legs, with the contract note's own breakdown
    the artefacts ->  a Parquet ledger, a summary JSON, and the same rows validated through
                      ``TradeRecord`` into the stored ``trades`` table (no schema change)
    the honesty   ->  the gate labels the result instead of refusing it, the caveats are
                      printed and stored, and ``--dry-run``/``--no-write`` write nothing

Nothing here touches the network: the only inputs are what is stored in DuckDB and the
artefact on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_OK
from core.backtest import EXIT_REASONS, LEDGER_COLUMNS
from core.constants import TRADE_COLUMNS
from core.storage import connect
from tests.support.ai import (
    cli_args,
    seed_source,
    shrink_config,
    write_signal_model,
)

#: The horizon the crafted model is fitted for.
HORIZON = 5


def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def backtests_dir(tmp_path) -> Path:
    return tmp_path / "data" / "backtests"


def models_dir(tmp_path) -> Path:
    return tmp_path / "data" / "models"


def seed(tmp_path, **config_overrides) -> Path:
    """A seeded database (features, labels and bars) plus the configuration directory."""
    seed_source(db_file(tmp_path))
    return shrink_config(tmp_path, **config_overrides)


def with_model(tmp_path, config_dir: Path, **kwargs) -> Path:
    """Add a real model artefact to a seeded ``tmp_path`` and return the config directory."""
    write_signal_model(models_dir(tmp_path), horizon=HORIZON, **kwargs)
    return config_dir


def run_backtest(tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode backtest`` with the fixture's database and configuration."""
    arguments = [
        *cli_args(tmp_path, mode="backtest", run_artifacts=run_artifacts),
        "--config-dir",
        str(config_dir),
    ]
    return ai_main([*arguments, *extra])


def ledger(tmp_path, horizon: int = HORIZON) -> pd.DataFrame:
    """This run's trade ledger, read back from the Parquet artefact."""
    paths = sorted(backtests_dir(tmp_path).glob(f"*_h{horizon}_*_ledger.parquet"))
    assert len(paths) == 1, [path.name for path in paths]
    return pd.read_parquet(paths[0])


def summary(tmp_path, horizon: int = HORIZON) -> dict:
    """This run's summary JSON, read back from disk."""
    paths = sorted(backtests_dir(tmp_path).glob(f"*_h{horizon}_*_summary.json"))
    assert len(paths) == 1, [path.name for path in paths]
    return json.loads(paths[0].read_text(encoding="utf-8"))


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# No model, nothing to do
# ---------------------------------------------------------------------------
def test_without_a_model_the_run_says_so_and_writes_nothing(tmp_path, capsys) -> None:
    """A backtest needs a model; the exit code is 0 because nothing went wrong."""
    config_dir = seed(tmp_path)
    exit_code = run_backtest(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    captured = capsys.readouterr()
    assert "no model artefact to replay" in captured.err
    assert "--mode train" in captured.err
    assert not list(backtests_dir(tmp_path).glob("*.parquet"))


def test_a_model_without_its_metadata_is_a_configuration_error(tmp_path, capsys) -> None:
    """Half an artefact cannot be scored, so the run refuses before touching the bars."""
    config_dir = seed(tmp_path)
    model = write_signal_model(models_dir(tmp_path), horizon=HORIZON)
    model.with_name(model.stem + "_metadata.json").unlink()
    exit_code = run_backtest(tmp_path, config_dir)
    assert exit_code == EXIT_CONFIG_ERROR
    assert "metadata is missing" in capsys.readouterr().err


def test_a_horizon_that_disagrees_with_the_model_is_a_configuration_error(
    tmp_path, capsys
) -> None:
    """``--model`` names an artefact, and a horizon that contradicts it is refused."""
    config_dir = seed(tmp_path)
    model = write_signal_model(models_dir(tmp_path), horizon=HORIZON)
    exit_code = run_backtest(tmp_path, config_dir, "--model", str(model), "--horizon", "10")
    assert exit_code == EXIT_CONFIG_ERROR
    assert "was trained for horizon 5m" in capsys.readouterr().err


def test_a_horizon_without_an_artefact_is_no_model(tmp_path, capsys) -> None:
    """Nothing was fitted for that horizon, which is a "train one first", not a crash."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    exit_code = run_backtest(tmp_path, config_dir, "--horizon", "10")
    assert exit_code == EXIT_OK
    assert "no model artefact to replay" in capsys.readouterr().err


def test_no_candles_in_scope_is_reported_as_no_data(tmp_path, capsys) -> None:
    """The bars are the market: without them there is nothing to replay."""
    seed_source(db_file(tmp_path), candles=False)
    config_dir = shrink_config(tmp_path)
    write_signal_model(models_dir(tmp_path), horizon=HORIZON)
    exit_code = run_backtest(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    assert "no stored 1m candles in scope" in capsys.readouterr().err
    assert not list(backtests_dir(tmp_path).glob("*.parquet"))


# ---------------------------------------------------------------------------
# The replay, the ledger and the cost model
# ---------------------------------------------------------------------------
def test_a_backtest_writes_the_ledger_the_summary_and_the_stored_ledger(tmp_path, capsys) -> None:
    """Three shapes of the same evidence: Parquet, JSON, and the ``trades`` table."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    exit_code = run_backtest(tmp_path, config_dir, run_artifacts=True)
    out = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "ANJANEYA-AI backtest" in out
    assert "every loser is in the ledger" in out
    assert "a threshold rule, not spec 24's decision layer" in out

    frame = ledger(tmp_path)
    assert tuple(frame.columns) == LEDGER_COLUMNS
    assert len(frame) > 0
    assert set(TRADE_COLUMNS) < set(frame.columns)
    payload = summary(tmp_path)
    assert payload["mode"] == "backtest"
    assert payload["horizon_minutes"] == HORIZON
    assert payload["model"]["horizon_minutes"] == HORIZON
    assert payload["stored_ledger_columns"] == list(TRADE_COLUMNS)
    assert payload["metrics"]["total_trades"] == len(frame)
    assert payload["counts"]["entries"] == len(frame)
    assert payload["counts"]["contracts"] == 2
    assert payload["counts"]["bars"] > 0
    assert payload["window"]["trading_days"] == 4
    assert payload["costs"]["caveat"]
    assert payload["segments"]
    assert payload["caveats"]

    stored = query(tmp_path, "SELECT count(*), sum(net_pnl) FROM trades")[0]
    assert stored[0] == len(frame)
    assert float(stored[1]) == pytest.approx(float(frame["net_pnl"].sum()))


def test_the_ledger_keeps_every_loser_and_charges_both_legs(tmp_path) -> None:
    """The fixture's CE path rises and its PE path falls, so both outcomes are in the file."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    assert run_backtest(tmp_path, config_dir) == EXIT_OK
    frame = ledger(tmp_path)
    assert (frame["net_pnl"] > 0).any()
    assert (frame["net_pnl"] < 0).any()
    assert (frame["costs"] > 0).all()
    assert (frame["mae"] <= 0).all()
    assert (frame["mfe"] >= 0).all()
    assert (frame["holding_minutes"] >= 0).all()
    # net = gross - costs, on every single row.
    assert (
        (frame["gross_pnl"] - frame["costs"] - frame["net_pnl"]).abs() < 1e-9
    ).all()
    # A loser is a losing *side*: the falling PE path is the one that loses.
    losers = frame[frame["net_pnl"] < 0]
    assert set(losers["option_type"]) == {"PE"}


def test_every_exit_reason_is_one_the_engine_defines(tmp_path) -> None:
    """No exit may be invented at the reporting layer."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    assert run_backtest(tmp_path, config_dir) == EXIT_OK
    frame = ledger(tmp_path)
    assert set(frame["exit_reason"]) <= set(EXIT_REASONS)
    counts = frame["exit_reason"].value_counts().to_dict()
    assert summary(tmp_path)["counts"]["exit_reasons"] == counts

    ctx = summary(tmp_path)["counts"]
    assert ctx["decision_bars"] + ctx["in_position_bars"] >= ctx["bars"]
    assert ctx["decisions"] > 0
    assert ctx["entries"] == len(frame)
    assert ctx["spread_fills"] == 0



# ---------------------------------------------------------------------------
# What the run may claim, and what it may not
# ---------------------------------------------------------------------------
def test_a_run_below_the_meaningful_backtest_gate_is_labelled_not_refused(tmp_path, capsys) -> None:
    """Spec 4.3's 120-day gate labels an exploratory backtest; it does not stop it."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    assert run_backtest(tmp_path, config_dir) == EXIT_OK
    out = capsys.readouterr().out
    assert "BELOW_GATE" in out
    assert "too little data to conclude anything" in out
    payload = summary(tmp_path)
    assert payload["verdict"]["results_label"] == "BELOW_GATE"
    assert payload["verdict"]["refused"] is False
    assert payload["verdict"]["production_model"] is False
    assert payload["gate"]["days"] == 4
    assert payload["gate"]["stage"] == "INSUFFICIENT"
    assert payload["gate"]["gates"]["meaningful_backtest_min_days"] == 120
    assert payload["gate"]["next_stage"] == "EXPLORATORY"
    assert any("stand-in for spec 24" in caveat for caveat in payload["caveats"])
    assert any("labels table is never touched" in caveat for caveat in payload["caveats"])
    assert any("verify" in caveat for caveat in payload["caveats"])


def test_the_backtest_never_writes_a_feature_label_or_candle_row(tmp_path) -> None:
    """The sources are read-only, and the stored ledger is the only table it touches."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    before = query(
        tmp_path,
        "SELECT (SELECT count(*) FROM features), (SELECT count(*) FROM labels), "
        "(SELECT count(*) FROM candles), (SELECT count(*) FROM trades)",
    )[0]
    assert run_backtest(tmp_path, config_dir) == EXIT_OK
    after = query(
        tmp_path,
        "SELECT (SELECT count(*) FROM features), (SELECT count(*) FROM labels), "
        "(SELECT count(*) FROM candles), (SELECT count(*) FROM trades)",
    )[0]
    assert before[:3] == after[:3]
    assert before[3] == 0 and after[3] == len(ledger(tmp_path))


def test_the_window_restricts_the_replay(tmp_path) -> None:
    """``--from``/``--to`` bound the bars, and the summary records the window it used."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    exit_code = run_backtest(tmp_path, config_dir, "--from", "2026-09-30", "--to", "2026-09-30")
    assert exit_code == EXIT_OK
    payload = summary(tmp_path)
    assert payload["window"]["trading_days"] == 1
    assert payload["window"]["from"].startswith("2026-09-30")
    assert set(ledger(tmp_path)["entry_timestamp"].astype(str).str[:10]) == {"2026-09-30"}


def test_a_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys) -> None:
    """A plan is a plan: no ledger, no summary, no rows."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    exit_code = run_backtest(tmp_path, config_dir, "--dry-run")
    out = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "backtest (dry run)" in out
    assert "writes to      : nothing" not in out  # the plan says where it *would* write
    assert "nothing (--no-write)" in out or "backtests" in out
    assert not list(backtests_dir(tmp_path).glob("*.parquet"))
    assert not list(backtests_dir(tmp_path).glob("*.json"))
    assert query(tmp_path, "SELECT count(*) FROM trades")[0][0] == 0


def test_no_write_replays_without_leaving_evidence_on_disk(tmp_path, capsys) -> None:
    """``--no-write`` still computes and reports; it simply stores nothing."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    exit_code = run_backtest(tmp_path, config_dir, "--no-write")
    out = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "trades         :" in out
    assert not list(backtests_dir(tmp_path).glob("*.parquet"))
    assert not list(backtests_dir(tmp_path).glob("*.json"))
    assert query(tmp_path, "SELECT count(*) FROM trades")[0][0] == 0
    assert "0 file(s), 0 trade row(s)" in out


def test_json_output_is_machine_readable(tmp_path, capsys) -> None:
    """``--json`` is the same run, in a shape a script can consume."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    exit_code = run_backtest(tmp_path, config_dir, "--json")
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_OK
    assert payload["mode"] == "backtest"
    assert payload["signal"]["min_probability"] > 0
    assert payload["execution"]["timeframe"] == "1m"
    assert payload["metrics"]["total_trades"] == len(ledger(tmp_path))
    assert isinstance(payload["segments"], list)
    assert {table["name"] for table in payload["segments"]} >= {
        "option_type",
        "time_of_day_bucket",
        "exit_reason",
    }


def test_min_probability_override_is_recorded_and_changes_the_run(tmp_path) -> None:
    """The knob a researcher turns first is stored with the value it was turned to."""
    config_dir = with_model(tmp_path, seed(tmp_path))
    assert run_backtest(tmp_path, config_dir, "--min-probability", "0.9") == EXIT_OK
    payload = summary(tmp_path)
    assert payload["signal"]["min_probability"] == pytest.approx(0.9)
    assert payload["counts"]["decisions"] == 0
    assert payload["metrics"]["total_trades"] == 0
    assert len(ledger(tmp_path)) == 0



# ---------------------------------------------------------------------------
# The chain the stage exists for: dataset -> train -> backtest
# ---------------------------------------------------------------------------
def test_the_real_train_mode_then_backtest_chain(tmp_path, capsys) -> None:
    """A model ``--mode train`` wrote is replayed by ``--mode backtest`` - no hand-made file.

    The fixture's four days sit below the training gate, so the training run needs
    ``--force``; the backtest then finds the artefact itself (no ``--model``) and reports
    whatever the model signalled. Zero trades is a legitimate outcome here - the assertion
    is that the chain runs, the ledger is written, and the run says how many decisions it
    saw, rather than that the fixture happens to contain a signal.
    """
    from apps.anjaneeya_ai.run import main as ai_main_run
    from tests.support.ai import FAST_TRAINING, ONE_DAY_FOLDS

    seed_source(db_file(tmp_path))
    config_dir = shrink_config(
        tmp_path, walk_forward=ONE_DAY_FOLDS, training=FAST_TRAINING
    )
    trained = ai_main_run(
        [
            *cli_args(tmp_path, mode="train"),
            "--config-dir",
            str(config_dir),
            "--horizon",
            str(HORIZON),
            "--force",
        ]
    )
    assert trained == EXIT_OK
    capsys.readouterr()

    exit_code = run_backtest(
        tmp_path, config_dir, "--horizon", str(HORIZON), "--min-probability", "0.26"
    )
    out = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "ANJANEYA-AI backtest" in out
    payload = summary(tmp_path)
    assert payload["model"]["run_id"]
    assert payload["model"]["results_label"] in {"BELOW_GATE", "EXPLORATORY", "PRELIMINARY"}
    assert payload["counts"]["decision_bars"] > 0
    assert payload["metrics"]["total_trades"] == len(ledger(tmp_path))

