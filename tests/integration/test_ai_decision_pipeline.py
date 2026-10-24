"""End-to-end ANJANEYA-AI decision layer (Phase 7, spec 24 / master-prompt phase 24).

``--mode decide`` is exercised through the real ``main()`` against a seeded database and a real
model artefact pair, so every assertion is about what actually happened:

    the inputs    ->  the stored feature rows, read back by the mode itself, and one model
                      artefact whose class follows ``option_type`` (a CE contract is predicted
                      UP_CONTINUATION, a PE contract DOWN_CONTINUATION)
    the record    ->  one JSON, one Markdown and one CSV under ``data/predictions/``, all
                      registered on the run manifest, one row per scored bar
    the controls  ->  ``--min-probability`` / ``--min-margin`` / ``--max-decisions-per-day``
                      change the record and are recorded as the values actually used;
                      ``--dry-run`` and ``--no-write`` write nothing
    the honesty   ->  the gate *labels* the run instead of refusing it, every caveat is printed
                      and stored, an empty database is reported with a reason and exit 0 (never
                      a crash), and nothing is written to DuckDB

Nothing here touches the network: the only inputs are the stored database and the artefact on
disk.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from apps.anjaneeya_ai.decision_writer import decisions_stem
from apps.anjaneeya_ai.explain_writer import model_stem
from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_NOT_IMPLEMENTED, EXIT_OK
from core.decision import DECISION_COLUMNS, DECISION_REASONS
from core.storage import bootstrap_database, connect
from tests.support.ai import cli_args, seed_source, shrink_config, write_signal_model
from tests.support.engine import CE_SYMBOL

#: The horizon the crafted model is fitted for.
HORIZON = 5

#: Bars per contract-day the fixture writes (``tests.support.ai.BARS``).
BARS = 60

#: The four fixture sessions.
DAYS = 4

#: Every row the fixture stores: two contracts, four days, sixty bars each.
EXPECTED_ROWS = DAYS * 2 * BARS


def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def out_dir(tmp_path) -> Path:
    return tmp_path / "data" / "predictions"


def models_dir(tmp_path) -> Path:
    return tmp_path / "data" / "models"


def seed(tmp_path, *, with_model: bool = True, seed_db: bool = True, **config_overrides) -> Path:
    """A seeded database (features, labels and bars), a model artefact and a config directory."""
    if seed_db:
        seed_source(db_file(tmp_path))
    else:
        bootstrap_database(db_file(tmp_path), schema_version=1)
    if with_model:
        write_signal_model(models_dir(tmp_path), horizon=HORIZON)
    return shrink_config(tmp_path, **config_overrides)


def run_decide(tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode decide`` with the fixture's database and configuration."""
    return ai_main(
        [
            *cli_args(tmp_path, mode="decide", run_artifacts=run_artifacts),
            "--config-dir",
            str(config_dir),
            *extra,
        ]
    )


def written(tmp_path) -> list[Path]:
    """Every artefact the mode wrote, sorted by name."""
    root = out_dir(tmp_path)
    return sorted(root.glob("*")) if root.exists() else []


def payload(tmp_path) -> dict:
    """The JSON artefact's document."""
    matches = sorted(out_dir(tmp_path).glob("*_decisions.json"))
    assert matches, "no JSON artefact was written"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def decisions_frame(tmp_path) -> pd.DataFrame:
    """The decisions CSV as a frame."""
    matches = sorted(out_dir(tmp_path).glob("*_decisions.csv"))
    assert matches, "no CSV artefact was written"
    return pd.read_csv(matches[0])


def table_counts(tmp_path) -> dict[str, int]:
    """Row counts of the tables the decision record must never write to."""
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return {
            table: int(connection.execute(f"select count(*) from {table}").fetchone()[0])
            for table in ("ticks", "candles", "features", "labels", "trades")
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_the_mode_writes_json_markdown_and_csv(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir) == EXIT_OK

    model_name = next(models_dir(tmp_path).glob("*_lightgbm.txt")).name
    stem = decisions_stem(model_stem(model_name))
    names = [path.name for path in written(tmp_path)]
    assert f"{stem}_decisions.json" in names
    assert f"{stem}_decisions.md" in names
    assert f"{stem}_decisions.csv" in names

    out = capsys.readouterr().out
    assert "ANJANEYA-AI decision layer (spec 24)" in out
    assert "  -- decisions (by reason) --" in out
    assert "  -- entries (by option side) --" in out
    assert "entry(ies)" in out


def test_one_decision_per_scored_row(tmp_path) -> None:
    """Nothing is dropped: a refusal is part of the record, not an omission."""
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir)
    document = payload(tmp_path)
    assert document["report"] == "spec 24 - the decision layer"
    assert document["spec"] == "24"
    assert document["summary"]["decisions"] == EXPECTED_ROWS
    assert document["summary"]["rows_in_scope"] == EXPECTED_ROWS
    assert len(document["decisions"]) == EXPECTED_ROWS
    assert document["summary"]["entries"] + document["summary"]["no_trade"] == EXPECTED_ROWS


def test_every_decision_carries_a_reason_from_the_closed_vocabulary(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir)
    document = payload(tmp_path)
    reasons = {str(row["reason"]) for row in document["decisions"]}
    assert reasons <= set(DECISION_REASONS)
    assert set(document["summary"]["reasons"]) == set(DECISION_REASONS)


def test_the_model_and_its_label_travel_with_the_record(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir)
    document = payload(tmp_path)
    assert document["model"]["name"].endswith("_lightgbm.txt")
    assert document["model"]["horizon_minutes"] == HORIZON
    assert document["horizon_minutes"] == HORIZON
    assert document["timeframe"] == "1m"


def test_the_gate_labels_the_run_instead_of_refusing_it(tmp_path) -> None:
    """Four captured days is INSUFFICIENT - and the record is still written, labelled."""
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir) == EXIT_OK
    document = payload(tmp_path)
    assert document["gate"]["stage"] == "INSUFFICIENT"
    assert document["results_label"] == "BELOW_GATE"
    assert document["verdict"]["refused"] is False
    # Relaxed 60 -> 45 (Phase 9); four days is still BELOW_GATE.
    assert document["gate"]["gates"]["ml_training_min_days"] == 45


def test_the_markdown_has_the_sections(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir)
    text = next(out_dir(tmp_path).glob("*_decisions.md")).read_text(encoding="utf-8")
    for heading in (
        "## 1. The policy",
        "## 2. What the policy decided",
        "## 3. Entries",
        "## Caveats",
    ):
        assert heading in text
    assert "spec 24's decision layer does not exist yet" not in text


def test_the_csv_is_the_whole_ladder_in_the_documented_order(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir)
    frame = decisions_frame(tmp_path)
    assert list(frame.columns) == list(DECISION_COLUMNS)
    assert len(frame) == EXPECTED_ROWS
    # the exit plan is *stated* in the record; no match, no price and no P&L is simulated here
    for forbidden in ("pnl", "entry_price", "exit_price", "fill_price"):
        assert forbidden not in frame.columns
    assert frame["quantity"].unique().tolist() == [75]


def test_the_mode_never_writes_to_duckdb(tmp_path) -> None:
    config_dir = seed(tmp_path)
    before = table_counts(tmp_path)
    run_decide(tmp_path, config_dir)
    assert table_counts(tmp_path) == before


def test_the_artifacts_are_registered_on_the_manifest(tmp_path) -> None:
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir, run_artifacts=True) == EXIT_OK
    runs = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert runs
    manifest = json.loads(runs[-1].read_text(encoding="utf-8"))
    assert manifest["mode"] == "decide"
    registered = {Path(str(path)).name for path in manifest["artifacts"]}
    assert any(name.endswith("_decisions.json") for name in registered)
    notes = " ".join(str(note) for note in manifest.get("notes", []))
    assert "entry(ies)" in notes


def test_the_mode_is_registered_so_it_is_not_planned(tmp_path) -> None:
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir) != EXIT_NOT_IMPLEMENTED



# ---------------------------------------------------------------------------
# The controls: the flags change the record, and the record says which flags ran
# ---------------------------------------------------------------------------
def test_min_probability_is_recorded_and_tightening_it_never_adds_entries(tmp_path) -> None:
    permissive = seed(tmp_path)
    run_decide(tmp_path, permissive, "--min-probability", "0.01")
    loose = payload(tmp_path)
    assert loose["policy"]["min_probability"] == 0.01

    strict = shrink_config(tmp_path)
    run_decide(tmp_path, strict, "--min-probability", "0.90")
    tight = payload(tmp_path)
    assert tight["policy"]["min_probability"] == 0.90
    assert tight["summary"]["entries"] <= loose["summary"]["entries"]


def test_min_margin_is_recorded_and_can_only_refuse(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir, "--min-margin", "0.5")
    document = payload(tmp_path)
    assert document["policy"]["min_margin"] == 0.5
    # nothing can enter without a 0.5 lead over the runner-up under this policy
    for row in document["decisions"]:
        if row["decision"] in ("ENTER_CE", "ENTER_PE"):
            assert row["margin"] >= 0.5


def test_the_daily_budget_is_honoured_and_reported(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(
        tmp_path, config_dir, "--min-probability", "0.01", "--max-decisions-per-day", "2"
    )
    document = payload(tmp_path)
    assert document["policy"]["max_decisions_per_day"] == 2
    per_day = document["summary"]["entries_by_day"]
    assert per_day, "the budget test needs at least one trading day"
    assert max(per_day.values()) <= 2


def test_the_daily_budget_reason_is_reachable_end_to_end(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(
        tmp_path, config_dir, "--min-probability", "0.01", "--max-decisions-per-day", "1"
    )
    document = payload(tmp_path)
    assert document["summary"]["reasons"]["DAILY_LIMIT_REACHED"] > 0


def test_a_single_contract_can_be_scoped(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir, "--symbol", CE_SYMBOL)
    document = payload(tmp_path)
    assert document["symbols"] == [CE_SYMBOL]
    assert {row["tradingsymbol"] for row in document["decisions"]} == {CE_SYMBOL}
    assert document["summary"]["decisions"] == DAYS * BARS


def test_the_window_narrows_the_rows(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir, "--from", "2026-10-01", "--to", "2026-10-01")
    document = payload(tmp_path)
    assert document["summary"]["decisions"] == 2 * BARS
    assert document["summary"]["sessions"] == 1


def test_dry_run_writes_nothing_and_prints_the_plan(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir, "--dry-run") == EXIT_OK
    out = capsys.readouterr().out
    assert "dry run, nothing is written" in out
    assert "would be decided" in out
    assert "a decision is not an order and not advice" in out
    assert written(tmp_path) == []


def test_no_write_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir, "--no-write") == EXIT_OK
    assert written(tmp_path) == []
    assert "nothing (--no-write)" in capsys.readouterr().out


def test_json_output_is_machine_readable(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    assert run_decide(tmp_path, config_dir, "--json") == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["mode"] == "decide"
    assert document["run_id"]
    assert document["summary"]["decisions"] == EXPECTED_ROWS
    assert document["write"] is True
    assert len(document["written"]) == 3



# ---------------------------------------------------------------------------
# The honesty: no model, no rows and no database are reported, never fatal
# ---------------------------------------------------------------------------
def test_no_model_is_reported_not_a_crash(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path, with_model=False)
    assert run_decide(tmp_path, config_dir) == EXIT_OK
    assert "no model artefact to decide with" in capsys.readouterr().err
    assert written(tmp_path) == []


def test_no_feature_rows_is_reported_not_a_crash(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path, seed_db=False)
    assert run_decide(tmp_path, config_dir) == EXIT_OK
    assert "no stored 1m feature rows in scope" in capsys.readouterr().err
    assert written(tmp_path) == []


def test_the_caveats_are_stored_and_printed(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir)
    document = payload(tmp_path)
    joined = " ".join(document["caveats"]).lower()
    assert "never places an order" in joined
    assert "a decision is not an edge" in joined
    assert "stated, not simulated" in joined
    assert "no look-ahead" in joined
    # the model's own label is part of the honesty block
    assert any("EXPLORATORY" in caveat for caveat in document["caveats"])


def test_the_console_says_what_a_decision_is_not(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    run_decide(tmp_path, config_dir, "--json")
    capsys.readouterr()
    run_decide(tmp_path, config_dir)
    out = capsys.readouterr().out
    assert "results    : BELOW_GATE" in out
    assert "gate       : " in out


def test_a_model_trained_on_another_timeframe_is_refused(tmp_path, capsys) -> None:
    """A decision on rows the model never saw is a configuration error, not a record."""
    config_dir = seed(tmp_path)
    from core.app_bootstrap import EXIT_CONFIG_ERROR

    assert run_decide(tmp_path, config_dir, "--timeframe", "5m") == EXIT_CONFIG_ERROR
    assert "was trained on" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The writer, in isolation
# ---------------------------------------------------------------------------
def test_the_writer_is_atomic_and_counted(tmp_path) -> None:
    from apps.anjaneeya_ai.decision_writer import DecisionWriter
    from core.decision import DecisionPolicy, DecisionRow, build_decision_report

    stamp = datetime(2026, 10, 1, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
    rows = [
        DecisionRow(
            timestamp=stamp,
            tradingsymbol="NIFTY01OCT2625000CE",
            option_type="CE",
            probabilities={"UP_CONTINUATION": 0.6, "RANGE": 0.4},
            timeframe="1m",
            trading_date=date(2026, 10, 1),
        )
    ]
    report = build_decision_report(rows, DecisionPolicy(), as_of=date(2026, 10, 1))
    writer = DecisionWriter(out_dir(tmp_path), run_id="run", config_hash="hash")
    paths = writer.write(report, stem="fixture")
    assert len(paths) == 3
    assert writer.stats.files == 3
    assert writer.stats.failures == 0
    assert writer.stats.decisions == 1
    assert writer.stats.entries == 1
    assert not list(out_dir(tmp_path).glob("*.tmp"))


def test_the_writer_counts_a_failure_rather_than_raising(tmp_path, monkeypatch) -> None:
    from apps.anjaneeya_ai.decision_writer import DecisionWriter
    from core.decision import DecisionPolicy, build_decision_report

    report = build_decision_report([], DecisionPolicy(), as_of=date(2026, 10, 1))
    writer = DecisionWriter(out_dir(tmp_path), run_id="run")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", explode)
    paths = writer.write(report, stem="fixture")
    # the two text artefacts fail and are counted; the CSV is written by pandas, not by
    # ``Path.write_text``, so it still lands - a partial failure, visible and never raised
    assert writer.stats.failures == 2
    assert writer.stats.files == 1
    assert writer.json_path("fixture") not in paths
    assert writer.csv_path("fixture") in paths

