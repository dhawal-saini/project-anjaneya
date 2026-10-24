"""End-to-end ANJANEYA-AI explanations (Stage 5.1, spec 26).

``--mode explain`` is exercised through the real ``main()`` against a seeded database and a
real model artefact (the pair ``--mode train`` writes), so every assertion is about what
actually happened:

    the input     ->  the stored feature rows, read back by the mode itself rather than
                      handed in as a frame
    the blocks    ->  spec 26's list, in the configured order, with the AI block last and the
                      model's own class distribution in it
    the honesty   ->  a ``NULL`` feature is rendered ``unknown`` and never 0, an explanation
                      carries the row's provenance, and every artefact carries the caveats
                      (including the model's own results label)
    the artefacts ->  one JSON, one Markdown and one CSV of lines under
                      ``data/reports/explanations/``, named after the model, registered on the
                      run manifest
    the refusals  ->  no model, no feature rows, a timeframe or horizon the model never saw,
                      ``--dry-run`` and ``--no-write`` each behave exactly as documented, and
                      nothing is written to DuckDB

Nothing here touches the network: the only inputs are the stored database and the model
artefact on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_OK
from core.constants import get_timezone
from core.storage import connect
from tests.support.ai import cli_args, seed_source, shrink_config, write_signal_model
from tests.support.engine import CE_SYMBOL, PE_SYMBOL

#: The stem the fixture model is written under (``write_signal_model``'s own name).
MODEL_NAME = "1m_h5_2026-09-28_2026-10-01_lightgbm.txt"

#: ``model.explainability.rows`` in the shipped configuration: the default ``--rows``.
DEFAULT_ROWS = 5

#: Spec 26's blocks plus the model's own, in the order the shipped configuration declares.
SECTIONS = ("nifty", "option", "price_action", "candle", "support_resistance", "context", "ai")

#: A stored feature the fixture leaves ``NULL`` (only ``support_touch_count`` is set).
NULL_FEATURE = "support_strength"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def models_dir(tmp_path) -> Path:
    return tmp_path / "data" / "models"


def explanations_dir(tmp_path) -> Path:
    return tmp_path / "data" / "reports" / "explanations"


def seed(tmp_path) -> Path:
    """A seeded database (features, labels, candles) plus the fixture's configuration."""
    seed_source(db_file(tmp_path))
    return shrink_config(tmp_path)


def write_model(tmp_path) -> Path:
    """The artefact pair ``--mode train`` would have written for the fixture."""
    return write_signal_model(models_dir(tmp_path))


def run_explain(tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode explain`` with the fixture's database and configuration."""
    arguments = [
        *cli_args(tmp_path, mode="explain", run_artifacts=run_artifacts),
        "--config-dir",
        str(config_dir),
    ]
    return ai_main([*arguments, *extra])


def artifacts(tmp_path) -> dict[str, Path]:
    """The three explanation artefacts, keyed by extension."""
    root = explanations_dir(tmp_path)
    found: dict[str, Path] = {}
    for suffix in (".json", ".md", ".csv"):
        for path in sorted(root.glob(f"*{suffix}")):
            found[suffix] = path
    return found


def payload(tmp_path) -> dict:
    """The JSON explanations, read back from disk."""
    return json.loads(artifacts(tmp_path)[".json"].read_text(encoding="utf-8"))


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


def newest_feature_moment(tmp_path) -> pd.Timestamp:
    """The newest stored feature timestamp, in exchange-local time."""
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        value = connection.execute("select max(timestamp) from features").fetchone()[0]
    finally:
        connection.close()
    return pd.Timestamp(value).tz_convert(get_timezone("Asia/Kolkata"))


# ---------------------------------------------------------------------------
# Nothing to explain
# ---------------------------------------------------------------------------
def test_without_a_model_the_run_says_so_and_writes_nothing(tmp_path, capsys) -> None:
    """An explanation needs a model; the exit code is 0 because nothing went wrong."""
    config_dir = seed(tmp_path)

    exit_code = run_explain(tmp_path, config_dir)

    assert exit_code == EXIT_OK
    captured = capsys.readouterr()
    assert "no model artefact to explain" in captured.err
    assert "--mode train" in captured.err
    assert not explanations_dir(tmp_path).exists()


def test_a_model_path_that_does_not_exist_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--model", str(tmp_path / "nope.txt"))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "does not exist" in capsys.readouterr().err


def test_a_store_without_feature_rows_says_so_and_writes_nothing(tmp_path, capsys) -> None:
    """The database is bootstrapped but empty: that is a NO_DATA, not a crash."""
    config_dir = shrink_config(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir)

    assert exit_code == EXIT_OK
    assert "no stored 1m feature rows in scope" in capsys.readouterr().err
    assert not explanations_dir(tmp_path).exists()


# ---------------------------------------------------------------------------
# The explanations themselves
# ---------------------------------------------------------------------------
def test_the_mode_writes_three_artefacts_named_after_the_model(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir)

    assert exit_code == EXIT_OK
    found = artifacts(tmp_path)
    assert set(found) == {".json", ".md", ".csv"}
    stem = MODEL_NAME[: -len(".txt")]
    assert found[".json"].name == f"{stem}_explanations.json"
    assert found[".md"].name == f"{stem}_explanations.md"
    assert found[".csv"].name == f"{stem}_entries.csv"
    printed = capsys.readouterr().out
    assert "ANJANEYA-AI explanations (spec 26)" in printed
    assert "NIFTY:" in printed
    assert "AI:" in printed


def test_every_block_of_spec_26_is_present_with_the_model_block_last(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_explain(tmp_path, config_dir)

    document = payload(tmp_path)

    assert document["counts"]["predictions"] == DEFAULT_ROWS
    assert document["policy"]["spec"] == "26"
    assert [block["key"] for block in document["policy"]["blocks"]] == list(SECTIONS[:-1])
    explanations = document["explanations"]
    assert len(explanations) == DEFAULT_ROWS
    for explanation in explanations:
        assert [section["key"] for section in explanation["sections"]] == list(SECTIONS)
        # every configured column is rendered exactly once, whatever its value
        columns = [
            entry["column"] for section in explanation["sections"] for entry in section["entries"]
        ]
        assert sorted(columns) == sorted(document["policy"]["columns"])


def test_the_ai_block_carries_the_model_own_class_distribution(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_explain(tmp_path, config_dir)

    document = payload(tmp_path)

    model = document["model"]
    assert model["algorithm"] == "lightgbm"
    assert model["horizon_minutes"] == 5
    for explanation in document["explanations"]:
        ai_section = explanation["sections"][-1]
        assert ai_section["key"] == "ai"
        assert ai_section["available"] is True
        names = [item["class"] for item in ai_section["probabilities"]]
        assert sorted(names) == sorted(model["classes"])
        ranks = [item["rank"] for item in ai_section["probabilities"]]
        assert ranks == sorted(ranks) == list(range(len(model["classes"])))
        probabilities = [item["probability"] for item in ai_section["probabilities"]]
        assert probabilities == sorted(probabilities, reverse=True)
        assert probabilities[0] == pytest.approx(explanation["prediction"]["probability"])
        assert explanation["prediction"]["class"] == ai_section["probabilities"][0]["class"]


def test_a_null_feature_is_reported_unknown_and_never_zero(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_explain(tmp_path, config_dir)

    document = payload(tmp_path)

    support = next(
        section
        for section in document["explanations"][0]["sections"]
        if section["key"] == "support_resistance"
    )
    line = next(entry for entry in support["entries"] if entry["column"] == NULL_FEATURE)
    assert line["known"] is False
    assert line["value"] is None
    assert line["text"] == "unknown"
    assert line["kind"] == "unknown"
    # and the same line in the flat CSV a reader pivots
    frame = pd.read_csv(artifacts(tmp_path)[".csv"])
    row = frame[
        (frame["section"] == "support_resistance") & (frame["column"] == NULL_FEATURE)
    ].iloc[0]
    assert bool(row["known"]) is False
    assert row["text"] == "unknown"


def test_the_audit_block_carries_the_row_provenance(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_explain(tmp_path, config_dir)

    document = payload(tmp_path)

    audit = document["explanations"][0]["audit"]
    assert audit["quality_flag"] == "OK"
    assert audit["source"] == "DERIVED"
    assert audit["run_id"]
    assert audit["fetch_timestamp"]


def test_the_newest_predictions_are_explained_first_and_rows_caps_them(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--rows", "2")

    assert exit_code == EXIT_OK
    document = payload(tmp_path)
    assert document["counts"]["predictions"] == 2
    stamps = [explanation["identity"]["timestamp"] for explanation in document["explanations"]]
    assert stamps == sorted(stamps, reverse=True)
    assert pd.Timestamp(stamps[0]).tz_convert(
        get_timezone("Asia/Kolkata")
    ) == newest_feature_moment(tmp_path)


def test_the_window_and_the_symbol_scope_are_honoured(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(
        tmp_path,
        config_dir,
        "--rows",
        "3",
        "--from",
        "2026-09-30",
        "--to",
        "2026-09-30",
        "--symbol",
        PE_SYMBOL,
    )

    assert exit_code == EXIT_OK
    document = payload(tmp_path)
    assert document["window"]["symbols"] == [PE_SYMBOL]
    symbols = {
        explanation["identity"]["tradingsymbol"] for explanation in document["explanations"]
    }
    assert symbols == {PE_SYMBOL}
    for explanation in document["explanations"]:
        stamp = pd.Timestamp(explanation["identity"]["timestamp"])
        assert stamp.tz_convert(get_timezone("Asia/Kolkata")).date().isoformat() == "2026-09-30"


# ---------------------------------------------------------------------------
# The refusals and the honesty rules
# ---------------------------------------------------------------------------
def test_a_timeframe_the_model_never_saw_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--timeframe", "5m")

    assert exit_code == EXIT_CONFIG_ERROR
    message = capsys.readouterr().err
    assert "was trained on 1m rows" in message
    assert not explanations_dir(tmp_path).exists()


def test_a_horizon_the_model_never_saw_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    model_path = write_model(tmp_path)

    exit_code = run_explain(
        tmp_path, config_dir, "--model", str(model_path), "--horizon", "10"
    )

    assert exit_code == EXIT_CONFIG_ERROR
    assert "was trained for horizon 5m" in capsys.readouterr().err


def test_a_horizon_with_no_artefact_says_which_horizons_exist(tmp_path, capsys) -> None:
    """``--horizon 10`` with only a 5m model is a NO_MODEL, not a crash (as in backtest)."""
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--horizon", "10")

    assert exit_code == EXIT_OK
    assert "no model artefact to explain" in capsys.readouterr().err
    assert not explanations_dir(tmp_path).exists()


def test_the_caveats_travel_with_every_artefact(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_explain(tmp_path, config_dir)

    document = payload(tmp_path)

    assert any("not a claim about why the market moved" in caveat for caveat in document["caveats"])
    assert any("this is not the decision layer" in caveat for caveat in document["caveats"])
    assert any("quality_flag" in caveat for caveat in document["caveats"])
    assert any("EXPLORATORY" in caveat for caveat in document["caveats"])
    markdown = artifacts(tmp_path)[".md"].read_text(encoding="utf-8")
    assert "## Caveats" in markdown
    assert f"### {CE_SYMBOL} @" in markdown or f"### {PE_SYMBOL} @" in markdown
    assert "**AI**" in markdown


def test_the_run_never_writes_to_the_database(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    before = counts(tmp_path)

    assert run_explain(tmp_path, config_dir, "--rows", "2") == EXIT_OK

    assert counts(tmp_path) == before


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--dry-run")

    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "ANJANEYA-AI explanations (spec 26) - dry run, nothing is written" in printed
    assert "rows       :" in printed
    assert "blocks     :" in printed
    assert not explanations_dir(tmp_path).exists()


def test_no_write_prints_the_explanations_and_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--rows", "1", "--no-write")

    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "NIFTY:" in printed
    assert "wrote      : nothing (--no-write)" in printed
    assert not explanations_dir(tmp_path).exists()


def test_json_output_is_the_same_document_the_artefact_carries(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, "--json")

    assert exit_code == EXIT_OK
    printed = json.loads(capsys.readouterr().out)
    document = payload(tmp_path)
    assert printed["mode"] == "explain"
    assert printed["model_path"] == str(models_dir(tmp_path) / MODEL_NAME)
    assert printed["counts"] == document["counts"]
    assert [item["identity"] for item in printed["explanations"]] == [
        item["identity"] for item in document["explanations"]
    ]


def test_run_artifacts_are_registered_on_the_manifest(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)

    exit_code = run_explain(tmp_path, config_dir, run_artifacts=True)

    assert exit_code == EXIT_OK
    runs = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert runs
    manifest = json.loads(runs[-1].read_text(encoding="utf-8"))
    assert manifest["mode"] == "explain"
    registered = {Path(str(path)).name for path in manifest["artifacts"]}
    assert any(name.endswith("_explanations.json") for name in registered)
    assert any(name.endswith("_entries.csv") for name in registered)



