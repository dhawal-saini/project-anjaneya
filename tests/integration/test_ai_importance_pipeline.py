"""End-to-end ANJANEYA-AI feature importance (Stage 5.2, spec 27).

``--mode importance`` is exercised through the real ``main()`` against a seeded database and a
real model artefact (the pair ``--mode train`` writes), so every assertion is about what
actually happened:

    the inputs    ->  the stored feature rows and labels, read back by the mode itself rather
                      than handed in as a frame; nothing is refitted
    the answer    ->  native importance off the booster, a permutation delta with a confidence
                      interval per feature, a total per source of spec 27's list, and a
                      ranking that finds the feature the fixture model's class really is a
                      function of
    the honesty   ->  a feature the sample cannot measure is *not measured* (never 0), a
                      feature no source claims is reported as unmapped, and the model's own
                      results label travels in the caveats
    the artefacts ->  one JSON, one Markdown and one CSV under ``data/reports/importance/``,
                      named after the model, registered on the run manifest
    the refusals  ->  no model, no feature rows, a timeframe or horizon the model never saw,
                      a single ``--repeats``, ``--dry-run`` and ``--no-write`` each behave
                      exactly as documented, and nothing is written to DuckDB

Nothing here touches the network: the only inputs are the stored database and the model
artefact on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_OK
from core.ml import CORRELATED_FEATURES_CAVEAT, IMPORTANCE_CAVEAT
from core.storage import connect
from tests.support.ai import (
    ONE_DAY_FOLDS,
    cli_args,
    seed_source,
    shrink_config,
    write_signal_model,
)

#: The stem the fixture model is written under (``write_signal_model``'s own name).
MODEL_NAME = "1m_h5_2026-09-28_2026-10-01_lightgbm.txt"

#: ``model.importance.repeats`` in the shipped configuration.
DEFAULT_REPEATS = 5

#: The number of stored feature columns a model is trained on (the trainable contract).
TRAINABLE_COLUMNS = 77

#: Spec 27's own sources, in the order the shipped configuration declares them.
SPEC_27_TOPICS = (
    "nifty_regime",
    "option_ma",
    "candle_behaviour",
    "support_resistance",
    "structure",
    "iv",
    "greeks",
    "oi",
    "option_residual",
    "time_of_day",
    "cycle_phase",
    "time_to_expiry",
)

#: The CSV's columns, as ``apps/anjaneeya_ai/importance_writer.py`` declares them.
CSV_COLUMNS = (
    "rank",
    "column",
    "topic",
    "group",
    "measured",
    "native",
    "native_share",
    "native_rank",
    "permutation_mean",
    "permutation_std",
    "ci_low",
    "ci_high",
    "null_share",
    "constant",
    "shap",
    "reason",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def models_dir(tmp_path) -> Path:
    return tmp_path / "data" / "models"


def reports_dir(tmp_path) -> Path:
    return tmp_path / "data" / "reports" / "importance"


def seed(tmp_path) -> Path:
    """A seeded database (features, labels, candles) plus the fixture's configuration."""
    seed_source(db_file(tmp_path))
    return shrink_config(tmp_path, walk_forward=ONE_DAY_FOLDS)


def write_model(tmp_path) -> Path:
    """The artefact pair ``--mode train`` would have written for the fixture."""
    return write_signal_model(models_dir(tmp_path))


def run_importance(tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode importance`` with the fixture's database and configuration."""
    return ai_main(
        [*cli_args(tmp_path, mode="importance", run_artifacts=run_artifacts), "--config-dir", str(config_dir), *extra]
    )


def artifacts(tmp_path) -> dict[str, Path]:
    """The three importance artefacts, keyed by extension."""
    root = reports_dir(tmp_path)
    found: dict[str, Path] = {}
    for suffix in (".json", ".md", ".csv"):
        for path in sorted(root.glob(f"*{suffix}")):
            found[suffix] = path
    return found


def payload(tmp_path) -> dict:
    """The JSON measurement, read back from disk."""
    return json.loads(artifacts(tmp_path)[".json"].read_text(encoding="utf-8"))


def features(payload_: dict) -> dict[str, dict]:
    """The payload's features, keyed by column."""
    return {item["column"]: item for item in payload_["features"]}


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


def topics(payload_: dict) -> dict[str, dict]:
    """The payload's sources, keyed by topic."""
    return {item["key"]: item for item in payload_["topics"]}


# ---------------------------------------------------------------------------
# The refusals: no model, a bad path, a bad timeframe or horizon, a bad repeat count
# ---------------------------------------------------------------------------
def test_without_a_model_the_run_says_so_and_writes_nothing(tmp_path, capsys) -> None:
    """A measurement needs a model; the exit code is 0 because nothing went wrong."""
    config_dir = seed(tmp_path)
    exit_code = run_importance(tmp_path, config_dir)
    captured = capsys.readouterr()
    assert exit_code == EXIT_OK
    assert "no model artefact to measure" in captured.err
    assert not reports_dir(tmp_path).exists()


def test_a_model_path_that_does_not_exist_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--model", str(tmp_path / "nope.txt"))
    assert exit_code == EXIT_CONFIG_ERROR
    assert "does not exist" in capsys.readouterr().err


def test_a_timeframe_the_model_never_saw_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--timeframe", "5m")
    captured = capsys.readouterr()
    assert exit_code == EXIT_CONFIG_ERROR
    assert "trained on 1m rows" in captured.err
    assert not reports_dir(tmp_path).exists()


def test_a_horizon_the_model_never_saw_is_a_no_model_not_a_crash(tmp_path, capsys) -> None:
    """``--horizon 10`` with only a 5m model is a NO_MODEL (as in explain and backtest)."""
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--horizon", "10")
    assert exit_code == EXIT_OK
    assert "no model artefact to measure" in capsys.readouterr().err


def test_one_repeat_is_refused_because_it_has_no_spread(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--repeats", "1")
    assert exit_code == EXIT_CONFIG_ERROR
    assert "at least 2" in capsys.readouterr().err


def test_no_feature_rows_in_scope_is_a_no_data(tmp_path, capsys) -> None:
    config_dir = shrink_config(tmp_path, walk_forward=ONE_DAY_FOLDS)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--from", "2030-01-01")
    assert exit_code == EXIT_OK
    assert "no stored 1m feature rows in scope" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The artefacts: three files, one measurement
# ---------------------------------------------------------------------------
def test_the_mode_writes_three_artefacts_named_after_the_model(tmp_path, capsys) -> None:
    """A ranking must be traceable to the model it describes, so the stem is the model's."""
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    written = artifacts(tmp_path)
    stem = Path(MODEL_NAME).stem
    assert set(written) == {".json", ".md", ".csv"}
    assert written[".json"].name == f"{stem}_importance.json"
    assert written[".md"].name == f"{stem}_importance.md"
    assert written[".csv"].name == f"{stem}_features.csv"
    printed = capsys.readouterr().out
    assert "ANJANEYA-AI feature importance (spec 27)" in printed
    assert "caveat:" in printed


def test_the_json_carries_the_measurement_the_sample_and_the_sources(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    document = payload(tmp_path)
    assert document["run_id"] and document["config_hash"]
    assert document["spec"] == "27"
    assert document["classes"] == [
        "UP_CONTINUATION",
        "DOWN_CONTINUATION",
        "RANGE",
        "REVERSAL",
    ]
    measurement = document["measurement"]
    assert measurement["metric"] == "log_loss"
    assert measurement["direction"] == "lower_is_better"
    assert measurement["repeats"] == 2
    assert measurement["seed"] == 42  # the model's own seed, since importance.seed is null
    assert measurement["baseline"] > 0
    assert measurement["features"] == TRAINABLE_COLUMNS
    assert document["sample"]["name"] == "holdout"
    assert document["sample"]["rows_used"] > 0
    assert document["sample"]["capped"] is False
    assert document["counts"]["features"] == TRAINABLE_COLUMNS
    assert document["counts"]["measured"] + document["counts"]["unmeasured"] == TRAINABLE_COLUMNS
    assert document["counts"]["shap_status"] == "unavailable"  # shap is not installed here
    assert ".[explain]" in document["shap"]["reason"]


def test_every_source_of_spec_27_is_present_in_the_answer(tmp_path) -> None:
    """The spec asks *which* sources carry the information, so all of them must be addressed."""
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    document = payload(tmp_path)
    by_key = topics(document)
    assert list(by_key) == [
        "nifty_regime",
        "option_ma",
        "candle_behaviour",
        "support_resistance",
        "structure",
        "price_action",
        "iv",
        "greeks",
        "oi",
        "option_residual",
        "time_of_day",
        "cycle_phase",
        "time_to_expiry",
        "day_of_week",
        "option_contract",
        "option_context",
    ]
    for key in SPEC_27_TOPICS:
        entry = by_key[key]
        assert entry["features"] >= 1
        if entry["permutation"]["measured"]:
            assert entry["permutation"]["ci_low"] <= entry["permutation"]["mean"]
            assert entry["permutation"]["mean"] <= entry["permutation"]["ci_high"]
        else:
            assert entry["permutation"]["reason"]
    # The fixture's stored rows carry almost nothing, so most features cannot be measured - and
    # the artefact says which, instead of reporting a zero for every one of them.
    unmeasured = [item for item in document["features"] if not item["permutation"]["measured"]]
    assert unmeasured
    for item in unmeasured:
        assert item["permutation"]["mean"] is None
        assert item["rank"] is None
        assert item["permutation"]["reason"]


def test_a_source_the_fixture_cannot_measure_says_which_and_why(tmp_path) -> None:
    """A source whose columns are all NULL is "these rows cannot say", not "no information"."""
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    greeks = topics(payload(tmp_path))["greeks"]
    assert greeks["features"] == 4
    assert greeks["measured_features"] <= greeks["features"]
    if greeks["measured_features"] == 0:
        reason = greeks["permutation"]["reason"]
        assert "not here to measure" in reason or "could be measured" in reason


def test_every_row_of_the_csv_carries_its_reason_and_the_columns_are_the_documented_ones(
    tmp_path,
) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    frame = pd.read_csv(artifacts(tmp_path)[".csv"])
    assert list(frame.columns) == list(CSV_COLUMNS)
    assert len(frame) == TRAINABLE_COLUMNS
    assert frame["column"].is_unique
    measured = frame[frame["measured"]]
    assert list(measured["rank"]) == sorted(measured["rank"])
    assert measured["permutation_mean"].notna().all()
    blocked = frame[~frame["measured"]]
    assert blocked["permutation_mean"].isna().all()
    assert blocked["reason"].notna().all()


def test_the_markdown_prints_the_sources_the_features_and_the_caveats(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    text = artifacts(tmp_path)[".md"].read_text(encoding="utf-8")
    assert "# ANJANEYA feature importance (spec 27)" in text
    assert "## Sources (spec 27's question)" in text
    assert "## Every feature" in text
    assert IMPORTANCE_CAVEAT in text
    assert CORRELATED_FEATURES_CAVEAT in text
    assert "the model measured is labelled EXPLORATORY" in text
    assert "| `nifty_regime` NIFTY regime |" in text


def store_class_labels(tmp_path) -> None:
    """Make the stored labels follow ``option_type`` - the relation the fixture model learned.

    The fixture's own labels are minute-cycled (:func:`tests.support.ai.label_rows`), so a model
    trained on the ``option_type`` -> class rule is confidently *wrong* on them, and permuting
    its one informative feature barely moves the score. That is a property of the fixture, not
    of the measurement: an importance run can only find a signal the store actually carries, so
    this puts the signal where the model is right about it. ``CE`` becomes ``UP_CONTINUATION``
    and ``PE`` becomes ``DOWN_CONTINUATION`` - exactly what ``signal_class_frame`` trains on.
    """
    connection = connect(db_file(tmp_path))
    try:
        connection.execute(
            "update labels set label = case when option_type = 'CE' "
            "then 'UP_CONTINUATION' else 'DOWN_CONTINUATION' end"
        )
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The measurement itself: does it find the signal, is it reproducible, is it read-only
# ---------------------------------------------------------------------------
def test_the_ranking_finds_the_feature_the_fixture_class_is_a_function_of(tmp_path) -> None:
    """The decidable case: the store's class follows ``option_type``, so that column ranks first.

    ``option_type`` is the one model input that carries the fixture's class (the NIFTY, candle
    and level columns are functions of the minute, and the class is not), so a measurement that
    cannot rank it first is not measuring the model that was loaded.
    """
    config_dir = seed(tmp_path)
    store_class_labels(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    document = payload(tmp_path)
    by_column = features(document)
    top = min(
        (item for item in document["features"] if item["rank"]),
        key=lambda item: item["rank"],
    )
    assert top["column"] == "option_type"
    assert top["topic"] == "option_contract"
    assert top["rank"] == 1
    assert top["permutation"]["mean"] > 0
    assert by_column["option_type"]["group"] == "identity"
    assert by_column["option_type"]["permutation"]["ci_low"] > 0  # not "no effect"
    assert topics(document)["option_contract"]["rank"] == 1
    assert topics(document)["option_contract"]["permutation"]["mean"] > 0


def test_two_runs_at_one_seed_produce_the_same_numbers(tmp_path) -> None:
    """Reproducibility, end to end: the same rows, the same seed, the same deltas."""
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    first = payload(tmp_path)
    run_importance(tmp_path, config_dir, "--repeats", "2")
    second = payload(tmp_path)
    assert first["measurement"] == second["measurement"]
    assert first["sample"] == second["sample"]
    assert [item["permutation"] for item in first["features"]] == [
        item["permutation"] for item in second["features"]
    ]
    assert [item["rank"] for item in first["features"]] == [
        item["rank"] for item in second["features"]
    ]


def test_the_run_never_writes_to_the_database(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    before = counts(tmp_path)
    assert run_importance(tmp_path, config_dir, "--repeats", "2") == EXIT_OK
    assert counts(tmp_path) == before


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--dry-run")
    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "ANJANEYA-AI feature importance (spec 27) - dry run, nothing is written" in printed
    assert "  sample     :" in printed
    assert "  sources    :" in printed
    assert "  measurement:" in printed
    assert not reports_dir(tmp_path).exists()


def test_no_write_prints_the_measurement_and_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--repeats", "2", "--no-write")
    assert exit_code == EXIT_OK
    printed = capsys.readouterr().out
    assert "ANJANEYA-AI feature importance (spec 27)" in printed
    assert "wrote      : nothing (--no-write)" in printed
    assert not reports_dir(tmp_path).exists()


def test_json_output_is_the_same_document_the_artefact_carries(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--repeats", "2", "--json")
    assert exit_code == EXIT_OK
    printed = json.loads(capsys.readouterr().out)
    document = payload(tmp_path)
    assert printed["mode"] == "importance"
    assert printed["model_path"] == str(models_dir(tmp_path) / MODEL_NAME)
    assert printed["counts"] == document["counts"]
    assert printed["measurement"] == document["measurement"]
    assert printed["topics"] == document["topics"]
    assert printed["writer"] == {
        "features": TRAINABLE_COLUMNS,
        "sources": 16,
        "files": 3,
        "failures": 0,
    }


def test_run_artifacts_are_registered_on_the_manifest(tmp_path) -> None:
    config_dir = seed(tmp_path)
    write_model(tmp_path)
    exit_code = run_importance(tmp_path, config_dir, "--repeats", "2", run_artifacts=True)
    assert exit_code == EXIT_OK
    runs = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert runs
    manifest = json.loads(runs[-1].read_text(encoding="utf-8"))
    assert manifest["mode"] == "importance"
    registered = {Path(str(path)).name for path in manifest["artifacts"]}
    assert any(name.endswith("_importance.json") for name in registered)
    assert any(name.endswith("_features.csv") for name in registered)
    notes = " ".join(str(note) for note in manifest.get("notes", []))
    assert IMPORTANCE_CAVEAT in notes




