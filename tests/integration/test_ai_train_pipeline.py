"""End-to-end ANJANEYA-AI training pipeline (Stage 4.2, specs 21-22).

``--mode train`` is exercised through the real ``main()`` against a seeded database and a
configuration whose walk-forward windows fit the fixture, so every assertion is made against
what actually happened:

    features + labels  ->  the same leak-free matrix ``--mode dataset`` writes, in memory
    the gate           ->  below the exploratory gate the run refuses (exit code 4) and
                           writes nothing; ``--force`` trains and keeps the BELOW_GATE label
    the folds          ->  train / validate per fold, baselines scored on the same rows,
                           per-fold metrics and an honest across-fold aggregate
    the final model    ->  fitted on every row outside the holdout, written as LightGBM's
                           own text plus metadata that can be read a year later
    the holdout        ->  untouched unless ``--score-holdout`` asks for it, once
    the sources        ->  the feature and label tables are never written

Nothing here touches the network: the only input is what is stored in DuckDB.
"""

from __future__ import annotations

import json

import pytest
import yaml

from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_GATE_REFUSED, EXIT_OK
from core.ml import BASELINE_NAMES
from tests.support.ai import (
    CLASSES,
    FAST_TRAINING,
    ONE_DAY_FOLDS,
    cli_args,
    seed_source,
    shrink_config,
)

#: Rows in one fold's validation window: one contract-day of labelled bars.
BARS = 60
LABELLED_PER_CONTRACT_DAY = BARS - 30 + 1
VALIDATE_ROWS = 2 * LABELLED_PER_CONTRACT_DAY


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def models_dir(tmp_path):
    return tmp_path / "data" / "models"


def seed(tmp_path, **config_overrides):
    """A seeded database plus the shrunk configuration directory to hand the CLI."""
    seed_source(db_file(tmp_path))
    config_dir = shrink_config(
        tmp_path,
        walk_forward=config_overrides.pop("walk_forward", ONE_DAY_FOLDS),
        training=config_overrides.pop("training", FAST_TRAINING),
        **config_overrides,
    )
    return config_dir


def run_train(tmp_path, config_dir, *extra: str, run_artifacts: bool = False) -> int:
    """``--mode train`` with the fixture's database and configuration."""
    arguments = [*cli_args(tmp_path, mode="train", run_artifacts=run_artifacts), "--config-dir", str(config_dir)]
    return ai_main([*arguments, *extra])


def metadata(tmp_path, horizon: int = 5) -> dict:
    """The metadata JSON of one horizon's model artefact."""
    paths = sorted(models_dir(tmp_path).glob(f"*_h{horizon}_*_metadata.json"))
    assert len(paths) == 1, [path.name for path in paths]
    return json.loads(paths[0].read_text(encoding="utf-8"))


def boosters(tmp_path, horizon: int = 5) -> list:
    """The LightGBM model text files on disk for one horizon."""
    return sorted(models_dir(tmp_path).glob(f"*_h{horizon}_*.txt"))


def query(tmp_path, sql: str):
    from core.storage import connect

    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The gate: refuse before doing any work, unless the operator insists
# ---------------------------------------------------------------------------
def test_below_the_exploratory_gate_the_run_refuses_and_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5")
    assert exit_code == EXIT_GATE_REFUSED
    captured = capsys.readouterr()
    assert "refusing to train" in captured.err
    assert "--force" in captured.err
    assert "exploratory" in captured.err
    assert not models_dir(tmp_path).exists() or not list(models_dir(tmp_path).iterdir())


def test_force_trains_but_keeps_the_below_gate_label(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert "BELOW_GATE" in out
    assert "INSUFFICIENT" in out
    assert metadata(tmp_path)["results_label"] == "BELOW_GATE"
    assert metadata(tmp_path)["verdict"]["production_model"] is False


def test_a_refused_run_records_the_refusal_in_the_manifest(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", run_artifacts=True)
    manifests = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert manifests, "the run should still write a manifest"
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "REFUSED"
    assert any("refused by the data gate" in note for note in payload["notes"])
    assert payload["data_coverage"]["stage"] == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------
def test_the_mode_trains_every_fold_and_writes_a_model(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert "horizon 5m : SUCCESS" in out
    assert "2 trained, 0 skipped" in out
    assert "final model" in out
    boosters_ = boosters(tmp_path)
    assert len(boosters_) == 1
    assert boosters_[0].name.startswith("1m_h5_")
    # LightGBM's own text: it must be loadable without ANJANEYA in the picture.
    text = boosters_[0].read_text(encoding="utf-8")
    assert "tree" in text and "feature_names" in text


def test_the_model_metadata_explains_the_model(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    payload = metadata(tmp_path)
    assert payload["dataset_name"].startswith("1m_h5_")
    assert payload["dataset_artefact"] == f"{payload['dataset_name']}.parquet"
    assert payload["model"]["algorithm"] == "lightgbm"
    assert payload["model"]["horizon_minutes"] == 5
    assert payload["model"]["classes"] == list(CLASSES)
    assert payload["model"]["policy"]["random_seed"] == 42
    assert payload["model"]["policy"]["early_stopping"] is False
    assert payload["model"]["importance"]
    assert set(payload["baseline_aggregates"]) == set(BASELINE_NAMES)
    assert payload["aggregates"]["log_loss"]["count"] == 2
    assert payload["trained_folds"] == 2 and payload["skipped_folds"] == 0
    assert payload["holdout"] is None
    assert payload["labels_timeframe_caveat"].startswith("stored label rows carry no timeframe")
    assert any("too few for a spread" in note for note in payload["notes"])



def test_the_fold_metrics_are_on_the_records_the_split_documents(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    payload = metadata(tmp_path)
    for fold in payload["folds"]:
        assert fold["status"] == "TRAINED"
        comparison = fold["comparison"]
        assert comparison["rows"] == fold["fold"]["validate_rows"] == VALIDATE_ROWS
        assert comparison["rows"] >= 30  # preliminary_min_samples
        assert comparison["label"] == "MEASURED"
        assert comparison["model"]["log_loss"] is not None
        assert set(comparison["baselines"]) == set(BASELINE_NAMES)
        for name, metrics in comparison["baselines"].items():
            assert metrics["rows"] == comparison["rows"]
            delta = comparison["log_loss_delta_vs_baseline"][name]
            assert delta == pytest.approx(metrics["log_loss"] - comparison["model"]["log_loss"])
        assert comparison["best_baseline"] in BASELINE_NAMES
        assert comparison["beats_all_baselines"] == (
            len(comparison["beats_baselines"]) == len(comparison["baselines"])
        )


def test_the_features_the_model_used_are_the_datasets_features(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    payload = metadata(tmp_path)
    features = payload["model"]["feature_columns"]
    assert len(features) == payload["dataset"]["features"]
    assert set(payload["model"]["categorical_columns"]) <= set(features)
    assert set(payload["model"]["importance"]) == set(features)
    # Nothing label-derived can be a feature, and the trainer would have refused if it were.
    assert not [name for name in features if name.startswith("future_")]
    assert "label" not in features and "label_code" not in features


def test_the_holdout_is_scored_only_when_asked(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    assert metadata(tmp_path)["holdout"] is None

    run_train(tmp_path, config_dir, "--horizon", "5", "--force", "--score-holdout")
    payload = metadata(tmp_path)
    assert payload["holdout"]["label"] == "HOLDOUT"
    assert payload["holdout"]["rows"] == VALIDATE_ROWS
    assert any("no longer untouched" in note for note in payload["notes"])
    # The final model is fitted on everything *outside* the holdout either way.
    assert payload["final_train_rows"] < payload["dataset"]["rows"]


def test_the_seed_flag_changes_the_model_and_is_recorded(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force", "--seed", "7")
    assert metadata(tmp_path)["model"]["policy"]["random_seed"] == 7


def test_no_write_trains_and_reports_nothing_on_disk(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5", "--force", "--no-write")
    assert exit_code == EXIT_OK
    assert "horizon 5m : SUCCESS" in capsys.readouterr().out
    assert not models_dir(tmp_path).exists() or not list(models_dir(tmp_path).iterdir())


def test_horizon_flags_narrow_the_output(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "15", "--force")
    assert boosters(tmp_path, horizon=15)
    assert not boosters(tmp_path, horizon=5)


def test_a_horizon_nobody_configured_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "7", "--force")
    assert exit_code == EXIT_CONFIG_ERROR
    assert "model.horizons_minutes" in capsys.readouterr().err



def test_an_unimplemented_algorithm_is_a_configuration_error(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    model = yaml.safe_load((config_dir / "model.yaml").read_text(encoding="utf-8"))
    model["algorithm"] = "xgboost"
    (config_dir / "model.yaml").write_text(
        yaml.safe_dump(model, sort_keys=False), encoding="utf-8"
    )
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    assert exit_code == EXIT_CONFIG_ERROR
    assert "not implemented" in capsys.readouterr().err


def test_a_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5", "--dry-run")
    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert "train (dry run)" in out
    assert "would write" in out
    assert "horizon 5m" in out
    assert not models_dir(tmp_path).exists() or not list(models_dir(tmp_path).iterdir())


def test_json_output_is_machine_readable(tmp_path, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_train(tmp_path, config_dir, "--horizon", "5", "--force", "--json")
    assert exit_code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "train"
    assert payload["training"]["random_seed"] == 42
    assert payload["baselines"] == list(BASELINE_NAMES)
    assert payload["verdict"]["results_label"] == "BELOW_GATE"
    horizon = payload["horizons"][0]
    assert horizon["horizon_minutes"] == 5
    assert horizon["status"] == "SUCCESS"
    assert horizon["report"]["trained_folds"] == 2
    assert horizon["report"]["aggregates"]["log_loss"]["mean"] > 0.0
    assert horizon["report"]["final_model"]["train_rows"] > 0


def test_the_feature_and_label_tables_are_not_written(tmp_path) -> None:
    config_dir = seed(tmp_path)
    features_before = query(tmp_path, "SELECT count(*) FROM features")[0][0]
    labels_before = query(tmp_path, "SELECT count(*) FROM labels")[0][0]
    run_train(tmp_path, config_dir, "--horizon", "5", "--force", "--score-holdout")
    assert query(tmp_path, "SELECT count(*) FROM features")[0][0] == features_before
    assert query(tmp_path, "SELECT count(*) FROM labels")[0][0] == labels_before


def test_a_second_run_replaces_the_artefacts_instead_of_piling_up(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    first = {path.name for path in models_dir(tmp_path).iterdir()}
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    assert {path.name for path in models_dir(tmp_path).iterdir()} == first


def test_no_feature_rows_reports_no_data_instead_of_training(tmp_path, capsys) -> None:
    from core.storage import bootstrap_database

    config_dir = shrink_config(tmp_path, walk_forward=ONE_DAY_FOLDS, training=FAST_TRAINING)
    bootstrap_database(db_file(tmp_path), schema_version=1)
    exit_code = run_train(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    assert "no stored feature rows" in capsys.readouterr().err
    assert not models_dir(tmp_path).exists() or not list(models_dir(tmp_path).iterdir())


def test_training_never_writes_a_dataset_artefact(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force")
    datasets = tmp_path / "data" / "datasets"
    assert not datasets.exists() or not list(datasets.iterdir())


def test_the_run_manifest_records_the_models_and_the_verdict(tmp_path) -> None:
    config_dir = seed(tmp_path)
    run_train(tmp_path, config_dir, "--horizon", "5", "--force", run_artifacts=True)
    manifests = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert payload["mode"] == "train"
    assert any("_lightgbm.txt" in artifact for artifact in payload["artifacts"])
    assert payload["data_coverage"]["results_label"] == "BELOW_GATE"
    assert payload["data_coverage"]["horizons"]["5"]["trained_folds"] == 2

