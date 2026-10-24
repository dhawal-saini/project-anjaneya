"""End-to-end ANJANEYA-AI dataset pipeline (Stage 4.1, specs 21-22).

The AI app's first mode is exercised through the real ``main()`` against a seeded database,
so every assertion is made against what actually landed on disk:

    features + labels ->  4 dataset artefacts (one per horizon) + 4 split manifests
    the matrix        ->  feature columns only, a class code per row, labels beside them
    the split         ->  the shipped policy's holdout verdict, or real folds under a
                          shrunken policy, with exact timestamps in the manifest
    the sources       ->  untouched: the dataset pass never writes a feature or label row

Nothing here touches the network: the only input is what is stored in DuckDB.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
import yaml

from apps.anjaneeya_ai.run import main as ai_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_FAILED, EXIT_OK
from core.constants import FEATURE_COLUMNS
from core.storage import connect
from tests.support.ai import (
    BARS,
    CLASSES,
    DAYS,
    HORIZONS,
    cli_args,
    seed_source,
)
from tests.support.engine import CE_SYMBOL, PE_SYMBOL

#: Labelled rows per horizon in the fixture (minutes 0..30 keep a 30-bar window).
LABELLED_PER_CONTRACT_DAY = BARS - max(HORIZONS) + 1
LABELLED_PER_HORIZON = len(DAYS) * 2 * LABELLED_PER_CONTRACT_DAY
#: Labelled rows per horizon for a single contract across every fixture day.
LABELLED_PER_CONTRACT = len(DAYS) * LABELLED_PER_CONTRACT_DAY


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def datasets_dir(tmp_path):
    return tmp_path / "data" / "datasets"


def artifacts(tmp_path, *suffixes: str) -> list:
    """Dataset artefacts on disk, filtered by suffix."""
    directory = datasets_dir(tmp_path)
    if not directory.exists():
        return []
    files = sorted(path for path in directory.iterdir() if path.is_file())
    return [path for path in files if not suffixes or path.name.endswith(suffixes)]


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def seeded(tmp_path, **kwargs):
    """A database with the fixture's feature and label rows, ready for the CLI."""
    seed_source(db_file(tmp_path), **kwargs)
    return tmp_path


def matrix(tmp_path, horizon: int = 5) -> pd.DataFrame:
    """Read one horizon's Parquet artefact back."""
    paths = [path for path in artifacts(tmp_path, ".parquet") if f"_h{horizon}_" in path.name]
    assert len(paths) == 1, [path.name for path in paths]
    return pd.read_parquet(paths[0])


def manifest(tmp_path, horizon: int = 5) -> dict:
    """Read one horizon's split manifest back."""
    paths = [path for path in artifacts(tmp_path, "_splits.json") if f"_h{horizon}_" in path.name]
    assert len(paths) == 1, [path.name for path in paths]
    return json.loads(paths[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The happy path: one matrix and one manifest per horizon
# ---------------------------------------------------------------------------
def test_the_mode_writes_one_matrix_and_manifest_per_horizon(tmp_path, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert len(artifacts(tmp_path, ".parquet")) == len(HORIZONS)
    assert len(artifacts(tmp_path, "_splits.json")) == len(HORIZONS)
    assert not artifacts(tmp_path, ".tmp")
    for horizon in HORIZONS:
        assert f"horizon {horizon}m : SUCCESS" in printed
    assert f"1m_h5_{DAYS[0].isoformat()}_{DAYS[-1].isoformat()}.parquet" in printed


def test_the_matrix_carries_the_features_the_class_and_its_code(tmp_path) -> None:
    seeded(tmp_path)
    ai_main(cli_args(tmp_path))

    frame = matrix(tmp_path, 5)

    assert len(frame) == LABELLED_PER_HORIZON
    assert set(frame["tradingsymbol"]) == {CE_SYMBOL, PE_SYMBOL}
    assert set(frame["label"]) <= set(CLASSES)
    assert list(frame["label_code"].astype(int)) == [
        list(CLASSES).index(str(value)) for value in frame["label"]
    ]
    # The forward numbers travel with the row - and never as features.
    assert "future_return" in frame.columns
    assert not [name for name in frame.columns if name.startswith("future_") and name in FEATURE_COLUMNS]


def test_the_matrix_keeps_the_identity_columns_a_split_needs(tmp_path) -> None:
    seeded(tmp_path)
    ai_main(cli_args(tmp_path))

    frame = matrix(tmp_path, 5)

    for name in ("timestamp", "tradingsymbol", "expiry", "option_type", "timeframe"):
        assert name in frame.columns
    assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
    assert list(frame["timestamp"]) == sorted(frame["timestamp"])
    assert frame["timeframe"].unique().tolist() == ["1m"]


def test_every_horizon_keeps_only_its_own_labels(tmp_path) -> None:
    seeded(tmp_path)
    ai_main(cli_args(tmp_path))

    for horizon in HORIZONS:
        frame = matrix(tmp_path, horizon)
        assert frame["horizon_minutes"].unique().tolist() == [horizon]
        assert len(frame) == LABELLED_PER_HORIZON


def test_the_split_manifest_documents_the_policy_and_the_holdout(tmp_path) -> None:
    seeded(tmp_path)
    ai_main(cli_args(tmp_path))

    payload = manifest(tmp_path, 5)

    assert payload["split"]["policy"]["train_window_days"] == 30
    assert payload["split"]["policy"]["embargo_minutes"] == 30
    assert payload["dataset"]["rows"] == LABELLED_PER_HORIZON
    assert payload["dataset"]["days"] == len(DAYS)
    assert payload["dataset"]["class_counts"]["RANGE"] > 0
    assert payload["feature_columns"] and "label" not in payload["feature_columns"]
    # Four days cannot fill a 30 + 10 + 30 day policy: the manifest says so in words.
    assert payload["split"]["folds"] == 0
    assert "needs 70 captured day(s)" in payload["split"]["reason"]
    assert payload["split"]["holdout"]["days"] == [day.isoformat() for day in DAYS]


def test_the_gate_verdict_is_printed_and_recorded(tmp_path, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "captured day(s)" in printed
    assert "INSUFFICIENT" in printed
    assert "4 captured day(s): too little data" in printed


# ---------------------------------------------------------------------------
# A policy small enough to actually fold
# ---------------------------------------------------------------------------
@pytest.fixture
def foldable_config(tmp_path, config_dir):
    """A copy of the shipped configuration whose walk-forward fits four days."""
    custom = tmp_path / "config"
    shutil.copytree(config_dir, custom)
    model = yaml.safe_load((custom / "model.yaml").read_text(encoding="utf-8"))
    model["walk_forward"].update(
        {
            "train_window_days": 2,
            "validate_window_days": 1,
            "step_days": 1,
            "holdout_days": 1,
            "embargo_minutes": 30,
        }
    )
    (custom / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return custom


def test_a_foldable_policy_produces_real_folds(tmp_path, foldable_config, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--config-dir", str(foldable_config)]))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    payload = manifest(tmp_path, 5)
    assert payload["split"]["folds"] == 1
    first = payload["split"]["splits"][0]
    assert first["train_days"] == [DAYS[0].isoformat(), DAYS[1].isoformat()]
    assert first["validate_days"] == [DAYS[2].isoformat()]
    assert first["train_rows"] > 0 and first["validate_rows"] > 0
    assert first["embargoed_rows"] == 0 and first["purged_rows"] == 0
    assert payload["split"]["holdout"]["days"] == [DAYS[-1].isoformat()]
    assert "fold 0: train" in printed


def test_the_validation_window_is_never_part_of_the_training_window(
    tmp_path, foldable_config
) -> None:
    seeded(tmp_path)
    ai_main(cli_args(tmp_path, extra=["--config-dir", str(foldable_config)]))

    for fold in manifest(tmp_path, 5)["split"]["splits"]:
        assert max(fold["train_days"]) < min(fold["validate_days"])
        assert fold["validate_first_timestamp"] > fold["train_last_timestamp"]


# ---------------------------------------------------------------------------
# Flags: scope, horizons, contracts, dry run, JSON
# ---------------------------------------------------------------------------
def test_horizon_flags_narrow_the_output(tmp_path) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--horizon", "5", "--horizon", "10"]))

    assert exit_code == EXIT_OK
    assert len(artifacts(tmp_path, ".parquet")) == 2
    assert sorted(path.name for path in artifacts(tmp_path, ".parquet")) == sorted(
        f"1m_h{h}_{DAYS[0].isoformat()}_{DAYS[-1].isoformat()}.parquet" for h in (5, 10)
    )


def test_a_horizon_nobody_configured_is_a_configuration_error(tmp_path, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--horizon", "7"]))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "not in model.horizons_minutes" in capsys.readouterr().err


def test_a_timeframe_that_is_not_enabled_must_be_enabled_in_the_config(
    tmp_path, config_dir, capsys
) -> None:
    custom = tmp_path / "config"
    shutil.copytree(config_dir, custom)
    settings = yaml.safe_load((custom / "settings.yaml").read_text(encoding="utf-8"))
    settings["market"]["timeframes"] = ["1m"]
    (custom / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    seeded(tmp_path)

    exit_code = ai_main(
        cli_args(tmp_path, extra=["--config-dir", str(custom), "--timeframe", "5m"])
    )

    assert exit_code == EXIT_CONFIG_ERROR
    assert "not enabled" in capsys.readouterr().err


def test_the_from_to_window_narrows_the_matrix(tmp_path) -> None:
    seeded(tmp_path)
    day = DAYS[1]

    exit_code = ai_main(
        cli_args(tmp_path, extra=["--from", day.isoformat(), "--to", day.isoformat()])
    )

    assert exit_code == EXIT_OK
    frame = matrix(tmp_path, 5)
    assert len(frame) == LABELLED_PER_CONTRACT_DAY * 2
    assert frame["timestamp"].dt.date.unique().tolist() == [day]
    assert f"1m_h5_{day.isoformat()}_{day.isoformat()}.parquet" in {
        path.name for path in artifacts(tmp_path, ".parquet")
    }


def test_a_bad_date_is_a_configuration_error(tmp_path, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--from", "28/09/2026"]))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "--from" in capsys.readouterr().err


def test_a_symbol_flag_narrows_the_matrix(tmp_path) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--symbol", PE_SYMBOL, "--horizon", "5"]))

    assert exit_code == EXIT_OK
    frame = matrix(tmp_path, 5)
    assert frame["tradingsymbol"].unique().tolist() == [PE_SYMBOL]
    assert len(frame) == LABELLED_PER_CONTRACT


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--dry-run"]))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "dry-run" in printed
    assert "split policy" in printed
    assert "would write" in printed
    assert artifacts(tmp_path) == []
    assert query(tmp_path, "SELECT count(*) FROM features")[0][0] > 0


def test_json_output_is_machine_readable(tmp_path, capsys) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--json", "--horizon", "5"]))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == EXIT_OK
    assert payload["mode"] == "dataset"
    assert payload["policy"]["timeframe"] == "1m"
    assert payload["gate"]["days"] == len(DAYS)
    assert payload["artifacts"]["datasets"] == 1
    assert payload["horizons"][0]["horizon_minutes"] == 5
    assert payload["horizons"][0]["dataset"]["rows"] == LABELLED_PER_HORIZON
    assert payload["horizons"][0]["split"]["folds"] == 0


# ---------------------------------------------------------------------------
# The sources stay untouched, the run is reproducible, failures are reported
# ---------------------------------------------------------------------------
def test_the_feature_and_label_tables_are_not_written(tmp_path) -> None:
    seeded(tmp_path)
    before = (
        query(tmp_path, "SELECT count(*) FROM features")[0][0],
        query(tmp_path, "SELECT count(*) FROM labels")[0][0],
        query(tmp_path, "SELECT count(*) FROM candles")[0][0],
    )

    ai_main(cli_args(tmp_path))

    after = (
        query(tmp_path, "SELECT count(*) FROM features")[0][0],
        query(tmp_path, "SELECT count(*) FROM labels")[0][0],
        query(tmp_path, "SELECT count(*) FROM candles")[0][0],
    )
    assert before == after
    # ... and no other table gained a row either.
    assert query(tmp_path, "SELECT count(*) FROM trades")[0][0] == 0


def test_a_second_run_replaces_the_artefacts_instead_of_piling_up(tmp_path) -> None:
    seeded(tmp_path)

    ai_main(cli_args(tmp_path))
    first = sorted(path.name for path in artifacts(tmp_path))
    rows = len(matrix(tmp_path, 5))
    ai_main(cli_args(tmp_path))

    assert sorted(path.name for path in artifacts(tmp_path)) == first
    assert len(matrix(tmp_path, 5)) == rows


def test_the_run_manifest_records_the_artefacts_and_the_gate(tmp_path) -> None:
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, run_artifacts=True))

    assert exit_code == EXIT_OK
    runs = [
        path
        for path in (tmp_path / "data" / "runs").iterdir()
        if (path / "manifest.json").exists()
    ]
    payload = json.loads((runs[-1] / "manifest.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "dataset"
    assert payload["status"] == "SUCCESS"
    assert any(name.endswith(".parquet") for name in payload["artifacts"])
    assert payload["data_coverage"]["days"] == len(DAYS)
    assert payload["data_coverage"]["stage"] == "INSUFFICIENT"
    assert any("data gate" in note for note in payload["notes"])


def test_no_feature_rows_reports_no_data(tmp_path, capsys) -> None:
    from core.storage import bootstrap_database

    bootstrap_database(db_file(tmp_path), schema_version=1)
    capsys.readouterr()

    exit_code = ai_main(cli_args(tmp_path))

    assert exit_code == EXIT_OK
    assert "no stored feature rows" in capsys.readouterr().err
    assert artifacts(tmp_path) == []


def test_every_run_states_the_labels_timeframe_caveat(tmp_path, capsys) -> None:
    """The labels table has no timeframe column, so the assumption is printed, not implied."""
    seeded(tmp_path)

    exit_code = ai_main(cli_args(tmp_path, extra=["--horizon", "5"]))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "carry no timeframe column" in printed
    assert "built at 1m" in printed


def test_a_label_row_with_no_feature_row_is_reported(tmp_path, capsys) -> None:
    """Both directions of an unmatched join are published - this is the label-side one."""
    seeded(tmp_path)
    connection = connect(db_file(tmp_path))
    try:
        connection.execute(
            """
            INSERT INTO labels
            SELECT timestamp + INTERVAL 70 MINUTE, tradingsymbol, instrument_token, expiry,
                   strike, option_type, horizon_minutes, future_return, future_max_up_move,
                   future_max_down_move, future_trend_change, label, label_version, source,
                   fetch_timestamp, quality_flag, run_id
            FROM labels WHERE horizon_minutes = 5 LIMIT 1
            """
        )
    finally:
        connection.close()

    exit_code = ai_main(cli_args(tmp_path, extra=["--horizon", "5"]))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "1 label row(s) without a feature" in printed
    assert len(matrix(tmp_path, 5)) == LABELLED_PER_HORIZON


def test_the_primary_key_stops_a_label_being_stored_twice(tmp_path) -> None:
    """A real duplicate is a storage error, so the builder can only ever see one row."""
    seeded(tmp_path)
    connection = connect(db_file(tmp_path))
    try:
        row = connection.execute(
            "SELECT timestamp, tradingsymbol, horizon_minutes FROM labels LIMIT 1"
        ).fetchone()
        with pytest.raises(Exception) as excinfo:
            connection.execute(
                """
                INSERT INTO labels
                SELECT timestamp, tradingsymbol, instrument_token, expiry, strike, option_type,
                       horizon_minutes, future_return, future_max_up_move, future_max_down_move,
                       future_trend_change, label, label_version, source, fetch_timestamp,
                       quality_flag, run_id
                FROM labels WHERE timestamp = ? AND tradingsymbol = ? AND horizon_minutes = ?
                """,
                [row[0], row[1], row[2]],
            )
        assert "Duplicate key" in str(excinfo.value)
    finally:
        connection.close()


def test_a_horizon_that_cannot_be_built_exits_non_zero(tmp_path, capsys, monkeypatch) -> None:
    """A dataset failure is reported per horizon, written nowhere, and exits non-zero."""
    import apps.anjaneeya_ai.run as ai_run
    from core.ml import DatasetError

    def failing(*args, **kwargs):
        raise DatasetError("simulated: the stored rows cannot be joined")

    seeded(tmp_path)
    monkeypatch.setattr(ai_run, "build_dataset", failing)

    exit_code = ai_main(cli_args(tmp_path, extra=["--horizon", "5"]))
    captured = capsys.readouterr()

    assert exit_code == EXIT_FAILED
    assert "horizon 5m : FAILED" in captured.out
    assert "simulated" in captured.out
    assert artifacts(tmp_path) == []


def test_an_empty_result_is_not_an_error(tmp_path, capsys) -> None:
    """A day with features but no labels at all is reported, not treated as a crash."""
    seeded(tmp_path)
    connection = connect(db_file(tmp_path))
    try:
        connection.execute("DELETE FROM labels")
    finally:
        connection.close()

    exit_code = ai_main(cli_args(tmp_path, extra=["--horizon", "5"]))
    printed = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "horizon 5m : NO_ROWS" in printed
    assert "0 labelled row(s)" in printed




