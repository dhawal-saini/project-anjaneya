"""End-to-end ANJANEYA-ENGINE feature pipeline (Stage 3.2e, specs 18/19).

Ticks are seeded through the real schema, ``build-candles`` runs through the real
``main()``, a NIFTY index series is seeded into ``index_history``, and then
``build-features`` is asserted against what landed on disk:

    candles        ->  feature rows, one per (timestamp, contract)
    data_coverage  ->  one verdict per (day, underlying, timeframe) for the feature stream
    Parquet        ->  one compact file per (timeframe, day), no ``.tmp`` left
    validate       ->  a PASS_WITH_WARNINGS report and its JSON file, per day

Nothing here touches the network: the only inputs are stored ticks and stored index
history.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from apps.anjaneeya_engine.run import main as engine_main
from core.app_bootstrap import EXIT_FAILED, EXIT_OK
from core.constants import INDEX_HISTORY_COLUMNS
from core.storage import connect, insert_rows
from tests.support.engine import (
    CE_SYMBOL,
    PE_SYMBOL,
    SESSION_START,
    WEEKEND,
    cli_args,
    make_index_row,
    seed_index_history,
    seed_ticks,
    session_ticks,
)

#: Observed minutes (09:15-09:20) - enough for one bar per timeframe.
MINUTES = tuple(range(0, 6))
#: A few minutes of synthetic NIFTY index history per timeframe.
INDEX_MINUTES = tuple(range(0, 6))


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def seed_all(tmp_path):
    """Ticks for the session plus the NIFTY index history the feature build needs."""
    seed_ticks(db_file(tmp_path), session_ticks(minutes=MINUTES))
    seed_index_history(db_file(tmp_path), [])
    connection = connect(db_file(tmp_path))
    try:
        rows = [
            make_index_row(
                timestamp=SESSION_START + timedelta(minutes=minute),
                timeframe="1m",
                open=25000.0 + minute,
                high=25002.0 + minute,
                low=24998.0 + minute,
                close=25001.0 + minute,
            )
            for minute in INDEX_MINUTES
        ]
        insert_rows(connection, "index_history", rows, mode="append", columns=INDEX_HISTORY_COLUMNS)
    finally:
        connection.close()
    return tmp_path


@pytest.fixture
def seeded(tmp_path):
    """A database with a session's ticks, its candles and a NIFTY index series."""
    seed_all(tmp_path)
    assert engine_main(cli_args(tmp_path)) == EXIT_OK  # build-candles
    return tmp_path


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_build_features_writes_one_row_per_timestamp_and_contract(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-features"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "build-features" in output
    assert "[SUCCESS] 2 contract(s)" in output
    assert "failures        : 0" in output

    counts = dict(
        query(seeded, "SELECT tradingsymbol, count(*) FROM features GROUP BY 1 ORDER BY 1")
    )
    assert counts == {CE_SYMBOL: 6, PE_SYMBOL: 6}
    assert query(seeded, "SELECT DISTINCT timeframe FROM features") == [("1m",)]
    assert query(seeded, "SELECT DISTINCT underlying FROM features") == [("NIFTY",)]
    assert query(seeded, "SELECT DISTINCT source FROM features") == [("DERIVED",)]


def test_a_feature_row_carries_the_engines_output(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-features"))

    row = query(
        seeded,
        "SELECT candle_body_ratio, candle_direction, option_atr, nifty_regime, "
        "distance_from_atm, time_of_day_bucket, day_of_week, support_strength "
        "FROM features WHERE tradingsymbol = "
        f"'{CE_SYMBOL}' AND timestamp = TIMESTAMPTZ '{SESSION_START.isoformat()}'",
    )

    assert len(row) == 1
    body_ratio, direction, option_atr, regime, distance, bucket, weekday, strength = row[0]
    assert body_ratio == pytest.approx(1.0)  # the minute's ticks all moved one way
    assert direction == "BULLISH"
    # A six-minute session cannot produce a 14-bar ATR: the column is NULL, and the
    # flag beside it says exactly that rather than a fabricated number.
    assert option_atr is None
    assert query(
        seeded, "SELECT DISTINCT option_insufficient_history FROM features"
    ) == [(True,)]
    assert regime is None or isinstance(regime, str)
    assert distance == pytest.approx(25000.0 - 25001.0)
    assert bucket == "OPEN_915_930"
    assert weekday == 1
    # No previous session and no confirmed swing yet: the S/R strength is NULL on the
    # first bar rather than a made-up 0.
    assert strength is None or 0.0 <= strength <= 1.0


def test_a_second_run_replaces_rows_instead_of_duplicating_them(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-features"))
    engine_main(cli_args(seeded, mode="build-features"))

    assert query(seeded, "SELECT count(*) FROM features") == [(12,)]
    assert query(seeded, "SELECT count(*) FROM data_coverage") == [(9,)]  # 8 candles + 1 feature row


def test_the_feature_stream_reports_its_own_coverage_row(seeded) -> None:
    """The feature verdict is keyed by the underlying so it cannot overwrite candles."""
    engine_main(cli_args(seeded, mode="build-features"))

    rows = query(
        seeded,
        "SELECT tradingsymbol, timeframe, expected_rows, actual_rows, quality_flag, note "
        "FROM data_coverage WHERE tradingsymbol = 'NIFTY'",
    )

    assert len(rows) == 1
    symbol, timeframe, expected, actual, flag, note = rows[0]
    assert (symbol, timeframe) == ("NIFTY", "1m")
    assert (expected, actual) == (750, 12)  # 375 bars expected per contract x 2
    assert flag == "GAP_FILLED"
    assert "features over 2 contract(s)" in note


def test_the_rows_are_mirrored_to_parquet(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-features"))

    target = seeded / "data" / "features" / "1m" / "2026-09-29.parquet"
    assert target.exists()
    assert list((seeded / "data" / "features").rglob("*.tmp")) == []
    assert query(seeded, f"SELECT count(*) FROM read_parquet('{target.as_posix()}')") == [(12,)]


def test_only_the_requested_timeframe_is_built(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-features", extra=["--timeframes", "15m"]))

    assert exit_code == EXIT_OK
    assert "at 15m" in capsys.readouterr().out
    assert query(seeded, "SELECT DISTINCT timeframe FROM features") == [("15m",)]


def test_asking_for_two_timeframes_is_a_configuration_error(seeded, capsys) -> None:
    exit_code = engine_main(
        cli_args(seeded, mode="build-features", extra=["--timeframes", "1m,5m"])
    )

    assert exit_code == 2
    assert "exactly" in capsys.readouterr().err


def test_a_dry_run_plans_the_work_and_writes_nothing(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-features", extra=["--dry-run"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "dry-run: build-features -> 1 day(s) at 1m" in output
    assert "2 contract(s)" in output
    assert "NIFTY index bar(s)" in output
    assert query(seeded, "SELECT count(*) FROM features") == [(0,)]


def test_the_json_summary_is_machine_readable(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-features", extra=["--json"]))

    assert exit_code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "build-features"
    assert report["timeframe"] == "1m"
    assert report["rows_written"] == 12
    assert report["coverage_rows"] == 1
    assert report["parquet_files"] == 1
    day = report["days"][0]
    assert day["status"] == "SUCCESS"
    assert day["contracts"] == 2
    assert day["expected_rows"] == 750
    assert day["failures"] == []
def test_a_day_without_candles_says_so_instead_of_inventing_rows(tmp_path, capsys) -> None:
    seed_all(tmp_path)  # ticks + index history, but no candles

    exit_code = engine_main(cli_args(tmp_path, mode="build-features"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "no candles" in output
    assert query(tmp_path, "SELECT count(*) FROM features") == [(0,)]


def test_a_missing_index_series_is_reported_and_only_nulls_the_nifty_block(tmp_path, capsys) -> None:
    seed_ticks(db_file(tmp_path), session_ticks(minutes=MINUTES))
    assert engine_main(cli_args(tmp_path)) == EXIT_OK  # candles only, no index history

    exit_code = engine_main(cli_args(tmp_path, mode="build-features"))

    assert exit_code == EXIT_OK
    assert "no NIFTY index history" in capsys.readouterr().err
    assert query(tmp_path, "SELECT count(*) FROM features WHERE nifty_regime IS NOT NULL") == [(0,)]
    assert query(tmp_path, "SELECT count(*) FROM features WHERE candle_body IS NOT NULL") == [(12,)]


def test_a_weekend_build_is_a_no_session(seeded, capsys) -> None:
    exit_code = engine_main(
        cli_args(seeded, mode="build-features", extra=["--date", WEEKEND.isoformat()])
    )

    assert exit_code == EXIT_OK
    assert "[NO_SESSION]" in capsys.readouterr().out


def test_one_unusable_contract_does_not_lose_the_others(seeded, capsys, monkeypatch) -> None:
    from apps.anjaneeya_engine import run as engine_run
    from core.features import FeatureBuildError

    real = engine_run.build_feature_frame

    def explode(**kwargs):
        if kwargs["candles"]["tradingsymbol"].iloc[0] == PE_SYMBOL:
            raise FeatureBuildError("synthetic: unusable contract")
        return real(**kwargs)

    monkeypatch.setattr(engine_run, "build_feature_frame", explode)

    exit_code = engine_main(cli_args(seeded, mode="build-features"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_FAILED
    assert "[PARTIAL]" in output
    assert "synthetic: unusable contract" in output
    assert query(seeded, "SELECT DISTINCT tradingsymbol FROM features") == [(CE_SYMBOL,)]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def test_validate_reports_pass_with_warnings_and_writes_a_report(seeded, capsys) -> None:
    engine_main(cli_args(seeded, mode="build-features"))
    capsys.readouterr()  # the build summary is not part of this assertion

    exit_code = engine_main(cli_args(seeded, mode="validate", extra=["--probes", "2"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "ANJANEYA-ENGINE validate" in output
    assert "[PASS_WITH_WARNINGS]" in output

    report_path = seeded / "data" / "reports" / "feature_validation" / "2026-09-29_1m.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "validate"
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["row_count"] == 12
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["unique_keys"]["status"] == "PASS"
    assert checks["ohlc_sanity"]["status"] == "PASS"
    assert checks["no_look_ahead"]["status"] == "PASS"
    assert checks["no_look_ahead"]["checked"] >= 1
    assert checks["missing_values"]["status"] == "WARN"


def test_validate_never_writes_to_the_database(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-features"))
    before = query(seeded, "SELECT count(*) FROM features")

    engine_main(cli_args(seeded, mode="validate"))

    assert query(seeded, "SELECT count(*) FROM features") == before


def test_validate_fails_loudly_when_the_stored_rows_are_wrong(seeded, capsys) -> None:
    """Plant an infinite value and the run must exit non-zero, naming the check."""
    engine_main(cli_args(seeded, mode="build-features"))
    capsys.readouterr()
    connection = connect(db_file(seeded))
    try:
        connection.execute("UPDATE features SET option_atr = 'Infinity'::DOUBLE")
    finally:
        connection.close()

    exit_code = engine_main(cli_args(seeded, mode="validate", extra=["--probes", "0"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_FAILED
    assert "[FAIL]" in output
    assert "FAIL infinite_values" in output


def test_validate_of_an_empty_scope_fails_with_a_reason(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="validate", extra=["--probes", "0"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_FAILED
    assert "rows_present" in output


def test_validate_dry_run_reads_only(seeded, capsys) -> None:
    engine_main(cli_args(seeded, mode="build-features"))
    capsys.readouterr()

    exit_code = engine_main(cli_args(seeded, mode="validate", extra=["--dry-run"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "dry-run: validate" in output
    assert not (seeded / "data" / "reports" / "feature_validation").exists()


def test_validate_json_output_is_machine_readable(seeded, capsys) -> None:
    engine_main(cli_args(seeded, mode="build-features"))
    capsys.readouterr()

    exit_code = engine_main(cli_args(seeded, mode="validate", extra=["--json", "--probes", "0"]))

    assert exit_code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "validate"
    assert report["days"][0]["status"] == "PASS_WITH_WARNINGS"


def test_the_run_manifest_records_the_feature_build(seeded) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-features", run_artifacts=True))

    assert exit_code == EXIT_OK
    rows = query(seeded, "SELECT app_name, mode, status FROM run_manifest")
    assert ("anjaneeya_engine", "build-features", "SUCCESS") in rows
