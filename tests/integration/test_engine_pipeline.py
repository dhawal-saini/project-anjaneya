"""End-to-end ANJANEYA-ENGINE candle-pipeline test.

Ticks are seeded through the *real* schema and writers, the app is driven through
its *real* ``main()``, and every assertion is made against what landed on disk:

    DuckDB candles  ->  1m/3m/5m/15m bars, open-time keyed, no duplicates
    data_coverage   ->  one verdict per (day, contract, timeframe)
    Parquet         ->  one compact file per timeframe per day, no ``.tmp`` left
    run_manifest    ->  the run's verdict, written after the engine released the
                        database (DuckDB allows one writer at a time)

Nothing here touches the network: the only input is stored ticks.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta

import pytest
import yaml

from apps.anjaneeya_engine.run import main as engine_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_FAILED, EXIT_OK
from core.storage import connect
from tests.support.engine import (
    CE_SYMBOL,
    DAY,
    IST,
    PE_SYMBOL,
    SESSION_END,
    SESSION_START,
    WEEKEND,
    cli_args,
    make_tick_row,
    seed_ticks,
    session_ticks,
    ticks_for_minutes,
)

#: Six observed minutes (09:15-09:20) - enough for every timeframe to split.
MINUTES = tuple(range(0, 6))
#: Bars a complete 6-minute session produces for two contracts.
EXPECTED_BARS = {"1m": 12, "3m": 4, "5m": 4, "15m": 2}


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def query(tmp_path, sql: str):
    """Run one read-only query through a short-lived connection."""
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


@pytest.fixture
def seeded(tmp_path):
    """A database holding a six-minute, two-contract session on :data:`DAY`."""
    seed_ticks(db_file(tmp_path), session_ticks(minutes=MINUTES))
    return tmp_path


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_build_candles_writes_every_timeframe_and_exits_zero(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "build-candles" in output
    assert "[SUCCESS] 2 contract(s)" in output
    assert "failures        : 0" in output

    counts = dict(query(seeded, "SELECT timeframe, count(*) FROM candles GROUP BY 1"))
    assert counts == EXPECTED_BARS


def test_the_first_bar_carries_the_minutes_ohlcv(seeded) -> None:
    engine_main(cli_args(seeded))

    row = query(
        seeded,
        "SELECT open, high, low, close, volume, source, quality_flag FROM candles "
        f"WHERE timeframe = '1m' AND tradingsymbol = '{CE_SYMBOL}' "
        f"AND timestamp = TIMESTAMPTZ '{SESSION_START.isoformat()}'",
    )

    assert row == [(120.5, 121.0, 120.5, 121.0, 10, "DERIVED", "OK")]


def test_a_second_run_replaces_rows_instead_of_duplicating_them(seeded) -> None:
    engine_main(cli_args(seeded))
    engine_main(cli_args(seeded))

    counts = dict(query(seeded, "SELECT timeframe, count(*) FROM candles GROUP BY 1"))
    assert counts == EXPECTED_BARS
    assert query(seeded, "SELECT count(*) FROM data_coverage") == [(8,)]
    assert query(
        seeded,
        "SELECT count(*) FROM (SELECT timestamp, timeframe, tradingsymbol, count(*) n "
        "FROM candles GROUP BY 1, 2, 3 HAVING n > 1)",
    ) == [(0,)]


def test_coverage_records_the_session_expectation_per_stream(seeded) -> None:
    engine_main(cli_args(seeded))

    one_minute = query(
        seeded,
        "SELECT expected_rows, actual_rows, gap_count, coverage_ratio, quality_flag "
        f"FROM data_coverage WHERE timeframe = '1m' AND tradingsymbol = '{CE_SYMBOL}'",
    )
    assert len(one_minute) == 1
    expected_rows, actual_rows, gap_count, coverage_ratio, quality_flag = one_minute[0]
    assert (expected_rows, actual_rows, gap_count) == (375, 6, 1)
    assert coverage_ratio == pytest.approx(6 / 375)
    assert quality_flag == "GAP_FILLED"

    fifteen = query(
        seeded, "SELECT expected_rows, actual_rows FROM data_coverage WHERE timeframe = '15m'"
    )
    assert fifteen == [(25, 1), (25, 1)]


def test_the_parquet_mirror_matches_the_stored_candles(seeded) -> None:
    engine_main(cli_args(seeded))

    target = seeded / "data" / "candles" / "1m" / "2026-09-29.parquet"
    assert target.exists()
    assert query(seeded, f"SELECT count(*) FROM read_parquet('{target.as_posix()}')") == [(12,)]
    assert sorted(path.name for path in (seeded / "data" / "candles").glob("*/*")) == [
        "2026-09-29.parquet",
        "2026-09-29.parquet",
        "2026-09-29.parquet",
        "2026-09-29.parquet",
    ]
    assert list((seeded / "data" / "candles").rglob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Honesty: what the engine refuses to build
# ---------------------------------------------------------------------------
def test_ticks_outside_the_session_never_reach_a_candle(seeded) -> None:
    seed_ticks(
        db_file(seeded),
        [
            make_tick_row(timestamp=SESSION_START.replace(hour=9, minute=10), last_price=900.0),
            make_tick_row(timestamp=SESSION_END, last_price=950.0),
            make_tick_row(timestamp=datetime(2026, 9, 27, 9, 20, tzinfo=IST), last_price=970.0),
        ],
    )

    engine_main(cli_args(seeded, extra=["--all"]))

    assert query(seeded, "SELECT count(*) FROM candles WHERE high > 500") == [(0,)]
    assert dict(query(seeded, "SELECT timeframe, count(*) FROM candles GROUP BY 1")) == EXPECTED_BARS


def test_a_contract_with_only_flagged_ticks_is_covered_not_built(tmp_path, capsys) -> None:
    seed_ticks(
        db_file(tmp_path),
        [
            *(make_tick_row(**tick) for tick in ticks_for_minutes(MINUTES)),
            *(
                make_tick_row(**dict(tick, quality_flag="SUSPECT"))
                for tick in ticks_for_minutes(
                    MINUTES, symbol=PE_SYMBOL, token=900_002, option_type="PE"
                )
            ),
        ],
    )

    exit_code = engine_main(cli_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "excluded    : SUSPECT=12 tick(s)" in output
    assert "empty       : 1 contract(s) had no usable tick" in output
    assert query(tmp_path, "SELECT DISTINCT tradingsymbol FROM candles") == [(CE_SYMBOL,)]
    assert query(
        tmp_path,
        f"SELECT count(*) FROM data_coverage WHERE tradingsymbol = '{PE_SYMBOL}' "
        "AND quality_flag = 'MISSING_TICK'",
    ) == [(4,)]


def test_a_flagged_contract_is_built_when_the_operator_asks_for_that_flag(tmp_path, capsys) -> None:
    """2026-10-27 on this machine: 93% of ticks were SUSPECT - not because they were
    bad, but because the *machine* clock sat behind the exchange
    (``exchange_time - received_at > 2s``). The tick still carries its exchange
    timestamp, so an operator must be able to admit those flags explicitly, and the
    run must say which flags it used.
    """
    seed_ticks(
        db_file(tmp_path),
        [
            *(make_tick_row(**tick) for tick in ticks_for_minutes(MINUTES)),
            *(
                make_tick_row(**dict(tick, quality_flag="SUSPECT"))
                for tick in ticks_for_minutes(
                    MINUTES, symbol=PE_SYMBOL, token=900_002, option_type="PE"
                )
            ),
        ],
    )

    exit_code = engine_main(cli_args(tmp_path, extra=["--quality", "OK,SUSPECT"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "tick quality    : OK,SUSPECT" in output
    assert "excluded" not in output
    assert query(tmp_path, "SELECT DISTINCT tradingsymbol FROM candles ORDER BY 1") == [
        (CE_SYMBOL,),
        (PE_SYMBOL,),
    ]
    assert query(
        tmp_path,
        f"SELECT count(*) FROM data_coverage WHERE tradingsymbol = '{PE_SYMBOL}' "
        "AND quality_flag = 'MISSING_TICK'",
    ) == [(0,)]


def test_an_unknown_quality_flag_is_refused_before_anything_is_built(tmp_path) -> None:
    assert engine_main(cli_args(tmp_path, extra=["--quality", "OK,BANANA"])) == EXIT_CONFIG_ERROR
    assert query(tmp_path, "SELECT count(*) FROM candles") == [(0,)]


def test_a_weekend_date_is_reported_as_no_session_and_writes_nothing(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, extra=["--date", WEEKEND.isoformat()]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "[NO_SESSION]" in output
    assert query(seeded, "SELECT count(*) FROM candles") == [(0,)]


def test_a_bad_date_is_a_configuration_error(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, extra=["--date", "29/09/2026"]))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "--date" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["7m", "", "1m,5m,7m"])
def test_an_unsupported_timeframe_is_a_configuration_error(seeded, capsys, value) -> None:
    exit_code = engine_main(cli_args(seeded, extra=["--timeframes", value]))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "configuration error" in capsys.readouterr().err


def test_a_timeframe_that_is_not_enabled_must_be_enabled_in_the_config(tmp_path, config_dir, capsys) -> None:
    """``--timeframes`` may narrow the configured set, never invent one."""
    custom = tmp_path / "config"
    shutil.copytree(config_dir, custom)
    settings = yaml.safe_load((custom / "settings.yaml").read_text(encoding="utf-8"))
    settings["market"]["timeframes"] = ["1m"]
    (custom / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    seed_ticks(db_file(tmp_path), session_ticks(minutes=MINUTES))

    exit_code = engine_main(
        cli_args(tmp_path, extra=["--config-dir", str(custom), "--timeframes", "5m"])
    )

    assert exit_code == EXIT_CONFIG_ERROR
    assert "not enabled" in capsys.readouterr().err


def test_no_stored_ticks_reports_no_data(tmp_path, capsys) -> None:
    assert engine_main(cli_args(tmp_path)) == EXIT_OK  # creates the schema
    capsys.readouterr()

    exit_code = engine_main(cli_args(tmp_path))

    assert exit_code == EXIT_OK
    assert "no stored ticks" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Scoping flags and dry runs
# ---------------------------------------------------------------------------
def test_symbol_and_timeframe_flags_narrow_the_build_without_clobbering(seeded) -> None:
    engine_main(cli_args(seeded))  # a full build first

    exit_code = engine_main(
        cli_args(seeded, extra=["--symbol", CE_SYMBOL, "--timeframes", "1m,5m"])
    )

    assert exit_code == EXIT_OK
    # The narrowed run replaced only that contract's 1m/5m scopes; every other
    # contract and timeframe kept the rows the full build wrote.
    counts = dict(query(seeded, "SELECT timeframe, count(*) FROM candles GROUP BY 1"))
    assert counts == {"1m": 12, "3m": 4, "5m": 4, "15m": 2}
    assert dict(
        query(
            seeded,
            f"SELECT timeframe, count(*) FROM candles WHERE tradingsymbol = '{PE_SYMBOL}' GROUP BY 1",
        )
    ) == {"1m": 6, "3m": 2, "5m": 2, "15m": 1}


def test_all_builds_every_day_that_has_ticks(seeded, capsys) -> None:
    shift = timedelta(days=1)
    seed_ticks(
        db_file(seeded),
        [
            make_tick_row(
                **{
                    **tick,
                    "timestamp": tick["timestamp"] + shift,
                    "fetch_timestamp": tick["fetch_timestamp"] + shift,
                }
            )
            for tick in ticks_for_minutes(MINUTES)
        ],
    )

    exit_code = engine_main(cli_args(seeded, extra=["--all"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "2 day(s)" in output
    assert query(
        seeded,
        "SELECT DISTINCT CAST(timezone('Asia/Kolkata', timestamp) AS DATE) AS day "
        "FROM candles ORDER BY day",
    ) == [(DAY,), (date(2026, 9, 30),)]
    assert (seeded / "data" / "candles" / "15m" / "2026-09-30.parquet").exists()


def test_a_dry_run_plans_the_work_and_writes_nothing(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, extra=["--dry-run"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "dry-run: build-candles -> 1 day(s)" in output
    assert "2 contract(s); a complete session = 1m=375, 3m=125, 5m=75, 15m=25 bar(s)" in output
    assert query(seeded, "SELECT count(*) FROM candles") == [(0,)]
    assert query(seeded, "SELECT count(*) FROM data_coverage") == [(0,)]
    assert not (seeded / "data" / "candles").exists()


def test_a_dry_run_without_a_database_explains_the_next_step(tmp_path, capsys) -> None:
    exit_code = engine_main(cli_args(tmp_path, extra=["--dry-run"]))

    assert exit_code == EXIT_OK
    assert "no DuckDB database" in capsys.readouterr().err


def test_the_json_summary_is_machine_readable(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, extra=["--json"]))

    assert exit_code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "build-candles"
    assert report["timeframes"] == ["1m", "3m", "5m", "15m"]
    assert report["candles_written"] == sum(EXPECTED_BARS.values())
    assert report["coverage_rows"] == 8
    assert report["parquet_files"] == 4
    day = report["days"][0]
    assert day["day"] == DAY.isoformat()
    assert day["status"] == "SUCCESS"
    assert day["bars"] == EXPECTED_BARS
    assert day["coverage"] == {"GAP_FILLED": 8}
    assert day["failures"] == []


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_one_unusable_contract_does_not_lose_the_others(seeded, capsys, monkeypatch) -> None:
    from apps.anjaneeya_engine import run as engine_run
    from apps.anjaneeya_engine.candles import CandleBuildError

    real = engine_run.build_1m_candles

    def explode(frame, *, tradingsymbol, **kwargs):
        if tradingsymbol == PE_SYMBOL:
            raise CandleBuildError("synthetic: unusable contract")
        return real(frame, tradingsymbol=tradingsymbol, **kwargs)

    monkeypatch.setattr(engine_run, "build_1m_candles", explode)

    exit_code = engine_main(cli_args(seeded))

    output = capsys.readouterr().out
    assert exit_code == EXIT_FAILED
    assert "[PARTIAL]" in output
    assert "synthetic: unusable contract" in output
    assert query(seeded, "SELECT DISTINCT tradingsymbol FROM candles") == [(CE_SYMBOL,)]


def test_the_run_manifest_is_written_after_the_database_is_released(seeded) -> None:
    """The engine holds the only writer slot, so a late manifest insert would vanish."""
    exit_code = engine_main(cli_args(seeded, run_artifacts=True))

    assert exit_code == EXIT_OK
    assert query(seeded, "SELECT app_name, mode, status FROM run_manifest") == [
        ("anjaneeya_engine", "build-candles", "SUCCESS")
    ]
    run_dirs = list((seeded / "data" / "runs").glob("anjaneeya_engine_*"))
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS"


