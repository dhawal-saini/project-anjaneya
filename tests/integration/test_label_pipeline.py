"""End-to-end ANJANEYA-ENGINE label pipeline (Stage 3.4b, spec 20).

A session's ticks are seeded through the real schema, ``build-candles`` runs through
the real ``main()`` and then ``build-labels`` runs - also through ``main()`` - so every
assertion is made against what actually landed on disk:

    candles        ->  label rows, one per (timestamp, contract, horizon)
    data_coverage  ->  the label verdict *next to* the candle verdict of the same day
    Parquet        ->  one file per day under ``data/labels/``, no ``.tmp`` left
    features       ->  untouched: the label pass never reads or writes a feature row

The two passes are deliberately separate (spec 20): a label is a function of the
candles *after* a moment, a feature of the candles *up to* it. These tests pin that
separation down at the level an operator sees it - the files and the tables.

Nothing here touches the network: the only input is stored ticks.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from apps.anjaneeya_engine.run import main as engine_main
from core.app_bootstrap import EXIT_OK
from core.storage import connect
from tests.support.engine import (
    CE_SYMBOL,
    PE_SYMBOL,
    SESSION_START,
    cli_args,
    make_feature,
    seed_features,
    seed_ticks,
    session_ticks,
)

#: Observed minutes (09:15-10:14) - 60 one-minute bars per contract. Enough for the
#: ATR(14) warm-up, the 5-bar prior move *and* a 30-minute forward window.
MINUTES = tuple(range(0, 60))
#: Bars per contract the candle build produces at 1m.
BARS = len(MINUTES)
#: The horizons ``config/model.yaml`` ships, and the rows they imply per contract.
HORIZONS = (5, 10, 15, 30)
ROWS_PER_CONTRACT = BARS * len(HORIZONS)
#: ``data_coverage`` is keyed (day, tradingsymbol, timeframe), so the label stream - which
#: is per *contract* like candles - spells its own timeframe rather than overwriting the
#: candle verdict (the tick stream does the same with ``"TICK"``).
LABEL_TIMEFRAME = "1m:labels"
#: The bars of a 09:15-15:30 session at 1m, the label stream's ``expected_rows`` divisor.
SESSION_BARS = 375


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
    """A database with an hour of two contracts' stored candles on the test day."""
    seed_ticks(db_file(tmp_path), session_ticks(minutes=MINUTES))
    assert engine_main(cli_args(tmp_path)) == EXIT_OK  # build-candles
    return tmp_path


def null_counts(seeded) -> dict[int, int]:
    """Unlabelled row count per horizon (both contracts), straight from the table."""
    rows = query(
        seeded,
        "SELECT horizon_minutes, count(*) FROM labels WHERE label IS NULL "
        "GROUP BY 1 ORDER BY 1",
    )
    return {int(horizon): int(count) for horizon, count in rows}


# ---------------------------------------------------------------------------
# The happy path: one row per (bar, horizon), whatever the label
# ---------------------------------------------------------------------------
def stamp(minutes: int) -> str:
    """SQL timestamp literal for ``minutes`` after the session open."""
    return (SESSION_START + timedelta(minutes=minutes)).isoformat()


def test_build_labels_writes_one_row_per_bar_and_horizon(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-labels"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "ANJANEYA-ENGINE build-labels" in output
    assert "horizons 5m, 10m, 15m, 30m" in output
    assert "contracts      : 2" in output

    shape = query(
        seeded,
        "SELECT tradingsymbol, horizon_minutes, count(*) FROM labels "
        "GROUP BY 1, 2 ORDER BY 1, 2",
    )
    assert shape == [
        (symbol, horizon, BARS)
        for symbol in sorted((CE_SYMBOL, PE_SYMBOL))
        for horizon in HORIZONS
    ]


def test_the_stored_rows_are_the_contract_and_the_run(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels", run_artifacts=True))

    assert query(seeded, "SELECT DISTINCT label_version, source FROM labels") == [
        ("v1", "DERIVED")
    ]


def test_a_bar_inside_the_window_carries_the_next_bars_move(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels"))

    # ``future_return`` is relative, and it is read from the *stored* candles rather than
    # recomputed from anything the label pass produced - so the two tables have to agree.
    rows = query(
        seeded,
        f"""
        WITH c AS (
            SELECT timestamp, close FROM candles
            WHERE timeframe = '1m' AND tradingsymbol = '{CE_SYMBOL}'
        )
        SELECT l.future_return, c2.close / c.close - 1.0
        FROM labels l
        JOIN c ON c.timestamp = l.timestamp
        JOIN c c2 ON c2.timestamp = l.timestamp + INTERVAL 5 MINUTE
        WHERE l.horizon_minutes = 5 AND l.tradingsymbol = '{CE_SYMBOL}'
        """,
    )
    assert len(rows) == BARS - 5  # every bar but the last five has a 5-minute window
    assert all(stored == pytest.approx(expected) for stored, expected in rows)


def test_the_excursions_are_signed_and_never_invented(seeded) -> None:
    """An up move floors at zero and a down move caps at zero - in *price* units."""
    engine_main(cli_args(seeded, mode="build-labels"))

    assert query(
        seeded,
        "SELECT count(*) FROM labels WHERE future_max_up_move < 0 "
        "OR future_max_down_move > 0",
    ) == [(0,)]
    # The seeded session only rises, so nothing ever went against the bar's close.
    assert query(
        seeded,
        "SELECT DISTINCT future_max_down_move FROM labels "
        "WHERE future_max_down_move IS NOT NULL",
    ) == [(0.0,)]


def test_a_clearing_move_with_a_rising_prior_is_a_continuation(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels"))

    assert query(
        seeded, "SELECT DISTINCT label FROM labels WHERE label IS NOT NULL ORDER BY 1"
    ) == [("UP_CONTINUATION",)]
    assert query(
        seeded, "SELECT count(*) FROM labels WHERE future_trend_change IS NOT NULL"
    ) == query(seeded, "SELECT count(*) FROM labels WHERE label IS NOT NULL")


# ---------------------------------------------------------------------------
# What a label cannot be
# ---------------------------------------------------------------------------
def test_a_window_that_leaves_the_session_leaves_the_row_unlabelled(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels"))

    # The last five bars cannot hold a 5-minute window, and the 30-minute horizon reaches
    # back thirty bars; the row is still there, saying "no label here".
    for horizon, minutes in ((5, 59), (30, 45)):
        row = query(
            seeded,
            "SELECT label, future_return, future_max_up_move, future_trend_change FROM labels "
            f"WHERE tradingsymbol = '{CE_SYMBOL}' AND horizon_minutes = {horizon} "
            f"AND timestamp = TIMESTAMPTZ '{stamp(minutes)}'",
        )
        assert row == [(None, None, None, None)]

    assert query(
        seeded,
        "SELECT label IS NOT NULL FROM labels WHERE tradingsymbol = "
        f"'{CE_SYMBOL}' AND horizon_minutes = 30 AND timestamp = TIMESTAMPTZ '{stamp(29)}'",
    ) == [(True,)]


def test_the_warm_up_bars_have_numbers_but_no_class(seeded) -> None:
    """ATR(14) - the threshold - does not exist yet, and neither does the prior move."""
    engine_main(cli_args(seeded, mode="build-labels"))

    row = query(
        seeded,
        "SELECT label, future_return FROM labels WHERE tradingsymbol = "
        f"'{CE_SYMBOL}' AND horizon_minutes = 5 AND timestamp = TIMESTAMPTZ '{stamp(0)}'",
    )
    assert row[0][0] is None  # no class: the threshold is not known 14 bars in
    assert row[0][1] is not None  # but the forward window is real, so the number is

    # 13 ATR warm-up bars + the horizon's own tail, for each of the two contracts.
    assert null_counts(seeded) == {
        5: 2 * (13 + 5),
        10: 2 * (13 + 10),
        15: 2 * (13 + 15),
        30: 2 * (13 + 30),
    }


# ---------------------------------------------------------------------------
# Coverage, the mirror, and the pass that must not happen
# ---------------------------------------------------------------------------
def test_the_label_verdict_sits_next_to_the_candle_verdict(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels"))

    rows = query(
        seeded,
        "SELECT timeframe, expected_rows, actual_rows FROM data_coverage "
        f"WHERE tradingsymbol = '{CE_SYMBOL}' AND timeframe IN ('1m', '{LABEL_TIMEFRAME}') "
        "ORDER BY timeframe",
    )
    # The candle verdict is still the candle verdict: 60 bars of a 375-bar session.
    assert rows[0] == ("1m", SESSION_BARS, BARS)
    # And the label verdict describes its own stream: bars x horizons.
    assert rows[1] == (LABEL_TIMEFRAME, SESSION_BARS * len(HORIZONS), ROWS_PER_CONTRACT)


def test_the_rows_are_mirrored_to_parquet(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels"))

    target = seeded / "data" / "labels" / "2026-09-29.parquet"
    assert target.exists()
    assert list((seeded / "data" / "labels").rglob("*.tmp")) == []
    assert query(seeded, f"SELECT count(*) FROM read_parquet('{target.as_posix()}')") == [
        (2 * ROWS_PER_CONTRACT,)
    ]


def test_the_feature_rows_are_never_touched(seeded) -> None:
    """Spec 20's separation, checked on the tables: labels do not read or rewrite features."""
    seed_features(db_file(seeded), [make_feature()])
    before = query(seeded, "SELECT tradingsymbol, option_atr, quality_flag FROM features")

    assert engine_main(cli_args(seeded, mode="build-labels")) == EXIT_OK

    assert query(seeded, "SELECT tradingsymbol, option_atr, quality_flag FROM features") == before
    assert query(seeded, "SELECT count(*) FROM labels") == [(2 * ROWS_PER_CONTRACT,)]


def test_a_second_run_replaces_rather_than_duplicates(seeded) -> None:
    engine_main(cli_args(seeded, mode="build-labels"))
    engine_main(cli_args(seeded, mode="build-labels"))

    assert query(seeded, "SELECT count(*) FROM labels") == [(2 * ROWS_PER_CONTRACT,)]
    assert query(
        seeded,
        "SELECT count(*) FROM (SELECT timestamp, tradingsymbol, horizon_minutes, count(*) n "
        "FROM labels GROUP BY 1, 2, 3 HAVING n > 1)",
    ) == [(0,)]



# ---------------------------------------------------------------------------
# Scoping, dry runs and the run manifest
# ---------------------------------------------------------------------------
def test_one_contract_can_be_named(seeded, capsys) -> None:
    exit_code = engine_main(
        cli_args(seeded, mode="build-labels", extra=["--symbol", CE_SYMBOL])
    )

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "contracts      : 1" in output
    assert query(seeded, "SELECT DISTINCT tradingsymbol FROM labels") == [(CE_SYMBOL,)]


def test_a_dry_run_reports_the_plan_and_writes_nothing(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-labels", extra=["--dry-run"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "build-labels dry-run" in output
    assert f"spelled '{LABEL_TIMEFRAME}'" in output
    assert "warm-up" in output
    assert list((seeded / "data" / "labels").rglob("*.parquet")) == []
    assert query(seeded, "SELECT count(*) FROM labels") == [(0,)]


def test_a_day_without_candles_says_so_instead_of_inventing_rows(tmp_path, capsys) -> None:
    seed_ticks(db_file(tmp_path), session_ticks(minutes=MINUTES))  # ticks, no candles

    exit_code = engine_main(cli_args(tmp_path, mode="build-labels"))

    output = capsys.readouterr().err
    assert exit_code == EXIT_OK
    assert "no stored candles found" in output


def test_the_json_report_is_machine_readable(seeded, capsys) -> None:
    engine_main(cli_args(seeded, mode="build-labels", extra=["--json"]))

    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "build-labels"
    assert report["policy"]["version"] == "v1"
    assert report["policy"]["horizons_minutes"] == list(HORIZONS)
    day = report["days"][0]
    assert day["status"] == "SUCCESS"
    assert day["contracts"] == 2
    assert day["rows"] == 2 * ROWS_PER_CONTRACT
    assert day["expected_rows"] == 2 * SESSION_BARS * len(HORIZONS)
    assert day["failures"] == []
    assert day["parquet"] == ["2026-09-29.parquet"]
    assert len(day["per_contract"]) == 2


def test_the_run_manifest_records_the_label_build(seeded) -> None:
    exit_code = engine_main(cli_args(seeded, mode="build-labels", run_artifacts=True))

    assert exit_code == EXIT_OK
    assert (
        "anjaneeya_engine",
        "build-labels",
        "SUCCESS",
    ) in query(seeded, "SELECT app_name, mode, status FROM run_manifest")

