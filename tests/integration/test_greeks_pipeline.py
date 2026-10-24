"""End-to-end IV/greeks enrichment (Stage 3.3, spec 16).

The bars are seeded with closes that are Black-Scholes prices at a known volatility, so
the whole pipeline can be judged against a number nobody had to guess:

    priced candles + NIFTY index -> enrich-greeks -> iv == sigma, deltas signed right
                                 -> build-features -> expected move + residual
                                 -> validate       -> PASS, with the greeks checks run

Nothing here touches the network: the only inputs are stored candles and stored index
history, and nothing is built from ticks (``--mode enrich-greeks`` reads bars).
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from apps.anjaneeya_engine.run import main as engine_main
from core.app_bootstrap import EXIT_CONFIG_ERROR, EXIT_FAILED, EXIT_OK
from core.storage import connect
from tests.support.engine import (
    CE_SYMBOL,
    PE_SYMBOL,
    SESSION_START,
    WEEKEND,
    cli_args,
    priced_window,
    seed_candles,
    seed_index_history,
    seed_ticks,
    session_ticks,
)

#: Bars the fixtures publish (09:15-09:20).
MINUTES = (0, 1, 2, 3, 4, 5)
SIGMA = 0.18


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def seed_day(tmp_path, *, with_index: bool = True, sigma: float = SIGMA):
    """A day of priced option bars for CE and PE, the NIFTY series, and a tick or two.

    The bars are seeded (never built from ticks) so that the IVs have an answer to find.
    The ticks are only there because ``build-features`` picks its contract *universe* from
    the tick table - the one input this stage did not change.
    """
    call = priced_window(minutes=MINUTES, sigma=sigma)
    put = priced_window(
        minutes=MINUTES, symbol=PE_SYMBOL, token=900_002, option_type="PE", sigma=sigma
    )
    seed_ticks(db_file(tmp_path), session_ticks(minutes=(0, 1)))
    seed_candles(db_file(tmp_path), [*call.rows(), *put.rows()])
    seed_index_history(db_file(tmp_path), [])
    if with_index:
        from core.constants import INDEX_HISTORY_COLUMNS
        from core.storage import insert_rows

        # One NIFTY series (both contracts were priced off the same one).
        connection = connect(db_file(tmp_path))
        try:
            insert_rows(
                connection,
                "index_history",
                call.index.to_dict("records"),
                mode="append",
                columns=INDEX_HISTORY_COLUMNS,
            )
        finally:
            connection.close()
    return tmp_path


@pytest.fixture
def seeded(tmp_path):
    """A database with priced candles and the matching NIFTY index series."""
    return seed_day(tmp_path)


def greeks(tmp_path, symbol=CE_SYMBOL):
    """``(iv, delta, gamma, theta, vega, days_to_expiry, distance_from_atm)`` of every bar."""
    return query(
        tmp_path,
        "SELECT timestamp, iv, delta, gamma, theta, vega, days_to_expiry, distance_from_atm "
        f"FROM candles WHERE tradingsymbol = '{symbol}' AND timeframe = '1m' "
        "ORDER BY timestamp",
    )


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_enrich_greeks_solves_the_volatility_the_prices_were_built_from(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "enrich-greeks" in output
    assert "2 contract(s)" in output
    assert "bars updated    : 12 of 12" in output  # two contracts, six 1m bars each

    rows = greeks(seeded)
    assert len(rows) == 6
    for _timestamp, iv, delta, gamma, theta, vega, days, distance in rows:
        assert iv == pytest.approx(SIGMA, abs=1e-6)
        assert 0.0 < delta < 1.0
        assert gamma > 0.0
        assert theta < 0.0
        assert vega > 0.0
        assert days is not None and distance is not None

    # The put is priced off the same level and gets the same volatility, with a put delta.
    put = greeks(seeded, PE_SYMBOL)
    assert all(row[1] == pytest.approx(SIGMA, abs=1e-6) for row in put)
    assert all(row[2] < 0.0 for row in put)


def test_the_summary_reports_the_model_and_the_solution_quality(seeded, capsys) -> None:
    engine_main(cli_args(seeded, mode="enrich-greeks"))

    output = capsys.readouterr().out
    assert "model           : black_scholes, r=0.065, q=0.0, price=close" in output
    assert "iv          : 12 solved, 0 unsolved" in output  # two contracts x six bars
    assert "price error" in output


def test_a_second_run_replaces_the_greeks_and_writes_no_duplicate_rows(seeded) -> None:
    engine_main(cli_args(seeded, mode="enrich-greeks"))
    before = greeks(seeded)
    engine_main(cli_args(seeded, mode="enrich-greeks"))

    assert query(seeded, "SELECT count(*) FROM candles") == [(12,)]
    assert greeks(seeded) == before


def test_the_json_summary_is_machine_readable(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks", extra=["--json"]))

    report = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_OK
    assert report["mode"] == "enrich-greeks"
    assert report["model"]["risk_free_rate"] == 0.065
    assert report["model"]["price_field"] == "close"
    assert report["bars"] == 12
    assert report["updated"] == 12
    assert report["unmatched"] == 0
    assert report["unsolved_iv"] == 0
    assert report["days"][0]["solved_iv"] == 12
    assert report["days"][0]["status"] == "SUCCESS"
    assert report["failures"] == []


def test_the_candle_mirror_is_refreshed_with_the_greeks(seeded) -> None:
    engine_main(cli_args(seeded, mode="enrich-greeks"))

    target = seeded / "data" / "candles" / "1m" / "2026-09-29.parquet"
    assert target.exists()
    assert list((seeded / "data" / "candles").rglob("*.tmp")) == []
    values = query(
        seeded,
        f"SELECT DISTINCT iv FROM read_parquet('{target.as_posix()}') WHERE iv IS NOT NULL",
    )
    assert len(values) == 1
    assert float(values[0][0]) == pytest.approx(SIGMA, abs=1e-6)


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------
def test_a_single_contract_can_be_enriched_alone(seeded) -> None:
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks", extra=["--symbol", CE_SYMBOL]))

    assert exit_code == EXIT_OK
    assert all(row[1] is not None for row in greeks(seeded, CE_SYMBOL))
    assert all(row[1] is None for row in greeks(seeded, PE_SYMBOL))


def test_a_single_timeframe_can_be_enriched(seeded) -> None:
    """``--timeframes`` is a subset here (unlike build-features, whose key has no timeframe)."""
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks", extra=["--timeframes", "1m"]))

    assert exit_code == EXIT_OK
    assert query(seeded, "SELECT DISTINCT timeframe FROM candles WHERE iv IS NOT NULL") == [("1m",)]


def test_a_disabled_timeframe_is_a_configuration_error(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks", extra=["--timeframes", "30m"]))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "30m" in capsys.readouterr().err


def test_a_dry_run_explains_the_model_and_writes_nothing(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks", extra=["--dry-run"]))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "dry-run: enrich-greeks" in output
    assert "black_scholes" in output
    assert "iv bracket" in output
    assert "2 contract(s), 12 bar(s)" in output
    assert "dry-run writes no greeks and no Parquet" in output
    assert query(seeded, "SELECT count(*) FROM candles WHERE iv IS NOT NULL") == [(0,)]


def test_a_weekend_is_said_out_loud_instead_of_enriched(seeded, capsys) -> None:
    exit_code = engine_main(
        cli_args(seeded, mode="enrich-greeks", extra=["--date", WEEKEND.isoformat()])
    )

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "[NO_SESSION]" in output
    assert "0 contract(s)" in output


def test_a_bad_date_is_a_configuration_error(seeded, capsys) -> None:
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks", extra=["--date", "29-09-2026"]))

    assert exit_code == EXIT_CONFIG_ERROR
    assert "not a valid ISO date" in capsys.readouterr().err


def test_a_database_without_candles_says_so_and_exits_zero(tmp_path, capsys) -> None:
    seed_index_history(db_file(tmp_path), [])
    exit_code = engine_main(cli_args(tmp_path, mode="enrich-greeks"))

    assert exit_code == EXIT_OK
    assert "no stored candles" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Honest failure
# ---------------------------------------------------------------------------
def test_without_index_history_the_iv_stays_null_and_the_run_says_why(tmp_path, capsys) -> None:
    seed_day(tmp_path, with_index=False)
    exit_code = engine_main(cli_args(tmp_path, mode="enrich-greeks"))

    assert exit_code == EXIT_OK
    assert query(tmp_path, "SELECT count(*) FROM candles WHERE iv IS NULL") == [(12,)]
    stored = query(
        tmp_path, "SELECT DISTINCT days_to_expiry FROM candles WHERE days_to_expiry IS NOT NULL"
    )
    assert stored  # the context columns are still written: they need no model
    rows = query(tmp_path, "SELECT count(*) FROM candles WHERE distance_from_atm IS NOT NULL")
    assert rows == [(0,)]  # ... except the ATM distance, which needs the level


def test_a_price_with_no_iv_is_left_null_and_counted(seeded, capsys) -> None:
    """A deep in-the-money close below intrinsic has no volatility: it must stay NULL."""
    connection = connect(db_file(seeded))
    try:
        connection.execute(
            "UPDATE candles SET open = 1.0, high = 1.0, low = 1.0, close = 1.0, "
            "oi = 1 WHERE tradingsymbol = ? AND timestamp = ?",
            [CE_SYMBOL, SESSION_START + timedelta(minutes=1)],
        )
    finally:
        connection.close()

    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "1 unsolved" in output
    row = query(
        seeded,
        "SELECT iv, delta FROM candles WHERE tradingsymbol = "
        f"'{CE_SYMBOL}' AND timestamp = TIMESTAMPTZ '{(SESSION_START + timedelta(minutes=1)).isoformat()}'",
    )
    assert row == [(None, None)]
    assert query(seeded, "SELECT count(*) FROM candles WHERE iv IS NOT NULL") == [(11,)]
    # The bar it *precedes* still has a delta, so its expectation is computed from it.
    neighbours = query(
        seeded,
        "SELECT iv FROM candles WHERE tradingsymbol = "
        f"'{CE_SYMBOL}' AND iv IS NOT NULL",
    )
    assert len(neighbours) == 5


def test_one_contract_that_cannot_be_enriched_does_not_sink_the_day(
    seeded, monkeypatch, capsys
) -> None:
    """A failure is local and loud: exit 1, the contract named, the rest still written."""
    from apps.anjaneeya_engine import run as engine_run

    real = engine_run.enrich_frame

    def exploding(frame, **kwargs):
        if not frame.empty and str(frame["tradingsymbol"].iloc[0]) == PE_SYMBOL:
            raise RuntimeError("no model today")
        return real(frame, **kwargs)

    monkeypatch.setattr(engine_run, "enrich_frame", exploding)
    exit_code = engine_main(cli_args(seeded, mode="enrich-greeks"))

    output = capsys.readouterr().out
    assert exit_code == EXIT_FAILED
    assert "[PARTIAL]" in output
    assert "no model today" in output
    assert all(row[1] is not None for row in greeks(seeded, CE_SYMBOL))
    assert all(row[1] is None for row in greeks(seeded, PE_SYMBOL))


# ---------------------------------------------------------------------------
# The closed loop: enrichment -> features -> validation
# ---------------------------------------------------------------------------
def test_build_features_uses_the_greeks_enrichment_published(seeded) -> None:
    engine_main(cli_args(seeded, mode="enrich-greeks"))
    # ``--date`` because this fixture stores candles, not ticks - build-features picks its
    # default day from ticks, and every mode's ``--date`` is the same override.
    engine_main(
        cli_args(seeded, mode="build-features", extra=["--date", "2026-09-29"])
    )

    row = query(
        seeded,
        "SELECT iv, delta, iv_change, expected_option_move, actual_option_move, option_residual "
        f"FROM features WHERE tradingsymbol = '{CE_SYMBOL}' ORDER BY timestamp LIMIT 1",
    )
    iv, delta, iv_change, expected, actual, residual = row[0]
    assert iv == pytest.approx(SIGMA, abs=1e-6)
    assert delta is not None
    # The first published bar is differenced against the *previous session's* last bar,
    # which does not exist here (the day starts fresh), so those three are NULL.
    assert iv_change is None and expected is None and actual is None and residual is None

    second = query(
        seeded,
        "SELECT iv_change, expected_option_move, actual_option_move, option_residual "
        f"FROM features WHERE tradingsymbol = '{CE_SYMBOL}' ORDER BY timestamp LIMIT 1 OFFSET 1",
    )[0]
    iv_change, expected, actual, residual = second
    assert iv_change == pytest.approx(0.0, abs=1e-6)
    assert expected == pytest.approx(delta * 4.0, rel=1e-3)  # delta * the NIFTY move
    assert actual is not None
    assert residual == pytest.approx(actual - expected, abs=1e-9)


def test_validation_reports_a_pass_once_the_greeks_are_there(seeded) -> None:
    engine_main(cli_args(seeded, mode="enrich-greeks"))
    engine_main(cli_args(seeded, mode="build-features", extra=["--date", "2026-09-29"]))
    exit_code = engine_main(cli_args(seeded, mode="validate", extra=["--date", "2026-09-29"]))

    assert exit_code == EXIT_OK
    report = json.loads(
        (seeded / "data" / "reports" / "feature_validation" / "2026-09-29_1m.json").read_text()
    )
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["greeks_sanity"]["status"] == "PASS"
    assert checks["relative_identity"]["status"] in {"PASS", "WARN"}
    assert "enrich-greeks" not in checks["missing_values"]["detail"]


def test_rebuilding_a_scope_replaces_the_greeks_with_the_new_bars(seeded) -> None:
    """A rebuilt bar arrives without greeks: the enrichment is a pass, not a property of the table.

    This is the documented order of operations (``build-candles`` -> ``enrich-greeks`` ->
    ``build-features``): a rebuild that produced fresh bars has nothing to say about their
    IVs, so the columns go back to ``NULL`` rather than carrying a stale number.
    """
    from core.constants import CANDLE_COLUMNS
    from core.storage import insert_rows

    engine_main(cli_args(seeded, mode="enrich-greeks"))
    assert query(seeded, "SELECT count(*) FROM candles WHERE iv IS NOT NULL") == [(12,)]

    connection = connect(db_file(seeded))
    try:
        # Exactly the two statements ``CandleWriter.replace_candles`` runs for a scope.
        connection.execute(
            "DELETE FROM candles WHERE timeframe = '1m' AND tradingsymbol = ?", [CE_SYMBOL]
        )
        insert_rows(
            connection,
            "candles",
            list(priced_window(minutes=MINUTES).rows()),
            mode="append",
            columns=CANDLE_COLUMNS,
        )
    finally:
        connection.close()

    assert query(seeded, "SELECT count(*) FROM candles WHERE iv IS NOT NULL") == [(6,)]  # the PE only
    assert query(seeded, "SELECT count(*) FROM candles WHERE iv IS NULL") == [(6,)]
