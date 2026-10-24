"""Tests for ``--mode universe`` (Stage 3.4a, spec 17) and the readers behind it.

Spec 17 asks a question an operator has to be able to answer before any research
is believable: *which strikes are in the universe, and do I actually hold data for
them?* These tests pin down the four things that make the answer trustworthy:

* the ATM comes from the configured reference (the index, or the futures contract)
  and from **stored** data - never from a guess,
* a contract the instrument master does not list is reported as ``NOT_LISTED``
  with its strike and expiry named, not quietly dropped,
* the candle and feature counts come from the database (so "we captured it" and
  "we have features for it" stay different statements),
* the mode writes nothing at all.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from apps.anjaneeya_engine.run import (
    _stored_futures_symbol,
    _universe_day,
    _universe_reference_price,
)
from apps.anjaneeya_engine.run import (
    main as engine_main,
)
from core.app_bootstrap import EXIT_FAILED, EXIT_OK
from core.storage import (
    candle_counts,
    connect,
    feature_counts,
    insert_rows,
    read_instruments,
    read_option_instruments,
)
from tests.support.engine import (
    DAY,
    EXPIRY,
    INDEX_SYMBOL,
    SESSION_START,
    UNIVERSE_STRIKES,
    cli_args,
    futures_instrument_row,
    make_candle,
    make_engine_context,
    make_feature,
    make_index_row,
    option_instrument_rows,
    seed_candles,
    seed_features,
    seed_index_history,
    seed_instruments,
)

TIMEFRAME = "1m"
#: DAY is a Tuesday, and Tuesday is the configured expiry weekday: the contract that
#: expires on the session day itself is the config's ``current_week``.
CURRENT_EXPIRY = date(2026, 9, 29)
#: The week after :data:`CURRENT_EXPIRY` - the config's ``next_week``.
NEXT_EXPIRY = date(2026, 10, 6)
#: The close the seeded NIFTY series ends on (the ATM anchor for most tests).
INDEX_CLOSE = 25014.0
#: Six contracts make up one expiry's ATM window in the shipped config.
CONTRACTS_PER_EXPIRY = len(UNIVERSE_STRIKES) * 2


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def ochl(close: float) -> dict[str, float]:
    """A consistent OHLC block around one close (the row model validates high/low)."""
    return {"open": close - 0.5, "high": close + 0.5, "low": close - 0.5, "close": close}


def contract_candles(
    *,
    symbol: str,
    token: int,
    strike: float,
    option_type: str,
    expiry: date = EXPIRY,
    bars: int = 3,
):
    """``bars`` 1m candles for one contract, inside the session window."""
    return [
        make_candle(
            timestamp=SESSION_START + timedelta(minutes=index),
            timeframe=TIMEFRAME,
            instrument_token=token,
            tradingsymbol=symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            **ochl(120.0 + index),
        )
        for index in range(bars)
    ]


def seed_day(
    tmp_path,
    *,
    expiries=(CURRENT_EXPIRY, NEXT_EXPIRY),
    listed_strikes=UNIVERSE_STRIKES,
    covered_strikes=None,
    with_features: bool = True,
    with_index: bool = True,
):
    """A session the universe mode can report on.

    ``listed_strikes`` is what the instrument master holds; ``covered_strikes`` the
    subset the database has candles (and features) for. The gap between the two is
    exactly what the mode has to expose, so a test can create either kind of hole.
    """
    bars = db_file(tmp_path)
    covered = {float(strike) for strike in (covered_strikes or listed_strikes)}
    instruments = option_instrument_rows(expiries=expiries, strikes=listed_strikes)
    instruments.append(futures_instrument_row())
    seed_instruments(bars, instruments)

    candle_rows = []
    feature_rows = []
    for instrument in instruments:
        if instrument.instrument_type not in {"CE", "PE"}:
            continue
        if float(instrument.strike) not in covered:
            continue
        rows = contract_candles(
            symbol=instrument.tradingsymbol,
            token=instrument.instrument_token,
            strike=instrument.strike,
            option_type=instrument.instrument_type,
            expiry=instrument.expiry,
        )
        candle_rows.extend(rows)
        if with_features:
            feature_rows.extend(
                make_feature(
                    timestamp=row.timestamp,
                    tradingsymbol=row.tradingsymbol,
                    instrument_token=row.instrument_token,
                    expiry=row.expiry,
                    strike=row.strike,
                    option_type=row.option_type,
                )
                for row in rows
            )
    seed_candles(bars, candle_rows)
    if feature_rows:
        seed_features(bars, feature_rows)
    if with_index:
        seed_index_history(
            bars,
            [
                make_index_row(
                    timestamp=SESSION_START + timedelta(minutes=offset),
                    open=INDEX_CLOSE - 3 + offset,
                    high=INDEX_CLOSE + 5 + offset,
                    low=INDEX_CLOSE - 5 + offset,
                    close=INDEX_CLOSE - 2 + offset,
                )
                for offset in range(3)
            ],
        )
    return tmp_path


def run(tmp_path, *, extra=None):
    """Run ``--mode universe`` against a seeded database."""
    return engine_main(cli_args(tmp_path, mode="universe", extra=extra))


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_universe_reports_the_configured_menu_with_its_data_counts(tmp_path, capsys) -> None:
    seed_day(tmp_path)

    exit_code = run(tmp_path)
    output = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "ANJANEYA-ENGINE universe - 2026-09-29" in output
    assert "reference      : spot @ 25,014.00 (index_history:NIFTY 50" in output
    assert "ATM strike     : 25,000 (interval 50, rounding nearest)" in output
    assert "strike ladder  : 24,800, 24,900, 25,000, 25,100, 25,200" in output
    assert "expiries       : 2026-09-29, 2026-10-06" in output
    assert f"selected       : {2 * CONTRACTS_PER_EXPIRY} contract(s)" in output
    # Every selected contract has three bars and three feature rows.
    assert "coverage       : OK=20" in output
    assert "complete       : yes" in output
    assert "NIFTY29SEP2625000CE" in output


def test_a_contract_without_bars_is_reported_not_hidden(tmp_path, capsys) -> None:
    """Capture subscribed to every strike; only the 25 000 ones actually have bars."""
    seed_day(tmp_path, covered_strikes=(25000,))

    assert run(tmp_path) == EXIT_OK
    output = capsys.readouterr().out

    assert "coverage       : NO_CANDLES=16, OK=4" in output
    assert "complete       : no" in output
    assert "NO_CANDLES" in output


def test_a_strike_the_master_does_not_list_is_named(tmp_path, capsys) -> None:
    seed_day(tmp_path, listed_strikes=(24900, 25000))

    assert run(tmp_path) == EXIT_OK
    output = capsys.readouterr().out

    assert "coverage       : NOT_LISTED=12, OK=8" in output
    assert "missing        : CE 25100 not listed for 2026-09-29" in output
    assert "missing        : PE 25100 not listed for 2026-09-29" in output
    # The list is capped at eight lines; the four remaining 06-Oct gaps are summarised.
    assert "missing        : ... and 4 more" in output


def test_feature_data_is_reported_separately_from_candle_data(tmp_path, capsys) -> None:
    """"We have bars" and "we have features" are different statements."""
    seed_day(tmp_path, with_features=False)

    assert run(tmp_path) == EXIT_OK
    output = capsys.readouterr().out

    assert "coverage       : NO_FEATURES=20" in output
    assert "complete       : no" in output


# ---------------------------------------------------------------------------
# Where the ATM price comes from
# ---------------------------------------------------------------------------
def test_price_flag_moves_the_atm(tmp_path, capsys) -> None:
    seed_day(tmp_path)

    assert run(tmp_path, extra=["--price", "25160"]) == EXIT_OK
    output = capsys.readouterr().out

    assert "--price (operator supplied)" in output
    assert "ATM strike     : 25,150" in output
    assert "strike ladder  : 24,950, 25,050, 25,150, 25,250, 25,350" in output


def test_a_rejected_price_is_a_configuration_error(tmp_path, capsys) -> None:
    seed_day(tmp_path)

    assert run(tmp_path, extra=["--price", "-5"]) == 2  # EXIT_CONFIG_ERROR
    assert "--price must be a positive number" in capsys.readouterr().err


def test_no_stored_reference_price_fails_the_run_instead_of_guessing(tmp_path, capsys) -> None:
    """Bars exist, so the day exists - but nothing places the ATM."""
    seed_day(tmp_path, with_index=False)

    exit_code = run(tmp_path)
    output = capsys.readouterr().out

    assert exit_code == EXIT_FAILED
    assert "reference      : UNRESOLVED" in output
    assert "no stored index_history or candle rows for NIFTY 50" in output


def test_the_index_series_can_come_from_candles_when_history_is_missing(
    tmp_path, capsys
) -> None:
    """An intraday capture stores index *ticks*; ``index_history`` may lag behind."""
    seed_day(tmp_path, with_index=False)
    connection = connect(db_file(tmp_path))
    try:
        insert_rows(
            connection,
            "candles",
            [
                make_candle(
                    timestamp=SESSION_START + timedelta(minutes=offset),
                    tradingsymbol=INDEX_SYMBOL,
                    instrument_token=256_265,
                    expiry=None,
                    strike=None,
                    option_type=None,
                    **ochl(25012.0 + offset),
                ).to_record(exclude_none=False)
                for offset in range(2)
            ],
            mode="append",
            columns=[
                "timestamp",
                "timeframe",
                "instrument_token",
                "tradingsymbol",
                "expiry",
                "strike",
                "option_type",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "oi",
                "source",
                "fetch_timestamp",
                "quality_flag",
                "run_id",
            ],
        )
    finally:
        connection.close()

    assert run(tmp_path) == EXIT_OK
    # The last stored index close (25 013) places the ATM at 25 000.
    assert "candles:NIFTY 50 @" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Report shapes, dry-run, and the read-only guarantee
# ---------------------------------------------------------------------------
def test_json_report_carries_the_same_facts(tmp_path, capsys) -> None:
    seed_day(tmp_path)

    assert run(tmp_path, extra=["--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)

    assert report["mode"] == "universe"
    day = report["days"][0]
    assert day["day"] == "2026-09-29"
    assert day["reference_label"] == "spot"
    assert day["reference_price"] == INDEX_CLOSE
    assert day["selection"]["atm_strike"] == 25000.0
    assert day["selection"]["offsets"] == [-200.0, -100.0, 0.0, 100.0, 200.0]
    assert day["complete"] is True
    assert day["counts"] == {"OK": 20}
    assert day["symbol_drift"] == 0
    first = day["contracts"][0]
    assert set(first) == {
        "expiry",
        "strike",
        "option_type",
        "tradingsymbol",
        "instrument_token",
        "candles",
        "features",
        "symbol_matches_template",
        "status",
    }
    assert first["status"] == "OK"
    assert first["candles"] == 3


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys) -> None:
    seed_day(tmp_path)
    before = query(tmp_path, "SELECT count(*) FROM candles")[0][0]

    assert run(tmp_path, extra=["--dry-run"]) == EXIT_OK
    output = capsys.readouterr().out

    assert "universe dry-run" in output
    assert "writes         : nothing" in output
    assert query(tmp_path, "SELECT count(*) FROM candles")[0][0] == before


def test_the_mode_leaves_every_table_untouched(tmp_path, capsys) -> None:
    seed_day(tmp_path)
    tables = ("candles", "features", "instruments", "index_history", "data_coverage")
    before = {table: query(tmp_path, f"SELECT count(*) FROM {table}")[0][0] for table in tables}

    assert run(tmp_path) == EXIT_OK
    capsys.readouterr()

    after = {table: query(tmp_path, f"SELECT count(*) FROM {table}")[0][0] for table in tables}
    assert after == before


def test_a_day_without_candles_reports_no_data(tmp_path, capsys) -> None:
    seed_instruments(db_file(tmp_path), option_instrument_rows())

    assert run(tmp_path) == EXIT_OK
    assert "no stored candles found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The futures reference (spec 17: "spot vs futures, configurable")
# ---------------------------------------------------------------------------
def futures_universe_config(config):
    """``option_universe`` with ``atm.reference`` switched to the futures contract."""
    atm = config.option_universe.atm.model_copy(update={"reference": "futures"})
    return config.model_copy(
        update={"option_universe": config.option_universe.model_copy(update={"atm": atm})}
    )


def test_futures_reference_price_comes_from_the_futures_contract(tmp_path, config) -> None:
    seed_day(tmp_path)
    connection = connect(db_file(tmp_path))
    try:
        # A futures bar whose close (25 152) snaps to 25 150 - not 25 000.
        rows = [
            make_candle(
                timestamp=SESSION_START + timedelta(minutes=offset),
                tradingsymbol="NIFTY27OCT26FUT",
                instrument_token=510_001,
                expiry=None,
                strike=None,
                option_type=None,
                **ochl(25152.0),
            ).to_record(exclude_none=False)
            for offset in range(2)
        ]
        insert_rows(connection, "candles", rows, mode="append", columns=list(rows[0].keys()))
    finally:
        connection.close()

    context = make_engine_context(futures_universe_config(config), mode="universe", bootstrap=False)
    connection = connect(db_file(tmp_path))
    try:
        price, source = _universe_reference_price(context, connection, day=DAY, explicit=None)
        result = _universe_day(
            context, connection, day=DAY, timeframe=TIMEFRAME, explicit_price=None
        )
    finally:
        connection.close()

    assert price == 25152.0
    assert source.startswith("candles:NIFTY27OCT26FUT @")
    assert result.selection is not None
    assert result.selection.reference_label == "futures"
    assert result.selection.atm_strike == 25150.0
    assert result.selection.offsets == (-200.0, -100.0, 0.0, 100.0, 200.0)


def test_futures_reference_without_a_listed_contract_is_reported(tmp_path, config) -> None:
    seed_day(tmp_path)
    context = make_engine_context(futures_universe_config(config), mode="universe", bootstrap=False)
    connection = connect(db_file(tmp_path))
    try:
        frame = read_instruments(connection, exchange="NFO", name="NIFTY")
        symbol, reason = _stored_futures_symbol(
            context, frame[frame["instrument_type"] != "FUT"], day=DAY
        )
    finally:
        connection.close()

    assert symbol is None
    assert reason == "the instruments table holds no futures row"


# ---------------------------------------------------------------------------
# The readers the mode is built on
# ---------------------------------------------------------------------------
def test_readers_report_what_is_actually_stored(tmp_path) -> None:
    seed_day(tmp_path, covered_strikes=(25000,))
    connection = connect(db_file(tmp_path))
    try:
        instruments = read_option_instruments(connection, exchange="NFO", name="NIFTY")
        assert len(instruments) == 2 * CONTRACTS_PER_EXPIRY  # 2 expiries x 3 x 2; FUT excluded
        assert set(instruments["instrument_type"]) == {"CE", "PE"}
        window = (SESSION_START, SESSION_START + timedelta(hours=6))
        bars = candle_counts(connection, start=window[0], end=window[1], timeframe="1m")
        assert sorted(bars.values()) == [3, 3, 3, 3]
        assert all(name.startswith("NIFTY") for name in bars)
        rows = feature_counts(connection, start=window[0], end=window[1], timeframe="1m")
        assert rows == bars
    finally:
        connection.close()
