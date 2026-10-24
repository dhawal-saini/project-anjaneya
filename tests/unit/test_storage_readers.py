"""Tests for :mod:`core.storage.readers` (the engine's read side).

The reader is the boundary between DuckDB's dtypes and the candle builder, so
these tests pin down exactly that boundary: which rows come back, in which order,
with which dtypes - and that a timestamp stored as UTC is still grouped by its
*IST* date.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from core.constants import CANDLE_COLUMNS, FEATURE_COLUMNS, INDEX_HISTORY_COLUMNS
from core.schemas import QualityFlag
from core.storage import (
    CoverageTracker,
    StreamStats,
    build_coverage_row,
    connect,
    first_candle_timestamp,
    insert_rows,
    quality_counts,
    read_candles,
    read_features,
    read_index_history,
    read_ticks,
    symbols_with_candles,
    symbols_with_ticks,
    trading_days_with_candles,
    trading_days_with_ticks,
)
from core.storage.readers import TICK_READ_COLUMNS
from tests.support.engine import (
    CE_SYMBOL,
    CE_TOKEN,
    DAY,
    EXPIRY,
    IST,
    PE_SYMBOL,
    PE_TOKEN,
    RUN_ID,
    SESSION_END,
    SESSION_START,
    at,
    make_candle,
    make_feature,
    make_index_row,
    make_tick,
    make_tick_row,
    seed_ticks,
    ticks_for_minutes,
)


def session_rows(minutes=(0, 1, 2)):
    """Ticks for both contracts, plus one SUSPECT and two out-of-window ticks."""
    rows = [make_tick_row(**tick) for tick in ticks_for_minutes(minutes)]
    rows += [
        make_tick_row(**tick)
        for tick in ticks_for_minutes(
            minutes, symbol=PE_SYMBOL, token=PE_TOKEN, option_type="PE", start_price=90.0
        )
    ]
    return rows


@pytest.fixture
def connection(isolated_paths):
    """A single read-write connection to a freshly seeded database.

    One connection, because DuckDB refuses a second one to the same file inside
    one process - the same constraint the CLI has to live with.
    """
    seed_ticks(isolated_paths["db_path"], session_rows())
    instance = connect(isolated_paths["db_path"])
    yield instance
    instance.close()


def window(start=SESSION_START, end=SESSION_END):
    return {"start": start, "end": end}


# ---------------------------------------------------------------------------
# read_ticks
# ---------------------------------------------------------------------------
def test_read_ticks_returns_the_window_ordered_by_symbol_then_time(connection) -> None:
    frame = read_ticks(connection, **window())

    assert len(frame) == 12  # two contracts with two ticks in three minutes
    assert list(frame.columns) == list(TICK_READ_COLUMNS)
    assert list(frame["tradingsymbol"].unique()) == [CE_SYMBOL, PE_SYMBOL]
    pairs = list(zip(frame["tradingsymbol"], frame["timestamp"], strict=True))
    assert pairs == sorted(pairs)  # ordered by symbol, then by time
    for symbol in (CE_SYMBOL, PE_SYMBOL):
        stamps = frame.loc[frame["tradingsymbol"] == symbol, "timestamp"].tolist()
        assert stamps == sorted(stamps)


def test_read_ticks_hands_the_builder_a_frame_it_can_use(connection) -> None:
    """The dtypes that matter: tz-aware timestamps and usable contract metadata."""
    frame = read_ticks(connection, symbols=[CE_SYMBOL], **window())

    assert frame["timestamp"].dt.tz is not None
    assert frame["instrument_token"].iloc[0] == CE_TOKEN
    assert frame["quality_flag"].iloc[0] == QualityFlag.OK.value

    from apps.anjaneeya_engine.candles import build_1m_candles

    bars, stats = build_1m_candles(frame, tradingsymbol=CE_SYMBOL, run_id=RUN_ID)
    assert len(bars) == 3
    assert stats.ticks == 6
    assert bars[0].timestamp == SESSION_START
    # DuckDB hands a DATE column back as a pandas Timestamp; the row contract
    # wants a plain date, so the builder must convert rather than rely on it.
    assert bars[0].expiry == EXPIRY
    assert not isinstance(bars[0].expiry, datetime)


def test_read_ticks_excludes_ticks_outside_the_window(connection) -> None:
    insert_rows(
        connection,
        "ticks",
        [
            make_tick_row(timestamp=SESSION_START - timedelta(minutes=5)),
            make_tick_row(timestamp=SESSION_END),
        ],
        mode="append",
    )

    inside = read_ticks(connection, **window())
    wider = read_ticks(
        connection, **window(SESSION_START - timedelta(minutes=10), SESSION_END + timedelta(minutes=1))
    )

    assert SESSION_START - timedelta(minutes=5) not in inside["timestamp"].tolist()
    assert len(wider) == len(inside) + 2


def test_read_ticks_only_returns_usable_ticks_by_default(connection) -> None:
    flagged = make_tick_row(
        timestamp=at(0, seconds=10), quality_flag=QualityFlag.SUSPECT.value
    )
    insert_rows(connection, "ticks", [flagged], mode="append")

    default = read_ticks(connection, symbols=[CE_SYMBOL], **window())
    everything = read_ticks(connection, symbols=[CE_SYMBOL], quality_flags=None, **window())
    only_flagged = read_ticks(
        connection, symbols=[CE_SYMBOL], quality_flags=[QualityFlag.SUSPECT], **window()
    )

    assert QualityFlag.SUSPECT.value not in default["quality_flag"].tolist()
    assert len(everything) == len(default) + 1
    assert only_flagged["quality_flag"].tolist() == [QualityFlag.SUSPECT.value]


def test_read_ticks_filters_by_symbol(connection) -> None:
    frame = read_ticks(connection, symbols=[PE_SYMBOL], **window())

    assert set(frame["tradingsymbol"]) == {PE_SYMBOL}


def test_read_ticks_rejects_an_unknown_column(connection) -> None:
    with pytest.raises(ValueError, match="Unknown tick columns"):
        read_ticks(connection, columns=["timestamp", "nope"], **window())


# ---------------------------------------------------------------------------
# Discovery queries
# ---------------------------------------------------------------------------
def test_trading_days_are_ist_dates_not_utc_dates(connection) -> None:
    """23:00 UTC on the 29th is 04:30 IST on the 30th - and must be grouped as such."""
    insert_rows(
        connection,
        "ticks",
        [
            {
                **make_tick(timestamp=datetime(2026, 9, 30, 4, 30, tzinfo=IST)),
                "fetch_timestamp": datetime(2026, 9, 30, 4, 30, tzinfo=IST),
            }
        ],
        mode="append",
    )

    days = trading_days_with_ticks(connection)

    assert days == [DAY, date(2026, 9, 30)]
    assert date(2026, 9, 29) in days and date(2026, 9, 30) in days


def test_symbols_with_ticks_is_window_scoped_and_sorted(connection) -> None:
    assert symbols_with_ticks(connection, start=SESSION_START, end=SESSION_END) == [
        CE_SYMBOL,
        PE_SYMBOL,
    ]
    assert symbols_with_ticks(connection, start=at(30), end=at(60)) == []


def test_quality_counts_reports_what_was_stored_per_flag(connection) -> None:
    insert_rows(
        connection,
        "ticks",
        [
            make_tick_row(timestamp=at(0, seconds=10), quality_flag=QualityFlag.SUSPECT.value),
            make_tick_row(timestamp=at(0, seconds=20), quality_flag=QualityFlag.STALE.value),
        ],
        mode="append",
    )

    counts = quality_counts(connection, start=SESSION_START, end=SESSION_END)
    scoped = quality_counts(connection, start=SESSION_START, end=SESSION_END, symbols=[CE_SYMBOL])

    assert counts == {"OK": 12, "STALE": 1, "SUSPECT": 1}
    assert scoped == {"OK": 6, "STALE": 1, "SUSPECT": 1}


# ---------------------------------------------------------------------------
# read_candles
# ---------------------------------------------------------------------------
def test_read_candles_is_empty_before_anything_is_built(connection) -> None:
    assert read_candles(connection).empty


def test_read_candles_filters_by_window_symbol_and_timeframe(connection) -> None:
    rows = [
        make_candle(timestamp=at(0), timeframe=timeframe, tradingsymbol=CE_SYMBOL)
        for timeframe in ("1m", "5m")
    ]
    rows.append(
        make_candle(
            timestamp=at(0),
            timeframe="1m",
            tradingsymbol="NIFTY 50",
            instrument_token=None,
            expiry=None,
            strike=None,
            option_type=None,
        )
    )
    rows.append(make_candle(timestamp=at(30), timeframe="1m"))
    insert_rows(connection, "candles", rows, mode="append", columns=CANDLE_COLUMNS)

    everything = read_candles(connection)
    one_symbol = read_candles(connection, symbols=[CE_SYMBOL])
    one_timeframe = read_candles(connection, timeframes=["5m"])
    scoped = read_candles(connection, start=SESSION_START, end=at(5), symbols=[CE_SYMBOL])

    assert len(everything) == 4
    assert set(one_symbol["tradingsymbol"]) == {CE_SYMBOL}
    assert set(one_timeframe["timeframe"]) == {"5m"}
    assert len(scoped) == 2
    pairs = list(zip(everything["tradingsymbol"], everything["timeframe"], strict=True))
    assert pairs == sorted(pairs)


def test_candles_come_back_in_a_stable_order(connection) -> None:
    insert_rows(
        connection,
        "candles",
        [
            make_candle(
                timestamp=at(2),
                timeframe="1m",
                tradingsymbol=PE_SYMBOL,
                strike=None,
                expiry=None,
                option_type=None,
                instrument_token=PE_TOKEN,
            ),
            make_candle(timestamp=at(0), timeframe="5m", tradingsymbol=CE_SYMBOL),
            make_candle(timestamp=at(0), timeframe="1m", tradingsymbol=CE_SYMBOL),
        ],
        mode="append",
        columns=CANDLE_COLUMNS,
    )

    frame = read_candles(connection, symbols=[CE_SYMBOL])

    pairs = list(zip(frame["timeframe"], frame["timestamp"], strict=True))
    assert pairs == [("1m", at(0)), ("5m", at(0))]
    assert set(frame.columns) >= {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# Candles: the greeks columns and the candle-side discovery queries (Stage 3.3)
# ---------------------------------------------------------------------------
def test_read_candles_serves_the_greeks_that_enrichment_fills(connection) -> None:
    """The read side stops hiding half a row: the greeks are served like any other column."""
    from core.storage import CANDLE_READ_COLUMNS

    assert set(CANDLE_READ_COLUMNS) == set(CANDLE_COLUMNS)
    insert_rows(
        connection,
        "candles",
        [make_candle(timestamp=at(0), iv=0.14, delta=0.52, gamma=0.0003, theta=-12.0, vega=8.5)],
        mode="append",
        columns=CANDLE_COLUMNS,
    )

    frame = read_candles(connection)

    assert float(frame["iv"].iloc[0]) == pytest.approx(0.14)
    assert float(frame["delta"].iloc[0]) == pytest.approx(0.52)
    assert float(frame["vega"].iloc[0]) == pytest.approx(8.5)


def test_trading_days_with_candles_groups_by_the_ist_date(connection) -> None:
    """The candle-side day discovery must agree with the tick-side one about "a day"."""
    insert_rows(
        connection,
        "candles",
        [
            make_candle(timestamp=at(0), timeframe="1m", tradingsymbol=CE_SYMBOL),
            make_candle(timestamp=at(0), timeframe="5m", tradingsymbol=CE_SYMBOL),
            make_candle(timestamp=at(30), timeframe="1m", tradingsymbol=CE_SYMBOL),
        ],
        mode="append",
        columns=CANDLE_COLUMNS,
    )

    assert trading_days_with_candles(connection) == [DAY]
    assert trading_days_with_candles(connection, timeframes=["5m"]) == [DAY]
    assert trading_days_with_candles(connection, timeframes=["3m"]) == []


def test_symbols_with_candles_filters_by_window_and_timeframe(connection) -> None:
    """The enrichment's work list: which contracts actually have bars in the session."""
    insert_rows(
        connection,
        "candles",
        [
            make_candle(timestamp=at(0), timeframe="1m", tradingsymbol=CE_SYMBOL),
            make_candle(timestamp=at(0), timeframe="5m", tradingsymbol=CE_SYMBOL),
            make_candle(
                timestamp=at(0),
                timeframe="1m",
                tradingsymbol=PE_SYMBOL,
                instrument_token=PE_TOKEN,
                option_type="PE",
            ),
            make_candle(timestamp=at(60), timeframe="1m", tradingsymbol=CE_SYMBOL),
        ],
        mode="append",
        columns=CANDLE_COLUMNS,
    )

    assert symbols_with_candles(connection, start=SESSION_START, end=SESSION_END) == [
        CE_SYMBOL,
        PE_SYMBOL,
    ]
    assert symbols_with_candles(
        connection, start=SESSION_START, end=SESSION_END, timeframes=["5m"]
    ) == [CE_SYMBOL]
    assert symbols_with_candles(connection, start=SESSION_START, end=at(10)) == [
        CE_SYMBOL,
        PE_SYMBOL,
    ]


# ---------------------------------------------------------------------------
# Shared coverage module (moved to core in Stage 3.1)
# ---------------------------------------------------------------------------
def test_coverage_vocabulary_lives_in_core_and_is_re_exported_by_capture() -> None:
    from apps.anjaneeya_capture import storage_writer

    assert storage_writer.CoverageTracker is CoverageTracker
    assert storage_writer.StreamStats is StreamStats
    assert storage_writer.build_coverage_row is build_coverage_row

    stats = StreamStats(tradingsymbol="NIFTY CE", timeframe="1m", rows=5, expected_rows=10)
    row = build_coverage_row(trading_date=DAY, stats=stats, run_id=RUN_ID)
    assert row.coverage_ratio == 0.5
    assert row.quality_flag is QualityFlag.OK



# ---------------------------------------------------------------------------
# read_features / read_index_history / first_candle_timestamp (Stage 3.2e)
def test_read_features_is_empty_before_anything_is_built(connection) -> None:
    assert read_features(connection).empty


def test_read_features_filters_by_window_symbol_and_timeframe(connection) -> None:
    insert_rows(
        connection,
        "features",
        [
            make_feature(timestamp=at(0), tradingsymbol=CE_SYMBOL, timeframe="1m"),
            make_feature(timestamp=at(1), tradingsymbol=CE_SYMBOL, timeframe="1m"),
            make_feature(timestamp=at(0), tradingsymbol=PE_SYMBOL, timeframe="5m"),
        ],
        mode="append",
    )

    everything = read_features(connection)
    one_symbol = read_features(connection, symbols=[CE_SYMBOL])
    one_timeframe = read_features(connection, timeframes=["5m"])
    scoped = read_features(connection, start=SESSION_START, end=at(1), symbols=[CE_SYMBOL])

    assert len(everything) == 3
    assert len(one_symbol) == 2
    assert len(one_timeframe) == 1
    assert list(scoped["timestamp"]) == [at(0)]


def test_read_features_carries_the_whole_stored_contract(connection) -> None:
    insert_rows(connection, "features", [make_feature()], mode="append")

    frame = read_features(connection)

    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame.iloc[0]["option_type"] == "CE"
    assert frame.iloc[0]["run_id"] == RUN_ID


def test_read_features_is_ordered_by_contract_then_time(connection) -> None:
    insert_rows(
        connection,
        "features",
        [
            make_feature(timestamp=at(1), tradingsymbol=PE_SYMBOL, option_type="PE"),
            make_feature(timestamp=at(2), tradingsymbol=CE_SYMBOL),
            make_feature(timestamp=at(0), tradingsymbol=CE_SYMBOL),
        ],
        mode="append",
    )

    frame = read_features(connection)

    assert list(zip(frame["tradingsymbol"], frame["timestamp"], strict=True)) == [
        (CE_SYMBOL, at(0)),
        (CE_SYMBOL, at(2)),
        (PE_SYMBOL, at(1)),
    ]


def test_read_index_history_scopes_by_window_symbol_and_timeframe(connection) -> None:
    insert_rows(
        connection,
        "index_history",
        [
            make_index_row(timestamp=at(0), timeframe="1m"),
            make_index_row(timestamp=at(1), timeframe="1m"),
            make_index_row(timestamp=at(0), timeframe="1d"),
        ],
        mode="append",
    )

    assert len(read_index_history(connection)) == 3
    assert len(read_index_history(connection, timeframes=["1m"])) == 2
    assert len(read_index_history(connection, start=SESSION_START, end=at(1))) == 2
    assert read_index_history(connection, symbols=["SOMETHING ELSE"]).empty
    assert list(read_index_history(connection).columns) == list(INDEX_HISTORY_COLUMNS)


def test_first_candle_timestamp_is_none_without_candles(connection) -> None:
    assert first_candle_timestamp(connection, tradingsymbol=CE_SYMBOL, timeframe="1m") is None


def test_first_candle_timestamp_finds_the_contracts_birth(connection) -> None:
    insert_rows(
        connection,
        "candles",
        [
            make_candle(timestamp=at(1)),
            make_candle(timestamp=at(0)),
            make_candle(timestamp=at(0), tradingsymbol=PE_SYMBOL, option_type="PE"),
        ],
        mode="append",
    )

    assert first_candle_timestamp(connection, tradingsymbol=CE_SYMBOL, timeframe="1m") == at(0)
    assert first_candle_timestamp(connection, tradingsymbol=PE_SYMBOL, timeframe="5m") is None
