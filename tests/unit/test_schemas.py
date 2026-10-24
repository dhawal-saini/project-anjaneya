"""Tests for :mod:`core.schemas` - the invariants every stored row must satisfy."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from core.constants import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    TICK_COLUMNS,
    TRADE_COLUMNS,
)
from core.schemas import (
    CONTRACT_PAIRS,
    CandleRow,
    ContractIdentity,
    FeatureRow,
    LabelClass,
    LabelRow,
    OptionType,
    QualityFlag,
    RunManifest,
    RunStatus,
    Source,
    TickRow,
    TradeRecord,
)

IST = ZoneInfo("Asia/Kolkata")
TS = datetime(2026, 9, 29, 9, 16, 0, tzinfo=IST)
EXPIRY = date(2026, 10, 1)
RUN_ID = "anjaneeya_capture_20260929-091500_abcdef12"


def make_tick(**overrides) -> dict:
    payload = {
        "timestamp": TS,
        "instrument_token": 12345,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "last_price": 123.45,
        "volume_traded": 1000,
        "oi": 5000,
        "bid": 123.0,
        "ask": 124.0,
        "bid_qty": 75,
        "ask_qty": 150,
        "source": "ZERODHA_WEBSOCKET",
        "fetch_timestamp": TS + timedelta(milliseconds=200),
        "quality_flag": "OK",
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return payload


def make_candle(**overrides) -> dict:
    payload = {
        "timestamp": TS,
        "timeframe": "1m",
        "instrument_token": 12345,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "open": 123.0,
        "high": 126.0,
        "low": 122.0,
        "close": 125.5,
        "volume": 500,
        "source": "DERIVED",
        "run_id": "run-1",
    }
    payload.update(overrides)
    return payload


def make_feature(**overrides) -> dict:
    payload = {
        "timestamp": TS,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "source": "DERIVED",
        "run_id": "run-1",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Column parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,model,columns", CONTRACT_PAIRS, ids=[p[0] for p in CONTRACT_PAIRS])
def test_model_fields_match_column_contract(name, model, columns) -> None:
    """Each row model must expose exactly the contracted columns."""
    assert set(model.model_fields) == set(columns), f"{name} drifted from its column contract"


def test_unknown_column_is_rejected() -> None:
    """A typo in a column name must never be silently accepted."""
    with pytest.raises(ValidationError):
        TickRow(**make_tick(not_a_real_column=1))


def make_snapshot(**overrides) -> dict:
    return {
        "timestamp": TS,
        "instrument_token": 12345,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "last_price": 123.0,
        "iv": 0.14,
        "oi": 5000,
        "fetch_timestamp": TS,
        "run_id": "run-1",
        **overrides,
    }


def make_label(**overrides) -> dict:
    return {
        "timestamp": TS,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "horizon_minutes": 5,
        "future_return": 0.012,
        "future_max_up_move": 0.02,
        "future_max_down_move": -0.01,
        "label": "UP_CONTINUATION",
        "label_version": "v1",
        "run_id": "run-1",
        **overrides,
    }


def make_trade(**overrides) -> dict:
    return {
        "trade_id": "t-1",
        "run_id": "run-1",
        "entry_timestamp": TS,
        "exit_timestamp": TS + timedelta(minutes=5),
        "tradingsymbol": "NIFTY26OCT25000CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "option_type": "CE",
        "entry_price": 123.0,
        "exit_price": 129.0,
        "quantity": 75,
        **overrides,
    }


def make_index_history(**overrides) -> dict:
    return {
        "timestamp": TS,
        "tradingsymbol": "NIFTY 50",
        "timeframe": "1m",
        "open": 25000.0,
        "high": 25010.0,
        "low": 24995.0,
        "close": 25005.0,
        "volume": 0,
        "oi": None,
        "source": "ZERODHA_REST",
        "fetch_timestamp": TS + timedelta(seconds=1),
        "run_id": "run-1",
        **overrides,
    }


def make_instrument(**overrides) -> dict:
    return {
        "instrument_token": 12345,
        "exchange": "NFO",
        "tradingsymbol": "NIFTY26OCT25000CE",
        "name": "NIFTY",
        "segment": "NFO-OPT",
        "instrument_type": "CE",
        "expiry": EXPIRY,
        "strike": 25000.0,
        "lot_size": 75,
        "tick_size": 0.05,
        "fetched_at": TS,
        "run_id": "run-1",
        **overrides,
    }


def make_coverage(**overrides) -> dict:
    return {
        "trading_date": EXPIRY,
        "tradingsymbol": "NIFTY26OCT25000CE",
        "timeframe": "TICK",
        "expected_rows": None,
        "actual_rows": 1200,
        "first_timestamp": TS,
        "last_timestamp": TS + timedelta(minutes=30),
        "gap_count": 2,
        "largest_gap_seconds": 45.0,
        "coverage_ratio": None,
        "quality_flag": "OK",
        "note": "captured",
        "run_id": "run-1",
        **overrides,
    }


VALID_PAYLOADS = {
    "TickRow": make_tick,
    "OptionChainSnapshotRow": make_snapshot,
    "IndexHistoryRow": make_index_history,
    "InstrumentRow": make_instrument,
    "CoverageRow": make_coverage,
    "CandleRow": make_candle,
    "FeatureRow": make_feature,
    "LabelRow": make_label,
    "TradeRecord": make_trade,
}


@pytest.mark.parametrize("name,model,columns", CONTRACT_PAIRS, ids=[p[0] for p in CONTRACT_PAIRS])
def test_every_contract_model_rejects_unknown_columns(name, model, columns) -> None:
    """Every row model forbids unknown keys, with an otherwise valid payload."""
    payload = {**VALID_PAYLOADS[name](), "bogus_column": 1}
    with pytest.raises(ValidationError):
        model(**payload)


@pytest.mark.parametrize("name,model,columns", CONTRACT_PAIRS, ids=[p[0] for p in CONTRACT_PAIRS])
def test_valid_payload_maps_onto_the_column_contract(name, model, columns) -> None:
    """A fully populated row serialises onto exactly the contracted columns."""
    record = model(**VALID_PAYLOADS[name]()).to_record(exclude_none=False)
    assert set(record) == set(columns)


# ---------------------------------------------------------------------------
# Ticks
# ---------------------------------------------------------------------------
def test_tick_accepts_a_well_formed_option_tick() -> None:
    tick = TickRow(**make_tick())
    assert tick.is_option is True
    assert tick.option_type is OptionType.CE
    assert tick.mid_price == pytest.approx(123.5)
    assert tick.quality_flag is QualityFlag.OK
    assert tick.source is Source.ZERODHA_WEBSOCKET


def test_tick_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        TickRow(**make_tick(timestamp=datetime(2026, 9, 29, 9, 16, 0)))  # noqa: DTZ001


def test_tick_rejects_partial_contract_identity() -> None:
    """Strike without option_type is a contract-identity bug (spec 6.4)."""
    with pytest.raises(ValidationError, match="expiry, strike and option_type"):
        TickRow(**make_tick(option_type=None))


def test_tick_rejects_fetch_before_exchange_timestamp() -> None:
    with pytest.raises(ValidationError, match="fetch_timestamp"):
        TickRow(**make_tick(fetch_timestamp=TS - timedelta(seconds=1)))


def test_tick_rejects_crossed_quote() -> None:
    with pytest.raises(ValidationError, match="bid"):
        TickRow(**make_tick(bid=130.0, ask=125.0))


def test_tick_rejects_negative_price() -> None:
    with pytest.raises(ValidationError):
        TickRow(**make_tick(last_price=-1.0))


def test_tick_index_row_has_no_option_identity() -> None:
    tick = TickRow(
        **make_tick(
            tradingsymbol="NIFTY 50",
            expiry=None,
            strike=None,
            option_type=None,
            instrument_token=256265,
            bid=None,
            ask=None,
            bid_qty=None,
            ask_qty=None,
        )
    )
    assert tick.is_option is False
    assert tick.mid_price is None


def test_to_record_converts_enums_and_drops_none() -> None:
    record = TickRow(**make_tick(bid=None, ask=None, bid_qty=None, ask_qty=None)).to_record()
    assert record["option_type"] == "CE"
    assert record["source"] == "ZERODHA_WEBSOCKET"
    assert "bid" not in record  # excluded because it is None
    assert set(record) <= set(TICK_COLUMNS)


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------
def test_candle_accepts_valid_ohlc() -> None:
    candle = CandleRow(**make_candle())
    assert candle.price_range == pytest.approx(4.0)
    assert candle.is_option is True


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"open": 127.0}, "above high"),
        ({"close": 121.0}, "below low"),
        ({"high": 100.0}, "cannot be below low"),
    ],
)
def test_candle_rejects_impossible_ohlc(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        CandleRow(**make_candle(**overrides))


def test_candle_rejects_unsupported_timeframe() -> None:
    with pytest.raises(ValidationError, match="Unsupported timeframe"):
        CandleRow(**make_candle(timeframe="7m"))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_feature_row_accepts_sparse_values() -> None:
    """Engine outputs arrive incrementally, so None must be allowed but preserved."""
    feature = FeatureRow(**make_feature(nifty_regime="BULLISH", days_to_expiry=1234.0))
    assert feature.nifty_regime.value == "BULLISH"
    assert feature.option_trend_score is None
    assert feature.contract_key == ("NIFTY", EXPIRY, 25000.0, "CE")


def test_feature_row_rejects_label_fields() -> None:
    """Structural look-ahead guard: labels must not leak into the feature row."""
    for field in ("future_return", "label", "label_version", "future_max_up_move"):
        with pytest.raises(ValidationError):
            FeatureRow(**make_feature(**{field: 1}))


def test_feature_row_has_no_label_columns() -> None:
    assert not set(FeatureRow.model_fields) & {
        "future_return",
        "future_max_up_move",
        "future_max_down_move",
        "future_trend_change",
        "label",
        "label_version",
    }
    assert set(FEATURE_COLUMNS) == set(FeatureRow.model_fields)


def test_feature_row_rejects_unknown_categorical_value() -> None:
    with pytest.raises(ValidationError):
        FeatureRow(**make_feature(nifty_regime="MOONISH"))


# ---------------------------------------------------------------------------
# Labels, trades, manifests
# ---------------------------------------------------------------------------
def test_label_row_accepts_valid_payload() -> None:
    label = LabelRow(**make_label())
    assert label.label is LabelClass.UP_CONTINUATION
    assert set(LABEL_COLUMNS) == set(LabelRow.model_fields)


def test_label_row_enforces_excursion_signs() -> None:
    with pytest.raises(ValidationError, match="future_max_up_move"):
        LabelRow(**make_label(future_max_up_move=-0.01))
    with pytest.raises(ValidationError, match="future_max_down_move"):
        LabelRow(**make_label(future_max_down_move=0.01))


def test_label_row_requires_positive_horizon() -> None:
    with pytest.raises(ValidationError):
        LabelRow(**make_label(horizon_minutes=0))


def test_trade_record_accepts_valid_payload() -> None:
    trade = TradeRecord(**make_trade())
    assert set(TRADE_COLUMNS) == set(TradeRecord.model_fields)
    assert trade.quantity == 75


def test_trade_record_rejects_exit_before_entry() -> None:
    with pytest.raises(ValidationError, match="exit_timestamp"):
        TradeRecord(**make_trade(exit_timestamp=TS - timedelta(minutes=1)))


def test_trade_record_rejects_zero_quantity() -> None:
    with pytest.raises(ValidationError):
        TradeRecord(**make_trade(quantity=0))


def test_run_manifest_defaults_to_started() -> None:
    manifest = RunManifest(run_id="r1", app_name="anjaneeya_capture", started_at=TS)
    assert manifest.status is RunStatus.STARTED
    assert manifest.package_versions == {}
    assert manifest.artifacts == []


def test_contract_identity_is_all_or_nothing() -> None:
    with pytest.raises(ValidationError, match="Partial option contract identity"):
        ContractIdentity(expiry=EXPIRY)
    identity = ContractIdentity(expiry=EXPIRY, strike=25000.0, option_type="PE")
    assert identity.is_option is True
    assert identity.key == ("NIFTY", EXPIRY, 25000.0, "PE")


def test_option_chain_snapshot_requires_full_identity() -> None:
    from core.schemas import OptionChainSnapshotRow

    snapshot = OptionChainSnapshotRow(**make_snapshot())
    assert snapshot.option_type is OptionType.CE
    assert set(OPTION_CHAIN_COLUMNS) == set(OptionChainSnapshotRow.model_fields)


# ---------------------------------------------------------------------------
# Index history / instrument master / coverage ledger (Phase 2 additions)
# ---------------------------------------------------------------------------
def test_index_history_accepts_a_daily_bar() -> None:
    from core.schemas import IndexHistoryRow

    bar = IndexHistoryRow(**make_index_history(timeframe="1d", fetch_timestamp=TS))
    assert bar.timeframe == "1d"
    assert bar.source is Source.ZERODHA_REST
    assert bar.oi is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"high": 24990.0},  # high below the body
        {"low": 25010.0},  # low above the body
        {"open": 25000.0, "close": 25005.0, "high": 25004.0},
        {"fetch_timestamp": TS - timedelta(seconds=1)},
    ],
)
def test_index_history_rejects_impossible_bars(overrides) -> None:
    from core.schemas import IndexHistoryRow

    with pytest.raises(ValidationError):
        IndexHistoryRow(**make_index_history(**overrides))


def test_instrument_row_normalises_exchange_and_requires_option_identity() -> None:
    from core.schemas import InstrumentRow

    instrument = InstrumentRow(**make_instrument(exchange="nfo"))
    assert instrument.exchange == "NFO"
    assert instrument.is_option is True

    with pytest.raises(ValidationError, match="needs both expiry and strike"):
        InstrumentRow(**make_instrument(strike=None))
    with pytest.raises(ValidationError, match="needs an expiry date"):
        InstrumentRow(**make_instrument(instrument_type="FUT", expiry=None, strike=0.0))


def test_index_instrument_needs_no_expiry_or_strike() -> None:
    from core.schemas import InstrumentRow

    index = InstrumentRow(
        **make_instrument(
            exchange="NSE",
            tradingsymbol="NIFTY 50",
            name="NIFTY 50",
            segment="INDICES",
            instrument_type=None,
            expiry=None,
            strike=None,
            lot_size=None,
        )
    )
    assert index.is_option is False
    assert index.expiry is None


def test_coverage_row_validates_the_ledger_invariants() -> None:
    from core.schemas import CoverageRow

    row = CoverageRow(**make_coverage())
    assert row.quality_flag is QualityFlag.OK
    assert row.coverage_ratio is None

    with pytest.raises(ValidationError, match="cannot precede"):
        CoverageRow(**make_coverage(first_timestamp=TS, last_timestamp=TS - timedelta(minutes=1)))
    with pytest.raises(ValidationError, match="exceeds expected_rows"):
        CoverageRow(**make_coverage(expected_rows=10, actual_rows=11))
    with pytest.raises(ValidationError):
        CoverageRow(**make_coverage(coverage_ratio=1.5))


def test_out_of_session_is_a_first_class_quality_flag() -> None:
    """Ticks outside the session are recorded with their own flag (never dropped)."""
    tick = TickRow(**make_tick(quality_flag="OUT_OF_SESSION"))
    assert tick.quality_flag is QualityFlag.OUT_OF_SESSION
