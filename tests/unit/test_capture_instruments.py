"""Tests for the instrument master, dump cache and option-universe resolution."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.instruments import (
    Instrument,
    InstrumentDumpUnavailable,
    InstrumentMaster,
    UniverseResolutionError,
    atm_strike_for,
    build_option_universe,
    dump_path,
    futures_contract,
    latest_cached_dump,
    load_dump_file,
    load_instrument_dump,
    parse_dump,
    render_symbol,
    resolve_atm_reference,
    save_dump_file,
    select_expiries,
    template_pattern,
    to_instrument_rows,
)
from core.config_loader import load_config
from core.market_calendar import NseCalendar
from tests.support.capture import (
    DAY,
    MONTH_EXPIRY,
    NEXT_WEEK_EXPIRY,
    NIFTY_FUT_TOKEN,
    NIFTY_SPOT_TOKEN,
    STRIKES,
    WEEK_EXPIRY,
    FakeKite,
    RecordingLogger,
    make_dump,
    make_master,
)

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 29, 8, 30, tzinfo=IST)
RUN_ID = "anjaneeya_capture_20260929-083000_abcdef12"


@pytest.fixture
def config(isolated_paths):
    """The shipped configuration, resolved against isolated paths."""
    return load_config()


@pytest.fixture
def calendar(config) -> NseCalendar:
    return NseCalendar(config.calendar)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_dump_normalises_provider_quirks() -> None:
    """Kite reports NaN/'' for an index's expiry/strike/lot size."""
    master = make_master()
    index = master.get(NIFTY_SPOT_TOKEN)
    assert index is not None
    assert index.name == "NIFTY 50"
    assert index.expiry is None
    assert index.strike is None
    assert index.lot_size is None
    assert index.instrument_type is None
    assert index.is_index is True
    assert index.tick_size == pytest.approx(0.05)


def test_parse_dump_counts_unusable_rows() -> None:
    rows = [
        {"instrument_token": 0, "exchange": "NSE", "tradingsymbol": "X"},
        {"instrument_token": 1, "exchange": "NSE"},
        {"instrument_token": 2, "exchange": "NSE", "tradingsymbol": "OK"},
    ]
    instruments, rejected = parse_dump(rows)
    assert rejected == 2
    assert [item.instrument_token for item in instruments] == [2]


def test_to_instrument_rows_skips_contracts_that_break_the_row_contract() -> None:
    valid = Instrument(
        instrument_token=1,
        exchange="nfo",
        tradingsymbol="A",
        instrument_type="FUT",
        expiry=WEEK_EXPIRY,
    )
    broken = Instrument(instrument_token=2, exchange="NFO", tradingsymbol="B", instrument_type="CE")
    rows = to_instrument_rows([valid, broken], run_id=RUN_ID, fetched_at=NOW)
    assert len(rows) == 1
    assert rows[0].exchange == "NFO"  # normalised by the row model
    assert rows[0].run_id == RUN_ID


# ---------------------------------------------------------------------------
# Master lookups
# ---------------------------------------------------------------------------
def test_master_looks_contracts_up_by_identity_not_by_symbol() -> None:
    master = make_master()
    option = master.option(
        exchange="NFO", name="NIFTY", expiry=WEEK_EXPIRY, strike=25000, option_type="CE"
    )
    assert option is not None
    assert option.strike == 25000.0
    assert option.option_type is not None and option.option_type.value == "CE"
    assert (
        master.option(exchange="NFO", name="NIFTY", expiry=WEEK_EXPIRY, strike=25999, option_type="CE")
        is None
    )
    assert master.option_expiries(exchange="NFO", name="NIFTY") == [WEEK_EXPIRY, NEXT_WEEK_EXPIRY]
    assert master.option_strikes(exchange="NFO", name="NIFTY", expiry=WEEK_EXPIRY) == list(STRIKES)


def test_master_symbol_and_token_lookups() -> None:
    master = make_master()
    assert master.get(NIFTY_SPOT_TOKEN).tradingsymbol == "NIFTY 50"
    assert master.get_by_symbol("nse", "NIFTY 50").instrument_token == NIFTY_SPOT_TOKEN
    assert master.get_by_symbol("NSE", "NOT LISTED") is None
    assert len(master) == len(master.tokens)


def test_master_futures_are_ordered_and_filterable() -> None:
    master = make_master()
    futures = master.futures(exchange="NFO", name="NIFTY", on_or_after=DAY)
    assert [f.instrument_token for f in futures] == [NIFTY_FUT_TOKEN]
    assert futures[0].expiry == MONTH_EXPIRY
    assert (
        master.futures(exchange="NFO", name="NIFTY", on_or_after=MONTH_EXPIRY + timedelta(days=1)) == []
    )


def test_master_describe_is_log_safe() -> None:
    description = make_master().describe()
    assert description["instruments"] > 0
    assert description["by_instrument_type"]["CE"] == len(STRIKES) * 2
    assert description["rejected_rows"] == 0


# ---------------------------------------------------------------------------
# ATM + expiries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("price", "rounding", "expected"),
    [
        (25012.0, "nearest", 25000.0),
        (25025.0, "nearest", 25050.0),
        (25025.0, "floor", 25000.0),
        (25025.0, "ceil", 25050.0),
        (25000.0, "nearest", 25000.0),
    ],
)
def test_atm_snapping(price: float, rounding: str, expected: float) -> None:
    assert atm_strike_for(price, interval=50, rounding=rounding) == expected


def test_atm_rejects_unknown_rounding_and_zero_interval() -> None:
    with pytest.raises(ValueError, match="Unknown ATM rounding"):
        atm_strike_for(25000, interval=50, rounding="sideways")
    with pytest.raises(ValueError, match="strike interval"):
        atm_strike_for(25000, interval=0)


def test_expiry_selectors_resolve_to_calendar_dates(calendar: NseCalendar) -> None:
    assert select_expiries(["current_week", "next_week"], day=DAY, calendar=calendar) == [
        WEEK_EXPIRY,
        NEXT_WEEK_EXPIRY,
    ]
    # Day *is* the September monthly expiry, so the current-month selector keeps it;
    # after that date has passed it rolls on to the next month's contract.
    assert select_expiries(["current_month"], day=DAY, calendar=calendar) == [WEEK_EXPIRY]
    assert select_expiries(["current_month"], day=date(2026, 10, 1), calendar=calendar) == [
        MONTH_EXPIRY
    ]
    assert select_expiries(["next_month"], day=DAY, calendar=calendar) == [
        calendar.monthly_expiry(2026, 10)
    ]


def test_expiry_selectors_deduplicate_and_keep_the_expiry_day(calendar: NseCalendar) -> None:
    # On the expiry day itself the contract still trades until the close.
    assert select_expiries(["current_week"], day=WEEK_EXPIRY, calendar=calendar) == [WEEK_EXPIRY]
    assert select_expiries(["current_week", "current_week"], day=DAY, calendar=calendar) == [WEEK_EXPIRY]


def test_unknown_expiry_selector_is_a_hard_error(calendar: NseCalendar) -> None:
    with pytest.raises(UniverseResolutionError, match="Unknown expiry selector"):
        select_expiries(["last_quarter"], day=DAY, calendar=calendar)


# ---------------------------------------------------------------------------
# Symbol templates (a cross-check, never a lookup)
# ---------------------------------------------------------------------------
def test_render_symbol_and_pattern() -> None:
    rendered = render_symbol(
        "NIFTY{expiry_short}{strike}{option_type}",
        expiry=MONTH_EXPIRY,
        strike=25000,
        option_type="CE",
        monthly_codes={10: "OCT"},
    )
    assert rendered == "NIFTYOCT25000CE"
    pattern = template_pattern("NIFTY{expiry_short}{strike}{option_type}")
    assert pattern.match("NIFTY06OCT2625000CE") is not None
    assert pattern.match("BANKNIFTY06OCT2625000CE") is None


# ---------------------------------------------------------------------------
# Universe resolution
# ---------------------------------------------------------------------------
def test_universe_contains_the_atm_window_for_both_expiries(config, calendar) -> None:
    universe = build_option_universe(
        universe_config=config.option_universe,
        master=make_master(),
        reference_price=25012.0,
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
    )
    assert universe.atm_strike == 25000.0
    assert universe.expiries == (WEEK_EXPIRY, NEXT_WEEK_EXPIRY)
    assert len(universe.options) == 20  # 2 expiries x 5 offsets x 2 sides
    assert universe.reference.instrument_token == NIFTY_SPOT_TOKEN
    assert universe.futures is not None and universe.futures.instrument_token == NIFTY_FUT_TOKEN
    assert universe.missing == ()
    assert len(universe.tokens) == 22
    assert universe.symbols[NIFTY_FUT_TOKEN] == "NIFTY26OCTFUT"
    assert universe.reference_label == "spot"
    assert universe.is_empty is False


def test_universe_reports_every_missing_contract_instead_of_hiding_it(config, calendar) -> None:
    master = make_master(strikes=(24900, 25000), expiries=(WEEK_EXPIRY,))
    universe = build_option_universe(
        universe_config=config.option_universe,
        master=master,
        reference_price=25000.0,
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
    )
    # ATM = 25000, offsets [-200, -100, 0, 100, 200]: -100 and 0 exist (4 contracts),
    # -200, +100 and +200 do not, and that gap must be reported rather than papered over.
    assert len(universe.options) == 4
    assert universe.missing == (
        f"no listed options for expiry {NEXT_WEEK_EXPIRY.isoformat()}",
        f"strike 24800 not listed for {WEEK_EXPIRY.isoformat()}",
        f"strike 25100 not listed for {WEEK_EXPIRY.isoformat()}",
        f"strike 25200 not listed for {WEEK_EXPIRY.isoformat()}",
    )
    assert universe.expiries == (WEEK_EXPIRY,)


def test_universe_is_empty_when_the_configured_expiries_are_not_listed(config, calendar) -> None:
    master = make_master(expiries=(MONTH_EXPIRY,))
    universe = build_option_universe(
        universe_config=config.option_universe,
        master=master,
        reference_price=25000.0,
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
    )
    assert universe.is_empty is True
    assert len(universe.missing) == 2  # both configured weekly expiries are absent


def test_universe_refuses_to_build_without_the_index(config, calendar) -> None:
    with pytest.raises(UniverseResolutionError, match="reference instrument"):
        build_option_universe(
            universe_config=config.option_universe,
            master=make_master(exchanges=("NFO",)),
            reference_price=25000.0,
            day=DAY,
            calendar=calendar,
            resolved_at=NOW,
        )


def test_universe_silently_accepts_a_matching_symbol_template(config, calendar) -> None:
    logger = RecordingLogger()
    build_option_universe(
        universe_config=config.option_universe,
        master=make_master(),
        reference_price=25012.0,
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
        logger=logger,
    )
    assert not logger.contains("symbol_templates", "WARNING")


def test_universe_warns_once_when_symbol_templates_drift(config, calendar) -> None:
    templates = config.option_universe.symbol_templates.model_copy(
        update={"weekly": "NIFTY{strike}{option_type}", "monthly": "NIFTY{strike}{option_type}"}
    )
    universe_config = config.option_universe.model_copy(update={"symbol_templates": templates})
    logger = RecordingLogger()
    build_option_universe(
        universe_config=universe_config,
        master=make_master(),
        reference_price=25012.0,
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
        logger=logger,
    )
    assert len(logger.warnings()) == 1
    assert "symbol_templates" in logger.warnings()[0]


# ---------------------------------------------------------------------------
# Dump cache
# ---------------------------------------------------------------------------
def test_dump_cache_round_trip(tmp_path) -> None:
    rows = make_dump()["NSE"]
    path = save_dump_file(rows, cache_dir=tmp_path, exchange="NSE", day=DAY, fetched_at=NOW)
    assert path == dump_path(tmp_path, "NSE", DAY)
    loaded, fetched_at = load_dump_file(path)
    # The dump contains NaN, which is never equal to itself - compare the
    # serialised form so the assertion is meaningful.
    assert json.dumps(loaded) == json.dumps(rows)
    assert fetched_at == NOW
    assert list(tmp_path.glob(".dump-*")) == []  # the temp file was replaced, not left behind


def test_dump_cache_rejects_an_incompatible_layout(tmp_path) -> None:
    path = tmp_path / "NSE_2026-09-29.json"
    path.write_text(
        json.dumps({"format_version": 999, "rows": [], "fetched_at": NOW.isoformat()}),
        encoding="utf-8",
    )
    with pytest.raises(InstrumentDumpUnavailable, match="format_version"):
        load_dump_file(path)


def test_dump_cache_reports_a_corrupt_file(tmp_path) -> None:
    path = tmp_path / "NSE_2026-09-29.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(InstrumentDumpUnavailable, match="cannot read"):
        load_dump_file(path)


def test_latest_cached_dump_picks_the_newest_file(tmp_path) -> None:
    save_dump_file([], cache_dir=tmp_path, exchange="NSE", day=DAY, fetched_at=NOW)
    later = DAY + timedelta(days=1)
    save_dump_file([], cache_dir=tmp_path, exchange="NSE", day=later, fetched_at=NOW)
    assert latest_cached_dump(tmp_path, "NSE") == dump_path(tmp_path, "NSE", later)
    assert latest_cached_dump(tmp_path, "NFO") is None
    assert latest_cached_dump(tmp_path / "missing", "NSE") is None


# ---------------------------------------------------------------------------
# Dump loading policy (cache -> download -> stale cache)
# ---------------------------------------------------------------------------
def test_fresh_cache_is_used_without_calling_the_api(tmp_path) -> None:
    for exchange, rows in make_dump().items():
        save_dump_file(rows, cache_dir=tmp_path, exchange=exchange, day=DAY, fetched_at=NOW)
    client = FakeKite()

    dump = load_instrument_dump(
        client,
        cache_dir=tmp_path,
        exchanges=("NSE", "NFO"),
        ttl_hours=12,
        now=NOW + timedelta(minutes=5),
        logger=RecordingLogger(),
    )
    assert dump.source == "cache"
    assert client.instrument_calls == []
    assert len(dump.master) == 22


def test_stale_cache_triggers_a_download_and_is_refreshed(tmp_path) -> None:
    for exchange, rows in make_dump().items():
        save_dump_file(rows, cache_dir=tmp_path, exchange=exchange, day=DAY, fetched_at=NOW)
    client = FakeKite()
    logger = RecordingLogger()

    dump = load_instrument_dump(
        client,
        cache_dir=tmp_path,
        exchanges=("NSE", "NFO"),
        ttl_hours=12,
        now=NOW + timedelta(hours=13),
        logger=logger,
    )
    assert dump.source == "download"
    assert client.instrument_calls == ["NSE", "NFO"]
    assert dump.fetched_at is not None
    assert logger.contains("downloaded NSE")


def test_force_download_ignores_a_fresh_cache(tmp_path) -> None:
    save_dump_file(make_dump()["NSE"], cache_dir=tmp_path, exchange="NSE", day=DAY, fetched_at=NOW)
    client = FakeKite()
    load_instrument_dump(
        client,
        cache_dir=tmp_path,
        exchanges=("NSE",),
        ttl_hours=12,
        now=NOW,
        force_download=True,
    )
    assert client.instrument_calls == ["NSE"]


def test_offline_without_a_cache_is_an_actionable_error(tmp_path) -> None:
    with pytest.raises(InstrumentDumpUnavailable, match="--mode instruments"):
        load_instrument_dump(
            None,
            cache_dir=tmp_path,
            exchanges=("NSE",),
            ttl_hours=12,
            now=NOW,
        )


def test_offline_falls_back_to_a_stale_cache_with_a_loud_warning(tmp_path) -> None:
    save_dump_file(make_dump()["NSE"], cache_dir=tmp_path, exchange="NSE", day=DAY, fetched_at=NOW)
    logger = RecordingLogger()
    dump = load_instrument_dump(
        None,
        cache_dir=tmp_path,
        exchanges=("NSE",),
        ttl_hours=12,
        now=NOW + timedelta(days=3),
        logger=logger,
    )
    assert dump.source == "cache"
    assert dump.master.get(NIFTY_SPOT_TOKEN) is not None
    assert any("STALE instrument dump" in message for message in logger.warnings())


def test_an_empty_download_never_replaces_a_usable_master(tmp_path) -> None:
    client = FakeKite(dump={"NSE": [], "NFO": []})
    with pytest.raises(InstrumentDumpUnavailable, match="empty instrument dump"):
        load_instrument_dump(
            client,
            cache_dir=tmp_path,
            exchanges=("NSE",),
            ttl_hours=12,
            now=NOW,
            logger=RecordingLogger(),
        )
    assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# The ATM anchor: spot vs futures (spec 17, Stage 3.4)
# ---------------------------------------------------------------------------
def futures_universe_config(config):
    """``option_universe`` with ``atm.reference`` switched to the futures contract."""
    atm = config.option_universe.atm.model_copy(update={"reference": "futures"})
    return config.option_universe.model_copy(update={"atm": atm})


def test_a_spot_reference_anchors_on_the_index(config, calendar) -> None:
    universe = build_option_universe(
        universe_config=config.option_universe,
        master=make_master(),
        reference_price=25012.0,
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
    )
    assert universe.reference_label == "spot"
    assert universe.atm_anchor.instrument_token == NIFTY_SPOT_TOKEN
    assert universe.atm_anchor_token == NIFTY_SPOT_TOKEN
    assert universe.describe()["atm_anchor"] == "NIFTY 50"


def test_a_futures_reference_snaps_the_futures_price_and_anchors_on_the_contract(
    config, calendar
) -> None:
    universe = build_option_universe(
        universe_config=futures_universe_config(config),
        master=make_master(),
        reference_price=25052.0,  # the futures quote, not the index's 25012
        day=DAY,
        calendar=calendar,
        resolved_at=NOW,
    )
    assert universe.reference_label == "futures"
    assert universe.atm_strike == 25050.0
    # The stream must watch the contract whose price placed the ATM, or an
    # intraday re-centre would snap the *index* while the config says "futures".
    assert universe.atm_anchor.instrument_token == NIFTY_FUT_TOKEN
    assert universe.atm_anchor_token == NIFTY_FUT_TOKEN
    assert universe.futures is not None and universe.futures.instrument_token == NIFTY_FUT_TOKEN


def test_a_futures_reference_without_a_listed_contract_is_a_hard_error(config, calendar) -> None:
    """Snapping a spot price while the config says "futures" would mis-place the ATM."""
    dump = make_dump()
    master = InstrumentMaster.from_dump(
        row
        for rows in dump.values()
        for row in rows
        if str(row.get("instrument_type", "")).upper() != "FUT"
    )
    with pytest.raises(UniverseResolutionError, match="no NIFTY futures contract"):
        build_option_universe(
            universe_config=futures_universe_config(config),
            master=master,
            reference_price=25052.0,
            day=DAY,
            calendar=calendar,
            resolved_at=NOW,
        )


def test_the_futures_contract_follows_the_configured_expiry_selector(config, calendar) -> None:
    universe_config = futures_universe_config(config)
    master = make_master()
    nearest = futures_contract(master, universe_config=universe_config, day=DAY)
    assert nearest is not None and nearest.instrument_token == NIFTY_FUT_TOKEN

    # The synthetic dump lists a single futures contract, so asking for the next
    # month must resolve to nothing rather than silently falling back to this one.
    next_month = universe_config.reference_instruments.nifty_futures.model_copy(
        update={"expiry_selector": "next_month"}
    )
    references = universe_config.reference_instruments.model_copy(
        update={"nifty_futures": next_month}
    )
    selector_config = universe_config.model_copy(update={"reference_instruments": references})
    assert futures_contract(master, universe_config=selector_config, day=DAY) is None


def test_resolve_atm_reference_reads_the_label_it_is_given(config, calendar) -> None:
    master = make_master()
    spot = resolve_atm_reference(master, universe_config=config.option_universe, day=DAY)
    assert spot is not None and spot.instrument_token == NIFTY_SPOT_TOKEN
    futures = resolve_atm_reference(
        master, universe_config=config.option_universe, day=DAY, label="futures"
    )
    assert futures is not None and futures.instrument_token == NIFTY_FUT_TOKEN
    assert futures_contract(master, universe_config=config.option_universe, day=DAY) is futures

