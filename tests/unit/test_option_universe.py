"""Tests for :mod:`core.options.universe` - spec 17's option selection policy.

The policy is what every later stage assumes: which expiries exist, which strikes
sit around the ATM, and how a reference price becomes a strike. Four of its
properties are worth protecting, because each one fails *quietly*:

* the ATM snap must not drift: an exact midpoint rounds up by a pinned rule
  rather than by whatever the binary floating point happens to produce,
* a strike offset that is not a multiple of the strike interval must be refused
  rather than silently producing a strike the exchange never lists,
* an expired expiry must be dropped (subscribing to it captures nothing),
* the ATM must stay *visible* as an anchor even when a config chooses a ladder
  that does not contain it (``includes_atm``), so a reader never has to assume.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from core.config_loader import load_config
from core.market_calendar import NseCalendar
from core.options.universe import (
    EXPIRY_SELECTORS,
    OPTION_SIDES,
    ContractSpec,
    OptionSelection,
    UniverseResolutionError,
    atm_strike_for,
    build_selection,
    current_monthly_expiry,
    next_monthly_expiry,
    render_symbol,
    select_contracts,
    select_expiries,
    strike_ladder,
    template_pattern,
)
from core.schemas import OptionType

IST = ZoneInfo("Asia/Kolkata")
#: Tuesday 2026-09-29 - the day the capture fixtures use too, and an expiry day
#: under the shipped configuration (Tuesday is the configured expiry weekday).
DAY = date(2026, 9, 29)
#: The weekly expiry of the week containing :data:`DAY`.
WEEK_EXPIRY = DAY
#: The following week's expiry.
NEXT_WEEK_EXPIRY = date(2026, 10, 6)
#: The monthly contract trading through October (last Tuesday of the month).
MONTH_EXPIRY = date(2026, 10, 27)


@pytest.fixture
def config(isolated_paths):
    """The shipped configuration, resolved against isolated paths."""
    return load_config()


@pytest.fixture
def calendar(config) -> NseCalendar:
    return NseCalendar(config.calendar)


# ---------------------------------------------------------------------------
# ATM snapping
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


def test_a_midpoint_rounds_up_and_the_rule_is_deterministic() -> None:
    """``nearest`` is ``floor(price / interval + 0.5)``: an exact midpoint goes up.

    The pinned midpoint matters because changing it would move the ATM of a live
    universe by one strike; the tie is a documented rule, not an accident of
    floating point.
    """
    assert atm_strike_for(25025.0, interval=50, rounding="nearest") == 25050.0
    assert atm_strike_for(25024.99, interval=50, rounding="nearest") == 25000.0
    assert atm_strike_for(25025.01, interval=50, rounding="nearest") == 25050.0


def test_atm_rejects_unknown_rounding_and_zero_interval() -> None:
    with pytest.raises(ValueError, match="Unknown ATM rounding"):
        atm_strike_for(25000, interval=50, rounding="sideways")
    with pytest.raises(ValueError, match="strike interval"):
        atm_strike_for(25000, interval=0)


# ---------------------------------------------------------------------------
# The strike ladder
# ---------------------------------------------------------------------------
def test_ladder_is_the_atm_plus_each_offset_ascending_and_deduplicated() -> None:
    assert strike_ladder(25000, [100, -100, 0, 100]) == (24900.0, 25000.0, 25100.0)


def test_ladder_accepts_a_config_that_excludes_the_atm() -> None:
    """A strangle is a legitimate ladder; the ATM simply is not in it."""
    assert strike_ladder(25000, [-200, -100, 100, 200]) == (24800.0, 24900.0, 25100.0, 25200.0)


def test_ladder_refuses_an_offset_off_the_strike_grid() -> None:
    with pytest.raises(ValueError, match="not a multiple of the strike interval"):
        strike_ladder(25000, [0, 25], interval=50)


def test_ladder_refuses_nonsense_offsets_and_prices() -> None:
    with pytest.raises(ValueError, match="at least one strike offset"):
        strike_ladder(25000, [])
    with pytest.raises(ValueError, match="not a number"):
        strike_ladder(25000, ["wide"])
    with pytest.raises(ValueError, match="not a finite number"):
        strike_ladder(25000, [math.inf])
    with pytest.raises(ValueError, match="non-positive strike"):
        strike_ladder(50, [-100])


# ---------------------------------------------------------------------------
# Expiries
# ---------------------------------------------------------------------------
def test_expiry_selectors_resolve_to_calendar_dates(calendar: NseCalendar) -> None:
    assert select_expiries(["current_week", "next_week"], day=DAY, calendar=calendar) == [
        WEEK_EXPIRY,
        NEXT_WEEK_EXPIRY,
    ]
    # Day *is* the September monthly expiry, so the current-month selector keeps it;
    # once it has passed, the selector rolls on to the next month's contract.
    assert select_expiries(["current_month"], day=DAY, calendar=calendar) == [WEEK_EXPIRY]
    assert select_expiries(["current_month"], day=date(2026, 10, 1), calendar=calendar) == [
        MONTH_EXPIRY
    ]
    assert select_expiries(["next_month"], day=DAY, calendar=calendar) == [
        calendar.monthly_expiry(2026, 10)
    ]


def test_the_expiry_day_still_trades_and_duplicates_collapse(calendar: NseCalendar) -> None:
    assert select_expiries(["current_week"], day=WEEK_EXPIRY, calendar=calendar) == [WEEK_EXPIRY]
    assert select_expiries(
        ["current_week", "current_week"], day=DAY, calendar=calendar
    ) == [WEEK_EXPIRY]


def test_unknown_expiry_selector_is_a_hard_error(calendar: NseCalendar) -> None:
    with pytest.raises(UniverseResolutionError, match="Unknown expiry selector"):
        select_expiries(["last_quarter"], day=DAY, calendar=calendar)


def test_monthly_expiry_helpers_roll_into_the_next_month(calendar: NseCalendar) -> None:
    """After October's monthly expiry, the live monthly contract is November's."""
    after = date(2026, 10, 30)
    assert current_monthly_expiry(after, calendar=calendar) == calendar.monthly_expiry(2026, 11)
    assert next_monthly_expiry(after, calendar=calendar) == calendar.monthly_expiry(2026, 12)


def test_every_advertised_selector_is_understood(calendar: NseCalendar) -> None:
    """``EXPIRY_SELECTORS`` is the contract the config validates against."""
    for selector in EXPIRY_SELECTORS:
        assert select_expiries([selector], day=DAY, calendar=calendar)


# ---------------------------------------------------------------------------
# Contracts and symbol rendering
# ---------------------------------------------------------------------------
def test_select_contracts_is_the_cross_product_in_a_stable_order() -> None:
    contracts = select_contracts(strikes=[25100.0, 25000.0], expiries=[WEEK_EXPIRY])
    assert [(contract.strike, contract.option_type) for contract in contracts] == [
        (25000.0, OptionType.CE),
        (25000.0, OptionType.PE),
        (25100.0, OptionType.CE),
        (25100.0, OptionType.PE),
    ]
    assert all(isinstance(contract, ContractSpec) for contract in contracts)


def test_select_contracts_collapses_duplicates_and_survives_empty_input() -> None:
    contracts = select_contracts(strikes=[25000.0, 25000.0], expiries=[WEEK_EXPIRY, WEEK_EXPIRY])
    assert len(contracts) == len(OPTION_SIDES)
    assert select_contracts(strikes=[], expiries=[WEEK_EXPIRY]) == ()
    assert select_contracts(strikes=[25000.0], expiries=[]) == ()


def test_contract_key_is_the_attribute_identity_every_table_uses() -> None:
    spec = ContractSpec(expiry=WEEK_EXPIRY, strike=25000, option_type=OptionType.CE)
    assert spec.key == (WEEK_EXPIRY, 25000.0, "CE")
    assert spec.describe() == {
        "expiry": WEEK_EXPIRY.isoformat(),
        "strike": 25000.0,
        "option_type": "CE",
    }


def test_symbol_rendering_and_pattern() -> None:
    rendered = render_symbol(
        "NIFTY{expiry_short}{strike}{option_type}",
        expiry=MONTH_EXPIRY,
        strike=25000,
        option_type="CE",
        monthly_codes={10: "OCT"},
    )
    assert rendered == "NIFTYOCT25000CE"
    spec = ContractSpec(MONTH_EXPIRY, 25000.0, OptionType.CE)
    assert (
        spec.symbol(template="NIFTY{expiry_short}{strike}{option_type}", monthly_codes={10: "OCT"})
        == rendered
    )
    pattern = template_pattern("NIFTY{expiry_short}{strike}{option_type}")
    assert pattern.match("NIFTY06OCT2625000CE") is not None
    assert pattern.match("BANKNIFTY06OCT2625000CE") is None


# ---------------------------------------------------------------------------
# The config-driven selection
# ---------------------------------------------------------------------------
def test_build_selection_follows_the_shipped_config(config, calendar: NseCalendar) -> None:
    selection = build_selection(
        universe_config=config.option_universe, reference_price=25012.0, day=DAY, calendar=calendar
    )
    assert isinstance(selection, OptionSelection)
    assert selection.underlying == "NIFTY"
    assert selection.exchange == "NFO"
    assert selection.reference_label == "spot"
    assert selection.atm_strike == 25000.0
    assert selection.strikes == (24800.0, 24900.0, 25000.0, 25100.0, 25200.0)
    assert selection.offsets == (-200.0, -100.0, 0.0, 100.0, 200.0)
    assert selection.includes_atm is True
    # DAY is an expiry Tuesday, so the shipped ``current_month`` selector resolves to the
    # same date as ``current_week`` and is de-duplicated: still two expiries.
    assert selection.expiries == (WEEK_EXPIRY, NEXT_WEEK_EXPIRY)
    assert len(selection.contracts) == 20  # 2 expiries x 5 strikes x 2 sides
    assert selection.is_empty is False


def test_build_selection_reports_a_ladder_without_the_atm(config, calendar: NseCalendar) -> None:
    universe_config = config.option_universe
    universe_config.strikes_offset = [-200, -100, 100, 200]
    selection = build_selection(
        universe_config=universe_config, reference_price=25000.0, day=DAY, calendar=calendar
    )
    assert selection.includes_atm is False
    assert selection.offsets == (-200.0, -100.0, 100.0, 200.0)


def test_build_selection_records_the_label_of_the_price_it_was_given(
    config, calendar: NseCalendar
) -> None:
    """The caller states where the price came from; the selection never assumes."""
    config.option_universe.atm.reference = "futures"
    selection = build_selection(
        universe_config=config.option_universe, reference_price=25052.0, day=DAY, calendar=calendar
    )
    assert selection.reference_label == "futures"
    assert selection.atm_strike == 25050.0


@pytest.mark.parametrize("price", [0.0, -25000.0, float("nan"), float("inf"), "cheap"])
def test_build_selection_refuses_a_price_that_cannot_place_an_atm(
    config, calendar: NseCalendar, price
) -> None:
    with pytest.raises(ValueError, match="reference price"):
        build_selection(
            universe_config=config.option_universe, reference_price=price, day=DAY, calendar=calendar
        )


def test_describe_is_json_safe_and_says_what_was_selected(config, calendar: NseCalendar) -> None:
    created = datetime(2026, 9, 29, 9, 15, tzinfo=IST)
    selection = build_selection(
        universe_config=config.option_universe,
        reference_price=25012.0,
        day=DAY,
        calendar=calendar,
        created_at=created,
    )
    described = selection.describe()
    assert described["atm_strike"] == 25000.0
    assert described["offsets"] == [-200.0, -100.0, 0.0, 100.0, 200.0]
    assert described["strikes"] == [24800.0, 24900.0, 25000.0, 25100.0, 25200.0]
    assert described["expiries"] == [WEEK_EXPIRY.isoformat(), NEXT_WEEK_EXPIRY.isoformat()]
    assert described["contracts"] == 20
    assert described["includes_atm"] is True
    assert described["created_at"] == created.isoformat()
    assert described["reference_label"] == "spot"
    assert described["strike_interval"] == 50.0
