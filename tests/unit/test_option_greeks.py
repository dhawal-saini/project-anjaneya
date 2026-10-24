"""Tests for :mod:`core.options.greeks` (IV and greeks, spec 16, Stage 3.3).

What is worth protecting here, in order of how badly it would hurt to get it wrong:

* **The greeks are the derivatives of the price they claim to be.** Every one of them is
  checked against a central finite difference of ``black_scholes_price`` - the one test
  that catches a formula slip (the first version of this module divided theta's decay term
  by ``sigma*sqrt(T)`` instead of ``sqrt(T)``, a 6x error that no test on a single golden
  number would have noticed).
* **Put-call parity.** ``call - put == S*exp(-qT) - K*exp(-rT)`` holds for any correct
  implementation, so it is a model-independent identity rather than a remembered number.
* **The IV solve round-trips, and refuses to lie.** A price produced from a known
  volatility must come back out of the bisection, and a price outside the model's
  no-arbitrage band must return ``NaN`` rather than the bracket's edge.
* **The time convention is explicit.** ``T`` runs from the bar's *close instant* to the
  expiry session end and is floored at one session; ``days_to_expiry`` is the calendar's
  own number and is never recomputed here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.options import (
    GREEKS_COLUMNS,
    MODEL_GREEK_COLUMNS,
    GreeksModel,
    black_scholes_greeks,
    black_scholes_price,
    enrich_option_frame,
    implied_volatility,
    option_price_series,
    time_to_expiry_years,
)

IST = "Asia/Kolkata"

#: A 30-day option, used wherever a derivative needs room to breathe.
T30 = 30.0 / 365.0

#: Strike / volatility grid the parity and round-trip tests walk.
STRIKES = [23000.0, 24500.0, 25000.0, 25500.0, 27000.0]
VOLATILITIES = [0.06, 0.10, 0.15, 0.25, 0.45]


def model(**overrides) -> GreeksModel:
    """The shipped assumptions, with a test overriding what it needs."""
    return GreeksModel(**overrides)


def price(spot, strike, years, volatility, option_type="CE", *, m=None) -> float:
    """Scalar price helper (keeps the assertions readable)."""
    return float(
        black_scholes_price(
            spot, strike, years, volatility, option_type=option_type, model=m or model()
        )
    )


def moments(*values: datetime) -> pd.DatetimeIndex:
    """A tz-aware index of instants from naive local datetimes."""
    return pd.DatetimeIndex([pd.Timestamp(value, tz=IST) for value in values])


# ---------------------------------------------------------------------------
# The price and its derivatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("option_type", ["CE", "PE"])
@pytest.mark.parametrize("strike", STRIKES)
def test_put_call_parity_holds_for_every_strike(strike, option_type) -> None:
    """``call - put`` is the discounted forward, whatever the volatility is."""
    m = model()
    for volatility in VOLATILITIES:
        call = price(25000.0, strike, T30, volatility, "CE", m=m)
        put = price(25000.0, strike, T30, volatility, "PE", m=m)
        expected = 25000.0 - strike * np.exp(-m.risk_free_rate * T30)
        assert call - put == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("option_type", ["CE", "PE"])
@pytest.mark.parametrize("strike", [24000.0, 25000.0, 26000.0])
def test_greeks_match_a_finite_difference_of_the_price(strike, option_type) -> None:
    """Every stored greek is the derivative of the price, in its documented units.

    ``vega`` is per **1 percentage point** of vol and ``theta`` per **calendar day**, so
    the finite differences step by 1 vol point and by 1/365 of a year - mixing those
    conventions up is the kind of mistake a user only notices after training on it.
    """
    m = model()
    spot, volatility = 25000.0, 0.18
    analytic = black_scholes_greeks(
        spot, strike, T30, volatility, option_type=option_type, model=m
    )
    step = 0.5
    numeric_delta = (
        price(spot + step, strike, T30, volatility, option_type, m=m)
        - price(spot - step, strike, T30, volatility, option_type, m=m)
    ) / (2 * step)
    numeric_gamma = (
        price(spot + step, strike, T30, volatility, option_type, m=m)
        - 2 * price(spot, strike, T30, volatility, option_type, m=m)
        + price(spot - step, strike, T30, volatility, option_type, m=m)
    ) / (step * step)
    numeric_vega = (
        price(spot, strike, T30, volatility + 0.01, option_type, m=m)
        - price(spot, strike, T30, volatility - 0.01, option_type, m=m)
    ) / 2.0
    numeric_theta = price(
        spot, strike, T30 - 1.0 / m.year_days, volatility, option_type, m=m
    ) - price(spot, strike, T30, volatility, option_type, m=m)

    assert float(analytic.delta[0]) == pytest.approx(numeric_delta, abs=1e-6)
    assert float(analytic.gamma[0]) == pytest.approx(numeric_gamma, rel=1e-4)
    assert float(analytic.vega[0]) == pytest.approx(numeric_vega, rel=1e-3)
    assert float(analytic.theta[0]) == pytest.approx(numeric_theta, rel=2e-2)


def test_greek_signs_and_identities() -> None:
    """Calls and puts share gamma/vega, their deltas differ by the carry factor."""
    m = model()
    call = black_scholes_greeks(25000.0, 25000.0, T30, 0.15, option_type="CE", model=m)
    put = black_scholes_greeks(25000.0, 25000.0, T30, 0.15, option_type="PE", model=m)

    assert 0.0 < float(call.delta[0]) < 1.0
    assert -1.0 < float(put.delta[0]) < 0.0
    assert float(call.delta[0] - put.delta[0]) == pytest.approx(
        float(np.exp(-m.dividend_yield * T30)), abs=1e-12
    )
    assert float(call.gamma[0]) == pytest.approx(float(put.gamma[0]), rel=1e-12)
    assert float(call.vega[0]) == pytest.approx(float(put.vega[0]), rel=1e-12)
    assert float(call.gamma[0]) > 0.0
    assert float(call.vega[0]) > 0.0
    assert float(call.theta[0]) < 0.0  # a long option loses value as time passes
    assert float(put.theta[0]) < 0.0


def test_deep_wings_still_produce_a_sane_delta() -> None:
    """The far strikes must not produce a nonsense delta."""
    m = model()
    deep_itm = black_scholes_greeks(25000.0, 10000.0, T30, 0.12, option_type="CE", model=m)
    deep_otm = black_scholes_greeks(25000.0, 60000.0, T30, 0.12, option_type="CE", model=m)
    assert float(deep_itm.delta[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(deep_otm.delta[0]) == pytest.approx(0.0, abs=1e-9)


def test_missing_inputs_price_as_nan_instead_of_a_number() -> None:
    """``NaN``/zero/negative inputs must never be priced as if they were real."""
    m = model()
    assert np.isnan(price(np.nan, 25000.0, T30, 0.15, m=m))
    assert np.isnan(price(25000.0, 0.0, T30, 0.15, m=m))
    assert np.isnan(price(25000.0, 25000.0, 0.0, 0.15, m=m))
    assert np.isnan(price(25000.0, 25000.0, T30, 0.0, m=m))
    assert np.isnan(price(25000.0, 25000.0, -T30, 0.15, m=m))
    assert np.isnan(price(25000.0, 25000.0, np.nan, 0.15, m=m))


def test_vectorised_and_scalar_calls_agree() -> None:
    """The vectorised path is the same arithmetic, so it must agree exactly."""
    m = model()
    spots = np.array([24990.0, 25000.0, 25010.0])
    prices = black_scholes_price(spots, 25000.0, T30, 0.15, option_type="CE", model=m)
    scalars = [price(float(spot), 25000.0, T30, 0.15, "CE", m=m) for spot in spots]
    assert list(prices) == pytest.approx(scalars, abs=0.0)


def test_unknown_scalar_option_type_raises_but_array_entries_go_nan() -> None:
    """A typo in code raises; a bad value in data is ``NaN``."""
    m = model()
    with pytest.raises(ValueError, match="unknown option_type"):
        price(25000.0, 25000.0, T30, 0.15, "XX", m=m)
    mixed = black_scholes_price(
        25000.0, 25000.0, T30, 0.15, option_type=np.array(["CE", "XX"]), model=m
    )
    assert np.isfinite(float(mixed[0]))
    assert np.isnan(float(mixed[1]))


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("option_type", ["CE", "PE"])
@pytest.mark.parametrize("strike", STRIKES)
def test_implied_volatility_round_trips(strike, option_type) -> None:
    """Pricing at a known volatility and solving it back recovers that volatility."""
    m = model()
    prices = np.array(
        [
            price(25000.0, strike, T30, volatility, option_type, m=m)
            for volatility in VOLATILITIES
        ]
    )
    solved = implied_volatility(prices, 25000.0, strike, T30, option_type=option_type, model=m)
    assert list(solved) == pytest.approx(VOLATILITIES, abs=1e-5)


def test_implied_volatility_refuses_a_price_outside_the_model() -> None:
    """No IV exists below intrinsic or above the underlying: that is ``NaN``, not an edge."""
    m = model()
    intrinsic = 25000.0 - 26000.0 * np.exp(-m.risk_free_rate * T30)  # in-the-money call
    below = implied_volatility(intrinsic - 1.0, 25000.0, 26000.0, T30, option_type="CE", model=m)
    above = implied_volatility(25000.1, 25000.0, 26000.0, T30, option_type="CE", model=m)
    zero = implied_volatility(0.0, 25000.0, 25000.0, T30, option_type="CE", model=m)
    negative = implied_volatility(-5.0, 25000.0, 25000.0, T30, option_type="CE", model=m)
    assert all(np.isnan(float(value)) for value in (below, above, zero, negative))


def test_implied_volatility_respects_the_configured_bracket() -> None:
    """A price only a 600% vol could explain has no IV in a 500% bracket (and vice versa)."""
    m = model(volatility_ceiling=5.0)
    too_expensive = price(25000.0, 25000.0, T30, 5.0, m=m) + 1.0
    assert np.isnan(
        float(implied_volatility(too_expensive, 25000.0, 25000.0, T30, option_type="CE", model=m))
    )
    floored = model(volatility_floor=0.05)
    too_cheap = price(25000.0, 25000.0, T30, 0.01, m=model())
    assert np.isnan(
        float(
            implied_volatility(
                too_cheap, 25000.0, 25000.0, T30, option_type="CE", model=floored
            )
        )
    )


def test_implied_volatility_returns_nan_for_missing_inputs() -> None:
    """A row without a price, spot, strike or time has no IV."""
    m = model()
    assert np.isnan(
        float(implied_volatility(np.nan, 25000.0, 25000.0, T30, option_type="CE", model=m))
    )
    assert np.isnan(
        float(implied_volatility(120.0, np.nan, 25000.0, T30, option_type="CE", model=m))
    )
    assert np.isnan(
        float(implied_volatility(120.0, 25000.0, np.nan, T30, option_type="CE", model=m))
    )
    assert np.isnan(
        float(implied_volatility(120.0, 25000.0, 25000.0, np.nan, option_type="CE", model=m))
    )


# ---------------------------------------------------------------------------
# Time to expiry
# ---------------------------------------------------------------------------
def test_time_to_expiry_counts_a_year_of_days() -> None:
    """One day before the expiry moment is 1/365 of a year; earlier is proportionally more."""
    m = model()
    close = moments(
        datetime(2026, 9, 30, 15, 30),  # exactly the expiry moment of 2026-10-01
        datetime(2026, 9, 30, 9, 30),  # six hours earlier
        datetime(2026, 9, 29, 15, 30),  # one day earlier
    )
    expiry = moments(datetime(2026, 10, 1, 15, 30))
    years = time_to_expiry_years(close, expiry, model=m)
    assert float(years[2]) == pytest.approx(2.0 / 365.0, abs=1e-12)
    assert float(years[1]) == pytest.approx(30.0 / (24 * 365.0), abs=1e-12)
    assert float(years[0]) == pytest.approx(1.0 / 365.0, abs=1e-12)


def test_time_to_expiry_floor_holds_on_the_expiry_day() -> None:
    """A bar at (or after) the expiry close gets the floor, never zero or a negative."""
    m = model()
    close = moments(
        datetime(2026, 10, 1, 15, 29),  # one minute before the expiry moment
        datetime(2026, 10, 1, 12, 0),  # three and a half hours before it
        datetime(2026, 10, 2, 9, 30),  # after it
    )
    expiry = moments(datetime(2026, 10, 1, 15, 30))
    years = time_to_expiry_years(close, expiry, model=m)
    assert all(float(value) == pytest.approx(m.min_time_to_expiry_years, abs=1e-15) for value in years)
    assert m.min_time_to_expiry_years > 0.0


def test_time_to_expiry_propagates_a_missing_date_as_nan() -> None:
    """No expiry date means no time to expiry - and therefore no greeks."""
    m = model()
    close = moments(datetime(2026, 9, 29, 9, 16))
    expiry = pd.DatetimeIndex([pd.NaT]).tz_localize(IST)
    assert np.isnan(float(time_to_expiry_years(close, expiry, model=m)[0]))


def test_time_to_expiry_rejects_naive_timestamps() -> None:
    """Naive input is a bug everywhere in ANJANEYA, not a default timezone."""
    m = model()
    naive = pd.DatetimeIndex([pd.Timestamp(2026, 9, 29, 9, 16)])
    expiry = moments(datetime(2026, 10, 1, 15, 30))
    with pytest.raises(ValueError, match="timezone-aware"):
        time_to_expiry_years(naive, expiry, model=m)


def test_time_to_expiry_broadcasts_a_single_expiry_and_rejects_mismatches() -> None:
    """One expiry for a whole column is allowed; a length that matches nothing is not."""
    m = model()
    close = moments(datetime(2026, 9, 29, 9, 16), datetime(2026, 9, 29, 9, 17))
    expiry = moments(datetime(2026, 10, 1, 15, 30))
    assert time_to_expiry_years(close, expiry, model=m).shape == (2,)
    with pytest.raises(ValueError, match="same length"):
        time_to_expiry_years(
            close,
            moments(
                datetime(2026, 10, 1, 15, 30),
                datetime(2026, 10, 2, 15, 30),
                datetime(2026, 10, 3, 15, 30),
            ),
            model=m,
        )


def test_time_to_expiry_ignores_the_callers_timezone() -> None:
    """The same instants expressed in another timezone give the same answer."""
    m = model()
    ist = moments(datetime(2026, 9, 29, 9, 16))
    expiry = moments(datetime(2026, 10, 1, 15, 30))
    assert float(time_to_expiry_years(ist, expiry, model=m)[0]) == pytest.approx(
        float(time_to_expiry_years(ist.tz_convert("UTC"), expiry, model=m)[0]), abs=1e-12
    )


# ---------------------------------------------------------------------------
# enrich_option_frame
# ---------------------------------------------------------------------------
def candle_frame(*, rows: int = 3, strike=25000.0, option_type="CE"):
    """A tiny one-contract candle frame whose closes are real prices at 18% vol."""
    m = model()
    start = pd.Timestamp(2026, 9, 29, 9, 15, tz=IST)
    expiry = pd.Timestamp(2026, 10, 1, 15, 30, tz=IST)
    records = []
    for position in range(rows):
        moment = start + timedelta(minutes=position)
        spot = 25000.0 + position * 5.0
        years = float(
            time_to_expiry_years(
                pd.DatetimeIndex([moment + timedelta(minutes=1)]),
                pd.DatetimeIndex([expiry]),
                model=m,
            )[0]
        )
        value = price(spot, strike, years, 0.18, option_type or "CE", m=m)
        records.append(
            {
                "timestamp": moment,
                "timeframe": "1m",
                "tradingsymbol": "NIFTY01OCT2625000CE",
                "strike": strike,
                "expiry": pd.Timestamp(2026, 10, 1).date(),
                "option_type": option_type,
                "open": value,
                "high": value + 1.0,
                "low": value - 1.0,
                "close": value,
                "spot": spot,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "timeframe",
                "tradingsymbol",
                "strike",
                "expiry",
                "option_type",
                "open",
                "high",
                "low",
                "close",
                "spot",
            ]
        )
    return pd.DataFrame(records)


def enrich(frame: pd.DataFrame, *, m=None, spot=None, days=None, price_field=None) -> pd.DataFrame:
    """Run the enrichment with sensible defaults for the aligned inputs."""
    m = m or model()
    timestamps = pd.to_datetime(frame["timestamp"])
    expiry = pd.DatetimeIndex([pd.Timestamp(2026, 10, 1, 15, 30, tz=IST)] * len(frame))
    return enrich_option_frame(
        frame,
        close_moments=(timestamps + timedelta(minutes=1)).to_numpy(),
        spot=frame["spot"] if spot is None else spot,
        expiry_moments=expiry.to_numpy(),
        days_to_expiry=np.full(len(frame), 2.0) if days is None else days,
        model=m,
        price_field=price_field,
    )


def test_enrich_adds_exactly_the_greek_columns_and_keeps_the_input() -> None:
    """The seven columns appear; the original columns and the caller's frame are untouched."""
    frame = candle_frame()
    before = frame.copy()
    enriched = enrich(frame)

    assert list(enriched.columns) == [
        *frame.columns,
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "days_to_expiry",
        "distance_from_atm",
    ]
    assert frame.equals(before)
    assert set(MODEL_GREEK_COLUMNS) == {"iv", "delta", "gamma", "theta", "vega"}
    assert set(GREEKS_COLUMNS) == {
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "days_to_expiry",
        "distance_from_atm",
    }


def test_enrich_recovers_the_volatility_behind_the_prices() -> None:
    """The prices were produced at 18% vol, so the IVs must come back at 18%."""
    enriched = enrich(candle_frame(rows=4))
    assert list(enriched["iv"]) == pytest.approx([0.18] * 4, abs=1e-6)
    assert all(0.0 < float(value) < 1.0 for value in enriched["delta"])
    assert all(float(value) > 0.0 for value in enriched["gamma"])
    assert all(float(value) < 0.0 for value in enriched["theta"])


def test_enrich_uses_the_spot_it_was_given_which_is_the_aligned_one() -> None:
    """``distance_from_atm`` is strike minus the aligned spot, not minus another level."""
    enriched = enrich(candle_frame(rows=2), spot=np.array([24900.0, 25100.0]))
    assert list(enriched["distance_from_atm"]) == pytest.approx([100.0, -100.0])


def test_enrich_carries_the_calendars_days_to_expiry_through() -> None:
    """The DTE column is an input (the calendar owns its definition), never re-derived."""
    enriched = enrich(candle_frame(rows=2), days=np.array([1234.0, 1233.0]))
    assert list(enriched["days_to_expiry"]) == pytest.approx([1234.0, 1233.0])


def test_enrich_leaves_a_row_without_a_contract_null() -> None:
    """A row with no strike or no option type has no model: its greeks stay NULL."""
    frame = candle_frame(rows=2)
    frame["strike"] = [None, 25000.0]
    frame["option_type"] = [None, None]
    enriched = enrich(frame)
    assert enriched["iv"].isna().all()
    assert enriched["delta"].isna().all()
    assert np.isnan(float(enriched["distance_from_atm"].iloc[0]))


def test_enrich_without_a_spot_leaves_the_greeks_null() -> None:
    """No NIFTY level at the bar's close means no IV - and no ATM distance either."""
    enriched = enrich(candle_frame(rows=3), spot=np.full(3, np.nan))
    assert enriched["iv"].isna().all()
    assert enriched["delta"].isna().all()
    assert enriched["distance_from_atm"].isna().all()


def test_enrich_with_an_unknown_option_type_inside_the_frame_goes_null() -> None:
    """A bad option type inside a frame is data: those rows get NULL greeks only."""
    frame = candle_frame(rows=3)
    frame["option_type"] = ["CE", "XX", "PE"]
    enriched = enrich(frame)
    assert not np.isnan(float(enriched["iv"].iloc[0]))
    assert np.isnan(float(enriched["iv"].iloc[1]))
    assert not np.isnan(float(enriched["iv"].iloc[2]))


def test_enrich_an_empty_frame_is_an_empty_frame() -> None:
    """An empty input produces the same shape back, with the greek columns present."""
    enriched = enrich(candle_frame(rows=0))
    assert enriched.empty
    for column in GREEKS_COLUMNS:
        assert column in enriched.columns


def test_enrich_rejects_misaligned_inputs_and_missing_columns() -> None:
    """A length that matches nothing, or a missing key column, raises instead of guessing."""
    frame = candle_frame(rows=3)
    with pytest.raises(ValueError, match="frame has 3 row"):
        enrich(frame, spot=np.array([25000.0, 25010.0]))
    with pytest.raises(ValueError, match="missing column"):
        enrich_option_frame(
            frame.drop(columns=["option_type"]),
            close_moments=frame["timestamp"].to_numpy(),
            spot=frame["spot"],
            expiry_moments=frame["timestamp"].to_numpy(),
            days_to_expiry=1.0,
            model=model(),
        )
    with pytest.raises(TypeError, match="DataFrame"):
        enrich_option_frame(
            [1, 2, 3],
            close_moments=[],
            spot=[],
            expiry_moments=[],
            days_to_expiry=[],
            model=model(),
        )


def test_option_price_series_follows_the_configured_field() -> None:
    """``hl2``/``ohlc4`` read the price from the bar's own fields, ``close`` stays the default."""
    frame = candle_frame(rows=2)
    m = model()
    close = option_price_series(frame, model=m)
    hl2 = option_price_series(frame, model=m, price_field="hl2")
    ohlc4 = option_price_series(frame, model=m, price_field="ohlc4")
    assert float(close.iloc[0]) == pytest.approx(float(frame["close"].iloc[0]))
    assert float(hl2.iloc[0]) == pytest.approx(float(frame["close"].iloc[0]))  # symmetric bar
    assert float(ohlc4.iloc[0]) == pytest.approx(float(frame["close"].iloc[0]))
    assert float(hl2.iloc[0]) - float(frame["low"].iloc[0]) == pytest.approx(1.0)


def test_option_price_series_rejects_an_unknown_field() -> None:
    """A typo in the config is a clear error, not a silent fallback to the close."""
    with pytest.raises(ValueError, match="unknown greeks.price_field"):
        option_price_series(candle_frame(), model=model(), price_field="vwap")
