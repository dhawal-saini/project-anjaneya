"""Option implied volatility and greeks (spec 16, Stage 3.3).

Spec 16 says: use the greeks the broker sends *if it sends them*, otherwise compute
them with Black-Scholes for European, cash-settled options. The capture layer stores
no greeks (Zerodha's feed carries none for NFO options), so every
``iv``/``delta``/``gamma``/``theta``/``vega`` in this platform is **computed here**.

The model, in the words a later reader needs
--------------------------------------------
Black-Scholes-Merton with a continuous dividend yield ``q``::

    d1 = (ln(S/K) + (r - q + vol^2 / 2) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    call = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
    put  = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

* ``S`` - the **NIFTY index** level at the bar's *close instant* (never the bar's
  open time: see the alignment note in ``core/features/feature_builder.py``).
* ``K`` - the contract's strike, ``T`` - time to expiry in ACT/365 years.
* ``r`` - ``greeks.risk_free_rate`` from ``config/indicators.yaml``. **It is a
  configured constant, not a fetched curve**, because a stored IV is only comparable
  across runs if the rate that produced it is pinned. The shipped baseline (6.5%) is
  the rounded India 10-year G-Sec yield.
* ``q`` - ``greeks.dividend_yield``, shipped as **0.0**: dividends are ignored. That
  is a documented approximation - NIFTY options are in practice priced off the futures
  (which carry the cost of carry), so a zero-dividend spot model absorbs the basis into
  the IV instead of modelling it. Change it only deliberately.
* ``T`` - measured from the bar's **close instant** to the **expiry session end**
  (``calendar.session_end(expiry)``, so a half day is handled), divided by
  ``greeks.year_days`` (365) and floored at ``greeks.min_time_to_expiry_minutes`` (one
  session). The floor matters: the last bars of an expiry day would otherwise divide by
  ~zero and report an IV that is really float noise.

Units of the stored columns (fixed here, because a silent convention change would
invalidate every model trained on them):

===========  ==========================================================================
``iv``       annualised, continuously compounded, as a **fraction** (0.14 = 14%)
``delta``    dV/dS, per 1 index point (call in [0, 1], put in [-1, 0])
``gamma``    d2V/dS2, per index point squared (>= 0 for a long option)
``vega``     dV/dvol **per 1 percentage point** of vol (the annual figure / 100)
``theta``    dV/dtime **per calendar day** (the annual figure / ``year_days``, <= 0)
===========  ==========================================================================

Limitations (spec 16 asks for them to be written down)

* A zero-dividend European model on the *spot* index approximates a market that prices
  off the futures; the residual basis lands in the IV.
* ``close`` is a traded price, not a mid: on a thin strike the bar's last trade can be
  several ticks off fair value and the IV inherits that. ``price_field: hl2`` exists for
  that reason, and records itself in the run summary when used.
* Volatility is assumed constant over the option's remaining life (no skew, no term
  structure): the IV is a *snapshot*, and minutes apart the same contract can
  legitimately show a very different one.
* Greek values are model outputs, not observations. They are as good as ``iv`` is.
* A price outside the ``[volatility_floor, volatility_ceiling]`` bracket has no IV under
  this model; the columns stay ``NULL`` - the honest answer - and the summary counts it.

What this module is *not*
-------------------------
It performs no I/O, reads no database and knows nothing about candles, contracts or
provenance: frames in, frames out. ``apps/anjaneeya_engine/enrichment.py`` owns the
alignment (which NIFTY level belongs to which bar) and ``--mode enrich-greeks`` owns
persisting the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

__all__ = [
    "GREEKS_COLUMNS",
    "MODEL_GREEK_COLUMNS",
    "GreeksModel",
    "black_scholes_greeks",
    "black_scholes_price",
    "enrich_option_frame",
    "implied_volatility",
    "option_price_series",
    "time_to_expiry_years",
]

#: Every column Stage 3.3 fills on a candle row (and which the feature row then
#: carries through as spec 16's option context).
GREEKS_COLUMNS = (
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "days_to_expiry",
    "distance_from_atm",
)

#: The subset produced by the pricing model itself. ``days_to_expiry`` comes from the
#: calendar and ``distance_from_atm`` from the strike and the spot - both are written by
#: :func:`enrich_option_frame`, but inventing either here would create a second
#: definition of a number the rest of the platform already owns.
MODEL_GREEK_COLUMNS = ("iv", "delta", "gamma", "theta", "vega")

#: Accepted spellings of a call / put option type (case-insensitive).
_CALL_TYPES = ("CE", "CALL")
_PUT_TYPES = ("PE", "PUT")

#: Slack used when testing whether a price sits inside the no-arbitrage band. The
#: comparison is on floats whose magnitudes are in rupees, so an absolute epsilon is the
#: right tool; a price exactly *on* the boundary has no IV and stays ``NULL``.
_BOUNDARY_SLACK = 1e-12

_SQRT_TWO_PI = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class GreeksModel:
    """The (documented) pricing assumptions behind every stored IV.

    Built from ``config/indicators.yaml``'s ``greeks`` section -
    :meth:`from_config` is the only place that knows the config field names, so the
    model is constructor-injectable in tests without a config file.
    """

    risk_free_rate: float = 0.065
    dividend_yield: float = 0.0
    price_field: str = "close"
    year_days: int = 365
    min_time_to_expiry_minutes: float = 375.0
    volatility_floor: float = 0.005
    volatility_ceiling: float = 5.0
    tolerance: float = 1e-6
    max_iterations: int = 100
    name: str = "black_scholes"

    @property
    def year_seconds(self) -> float:
        """Seconds in the model's year (ACT/365 by default)."""
        return float(self.year_days) * 24.0 * 3600.0

    @property
    def min_time_to_expiry_years(self) -> float:
        """The floor applied to every ``T`` (one session's worth of a year)."""
        return float(self.min_time_to_expiry_minutes) * 60.0 / self.year_seconds

    @classmethod
    def from_config(cls, config: Any) -> GreeksModel:
        """Build the model from the ``greeks`` section of an ``IndicatorsConfig``."""
        section = getattr(config, "greeks", config)
        return cls(
            risk_free_rate=float(section.risk_free_rate),
            dividend_yield=float(section.dividend_yield),
            price_field=str(section.price_field),
            year_days=int(section.year_days),
            min_time_to_expiry_minutes=float(section.min_time_to_expiry_minutes),
            volatility_floor=float(section.volatility_floor),
            volatility_ceiling=float(section.volatility_ceiling),
            tolerance=float(section.iv_tolerance),
            max_iterations=int(section.iv_max_iterations),
            name=str(getattr(section, "model", "black_scholes")),
        )

    def describe(self) -> dict[str, Any]:
        """The assumptions, as recorded in the run summary (reproducibility)."""
        return {
            "model": self.name,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "price_field": self.price_field,
            "year_days": self.year_days,
            "min_time_to_expiry_minutes": self.min_time_to_expiry_minutes,
            "volatility_floor": self.volatility_floor,
            "volatility_ceiling": self.volatility_ceiling,
            "iv_tolerance": self.tolerance,
            "iv_max_iterations": self.max_iterations,
        }


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _normal_cdf(value: np.ndarray) -> np.ndarray:
    """Standard normal CDF, accurate to machine precision (``scipy.special.ndtr``)."""
    return ndtr(value)


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    """Standard normal density."""
    return np.exp(-0.5 * value * value) / _SQRT_TWO_PI


def _floats(*values: Any) -> tuple[np.ndarray, ...]:
    """Coerce every argument to a float array (objects/``None`` become ``NaN``).

    A scalar stays a 0-dimensional array, so a scalar call returns a scalar and a
    whole-column call returns an array - the caller never has to know which it passed.
    """
    coerced: list[np.ndarray] = []
    for value in values:
        if value is None:
            coerced.append(np.asarray(np.nan))
            continue
        try:
            numeric = pd.to_numeric(value, errors="coerce")
        except (TypeError, ValueError):  # pragma: no cover - defensive
            numeric = pd.to_numeric(pd.Series(value), errors="coerce")
        coerced.append(np.asarray(numeric, dtype="float64"))
    return tuple(coerced)


def _option_masks(option_type: Any, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """``(is_call, is_put, unknown)`` masks for a scalar or array option type.

    A *scalar* unknown type is a programming error and raises; an *array* containing
    unknown entries (a NULL column, a hand-edited row) is data, so those entries are
    simply marked unknown and their greeks stay ``NULL``.
    """
    values = np.asarray(option_type, dtype=object)
    empty = np.zeros(shape, dtype=bool)
    if values.ndim == 0:
        text = str(values.item()).strip().upper()
        if text in _CALL_TYPES:
            return np.ones(shape, dtype=bool), empty.copy(), empty.copy()
        if text in _PUT_TYPES:
            return empty.copy(), np.ones(shape, dtype=bool), empty.copy()
        raise ValueError(
            f"unknown option_type {option_type!r}; expected a call "
            f"({'/'.join(_CALL_TYPES)}) or a put ({'/'.join(_PUT_TYPES)})"
        )
    labels = np.array(
        ["" if value is None else str(value).strip().upper() for value in values.ravel()],
        dtype=object,
    ).reshape(values.shape)
    is_call = np.isin(labels, _CALL_TYPES)
    is_put = np.isin(labels, _PUT_TYPES)
    return is_call, is_put, ~(is_call | is_put)


def _time_fields(
    spot: Any, strike: Any, time_to_expiry: Any, volatility: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Broadcast the four model inputs and mask the entries the model can use."""
    spot_values, strike_values, years, vols = _floats(spot, strike, time_to_expiry, volatility)
    shape = np.broadcast_shapes(
        spot_values.shape, strike_values.shape, years.shape, vols.shape
    )
    spot_values = np.broadcast_to(spot_values, shape).astype("float64")
    strike_values = np.broadcast_to(strike_values, shape).astype("float64")
    years = np.broadcast_to(years, shape).astype("float64")
    vols = np.broadcast_to(vols, shape).astype("float64")
    usable = (
        np.isfinite(spot_values)
        & np.isfinite(strike_values)
        & np.isfinite(years)
        & np.isfinite(vols)
        & (spot_values > 0.0)
        & (strike_values > 0.0)
        & (years > 0.0)
        & (vols > 0.0)
    )
    return spot_values, strike_values, years, vols, usable, shape


def _safe(value: np.ndarray, usable: np.ndarray, replacement: float) -> np.ndarray:
    """Replace unusable entries so the arithmetic below never sees 0/NaN/negative."""
    return np.where(usable, value, replacement)


def _d1_d2(
    spot: np.ndarray,
    strike: np.ndarray,
    years: np.ndarray,
    volatility: np.ndarray,
    model: GreeksModel,
    usable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``d1``/``d2``, with unusable entries replaced by a harmless constant."""
    spot_ = _safe(spot, usable, 1.0)
    strike_ = _safe(strike, usable, 1.0)
    years_ = _safe(years, usable, model.min_time_to_expiry_years)
    vol_ = _safe(volatility, usable, 0.2)
    root = vol_ * np.sqrt(years_)
    d1 = (
        np.log(spot_ / strike_)
        + (model.risk_free_rate - model.dividend_yield + 0.5 * vol_ * vol_) * years_
    ) / root
    return d1, d1 - root


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def black_scholes_price(
    spot: Any,
    strike: Any,
    time_to_expiry: Any,
    volatility: Any,
    *,
    option_type: Any,
    model: GreeksModel,
) -> np.ndarray:
    """European call/put price under Black-Scholes-Merton (continuous yield).

    Every argument broadcasts; an entry whose inputs are missing (``NaN``), zero or
    negative - or whose ``option_type`` is neither a call nor a put - prices as
    ``NaN`` rather than as a number that would then be trusted.

    A *scalar* unknown ``option_type`` raises :class:`ValueError` (that is a bug in
    the calling code, not data); an array may contain unknown entries.
    """
    spot_values, strike_values, years, vols = _floats(spot, strike, time_to_expiry, volatility)
    spot_values, strike_values, years, vols, usable, shape = _time_fields(
        spot_values, strike_values, years, vols
    )
    is_call, _, unknown = _option_masks(option_type, shape)
    usable &= ~unknown
    d1, d2 = _d1_d2(spot_values, strike_values, years, vols, model, usable)

    years_ = _safe(years, usable, model.min_time_to_expiry_years)
    spot_ = _safe(spot_values, usable, 1.0)
    strike_ = _safe(strike_values, usable, 1.0)
    carry = np.exp(-model.dividend_yield * years_)
    discount = np.exp(-model.risk_free_rate * years_)
    call = spot_ * carry * _normal_cdf(d1) - strike_ * discount * _normal_cdf(d2)
    put = strike_ * discount * _normal_cdf(-d2) - spot_ * carry * _normal_cdf(-d1)
    price = np.where(is_call, call, put)
    return np.where(usable, price, np.nan)


def black_scholes_greeks(
    spot: Any,
    strike: Any,
    time_to_expiry: Any,
    volatility: Any,
    *,
    option_type: Any,
    model: GreeksModel,
) -> pd.DataFrame:
    """``delta``/``gamma``/``theta``/``vega`` for a grid of inputs.

    Returns a ``DataFrame`` with exactly :data:`MODEL_GREEK_COLUMNS` minus ``iv``
    (the inputs' volatility is given, not solved), in the market conventions the
    module docstring fixes: delta per index point, gamma per point squared, vega per
    **1 percentage point** of vol, theta per **calendar day**.
    """
    spot_values, strike_values, years, vols = _floats(spot, strike, time_to_expiry, volatility)
    spot_values, strike_values, years, vols, usable, shape = _time_fields(
        spot_values, strike_values, years, vols
    )
    is_call, _, unknown = _option_masks(option_type, shape)
    usable &= ~unknown
    d1, d2 = _d1_d2(spot_values, strike_values, years, vols, model, usable)

    years_ = _safe(years, usable, model.min_time_to_expiry_years)
    spot_ = _safe(spot_values, usable, 1.0)
    strike_ = _safe(strike_values, usable, 1.0)
    vol_ = _safe(vols, usable, 0.2)
    root = vol_ * np.sqrt(years_)
    carry = np.exp(-model.dividend_yield * years_)
    discount = np.exp(-model.risk_free_rate * years_)
    density = _normal_pdf(d1)

    call_delta = carry * _normal_cdf(d1)
    put_delta = carry * (_normal_cdf(d1) - 1.0)
    delta = np.where(is_call, call_delta, put_delta)

    gamma = carry * density / (spot_ * root)

    vega_per_unit = spot_ * carry * density * np.sqrt(years_)
    # Stored per percentage point of vol: the number a trader reads off a screen.
    vega = vega_per_unit / 100.0

    # Theta's decay term is dV/dT for the diffusion part: -S*phi(d1)*sigma / (2*sqrt(T)).
    # Dividing by ``2*root`` instead (sigma*sqrt(T)) silently scaled theta by 1/sigma.
    decay = -spot_ * carry * density * vol_ / (2.0 * np.sqrt(years_))
    call_theta = decay - model.risk_free_rate * strike_ * discount * _normal_cdf(d2) + (
        model.dividend_yield * spot_ * carry * _normal_cdf(d1)
    )
    put_theta = decay + model.risk_free_rate * strike_ * discount * _normal_cdf(-d2) - (
        model.dividend_yield * spot_ * carry * _normal_cdf(-d1)
    )
    # Stored per calendar day: a weekly option's theta is meaningless as an annual rate.
    theta = np.where(is_call, call_theta, put_theta) / float(model.year_days)

    return pd.DataFrame(
        {
            "delta": np.atleast_1d(np.where(usable, delta, np.nan)),
            "gamma": np.atleast_1d(np.where(usable, gamma, np.nan)),
            "theta": np.atleast_1d(np.where(usable, theta, np.nan)),
            "vega": np.atleast_1d(np.where(usable, vega, np.nan)),
        }
    )


def implied_volatility(
    price: Any,
    spot: Any,
    strike: Any,
    time_to_expiry: Any,
    *,
    option_type: Any,
    model: GreeksModel,
) -> np.ndarray:
    """Solve the implied volatility of each ``price`` (bisection, vectorised).

    The search runs on ``[volatility_floor, volatility_ceiling]`` and is *bracketed*
    before it starts: a price at or below the model value at the floor, or at or above
    the value at the ceiling, has **no** implied volatility under this model and comes
    back as ``NaN`` instead of as the bracket's edge (which is what a battery of
    Newton-Raphson implementations quietly returns). Missing inputs - ``NaN`` price,
    no spot, no strike, zero time, an unknown option type - are ``NaN`` too.

    Deterministic and dependency-free: the bisection is written here rather than
    delegated to a scalar root finder, so a whole contract's bars are solved in one
    vectorised pass.
    """
    prices, spot_values, strike_values, years = _floats(price, spot, strike, time_to_expiry)
    shape = np.broadcast_shapes(
        prices.shape, spot_values.shape, strike_values.shape, years.shape
    )
    prices = np.broadcast_to(prices, shape).astype("float64")
    spot_values = np.broadcast_to(spot_values, shape).astype("float64")
    strike_values = np.broadcast_to(strike_values, shape).astype("float64")
    years = np.broadcast_to(years, shape).astype("float64")
    is_call, _, unknown = _option_masks(option_type, shape)

    usable = (
        np.isfinite(prices)
        & np.isfinite(spot_values)
        & np.isfinite(strike_values)
        & np.isfinite(years)
        & (prices > 0.0)
        & (spot_values > 0.0)
        & (strike_values > 0.0)
        & (years > 0.0)
        & ~unknown
    )
    # ``black_scholes_price`` takes data (an array of labels), not masks, so unknown
    # entries are excluded through ``usable`` and arrive at it as a valid label - their
    # price is simply never used.
    labels = np.where(is_call, "CE", "PE")
    floor, ceiling = model.volatility_floor, model.volatility_ceiling
    at_floor = black_scholes_price(
        spot_values, strike_values, years, np.full(shape, floor), option_type=labels, model=model
    )
    at_ceiling = black_scholes_price(
        spot_values, strike_values, years, np.full(shape, ceiling), option_type=labels, model=model
    )
    inside = (
        usable & (prices > at_floor + _BOUNDARY_SLACK) & (prices < at_ceiling - _BOUNDARY_SLACK)
    )

    low = np.full(shape, floor, dtype="float64")
    high = np.full(shape, ceiling, dtype="float64")
    active = inside.copy()
    for _ in range(int(model.max_iterations)):
        if not active.any():
            break
        middle = 0.5 * (low + high)
        middle_price = black_scholes_price(
            spot_values, strike_values, years, middle, option_type=labels, model=model
        )
        below = middle_price < prices
        low = np.where(active & below, middle, low)
        high = np.where(active & ~below, middle, high)
        active &= (high - low) > model.tolerance

    return np.where(inside, 0.5 * (low + high), np.nan)


def option_price_series(
    frame: pd.DataFrame, *, model: GreeksModel, price_field: str | None = None
) -> pd.Series:
    """The candle field the IV is solved from, as a float ``Series``.

    ``price_field`` defaults to the configured one (``greeks.price_field``). The column
    is *named* rather than assumed so the run summary can record which price produced
    the stored IVs - ``close`` is a traded price, ``hl2``/``ohlc4`` a crude mid.
    """
    field = (price_field or model.price_field or "close").strip().lower()
    if field == "close":
        needed = ("close",)
        values = pd.to_numeric(frame["close"], errors="coerce") if "close" in frame else None
    elif field == "hl2":
        needed = ("high", "low")
        values = (
            (
                pd.to_numeric(frame["high"], errors="coerce")
                + pd.to_numeric(frame["low"], errors="coerce")
            )
            / 2.0
            if set(needed) <= set(frame.columns)
            else None
        )
    elif field == "ohlc4":
        needed = ("open", "high", "low", "close")
        values = (
            sum(pd.to_numeric(frame[column], errors="coerce") for column in needed) / 4.0
            if set(needed) <= set(frame.columns)
            else None
        )
    else:
        raise ValueError(
            f"unknown greeks.price_field {price_field or model.price_field!r}; expected "
            "one of close, hl2, ohlc4"
        )
    if values is None:
        raise ValueError(
            f"the frame is missing column(s) {list(needed)} needed for price_field={field!r}"
        )
    return values.astype("float64").rename("option_price")


def time_to_expiry_years(close_moments: Any, expiry_moments: Any, *, model: GreeksModel) -> np.ndarray:
    """ACT/``year_days`` years from each bar's **close instant** to each expiry moment.

    Both inputs must be timezone-aware (a naive timestamp is a bug everywhere in
    ANJANEYA) and are compared in UTC, so a caller's timezone cannot change the answer.
    The result is floored at :attr:`GreeksModel.min_time_to_expiry_years` - never zero,
    never negative - and is ``NaN`` where either input is missing.
    """
    closes = _aware_moments(close_moments, name="close_moments")
    expiries = _aware_moments(expiry_moments, name="expiry_moments")
    if len(closes) == 1 and len(expiries) != 1:
        closes = closes.repeat(len(expiries))
    elif len(expiries) == 1 and len(closes) != 1:
        expiries = expiries.repeat(len(closes))
    elif len(closes) != len(expiries):
        raise ValueError(
            f"close_moments ({len(closes)}) and expiry_moments ({len(expiries)}) must have "
            "the same length, or one of them must be a single value"
        )
    seconds = (expiries - closes).total_seconds().to_numpy(dtype="float64")
    years = seconds / model.year_seconds
    return np.maximum(years, model.min_time_to_expiry_years)


def _aware_moments(value: Any, *, name: str) -> pd.DatetimeIndex:
    """A UTC ``DatetimeIndex`` from a tz-aware column/array/scalar of datetimes.

    A column that is *entirely* missing carries no timezone to lose (there is no
    timestamp in it), so it is accepted as UTC ``NaT`` - which is what makes an option
    frame with no expiry at all produce ``NULL`` greeks instead of raising.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(value, errors="coerce"))
    if stamps.tz is None:
        if len(stamps) and bool(stamps.isna().all()):
            return stamps.tz_localize("UTC")
        raise ValueError(
            f"{name} must be timezone-aware; ANJANEYA only stores tz-aware timestamps "
            "(see settings.time.forbid_naive_timestamps)"
        )
    return stamps.tz_convert("UTC")


def _aligned(value: Any, length: int, *, name: str) -> np.ndarray:
    """A ``value`` covering ``length`` rows, positionally (never by label).

    A scalar broadcasts; anything else must already be aligned to the frame's rows -
    the caller builds it from that frame - so a length that matches nothing is a bug
    and raises instead of silently mis-assigning greeks to the wrong bars.
    """
    if isinstance(value, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        size = len(value)
        if size != length:
            if size == 1:
                return np.repeat(np.asarray(value), length)
            raise ValueError(
                f"{name} has {size} value(s) but the frame has {length} row(s); pass a "
                "column of the frame (or build the enrichment inputs from it)"
            )
        return np.asarray(value)
    return np.repeat(np.asarray(value), length)


def enrich_option_frame(
    frame: pd.DataFrame,
    *,
    close_moments: Any,
    spot: Any,
    expiry_moments: Any,
    days_to_expiry: Any,
    model: GreeksModel,
    price_field: str | None = None,
) -> pd.DataFrame:
    """Add every :data:`GREEKS_COLUMNS` column to a candle frame (spec 16).

    ``frame`` is one contract's candles at one timeframe, in any order (the result keeps
    the input's index and row order). The four inputs are **aligned positionally** to
    that frame and are deliberately unambiguous about *when* they are true:

    ``close_moments``
        the instant each bar closes (open time + timeframe). The IV is solved for the
        price observed at the close, so the spot that belongs with it is the one at the
        close too - never the bar's open time.
    ``spot``
        the underlying level at that close instant (``NaN`` where unknown).
    ``expiry_moments``
        the moment the contract expires; ``days_to_expiry`` is *not* derived from it,
        because which definition of ``days_to_expiry`` the platform uses is a calendar
        setting (``config/calendar.yaml``).
    ``days_to_expiry``
        the calendar's own value for the bar, carried through untouched.

    Only rows that identify a *contract* (a strike **and** an option type) and whose
    price, spot and time to expiry are all usable get greeks; everything else stays
    ``NULL``, including a price outside the volatility bracket. ``distance_from_atm`` is
    ``strike - spot`` - the same definition the feature builder uses - and is written
    even for rows whose IV cannot be solved, because it needs no model.

    The input frame is never mutated.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"enrich_option_frame expects a DataFrame, got {type(frame).__name__}")
    length = len(frame)
    enriched = frame.copy()
    if length == 0:
        for column in GREEKS_COLUMNS:
            enriched[column] = pd.Series(dtype="float64")
        return enriched

    missing = [column for column in ("strike", "option_type") if column not in frame.columns]
    if missing:
        raise ValueError(
            f"the candle frame is missing column(s) {missing}; without them a row does not "
            "say which contract it prices"
        )

    strikes = pd.to_numeric(_aligned_series(frame["strike"], length, name="strike"), errors="coerce")
    spots = pd.to_numeric(_aligned_series(spot, length, name="spot"), errors="coerce")
    days = pd.to_numeric(
        _aligned_series(days_to_expiry, length, name="days_to_expiry"), errors="coerce"
    )
    option_types = _aligned(frame["option_type"], length, name="option_type")
    prices = option_price_series(frame, model=model, price_field=price_field)
    years = time_to_expiry_years(close_moments, expiry_moments, model=model)

    volatility = implied_volatility(
        prices.to_numpy(),
        spots.to_numpy(),
        strikes.to_numpy(),
        years,
        option_type=option_types,
        model=model,
    )
    greeks = black_scholes_greeks(
        spots.to_numpy(),
        strikes.to_numpy(),
        years,
        volatility,
        option_type=option_types,
        model=model,
    )

    enriched["iv"] = volatility
    for column in greeks.columns:
        enriched[column] = greeks[column].to_numpy()
    enriched["days_to_expiry"] = days.to_numpy()
    # Same definition as the feature builder (and the same sign: positive = strike
    # above the spot, i.e. out of the money for a call).
    enriched["distance_from_atm"] = (strikes - spots).to_numpy()
    return enriched


def _aligned_series(value: Any, length: int, *, name: str) -> pd.Series:
    """``_aligned`` for a value that has to become a pandas ``Series``."""
    if isinstance(value, pd.Series):
        return value if len(value) == length else pd.Series(_aligned(value, length, name=name))
    return pd.Series(_aligned(value, length, name=name))

