"""Option instruments and contract maths.

Implemented (Stage 3.3)

* :mod:`~core.options.option_trend` - spec 10's lightweight per-contract trend
  engine: a short moving-average stack, the alignment/slope/spread/momentum
  readings, and the explicit insufficient-history policies for weekly contracts.
* :mod:`~core.options.greeks` - spec 16's IV/greeks model: Black-Scholes-Merton
  (European, cash-settled) with a configured risk-free rate and dividend
  assumption, a bracketed bisection for the implied volatility, and the frame-level
  enrichment that writes ``iv``/``delta``/``gamma``/``theta``/``vega`` onto candles.
  Every assumption is documented in the module docstring and recorded in the run
  summary.
* :mod:`~core.options.universe` - spec 17's option selection: the ATM anchor (spot or
  futures), the configurable strike ladder, expiry selection, and the
  ``(expiry, strike, option_type)`` contract menu that capture subscribes to and the
  engine reports against.
"""

from __future__ import annotations

from core.options.greeks import (
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
from core.options.option_trend import (
    OPTION_TREND_FEATURE_COLUMNS,
    average_stack,
    option_metrics,
    option_trend_features,
)
from core.options.universe import (
    ATM_ROUNDINGS,
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

__all__ = [
    "ATM_ROUNDINGS",
    "EXPIRY_SELECTORS",
    "GREEKS_COLUMNS",
    "MODEL_GREEK_COLUMNS",
    "OPTION_SIDES",
    "OPTION_TREND_FEATURE_COLUMNS",
    "ContractSpec",
    "GreeksModel",
    "OptionSelection",
    "UniverseResolutionError",
    "atm_strike_for",
    "average_stack",
    "black_scholes_greeks",
    "black_scholes_price",
    "build_selection",
    "current_monthly_expiry",
    "enrich_option_frame",
    "implied_volatility",
    "next_monthly_expiry",
    "option_metrics",
    "option_price_series",
    "option_trend_features",
    "render_symbol",
    "select_contracts",
    "select_expiries",
    "strike_ladder",
    "template_pattern",
    "time_to_expiry_years",
]


