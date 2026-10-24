"""Turn stored candles into IV/greeks (spec 16, Stage 3.3).

What this module owns, and what it deliberately does not
-------------------------------------------------------
:mod:`core.options.greeks` is the *model* - prices, greeks, the implied-volatility
search. It knows nothing about candles, the index or the calendar. This module is the
thin layer that answers the one question the model cannot: **which NIFTY level belongs
to which bar?** That answer is a platform-wide rule, not a local choice:

* a bar's numbers are known at its **close instant** (open time + timeframe), so the
  index level an option bar is priced against is the one from ``merge_asof`` on close
  instants - never the level at the bar's *open* time, which would be a one-bar
  look-ahead;
* the index series is first resampled onto the enrichment timeframe with
  :func:`core.features.feature_builder.aggregate_index_frame`, i.e. the *same* function
  and the *same* wall-clock bucket rule the feature builder uses. A 15m feature row
  therefore sees the NIFTY level a 15m candle's IV was solved against - two definitions
  of "the spot at this bar" would be a quiet source of disagreement between the stored
  greeks and the stored features.

``days_to_expiry`` is the **calendar's** value for the bar's own timestamp, exactly the
one the feature row carries (``config/calendar.yaml`` decides the definition), while the
model's ``T`` is ACT/365 from the close instant to the expiry session end. They are
different quantities on purpose, and both are documented in
:mod:`core.options.greeks`.

Importing this module performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from core.constants import DEFAULT_TIMEZONE
from core.features import aggregate_index_frame, close_moments, timeframe_of
from core.options import (
    GreeksModel,
    black_scholes_price,
    enrich_option_frame,
    option_price_series,
    time_to_expiry_years,
)

__all__ = ["EnrichmentResult", "enrich_frame", "spot_for_bars"]


@dataclass(slots=True)
class EnrichmentResult:
    """What one ``(contract, timeframe, day)`` enrichment pass produced.

    The counters are the point of the dataclass: an enrichment run reports how many bars
    got a real IV, how many could not (no spot, no expiry, or a price outside the model's
    volatility bracket) and how far the solved IVs are from reproducing the observed
    prices. A number that is not reported here is a number nobody checks.
    """

    frame: pd.DataFrame
    bars: int = 0
    solved_iv: int = 0
    unsolved_iv: int = 0
    missing_spot: int = 0
    max_price_error: float = float("nan")

    @property
    def solved_ratio(self) -> float:
        """Share of bars with a solved IV (0.0 when there are no bars)."""
        return float(self.solved_iv) / float(self.bars) if self.bars else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape for the run summary."""
        return {
            "bars": self.bars,
            "solved_iv": self.solved_iv,
            "unsolved_iv": self.unsolved_iv,
            "missing_spot": self.missing_spot,
            "solved_ratio": round(self.solved_ratio, 6),
            "max_price_error": (
                None if not np.isfinite(self.max_price_error) else float(self.max_price_error)
            ),
        }


def _as_date(value: Any) -> date | None:
    """A ``datetime.date`` from whatever the storage layer handed back."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    return value


def spot_for_bars(
    frame: pd.DataFrame,
    *,
    index_frame: pd.DataFrame | None,
    timeframe: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.Series:
    """The underlying level at each bar's **close instant** (``NaN`` when unknown).

    ``index_frame`` is the NIFTY series as stored (any resolution); it is resampled onto
    ``timeframe`` with the feature builder's own bucket rule and then joined backwards on
    close instants, so a 15m index bar can never reach a 1m candle that closed before it.
    The result keeps ``frame``'s index and row order, and a bar with no index bar at or
    before its close gets ``NaN`` - which is what makes that bar's greeks ``NULL`` rather
    than solved against a level from an hour later.
    """
    if frame is None or len(frame) == 0 or "timestamp" not in frame.columns:
        return pd.Series(dtype="float64")
    if index_frame is None or len(index_frame) == 0:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    aggregated = aggregate_index_frame(
        index_frame, timeframe=timeframe, timezone_name=timezone_name
    )
    if aggregated is None or len(aggregated) == 0:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    right = aggregated[["timestamp", "close"]].rename(columns={"close": "spot"}).copy()
    right["_moment"] = close_moments(right, timeframe=timeframe_of(aggregated, default=timeframe))
    # Deterministic when the index series itself carries a duplicate close instant: the
    # newest bar wins.
    right = right.sort_values("_moment", kind="stable").drop_duplicates("_moment", keep="last")

    left = pd.DataFrame(
        {
            "_moment": close_moments(frame, timeframe=timeframe).to_numpy(),
            "_position": np.arange(len(frame)),
        }
    ).sort_values("_moment", kind="stable")
    merged = pd.merge_asof(
        left,
        right[["_moment", "spot"]],
        on="_moment",
        direction="backward",
        allow_exact_matches=True,
    )
    values = pd.to_numeric(merged["spot"], errors="coerce").to_numpy(dtype="float64")
    restored = np.full(len(frame), np.nan, dtype="float64")
    restored[merged["_position"].to_numpy(dtype="int64")] = values
    return pd.Series(restored, index=frame.index, dtype="float64")


def enrich_frame(
    frame: pd.DataFrame,
    *,
    index_frame: pd.DataFrame | None,
    timeframe: str,
    calendar: Any,
    model: GreeksModel,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> EnrichmentResult:
    """Add spec 16's greeks and option context to one contract's candle frame.

    ``calendar`` supplies two things and is asked for both, so the platform has exactly
    one answer to each question: the expiry moment (``session_end(expiry)``, which
    handles a half day) and ``days_to_expiry`` (whose definition is a
    ``config/calendar.yaml`` setting).
    """
    if frame is None or len(frame) == 0:
        return EnrichmentResult(frame=pd.DataFrame() if frame is None else frame.copy())

    length = len(frame)
    spot = spot_for_bars(
        frame, index_frame=index_frame, timeframe=timeframe, timezone_name=timezone_name
    )
    moments = pd.to_datetime(frame["timestamp"], errors="raise")
    if getattr(moments.dt, "tz", None) is None:
        raise ValueError(
            "candle timestamps are naive; ANJANEYA only stores timezone-aware data "
            "(see settings.time.forbid_naive_timestamps)"
        )

    expiries = (
        [_as_date(value) for value in frame["expiry"]] if "expiry" in frame.columns else [None] * length
    )
    expiry_moments = pd.Series(
        [calendar.session_end(value) if value is not None else pd.NaT for value in expiries],
        index=frame.index,
    )
    days = pd.Series(
        [
            calendar.days_to_expiry(moment, expiry) if expiry is not None else np.nan
            for moment, expiry in zip(moments, expiries, strict=False)
        ],
        index=frame.index,
        dtype="float64",
    )

    enriched = enrich_option_frame(
        frame,
        close_moments=close_moments(frame, timeframe=timeframe).to_numpy(),
        spot=spot.to_numpy(),
        expiry_moments=expiry_moments.to_numpy(),
        days_to_expiry=days.to_numpy(),
        model=model,
    )
    result = EnrichmentResult(frame=enriched, bars=length)
    result.solved_iv = int(enriched["iv"].notna().sum())
    result.unsolved_iv = length - result.solved_iv
    is_option = pd.to_numeric(enriched["strike"], errors="coerce").notna()
    result.missing_spot = int((spot.isna() & is_option).sum())
    result.max_price_error = _max_price_error(
        enriched, spot=spot, expiry_moments=expiry_moments, model=model
    )
    return result


def _max_price_error(
    enriched: pd.DataFrame, *, spot: pd.Series, expiry_moments: pd.Series, model: GreeksModel
) -> float:
    """Worst ``|observed price - model price at the solved IV|`` over the frame.

    The model's own self-check: the IV was solved *from* a price, so pricing that same
    price back with the solved volatility must reproduce it. A large number here means
    the bisection did not converge (or its bracket was too narrow) - not that the market
    moved. It is the one QC figure a stored IV can be judged by on its own.
    """
    solved = enriched["iv"].notna()
    if not solved.any():
        return float("nan")
    try:
        prices = option_price_series(enriched, model=model)
    except ValueError:  # pragma: no cover - the caller always passes candle rows
        return float("nan")
    timeframe = _timeframe_of_frame(enriched)
    years = time_to_expiry_years(
        close_moments(enriched, timeframe=timeframe).to_numpy(),
        expiry_moments.to_numpy(),
        model=model,
    )
    model_prices = black_scholes_price(
        spot.to_numpy(),
        pd.to_numeric(enriched["strike"], errors="coerce").to_numpy(),
        years,
        enriched["iv"].to_numpy(),
        option_type=enriched["option_type"].to_numpy(),
        model=model,
    )
    error = np.abs(prices.to_numpy() - model_prices)[solved.to_numpy()]
    finite = error[np.isfinite(error)]
    return float(finite.max()) if finite.size else float("nan")


def _timeframe_of_frame(frame: pd.DataFrame) -> str | None:
    """The frame's own timeframe label, or ``None`` when it does not carry one."""
    if "timeframe" not in frame.columns:
        return None
    labels = {str(label) for label in frame["timeframe"].dropna().unique()}
    return labels.pop() if len(labels) == 1 else None
