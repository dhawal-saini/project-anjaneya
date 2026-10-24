"""Project-wide constants and column contracts.

Everything here is a *structural* contract (column names, timezone helper,
timeframes). Any *tunable* value belongs in ``config/*.yaml`` instead.

Keeping the column orders in one module lets storage, schemas, tests and the
future feature engine agree on a single source of truth.
"""

from __future__ import annotations

from datetime import tzinfo
from zoneinfo import ZoneInfo

__all__ = [
    "DEFAULT_TIMEZONE",
    "SUPPORTED_TIMEFRAMES",
    "TIMEFRAME_MINUTES",
    "HISTORICAL_INTERVALS",
    "INTERVAL_TIMEFRAMES",
    "TICK_COLUMNS",
    "CANDLE_COLUMNS",
    "OPTION_CHAIN_COLUMNS",
    "PROVENANCE_COLUMNS",
    "INDEX_HISTORY_COLUMNS",
    "INSTRUMENT_COLUMNS",
    "COVERAGE_COLUMNS",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "TRADE_COLUMNS",
    "get_timezone",
    "timeframe_to_minutes",
    "interval_to_timeframe",
]

#: Default research timezone. All timestamps are stored/exposed with this tz.
DEFAULT_TIMEZONE = "Asia/Kolkata"

#: Candles are built in parallel for each of these timeframes.
SUPPORTED_TIMEFRAMES = ("1m", "3m", "5m", "15m")

TIMEFRAME_MINUTES: dict[str, int] = {"1m": 1, "3m": 3, "5m": 5, "15m": 15}

#: Interval strings accepted by Kite's ``historical_data`` endpoint. This is the
#: structural contract; ``core.auth.zerodha_client.VALID_INTERVALS`` re-exports it
#: so the SDK wrapper and the configuration loader can never disagree.
HISTORICAL_INTERVALS = (
    "minute",
    "3minute",
    "5minute",
    "10minute",
    "15minute",
    "30minute",
    "60minute",
    "day",
)

#: Kite interval -> ANJANEYA timeframe label. Stored in the ``timeframe`` column
#: so REST-backfilled index rows and WebSocket-built candles speak one language
#: (``"1m"``), never the provider's ("minute").
INTERVAL_TIMEFRAMES: dict[str, str] = {
    "minute": "1m",
    "3minute": "3m",
    "5minute": "5m",
    "10minute": "10m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "1h",
    "day": "1d",
}

#: Provenance / audit columns carried by every stored row (spec 6.5).
PROVENANCE_COLUMNS = ("source", "fetch_timestamp", "quality_flag", "run_id")

# -- raw ticks (spec 4.6) ---------------------------------------------------
TICK_COLUMNS = (
    "timestamp",
    "instrument_token",
    "tradingsymbol",
    "expiry",
    "strike",
    "option_type",
    "last_price",
    "volume_traded",
    "oi",
    "bid",
    "ask",
    "bid_qty",
    "ask_qty",
    *PROVENANCE_COLUMNS,
)

# -- option chain snapshots -------------------------------------------------
OPTION_CHAIN_COLUMNS = (
    "timestamp",
    "instrument_token",
    "tradingsymbol",
    "expiry",
    "strike",
    "option_type",
    "last_price",
    "bid",
    "ask",
    "volume",
    "oi",
    "iv",
    *PROVENANCE_COLUMNS,
)

# -- NIFTY index history (spec 6.1) ----------------------------------------
#: REST-backfilled index OHLCV. ``timeframe`` holds the ANJANEYA label derived
#: from the Kite interval (see :data:`INTERVAL_TIMEFRAMES`), and the primary key
#: is ``(timestamp, tradingsymbol, timeframe)`` so a re-fetch is idempotent.
INDEX_HISTORY_COLUMNS = (
    "timestamp",
    "tradingsymbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
    *PROVENANCE_COLUMNS,
)

#: Instrument master rows (Kite's daily instruments dump). Reference data, so it
#: carries ``fetched_at`` instead of the full provenance block.
INSTRUMENT_COLUMNS = (
    "instrument_token",
    "exchange",
    "tradingsymbol",
    "name",
    "segment",
    "instrument_type",
    "expiry",
    "strike",
    "lot_size",
    "tick_size",
    "fetched_at",
    "run_id",
)

#: Data-coverage ledger (spec 4.3 / 6.5). One row per
#: ``(trading_date, tradingsymbol, timeframe)``; ``quality_flag`` is NOT NULL
#: because a coverage row without a verdict is not auditable.
COVERAGE_COLUMNS = (
    "trading_date",
    "tradingsymbol",
    "timeframe",
    "expected_rows",
    "actual_rows",
    "first_timestamp",
    "last_timestamp",
    "gap_count",
    "largest_gap_seconds",
    "coverage_ratio",
    "quality_flag",
    "note",
    "run_id",
)

# -- candles (spec 6.3) -----------------------------------------------------
CANDLE_COLUMNS = (
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
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "days_to_expiry",
    "distance_from_atm",
    *PROVENANCE_COLUMNS,
)

# -- unified feature row (spec 18) -----------------------------------------
#: Stable, versioned feature contract. Feature names must never be renamed
#: casually: they are persisted, referenced by models and used in reports.
FEATURE_COLUMNS = (
    # identity / context
    "timestamp",
    "underlying",
    "tradingsymbol",
    "instrument_token",
    "expiry",
    "strike",
    "option_type",
    "timeframe",
    # NIFTY group
    "nifty_fast_score",
    "nifty_medium_score",
    "nifty_long_score",
    "nifty_composite_trend_score",
    "nifty_trend_acceleration",
    "nifty_trend_persistence",
    "nifty_regime",
    # OPTION group
    "option_trend_score",
    "option_ma_alignment",
    "option_ma_slope",
    "option_normalized_slope",
    "option_ema20_ema200_spread",
    "option_spread_velocity",
    "option_atr",
    "option_momentum",
    "option_trend_persistence",
    "option_insufficient_history",
    # CANDLE group
    "candle_body",
    "candle_range",
    "candle_upper_wick",
    "candle_lower_wick",
    "candle_body_ratio",
    "candle_upper_wick_ratio",
    "candle_lower_wick_ratio",
    "candle_direction",
    "candle_gap",
    "candle_engulf_ratio",
    "candle_inside_flag",
    "candle_outside_flag",
    "candle_body_ratio_change",
    "candle_wick_ratio_change",
    "candle_close_position_within_prev_range",
    # LEVELS group
    "level_today_open",
    "level_today_high_running",
    "level_today_low_running",
    "level_previous_day_open",
    "level_previous_day_high",
    "level_previous_day_low",
    "level_previous_day_close",
    "distance_to_today_high",
    "distance_to_today_low",
    "distance_to_previous_day_high",
    "distance_to_previous_day_low",
    # LEVEL_EVENTS group
    "event_prev_day_high_state",
    "event_prev_day_low_state",
    "event_today_high_state",
    "event_today_low_state",
    # STRUCTURE group
    "swing_high_1",
    "swing_low_1",
    "swing_label",
    "market_structure_state",
    "distance_from_swing_high",
    "distance_from_swing_low",
    # SUPPORT / RESISTANCE group
    "support_strength",
    "resistance_strength",
    "support_touch_count",
    "support_rejection_count",
    "resistance_touch_count",
    "resistance_rejection_count",
    # CONTEXT group
    "distance_from_atm",
    "days_to_expiry",
    "expiry_cycle_phase",
    "time_of_day_bucket",
    "day_of_week",
    "iv",
    "iv_change",
    "delta",
    "gamma",
    "theta",
    "vega",
    "oi",
    "oi_change",
    "volume",
    # RELATIVE group
    "expected_option_move",
    "actual_option_move",
    "option_residual",
    # provenance
    *PROVENANCE_COLUMNS,
)

# -- label row (spec 20) ----------------------------------------------------
LABEL_COLUMNS = (
    "timestamp",
    "tradingsymbol",
    "instrument_token",
    "expiry",
    "strike",
    "option_type",
    "horizon_minutes",
    "future_return",
    "future_max_up_move",
    "future_max_down_move",
    "future_trend_change",
    "label",
    "label_version",
    *PROVENANCE_COLUMNS,
)

# -- backtest trade ledger (spec 23) ---------------------------------------
TRADE_COLUMNS = (
    "trade_id",
    "run_id",
    "entry_timestamp",
    "exit_timestamp",
    "tradingsymbol",
    "expiry",
    "strike",
    "option_type",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_reason",
    "exit_reason",
    "nifty_state",
    "option_state",
    "candle_state",
    "support_resistance_state",
    "ai_prediction",
    "ai_probability",
    "gross_pnl",
    "costs",
    "net_pnl",
    "mfe",
    "mae",
    "holding_minutes",
    "time_of_day_bucket",
    "expiry_cycle_phase",
    "days_to_expiry",
)


def get_timezone(name: str = DEFAULT_TIMEZONE) -> tzinfo:
    """Return a :class:`zoneinfo.ZoneInfo` for ``name``.

    Raises a helpful error when the IANA database is unavailable — on Windows
    that means the ``tzdata`` package is missing (it is a hard dependency of
    ANJANEYA precisely because of this).
    """
    try:
        return ZoneInfo(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Timezone {name!r} could not be resolved ({exc}). "
            "On Windows this usually means the `tzdata` package is not installed; "
            "run `uv sync` (or `pip install tzdata`) to fix it."
        ) from exc


def timeframe_to_minutes(timeframe: str) -> int:
    """Map a timeframe label (``"5m"``) to minutes.

    Raises ``ValueError`` for unsupported labels so that a typo can never
    silently produce wrong candles.
    """
    try:
        return TIMEFRAME_MINUTES[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; expected one of {SUPPORTED_TIMEFRAMES}"
        ) from None


def interval_to_timeframe(interval: str) -> str:
    """Map a Kite historical interval (``"minute"``) to a timeframe label (``"1m"``).

    Raises ``ValueError`` for an unknown interval — a silently mislabelled
    timeframe would later mix 1-minute and daily bars in one candle table.
    """
    try:
        return INTERVAL_TIMEFRAMES[interval]
    except KeyError:
        raise ValueError(
            f"Unsupported historical interval {interval!r}; expected one of {HISTORICAL_INTERVALS}"
        ) from None
