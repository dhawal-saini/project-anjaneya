"""Pydantic data contracts shared by all three ANJANEYA applications.

These schemas are the *data contract* of the project. They are used for

* validating rows before they are written to DuckDB / Parquet,
* validating rows read back from storage,
* documenting the meaning of every stored field, and
* guaranteeing the invariants the research engines rely on
  (timezone-aware timestamps, valid OHLC, contract identity integrity).

Design notes
------------
* ``timestamp`` and ``fetch_timestamp`` are declared as
  :data:`pydantic.AwareDatetime`, so a naive datetime is rejected outright.
* Option rows must always carry the full contract identity
  ``(expiry, strike, option_type)`` — the master spec forbids mixing
  contracts, so a half-identified option row is a hard error.
* ``extra="forbid"`` is deliberate: a typo in a column name must fail loudly
  instead of silently creating a new column.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from core.constants import (
    CANDLE_COLUMNS,
    COVERAGE_COLUMNS,
    FEATURE_COLUMNS,
    INDEX_HISTORY_COLUMNS,
    INSTRUMENT_COLUMNS,
    LABEL_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    TICK_COLUMNS,
    TRADE_COLUMNS,
)

__all__ = [
    "OptionType",
    "CandleDirection",
    "ExpiryCyclePhase",
    "TimeOfDayBucket",
    "RegimeState",
    "LevelState",
    "MarketStructureState",
    "LabelClass",
    "QualityFlag",
    "Source",
    "RunStatus",
    "ContractIdentity",
    "TickRow",
    "OptionChainSnapshotRow",
    "IndexHistoryRow",
    "InstrumentRow",
    "CoverageRow",
    "CandleRow",
    "FeatureRow",
    "LabelRow",
    "TradeRecord",
    "RunManifest",
]

#: Numerical tolerance used when comparing prices (never compare floats with ==).
PRICE_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class OptionType(StrEnum):
    """Option side. Values match the NSE ``tradingsymbol`` suffix."""

    CE = "CE"
    PE = "PE"


class CandleDirection(StrEnum):
    """Direction of a single candle body."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    FLAT = "FLAT"


class ExpiryCyclePhase(StrEnum):
    """Where the current session sits inside the weekly expiry cycle (RQ1)."""

    START_OF_CYCLE = "START_OF_CYCLE"
    MID_CYCLE = "MID_CYCLE"
    EXPIRY_DAY = "EXPIRY_DAY"


class TimeOfDayBucket(StrEnum):
    """Intraday analysis buckets (RQ2). Names are config-driven and stable."""

    OPEN_915_930 = "OPEN_915_930"
    MORNING_930_1100 = "MORNING_930_1100"
    MIDDAY_1100_1300 = "MIDDAY_1100_1300"
    AFTERNOON_1300_1430 = "AFTERNOON_1300_1430"
    CLOSE_1430_1515 = "CLOSE_1430_1515"
    #: Timestamps outside every configured bucket (pre-open, post-analysis).
    OUT_OF_SESSION = "OUT_OF_SESSION"


class RegimeState(StrEnum):
    """NIFTY regime. Descriptive only — this engine never emits BUY/SELL."""

    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    BULLISH_DEVELOPING = "BULLISH_DEVELOPING"
    NEUTRAL = "NEUTRAL"
    BEARISH_DEVELOPING = "BEARISH_DEVELOPING"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class LevelState(StrEnum):
    """State of a price level relative to price (spec 13).

    Every one of these is timestamped and non-repainting: once a state has
    been emitted for a bar it must never be rewritten by later data.
    """

    NOT_TESTED = "NOT_TESTED"
    APPROACHING = "APPROACHING"
    TOUCHED = "TOUCHED"
    CROSSED_ABOVE = "CROSSED_ABOVE"
    CROSSED_BELOW = "CROSSED_BELOW"
    CLOSED_ABOVE = "CLOSED_ABOVE"
    CLOSED_BELOW = "CLOSED_BELOW"
    HELD_ABOVE = "HELD_ABOVE"
    HELD_BELOW = "HELD_BELOW"
    REJECTED = "REJECTED"
    RETESTED = "RETESTED"
    FAILED_BREAKOUT = "FAILED_BREAKOUT"
    FAILED_BREAKDOWN = "FAILED_BREAKDOWN"


class MarketStructureState(StrEnum):
    """Market structure classification derived from confirmed swings."""

    BULLISH_STRUCTURE = "BULLISH_STRUCTURE"
    BEARISH_STRUCTURE = "BEARISH_STRUCTURE"
    RANGE = "RANGE"
    TRANSITION = "TRANSITION"


class LabelClass(StrEnum):
    """Behavioural labels. ANJANEYA never predicts BUY/SELL directly."""

    UP_CONTINUATION = "UP_CONTINUATION"
    DOWN_CONTINUATION = "DOWN_CONTINUATION"
    RANGE = "RANGE"
    REVERSAL = "REVERSAL"


class QualityFlag(StrEnum):
    """Data quality marker carried by every stored row (spec 6.5)."""

    OK = "OK"
    MISSING_TICK = "MISSING_TICK"
    GAP_FILLED = "GAP_FILLED"
    STALE = "STALE"
    SUSPECT = "SUSPECT"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    #: Tick that arrived outside the configured NSE session (pre-open ticks,
    #: post-close ticks, ticks on a non-trading day). Such rows are recorded -
    #: never silently dropped - but must be excluded from session analysis.
    OUT_OF_SESSION = "OUT_OF_SESSION"


class Source(StrEnum):
    """Where a row came from. Used to trace every number back to its origin."""

    ZERODHA_WEBSOCKET = "ZERODHA_WEBSOCKET"
    ZERODHA_REST = "ZERODHA_REST"
    DERIVED = "DERIVED"
    TEST_FIXTURE = "TEST_FIXTURE"


class RunStatus(StrEnum):
    """Lifecycle status recorded in a run manifest."""

    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


# ---------------------------------------------------------------------------
# Base model + contract identity
# ---------------------------------------------------------------------------
class AnjaneyaModel(BaseModel):
    """Base for every ANJANEYA row contract.

    ``extra="forbid"``        - unknown columns are rejected (typo protection).
    ``validate_assignment``   - invariants stay true after mutation.
    ``use_enum_values=False`` - callers get real enum members, not bare strings,
                                so ``row.option_type is OptionType.CE`` works.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )

    def to_record(self, exclude_none: bool = True) -> dict[str, Any]:
        """Return a plain ``dict`` ready for storage.

        Enum members are converted to their string values and datetimes stay
        timezone-aware. ``exclude_none`` (default) keeps sparse rows sparse
        instead of writing a wall of NULLs.
        """
        data = self.model_dump(mode="python", exclude_none=exclude_none)
        for key, value in list(data.items()):
            if isinstance(value, StrEnum):
                data[key] = value.value
        return data


class ContractIdentity(AnjaneyaModel):
    """The identity of a tradable contract (spec 6.4).

    Every option row MUST be identified by all four components. Indicators and
    features are never computed across different contracts.
    """

    underlying: str = Field(default="NIFTY", min_length=1)
    expiry: date | None = None
    strike: float | None = Field(default=None, ge=0)
    option_type: OptionType | None = None

    @model_validator(mode="after")
    def _all_or_none(self) -> ContractIdentity:
        parts = (self.expiry, self.strike, self.option_type)
        provided = [p for p in parts if p is not None]
        if provided and len(provided) != len(parts):
            raise ValueError(
                "Partial option contract identity: expiry, strike and option_type must "
                "either all be provided or all be omitted "
                f"(got expiry={self.expiry}, strike={self.strike}, option_type={self.option_type})"
            )
        return self

    @property
    def is_option(self) -> bool:
        """True when this identity describes an option contract."""
        return self.option_type is not None

    @property
    def key(self) -> tuple[str, date | None, float | None, str | None]:
        """Hashable identity tuple, safe to use as a ``groupby`` key."""
        return (
            self.underlying,
            self.expiry,
            self.strike,
            self.option_type.value if self.option_type else None,
        )


# ---------------------------------------------------------------------------
# Raw market data rows
# ---------------------------------------------------------------------------
class TickRow(AnjaneyaModel):
    """One WebSocket/quote tick stored append-only (spec 4.6).

    Ticks are NEVER overwritten: the tick table is the immutable historical
    record from which everything else is derived.
    """

    timestamp: AwareDatetime = Field(description="Exchange tick time (IST, tz-aware).")
    instrument_token: int = Field(gt=0, description="Zerodha instrument token.")
    tradingsymbol: str = Field(min_length=1)
    expiry: date | None = Field(default=None, description="Required for options.")
    strike: float | None = Field(default=None, ge=0)
    option_type: OptionType | None = None
    last_price: float = Field(ge=0)
    volume_traded: int | None = Field(default=None, ge=0, description="Cumulative day volume.")
    oi: int | None = Field(default=None, ge=0, description="Open interest, where published.")
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    bid_qty: int | None = Field(default=None, ge=0)
    ask_qty: int | None = Field(default=None, ge=0)
    source: Source
    fetch_timestamp: AwareDatetime = Field(
        description="When ANJANEYA received the tick (always >= timestamp)."
    )
    quality_flag: QualityFlag = QualityFlag.OK
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> TickRow:
        if self.fetch_timestamp < self.timestamp:
            raise ValueError("fetch_timestamp cannot precede the exchange timestamp")
        parts_provided = [p for p in (self.expiry, self.strike, self.option_type) if p is not None]
        if parts_provided and len(parts_provided) != 3:
            raise ValueError(
                "Option ticks must carry expiry, strike and option_type together "
                f"(got expiry={self.expiry}, strike={self.strike}, option_type={self.option_type})"
            )
        if self.bid is not None and self.ask is not None and self.bid > self.ask + PRICE_EPSILON:
            raise ValueError(f"bid ({self.bid}) cannot exceed ask ({self.ask})")
        return self

    @property
    def is_option(self) -> bool:
        """True when this tick belongs to an option contract."""
        return self.option_type is not None

    @property
    def mid_price(self) -> float | None:
        """Mid price when a two-sided quote exists, else ``None``."""
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


class OptionChainSnapshotRow(AnjaneyaModel):
    """A periodically polled option-chain quote (OI/IV snapshot).

    Polling complements the WebSocket: it provides a second, independent
    source for OI/IV/OI-change when the tick stream is sparse.
    """

    timestamp: AwareDatetime
    instrument_token: int = Field(gt=0)
    tradingsymbol: str = Field(min_length=1)
    expiry: date
    strike: float = Field(ge=0)
    option_type: OptionType
    last_price: float = Field(ge=0)
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    volume: int | None = Field(default=None, ge=0)
    oi: int | None = Field(default=None, ge=0)
    iv: float | None = Field(default=None, ge=0, description="Implied vol as a fraction.")
    source: Source = Source.ZERODHA_REST
    fetch_timestamp: AwareDatetime
    quality_flag: QualityFlag = QualityFlag.OK
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> OptionChainSnapshotRow:
        if self.bid is not None and self.ask is not None and self.bid > self.ask + PRICE_EPSILON:
            raise ValueError(f"bid ({self.bid}) cannot exceed ask ({self.ask})")
        return self


# ---------------------------------------------------------------------------
# REST-backfilled index history (spec 6.1)
# ---------------------------------------------------------------------------
class IndexHistoryRow(AnjaneyaModel):
    """One historical OHLCV bar for the NIFTY index (REST backfill).

    The index is captured from the historical-candle endpoint because the
    WebSocket only streams its *last price*; the whole point of the backfill is
    a durable multi-hundred-day OHLCV history for trend/regime research.

    ``timeframe`` is the ANJANEYA label (``"1m"``, ``"1d"``), never Kite's
    interval string, and ``(timestamp, tradingsymbol, timeframe)`` is the
    primary key - so re-running a backfill updates the same rows instead of
    duplicating them.
    """

    timestamp: AwareDatetime = Field(description="Bar open time (IST, tz-aware).")
    tradingsymbol: str = Field(min_length=1, description="e.g. 'NIFTY 50'.")
    timeframe: str = Field(min_length=1, description="ANJANEYA timeframe label.")
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: int | None = Field(default=None, ge=0, description="Index volume (0/absent).")
    oi: int | None = Field(default=None, ge=0)
    source: Source = Source.ZERODHA_REST
    fetch_timestamp: AwareDatetime = Field(
        description="When ANJANEYA fetched the bar (always >= timestamp)."
    )
    quality_flag: QualityFlag = QualityFlag.OK
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> IndexHistoryRow:
        if self.high + PRICE_EPSILON < self.low:
            raise ValueError(f"high ({self.high}) cannot be below low ({self.low})")
        if self.high + PRICE_EPSILON < max(self.open, self.close):
            raise ValueError(
                f"high ({self.high}) is below open/close ({self.open}/{self.close})"
            )
        if self.low - PRICE_EPSILON > min(self.open, self.close):
            raise ValueError(
                f"low ({self.low}) is above open/close ({self.open}/{self.close})"
            )
        if self.fetch_timestamp < self.timestamp:
            raise ValueError("fetch_timestamp cannot precede the bar timestamp")
        return self


# ---------------------------------------------------------------------------
# Instrument master (Kite's daily instrument dump)
# ---------------------------------------------------------------------------
class InstrumentRow(AnjaneyaModel):
    """One row of the Kite instrument dump, reduced to the fields we use.

    The dump is the authoritative mapping from ``tradingsymbol`` to
    ``instrument_token``. ANJANEYA resolves contracts by *attributes*
    (exchange, underlying, expiry, strike, option type) rather than by
    hand-built symbol strings, because NSE changes symbol formats.
    """

    instrument_token: int = Field(gt=0)
    exchange: str = Field(min_length=1)
    tradingsymbol: str = Field(min_length=1)
    name: str | None = Field(default=None, description="Underlying name, e.g. 'NIFTY'.")
    segment: str | None = Field(default=None, description="e.g. 'NFO-OPT'.")
    instrument_type: str | None = Field(default=None, description="CE / PE / FUT / ...")
    expiry: date | None = None
    strike: float | None = Field(default=None, ge=0)
    lot_size: int | None = Field(default=None, gt=0)
    tick_size: float | None = Field(default=None, gt=0)
    fetched_at: AwareDatetime = Field(description="When the dump was downloaded.")
    run_id: str = Field(min_length=1)

    @field_validator("exchange", mode="after")
    @classmethod
    def _upper_exchange(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _validate(self) -> InstrumentRow:
        instrument_type = (self.instrument_type or "").upper()
        if instrument_type in {"CE", "PE"}:
            if self.expiry is None or self.strike is None:
                raise ValueError(
                    f"option instrument {self.tradingsymbol!r} needs both expiry and strike "
                    f"(got expiry={self.expiry}, strike={self.strike})"
                )
        elif instrument_type == "FUT" and self.expiry is None:
            raise ValueError(f"futures instrument {self.tradingsymbol!r} needs an expiry date")
        return self

    @property
    def is_option(self) -> bool:
        """True for a CE/PE contract."""
        return (self.instrument_type or "").upper() in {"CE", "PE"}


# ---------------------------------------------------------------------------
# Data coverage ledger (spec 4.3 / 6.5)
# ---------------------------------------------------------------------------
class CoverageRow(AnjaneyaModel):
    """One ``(trading_date, tradingsymbol, timeframe)`` coverage verdict.

    Written by ANJANEYA-CAPTURE for every instrument it captured, so "how much
    of today did we actually record?" is a SQL query rather than an anecdote.
    ``expected_rows`` stays ``None`` for streams whose expected volume is not
    predictable (tick streams), in which case ``coverage_ratio`` is ``None`` and
    the verdict rests on ``gap_count`` / ``largest_gap_seconds``.
    """

    trading_date: date
    tradingsymbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1, description="'TICK' for the tick stream.")
    expected_rows: int | None = Field(default=None, ge=0)
    actual_rows: int | None = Field(default=None, ge=0)
    first_timestamp: AwareDatetime | None = None
    last_timestamp: AwareDatetime | None = None
    gap_count: int | None = Field(default=None, ge=0)
    largest_gap_seconds: float | None = Field(default=None, ge=0)
    coverage_ratio: float | None = Field(default=None, ge=0, le=1)
    quality_flag: QualityFlag
    note: str | None = None
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> CoverageRow:
        if (
            self.first_timestamp is not None
            and self.last_timestamp is not None
            and self.last_timestamp < self.first_timestamp
        ):
            raise ValueError("last_timestamp cannot precede first_timestamp")
        if (
            self.expected_rows is not None
            and self.actual_rows is not None
            and self.actual_rows > self.expected_rows
        ):
            raise ValueError(
                f"actual_rows ({self.actual_rows}) exceeds expected_rows ({self.expected_rows}); "
                "the expectation is wrong - fix it rather than fudging the ratio"
            )
        return self


# ---------------------------------------------------------------------------
# Derived candles
# ---------------------------------------------------------------------------
class CandleRow(AnjaneyaModel):
    """One OHLCV candle (spec 4.7 / 6.3).

    Candles are built strictly from stored ticks and never mix ticks from
    different contracts. The OHLC invariants below are what protects every
    downstream indicator from garbage input.
    """

    timestamp: AwareDatetime = Field(description="Candle OPEN time (IST, tz-aware).")
    timeframe: str = Field(description="One of: " + ", ".join(SUPPORTED_TIMEFRAMES))
    instrument_token: int | None = Field(default=None, gt=0)
    tradingsymbol: str = Field(min_length=1)
    expiry: date | None = None
    strike: float | None = Field(default=None, ge=0)
    option_type: OptionType | None = None
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: int | None = Field(default=None, ge=0)
    oi: int | None = Field(default=None, ge=0)
    iv: float | None = Field(default=None, ge=0)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    days_to_expiry: float | None = None
    distance_from_atm: float | None = None
    source: Source
    fetch_timestamp: AwareDatetime | None = None
    quality_flag: QualityFlag = QualityFlag.OK
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> CandleRow:
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe {self.timeframe!r}; expected one of {SUPPORTED_TIMEFRAMES}"
            )
        if self.high + PRICE_EPSILON < self.low:
            raise ValueError(f"high ({self.high}) cannot be below low ({self.low})")
        for name in ("open", "close"):
            value = getattr(self, name)
            if value > self.high + PRICE_EPSILON:
                raise ValueError(f"{name} ({value}) above high ({self.high})")
            if value < self.low - PRICE_EPSILON:
                raise ValueError(f"{name} ({value}) below low ({self.low})")
        parts_provided = [p for p in (self.expiry, self.strike, self.option_type) if p is not None]
        if parts_provided and len(parts_provided) != 3:
            raise ValueError(
                "Option candles must carry expiry, strike and option_type together "
                f"(got expiry={self.expiry}, strike={self.strike}, option_type={self.option_type})"
            )
        return self

    @property
    def is_option(self) -> bool:
        """True when this candle belongs to an option contract."""
        return self.option_type is not None

    @property
    def price_range(self) -> float:
        """High minus low (never negative)."""
        return self.high - self.low


# ---------------------------------------------------------------------------
# Unified feature row (spec 18)
# ---------------------------------------------------------------------------
#: Fields that must NEVER appear in a feature row. Labels are generated in a
#: physically separate stage (spec 20/30), so their presence here is a bug.
LABEL_FIELD_NAMES = (
    "future_return",
    "future_max_up_move",
    "future_max_down_move",
    "future_trend_change",
    "label",
    "label_version",
)


class FeatureRow(AnjaneyaModel):
    """One unified feature row per ``(timestamp, option contract)`` (spec 18).

    Feature values are deliberately optional: early in the accumulation cycle
    (and for short-lived weekly contracts) many engines cannot produce a
    value yet. ``None`` means "not available at this timestamp" - it is never
    silently replaced with 0 or with a future value.

    ``time_of_day_bucket`` is a plain string because bucket names are
    configurable in ``config/calendar.yaml``; the other categorical fields use
    enums because their vocabulary is fixed by the research questions.
    """

    # -- identity / context -------------------------------------------------
    timestamp: AwareDatetime
    underlying: str = Field(default="NIFTY", min_length=1)
    tradingsymbol: str = Field(min_length=1)
    instrument_token: int | None = Field(default=None, gt=0)
    expiry: date | None = None
    strike: float | None = Field(default=None, ge=0)
    option_type: OptionType | None = None
    timeframe: str | None = None

    # -- NIFTY group --------------------------------------------------------
    nifty_fast_score: float | None = None
    nifty_medium_score: float | None = None
    nifty_long_score: float | None = None
    nifty_composite_trend_score: float | None = None
    nifty_trend_acceleration: float | None = None
    nifty_trend_persistence: float | None = None
    nifty_regime: RegimeState | None = None

    # -- OPTION group -------------------------------------------------------
    option_trend_score: float | None = None
    option_ma_alignment: float | None = None
    option_ma_slope: float | None = None
    option_normalized_slope: float | None = None
    option_ema20_ema200_spread: float | None = None
    option_spread_velocity: float | None = None
    option_atr: float | None = None
    option_momentum: float | None = None
    option_trend_persistence: float | None = None
    option_insufficient_history: bool | None = None

    # -- CANDLE group -------------------------------------------------------
    candle_body: float | None = None
    candle_range: float | None = None
    candle_upper_wick: float | None = None
    candle_lower_wick: float | None = None
    candle_body_ratio: float | None = None
    candle_upper_wick_ratio: float | None = None
    candle_lower_wick_ratio: float | None = None
    candle_direction: CandleDirection | None = None
    candle_gap: float | None = None
    candle_engulf_ratio: float | None = None
    candle_inside_flag: bool | None = None
    candle_outside_flag: bool | None = None
    candle_body_ratio_change: float | None = None
    candle_wick_ratio_change: float | None = None
    candle_close_position_within_prev_range: float | None = None

    # -- LEVELS group -------------------------------------------------------
    level_today_open: float | None = None
    level_today_high_running: float | None = None
    level_today_low_running: float | None = None
    level_previous_day_open: float | None = None
    level_previous_day_high: float | None = None
    level_previous_day_low: float | None = None
    level_previous_day_close: float | None = None
    distance_to_today_high: float | None = None
    distance_to_today_low: float | None = None
    distance_to_previous_day_high: float | None = None
    distance_to_previous_day_low: float | None = None

    # -- LEVEL_EVENTS group -------------------------------------------------
    event_prev_day_high_state: LevelState | None = None
    event_prev_day_low_state: LevelState | None = None
    event_today_high_state: LevelState | None = None
    event_today_low_state: LevelState | None = None

    # -- STRUCTURE group ----------------------------------------------------
    swing_high_1: float | None = None
    swing_low_1: float | None = None
    swing_label: str | None = None
    market_structure_state: MarketStructureState | None = None
    distance_from_swing_high: float | None = None
    distance_from_swing_low: float | None = None

    # -- SUPPORT / RESISTANCE group ----------------------------------------
    support_strength: float | None = None
    resistance_strength: float | None = None
    support_touch_count: int | None = Field(default=None, ge=0)
    support_rejection_count: int | None = Field(default=None, ge=0)
    resistance_touch_count: int | None = Field(default=None, ge=0)
    resistance_rejection_count: int | None = Field(default=None, ge=0)

    # -- CONTEXT group ------------------------------------------------------
    distance_from_atm: float | None = None
    days_to_expiry: float | None = Field(default=None, ge=0)
    expiry_cycle_phase: ExpiryCyclePhase | None = None
    #: Name of the configured time-of-day bucket; see config/calendar.yaml.
    time_of_day_bucket: str | None = None
    #: Monday = 0 ... Sunday = 6 (Python weekday convention).
    day_of_week: int | None = Field(default=None, ge=0, le=6)
    iv: float | None = Field(default=None, ge=0)
    iv_change: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    oi: int | None = Field(default=None, ge=0)
    oi_change: int | None = None
    volume: int | None = Field(default=None, ge=0)

    # -- RELATIVE group -----------------------------------------------------
    expected_option_move: float | None = None
    actual_option_move: float | None = None
    option_residual: float | None = None

    # -- provenance ---------------------------------------------------------
    source: Source
    fetch_timestamp: AwareDatetime | None = None
    quality_flag: QualityFlag = QualityFlag.OK
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_label_fields(self) -> FeatureRow:
        """Structural look-ahead guard (spec 20/30).

        Labels live in a physically separate dataset. Because ``extra="forbid"``
        already rejects unknown keys, this validator exists to give a *clear*
        error message when somebody tries to smuggle a label into the feature
        row via a model-level dict.
        """
        for name in LABEL_FIELD_NAMES:
            if name in type(self).model_fields:  # pragma: no cover - defensive
                raise ValueError(f"Feature schema must not contain label field {name!r}")
        return self

    @property
    def contract_key(self) -> tuple[str, date | None, float | None, str | None]:
        """Contract identity of this feature row."""
        return (
            self.underlying,
            self.expiry,
            self.strike,
            self.option_type.value if self.option_type else None,
        )


# ---------------------------------------------------------------------------
# Labels (spec 20) - physically separate from features
# ---------------------------------------------------------------------------
class LabelRow(AnjaneyaModel):
    """Forward-looking behaviour measured *after* a feature timestamp (spec 20).

    Sign conventions (documented so nobody has to guess):

    * ``future_return``          signed forward return over the horizon.
    * ``future_max_up_move``     maximum favourable excursion, ``>= 0``.
    * ``future_max_down_move``   maximum adverse excursion, ``<= 0``.
    * ``future_trend_change``    change in the trend/regime descriptor.
    * ``label``                  discretised class used for training.

    Labels are produced by a dedicated pass over future candles. They must
    never be merged into the feature table at generation time.
    """

    timestamp: AwareDatetime
    tradingsymbol: str = Field(min_length=1)
    instrument_token: int | None = Field(default=None, gt=0)
    expiry: date | None = None
    strike: float | None = Field(default=None, ge=0)
    option_type: OptionType | None = None
    horizon_minutes: int = Field(gt=0)
    future_return: float | None = None
    future_max_up_move: float | None = None
    future_max_down_move: float | None = None
    future_trend_change: float | None = None
    label: LabelClass | None = None
    label_version: str = Field(min_length=1)
    source: Source = Source.DERIVED
    fetch_timestamp: AwareDatetime | None = None
    quality_flag: QualityFlag = QualityFlag.OK
    run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> LabelRow:
        if self.future_max_up_move is not None and self.future_max_up_move < 0:
            raise ValueError("future_max_up_move is an upward excursion and must be >= 0")
        if self.future_max_down_move is not None and self.future_max_down_move > 0:
            raise ValueError("future_max_down_move is a downward excursion and must be <= 0")
        return self


# ---------------------------------------------------------------------------
# Backtest ledger (spec 23) - losing trades are recorded, never hidden
# ---------------------------------------------------------------------------
class TradeRecord(AnjaneyaModel):
    """A single simulated trade. Includes every losing trade (spec 23)."""

    trade_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    entry_timestamp: AwareDatetime
    exit_timestamp: AwareDatetime | None = None
    tradingsymbol: str = Field(min_length=1)
    expiry: date | None = None
    strike: float | None = Field(default=None, ge=0)
    option_type: OptionType | None = None
    entry_price: float = Field(ge=0)
    exit_price: float | None = Field(default=None, ge=0)
    quantity: int = Field(gt=0)
    entry_reason: str | None = None
    exit_reason: str | None = None
    nifty_state: str | None = None
    option_state: str | None = None
    candle_state: str | None = None
    support_resistance_state: str | None = None
    ai_prediction: str | None = None
    ai_probability: float | None = Field(default=None, ge=0, le=1)
    gross_pnl: float | None = None
    costs: float | None = Field(default=None, ge=0)
    net_pnl: float | None = None
    mfe: float | None = Field(default=None, description="Maximum favourable excursion.")
    mae: float | None = Field(default=None, description="Maximum adverse excursion.")
    holding_minutes: float | None = Field(default=None, ge=0)
    time_of_day_bucket: str | None = None
    expiry_cycle_phase: ExpiryCyclePhase | None = None
    days_to_expiry: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate(self) -> TradeRecord:
        if self.exit_timestamp is not None and self.exit_timestamp < self.entry_timestamp:
            raise ValueError("exit_timestamp cannot precede entry_timestamp")
        return self


# ---------------------------------------------------------------------------
# Run manifest (spec 36)
# ---------------------------------------------------------------------------
class RunManifest(AnjaneyaModel):
    """Reproducibility record written by every ANJANEYA run (spec 36)."""

    run_id: str = Field(min_length=1)
    app_name: str = Field(min_length=1)
    mode: str | None = None
    status: RunStatus = RunStatus.STARTED
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    config_hash: str | None = None
    config_dir: str | None = None
    git_commit: str | None = None
    python_version: str | None = None
    platform: str | None = None
    package_versions: dict[str, str | None] = Field(default_factory=dict)
    random_seed: int | None = None
    data_coverage: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Import-time contract parity check
# ---------------------------------------------------------------------------
CONTRACT_PAIRS: tuple[tuple[str, type[AnjaneyaModel], tuple[str, ...]], ...] = (
    ("TickRow", TickRow, TICK_COLUMNS),
    ("OptionChainSnapshotRow", OptionChainSnapshotRow, OPTION_CHAIN_COLUMNS),
    ("IndexHistoryRow", IndexHistoryRow, INDEX_HISTORY_COLUMNS),
    ("InstrumentRow", InstrumentRow, INSTRUMENT_COLUMNS),
    ("CoverageRow", CoverageRow, COVERAGE_COLUMNS),
    ("CandleRow", CandleRow, CANDLE_COLUMNS),
    ("FeatureRow", FeatureRow, FEATURE_COLUMNS),
    ("LabelRow", LabelRow, LABEL_COLUMNS),
    ("TradeRecord", TradeRecord, TRADE_COLUMNS),
)


def assert_contract_parity() -> None:
    """Raise if a column contract and its pydantic model have drifted apart.

    This is a pure, side-effect-free consistency check (no I/O). Running it at
    import time means a column was renamed in one place but not the other is
    caught the moment any app or test touches the schemas - rather than at
    09:15 on a trading day.
    """
    problems: list[str] = []
    for name, model, columns in CONTRACT_PAIRS:
        missing = sorted(set(columns) - set(model.model_fields))
        extra = sorted(set(model.model_fields) - set(columns))
        if missing:
            problems.append(f"{name} is missing contracted columns: {missing}")
        if extra:
            problems.append(f"{name} has fields absent from the column contract: {extra}")
    if problems:
        raise RuntimeError("ANJANEYA schema/column contract mismatch: " + "; ".join(problems))


assert_contract_parity()
