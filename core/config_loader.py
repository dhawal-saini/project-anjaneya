"""Typed configuration loading for ANJANEYA.

Every tunable in ANJANEYA lives in ``config/*.yaml``. This module turns those
files into validated pydantic objects so that

* a typo in a config key is a hard failure, not a silently ignored setting,
* the resolved configuration can be hashed and stored in the run manifest
  (spec 36 - reproducibility), and
* secrets never appear in YAML: files reference ``${ENV_VAR}`` placeholders
  which are resolved from the process environment (optionally from ``.env``).

Importing this module performs **no I/O**; all file access happens inside
:func:`load_config`, which the applications call explicitly.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import date, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.constants import (
    DEFAULT_TIMEZONE,
    FEATURE_COLUMNS,
    HISTORICAL_INTERVALS,
    SUPPORTED_TIMEFRAMES,
)

__all__ = [
    "ConfigError",
    "CONFIG_FILES",
    "AnjaneyaConfig",
    "load_config",
    "load_yaml_file",
    "resolve_env_placeholders",
    "referenced_env_vars",
    "config_file_hash",
]

#: Placeholder syntax: ``${VAR}`` or ``${VAR:-fallback}``.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: The six configuration files ANJANEYA loads (spec 31).
CONFIG_FILES = (
    "settings.yaml",
    "data_sources.yaml",
    "option_universe.yaml",
    "indicators.yaml",
    "model.yaml",
    "calendar.yaml",
)


class ConfigError(RuntimeError):
    """Raised when configuration is missing, unparseable or inconsistent."""


def resolve_env_placeholders(value: Any, environ: dict[str, str] | None = None) -> Any:
    """Recursively resolve ``${VAR}`` / ``${VAR:-default}`` placeholders.

    ``environ`` defaults to :data:`os.environ`. A placeholder without a default
    whose variable is unset raises :class:`ConfigError` - configuration that
    silently becomes ``"None"`` is exactly the kind of bug that ruins a
    multi-month data capture run.
    """
    env = os.environ if environ is None else environ

    def _resolve_string(text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env and env[name] != "":
                return env[name]
            if default is not None:
                return default
            raise ConfigError(
                f"Environment variable {name!r} referenced by configuration is not set "
                "and no default was provided. Add it to your .env file (see .env.example)."
            )

        return _ENV_PATTERN.sub(_replace, text)

    if isinstance(value, str):
        return _resolve_string(value)
    if isinstance(value, dict):
        return {key: resolve_env_placeholders(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_placeholders(item, env) for item in value]
    return value


def referenced_env_vars(value: Any) -> set[str]:
    """Return every environment variable name referenced by a config object.

    Used by tests to guarantee ``.env.example`` documents all of them.
    """
    found: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            found.update(match.group(1) for match in _ENV_PATTERN.finditer(node))
        elif isinstance(node, dict):
            for item in node.values():
                _walk(item)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(value)
    return found


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, raising :class:`ConfigError` on problems."""
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if parsed is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(parsed, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(parsed).__name__}")
    return parsed


def config_file_hash(config_dir: Path, files: tuple[str, ...] = CONFIG_FILES) -> str:
    """SHA-256 over the raw bytes of every config file (spec 36).

    Hashing the *raw* text (before env resolution) means the hash identifies the
    committed configuration, while secrets stay out of the hash and out of logs.
    """
    digest = hashlib.sha256()
    for name in sorted(files):
        path = Path(config_dir) / name
        digest.update(name.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


class _StrictConfigModel(BaseModel):
    """Base for every config section: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


def _validate_timezone(value: str) -> str:
    """Timezone strings must be IANA names; validation happens on first use."""
    if not value or "/" not in value:
        raise ValueError(f"{value!r} is not an IANA timezone name like 'Asia/Kolkata'")
    return value


# ---------------------------------------------------------------------------
# settings.yaml
# ---------------------------------------------------------------------------
class ProjectConfig(_StrictConfigModel):
    """Static project metadata (bumped when the data contract changes)."""

    name: str = "ANJANEYA"
    full_title: str = "ANJANEYA - NIFTY50 Options AI Research Engine"
    code_name: str = "ANJANEYA"
    version: str
    schema_version: int = Field(ge=1)


class TimeConfig(_StrictConfigModel):
    """Time handling policy."""

    timezone: str = DEFAULT_TIMEZONE
    forbid_naive_timestamps: bool = True

    @field_validator("timezone")
    @classmethod
    def _check_tz(cls, value: str) -> str:
        return _validate_timezone(value)


class MarketConfig(_StrictConfigModel):
    """Trading session bounds and candle timeframes."""

    session_start: time
    session_end: time
    analysis_end: time
    timeframes: list[str] = Field(default_factory=lambda: ["1m", "3m", "5m", "15m"])

    @field_validator("timeframes")
    @classmethod
    def _supported_timeframes(cls, value: list[str]) -> list[str]:
        unknown = [tf for tf in value if tf not in SUPPORTED_TIMEFRAMES]
        if unknown:
            raise ValueError(
                f"Unsupported candle timeframes {unknown}; supported: {SUPPORTED_TIMEFRAMES}"
            )
        if not value:
            raise ValueError("At least one candle timeframe must be configured")
        return value

    @model_validator(mode="after")
    def _ordered(self) -> MarketConfig:
        if not (self.session_start < self.session_end):
            raise ValueError("session_start must be earlier than session_end")
        if not (self.session_start <= self.analysis_end <= self.session_end):
            raise ValueError("analysis_end must lie within [session_start, session_end]")
        return self


class DataDirsConfig(_StrictConfigModel):
    """Sub-directory names under the data root."""

    raw: str = "raw"
    candles: str = "candles"
    features: str = "features"
    labels: str = "labels"
    datasets: str = "datasets"
    models: str = "models"
    predictions: str = "predictions"
    backtests: str = "backtests"
    reports: str = "reports"
    logs: str = "logs"
    runs: str = "runs"


class DataConfig(_StrictConfigModel):
    """Where captured and derived data lives."""

    root: str = "./data"
    db_path: str = "./db/anjaneeya.duckdb"
    dirs: DataDirsConfig = Field(default_factory=DataDirsConfig)
    parquet_compression: Literal["zstd", "snappy", "gzip", "lz4", "none"] = "zstd"


class LoggingConfig(_StrictConfigModel):
    """Logging sinks (loguru based)."""

    level: str = "INFO"
    console: bool = True
    file: bool = True
    rotation: str = "00:00"
    retention: str = "30 days"
    json_sink: bool = False
    json_sink_name: str = "anjaneeya.jsonl"
    intercept_stdlib: bool = True

    @field_validator("level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"Unknown log level {value!r}; expected one of {sorted(allowed)}")
        return upper


class RunTrackingConfig(_StrictConfigModel):
    """Run manifest / reproducibility settings (spec 36)."""

    enabled: bool = True
    record_git_commit: bool = True
    record_packages: list[str] = Field(default_factory=list)


class EngineConfig(_StrictConfigModel):
    """ANJANEYA-ENGINE behaviour that is not itself a market parameter.

    Deliberately tiny: everything that *defines* a candle (the session window,
    the timeframes) lives in ``market`` / ``calendar.yaml`` so there is only one
    place to look. This section only holds the derived-output switches.
    """

    #: Mirror the candle table to ``data/candles/<timeframe>/<YYYY-MM-DD>.parquet``
    #: in addition to DuckDB. The mirror is derived data: it can always be
    #: rebuilt from ticks, so turning it off never loses information.
    mirror_parquet: bool = True


class SettingsConfig(_StrictConfigModel):
    """Root model of ``config/settings.yaml``."""

    project: ProjectConfig
    time: TimeConfig
    market: MarketConfig
    data: DataConfig
    logging: LoggingConfig
    run_tracking: RunTrackingConfig
    #: Optional so that a Phase-1 style ``settings.yaml`` still loads; the
    #: shipped file documents the section explicitly.
    engine: EngineConfig = Field(default_factory=EngineConfig)


# ---------------------------------------------------------------------------
# data_sources.yaml
# ---------------------------------------------------------------------------
class TimeoutsConfig(_StrictConfigModel):
    """HTTP timeouts (seconds). Kite Connect's own default is 7s."""

    request_seconds: float = Field(default=7.0, gt=0)
    login_seconds: float = Field(default=15.0, gt=0)


class RateLimitsConfig(_StrictConfigModel):
    """Requests per second allowed per endpoint class (spec 4.5)."""

    quote_per_second: float = Field(default=1.0, gt=0)
    historical_per_second: float = Field(default=3.0, gt=0)
    other_per_second: float = Field(default=10.0, gt=0)
    #: <1.0 = more conservative than the documented broker limits.
    safety_factor: float = Field(default=0.9, gt=0, le=1)


class TokenStoreConfig(_StrictConfigModel):
    """Where the Kite access token is cached between runs."""

    path: str
    validity_cutoff: time = time(6, 0)
    refresh_on_startup: bool = True
    min_remaining_minutes: int = Field(default=30, ge=0)


class RetryConfig(_StrictConfigModel):
    """Retry / backoff policy for every REST call (spec 4.5)."""

    max_attempts: int = Field(default=5, ge=1)
    initial_backoff_seconds: float = Field(default=1.0, gt=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    max_backoff_seconds: float = Field(default=60.0, gt=0)
    jitter: bool = True
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])


class ClockSkewConfig(_StrictConfigModel):
    """How a tick's exchange timestamp is judged against the local clock.

    ``trust: exchange`` (the default) treats the exchange timestamp as the truth and
    uses the local clock only when a tick carries no exchange time at all. That is
    the correct rule for a machine whose clock drifts: the flag exists to catch bad
    *data*, and a slow PC clock is not bad data. On 2026-10-27 one such machine
    stored 899,915 ticks flagged ``SUSPECT`` that all carried valid exchange stamps.

    ``trust: machine`` restores the strict rule - a tick whose exchange stamp is more
    than ``tolerance_seconds`` ahead of the local receive time is ``SUSPECT``.

    ``report_offset`` publishes the measured ``exchange - received`` offset in the
    heartbeat, so a drifting clock stays visible without being written into the data.
    """

    trust: Literal["exchange", "machine"] = "exchange"
    tolerance_seconds: float = Field(default=2.0, ge=0)
    report_offset: bool = True


class WebsocketConfig(_StrictConfigModel):
    """KiteTicker connection budget (spec 4.4/4.6).

    Facts this encodes: at most ``max_instruments`` instruments per socket and
    ``max_connections`` sockets per API key; ``mode`` determines the packet size
    (ltp 8B, quote 44B, full 184B) and therefore which fields exist at all -
    only ``full`` carries OI and market depth.
    """

    max_instruments: int = Field(default=3000, gt=0)
    max_connections: int = Field(default=3, gt=0)
    mode: Literal["ltp", "quote", "full"] = "full"
    connect_timeout_seconds: float = Field(default=30.0, gt=0)
    reconnect_initial_seconds: float = Field(default=2.0, gt=0)
    reconnect_multiplier: float = Field(default=2.0, ge=1)
    reconnect_max_seconds: float = Field(default=60.0, gt=0)
    #: Handed to ``KiteTicker(reconnect_max_tries=...)``; the SDK owns its own
    #: exponential backoff, we only bound the number of attempts.
    max_reconnect_attempts: int = Field(default=50, ge=0)
    stall_timeout_seconds: float = Field(default=60.0, gt=0)
    #: Exchange time is authoritative; see :class:`ClockSkewConfig`.
    clock_skew: ClockSkewConfig = Field(default_factory=ClockSkewConfig)

    @model_validator(mode="after")
    def _check_backoff(self) -> WebsocketConfig:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError(
                "reconnect_max_seconds must be >= reconnect_initial_seconds "
                f"({self.reconnect_max_seconds} < {self.reconnect_initial_seconds})"
            )
        return self


class HeartbeatConfig(_StrictConfigModel):
    """Capture heartbeat file (the machine-readable "is capture alive?" signal)."""

    interval_seconds: float = Field(default=10.0, gt=0)
    #: A heartbeat older than this means the capture process is unhealthy.
    stale_after_seconds: float = Field(default=90.0, gt=0)
    filename: str = Field(default="capture.json", min_length=1)

    @model_validator(mode="after")
    def _stale_after_interval(self) -> HeartbeatConfig:
        if self.stale_after_seconds <= self.interval_seconds:
            raise ValueError(
                "heartbeat.stale_after_seconds must exceed heartbeat.interval_seconds "
                f"({self.stale_after_seconds} <= {self.interval_seconds}), otherwise a healthy "
                "process always looks stale"
            )
        return self


class GapConfig(_StrictConfigModel):
    """Tick-stream gap detection. Tick streams have no fixed row count."""

    tick_gap_seconds: float = Field(default=30.0, gt=0)
    #: Gaps are logged at most this many times per session, then rolled up into
    #: the coverage ledger - a broken feed must not produce a 50 MB log file.
    log_limit_per_session: int = Field(default=200, ge=0)


class CaptureStorageConfig(_StrictConfigModel):
    """DuckDB batching plus the Parquet mirror written alongside it."""

    flush_rows: int = Field(default=5000, ge=1)
    flush_seconds: float = Field(default=5.0, gt=0)
    #: Hard cap: beyond this the buffer spills to a Parquet part file rather
    #: than dropping rows (a slow disk must not cost us ticks).
    max_buffer_rows: int = Field(default=100_000, ge=1)
    mirror_parquet: bool = True
    parquet_compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd"
    parquet_partition: Literal["day", "none"] = "day"

    @model_validator(mode="after")
    def _buffer_ge_flush(self) -> CaptureStorageConfig:
        if self.max_buffer_rows < self.flush_rows:
            raise ValueError(
                f"storage.max_buffer_rows ({self.max_buffer_rows}) must be >= "
                f"storage.flush_rows ({self.flush_rows})"
            )
        return self


class InstrumentCacheConfig(_StrictConfigModel):
    """Where Kite's daily instrument dump is cached on disk."""

    cache_dir: str = Field(default="raw/instruments", min_length=1)
    refresh_ttl_hours: float = Field(default=12.0, gt=0)
    exchanges: list[str] = Field(default_factory=lambda: ["NSE", "NFO"])

    @field_validator("exchanges")
    @classmethod
    def _exchanges(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip().upper() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("instruments.exchanges must contain at least one exchange")
        return sorted(set(cleaned))


class HistoricalFetchConfig(_StrictConfigModel):
    """REST historical-candle fetch policy (index backfill / daily top-up).

    Kite applies its own (undocumented, per-interval) range limits and returns
    an error rather than truncating. ANJANEYA therefore *plans* its windows from
    ``max_days_per_request`` and halves a window when the server still objects.
    """

    default_interval: str = Field(default="minute", min_length=1)
    max_days_per_request: dict[str, int] = Field(
        default_factory=lambda: {
            "minute": 30,
            "3minute": 60,
            "5minute": 90,
            "10minute": 120,
            "15minute": 180,
            "30minute": 240,
            "60minute": 365,
            "day": 1800,
        }
    )
    adaptive_split_on_error: bool = True
    max_split_depth: int = Field(default=6, ge=0)

    @field_validator("default_interval")
    @classmethod
    def _known_interval(cls, value: str) -> str:
        if value not in HISTORICAL_INTERVALS:
            raise ValueError(
                f"historical.default_interval {value!r} is not a Kite interval; "
                f"expected one of {HISTORICAL_INTERVALS}"
            )
        return value

    @field_validator("max_days_per_request")
    @classmethod
    def _known_keys(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = sorted(set(value) - set(HISTORICAL_INTERVALS))
        if unknown:
            raise ValueError(
                f"historical.max_days_per_request has unknown interval(s) {unknown}; "
                f"expected keys from {HISTORICAL_INTERVALS}"
            )
        invalid = sorted(name for name, days in value.items() if days <= 0)
        if invalid:
            raise ValueError(f"historical.max_days_per_request days must be > 0, got {invalid}")
        return value

    @model_validator(mode="after")
    def _default_interval_is_configured(self) -> HistoricalFetchConfig:
        if self.default_interval not in self.max_days_per_request:
            raise ValueError(
                "historical.max_days_per_request has no entry for the default interval "
                f"{self.default_interval!r}; add one or change default_interval"
            )
        return self

    def window_days(self, interval: str) -> int:
        """Maximum days per request for ``interval`` (falling back to the smallest)."""
        if interval in self.max_days_per_request:
            return self.max_days_per_request[interval]
        return min(self.max_days_per_request.values())


class OptionChainPollConfig(_StrictConfigModel):
    """Periodic option-chain (OI/IV) polling configuration."""

    interval_seconds: int = Field(default=60, gt=0)
    quotes_per_request: int = Field(default=500, gt=0)
    use_websocket_mode_quote: bool = True


class CaptureConfig(_StrictConfigModel):
    """ANJANEYA-CAPTURE session policy (spec 4.4, 4.6, 6.1)."""

    session_start_guard_minutes: int = Field(default=15, ge=0)
    wait_for_session_open: bool = True
    auto_stop_at_session_end: bool = True
    smoke_minutes: int = Field(default=3, gt=0)
    #: ``None`` disables re-centring; a number re-resolves the strike window at
    #: that cadence so the window follows the money as NIFTY moves.
    universe_refresh_minutes: int | None = Field(default=30, gt=0)
    out_of_session_ticks: Literal["record", "drop"] = "record"
    dedupe_identical_ticks: bool = False
    status_interval_seconds: float = Field(default=60.0, gt=0)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    gaps: GapConfig = Field(default_factory=GapConfig)
    storage: CaptureStorageConfig = Field(default_factory=CaptureStorageConfig)
    instruments: InstrumentCacheConfig = Field(default_factory=InstrumentCacheConfig)
    historical: HistoricalFetchConfig = Field(default_factory=HistoricalFetchConfig)


class ZerodhaConfig(_StrictConfigModel):
    """Zerodha Kite Connect data source (credential *names* only)."""

    enabled: bool = True
    api_key_env: str = "ZERODHA_API_KEY"
    api_secret_env: str = "ZERODHA_API_SECRET"
    user_id_env: str = "ZERODHA_USER_ID"
    password_env: str = "ZERODHA_PASSWORD"
    totp_key_env: str = "ZERODHA_TOTP_KEY"
    redirect_url_env: str = "ZERODHA_REDIRECT_URL"
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    token_store: TokenStoreConfig
    retry: RetryConfig = Field(default_factory=RetryConfig)
    websocket: WebsocketConfig = Field(default_factory=WebsocketConfig)
    option_chain: OptionChainPollConfig = Field(default_factory=OptionChainPollConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)


class DataSourcesConfig(_StrictConfigModel):
    """Root model of ``config/data_sources.yaml``."""

    zerodha: ZerodhaConfig


# ---------------------------------------------------------------------------
# option_universe.yaml
# ---------------------------------------------------------------------------
class UniverseInstrumentConfig(_StrictConfigModel):
    """Contract sizing / spacing used by the universe builder."""

    lot_size: int = Field(default=75, gt=0)
    strike_interval: int = Field(default=50, gt=0)


class AtmConfig(_StrictConfigModel):
    """How the at-the-money strike is chosen."""

    reference: Literal["spot", "futures"] = "spot"
    rounding: Literal["nearest", "floor", "ceil"] = "nearest"


class SymbolTemplatesConfig(_StrictConfigModel):
    """NSE symbol templates (they change over time, so they are config)."""

    weekly: str
    monthly: str
    monthly_codes: dict[int, str]

    @model_validator(mode="after")
    def _check_codes(self) -> SymbolTemplatesConfig:
        missing = [month for month in range(1, 13) if month not in self.monthly_codes]
        if missing:
            raise ValueError(f"monthly_codes is missing months {missing}")
        return self


class ReferenceInstrumentConfig(_StrictConfigModel):
    """An always-subscribed non-option instrument.

    Either an explicit ``tradingsymbol`` (the NIFTY spot index) or an
    ``expiry_selector`` for instruments whose symbol changes every month (the
    NIFTY futures contract) - exactly one of the two must be given.
    """

    exchange: str
    tradingsymbol: str | None = None
    expiry_selector: Literal["current_month", "next_month", "current_week"] | None = None

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> ReferenceInstrumentConfig:
        provided = [item for item in (self.tradingsymbol, self.expiry_selector) if item is not None]
        if len(provided) != 1:
            raise ValueError(
                "reference instrument needs exactly one of tradingsymbol or expiry_selector, "
                f"got tradingsymbol={self.tradingsymbol!r}, expiry_selector={self.expiry_selector!r}"
            )
        return self


class ReferenceInstrumentsConfig(_StrictConfigModel):
    """Reference instruments (spot index + current futures contract)."""

    nifty_spot: ReferenceInstrumentConfig
    nifty_futures: ReferenceInstrumentConfig


class IndexHistoryConfig(_StrictConfigModel):
    """REST backfill policy for the NIFTY index (spec 6.1)."""

    instruments: list[str] = Field(default_factory=lambda: ["nifty_spot"])
    backfill_days: int = Field(default=400, gt=0)
    topup_days: int = Field(default=10, gt=0)
    #: Kite interval used for the long backfill (``day`` by default).
    backfill_interval: str = Field(default="day", min_length=1)
    #: Kite interval used for the incremental top-up (``minute`` by default).
    topup_interval: str = Field(default="minute", min_length=1)

    @field_validator("backfill_interval", "topup_interval")
    @classmethod
    def _known_interval(cls, value: str) -> str:
        if value not in HISTORICAL_INTERVALS:
            raise ValueError(
                f"index_history interval {value!r} is not a Kite interval; "
                f"expected one of {HISTORICAL_INTERVALS}"
            )
        return value

    @field_validator("instruments")
    @classmethod
    def _instruments(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("index_history.instruments must name at least one reference instrument")
        return value


class OptionUniverseConfig(_StrictConfigModel):
    """Root model of ``config/option_universe.yaml``."""

    underlying: str = "NIFTY"
    exchange: str = "NFO"
    currency: str = "INR"
    instrument: UniverseInstrumentConfig = Field(default_factory=UniverseInstrumentConfig)
    strikes_offset: list[int]
    expiries: list[Literal["current_week", "next_week", "current_month", "next_month"]]
    atm: AtmConfig = Field(default_factory=AtmConfig)
    symbol_templates: SymbolTemplatesConfig
    reference_instruments: ReferenceInstrumentsConfig
    index_history: IndexHistoryConfig = Field(default_factory=IndexHistoryConfig)

    @field_validator("strikes_offset")
    @classmethod
    def _offsets(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("strikes_offset must contain at least one offset")
        interval = 50
        invalid = [offset for offset in value if offset % interval != 0]
        if invalid:
            raise ValueError(f"strikes_offset must be NIFTY strike multiples of 50, got {invalid}")
        return sorted(set(value))

    @field_validator("expiries")
    @classmethod
    def _expiries(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("expiries must contain at least one expiry selector")
        return value


# ---------------------------------------------------------------------------
# indicators.yaml
# ---------------------------------------------------------------------------
class MovingAverageSpec(_StrictConfigModel):
    """One moving-average definition (spec 8 / 10)."""

    name: str = Field(min_length=1)
    type: Literal["ema", "sma"]
    period: int = Field(gt=1)
    source: Literal["open", "high", "low", "close", "hl2", "hlc3"] = "close"
    #: Required for the NIFTY score combinations, optional for the option engine.
    role: Literal["fast", "medium", "long"] | None = None


class SlopeConfig(_StrictConfigModel):
    """Slope lookback for the normalized-slope indicator (spec 10)."""

    k: int = Field(default=5, gt=0)


class PersistenceConfig(_StrictConfigModel):
    """How many bars must agree to call a trend persistent."""

    window: int = Field(default=5, gt=1)


class SpreadVelocityConfig(_StrictConfigModel):
    """Lookback for spread velocity (spec 10)."""

    k: int = Field(default=5, gt=0)


class AtrConfig(_StrictConfigModel):
    """Average True Range configuration."""

    period: int = Field(default=14, gt=1)
    method: Literal["wilder", "simple"] = "wilder"


class MomentumConfig(_StrictConfigModel):
    """Lookback for the ATR-normalised momentum feature (spec 10)."""

    period: int = Field(default=10, gt=1)


class NiftyMovingAveragesConfig(_StrictConfigModel):
    """Three MA combinations evaluated side by side (spec 8)."""

    combination_1: list[MovingAverageSpec]
    combination_2: list[MovingAverageSpec]
    combination_3: list[MovingAverageSpec]

    @model_validator(mode="after")
    def _validate_combinations(self) -> NiftyMovingAveragesConfig:
        for field_name in ("combination_1", "combination_2", "combination_3"):
            specs = getattr(self, field_name)
            if not specs:
                raise ValueError(f"{field_name} must define at least one moving average")
            names = [spec.name for spec in specs]
            if len(names) != len(set(names)):
                raise ValueError(f"{field_name} contains duplicate moving-average names: {names}")
            missing_roles = [spec.name for spec in specs if spec.role is None]
            if missing_roles:
                raise ValueError(
                    f"{field_name} must assign a role (fast/medium/long) to every moving "
                    f"average; missing on {missing_roles}"
                )
        return self


class ScoreWeightsConfig(_StrictConfigModel):
    """Weights for the fast/medium/long score aggregation."""

    fast: float = Field(ge=0, le=1)
    medium: float = Field(ge=0, le=1)
    long: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> ScoreWeightsConfig:
        total = self.fast + self.medium + self.long
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fast + medium + long weights must sum to 1.0, got {total}")
        return self


class RegimeThresholdsConfig(_StrictConfigModel):
    """Composite-score cut-offs for the NIFTY regime labels."""

    strong_bullish: float = Field(gt=0, le=1)
    bullish: float = Field(gt=0, le=1)
    bullish_developing: float = Field(gt=0, le=1)
    bearish_developing: float = Field(lt=0, ge=-1)
    bearish: float = Field(lt=0, ge=-1)
    strong_bearish: float = Field(lt=0, ge=-1)

    @model_validator(mode="after")
    def _monotonic(self) -> RegimeThresholdsConfig:
        ordered = (
            self.strong_bearish,
            self.bearish,
            self.bearish_developing,
            0.0,
            self.bullish_developing,
            self.bullish,
            self.strong_bullish,
        )
        if any(later <= earlier for earlier, later in zip(ordered, ordered[1:], strict=False)):
            raise ValueError(
                "regime thresholds must increase strictly from strong_bearish to strong_bullish"
            )
        return self


class NiftyScoresConfig(_StrictConfigModel):
    """NIFTY score weights and regime thresholds (spec 9)."""

    weights: ScoreWeightsConfig
    regime_thresholds: RegimeThresholdsConfig


class SpreadPairConfig(_StrictConfigModel):
    """The two averages the published option spread is measured between (spec 10).

    Spec 10 names the reading ``EMA20_EMA200_spread`` and spec 18 fixes the stored
    column name, so the pair is *named* in the config rather than inferred from the
    stack: the shipped stack also carries EMA 8 low/high, and "fastest minus
    slowest" would have silently meant EMA 8 - EMA 200.
    """

    fast: str = Field(min_length=1)
    slow: str = Field(min_length=1)


class OptionTrendConfig(_StrictConfigModel):
    """Option-side trend engine (spec 10) - intentionally short-window."""

    moving_averages: list[MovingAverageSpec]
    #: flag | skip | shorter_window - see config/indicators.yaml comments.
    insufficient_history_policy: Literal["flag", "skip", "shorter_window"] = "flag"
    min_candles_for_full_set: int = Field(default=200, gt=0)
    #: Which two configured averages the spread columns sit between. ``None`` falls
    #: back to the slowest minus the fastest of ``moving_averages``.
    spread_pair: SpreadPairConfig | None = None

    @field_validator("moving_averages")
    @classmethod
    def _non_empty_unique(cls, value: list[MovingAverageSpec]) -> list[MovingAverageSpec]:
        if not value:
            raise ValueError("option_trend.moving_averages must not be empty")
        names = [spec.name for spec in value]
        if len(names) != len(set(names)):
            raise ValueError(f"option_trend moving-average names must be unique, got {names}")
        return value

    @model_validator(mode="after")
    def _spread_pair_is_real(self) -> OptionTrendConfig:
        if self.spread_pair is None:
            return self
        by_name = {spec.name: spec for spec in self.moving_averages}
        for side, name in (("fast", self.spread_pair.fast), ("slow", self.spread_pair.slow)):
            if name not in by_name:
                raise ValueError(
                    f"option_trend.spread_pair.{side} names {name!r}, which is not one of "
                    f"option_trend.moving_averages {sorted(by_name)}"
                )
        if by_name[self.spread_pair.fast].period > by_name[self.spread_pair.slow].period:
            raise ValueError(
                "option_trend.spread_pair.fast must be the shorter-period average of the pair "
                f"({self.spread_pair.fast} is slower than {self.spread_pair.slow})"
            )
        return self



class CandleBehaviourConfig(_StrictConfigModel):
    """Candle shape classification parameters (spec 11)."""

    epsilon: float = Field(default=1e-9, gt=0)
    small_body_ratio: float = Field(default=0.25, ge=0, le=1)
    long_wick_ratio: float = Field(default=0.50, ge=0, le=1)
    engulf_min_ratio: float = Field(default=1.0, gt=0)


class SwingConfig(_StrictConfigModel):
    """Swing detection - ``right_bars`` is a confirmation delay, never zero."""

    left_bars: int = Field(default=3, ge=1)
    right_bars: int = Field(default=3, ge=1)
    max_swings: int = Field(default=10, ge=1)


class LevelInteractionConfig(_StrictConfigModel):
    """ATR-normalized proximity thresholds for level interaction states."""

    approaching_atr_fraction: float = Field(default=0.25, gt=0)
    touch_atr_fraction: float = Field(default=0.05, gt=0)
    min_closes_to_hold: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> LevelInteractionConfig:
        if self.touch_atr_fraction >= self.approaching_atr_fraction:
            raise ValueError("touch_atr_fraction must be smaller than approaching_atr_fraction")
        return self


class SupportResistanceWeightsConfig(_StrictConfigModel):
    """Weights of the S/R strength score (must sum to 1.0)."""

    touches: float = Field(ge=0, le=1)
    rejections: float = Field(ge=0, le=1)
    holds: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    distance: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums(self) -> SupportResistanceWeightsConfig:
        total = self.touches + self.rejections + self.holds + self.recency + self.distance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"S/R strength weights must sum to 1.0, got {total}")
        return self


class SupportResistanceConfig(_StrictConfigModel):
    """Support/resistance strength model (spec 15)."""

    weights: SupportResistanceWeightsConfig


class GreeksConfig(_StrictConfigModel):
    """Option IV/greeks model (spec 16, Stage 3.3).

    Every number here is a *documented assumption* that gets recorded in the run
    summary, not a fitted parameter: an IV is only comparable across runs if the
    rate, the dividend assumption and the time convention that produced it are the
    same. Nothing is fetched from a curve - see ``core/options/greeks.py``.
    """

    #: Only ``black_scholes`` exists today; the field is here so a second model
    #: (e.g. a local-vol or futures-carry model) is a config value, not a rewrite.
    model: Literal["black_scholes"] = "black_scholes"
    #: Annualised, *continuously compounded* risk-free rate (0.065 = 6.5%).
    risk_free_rate: float = Field(default=0.065, ge=0.0, le=1.0)
    #: Continuous dividend yield of the underlying. ``0.0`` means "ignore carry":
    #: NIFTY options are priced off the index, and the index's dividends are not
    #: observable tick by tick (spec 16 documents this limitation).
    dividend_yield: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Which candle field the IV is solved from: ``close`` (the bar's last trade,
    #: the same price ``actual_option_move`` uses), ``hl2`` or ``ohlc4``.
    price_field: Literal["close", "hl2", "ohlc4"] = "close"
    #: ACT/``year_days`` year used to turn a time-to-expiry into a year fraction.
    year_days: int = Field(default=365, gt=0)
    #: Floor on the time to expiry, in minutes. Without it the last bar of an
    #: expiry day divides by ~zero and produces an IV of whatever the float noise
    #: says; one session (375 minutes) is the documented floor.
    min_time_to_expiry_minutes: float = Field(default=375.0, gt=0)
    #: IV search bracket, as annualised vol fractions (0.005 = 0.5%).
    volatility_floor: float = Field(default=0.005, gt=0)
    volatility_ceiling: float = Field(default=5.0, gt=0)
    #: Bisection stops once the bracket is narrower than this (in vol units).
    iv_tolerance: float = Field(default=1e-6, gt=0)
    #: Safety net for a degenerate bracket; the bisection normally stops far earlier.
    iv_max_iterations: int = Field(default=100, gt=1)

    @model_validator(mode="after")
    def _bracket_increasing(self) -> GreeksConfig:
        if self.volatility_ceiling <= self.volatility_floor:
            raise ValueError(
                "greeks.volatility_ceiling must be above greeks.volatility_floor "
                f"(got {self.volatility_ceiling} <= {self.volatility_floor})"
            )
        return self


class LevelsConfig(_StrictConfigModel):
    """Levels, level-interaction and S/R configuration."""

    swing: SwingConfig = Field(default_factory=SwingConfig)
    interaction: LevelInteractionConfig = Field(default_factory=LevelInteractionConfig)
    support_resistance: SupportResistanceConfig


class IndicatorsConfig(_StrictConfigModel):
    """Root model of ``config/indicators.yaml``."""

    slope: SlopeConfig = Field(default_factory=SlopeConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    spread_velocity: SpreadVelocityConfig = Field(default_factory=SpreadVelocityConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    atr: AtrConfig = Field(default_factory=AtrConfig)
    nifty_moving_averages: NiftyMovingAveragesConfig
    nifty_scores: NiftyScoresConfig
    option_trend: OptionTrendConfig
    candle: CandleBehaviourConfig = Field(default_factory=CandleBehaviourConfig)
    #: Option pricing assumptions (spec 16). Optional so a Phase-1 style
    #: ``indicators.yaml`` still loads; the shipped file documents the section.
    greeks: GreeksConfig = Field(default_factory=GreeksConfig)
    levels: LevelsConfig


# ---------------------------------------------------------------------------
# model.yaml
# ---------------------------------------------------------------------------
class DataGatesConfig(_StrictConfigModel):
    """Minimum accumulation before each class of conclusion is allowed (4.3)."""

    exploratory_min_days: int = Field(default=20, gt=0)
    # Mirrors config/model.yaml's relaxed gate (60 -> 45, Phase 9) so an omitted key and the
    # shipped file can never describe two different policies.
    ml_training_min_days: int = Field(default=45, gt=0)
    meaningful_backtest_min_days: int = Field(default=120, gt=0)
    seasonality_min_days: int = Field(default=250, gt=0)

    @model_validator(mode="after")
    def _increasing(self) -> DataGatesConfig:
        gates = (
            self.exploratory_min_days,
            self.ml_training_min_days,
            self.meaningful_backtest_min_days,
            self.seasonality_min_days,
        )
        if any(later < earlier for earlier, later in zip(gates, gates[1:], strict=False)):
            raise ValueError("data gates must be non-decreasing from exploratory to seasonality")
        return self


class LabelsConfig(_StrictConfigModel):
    """Label construction policy (spec 20)."""

    normalization: Literal["atr", "percent", "fixed"] = "atr"
    normalization_param: float = Field(default=0.5, gt=0)
    percent_threshold: float = Field(default=0.0015, gt=0)
    fixed_threshold: float = Field(default=1.0, gt=0)
    #: How far back the "prior move" is measured when a clearing move is called a
    #: continuation or a reversal (``core/labels/label_builder.py``).
    trend_lookback: int = Field(default=5, gt=0)
    classes: list[str] = Field(
        default_factory=lambda: [
            "UP_CONTINUATION",
            "DOWN_CONTINUATION",
            "RANGE",
            "REVERSAL",
        ]
    )
    version: str = Field(default="v1", min_length=1)

    @field_validator("classes")
    @classmethod
    def _known_classes(cls, value: list[str]) -> list[str]:
        allowed = {"UP_CONTINUATION", "DOWN_CONTINUATION", "RANGE", "REVERSAL"}
        unknown = [item for item in value if item not in allowed]
        if unknown:
            raise ValueError(f"Unknown label classes {unknown}; allowed: {sorted(allowed)}")
        if len(value) < 2:
            raise ValueError("At least two label classes are required")
        return value


class DatasetConfig(_StrictConfigModel):
    """The training matrix (spec 21's ``dataset_builder``).

    One dataset is built per configured horizon, keyed ``(timestamp, tradingsymbol)``:
    the feature row as it stood at that moment, joined to the forward label of the same
    moment. Which *timeframe* those rows come from is a modelling choice (the features
    table is keyed by timestamp + contract, with the timeframe as a column), and so is
    the split between numeric and categorical features - a tree model needs to be told
    which columns are labels rather than numbers, and only the researcher knows.
    """

    #: Which candle timeframe a training row is built from (must be a configured one).
    timeframe: str = "1m"
    #: Feature columns the model should treat as categorical. Validated to be real
    #: feature columns: a typo here would otherwise silence a feature instead of
    #: failing, which is exactly the kind of thing that goes unnoticed for months.
    categorical_columns: list[str] = Field(
        default_factory=lambda: [
            "option_type",
            "nifty_regime",
            "candle_direction",
            "swing_label",
            "market_structure_state",
            "event_prev_day_high_state",
            "event_prev_day_low_state",
            "event_today_high_state",
            "event_today_low_state",
            "expiry_cycle_phase",
            "time_of_day_bucket",
        ]
    )

    @field_validator("timeframe")
    @classmethod
    def _known_timeframe(cls, value: str) -> str:
        if value not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"dataset.timeframe {value!r} is not supported; supported: "
                f"{', '.join(SUPPORTED_TIMEFRAMES)}"
            )
        return value

    @field_validator("categorical_columns")
    @classmethod
    def _known_columns(cls, value: list[str]) -> list[str]:
        names = list(value)
        if len(set(names)) != len(names):
            raise ValueError(f"dataset.categorical_columns must be unique, got {names}")
        # Imported lazily so the loader keeps importing nothing but constants at module
        # import: the guard below is the only place that needs the schema's contract.
        from core.ml.dataset import assert_no_label_leakage

        unknown = [name for name in names if name not in FEATURE_COLUMNS]
        if unknown:
            raise ValueError(
                f"dataset.categorical_columns names unknown feature column(s) {unknown}; "
                "they must be columns of the stored feature row (core/constants.py)"
            )
        assert_no_label_leakage(names)
        return names


class WalkForwardConfig(_StrictConfigModel):
    """Purged walk-forward validation (spec 22)."""

    train_window_days: int = Field(default=90, gt=0)
    validate_window_days: int = Field(default=15, gt=0)
    embargo_minutes: int = Field(default=30, ge=0)
    holdout_days: int = Field(default=30, gt=0)
    step_days: int = Field(default=15, gt=0)
    purge: bool = True

    @model_validator(mode="after")
    def _embargo_covers_horizon(self) -> WalkForwardConfig:
        if self.embargo_minutes <= 0:
            raise ValueError(
                "embargo_minutes must be > 0: it has to cover at least the longest label horizon"
            )
        if self.step_days > self.train_window_days:
            raise ValueError("step_days must not exceed train_window_days")
        return self


class TrainingConfig(_StrictConfigModel):
    """The LightGBM trainer's numbers (spec 21).

    Deliberately an explicit set of named fields rather than a free-form ``params`` dict:
    a value that is named is validated here, hashed into the config hash, and copied into
    every model artefact - which is the whole of "this model can be retrained". A dict
    would be none of those things.

    Nothing here is tuned on the validation folds by default. In particular
    ``early_stopping_rounds`` is **0 (off)**: stopping on the fold's own validation window
    would make the number reported for that window optimistic, and a research engine whose
    headline metric is chosen by the metric is not measuring anything. Turn it on
    knowingly - the run then says so in its report and in the model metadata.

    ``class_weight: balanced`` is the one concession to the label distribution: the four
    classes are not equally frequent, and an unweighted tree can reach a good log loss by
    never predicting the rare one.
    """

    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    num_leaves: int = Field(default=31, ge=2)
    #: ``-1`` leaves the depth unbounded (LightGBM's default; ``num_leaves`` still applies).
    max_depth: int = Field(default=-1, ge=-1)
    min_child_samples: int = Field(default=40, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l1: float = Field(default=0.0, ge=0)
    lambda_l2: float = Field(default=0.0, ge=0)
    class_weight: Literal["none", "balanced"] = "none"
    early_stopping_rounds: int = Field(default=0, ge=0)
    #: ``0`` means "every core"; the trainer pins LightGBM's own thread count to this.
    n_jobs: int = Field(default=0, ge=0)


class CostsConfig(_StrictConfigModel):
    """Indian F&O cost model (spec 23). Verify rates before trusting P&L."""

    brokerage_per_order: float = Field(default=20.0, ge=0)
    stt_rate: float = Field(default=0.000625, ge=0)
    exchange_txn_rate: float = Field(default=0.0003503, ge=0)
    sebi_rate: float = Field(default=0.000001, ge=0)
    gst_rate: float = Field(default=0.18, ge=0, le=1)
    stamp_duty_rate: float = Field(default=0.00003, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    use_captured_spread: bool = False


class ExecutionConfig(_StrictConfigModel):
    """How a signal becomes a fill (no look-ahead allowed)."""

    signal_at: Literal["candle_close"] = "candle_close"
    fill_at: Literal["next_open", "bid_ask"] = "next_open"
    latency_ms: int = Field(default=0, ge=0)


class BacktestSignalConfig(_StrictConfigModel):
    """The threshold rule that turns a predicted distribution into a position (spec 23).

    This is deliberately the *only* signal rule ANJANEYA implements, and it is a stand-in
    for spec 24's decision layer rather than a claim about edges: enter the option side whose
    directional class the model gives at least ``min_probability``, optionally only when that
    class is the model's top pick. Every backtest report prints that sentence, and
    ``model.labels.classes`` is cross-checked against these names so a typo cannot produce a
    rule that never fires.
    """

    mode: Literal["probability_threshold"] = "probability_threshold"
    min_probability: float = Field(default=0.40, gt=0, le=1)
    require_argmax: bool = True
    directions: dict[str, str] = Field(
        default_factory=lambda: {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}
    )

    @field_validator("directions")
    @classmethod
    def _option_types(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("backtest.signal.directions must map at least one option type")
        unknown = sorted(set(value) - {"CE", "PE"})
        if unknown:
            raise ValueError(
                f"backtest.signal.directions has unknown option type(s) {unknown}; "
                "expected CE and/or PE"
            )
        blank = sorted(name for name, klass in value.items() if not str(klass).strip())
        if blank:
            raise ValueError(f"backtest.signal.directions has an empty class for {blank}")
        return {str(name).upper(): str(klass) for name, klass in value.items()}


class BacktestSizingConfig(_StrictConfigModel):
    """How big one simulated position is (one NIFTY lot is 75 units, configurable)."""

    lots: int = Field(default=1, ge=1)
    lot_size: int = Field(default=75, ge=1)

    @property
    def quantity(self) -> int:
        """Units per trade (``lots * lot_size``)."""
        return self.lots * self.lot_size


class BacktestExitConfig(_StrictConfigModel):
    """When a position ends. Every exit is configurable and every one is counted."""

    #: ``None`` turns that exit off entirely.
    stop_loss_atr: float | None = Field(default=1.0, gt=0)
    target_atr: float | None = Field(default=1.5, gt=0)
    max_holding_minutes: float | None = Field(default=30.0, gt=0)
    #: A research backtest on weekly options does not hold overnight: the position is
    #: closed at the last bar of its session.
    flat_at_session_end: bool = True
    #: Bars cannot say which of stop/target came first when a bar touches both.
    intrabar_priority: Literal["stop_first", "target_first"] = "stop_first"
    exit_on_opposite_signal: bool = True


class BacktestConfig(_StrictConfigModel):
    """Root of ``model.yaml``'s ``backtest:`` section (spec 23)."""

    signal: BacktestSignalConfig = Field(default_factory=BacktestSignalConfig)
    sizing: BacktestSizingConfig = Field(default_factory=BacktestSizingConfig)
    exit: BacktestExitConfig = Field(default_factory=BacktestExitConfig)


class DecisionConfig(_StrictConfigModel):
    """Spec 24's decision layer (``--mode decide``): a distribution plus a rule, per bar.

    This is the *separate* piece the master prompt names ("ANJANEYA never predicts BUY/SELL.
    It predicts future behaviour over multiple horizons; the decision layer (Phase 24) is
    separate"). It lives beside ``backtest.signal`` rather than replacing it on purpose: the
    backtest's threshold rule is a *stand-in* that exists so costs can be measured, and this
    block is the policy an operator reads, tunes and holds to account - the two are allowed to
    differ, and ``--mode decide`` records which one made a decision.

    Values parallel ``backtest.signal`` so a reader recognises them, but they are independent:
    changing one never silently changes the other. ``directions`` values are cross-checked
    against ``labels.classes`` at load, exactly as the backtest's are, so a typo cannot produce
    a rule that never fires. Nothing here places an order and nothing here simulates an exit -
    ``stop_loss_atr``/``target_atr``/``max_holding_minutes`` are recorded as the plan the
    decision *intended*, and replaying them against stored prices is ``--mode backtest``'s job.
    """

    #: The master-spec phase this block implements (printed in every artefact).
    spec: str = "24"
    #: The directional class the option side needs must reach at least this probability.
    min_probability: float = Field(default=0.40, gt=0, le=1)
    #: ``true`` = only decide when the directional class is also the model's top class.
    require_argmax: bool = True
    #: The smallest lead over the runner-up class an entry may have (``0`` = no floor).
    min_margin: float = Field(default=0.0, ge=0, lt=1)
    #: ``option_type -> the labels.classes name`` that side needs (checked at load).
    directions: dict[str, str] = Field(
        default_factory=lambda: {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}
    )
    lots: int = Field(default=1, ge=1)
    #: One NIFTY lot. Verify against the current contract specification before reading a size.
    lot_size: int = Field(default=75, ge=1)
    #: The most entries one trading day may carry; ``None`` = no throttle.
    max_decisions_per_day: int | None = Field(default=None, ge=1)
    #: The exit plan the decision records (stated, never simulated here).
    stop_loss_atr: float | None = Field(default=1.0, gt=0)
    target_atr: float | None = Field(default=1.5, gt=0)
    max_holding_minutes: float | None = Field(default=30.0, gt=0)
    flat_at_session_end: bool = True

    @property
    def quantity(self) -> int:
        """Units one entry would size to (``lots * lot_size``)."""
        return self.lots * self.lot_size

    @field_validator("directions")
    @classmethod
    def _option_types(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("decision.directions must map at least one option type")
        unknown = sorted(set(value) - {"CE", "PE"})
        if unknown:
            raise ValueError(
                f"decision.directions has unknown option type(s) {unknown}; "
                "expected CE and/or PE"
            )
        blank = sorted(name for name, klass in value.items() if not str(klass).strip())
        if blank:
            raise ValueError(f"decision.directions has an empty class for {blank}")
        return {str(name).upper(): str(klass) for name, klass in value.items()}


class EvaluationConfig(_StrictConfigModel):
    """Evaluation, baselines and multiple-testing policy (spec 21/25)."""

    probability_metrics: list[str] = Field(
        default_factory=lambda: ["log_loss", "brier_score", "calibration_curve"]
    )
    baselines: list[str] = Field(
        default_factory=lambda: ["always_up", "always_range", "random", "nifty_regime_only"]
    )
    preliminary_min_samples: int = Field(default=30, gt=0)
    multiple_testing_correction: Literal["none", "benjamini_hochberg", "bonferroni"] = (
        "benjamini_hochberg"
    )
    significance_level: float = Field(default=0.05, gt=0, lt=1)


class ExplainabilitySectionConfig(_StrictConfigModel):
    """One block of an explanation (spec 26's ``NIFTY``, ``OPTION``, ... list).

    The spec says the values it names are *examples*, so a section is defined here rather
    than in the code: ``groups`` names stored feature *groups*
    (:data:`core.features.feature_builder.FEATURE_GROUPS`) and ``columns`` adds individual
    stored feature columns. Both are validated against the feature contract at load, so a
    typo is a failure instead of an explanation that quietly says nothing.
    """

    #: Stable identifier, used as the JSON key and the Markdown anchor.
    key: str = Field(min_length=1)
    #: Human title, printed and written as the section heading.
    title: str = Field(min_length=1)
    #: Feature groups to render, in group order.
    groups: list[str] = Field(default_factory=list)
    #: Individual feature columns to render after the groups.
    columns: list[str] = Field(default_factory=list)

    @field_validator("groups", "columns")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        names = list(value)
        if len(set(names)) != len(names):
            raise ValueError(f"explainability section lists a name twice: {names}")
        return names

    @model_validator(mode="after")
    def _something_to_render(self) -> ExplainabilitySectionConfig:
        if not self.groups and not self.columns:
            raise ValueError(
                f"explainability section {self.key!r} names neither groups nor columns, so "
                "it would render nothing; remove it or give it a source"
            )
        return self


class ExplainabilityConfig(_StrictConfigModel):
    """Spec 26's per-prediction explanation (``--mode explain``).

    Spec 26 asks for a *plain structured* explanation of every prediction and lists the
    blocks it expects; the list is described as examples, so the blocks - and the columns
    inside them - are configuration. ``probabilities`` is the "top class probabilities" the
    spec asks for, taken from the model's own metadata rather than from a fixed class list.
    """

    #: The master-spec section this structure comes from (printed in every artefact).
    spec: str = "26"
    #: How many of the model's class probabilities an explanation prints.
    probabilities: int = Field(default=3, ge=1)
    #: Decimals for every number in an explanation.
    digits: int = Field(default=4, ge=0, le=12)
    #: How many predictions ``--mode explain`` describes when ``--rows`` is not given.
    rows: int = Field(default=5, ge=1)
    #: Printed and stored where a stored feature is NULL at that instant: an explanation
    #: may never turn "unknown" into 0, and the word belongs in the configuration next to
    #: the rest of the presentation rules.
    missing_text: str = "unknown"
    sections: list[ExplainabilitySectionConfig] = Field(
        default_factory=lambda: [
            ExplainabilitySectionConfig(key="nifty", title="NIFTY", groups=["nifty"]),
            ExplainabilitySectionConfig(key="option", title="OPTION", groups=["option"]),
            ExplainabilitySectionConfig(
                key="price_action",
                title="PRICE ACTION",
                groups=["levels", "level_events", "structure"],
            ),
            ExplainabilitySectionConfig(key="candle", title="CANDLE", groups=["candle"]),
            ExplainabilitySectionConfig(
                key="support_resistance",
                title="SUPPORT / RESISTANCE",
                groups=["support_resistance"],
            ),
            ExplainabilitySectionConfig(
                key="context", title="CONTEXT", groups=["context", "relative"]
            ),
        ]
    )

    @field_validator("sections")
    @classmethod
    def _unique_keys(
        cls, value: list[ExplainabilitySectionConfig]
    ) -> list[ExplainabilitySectionConfig]:
        if not value:
            raise ValueError(
                "explainability.sections must not be empty: an explanation with no "
                "sections would print nothing"
            )
        keys = [section.key for section in value]
        if len(set(keys)) != len(keys):
            raise ValueError(f"explainability.sections keys must be unique, got {keys}")
        return value

    @model_validator(mode="after")
    def _names_are_stored_features(self) -> ExplainabilityConfig:
        """Every group and column must exist, and no column may be explained twice."""
        # Imported lazily so the loader keeps importing nothing but constants at module
        # import: this is the one place that needs the feature contract's group map.
        from core.features.feature_builder import FEATURE_GROUPS
        from core.ml.explainer import AI_SECTION_KEY, RESERVED_EXPLANATION_GROUPS

        seen: dict[str, str] = {}
        for section in self.sections:
            if section.key == AI_SECTION_KEY:
                raise ValueError(
                    f"explainability.sections uses the reserved key {AI_SECTION_KEY!r}, which "
                    "belongs to the model's own block (always appended last); rename the "
                    "configured section"
                )
            reserved = [name for name in section.groups if name in RESERVED_EXPLANATION_GROUPS]
            if reserved:
                raise ValueError(
                    f"explainability section {section.key!r} names the reserved feature "
                    f"group(s) {reserved}; identity is carried by every explanation's header "
                    "and provenance by its audit block, so neither is a section"
                )
            unknown_groups = [name for name in section.groups if name not in FEATURE_GROUPS]
            if unknown_groups:
                raise ValueError(
                    f"explainability section {section.key!r} names unknown feature "
                    f"group(s) {unknown_groups}; known groups: {sorted(FEATURE_GROUPS)}"
                )
            unknown_columns = [name for name in section.columns if name not in FEATURE_COLUMNS]
            if unknown_columns:
                raise ValueError(
                    f"explainability section {section.key!r} names unknown feature "
                    f"column(s) {unknown_columns}; they must be columns of the stored "
                    "feature row (core/constants.py)"
                )
            columns = list(section.columns)
            for name in section.groups:
                columns.extend(FEATURE_GROUPS[name])
            for column in columns:
                if column in seen:
                    raise ValueError(
                        f"feature column {column!r} is explained twice (sections "
                        f"{seen[column]!r} and {section.key!r}); one column, one line"
                    )
                seen[column] = section.key
        return self


def _default_importance_topics() -> list[ImportanceTopicConfig]:
    """The shipped map from spec 27's sources onto the stored feature contract.

    Twelve of these are spec 27's own list; the last four claim columns the list does not name
    (the option side, the day of the week, moneyness/volume and the price-action levels) so
    that no model input is left unattributed, and they say so in their titles.
    """
    return [
        ImportanceTopicConfig(key="nifty_regime", title="NIFTY regime", groups=["nifty"]),
        ImportanceTopicConfig(key="option_ma", title="option moving averages", groups=["option"]),
        ImportanceTopicConfig(key="candle_behaviour", title="candle behaviour", groups=["candle"]),
        ImportanceTopicConfig(
            key="support_resistance", title="support / resistance", groups=["support_resistance"]
        ),
        ImportanceTopicConfig(key="structure", title="market structure", groups=["structure"]),
        ImportanceTopicConfig(
            key="price_action",
            title="price action, levels (not named by spec 27)",
            groups=["levels", "level_events"],
        ),
        ImportanceTopicConfig(key="iv", title="implied volatility", columns=["iv", "iv_change"]),
        ImportanceTopicConfig(
            key="greeks", title="greeks", columns=["delta", "gamma", "theta", "vega"]
        ),
        ImportanceTopicConfig(key="oi", title="open interest", columns=["oi", "oi_change"]),
        ImportanceTopicConfig(
            key="option_residual",
            title="option residual (expected vs actual move)",
            columns=["expected_option_move", "actual_option_move", "option_residual"],
        ),
        ImportanceTopicConfig(
            key="time_of_day", title="time of day", columns=["time_of_day_bucket"]
        ),
        ImportanceTopicConfig(
            key="cycle_phase", title="expiry-cycle phase", columns=["expiry_cycle_phase"]
        ),
        ImportanceTopicConfig(
            key="time_to_expiry", title="time to expiry", columns=["days_to_expiry"]
        ),
        ImportanceTopicConfig(
            key="day_of_week", title="day of week (not named by spec 27)", columns=["day_of_week"]
        ),
        ImportanceTopicConfig(
            key="option_contract",
            title="option side (not named by spec 27)",
            columns=["option_type"],
        ),
        ImportanceTopicConfig(
            key="option_context",
            title="option context: moneyness and volume (not named by spec 27)",
            columns=["distance_from_atm", "volume"],
        ),
    ]


class ImportanceTopicConfig(_StrictConfigModel):
    """One source of spec 27's list (``NIFTY regime``, ``option MA``, ``IV``, ...).

    The spec's list is the *question* and the feature contract is the material: a topic maps
    stored feature *groups* (:data:`core.features.feature_builder.FEATURE_GROUPS`) and
    individual feature *columns* onto one source, so "where does the information come from"
    is answered by a configured map rather than by a name pattern in the code.
    """

    #: Stable identifier, used as the JSON key and the CSV's source column.
    key: str = Field(min_length=1)
    #: Human title, printed and written as the source's name.
    title: str = Field(min_length=1)
    #: Feature groups this source claims, in group order.
    groups: list[str] = Field(default_factory=list)
    #: Individual feature columns this source claims, after the groups.
    columns: list[str] = Field(default_factory=list)

    @field_validator("groups", "columns")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        names = list(value)
        if len(set(names)) != len(names):
            raise ValueError(f"importance topic lists a name twice: {names}")
        return names

    @model_validator(mode="after")
    def _something_to_claim(self) -> ImportanceTopicConfig:
        if not self.groups and not self.columns:
            raise ValueError(
                f"importance topic {self.key!r} names neither groups nor columns, so it would "
                "claim nothing; remove it or give it a source"
            )
        return self


class ImportanceConfig(_StrictConfigModel):
    """Spec 27's feature importance (``--mode importance``).

    Spec 27 asks for native importance, permutation importance and (optionally) SHAP, and then
    for an answer to "where does the predictive information come from". So both halves are
    configuration: how the measurement is made - permutations, scoring rule, which rows,
    whether SHAP may be imported - and the map from its sources onto the stored feature
    contract.
    """

    #: The master-spec section this structure comes from (printed in every artefact).
    spec: str = "27"
    #: Decimals for every number in an artefact.
    digits: int = Field(default=4, ge=0, le=12)
    #: Permutations per feature. At least 2: the interval is the point of the exercise.
    repeats: int = Field(default=5, ge=2)
    #: The scoring rule a permutation delta is measured with.
    metric: Literal["log_loss", "accuracy"] = "log_loss"
    #: Confidence level of every interval.
    confidence: float = Field(default=0.95, gt=0, lt=1)
    #: ``holdout`` (spec 22's reserved days) or ``matrix`` (every stored row).
    sample: Literal["holdout", "matrix"] = "holdout"
    #: The most rows a measurement uses (the newest ones); the rest are counted, not hidden.
    max_rows: int = Field(default=2000, ge=1)
    #: What to do about the optional SHAP attribution: record why it cannot run (``auto``),
    #: refuse to run without it (``on``) or never import it (``off``).
    shap: Literal["auto", "on", "off"] = "auto"
    #: The most rows SHAP is asked to explain.
    shap_rows: int = Field(default=500, ge=1)
    #: Seed for the permutations; ``None`` uses ``model.random_seed``.
    seed: int | None = Field(default=None, ge=0)
    topics: list[ImportanceTopicConfig] = Field(default_factory=_default_importance_topics)

    @field_validator("topics")
    @classmethod
    def _topics_present(cls, value: list[ImportanceTopicConfig]) -> list[ImportanceTopicConfig]:
        if not value:
            raise ValueError(
                "importance.topics must not be empty: spec 27 asks which sources the "
                "information comes from"
            )
        keys = [topic.key for topic in value]
        if len(set(keys)) != len(keys):
            raise ValueError(f"importance.topics keys must be unique, got {keys}")
        return value

    @model_validator(mode="after")
    def _names_are_stored_features(self) -> ImportanceConfig:
        """Every group and column must exist, and no column may be claimed twice."""
        # Imported lazily so the loader keeps importing nothing but constants at module
        # import: this is the one place that needs the feature contract's group map.
        from core.features.feature_builder import FEATURE_GROUPS
        from core.ml.importance import RESERVED_IMPORTANCE_GROUPS, UNMAPPED_TOPIC_KEY

        if UNMAPPED_TOPIC_KEY in {topic.key for topic in self.topics}:
            raise ValueError(
                f"importance.topics uses the reserved key {UNMAPPED_TOPIC_KEY!r}, which belongs "
                "to the features no configured source claims; rename the configured topic"
            )
        seen: dict[str, str] = {}
        for topic in self.topics:
            reserved = [name for name in topic.groups if name in RESERVED_IMPORTANCE_GROUPS]
            if reserved:
                raise ValueError(
                    f"importance topic {topic.key!r} claims the reserved feature group(s) "
                    f"{reserved}; identity is how a row is found and provenance describes its "
                    "writer, so neither is a source of market information"
                )
            unknown_groups = [name for name in topic.groups if name not in FEATURE_GROUPS]
            if unknown_groups:
                raise ValueError(
                    f"importance topic {topic.key!r} names unknown feature group(s) "
                    f"{unknown_groups}; known groups: {sorted(FEATURE_GROUPS)}"
                )
            unknown_columns = [name for name in topic.columns if name not in FEATURE_COLUMNS]
            if unknown_columns:
                raise ValueError(
                    f"importance topic {topic.key!r} names unknown feature column(s) "
                    f"{unknown_columns}; they must be columns of the stored feature row "
                    "(core/constants.py)"
                )
            claimed = list(topic.columns)
            for name in topic.groups:
                claimed.extend(FEATURE_GROUPS[name])
            for column in claimed:
                if column in seen:
                    raise ValueError(
                        f"feature column {column!r} is claimed by two sources (topics "
                        f"{seen[column]!r} and {topic.key!r}); one column, one source"
                    )
                seen[column] = topic.key
        return self

class ChartsConfig(_StrictConfigModel):
    """Spec 28's charts (``--mode charts``): how a figure is drawn and written.

    "Visualization is not optional", so this section is where the *presentation* lives - the
    figure size, the resolution, the file format, the colormaps and how many features the
    importance chart shows. It never decides *what* is drawn: the chart families come from the
    spec and the data, and this section only says how they look. ``ChartPolicy.from_config``
    reads these fields; the dataclass carries the same defaults, so a hand-built policy and the
    shipped one agree.
    """

    #: The master-spec section these charts come from (printed beside every figure's title).
    spec: str = "28"
    #: Decimals for a printed value on a chart.
    digits: int = Field(default=2, ge=0, le=12)
    #: Raster resolution of a written figure.
    dpi: int = Field(default=120, ge=30, le=600)
    #: Figure width/height in inches (wide charts) and the taller height for the bars chart.
    width: float = Field(default=14.0, gt=0)
    height: float = Field(default=7.0, gt=0)
    tall_height: float = Field(default=9.0, gt=0)
    #: File extension of a written figure (``png``/``svg``/``pdf``).
    format: str = Field(default="png", min_length=1)
    #: Colormaps: a diverging one for a P&L (positive/negative) and a sequential one otherwise.
    colormap: str = Field(default="RdYlGn", min_length=1)
    sequential_colormap: str = Field(default="viridis", min_length=1)
    #: How many features the importance bars show (the most important ones).
    top_features: int = Field(default=20, ge=1)
    #: The most points any single line is drawn from (a dense series is decimated to fit).
    max_points: int = Field(default=30_000, ge=2)
    #: The most event markers drawn on one price chart (the newest are kept).
    max_markers: int = Field(default=60, ge=1)




class AccumulationConfig(_StrictConfigModel):
    """Spec 29's continuous data-accumulation report (``--mode accumulation``).

    Phase 25 asks for a weekly summary of data captured, coverage, gaps, sample sizes and
    readiness for retraining. This section holds the *presentation and threshold* knobs; the
    four spec 4.3 gate values stay in ``data_gates`` so the report and ``--mode train`` read
    one ladder, not two. ``AccumulationPolicy.from_config`` reads these fields; the dataclass
    carries the same defaults, so a hand-built policy and the shipped one agree.
    """

    #: The master-spec section this report comes from (printed in every artefact).
    spec: str = "29"
    #: How many weeks back the captured-day series is bucketed (the report's own window).
    window_weeks: int = Field(default=4, ge=1)
    #: A capture gap longer than this many calendar days is called out as "capture may have
    #: stopped" - the one failure a weekly report exists to notice.
    stale_days: int = Field(default=10, ge=1)
    #: A day counts as *complete* at or above this coverage ratio (spec 4.3's ledger field).
    healthy_coverage_ratio: float = Field(default=0.9, gt=0, le=1)
    #: The sample-size floors a retrain is worth considering against.
    retrain_min_feature_rows: int = Field(default=5000, ge=1)
    retrain_min_label_rows: int = Field(default=5000, ge=1)
    #: A model older than this many days, on enough data, is flagged as due for a retrain.
    retrain_max_model_age_days: int = Field(default=30, ge=1)


class ModelConfig(_StrictConfigModel):
    """Root model of ``config/model.yaml``."""

    random_seed: int = Field(default=42, ge=0)
    horizons_minutes: list[int]
    algorithm: Literal["lightgbm", "xgboost", "catboost", "logistic_regression"]
    fallback_algorithm: (
        Literal["lightgbm", "xgboost", "catboost", "logistic_regression"] | None
    ) = None
    data_gates: DataGatesConfig = Field(default_factory=DataGatesConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    importance: ImportanceConfig = Field(default_factory=ImportanceConfig)
    charts: ChartsConfig = Field(default_factory=ChartsConfig)
    accumulation: AccumulationConfig = Field(default_factory=AccumulationConfig)

    @field_validator("horizons_minutes")
    @classmethod
    def _horizons(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("horizons_minutes must not be empty")
        if any(horizon <= 0 for horizon in value):
            raise ValueError("every horizon must be a positive number of minutes")
        if len(set(value)) != len(value):
            raise ValueError(f"horizons_minutes must be unique, got {value}")
        return sorted(value)

    @model_validator(mode="after")
    def _embargo_vs_horizon(self) -> ModelConfig:
        longest = max(self.horizons_minutes)
        if self.walk_forward.embargo_minutes < longest:
            raise ValueError(
                f"walk_forward.embargo_minutes ({self.walk_forward.embargo_minutes}) must be "
                f">= the longest label horizon ({longest} minutes) to prevent leakage"
            )
        return self

    @model_validator(mode="after")
    def _signal_directions_are_label_classes(self) -> ModelConfig:
        """A signal that names a class the labels never produce could never fire."""
        known = {str(name) for name in self.labels.classes}
        unknown = sorted(
            {str(name) for name in self.backtest.signal.directions.values()} - known
        )
        if unknown:
            raise ValueError(
                f"backtest.signal.directions names class(es) {unknown}, which "
                f"labels.classes does not define ({sorted(known)}); every signal would be "
                "dead on arrival"
            )
        return self

    @model_validator(mode="after")
    def _decision_directions_are_label_classes(self) -> ModelConfig:
        """A decision that names a class the labels never produce could never fire."""
        known = {str(name) for name in self.labels.classes}
        unknown = sorted({str(name) for name in self.decision.directions.values()} - known)
        if unknown:
            raise ValueError(
                f"decision.directions names class(es) {unknown}, which labels.classes does "
                f"not define ({sorted(known)}); every decision would be dead on arrival"
            )
        return self


# ---------------------------------------------------------------------------
# calendar.yaml
# ---------------------------------------------------------------------------
class SessionWindowConfig(_StrictConfigModel):
    """A session window in IST."""

    start: time
    end: time

    @model_validator(mode="after")
    def _ordered(self) -> SessionWindowConfig:
        if self.start >= self.end:
            raise ValueError(f"session start ({self.start}) must precede end ({self.end})")
        return self


class SpecialSessionConfig(_StrictConfigModel):
    """An ad-hoc session (e.g. muhurat trading) that overrides the regular one."""

    date: date
    start: time
    end: time
    name: str = "special session"

    @model_validator(mode="after")
    def _ordered(self) -> SpecialSessionConfig:
        if self.start >= self.end:
            raise ValueError(f"special session {self.date} start must precede end")
        return self


class SessionsConfig(_StrictConfigModel):
    """Regular / half-day sessions plus explicit exception dates."""

    regular: SessionWindowConfig
    half_day: SessionWindowConfig
    half_day_dates: list[date] = Field(default_factory=list)
    special_sessions: list[SpecialSessionConfig] = Field(default_factory=list)


class WeeklyExpiryConfig(_StrictConfigModel):
    """Weekly expiry weekday and holiday-roll policy.

    The default mirrors the *shipped* ``config/calendar.yaml`` (Tuesday, the
    weekday every NIFTY expiry in ``data/raw/instruments/NFO_2026-09-30.json``
    falls on). It is only a fallback for configurations that omit the block;
    the weekday itself is always configuration, never a code assumption.
    """

    weekday: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ] = "Tuesday"
    holiday_roll: Literal["previous_trading_day", "next_trading_day", "none"] = (
        "previous_trading_day"
    )


class MonthlyExpiryConfig(_StrictConfigModel):
    """Monthly expiry rule."""

    rule: Literal["last_weekly_expiry_of_month"] = "last_weekly_expiry_of_month"


class HolidayEntry(_StrictConfigModel):
    """A single market holiday. ``verified`` flags provisional dates."""

    date: date
    name: str = Field(min_length=1)
    verified: bool = False


class TimeOfDayBucketConfig(_StrictConfigModel):
    """A named intraday bucket ``[start, end)``."""

    name: str = Field(min_length=1)
    start: time
    end: time

    @model_validator(mode="after")
    def _ordered(self) -> TimeOfDayBucketConfig:
        if self.start >= self.end:
            raise ValueError(f"bucket {self.name}: start must precede end")
        return self


class DaysToExpiryConfig(_StrictConfigModel):
    """Which DTE definition is in force (must be documented - it matters)."""

    definition: Literal["calendar_days", "trading_days", "minutes_to_close"] = "minutes_to_close"
    max_minutes: int = Field(default=10080, gt=0)


class StartOfCycleConfig(_StrictConfigModel):
    """START_OF_CYCLE boundary definition (RQ1)."""

    days_after_previous_expiry: int = Field(default=1, ge=1)


class ExpiryCycleConfig(_StrictConfigModel):
    """Expiry-cycle phase boundaries (RQ1)."""

    start_of_cycle: StartOfCycleConfig = Field(default_factory=StartOfCycleConfig)


class CalendarConfig(_StrictConfigModel):
    """Root model of ``config/calendar.yaml``."""

    timezone: str = DEFAULT_TIMEZONE
    sessions: SessionsConfig
    weekly_expiry: WeeklyExpiryConfig = Field(default_factory=WeeklyExpiryConfig)
    monthly_expiry: MonthlyExpiryConfig = Field(default_factory=MonthlyExpiryConfig)
    #: False until the holiday list has been reconciled with the NSE circular.
    holidays_verified: bool = False
    holidays: list[HolidayEntry] = Field(default_factory=list)
    days_to_expiry: DaysToExpiryConfig = Field(default_factory=DaysToExpiryConfig)
    expiry_cycle: ExpiryCycleConfig = Field(default_factory=ExpiryCycleConfig)
    time_of_day_buckets: list[TimeOfDayBucketConfig]

    @field_validator("timezone")
    @classmethod
    def _tz(cls, value: str) -> str:
        return _validate_timezone(value)

    @model_validator(mode="after")
    def _consistent(self) -> CalendarConfig:
        seen: dict[date, str] = {}
        for entry in self.holidays:
            if entry.date in seen:
                raise ValueError(
                    f"duplicate holiday {entry.date.isoformat()} "
                    f"({seen[entry.date]!r} and {entry.name!r})"
                )
            seen[entry.date] = entry.name

        if not self.time_of_day_buckets:
            raise ValueError("time_of_day_buckets must not be empty")
        names = [bucket.name for bucket in self.time_of_day_buckets]
        if len(names) != len(set(names)):
            raise ValueError(f"time_of_day_buckets names must be unique, got {names}")

        ordered = sorted(self.time_of_day_buckets, key=lambda bucket: bucket.start)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start < previous.end:
                raise ValueError(
                    f"time_of_day_buckets must not overlap: {previous.name} ends "
                    f"{previous.end} but {current.name} starts {current.start}"
                )
        return self

    @property
    def holiday_dates(self) -> set[date]:
        """Set of holiday dates for fast membership tests."""
        return {entry.date for entry in self.holidays}

    @property
    def unverified_holidays(self) -> list[str]:
        """Holiday names whose dates still need reconciling with the NSE circular."""
        return [entry.name for entry in self.holidays if not entry.verified]


# ---------------------------------------------------------------------------
# The resolved configuration
# ---------------------------------------------------------------------------
class AnjaneyaConfig(BaseModel):
    """The fully resolved, validated ANJANEYA configuration.

    Instances are created by :func:`load_config`. The object is intentionally
    read-only-ish: applications receive it, they do not mutate it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: SettingsConfig
    data_sources: DataSourcesConfig
    option_universe: OptionUniverseConfig
    indicators: IndicatorsConfig
    model: ModelConfig
    calendar: CalendarConfig
    #: Directory the config was loaded from.
    config_dir: Path
    #: Repository root (parent of ``config_dir``) - relative paths resolve here.
    base_dir: Path
    #: SHA-256 of the raw config files, recorded in every run manifest.
    config_hash: str

    # -- path helpers -------------------------------------------------------
    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a possibly-relative configured path against the repo root."""
        path = Path(value)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def data_root(self) -> Path:
        """Absolute data root directory."""
        return self.resolve_path(self.settings.data.root)

    @property
    def db_path(self) -> Path:
        """Absolute DuckDB file path."""
        return self.resolve_path(self.settings.data.db_path)

    def data_dir(self, kind: str) -> Path:
        """Absolute path of a configured data sub-directory.

        ``kind`` is a field of :class:`DataDirsConfig`, e.g. ``"candles"``.
        """
        dirs = self.settings.data.dirs
        if not hasattr(dirs, kind):
            raise ConfigError(
                f"Unknown data directory {kind!r}; expected one of {sorted(type(dirs).model_fields)}"
            )
        return self.data_root / getattr(dirs, kind)

    @property
    def log_dir(self) -> Path:
        """Absolute log directory."""
        return self.data_dir("logs")

    @property
    def runs_dir(self) -> Path:
        """Absolute run-artifact directory."""
        return self.data_dir("runs")

    @property
    def token_store_path(self) -> Path:
        """Absolute path of the Zerodha access-token cache."""
        return self.resolve_path(self.data_sources.zerodha.token_store.path)

    @property
    def timezone_name(self) -> str:
        """Name of the research timezone (must match settings and calendar)."""
        return self.settings.time.timezone

    @property
    def instrument_cache_dir(self) -> Path:
        """Absolute directory holding the cached Kite instrument dumps.

        ``raw`` is a configured data directory, so a relative
        ``capture.instruments.cache_dir`` hangs off it.
        """
        return self.data_dir("raw") / self.data_sources.zerodha.capture.instruments.cache_dir

    @property
    def heartbeat_dir(self) -> Path:
        """Absolute directory holding the capture heartbeat file."""
        return self.data_dir("runs") / "heartbeat"

    def heartbeat_path(self) -> Path:
        """Absolute path of the capture heartbeat JSON file."""
        return self.heartbeat_dir / self.data_sources.zerodha.capture.heartbeat.filename

    @property
    def raw_dir(self) -> Path:
        """Absolute root of the raw data tree (Parquet mirrors live here)."""
        return self.data_dir("raw")

    def parquet_dir(self, stream: str) -> Path:
        """Absolute Parquet mirror directory for one raw stream.

        The layout reuses the directories Phase 1 reserved under ``data/raw``:
        ``data/raw/ticks/<YYYY-MM-DD>/part-*.parquet`` (compacted into
        ``<YYYY-MM-DD>.parquet`` at the end of a session) and likewise
        ``option_chain_snapshots``.
        """
        return self.raw_dir / stream

    def ensure_directories(self) -> list[Path]:
        """Create every configured data directory (explicit I/O, not import-time).

        Returns the list of directories that were created/exist.
        """
        created: list[Path] = []
        for kind in sorted(type(self.settings.data.dirs).model_fields):
            path = self.data_dir(kind)
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        for path in (self.db_path.parent, self.token_store_path.parent, self.heartbeat_dir):
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        return created

    def summary(self) -> dict[str, Any]:
        """Compact, log-safe summary (never contains secrets)."""
        return {
            "project": self.settings.project.name,
            "version": self.settings.project.version,
            "schema_version": self.settings.project.schema_version,
            "timezone": self.timezone_name,
            "config_dir": str(self.config_dir),
            "config_hash": self.config_hash,
            "data_root": str(self.data_root),
            "db_path": str(self.db_path),
            "timeframes": list(self.settings.market.timeframes),
            "horizons_minutes": list(self.model.horizons_minutes),
            "algorithm": self.model.algorithm,
            "expiry_weekday": self.calendar.weekly_expiry.weekday,
            "holidays_verified": self.calendar.holidays_verified,
            "holidays_total": len(self.calendar.holidays),
            "unverified_holidays": len(self.calendar.unverified_holidays),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_config(
    config_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    base_dir: str | Path | None = None,
) -> AnjaneyaConfig:
    """Load and validate every configuration file.

    Parameters
    ----------
    config_dir:
        Directory holding the six YAML files. Defaults to ``<repo>/config``.
    environ:
        Environment mapping used to resolve ``${VAR}`` placeholders. Defaults
        to :data:`os.environ`. Tests pass an explicit dict to stay hermetic.
    base_dir:
        Directory that relative paths resolve against. Defaults to the parent
        of ``config_dir`` (the repository root).

    Raises
    ------
    ConfigError
        If a file is missing/unreadable or a placeholder cannot be resolved.
    pydantic.ValidationError
        If the values are structurally valid YAML but semantically invalid
        (unknown key, out-of-range number, overlapping buckets, ...).
    """
    from pydantic import ValidationError

    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent / "config"
    config_path = Path(config_dir).resolve()
    if not config_path.is_dir():
        raise ConfigError(f"Configuration directory not found: {config_path}")

    resolved_base = Path(base_dir).resolve() if base_dir else config_path.parent

    raw: dict[str, dict[str, Any]] = {
        name: load_yaml_file(config_path / name) for name in CONFIG_FILES
    }
    substituted = {name: resolve_env_placeholders(body, environ) for name, body in raw.items()}

    try:
        return AnjaneyaConfig(
            settings=SettingsConfig.model_validate(substituted["settings.yaml"]),
            data_sources=DataSourcesConfig.model_validate(substituted["data_sources.yaml"]),
            option_universe=OptionUniverseConfig.model_validate(substituted["option_universe.yaml"]),
            indicators=IndicatorsConfig.model_validate(substituted["indicators.yaml"]),
            model=ModelConfig.model_validate(substituted["model.yaml"]),
            calendar=CalendarConfig.model_validate(substituted["calendar.yaml"]),
            config_dir=config_path,
            base_dir=resolved_base,
            config_hash=config_file_hash(config_path),
        )
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid ANJANEYA configuration in {config_path}:\n{exc}\n"
            "Fix the YAML values - ANJANEYA refuses to run with silently ignored settings."
        ) from exc
