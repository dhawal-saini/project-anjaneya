-- ===========================================================================
-- ANJANEYA - DuckDB schema (schema_version 1)
-- ===========================================================================
-- Design rules
--   * Raw tables (ticks, snapshots, index_history) are APPEND-ONLY. They are
--     the immutable record from which every derived table can be rebuilt.
--   * All timestamps are TIMESTAMPTZ; the session timezone is set to
--     Asia/Kolkata by the bootstrap so comparisons are exchange-local.
--   * Every table carries provenance: source, fetch_timestamp, quality_flag,
--     run_id - so any number can be traced back to the run that produced it.
--   * Running this file twice must be a no-op (IF NOT EXISTS everywhere).
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Schema bookkeeping
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER     NOT NULL,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    description  VARCHAR
);

-- ---------------------------------------------------------------------------
-- Instrument master (populated from Kite's instruments dump)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS instruments (
    instrument_token BIGINT      NOT NULL,
    exchange         VARCHAR     NOT NULL,
    tradingsymbol    VARCHAR     NOT NULL,
    name             VARCHAR,
    segment          VARCHAR,
    instrument_type  VARCHAR,
    expiry           DATE,
    strike           DOUBLE,
    lot_size         INTEGER,
    tick_size        DOUBLE,
    fetched_at       TIMESTAMPTZ NOT NULL,
    run_id           VARCHAR     NOT NULL,
    PRIMARY KEY (instrument_token)
);

-- ---------------------------------------------------------------------------
-- Raw ticks - append-only, never updated (spec 4.6)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticks (
    timestamp        TIMESTAMPTZ NOT NULL,
    instrument_token BIGINT      NOT NULL,
    tradingsymbol    VARCHAR     NOT NULL,
    expiry           DATE,
    strike           DOUBLE,
    option_type      VARCHAR,
    last_price       DOUBLE      NOT NULL,
    volume_traded    BIGINT,
    oi               BIGINT,
    bid              DOUBLE,
    ask              DOUBLE,
    bid_qty          BIGINT,
    ask_qty          BIGINT,
    source           VARCHAR     NOT NULL,
    fetch_timestamp  TIMESTAMPTZ NOT NULL,
    quality_flag     VARCHAR     NOT NULL,
    run_id           VARCHAR     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_token_ts ON ticks (instrument_token, timestamp);

-- ---------------------------------------------------------------------------
-- Option chain snapshots (periodic OI / IV polling)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    timestamp        TIMESTAMPTZ NOT NULL,
    instrument_token BIGINT      NOT NULL,
    tradingsymbol    VARCHAR     NOT NULL,
    expiry           DATE        NOT NULL,
    strike           DOUBLE      NOT NULL,
    option_type      VARCHAR     NOT NULL,
    last_price       DOUBLE      NOT NULL,
    bid              DOUBLE,
    ask              DOUBLE,
    volume           BIGINT,
    oi               BIGINT,
    iv               DOUBLE,
    source           VARCHAR     NOT NULL,
    fetch_timestamp  TIMESTAMPTZ NOT NULL,
    quality_flag     VARCHAR     NOT NULL,
    run_id           VARCHAR     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_contract_ts
    ON option_chain_snapshots (expiry, strike, option_type, timestamp);

-- ---------------------------------------------------------------------------
-- NIFTY index history (REST backfill + daily top-up)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS index_history (
    timestamp        TIMESTAMPTZ NOT NULL,
    tradingsymbol    VARCHAR     NOT NULL,
    timeframe        VARCHAR     NOT NULL DEFAULT '1m',
    open             DOUBLE      NOT NULL,
    high             DOUBLE      NOT NULL,
    low              DOUBLE      NOT NULL,
    close            DOUBLE      NOT NULL,
    volume           BIGINT,
    oi               BIGINT,
    source           VARCHAR     NOT NULL,
    fetch_timestamp  TIMESTAMPTZ NOT NULL,
    quality_flag     VARCHAR     NOT NULL,
    run_id           VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp, tradingsymbol, timeframe)
);

-- ---------------------------------------------------------------------------
-- Candles derived from ticks (spec 6.3). The primary key makes candle
-- rebuilding idempotent: re-running the engine for a day replaces, not doubles.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS candles (
    timestamp         TIMESTAMPTZ NOT NULL,
    timeframe         VARCHAR     NOT NULL,
    instrument_token  BIGINT,
    tradingsymbol     VARCHAR     NOT NULL,
    expiry            DATE,
    strike            DOUBLE,
    option_type       VARCHAR,
    open              DOUBLE      NOT NULL,
    high              DOUBLE      NOT NULL,
    low               DOUBLE      NOT NULL,
    close             DOUBLE      NOT NULL,
    volume            BIGINT,
    oi                BIGINT,
    iv                DOUBLE,
    delta             DOUBLE,
    gamma             DOUBLE,
    theta             DOUBLE,
    vega              DOUBLE,
    days_to_expiry    DOUBLE,
    distance_from_atm DOUBLE,
    source            VARCHAR     NOT NULL,
    fetch_timestamp   TIMESTAMPTZ,
    quality_flag      VARCHAR     NOT NULL,
    run_id            VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp, timeframe, tradingsymbol)
);


-- ---------------------------------------------------------------------------
-- Unified feature rows (spec 18). One row per (timestamp, contract).
-- Labels are NEVER stored here - see the `labels` table (spec 20/30).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    timestamp                              TIMESTAMPTZ NOT NULL,
    underlying                             VARCHAR     NOT NULL,
    tradingsymbol                          VARCHAR     NOT NULL,
    instrument_token                       BIGINT,
    expiry                                 DATE,
    strike                                 DOUBLE,
    option_type                            VARCHAR,
    timeframe                              VARCHAR,
    nifty_fast_score                       DOUBLE,
    nifty_medium_score                     DOUBLE,
    nifty_long_score                       DOUBLE,
    nifty_composite_trend_score            DOUBLE,
    nifty_trend_acceleration               DOUBLE,
    nifty_trend_persistence                DOUBLE,
    nifty_regime                           VARCHAR,
    option_trend_score                     DOUBLE,
    option_ma_alignment                    DOUBLE,
    option_ma_slope                        DOUBLE,
    option_normalized_slope                DOUBLE,
    option_ema20_ema200_spread             DOUBLE,
    option_spread_velocity                 DOUBLE,
    option_atr                             DOUBLE,
    option_momentum                        DOUBLE,
    option_trend_persistence               DOUBLE,
    option_insufficient_history            BOOLEAN,
    candle_body                            DOUBLE,
    candle_range                           DOUBLE,
    candle_upper_wick                      DOUBLE,
    candle_lower_wick                      DOUBLE,
    candle_body_ratio                      DOUBLE,
    candle_upper_wick_ratio                DOUBLE,
    candle_lower_wick_ratio                DOUBLE,
    candle_direction                       VARCHAR,
    candle_gap                             DOUBLE,
    candle_engulf_ratio                    DOUBLE,
    candle_inside_flag                     BOOLEAN,
    candle_outside_flag                    BOOLEAN,
    candle_body_ratio_change               DOUBLE,
    candle_wick_ratio_change               DOUBLE,
    candle_close_position_within_prev_range DOUBLE,
    level_today_open                       DOUBLE,
    level_today_high_running               DOUBLE,
    level_today_low_running                DOUBLE,
    level_previous_day_open                DOUBLE,
    level_previous_day_high                DOUBLE,
    level_previous_day_low                 DOUBLE,
    level_previous_day_close               DOUBLE,
    distance_to_today_high                 DOUBLE,
    distance_to_today_low                  DOUBLE,
    distance_to_previous_day_high          DOUBLE,
    distance_to_previous_day_low           DOUBLE,
    event_prev_day_high_state              VARCHAR,
    event_prev_day_low_state               VARCHAR,
    event_today_high_state                 VARCHAR,
    event_today_low_state                  VARCHAR,
    swing_high_1                           DOUBLE,
    swing_low_1                            DOUBLE,
    swing_label                            VARCHAR,
    market_structure_state                 VARCHAR,
    distance_from_swing_high               DOUBLE,
    distance_from_swing_low                DOUBLE,
    support_strength                       DOUBLE,
    resistance_strength                    DOUBLE,
    support_touch_count                    INTEGER,
    support_rejection_count                INTEGER,
    resistance_touch_count                 INTEGER,
    resistance_rejection_count             INTEGER,
    distance_from_atm                      DOUBLE,
    days_to_expiry                         DOUBLE,
    expiry_cycle_phase                     VARCHAR,
    time_of_day_bucket                     VARCHAR,
    day_of_week                            INTEGER,
    iv                                     DOUBLE,
    iv_change                              DOUBLE,
    delta                                  DOUBLE,
    gamma                                  DOUBLE,
    theta                                  DOUBLE,
    vega                                   DOUBLE,
    oi                                     BIGINT,
    oi_change                              BIGINT,
    volume                                 BIGINT,
    expected_option_move                   DOUBLE,
    actual_option_move                     DOUBLE,
    option_residual                        DOUBLE,
    source                                 VARCHAR     NOT NULL,
    fetch_timestamp                        TIMESTAMPTZ,
    quality_flag                           VARCHAR     NOT NULL,
    run_id                                 VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp, tradingsymbol)
);


-- ---------------------------------------------------------------------------
-- Labels (spec 20). Physically separate from `features`, one row per
-- (timestamp, contract, horizon). Sign conventions are documented in
-- core/schemas.py::LabelRow.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS labels (
    timestamp            TIMESTAMPTZ NOT NULL,
    tradingsymbol        VARCHAR     NOT NULL,
    instrument_token     BIGINT,
    expiry               DATE,
    strike               DOUBLE,
    option_type          VARCHAR,
    horizon_minutes      INTEGER     NOT NULL,
    future_return        DOUBLE,
    future_max_up_move   DOUBLE,
    future_max_down_move DOUBLE,
    future_trend_change  DOUBLE,
    label                VARCHAR,
    label_version        VARCHAR     NOT NULL,
    source               VARCHAR     NOT NULL,
    fetch_timestamp      TIMESTAMPTZ,
    quality_flag         VARCHAR     NOT NULL,
    run_id               VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp, tradingsymbol, horizon_minutes)
);

-- ---------------------------------------------------------------------------
-- Backtest trade ledger (spec 23). Every trade is recorded, including every
-- loser - a P&L number without its full ledger is not evidence.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    trade_id                    VARCHAR     NOT NULL,
    run_id                      VARCHAR     NOT NULL,
    entry_timestamp             TIMESTAMPTZ NOT NULL,
    exit_timestamp              TIMESTAMPTZ,
    tradingsymbol               VARCHAR     NOT NULL,
    expiry                      DATE,
    strike                      DOUBLE,
    option_type                 VARCHAR,
    entry_price                 DOUBLE      NOT NULL,
    exit_price                  DOUBLE,
    quantity                    INTEGER     NOT NULL,
    entry_reason                VARCHAR,
    exit_reason                 VARCHAR,
    nifty_state                 VARCHAR,
    option_state                VARCHAR,
    candle_state                VARCHAR,
    support_resistance_state    VARCHAR,
    ai_prediction               VARCHAR,
    ai_probability              DOUBLE,
    gross_pnl                   DOUBLE,
    costs                       DOUBLE,
    net_pnl                     DOUBLE,
    mfe                         DOUBLE,
    mae                         DOUBLE,
    holding_minutes             DOUBLE,
    time_of_day_bucket          VARCHAR,
    expiry_cycle_phase          VARCHAR,
    days_to_expiry              DOUBLE,
    PRIMARY KEY (trade_id)
);

-- ---------------------------------------------------------------------------
-- Data coverage ledger (spec 4.3 gates / spec 6.5 quality).
-- One row per (trading_date, symbol, timeframe) so gaps are auditable.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_coverage (
    trading_date      DATE        NOT NULL,
    tradingsymbol     VARCHAR     NOT NULL,
    timeframe         VARCHAR     NOT NULL,
    expected_rows     INTEGER,
    actual_rows       INTEGER,
    first_timestamp   TIMESTAMPTZ,
    last_timestamp    TIMESTAMPTZ,
    gap_count         INTEGER,
    largest_gap_seconds DOUBLE,
    coverage_ratio    DOUBLE,
    quality_flag      VARCHAR     NOT NULL,
    note              VARCHAR,
    run_id            VARCHAR     NOT NULL,
    PRIMARY KEY (trading_date, tradingsymbol, timeframe)
);

-- ---------------------------------------------------------------------------
-- Run manifests (spec 36) - one row per app execution.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_manifest (
    run_id            VARCHAR     NOT NULL,
    app_name          VARCHAR     NOT NULL,
    mode              VARCHAR,
    status            VARCHAR     NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ,
    config_hash       VARCHAR,
    config_dir        VARCHAR,
    git_commit        VARCHAR,
    python_version    VARCHAR,
    platform          VARCHAR,
    random_seed       INTEGER,
    package_versions  VARCHAR,
    data_coverage     VARCHAR,
    artifacts         VARCHAR,
    notes             VARCHAR,
    manifest_path     VARCHAR,
    PRIMARY KEY (run_id)
);
