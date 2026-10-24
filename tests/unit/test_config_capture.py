"""Tests for the Phase 2 capture configuration blocks.

The capture app reads a lot of policy from YAML (session guard, heartbeat,
gap thresholds, batching, historical windows). These tests pin the *semantics*
that matter operationally: a broken value must fail at startup, never at 09:15.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config_loader import (
    CaptureConfig,
    CaptureStorageConfig,
    HeartbeatConfig,
    HistoricalFetchConfig,
    InstrumentCacheConfig,
    WebsocketConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Shipped configuration
# ---------------------------------------------------------------------------
def test_shipped_capture_config_is_coherent(config) -> None:
    capture = config.data_sources.zerodha.capture
    assert capture.heartbeat.stale_after_seconds > capture.heartbeat.interval_seconds
    assert capture.storage.max_buffer_rows >= capture.storage.flush_rows
    assert capture.instruments.exchanges == ["NFO", "NSE"]
    assert capture.historical.window_days("minute") == 30
    assert capture.historical.window_days("day") == 1800
    # An unknown interval must fall back to the most conservative ceiling.
    assert capture.historical.window_days("nope") == 30


def test_capture_paths_hang_off_the_data_root(config) -> None:
    assert config.instrument_cache_dir == config.data_dir("raw") / "instruments"
    assert config.heartbeat_path() == config.heartbeat_dir / "capture.json"
    assert config.raw_dir == config.data_dir("raw")
    assert config.parquet_dir("ticks") == config.data_dir("raw") / "ticks"


def test_index_history_intervals_are_kite_intervals(config) -> None:
    index_history = config.option_universe.index_history
    assert index_history.backfill_interval == "day"
    assert index_history.topup_interval == "minute"


# ---------------------------------------------------------------------------
# Validation that prevents silent misbehaviour
# ---------------------------------------------------------------------------
def test_heartbeat_must_be_stale_later_than_it_is_written() -> None:
    with pytest.raises(ValueError, match="stale_after_seconds must exceed"):
        HeartbeatConfig(interval_seconds=30, stale_after_seconds=30)


def test_buffer_must_hold_a_full_flush() -> None:
    with pytest.raises(ValueError, match="max_buffer_rows"):
        CaptureStorageConfig(flush_rows=1000, max_buffer_rows=10)


def test_exchanges_are_normalised_and_deduplicated() -> None:
    assert InstrumentCacheConfig(exchanges=[" nfo ", "NSE", "nfo"]).exchanges == ["NFO", "NSE"]
    with pytest.raises(ValueError, match="at least one exchange"):
        InstrumentCacheConfig(exchanges=["", "  "])


def test_unknown_historical_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a Kite interval"):
        HistoricalFetchConfig(default_interval="second")


def test_unknown_interval_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown interval"):
        HistoricalFetchConfig(max_days_per_request={"minute": 30, "tick": 5})


def test_non_positive_window_days_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        HistoricalFetchConfig(max_days_per_request={"minute": 0})


def test_default_interval_must_have_a_window() -> None:
    with pytest.raises(ValueError, match="no entry for the default interval"):
        HistoricalFetchConfig(default_interval="3minute", max_days_per_request={"minute": 30})


def test_reconnect_max_must_be_at_least_the_initial_delay() -> None:
    with pytest.raises(ValueError, match="reconnect_max_seconds"):
        WebsocketConfig(reconnect_initial_seconds=10, reconnect_max_seconds=5)


def test_out_of_session_policy_only_accepts_known_values() -> None:
    assert CaptureConfig(out_of_session_ticks="record").out_of_session_ticks == "record"
    assert CaptureConfig(out_of_session_ticks="drop").out_of_session_ticks == "drop"
    with pytest.raises(ValueError):
        CaptureConfig(out_of_session_ticks="ignore")


def test_unknown_capture_key_is_a_hard_error(tmp_path: Path, config_dir: Path) -> None:
    """A typo in a capture setting must fail loudly (extra="forbid")."""
    import shutil

    import yaml

    scratch = tmp_path / "config"
    shutil.copytree(config_dir, scratch)
    path = scratch / "data_sources.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["zerodha"]["capture"]["flush_second"] = 5  # typo of `storage.flush_seconds`
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    from core.config_loader import ConfigError

    with pytest.raises(ConfigError):
        load_config(scratch, base_dir=tmp_path)
