"""Tests for :mod:`core.constants` - the structural contract of the project."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.constants import (
    CANDLE_COLUMNS,
    DEFAULT_TIMEZONE,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    PROVENANCE_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    TICK_COLUMNS,
    TIMEFRAME_MINUTES,
    TRADE_COLUMNS,
    get_timezone,
    timeframe_to_minutes,
)

ALL_CONTRACTS = {
    "TICK_COLUMNS": TICK_COLUMNS,
    "OPTION_CHAIN_COLUMNS": OPTION_CHAIN_COLUMNS,
    "CANDLE_COLUMNS": CANDLE_COLUMNS,
    "FEATURE_COLUMNS": FEATURE_COLUMNS,
    "LABEL_COLUMNS": LABEL_COLUMNS,
    "TRADE_COLUMNS": TRADE_COLUMNS,
}

#: Contracts that describe a timestamped market/dataset row. The trade ledger is
#: keyed by ``trade_id`` and is not a raw observation, so it is excluded.
TIMESTAMPED_CONTRACTS = {k: v for k, v in ALL_CONTRACTS.items() if k != "TRADE_COLUMNS"}


def test_timezone_resolves_on_this_platform() -> None:
    """Asia/Kolkata must resolve - on Windows that requires the tzdata package."""
    tz = get_timezone(DEFAULT_TIMEZONE)
    # A fixed historical instant: IST is UTC+05:30 with no DST.
    offset = datetime(2026, 1, 1, tzinfo=tz).utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 5.5 * 3600


def test_get_timezone_reports_unknown_zone_helpfully() -> None:
    """An unknown zone must raise a message that names the timezone."""
    with pytest.raises(RuntimeError, match="Mars/Olympus_Mons"):
        get_timezone("Mars/Olympus_Mons")


def test_timeframe_lookup_and_error() -> None:
    assert timeframe_to_minutes("5m") == 5
    assert set(TIMEFRAME_MINUTES) == set(SUPPORTED_TIMEFRAMES)
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        timeframe_to_minutes("7m")


@pytest.mark.parametrize("name,columns", sorted(TIMESTAMPED_CONTRACTS.items()))
def test_column_contracts_are_unique_and_start_with_timestamp(
    name: str, columns: tuple[str, ...]
) -> None:
    """Each market-data contract must be non-empty, unique and start with ``timestamp``."""
    assert columns, f"{name} is empty"
    assert len(columns) == len(set(columns)), f"{name} contains duplicate column names"
    assert columns[0] == "timestamp", f"{name} must start with the timestamp column"


@pytest.mark.parametrize("name,columns", sorted(TIMESTAMPED_CONTRACTS.items()))
def test_every_contract_carries_provenance(name: str, columns: tuple[str, ...]) -> None:
    """Every stored observation must be traceable (spec 6.5)."""
    assert set(PROVENANCE_COLUMNS) <= set(columns), f"{name} is missing provenance columns"


def test_trade_ledger_is_keyed_and_costs_are_tracked() -> None:
    """The ledger is not a raw observation, but it must stay auditable."""
    assert TRADE_COLUMNS[0] == "trade_id"
    assert len(TRADE_COLUMNS) == len(set(TRADE_COLUMNS))
    assert "run_id" in TRADE_COLUMNS
    # A P&L number without its cost breakdown is not evidence (spec 23).
    assert {"gross_pnl", "costs", "net_pnl"} <= set(TRADE_COLUMNS)
    # Losers and winners are both recorded, with excursion detail.
    assert {"mfe", "mae", "exit_reason"} <= set(TRADE_COLUMNS)
