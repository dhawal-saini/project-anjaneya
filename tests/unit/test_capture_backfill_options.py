"""Tests for ``--mode backfill-options`` in the capture app.

Kite serves active options but never expired ones, so this mode bootstraps candles
for the contracts listed *today* and writes them to the keyed ``candles`` table.
Two pieces carry all the risk and are pinned here: the ``--years`` window override
and the mapping of a fetched REST bar into an option candle that carries its full
contract identity.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

from apps.anjaneeya_capture import run as capture_run
from apps.anjaneeya_capture.instruments import Instrument
from core.schemas import IndexHistoryRow, OptionType


class _IndexHistory:
    def __init__(self, days: int) -> None:
        self.backfill_days = days


def test_history_days_defaults_to_the_config() -> None:
    assert capture_run._history_days(SimpleNamespace(years=None), _IndexHistory(730)) == 730


def test_history_days_honours_the_years_override() -> None:
    assert capture_run._history_days(SimpleNamespace(years=1), _IndexHistory(400)) == 365
    assert capture_run._history_days(SimpleNamespace(years=2), _IndexHistory(400)) == 730


def test_candle_from_index_row_carries_the_option_identity() -> None:
    moment = datetime(2026, 10, 1, 9, 16, tzinfo=UTC)
    row = IndexHistoryRow(
        timestamp=moment,
        tradingsymbol="NIFTY26O0622400CE",
        timeframe="1m",
        open=100.0,
        high=120.0,
        low=90.0,
        close=110.0,
        volume=1500,
        oi=None,
        fetch_timestamp=moment,
        run_id="run-test",
    )
    instrument = Instrument(
        instrument_token=10415874,
        exchange="NFO",
        tradingsymbol="NIFTY26O0622400CE",
        instrument_type="CE",
        expiry=date(2026, 10, 6),
        strike=22400.0,
    )

    candle = capture_run._candle_from_index_row(row, instrument)

    assert candle.tradingsymbol == "NIFTY26O0622400CE"
    assert candle.instrument_token == 10415874
    assert candle.option_type == OptionType.CE
    assert candle.strike == 22400.0
    assert candle.expiry == date(2026, 10, 6)
    assert candle.close == 110.0


def test_backfill_options_is_a_registered_mode() -> None:
    from core.app_bootstrap import APP_CAPTURE, DEFAULT_MODES

    assert "backfill-options" in DEFAULT_MODES[APP_CAPTURE]
    assert "backfill-options" in capture_run.MODE_PLAN
