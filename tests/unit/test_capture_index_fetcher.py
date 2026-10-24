"""Tests for the index-history fetcher: window planning and adaptive retry."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_capture.nifty_index_fetcher import (
    FetchTarget,
    FetchWindow,
    IndexHistoryFetcher,
    parse_candles,
    plan_windows,
    split_window,
)
from core.config_loader import load_config
from tests.support.capture import NIFTY_SPOT_TOKEN, FakeKite, RecordingLogger

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 29, 15, 35, tzinfo=IST)
RUN_ID = "anjaneeya_capture_20260929-153500_abcdef12"
TARGET = FetchTarget(instrument_token=NIFTY_SPOT_TOKEN, tradingsymbol="NIFTY 50")


@pytest.fixture
def historical_config(isolated_paths):
    return load_config().data_sources.zerodha.capture.historical


def make_candles(start: datetime, *, minutes: int = 5, step_minutes: int = 1) -> list[dict]:
    """A short run of well-formed candles."""
    candles = []
    for index in range(minutes):
        moment = start + timedelta(minutes=index * step_minutes)
        base = 25000.0 + index
        candles.append(
            {
                "date": moment,
                "open": base,
                "high": base + 5,
                "low": base - 5,
                "close": base + 2,
                "volume": 0,
                "oi": 0,
            }
        )
    return candles


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------
def test_plan_windows_covers_the_range_contiguously() -> None:
    windows = plan_windows(NOW - timedelta(days=100), NOW, max_days=30)
    assert len(windows) == 4
    assert windows[0].start == NOW - timedelta(days=100)
    assert windows[-1].end == NOW
    for previous, current in zip(windows, windows[1:], strict=False):
        # No gap and no overlap (a shared boundary second is allowed).
        assert current.start - previous.end <= timedelta(seconds=1)
        assert previous.days <= 30


def test_plan_windows_handles_degenerate_ranges() -> None:
    assert plan_windows(NOW, NOW, max_days=30) == [FetchWindow(start=NOW, end=NOW)]
    assert plan_windows(NOW, NOW - timedelta(days=1), max_days=30) == []
    with pytest.raises(ValueError, match="max_days"):
        plan_windows(NOW, NOW, max_days=0)


def test_split_window_halves_and_refuses_single_days() -> None:
    window = FetchWindow(start=NOW - timedelta(days=10), end=NOW)
    split = split_window(window)
    assert split is not None
    left, right = split
    assert left.start == window.start
    assert right.end == window.end
    assert left.end < right.start
    assert split_window(FetchWindow(start=NOW, end=NOW + timedelta(hours=12))) is None


# ---------------------------------------------------------------------------
# Candle parsing
# ---------------------------------------------------------------------------
def test_parse_candles_labels_the_timeframe_and_localises_naive_timestamps() -> None:
    raw = make_candles(datetime(2026, 9, 29, 9, 15))
    rows, rejected = parse_candles(
        raw, interval="minute", tradingsymbol="NIFTY 50", run_id=RUN_ID, fetched_at=NOW
    )
    assert rejected == 0
    assert len(rows) == 5
    assert rows[0].timeframe == "1m"
    assert rows[0].timestamp.tzinfo is not None
    assert rows[0].timestamp == datetime(2026, 9, 29, 9, 15, tzinfo=IST)
    assert rows[0].fetch_timestamp >= rows[0].timestamp
    assert rows[0].source.value == "ZERODHA_REST"


def test_parse_candles_accepts_dates_and_iso_strings() -> None:
    raw = [
        {"date": date(2026, 9, 28), "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        {"date": "2026-09-27T09:15:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        {"date": "2026-09-26T03:45:00+00:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
    ]
    rows, rejected = parse_candles(
        raw, interval="day", tradingsymbol="NIFTY 50", run_id=RUN_ID, fetched_at=NOW
    )
    assert rejected == 0
    assert [row.timeframe for row in rows] == ["1d", "1d", "1d"]
    assert rows[0].timestamp.date() == date(2026, 9, 28)
    assert rows[2].timestamp == datetime(2026, 9, 26, 9, 15, tzinfo=IST)


@pytest.mark.parametrize(
    "candle",
    [
        {"date": None, "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        {"date": "not-a-date", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        {"date": datetime(2026, 9, 29, 9, 15), "open": None, "high": 2, "low": 0.5, "close": 1.5},
        {"date": datetime(2026, 9, 29, 9, 15), "open": 1, "high": 0.1, "low": 0.5, "close": 1.5},
        {"date": datetime(2026, 9, 29, 9, 15), "open": -1, "high": 2, "low": 0.5, "close": 1.5},
        "not-a-mapping",
    ],
)
def test_parse_candles_counts_bad_candles_instead_of_raising(candle) -> None:
    rows, rejected = parse_candles(
        [candle], interval="minute", tradingsymbol="NIFTY 50", run_id=RUN_ID, fetched_at=NOW
    )
    assert rows == []
    assert rejected == 1


def test_parse_candles_drops_rows_outside_the_planned_window() -> None:
    window = FetchWindow(start=NOW - timedelta(minutes=3), end=NOW)
    raw = make_candles(NOW - timedelta(minutes=5), minutes=10)
    rows, rejected = parse_candles(
        raw,
        interval="minute",
        tradingsymbol="NIFTY 50",
        run_id=RUN_ID,
        fetched_at=NOW,
        window=window,
    )
    assert 0 < len(rows) <= 4
    assert rejected == 10 - len(rows)


def test_parse_candles_rejects_an_unknown_interval() -> None:
    with pytest.raises(ValueError, match="Unsupported historical interval"):
        parse_candles(
            [], interval="tick", tradingsymbol="NIFTY 50", run_id=RUN_ID, fetched_at=NOW
        )

# ---------------------------------------------------------------------------
# The fetcher
# ---------------------------------------------------------------------------
def make_fetcher(historical_config, client, logger=None) -> IndexHistoryFetcher:
    return IndexHistoryFetcher(
        client,
        historical_config=historical_config,
        run_id=RUN_ID,
        logger=logger,
        now_fn=lambda: NOW,
    )


def test_backfill_uses_the_configured_window_size(historical_config) -> None:
    # 100 days of minute data at <= 30 days per request => 4 requests.
    client = FakeKite()
    fetcher = make_fetcher(historical_config, client)
    result = fetcher.backfill([TARGET], days=100, interval="minute")

    assert result.windows_planned == 4
    assert result.requests == 4
    assert result.splits == 0
    assert result.ok is True
    assert client.history_calls[0][3] == "minute"
    assert client.history_calls[0][0] == NIFTY_SPOT_TOKEN


def test_backfill_returns_rows_with_an_audit_trail(historical_config) -> None:
    start = NOW - timedelta(days=1)
    client = FakeKite(history={(NIFTY_SPOT_TOKEN, "minute"): make_candles(start, minutes=3)})
    fetcher = make_fetcher(historical_config, client)
    result = fetcher.backfill([TARGET], days=2, interval="minute")

    assert len(result.rows) == 3
    assert result.symbols == ["NIFTY 50"]
    assert result.intervals == ["minute"]
    assert result.describe()["rows"] == 3


def test_top_up_refetches_only_the_recent_window(historical_config) -> None:
    client = FakeKite()
    fetcher = make_fetcher(historical_config, client)
    result = fetcher.top_up([TARGET], days=10, interval="minute")
    # 10 days at <= 30 days per request is a single request.
    assert result.windows_planned == 1
    first, last = client.history_calls[0][1], client.history_calls[0][2]
    assert (last - first) <= timedelta(days=10)


def test_a_range_error_is_survived_by_halving_the_window(historical_config) -> None:
    """Kite answers a too-wide range with an error; we halve and retry."""

    class FlakyKite(FakeKite):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.too_wide = 0

        def historical_data(self, token, from_date, to_date, interval, **kwargs):  # type: ignore[override]
            self.history_calls.append((token, from_date, to_date, interval, False, False))
            if (to_date - from_date) > timedelta(days=20):
                self.too_wide += 1
                raise RuntimeError("invalid range: too many days requested")
            return make_candles(from_date, minutes=2)

    client = FlakyKite()
    logger = RecordingLogger()
    fetcher = make_fetcher(historical_config, client, logger)

    result = fetcher.backfill([TARGET], days=100, interval="minute")

    assert result.ok is True
    assert result.splits > 0
    assert logger.contains("halving the window", "WARNING")
    assert len(result.rows) > 0


def test_a_permanent_failure_is_recorded_without_losing_the_other_windows(historical_config) -> None:
    class BrokenKite(FakeKite):
        def historical_data(self, token, from_date, to_date, interval, **kwargs):  # type: ignore[override]
            self.history_calls.append((token, from_date, to_date, interval, False, False))
            raise RuntimeError("upstream exploded")

    client = BrokenKite()
    logger = RecordingLogger()
    fetcher = make_fetcher(historical_config, client, logger)
    result = fetcher.backfill([TARGET], days=100, interval="minute")

    assert result.ok is False
    assert result.rows == []
    assert len(result.failures) > 0
    assert result.requests > result.windows_planned  # retries happened
    assert logger.contains("failed permanently", "ERROR")


def test_adaptive_split_can_be_disabled(historical_config) -> None:
    config = historical_config.model_copy(update={"adaptive_split_on_error": False})

    class BrokenKite(FakeKite):
        def historical_data(self, token, from_date, to_date, interval, **kwargs):  # type: ignore[override]
            raise RuntimeError("nope")

    fetcher = IndexHistoryFetcher(
        BrokenKite(), historical_config=config, run_id=RUN_ID, now_fn=lambda: NOW
    )
    result = fetcher.backfill([TARGET], days=100, interval="minute")
    assert result.splits == 0
    # One failure per planned window, no retries.
    assert len(result.failures) == result.windows_planned == 4


def test_an_unconfigured_interval_is_rejected(historical_config) -> None:
    config = historical_config.model_copy(update={"max_days_per_request": {"minute": 30}})
    fetcher = IndexHistoryFetcher(
        FakeKite(), historical_config=config, run_id=RUN_ID, now_fn=lambda: NOW
    )
    with pytest.raises(ValueError, match="no configured window size"):
        fetcher.fetch_range(TARGET, "day", NOW - timedelta(days=10), NOW)


def test_an_empty_range_makes_no_request(historical_config) -> None:
    client = FakeKite()
    fetcher = make_fetcher(historical_config, client)
    result = fetcher.fetch_range(TARGET, "minute", NOW, NOW - timedelta(days=1))
    assert result.windows_planned == 0
    assert client.history_calls == []


def test_plan_describes_the_request_budget_without_calling_the_api(historical_config) -> None:
    client = FakeKite()
    fetcher = make_fetcher(historical_config, client)
    planned = fetcher.plan([TARGET], days=65, interval="minute")
    assert len(planned) == 3
    assert "NIFTY 50" in planned[0]
    assert client.history_calls == []


def test_backfill_covers_several_targets(historical_config) -> None:
    other = FetchTarget(instrument_token=999999, tradingsymbol="NIFTY 50 (mirror)")
    client = FakeKite()
    fetcher = make_fetcher(historical_config, client)
    result = fetcher.backfill([TARGET, other], days=1, interval="minute")
    assert result.symbols == ["NIFTY 50", "NIFTY 50 (mirror)"]
    assert len(client.history_calls) == 2

