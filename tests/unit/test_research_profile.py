"""Unit tests for ``core/research/profile.py`` (spec 24.7 ATR profile, spec 24.8 intraday path).

These two items are the only ones that describe *the market* rather than the strategy, so the
tests are about the two ways a market profile misleads:

* **a bucket boundary that is off by one bar.** A stored bar is timestamped at its **open**,
  and the profile buckets it by when it *closed* (``timestamp + timeframe``) - the same
  convention ``--mode build-features`` uses for ``time_of_day_bucket``. The first test pins
  that, because a one-bar shift would move the 09:15 bar into the previous session.
* **an average of things that are not comparable.** Two symbols in one call is refused
  outright, and a bar outside every configured bucket is counted (``OUT_OF_SESSION``) rather
  than folded into the nearest bucket.

The intraday path is checked numerically: a monotonically rising session must produce a
monotonically rising average path, and the path must start at the bucket's *fill* (the open of
the first bar whose close lands in the bucket), not at the session's open.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.config_loader import CalendarConfig
from core.market_calendar import NseCalendar
from core.research.profile import (
    BAR_COLUMNS,
    OUT_OF_SESSION,
    ProfileError,
    bucket_order,
    build_bucket_profiles,
    build_intraday_paths,
)

IST = ZoneInfo("Asia/Kolkata")
SESSION = date(2026, 9, 28)  # a Monday
OTHER = date(2026, 10, 1)  # the Thursday after that week's Tuesday expiry


def calendar() -> NseCalendar:
    """A calendar whose buckets are wide enough to make each assertion unambiguous."""
    return NseCalendar(
        CalendarConfig.model_validate(
            {
                "timezone": "Asia/Kolkata",
                "sessions": {
                    "regular": {"start": "09:15", "end": "15:30"},
                    "half_day": {"start": "09:15", "end": "13:00"},
                    "half_day_dates": [],
                    "special_sessions": [],
                },
                "holidays": [],
                "time_of_day_buckets": [
                    {"name": "OPEN_915_945", "start": "09:15", "end": "09:45"},
                    {"name": "REST_945_1530", "start": "09:45", "end": "15:30"},
                ],
            }
        )
    )


def bar(
    moment: datetime,
    *,
    open_: float,
    close: float,
    high: float | None = None,
    low: float | None = None,
    symbol: str = "NIFTY 50",
    timeframe: str = "1m",
) -> dict:
    """One index bar with the stored shape (``timestamp`` is the bar's open)."""
    return {
        "timestamp": moment,
        "tradingsymbol": symbol,
        "timeframe": timeframe,
        "open": open_,
        "high": high if high is not None else max(open_, close) + 1.0,
        "low": low if low is not None else min(open_, close) - 1.0,
        "close": close,
    }


def bars(
    *,
    days: tuple[date, ...] = (SESSION,),
    bars_per_day: int = 30,
    minute_start: int = 0,
    symbol: str = "NIFTY 50",
) -> pd.DataFrame:
    """One-minute bars per session, each closing one point above its open."""
    rows = []
    for day in days:
        for index in range(bars_per_day):
            moment = datetime.combine(day, time(9, 15), tzinfo=IST) + timedelta(
                minutes=minute_start + index
            )
            base = 100.0 + index
            rows.append(bar(moment, open_=base, close=base + 1.0, symbol=symbol))
    frame = pd.DataFrame(rows, columns=list(BAR_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


# ---------------------------------------------------------------------------
# 24.7 - the volatility profile
# ---------------------------------------------------------------------------
def test_the_profile_buckets_a_bar_by_when_it_closed() -> None:
    frame = bars(days=(SESSION, OTHER), bars_per_day=30)

    table = build_bucket_profiles(frame, calendar(), atr_period=3, min_sessions=2)

    assert [name for name, _, _ in calendar().bucket_boundaries()] == bucket_order(calendar())
    assert [profile.bucket for profile in table.profiles] == ["OPEN_915_945", "REST_945_1530"]
    first = table.profiles[0]
    # the 09:15 bar closes at 09:16, so it is in OPEN; the 09:44 bar closes at 09:45 and is
    # not. Both sessions are MID_CYCLE, so this cell holds 29 bars from each of them.
    assert first.bars == 58
    assert first.sessions == 2
    assert first.range_mean == pytest.approx(3.0)
    rest = table.profiles[1]
    assert rest.bars == 2  # the 09:44 bar of each session
    assert first.atr_mean is not None and first.atr_mean > 0
    assert first.preliminary is False
    assert table.bars == 60
    assert table.dropped_out_of_session == 0
    assert table.symbol == "NIFTY 50"


def test_a_bar_outside_every_bucket_is_counted_not_folded_in() -> None:
    frame = bars(days=(SESSION,), bars_per_day=30)
    early = pd.DataFrame(
        [bar(datetime.combine(SESSION, time(9, 5), tzinfo=IST), open_=99.0, close=99.5)],
        columns=list(BAR_COLUMNS),
    )
    early["timestamp"] = pd.to_datetime(early["timestamp"])
    frame = pd.concat([frame, early], ignore_index=True)

    table = build_bucket_profiles(frame, calendar(), atr_period=3, min_sessions=1)

    assert table.dropped_out_of_session == 1
    # the dropped bar is *not* inside any profile, and not part of the session count
    assert sum(profile.bars for profile in table.profiles) == 30
    assert table.sessions == 1
    assert all(profile.bucket != OUT_OF_SESSION for profile in table.profiles)


def test_a_cell_built_from_one_session_is_preliminary() -> None:
    frame = bars(days=(SESSION,), bars_per_day=30)

    table = build_bucket_profiles(frame, calendar(), atr_period=3, min_sessions=2)

    assert table.profiles
    assert all(profile.preliminary for profile in table.profiles)
    assert table.preliminary_cells == [profile.label for profile in table.profiles]


def test_the_profile_refuses_input_it_cannot_average() -> None:
    frame = bars(days=(SESSION,), bars_per_day=5)

    two_symbols = pd.concat(
        [frame, frame.assign(tradingsymbol="NIFTY26OCTFUT")], ignore_index=True
    )
    with pytest.raises(ProfileError, match="2 symbols"):
        build_bucket_profiles(two_symbols, calendar())

    naive = frame.copy()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(ProfileError, match="naive"):
        build_bucket_profiles(naive, calendar())

    with pytest.raises(ProfileError, match="missing column"):
        build_bucket_profiles(frame.drop(columns=["high"]), calendar())

    with pytest.raises(ProfileError, match="no bars"):
        build_bucket_profiles(frame.iloc[0:0], calendar())



# ---------------------------------------------------------------------------
# 24.8 - the average cumulative intraday path
# ---------------------------------------------------------------------------
def test_the_path_starts_at_the_buckets_fill_and_rises_with_a_rising_session() -> None:
    frame = bars(days=(SESSION, OTHER), bars_per_day=35)

    paths = {path.bucket: path for path in build_intraday_paths(frame, calendar(), min_sessions=2)}
    opening = paths["OPEN_915_945"]

    # the fill is the open of the 09:15 bar, and that bar's close (09:16) is the first point
    assert opening.base_minute == "09:16"
    assert opening.sessions == 2
    assert opening.preliminary is False
    assert opening.points[0].bar_index == 1
    # the 09:15 bar opens at 100 and closes at 101 -> the first point is +1%
    assert opening.points[0].mean_return_pct == pytest.approx(1.0)
    assert opening.points[0].sample == 2
    means = [point.mean_return_pct for point in opening.points]
    assert means == sorted(means)
    # the path does not stop at the bucket's end: it follows the session to its last bar
    assert opening.points[-1].minute == "09:50"
    assert opening.points[-1].bar_index == 35


def test_a_path_below_the_session_rule_is_labelled_and_absent_when_unreachable() -> None:
    frame = bars(days=(SESSION,), bars_per_day=10)

    paths = build_intraday_paths(frame, calendar(), min_sessions=2)

    ids = {path.bucket: path for path in paths}
    assert "OPEN_915_945" in ids and ids["OPEN_915_945"].preliminary is True
    # only ten bars: nothing reaches the REST bucket at all, so it is absent, not empty
    assert "REST_945_1530" not in ids
    assert all(point.preliminary for point in ids["OPEN_915_945"].points)


def test_a_dropped_minute_is_dropped_rather_than_zero_filled() -> None:
    """The second session is missing a minute; that minute's sample must shrink, not read 0."""
    first = bars(days=(SESSION,), bars_per_day=20)
    second = bars(days=(OTHER,), bars_per_day=20, minute_start=10)
    frame = pd.concat([first, second], ignore_index=True)

    paths = {path.bucket: path for path in build_intraday_paths(frame, calendar(), min_sessions=1)}
    opening = paths["OPEN_915_945"]
    by_minute = {point.minute: point for point in opening.points}

    # session 1 closes 09:16..09:35, session 2 (starting 09:25) closes 09:26..09:45
    assert by_minute["09:30"].sample == 2
    assert by_minute["09:35"].sample == 2
    assert by_minute["09:43"].sample == 1
    assert opening.sessions == 2

