"""Tests for :class:`core.market_calendar.nse_calendar.NseCalendar`.

The calendar is where silent bugs are most expensive: a wrong expiry weekday or
a missed holiday quietly corrupts every expiry-relative feature. These tests use
the *shipped* configuration, so they also act as a regression guard on
``config/calendar.yaml``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from core.config_loader import CalendarConfig
from core.market_calendar import (
    DEFAULT_HOLIDAY_RISK_DAYS,
    OUT_OF_SESSION_BUCKET,
    WEEKDAY_INDEX,
    NseCalendar,
)
from core.schemas import ExpiryCyclePhase


@pytest.fixture
def calendar(config) -> NseCalendar:
    return NseCalendar(config.calendar)


def at(day: date, hour: int, minute: int, calendar: NseCalendar) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=calendar.timezone)


def build_calendar(**overrides) -> NseCalendar:
    """Build a small calendar directly, for behaviours the shipped config lacks."""
    payload = {
        "timezone": "Asia/Kolkata",
        "sessions": {
            "regular": {"start": "09:15", "end": "15:30"},
            "half_day": {"start": "09:15", "end": "13:00"},
            "half_day_dates": [],
            "special_sessions": [],
        },
        "holidays": [],
        "time_of_day_buckets": [{"name": "OPEN", "start": "09:15", "end": "09:30"}],
    }
    payload.update(overrides)
    return NseCalendar(CalendarConfig.model_validate(payload))


# ---------------------------------------------------------------------------
# Configuration wiring
# ---------------------------------------------------------------------------
def test_calendar_reflects_configuration(calendar: NseCalendar, config) -> None:
    assert calendar.config is config.calendar
    assert calendar.timezone.key == "Asia/Kolkata"
    assert calendar._expiry_weekday == WEEKDAY_INDEX[config.calendar.weekly_expiry.weekday]


def test_configured_weekday_is_a_business_day(config) -> None:
    """An expiry on a weekend would make every weekly expiry meaningless."""
    weekday = config.calendar.weekly_expiry.weekday
    assert WEEKDAY_INDEX[weekday] < 5, f"expiry weekday {weekday!r} is a weekend day"


def test_unverified_holidays_are_surfaced(config) -> None:
    """The holiday list is provisional; the code must say so, not hide it."""
    if config.calendar.holidays_verified:
        assert config.calendar.unverified_holidays == []
    else:
        assert config.calendar.unverified_holidays, (
            "holidays_verified is false but every holiday is marked verified"
        )


#: NSE's official 2026 trading-holiday list, **F&O segment**, as fetched from
#: ``https://www.nseindia.com/api/holiday-master?type=trading`` on 2026-10-01.
#: Pinned here on purpose: this is the reconciliation record, so the next annual
#: check has to write the new list down in both places and the shipped config
#: cannot drift away from what was verified unnoticed.
NSE_FO_2026_HOLIDAYS = frozenset(
    {
        date(2026, 1, 15),
        date(2026, 1, 26),
        date(2026, 2, 15),
        date(2026, 3, 3),
        date(2026, 3, 21),
        date(2026, 3, 26),
        date(2026, 3, 31),
        date(2026, 4, 3),
        date(2026, 4, 14),
        date(2026, 5, 1),
        date(2026, 5, 28),
        date(2026, 6, 26),
        date(2026, 8, 15),
        date(2026, 9, 14),
        date(2026, 10, 2),
        date(2026, 10, 20),
        date(2026, 11, 8),
        date(2026, 11, 10),
        date(2026, 11, 24),
        date(2026, 12, 25),
    }
)


def test_the_shipped_holiday_list_matches_the_nse_fo_2026_list(config) -> None:
    """The reconciliation itself, pinned against the exchange's own list.

    The seed this replaced was not merely provisional: five *weekday* dates were
    wrong, so five real sessions would have exited ``NO_SESSION`` and their ticks
    been lost forever, and seven official holidays were missing, so the capture
    would have run on days the exchange was shut. Both failure modes are silent,
    which is why the expected set is written out rather than derived.
    """
    assert {entry.date for entry in config.calendar.holidays} == set(NSE_FO_2026_HOLIDAYS)


def test_the_reconciliation_record_matches_the_committed_upstream_copy(config, repo_root) -> None:
    """The reconciliation is only as good as the copy of NSE's answer it rests on.

    ``docs/evidence/nse_holidays_2026.json`` is the raw, unedited response the shipped
    list was taken from. It is committed because the endpoint serves one year at a
    time and will have moved on by the time anyone re-checks this, so without it the
    only record of what NSE actually said would be this repository's own summary of
    it. With it, test, config and evidence agree with a *stored primary source*
    rather than with one another - and the descriptions are compared too, not just
    the dates, so a reworded festival is a diff rather than a silent reinterpretation.
    """
    import json

    payload = json.loads(
        (repo_root / "docs" / "evidence" / "nse_holidays_2026.json").read_text(encoding="utf-8")
    )
    upstream = {
        datetime.strptime(row["tradingDate"], "%d-%b-%Y").date(): row["description"]
        for row in payload["FO"]
    }
    assert set(upstream) == set(NSE_FO_2026_HOLIDAYS)
    for entry in config.calendar.holidays:
        assert entry.name == upstream[entry.date]


def test_the_reconciled_list_is_marked_verified(config) -> None:
    """``holidays_verified: true`` is a claim, and every entry has to back it up.

    ``check_calendar`` and the pre-session advisory both count entries with
    ``verified: false``, so one stale flag would keep warning about dates that
    have in fact been checked - and the warning would stop meaning anything.
    """
    assert config.calendar.holidays_verified is True
    assert config.calendar.unverified_holidays == []


def test_every_reconciled_date_closes_the_shipped_calendar(calendar: NseCalendar) -> None:
    """Each confirmed date is a non-trading day, weekends included."""
    for holiday in sorted(NSE_FO_2026_HOLIDAYS):
        assert calendar.is_holiday(holiday) is True
        assert calendar.is_trading_day(holiday) is False
        assert calendar.holiday_name(holiday)


def test_both_dump_confirmed_rolls_are_configured(calendar: NseCalendar) -> None:
    """Two dates are confirmed by a second source: the exchange's instrument dump.

    ``data/raw/instruments/NFO_2026-09-30.json`` lists 2026-10-19 and 2026-11-23
    as NIFTY option expiries, and both are Mondays. Weekly expiry is Tuesday, so
    they are ``previous_trading_day`` rolls for the Dussehra (2026-10-20) and Guru
    Nanak (2026-11-24) Tuesdays - and a Tuesday NSE does *not* close produces no
    such roll, so the roll is evidence from a source that never touched the
    holiday API.
    """
    for tuesday, monday in (
        (date(2026, 10, 20), date(2026, 10, 19)),
        (date(2026, 11, 24), date(2026, 11, 23)),
    ):
        assert tuesday.weekday() == 1, "weekly expiry is Tuesday"
        assert calendar.is_holiday(tuesday) is True
        assert calendar.weekly_expiry_of_week(monday) == monday
        assert calendar.is_expiry_day(monday) is True


def test_a_special_session_opens_a_day_the_defaults_would_close() -> None:
    """An explicitly configured session is a trading day - weekend and holiday or not.

    NSE lists 2026-11-08 as ``Diwali Laxmi Pujan*``: the asterisk marks a *muhurat*
    session, and the date is a Sunday. Weekend and holiday would each veto it, so a
    bare ``not weekend and not holiday`` can never express a muhurat session - which
    is precisely why ``special_sessions`` outranks both. The day's other facts are
    unchanged: it is still a Sunday and still NSE's holiday.
    """
    calendar = build_calendar(
        holidays=[{"date": "2026-11-08", "name": "Diwali Laxmi Pujan*", "verified": True}],
        sessions={
            "regular": {"start": "09:15", "end": "15:30"},
            "half_day": {"start": "09:15", "end": "13:00"},
            "half_day_dates": [],
            "special_sessions": [
                {"date": "2026-11-08", "start": "18:15", "end": "19:15", "name": "Muhurat"}
            ],
        },
    )
    day = date(2026, 11, 8)
    # The facts about the day do not change.
    assert calendar.is_weekend(day) is True
    assert calendar.is_holiday(day) is True
    assert calendar.holiday_name(day) == "Diwali Laxmi Pujan*"
    # The declaration wins, and it names the window.
    assert calendar.is_special_session(day) is True
    assert calendar.special_session(day) == (time(18, 15), time(19, 15))
    assert calendar.is_trading_day(day) is True
    assert calendar.session_window(day) == (time(18, 15), time(19, 15))
    # ... so the venue is open during it, which is what a capture actually asks.
    assert calendar.is_market_open(at(day, 18, 30, calendar)) is True
    assert calendar.is_market_open(at(day, 12, 0, calendar)) is False


def test_a_special_session_opens_a_weekday_holiday_too() -> None:
    """Muhurat falls midweek in most years - on a date NSE also lists as a holiday.

    The holiday veto, not the weekend one, is what has to yield here. Pinned with a
    past year so the rule is not tied to 2026's Sunday.
    """
    calendar = build_calendar(
        holidays=[{"date": "2025-10-21", "name": "Diwali Laxmi Pujan", "verified": True}],
        sessions={
            "regular": {"start": "09:15", "end": "15:30"},
            "half_day": {"start": "09:15", "end": "13:00"},
            "half_day_dates": [],
            "special_sessions": [
                {"date": "2025-10-21", "start": "18:15", "end": "19:15", "name": "Muhurat"}
            ],
        },
    )
    day = date(2025, 10, 21)
    assert day.weekday() == 1  # a Tuesday: only the holiday entry vetoed it
    assert calendar.is_holiday(day) is True
    assert calendar.is_special_session(day) is True
    assert calendar.is_trading_day(day) is True
    assert calendar.session_window(day) == (time(18, 15), time(19, 15))


def test_the_2026_muhurat_window_is_ready_but_not_invented(config) -> None:
    """The mechanism exists; the missing fact is NSE's announced timing.

    The shipped ``special_sessions`` list is empty, so 2026-11-08 is still reported as
    closed - which is what the file actually knows. The remaining gap is therefore a
    *data* gap (one circular, one config line) rather than the structural one it was:
    the calendar can now represent a session on a weekend or a holiday at all.
    """
    shipped = NseCalendar(config.calendar)
    assert shipped.is_special_session(date(2026, 11, 8)) is False
    assert shipped.special_session(date(2026, 11, 8)) is None
    assert shipped.is_trading_day(date(2026, 11, 8)) is False
    assert shipped.holiday_name(date(2026, 11, 8)) == "Diwali Laxmi Pujan*"


def test_describe_is_json_friendly(calendar: NseCalendar) -> None:
    import json

    payload = calendar.describe()
    json.dumps(payload, default=str)
    assert payload["timezone"] == "Asia/Kolkata"
    assert payload["buckets"]


# ---------------------------------------------------------------------------
# Day classification and sessions
# ---------------------------------------------------------------------------
def test_weekend_and_holiday_classification(calendar: NseCalendar) -> None:
    republic_day = date(2026, 1, 26)  # configured holiday, a Monday
    saturday = date(2026, 9, 26)

    assert calendar.is_holiday(republic_day) is True
    assert calendar.is_trading_day(republic_day) is False
    assert calendar.holiday_name(republic_day) == "Republic Day"
    assert calendar.is_weekend(saturday) is True
    assert calendar.is_trading_day(saturday) is False
    assert calendar.is_trading_day(date(2026, 9, 29)) is True
    assert calendar.holiday_name(date(2026, 9, 29)) is None


def test_session_window_and_bounds(calendar: NseCalendar, config) -> None:
    regular = date(2026, 9, 29)
    assert calendar.session_window(regular) == (
        config.calendar.sessions.regular.start,
        config.calendar.sessions.regular.end,
    )
    assert (calendar.session_start(regular).hour, calendar.session_start(regular).minute) == (9, 15)
    assert (calendar.session_end(regular).hour, calendar.session_end(regular).minute) == (15, 30)


def test_special_session_overrides_half_day_and_regular() -> None:
    """A muhurat session must win over both the half-day and regular windows."""
    cal = build_calendar(
        sessions={
            "regular": {"start": "09:15", "end": "15:30"},
            "half_day": {"start": "09:15", "end": "13:00"},
            "half_day_dates": ["2026-11-09"],
            "special_sessions": [
                {"date": "2026-11-08", "start": "18:15", "end": "19:15", "name": "Muhurat"}
            ],
        }
    )
    assert cal.session_window(date(2026, 11, 9)) == (time(9, 15), time(13, 0))
    assert cal.is_half_day(date(2026, 11, 9)) is True
    assert cal.session_window(date(2026, 11, 8)) == (time(18, 15), time(19, 15))


def test_market_open_and_minutes(calendar: NseCalendar) -> None:
    day = date(2026, 9, 29)
    assert calendar.is_market_open(at(day, 9, 14, calendar)) is False
    assert calendar.is_market_open(at(day, 9, 15, calendar)) is True
    assert calendar.is_market_open(at(day, 15, 29, calendar)) is True
    assert calendar.is_market_open(at(day, 15, 30, calendar)) is False
    assert calendar.is_market_open(at(date(2026, 1, 26), 10, 0, calendar)) is False

    assert calendar.minutes_to_close(at(day, 15, 0, calendar)) == pytest.approx(30.0)
    assert calendar.minutes_from_open(at(day, 9, 45, calendar)) == pytest.approx(30.0)


def test_naive_datetime_is_rejected(calendar: NseCalendar) -> None:
    with pytest.raises(ValueError, match="Naive datetime"):
        calendar.minutes_to_close(datetime(2026, 9, 29, 15, 0))  # noqa: DTZ001
    with pytest.raises(ValueError, match="Naive datetime"):
        calendar.time_of_day_bucket(datetime(2026, 9, 29, 10, 0))  # noqa: DTZ001


# ---------------------------------------------------------------------------
# Trading-day navigation
# ---------------------------------------------------------------------------
def test_next_and_previous_trading_day_skip_weekends(calendar: NseCalendar) -> None:
    friday = date(2026, 9, 25)
    assert calendar.next_trading_day(friday) == date(2026, 9, 28)  # Monday
    assert calendar.previous_trading_day(date(2026, 9, 28)) == friday
    assert calendar.next_trading_day(date(2026, 9, 28), inclusive=True) == date(2026, 9, 28)


def test_next_trading_day_skips_a_holiday(calendar: NseCalendar) -> None:
    # 2026-01-26 (Monday) is Republic Day -> the next session is Tuesday.
    assert calendar.next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)


def test_trading_days_range_is_inclusive_and_ordered(calendar: NseCalendar) -> None:
    days = calendar.trading_days(date(2026, 9, 25), date(2026, 9, 30))
    assert days == [date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)]
    assert calendar.count_trading_days(date(2026, 9, 25), date(2026, 9, 30)) == 4
    assert calendar.trading_days(date(2026, 9, 30), date(2026, 9, 25)) == []


def test_roll_holiday_policies() -> None:
    """Each configured roll policy must be honoured, including 'none'."""
    previous = build_calendar(
        holidays=[{"date": "2026-09-24", "name": "Test holiday"}],
        weekly_expiry={"weekday": "Thursday", "holiday_roll": "previous_trading_day"},
    )
    assert previous.roll_holiday(date(2026, 9, 24)) == date(2026, 9, 23)
    assert previous.roll_holiday(date(2026, 9, 22)) == date(2026, 9, 22)

    following = build_calendar(
        holidays=[{"date": "2026-09-24", "name": "Test holiday"}],
        weekly_expiry={"weekday": "Thursday", "holiday_roll": "next_trading_day"},
    )
    assert following.roll_holiday(date(2026, 9, 24)) == date(2026, 9, 25)

    unchanged = build_calendar(
        holidays=[{"date": "2026-09-24", "name": "Test holiday"}],
        weekly_expiry={"weekday": "Thursday", "holiday_roll": "none"},
    )
    assert unchanged.roll_holiday(date(2026, 9, 24)) == date(2026, 9, 24)


# ---------------------------------------------------------------------------
# Expiries
# ---------------------------------------------------------------------------
def test_weekly_expiry_is_the_configured_weekday(calendar: NseCalendar) -> None:
    # 2026-09-29 is a Tuesday, and Tuesday is the configured expiry weekday, so
    # that session *is* this week's expiry.
    assert calendar.weekly_expiry_of_week(date(2026, 9, 29)) == date(2026, 9, 29)
    assert calendar.current_weekly_expiry(date(2026, 9, 29)) == date(2026, 9, 29)
    assert calendar.next_weekly_expiry(date(2026, 9, 29)) == date(2026, 10, 6)
    assert calendar.previous_weekly_expiry(date(2026, 9, 29)) == date(2026, 9, 22)
    # The Wednesday after expiry already trades next week's contract.
    assert calendar.current_weekly_expiry(date(2026, 9, 30)) == date(2026, 10, 6)


def test_current_expiry_rolls_forward_after_expiry_day(calendar: NseCalendar) -> None:
    """On the Wednesday after the Tuesday expiry, the live contract is next week's."""
    wednesday = date(2026, 9, 30)
    assert calendar.current_weekly_expiry(wednesday) == date(2026, 10, 6)
    assert calendar.is_expiry_day(wednesday) is False
    assert calendar.is_expiry_day(date(2026, 9, 29)) is True


def test_expiry_on_a_holiday_rolls_back(calendar: NseCalendar) -> None:
    """Dussehra (2026-10-20) is a configured holiday *and* a Tuesday: expiry moves back.

    The NFO instrument dump fetched 2026-09-30 agrees - it lists the NIFTY option
    expiry of that week on Monday 2026-10-19.
    """
    assert calendar.is_holiday(date(2026, 10, 20)) is True
    assert calendar.weekly_expiry_of_week(date(2026, 10, 15)) == date(2026, 10, 19)
    assert calendar.is_expiry_day(date(2026, 10, 19)) is True


def test_monthly_expiry_is_the_last_weekly_expiry_of_the_month(calendar: NseCalendar) -> None:
    assert calendar.monthly_expiry(2026, 9) == date(2026, 9, 29)
    assert calendar.monthly_expiry(2026, 10) == date(2026, 10, 27)
    assert calendar.is_monthly_expiry_day(date(2026, 9, 29)) is True
    assert calendar.is_monthly_expiry_day(date(2026, 9, 22)) is False
    with pytest.raises(ValueError, match="month must be in 1..12"):
        calendar.monthly_expiry(2026, 13)


def test_current_monthly_expiry_rolls_into_the_next_month(calendar: NseCalendar) -> None:
    """`current_*` means "still trading", exactly as for the weekly series.

    On 2026-09-30 the September monthly contract (2026-09-29) has already
    expired, so the live monthly is October's; reporting the September date under
    a `current_` key is how an operator ends up sizing the wrong contract.
    """
    assert calendar.current_monthly_expiry(date(2026, 9, 30)) == date(2026, 10, 27)
    # Before the monthly expiry the month's own contract is the live one...
    assert calendar.current_monthly_expiry(date(2026, 9, 15)) == date(2026, 9, 29)
    # ... and on the expiry day itself it is still the current contract.
    assert calendar.current_monthly_expiry(date(2026, 9, 29)) == date(2026, 9, 29)
    # The roll crosses the year boundary rather than raising.
    assert calendar.current_monthly_expiry(date(2026, 12, 30)) == calendar.monthly_expiry(
        2027, 1
    )


def test_describe_reports_the_live_contracts(calendar: NseCalendar) -> None:
    """The status payload's `current_*` keys are the contracts a trader holds."""
    payload = calendar.describe(today=date(2026, 9, 30))
    assert payload["current_weekly_expiry"] == "2026-10-06"
    assert payload["current_monthly_expiry"] == "2026-10-27"


def test_expiries_between_walks_the_calendar(calendar: NseCalendar) -> None:
    weekly = calendar.expiries_between(date(2026, 9, 1), date(2026, 9, 30))
    assert weekly == [
        date(2026, 9, 1),
        date(2026, 9, 8),
        date(2026, 9, 15),
        date(2026, 9, 22),
        date(2026, 9, 29),
    ]
    monthly = calendar.expiries_between(date(2026, 9, 1), date(2026, 9, 30), monthly_only=True)
    assert monthly == [date(2026, 9, 29)]


# ---------------------------------------------------------------------------
# Days to expiry - all three supported definitions
# ---------------------------------------------------------------------------
def test_days_to_expiry_minutes_to_close(config, calendar: NseCalendar) -> None:
    """The shipped configuration uses minutes-to-close."""
    assert config.calendar.days_to_expiry.definition == "minutes_to_close"
    expiry = date(2026, 10, 1)
    # 09:15 to 15:30 on the expiry day is 375 minutes.
    assert calendar.days_to_expiry(at(expiry, 9, 15, calendar), expiry) == pytest.approx(375.0)
    # One hour before the close on the expiry day.
    assert calendar.days_to_expiry(at(expiry, 14, 30, calendar), expiry) == pytest.approx(60.0)
    # After the expiry close the value floors at zero, never negative.
    assert calendar.days_to_expiry(at(expiry, 15, 45, calendar), expiry) == 0.0
    # The configured cap is applied.
    earlier = at(date(2026, 9, 29), 9, 15, calendar)
    assert calendar.days_to_expiry(earlier, expiry) <= config.calendar.days_to_expiry.max_minutes


def test_days_to_expiry_calendar_days_definition() -> None:
    cal = build_calendar(days_to_expiry={"definition": "calendar_days", "max_minutes": 10080})
    expiry = date(2026, 10, 1)
    assert cal.days_to_expiry(at(date(2026, 9, 29), 10, 0, cal), expiry) == pytest.approx(2.0)
    assert cal.days_to_expiry(at(expiry, 10, 0, cal), expiry) == pytest.approx(0.0)
    assert cal.days_to_expiry(at(date(2026, 10, 5), 10, 0, cal), expiry) == 0.0


def test_days_to_expiry_trading_days_definition() -> None:
    cal = build_calendar(
        holidays=[{"date": "2026-09-30", "name": "Test holiday"}],
        days_to_expiry={"definition": "trading_days", "max_minutes": 10080},
    )
    expiry = date(2026, 10, 1)
    # 29th (Tue) and 1st (Thu) are sessions; the 30th is a holiday.
    assert cal.days_to_expiry(at(date(2026, 9, 29), 10, 0, cal), expiry) == pytest.approx(2.0)
    assert cal.days_to_expiry(at(expiry, 10, 0, cal), expiry) == pytest.approx(1.0)
    assert cal.days_to_expiry(at(date(2026, 10, 2), 10, 0, cal), expiry) == 0.0


# ---------------------------------------------------------------------------
# Expiry-cycle phase (RQ1)
# ---------------------------------------------------------------------------
def test_expiry_cycle_phase_sequence(calendar: NseCalendar) -> None:
    """Session after expiry = START_OF_CYCLE, mid-week = MID_CYCLE, expiry = EXPIRY_DAY."""
    expiry = date(2026, 9, 29)  # a Tuesday
    assert calendar.expiry_cycle_phase(expiry) is ExpiryCyclePhase.EXPIRY_DAY
    # Wednesday 23 Sep follows the 22 Sep expiry: the first session of the new cycle.
    assert calendar.expiry_cycle_phase(date(2026, 9, 23)) is ExpiryCyclePhase.START_OF_CYCLE
    assert calendar.expiry_cycle_phase(date(2026, 9, 24)) is ExpiryCyclePhase.MID_CYCLE
    assert calendar.expiry_cycle_phase(date(2026, 9, 28)) is ExpiryCyclePhase.MID_CYCLE


def test_expiry_cycle_phase_uses_the_supplied_expiry(calendar: NseCalendar) -> None:
    """An explicit expiry (e.g. next week's contract) must be respected."""
    assert (
        calendar.expiry_cycle_phase(date(2026, 9, 29), date(2026, 10, 6))
        is ExpiryCyclePhase.MID_CYCLE
    )
    # Already past the supplied expiry -> EXPIRY_DAY by definition.
    assert (
        calendar.expiry_cycle_phase(date(2026, 9, 29), date(2026, 9, 22))
        is ExpiryCyclePhase.EXPIRY_DAY
    )


def test_start_of_cycle_window_is_configurable() -> None:
    cal = build_calendar(expiry_cycle={"start_of_cycle": {"days_after_previous_expiry": 2}})
    # 2026-09-22 is the expiry; the 23rd and 24th are the next two sessions.
    assert cal.expiry_cycle_phase(date(2026, 9, 23)) is ExpiryCyclePhase.START_OF_CYCLE
    assert cal.expiry_cycle_phase(date(2026, 9, 24)) is ExpiryCyclePhase.START_OF_CYCLE
    assert cal.expiry_cycle_phase(date(2026, 9, 25)) is ExpiryCyclePhase.MID_CYCLE


# ---------------------------------------------------------------------------
# Time-of-day buckets (RQ2)
# ---------------------------------------------------------------------------
def test_time_of_day_buckets_are_half_open(calendar: NseCalendar) -> None:
    day = date(2026, 9, 29)
    assert calendar.time_of_day_bucket(at(day, 9, 15, calendar)) == "OPEN_915_930"
    assert calendar.time_of_day_bucket(at(day, 9, 29, calendar)) == "OPEN_915_930"
    # 09:30 belongs to the next bucket: the bucket end is exclusive.
    assert calendar.time_of_day_bucket(at(day, 9, 30, calendar)) == "MORNING_930_1100"
    assert calendar.time_of_day_bucket(at(day, 12, 0, calendar)) == "MIDDAY_1100_1300"
    assert calendar.time_of_day_bucket(at(day, 14, 0, calendar)) == "AFTERNOON_1300_1430"
    assert calendar.time_of_day_bucket(at(day, 15, 14, calendar)) == "CLOSE_1430_1515"


def test_time_of_day_bucket_outside_the_session(calendar: NseCalendar) -> None:
    day = date(2026, 9, 29)
    assert calendar.time_of_day_bucket(at(day, 8, 0, calendar)) == OUT_OF_SESSION_BUCKET
    assert calendar.time_of_day_bucket(at(day, 15, 15, calendar)) == OUT_OF_SESSION_BUCKET
    assert calendar.time_of_day_bucket(at(day, 23, 59, calendar)) == OUT_OF_SESSION_BUCKET
    # A bare ``time`` (no date) is accepted too - useful for feature code.
    assert calendar.time_of_day_bucket(time(9, 15)) == "OPEN_915_930"


def test_time_of_day_buckets_cover_the_analysis_window(config, calendar: NseCalendar) -> None:
    """Every minute of the analysis window must land in a real bucket.

    A gap would silently produce OUT_OF_SESSION values for genuine signals, which
    would then be dropped from research question RQ2 - a quiet source of bias.
    """
    start = config.settings.market.session_start
    end = config.settings.market.analysis_end
    minute = datetime.combine(date(2026, 9, 29), start, tzinfo=calendar.timezone)
    last = datetime.combine(date(2026, 9, 29), end, tzinfo=calendar.timezone)
    while minute < last:
        assert calendar.time_of_day_bucket(minute) != OUT_OF_SESSION_BUCKET, (
            f"{minute.time()} is not covered by any configured bucket"
        )
        minute += timedelta(minutes=1)


def test_bucket_boundaries_are_sorted(calendar: NseCalendar, config) -> None:
    boundaries = calendar.bucket_boundaries()
    assert [name for name, *_ in boundaries] == [
        bucket.name for bucket in config.calendar.time_of_day_buckets
    ]
    starts = [start for _, start, _ in boundaries]
    assert starts == sorted(starts)


def test_local_date_uses_the_calendar_timezone(calendar: NseCalendar) -> None:
    """A UTC timestamp just after midnight belongs to the next IST day."""

    utc_moment = datetime(2026, 9, 29, 20, 0, tzinfo=UTC)  # 01:30 IST on the 30th
    assert calendar.local_date(utc_moment) == date(2026, 9, 30)


# ---------------------------------------------------------------------------
# Holidays: the entry, and what depends on a provisional date
# ---------------------------------------------------------------------------
def provisional_calendar() -> NseCalendar:
    """A calendar with one confirmed and two provisional holidays."""
    return build_calendar(
        holidays=[
            {"date": "2026-10-02", "name": "Gandhi Jayanti", "verified": True},
            {"date": "2026-10-05", "name": "Provisional Monday", "verified": False},
            {"date": "2026-11-09", "name": "Provisional November", "verified": False},
        ]
    )


def test_holiday_entry_exposes_the_verified_flag() -> None:
    calendar = provisional_calendar()
    entry = calendar.holiday_entry(date(2026, 10, 5))
    assert entry is not None
    assert entry.name == "Provisional Monday"
    assert entry.verified is False
    assert calendar.holiday_name(date(2026, 10, 5)) == "Provisional Monday"
    assert calendar.holiday_entry(date(2026, 10, 6)) is None
    assert calendar.holiday_name(date(2026, 10, 6)) is None


def test_holidays_between_is_inclusive_and_ordered() -> None:
    calendar = provisional_calendar()
    entries = calendar.holidays_between(date(2026, 10, 2), date(2026, 10, 5))
    assert [entry.date for entry in entries] == [date(2026, 10, 2), date(2026, 10, 5)]


def test_holidays_between_filters_on_the_verified_flag() -> None:
    """The guard needs exactly the dates nobody has confirmed yet."""
    calendar = provisional_calendar()
    unverified = calendar.holidays_between(date(2026, 10, 1), date(2026, 10, 31), verified=False)
    assert [entry.date for entry in unverified] == [date(2026, 10, 5)]
    confirmed = calendar.holidays_between(date(2026, 10, 1), date(2026, 10, 31), verified=True)
    assert [entry.date for entry in confirmed] == [date(2026, 10, 2)]
    everything = calendar.holidays_between(date(2026, 10, 1), date(2026, 11, 30))
    assert [entry.date.isoformat() for entry in everything] == [
        "2026-10-02",
        "2026-10-05",
        "2026-11-09",
    ]


def test_holidays_between_rejects_a_backwards_window() -> None:
    calendar = provisional_calendar()
    with pytest.raises(ValueError, match="precedes start"):
        calendar.holidays_between(date(2026, 10, 5), date(2026, 10, 2))


def test_a_provisional_holiday_is_a_closed_day_until_it_is_confirmed() -> None:
    """The whole point: a provisional date closes the market in code, which is why the
    operator has to see it before the session and why ``verified`` cannot be decoration."""
    calendar = provisional_calendar()
    assert calendar.is_trading_day(date(2026, 10, 5)) is False
    assert calendar.holidays_between(date(2026, 10, 1), date(2026, 10, 10), verified=False)


def test_unverified_holiday_risk_uses_the_given_day_and_the_shared_window() -> None:
    """The apps' pre-session query: provisional dates from ``today`` over a short window."""
    calendar = provisional_calendar()

    assert [entry.date for entry in calendar.unverified_holiday_risk(today=date(2026, 10, 1))] == [
        date(2026, 10, 5)
    ]
    # The window is inclusive: seven days from 2026-09-28 reaches 2026-10-05 exactly.
    assert [entry.date for entry in calendar.unverified_holiday_risk(today=date(2026, 9, 28))] == [
        date(2026, 10, 5)
    ]
    assert calendar.unverified_holiday_risk(today=date(2026, 9, 20)) == []
    assert calendar.unverified_holiday_risk(today=date(2026, 10, 6)) == []
    # A custom window is honoured (the guard in the apps uses the default).
    assert [
        entry.date for entry in calendar.unverified_holiday_risk(today=date(2026, 10, 6), days=40)
    ] == [date(2026, 11, 9)]
    assert DEFAULT_HOLIDAY_RISK_DAYS == 7

