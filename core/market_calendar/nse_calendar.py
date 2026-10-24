"""NSE market calendar: sessions, holidays, expiries and research time buckets.

Why this module exists
----------------------
Almost every ANJANEYA research question is a question about *time*: the weekly
expiry cycle, the time of day, days-to-expiry, the session window. Getting the
calendar wrong invalidates results silently, so the calendar is a first-class,
tested engine rather than a helper sprinkled through the codebase.

Rules honoured
--------------
* Every input/output ``datetime`` is timezone-aware (Asia/Kolkata by default);
  a naive datetime raises immediately.
* Weekday of weekly expiry, holiday-roll policy, monthly-expiry rule and
  time-of-day buckets are **configuration**, never code (see
  ``config/calendar.yaml``).
* Nothing here mutates shared state, reads files or touches the network.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from core.config_loader import CalendarConfig, HolidayEntry, TimeOfDayBucketConfig
from core.constants import get_timezone
from core.schemas import ExpiryCyclePhase, TimeOfDayBucket

__all__ = ["DEFAULT_HOLIDAY_RISK_DAYS", "WEEKDAY_INDEX", "NseCalendar", "OUT_OF_SESSION_BUCKET"]

#: Weekday name -> Python ``date.weekday()`` index (Monday == 0).
WEEKDAY_INDEX: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

#: Bucket name used for timestamps that fall outside every configured bucket.
OUT_OF_SESSION_BUCKET = TimeOfDayBucket.OUT_OF_SESSION.value

#: How far ahead a still-provisional holiday is worth flagging, by default, in the apps'
#: ``--mode status`` and in a capture session's pre-session advisory. Seven days is roughly
#: "this week's sessions": the operator has time to check the NSE circular before the date
#: arrives, which is the only moment it can still be fixed.
DEFAULT_HOLIDAY_RISK_DAYS = 7


class NseCalendar:
    """The NSE trading calendar derived from ``config/calendar.yaml``.

    The object holds no mutable state, so it is safe to share between threads.
    """

    def __init__(self, calendar: CalendarConfig) -> None:
        self.config = calendar
        self.timezone = get_timezone(calendar.timezone)
        self._holidays: frozenset[date] = frozenset(calendar.holiday_dates)
        self._half_days: frozenset[date] = frozenset(calendar.sessions.half_day_dates)
        self._special: dict[date, tuple[time, time]] = {
            session.date: (session.start, session.end)
            for session in calendar.sessions.special_sessions
        }
        self._expiry_weekday: int = WEEKDAY_INDEX[calendar.weekly_expiry.weekday]
        self._buckets: tuple[TimeOfDayBucketConfig, ...] = tuple(calendar.time_of_day_buckets)

    # -- representation -----------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NseCalendar(tz={self.config.timezone!r}, "
            f"expiry_weekday={self.config.weekly_expiry.weekday!r}, "
            f"holidays={len(self._holidays)}, buckets={len(self._buckets)})"
        )

    # -- helpers ------------------------------------------------------------
    def _aware(self, moment: datetime) -> datetime:
        """Convert ``moment`` to the calendar timezone, rejecting naive input."""
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            raise ValueError(
                "Naive datetime received. ANJANEYA requires timezone-aware timestamps; "
                "attach tzinfo (e.g. datetime(..., tzinfo=ZoneInfo('Asia/Kolkata')))."
            )
        return moment.astimezone(self.timezone)

    def local_date(self, moment: datetime) -> date:
        """Exchange-local (IST) calendar date of an aware timestamp."""
        return self._aware(moment).date()

    def now(self) -> datetime:
        """Current time in the calendar timezone."""
        return datetime.now(self.timezone)

    # -- day classification -------------------------------------------------
    def is_weekend(self, day: date) -> bool:
        """True for Saturday/Sunday."""
        return day.weekday() >= 5

    def is_holiday(self, day: date) -> bool:
        """True when ``day`` is a configured market holiday."""
        return day in self._holidays

    def is_trading_day(self, day: date) -> bool:
        """True when the market has a session on ``day``.

        Precedence: an explicitly configured **special session** wins over the
        weekend and holiday defaults (see :meth:`is_special_session`). That
        override is the setting's whole purpose - a muhurat session is held on a
        date NSE *also* lists as a trading holiday, and falls on a Saturday or
        Sunday in some years, so without it the one session the operator has
        explicitly declared is exactly the one the calendar would deny, and a
        capture would exit ``NO_SESSION`` on a day the exchange traded.

        Every other day is the plain ``not weekend and not holiday``.
        """
        if day in self._special:
            return True
        return not self.is_weekend(day) and not self.is_holiday(day)

    def is_special_session(self, day: date) -> bool:
        """True when ``day`` carries an explicitly configured ad-hoc session.

        Such a day is a trading day *by declaration*, even when it is a weekend
        or a configured holiday - see :meth:`is_trading_day`.
        """
        return day in self._special

    def special_session(self, day: date) -> tuple[time, time] | None:
        """The configured ad-hoc ``(start, end)`` for ``day``, or ``None``."""
        return self._special.get(day)

    def is_half_day(self, day: date) -> bool:
        """True when ``day`` uses the configured half-day session."""
        return day in self._half_days

    def holiday_name(self, day: date) -> str | None:
        """Name of the holiday, if ``day`` is a holiday."""
        entry = self.holiday_entry(day)
        return entry.name if entry is not None else None

    def holiday_entry(self, day: date) -> HolidayEntry | None:
        """The configured holiday entry for ``day`` (with its ``verified`` flag), or ``None``."""
        for entry in self.config.holidays:
            if entry.date == day:
                return entry
        return None

    def holidays_between(
        self, start: date, end: date, *, verified: bool | None = None
    ) -> list[HolidayEntry]:
        """Configured holidays in ``[start, end]``, oldest first.

        ``verified`` filters on the provisional/confirmed flag: ``False`` returns only the
        dates that still need reconciling with the NSE circular - the dates that can make a
        capture session exit ``NO_SESSION`` on a day that is really a trading day, and that
        the operator therefore has to see *before* the session, not after it.
        """
        if end < start:
            raise ValueError(f"end ({end.isoformat()}) precedes start ({start.isoformat()})")
        return [
            entry
            for entry in sorted(self.config.holidays, key=lambda item: item.date)
            if start <= entry.date <= end and (verified is None or entry.verified is verified)
        ]

    def unverified_holiday_risk(
        self, *, days: int = DEFAULT_HOLIDAY_RISK_DAYS, today: date | None = None
    ) -> list[HolidayEntry]:
        """Provisional holidays from ``today`` (inclusive) to ``today + days``.

        The one query the apps ask before a session: a provisional date inside this window can
        turn a real trading day into ``NO_SESSION``, and the capture cannot be repeated later.
        """
        start = today or self.now().date()
        return self.holidays_between(start, start + timedelta(days=days), verified=False)

    # -- sessions -----------------------------------------------------------
    def session_window(self, day: date) -> tuple[time, time]:
        """Session ``(start, end)`` for ``day``.

        Precedence: special session > half-day > regular session.
        """
        if day in self._special:
            return self._special[day]
        if self.is_half_day(day):
            window = self.config.sessions.half_day
            return window.start, window.end
        window = self.config.sessions.regular
        return window.start, window.end

    def session_start(self, day: date) -> datetime:
        """Aware session start datetime for ``day``."""
        start, _ = self.session_window(day)
        return datetime.combine(day, start, tzinfo=self.timezone)

    def session_end(self, day: date) -> datetime:
        """Aware session end datetime for ``day``."""
        _, end = self.session_window(day)
        return datetime.combine(day, end, tzinfo=self.timezone)

    def is_market_open(self, moment: datetime) -> bool:
        """True when ``moment`` falls inside a trading session ``[start, end)``."""
        local = self._aware(moment)
        day = local.date()
        if not self.is_trading_day(day):
            return False
        return self.session_start(day) <= local < self.session_end(day)

    def minutes_to_close(self, moment: datetime) -> float:
        """Minutes from ``moment`` to that local day's session close (may be negative)."""
        local = self._aware(moment)
        return (self.session_end(local.date()) - local).total_seconds() / 60.0

    def minutes_from_open(self, moment: datetime) -> float:
        """Minutes since that local day's session open (may be negative before open)."""
        local = self._aware(moment)
        return (local - self.session_start(local.date())).total_seconds() / 60.0

    # -- trading-day navigation --------------------------------------------
    def next_trading_day(self, day: date, *, inclusive: bool = False) -> date:
        """First trading day on/after ``day``.

        With ``inclusive=False`` (default) the search starts at ``day + 1``, so
        the result is strictly after ``day``.
        """
        candidate = day if inclusive else day + timedelta(days=1)
        for _ in range(400):  # a year of holidays cannot exceed this
            if self.is_trading_day(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise RuntimeError(  # pragma: no cover - defensive
            f"No trading day found after {day.isoformat()}; check the holiday configuration"
        )

    def previous_trading_day(self, day: date, *, inclusive: bool = False) -> date:
        """Last trading day on/before ``day`` (strictly before it unless inclusive)."""
        candidate = day if inclusive else day - timedelta(days=1)
        for _ in range(400):
            if self.is_trading_day(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise RuntimeError(  # pragma: no cover - defensive
            f"No trading day found before {day.isoformat()}; check the holiday configuration"
        )

    def roll_holiday(self, day: date) -> date:
        """Apply the configured holiday-roll policy to a candidate expiry date.

        * ``previous_trading_day`` - NSE's normal behaviour for an expiry that
          falls on a holiday.
        * ``next_trading_day``     - the alternative used by some exchanges.
        * ``none``                 - return the candidate unchanged.
        """
        if self.is_trading_day(day):
            return day
        policy = self.config.weekly_expiry.holiday_roll
        if policy == "previous_trading_day":
            return self.previous_trading_day(day)
        if policy == "next_trading_day":
            return self.next_trading_day(day)
        return day

    def trading_days(self, start: date, end: date) -> list[date]:
        """Trading days in the inclusive range ``[start, end]`` (ascending)."""
        if end < start:
            return []
        return [
            start + timedelta(days=offset)
            for offset in range((end - start).days + 1)
            if self.is_trading_day(start + timedelta(days=offset))
        ]

    def count_trading_days(self, start: date, end: date) -> int:
        """Number of trading days in the inclusive range ``[start, end]``."""
        return len(self.trading_days(start, end))

    # -- expiries -----------------------------------------------------------
    def weekly_expiry_of_week(self, day: date) -> date:
        """The weekly expiry belonging to the ISO week that contains ``day``.

        The scheduled weekday comes from configuration (Tuesday for NIFTY since
        NSE moved it off Thursday; never hard-coded), and the result is passed
        through :meth:`roll_holiday`.
        """
        candidate = day + timedelta(days=(self._expiry_weekday - day.weekday()) % 7)
        return self.roll_holiday(candidate)

    def current_weekly_expiry(self, day: date) -> date:
        """The next live weekly expiry on/after ``day``.

        This is the expiry a trader would currently be holding: when ``day`` is
        already past this week's expiry (e.g. the Wednesday after a Tuesday
        expiry), the following week's expiry is returned.
        """
        candidate = self.weekly_expiry_of_week(day)
        if candidate < day:
            candidate = self.weekly_expiry_of_week(day + timedelta(days=7))
        return candidate

    def next_weekly_expiry(self, day: date) -> date:
        """The weekly expiry strictly after the one returned by
        :meth:`current_weekly_expiry` (used for next-week contracts)."""
        return self.weekly_expiry_of_week(self.current_weekly_expiry(day) + timedelta(days=7))

    def previous_weekly_expiry(self, day: date) -> date:
        """The most recent weekly expiry strictly before ``day``."""
        candidate = self.weekly_expiry_of_week(day)
        while candidate >= day:
            candidate = self.weekly_expiry_of_week(candidate - timedelta(days=7))
        return candidate

    def monthly_expiry(self, year: int, month: int) -> date:
        """Monthly expiry = the last weekly expiry that falls inside the month.

        When the scheduled expiry weekday lands in the next month (September 2026
        ends on a Wednesday, so the Tuesday of its last week is 29 September and
        the next one is 6 October), the *previous* week's expiry is the month's
        monthly expiry - which is exactly what NSE's monthly contract does.
        """
        if not 1 <= month <= 12:
            raise ValueError(f"month must be in 1..12, got {month}")
        last_day = (
            date(year, 12, 31)
            if month == 12
            else date(year, month + 1, 1) - timedelta(days=1)
        )
        candidate = self.weekly_expiry_of_week(last_day)
        while candidate > last_day:
            candidate = self.weekly_expiry_of_week(candidate - timedelta(days=7))
        return candidate

    def current_monthly_expiry(self, day: date) -> date:
        """The monthly contract currently trading: the monthly expiry on/after ``day``.

        When the month's own monthly expiry has already passed (the last days of
        the expiry month), the next month's contract is the live one - the same
        rule :meth:`current_weekly_expiry` applies to the weekly series, so
        ``current_*_expiry`` never names a contract that has stopped trading.
        """
        candidate = self.monthly_expiry(day.year, day.month)
        if candidate < day:
            year, month = (
                (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
            )
            candidate = self.monthly_expiry(year, month)
        return candidate

    def is_expiry_day(self, day: date) -> bool:
        """True when ``day`` is a weekly (and possibly monthly) expiry session."""
        return self.is_trading_day(day) and self.weekly_expiry_of_week(day) == day

    def is_monthly_expiry_day(self, day: date) -> bool:
        """True when ``day`` is the monthly expiry session."""
        return self.is_expiry_day(day) and self.monthly_expiry(day.year, day.month) == day

    def expiries_between(self, start: date, end: date, *, monthly_only: bool = False) -> list[date]:
        """All expiry dates in the inclusive range ``[start, end]``."""
        result: list[date] = []
        cursor = self.current_weekly_expiry(start)
        while cursor <= end:
            if not monthly_only or self.is_monthly_expiry_day(cursor):
                result.append(cursor)
            cursor = self.next_weekly_expiry(cursor)
        return result

    # -- days to expiry -----------------------------------------------------
    def days_to_expiry(self, moment: datetime, expiry: date) -> float:
        """Days-to-expiry using the definition configured in ``calendar.yaml``.

        * ``calendar_days``    - raw date difference (integer-valued),
        * ``trading_days``     - number of remaining trading sessions,
        * ``minutes_to_close`` - minutes until the expiry session close
          (capped at ``days_to_expiry.max_minutes``).

        The value is never negative: after the expiry close it is 0.
        """
        local = self._aware(moment)
        definition = self.config.days_to_expiry.definition

        if definition == "calendar_days":
            value = float((expiry - local.date()).days)
            return max(value, 0.0)

        if definition == "trading_days":
            # Remaining sessions: current day (if it is a trading day and the
            # expiry is not yet past) plus every trading day up to the expiry.
            start = self.next_trading_day(local.date(), inclusive=True)
            if start > expiry:
                return 0.0
            return float(self.count_trading_days(start, expiry))

        minutes = (self.session_end(expiry) - local).total_seconds() / 60.0
        if minutes <= 0:
            return 0.0
        return min(minutes, float(self.config.days_to_expiry.max_minutes))

    # -- expiry cycle phase (RQ1) -------------------------------------------
    def expiry_cycle_phase(self, day: date, expiry: date | None = None) -> ExpiryCyclePhase:
        """Classify a trading day into the weekly expiry cycle.

        * ``EXPIRY_DAY``      - the day *is* the effective expiry,
        * ``START_OF_CYCLE``  - within ``start_of_cycle.days_after_previous_expiry``
          trading days after the previous expiry,
        * ``MID_CYCLE``       - anything else in between.

        ``expiry`` may be supplied explicitly (e.g. for a mid-week contract);
        otherwise the next live weekly expiry is used.
        """
        effective_expiry = expiry or self.current_weekly_expiry(day)
        if day >= effective_expiry:
            return ExpiryCyclePhase.EXPIRY_DAY

        previous_expiry = self.previous_weekly_expiry(day)
        # Sessions strictly *after* the previous expiry (the expiry day itself
        # belongs to the expired cycle, not to the new one).
        sessions_since_previous = self.count_trading_days(
            self.next_trading_day(previous_expiry), day
        )
        threshold = self.config.expiry_cycle.start_of_cycle.days_after_previous_expiry
        if sessions_since_previous <= threshold:
            return ExpiryCyclePhase.START_OF_CYCLE
        return ExpiryCyclePhase.MID_CYCLE

    # -- time-of-day buckets (RQ2) -----------------------------------------
    def time_of_day_bucket(self, moment: datetime | time) -> str:
        """Name of the configured time-of-day bucket containing the moment.

        Buckets are half-open ``[start, end)``. A timestamp outside every
        configured bucket returns ``OUT_OF_SESSION`` rather than raising, so a
        slightly-off tick cannot crash a capture run.
        """
        local_time = self._aware(moment).time() if isinstance(moment, datetime) else moment
        for bucket in self._buckets:
            if bucket.start <= local_time < bucket.end:
                return bucket.name
        return OUT_OF_SESSION_BUCKET

    def bucket_boundaries(self) -> list[tuple[str, time, time]]:
        """Configured buckets as ``(name, start, end)`` tuples (ascending)."""
        return [
            (bucket.name, bucket.start, bucket.end)
            for bucket in sorted(self._buckets, key=lambda item: item.start)
        ]

    # -- introspection ------------------------------------------------------
    def describe(self, *, today: date | None = None) -> dict[str, object]:
        """Human-readable summary, used by ``scripts/health_check.py``.

        ``today`` is injectable so a report can state what the calendar said on a
        given day; it defaults to the current date in the market timezone.
        """
        today = today or self.now().date()
        return {
            "timezone": self.config.timezone,
            "expiry_weekday": self.config.weekly_expiry.weekday,
            "holiday_roll": self.config.weekly_expiry.holiday_roll,
            "holidays_total": len(self._holidays),
            "holidays_verified": self.config.holidays_verified,
            "unverified_holidays": self.config.unverified_holidays,
            "regular_session": (
                self.config.sessions.regular.start.isoformat(),
                self.config.sessions.regular.end.isoformat(),
            ),
            "days_to_expiry_definition": self.config.days_to_expiry.definition,
            "buckets": [name for name, *_ in self.bucket_boundaries()],
            "today": today.isoformat(),
            "today_is_trading_day": self.is_trading_day(today),
            "today_bucket_sample": self.time_of_day_bucket(
                self.session_start(today) + timedelta(minutes=1)
            ),
            "current_weekly_expiry": self.current_weekly_expiry(today).isoformat(),
            "next_weekly_expiry": self.next_weekly_expiry(today).isoformat(),
            "current_monthly_expiry": self.current_monthly_expiry(today).isoformat(),
        }
