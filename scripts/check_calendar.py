"""Is the calendar safe to trade on? - the pre-session guard for the NSE holiday list.

``config/calendar.yaml`` carries the NSE holiday list, and every date in it is flagged
``verified: false`` until it has been reconciled with the official NSE circular. That flag
is not cosmetic: the calendar decides whether a capture session runs at all. A provisional
date that turns out to be a *trading* day makes ``--mode capture`` exit ``NO_SESSION`` and
the session is simply never captured - the ticks do not exist anywhere afterwards, so the
loss is permanent.

This script answers the two questions an operator has the evening before:

1. **Which of the next N days trade, and why?** - the session window for each day, the
   weekly and monthly expiries, and the reason for every closure.
2. **Is any of them closed on a date nobody has confirmed yet?** - every provisional
   holiday inside the risk window is named, with the number of days left to fix it.

Usage::

    uv run python scripts/check_calendar.py                # next 30 days, 10-day risk window
    uv run python scripts/check_calendar.py --days 7
    uv run python scripts/check_calendar.py --as-of 2026-10-01 --json
    uv run python scripts/check_calendar.py --within 0     # only fail on today

Exit codes: 0 = nothing depends on a provisional date inside the risk window and (with
``--require-trading-day``) the day trades, 1 = something is risky or the day is closed,
2 = the configuration could not be read.

The script is offline and read-only: it reads the YAML, prints, and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/check_calendar.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.config_loader import AnjaneyaConfig, ConfigError, load_config  # noqa: E402
from core.market_calendar import NseCalendar  # noqa: E402

EXIT_OK = 0
EXIT_RISK = 1
EXIT_CONFIG_ERROR = 2

#: How many days ahead a provisional date is worth failing the check over.
DEFAULT_WITHIN = 10
#: How many days of the calendar to print.
DEFAULT_DAYS = 30


@dataclass(slots=True)
class DayRow:
    """One calendar day's verdict."""

    day: date
    weekday: str
    trading: bool
    reason: str = ""
    session: str | None = None
    expiry: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "date": self.day.isoformat(),
            "weekday": self.weekday,
            "trading": self.trading,
            "reason": self.reason,
            "session": self.session,
            "expiry": list(self.expiry),
        }


def _iso_date(value: str) -> date:
    """``argparse`` type for ``YYYY-MM-DD``."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/check_calendar.py",
        description="Show the next trading days, their expiries, and any provisional holiday "
        "they depend on.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML configuration.")
    parser.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start the listing here. Default: today in the calendar timezone.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="How many calendar days to list.")
    parser.add_argument(
        "--within",
        type=int,
        default=DEFAULT_WITHIN,
        help="Fail (exit 1) when a provisional holiday falls this many days ahead or sooner.",
    )
    parser.add_argument(
        "--require-trading-day",
        action="store_true",
        help="Fail (exit 1) unless the --as-of day is a trading day - the guard a scheduler "
        "uses to skip holidays and weekends.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the report as JSON.")
    return parser.parse_args(list(argv) if argv is not None else None)


def build_days(calendar: NseCalendar, *, start: date, days: int) -> list[DayRow]:
    """Classify ``days`` calendar days from ``start`` (inclusive)."""
    rows: list[DayRow] = []
    for offset in range(max(days, 0)):
        day = start + timedelta(days=offset)
        trading = calendar.is_trading_day(day)
        row = DayRow(day=day, weekday=day.strftime("%a"), trading=trading)
        if trading:
            window = calendar.session_window(day)
            row.session = f"{window[0]:%H:%M}-{window[1]:%H:%M}"
        else:
            entry = calendar.holiday_entry(day)
            if entry is not None:
                state = "verified" if entry.verified else "PROVISIONAL"
                row.reason = f"holiday: {entry.name} ({state})"
            else:
                row.reason = "weekend"
        tags: list[str] = []
        if calendar.is_special_session(day):
            # A day that is a weekend or a holiday but trades anyway - the operator
            # declared it. Named explicitly, because a Sunday showing as TRADING is
            # otherwise indistinguishable from a calendar bug.
            tags.append("special session")
        if day in {calendar.current_weekly_expiry(day), calendar.next_weekly_expiry(day)}:
            tags.append("weekly expiry")
        if calendar.is_monthly_expiry_day(day):
            tags.append("monthly expiry")
        row.expiry = tuple(tags)
        rows.append(row)
    return rows


def build_report(
    config: AnjaneyaConfig, calendar: NseCalendar, *, start: date, days: int, within: int
) -> dict[str, Any]:
    """The whole check, as data."""
    rows = build_days(calendar, start=start, days=days)
    horizon = start + timedelta(days=max(within, 0))
    risky = calendar.holidays_between(start, horizon, verified=False)
    # The holiday list is a whole year's calendar, so it holds dates that are already past:
    # "next" has to mean the first provisional date on or after ``start``, not the first in
    # the file.
    upcoming = calendar.holidays_between(start, date.max, verified=False)
    unverified_total = len(calendar.holidays_between(date.min, date.max, verified=False))
    return {
        "as_of": start.isoformat(),
        "as_of_weekday": start.strftime("%A"),
        "as_of_is_trading_day": calendar.is_trading_day(start),
        "days": days,
        "within": within,
        "timezone": config.timezone_name,
        "holidays_total": len(config.calendar.holidays),
        "holidays_verified": config.calendar.holidays_verified,
        "unverified_total": unverified_total,
        "session": {
            "regular_start": config.calendar.sessions.regular.start.isoformat(),
            "regular_end": config.calendar.sessions.regular.end.isoformat(),
            "weekly_expiry_weekday": config.calendar.weekly_expiry.weekday,
            "holiday_roll": config.calendar.weekly_expiry.holiday_roll,
        },
        "expiries": {
            "current_weekly": calendar.current_weekly_expiry(start).isoformat(),
            "next_weekly": calendar.next_weekly_expiry(start).isoformat(),
            "current_monthly": calendar.monthly_expiry(start.year, start.month).isoformat(),
        },
        "calendar_days": [row.to_dict() for row in rows],
        "trading_days": sum(1 for row in rows if row.trading),
        "window_end": rows[-1].day.isoformat() if rows else start.isoformat(),
        "risk_window": {"start": start.isoformat(), "end": horizon.isoformat(), "days": within},
        "risky_holidays": [
            {
                "date": entry.date.isoformat(),
                "name": entry.name,
                "days_away": (entry.date - start).days,
            }
            for entry in risky
        ],
        "next_unverified": (
            {
                "date": upcoming[0].date.isoformat(),
                "name": upcoming[0].name,
                "days_away": (upcoming[0].date - start).days,
            }
            if upcoming
            else None
        ),
    }


def render_report(report: dict[str, Any]) -> str:
    """The human-readable form of :func:`build_report`."""
    closed_and_required = bool(report.get("require_trading_day")) and not report[
        "as_of_is_trading_day"
    ]
    session = report["session"]
    expiries = report["expiries"]
    risk = report["risk_window"]
    provisional = "" if report["holidays_verified"] else "  (holidays_verified: false)"
    state = "trading day" if report["as_of_is_trading_day"] else "NOT a trading day"
    lines = [
        f"ANJANEYA calendar check - as of {report['as_of']} ({report['as_of_weekday']}): {state}",
        f"  timezone        : {report['timezone']}",
        f"  regular session : {session['regular_start'][:5]}-{session['regular_end'][:5]}",
        f"  weekly expiry   : {session['weekly_expiry_weekday']} (roll: {session['holiday_roll']})",
        f"  holidays        : {report['holidays_total']} configured, "
        f"{report['unverified_total']} PROVISIONAL{provisional}",
        f"  expiries        : weekly {expiries['current_weekly']} "
        f"(next {expiries['next_weekly']}), monthly {expiries['current_monthly']}",
        "",
        f"days {report['as_of']} -> {report['window_end']}: {report['trading_days']} trading of "
        f"{report['days']} calendar day(s)",
    ]
    for row in report["calendar_days"]:
        if row["trading"]:
            detail = row["session"] or ""
            if row["reason"]:
                detail = f"{detail} {row['reason']}".strip()
        else:
            detail = row["reason"]
        tags = f"  [{', '.join(row['expiry'])}]" if row["expiry"] else ""
        state = "TRADING" if row["trading"] else "CLOSED"
        lines.append(f"  {row['date']}  {row['weekday']}  {state:<7} {detail:<44}{tags}")
    lines.append("")
    if report["risky_holidays"]:
        lines.append(
            f"risk: {len(report['risky_holidays'])} provisional holiday(s) within {risk['days']} "
            f"day(s) (to {risk['end']})"
        )
        for item in report["risky_holidays"]:
            lines.append(
                f"  {item['date']}  {item['name']}  ({item['days_away']} day(s) away) - confirm it "
                "against the NSE circular, then set verified: true in config/calendar.yaml"
            )
        lines.append(
            "verdict: RISK - a provisional date that is really a trading day makes --mode capture "
            "exit NO_SESSION, and the ticks are gone for good"
        )
    else:
        lines.append(
            f"risk: none - no provisional holiday falls between {risk['start']} and {risk['end']}"
        )
        following = report["next_unverified"]
        if following:
            lines.append(
                f"note: the next provisional holiday is {following['date']} ({following['name']}) "
                f"in {following['days_away']} day(s) - outside the window, still worth confirming"
            )
        if closed_and_required:
            lines.append("verdict: CLOSED - --require-trading-day was asked for a trading day")
        else:
            lines.append("verdict: OK")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 clear, 1 risky or closed, 2 configuration."""
    args = parse_args(argv)
    try:
        config = load_config(Path(args.config_dir) if args.config_dir else None)
    except ConfigError as exc:
        print(f"ANJANEYA calendar: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    calendar = NseCalendar(config.calendar)
    start = args.as_of or calendar.now().date()
    report = build_report(config, calendar, start=start, days=args.days, within=args.within)
    report["require_trading_day"] = bool(args.require_trading_day)

    print(json.dumps(report, indent=2) if args.as_json else render_report(report))
    if report["risky_holidays"]:
        return EXIT_RISK
    if args.require_trading_day and not report["as_of_is_trading_day"]:
        return EXIT_RISK
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

