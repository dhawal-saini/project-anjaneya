"""Refresh the NSE trading-holiday list in ``config/calendar.yaml`` from NSE's own API.

``config/calendar.yaml`` decides whether a capture session runs at all, and every
date in it is only trustworthy once it has been reconciled with NSE. This script
does that reconciliation mechanically instead of by hand:

1. **fetch** ``https://www.nseindia.com/api/holiday-master?type=trading`` (a warm-up
   request first, because the endpoint needs NSE's cookies);
2. **record** the raw response at ``docs/evidence/nse_holidays_<year>.json`` - the
   unedited evidence the tests assert the YAML against;
3. **diff** the fetched F&O dates against ``config/calendar.yaml``: dates added,
   removed, or renamed;
4. **write** (``--write``) the holidays block back into the YAML in the file's own
   compact style, preserving every comment elsewhere in the file.

The default is a read-only **check**: it reports drift and exits non-zero, so a
scheduled job can shout without ever editing the calendar.

Usage::

    uv run python scripts/refresh_calendar.py                 # check (network)
    uv run python scripts/refresh_calendar.py --write         # check and update the YAML
    uv run python scripts/refresh_calendar.py --input saved.json   # offline diff
    uv run python scripts/refresh_calendar.py --dry-run --json

Exit codes: 0 = in sync (or ``--write`` applied), 1 = drift found, 2 = the fetch or
the configuration failed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/refresh_calendar.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.config_loader import AnjaneyaConfig, ConfigError, load_config  # noqa: E402
from core.constants import get_timezone  # noqa: E402

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_CONFIG_ERROR = 2

#: NSE's trading-holiday master (segments: CM, FO, CD, CBM, ...).
DEFAULT_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
#: A plain GET here first so NSE sets the cookies the API request then needs.
DEFAULT_WARMUP_URL = "https://www.nseindia.com/"
DEFAULT_SEGMENT = "FO"
DEFAULT_TIMEOUT = 30.0
#: NSE serves the API only to a browser-looking client.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: The YAML block this script owns: ``holidays:`` and its indented (or blank) lines.
HOLIDAYS_BLOCK_RE = re.compile(r"^holidays:[ \t]*\n(?:[ \t].*\n|\n)*", re.MULTILINE)
#: The ``holidays_verified:`` flag line.
HOLIDAYS_VERIFIED_RE = re.compile(r"^holidays_verified:[ \t]*\w+[ \t]*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Holiday:
    """One fetched holiday, normalised to an ISO date."""

    day: date
    name: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/refresh_calendar.py",
        description="Reconcile config/calendar.yaml with NSE's trading-holiday master.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML configuration.")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Calendar year to reconcile. Default: the current year in the research timezone.",
    )
    parser.add_argument("--segment", default=DEFAULT_SEGMENT, help="NSE holiday segment (FO for F&O).")
    parser.add_argument("--url", default=DEFAULT_URL, help="NSE holiday-master endpoint.")
    parser.add_argument("--warmup-url", default=DEFAULT_WARMUP_URL, help="URL fetched first for cookies.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout, seconds.")
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help="Read a saved holiday-master JSON instead of fetching (offline diff/tests).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the holidays block in config/calendar.yaml from the fetched dates.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Where the raw JSON is recorded. Default: <repo>/docs/evidence.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do everything except write any file.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit the report as JSON.")
    return parser.parse_args(list(argv) if argv is not None else None)



def fetch_payload(*, url: str, warmup_url: str, timeout: float) -> dict[str, object]:
    """Fetch NSE's holiday master, warming up a cookie jar first."""
    import http.cookiejar

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    warmup = urllib.request.Request(warmup_url, headers={"User-Agent": USER_AGENT})
    with opener.open(warmup, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        response.read()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": warmup_url,
        },
    )
    with opener.open(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        return json.loads(response.read().decode("utf-8"))


def holidays_for_segment(payload: dict[str, object], segment: str, *, year: int) -> list[Holiday]:
    """Normalise the ``segment`` rows into sorted :class:`Holiday` objects for ``year``."""
    rows = payload.get(segment) or []
    if not isinstance(rows, list):
        return []
    out: dict[date, Holiday] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("tradingDate") or "").strip()
        name = str(row.get("description") or "").strip()
        if not raw or not name:
            continue
        try:
            day = datetime.strptime(raw, "%d-%b-%Y").date()
        except ValueError:
            continue
        if day.year != year:
            continue
        out[day] = Holiday(day=day, name=name)
    return [out[day] for day in sorted(out)]


def current_holidays(config: AnjaneyaConfig, *, year: int) -> list[Holiday]:
    """The holidays already configured for ``year``."""
    return [
        Holiday(day=entry.date, name=entry.name)
        for entry in config.calendar.holidays
        if entry.date.year == year
    ]


def diff_holidays(
    fetched: list[Holiday], configured: list[Holiday]
) -> dict[str, list[dict[str, str]]]:
    """Compare two holiday lists by date; report added, removed and renamed dates."""
    have = {item.day: item.name for item in fetched}
    got = {item.day: item.name for item in configured}
    added = [{"date": d.isoformat(), "name": have[d]} for d in sorted(have.keys() - got.keys())]
    removed = [{"date": d.isoformat(), "name": got[d]} for d in sorted(got.keys() - have.keys())]
    renamed = [
        {"date": d.isoformat(), "configured": got[d], "fetched": have[d]}
        for d in sorted(have.keys() & got.keys())
        if have[d] != got[d]
    ]
    return {"added": added, "removed": removed, "renamed": renamed}


def format_holidays_block(fetched: list[Holiday], *, year: int, fetched_on: date) -> str:
    """Render the ``holidays:`` YAML block in the file's own compact style."""
    lines = [
        "holidays:",
        f"  # -- {year}: NSE F&O trading holidays ({len(fetched)} dates, NSE's own descriptions) --",
        f"  # Refreshed by scripts/refresh_calendar.py from the NSE holiday master on "
        f"{fetched_on.isoformat()}.",
    ]
    for holiday in fetched:
        lines.append(
            f'  - {{ date: "{holiday.day.isoformat()}", name: "{holiday.name}", verified: true }}'
        )
    return "\n".join(lines) + "\n"


def rewrite_calendar(text: str, fetched: list[Holiday], *, year: int, fetched_on: date) -> str:
    """Return ``text`` with its ``holidays:`` block replaced and ``holidays_verified`` set true."""
    block = format_holidays_block(fetched, year=year, fetched_on=fetched_on)
    if HOLIDAYS_BLOCK_RE.search(text) is None:
        raise ValueError("config/calendar.yaml has no 'holidays:' block to replace")
    updated = HOLIDAYS_BLOCK_RE.sub(block + "\n", text, count=1)
    return HOLIDAYS_VERIFIED_RE.sub("holidays_verified: true", updated, count=1)


def evidence_path(evidence_dir: Path | None, *, year: int) -> Path:
    """Where the raw NSE response is recorded."""
    base = evidence_dir if evidence_dir is not None else REPO_ROOT / "docs" / "evidence"
    return base / f"nse_holidays_{year}.json"


def render(report: dict[str, object]) -> str:
    """Render the refresh report for an operator."""
    diff = report["diff"]
    lines = [
        f"ANJANEYA calendar refresh - {report['year']} ({report['segment']})",
        f"  source         : {report['source']}",
        f"  fetched        : {report['fetched_count']} holiday date(s)",
        f"  configured     : {report['configured_count']} holiday date(s)",
    ]
    for label, verb in (("added", "in NSE, missing from the YAML"),
                        ("removed", "in the YAML, not in NSE"),
                        ("renamed", "same date, different description")):
        rows = diff[label]
        lines.append(f"  {label:<15}: {len(rows)} ({verb})")
        for row in rows:
            detail = row.get("name") or f"{row.get('configured')} -> {row.get('fetched')}"
            lines.append(f"      {row['date']}  {detail}")
    if report["dry_run"]:
        lines.append("  mode           : DRY-RUN - nothing was written")
    else:
        lines.append(f"  evidence       : {report['evidence_path']}")
        lines.append("  calendar       : WRITTEN" if report["wrote_calendar"] else "  calendar       : unchanged")
    lines.append(f"  verdict        : {'DRIFT' if report['drift'] else 'IN SYNC'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Fetch, record, diff and optionally rewrite the calendar. Returns 0/1/2."""
    args = parse_args(argv)
    try:
        config = load_config(Path(args.config_dir) if args.config_dir else None)
    except ConfigError as exc:
        print(f"ANJANEYA calendar refresh: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    now = datetime.now(get_timezone(config.timezone_name))
    year = args.year or now.year
    try:
        if args.input:
            payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
            source = f"file:{args.input}"
        else:
            payload = fetch_payload(url=args.url, warmup_url=args.warmup_url, timeout=args.timeout)
            source = args.url
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        print(
            f"ANJANEYA calendar refresh: could not fetch {args.url}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR

    fetched = holidays_for_segment(payload, args.segment, year=year)
    if not fetched:
        print(
            f"ANJANEYA calendar refresh: NSE returned no {args.segment} holidays for {year}; "
            "nothing to reconcile (check the segment name and the year).",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR

    configured = current_holidays(config, year=year)
    diff = diff_holidays(fetched, configured)
    drift = bool(diff["added"] or diff["removed"] or diff["renamed"])
    ev_path = evidence_path(Path(args.evidence_dir) if args.evidence_dir else None, year=year)
    calendar_file = config.config_dir / "calendar.yaml"

    wrote_calendar = False
    if not args.dry_run:
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        ev_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if args.write:
            original = calendar_file.read_text(encoding="utf-8")
            calendar_file.write_text(
                rewrite_calendar(original, fetched, year=year, fetched_on=now.date()),
                encoding="utf-8",
            )
            wrote_calendar = True

    report: dict[str, object] = {
        "year": year,
        "segment": args.segment,
        "source": source,
        "fetched_count": len(fetched),
        "configured_count": len(configured),
        "diff": diff,
        "drift": drift,
        "dry_run": bool(args.dry_run),
        "evidence_path": str(ev_path),
        "wrote_calendar": wrote_calendar,
    }
    print(json.dumps(report, indent=2) if args.as_json else render(report))

    if wrote_calendar:
        return EXIT_OK
    return EXIT_DRIFT if drift else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
