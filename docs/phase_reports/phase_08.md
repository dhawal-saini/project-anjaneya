# Phase 8 report — the calendar, reconciled (the one input the exchange could settle)

**Status: Stage 8.1 (the 2026 NSE trading-holiday list), stage 8.2 (a special session outranks
the weekend and holiday defaults) and stage 8.3 (the live proof, split so a closed-market run can
still pass) complete.**

Phase 7 closed the master prompt's phase list: phase 24, the decision layer, was the last *named*
phase, and `--mode accumulation` (stage 6.1) had already shipped the report on itself. So this
phase has no spec number and no master-prompt line behind it. It exists because of what the
repository says about itself when you ask it where it actually stands:

```
db/anjaneeya.duckdb          schema only
  ticks                      0        candles     0        features   0
  labels                     0        option_chain_snapshots 0      trades 0
  data_coverage              0
```

The pipeline is complete and has never seen a single real tick. Every number it can produce today
is synthetic, and the next real work is *capture* — which needs credentials, a live session and
market hours. There is exactly one input that could be settled without any of those, and the
system nagged about it on every single invocation:

```
calendar: 8 holiday date(s) are PROVISIONAL (holidays_verified: false).
          Reconcile them with the official NSE circular before relying on them.
```

That is `core/app_bootstrap.py` warning that the calendar it protects every session with was a
**seed** — and the runbook's §1.3 is explicit about what a wrong date costs: *"a holiday that is
really a trading day makes the session exit `NO_SESSION`, and the day's ticks are lost forever."*
A calendar cannot be captured later; it either knows a date in advance or the session is gone.
This phase reconciles it against the exchange's own published list.

## 1. Files created

* `docs/evidence/nse_holidays_2026.json` — the raw, unedited response body from
  `https://www.nseindia.com/api/holiday-master?type=trading`, fetched 2026-10-01. Committed
  because the endpoint serves **one year at a time**: by the time anyone re-checks these dates it
  will be serving 2027, and without a stored copy the only surviving record of what NSE actually
  said would be this repository's own summary of it. It is evidence, not configuration: nothing
  reads it at runtime, and `tests/unit/test_market_calendar.py` only reads it to check that the
  config still agrees with it.
* `docs/phase_reports/phase_08.md` — this report.

## 2. Files modified

* `config/calendar.yaml` — the `holidays:` block replaced with NSE's official 2026 F&O list (20
  dates, NSE's own descriptions), `holidays_verified: true`, and a provenance comment naming the
  endpoint, the fetch date, the second confirming source and the muhurat caveat. The stale
  `half_day_dates: []` example (`["2026-11-09"]`) was corrected: that date is not a half day, it
  was simply one of the seed's wrong ones.
* `tests/unit/test_market_calendar.py` — 8 new tests (§5) that pin the reconciliation and the
  session precedence.
* `core/market_calendar/nse_calendar.py` — `is_trading_day` now gives an explicitly configured
  special session precedence over the weekend and holiday tests, and `is_special_session` /
  `special_session` were added so a caller can tell a declared session from an ordinary one. No
  change for a configuration without `special_sessions`.
* `scripts/check_calendar.py` — a declared session is tagged `[special session]` in the day list,
  so a Sunday that renders `TRADING` is legible rather than alarming.
* `docs/runbook.md` §1.3 — rewritten from *"the list is provisional; here is how to clear it"* to
  *"the list is reconciled for 2026, here is how it was done, here is the procedure for next
  year"*, including the endpoint, the F&O segment, and the instrument-dump cross-check.
* `README.md` — the "Honest caveats" bullet that said the dates are provisional now says what they
  are, how they were checked, and that the claim covers **2026 only**; a second bullet states the
  muhurat situation and why no window is shipped; the status line and the status table name this
  phase; and the phase-report index gained this file.

## 3. What was implemented

### The seed was wrong, and wrong in the expensive direction

Not "provisional" — *incorrect*. Diffing the seed against NSE's F&O list, by date and weekday:

| Seed behaviour | Dates | Consequence |
| --- | --- | --- |
| **Closed a real trading day** | 2026-03-04 (Wed), 2026-03-20 (Fri), 2026-05-27 (Wed), 2026-08-26 (Wed), 2026-11-09 (Mon) | `NO_SESSION`; the day's ticks would have been lost forever |
| **Left a real holiday open** | 2026-01-15 (Thu), 2026-03-03 (Tue), 2026-03-26 (Thu), 2026-03-31 (Tue), 2026-05-28 (Thu), 2026-06-26 (Fri), 2026-09-14 (Mon) | capture runs on a closed exchange, finds nothing, exits `0` |
| **Correct** | 2026-01-26, 2026-04-03, 2026-04-14, 2026-05-01, 2026-08-15, 2026-10-02, 2026-10-20, 2026-11-10, 2026-11-24, 2026-12-25 | 10 of 15 |

Twelve of the fifteen seed entries were wrong on a weekday. Four of them were off by exactly one
day (Holi, Id-Ul-Fitr, Bakri Id, Diwali Laxmi Pujan), one by nineteen (Ganesh Chaturthi), and seven
official holidays were missing outright. The seed's own comments classified them "PROVISIONAL —
verify against NSE circular", which is why they were never trusted; what the reconciliation showed
is that "provisional" was understating it.

### The method: fetch, do not re-derive

`fetch_web_content` timed out against `nseindia.com` (the site refuses non-browser clients), but
`curl` with browser headers reached the JSON API directly in one request. That matters: the list in
the config is **NSE's own** dates and strings, copied — not a second-hand festival table and not a
re-derivation from a panchang. The next annual check is therefore a diff against the same endpoint,
which is a job that costs one command, rather than a re-research of when Diwali falls.

### The confirmations: a second source that never saw the holiday list

Two of the twenty dates were already provable from data **already in the repository**. NIFTY weekly
expiry is Tuesday. `data/raw/instruments/NFO_2026-09-30.json` — the exchange's own contract dump,
fetched 2026-09-30, entirely independent of the holiday API — lists
```
2026-10-06 Tue   2026-10-13 Tue   2026-10-19 Mon   2026-10-27 Tue
2026-11-03 Tue   2026-11-23 Mon   2026-12-29 Tue   ...
```

Two Mondays, where every other near term expiry is a Tuesday: **2026-10-19** is exactly the
`previous_trading_day` roll for the 2026-10-20 Tuesday, and **2026-11-23** the same for
2026-11-24. A Tuesday NSE does not close produces no such roll, so the exchange's own contract
listing independently confirms Dussehra and Guru Nanak Jayanti. Those two rolls are now asserted
in `tests/unit/test_market_calendar.py`, so the cross-check is a test and not a paragraph.

### The muhurat problem, and the precedence that fixes it

NSE lists 2026-11-08 as **`Diwali Laxmi Pujan*`**. The asterisk marks a date that carries a
*muhurat* trading session. It is also a **Sunday**, and:

```python
def is_trading_day(self, day: date) -> bool:
    if day in self._special:            # an explicit declaration wins
        return True
    return not self.is_weekend(day) and not self.is_holiday(day)
```

Two things were wrong, and only one of them was the calendar's fault. The **first** was a real
defect in `NseCalendar`: `sessions.special_sessions` could name a window (`session_window` does
return it — precedence there is *special > half-day > regular*) but could not make the day a
trading day, because the weekend test and the holiday entry both vetoed it afterwards. That is not
a corner case. A muhurat session is held on a date NSE *also* lists as a trading holiday, and falls
on a Saturday or Sunday in some years, so `not weekend and not holiday` can never express one — the
single session an operator had explicitly declared was exactly the one the calendar would deny, and
a capture would exit `NO_SESSION` on a day the exchange traded. **An explicit declaration now
outranks the defaults**: if a date is in `special_sessions`, it is a trading day, period, and the
weekend and holiday *facts* about it stay visible (`is_weekend` and `holiday_name` still answer
truthfully) while they stop vetoing. `is_special_session` and `special_session` were added so a
caller can tell a declared session from an ordinary one, and `check_calendar.py` tags such a day
`[special session]`, because a Sunday rendering as `TRADING` is otherwise indistinguishable from a
bug.

The **second** problem is not the code's to solve. NSE announces the muhurat window in a separate
circular, and no figure for 2026-11-08 has been published that this stage could verify — so none
was written. `special_sessions` is still empty, the date is still reported closed, and the gap is
now a **data** gap: one circular, one config line. That distinction is the whole point of the
change. "The calendar cannot represent this" is a defect someone would have had to find through a
missing session; "the window is not published yet" is a fact with a one-line remedy. Both are
tested, and the config comment says which is which at the point a reader meets the asterisk.

### Scope, stated rather than implied

`holidays_verified: true` is a claim about **2026 only**. The endpoint serves one year at a time and
returned no 2027 dates, so the flag now means "the dates in this file have been checked", not "the
calendar is safe forever". The config comment, the runbook and the README each say so, and each
says the flag must go back to `false` before 2027-01-01 — at which point the seed-versus-truth
problem returns, but this time with a standing warning rather than silence. The annual
reconciliation was already in the runbook's periodic table; it now has a procedure one command
long.

## 4. Configuration changed

```yaml
holidays_verified: true          # was false
holidays:                        # 20 entries, was 15
  - { date: "2026-01-15", name: "Municipal Corporation Election - Maharashtra", verified: true }
  - { date: "2026-01-26", name: "Republic Day", verified: true }
  ...
  - { date: "2026-12-25", name: "Christmas", verified: true }
```

Every entry is `verified: true` and every name is NSE's own description verbatim, including
`Diwali Laxmi Pujan*` with the asterisk — the file is a faithful copy of the source, so a future
diff shows real disagreements instead of cosmetic ones. Three of the twenty are weekend dates
(2026-02-15, 2026-03-21, 2026-08-15); they are kept for the same reason, and are inert because
`is_trading_day` already excludes weekends.

The config hash moved with the change: `08911c239062` -> `4ba62ff60b3c`. Every run manifest records
it, so artefacts produced before and after this phase are distinguishable by hash, and the number
moving is itself the evidence that the calendar a result rested on is the one that was reconciled.
Stage 8.2 then moved it again, to `d3f096805fa5`: the code change altered no shipped *value*, but the
rewritten muhurat comment changed the raw bytes the hash is taken over — which is the intended
behaviour (the hash identifies the committed file, comments included), and worth stating so the
second move is not mistaken for a second change of dates.

## 4b. Configuration that deliberately did **not** change

```yaml
sessions:
  special_sessions: []           # still empty - see §3 for why
```

The precedence change is in `NseCalendar`, not in this file. `special_sessions` stays empty because
NSE has not published the 2026-11-08 muhurat window; writing one in would have made a comment-free
config look authoritative about a timing nobody verified. The value of the stage is that the *next*
line of configuration is now sufficient, where before no line was.

## 5. Tests created

Eight, in `tests/unit/test_market_calendar.py` — which uses the *shipped* config, making it the
regression guard on `config/calendar.yaml`:

* `test_the_shipped_holiday_list_matches_the_nse_fo_2026_list` — the reconciliation itself, pinned
  as a hard-coded set of 20 dates. When the annual job is done the new list has to be written in
  two places, so it cannot be "reconciled" by editing the config alone.
* `test_the_reconciliation_record_matches_the_committed_upstream_copy` — reads
  `docs/evidence/nse_holidays_2026.json` and compares **both dates and descriptions**. Test, config
  and evidence therefore agree with a stored primary source, not with one another.
* `test_the_reconciled_list_is_marked_verified` — `holidays_verified is True` *and*
  `unverified_holidays == []`, so the flag cannot be flipped without the entries backing it.
* `test_every_reconciled_date_closes_the_shipped_calendar` — each of the 20 is a non-trading day
  with a name, weekends included.
* `test_both_dump_confirmed_rolls_are_configured` — the two independent confirmations above
  (2026-10-19 for Dussehra, 2026-11-23 for Guru Nanak), asserted against the calendar engine.
* `test_a_special_session_opens_a_day_the_defaults_would_close` — a configured special session on
  a Sunday that is *also* a holiday: the day trades, on the declared window, `is_market_open` is
  true inside it and false outside — while `is_weekend` and `holiday_name` still report the truth.
* `test_a_special_session_opens_a_weekday_holiday_too` — the same rule where only the *holiday*
  veto applied (pinned on 2025-10-21, a Tuesday, so the rule is not bound to 2026's Sunday).
* `test_the_2026_muhurat_window_is_ready_but_not_invented` — the shipped config has no entry, so
  2026-11-08 is still closed. The gap is now a missing *timing*, not a missing capability.

## 6. Test results (actual output)

```
$ python -m pytest tests/unit/test_market_calendar.py -q
45 passed in 8.05s          # was 37 before this phase; +8

$ python -m pytest tests/unit -q
1666 passed in 183.57s (0:03:03)   # was 1658; exactly +8, so nothing else moved

$ ruff check .
All checks passed!
```

The suite total moving by exactly the eight new tests is the point: no other test depended on the
seed's dates, which is what the file's own header claims ("these tests use the *shipped*
configuration, so they also act as a regression guard") and what `test_check_calendar.py` means
when it says its fixtures "do not depend on the shipped holiday list being provisional".

`scripts/check_calendar.py` — the operator's own tool, and the thing the flag existed for:

```
ANJANEYA calendar check - as of 2026-10-01 (Thursday): trading day
  timezone        : Asia/Kolkata
  regular session : 09:15-15:30
  weekly expiry   : Tuesday (roll: previous_trading_day)
  holidays        : 20 configured, 0 PROVISIONAL
  expiries        : weekly 2026-10-06 (next 2026-10-13), monthly 2026-10-27

days 2026-10-01 -> 2026-10-30: 20 trading of 30 calendar day(s)
  2026-10-19  Mon  TRADING 09:15-15:30                                   [weekly expiry]
  2026-10-20  Tue  CLOSED  holiday: Dussehra (verified)
  2026-10-27  Tue  TRADING 09:15-15:30                                   [weekly expiry, monthly expiry]

risk: none - no provisional holiday falls between 2026-10-01 and 2026-10-11
verdict: OK
EXIT=0
```

`20 configured, 0 PROVISIONAL`, exit `0`, and the Dussehra roll rendering as the Monday expiry the
contract dump listed. The bootstrap advisory is gone from every app's startup:

```
$ python -m apps.anjaneeya_ai.run --mode decide --dry-run
... config=4ba62ff60b3c schema_version=1 timezone=Asia/Kolkata dry_run=True
ANJANEYA-AI: no model artefact to decide with ... (--mode train) or pass --model PATH.
... ANJANEYA anjaneeya_ai finished: status=NO_MODEL exit=0
```

No holiday note, no warning, still `exit 0`. The one input that could be settled without a market
session is settled.

## 7. Example output (small, real)

The whole reconciliation, as it happened:

```
$ curl -H 'User-Agent: Mozilla/5.0 ...' \
      'https://www.nseindia.com/api/holiday-master?type=trading' -o nse_hol.json
HTTP 200 bytes 34022

$ python -c "import json; d=json.load(open('nse_hol.json')); print(list(d.keys()))"
['CBM', 'CD', 'CM', 'CMOT', 'COM', 'EGR', 'FO', 'IRD', 'MF', 'NDM', 'NTRP', 'SLBS']

$ python -c "...print the FO segment..."
2026-01-15 Thu  Municipal Corporation Election - Maharashtra
2026-01-26 Mon  Republic Day
2026-02-15 Sun  Mahashivratri                                    WEEKEND
...
2026-11-08 Sun  Diwali Laxmi Pujan*                              WEEKEND
2026-11-10 Tue  Diwali-Balipratipada
2026-11-24 Tue  Prakash Gurpurb Sri Guru Nanak Dev
2026-12-25 Fri  Christmas
count 20
```

The `FO` segment was the one taken: it is the segment whose holidays govern the futures and options
market, and it is what ANJANEYA trades. `CD`, `IRD`, `EGR` and `COM` disagree with it — `CD` has 24
dates, including bank-only closures such as Annual Bank Closing, Chatrapati Shivaji Maharaj Jayanti
and Id-E-Milad. Taking the wrong segment would have closed days the F&O market trades, which is the
same failure as the seed's, only better sourced.

The muhurat precedence, demonstrated directly on the date NSE asterisks (a scratch calendar that
*does* carry a window, since the shipped one does not):

```
$ python -c "from ... import ...; cal = NseCalendar(<2026-11-08 in special_sessions>)"
date            2026-11-08 Sunday
is_weekend      True
is_holiday      True
holiday_name    Diwali Laxmi Pujan*
special_session True (datetime.time(18, 15), datetime.time(19, 15))
is_trading_day  True
session_window  (datetime.time(18, 15), datetime.time(19, 15))
```

The day stays a Sunday and stays NSE's holiday — those facts are not rewritten — and it trades
anyway, on the declared window, because an explicit declaration outranks the defaults. With the
shipped `special_sessions: []` the same date answers `is_trading_day ... False`, which is exactly
the distance between the calendar's capability and the one missing fact.

The operator's own tool, run over a copy of the config that *does* declare the session:

```
$ python scripts/check_calendar.py --config-dir <scratch> --as-of 2026-11-06 --days 5
days 2026-11-06 -> 2026-11-10: 3 trading of 5 calendar day(s)
  2026-11-06  Fri  TRADING 09:15-15:30
  2026-11-07  Sat  CLOSED  weekend
  2026-11-08  Sun  TRADING 18:15-19:15                                   [special session]
  2026-11-09  Mon  TRADING 09:15-15:30                                   [weekly expiry]
  2026-11-10  Tue  CLOSED  holiday: Diwali-Balipratipada (verified)
verdict: OK
EXIT=0
```

A Sunday on the trading list is the one line that would otherwise look like a bug, so it says
`[special session]` in as many words. The shipped config produces the same days with
`2026-11-08  Sun  CLOSED  holiday: Diwali Laxmi Pujan* (verified)` instead.

## 8. Known limitations

* **The list covers 2026 and nothing else.** 2027 is absent because the endpoint serves one year at
  a time. A 2027 weekday is therefore treated as a trading day until the list is extended, and
  `holidays_verified: true` says nothing about it. The flag must be reset before 2027-01-01.
* **The muhurat session on 2026-11-08 has no verified window** (see §3). The *capability* gap is
  closed — a `special_sessions` entry now opens a day the weekend and holiday defaults would close —
  but NSE announces muhurat timings in a separate circular and no 2026 figure was verified here, so
  `special_sessions` is still empty and the date is still reported closed. One circular plus one
  config line makes it capturable; until then the calendar says what it knows.
* **Weekend dates are listed as holidays.** Faithful to NSE, inert to the engine, and a reminder
  that the file is a copy of the source rather than a normalised derivative.
* **No back-fill of earlier years.** The seed never had 2025 dates either; nothing in the repository
  trades 2025, so the year the data would cover is the year the file describes.
* **The reconciliation is a point-in-time check.** NSE can amend the list mid-year — it added the
  2026-01-15 Maharashtra election closure, an event no almanac predicts. Nothing polls for that;
  `check_calendar.py` in the morning routine is what would catch a change, and only for dates it is
  asked about.
* **This fixes the calendar, not the data.** Everything downstream still reports PRELIMINARY,
  because the database is still empty. A correct calendar cannot make an untrained model right.

## 9. Assumptions made

* **The F&O segment is the right one.** ANJANEYA trades NIFTY options on NFO, and `FO` is that
  segment's list. `CM` agrees with it on all 20 dates for 2026 — consistent with a segment whose
  holidays are a superset — but `FO` is what was copied.
* **The endpoint is the official list.** It is served by `nseindia.com` itself, not a broker or a
  news site; the response carries segment keys, `Sr_no`s and session columns that a derived table
  would not have.
* **A roll is evidence.** Assuming NSE's expiry weekday is Tuesday (already independently confirmed
  against the same instrument dump in Phase 3), a Monday expiry can only be a roll, and a roll only
  happens for a closure. That is what lets the dump confirm two dates without touching the API.
* **The evidence file is worth its 34 KB.** It is the only copy of what NSE said that will still
  exist once the endpoint has moved on.
* **"Verified" means checked, not correct forever.** The flag records that someone compared this
  list with the exchange's on 2026-10-01; it cannot record that NSE will not amend it later.
* **A `special_sessions` entry is a declaration, not a hint.** Giving it precedence over the weekend
  and holiday defaults assumes the operator only writes a window NSE has published. That is exactly
  why the entry for 2026-11-08 is absent rather than guessed: the precedence makes a wrong window
  dangerous (it opens a day the market is shut), so it is entered only from a circular, and the day
  the window is not known the calendar stays closed.
* **English/IST/NIFTY assumptions** are inherited unchanged from every earlier phase.

## 10. Next phase will implement

Nothing named remains, and this phase added none — it removed the last *known false input*. What is
left is the thing no phase can do from here: **capture real sessions.** The calendar was the one
input that could be settled without market hours, and it is now settled, so the next step is a
`--mode capture` run on a real trading day, then the daily pipeline, then `--mode accumulation`
until `ml_training_min_days` opens.

Two caveats travel with that, both recorded above rather than left implicit: the holiday list needs
re-checking before 2027, and the muhurat session on 2026-11-08 cannot be captured until NSE
publishes its window (the calendar is now *able* to hold it — see §3 — so that is a data gap with a
one-line remedy, not a design limit). Everything else in the repository is waiting on data, and
this phase was the last thing that could be fixed without it.

---

## 11. Stage 8.3 — the live proof, split so an evening run can pass

### The defect

`tests/live/test_live_capture_smoke.py` was **one** test that ran six steps in order and called
`pytest.skip` at step 6 when the session had already closed:

```python
assert summary.status in {"SUCCESS", "PARTIAL", "NO_SESSION"}, summary.describe()
if summary.status == "NO_SESSION":
    pytest.skip(f"outside the trading session: {summary.reason}")
```

Steps 1–5 — resolve the token, authenticate, download the ~10 MB instrument dump, write it to
DuckDB, anchor the ATM on a real price, resolve the universe — had **already run and passed** by the
time that skip fired. pytest does not partially credit a skipped test, so a run at 19:00 reported

```
1 skipped
```

which is byte-for-byte what a machine with *no credentials at all* reports. The one signal an
operator checking the broker wiring after the close could have had was thrown away by a skip that
was about the next step, not the ones behind it. Recovering it is a testing problem, not a broker
problem, and it is the difference between "the live path is unverified" and "the live path is
verified except for ticks".

### The fix

Three tests, and one module-scoped fixture so the expensive shared setup still happens exactly once:

| Test | Asserts | Needs |
| --- | --- | --- |
| `test_the_broker_path_authenticates_and_stores_the_master` | `profile()` returns a user id; the dump has >100 instruments; the writer's row count equals `SELECT count(*) FROM instruments` | a live token — **any hour** |
| `test_the_universe_resolves_from_a_real_price` | the price is > 0 and reports its source; while the market is open the source must be `rest-ltp` (never a stale dump row or a stored bar); the universe is not empty | a live token + one index quote |
| `test_a_one_minute_session_is_stored_end_to_end` | one poll row, one snapshot on disk, the heartbeat says `STOPPED`, and a real tick in DuckDB **while the market is open** | a live token + the exchange open |

The fixture (`connected`) authenticates and downloads the master once for all three: the dump is
~10 MB from a rate-limited endpoint, and three downloads to prove one connection would be waste.
Its failure *is* the answer to "does the broker path work?" — the traceback names the cause, and an
expired token says so in its own words.

Test 2 skips only on the one condition it cannot control, and narrowly: a `SessionError` from
`reference_price` means the market is closed, the dump carries no index price (Kite reports `0` for
index rows) and the scratch database has no `index_history` yet. That is an environment
limitation — it says so, with the underlying message, rather than reporting a broken wire.

### Verification

```
$ ANJANEYA_RUN_LIVE=1 python -m pytest tests/live --collect-only -q
tests/live/test_live_capture_smoke.py::test_the_broker_path_authenticates_and_stores_the_master
tests/live/test_live_capture_smoke.py::test_the_universe_resolves_from_a_real_price
tests/live/test_live_capture_smoke.py::test_a_one_minute_session_is_stored_end_to_end

3 tests collected in 3.13s

$ python -m ruff check tests/live/test_live_capture_smoke.py
All checks passed!
```

Collection is the honest limit of what can be verified from here: *running* the three tests needs a
live Kite token, and the cached one expired at 2026-10-01T06:00:00+05:30, the daily cutoff. Getting
past that is one command and a browser — `python scripts/daily_login.py` or
`python scripts/login_and_start.py` — after which the first test is a genuine pass at any hour.
That is exactly the gap this stage closes for the operator, so it is stated rather than worked
around.

### What moved

| File | Change |
| --- | --- |
| `tests/live/test_live_capture_smoke.py` | one test → three, plus the `connected` fixture and a module docstring that explains the split and the closed-market semantics |
| `docs/runbook.md` | new §1.7: the command, the opt-in rule, and the per-test "after the close" table |
| `README.md` | the Testing section now says the live suite is three tests and that the broker half passes at any hour |
| `docs/phase_reports/phase_08.md` | this section, and the header status line |

No production code changed: `apps/` and `core/` are untouched, so the offline suite is unaffected —
the default run still reports the single module-level skip it always did.


