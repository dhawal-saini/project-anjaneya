# ANJANEYA — operator runbook

Everything an operator has to do to turn a trading day into stored, validated data and a
backtest, in order, with the exact commands and what to do when one of them fails.

The two rules that matter most:

* **A missed session cannot be recovered.** Zerodha serves no historical expired-option
  data, so a day whose ticks were not captured is a day the study will never have. The
  morning routine below exists to make that impossible.
* **DuckDB allows one writer.** Never run the capture app and the engine/AI apps at the same
  time. The pipeline and the capture session are separate windows of the day (see §2).

Tested on Windows 11 (PowerShell 7 / Git Bash) with Python 3.13. Commands are shown for both
shell families; `scripts/*.ps1` and `scripts/*.sh` are equivalent wrappers.

---

## 0. The short version

| When | Command | Why |
| --- | --- | --- |
| once (Day 0) | `python -m apps.anjaneeya_capture.run --mode instruments` | the contract master every strike and expiry is resolved against (~4 min and 13 MB the first time, then cached per day) |
| once (Day 0) | `.\scripts\run_backfill.ps1` | 730 days (~2 years) of NIFTY index OHLCV: `enrich-greeks` reads it first and the pre-open plan anchors on it |
| once (Day 0) | `python -m apps.anjaneeya_capture.run --mode backfill-options` | optional: seed candles for the option contracts Kite lists *today* (it never serves expired options) |
| once (Day 0), Windows | `.\scripts\schedule_tasks.ps1 install` | register the weekday Morning + PostClose tasks (calendar-gated) |
| every morning, before 09:00 | `python scripts/run_morning.py` | **the morning routine in one guarded command**: calendar → universe gate → login → capture |
| every morning (if you schedule it) | *(the Morning task runs the same command)* | `\Anjaneya\Anjaneya-Morning` fires at 09:05, Mon-Fri |
| every evening, after 15:35 | `.\scripts\run_daily_pipeline.ps1` | candles → greeks → features → labels → validate → dataset → train → backtest |
| weekly | `.\scripts\run_weekly.ps1` | the daily chain **plus** report → explain → importance → charts → accumulation |
| any time something looks wrong | `python scripts/doctor.py` | readiness: login route, token expiry, dump age, database writable, scheduler state |
| monthly, or on an NSE circular | `python scripts/refresh_calendar.py [--write]` | reconcile `config/calendar.yaml` with NSE's holiday master |

**The daily loop is now two commands** - `python scripts/run_morning.py` in the morning and
`.\scripts\run_daily_pipeline.ps1` after the close - plus `.\scripts\schedule_tasks.ps1 install`
once to have Windows run both through `scripts/run_scheduled.ps1`, which gates every stage on
`check_calendar.py --require-trading-day --within 0` so a holiday is a clean no-op.

**Or use the front door.** `scripts/anj.py` puts the whole system behind three verbs. It
re-implements nothing - each verb shells out to exactly the scripts in the table above - so
the two commands a day become:

| When | Command |
| --- | --- |
| once (Day 0) | `uv run python scripts/anj.py history --years 2` — instruments + 2y NIFTY index + option history, in order |
| every trading day | `uv run python scripts/anj.py day` — the guarded session, then candles → greeks → features → labels → validate → dataset, then the suggestion |
| every evening | `uv run python scripts/anj.py learn` — dataset → train → backtest → report → explain → importance → charts → accumulation |

`day` stops at the dataset on purpose (`--skip train,backtest`): a model trained before the
close is trained on an incomplete day, which is why training is the evening's own command.
`day --no-capture` runs the post-close math alone (a holiday, or after the fact),
`day --skip-postclose` stops after the session, and `--dry-run` is accepted by every verb to
plan each child without writing anything. Exit codes are the shared vocabulary: `0` ok, `1` a
step failed, `2` configuration, `4` the accumulation gate refused (pass `--force` to `learn`
to train anyway). The individual commands below all still work and are unchanged.

**In a browser (optional).** `uv run streamlit run scripts/anj_ui.py` opens a console over the
same facade: a token card with a TOTP refresh, `history` (`Plan` = `--dry-run`, `Fetch` = live),
the first/last 10 rows of any table (read-only, handle closed per query), the `day`/`learn`
buttons with switches that become `learn`'s `--skip`, and the newest reports and charts. Install
it with `uv sync --extra ui --extra dev`; the extra is optional so the capture session never
needs a web framework. It refuses a second job while one runs - DuckDB has a single writer - and
never renders a token.

`run_morning.py` exists to make the two silent losses in §1.4 impossible: on a non-trading day
it prints `NO_SESSION` and exits `0` **without touching the broker**, and when the option
universe resolves to **zero** contracts it exits `4` **without starting the session** (a session
started that way records the index and no options, and still exits `0`). Exit codes: `0` ok, `1`
the login or the capture failed, `2` configuration/credentials, `4` the universe was empty.
| weekly (or after any backtest) | `python -m apps.anjaneeya_ai.run --mode report` | spec 24's ten answers over the newest ledger (JSON + Markdown + CSV) |
| weekly (after a training run) | `python -m apps.anjaneeya_ai.run --mode explain` | what the newest model was looking at, prediction by prediction (JSON + Markdown + CSV) |
| weekly (after a training run) | `python -m apps.anjaneeya_ai.run --mode importance` | which inputs actually carried the information, feature by feature and source by source (JSON + Markdown + CSV) |
| weekly (after a backtest and an importance run) | `python -m apps.anjaneeya_ai.run --mode charts` | spec 28's figures: the P&L heatmaps, the CE/PE curves, regime performance, the importance bars and the NIFTY / option price charts (one image each + an index) |
| any time something looks wrong | `python scripts/health_check.py` | configuration, calendar, database, heartbeat, data volume |

---

## 1. Day 0 — first-time setup

### 1.1 Environment

```powershell
uv sync                  # or: python -m venv .venv; .venv\Scripts\pip install -e .
.\.venv\Scripts\python.exe -c "import duckdb, lightgbm; print('ok')"
```

The wrappers pick the interpreter in this order: `ANJANEYA_PYTHON`, `.venv`, `uv run python`,
then a bare `python`. If `import duckdb` fails, every other command in this runbook will fail
too — fix it here, not later.

### 1.2 Credentials

```powershell
Copy-Item .env.example .env      # then edit .env
```

Five variables are needed, and nothing else is:

```
ZERODHA_API_KEY=...        # Kite Connect app key
ZERODHA_API_SECRET=...     # Kite Connect app secret
ZERODHA_USER_ID=...        # Zerodha login id
ZERODHA_PASSWORD=...       # Zerodha login password
ZERODHA_TOTP_KEY=...       # base32 TOTP secret (the "authenticator" key, not a code)
```

`ZERODHA_TOTP_KEY` is only used by the *automated* login. Without it (or when the broker
blocks TOTP automation) the browser login still works — it needs nothing but the API key and
secret. `ZERODHA_REDIRECT_URL` is optional: set it to a loopback address **and register the
same URL in the Kite app console** to let the login catch the `request_token` itself.

Real environment variables always win over the file, and no value is ever printed — only
masked tails.

Prove it, once:

```powershell
python scripts/daily_login.py --json
```

Exit `0` means a token is cached under `data/raw/.tokens/zerodha_token.json` and `profile()`
confirmed it. Exit `2` means credentials are missing or unreadable; exit `1` means the broker
refused the login. If TOTP automation is blocked, the fallback is manual:

```powershell
python scripts/daily_login.py --request-token <request_token from the redirect URL>
```

Better, do the manual part *once* rather than every morning: register a loopback redirect URL
in the Kite app console (redirect URLs are configured once, at developers.kite.trade), put the
same URL in `.env` as `ZERODHA_REDIRECT_URL` (e.g. `http://127.0.0.1:8444/`), and from then on
one command logs in and starts the capture (§2.1):

```powershell
python scripts/login_and_start.py
```

### 1.3 The calendar — the step that protects the capture

`config/calendar.yaml` **ships a reconciled NSE holiday list for 2026**
(`holidays_verified: true`, every entry `verified: true`), checked on 2026-10-01 against NSE's
own `holiday-master` endpoint, F&O segment. It matters more than any other config here: **a
holiday that is really a trading day makes the session exit `NO_SESSION`, and the day's ticks
are lost forever** — while a trading day mistaken for a holiday is worse still, because the
capture runs, finds nothing and exits `0`. Both failures are silent.

```powershell
python scripts/check_calendar.py --days 30          # 0 = clear, 1 = risky, 2 = config error
python scripts/check_calendar.py --json
```

The script names every provisional holiday inside the risk window (`--within`, default 10
days), the current/next weekly expiry, the monthly expiry and the session window for each
day. `holidays_verified: true` covers **2026 only** — it must be set back to `false` (or the
list extended) before 2027-01-01, when these entries stop covering the sessions being traded.

To (re-)reconcile, for each date in question:

1. check the date against NSE's official list — the endpoint the shipped list was taken from
   is `https://www.nseindia.com/api/holiday-master?type=trading`; take the **`FO`** segment,
   which is what ANJANEYA trades,
2. edit `config/calendar.yaml`: replace the entry with NSE's own date and description, and set
   `verified: true`,
3. when every date is confirmed, set `holidays_verified: true` at the top of the file,
4. re-run `tests/unit/test_market_calendar.py`: the expected list is pinned there too, so a
   config that disagrees with the record fails rather than passing quietly.

A second, independent source is worth using: `data/raw/instruments/NFO_<date>.json`. Every
NIFTY expiry in that dump is a Tuesday, and a **Monday** expiry is a `previous_trading_day`
roll — the exchange telling you the Tuesday after it was closed (2026-10-19 for Dussehra,
2026-11-23 for Guru Nanak). A date NSE does not close produces no such roll.

One date does not fit the open/closed rule at all: **2026-11-08, `Diwali Laxmi Pujan*`**, the
asterisk marking a *muhurat* evening session, on a Sunday. An entry in
`sessions.special_sessions` makes that day a trading day regardless of the weekend and the
holiday list — an explicit declaration outranks the defaults, which is the only way a muhurat
session can ever be represented, because NSE also lists the date as a holiday and in most years
it falls midweek. One is deliberately **not** shipped: NSE publishes the muhurat window in a
separate circular, and no 2026 timing had been verified when the list was reconciled, so the day
is still reported closed. When the circular is out, add it and the day becomes capturable:

```yaml
special_sessions:
  - { date: "2026-11-08", start: "<announced start>", end: "<announced end>", name: "Muhurat trading" }
```

The capture app also refuses to be quiet about this: a live `--mode capture` session prints a
warning for every provisional holiday within 7 days, before it does anything else. With the
2026 list reconciled that warning is now silent, which is the point — it will speak again the
moment the list goes stale.

### 1.4 The universe — decide **before** the first capture

`config/option_universe.yaml` decides what ANJANEYA subscribes to, and therefore what data
can exist later:

```yaml
strikes_offset: [-200, -100, 0, 100, 200]           # ATM ±200 in 100-point steps
expiries: [current_week, next_week, current_month]  # weekly rollover + the monthly cycle
```

Both limit the *ticks* **and** the option-chain snapshots (the poller polls the subscribed
symbols, not the full chain), so a strike that is not in this list is a strike the study will
never have for any past day. Widening is cheap from here and impossible afterwards; narrowing
is a filter you can apply at analysis time instead. **This was already widened once**: Phase 9
took it from `[-100, 0, 100]` and two expiries to the five strikes and three expiries above, so
the first capture subscribes ~30 option contracts. If the research question needs still more
context, change it **now**, in one edit, before the first session:

```yaml
strikes_offset: [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]
expiries:
  - current_week
  - next_week
  - current_month
  - next_month        # the following monthly cycle as well
```

Then look at the plan before committing to it for a whole session:

```powershell
python -m apps.anjaneeya_capture.run --mode capture --dry-run
```

The plan prints the exact symbols, strikes and expiries that would be subscribed, plus the
request budget. Every other step in this runbook is independent of that choice.

**Read two lines of that output every time.** `subscriptions` must be more than zero option
contracts, and the `expiries (...)` beside the ATM strike must be dates the exchange really
lists (`data/raw/instruments/NFO_<date>.json`). This is what the shipped configuration
produced on 2026-09-30, while `weekly_expiry.weekday` was still `Thursday` and the exchange
listed Tuesday expiries:

```
ATM strike          : 22600 (expiries 2026-10-01)
subscriptions       : 0 option contract(s) + 2 reference instrument(s) in 1 batch(es)
missing contracts   : 2 (e.g. no listed options for expiry 2026-10-01)
```

Empty `expiries ()` plus `no listed options for expiry <date>` means `weekly_expiry.weekday`
in `config/calendar.yaml` disagrees with the exchange. **Fix it, then re-run the plan**: with
`weekday: Tuesday` - which is what the file ships since 2026-09-30 - the very same command
resolves

```
ATM strike          : 22600 (expiries 2026-10-06, 2026-10-13)
subscriptions       : 12 option contract(s) + 2 reference instrument(s) in 1 batch(es)
```

A session planned against the wrong weekday is silent data loss: it subscribes nothing,
captures the index and no options, and still exits `0`. NSE has moved NIFTY weekly expiry
before, so check the dump the day before you start caring about a new expiry cycle.

The plan is offline, but it needs the instrument master and one price: on a fresh machine run
§1.5 first, or it stops with "no reference price available" (exit `2`) and there is nothing to
read. The price comes from the live quote when the market is open, the dump's `last_price`
otherwise, and - for an index, whose dump price is always `0` - from the newest stored
`index_history` close.

### 1.5 Instrument master and index history

```powershell
python -m apps.anjaneeya_capture.run --mode instruments     # today's Kite dump -> DuckDB
.\scripts\run_backfill.ps1                                  # 730 days (~2 years) of NIFTY index OHLCV
.\scripts\run_backfill.ps1 --years 3                        # override the window for one run
python -m apps.anjaneeya_capture.run --mode backfill-options --dry-run   # plan the option seed
python -m apps.anjaneeya_capture.run --mode backfill-options             # seed the listed options
```

The instrument master is cached, so later dry runs work offline. `index_history` is what
`enrich-greeks` reads first (candles are only the fallback), so run the backfill before the
first pipeline run — otherwise the IV columns stay `NULL`, and the pipeline will say so in
its pre-flight.

`--mode backfill-options` is an optional bootstrap: Kite serves the history of **active**
options and never expired ones, so it can only seed the contracts listed *today* (the widened
universe - five strikes, three expiries, both option types - is ~30 contracts). It writes them
to the keyed `candles` table with `source=zerodha_rest`, so a later tick-built candle for the
same minute simply overwrites it. It is not a substitute for the live capture; a day without
captured ticks is still a day the study loses.

### 1.6 Prove the whole chain, without spending a session

```powershell
.\scripts\start_capture.ps1 --smoke               # a few minutes of real ticks
.\scripts\run_daily_pipeline.ps1 --dry-run        # plans all eight steps, writes nothing
.\scripts\run_daily_pipeline.ps1                  # builds the day (expect exit 4 - see §5)
```

`--smoke` uses `capture.smoke_minutes` from `config/data_sources.yaml` and shuts down
cleanly. A pipeline dry run resolves the day, counts the ticks and index rows it finds, and
prints the exact command line of every step it would run.

### 1.7 Prove the broker path, at any hour

```powershell
$env:ANJANEYA_RUN_LIVE = "1"
uv run pytest tests/live -m live -q
```

Run this once after the first `scripts/daily_login.py`, and again whenever something about the
broker — the token, the endpoint, the instrument dump — looks wrong. It is the only part of the
suite that touches the network, so it is opt-in: the switch must be exported, and `ANJANEYA_RUN_LIVE`
in `.env` is deliberately ignored, so enabling live calls is never a side effect of a credentials
file.

It is three tests because the live path needs three different things, and an evening run has to
tell you *which* one is proven:

| Test | Needs | After the close |
| --- | --- | --- |
| `broker_path_authenticates_and_stores_the_master` | a live token | **passes** |
| `the_universe_resolves_from_a_real_price` | + one index quote | passes while the broker still quotes the index after the close (it normally does), otherwise skips with the reason |
| `a_one_minute_session_is_stored_end_to_end` | + the exchange open | skips — a closed market is not a defect |

So a run at 19:00 is still worth doing: `1 passed` at worst and `2 passed` at best already proves the
token, the instrument dump download and the universe, leaving only the ticks unproven until the next
session. `0 passed` with an error names the cause — an expired token says so, and the fix is
`python scripts/daily_login.py`.

---

## 2. Every trading day

### 2.0 The day, step by step

Six steps, in this order, every trading day. Each one says what "ok" looks like; §6 has the
remedies. The sections below (2.1-2.4) explain the same commands in more depth.

**Before the open**

1. **Is today a trading day?**

   ```powershell
   python scripts/check_calendar.py --days 7
   ```

   Expect `verdict: OK` and exit `0`, with every date reading `TRADING 09:15-15:30` (or
   `CLOSED` for a weekend/holiday you can confirm). Exit `1` names a provisional holiday
   inside the window: reconcile it with the NSE circular **now** and set `verified: true` on
   that entry in `config/calendar.yaml`. Unreconciled, a date that is really a trading day
   costs the whole session.

2. **What will the session subscribe to?**

   ```powershell
   python -m apps.anjaneeya_capture.run --mode capture --dry-run
   ```

   Expect `subscriptions : N option contract(s)` with `N > 0` and `expiries (...)` naming
   dates the exchange really lists (§1.4). Exit `2`, "no reference price available", means
   the instrument master or the index history is missing - run §1.5.

3. **Log in and start the session** (one command; it runs until 15:30)

   ```powershell
   .\.venv\Scripts\python.exe scripts/login_and_start.py
   ```

   Expect the browser to open (finish the 2FA there), then `==> ANJANEYA-CAPTURE session
   (mode=capture)`. Exit `1` = the login failed and **nothing** was started; exit `2` =
   `.env` problem; exit `3` = you pressed Ctrl-C. A re-run reuses the cached token, so it is
   always safe to try again. If the browser never comes back, the Kite app's Redirect URL is
   not the loopback address this machine listens on: re-run with `--paste` and paste the
   redirect URL you landed on (§1.2).

**While the session runs (09:15-15:30)**

4. **Leave that window alone** - it is the only process allowed to touch
   `db/anjaneeya.duckdb`. No pipeline, no engine, no AI app, and never a second capture.
   Watch it from a *second* window instead:

   ```powershell
   python scripts/health_check.py
   Get-Content data\runs\heartbeat\capture.json
   ```

   The heartbeat is rewritten every 10 s; older than 90 s (`stale_after_seconds`) means the
   session is not healthy even if the window is still open. `PARTIAL` (exit `1`) means
   something was missing - see §6 - and `coverage_ratio` in `data_coverage` is the day's
   scorecard.

**After the close (15:35, once the capture window has exited)**

5. **Build the day**

   ```powershell
   .\scripts\run_daily_pipeline.ps1
   ```

   Expect the eight-step table and exit `0`. Exit `4` is the training gate refusing (§5), not
   a failure; exit `2` means "no ticks are stored for this day" (a holiday, or a capture that
   never ran). Every child's output is in `data/logs/daily_pipeline_<date>.log`.

6. **Confirm the day landed**

   ```powershell
   python scripts/health_check.py
   python -m apps.anjaneeya_ai.run --mode status
   ```

   `captured_trading_days` must be one higher than yesterday, and `ticks`/`candles` must have
   rows for today. If it did not move, the session did not store anything: read step 3's exit
   code and the day's log before touching anything else.

The arithmetic is unforgiving: steps 1-3 are what prevent an unrecoverable day, steps 5-6 are
what make the day usable.


### 2.1 Before the open (by 09:00)

**One guarded command does the whole morning** — the calendar guard, the universe gate, the
login and the capture session, in that order, fail-fast:

```powershell
python scripts/run_morning.py                 # calendar -> universe gate -> login -> capture
python scripts/run_morning.py --dry-run       # plan only: what it would subscribe to, no network
python scripts/run_morning.py --refresh-instruments   # re-download the dump first if it is stale
python scripts/run_morning.py --json          # one machine-readable summary for a log
```

It refuses the two ways a day is lost silently: on a non-trading day it prints `NO_SESSION`
and exits `0` without touching the broker; when the option universe resolves to **zero**
contracts it exits `4` **without starting the session** (a session started that way records
the index and no options, and still exits `0` — the §1.4 trap). Run `python scripts/doctor.py`
first if you are unsure whether the machine is ready (token, dump age, database, scheduler).

The steps it automates, by hand, are still worth knowing — they are what it runs:

```powershell
python scripts/check_calendar.py --days 7                        # is today really a trading day?
python -m apps.anjaneeya_capture.run --mode capture --dry-run    # N > 0 contracts? real expiries?
python scripts/login_and_start.py                                # login + capture session, in one go
```

* `check_calendar.py` exit `1` → reconcile the named provisional holiday with the NSE
  circular **now**. If the date really is a trading day, set `verified: true` (or delete the
  entry) — otherwise the session refuses to run and the day is gone.
* the capture dry run is offline and takes seconds: it prints the ATM anchor, the strikes,
  the expiries and the request budget the session is about to spend. `0 option contract(s)`
  or an `expiries (...)` the exchange does not list means a wrong `weekly_expiry.weekday`
  (§1.4) — fix it before 09:15, not after.
* `login_and_start.py`: a cached token is reused; otherwise the browser opens the Kite login
  (finish the 2FA there), the `request_token` is caught on the loopback redirect URL, and only
  after `profile()` confirms the token does the capture session start. Exit `1` = the login
  failed and **nothing** was started; exit `2` = `.env` problem; exit `3` = you pressed Ctrl-C.
  Useful variants:

| Variant | Use it when |
| --- | --- |
| `--no-browser` | you want to open the URL in a specific browser (the URL is printed) |
| `--paste` | the app's redirect URL is not loopback: paste the redirect URL/`request_token` |
| `--no-start` | login only; start the capture later with `.\scripts\start_capture.ps1 -SkipLogin` |
| `--dry-run` | see which login path and capture command would be used, without doing either |
| `-- --smoke --until 15:30` | everything after `--` goes to the capture app |

The two-step route still works if you prefer it:

```powershell
python scripts/daily_login.py --json          # fresh token (or reuse the cached one)
python -m apps.anjaneeya_capture.run --mode capture --dry-run
```

* `daily_login.py` exit `0` and `"verified": true` → good. Exit `2` → `.env` problem; exit
  `1` → broker problem (use `--request-token`).

### 2.2 During the session (09:15–15:30)

```powershell
.\scripts\start_capture.ps1
```

The wrapper runs the login (already done above, so it usually reuses the token) and then
`--mode capture`. The session:

* subscribes the universe over the WebSocket and appends every tick,
* polls the option chain into `option_chain_snapshots` (OI/IV/quote),
* tops up the NIFTY index history,
* re-centres the strike window every `capture.universe_refresh_minutes`,
* writes a heartbeat to `data/runs/heartbeat/capture.json` every
  `capture.heartbeat.interval_seconds` (10 s by default; older than
  `stale_after_seconds`, 90 s, means the session is not healthy).

While it runs, this is the only process allowed to touch the database. Do not start the
engine, the pipeline or anything else that writes.

What the exit status means:

| Status | Exit | Meaning |
| --- | --- | --- |
| `SUCCESS` | 0 | the session captured what it was asked to |
| `NO_SESSION` | 0 | a holiday/weekend/after-close: nothing to do, not an error |
| `PARTIAL` | 1 | ticks were captured but something was missing — see §6 |
| `FAILED` | 1 | the session broke; the ticks that did land are still on disk |

### 2.3 After the close (15:35 onwards)

Wait until the capture process has exited, then run the one command for the day:

```powershell
.\scripts\run_daily_pipeline.ps1
```

It runs these eight steps, in order, fail-fast:

| # | Step | Mode | What it produces |
| --- | --- | --- | --- |
| 1 | candles | `build-candles` | 1m/3m/5m/15m bars + the `data_coverage` ledger |
| 2 | greeks | `enrich-greeks` | IV, delta… on the stored bars (reads `index_history` first) |
| 3 | features | `build-features` | the section-5 feature matrix |
| 4 | labels | `build-labels` | forward-return labels |
| 5 | validate | `validate` | the no-look-ahead validation report |
| 6 | dataset | `dataset` | the training matrix + time-ordered splits |
| 7 | train | `train` | a LightGBM artefact under `data/models/` (see §5) |
| 8 | backtest | `backtest` | the spec-23 replay ledger + summary under `data/backtests/` |

Useful flags:

```powershell
.\scripts\run_daily_pipeline.ps1 --dry-run --json          # plan only, machine-readable
.\scripts\run_daily_pipeline.ps1 --date 2026-10-01         # a specific day
.\scripts\run_daily_pipeline.ps1 --force                   # train even below the gate
.\scripts\run_daily_pipeline.ps1 --skip validate,backtest  # rebuild the engine half only
.\scripts\run_daily_pipeline.ps1 --continue-on-error       # keep going after a failure
```

Exit codes:

| Exit | Meaning |
| --- | --- |
| 0 | every step that ran succeeded |
| 1 | a step failed — the console names it, stderr says which |
| 2 | configuration problem: unknown `--skip` step, no ticks for the day, locked database |
| 4 | nothing failed, but `--mode train` refused below spec 4.3's gate (see §5) |

Three details worth knowing:

* the engine steps build the day given by `--date` (default: the newest day with stored
  ticks). The AI steps always read **all** history — a model trained on one day is not a
  model;
* `--timeframe` narrows `features`, `labels`, `validate` and the dataset. Candles and greeks
  always build every configured timeframe;
* `--horizon` is repeatable for `dataset`/`train`, but a backtest replays one model and a
  model is one horizon, so use a single `--horizon` or `--skip backtest`.

### 2.4 Afterwards

```powershell
python scripts/health_check.py                 # 0 = every section ok
Get-ChildItem data\logs\daily_pipeline_*.log | Sort-Object -Last 1 | Get-Content
```

The pipeline writes one log per day to `data/logs/daily_pipeline_<date>.log` (every child's
stdout/stderr, the plan, the step table and the verdict), and every app writes its own run
manifest under `data/runs/<app>_<timestamp>/manifest.json`.

**The research report is a separate, explicit step** — it is not part of the pipeline, because
it answers questions about *the whole ledger*, not about one day:

```powershell
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode report
```

It reads the newest ledger under `data\backtests\` (or `--ledger PATH`) plus the stored index
bars, and writes `<ledger stem>_report.json`, `<ledger stem>_report.md` and
`<ledger stem>_cells.csv` under `data\reports\research\`. Read the Markdown one: it lists spec
24's ten answers with every sample beside its number, prints `PRELIMINARY` wherever the sample
is below `evaluation.preliminary_min_samples`, and repeats the caveats (threshold signal,
configurable costs, the 250-day seasonality gate) under the tables. `--dry-run` and
`--no-write` print the same report without writing the three files. See README's *"Answering
the research questions"* for what each of the ten items contains.

### 2.5 The weekly accumulation check (spec 29)

Everything here rests on one assumption — that enough was captured — and this is the command
that checks it. Run it once a week (Monday morning is the natural slot):

```powershell
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode accumulation
```

It reads the stored tables and the `data_coverage` ledger read-only, and writes
`accumulation_<date>_accumulation.json`, `..._accumulation.md` and `..._tables.csv` under
`data\reports\accumulation\`. The Markdown one is the one to read: five sections — data
captured (sample sizes), coverage, captured days by week, readiness for retraining and warnings
— with the caveats under them.

What to look for, in order:

1. **`gaps` and the days-with-gaps table.** A day with a non-zero gap count is a day whose bars
   are missing minutes; it is listed worst-first so the re-capture decision is obvious.
2. **`recommended`.** `yes` means either there is enough data and no model yet, or the captured
   data is newer than the newest model, or the model is older than
   `model.accumulation.retrain_max_model_age_days`. It is a scheduling hint, not a verdict — the
   retrain itself still has to beat the walk-forward numbers.
3. **`ready      :` and the four checklist lines.** Each prints its `have` and its `need`, so
   "not ready" always names the gate or the sample that is short.
4. **`warnings`.** The stale-capture line ("capture may have stopped") is the one that matters
   most: it is the only place in the system that notices a week of silence.

`--weeks N` widens or narrows the window, `--json` emits the whole document, and `--dry-run` /
`--no-write` print it without writing anything. The report describes the *dataset*, never the
market — a green line means "there is enough recorded data to try something", nothing more.

### 2.6 What the rule would do (spec 24, the decision layer)

Master-prompt phase 24 puts the decision layer "later, separate — only after features, model
and backtest are validated". `--mode decide` is that separate step, and it is the last thing to
run in a research cycle: it records what **one configured rule** would have done with every
stored bar.

```powershell
# the newest model, over every stored row
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode decide
```

It writes `..._decisions.json`, `..._decisions.md` and `..._decisions.csv` under
`data\predictions\`, named after the model that produced them. Read the Markdown for the policy
and the entries; open the CSV when you want to filter on `reason`.

What to look for, in order:

1. **`results`** and **`gate`**. A decision made from a `BELOW_GATE` model is labelled
   `BELOW_GATE`: it inherits the model's standing and adds nothing to it. This is the first line
   to read because it decides whether the rest is worth reading.
2. **The reason counts.** `--mode decide` reports *why* it declined, so `BELOW_MIN_PROBABILITY`
   dominating means the policy is stricter than the model is confident; `NOT_ARGMAX` dominating
   means the directional class rarely wins.
3. **`entry rate`.** A rule that fires on 90% of bars is a busy rule, not a discovery. Nothing
   here is optimised — the thresholds are hand-set configuration and were fitted to no outcome.
4. **The entries table.** Every entry prints the probability and the margin behind it, plus the
   size and the exit plan the record *intends*.

`--min-probability`, `--min-margin` and `--max-decisions-per-day` override the rule for one run
(the record stores the values actually used, so a number stays reproducible); `--symbol`,
`--from`/`--to`, `--timeframe`, `--horizon` and `--model` scope it; `--json`, `--dry-run` and
`--no-write` behave as everywhere else.

It **places no order** — there is no broker anywhere in this repository — and it **simulates no
exit**: the stop, the target, the holding time and the session-end flatten are stated in the
record, and replaying them against stored prices with costs is `--mode backtest`'s job. Decide
produces a decision; backtest prices it.


---

## 3. Weekly and monthly checks

| Cadence | Check | Command |
| --- | --- | --- |
| weekly | anything provisional inside the next 30 days? | `python scripts/check_calendar.py --days 30` |
| weekly | every section still healthy? | `python scripts/health_check.py` |
| weekly | the long NIFTY history is intact | `.\scripts\run_backfill.ps1` |
| weekly | how much data is actually there? | `python -m apps.anjaneeya_ai.run --mode status` |
| weekly | captured days, coverage, gaps, sample sizes and whether a retrain is due | `python -m apps.anjaneeya_ai.run --mode accumulation` |
| weekly | what the ledger says, with samples and corrected p-values | `python -m apps.anjaneeya_ai.run --mode report` |
| weekly | what the newest model was looking at | `python -m apps.anjaneeya_ai.run --mode explain` |
| weekly | which inputs the newest model actually uses | `python -m apps.anjaneeya_ai.run --mode importance` |
| weekly | a look at the same results - heatmaps, CE/PE curves, regime bars, importance bars, price charts | `python -m apps.anjaneeya_ai.run --mode charts` |
| weekly | gaps in the week's bars/ticks | `python -m apps.anjaneeya_engine.run --mode universe` |
| monthly | disk usage (`data/candles`, `data/runs`, `data/raw`) | `Get-ChildItem data -Recurse \| Measure-Object -Property Length -Sum` |
| yearly (December) | reconcile the **next** year's holiday list | `python scripts/check_calendar.py --as-of 2027-01-01 --days 365` |

`data_coverage` is the ledger of every stream the capture touched. A day that looks "fine"
but has `coverage_ratio` well below 1 or a non-zero `gap_count` is a day whose bars will be
missing minutes — check it before trusting a week's analysis.

---

## 4. Scheduling it

Schedulers are optional. If the operator is present anyway, run §2.1–2.3 by hand: it takes
ten seconds and you get to see what the calendar and the pre-flight say.

If you do schedule it, use **two** tasks, because the capture must exit before the pipeline
starts (one writer per DuckDB file).

### Windows Task Scheduler

One command installs both weekday jobs with every setting in place — no UI clicking:

```powershell
.\scripts\schedule_tasks.ps1 install                        # 09:05 Morning, 15:45 PostClose
.\scripts\schedule_tasks.ps1 install -MorningTime 09:10 -PostCloseTime 15:50
.\scripts\schedule_tasks.ps1 status                         # state, last result, next run
.\scripts\schedule_tasks.ps1 remove                         # unregister both
```

`install` needs an elevated PowerShell session. It registers `\Anjaneya\Anjaneya-Morning`
and `\Anjaneya\Anjaneya-PostClose`, each running `scripts/run_scheduled.ps1`, which first
runs `check_calendar.py --require-trading-day --within 0` and exits cleanly on a holiday.
The tasks are created to:

* **run whether the user is logged on or not** (`S4U`, no stored password — for an
  unattended login you need `ZERODHA_USER_ID` + `ZERODHA_PASSWORD` + `ZERODHA_TOTP_KEY`,
  see §1.2 and `scripts/doctor.py`);
* run with the **highest privileges**;
* **ignore a new instance** while one is still running (`MultipleInstances=IgnoreNew`, so a
  second run never opens the database twice);
* **stop after 8 hours** (well past the longest session plus pipeline);
* start as soon as possible if the start time was missed (`StartWhenAvailable`).

The pipeline at 15:45 assumes the capture finished at 15:30. If a session ever runs long
(`--until`), move the pipeline later rather than letting the two overlap.

### cron (Linux, macOS, WSL, Git Bash)

```cron
# ANJANEYA - one guarded morning session and one post-close pipeline, weekdays only.
5 9  * * 1-5 cd /path/to/Project-Anjaneya && ./scripts/run_morning.sh             >> data/logs/cron_capture.log 2>&1
45 15 * * 1-5 cd /path/to/Project-Anjaneya && ./scripts/run_daily_pipeline.sh --json >> data/logs/cron_pipeline.log 2>&1
# weekly cadence (the daily chain plus report -> accumulation)
0 9 * * 6 cd /path/to/Project-Anjaneya && ./scripts/run_weekly.sh --force        >> data/logs/cron_weekly.log 2>&1
```

A cron job starts with a minimal environment: set `ANJANEYA_PYTHON=/path/to/.venv/bin/python`
in the crontab if the wrapper cannot find the virtual environment itself.

On a holiday the capture exits `0` (`NO_SESSION`) and the pipeline exits `2` ("no ticks are
stored") — that is the pipeline saying there was nothing to build, not a broken night. To
keep the scheduler honest, gate the evening line on the calendar instead of filtering the
alert afterwards:

```cron
45 15 * * 1-5 cd /path/to/Project-Anjaneya && python scripts/check_calendar.py --require-trading-day \
  && ./scripts/run_daily_pipeline.sh --json >> data/logs/cron_pipeline.log 2>&1
```

`--require-trading-day` exits `1` on a weekend or holiday, so the chain is only attempted on
a day the market was actually open.



---

## 5. When will there be a model?

The pipeline always builds data; whether it produces a **model** depends on how many days
have been captured. `config/model.yaml` enforces spec 4.3's gates:

| Captured days | What happens |
| --- | --- |
| 1 | `candles`…`dataset` succeed. `train` **exits 4** (below the exploratory gate) and the pipeline skips the backtest — exit code `4`, not a failure |
| 1 – 19 | the same, unless `--force` is passed |
| 20 – 44 | `train` runs and labels the result `BELOW_GATE`/`INSUFFICIENT`; with `--force` an artefact is written |
| 45 – 119 | `train` is allowed: results are `EXPLORATORY` |
| 120 – 249 | a backtest is "meaningful" (`meaningful_backtest_min_days`) |
| 250 + | seasonality analysis is allowed |

One caveat that surprises people: gate aside, **the walk-forward split needs enough days to
form a fold**. With the shipped `walk_forward` (`train_window_days: 30`,
`validate_window_days: 10`, `holdout_days: 30`) `--mode train` reports

```
horizon 5m: no folds were built: needs 70 captured day(s) (holdout 30 + train 30 + validate 10)
```

and writes nothing — even with `--force`. That is not a bug: it is the honest answer that 70
trading days is the smallest sample the configured walk-forward can test on. Options:

* wait (70 captured days ≈ 3.5 months of trading) — the recommended path, because a model
  trained on a shorter window is a weaker claim, not a faster one;
* deliberately lower `walk_forward.train_window_days` / `validate_window_days` /
  `holdout_days` and record it as a design decision, knowing every metric afterwards carries
  that caveat.

`--mode backtest` with no model artefact prints `no model artefact to replay` and exits `0`
(nothing to do). The pipeline avoids that case: after a gate refusal it marks the backtest
`SKIPPED` rather than replaying yesterday's model as if it were today's.

`--mode report` with no ledger prints `no trade ledger to analyse` and exits `0`, and it
**never refuses** to report: below the seasonality gate the weekday and cycle-phase tables are
labelled `PRELIMINARY` (and cells below the sample rule are not tested at all), but the numbers
are printed, because knowing what a four-day ledger looks like is how you learn to distrust it
early.

`--mode explain` with no model artefact prints `no model artefact to explain` and exits `0`
(train one first, or pass `--model PATH`). It applies **no** gate: an explanation describes the
inputs of a prediction, so refusing it for a small sample would hide the very thing an operator
needs to look at. The model's own label travels in the artefact instead, and so does the
sentence that an explanation is not a recommendation (the decision layer is the separate
`--mode decide`).

`--mode importance` with no model artefact prints `no model artefact to measure under` and exits
`0` (train one first, or pass `--model PATH`). It applies **no** gate either: a feature ranking
describes a model's use of its inputs, and the model's own results label travels in every
artefact (`the model measured is labelled EXPLORATORY ...: this ranking describes that model,
not a validated one`). It never refits and never writes to the database: it permutes a column of
the model's own training matrix, scores it against the baseline the untouched matrix produces,
repeats it at the configured seed and reports the interval - per feature and per source - with
the rows measured named in a caveat (spec 22's holdout days when the split has any, the whole
matrix - labelled in-sample - when it does not). A feature the sample cannot measure (all
`NULL`, or a single value) is reported as *not measured* with its reason, never as a delta of
`0`. SHAP is optional: a missing package is a recorded status unless `model.importance.shap` is
`on`, in which case it is a configuration error (`exit 2`). A `--repeats` below 2, a timeframe or
horizon the artefact never saw, and a malformed `--model` are all `exit 2` as well. Artefacts go
under `data/reports/importance/`, named after the model they rank.

`--mode charts` (spec 28) draws figures from what already exists - the newest ledger (or
`--ledger PATH`), the newest importance artefact (or `--importance PATH`) and the stored candles,
features and index bars - and writes one image per chart plus one index Markdown under
`data/reports/charts/`. It never recomputes a number: the heatmaps, curves and regime bars are the
same ledger `--mode report` reads, and the bars are the same artefact `--mode importance` wrote, so
a picture cannot disagree with the number beside it. Each chart is attempted independently, so a
missing ledger (or importance artefact, or stored candles) prints the reason and skips exactly the
charts that need it rather than failing the run - and a run that can build nothing exits `0` with
`no chart could be built`. `--only heatmap|curves|regime|importance|price` (or an exact chart name
such as `--only curves_ce_vs_pe`) narrows the run; `--metric native` charts the native column of the
importance artefact instead of the permutation delta. `--dry-run` lists what would be built without
importing matplotlib; `--no-write` draws and reports without leaving a file. Nothing is written to
DuckDB. Exit codes: `0` ok, `1` a crash, `2` configuration.

---

## 6. Failure playbook

| Symptom | Meaning | Action |
| --- | --- | --- |
| `check_calendar.py` exits `1`, "PROVISIONAL" | a holiday nobody has confirmed is inside the window | check the NSE circular; set `verified: true` or delete the entry (§1.3) |
| capture exits `NO_SESSION` on a day the market was open | the calendar closed a real trading day | fix `config/calendar.yaml`; that day's ticks are lost — this is why §2.1 exists |
| capture exits `FAILED` mid-session | the session broke; the ticks that landed are still on disk | run the pipeline anyway (`--continue-on-error`), then read `data_coverage` for the day; gaps are recorded, never fabricated |
| pipeline exit `2`, "the ticks table is empty" | nothing has ever been captured | run a capture session (`start_capture --smoke` proves the chain first) |
| pipeline exit `2`, "no ticks are stored for `<date>`" | that day has no capture (holiday, failure, or a wrong `--date`) | use the day the pre-flight prints, or `--date` a day that is stored |
| pipeline exit `4` | `train` below the gate | see §5 |
| `DatabaseLockedError` / "Could not set lock on file" | the capture session is still running, or a stale process holds the file | stop it, confirm the heartbeat has gone quiet, re-run — every step is idempotent |
| heartbeat `status: RUNNING` with a large `age_seconds` | the capture process died without a clean shutdown | read the capture log for the last tick, then trust `data_coverage`, not the heartbeat |
| candles `0 bars` / `NO_CANDLES` | the day has ticks but the build produced nothing | check the tick window and the day the engine resolved (`--date`); re-run `--mode build-candles` |
| many `NULL` IVs after `enrich-greeks` | `index_history` has no NIFTY rows for that day | `run_backfill.ps1`, then re-run `enrich-greeks` (`--skip candles,features,labels,validate,dataset,train,backtest`) |
| `validate` reports `FAIL no_look_ahead` | a feature changed when the window was truncated: something reads a later row | rebuild the features with `build-features`; treat any analysis on the flagged rows as void |
| `daily_login.py` exit `1` | the broker refused the login | `--request-token <token from the redirect URL>` |
| `login_and_start.py` exit `1` (login failed) | the token was never obtained, so **no** capture was started | read the error above it: a `.env`/API-key problem, or `--paste` when the redirect URL is not loopback |
| `login_and_start.py`: "Timed out … waiting for the browser" | the browser never came back to the redirect URL this machine is listening on | register `http://127.0.0.1:8444/` **exactly** as the Kite app's Redirect URL (or pass `--redirect-url`/`--port` to match it); `--paste` always works as a fallback |
| `login_and_start.py` exit `3` | you pressed Ctrl-C during the login wait | re-run; nothing was cached and nothing was started |
| `run_backfill` fails | no fresh NIFTY index history | not fatal: the pipeline warns and IVs fall back; retry in the evening |
| capture dry run prints `0 option contract(s)` and `no listed options for expiry <date>` | `weekly_expiry.weekday` in `config/calendar.yaml` disagrees with what the exchange actually lists | check `data/raw/instruments/NFO_<date>.json` for the real expiry dates (it listed Tuesdays on 2026-09-30), fix the weekday, re-run the dry run - a session planned this way subscribes nothing and still exits `0` |
| capture dry run exits `2`, "no reference price available" | the plan has no way to anchor the ATM: no live quote (market closed), `last_price: 0` in the dump (always the case for an index) and no stored `index_history` close | `.\\scripts\\run_backfill.ps1` once (§1.5), or plan while the market is open |
| first `--mode instruments` seems to hang | it is not hung: it writes ~46k rows one at a time (~4 minutes) and downloads ~13 MB of dumps into `data/raw/instruments/` | wait for `instrument master ... stored rows`, or check `data/logs/anjaneeya_capture.log` for progress |
| `--mode report` exits `0` with "no trade ledger to analyse" | no backtest artefact exists under `data/backtests/` (or `--ledger` names a file that is not there) | run `--mode backtest` first (it needs a model), or point `--ledger` at the Parquet you mean |
| `--mode report` exits `1`, "is not a trade ledger" | the Parquet is not the artefact `--mode backtest` wrote (its columns are not the trade contract) | pass the real `*_ledger.parquet`; a features/dataset dump is a different file with a similar name |
| `--mode report`: items 7 and 8 say "not measured" | there are no stored 1m NIFTY index bars in the ledger's window | `.\scripts\run_backfill.ps1` (§1.5), then re-run `--mode report` |
| `--mode report`: no cell in `9_significance` | every cell is below `evaluation.preliminary_min_samples` on one side of the comparison | that is the answer, not a failure: capture more sessions, or lower the rule knowingly |
| `--mode report`: rows of `PRELIMINARY` | the sample is below the configured minimum (or the ledger is shorter than the 250-day seasonality gate) | read them as observations; see §5 for the gate ladder |

When in doubt, run `python scripts/health_check.py` — it is offline, read-only, and reports
configuration, calendar, database, credentials, capture heartbeat and data progress in one
pass.

---

## 7. Where everything lives

| Path | What |
| --- | --- |
| `db/anjaneeya.duckdb` | the database: ticks, candles, `index_history`, features, labels, trades, `data_coverage`, `run_manifest` |
| `data/logs/` | app logs, plus one `daily_pipeline_<date>.log` per pipeline run |
| `data/runs/` | one manifest per app run (config hash, schema version, notes, exit status) |
| `data/runs/heartbeat/capture.json` | the capture session's liveness and progress |
| `data/raw/.tokens/` | the cached Kite access token (treat it as a secret) |
| `scripts/login_and_start.py` | the one-command morning: browser login caught on the loopback redirect URL, then the capture session |
| `data/candles/<timeframe>/` | Parquet mirrors of the stored bars |
| `data/features/`, `data/labels/` | Parquet mirrors of the derived tables |
| `data/datasets/` | the training matrices and their split definitions |
| `data/models/` | LightGBM artefacts (`*_h<horizon>_*.txt`) plus their metadata JSON |
| `data/backtests/` | the replay ledger and summary per backtest |
| `data/reports/feature_validation/` | the `validate` reports |
| `data/reports/research/` | spec 24's research reports (JSON + Markdown + one cells CSV per ledger) |
| `data/reports/explanations/` | spec 26's explanations (JSON + Markdown + one entries CSV per model) |
| `data/reports/importance/` | spec 27's feature-importance reports (JSON + Markdown + one features CSV per model) |
| `data/reports/charts/` | spec 28's figures (one image per chart + one index Markdown) |
| `data/reports/accumulation/` | spec 29's weekly accumulation reports (JSON + Markdown + one per-table CSV per day) |
| `data/predictions/` | spec 24's decision records (`--mode decide`: JSON + Markdown + decisions CSV per model) |

Environment overrides, useful for a scratch run that must not touch the real data:
`ANJANEYA_DATA_ROOT` and `ANJANEYA_DB_PATH` — both are also `--data-root` / `--db-path` flags
on every app and on the pipeline.

---

## 8. The rules that keep the data honest

1. **One writer.** Capture, engine and AI never share DuckDB at the same moment. Capture in
   the morning, pipeline in the evening; anything else is a bug, not a shortcut.
2. **Ticks are append-only.** Never edit or delete a tick row: `data_coverage` and every
   derived table are rebuilt from them, and the ledger is the record of what was really
   captured.
3. **Every derived table is rebuildable.** If candles, features or labels look wrong, re-run
   the build for that day — never hand-write a derived value.
4. **The calendar is data.** A holiday the market did not take costs a whole session: fix the
   date, then set `verified: true`.
5. **The universe is a one-way door.** Widen `strikes_offset`/`expiries` before the first
   capture; narrowing later is free, recovering the missing strikes is impossible.
6. **Gates are the point.** A model below 60 captured days, or a backtest below 120, is not a
   result. `--force` exists to smoke-test the chain, not to publish a number.

