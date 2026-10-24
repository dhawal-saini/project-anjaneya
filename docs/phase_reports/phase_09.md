# Phase 9 report — the routine, automated (one installer, two commands a day)

**Status: Stage 9.1 (`run_morning`), 9.2 (`--weekly`), 9.3 (`doctor`), 9.4
(`refresh_calendar` + `schedule_tasks`) and 9.5 (the data bootstrap: a 2-year index backfill,
an active-contract option seed and relaxed exploratory walk-forward windows) complete.**

Phase 8 left the repository in a state where every *stage* existed and every *day* was still
manual. The runbook's §2 was a checklist of nine commands to run in order before 09:00 — log in,
check the calendar, dry-run the universe, start the capture — and the two ways to lose a day were
both silent:

* a capture started on a day the calendar wrongly calls a trading day exits `NO_SESSION` and the
  session is simply never captured (the ticks do not exist anywhere afterwards);
* a capture whose option universe resolves to **zero** contracts subscribes the index and no
  options, records that, and still **exits `0`** (`config/calendar.yaml`'s own comment calls this
  out: it happened when `weekly_expiry.weekday` disagreed with the exchange).

Neither is discoverable until hours later, and neither is recoverable. Phase 9 turns the morning
into one guarded command that cannot take either path, wraps the whole day in two commands plus a
one-shot installer, adds a weekly cadence, and bootstraps the data the relaxed exploratory model
needs.

## 1. Files created

* `scripts/run_morning.py` (+ `.ps1`, `.sh`) — the guarded morning routine:
  calendar guard → universe gate → login → capture. Exit codes `0` ok / `1` login-or-capture
  failed / `2` configuration / `4` the universe was empty and the session was **not** started.
* `scripts/run_scheduled.ps1` — the calendar-gated entry point the Tasks call: it runs
  `check_calendar.py --require-trading-day --within 0` first and exits `0` (a clean no-op) on a
  non-trading day, then dispatches the `morning` or `postclose` stage.
* `scripts/schedule_tasks.ps1` — `install` / `status` / `remove` for the two weekday tasks
  (`\Anjaneya\Anjaneya-Morning`, `\Anjaneya\Anjaneya-PostClose`).
* `scripts/doctor.py` — the pre-flight readiness report: it reuses `health_check.py`'s sections
  and adds the login route (TOTP vs manual), the cached token's expiry, the instrument-dump age,
  a database write probe and the scheduler state.
* `scripts/refresh_calendar.py` — fetches NSE's `holiday-master?type=trading`, records the raw
  response under `docs/evidence/`, diffs it against `config/calendar.yaml` and (with `--write`)
  rewrites the `holidays:` block while preserving every comment in the file.
* `scripts/run_weekly.ps1` / `scripts/run_weekly.sh` — the daily chain plus the weekly cadence.
* `tests/unit/test_run_morning.py`, `test_run_weekly.py`, `test_doctor.py`,
  `test_refresh_calendar.py`, `test_capture_backfill_options.py`, `test_schedule_tasks.py`.

## 2. Files changed

* `scripts/run_daily_pipeline.py` — added the opt-in `--weekly` flag and `WEEKLY_STEPS`
  (`report → explain → importance → charts → accumulation`). Without `--weekly` the chain is the
  eight daily steps it always was, so the default behaviour is unchanged.
* `core/app_bootstrap.py` — registered the `backfill-options` capture mode.
* `apps/anjaneeya_capture/run.py` — added `--mode backfill-options` (resolve the listed universe,
  fetch each contract's candles, write them to the keyed `candles` table) and a `--years` window
  override for `--mode backfill`.
* `apps/anjaneeya_capture/storage_writer.py` — `write_candles()` plus a `candles` counter.
* `config/option_universe.yaml` — `strikes_offset: [-200, -100, 0, 100, 200]`, `expiries:
  [current_week, next_week, current_month]`, `backfill_days: 400 → 730`.
* `config/model.yaml` — `data_gates.ml_training_min_days: 60 → 45`; `walk_forward`
  `90/15 → 30/10` (step `15`), `embargo_minutes` left at `30` = the maximum label horizon.
* `core/accumulation.py`, `core/config_loader.py` — the `ml_training_min_days` *default* moved
  `60 → 45` in both, so the dataclass default, the config schema's fallback and
  `config/model.yaml` cannot describe two different policies.
* `docs/runbook.md`, `README.md`, `.env.example` — rewritten around the one-installer /
  two-commands-a-day routine.

## 3. The two silent losses, closed

`run_morning.py` is the whole point of the phase, and it is deliberately boring: it does the four
things §2 used to ask a human to do, in the only order that cannot lie.

* **`_calendar_verdict`** asks `NseCalendar.is_trading_day(today)` before anything else. A
  non-trading day prints `NO_SESSION` and returns `EXIT_OK` — a holiday is not an error, and the
  broker is never touched.
* **`_universe_gate`** runs `--mode capture --dry-run` (fully offline, from the cached dump),
  parses the `subscriptions N option contract(s)` line and:
  * `N ≥ 1` → the gate opens;
  * `N == 0` → `EXIT_GATE_REFUSED`, and the session is **not** started;
  * no such line, or a non-zero dry-run → a gate refusal or a configuration error, never a session.

Only when the gate is open does it run `scripts/daily_login.py` (via `login_and_start.run_login`,
so there is one login implementation) and, only after that verified, `start_capture`. A dead token
can therefore never reach the session as a silent no-capture run — exactly the failure the existing
`login_and_start.py` was built to prevent, now enforced on the scheduled path too.

## 4. Data bootstrap (stage 9.5)

Three changes make the first exploratory model reachable sooner, and each is honest about what it
can and cannot do:

* **The index window doubled** (`backfill_days: 730`) so `index_history` carries ~2 years of daily
  NIFTY OHLCV — past the spec-4.3 seasonality gate (250 trading days) — with `--years N` to
  override it for one run.
* **The capture universe widened** to five strikes × three expiries. This is a *one-way door*: a
  session only ever captures the contracts subscribed during it, and Kite serves no historical
  expired options, so the widening is documented as a decide-before-first-session choice. The
  offline dry-run now resolves **30** contracts (it was 12).
* **`--mode backfill-options`** seeds candles for the contracts Kite currently lists. It is
  explicitly *not* a substitute for the live capture: the provider serves active options only, so
  it cannot reconstruct a past dataset. Bars land in the keyed `candles` table with
  `source=zerodha_rest`, so a later tick-built candle for the same minute overwrites it.
* **The exploratory windows relaxed** (`walk_forward 30/10/15`, `ml_training_min_days: 45`). The two
  move together — a 30-day train window needs ~30 captured days, not 60 — and `embargo_minutes`
  stays at 30, which is `max(horizons_minutes)`.

## 5. Tests

Six new test modules, all offline and hermetic (child processes and the network replaced by
recorders), plus the existing 2008-test baseline:

* `test_run_morning.py` — the zero-contract gate (exit `4`, no session started), the holiday path
  (clean `NO_SESSION`, no child spawned), the login-before-start order, the login-failure path (no
  session started) and `--dry-run` doing neither.
* `test_run_weekly.py` — the daily chain is unchanged without `--weekly`; the cadence steps appear
  only when asked for, in order; a weekly step targets the AI app and needs no ticks.
* `test_doctor.py` — the report contract (`is_healthy`, every section, the verdict), the write
  probe and `main --json`.
* `test_refresh_calendar.py` — NSE payload parsing and year filter, the drift diff, and a YAML
  rewrite that keeps every comment and still reparses; `main --input … --dry-run` reports drift
  with no writes.
* `test_capture_backfill_options.py` — the `--years` window and the REST-bar → option-candle
  mapping (contract identity included).
* `test_schedule_tasks.py` — the scheduler wrappers still carry `IgnoreNew`, S4U, `Highest`, the
  8-hour cap and the `--require-trading-day --within 0` gate.

The modules that guard the *shipped* configuration were updated to the Phase-9 choices so they keep
asserting the real values rather than remembered ones: the widened ladder and expiry list
(`test_option_universe.py`), the engine's ladder/coverage counts and the fake instrument master
(`test_engine_universe.py`, `tests/support/engine.py`), the capture universe sizes
(`test_capture_instruments.py`, `tests/support/capture.py`), the relaxed gates and windows
(`test_config_loader.py`, `test_ml_walk_forward.py`), and the AI pipeline gate expectations
(`test_ai_accumulation_pipeline.py`, `test_ai_decision_pipeline.py`, `test_ai_dataset_pipeline.py`).
Two *source* defaults were aligned with the relaxed gate as well - `AccumulationPolicy`
(`core/accumulation.py`) and `DataGatesConfig` (`core/config_loader.py`) - because a default that
disagrees with `config/model.yaml` is a second, hidden policy. No assertion was weakened to make a
test pass; each now pins the value the project actually ships.

## 6. What is not done here (and why)

* **A live session has not been captured.** Today's session (Thursday 2026-10-01) has already closed
  uncaptured and 2026-10-02 is Gandhi Jayanti, so the first session this project can capture is
  **Monday 2026-10-05**. The live-token path is
  proven as far as the environment allows: `doctor.py` reports `ZERODHA_API_KEY`/`SECRET`/`USER_ID`
  present, `ZERODHA_TOTP_KEY` absent (so the scheduled login needs it, or the interactive
  `login_and_start.py`), and the cached token expired at 06:00 as expected.
* **The instrument dump is stale** (`NSE_2026-09-30.json`, ~25 h against a 12 h TTL). `doctor.py`
  flags it; `run_morning.py --refresh-instruments` clears it before the first session.
* **The muhurat session (2026-11-08) is still a placeholder.** NSE publishes the window in a
  separate circular; `config/calendar.yaml` documents where the verbatim window goes in
  `special_sessions` and deliberately writes nothing until it can be quoted.

## 7. Verification

* `ruff check` clean on every file touched.
* New tests: 36 passed. Targeted suites for the affected apps
  (`test_capture_run.py`, `test_app_entrypoints.py`, `test_capture_storage_writer.py`,
  `test_check_calendar.py`, `test_daily_pipeline.py`): 121 passed.
* Live smoke, offline: `run_morning.py --dry-run` resolved the trading day and **30** option
  contracts on 2026-10-01, and on the holiday 2026-10-02 stopped at `NO_SESSION` (*"nothing to do;
  the capture is not started"*) with exit `0`; `run_morning.py --dry-run --json` reported
  `"subscriptions": 30`; `--mode backfill-options --dry-run` resolved 30 contracts (0 missing) and
  planned 750 minute requests over 730 days; `doctor.py` produced its full report, its `data
  progress` section echoing the relaxed thresholds (20/45/120/250) straight from the config;
  `refresh_calendar.py --input
  docs/evidence/nse_holidays_2026.json` reported **in sync** (20/20 against the committed evidence);
  `check_calendar.py --require-trading-day --within 0` exited `0` for `--as-of 2026-10-01` and `1`
  for `--as-of 2026-10-02` (Gandhi Jayanti), so the guard is exercised in both directions; and, run
  on the holiday itself, `scripts/run_scheduled.ps1 -Stage morning` printed *"today is not a
  confirmed trading day - morning skipped"* and exited **0** without touching the capture - a
  holiday is a clean no-op, exactly as designed. Every PowerShell wrapper parsed.
* Full `pytest -q` re-run after the changes: **2044 passed, 1 skipped, 0 failed** - exactly the
  pre-change baseline (2008 passed, 1 skipped) plus the 36 new tests. The first sweep caught one
  place that still remembered the old gate: `AccumulationPolicy` (`core/accumulation.py`) and
  `DataGatesConfig` (`core/config_loader.py`) both defaulted `ml_training_min_days` to `60` and are
  now `45`, so the dataclass default, the schema default and `config/model.yaml` cannot drift apart.

The two commands that matter, re-run on the day of this report (2026-10-02, Gandhi Jayanti) so the
holiday path and the gate are both visible in real output rather than described:

```
$ uv run python scripts/run_morning.py --dry-run --json      # a holiday: the guard answers first
{ ...                                     # "..." = absolute paths, elided for width
  "dry_run": true,
  "date": "2026-10-02",
  "trading_day": false,
  "calendar_warnings": [],
  "status": "NO_SESSION",
  "subscriptions": null, "login": null, "capture_exit_code": null }

$ uv run python -m apps.anjaneeya_capture.run --mode capture --dry-run
WARNING | using a STALE instrument dump for NFO (1d old), NSE (1d old) - expiries/strikes
          listed after that date are invisible until a download succeeds
ANJANEYA-CAPTURE dry-run: capture
session date        : 2026-10-02 (NOT a trading day)
reference price     : 22620.45 (index-history:NIFTY 50 (last stored close))
ATM strike          : 22600 (expiries 2026-10-06, 2026-10-13, 2026-10-27)
subscriptions       : 30 option contract(s) + 2 reference instrument(s) in 1 batch(es)
option-chain polls  : 1 request(s) per cycle
index backfill      : 1 planned request(s)
note                : not a trading day: a session would exit immediately
note                : instrument master taken from the on-disk cache
  websocket     : enabled
```

Read together they are the whole design in two lines: the guard stops the day (`NO_SESSION`, exit
`0`, the broker never touched) while the universe behind it still resolves all **30** contracts from
the cached dump — five strikes around ATM 22600 across the 10-06 / 10-13 / 10-27 expiries — so the
gate is proven open (30 ≥ 1) on the same run that proves the calendar is shut.

## 8. Known limitations

* **A missed morning is clean, not recovered.** Both tasks are registered with `StartWhenAvailable`,
  so a machine asleep at 09:05 starts the stage when it wakes — and if that is after the close the
  capture returns `NO_SESSION` (*"the session closed at 15:30"*) with exit `0`
  (`auto_stop_at_session_end: true`). The automation guarantees a *corrupt* day cannot be recorded;
  it cannot record a day the machine slept through, and no later run can back-fill those ticks.
* **Unattended login depends on one secret.** Without `ZERODHA_TOTP_KEY` the scheduled morning stops
  at the login step (exit `1`, session never started) — correctly, but it stops. The interactive
  route (`scripts/login_and_start.py`, or `run_morning.py --request-token …`) needs no TOTP key and
  no secret in the environment; it needs a human at 09:00 instead.
* **The widened universe is a one-way door.** Five strikes × three expiries is what *future* sessions
  will hold. Changing `strikes_offset`/`expiries` again changes the next session, never a past one:
  `--mode backfill-options` can seed contracts Kite still lists, and nothing beyond that exists.
* **The gate resolves from a cache; the session re-resolves after login.** `run_morning.py` is
  installed without `--refresh-instruments`, so the offline gate reads the on-disk dump (12 h TTL)
  and says so out loud (the warning quoted in §7). The two counts can therefore differ. That is safe
  in the direction that matters — the gate only ever refuses on *zero* — but the first probe of the
  day is only as current as the cache, which is why §6 lists one `--refresh-instruments` run before
  the first session.
* **The relaxed numbers are exploratory by construction.** `90/15 → 30/10` and
  `ml_training_min_days 60 → 45` let the *first* walk-forward run on weeks instead of months; they
  did not make weeks enough. `--mode train` still refuses below 45 captured days (exit `4`, and
  `run_daily_pipeline.py` reports `BELOW_GATE` and blocks the backtest step rather than failing),
  everything under `meaningful_backtest_min_days: 120` stays PRELIMINARY, and spec 4.3's seasonality
  gate still needs 250 trading days. `config/model.yaml` says in its own comment to widen them back.
* **`walk_forward.holdout_days: 30` is untouched.** Total days before a first honest fold is
  therefore 30 train + 10 validate + 30 holdout = **70**, not 45. The gate and the holdout answer
  different questions and only the gate was relaxed.
* **The weekly wrapper carries no calendar guard.** `run_weekly.ps1`/`.sh` call
  `run_daily_pipeline.py --weekly` directly; the guard lives in `run_scheduled.ps1`, which is what
  the installed tasks run (`morning` and `postclose` both). Harmless — the weekly chain reads stored
  data — but a hand-run weekly on a holiday is a wasted reprocess, not a lost session.
* **The scheduler is a Windows Task Scheduler installer.** `run_morning.sh` and `run_weekly.sh`
  honour `ANJANEYA_PYTHON` so cron can run the same chain, but only the PowerShell path has an
  installer, an S4U principal and an `install`/`status`/`remove` lifecycle.
* **`refresh_calendar.py` cannot see the two gaps it documents** (see §6): the NSE holiday master
  serves one year at a time, so 2027 is invisible until NSE publishes it, and the muhurat window
  lives in a circular the endpoint does not carry.
* **Nothing above PRELIMINARY is measurable yet.** The database held no captured session when this
  report was written. Every gate, window and report above is correct *and starved* — Phase 9 made the
  data reachable sooner; it did not make the data exist.

## 9. Assumptions made

* **Thirty contracts is a safe subscription count.** Five strikes × three expiries × two sides
  resolves to ~30 tick subscriptions — inside the WebSocket connection limits and the poller's rate
  budget (`capture.rate_limits`) — and both the gate and the session print the number before 09:15,
  so a change is visible at the moment it happens rather than in the coverage ledger the next day.
* **Deciding the universe before the first session is the only moment it is free.** The widening is
  justified by a dry-run only: Kite lists all three expiries today (`30 contract(s)`, `0 missing`), so
  the ladder resolves without a single captured tick, and after the first session the same change
  would cost a permanent hole in the option history.
* **45 / 30 / 10 are exploratory, not fitted.** They were chosen for *reachability* — how few captured
  days can still produce a walk-forward with an untouched holdout — and no data selected them, because
  no data existed. Spec 34's "do not optimise too early" is satisfied trivially here; the live risk is
  the opposite one, and §8 states it.
* **A holiday is exit `0`.** The scheduled wrapper treats a non-trading day as a clean no-op so Task
  Scheduler's history shows success for a day with nothing to do; a red task on every holiday is how
  an operator learns to ignore a red task.
* **09:05 and 15:45 are the right trigger times.** 09:05 is before the 09:15 open and the session's
  own `wait_for_session_open: true` does the waiting; 15:45 is after the 15:30 close and after the
  capture's last write. Both are defaults, movable with
  `schedule_tasks.ps1 install -MorningTime/-PostCloseTime` and no code change.
* **`--write` may only rewrite the `holidays:` block.** `refresh_calendar.py` preserves every comment
  and every other key byte-for-byte, and will not invent a `special_sessions` entry from a summary
  page; a window is entered by hand from the circular, never parsed (Phase 8's reasoning, unchanged).
* **The installer is elevated, idempotent and password-free.** `install` re-registers with `-Force`,
  the principal is S4U (run whether logged on or not, no stored password), and the task settings
  assume nothing about the machine's power or idle policy.
* **`run_scheduled.ps1` is the only entry point the installed tasks call**, so the calendar guard
  (`check_calendar.py --require-trading-day --within 0`) is applied to the post-close stage too, not
  only the morning one.
* **English/IST/NIFTY assumptions** are inherited unchanged from every earlier phase.

## 10. Next phase will implement

Phase 8's §10 ended with *"the next step is a `--mode capture` run on a real trading day"*. Phase 9
did not take that step — it turned the step into one guarded command, wrapped the day in two, and
brought the data those commands need within reach. The master prompt's phase list is still **complete**
(phase 25 shipped at stage 6.1, phase 24 at stage 7.1) and this phase added no new named stage, so
nothing named remains to implement. What is left is execution, in this order:

1. **Set `ZERODHA_TOTP_KEY` in `.env`** so the scheduled login is unattended. `doctor.py` is the
   check — it names the missing key up front rather than letting the morning fail at 09:05.
2. **`run_morning.py --refresh-instruments`** once, to replace the stale dump §7 shows the gate
   reading.
3. **Monday 2026-10-05 — the first real session**: the installed task at 09:05, or by hand
   (`python scripts/daily_login.py`, then `scripts/start_capture.ps1`).
4. **Post-close**: `run_daily_pipeline.py` — candles → greeks → features → labels → validate →
   dataset → train → backtest. Until 45 captured days exist the train step is the normal stopping
   point (`BELOW_GATE`, exit `4`, backtest blocked, not failed).
5. **Then `--mode accumulation` week by week**, until the gates open for real: **45** days for an
   exploratory model, **70** for the first walk-forward with its untouched 30-day holdout, **120/250**
   before a backtest or a seasonality claim is worth believing. Until then `--mode report`,
   `--mode decide` and every artefact they write say PRELIMINARY / BELOW_GATE — which is the
   intended reading, not a defect.

Two items are external and neither is code: NSE's muhurat circular for 2026-11-08 (one
`special_sessions` line once the window can be quoted — §6) and the 2027 holiday list before
2027-01-01 (§8).

A *Phase 10*, if one is ever written, will be defined by the data rather than by the master prompt:
widening the walk-forward windows back to `90/15`, retraining on real accumulation, and reporting what
the first honest out-of-sample fold says — including if it says nothing. The repository is now waiting
on exactly one thing it cannot do for itself: the market opening on a day the calendar calls a trading
day.


