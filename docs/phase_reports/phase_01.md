# ANJANEYA — Phase 1 completion report

**Phase:** 1 of the multi-phase roadmap — skeleton, configuration, foundations
**Date:** 2026-09-29
**Status:** COMPLETE (verified by the commands at the end of this document)
**Runtime:** Python 3.13.12, uv 0.11.2, Windows x86_64

---

## 1. Objective

Create the project skeleton that every later phase builds on:

* six fully validated YAML configuration files (no parameter may be hard-coded),
* the shared `core` library: column contracts, row schemas, configuration
  loader, NSE market calendar, DuckDB schema and bootstrap, logging with secret
  redaction, run manifests,
* the Zerodha Kite Connect access layer: credentials, TOTP login, request-token
  exchange, token cache, rate limiter, retrying REST client,
* three application entry points (`capture`, `engine`, `ai`) with a working
  `--mode status`,
* an offline test suite and operator tooling.

Deliberately **out of scope**: any actual market-data capture, indicator
computation, model training or backtesting. Those modes exist as CLI choices but
exit with code 3 and an explanation until their phase lands.

---

## 2. Deliverables

### 2.1 Repository skeleton

```
pyproject.toml            uv/setuptools project, pinned deps, pytest+ruff config
.python-version           3.13
.env.example              every environment variable the project reads
.gitignore                raw data, tokens, database and artifacts never committed
README.md                 purpose, quickstart, rules, honest caveats
config/                   settings, data_sources, option_universe, indicators,
                          model, calendar
core/                     constants, schemas, config_loader, logging_setup,
                          run_context, app_bootstrap, rate_limiter,
                          market_calendar/, storage/, auth/ + 8 package skeletons
apps/anjaneeya_capture/   run.py (+ package init)
apps/anjaneeya_engine/    run.py (+ package init)
apps/anjaneeya_ai/        run.py (+ package init)
tests/unit/ (9 files)     constants, config, schemas, calendar, rate limiter,
                          auth, client, run context, logging
tests/integration/ (2)    DuckDB bootstrap + DDL/model parity; app entry points
scripts/health_check.py   offline installation diagnostic
docs/phase_reports/       this report
```

### 2.2 Configuration

| File | Contents | Validation highlights |
| --- | --- | --- |
| `settings.yaml` | project metadata, time policy, session bounds, timeframes, data paths, logging, run tracking | session ordering, `analysis_end` inside the session, supported timeframes, log levels |
| `data_sources.yaml` | Zerodha credential **names**, timeouts, rate limits, token store, retry, websocket, option-chain polling | all limits > 0, `safety_factor ∈ (0,1]`, sane retry status codes |
| `option_universe.yaml` | underlying, strike offsets, expiries, ATM rule, symbol templates, reference instruments, index history | strikes are multiples of 50, month codes complete, expiry selectors non-empty |
| `indicators.yaml` | slope/persistence/spread lookbacks, ATR, three NIFTY MA combinations, option trend, candle rules, levels, S/R weights | score weights sum to 1.0, MA names unique with roles assigned, S/R weights sum to 1.0, touch < approach |
| `model.yaml` | horizons, algorithms, data gates, labels, walk-forward, Indian costs, execution, evaluation | gates non-decreasing, embargo ≥ longest horizon, horizons unique and sorted |
| `calendar.yaml` | sessions, expiry weekday + holiday roll, holidays, DTE definition, cycle phases, time-of-day buckets | no duplicate holidays, buckets non-empty/unique/non-overlapping |

Environment placeholders (`${VAR}`, `${VAR:-default}`) are resolved from the
process environment; a missing variable without a default is a hard error, and
`.env.example` parity is enforced by a test.

### 2.3 Core library

| Module | Responsibility | Notable safeguards |
| --- | --- | --- |
| `constants.py` | timezone helper, timeframe map, column contracts | resolves `Asia/Kolkata`; raises a Windows-specific hint when `tzdata` is missing |
| `schemas.py` | `TickRow`, `OptionChainSnapshotRow`, `CandleRow`, `FeatureRow`, `LabelRow`, `TradeRecord`, `RunManifest`, enums | tz-aware timestamps only, complete-or-absent contract identity, OHLC invariants, no bid above ask, label fields forbidden in feature rows, DDL/model parity asserted at import |
| `config_loader.py` | typed configuration, env substitution, config hashing | `extra="forbid"` everywhere; cross-field invariants; SHA-256 of raw YAML for reproducibility |
| `market_calendar/` | sessions, holidays, expiry maths, DTE, cycle phase, buckets | naive datetimes rejected; holiday-roll policies honoured; buckets must tile the analysis window |
| `storage/` | `schema.sql` (11 tables) + idempotent bootstrap | single source of DDL, schema-version bookkeeping, refuses a newer on-disk schema, IST session timezone |
| `auth/` | credentials, TOTP login, request-token exchange, token cache, REST client | secrets never logged; masked `describe()`; 403 → actionable `SessionExpiredError` |
| `rate_limiter.py` | token buckets per endpoint class + exponential backoff | injectable clock/sleep; 429 pauses the whole bucket; non-transient errors are not retried |
| `logging_setup.py` | loguru sinks (console, per-app file, optional JSON-lines) | secret redaction by key **and** by value; idempotent reconfiguration; stdlib interception |
| `run_context.py` | `run_id`, config hash, git commit, package versions, manifest | failed runs also write a manifest |
| `app_bootstrap.py` | shared startup for the three apps | deterministic order, dry-run support, run manifest recorded in DuckDB, consistent exit codes |

### 2.4 Application entry points

All three apps expose the same CLI:

```
--mode {status, <app modes>}  --config-dir  --data-root  --db-path
--log-level  --dry-run  --no-run-artifacts  --json
```

* `--mode status` prints configuration, calendar, database and credential
  *presence* state, creates the directories and the DuckDB file, writes a run
  manifest, and performs **no network access**.
* Unimplemented modes exit `3` after printing what is planned and where.
* Invalid configuration exits `2` with the validation error.

Exit codes: `0` ok, `1` runtime failure, `2` configuration error,
`3` not implemented yet.

---

## 3. Verification performed

| Check | Command | Result |
| --- | --- | --- |
| Environment / timezone | `python -c "ZoneInfo('Asia/Kolkata')"` | resolves (tzdata 2026.4 installed) |
| Dependency versions | in-venv import check | pandas 2.3.3, duckdb 1.5.6, pydantic 2.13.5, kiteconnect 5.x |
| Config validation | `load_config()` | all six files load; summary printed |
| Calendar | `NseCalendar(...).describe()` | Tuesday expiry, 15 holidays (8 provisional, flagged), 5 buckets |
| DuckDB bootstrap | `bootstrap_database(tmp)` | 11 tables created; a second run is a no-op |
| Health check | `python scripts/health_check.py` | RESULT: HEALTHY (exit 0) |
| Status modes | `python -m apps.<app>.run --mode status` ×3 | exit 0, full report |
| Planned modes | `--mode capture` / `build-candles` / `train` | exit 3 with an explicit explanation |
| Test suite | `pytest -q` | **230 passed** |
| Coverage (Phase-1 modules only) | `pytest --cov=core.schemas --cov=core.config_loader ...` | 94% statements, 84% branches |
| Lint | `ruff check .` | clean |
| DDL/model parity | `pytest tests/integration -k parity` | ticks, candles, features, labels, trades match their pydantic models exactly |
| Concurrency guard | `pytest -k locked` | a second process gets a clear "single writer" error, not a raw IO failure |

### Bugs found and fixed during verification

1. **`weekly_expiry.weekday` shipped as `Thursday`.** NSE had moved the NIFTY weekly
   expiry to Tuesday by the time the first dump was fetched, so the capture plan
   resolved zero option contracts while still exiting `0`. Corrected to `Tuesday`
   on 2026-09-30 against `data/raw/instruments/NFO_2026-09-30.json` (every listed
   NIFTY expiry is a Tuesday, and the Monday expiries are holiday rolls). The
   weekday stays configuration; `config/calendar.yaml` carries the evidence.
2. **`monthly_expiry` was wrong for months that end before the expiry weekday.**
   September 2026 ends on a Wednesday, so "the Thursday of the last week" landed
   on 1 October. Now the last weekly expiry *inside* the month wins.
3. **`START_OF_CYCLE` counted the expiry day itself**, making the first session
   of a new cycle look like mid-cycle. The count now starts strictly after the
   previous expiry.
4. **Request-token parsing stopped at `-` or `_`.** The regex now accepts any
   non-`&` character.
5. **Token buckets smaller than one request could never serve a request**
   (capacity is now at least one token).
6. **Stdlib log records crashed the sinks** with `KeyError: 'app'`; the app name
   is now a default `extra`.
7. **`enqueue=True` made file sinks lossy in short-lived runs** — `enqueue` is
   now opt-in, so output is deterministic by default.
8. **Importing `kiteconnect` opens a loopback socket** (Twisted reactor
   install), which the network guard initially blocked; the guard now allows
   loopback only.
9. **`Sequence` was missing from the client's imports** (surfaced by `ruff`
   F821); with `from __future__ import annotations` this was invisible at
   runtime but would have broken type checking.
10. **A locked DuckDB file produced a raw IO error.** DuckDB allows one writing
   process per file, so running two ANJANEYA apps at once failed with
   "being used by another process". That is now mapped to a
   `DatabaseLockedError` with an actionable message (and covered by a
   cross-process test).
11. **`current_monthly_expiry` in the status payload could name an expired
   contract.** `describe()` reported the *calendar month's* monthly expiry, so on
   2026-09-30 it printed `2026-09-29` under a `current_` key while the weekly key
   next to it rolled forward. The rule now lives once, on the calendar
   (`NseCalendar.current_monthly_expiry`, used by `core.options.universe` too),
   and `describe(today=...)` is injectable so the value can be tested per date.

---

## 4. Knowingly incomplete / unverified

* **Automated Kite login is untested against live Zerodha.** No credentials were
  available. The flow is fully unit-tested against a stubbed HTTP layer and uses
  the documented `/api/login` → `/api/twofa` sequence, but the first live run
  should be treated as a real test. The `live` marker exists for exactly this
  (`ANJANEYA_RUN_LIVE=1`).
* **Holiday dates are provisional** (`holidays_verified: false`): 7 fixed-date
  holidays are high confidence, 8 festival dates are marked provisional and are
  surfaced by the health check and in every manifest.
* **Cost-model constants are baseline values** taken from documented Indian F&O
  rates; re-verify before trusting any P&L.
* **`notebooks/` is empty** until the first exploratory-analysis phase.
* **No WebSocket code yet** — `KiteTicker` wiring arrives with the capture phase.
* **No indicators, features, labels, models or backtests yet.** Everything under
  `core/indicators`, `core/nifty`, `core/options`, `core/price_action`,
  `core/features`, `core/labels`, `core/ml`, `core/backtest` and
  `core/visualization` is a docstring-only placeholder that fixes the module
  boundary for later phases.

---

## 5. What the next phase builds on

Later phases can rely on:

* `bootstrap_app()` — identical startup, logging, run context and exit codes for
  every app, so no phase re-implements plumbing;
* `AnjaneyaConfig` — one validated object carrying every parameter, hashed per run;
* `NseCalendar` — expiry, DTE, cycle-phase and bucket maths already correct and
  tested;
* `ZerodhaClient` — rate limiting, retry, timeouts and session-expiry handling in
  place, with an injectable SDK for tests;
* the DuckDB schema plus schema-version discipline for tick/candle/feature stores;
* the offline test discipline (no network, isolated paths) that keeps later
  phases fast and honest.

