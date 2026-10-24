# Phase 2 report — ANJANEYA-CAPTURE (live acquisition)

**Status: complete.** WebSocket tick capture, option-chain polling and the NIFTY
index backfill are implemented, wired into the CLI, covered by the offline suite
and reproducible end to end against fakes. The live smoke test exists but could
not be executed here (no real `ZERODHA_*` credentials — see *Not verified*).

Delivered against the approved plan, with no changes to the Phase 1 DuckDB schema
(`schema_version` stays **1** — Phase 2 added no columns, so an existing database
needs no migration).

Follow-up folded into this phase: `.env` is now loaded explicitly by every entry
point (`core/env.py`). Before that, a correctly filled-in `.env` was ignored —
the file existed, nothing read it, and `scripts/daily_login.py` failed with
"Missing ZERODHA_API_KEY" even though the credentials were right.

---

## 1. What was built

| Module | Responsibility |
| --- | --- |
| `apps/anjaneeya_capture/instruments.py` | Kite daily dump cache (`data/raw/instruments/<EXCHANGE>_<date>.json`), `InstrumentMaster` lookups, ATM snapping, expiry selection, `OptionUniverse` resolution |
| `apps/anjaneeya_capture/nifty_index_fetcher.py` | REST index backfill/top-up: self-planned windows, adaptive halving on range errors, IST normalisation, idempotent upsert |
| `apps/anjaneeya_capture/storage_writer.py` | Batched DuckDB appends, daily Parquet mirror (part files + compaction), coverage tracker/ledger |
| `apps/anjaneeya_capture/heartbeat.py` | Atomic heartbeat JSON (the only live view while DuckDB is locked) |
| `apps/anjaneeya_capture/websocket_listener.py` | `KiteTickerAdapter`, tick→row conversion, subscription batching, gap detection, stall watchdog + restart |
| `apps/anjaneeya_capture/option_chain_poller.py` | Periodic REST quote snapshots (OI, two-sided quotes), batching, partial-failure tolerance |
| `apps/anjaneeya_capture/session.py` | One session start to finish: window checks, universe, subscribe, poll, monitor, refresh, shutdown, coverage, summary |
| `apps/anjaneeya_capture/run.py` | CLI: `capture`, `option-chain`, `backfill`, `instruments`, `status` + `--smoke`, `--smoke-minutes`, `--until`, `--force-refresh`, `--request-token` |

Supporting changes:

* `core/constants.py` — added `INSTRUMENT_COLUMNS`, `COVERAGE_COLUMNS`,
  `HISTORICAL_INTERVALS`, `INTERVAL_TIMEFRAMES`, `interval_to_timeframe()`, and
  extended `INDEX_HISTORY_COLUMNS` to the DDL's real shape (symbol + timeframe).
* `core/schemas.py` — added `IndexHistoryRow`, `InstrumentRow`, `CoverageRow` and
  the `OUT_OF_SESSION` quality flag; all three are in `CONTRACT_PAIRS`.
* `core/storage/writers.py` — new: the single place that turns row objects into
  SQL (`insert_rows` with append/replace/ignore, identifier validation).
* `core/config_loader.py` — `ZerodhaConfig.capture` (session policy, heartbeat,
  gaps, storage, instruments, historical fetch) + websocket reconnect budget +
  `index_history` intervals; path helpers (`instrument_cache_dir`,
  `heartbeat_path()`, `parquet_dir()`).
* `core/app_bootstrap.py` — `parse_args(configure_parser=...)` hook, `instruments`
  mode registered, optional heartbeat/capture-policy sections in `render_status`.
* `core/env.py` — **new**: the only place a `.env` file is read. Explicit
  (called by `bootstrap_app`, `scripts/daily_login.py`, `scripts/health_check.py`
  *before* the configuration is loaded, so `${VAR}` placeholders resolve and the
  `ZERODHA_*` values reach the log-redaction list); a real environment variable
  always wins; blank values mean "unset"; a missing file is a no-op;
  `ANJANEYA_SKIP_DOTENV=1` disables it (the test suite sets this).
* `scripts/daily_login.py` — one-command Kite login (TOTP or `--request-token`)
  that caches and verifies the token; never prints a secret.
* `scripts/start_capture.{ps1,sh}`, `scripts/run_backfill.{ps1,sh}` — wrapper
  scripts (login → capture; backfill).
* `scripts/health_check.py` — new `capture` section: heartbeat freshness, capture
  policy, coverage ledger, plus a "database is locked, capture is probably
  running" explanation.

---

## 2. Provider facts that shaped the design (verified, not assumed)

All of these were read out of the installed `kiteconnect` SDK or the shipped
configuration before writing code:

1. **Tick timestamps are machine-local, not IST.** The SDK builds
   `exchange_timestamp` with `datetime.fromtimestamp(epoch)`, which yields a
   *naive datetime in the machine's own timezone*. `localize_exchange_timestamp`
   therefore re-interprets it through the machine's zone before converting to
   IST (a no-op when the machine runs IST). Doing the naive thing — treating the
   value as IST — would shift every tick by the machine's UTC offset. Covered by
   a test proving the same wall-clock value means different instants in IST and
   US/Eastern.
2. **The tick payload has no `bid`/`ask`.** They are derived from
   `depth["buy"][0]` / `depth["sell"][0]`; a one-sided book yields `NULL` (never
   a fabricated price), and a crossed book is dropped rather than stored.
3. **Packet lengths decide the fields.** ltp 8B, quote 44B, full 184B, index
   28/32B — only `full` carries OI and depth, which is why `websocket.mode: full`
   is shipped and a non-`full` mode produces a loud session note.
4. **The 1-byte keep-alive never reaches `on_ticks`** (the SDK filters payloads
   `<= 4` bytes). Liveness is therefore tracked on `on_message` —
   connectivity — so a quiet far-OTM strike is never mistaken for a dead socket.
5. **The SDK re-subscribes by itself** (`_on_open` → `resubscribe`, replaying
   token *and* mode). We do not duplicate that; our watchdog only restarts the
   socket when the SDK has *given up* (`on_noreconnect`), which also resets its
   internal attempt counter — necessary for a session to survive a long outage.
6. **`subscribe()` alone marks tokens as QUOTE.** `set_mode` must follow it per
   batch, or a "full" session would silently receive quote packets.
7. **Kite publishes no implied volatility anywhere.** `iv` is stored as `NULL`
   and will be derived from price in a later phase.
8. **Symbol strings are not a contract.** Contracts are resolved by attributes
   (exchange, underlying, expiry, strike, side); `symbol_templates` are a
   cross-check that emits one bounded warning when the dump's format disagrees.

---

## 3. Key design decisions

* **DuckDB append-only + daily Parquet mirror.** Ticks and snapshots are
  append-only; `index_history`, `instruments` and `data_coverage` are keyed and
  therefore idempotent to re-fetch. The mirror lives in the directories Phase 1
  reserved under the raw tree (`data/raw/ticks/<date>/…`, likewise
  `option_chain_snapshots`): immutable `part-*.parquet` files per flush, compacted
  into one `<date>.parquet` per stream at shutdown, so a crash loses nothing and
  a scan stays cheap.
* **Never lose a tick.** If a DuckDB flush fails, the batch is still written to
  Parquet and counted as `spilled_rows`; only a failure of *both* paths is
  reported as `dropped_rows` (and that makes the session `PARTIAL`, exit 1).
* **Universe resolved at runtime, re-centred while running.** Strikes are chosen
  from the live ATM at 09:15 and re-resolved every
  `capture.universe_refresh_minutes`; the refresh is a token *diff*, and it is
  refused when the new window would resolve fewer contracts than the current one
  (a stale or incomplete dump must never shrink coverage).
* **No fixed row expectation for ticks.** Tick streams have no predictable row
  count, so coverage rests on first/last timestamp plus documented gaps
  (`capture.gaps.tick_gap_seconds`, capped logging, rolled into `data_coverage`)
  rather than a fabricated ratio.
* **Session semantics are explicit.** A holiday or an after-close start exits 0
  with status `NO_SESSION`; starting more than `session_start_guard_minutes`
  before the open is refused rather than waiting all morning; a recovered stall
  or a tick-less session is `PARTIAL` and exits 1.
* **Credentials come from the environment, `.env` only fills gaps.** `core/env.py`
  is an explicit call rather than an import-time side effect, and it runs *before*
  `load_config()`, which is what makes `${VAR}` placeholders in the YAML files and
  the log-redaction list see the same values. Precedence is deliberately one-way:
  a shell/CI variable always beats the file, so a `.env` can never surprise a
  pipeline. `ANJANEYA_SKIP_DOTENV=1` turns the file off entirely, and the test
  suite sets it — a developer's real account is unreachable from `pytest`.
* **Offline observability.** `--dry-run` (and the `instruments`/`backfill`
  planning paths) works with no credentials, using the cached dump and its
  previous-close price.


---

## 4. Verification performed

* `uv run pytest -q` → **497 passed, 1 skipped** (the live test, skipped without
  credentials). `uv run ruff check .` → clean.
* `.env` loading verified end to end: `tests/unit/test_env_loader.py` (precedence,
  blank values, missing file, the skip flag, and that no value is ever exposed) and
  `tests/integration/test_env_loading.py`, which drives `health_check`,
  `daily_login` and the capture CLI from a scratch checkout that contains a real
  `.env`. The regression is pinned: without the file the login exits 2
  ("missing credentials"), with it, the same run gets past the credential check and
  stops at `--request-token`.
* `uv run python scripts/health_check.py` → exit 0, with the new capture section.
* CLI smoke (isolated data root, no network): `--mode status`, plus `--dry-run`
  for `instruments`, `backfill`, `capture` and `option-chain` — all offline, all
  showing the resolved universe, ATM strike and request budget.
* Shell/PowerShell wrappers executed: `run_backfill.{sh,ps1} --dry-run`,
  `start_capture.sh --dry-run`, `start_capture.ps1 -SkipLogin --dry-run`.
* `tests/integration/test_capture_pipeline.py` runs a whole session against fakes
  (including a mid-session reconnect) and asserts on DuckDB row counts, Parquet
  contents, the coverage ledger, the heartbeat and the run manifest.
* Credential-less behaviour asserted: `instruments`/`backfill` exit 2 with an
  actionable message; a live session raises `SessionError` (exit 2) instead of
  pretending to capture.

Test layout added: `tests/support/capture.py` (shape-accurate fakes),
`tests/unit/test_capture_{instruments,index_fetcher,storage_writer,websocket,option_poller,heartbeat,session,config}.py`,
`tests/unit/test_schema_contract.py` (DDL ↔ model parity),
`tests/integration/test_capture_{pipeline,scripts}.py`, `tests/live/`.

---

## 5. Not verified (be explicit)

* **No live data was captured.** There were no real `ZERODHA_*` credentials in
  this environment, so the live smoke test
  (`tests/live/test_live_capture_smoke.py`) reports *skipped*. Until it is run
  once with credentials, these remain broker-dependent assumptions:
  * the actual per-interval historical-candle limits (our windows are
    conservative and self-correcting, but the configured values are not measured);
  * the exact `tradingsymbol` format produced by the dump in the current month
    (lookups are attribute-based, so this only affects the template cross-check);
  * connect/subscribe latency and the real tick rate per instrument (which
    decides whether the default gap threshold is well chosen).
* **The automated TOTP login path has still not run against live Zerodha** (same
  reason as Phase 1). Its unit tests cover the request/response handling.
* **The manual `--request-token` login has not been exercised against live
  Zerodha here.** The `.env` → `scripts/daily_login.py` path is covered offline up
  to (but not including) the network call: a scratch checkout with a real `.env`
  proves the credentials are found, and the test then asserts the run stops at the
  step that needs the operator (`--request-token`). No actual `request_token` has
  been exchanged in this environment.
* **Holiday dates in `config/calendar.yaml` remain provisional**; a session on a
  mis-listed holiday would be refused, or run with wrong session bounds.
* **The token-store path is repository-relative** (`./data/raw/.tokens/...`, a
  Phase 1 contract) and therefore does not follow `--data-root`. Left unchanged
  deliberately so an existing user's token file does not move; `--data-root`
  affects everything else.
* **Coverage `expected_rows` is always `NULL` for tick streams** — by design, but
  it means tick coverage is qualitative (gaps) rather than a percentage.

---

## 6. How to use it

```bash
# once: provide the credentials (see README "Credentials"); .env is git-ignored
cp .env.example .env

# once a day - either log in manually first (no password on disk) ...
#   https://kite.trade/connect/login?api_key=<your_api_key> -> copy request_token
uv run python scripts/daily_login.py --request-token <token>
# ... or let the automated TOTP login do it (needs user id/password/TOTP key)
uv run python scripts/daily_login.py

# answer "did my .env get picked up?" in one command (path + names, never values)
uv run python scripts/health_check.py

# see exactly what would happen (offline, no credentials needed)
uv run python -m apps.anjaneeya_capture.run --mode capture --dry-run

# capture one session (or a short smoke test)
uv run python -m apps.anjaneeya_capture.run --mode capture
./scripts/start_capture.sh --smoke

# build the index history, then top it up daily
uv run python -m apps.anjaneeya_capture.run --mode backfill

# while capture is running
cat data/runs/heartbeat/capture.json
```

Phase 2 writes: one row per tick in `ticks`, one row per contract per poll in
`option_chain_snapshots`, one row per (day, symbol, timeframe) in
`data_coverage`, the instrument master in `instruments`, index bars in
`index_history` — plus one Parquet file per stream per day under
`data/raw/<stream>/<date>.parquet`.

---

## 7. What Phase 3 (engine) can rely on

* `ticks` and `option_chain_snapshots` are append-only with full provenance
  (`source`, `fetch_timestamp`, `quality_flag`, `run_id`).
* `index_history` holds real OHLCV for the index at `1d` (from the backfill) and
  `1m` (from the daily top-up), keyed and idempotent.
* `data_coverage` states, per day/symbol/timeframe, how much was captured — the
  data gates in `model.yaml` can now be measured rather than guessed.
* Ticks outside the session are stored with `quality_flag = OUT_OF_SESSION`
  (or dropped, per configuration) and must be excluded from session analysis.
* Bid/ask and OI exist only for `full`-mode option ticks; anything built on them
  must tolerate `NULL` rather than assume a two-sided book.

---

## 8. Follow-ups deliberately left for later

* IV derivation (from price, via the option greeks module) — Phase 4+.
* A `recover` command that re-imports Parquet part files into DuckDB after a
  spill (today the files are on disk and readable, but there is no importer yet).
* Measuring the per-interval historical limits and tightening
  `capture.historical.max_days_per_request` once real requests have been made.
* A convenience wrapper that opens the Kite login URL and extracts
  `request_token` from the redirect automatically. Today the manual login is a
  documented ~30-second browser step (and the *only* remaining manual step before
  a capture session, since no broker password is stored on disk).

