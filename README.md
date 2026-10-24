# ANJANEYA

**NIFTY50 Options AI Research Engine** — a research and backtesting platform for
studying *when* NIFTY option contracts move, *why* they move, and whether that
behaviour is repeatable under realistic Indian transaction costs.

> **ANJANEYA is a research system, not a trading system.**
> It never places orders, never talks to an order-management API and never
> emits BUY/SELL signals. Its only outputs are measured observations about
> market behaviour, plus an explicit statement of how much data those
> observations rest on.

---

## Status: Phase 9 complete (Phase 9 automated the daily routine: one installer + two commands a day, a supervised morning routine, a weekly cadence, a pre-flight doctor, an NSE calendar refresher and a 2-year index backfill)

| Area | State |
| --- | --- |
| Configuration system (6 YAML files, fully validated) | ✅ done |
| Shared `core` library, market calendar, DuckDB schema, logging, run manifests | ✅ done |
| Zerodha auth (TOTP login + request token + token cache), rate limiter, REST client | ✅ done |
| Three app entry points with `--mode status` | ✅ done |
| Instrument master (daily dump cache, attribute-based universe resolution) | ✅ done |
| WebSocket tick capture (reconnect, stall watchdog, depth→bid/ask, IST timestamps) | ✅ done |
| Option-chain OI/quotes polling + NIFTY index backfill/top-up (REST) | ✅ done |
| Batched DuckDB writes + daily Parquet mirror + coverage ledger + heartbeat | ✅ done |
| **Candles: 1m from ticks, 3m/5m/15m merged from 1m, coverage + Parquet mirror** | ✅ Phase 3.1 |
| **Indicator library (SMA/EMA, ATR, slope, spread, persistence, statistics)** | ✅ Stage 3.2a |
| **Price-action engines (candle behaviour, swings, structure, levels)** | ✅ Stage 3.2b |
| **Level interaction state machine + support/resistance strengths** | ✅ Stage 3.2c |
| **NIFTY trend engine (MA combinations, scores, composite, regime) + option trend** | ✅ Stage 3.2d |
| **Unified feature row (specs 8-16) + spec 19 validation + JSON report** | ✅ Stage 3.2e |
| **IV / greeks enrichment (Black-Scholes, spec 16) written onto stored candles** | ✅ Stage 3.3 |
| **Option universe (spec 17): ATM anchor, strike ladder, expiries, `--mode universe`** | ✅ Stage 3.4a |
| **Labels (spec 20): forward returns, excursions, trend change, discretised class** | ✅ Stage 3.4b |
| **Training matrix (spec 21) + purged walk-forward splits (spec 22), `--mode dataset`** | ✅ Stage 4.1 |
| **LightGBM trainer + spec 21 baselines + probability metrics, `--mode train`** | ✅ Stage 4.2 |
| **Event-driven backtest: next-open fills, spec 23 Indian cost model, trade ledger, `--mode backtest`** | ✅ Stage 4.3 |
| **One-command daily pipeline (`run_daily_pipeline`) + calendar guard (`check_calendar`) + operator runbook** | ✅ Stage 4.3 |
| **Research report (specs 24-25): time-of-day / cycle-phase matrices, interactions, holding time, ATR profile, intraday path, corrected significance, `--mode report`** | ✅ Stage 4.4 |
| **Explainability (spec 26): per-prediction blocks from the stored row + the model's own class probabilities, `--mode explain`** | ✅ Stage 5.1 |
| **Feature importance (spec 27): native gain + permutation importance with confidence intervals per feature and per source, optional SHAP, `--mode importance`** | ✅ Stage 5.2 |
| **Visualization (spec 28): P&L heatmaps, CE/PE curves, regime-conditional performance, importance bars and NIFTY / option price charts, `--mode charts`** | ✅ Stage 5.3 |
| **Continuous data-accumulation report (spec 29): captured days, coverage and gaps, sample sizes and readiness for retraining, week by week, `--mode accumulation`** | ✅ Stage 6.1 |
| **Decision layer (spec 24 / master-prompt phase 24): `--mode decide` records what one configured rule would do with every stored bar - `ENTER_CE`/`ENTER_PE`/`NO_TRADE`, a reason code, the size and the stated exit plan, under `data/predictions/`** | ✅ Stage 7.1 |
| **Calendar reconciled against NSE's own 2026 F&O holiday list (20 dates, `holidays_verified: true`, raw response committed as evidence), and an explicit special session now outranks the weekend and holiday defaults** | ✅ Stages 8.1-8.2 |
| **Daily automation: supervised `run_morning` (calendar guard → universe gate → login → capture), opt-in `--weekly` cadence steps, `run_scheduled.ps1` + `schedule_tasks.ps1` (weekday tasks), `doctor.py` pre-flight and `refresh_calendar.py`** | ✅ Stage 9.1-9.4 |
| **Widened capture universe (5 strikes × 3 expiries), a ~2-year index backfill (`backfill_days: 730`, `--years`), an active-contract `--mode backfill-options` bootstrap, and relaxed exploratory walk-forward windows (30/10/15, `ml_training_min_days: 45`)** | ✅ Stage 9.5 |

Capture modes are **gated on real credentials**: without a Kite session they stop
with `exit 2` and an actionable message instead of pretending to record anything.
`--mode build-candles`, `--mode enrich-greeks`, `--mode universe`,
`--mode build-features`, `--mode build-labels` and `--mode validate` (engine) and
`--mode dataset` / `--mode train` / `--mode backtest` / `--mode report` /
`--mode explain` / `--mode importance` / `--mode charts` / `--mode accumulation` /
`--mode decide` (AI) need **no
credentials and no network** - they only read what is already stored (ticks, candles,
features, labels, the NIFTY index history, a model artefact, a trade ledger, an
importance artefact, the coverage ledger).
`--mode train` refuses to run at all below spec 4.3's exploratory gate (`exit 4`) unless
you pass `--force`; `--mode backtest` and `--mode decide` never refuse, they *label* the run
(`BELOW_GATE`) and print the gate it fell short of; `--mode report` labels every small cell
`PRELIMINARY` and refuses to test it at all.

---

## What the system will do (roadmap in one screen)

1. **ANJANEYA-CAPTURE** — record NIFTY index + option ticks over the WebSocket,
   poll option-chain OI/IV snapshots, backfill index history via REST.
2. **ANJANEYA-ENGINE** — turn ticks into candles, then into the engines under
   study (NIFTY trend scores, option trend, candle behaviour, levels &
   level-events, swings & structure, support/resistance) and finally into one
   unified feature row per timestamp.
3. **ANJANEYA-AI** — build leak-free datasets, train with purged walk-forward
   validation, backtest with Indian costs, report findings — including
   negative ones — explain each prediction in plain structured form, rank where
   the model's information came from, and draw the figures that make it
   checkable by eye.

Two research questions are answered explicitly throughout:

* **RQ1** — does time-distance from weekly expiry change option behaviour?
* **RQ2** — does the intraday time bucket change option behaviour?

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (the project pins Python 3.13 via
`.python-version`).

```bash
# 1. create the environment (installs tzdata - mandatory on Windows)
uv sync --extra dev

# 2. check the installation without touching the network
uv run python scripts/health_check.py

# 3. see what each app reports
uv run python -m apps.anjaneeya_capture.run --mode status
uv run python -m apps.anjaneeya_engine.run  --mode status
uv run python -m apps.anjaneeya_ai.run      --mode status

# 4. run the offline test suite
uv run pytest -q
```

### The whole system in three commands

`scripts/anj.py` is the front door. It re-implements nothing - it delegates to the scripts
below - so this is the only surface an operator has to remember:

```bash
uv run python scripts/anj.py history --years 2   # once: instruments + 2y NIFTY index + option history
uv run python scripts/anj.py day                 # each trading day: the session, then the math + suggestion
uv run python scripts/anj.py learn               # each evening: train on everything, backtest, report
```

`day` deliberately stops at the dataset (`--skip train,backtest`): a model trained before the
close is trained on an incomplete day. `--no-capture` runs the post-close math alone (a
holiday, or after the fact) and `--dry-run` plans every child without writing anything. On a
non-trading day the calendar guard prints `NO_SESSION` and exits `0`. Exit codes are the
vocabulary every app already shares: `0` ok, `1` a step failed, `2` configuration, `4` the
accumulation gate refused.

### The same three verbs in a browser (optional)

`scripts/anj_ui.py` is a Streamlit console over that facade - a thin client, not a second
implementation: every button spawns the same `anj.py` command you could type, and the CLI stays
the source of truth. It is an optional extra so the capture session, the engine and the AI app
never need a web framework:

```bash
uv sync --extra ui
uv run streamlit run scripts/anj_ui.py
```

`uv sync` replaces the environment, so add `--extra dev` (`uv sync --extra ui --extra dev`)
if you want `pytest` and `ruff` back in the same virtualenv.

| Tab | What it does |
| --- | --- |
| Token | the cached token masked, its expiry, a TOTP refresh, and a `request_token` paste box (exchanged in-process, never logged) |
| History | `anj.py history` - `Plan` is `--dry-run`, `Fetch` is the real thing, both stream into a log |
| Data | first/last 10 rows and the row count of one table, read-only, handle closed before the page draws |
| Run | `day`, `day --no-capture`, `learn`, with switches that become `learn`'s `--skip` |
| Results | the newest reports and charts under `data/reports/` |

Two rules the console holds to: **one job at a time** (DuckDB has one writer, so a second click
is refused instead of locking the database) and **no secret on screen** (the token is masked, the
`request_token` box is a password field). A job runs detached and writes its own status file, so
closing the browser tab does not kill a running backfill. The switches only reshape `learn`: the
`indicator` and `price_action` work is computed *inside* `features` and the project places no
orders, so those switches are shown disabled with the reason rather than faked.

Everything below remains available and unchanged.

### The daily routine: one installer, two commands a day

```bash
# once (Windows): register the weekday Morning + PostClose tasks (calendar-gated)
pwsh ./scripts/schedule_tasks.ps1 install

# every morning - one guarded command (or let the Morning task run it)
uv run python scripts/run_morning.py
uv run python scripts/run_morning.py --dry-run              # plan only, no network
uv run python scripts/run_morning.py --refresh-instruments  # odd the dump first

# every evening, after 15:35
./scripts/run_daily_pipeline.sh

# weekly: the daily chain plus report -> explain -> importance -> charts -> accumulation
./scripts/run_weekly.sh

# any time something looks wrong
uv run python scripts/doctor.py          # readiness: token, dump age, db, scheduler
uv run python scripts/refresh_calendar.py --write   # reconcile the calendar with NSE
```

`run_morning.py` refuses the two ways a day is lost silently: on a non-trading day it
prints `NO_SESSION` and exits `0` without touching the broker, and when the option
universe resolves to **zero** contracts it exits `4` **without starting the session**
(the index-only-capture trap). `schedule_tasks.ps1` wraps both stages in
`scripts/run_scheduled.ps1`, which gates on `check_calendar.py --require-trading-day
--within 0`, so a task firing on a holiday is a clean no-op. Exit codes: `0` ok, `1`
login/capture failed, `2` configuration, `4` the universe was empty.

### Recording a session (ANJANEYA-CAPTURE)

```bash
# 0. once per day: log in and cache an access token
uv run python scripts/daily_login.py

# 1. see exactly what a session would subscribe to, without calling the broker
uv run python -m apps.anjaneeya_capture.run --mode capture --dry-run

# 2. capture one session (waits for the 09:15 open; stops at 15:30)
uv run python -m apps.anjaneeya_capture.run --mode capture
#    ... or use the wrappers, which log in first:
./scripts/start_capture.sh --smoke          # 3-minute smoke session
./scripts/start_capture.sh --until 15:30
pwsh ./scripts/start_capture.ps1 -SkipLogin --dry-run

# 3. build the long NIFTY index history (run once; ~1 request per 400 days)
uv run python -m apps.anjaneeya_capture.run --mode backfill
./scripts/run_backfill.sh --dry-run

# 4. is the calendar safe? (0 = clear, 1 = a provisional holiday could cost a session)
uv run python scripts/check_calendar.py --days 30
```

### One command for the day (after the close)

The whole post-close chain - candles, greeks, features, labels, validate, dataset, train,
backtest - in one command, with one dated log and one verdict:

```bash
./scripts/run_daily_pipeline.sh                 # the newest day with stored ticks
./scripts/run_daily_pipeline.sh --dry-run       # plan all eight steps, write nothing
./scripts/run_daily_pipeline.sh --force         # train below spec 4.3's gate (smoke tests)
./scripts/run_daily_pipeline.sh --skip validate,backtest
```

Exit codes: `0` all good, `1` a step failed, `2` configuration problem, `4` only the training
gate refused (`--force` overrides it). Run it **after** the capture session has exited -
DuckDB allows one writer. `docs/runbook.md` has the morning routine, the failure playbook,
scheduling snippets and the answer to "when will there be a model?".

While a session runs, `--mode status` **cannot** read DuckDB (DuckDB allows one
read-write process). Use the heartbeat file instead:

```bash
uv run python -m apps.anjaneeya_capture.run --mode status   # shows the heartbeat
cat data/runs/heartbeat/capture.json
```

### Building candles (ANJANEYA-ENGINE)

No credentials, no network - the engine only reads what capture already stored.

```bash
# what would be built, and how big a complete session is (writes nothing)
uv run python -m apps.anjaneeya_engine.run --mode build-candles --dry-run

# build the most recent day that has ticks
uv run python -m apps.anjaneeya_engine.run --mode build-candles

# one specific day / one contract / a subset of timeframes
uv run python -m apps.anjaneeya_engine.run --mode build-candles --date 2026-09-29
uv run python -m apps.anjaneeya_engine.run --mode build-candles --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_engine.run --mode build-candles --timeframes 1m,5m

# every day that has ticks, oldest first (machine-readable summary)
uv run python -m apps.anjaneeya_engine.run --mode build-candles --all --json
```

What it does, per `(day, contract)`:

1. reads the day's **`quality_flag = 'OK'`** ticks inside the calendar's session
   window (a `SUSPECT`/`STALE`/`OUT_OF_SESSION` tick can never shape a candle);
2. builds **1m bars keyed by their open time in IST** (the 09:15 bar holds
   `[09:15:00, 09:16:00)`);
3. merges **3m/5m/15m from those 1m bars** (never a second pass over ticks), so
   the 15m open is always the first 1m open;
4. **replaces** the affected `(day, timeframe, contract)` scope in `candles`
   (delete-then-append: a shorter rebuild can never leave the previous run's
   later bars behind);
5. writes a **`data_coverage`** verdict per `(day, contract, timeframe)` with the
   exact expected bar count (375/125/75/25 for a normal session) and the gaps;
6. mirrors the result to `data/candles/<timeframe>/<YYYY-MM-DD>.parquet`.

Exit codes: `0` ok (including "nothing to do"), `1` some contract failed (the run
summary names it; the others are still written), `2` bad configuration.

### Pricing them: IV and greeks (ANJANEYA-ENGINE)

The capture feed carries no greeks, so ANJANEYA computes them: `--mode enrich-greeks`
solves each option bar's **implied volatility** from the configured price field against
the **NIFTY level at the bar's close instant**, derives `delta`/`gamma`/`theta`/`vega`
from it, and `UPDATE`s exactly those columns (plus `days_to_expiry` and
`distance_from_atm`) onto the **stored bars**. It never rebuilds an OHLCV row.

```bash
# which contracts, how many bars, and the model's assumptions (writes nothing)
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --dry-run

# price the most recent day that has candles (all configured timeframes)
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks

# one day / one contract / a subset of timeframes / machine-readable summary
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --date 2026-09-29
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --timeframes 1m,5m --json
```

The model (spec 16's "document it" clause, and the same text lives in
[`core/options/greeks.py`](core/options/greeks.py)):

* **Black-Scholes-Merton, European, cash-settled**, with a continuous dividend yield.
* **Risk-free rate is a configured constant** (`indicators.greeks.risk_free_rate`,
  shipped as 6.5%), *not* a fetched curve: an IV is only comparable across runs if the
  rate that produced it is pinned. Every run prints the assumptions it used, and a
  changed assumption is a **rebuild of the greeks**, not a silent re-interpretation.
* **Dividends are ignored** (`dividend_yield: 0.0`). NIFTY options are in practice priced
  off the futures, so a zero-dividend *spot* model absorbs the basis into the IV. That is
  a documented approximation, not an accident.
* **`T` runs from the bar's close instant to the expiry session end** (ACT/365) and is
  floored at one session, so the last bars of an expiry day cannot divide by ~zero.
  `days_to_expiry` stays the **calendar's** value (a `calendar.yaml` setting).
* **A price with no IV stays `NULL`.** A quote below intrinsic (or above the underlying)
  has no implied volatility inside the model's bracket, and the run prints how many bars
  that was, side by side with `max |price - BS(iv)|` - the solver's own self-check.
* Columns are quoted in fixed units: delta per index point, gamma per point squared,
  **vega per 1 vol point**, **theta per calendar day**, `iv` as an annualised fraction.

Order of operations for a fresh day:

```bash
uv run python -m apps.anjaneeya_engine.run --mode build-candles    # ticks -> bars
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks    # bars -> iv/greeks
uv run python -m apps.anjaneeya_engine.run --mode build-features   # bars -> feature rows
uv run python -m apps.anjaneeya_engine.run --mode validate         # spec 19 over the rows
uv run python -m apps.anjaneeya_engine.run --mode build-labels     # bars -> labels (separate pass)
uv run python -m apps.anjaneeya_ai.run     --mode dataset          # features + labels -> matrices
```

A candle rebuild replaces the bar, so its greeks must be re-derived: run
`build-candles` again and the affected bars go back to `NULL` until `enrich-greeks`
runs for that scope. `build-features` publishes whatever is there - an unenriched bar
simply has no `iv`, and with no `delta` the expected move and residual stay `NULL`
rather than being invented.

### Building the feature rows and validating them (ANJANEYA-ENGINE)

`build-features` stitches every engine (NIFTY trend/regime, option trend, candle
behaviour, levels, level-events, structure, support/resistance, option context) into
**one row per `(timestamp, contract)`**; `validate` re-reads what was stored and runs
spec 19's checks over it, including a no-look-ahead recomputation.

```bash
# what would be built, from which window, and whether the NIFTY index is there
uv run python -m apps.anjaneeya_engine.run --mode build-features --dry-run

# build the most recent day's candles into feature rows (default timeframe: the
# first configured one, i.e. 1m)
uv run python -m apps.anjaneeya_engine.run --mode build-features

# a specific day, contract or timeframe (exactly one timeframe per run)
uv run python -m apps.anjaneeya_engine.run --mode build-features --date 2026-09-29 --timeframes 5m
uv run python -m apps.anjaneeya_engine.run --mode build-features --symbol NIFTY01OCT2625000CE

# spec 19 over the stored rows: 3 no-look-ahead probes per contract, JSON report
uv run python -m apps.anjaneeya_engine.run --mode validate --probes 3
uv run python -m apps.anjaneeya_engine.run --mode validate --json > validation.json
```

What `build-features` does, per `(day, contract)`:

1. reads that contract's candles from **`max(session - required_lookback_bars - one
   session, its first stored bar)`** to the session close (2062 1m bars of warm-up on
   the shipped config, so the long NIFTY averages and the level state machine have
   real history);
2. reads the **NIFTY index** rows for the same window, resampling the 1m series onto
   the feature timeframe when a coarser one is not stored;
3. computes the contract block and the NIFTY block, and joins them by **bar close
   instant** (a 15m index bar can never reach a 1m candle before it has closed);
4. keeps only the bars inside the session and **replaces the whole contract-day** in
   `features` (the table's key has no timeframe, so a day is authoritative for one
   series);
5. writes one **`data_coverage`** verdict per `(day, underlying, timeframe)` (keyed by
   `NIFTY`, so the per-contract candle verdicts are never overwritten) and mirrors the
   rows to `data/features/<timeframe>/<YYYY-MM-DD>.parquet`.

`validate` writes `data/reports/feature_validation/<YYYY-MM-DD>_<timeframe>.json` and
exits `1` when any check **fails** (a warning - an all-`NULL` column, a session gap -
does not fail a run). It never writes to the database: it reads the stored rows plus
the inputs they were built from.

### The forward-looking half: labels (ANJANEYA-ENGINE, spec 20)

A feature is a function of the candles *up to* a moment; a **label** is a function of the
candles *after* it. They are computed in **two separate passes** on purpose - mixing them
in one is how look-ahead bias gets into a research pipeline - and
`core.schemas.FeatureRow` structurally refuses the label columns, so the separation is
enforced by the row contract rather than by discipline.

```bash
# what would be built, at which horizons, from which warm-up (writes nothing)
uv run python -m apps.anjaneeya_engine.run --mode build-labels --dry-run

# label the most recent day that has candles (exactly one timeframe per run)
uv run python -m apps.anjaneeya_engine.run --mode build-labels
uv run python -m apps.anjaneeya_engine.run --mode build-labels --date 2026-09-29 --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_engine.run --mode build-labels --json > labels.json
```

Every session bar of every contract with candles gets **one row per configured horizon**
(5/10/15/30 minutes in the shipped `config/model.yaml`) in the `labels` table:

| Column | Unit | Meaning |
| --- | --- | --- |
| `future_return` | relative | `close[t+h] / close[t] - 1` |
| `future_max_up_move` | price | the best level the market reached in the window, `>= 0` |
| `future_max_down_move` | price | the worst level reached, `<= 0` |
| `future_trend_change` | threshold units | the normalised forward move minus the normalised move into `t` |
| `label` | class | `UP_CONTINUATION` / `DOWN_CONTINUATION` / `RANGE` / `REVERSAL` |

The threshold is configured (`normalization: atr` = `0.5 * ATR(14)`, Wilder; `percent` and
`fixed` also work) and is always read **at or before `t`**, so the forward window can never
influence the scale it is judged by. Three rules keep a label honest:

* it needs its **whole forward window inside one session** (`t + horizon <= session_end`,
  every minute present) - the last bars of a session carry `NULL` label fields rather than
  a move that spans the overnight gap, and the row still exists so the hole is a fact
  rather than a silence;
* a continuation has to continue something: the prior move is `close[t] -
  close[t - trend_lookback]`, and with no prior bar (or a flat one) no continuation is
  claimed;
* a threshold that does not exist yet (ATR warm-up) leaves the class `NULL` while the
  forward *numbers* are still published.

The pass writes the `labels` table (delete-then-append per contract-day, all horizons), one
`data_coverage` verdict per contract spelled `1m:labels` (so it can never overwrite the
candle verdict for the same `(day, contract, timeframe)`) and the mirror
`data/labels/<YYYY-MM-DD>.parquet`. It never reads or writes a feature row:
`build-labels` is happy to run before (or without) `build-features`.


### The training matrix and its splits (ANJANEYA-AI, specs 21-22)

`--mode dataset` is where the causal half and the forward-looking half of the data meet -
in a research artefact, never in a live row. It joins each stored feature row to the label
of the **same instant** (`(timestamp, tradingsymbol)` with `validate="one_to_one"`; no
`merge_asof`, no tolerance), keeps only the rows that carry a class, and writes one Parquet
matrix plus one split manifest per configured horizon:

```
data/datasets/1m_h15_2026-09-01_2026-09-30.parquet
data/datasets/1m_h15_2026-09-01_2026-09-30_splits.json
```

```bash
# what a dataset would contain, at which horizons, split how (writes nothing)
uv run python -m apps.anjaneeya_ai.run --mode dataset --dry-run

# build every configured horizon / one horizon / a window / one contract / JSON
uv run python -m apps.anjaneeya_ai.run --mode dataset
uv run python -m apps.anjaneeya_ai.run --mode dataset --horizon 15
uv run python -m apps.anjaneeya_ai.run --mode dataset --from 2026-09-01 --to 2026-09-30
uv run python -m apps.anjaneeya_ai.run --mode dataset --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode dataset --json
```

Five things are worth knowing about the matrix:

* **labels are structurally absent from the features.** `future_return`, the excursions,
  `future_trend_change`, `label`, `label_code`, `label_version` and `horizon_minutes` are
  refused as features by name *and* by the `future_` prefix, so no config knob can put a
  forward number into `X`. The class ships twice: as the configured string (`label`) and as
  its integer index in `model.labels.classes` (`label_code`).
* **the join is counted, not trusted.** The run prints feature rows in, labelled rows out,
  rows whose label could not be computed (a forward window that left the session, an ATR
  warm-up) and unmatched rows in both directions - so "why is the matrix smaller than the
  feature table?" always has a number for an answer. Nothing is imputed, and constant or
  mostly-NULL columns are reported rather than silently dropped.
* **categoricals are declared, not guessed.** `model.dataset.categorical_columns` (validated
  against the stored feature contract) ships as pandas `category` dtype, and every other
  feature stays numeric - the artefact records the vocabulary so a model can be told about
  it later.
* **the split is purged walk-forward with a real holdout.** Spec 22's
  `walk_forward` policy: whole trading days in order, the last `holdout_days` removed from
  *every* window, an embargo that must cover the longest label horizon (the code refuses
  `embargo < horizon`), and every fold reporting its exact first/last train and validation
  timestamps, both cut-offs, and how many rows the guards removed. When there are not enough
  days yet, the report says how many are still missing instead of inventing folds.
* **the data gates are printed and stored.** The spec 4.3 stage
  (EXPLORATORY / ML_TRAINING / MEANINGFUL_BACKTEST / SEASONALITY) is computed from the days
  the matrix actually covers, warned about below `ml_training_min_days`, and recorded in the
  run manifest - so a four-day artefact says so in its own metadata.
* **one caveat is printed on every run**: the stored `labels` rows have no `timeframe`
  column, so a matrix assumes they were built at the dataset's timeframe. Rebuild the labels
  (`--mode build-labels`) if that is not the case.


### Training a model on it (ANJANEYA-AI, specs 21-22)

`--mode train` consumes that same matrix in memory - it does not re-implement the join - and
runs spec 22's walk-forward: for every fold it fits a LightGBM multiclass model on the fold's
surviving rows, scores the fold's validation window, scores spec 21's baselines on **exactly
the same rows**, and finally refits one model on every row outside the holdout. The model
output is a probability per label class - ANJANEYA never predicts BUY/SELL (the decision
layer is a later phase).

```
data/models/1m_h15_2026-09-01_2026-09-30_lightgbm.txt
data/models/1m_h15_2026-09-01_2026-09-30_lightgbm_metadata.json
```

```bash
# what would be trained, at which horizons, split how, gated how (writes nothing)
uv run python -m apps.anjaneeya_ai.run --mode train --dry-run

# train every configured horizon (the run's default timeframe and scope)
uv run python -m apps.anjaneeya_ai.run --mode train

# one horizon, one contract, a window, a different seed, machine-readable
uv run python -m apps.anjaneeya_ai.run --mode train --horizon 15
uv run python -m apps.anjaneeya_ai.run --mode train --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode train --from 2026-09-01 --to 2026-09-30
uv run python -m apps.anjaneeya_ai.run --mode train --seed 7 --json

# score the final model on the untouched holdout, once
uv run python -m apps.anjaneeya_ai.run --mode train --score-holdout

# too little data? the run says so and refuses; --force trains anyway, labelled BELOW_GATE
uv run python -m apps.anjaneeya_ai.run --mode train --force

# train and report without writing anything under data/models/
uv run python -m apps.anjaneeya_ai.run --mode train --no-write
```

Seven things are worth knowing about the run:

* **the gate is enforced before any work.** Below `data_gates.exploratory_min_days` (20
  captured days) the run prints how many days it saw, how many the exploratory and
  production gates need, and exits **4** without training. Between the exploratory gate and
  `ml_training_min_days` (45) the results are labelled `EXPLORATORY`; at or above it the
  results stay `PRELIMINARY` until a backtest can be run (`meaningful_backtest_min_days`,
  120). `--force` trains below the gate and keeps the `BELOW_GATE` label - a forced model is
  never a production model.
* **the baselines are the point.** `always_up`, `always_range`, `random` and
  `nifty_regime_only` are fitted on each fold's *training* rows and scored on the fold's
  validation rows, so both sides are out-of-sample. Every fold reports the log-loss delta
  against each baseline, and the across-fold table keeps the raw fold values (a mean alone
  would hide a model that is excellent twice and useless once).
* **probability metrics, reported honestly.** Log loss, Brier score, accuracy, and (when
  configured) a calibration curve with its expected calibration error. A metric the config
  did not ask for is reported as `null`, never as `0.0`. A confidently wrong one-hot is
  clipped at `1e-9` and **counted** as clipped, so a log loss of 20 cannot be mistaken for a
  finely-measured number.
* **nothing is silently fitted on one class.** A fold whose training window carries a single
  class is *skipped* and the reason is printed and stored; the run continues with the folds
  that can be trained, and says how many were skipped.
* **the holdout is not touched unless you ask.** Without `--score-holdout` the final model is
  fitted on every row outside the holdout and the holdout is left alone (that is what spec
  22 reserves it for). With the flag it is scored once, and the run records that it is no
  longer a holdout.
* **determinism is built in.** `model.random_seed` (or `--seed`) pins every LightGBM
  sub-seed; `deterministic=True` and `force_row_wise=True` are set. Train the same matrix
  twice and you get the same model - which is what makes a reported number re-derivable.
* **the artefact explains itself.** The metadata JSON carries the dataset it was trained on,
  the whole training policy, the feature and categorical vocabulary (so an unseen level at
  scoring time becomes *missing*, not a different integer code), the per-fold metrics and
  baselines, the aggregates, the gate verdict, the caveats, and the feature importances.


### Replaying it afterwards (ANJANEYA-AI, spec 23)

`--mode backtest` replays the stored bars against one model artefact - no network, no
credentials, no hand-written prices:

```
signals   read at the close of candle T          (the config's signal_at)
fills     at the open of T+1 (+ latency bars)    (never at the close that produced them)
exits     stop / target (ATR multiples), a time limit, the opposite signal, session end
costs     brokerage, STT (sell), exchange txn, SEBI fee, GST on the charges,
          stamp duty (buy), slippage, + an optional captured bid/ask spread
ledger    one row per *closed* trade, losses included, in spec 23's own column set
```

```bash
# what would be replayed: which model, which bars, which exits, which costs (writes nothing)
uv run python -m apps.anjaneeya_ai.run --mode backtest --dry-run

# replay the newest model for the configured dataset timeframe
uv run python -m apps.anjaneeya_ai.run --mode backtest

# a specific artefact, a wider threshold, one contract, a window, machine-readable
uv run python -m apps.anjaneeya_ai.run --mode backtest --model data/models/1m_h15_..._lightgbm.txt
uv run python -m apps.anjaneeya_ai.run --mode backtest --min-probability 0.45
uv run python -m apps.anjaneeya_ai.run --mode backtest --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode backtest --from 2026-09-01 --to 2026-09-30 --json

# replay without leaving evidence (no ledger, no summary, no rows)
uv run python -m apps.anjaneeya_ai.run --mode backtest --no-write
```

It writes:

```
data/backtests/1m_h5_2026-09-01_2026-09-30_ledger.parquet   # the trades, one row each
data/backtests/1m_h5_2026-09-01_2026-09-30_summary.json     # policies, counters, metrics, caveats
+ the same 28 columns appended to the stored `trades` table (spec 23's ledger contract)
```

and prints the gate, the verdict, the signal rule, both sides' costs, the counters (how many
bars had context, a prediction, a decision, a fill), the p&l with costs called out, the win
rate / payoff / profit factor / expectancy, the drawdown, holding time, exposure, the return
distribution, the Sharpe-like ratio, every exit reason, and one line per segment (option
side, time-of-day, cycle phase, weekday, days to expiry, strike distance, regime, exit
reason, expiry) with its **sample size** and a `[PRELIMINARY]` flag below
`evaluation.preliminary_min_samples`.

Five things are worth knowing about the run:

* **the signal rule is a threshold, and the run says so.** `backtest.signal` enters the
  option side whose directional class (`CE <- UP_CONTINUATION`, `PE <- DOWN_CONTINUATION`)
  the model gives at least `min_probability` for, optionally only when it is the model's top
  class. Spec 24's decision layer is separate (`--mode decide`) and this replay does not use it:
  this measures **costs**, not an edge,
  and every report and summary says that in a caveat.
* **no look-ahead is structural.** A signal can only fill at `T+1+latency`, inside the same
  session; a signal with no fill bar is dropped and counted. Every ledger field that
  describes the world (`nifty_state`, the bucket, the cycle phase, the ATR) is read from the
  *signal* bar's feature row.
* **losses are the point.** Every closed trade is a row - with its costs, its MFE/MAE and
  its exit reason. `--mode report` is built on this ledger precisely because nothing here
  filters it.
* **the labels table is never read.** A backtest is driven by prices, features and the model;
  a trade can therefore never be justified by its own future.
* **an exploratory backtest is labelled, not refused.** Below spec 4.3's
  `meaningful_backtest_min_days` (120 captured days) the run exits **0**, prints `BELOW_GATE`
  and stores that verdict in the summary - unlike `--mode train`, which refuses to fit a
  model it would have to call production.

Rates are configuration, not truth: `config/model.yaml`'s `costs:` block holds every charge,
and the summary carries the sentence that says to verify them against a current contract note
before quoting any P&L.


### Answering the research questions: the report (ANJANEYA-AI, spec 24)

Spec 24 is the project's *primary research output*, and it is a list of questions rather than a
chart gallery. `--mode report` answers all ten of them from the ledger `--mode backtest`
wrote - **losers included** - plus the stored NIFTY index bars for the two that are about the
market instead of the strategy:

```bash
# the ten items over the newest ledger under data/backtests/ (writes 3 artefacts)
uv run python -m apps.anjaneeya_ai.run --mode report

# a specific ledger / machine-readable / without writing anything
uv run python -m apps.anjaneeya_ai.run --mode report --ledger data/backtests/1m_h5_2026-09-28_2026-10-01_ledger.parquet
uv run python -m apps.anjaneeya_ai.run --mode report --json > report.json
uv run python -m apps.anjaneeya_ai.run --mode report --dry-run      # prints, writes nothing
uv run python -m apps.anjaneeya_ai.run --mode report --no-write
```

It writes one JSON document, one Markdown report and one flat CSV of cells under
`data/reports/research/`, all three rendered from one in-memory object (so they cannot
disagree), named after the ledger they came from:

```
data/reports/research/1m_h5_2026-09-28_2026-10-01_report.json   # the whole thing
data/reports/research/1m_h5_2026-09-28_2026-10-01_report.md     # the ten items as tables
data/reports/research/1m_h5_2026-09-28_2026-10-01_cells.csv     # one row per cell + verdict
```

What the ten items are, and how each one is built:

| # | Question | Answer |
| --- | --- | --- |
| 1 | performance by time-of-day bucket x option side | matrix of the ledger's closed trades |
| 2 | performance by time-of-day bucket x expiry-cycle phase | matrix (`EXPIRY_DAY`, `START_OF_CYCLE`, `MID_CYCLE`) |
| 3 | performance by day of week | segment table (spec 25's cut) |
| 4 | performance by days to expiry | segment table, bucketed (`EXPIRY_DAY` .. `4_PLUS_DAYS`) |
| 5 | interactions: regime x time-of-day, regime x cycle phase | two matrices |
| 6 | holding-time distribution by bucket | per-bucket distribution (p5/p25/p50/p75/p95) |
| 7 | volatility (ATR) profile by bucket and cycle phase | from the **stored index bars**, with the configured ATR |
| 8 | average cumulative intraday path per bucket | mean `close / bucket-open - 1` per minute, over sessions |
| 9 | significance, corrected for multiple testing | z-test on the win rate + Welch on net P&L, per cell |
| 10 | PRELIMINARY labels below the sample rule | carried by every cell, row and path |

Six things are worth knowing about the run:

* **every number carries its sample, and a small one is never tested.** A cell (or the baseline
  it is compared against) below `evaluation.preliminary_min_samples` is marked
  `PRELIMINARY` and its p-values stay `null` - a p-value from six trades is noise dressed as
  evidence. The family size the correction used is printed next to the result.
* **the correction is the configured one, applied honestly.** `evaluation.multiple_testing_correction`
  is `benjamini_hochberg` (or `bonferroni` / `none`), each matrix is one family, and only the
  tests that actually *ran* count towards the family size.
* **an untestable comparison is refused, not fudged.** A cell whose trades all won (no variance
  to test) or whose P&L never varied has `null` p-values and the reason in the JSON.
* **"not measured" never looks like "zero".** Without index bars, items 7 and 8 are present in
  the document with the reason (`volatility_unavailable`, `intraday_paths_unavailable`), and a
  bucket that never traded is an **absent** cell - never a profit of 0.0.
* **the market items are the market's, the ledger items are the strategy's.** Items 7 and 8
  are computed from the stored NIFTY 1m bars over the ledger's own window (a wider window would
  describe days the ledger never traded); items 1-6 and 9 come from the trades alone.
* **the caveats are part of the document.** The threshold signal rule, the "costs are
  configuration" sentence, the correction's family size and the seasonality gate
  (`data_gates.seasonality_min_days` = 250 captured days for any weekday/cycle-phase claim) are
  fields in the JSON and printed under the tables in the Markdown.

`--mode report` reads the database **read-only** and writes nothing to it: it never touches a
feature row, a label row or a trade, so a report can never change the evidence it describes.
Exit codes: `0` ok (including "there is no ledger yet"), `1` a ledger that is not a ledger
(a Parquet without the trade contract's columns), `2` configuration.


### Explaining a prediction: spec 26 (ANJANEYA-AI, `--mode explain`)

Spec 26 asks for every prediction to be explainable "in plain structured form" and lists the
blocks it expects - NIFTY, OPTION, PRICE ACTION, CANDLE, S/R, CONTEXT and the model's top
class probabilities - adding that those values are *examples*. So the blocks, and the columns
inside them, are configuration (`model.explainability`), not code:

```bash
# the newest 5 stored predictions, scored with the newest model, as blocks
uv run python -m apps.anjaneeya_ai.run --mode explain

# how many predictions / one contract / a window / a different model
uv run python -m apps.anjaneeya_ai.run --mode explain --rows 20
uv run python -m apps.anjaneeya_ai.run --mode explain --symbol NIFTY06OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode explain --from 2026-09-30 --to 2026-10-01
uv run python -m apps.anjaneeya_ai.run --mode explain --model data/models/1m_h5_..._lightgbm.txt

# machine-readable / print the plan / print without writing
uv run python -m apps.anjaneeya_ai.run --mode explain --json > explanations.json
uv run python -m apps.anjaneeya_ai.run --mode explain --dry-run
uv run python -m apps.anjaneeya_ai.run --mode explain --no-write
```

It writes one JSON, one Markdown and one flat CSV of *lines* under
`data/reports/explanations/`, all three from one in-memory object, named after the model
artefact they came from:

```
data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_explanations.json
data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_explanations.md
data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_entries.csv   # one row per line
```

```
NIFTY06OCT2625000CE @ 2026-10-01T09:20:00+05:30
  prediction : UP_CONTINUATION p=0.7991 (runner-up DOWN_CONTINUATION 0.2009, margin 0.5983)
  NIFTY:
    nifty_composite_trend_score : 0.45
    nifty_regime                : BEARISH
    nifty_fast_score            : unknown
    ...
  SUPPORT / RESISTANCE:
    support_strength            : unknown
    support_touch_count         : 1
  CONTEXT:
    days_to_expiry              : 2
    expiry_cycle_phase          : MID_CYCLE
    ...
  AI:
    UP_CONTINUATION   : 0.7991  <- predicted
    DOWN_CONTINUATION : 0.2009
    RANGE             : 0
    (top 3 of 4 class probabilities; the stored artefact carries every one of them)
```

Five rules make an explanation readable rather than decorative:

* **the blocks are configuration.** `model.explainability.sections` names stored feature
  *groups* (`core/features/feature_builder.FEATURE_GROUPS`) or individual feature *columns*;
  every name is checked against the feature contract at load, and a column may appear in
  exactly one block. Adding a feature to the contract and forgetting to explain it is a
  visible gap rather than a silent one.
* **a `NULL` feature is `unknown`, never 0.** `model.explainability.missing_text` is the word;
  "we do not know" and "it was zero" are different statements about a market, and the JSON
  carries `known: false` beside it.
* **a column the policy names but the row does not carry is a refusal**, not a blank - the same
  rule the predictor applies to the model's own feature list.
* **the AI block is the model's.** The classes, their order and its own pick come from the
  loaded artefact's metadata (never a hard-coded class list); the JSON stores the whole
  distribution, the console shows the top `model.explainability.probabilities` of them.
* **an explanation is not a recommendation.** Every artefact carries the caveats that it
  describes inputs rather than causes, that the cited row's `fetch_timestamp`/`quality_flag`
  are in the audit block, that the decision layer is a separate step (`--mode decide`) and this
  is not it - and the results
  label of the model being explained.

Identity (contract, expiry, strike, timestamp, timeframe) is in every header and the explained
row's provenance (`source`, `fetch_timestamp`, `quality_flag`, `run_id`) in its audit block,
which is why the `identity` and `provenance` feature groups are reserved and cannot be used as
blocks. `--mode explain` reads the database **read-only** and applies no data-accumulation gate
(an explanation describes an input, it is not a result). Exit codes: `0` ok, including "no
model artefact yet" and "no feature rows in scope"; `1` a crash; `2` configuration, including a
model whose timeframe or horizon is not the one being explained.



### Saying where the information came from: spec 27 (ANJANEYA-AI, `--mode importance`)

Spec 26 describes *what* the model was looking at for one prediction. Spec 27 asks the harder
question about the model as a whole: **which** of its inputs actually carried the information -
native importance, permutation importance, optionally SHAP - and then whether the information
comes from the NIFTY regime, the option's moving averages, candle behaviour, S/R, structure, IV,
the greeks, OI, the option residual, time of day, cycle phase or time to expiry. Its last line
is the rule: *"Do not assume. Measure. Report with confidence intervals where practical."*

```bash
# measure the newest model on its own training matrix (spec 22's reserved holdout days)
uv run python -m apps.anjaneeya_ai.run --mode importance

# a specific model, a window, one contract, more shuffles
uv run python -m apps.anjaneeya_ai.run --mode importance --model data/models/1m_h5_..._lightgbm.txt
uv run python -m apps.anjaneeya_ai.run --mode importance --from 2026-09-01 --to 2026-10-01
uv run python -m apps.anjaneeya_ai.run --mode importance --symbol NIFTY06OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode importance --repeats 20

# machine-readable / print the plan / print without writing
uv run python -m apps.anjaneeya_ai.run --mode importance --json > importance.json
uv run python -m apps.anjaneeya_ai.run --mode importance --dry-run
uv run python -m apps.anjaneeya_ai.run --mode importance --no-write
```

One JSON, one Markdown and one flat CSV per feature under `data/reports/importance/`, named
after the model artefact they describe:

```
data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_importance.json
data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_importance.md
data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_features.csv   # one row per feature
```

```
ANJANEYA-AI feature importance (spec 27) - 74 of 77 feature(s) measured over 2000 row(s), 5 permutation(s) at seed 42
  model      : .../1m_h5_2026-09-28_2026-10-01_lightgbm.txt (lightgbm, horizon 5m, EXPLORATORY)
  sample     : holdout - 2000 row(s) over 10 day(s), 2026-09-17T09:15:00+05:30 .. 2026-10-01T15:29:00+05:30
  metric     : log_loss (lower_is_better), baseline 1.2074
  sources    : 10 of 16 measured
    #1 option_ma (option moving averages): delta 0.0812 [0.0704, 0.092] over 10 feature(s)
    #2 nifty_regime (NIFTY regime): delta 0.0519 [0.0441, 0.0597] over 7 feature(s)
    #3 candle_behaviour (candle behaviour): delta 0.0226 [0.0198, 0.0254] over 15 feature(s)
    ...
    -  iv: none of its 2 feature(s) could be measured in this sample: every value of this column is NULL in the sample, so the model is fed the same missing value either way
  unmeasured : 3 feature(s) the sample could not say anything about (all NULL or constant); each reason is in the artefact
  SHAP       : unavailable - shap is an optional extra of ANJANEYA (pip install -e '.[explain]')
```

Six rules make the numbers readable rather than decorative:

* **the measurement is configured, not asserted.** `model.importance` decides how many times
  each feature is permuted (at least 2), which scoring rule (`log_loss`, or `accuracy`), which
  rows (`holdout` - spec 22's reserved days - or the whole `matrix`), the row cap, the seed and
  the confidence level. Every artefact prints the seed and the baseline it was measured against.
* **the sources are configured too.** `model.importance.topics` maps stored feature *groups* and
  columns onto spec 27's list, checked against the feature contract at load, with a column
  claimed by two sources refused. A feature no source claims is reported under `unmapped` -
  never attributed to a source that did not measure it.
* **native importance comes off the model, not out of the code.** The gain numbers are read out
  of the loaded booster (the same object that scores rows), never from a class list or a copy
  in a JSON beside it.
* **a feature the sample cannot measure is *not measured*, never 0.** All-`NULL` columns and
  columns holding a single value are reported with the reason: a delta of `0` would read as
  "this feature carries no information" when the truth is "these rows cannot say".
* **the interval is arithmetic on repeats.** The permutation is repeated at a fixed seed and the
  per-repeat deltas are kept, so a feature's interval and a *source's* interval (the sum of its
  features' deltas per repeat) are computed, not judged.
* **SHAP is optional in the strongest sense.** It is not a dependency; `model.importance.shap`
  decides whether a missing package is a recorded reason (`auto`, the default), a configuration
  error (`on`) or never asked about (`off`). A failure *inside* SHAP is a status on an optional
  measurement - the other two are already measured.

`--mode importance` reads the database **read-only**, never refits a model, and applies no
data-accumulation gate: a ranking describes a model, so the model's own results label travels
in the caveats instead of a refusal. Exit codes are the same as `--mode explain`: `0` ok
(including "no model artefact yet"), `1` a crash, `2` configuration - including a timeframe or
horizon the artefact never saw, and a `--repeats` below 2.

### Drawing it so it can be checked by eye: spec 28 (ANJANEYA-AI, `--mode charts`)

Spec 28 is two sentences of instruction and a list of pictures: *"Charts must be produced for
validation"*, *"Visualization is not optional"*. It asks for the NIFTY series (price, MA lines,
trend score, regime) and one option contract (price with `EMA8 high`, `EMA8 low`, `EMA20`,
`EMA55`, `EMA200`), both overlaid with today's running high/low, the previous day's high/low,
support/resistance, breakout/rejection markers, structure labels and the model's prediction - and
then five research charts: the time-of-day × cycle-phase P&L heatmap, the time-of-day × day-of-week
heatmap, the CE-vs-PE P&L curves, regime-conditional performance and the feature-importance bars.

```bash
# draw everything the stored data can support
uv run python -m apps.anjaneeya_ai.run --mode charts

# one family, one chart, a specific ledger, a specific importance artefact
uv run python -m apps.anjaneeya_ai.run --mode charts --only heatmap
uv run python -m apps.anjaneeya_ai.run --mode charts --only curves_ce_vs_pe
uv run python -m apps.anjaneeya_ai.run --mode charts --ledger data/backtests/1m_h5_..._ledger.parquet
uv run python -m apps.anjaneeya_ai.run --mode charts --importance data/reports/importance/..._importance.json

# chart the native column of the artefact / plan it / draw without writing
uv run python -m apps.anjaneeya_ai.run --mode charts --metric native
uv run python -m apps.anjaneeya_ai.run --mode charts --dry-run
uv run python -m apps.anjaneeya_ai.run --mode charts --no-write
```

Every figure is one image file plus one index Markdown under `data/reports/charts/`, named after
the ledger (or the model) it drew:

```
data/reports/charts/1m_h5_2026-09-28_2026-10-01_heatmap_time_of_day_cycle_phase.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_heatmap_time_of_day_day_of_week.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_curves_ce_vs_pe.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_regime_performance.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_importance_bars.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_price_nifty.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_price_option.png
data/reports/charts/1m_h5_2026-09-28_2026-10-01_charts.md        # the index that names them
```

```
ANJANEYA-AI charts (spec 28) - 7 figure(s), stem 1m_h5_2026-09-28_2026-10-01
  heatmap_time_of_day_cycle_phase: 15 cell(s), 11 preliminary
  heatmap_time_of_day_day_of_week: 15 cell(s), 11 preliminary
  curves_ce_vs_pe: 2 side(s) from 48 trade(s)
  regime_performance: 2 regime(s)
  importance_bars: 20 feature(s) from 1m_h5_..._importance.json
  price_nifty: 240 bar(s)
  price_option: 96 bar(s)
  wrote      : .../data/reports/charts/1m_h5_2026-09-28_2026-10-01_price_option.png
  ...
  skipped    : the P&L charts need a trade ledger under .../data/backtests - run --mode backtest first, or pass --ledger PATH
```

Three rules make a chart evidence rather than decoration:

* **the numbers are not recomputed.** The heatmaps, the CE/PE curves and the regime bars come from
  the *same ledger* `--mode report` reads, and the bars come from the artefact `--mode importance`
  wrote. A chart can therefore never disagree with the number beside it; it is a second view of one
  measurement, not a second measurement.
* **a value that is not there is left blank, never drawn as zero.** An empty heatmap cell prints
  only its trade count and no colour; a feature the sample could not measure has no bar. The whole
  zero-versus-unknown distinction the other stages enforce survives into the picture.
* **the sample travels with the picture.** Every heatmap cell prints its trade count and a cell
  below `evaluation.preliminary_min_samples` is hatched as `PRELIMINARY`; the importance bars carry
  their confidence interval; the option chart marks *where* the model entered. A chart that hides
  how little stands behind a cell would be decoration, which is the one thing spec 28 forbids.

The mode is read-only: it writes images and the index, never a feature row, a label row or a trade.
Each chart is attempted independently, so a missing ledger (or importance artefact, or stored
candles) skips exactly the charts that need it, with the reason printed, instead of failing the
run. `--dry-run` lists what would be built without importing matplotlib at all. Exit codes are the
same as the other reading modes: `0` ok (including "no chart could be built"), `1` a crash, `2`
configuration.

`model.charts` is where the presentation lives - figure size, resolution, file format, the diverging
colormap for a P&L and the caps (`top_features`, `max_points`, `max_markers`) that keep a dense
series or a busy overlay from making an unreadable picture. Nothing about *how a chart looks* is
hard-coded, and nothing here decides *what* a chart means.

---

### The decision layer: what the rule would do, spec 24 (ANJANEYA-AI, `--mode decide`)

The master prompt ends its phase list with **"PHASE 24 - Decision Layer (later, separate - only
after features, model and backtest are validated)"**, and says twice that ANJANEYA "never
predicts BUY/SELL. It predicts future behaviour over multiple horizons; the decision layer is
separate". `--mode decide` is that separate step: it scores the stored feature rows with one
model artefact and records what **one configured rule** would have done with each bar -
`ENTER_CE` / `ENTER_PE` / `NO_TRADE`, the reason code, the size and the exit plan:

```bash
# the newest model over every stored row (writes JSON + Markdown + a decisions CSV)
uv run python -m apps.anjaneeya_ai.run --mode decide

# one model, one horizon, a window, one contract, a tighter rule
uv run python -m apps.anjaneeya_ai.run --mode decide --model data/models/1m_h5_..._lightgbm.txt
uv run python -m apps.anjaneeya_ai.run --mode decide --horizon 5 --from 2026-09-28 --to 2026-10-01
uv run python -m apps.anjaneeya_ai.run --mode decide --symbol NIFTY06OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode decide --min-probability 0.55 --min-margin 0.10

# at most two entries a day, machine-readable / print the plan / print without writing
uv run python -m apps.anjaneeya_ai.run --mode decide --max-decisions-per-day 2 --json > decisions.json
uv run python -m apps.anjaneeya_ai.run --mode decide --dry-run
uv run python -m apps.anjaneeya_ai.run --mode decide --no-write
```

One JSON, one Markdown and one decisions CSV under `data/predictions/`, named after the model
that produced them (so a record names the policy **and** the model):

```
data/predictions/1m_h5_2026-09-28_2026-10-01_lightgbm_decisions.json   # the whole record
data/predictions/1m_h5_2026-09-28_2026-10-01_lightgbm_decisions.md     # sections + every entry
data/predictions/1m_h5_2026-09-28_2026-10-01_lightgbm_decisions.csv    # one row per scored bar
```

```
ANJANEYA-AI decision layer (spec 24) - as of 2026-10-01
  headline   : 480 decision(s) over 4 day(s): 96 entry(ies) (20.0%), 384 no-trade
  policy     : enter the option side whose directional class reaches p >= 0.40 and the model's top class
  results    : BELOW_GATE
  gate       : 4 captured day(s): too little data to conclude anything [INSUFFICIENT]; 16 more day(s) to EXPLORATORY
  model      : 1m_h5_2026-09-28_2026-10-01_lightgbm.txt (horizon 5m)
  -- decisions (by reason) --
  ENTERED                        : 96
  BELOW_MIN_PROBABILITY          : 384
  ...
  -- entries (by option side) --
  CE                             : 48
  PE                             : 48
```

Five rules keep the record honest:

* **it is a policy, not a model.** The probabilities come from one stored artefact; the rule that
  turns them into ENTER or NO_TRADE is configuration (`model.decision`), recorded in the artefact
  with the values actually used. The backtest's threshold rule (`backtest.signal`) is left
  exactly as it was: the two are independent, and each says which one ran.
* **it states an exit plan; it does not simulate one.** The stop, the target, the maximum holding
  and the session-end flatten are *echoed* from the policy into every row. There is no fill
  price, no P&L and no match anywhere in the record - replaying that plan against stored prices,
  with costs, is `--mode backtest`'s job, and this mode deliberately does not do it twice.
* **every refusal has a reason.** The vocabulary is closed and ordered
  (`UNMAPPED_OPTION_TYPE`, `DIRECTIONAL_CLASS_NOT_MODELLED`, `MISSING_PROBABILITY`,
  `BELOW_MIN_PROBABILITY`, `NOT_ARGMAX`, `MARGIN_BELOW_FLOOR`, `DAILY_LIMIT_REACHED`), so "why is
  this row a NO_TRADE?" is a value, not a paragraph - and the counts print every code, zeroes
  included.
* **the day's budget is deterministic.** `--max-decisions-per-day` goes to the *earliest* signals
  of the day across every symbol, and a row that already failed an earlier rung keeps its own
  reason instead of being relabelled.
* **the gate labels, it never refuses.** Like `--mode backtest`, the run is written and labelled
  (`BELOW_GATE`, `EXPLORATORY`, ...) with the gate it fell short of, and the model's own
  `results_label` travels in every artefact - a decision inherits the model's standing.

It places **no order** (there is no broker in this repository), writes nothing to DuckDB, and
says in every artefact that a decision is not an edge: the thresholds are arbitrary
configuration, chosen by hand and fitted to no outcome.

---

### Is there enough data yet? spec 29 (ANJANEYA-AI, `--mode accumulation`)

Everything above rests on one assumption: that enough was captured. Spec 29 (master-prompt
phase 25) makes that assumption checkable on a schedule - a **weekly summary of data captured,
coverage, gaps, sample sizes and readiness for retraining**:

```bash
# the weekly report (writes JSON + Markdown + a per-table CSV)
uv run python -m apps.anjaneeya_ai.run --mode accumulation

# a wider or narrower window (weeks), and the machine-readable shape
uv run python -m apps.anjaneeya_ai.run --mode accumulation --weeks 12
uv run python -m apps.anjaneeya_ai.run --mode accumulation --json > accumulation.json

# what would be read, without writing anything / without touching the database
uv run python -m apps.anjaneeya_ai.run --mode accumulation --dry-run
uv run python -m apps.anjaneeya_ai.run --mode accumulation --no-write
```

One JSON, one Markdown and one per-table CSV under `data/reports/accumulation/`, named after
the day they describe:

```
data/reports/accumulation/accumulation_2026-10-01_accumulation.json   # the whole report
data/reports/accumulation/accumulation_2026-10-01_accumulation.md     # the five sections
data/reports/accumulation/accumulation_2026-10-01_tables.csv          # one row per stored table
```

```
ANJANEYA-AI accumulation report (spec 29) - as of 2026-10-01
  headline   : 34 captured trading day(s) [EXPLORATORY], last capture 1 day(s) ago; no retrain due
  window     : 2026-09-04 .. 2026-10-01 (4 week(s), 34 captured day(s) of 20 trading day(s))
  gate       : 34 captured day(s): exploratory analysis only [EXPLORATORY]; 26 more day(s) to ML_TRAINING
  -- sample sizes (stored rows) --
  ticks      : 12,345,678 row(s) over 34 day(s) (...)
  ...
  -- coverage (data_coverage) --
  days       : 34 captured
  gaps       : 1 day(s) with gaps; worst 2026-09-15 (3 gap(s))
  quality    : {'OK': 135, 'GAP_FILLED': 4}
  -- readiness for retraining --
  [OK] spec 4.3's ML-training gate (45 days): have 34, need 45 (gate stage EXPLORATORY)
  ...
  -- retrain --
  recommended: no - not ready: captured_days, label_rows
  ...
```

Four rules make the report evidence rather than a dashboard:

* **the gate is spec 4.3's, re-read not re-implemented.** The ladder lives in one place
  (`core/gates.py`) and the report asks it, so a Monday report and a `--mode train` refusal can
  never disagree about which gate is open.
* **a captured day is not a complete day.** Coverage comes straight from the `data_coverage`
  ledger: days with gaps are listed worst-first, days below the configured ratio floor are
  called out, and the coverage requirement fails on a single recorded gap - the one failure an
  operator can still act on.
* **"ready" and "recommended" are different questions.** `ready_for_retraining` is a checklist
  (the gate, gap-free coverage, feature rows, label rows) where every line prints its `have`
  and its `need`; `retrain_recommended` additionally asks whether the captured data is newer
  than the newest model, or the model older than `retrain_max_model_age_days`.
* **a number is printed only when it was measured.** A table that cannot be read is reported as
  *unavailable with the reason*, never as zero, and a count-only table (the instrument master)
  carries no day span rather than "over 0 day(s)".

The mode reads the database **read-only** - it writes no tick, candle, feature, label or trade -
so a report can never change the dataset it describes. A missing ledger, an empty database or a
missing table is reported with a reason and `exit 0`, never a crash. `model.accumulation` holds
the presentation and the sample-size floors; the four gate *values* stay in `model.data_gates`.

---

### Which contracts: the option universe (ANJANEYA-ENGINE, spec 17)

Spec 17 is a *selection*, and selection is a question you have to be able to answer
from data: which expiries trade, which strikes sit around at-the-money, and do we
actually hold bars for them? `--mode universe` answers exactly that, offline, from
the stored database:





```bash
# the configured menu for the latest day with candles, checked against the database
uv run python -m apps.anjaneeya_engine.run --mode universe

# a specific day / a hand-placed ATM / machine-readable
uv run python -m apps.anjaneeya_engine.run --mode universe --date 2026-09-29
uv run python -m apps.anjaneeya_engine.run --mode universe --price 25160
uv run python -m apps.anjaneeya_engine.run --mode universe --json
```

It prints the **reference** (spot index or futures contract, with the price and where
it came from), the **ATM strike**, the **strike ladder** with its offsets, the
selected **expiries**, and then one row per contract: its stored `tradingsymbol`,
candle and feature row counts, and a status - `OK`, `NO_CANDLES`, `NO_FEATURES` or
`NOT_LISTED`. A contract the instrument master does not hold is **named** (strike,
side, expiry), never silently dropped, and `complete: yes` is printed only when every
selected contract has both bars and features. The mode opens the database
**read-only** and writes nothing at all - not even a coverage row.

The policy itself lives in [`core/options/universe.py`](core/options/universe.py)
(`build_selection`), so the same rules produce the same contracts everywhere:
ATM snapping (`nearest` is `floor(price/interval + 0.5)`; an exact midpoint rounds
**up**), the strike ladder (`strikes_offset`, each offset required to be a whole
number of strike intervals), and expiry selection (`current_week`, `next_week`,
`current_month`, `next_month` - resolved through `calendar.yaml`). `atm.reference`
chooses the anchor: with `futures`, capture prices *and* watches the futures contract
named by `reference_instruments.nifty_futures.expiry_selector`, and a master holding
no futures contract is a hard error rather than a universe placed on the wrong price.

### Credentials (needed for the capture modes)

```bash
cp .env.example .env      # then fill it in; .env is git-ignored
```

`.env` is loaded automatically by every entry point (`core/env.py`) from the
directory above `config/`. Three rules make it safe to keep around:

* a real environment variable **always wins** over the file, so CI secrets and
  one-off `ZERODHA_API_KEY=... uv run ...` overrides behave as expected;
* blank values are ignored, so a half-filled file is not the same as a wrong one;
* secrets are never printed or logged - `scripts/daily_login.py` and
  `scripts/health_check.py` report the file path and the variable *names* only.

To ignore the file for one shell: `ANJANEYA_SKIP_DOTENV=1` (the test suite always
sets this, which is why the offline tests can never reach your real account).

Two supported modes:

* **Automated**: `ZERODHA_USER_ID` + `ZERODHA_PASSWORD` + `ZERODHA_TOTP_KEY`
  (from the 2FA setup QR) → ANJANEYA logs in, exchanges the one-time
  `request_token`, and caches the `access_token` until ~06:00 IST next day.
* **Manual**: log in at
  `https://kite.trade/connect/login?api_key=<your_api_key>`, copy the
  `request_token` from the redirect URL, and pass it either to
  `scripts/daily_login.py --request-token <token>` (recommended: it verifies the
  token) or to a capture run via `--request-token <token>`.

Secrets are read **only** from the environment. YAML files hold the *names* of
the variables, never the values, and log records are scrubbed of any value that
matches a configured secret.

---

## Layout

```
config/               six YAML files - every tunable in the system
  settings.yaml         project metadata, paths, logging, run tracking
  data_sources.yaml     Zerodha credential names, rate limits, retry, websocket,
                        capture session policy (heartbeat, gaps, batching, fetch)
  option_universe.yaml  strikes, expiries, symbol templates, index history
  indicators.yaml       MA combinations, scores, candle/levels parameters
  model.yaml            horizons, data gates, dataset, labels, walk-forward, costs,
                        execution, backtest, evaluation, explainability (spec 26),
                        importance (spec 27), charts (spec 28), accumulation (spec 29)
  calendar.yaml         sessions, expiry weekday, holidays, time buckets

core/                 shared library (pure logic; no side effects on import)
  constants.py          column contracts, timezone helper, interval/timeframe map
  schemas.py            pydantic row contracts (tick/snapshot/index/instrument/
                        coverage/candle/feature/label/trade)
  config_loader.py      validated configuration + config hashing
  env.py                explicit `.env` loading (a real env var always wins)
  gates.py              spec 4.3's data-accumulation gates as one pure function (the
                        ladder `--mode train` refuses on and `--mode accumulation` reports)
  accumulation.py       spec 29's report model: the window, the gate, the coverage/gap
                        summary, the readiness checklist, the retrain recommendation and
                        the JSON/Markdown/console renderings (pure, no I/O)
  market_calendar/      NSE sessions, holidays, expiries, DTE, cycle phase
  storage/              DuckDB bootstrap + schema.sql, the shared row readers,
                        row writers and the coverage verdicts
    accumulation.py       spec 29's reader: table sample sizes/spans, the coverage
                          ledger collapsed per day, and the weekly captured-day buckets
  auth/                 Zerodha credentials, TOTP login, token cache, REST client
  rate_limiter.py       token buckets + exponential backoff (429 aware)
  logging_setup.py      loguru sinks with secret redaction
  run_context.py        run_id, config hash, git commit, versions
  app_bootstrap.py      the shared startup sequence for all three apps
  indicators/ nifty/ price_action/ backtest/
                        package skeletons with documented module boundaries
  ml/                   the training side (specs 21-22)
    dataset.py            features + labels -> the leak-free training matrix, the
                          label-leakage guard, and the spec 4.3 data gates
    walk_forward.py       spec 22's purged walk-forward splitter (holdout, embargo,
                          purge), with exact timestamps per fold - and the exact
                          (train, validate) frames a fold hands to the trainer
    trainer.py            LightGBM multiclass training (fixed seeds, single-class
                          refusal) and the whole walk-forward run: fold metrics,
                          baselines on the same rows, one final model outside the holdout
    baselines.py          spec 21's always-up / always-range / random / regime-only
                          reference points, each fitted on the training fold
    evaluation.py         log loss, Brier score, accuracy, calibration curves + ECE,
                          the model-vs-baseline comparison, and spec 4.3's gate as a label
    predictor.py          scoring a stored artefact: training feature order, categorical
                          vocabulary, one class distribution per row
    explainer.py          spec 26's explanations: configured blocks of the stored feature row
                          plus the model's own class probabilities, a NULL feature rendered
                          as unknown, a missing column refused, and the caveats attached
    importance.py         spec 27's feature importance: the model's own gain off the booster,
                          permutation importance at a fixed seed with confidence intervals
                          per feature and per source, an optional SHAP attribution that says
                          why it did not run, and the features no source claims kept visible
  visualization/        spec 28's charts (pure: data in, a matplotlib figure out)
    charts.py             the figure builders: the price chart (price, MA lines, level/running
                          overlays, breakout/rejection/structure markers, an optional trend-score
                          axis and regime shading), the P&L heatmap, the line curves, the bar
                          chart and the importance bars - each refusing an empty picture and
                          leaving a value that is not there blank rather than drawing a zero
    sources.py            the loaders: the ledger grouped into a heatmap (samples and
                          PRELIMINARY flags), the CE/PE curves, the regime bars, the importance
                          bars from a stored artefact, and the price data from stored candles /
                          features / index bars (the option MA stack recomputed with the engine's
                          own function)
  labels/               the forward-looking half of the dataset (spec 20)
    label_builder.py      one contract's candles -> LABEL_COLUMNS (multi-horizon)
  options/               option instruments and contract maths
    option_trend.py       spec 10's per-contract trend engine
    greeks.py             spec 16's IV/greeks model (Black-Scholes-Merton) - the
                          documented pricing assumptions live here
    universe.py           spec 17's option selection: ATM anchor (spot/futures),
                          strike ladder, expiries, the contract menu
  features/             the unified feature row (spec 18) and its validation (spec 19)
    feature_builder.py    one contract's candles -> exactly FEATURE_COLUMNS
    feature_validation.py spec 19's checks, as a report that never drops anything
  research/             spec 24's answers over one trade ledger (no writes, no network)
    segments.py           the matrices (bucket x side/phase, regime interactions), the
                          single-axis cuts and the holding-time profile - each cell with
                          its sample and its PRELIMINARY flag
    significance.py       per-cell tests against the rest of the ledger (win-rate z-test,
                          Welch on net P&L) + Benjamini-Hochberg / Bonferroni, and no
                          test at all below the configured sample rule
    profile.py            the market's own profile from stored index bars: the ATR/range
                          table by bucket x cycle phase, and the average cumulative
                          intraday path per bucket
    report.py             the ten items as one document (JSON + Markdown), with the
                          policy, the samples and the caveats attached to the numbers

apps/
  anjaneeya_capture/    data acquisition (Phase 2)
    instruments.py        instrument master, dump cache, universe resolution
    websocket_listener.py KiteTicker adapter, tick conversion, stall watchdog
    option_chain_poller.py periodic OI/quote snapshots over REST
    nifty_index_fetcher.py index history backfill with adaptive window halving
    storage_writer.py     batched DuckDB writes + Parquet mirror + coverage
    heartbeat.py          the machine-readable "is capture alive?" file
    session.py            one session, start to finish
    run.py                the CLI (modes: capture/option-chain/backfill/instruments/status)
  anjaneeya_engine/     ticks -> candles -> greeks -> features
    candles.py            tick frame -> 1m bars, higher timeframes merged from 1m
    enrichment.py         which NIFTY level belongs to which bar (the close-instant join)
    storage_writer.py     replace-by-scope candle/feature/label writes, greek updates,
                          Parquet mirror
    run.py                the CLI (modes: build-candles/enrich-greeks/universe/
                          build-features/build-labels/validate/status)
  anjaneeya_ai/         datasets -> model -> backtest -> report -> explanations
    dataset_writer.py     atomic dataset artefacts: the Parquet matrix + split manifest
    model_writer.py       atomic model artefacts: LightGBM's model text + the metadata
                          JSON that explains how it was produced
    backtest_writer.py    atomic ledger artefacts: the trade Parquet + the run summary JSON
    report_writer.py      atomic research artefacts: the spec 24 JSON + Markdown + cells CSV
    explain_writer.py     atomic explanation artefacts: the spec 26 JSON + Markdown + the
                          flat CSV of explanation lines
    importance_writer.py  atomic importance artefacts: the spec 27 JSON + Markdown + the flat
                          CSV of features (one row per feature, ranked)
    chart_writer.py       atomic chart artefacts: one image per spec 28 figure + the index
                          Markdown that names them, each figure closed after it is saved
    accumulation_writer.py atomic accumulation artefacts: the spec 29 JSON + Markdown + the
                          per-table CSV, from one in-memory report
    run.py                the CLI (modes: dataset/train/backtest/report/explain/importance/
                          charts/accumulation/status)

data/                 captured and derived data (git-ignored, never committed)
  raw/instruments/      cached daily Kite instrument dumps
  raw/ticks/            Parquet mirror of the tick stream (one file per day)
  raw/option_chain_snapshots/  Parquet mirror of the OI/quote snapshots
  candles/<timeframe>/  Parquet mirror of the derived candles (one file per day)
  features/<timeframe>/ Parquet mirror of the unified feature rows (one file per day)
  labels/               Parquet mirror of the label rows (one file per day)
  datasets/             one training matrix + split manifest per (timeframe, horizon)
  models/               one LightGBM model text + metadata JSON per training run
  backtests/            one trade ledger Parquet + run summary JSON per backtest
  reports/feature_validation/  spec 19's per-day JSON validation reports
  reports/research/     spec 24's research reports (JSON + Markdown + one CSV per ledger)
  reports/explanations/ spec 26's explanations (JSON + Markdown + one CSV of lines per model)
  reports/importance/   spec 27's feature importance (JSON + Markdown + one CSV of features)
  reports/charts/       spec 28's figures (one image per chart + one index Markdown)
  reports/accumulation/ spec 29's weekly accumulation reports (JSON + Markdown + per-table CSV)
  runs/heartbeat/       capture heartbeat JSON
db/                   local DuckDB analytical database (git-ignored)
scripts/              operator tooling
  daily_login.py        one-command Kite login + token verification
  login_and_start.py    browser login then the capture session, one command
  run_morning.py        guarded morning: calendar -> universe gate -> login -> capture
  health_check.py       offline health report (includes capture state)
  doctor.py             pre-flight readiness (token, dump age, db, scheduler)
  check_calendar.py     trading days, expiries, provisional-holiday guard
  refresh_calendar.py   reconcile config/calendar.yaml with NSE's holiday master
  run_daily_pipeline.py candles -> greeks -> features -> labels -> validate ->
                        dataset -> train -> backtest (+ --weekly cadence steps)
  start_capture.{ps1,sh} run the daily login then a capture session
  run_backfill.{ps1,sh} run the index-history backfill
  run_daily_pipeline.{ps1,sh} run the whole post-close chain
  run_morning.{ps1,sh}  the guarded morning routine
  run_weekly.{ps1,sh}   the daily chain plus the weekly cadence steps
  run_scheduled.ps1     calendar-gated entry point for the scheduled tasks
  schedule_tasks.ps1    install/status/remove the weekday Morning + PostClose tasks
tests/                offline unit + integration tests (fakes in tests/support/)
docs/runbook.md       the operator's day: setup, daily routine, failure playbook
docs/phase_reports/   per-phase completion reports
deepseek_text_*.txt   the master specification this project is built against
```

---

## Operating rules the code enforces

1. **No look-ahead, ever.** Feature rows physically cannot contain label
   fields, the dataset builder refuses to treat a label-derived column as a
   feature, and the walk-forward embargo must be ≥ the longest label horizon
   (checked when the configuration loads *and* when the splitter is built).
2. **Every timestamp is timezone-aware.** A naive `datetime` raises
   immediately. `tzdata` is a hard dependency because Windows ships no IANA
   database.
3. **Contract identity is complete or absent.** `(expiry, strike, option_type)`
   must travel together, so features can never mix contracts.
4. **Raw ticks are append-only.** Everything downstream is rebuilt from them;
   nothing overwrites the record of what the market actually did.
5. **Parameters live in YAML.** No thresholds, windows or weights in Python.
6. **Data gates are respected.** Findings below 20 / 60 / 120 / 250 captured
   trading days are labelled EXPLORATORY / PRELIMINARY accordingly,
   `health_check.py` reports progress against those gates, and `--mode train`
   refuses to train at all below the exploratory gate (exit 4) unless forced.
7. **Runs are reproducible.** Every run writes a manifest (config hash, git
   commit, versions, seed, artifacts) to `data/runs/<run_id>/manifest.json` and
   a matching row in the `run_manifest` table.
8. **One writer at a time.** DuckDB allows a single read-write process per file,
   so ANJANEYA's apps are run sequentially against `db/anjaneeya.duckdb`
   (capture during the session; engine and AI afterwards). If a second process
   tries anyway, it stops with an explicit "database is already open" message
   instead of a raw IO error.
9. **Derived data is replaced, never merged.** Rebuilding a day's candles deletes
   that `(day, timeframe, contract)` scope first and appends the fresh rows, so a
   shorter rebuild cannot leave the previous run's later bars behind. Raw data
   (rule 4) is never deleted; derived data always is replaceable.
10. **A missing value stays missing.** No bar is invented for a minute without
    ticks, no `NULL` greeks become `0`, and a volume counter reset is counted and
    reported rather than being absorbed into a negative number.

---

## Honest caveats (read before trusting any number)

* **Holiday dates in `config/calendar.yaml` are reconciled for 2026, and 2026
  only.** `holidays_verified: true` and every entry `verified: true`, checked on
  2026-10-01 against NSE's own `holiday-master` endpoint (F&O segment). Before
  that, the list was a seed whose five incorrect weekday dates would each have
  cost a real session and whose seven missing holidays would each have started a
  capture on a closed exchange. `health_check.py` still reports the flag and every
  run manifest still records it, so a list that goes stale goes stale *loudly*:
  set it back to `false` before 2027-01-01. Expiry-related conclusions depend on
  this list.
* **A muhurat session has to be entered by hand, and is not.** NSE lists
  2026-11-08 as `Diwali Laxmi Pujan*`, and the asterisk marks an evening session
  on a Sunday. `NseCalendar.is_trading_day` now gives an explicit
  `sessions.special_sessions` entry precedence over both the weekend test and the
  holiday list — the only precedence that can express a muhurat session, since NSE
  lists those dates as holidays too and they fall midweek in most years. No entry
  is shipped, because NSE announces the window in a separate circular and no 2026
  figure has been verified here: the day is still reported closed, which is what
  the file actually knows. Add the announced window to `special_sessions` when the
  circular lands.
* **The cost model is a configurable baseline.** Brokerage, STT, exchange fees,
  GST, stamp duty, SEBI charges and slippage are parameters; they change, and
  the values in `config/model.yaml` must be verified against current
  broker/statutory rates before any P&L is taken seriously.
* **NIFTY weekly expiry has been both Thursday and Tuesday** in recent years,
  so the weekday is configuration, not an assumption. The shipped
  `config/calendar.yaml` says `Tuesday`, verified on 2026-09-30 against the
  exchange's own `NFO` instrument dump (every listed NIFTY expiry is a Tuesday,
  and the Monday expiries are holiday rolls). Re-check it whenever a session
  captures nothing: `python -m apps.anjaneeya_capture.run --mode capture
  --dry-run` must print a non-zero `subscriptions` line.
* **Historical option data cannot simply be downloaded.** Zerodha does not serve
  expired option contracts, so ANJANEYA accumulates its own option history by
  running the capture app every session. That accumulation is the gating
  constraint on the whole project — hence the data gates above.
* **The automated login flow uses Kite's non-public `/api/login` + `/api/twofa`
  endpoints** because the official SDK does not expose them. It is isolated,
  mockable, and covered by tests, but it has not been exercised against live
  Zerodha from this machine (no credentials were available during Phases 1-2).
  The live smoke test (`tests/live/`) is the intended proof and skips itself
  without credentials.
* **Kite publishes no implied volatility on any endpoint.** The tick payload and
  the REST quote contain OI, volume and depth, but no IV, so the capture app
  stores `iv = NULL`. IV is derived from price and OI in a later phase — it is
  never invented at capture time.
* **Only market depth gives a two-sided quote.** The tick payload has no
  `bid`/`ask` fields, so they are derived from the top of the book. In `ltp` or
  `quote` websocket mode they are therefore `NULL`; the shipped configuration
  uses `full` mode for exactly this reason, and a session warns loudly if the
  mode is changed.
* **Symbol formats are not assumed.** Contracts are resolved from Kite's daily
  instrument dump by *attributes* (exchange, underlying, expiry, strike, option
  type), and the configured `symbol_templates` are only a cross-check that warns
  when NSE changes a format.
* **Per-interval historical-candle limits are not documented by Kite.** The
  backfill plans its own windows from configuration and halves a window when the
  server objects, so an unexpected limit degrades throughput rather than failing
  the run.
* **Capture is only as good as its uptime.** The coverage ledger records gaps
  (silences longer than `capture.gaps.tick_gap_seconds` while the market is open)
  per stream, so "how much of the session did we actually record?" is answerable
  afterwards instead of assumed.
* **A candle covers most of the session, not all of it.** A minute with no tick
  produces **no bar** rather than a synthetic one, so a thin contract legitimately
  has fewer than 375 bars and `data_coverage.coverage_ratio` says how many are
  missing (the bar count is otherwise reported as if the session were complete).
* **Candle volume is a partial sum, deliberately.** Kite publishes *cumulative*
  day volume, so a bar's volume is the sum of the increments observed inside it.
  The first tick of the day has no predecessor, so the 09:15 bar starts counting
  from the first tick we actually saw, and any negative increment (a
  mid-session counter reset) contributes zero and is counted in the run summary
  as a `volume reset` instead of silently becoming a negative number.
* **Higher timeframes are merged, not rebuilt.** 3m/5m/15m come from the 1m bars,
  so they are only as good as the 1m layer - and a 15m bar with a missing 1m
  input carries that minute's missing volume. The gap is reported at 1m level,
  which is where it actually happened.
* **The greeks are a model, and the model is configured.** `iv`, `delta`, `gamma`,
  `theta` and `vega` are computed by `--mode enrich-greeks` (Black-Scholes-Merton, a
  configured rate and dividend assumption, ACT/365 `T` floored at one session) - they are
  **not** what the exchange published, because the feed does not publish them. Read the
  assumptions before comparing two IVs, and rebuild the greeks if an assumption changes.
  A bar whose price has no IV in the bracket stays `NULL`.
* **The feature row is derivable, the greeks are not.**
  `expected_option_move = delta[t-1] * nifty_move` uses the delta of the bar *before* the
  move (using `delta[t]` would let the move partly define its own expectation and would
  mechanically shrink `|option_residual|`), and all three relative columns are `NULL`
  when there is no delta - which is exactly what an unenriched build publishes.
* **A candle rebuild invalidates the greeks on the rebuilt scope.** They are columns on a
  bar that was replaced, so they go back to `NULL`: the order is `build-candles` ->
  `enrich-greeks` -> `build-features`, and the enrichment pass is cheap enough to re-run
  over a day.
* **The feature row is one row per `(timestamp, contract)`, so a build covers one
  timeframe.** `build-features` refuses `--timeframes 1m,5m` (the two series share
  every bar's open time and the table's key has no timeframe in it) and a rebuild
  replaces the whole contract-day. Switching the stored timeframe is a rebuild of that
  day, not a second set of rows. `enrich-greeks`, which writes to the *candle* table,
  takes any subset of timeframes.
* **The `nifty_*` columns need `index_history`.** Without it they are `NULL` - warned
  about, summarised in the run output and noted in the coverage row - and
  `nifty_long_score` stays `NULL` until the lookback window really holds 2062 1m bars
  (`required_lookback_bars`), which the CLI reads for every day it builds. The same
  missing series makes `iv` `NULL`: an IV needs a level at the bar's close.
* **Bucket boundaries are wall-clock aligned.** Bars are floored against IST
  midnight, which matches every NSE session (09:15-15:30, half days, the
  hour-aligned special sessions). A hypothetical session starting at 09:16 would
  produce a 09:15 bucket, which is documented rather than hidden.
* **Engine conventions are fixed before the feature row exists.** The indicator
  and price-action layers already decide how a feature will read, and each choice
  is recorded in the code and in
  [`docs/phase_reports/phase_03.md`](docs/phase_reports/phase_03.md): every
  `distance_*` is in **ATR units** (the raw level sits beside it in the same row),
  `NULL`/`NaN` means *not computed yet* and is never a `0`, an EMA/SMA is only
  published once its window is complete, and a swing is attached to the bar that
  could first have *known* it - never to an earlier one.
* **A level has a state, and only a *held* break can fail.** `NOT_TESTED` ->
  `APPROACHING` -> `TOUCHED` -> `CLOSED_ABOVE`/`CLOSED_BELOW` -> `HELD_*` ->
  `RETESTED`, with `FAILED_BREAKOUT`/`FAILED_BREAKDOWN` reserved for a break that
  never reached a hold. The state column summarises the last thing that happened;
  the per-bar event flags are the exact record, and support/resistance counts are
  built from those same flags so the two can never disagree.
* **A model metric is not a backtest.** `--mode train` reports log loss, Brier score,
  calibration and the baselines' scores on the *validation* windows of a walk-forward run -
  that is a statement about how well the model ranked the future it was shown, before
  spreads, costs, fills and position sizing. Spec 4.3 makes results statistically meaningful
  only once a meaningful backtest can be run (120 captured days), and every training run
  prints and stores that caveat next to its numbers.
* **Few folds is not a result.** With only a handful of captured days a walk-forward run
  produces two or three folds; the across-fold mean and standard deviation then describe
  three observations, not a future. The report says so in the notes, and the raw per-fold
  values travel in the model metadata so nobody has to trust the average.
* **The stored labels carry no timeframe.** A training matrix (and therefore a model)
  assumes the label rows were built at the dataset's timeframe; every run prints that
  assumption. Rebuild the labels (`--mode build-labels`) if the assumption is wrong - the
  candidate fix is a `timeframe` column on `labels` (schema version 2), which is a policy
  decision rather than something to paper over.
* **The report's p-values are large-sample approximations.** `--mode report` tests a cell
  against the rest of the ledger with a two-proportion z-test (win rate) and Welch's t-test
  (mean net P&L), corrected across each matrix's own family. There is no exact or
  permutation alternative yet, so `SIGNIFICANT` means "the normal approximation cleared the
  corrected level on this ledger", not "an edge exists" - and the sample rule
  (`evaluation.preliminary_min_samples`) refuses to test the cells where the approximation
  would be worst.
* **A report describes the ledger it was handed, bugs and all.** It never re-runs the
  backtest, by design: if the replay was wrong, the report will faithfully describe the
  wrong replay rather than disagree with it. The ledger and the report are meant to be read
  together.
* **The report's CSV is one row per matrix cell.** The weekday and days-to-expiry cuts, the
  holding-time profile, the ATR profile and the intraday paths are in the JSON and Markdown
  but not (yet) in the CSV.
* **The ATR profile and the intraday path describe NIFTY, not the options.** They are built
  from the stored index bars; a per-side premium profile would need the option ATR and a
  premium series, and nothing here claims it.
* **An explanation describes inputs, not causes.** `--mode explain` quotes the stored feature
  row of one instant and the class distribution one model produced from it. A `NULL` feature
  is printed as `unknown` (never 0), a line the run could not fill says why, and a column the
  configuration names but the row does not carry is a refusal. It does not attribute a market
  move to a feature, it does not turn a probability into a position (the decision layer is the
  separate `--mode decide`), and it is only as fresh as the row it quotes - whose `fetch_timestamp`
  and `quality_flag` sit in its audit block. Explaining a model trained below the gate inherits
  that label, and every artefact names the model it explained.
* **An importance ranking is about one model, not about the market.** `--mode importance` scores
  a permuted column with every other column held at its stored value, so two features that
  carry the same information share (and can split) the credit, and a feature that is a
  combination of others keeps less than its influence suggests. A feature the sample cannot
  measure - all `NULL`, or a single value - is reported as *not measured* with the reason,
  never as a delta of 0. The interval is over repeated permutations at one seed, not a
  bootstrap over rows, and it describes this model's use of these inputs: a model below spec
  4.3's gate is ranked as the unvalidated model it is, and every artefact carries that label.
  The rows measured are named too: spec 22's reserved holdout days when the split has any, the
  whole stored matrix (with an explicit in-sample caveat) when it does not, capped to the
  newest `importance.max_rows` rows when it is larger than that.
* **A green accumulation report is not a green light.** `--mode accumulation` describes the
  *dataset*: a satisfied requirement means there is enough recorded data to try something, never
  that something is true. `ready_for_retraining` is a checklist about sample sizes and coverage,
  and `retrain_recommended` only compares the newest model's date with the newest captured day -
  whether a retrain actually helps is decided by purged walk-forward validation, not here. The
  report also cannot see a day that was never attempted: it counts what the coverage ledger
  recorded, so a session nobody started is invisible to it (the capture heartbeat and
  `scripts/check_calendar.py` are what notice that).

---

## Testing

```bash
uv run pytest -q                                  # offline suite
uv run pytest -q --cov=core --cov=apps            # with coverage
ANJANEYA_RUN_LIVE=1 uv run pytest -q -m live      # needs real credentials
```

The live suite (`tests/live/`) is three tests, not one, because the live path needs three
different things and a run outside market hours has to report which of them is proven. The first
(`...authenticates_and_stores_the_master`) needs nothing but a live token, so it **passes at any
hour** — evening, weekend or holiday; the second adds a real index quote for the universe; the
third is the only one that needs the exchange open, and it skips with the reason otherwise (a
closed market is not a defect). Run it after `scripts/daily_login.py`; before that it cannot pass,
because there is no token to prove.

Test discipline is enforced mechanically by `tests/conftest.py`:

* an autouse guard blocks any **off-machine** socket connection, so the suite can
  never accidentally call Kite (loopback stays allowed because importing
  `kiteconnect` installs a Twisted reactor that opens a local socket pair);
* `ANJANEYA_DATA_ROOT` and `ANJANEYA_DB_PATH` are redirected into `tmp_path`, so
  tests never write into the repository's `data/` or `db/`.

Capture is tested the same way the rest of the system is: with *shape-accurate*
fakes in `tests/support/capture.py` (a synthetic instrument dump, a `KiteTicker`
double that replays the subscribe/set_mode protocol, a REST client double). The
integration test drives a whole session — ticks, a mid-session reconnect, OI
snapshots, coverage, heartbeat, Parquet mirror — and asserts on what landed on
disk.

The engine is tested the same way, with `tests/support/engine.py` providing tick
streams, seeded databases, synthetic candle/index windows and the real `AppContext`.
`test_engine_pipeline.py` seeds ticks, runs the CLI and then asserts on the candles,
the coverage verdicts and the Parquet files that resulted — including a second run
proving that a rebuild *replaces* rather than duplicates. `test_feature_pipeline.py`
goes one stage further: ticks → candles → feature rows → `--mode validate`, asserting
on the stored rows, the feature coverage row, the Parquet mirror and the JSON
validation report. The two checks worth knowing about are the ones that cannot be
faked — a 15m index bar is proven unreachable from a 1m candle that has not closed
yet, and validation is proven to *catch* a deliberately future-peeking engine.

The operator scripts get the same treatment. `test_daily_pipeline.py` replaces the
child processes with a recorder and asserts the exact argv of all eight steps (the
day, the single-timeframe flag, the horizons, `--force` only where it belongs, and a
dry run that writes nothing); `test_check_calendar.py` drives the holiday guard
through a scratch calendar, including the case that matters — a *provisional* date
inside the risk window must exit non-zero and be named. `test_daily_pipeline_e2e.py`
then runs the real chain with real child processes in two configurations: the shipped
one (engine half green, `train` refusing below the gate exactly as it should) and one
whose walk-forward fits the fixture (a real model artefact and a real backtest ledger
on disk). The automation added in Phase 9 is pinned the same way, child processes and
network replaced by recorders: `test_run_morning.py` proves the two silent-loss
refusals (a holiday is a clean `NO_SESSION`; a zero-contract universe exits `4`
**without** starting the session) and that the happy path logs in before it starts;
`test_run_weekly.py` proves the daily chain is unchanged without `--weekly` and that
the cadence steps appear only when asked for, in order; `test_doctor.py` pins the
readiness report contract; `test_refresh_calendar.py` proves the NSE payload parses,
the diff is right and a YAML rewrite keeps every comment; `test_capture_backfill_options.py`
covers the `--years` window and the REST-bar → option-candle mapping; and
`test_schedule_tasks.py` asserts the scheduler wrappers still carry the settings that
matter (no overlapping instances, S4U, highest privileges, an 8-hour cap, the calendar
gate).

The report stage is tested at both levels. The unit tests pin the arithmetic that can be
checked by hand (the two-proportion z-test against its textbook formula, Benjamini-Hochberg
against hand-computed values *and* against SciPy's own `false_discovery_control`, the
ATR/path aggregates on synthetic bars) and the honesty rules that make the tables readable
(a cell exactly at the sample threshold, a missing intersection that must stay absent, a
bucket boundary that must move with the bar's *close*, a `NaN` that must come back as `None`).
`test_ai_report_pipeline.py` then runs the real `main()` over a real Parquet ledger and a
seeded database: three artefacts whose stem mirrors the ledger, the ten items with their
samples, corrected significance, the market items from the *stored* index bars, the
Markdown/CSV renderings, `--dry-run`/`--no-write`/`--json`, and a before/after count of every
stored table proving a report writes nothing to DuckDB.

The explain stage is tested the same way. `test_ml_explainer.py` pins the three rules that
carry spec 26: the shipped configuration and the module's own defaults resolve to *the same*
blocks (and a hand-built policy explains exactly the column it names, so nothing is
hard-coded), a `NULL` feature comes back as the configured word with `known: false` while a
real `0` stays a measurement, and a column the policy names but the row does not carry is a
refusal rather than a blank. The model's block is checked to rank whatever distribution it was
handed (ties keeping their declared order), to prefer a scored row's own pick over a
recomputation, and to say "not scored" when there is no model at all.
`test_ai_explain_pipeline.py` runs the real `main()` against a seeded database and a real
artefact pair: three artefacts named after the model, spec 26's blocks in the configured order
with the AI block last and the model's whole class distribution in it, a `NULL` feature
`unknown` in both the JSON and the CSV, the row's audit block, newest-first ordering with
`--rows`, the window/`--symbol` scoping, the refusals (no model, no rows, a timeframe or
horizon the model never saw) and `--dry-run`/`--no-write` each writing nothing - with a
before/after count of `features`, `labels` and `trades` proving the mode never writes to
DuckDB.

The importance stage is tested at both levels too. `test_ml_importance.py` pins the rules that
carry spec 27: the shipped configuration and the module's own defaults resolve to the same
sources (and every trainable feature is claimed by exactly one of them), a policy that claims a
column twice or asks for a single permutation is refused, permutation importance finds the
feature a synthetic class is a function of *ahead of the noise beside it* and produces the same
deltas twice at one seed, a permuted categorical keeps the model's own vocabulary (the bug that
cost a run while this stage was built), a feature the sample cannot measure is reported as not
measured rather than as a zero, and SHAP distinguishes missing / off / stubbed / broken. The
per-source totals are checked to be the sum of their features' repeats, and the three structural
caveats are checked to be present even in a hand-built report. `test_ai_importance_pipeline.py`
runs the real `main()` against a seeded database and a real artefact pair: three artefacts named
after the model, all sixteen sources with their intervals, the sample named (spec 22's holdout),
a ranking that finds the one column the fixture's class really is a function of, determinism
across two runs, the refusals (no model, no rows, a bad timeframe or horizon, `--repeats 1`) and
`--dry-run`/`--no-write` each writing nothing - with a before/after count of `features`, `labels`
and `trades` proving the mode never writes to DuckDB.

The visualization stage is tested at both levels as well. `test_visualization_charts.py` pins the
rules that carry spec 28: the shipped `charts:` block and the module's own defaults resolve to the
same presentation, every builder renders a figure and refuses an impossible one (an empty price
series, an empty grid, a one-point curve), a heatmap cell nothing traded in is left blank while a
small one is flagged PRELIMINARY, the importance bars rank by absolute value and attach the
interval only to the permutation metric, the CE/PE curves are cumulative and ordered by entry
time, the option MA stack is recomputed from the contract's own candles, and the level-event
markers fire on the breaks and rejections but not on the sustained `HELD_*` states.
`test_ai_charts_pipeline.py` runs the real `main()` against a seeded database, a real ledger and a
real importance artefact: one figure per chart plus the index, the index naming every figure,
`--only` narrowing to a family, `--metric native` charting that column, a missing ledger (or
importance artefact) skipping exactly the charts that need it with the reason printed,
`--dry-run`/`--no-write` writing nothing, and a before/after count of `features`, `labels` and
`trades` proving the mode never writes to DuckDB.

`test_ai_decision_pipeline.py` runs the real `main()` against a seeded database and a real model
artefact: one decision per scored bar, three artefacts written and registered on the manifest,
the reason vocabulary closed, the exit plan echoed but never simulated, the daily budget honoured
and its reason reachable, `--min-probability`/`--min-margin`/`--max-decisions-per-day`/`--symbol`/
`--from`/`--to` changing the record and being recorded as the values used, `--dry-run`/`--no-write`
writing nothing, the gate labelling (never refusing) an `INSUFFICIENT` run, a missing model or an
empty database reported with a reason and `exit 0`, and a before/after count proving the mode
never writes to DuckDB.

See [`docs/phase_reports/phase_01.md`](docs/phase_reports/phase_01.md),
[`docs/phase_reports/phase_02.md`](docs/phase_reports/phase_02.md),
[`docs/phase_reports/phase_03.md`](docs/phase_reports/phase_03.md),
[`docs/phase_reports/phase_04.md`](docs/phase_reports/phase_04.md),
[`docs/phase_reports/phase_05.md`](docs/phase_reports/phase_05.md),
[`docs/phase_reports/phase_06.md`](docs/phase_reports/phase_06.md),
[`docs/phase_reports/phase_07.md`](docs/phase_reports/phase_07.md) and
[`docs/phase_reports/phase_08.md`](docs/phase_reports/phase_08.md) for the phase
reports, including what was verified and what remains untested.


