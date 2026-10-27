# ANJANEYA — daily playbook

One page, manual operation. No Task Scheduler, no TOTP secret: the Kite token is
caught in a browser and cached once a day.

> **Golden rule:** DuckDB has **one writer**. Never run two of these at the same
> time. The console refuses a second job; the command line will not.

---

## 0. One-time setup (already done on this machine)

| Item | Where |
| --- | --- |
| Start the console | `uv run streamlit run scripts/anj_ui.py` → http://localhost:8501 |
| Credentials | `.env` (`ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_USER_ID`, `ZERODHA_PASSWORD`, `ZERODHA_REDIRECT_URL`) |
| Corporate TLS proxy | `.env` → `REQUESTS_CA_BUNDLE` → `data/certs/anjaneeya-ca-bundle.pem` (see `data/certs/README.md`) |
| Safety check | `uv run python scripts/doctor.py` → must end with `RESULT: READY` |
| Cached token | `data/raw/.tokens/zerodha_token.json` |

---

## 1. Morning — start at **09:00**, not 09:15

The session has to log in, resolve the option universe and top up the NIFTY index
*before* the open. The app then waits for 09:15 by itself
(`capture.wait_for_session_open: true`, guard window 15 min).

### Step 1 — one click: **Run → “Morning — browser login + capture”**

That single job does the whole morning:

1. opens the Kite login in your browser (password + 2FA),
2. catches the `request_token` on `http://127.0.0.1:5000/`,
3. exchanges and **verifies** it (`profile()`), caching the token until 06:00 tomorrow,
4. starts the capture, which waits for 09:15 and stops at 15:30.

If today's token is already cached and valid, the browser is skipped and the capture
starts immediately — so pressing it twice is harmless.

**Fallback A — paste the token** (no browser on this machine, or the job timed out):

1. Open `https://kite.trade/connect/login?api_key=<your ZERODHA_API_KEY>&v=3`
2. Log in and finish the 2FA.
3. The browser lands on `http://127.0.0.1:5000/?request_token=XXXX&action=login&status=success`
   — the page will **not** load (nothing is listening). That is expected.
   Copy the whole URL, or just the value after `request_token=`.
4. Console → **Token** tab → paste into *request_token or the redirect URL* →
   **Exchange request token**.

**Fallback B — the terminal:**

```powershell
.\.venv\Scripts\python.exe scripts\login_and_start.py     # login + capture
```

**If you paste by hand:**

```powershell
uv run python scripts/daily_login.py --request-token "<token or redirect URL>"
```

### Step 2 — start the session

| | |
| --- | --- |
| **Console** | **Run** → **Day — capture + post-close math** |
| **CLI** | `uv run python scripts/login_and_start.py` (login + capture) or `uv run python scripts/run_morning.py` (cached token) |
| **Does** | calendar guard → universe gate → login → index top-up → capture → at 15:30 the post-close math |
| **Stops** | by itself at 15:30 |

A holiday is a clean `NO_SESSION` (exit 0). An empty option universe is exit 4 and
the session is **not** started (deliberate — an index-only capture is a lost day).

### Step 3 (optional) — look at the plan before 09:15

**History** → *Plan* (no writes), or:

```powershell
uv run python -m apps.anjaneeya_capture.run --mode capture --dry-run
```

It prints the ATM anchor, the strikes, the expiries and the request budget.
`0 option contract(s)` means a calendar/expiry misconfiguration — fix it before
the open, not after.

---

## 2. During the session (09:15–15:30)

* **Press nothing else.** One writer.
* The **Data** tab shows a warning while the capture holds the lock — expected,
  not an error.
* Progress: `data/runs/heartbeat/capture.json` (rewritten every 10 s) and
  `data/logs/anjaneeya_capture.log`.
* A job started from the console survives closing the browser tab.

---

## 3. After the close — 15:35 onwards

Order matters. Step 1 feeds step 2.

| # | What | Console | CLI |
| --- | --- | --- | --- |
| 1 | **Top up the NIFTY index** (network, seconds) | **Run** → *Top up index — NIFTY daily + 1m* | `pwsh ./scripts/run_backfill.ps1` |
| 2 | **Rebuild the day** (offline) | **Run** → *Day — math only* | `uv run python scripts/anj.py day --no-capture` |
| 3 | **Train / backtest / reports** (weekly or when ready) | **Run** → *Learn* | `uv run python scripts/anj.py learn` |

**Why step 1 cannot be skipped:** the session tops the index up at **09:05**, which
covers everything up to the *previous* day. The engine builds the day's `nifty_*`
block from `index_history` only. Without the 15:35 top-up, that day's NIFTY
features (and its IV/greeks) are degraded or `NULL`.

**What step 2 runs:** `candles → greeks → features → labels → validate → dataset`,
then the decision layer (`--mode decide`). It deliberately **skips `train` and
`backtest`** — a model trained before the close is trained on an incomplete day.

**What step 3 runs:** `run_daily_pipeline.py --weekly` = `dataset → train →
backtest → report → explain → importance → charts → accumulation`.
(`pwsh ./scripts/run_daily_pipeline.ps1` runs the same chain **without** the four
research artefacts; `pwsh ./scripts/run_weekly.ps1` is the weekly one.)

**If a session started late or died mid-day**, heal the missing *candles* from REST
before the pipeline (ticks cannot be recovered, 1-minute bars can):

```powershell
.\.venv\Scripts\python.exe -m apps.anjaneeya_capture.run --mode backfill-options --years 0.01
```

**If this PC's clock is behind the exchange**, every tick is flagged `SUSPECT`
(`exchange_time - received_at > 2 s`) and the default build keeps almost none of
them. Until the clock is synced, build the day with the flag explicitly:

```powershell
.\.venv\Scripts\python.exe -m apps.anjaneeya_engine.run --mode build-candles --all --quality OK,SUSPECT
```

The tick's own **exchange timestamp** is still what is used - the flag only says the
machine clock disagreed with it. Every run prints `tick quality : OK,SUSPECT` so a
day built this way is never mistaken for a clean one. Syncing the clock retires it:
`w32tm /config /manualpeerlist:"time.windows.com" /syncfromflags:manual /update`
then `w32tm /resync` (elevated shell).

---

## 4. Where the outputs land

| Artefact | Path |
| --- | --- |
| Database | `db/anjaneeya.duckdb` |
| Raw ticks / instruments | `data/raw/` |
| Parquet mirrors | `data/candles/`, `data/features/`, `data/labels/` |
| Datasets | `data/datasets/` |
| Models | `data/models/<timeframe>_h<horizon>_<from>_<to>_lightgbm.txt` |
| Backtest ledger | `data/backtests/` |
| Reports / charts | `data/reports/{explanations,importance,charts,accumulation}/` |
| Logs | `data/logs/` (pipeline: `daily_pipeline_<date>.log`) |
| UI job log | `data/logs/ui/` |

**Console → Results** shows the newest reports and charts; **Data** shows the first
or last rows of any table.

---

## 5. How the model trains (the plain version)

```
ticks → candles → iv/greeks → features ─┐
                            → labels  ──┴→ dataset → train → backtest → reports
```

* **features** = one row per `(timestamp, contract)`: NIFTY trend/regime, option
  trend, candle behaviour, levels, structure, support/resistance, option context.
* **labels** = forward-looking, built in a separate pass: for each horizon
  **5 / 10 / 15 / 30 min** → `UP_CONTINUATION`, `DOWN_CONTINUATION`, `RANGE`,
  `REVERSAL`.
* **dataset** joins them (1m rows), refuses label-derived columns as features, and
  splits with **purged walk-forward**: 30-day train window, 10-day validation,
  **30-min embargo** (≥ the longest horizon), step 15 days, and the last
  **30 days held out** untouched.
* **train** fits LightGBM per horizon (300 trees, lr 0.05, seed 42) against four
  baselines (always-up / always-range / random / regime-only) on the *same* rows.
  A single-class frame is refused, never scored as 100%. The final artefact is
  refit on everything outside the holdout; the holdout is scored **once**.
* Every run **retrains from scratch** on all stored history. There is no
  incremental update — "live" data joins through the same daily feature/label rows.

### When will there be a model?

| Captured trading days (from `ticks`) | What happens |
| --- | --- |
| < 20 | `train` exits **4** (accumulation gate) |
| 20–44 | gate still refuses; `--force` runs, but the splitter needs 70 days → **no model is written** |
| 45–69 | gate open, run completes, still **0 folds → no model artefact** |
| **≥ 70** | first fold (`holdout 30 + train 30 + validate 10`), and the final model is fitted; +1 fold per 15 days |
| ≥ 120 / ≥ 250 | meaningful backtest / seasonality claims |

The trainer returns as soon as the split has no folds (`core/ml/trainer.py`), so
`--force` can never buy you a model early — it only skips the *gate verdict*.
The honest answer with the shipped windows: **the first model appears after ~70
captured sessions**, and until then `--force` produces a report, not an artefact.

**The one lever:** shrink the windows in `config/model.yaml`
(`holdout_days`, `walk_forward.train_window_days`) to lower `days_required` —
e.g. `15 + 20 + 10 = 45`. That changes what the metrics mean, so decide it
deliberately rather than as a shortcut.

---

## 6. Rules that bite

1. **The token dies at 06:00 IST.** A login done tonight is useless tomorrow
   morning. One login per day, every day.
2. **One writer.** Capture, pipeline, backfill, console job — never two at once.
3. **Order after the close:** index top-up → rebuild the day → (then) train.
4. **`--all` is tick-driven.** Days that came from the REST option backfill have
   candles but no ticks, so they need an explicit `--date`.
5. **Expiry day moves the universe.** Strikes are ATM ±200 in 50-point steps for
   the current week, next week and the current month; the window re-centres every
   30 minutes. Tomorrow's planned set is printed by the dry run.
6. **A capture started after 15:30 stops immediately** — correct behaviour, not a
   bug.

---

## 7. Troubleshooting

| Symptom | Cause | What to do |
| --- | --- | --- |
| `Kite rejected the request_token exchange` | token already used, or older than ~2 min | generate a fresh one; the token is single-use |
| `SSL: CERTIFICATE_VERIFY_FAILED` | CA bundle missing/stale | see `data/certs/README.md`; then `doctor.py` |
| `doctor.py` → `PROBLEM instrument dump` | dump never downloaded | **History** → *Fetch* (or `--mode instruments`) |
| Pipeline exit **2** | no ticks stored for that day | the session did not run; pass `--date` for another day |
| Pipeline exit **4** | accumulation gate | expected; `--force` runs it, but no model is written until 70 captured days |
| All `nifty_*` are `NULL` | the 15:35 index top-up was skipped | top up, then rebuild the day |
| `dataset` has no rows | features/labels not built for that day | rebuild the day first |
| Console says a job is running but nothing is | a status file outlived its process | the console now checks the **pid**; press **Forget job** if a card still sticks |
| Job exited with `3221225786` (`0xC000013A`) | the old detached launcher was killed by a console control event | fixed: jobs now run in a hidden console |
| `Data` tab warns about the database | a capture holds the write lock | expected during 09:15–15:30 |

---

## 8. One-screen cheat sheet

```powershell
# morning (09:00)
uv run streamlit run scripts/anj_ui.py            # console
#   Run -> Morning - browser login + capture      (one click: token + capture)
#   ...or without the console:
#   .\.venv\Scripts\python.exe scripts\login_and_start.py

# after the close (15:35)
#   Run -> Top up index            |  .\.venv\Scripts\python.exe -m apps.anjaneeya_capture.run --mode backfill
#   Run -> Day - math only         |  .\.venv\Scripts\python.exe scripts\anj.py day --no-capture

# weekly (or when there is enough data)
uv run python scripts/anj.py learn                # train + backtest + reports

# any time something looks wrong
uv run python scripts/doctor.py
uv run python scripts/check_calendar.py --days 7
```
