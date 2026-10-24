# Phase 6 report — the continuous data-accumulation report

**Status: Stage 6.1 (spec 29 — the continuous data-accumulation report) complete.**

Phase 5 ended with a model, a research report, per-prediction explanations, a feature-importance
ranking and the charts that make the numbers checkable by eye. All of it rests on one assumption:
that enough was captured. Master spec section 4.2 is explicit about why that assumption cannot be
left implicit — NIFTY options history "is BUILT BY ANJANEYA-CAPTURE, captured live, day by day",
"the options dataset grows over time", and "ANJANEYA-AI must wait for enough accumulated days
before training a serious model" — and section 4.3 then requires ANJANEYA to "track and report data
coverage" against four milestones (20/60/120/250 captured trading days). Master-prompt phase 25
gives that requirement its recurring shape: a **weekly summary of data captured, coverage, gaps,
sample sizes and readiness for retraining**.

`--mode accumulation` is that summary. It reads the stored tables and the `data_coverage` ledger
**read-only**, applies no gate of its own (reporting the gate is the point), and writes one JSON,
one Markdown and one per-table CSV under `data/reports/accumulation/`, named after the day they
describe (`accumulation_2026-10-01_*`). `schema_version` stays **1**: the mode writes no tick, no
candle, no feature row, no label row and no trade.

Four rules made the report evidence rather than a dashboard:

* **The gate is spec 4.3's, re-read rather than re-implemented.** The ladder
  (`GATE_STAGES`/`GateStatus`/`data_gate_status`) moved out of `core/ml/dataset.py` into a new,
  model-free `core/gates.py` and is re-exported from `core.ml` for every existing caller. There is
  exactly one ladder, so a Monday report and a `--mode train` refusal can never disagree about which
  gate is open — and the report no longer needs pandas or LightGBM to ask the question.
* **A captured day is not a complete day.** Coverage comes straight from the ledger: days with gaps
  are listed worst-first, days below the configured ratio floor are called out, and the coverage
  requirement fails on a single recorded gap — the one coverage failure an operator can still act
  on.
* **"Ready" and "recommended" are different questions.** `ready_for_retraining` is a four-line
  checklist (the gate, gap-free coverage, feature rows, label rows) where every line prints its
  `have` and its `need`; `retrain_recommended` additionally compares the newest captured day with
  the newest model artefact and the model's age against `retrain_max_model_age_days`.
* **A number is printed only when it was measured.** A table that cannot be read is reported as
  *unavailable with the reason*, never as zero; a count-only table (the instrument master) carries
  no day span rather than "over 0 day(s)"; and a `NULL` `coverage_ratio` (a tick stream has no
  expected row count) is treated as *not evidence of completeness* rather than quietly passing.

Every artefact carries the same five caveats, and the first is the one that matters: this report
describes the *dataset*, not the market. A satisfied requirement means there is enough recorded
data to try something, never that something is true.

---

## 1. What was built

| File | What it does |
| --- | --- |
| `core/gates.py` (new) | Spec 4.3's data-accumulation gates as one pure function: `GATE_STAGES`, `GATE_SENTENCES`, `GateStatus` (with `ready_for_training`, `to_dict`, `describe`) and `data_gate_status`. Extracted from `core/ml/dataset.py` so a model-free report can reach it; imports nothing heavier than `dataclasses`. |
| `core/accumulation.py` (new) | Spec 29's report model, pure and I/O-free: `AccumulationPolicy` (the knobs, `from_config`, `gate_status`), `TableProfile` (a table's rows/days/span, or why it could not be read), `CoverageDay` (one ledger day collapsed: streams, complete/gap/missing/suspect, worst ratio, gap total, per-flag counts), `WeekCapture`, `ModelRecord`, `AccumulationInputs` (the reader's container), `ReadinessItem`, `AccumulationReport` (with `gap_days`, `incomplete_days`, `latest_capture`, `missing_requirements`, `ready_for_retraining`, `headline`, `to_dict`, `to_markdown`), `build_accumulation_report`, `render_lines` and `render_markdown`. |
| `core/storage/accumulation.py` (new) | The read side: `TIMESTAMP_COLUMNS`/`ACCUMULATION_TABLES` (which column each table's span is measured from), `read_table_profile`/`read_table_profiles`, `read_coverage_days` (the ledger grouped by `(trading_date, quality_flag)` in one pass, collapsed in Python into both per-day verdicts and the database-wide flag histogram), `read_weekly_capture` (distinct captured days bucketed into the last N weeks ending on `as_of`), and `read_accumulation_inputs`. A table that cannot be read is reported, never raised. |
| `apps/anjaneeya_ai/accumulation_writer.py` (new) | Atomic artefacts (temp file, then rename), failures counted rather than raised: `<stem>_accumulation.json` (with the run identity in front), `<stem>_accumulation.md` and `<stem>_tables.csv` (one row per table, `days` kept as a nullable integer so a count-only table has a real blank). `accumulation_stem(as_of)` is `accumulation_<YYYY-MM-DD>`. |
| `apps/anjaneeya_ai/run.py` (modified) | `--mode accumulation`: `_accumulation_policy`, `_newest_model_record` (the newest `*_lightgbm.txt` under `data/models/`, with its horizon from its own metadata), `_accumulation_window`, `_accumulation_plan_lines` (the `--dry-run` plan), `_accumulation_payload` (the `--json` shape) and `_run_accumulation`; the `--weeks` flag; `MODE_PLAN`/`AVAILABLE_MODES`/`_dispatch` wired; the `--no-write` help text extended. |
| `core/config_loader.py` (modified) | `AccumulationConfig` (a validated `accumulation:` block) and its field on `ModelConfig`. |
| `core/app_bootstrap.py` (modified) | `accumulation` registered in `DEFAULT_MODES[APP_AI]`. |
| `config/model.yaml` (modified) | The shipped `accumulation:` block. |
| `core/ml/dataset.py` (modified) | The gate block replaced by a re-export from `core.gates` (its `__all__` and every existing import are unchanged), so `core.ml.GateStatus` / `core.ml.data_gate_status` still work. |
| `core/storage/__init__.py` (modified) | The new reader functions exported. |
| `README.md`, `docs/runbook.md` (modified) | Status table, the `--mode accumulation` section, the layout, the caveats, and the runbook's weekly routine (§2.5, §3, §7). |
| `docs/phase_reports/phase_06.md` (new) | This report. |

### Configuration added

One new block in `config/model.yaml`, validated by `AccumulationConfig`:

```yaml
accumulation:
  spec: "29"
  window_weeks: 4                      # the report's window (last N weeks ending on as_of)
  stale_days: 10                       # newer than this and the report says capture may have stopped
  healthy_coverage_ratio: 0.9          # a captured day counts as complete at or above this
  retrain_min_feature_rows: 5000       # sample-size floors the retrain checklist compares against
  retrain_min_label_rows: 5000
  retrain_max_model_age_days: 30       # a model older than this, on enough data, is due a retrain
```

The four **gate values** are deliberately *not* here: they stay in `data_gates:` so this report and
`--mode train` read one ladder, not two. `AccumulationPolicy` carries the same defaults, so a
hand-built policy (a test, a notebook) and the shipped one agree — a test asserts exactly that.

### Tests created

| File | Tests | What it covers |
| --- | --- | --- |
| `tests/unit/test_accumulation.py` (new) | 45 | The whole model without a database: the table profile's present/empty/unavailable states; `CoverageDay.is_complete` (a gap fails, a `NULL` ratio is not completeness); the policy defaults matching `config/model.yaml`; the gate re-reading spec 4.3's ladder (parametrised over 0/20/60/120 days); the window arithmetic and its `--weeks` override; captured-day counting and in-window filtering; `days_since_last_capture`; week buckets; `gap_days` ordering and `incomplete_days`; the four readiness lines (`have`/`need`/`met`/`detail`, a missing table failing a line); `ready_for_retraining`; the retrain recommendation in all four states (not ready, no model, data newer than the model, model too old, model current); every warning (stale, nothing captured, unreadable ledger, gaps, insufficient gate, no table read); the caveats; and the three renderings (`to_dict`, `render_lines`, `render_markdown`). |
| `tests/integration/test_ai_accumulation_pipeline.py` (new) | 19 | The real `main()` against a seeded database and a **frozen clock**: the three artefacts are written and named after the report day; the JSON document; the gate being spec 4.3's; the Markdown sections; the per-table CSV; the report never writing to `ticks`/`candles`/`features`/`labels`/`trades`; `--no-write`, `--dry-run`, `--json` and `--weeks`; a gap day being reported and failing the coverage line; an empty database being reported (exit 0) rather than crashing; the newest model artefact being named in the report; the artefacts being registered on the run manifest; the mode never taking the `NOT_IMPLEMENTED` path; the reader collapsing the ledger per day and reporting a missing table; the writer being atomic and counted; and the `accumulation:` config block being honoured. |

**Net test delta: +64** (45 unit + 19 integration). No test was modified.

---

## 2. Design decisions worth recording

**The gate moved to `core/gates.py`.** The report must ask "which spec 4.3 gate is open?" without
importing `core.ml` — which drags in pandas and (through `core/ml/__init__.py`) LightGBM. A weekly
report that costs a LightGBM import to print a day count is a report nobody will run. The ladder is
now a 100-line pure module; `core/ml/dataset.py` imports and re-exports it, so `core.ml.GateStatus`,
`core.ml.GATE_STAGES` and `core.ml.data_gate_status` are unchanged for every existing caller and
test. There is exactly one ladder in the codebase, which was the point.

**The report is one pure object with three renderings.** `AccumulationReport` renders to JSON, to
Markdown and to the console block from the same in-memory object, so the three files cannot
disagree — the same rule `--mode report`, `--mode explain` and `--mode importance` follow.

**`as_of` is a parameter, not `datetime.now()`.** `build_accumulation_report(..., as_of=...)` takes
the day it describes, and the app supplies it from `context.calendar` (exchange-local). Without
that, the window would silently change under a test, and a report could not be reproduced. The
integration test freezes `NseCalendar.now()` and asserts the exact window start.

**The reader measures, the model decides.** `core/storage/accumulation.py` only produces facts
(row counts, spans, per-day ledger collapses, week buckets); every judgement (what counts as
complete, which readiness lines exist, whether a retrain is due, which words go in a warning) lives
in `core/accumulation.py`. A number is measured once, so no two sections of a report can re-read the
database and drift apart.

**Coverage is judged by gaps, not by ratio alone.** A tick stream has no `expected_rows`, so its
`coverage_ratio` is `NULL` for every day — which means "all days have a ratio ≥ 0.9" would fail
forever on a healthy install. The readiness line therefore fails on a *recorded gap* (an actionable
failure) and the ratio floor is reported as an observation instead. `CoverageDay.is_complete`
treats a `NULL` ratio as *not* evidence of completeness, so the conservative reading and the
actionable reading are separate and both visible.

**`days` is nullable on `TableProfile`.** The instrument master has no timestamp column; printing
"over 0 day(s)" for it would be a small lie of exactly the kind this project refuses. `None` means
"counted, no span exists", `0` means "has a timestamp and spans no day".

**No pruning.** Every other AI mode writes artefacts named after their input (a ledger, a model) and
they accumulate deliberately. An accumulation report is named after the day it describes, so one
file per week *is* the series; deleting old ones would delete the history the report exists to
build.

**The mode applies no gate.** `--mode train` refuses below the exploratory gate; `--mode
backtest` labels its result. This mode *reports* the gate and always exits 0 — refusing to report
on the data because there is not enough data would be the one refusal that makes the situation
worse.

---

## 3. How to use it

```powershell
# the weekly report (writes JSON + Markdown + a per-table CSV)
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode accumulation

# wider or narrower window, machine-readable shape
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode accumulation --weeks 12
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode accumulation --json

# plan it without writing; or print it without writing
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode accumulation --dry-run
.\.venv\Scripts\python.exe -m apps.anjaneeya_ai.run --mode accumulation --no-write
```

Exit codes match the other reading modes: **0** ok (including "nothing has been captured yet"),
**1** a crash, **2** a configuration problem. There is no gate refusal — reporting the gate is the
point.

The artefacts land under `data/reports/accumulation/`:

```
data/reports/accumulation/accumulation_2026-10-01_accumulation.json   # the whole report
data/reports/accumulation/accumulation_2026-10-01_accumulation.md     # the five sections
data/reports/accumulation/accumulation_2026-10-01_tables.csv          # one row per stored table
```

---

## 4. Example output (real, from this repository's database)

`uv run python -m apps.anjaneeya_ai.run --mode accumulation` against the working database (which has
the instrument master and the NIFTY index backfill, but no captured sessions yet):

```
ANJANEYA-AI accumulation report (spec 29) - as of 2026-10-01
  headline   : 0 captured trading day(s) [INSUFFICIENT]; no retrain due
  window     : 2026-09-04 .. 2026-10-01 (4 week(s), 0 captured day(s) of 20 trading day(s))
  gate       : 0 captured day(s): too little data to conclude anything [INSUFFICIENT]; 20 more day(s) to EXPLORATORY
  -- sample sizes (stored rows) --
  instruments : 46,489 row(s)
  ticks      : 0 row(s) over 0 day(s)
  option_chain_snapshots : 0 row(s) over 0 day(s)
  index_history : 3,271 row(s) over 271 day(s) (2025-08-28T00:00:00+05:30 .. 2026-09-30T15:29:00+05:30)
  candles    : 0 row(s) over 0 day(s)
  features   : 0 row(s) over 0 day(s)
  labels     : 0 row(s) over 0 day(s)
  trades     : 0 row(s) over 0 day(s)
  data_coverage : 0 row(s) over 0 day(s)
  run_manifest : 30 row(s) over 3 day(s) (2026-09-29T22:31:42+05:30 .. 2026-10-01T15:03:09+05:30)
  -- coverage (data_coverage) --
  days       : 0 captured
  gaps       : none recorded
  below floor: 0 day(s) under 0.9
  -- captured days by week --
  2026-09-04 .. 2026-09-10 : 0 day(s)
  2026-09-11 .. 2026-09-17 : 0 day(s)
  2026-09-18 .. 2026-09-24 : 0 day(s)
  2026-09-25 .. 2026-10-01 : 0 day(s)
  -- readiness for retraining --
  [MISSING] spec 4.3's ML-training gate (60 days): have 0, need 60 (gate stage INSUFFICIENT)
  [MISSING] captured days recorded no coverage gaps: have 0, need 0 (nothing has been captured yet)
  [MISSING] at least 5000 stored features row(s): have 0, need 5000 (0 row(s) stored)
  [MISSING] at least 5000 stored labels row(s): have 0, need 5000 (0 row(s) stored)
  ready      : no
  -- retrain --
  newest     : none
  recommended: no - not ready: captured_days, coverage, feature_rows, label_rows
  -- warnings --
  - no trading day has been captured yet: run `--mode capture` on a trading day, then re-read this report
  - spec 4.3: too little data to conclude anything - keep capturing before reading anything into a report
  -- caveats --
  * this report describes the *dataset*, not the market: a satisfied requirement means there is enough recorded data to try something, never that something is true
  * a captured day is not necessarily a complete day - read coverage_ratio and gap_count, not the day count alone (a day whose ratio is below the configured floor is called out)
  * every sample size is count(*) of what is stored; nothing is extrapolated, imputed or smoothed, and a missing table is reported as unavailable rather than as zero
  * the coverage ledger is written by ANJANEYA-CAPTURE and ANJANEYA-ENGINE; a day can have ticks and still have no candles until --mode build-candles has run
  * readiness for retraining is a checklist about the data, not a claim about model quality: whether a retrain helps is decided by purged walk-forward validation, not here
  wrote      : ...\data\reports\accumulation\accumulation_2026-10-01_accumulation.json
  wrote      : ...\data\reports\accumulation\accumulation_2026-10-01_accumulation.md
  wrote      : ...\data\reports\accumulation\accumulation_2026-10-01_tables.csv
```

Note the two honest details an operator should read first: `instruments : 46,489 row(s)` carries
**no** day span (the table has no timestamp, so it does not claim one), and the readiness lines print
`have 0, need 0` for the coverage requirement with the reason *"nothing has been captured yet"*
rather than a bare `PASS`.

---

## 5. Assumptions and open questions

* **The report counts what the ledger recorded.** A session nobody started leaves no coverage row,
  so it is invisible here — the gap is between "attempted and gappy" and "never attempted". The
  capture heartbeat (`data/runs/heartbeat/capture.json`) and `scripts/check_calendar.py` remain the
  things that notice a session that never ran. A future stage could cross-check expected trading days
  (from the calendar) against captured ones and name the missing days explicitly; the report already
  receives `trading_days_in_window` exactly so that comparison exists, but it does not yet list the
  absent days.
* **"Captured days" is distinct `trading_date` in `data_coverage`.** A day with one stream row and a
  day with four hundred both count as one day. That is deliberate — the gate is about *days*, spec
  4.3's unit — and the per-day stream counts are what tell them apart.
* **The retrain recommendation is a date comparison, not a quality judgement.** It compares the
  newest captured day with the newest model artefact's mtime and the model's age against
  `retrain_max_model_age_days`. It never re-scores the model, and it says so. Whether a retrain helps
  is a walk-forward question.
* **The sample-size floors (5000 feature rows, 5000 label rows) are defaults, not measurements.**
  They are the "is there enough data to be worth starting a training run" line, chosen to be
  obviously too small to be a quality bar. They live in configuration precisely so they can be
  argued with.
* **`data_coverage`'s `note` column is not parsed.** The per-day verdict is built from
  `quality_flag`, `gap_count`, `largest_gap_seconds` and `coverage_ratio` — the machine-readable
  columns — never from a sentence, which would make the report's numbers depend on a free-text
  format.
* **Week buckets are aligned to the report day, not to ISO weeks.** `--weeks 4` on a Monday covers
  the four 7-day windows ending that Monday. That is what makes the number stable between runs on
  different days of the week; it is documented in the module and printed in the Markdown.
* **`--weeks` larger than the whole history is harmless.** The buckets simply report more zero
  weeks; `captured_days` is the ledger's own count and is not windowed.

---

## 6. Master-prompt checklist (§33)

| Item | Where |
| --- | --- |
| **1. Files created** | `core/gates.py`, `core/accumulation.py`, `core/storage/accumulation.py`, `apps/anjaneeya_ai/accumulation_writer.py`, `tests/unit/test_accumulation.py`, `tests/integration/test_ai_accumulation_pipeline.py`, `docs/phase_reports/phase_06.md`. |
| **2. Files modified** | `apps/anjaneeya_ai/run.py`, `core/ml/dataset.py`, `core/config_loader.py`, `core/app_bootstrap.py`, `core/storage/__init__.py`, `config/model.yaml`, `README.md`, `docs/runbook.md`. |
| **3. What was implemented** | Spec 29 (master-prompt phase 25) — the continuous data-accumulation report: `--mode accumulation` reads the stored tables and the `data_coverage` ledger read-only and writes a JSON, a Markdown and a per-table CSV covering data captured, coverage, gaps, sample sizes and readiness for retraining, with the newest model artefact's identity and age. |
| **4. Configuration added** | `model.yaml` → `accumulation:` (spec 29, `window_weeks`, `stale_days`, `healthy_coverage_ratio`, `retrain_min_feature_rows`, `retrain_min_label_rows`, `retrain_max_model_age_days`), validated by `AccumulationConfig`. The four spec 4.3 gate values stay in `data_gates:`. |
| **5. Tests created** | 45 unit (`tests/unit/test_accumulation.py`) + 19 integration (`tests/integration/test_ai_accumulation_pipeline.py`); **+64**. |
| **6. Test results (actual output)** | See §7 below. |
| **7. Example output (small, real)** | §4 — a verbatim run against this repository's database. |
| **8. Known limitations** | §5 above: a never-attempted session is invisible; "captured days" counts days not streams; the retrain recommendation is a date comparison; the sample-size floors are defaults; `note` is not parsed; week buckets are report-aligned. `schema_version` unchanged (no schema change); the report never writes a row to any table. |
| **9. Assumptions made** | The coverage ledger is the authority on what was captured; `trading_date` is the day unit; the newest `*_lightgbm.txt` under `data/models/` is "the model"; the calendar's `count_trading_days` is inclusive of both ends. |
| **10. Next phase will implement** | Master-prompt phase 24 — the decision layer — remains the one substantive piece not built, and the master prompt defers it explicitly ("later, separate… only after features, model and backtest are validated"). Everything below it now reports on itself: the gates (`core/gates.py`), the data (`--mode accumulation`), the model, the backtest and the report. Phase 25 is complete with this stage. |

---

## 7. Test results (actual output)

```
uv run ruff check .                 ->  All checks passed!

uv run pytest tests/unit/test_accumulation.py -q
    ->  45 passed

uv run pytest tests/integration/test_ai_accumulation_pipeline.py -q
    ->  19 passed in 160.21s (0:02:40)

# regression batch for the gate extraction + the touched modules:
uv run pytest tests/unit/test_config_loader.py tests/unit/test_ml_dataset.py \
                tests/unit/test_schema_contract.py tests/unit/test_engine_writer.py \
                tests/unit/test_capture_storage_writer.py tests/unit/test_accumulation.py -q
    ->  251 passed in 25.10s

uv run pytest tests/integration/test_app_entrypoints.py -q
    ->  36 passed in 18.73s      # includes every DEFAULT_MODES[APP_AI] entry, incl. accumulation

uv run pytest -q                     # the full offline suite
    ->  1929 passed, 1 skipped in 1731.70s (0:28:51)

        SKIPPED [1] tests/live/test_live_capture_smoke.py:60:
            live capture test requires ANJANEYA_RUN_LIVE=1 and real ZERODHA_*
            credentials (environment or .env)
```

The suite went 1865 -> 1929 passing (**+64**, the 45 unit and 19 integration tests above); the one
skip is the pre-existing live-capture smoke test, which is gated on `ANJANEYA_RUN_LIVE=1` and real
broker credentials and is not affected by this stage. No failures, no errors, no new warnings — the
gate extraction from `core/ml/dataset.py` into `core/gates.py` broke nothing across the 1865 tests
that already existed.





