# Phase 7 report — the decision layer (spec 24, master-prompt phase 24)

**Status: Stage 7.1 (`--mode decide`) complete.**

The master prompt ends its phase list with a line that it then repeats twice in the
configuration it ships:

```
PHASE 24 — Decision Layer (later, separate)
  Only after features, model and backtest are validated.
```

> "ANJANEYA never predicts BUY/SELL. It predicts future behaviour over multiple horizons; the
> decision layer (Phase 24) is separate." — `config/model.yaml`, twice

Every phase before this one was careful to leave that piece *visible as missing* rather than
fake it. `SignalPolicy` describes itself as "**not** spec 24's decision layer … the smallest
honest thing that lets a backtest measure costs", and `DECISION_LAYER_CAVEAT` travelled with
every explanation, importance ranking, chart and research report, saying so out loud.

Phase 7 builds the separate piece, and builds it as a **record**, not a signal: `--mode decide`
scores the stored feature rows with one model artefact and writes what **one configured rule**
would have done with each bar — `ENTER_CE` / `ENTER_PE` / `NO_TRADE`, the reason code, the size
and the exit plan — to `data/predictions/`. It is read-only against DuckDB, it places no order,
and it simulates no exit.

Five rules made it a record rather than advice:

* **The policy is configuration, and the record names it.** `model.decision` holds
  `min_probability`, `require_argmax`, `min_margin`, the side→class map, the size, the optional
  daily budget and the exit plan. The values *actually used* (including any `--min-probability`
  override) are written into the artefact, so a decision stays re-derivable. The backtest's
  `backtest.signal` block was left exactly as it was: the two policies are independent, and each
  artefact says which one ran.
* **The reason vocabulary is closed, ordered and tested.** `DECISION_REASONS` is the ladder
  (`ENTERED`, `UNMAPPED_OPTION_TYPE`, `DIRECTIONAL_CLASS_NOT_MODELLED`, `MISSING_PROBABILITY`,
  `BELOW_MIN_PROBABILITY`, `NOT_ARGMAX`, `MARGIN_BELOW_FLOOR`, `DAILY_LIMIT_REACHED`), the first
  failing rung names the reason, and the rollout prints every code — zeroes included — so "why is
  this a NO_TRADE?" is a value a test can assert rather than a paragraph a reader must parse.
* **The exit plan is stated, never simulated.** `stop_loss_atr`, `target_atr`,
  `max_holding_minutes` and `flat_at_session_end` are *echoed* into every row. There is no fill
  price, no match and no P&L anywhere in the module — replaying that plan against stored prices
  with costs is `--mode backtest`'s job, and doing it twice would create exactly the two-engines-
  disagree failure the project keeps designing out.
* **The day's budget is deterministic, and it is a throttle, not a filter.** When
  `max_decisions_per_day` is set the rows are walked in total order
  (`trading_date`, `timestamp`, `tradingsymbol`), so the budget goes to the *earliest* signals of
  the day across every symbol — and a row that already failed an earlier rung keeps its own
  reason instead of being relabelled `DAILY_LIMIT_REACHED`.
* **The gate labels; it never refuses.** Like `--mode backtest`, the mode writes its record and
  labels it (`BELOW_GATE`) with the gate it fell short of, and the model's own `results_label`
  travels in the artefact. A decision inherits the model's standing and adds nothing to it.

The mode reuses the project's existing seams rather than inventing new ones: `core.gates`'
ladder, `core.ml`'s `ModelPredictor`/`ModelArtefact` pair (feature order and categorical
vocabulary included), `core.storage`'s `read_features`, `trading_days` for the captured-day
count, `gate_verdict` for the label, and the atomic-writer pattern the accumulation stage
established. `schema_version` stays **1**: the mode writes no tick, candle, feature, label or
trade row — the only thing it writes is files under `data/predictions/`, the directory the
runbook has been reserving for this since Phase 1.


## 1. Files created

| File | What it is |
| --- | --- |
| `core/decision.py` | The pure model: `DecisionPolicy`, `Decision`, `DecisionRow`, `DecisionReport`, `build_decision_report`, `render_lines`, `render_markdown`, the reason vocabulary and the caveats. Imports nothing heavier than `core.gates`; no pandas, no LightGBM, no I/O. |
| `apps/anjaneeya_ai/decision_writer.py` | The writer: one `DecisionReport` → `_decisions.json`, `_decisions.md` and `_decisions.csv` under `data/predictions/`, atomically, counting every failure instead of raising. |
| `tests/unit/test_decision.py` | 44 unit tests: the policy ladder, the report's counts, the daily budget, the renderers and the configuration binding. |
| `tests/integration/test_ai_decision_pipeline.py` | 26 end-to-end tests through the real `main()` against a seeded database and a real model artefact pair. |
| `docs/phase_reports/phase_07.md` | This report. |

## 2. Files modified

| File | Change |
| --- | --- |
| `config/model.yaml` | New `decision:` block (spec 24's policy), with the comment that records why it is separate from `backtest.signal`; the `backtest.signal` comment updated (the decision layer now exists, and the replay still does not use it). |
| `core/config_loader.py` | `DecisionConfig`, `ModelConfig.decision`, and a `_decision_directions_are_label_classes` validator mirroring the existing signal one. |
| `core/app_bootstrap.py` | `decide` registered in `DEFAULT_MODES[APP_AI]`. |
| `apps/anjaneeya_ai/run.py` | `--mode decide` — `_decision_policy`, `_decision_caveats`, `_decision_rows`, `_decision_plan_lines`, `_decision_payload`, `_run_decide`, the `MODE_PLAN`/`AVAILABLE_MODES` entries, the `--min-margin`/`--max-decisions-per-day` flags, the `--min-probability` help (now backtest *and* decide) and the dispatch. |
| `core/ml/explainer.py` | `DECISION_LAYER_CAVEAT` reworded: the decision layer now exists, and *this* is still not it. |
| `core/backtest/engine.py` | The `SignalPolicy` docstring, the module docstring and the unknown-mode error now say "the decision layer is the separate `--mode decide`, and this replay does not use it" instead of "does not exist yet". |
| `core/research/report.py` | The ledger caveat reworded the same way. |
| `tests/integration/test_ai_explain_pipeline.py` | The one assertion that pinned the old caveat sentence now pins the new one. |
| `README.md` | A `--mode decide` section, the status row, the credential-free mode list, the backtest and explanation caveats, and the phase-report index. |
| `docs/runbook.md` | A §2.6 operator section for the decision layer, the explanation caveat, and the `data/predictions/` row of "where everything lives" (it was reserved; it is now written). |

## 3. What was implemented

* **The policy ladder, as one pure function.** `DecisionPolicy.decide(option_type, probabilities,
  modelled_classes)` returns `(decision, reason, numbers)` and nothing else. `modelled_classes`
  separates "the model cannot speak about this side" (`DIRECTIONAL_CLASS_NOT_MODELLED`) from
  "this row scored low" and from "the model gave no probability for it"
  (`MISSING_PROBABILITY`) — three facts the project would otherwise conflate.
* **The report, as one object with four readings.** `DecisionReport` exposes `entries`,
  `no_trade`, `sessions`, `entry_rate`, `reason_counts` (the whole vocabulary, including zeroes),
  `entries_by_option_type`, `entries_by_day`, `results_label`, `headline()`, `to_dict()` and
  `to_markdown()`; `render_lines()` renders the console block. JSON, Markdown and the console are
  three views of one in-memory object, so they cannot disagree.
* **The daily budget, deterministically.** `build_decision_report` walks the rows in total order
  and spends the day's budget on the earliest signals, leaving every already-refused row with its
  own reason.
* **The writer, atomically.** `DecisionWriter` shares its JSON/Markdown/CSV behaviour with the
  accumulation writer (temp file → `os.replace`, failures counted, never raised) and imports
  `DECISION_COLUMNS` from the model, so the CSV contract and the JSON rows stay one tuple.
* **The mode, read-only and gate-labelled.** `_run_decide` resolves the model (shared
  `_resolve_model_artefact`), refuses a timeframe mismatch (shared `_check_artefact_timeframe`),
  reads features, counts captured days with `trading_days`, gets the gate from `core.gates` and
  the label from `gate_verdict`, scores, decides, writes, registers every artefact on the run
  manifest, and logs/notes the headline, the verdict and every caveat. No model → `NO_MODEL`
  (`exit 0`); no scored rows → `NO_DATA` (`exit 0`); both are reported, neither is a crash.

## 4. Configuration added

```yaml
decision:
  spec: "24"
  min_probability: 0.40        # the directional class must reach this
  require_argmax: true         # ... and be the model's top class
  min_margin: 0.0              # ... with at least this lead over the runner-up
  directions:
    CE: UP_CONTINUATION        # checked against labels.classes at load
    PE: DOWN_CONTINUATION
  lots: 1
  lot_size: 75
  max_decisions_per_day: null  # null = no throttle
  stop_loss_atr: 1.0           # the exit plan the record states (never simulated here)
  target_atr: 1.5
  max_holding_minutes: 30
  flat_at_session_end: true
```

The values deliberately parallel `backtest.signal`, but they are a *separate* block: changing one
never silently changes the other. Every value is validated at load (ranges, the `CE`/`PE`
vocabulary, and `directions` against `labels.classes`), hashed into the config hash, and copied
into each artefact.

## 5. Tests created

* `tests/unit/test_decision.py` (44) — every rung of the ladder in isolation (`UNMAPPED`,
  `NOT_MODELLED`, `MISSING`, `BELOW_MIN_PROBABILITY`, `NOT_ARGMAX`, `MARGIN_BELOW_FLOOR`), the 13
  impossible-policy constructions, the reason vocabulary's containment, the budget's determinism
  and its per-day scope, the "the budget must not relabel a refused row" rule, the exit plan being
  present with no P&L column anywhere, "an empty report has no rate rather than a zero", the
  JSON/Markdown/console shapes, and the config binding (including `None` meaning "not
  overridden").
* `tests/integration/test_ai_decision_pipeline.py` (26) — the real `main()`: three artefacts per
  run and their names, one decision per scored row, the closed reason vocabulary over 480 rows,
  the model and its label travelling, the gate labelling an `INSUFFICIENT` run, the CSV contract
  and the absence of a simulated exit, no writes to DuckDB, manifest registration, the
  registered-mode guard, the four overrides changing the record *and* being recorded,
  `--symbol`/`--from`/`--to` scoping, `--dry-run`/`--no-write` writing nothing, `--json` being
  machine-readable, a missing model and an empty database being reported rather than fatal, a
  cross-timeframe model being refused, and the writer's atomic/counted behaviour (including a
  counted partial failure).


## 6. Test results (actual output)

Lint first — the whole repository, not just the new files:

```
$ .venv/Scripts/python.exe -m ruff check .
All checks passed!
```

The unit suite (every pure module, no model training):

```
$ .venv/Scripts/python.exe -m pytest tests/unit -q
1658 passed in 140.97s (0:02:20)
```

(The 44 new `test_decision.py` cases are inside that count; the file ran green on its own as
`44 passed in 5.38s` before the whole-suite run.)

The integration batches either side of the change, run through the real `main()`:

```
$ pytest tests/integration/test_ai_decision_pipeline.py tests/unit/test_decision.py -q
70 passed in 248.30s (0:04:08)

$ pytest tests/integration/test_ai_explain_pipeline.py tests/integration/test_ai_backtest_pipeline.py \
         tests/integration/test_ai_report_pipeline.py tests/unit/test_research_report.py -q
61 passed in 617.44s (0:10:17)

$ pytest tests/integration/test_app_entrypoints.py tests/unit/test_config_loader.py \
         tests/unit/test_ml_explainer.py -q
152 passed in 26.46s
```

Three things those runs pin down:

* `test_app_entrypoints.py` walks **every** mode of every app with `--dry-run` and refuses the
  answer "not implemented" (exit 3) for any of them — `decide` included — which is why the mode
  had to land with its implementation and an exhaustive `MODE_PLAN` entry.
* `test_ai_explain_pipeline.py` is where the reworded `DECISION_LAYER_CAVEAT` is asserted, and
  `test_ai_backtest_pipeline.py` is where `SignalPolicy`'s "a threshold rule, not spec 24's
  decision layer" sentence is asserted: both still pass, because the *claim* was kept true - the
  backtest and the explanation are still not the decision layer.
* `test_config_loader.py` validates the new `decision:` block (its ranges, its `CE`/`PE`
  vocabulary and its classes against `labels.classes`), and the config hash moves with it.

One caveat about *how* this was verified, recorded rather than glossed: this machine ran the
integration tier at roughly 15 s per test (each integration test fits a fresh LightGBM model and
writes a real database), so the whole 342-test integration directory was **not** re-run end to
end in this session - the unit suite and the seven files above were. Those seven cover every file
this stage touched (`core/decision.py`, `decision_writer.py`, `core/config_loader.py`,
`core/app_bootstrap.py`, `apps/anjaneeya_ai/run.py`, `core/ml/explainer.py`,
`core/backtest/engine.py`, `core/research/report.py`); the untouched integration files (capture,
engine, features, labels, greeks, the daily pipeline, and the charts/importance/dataset/train
pipelines) import none of them and were last green in Phase 6. The `decision:` block they all
load through `bootstrap_app` is the one shared dependency, and it is covered by
`test_config_loader.py` and by the bootstrap in every test that did run.


## 7. Example output (small, real)

The fixture run the integration tests exercise — a seeded four-session database (two contracts,
60 bars a day, 480 scored rows) and a real LightGBM artefact fitted on the fixture's own feature
columns. It is deliberately a *busy* rule: the fixture's class follows `option_type`, so every
row clears the shipped 0.40 floor, which is exactly the situation the "an entry rate is not an
edge" caveat exists for.

```
ANJANEYA-AI decision layer (spec 24) - as of 2026-10-01
  headline   : 480 decision(s) over 4 day(s): 480 entry(ies) (100.0%), 0 no-trade
  policy     : enter the option side whose directional class reaches p >= 0.40 and the model's top class
  results    : BELOW_GATE
  gate       : 4 captured day(s): too little data to conclude anything [INSUFFICIENT]; 16 more day(s) to EXPLORATORY
  window     : unknown .. unknown
  timeframe  : 1m
  model      : 1m_h5_2026-09-28_2026-10-01_lightgbm.txt (horizon 5m)
  -- decisions (by reason) --
  ENTERED                        : 480
  UNMAPPED_OPTION_TYPE           : 0
  DIRECTIONAL_CLASS_NOT_MODELLED : 0
  MISSING_PROBABILITY            : 0
  BELOW_MIN_PROBABILITY          : 0
  NOT_ARGMAX                     : 0
  MARGIN_BELOW_FLOOR             : 0
  DAILY_LIMIT_REACHED            : 0
  -- entries (by option side) --
  CE                             : 240
  PE                             : 240
  -- entries (by trading day) --
  2026-09-28                     : 120
  2026-09-29                     : 120
  2026-09-30                     : 120
  2026-10-01                     : 120
  wrote      : data/predictions/1m_h5_2026-09-28_2026-10-01_lightgbm_decisions.json
  wrote      : ..._lightgbm_decisions.md
  wrote      : ..._lightgbm_decisions.csv
```

`--dry-run` instead prints the plan and writes nothing (`rows : 480 stored feature row(s) would be
decided`), and `--min-probability 0.90 --max-decisions-per-day 2` turns those 480 entries into a
handful, with `DAILY_LIMIT_REACHED` and `BELOW_MIN_PROBABILITY` appearing in the counts.

## 8. Known limitations

* **The exit plan is not evaluated.** `--mode decide` records the stop, the target, the holding
  time and the session-end flatten; it never checks whether they would have been hit. The
  decision is therefore untested even as a *plan* until `--mode backtest` replays it. This is
  deliberate (one engine, not two), but it means a decision record is still only a record.
* **Repeated entries on one contract are not deduplicated.** The policy is evaluated
  independently on every bar; without `max_decisions_per_day`, a strongly-predicted contract
  yields an entry on every bar it appears in. The record does not model a position's life, so
  `entries` counts *decision points*, not trades — and the mode says so rather than pretending
  otherwise.
* **`max_decisions_per_day` is a throttle, not a portfolio.** It caps decision points per IST
  trading day and is spent in time order; it knows nothing about overlapping positions, capital
  or correlation. With no broker and no account model, anything more would be invention.
* **The rule is not fitted and not evaluated.** The thresholds are hand-set configuration. There
  is no optimisation, no walk-forward selection and no metric here — a decision record has no
  P&L by construction.
* **The decision is only as good as the model's row.** A `NULL` feature is passed through to
  `ModelPredictor` exactly as the explanation stage does; a low-quality or stale row decides on
  low-quality or stale inputs, and the probability inherits that.
* **The Markdown prints entries, not decisions.** The full list is in the JSON and the CSV; the
  Markdown counts the refusals by reason rather than tabulating them, so a 100,000-row record
  stays readable. Nothing is dropped, but reading the reasons row-by-row means opening the CSV.
* **One model, one timeframe, one horizon per run.** Shared with `--mode backtest`,
  `--mode explain` and `--mode importance`: several horizons at once is refused, and a model
  trained on another timeframe is a configuration error.
* **`schema_version` is unchanged (1).** The mode writes no database row at all.


## 9. Assumptions made

* **The model artefact is the model.** Nothing re-trains or re-calibrates; the probabilities are
  whatever the newest `*_lightgbm.txt` (or `--model`) produced.
* **The stored feature row is the input.** As in `--mode explain` and `--mode backtest`, the
  decision is made from the row the engine wrote, not from a recomputation.
* **Stored timestamps are IST-aware** (spec 6.6), so a row's `.date()` *is* the exchange-local
  trading day the daily budget is counted against. A naive timestamp would silently shift the
  day boundary — the schema guarantees it cannot.
* **`option_type` and `timeframe` are carried by the feature row** (they are part of the stored
  contract), so a decision can be attributed to a contract and a timeframe without a join. A
  frame missing either is a refusal, not a default.
* **The captured-day count comes from the scored rows**, exactly as `--mode backtest` counts the
  days its bars span — not from the coverage ledger (that is `--mode accumulation`'s job).
* **The gate never blocks a decision record**, because a decision is a description of a rule, not
  a result. The label carries the warning instead.
* **`--min-probability` is shared with `--mode backtest`.** One mode runs per invocation, so one
  flag cannot mean two things in one run; both artefacts record the value actually used.
* **English/IST/NIFTY assumptions** are inherited unchanged from every earlier phase.

## 10. Next phase will implement

With Phase 7 the master prompt's phase list is **complete**: phase 25 shipped as stage 6.1
(`--mode accumulation`), and phase 24 — the decision layer, the last named phase — ships here as
stage 7.1. Nothing in the master prompt is now unimplemented.

The honest next step is therefore not another named phase but *data*. `--mode accumulation`
already reports the captured-day count that gates everything above it, and `--mode decide`
demonstrably works on the four sessions the fixtures can build. The remaining work is the one the
master prompt cannot do for us: capture real sessions until the gate opens
(`ml_training_min_days`, then `meaningful_backtest_min_days`), re-run `--mode decide` with a
production model, and find out whether the decision record ever stops being labelled
`BELOW_GATE`. Until then every number here, the decision layer's included, is PRELIMINARY by
construction — which is the one thing the record will never hide.

> **Added in Phase 8.** "Capture real sessions" turned out to have one prerequisite that needed no
> market hours: the calendar. This report's own run was made on a `config/calendar.yaml` whose
> holiday list was still a seed, twelve of whose fifteen entries were wrong on a weekday — five of
> them closing days the exchange trades. Phase 8 reconciled it against NSE's official list and
> pinned the result; the config hash in the example output above has moved with it. See
> [`phase_08.md`](phase_08.md).
