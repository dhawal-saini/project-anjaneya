# Phase 4 report — ANJANEYA-AI (Stages 4.1-4.4: the matrix, its purged splits, the model, the backtest, and spec 24's report)

**Status: Stages 4.1, 4.2, 4.3 and 4.4 complete.** `--mode dataset` joins the stored feature rows to
the forward labels of the *same* instant (spec 21's `dataset_builder`), cuts the result with
spec 22's purged walk-forward splitter, and writes one Parquet matrix plus one split manifest
per configured horizon under `data/datasets/`. `--mode train` then consumes that same matrix
in memory: for every fold it fits a LightGBM multiclass model on the fold's surviving rows,
scores the fold's validation window, scores spec 21's baselines (always-up, always-range,
random, regime-only) on exactly the same rows, and finally refits one model on every row
outside the holdout, writing LightGBM's model text plus a metadata JSON that explains it
under `data/models/`. `--mode backtest` replays the stored bars against one of those models -
signal at a candle's close, fill at the next bar's open, stop/target/time/signal/session-end
exits - and writes a trade ledger with spec 23's Indian cost model charged on both legs.
All three modes read the database and write nothing else: no feature row, no label row, no
schema change (`schema_version` stays **1**).

Five rules from earlier phases are now mechanical rather than aspirational:

* **labels can never become features.** `core/ml/dataset.py` refuses any column named
  `label`/`label_code`/`label_version`/`horizon_minutes` or anything starting with
  `future_`, so no config knob (including `dataset.categorical_columns`) can smuggle a
  forward number into `X` - and `core/ml/trainer.py` runs the same guard immediately before
  the matrix reaches LightGBM.
* **the embargo must cover the horizon.** `SplitPolicy` refuses `embargo < horizon`
  outright, on top of the same check the configuration loader already performs.
* **the trainer sees exactly the rows the split counted.** `fold_frames()` rebuilds the
  guards from the fold's own stored cut-offs, so a training window can never quietly include
  a row the report says was embargoed.
* **the data gate refuses before any work happens.** Below `exploratory_min_days` a training
  run exits **4** with the numbers it needs, unless the operator passes `--force` - and a
  forced run keeps the `BELOW_GATE` label forever.
* **a signal can only fill where it could have.** In `core/backtest/engine.py` a decision read
  at bar *T* fills at bar *T+1+latency* or not at all (counted); the ledger's context fields
  come from the *signal* bar; and the labels table is never opened by a backtest at all.

---

## 1. What was built

| Module | Responsibility |
| --- | --- |
| `core/ml/dataset.py` (new body) | `DatasetPolicy` (the modelling choices), `build_dataset` (the exact-timestamp join, the class encoding, the counters), `Dataset` (matrix + identity + target, with honest reporting of constant/NULL/missing columns) and `data_gate_status` (spec 4.3's gates as a pure function). |
| `core/ml/walk_forward.py` (new body) | `SplitPolicy` (spec 22's numbers, self-validating), `Fold`/`HoldoutWindow`/`DatasetSplit` (exact timestamps, rows, class counts, embargo/purge counters) and `purged_walk_forward_splits` - whole trading days in order, holdout removed first, the boundary guarded by the embargo and counted by the purge. |
| `apps/anjaneeya_ai/dataset_writer.py` (new) | Atomic artefact writing: `<timeframe>_h<horizon>_<first>_<last>.parquet` plus `..._splits.json`, temp-file-then-rename, failures counted and logged rather than raised. |
| `apps/anjaneeya_ai/run.py` (extended) | `--mode dataset` with `--from/--to/--timeframe/--horizon/--symbol/--dry-run/--json`, a human and a JSON summary, per-horizon failure isolation, artifacts registered in the run manifest, exit codes 0/1/2/3. |

Supporting changes:

* `core/config_loader.py` — a validated `dataset:` section (`timeframe`, and
  `categorical_columns` checked against `FEATURE_COLUMNS` *and* against the leakage guard);
  `ModelConfig.dataset`.
* `config/model.yaml` — the `dataset:` section, with the reason for every default in a
  comment.
* `core/ml/__init__.py` — re-exports the new vocabulary; `baselines`/`trainer`/`evaluation`
  stay docstring-only placeholders until the model stage lands.
* `core/app_bootstrap.py` — `dataset` added to ANJANEYA-AI's `--mode` choices.
* `tests/support/ai.py` (new) — four synthetic contract-days of feature and label rows with
  the parts a dataset cares about: a session tail whose label is NULL, all four classes, a
  real categorical vocabulary, and the stored row contract.

### Stage 4.2 — the model

| Module | Responsibility |
| --- | --- |
| `core/ml/trainer.py` (new body) | `TrainingPolicy` (every number the trainer uses, self-validating, algorithm-gated), `TrainedModel` (the fitted booster plus the metadata that scores it: feature order, categorical vocabulary, importances), `train_model` (the refusals and the fit) and the walk-forward *run* itself: `train_and_evaluate`, `FoldResult`, `TrainingReport`. |
| `core/ml/baselines.py` (new body) | Spec 21's four reference points, each fitted on the training fold only: `always_up`, `always_range`, `random` (seeded, deterministic), `nifty_regime_only` (the training fold's class distribution per regime, add-one smoothed, unseen/missing levels falling back to the prior). |
| `core/ml/evaluation.py` (new body) | `EvaluationPolicy`, the probability metrics (`log_loss`, `brier_score`, `accuracy`, `calibration_curve` with expected calibration error), `ProbabilityMetrics`, `MetricAggregate`, `FoldComparison` / `compare_to_baselines` (the model vs every baseline on the same rows), and `gate_verdict` (spec 4.3 as a label: `BELOW_GATE` / `EXPLORATORY` / `PRELIMINARY`). |
| `core/ml/walk_forward.py` (extended) | `fold_frames`, `holdout_frame`, `usable_frame` - the exact rows a fold may train on, the untouched holdout, and everything outside it - plus a `purge` flag on `Fold` recording whether the purge was applied or only counted. |
| `apps/anjaneeya_ai/model_writer.py` (new) | Atomic model artefacts: LightGBM's own model text plus `<dataset_name>_<algorithm>_metadata.json` carrying the dataset, the training policy, the per-fold metrics and baselines, the aggregates, the gate verdict, the caveats and the importances. |
| `apps/anjaneeya_ai/run.py` (extended) | `--mode train` with `--seed/--score-holdout/--force/--no-write` on top of the dataset flags, a human and a JSON summary, the gate refusal (exit 4), per-horizon isolation, and the model artefacts in the run manifest. |

Supporting changes for Stage 4.2:

* `core/config_loader.py` — a validated `training:` section (`n_estimators`, `learning_rate`,
  `num_leaves`, `max_depth`, `min_child_samples`, the fractions, the lambdas, `class_weight`,
  `early_stopping_rounds`, `n_jobs`); `ModelConfig.training`.
* `config/model.yaml` — the `training:` block, with the reason for every default (and why
  early stopping is off).
* `core/app_bootstrap.py` — `EXIT_GATE_REFUSED = 4`: "this run refused the work on purpose",
  distinct from a crash (1) and from a configuration error (2).
* `tests/support/ai.py` (extended) — `shrink_config` (a configuration whose walk-forward and
  tree sizes fit a four-day fixture), `signal_frame` (a frame whose class is learnable from
  two of its features), `ONE_DAY_FOLDS` and `FAST_TRAINING`.
### Stage 4.3 — the backtest

| Module | Responsibility |
| --- | --- |
| `core/ml/predictor.py` (new body) | Spec 21's missing `predictor`: `ModelArtefact.load` reads the artefact **pair** (LightGBM's model text + the metadata JSON beside it), `ModelPredictor.matrix` rebuilds the training feature order and the training categorical vocabulary (an unseen level becomes *missing*, never a new code), and `predict_frame` returns the keys plus one `p_<class>` column per class and the model's top pick. A booster without its metadata, or a pair whose feature counts disagree, is refused. |
| `core/backtest/costs.py` (new body) | The Indian F&O contract note: `CostPolicy` (brokerage per order, STT on sell premium, exchange txn, SEBI fee, GST on the charges, stamp duty on the buy side, slippage in bps, an optional captured spread), `CostModel.order`/`round_trip` returning an `OrderCost`/`TradeCosts` with every charge named, and counters for spreads used/ignored. Refuses a negative rate and a non-positive spread. |
| `core/backtest/engine.py` (new body) | `SignalPolicy` (the threshold rule, a documented stand-in for spec 24) and `SignalDecision`; `ExecutionPolicy` (fill mode, latency rounded up to bars, sizing from lots x lot size, ATR stop/target, holding limit, session-end flatness, intrabar priority); `Trade` (the 28 stored columns plus three research columns); `BacktestCounts` (every rejection counted); `BacktestResult`; and `Backtester.run` - the session-by-session replay with gap/stop/target/time/signal/session-end exits, MFE/MAE and both legs charged. `trading_days()` defines a day the way the calendar and the gates do. |
| `core/backtest/metrics.py` (new body) | Spec 25: `TradeMetrics` (win rate, averages, payoff, profit factor, expectancy, gross/costs/net, cost ratio, drawdown, holding time, exposure, a per-trade and an annualised Sharpe-like ratio), `DistributionSummary` (min/max/mean/sd + percentiles), `max_drawdown`, and the segmented tables (`option_type`, `time_of_day_bucket`, `expiry_cycle_phase`, `day_of_week`, `days_to_expiry`, `strike_distance`, `nifty_state`, `exit_reason`, `expiry`) with their sample size and a `PRELIMINARY` flag. |
| `apps/anjaneeya_ai/backtest_writer.py` (new) | Atomic artefacts: `<timeframe>_h<horizon>_<first>_<last>_ledger.parquet` (spec 23's columns + the research extras), `..._summary.json` (policies, model, counters, metrics, segments, caveats), and the same rows validated through `TradeRecord` into the stored `trades` table. |
| `apps/anjaneeya_ai/run.py` (extended) | `--mode backtest` with `--model`, `--min-probability`, `--from/--to/--timeframe/--horizon/--symbol/--dry-run/--json/--no-write`, the model discovery (newest artefact, or an explicit path, with the horizon cross-checked), a human and a JSON summary, the gate *labelled* (never refused), and the artefacts registered in the run manifest. |

Supporting changes for Stage 4.3:

* `core/config_loader.py` — a validated `backtest:` section: `BacktestSignalConfig`
  (`mode`, `min_probability`, `require_argmax`, `directions`), `BacktestSizingConfig`
  (`lots`, `lot_size`, `quantity`), `BacktestExitConfig` (`stop_loss_atr`, `target_atr`,
  `max_holding_minutes`, `flat_at_session_end`, `intrabar_priority`,
  `exit_on_opposite_signal`); `ModelConfig.backtest`. A direction that
  `labels.classes` does not define is a load-time error, because every signal would be dead
  on arrival.
* `config/model.yaml` — the `backtest:` block, with the reason for every default (including
  why the signal rule is a threshold and why the intrabar tie goes to the stop).
* `core/ml/trainer.py` — `_feature_matrix` promoted to a public `feature_matrix`, because the
  predictor must build *the same* matrix the trainer built.
* `core/ml/__init__.py` — re-exports the predictor vocabulary (`ModelArtefact`, `ModelPredictor`,
  `ModelError`, `PREDICTION_COLUMNS`, `predictor_columns`, `metadata_path_for`, `feature_matrix`).
* `core/backtest/__init__.py` — re-exports the cost, engine and metric vocabulary.
* `tests/support/ai.py` (extended) — stored 1m bars with a predictable premium path (a CE
  drifts up, a PE down), `signal_class_frame` (a target that follows `option_type`, so a
  crafted model fires on both sides and the replay yields a winner *and* a loser),
  `write_signal_model` (a real artefact pair, written with `TrainedModel.to_manifest()`),
  and a `backtest:` override in `shrink_config`.
* `tests/integration/test_app_entrypoints.py` — ANJANEYA-AI's unplumbed mode moves from
  `backtest` to `report`: `--mode backtest` now runs, so the "a mode that exists but is not
  implemented exits 2" guard needed a mode that still has not been built.
* `tests/integration/test_ai_dataset_pipeline.py` — that file's "building a dataset writes no
  stored data" check now counts `candles` alongside `features` and `labels`, because the
  backtest stage reads candles for real and the assertion is worth making explicit.

Supporting changes for the operator half of Stage 4.3 (the data-capture runbook):

* `scripts/run_daily_pipeline.py` + `scripts/run_daily_pipeline.{ps1,sh}` — one command for
  the evening: the eight steps in dependency order (`candles → greeks → features → labels →
  validate → dataset → train → backtest`), a declarative step table, a read-only pre-flight
  (day resolution, tick and `index_history` counts, calendar contradictions, provisional
  holidays within 7 days), `--date/--timeframe/--horizon/--skip/--force/--dry-run/
  --continue-on-error/--json/--log-file` plus the global path flags forwarded to every child,
  one dated log (`data/logs/daily_pipeline_<date>.log`) and one verdict. Exit codes: `0` ok,
  `1` a step failed, `2` configuration, `4` only the training gate refused.
* `scripts/check_calendar.py` — the pre-session guard: the next N days with their session
  window, why each one is closed, the current/next weekly and the monthly expiry, the
  provisional holidays inside a risk window (`--within`, default 10 days) and
  `--require-trading-day` for a scheduler. Exit `1` when a provisional date can cost a
  session, `2` when the configuration cannot be read.
* `core/market_calendar/nse_calendar.py` — `holiday_entry()` and `holidays_between(start,
  end, verified=...)`, so "which dates has nobody confirmed yet" is a calendar query rather
  than a loop over the config in two different scripts.
* `apps/anjaneeya_capture/run.py` — every live session prints (and logs) an advisory naming
  the provisional holidays within 7 days, before it does anything else. Advisory, not a
  refusal: the operator may know something the YAML does not.
* `docs/runbook.md` — the operator's day: first-time setup (credentials, calendar, the
  one-way universe decision, instrument master, index history), the morning/evening routine,
  weekly checks, Windows Task Scheduler and cron snippets, the failure playbook, the
  gate/walk-forward arithmetic, and the six rules that keep the data honest.
* `tests/unit/test_daily_pipeline.py`, `tests/unit/test_check_calendar.py`,
  `tests/unit/test_market_calendar.py` (extended), and
  `tests/integration/test_daily_pipeline_e2e.py` (the real chain, real child processes:
  engine half gated on the shipped config; dataset → train → backtest on a walk-forward that
  fits the fixture).

---

## 2. Design decisions worth recording

**The join is exact, and there is no fallback.** `build_dataset` merges on
`(timestamp, tradingsymbol)` with `validate="one_to_one"`. No `merge_asof`, no tolerance:
a label 30 seconds away describes a different decision moment, and borrowing it is the kind
of error that makes every number in a report slightly wrong without ever announcing itself.
Both directions of an unmatched join are counted separately (`features_without_label`,
`labels_without_feature`), so "why is the matrix smaller than the feature table?" always has
a number for an answer.

**A row without a class is not a training row, but it is not a secret either.** The label
builder deliberately stores a row whose forward window did not fit inside the session with
`label = NULL`. The dataset drops it from the matrix and publishes the count beside the
count it kept (`unlabelled_rows` vs `labelled_rows`), which is what makes an ATR warm-up or
a truncated session visible in a run summary instead of invisible in a row count.

**Features are the stored contract minus identity minus labels.** `feature_columns` is
derived by filtering `FEATURE_COLUMNS` (never by taking "everything else"), so a column that
appears in the frame by accident cannot become a feature, the column order is identical
between runs, and a column missing from the frame is *reported*
(`missing_feature_columns`) rather than filled. `option_type` is deliberately kept: CE vs PE
is part of what RQ1/RQ2 ask about, so it ships as a categorical feature.

**Categorical columns ship as pandas `category` dtype.** The Parquet artefact then records
the vocabulary, and the trainer can hand a tree model the codes it wants without
re-deriving them from strings. Nothing is encoded *here*: encoding is a fold-level decision
(spec 22), so the artefact stays readable.

**Nothing is imputed, nothing is dropped for being awkward.** Constant and mostly-NULL
columns are reported (`constant_columns`, `null_shares`) and left in place: feature
selection must be fit on training folds only, and a column that is constant across the whole
scope may still vary inside a fold.

**Splits run on the days the dataset has, not on a calendar.** A gap (a holiday, a day with
no capture) simply is not a day, so no fold window is padded with empty days. The final
`holdout_days` are removed from *every* window before the walk starts - the untouched period
spec 22 demands - and the fold report carries the exact first/last timestamps of both
windows, the two cut-offs, and how many rows each guard removed.

**The embargo is the rule; the purge is the same rule expressed through the label window.**
The embargo drops every training row within `embargo_minutes` of the validation window; the
purge counts every training row whose label window `[t, t + horizon)` would reach into it.
With an embargo that covers the longest horizon the purged set is a subset of the embargoed
set - and that is the point: the splitter *refuses* `embargo < horizon`, so the two guards
can never disagree. `purge: false` in `walk_forward` therefore changes nothing but the
report, which is exactly what the counters are for.

**The gates are a function, not a sentence.** `data_gate_status(days, exploratory=...,
ml_training=..., meaningful_backtest=..., seasonality=...)` returns the stage, what that
stage permits, and how many days are missing to the next one. `--mode dataset` prints it,
logs a warning below `ml_training_min_days`, and puts it in the run manifest's notes - so a
research artefact built on four days says so in its own metadata.

**One matrix per horizon, named after what it covers.** A dataset is a frozen research
artefact, so it is written as `1m_h5_2026-09-28_2026-10-01.parquet` beside
`1m_h5_2026-09-28_2026-10-01_splits.json`. A second run replaces them (temp file, then
rename), so an interrupted run cannot leave a half-written matrix behind.

**One honest limitation, printed on every run.** The `labels` table is keyed
`(timestamp, tradingsymbol, horizon_minutes)` and has **no `timeframe` column**, so nothing
in the stored rows records whether a label came from 1m or 5m candles. A matrix built from
5m bars and one built from the 1m rows at the same instants are indistinguishable to the
join - and the label build's delete-then-append per day makes mixing timeframes within one
contract-day impossible rather than merely unlikely. `--mode dataset` therefore prints (and
records as a run note) that the stored labels are *assumed* to have been built at the
dataset's timeframe, and `core/ml/dataset.py` refuses a repeated label key with a message
that names this exact trap. Closing it properly means a `timeframe` column on `labels` - a
schema change, deliberately not taken in a stage whose contract is "no schema change".

---

## 2b. Stage 4.2 design decisions worth recording

**The trainer refuses more than it fits, on purpose.** An empty training frame, a frame with
one class, a frame without class codes, codes outside the configured classes, a duplicated or
missing feature column, a categorical column that is not a feature, a label-derived column
offered as a feature, and an algorithm this stage has not implemented - each one raises
`TrainingError` naming exactly what is wrong. A single-class frame is the important case:
LightGBM would happily fit it and every metric would be perfect, so the *walk-forward run*
catches that error and reports the fold as `SKIPPED` with its reason, rather than aborting a
run that is otherwise measuring something - and a skipped fold is never counted as a good one.

**Nothing is tuned on the data it is measured on.** `early_stopping_rounds` defaults to
**0** (off), and when it is on the report and the model metadata both say that the number of
trees was chosen on each fold's own validation window. Every LightGBM sub-seed is pinned to
the single configured seed with `deterministic=True` and `force_row_wise=True`, so two runs
of the same command over the same matrix produce the same model - which is what makes a
reported number re-derivable.

**The baselines are fitted per fold, not globally.** `nifty_regime_only` learns the training
fold's class distribution per regime; a baseline fitted on everything would beat the model
for free. `random` re-seeds on every call, so the same frame and seed always give the same
probabilities regardless of call order. The constant baselines are deliberately *not*
smoothed - their log loss is dominated by the clipping, which is the honest way to say "this
rule is wrong whenever it is wrong".

**A metric that was not measured is `None`, never `0.0`.** `ProbabilityMetrics` computes
exactly the metrics `evaluation.probability_metrics` asks for; anything else is `null` in the
JSON. The model-vs-baseline comparison always measures log loss, Brier score *and* accuracy on
both sides whatever the configuration says, because a delta on one metric is not a
comparison: a model can win on log loss while its argmax never matches anything.

**Clipping is applied *and counted*.** A confidently wrong one-hot costs infinite log loss;
`core/ml/evaluation.py` clips to `1e-9` and renormalises the row, then reports how many rows
needed it (`clipped_predictions`) - a log loss of 20 that is really "the model said never"
must not read like a finely-measured number.

**The across-fold table keeps its values.** `MetricAggregate` carries the raw per-fold numbers
alongside mean/min/max/sd, and any run with fewer than three trained folds says in its notes
that the spread is descriptive rather than an estimate. No hypothesis test is performed here:
`evaluation.significance_level` and the multiple-testing correction belong to the reporting
stage (spec 25), where there is a family of comparisons to correct for.

**The final model is fitted on everything outside the holdout - and the holdout is only
scored when asked.** `usable_frame()` is what the artefact is trained on;
`holdout_frame()` is what it is checked against, and only with `--score-holdout`. When that
flag is passed the report and metadata both record that the holdout has been looked at once
and is no longer untouched - a holdout you can consult twice is not a holdout, and the run
says so rather than quietly allowing it.

---

## 2c. Stage 4.3 design decisions worth recording

**The bridge from a prediction to a position is a threshold, and every artefact says so.**
Spec 21's model outputs a distribution over *behaviour* classes; spec 23 asks for a backtest;
spec 24's decision layer is explicitly later. `SignalPolicy` fills that gap with the smallest
honest rule available - enter the option side whose directional class the model gives at least
`min_probability` for (`CE` follows `UP_CONTINUATION`, `PE` follows `DOWN_CONTINUATION`),
optionally requiring that class to be the model's top pick. It is printed by the report ("a
threshold rule, not spec 24's decision layer"), stored in the summary, and repeated as a
caveat, so no number produced by it can be read as an edge. `ModelConfig` refuses a direction
that `labels.classes` does not define - a rule that could never fire is a configuration bug,
not a quiet zero.

**No look-ahead is structural, not a promise.** A signal is read at the close of bar *T* and
can only fill at the open of bar *T+1+latency*; when that bar does not exist inside the same
session the signal is dropped and counted (`no_fill_bar`). The intrabar tie case - a bar that
touches both the stop and the target - cannot be resolved from bars, so it is resolved by
`exit.intrabar_priority` (stop-first by default, the pessimistic choice) *and counted*
(`both_touched`). Every contextual ledger field (`nifty_state`, `candle_state`, the bucket,
the cycle phase, the ATR the stop came from) is read from the **signal** bar's feature row,
never from the fill bar's. And a backtest never opens the labels table at all.

**Friction is a cost, not a price.** `entry_price`/`exit_price` are the stored bars' own
prices; every rupee of brokerage, STT, exchange charge, SEBI fee, GST, stamp duty and
slippage is charged into `costs`, in one object (`OrderCost`) whose fields are the contract
note's own lines. A reader can therefore subtract the friction and see the frictionless
number, which is exactly what a cost model is for. GST sits on (brokerage + exchange + SEBI)
and never on the premium or on STT - and a captured spread, when a run enables one, *replaces*
slippage on that side rather than being billed on top of it.

**Every rejection is a number.** `BacktestCounts` reports how many bars had no feature row,
no prediction, no decision, no fill bar, and how many positions each rule closed. "Why are
there only two trades?" is answered by the run rather than by a debugging session - and each
counter has a hand-built fixture in `tests/unit/test_backtest_engine.py` that moves it on its
own.

**A trade that cannot be described is refused, not stored.** The engine refuses a candle
frame with a non-positive or non-finite price (a premium that is not a positive number is a
data defect), a probability row that is not a distribution (`sum != 1` within 1e-3, or a value
outside `[0, 1]`), a frame missing a column the policy needs, and an ATR exit without an
`option_atr` column. At the other end the writer validates every row through `TradeRecord`
before it reaches the stored table, so a malformed ledger row can only ever exist in memory.

**The ledger is spec 23's contract *plus* what the table cannot hold.** The stored `trades`
table already existed in schema version 1 with 28 columns, so the backtest writes exactly
those - and carries three research-only columns (`distance_from_atm`, `day_of_week`,
`bid_ask_spread`) in the Parquet ledger and the segmented metrics, which is what keeps
`schema_version` at **1**. Losing the strike-distance segmentation to a schema change was not
worth it; pretending the column was stored was not an option.

**Holding time is measured to the fill, and the time exit is checked the same way.** A bar is
stamped at its start, so a position flattened at a bar's *close* was held one bar longer than
the two timestamps suggest. `_Exit.at_close` records which kind of fill ended the trade, and
the ledger's `holding_minutes` uses it - which is why a `TIME_EXIT` trade always reports at
least `max_holding_minutes` of holding. Without that, a 30-minute limit would have produced
29-minute trades and looked like a bug.

**A backtest is labelled, never refused.** `--mode train` refuses below the exploratory gate
because it would be fitting a model it has to call production. A backtest below
`meaningful_backtest_min_days` (120 days) is *exactly* what an exploratory research engine
should allow - so the backtest runs, exits 0, prints `BELOW_GATE` with the day counts, and
stores the verdict in its summary. The claim "statistically meaningful after costs" (RQ4) is
earned by the gate, not asserted by the run.

**A run with no signals is a run with no signals.** If nothing clears the threshold the
ledger is empty, the metrics are `None` rather than `0.0` (no profit factor without trades), and
the counters say how many decisions were seen. That is a result - and it is the result the real
four-day fixture produces, which is why the integration tests assert on structure (the ledger,
the counters, the caveats) rather than on a trade count.

**The artefact carries its own explanation.** The metadata JSON stores the dataset it came
from (`dataset_name` + `dataset_artefact`), the whole training policy, the feature columns and
the categorical vocabulary. The vocabulary matters: at scoring time a categorical level the
model never saw is cast to *missing* rather than to a different integer code - the failure
that silently turns a prediction into noise.

**Spec 4.3 is a refusal, not a warning.** Below `exploratory_min_days` the mode exits **4**
(`EXIT_GATE_REFUSED`) after printing the days it saw and the days each gate needs, and writes
nothing. `--force` turns it into a run that keeps the `BELOW_GATE` label in the report,
the metadata and the run manifest. Above the gate, `EXPLORATORY` and `PRELIMINARY` are the
labels - a model metric is not a backtest, and every run stores that caveat next to its
numbers.

---

## 3. Verification performed

`uv run pytest -q` — **1649 passed, 1 skipped** (the count after Stage 4.4; §6.4 lists the
focused commands for that stage, and the 1 skip is the opt-in live capture smoke test in
`tests/live/`, which needs real credentials). The three AI integration files are the slow part - they
train real LightGBM models on the seeded store - and they run first (path order), so the
suite is best run in the background with the log tailed. The operator half of this stage adds
71 tests: the two new unit files (38 + 15), the capture advisory and status report (8), six
more calendar tests, and four end-to-end pipeline tests that spawn the real apps.
`uv run ruff check .` — clean.

New tests so far (Stage 4.1: 55 + 25 + 25 = 105; Stage 4.2: 60 + 19 + 43 + 6 + 20 = 148;
Stage 4.3 below: 18 + 46 + 15 + 11 + 14 + 16 = 120, plus the 71 operator tests;
**444 in Phase 4**):

* `tests/unit/test_ml_dataset.py` (55) — the policy read from the shipped configuration and
  overridden; the leakage guard by name *and* by the `future_` prefix; the join's
  bookkeeping (feature rows in, labelled rows out, unlabelled and both directions of
  unmatched rows counted, the counts forced to add up); the join proved to be *exact* (a
  label two minutes away matches nothing); class codes following the configured order; a
  stored class nobody configured, two label versions in one matrix, a repeated key on either
  side, a missing key column, a naive timestamp, an unconfigured horizon and a configured
  categorical column absent from the frame - each refused with a message that names it; the
  feature-column contract (the stored order, identity and provenance excluded, `option_type`
  kept, every label-derived name excluded, categoricals as `category` dtype, the matrix
  carrying features only); constant/NULL/missing columns reported and never removed; the
  empty answers (no features, no labels for that horizon, every label NULL) each keeping
  their counts; and the spec 4.3 gate ladder at 0/19/20/59/60/119/120/249/250/500 days.
* `tests/unit/test_ml_walk_forward.py` (25) — the policy from the shipped configuration (and
  the four ways a policy that cannot be leak-free is refused, including
  `embargo < horizon`); folds walking day by day with the validation window strictly after
  the training window; the holdout being the last days and in no fold; a step larger than
  the training window; days the dataset does not have never invented; a short dataset
  returning no folds *with the number of days still missing*; the empty dataset; the embargo
  removing exactly the tail of the training window (29 rows with a 30-minute embargo, and
  the row *on* the cut-off kept); the purge counting the rows whose label window reaches the
  validation window and being a subset of the embargoed set; `purge=False` changing nothing
  but the report; a whole-session fixture where both guards are trivially satisfied;
  per-fold class counts adding up; the split totals being the sum of its folds; the manifest
  being JSON-ready with ISO timestamps; and the same dataset splitting identically twice.
* `tests/integration/test_ai_dataset_pipeline.py` (25) — the real CLI over a seeded database
  (four contract-days): one matrix *and* one manifest per horizon, no `.tmp` left behind;
  the matrix's columns (class and code, the forward numbers beside them, the identity
  columns, the timeframes, the ordering); each horizon keeping only its own labels; the
  manifest's policy, dataset description and holdout; the gate verdict printed and recorded;
  a shrunken `walk_forward` producing real folds with exact timestamps (and the validation
  window proven to follow the training window); `--horizon`, `--from/--to`, `--symbol`,
  `--dry-run` (plan printed, nothing written) and `--json`; a bad date and an unconfigured
  horizon as configuration errors, and a timeframe nobody enabled too; the sources
  *untouched* (features, labels, trades and candles all unchanged); a second run replacing
  the artefacts instead of piling up; the run manifest listing the artefacts, the gate and
  the labels-timeframe caveat; no feature rows and no labelled rows reported as NO_DATA /
  NO_ROWS rather than errors; a label row with no feature row counted; the primary key
  preventing a stored duplicate; and a simulated build failure reported per horizon, written
  nowhere, with exit code 1.

Manual check (synthetic four-day store, 2 contracts × 60 one-minute bars, scratch
directory): `--mode dataset` runs in **2.2 s** end to end (about 1.5 s of that is the
`uv run` startup) and writes 8 files (4 horizons × 2), 992 matrix rows in total. The printed
summary names the gate (`4 captured day(s): too little data ... [INSUFFICIENT]; 16 more
day(s) to EXPLORATORY`), the label-timeframe caveat, the class mix
(`UP_CONTINUATION 64, DOWN_CONTINUATION 64, RANGE 64, REVERSAL 56`), the feature split
(`77 (66 numeric, 11 categorical; 70 constant here, 0 missing from the frame)`), and the
split: with the shipped `walk_forward` policy it is `no folds - needs 135 captured day(s)`,
and with a shrunk policy (train 2 / validate 1 / holdout 1) it becomes
`1 fold(s) over 4 day(s), holdout 1 day(s) untouched` /
`fold 0: train 124 row(s) (0 removed: 0 embargoed, 0 purged) -> validate 62 row(s)`. The
fixture's rows stop at 10:14, so a 30-minute embargo across a session boundary removes
nothing - which is the honest result for session-bounded data, and the splitter's unit
tests use a fixture whose rows run through a calendar day to make the guard actually bite.
Running the mode against the repository's own (feature-free) database prints the "no stored
feature rows" message and writes nothing, which is what a fresh checkout should do.

Stage 4.2 tests:

* `tests/unit/test_ml_trainer.py` (60) — the policy from the shipped configuration (seed
  override, `describe()`), the LightGBM parameters pinning every sub-seed with
  `deterministic=True` and refusing a single-class problem, the balanced class weights
  (inverse counts, zero for an absent class, sklearn's convention that the weights sum to the
  sample count), fourteen policy refusals, and the trainer's own refusals: an empty frame, a
  single-class frame, a frame without class codes, non-integer codes, codes outside the
  configured classes, no features, duplicate features, a missing feature, a categorical that is
  not a feature, a `future_*` column offered as a feature, a non-numeric column that is not
  declared categorical, one configured class. Then the learning: the model beats the 25% base
  rate on a frame whose class is decided by two features, probabilities are a distribution per
  row, training twice is identical, a different seed differs, an unseen categorical level and a
  NaN feature still score (as *missing*, not as a code), the vocabulary is recorded, importance
  covers every feature and the two signal columns out-rank the noise, and the manifest records
  the algorithm, horizon, classes, seed and early-stopping state. The walk-forward run: every
  fold scored over exactly its validation rows, `fold_frames()` matching the fold's own counts,
  the final model fitted on everything outside the holdout, the holdout untouched unless asked
  (and saying so when it is), every baseline scored on the same rows with consistent deltas, a
  single-class training fold *skipped* while the other fold trains, a run with no trainable fold
  reporting rather than crashing, a split that cannot fold explaining why, the verdict and its
  caveat travelling into the report, early stopping flagged as optimistic, and the whole report
  JSON-serialisable.
* `tests/unit/test_ml_baselines.py` (19) — the four configured baselines are exactly the four
  implemented ones; each returns a distribution; constant baselines give their class
  probability 1 and need that class configured; `random` is reproducible for a seed, differs
  for another, and ignores the frame's content; `nifty_regime_only` learns the training fold's
  per-regime distribution (checked against a hand-computed smoothed frequency), falls back to
  the prior for an unseen level and for NaN, does not invent a "nan" level, needs its column,
  and smooths a level seen once; plus the refusals (unknown name, empty list, empty frame,
  duplicate classes, non-positive smoothing).
* `tests/unit/test_ml_evaluation.py` (43) — hand-checkable arithmetic (a perfect prediction
  scores 0, a uniform one scores `log k`, a confidently wrong one-hot scores the clipping floor
  and is *counted* as clipped, clipping leaves a reasonable prediction alone), the
  shape/dtype/value refusals, the calibration curve (a uniform prediction is perfectly
  calibrated, an over-confident one reports the gap, empty bins are `None` rather than `0`, a
  curve needs two bins, every bin serialises), the policy (from the shipped configuration, and
  refusing an unknown metric, a non-positive sample floor and a single bin), "only the requested
  metrics are computed" (`None`, not `0.0`), the aggregates (mean/min/max/sd, raw values kept,
  `None` ignored, an empty aggregate saying so), the model-vs-baseline comparison (deltas, the
  best baseline, when the baseline wins, the three always-measured metrics, a custom label for a
  holdout, `PRELIMINARY` below 30 rows and `MEASURED` above), and every gate stage's label
  including `--force`.
* `tests/unit/test_ml_walk_forward.py` (+6, 31 total) — `fold_frames` returning exactly the rows
  the fold counted with the guards in them, disjoint from the validation rows, agreeing with
  `purge: false`; the holdout frame being the untouched window; the usable frame being
  everything else; and a split without a holdout handing out the whole dataset.
* `tests/integration/test_ai_train_pipeline.py` (20) — the real CLI over the seeded fixture and
  a shrunken configuration: refusal below the exploratory gate (exit 4, nothing written,
  `--force` mentioned), the refusal recorded in the run manifest, `--force` training with
  `BELOW_GATE` in the report/metadata/coverage, the fold metrics matching the split's own
  records (row counts, baselines on the same rows, deltas consistent with the log losses,
  `MEASURED` labels), the model's features being the dataset's features with nothing
  label-derived among them, LightGBM's model text loadable as a model file, the metadata
  explaining the model (policy, classes, importances, baseline aggregates, notes, the labels
  caveat), the holdout scored only with `--score-holdout` (and still excluded from the fit),
  `--seed` recorded, `--no-write`, `--horizon`, an unconfigured horizon and an unimplemented
  algorithm as configuration errors, the dry run, `--json`, the feature/label tables untouched,
  a second run replacing rather than piling up, no feature rows reported as NO_DATA, and
  training never writing a dataset artefact.

Manual check (the same synthetic four-day store): `--mode train` against the shipped gates
refuses with `exit 4` and prints `4 day(s) covered by the matrix`, the gate line
(`16 more day(s) to EXPLORATORY`) and both thresholds. With `--force`, one horizon runs in
about **7 s** (including `uv run` startup) and reports: 2 folds over 4 days with 1 day held
out, each fold validating on 62 rows with `log loss 0.7195, brier score 0.1032, accuracy
0.8710, ECE 0.1628`; baselines `always_up 15.3753`, `always_range 15.3753`, `random 1.4983`,
`nifty_regime_only 0.1670`; the final model fitted on 186 rows outside the holdout; two files
written. **`nifty_regime_only` beats the model on this fixture** (by 0.5525) - the fixture
cycles its regime *with* the label it is trying to predict, so the baseline has the answer and
the model does not. That is the report doing its job: on a synthetic four-day store the honest
headline is "the best baseline wins", and it is printed as such instead of being hidden by a
flattering average. With `--score-holdout` the holdout is scored once (62 rows, log loss
0.3787) and the run records that it is no longer untouched. The two failure paths were
exercised too: `--horizon 7` and `algorithm: xgboost` both exit 2 with the reason named.

Stage 4.3 tests:

* `tests/unit/test_backtest_costs.py` (18) — the contract note charge by charge: a buy leg
  pays no STT and does pay stamp duty (hand-computed totals, to six decimals), a sell leg the
  mirror image, GST on (brokerage + exchange + SEBI) and provably *not* on the premium or on
  the statutory total, slippage as a cost with the fill price untouched, the round trip's
  total/turnover/per-unit, a captured spread replacing slippage when enabled (and its half
  charged once), a spread ignored *and counted* when the policy forbids it, a non-positive
  spread refused, seven policy refusals (negative rates, `gst > 1`), an unknown side / zero
  quantity / negative price, and `describe()` carrying every rate plus the caveat.
* `tests/unit/test_backtest_engine.py` (46) — a signal filling at the next bar's **open** (and
  at `T+1+latency` when latency is on, with `latency_ms` -> bars rounding up), a signal on the
  last bar dropped and counted, a latency wider than the session dropping everything, a
  below-threshold bar counted but not traded, `require_argmax` blocking a strong
  non-winning class; the exits: target at the target price, stop at the stop price, a **gap
  through the stop at the open**, a bar touching both resolved by `intrabar_priority` and
  counted, the time exit at the close of the bar that reached the limit (with
  `holding_minutes` matching the rule that closed it), the opposite signal at the next open,
  flat at session end, `DATA_END` when session-end flatness is off, and never two positions at
  once; the ledger: `to_record()` in exactly `TRADE_COLUMNS` order, `to_row()` adding the three
  research columns, the context of the *signal* bar (regime, candle state, the composed S/R
  state, bucket, phase, DTE, distance, weekday, expiry, strike), a **losing** trade with its
  costs (`net == gross - costs`, `mae < 0`), MFE/MAE spanning the bars held, two runs producing
  identical ledgers, exposure from the bar counts, `trading_days()` using the IST session; the
  fill modes (a captured spread charged on both legs and named in `entry_reason`; a missing
  spread counted and fallen back on); and the refusals (missing columns in each frame, no
  `option_type` anywhere, an ATR exit without `option_atr`, a non-positive price, empty
  candles, a probability row that is not a distribution, five impossible signal policies,
  eight impossible execution policies).
* `tests/unit/test_backtest_metrics.py` (15) — hand-checkable metrics (win rate, averages,
  payoff, profit factor, expectancy, cost ratio), a metric that cannot be computed reported as
  `None` (no loss -> no profit factor; no trades -> nothing at all), the drawdown walk (with
  the percentage absent when the peak never rose), the distribution's percentiles and sample
  sd, exposure needing the bar counts, the Sharpe-like ratio as expectancy over volatility,
  an open trade refused rather than marked to market; the segments: grouping with the sample
  kept, the PRELIMINARY flag on both sides of the threshold, rows ordered by net P&L,
  `UNKNOWN` for missing context rather than a dropped row, the documented DTE and
  strike-distance buckets, the weekday read from the entry timestamp, every configured segment
  building (empty when there is nothing to group), and an unknown segment refused.
* `tests/unit/test_ml_predictor.py` (11) — loading the artefact **pair** (feature order,
  vocabulary, horizon, algorithm, training rows), a booster with no metadata refused, a missing
  booster refused, a JSON file with no `model` block refused, a pair whose feature counts
  disagree refused; scoring: the keys and one column per class in the model's order,
  probabilities summing to 1, a missing feature column refused by name, a frame without the
  join keys refused, an unseen categorical level becoming *missing* (with the matrix showing
  `NaN`) instead of a new code, determinism, and the fixture model's learned signal
  (`option_type` -> class) proven before the backtest tests rely on it.
* `tests/unit/test_config_loader.py` (+14, 40 total) — the shipped `backtest:` block as the
  documented baseline, a direction class `labels.classes` does not define refused, an unknown
  option type refused, nine impossible values refused where they are loaded, `null` turning
  stop and holding limit *off* rather than defaulting them, and a typo'd key refused.
* `tests/integration/test_ai_backtest_pipeline.py` (16) — the real CLI over the seeded fixture
  and a real artefact: no model at all (exit 0, an actionable message, nothing written), a
  model without its metadata (exit 2), a horizon that contradicts the artefact (exit 2), a
  horizon nobody fitted (`NO_MODEL`), no candles in scope (`NO_DATA`); then the run itself -
  the Parquet ledger in `LEDGER_COLUMNS` order, the summary's model/policies/counters/metrics/
  segments/caveats, the stored `trades` rows agreeing with the ledger's count and net P&L
  sum, every loser kept with both legs charged (`net == gross - costs` on every row, and the
  losing side being the one the fixture makes lose), every exit reason one the engine defines
  and the counts matching the ledger, the gate **labelled** `BELOW_GATE` without refusing, the
  feature/label/candle tables untouched, `--from/--to` bounding the replay, `--dry-run` and
  `--no-write` writing nothing, `--json` parsed back, `--min-probability` recorded and
  changing the run (0.9 -> no decisions, empty ledger), and finally the real
  `dataset -> train --force -> backtest` chain finding the model `--mode train` wrote.

Manual check (the same four-day store plus a model fitted on `option_type` -> class, so the
signal fires on both sides): `--mode backtest` runs in about **1.5 s** end to end and reports
`480 bar(s) over 8 contract-session(s) across 2 contract(s); 144 flat-bar decision point(s)`,
`144 signal(s) fired, 136 entered, 8 had no fill bar`, and
`136 closed (64 winner(s), 72 loser(s), 0 scratch)` with `gross 2,443.50, costs 8,983.69, net
-6,540.19` - the friction is **367.7% of the gross edge**, win rate `47.06%`, profit factor
`0.4149`, expectancy `-48.09`, max drawdown `11,178.78`, average holding `2.47 min`, exposure
`98.33%`, exits `SESSION_END 4, STOP_LOSS 68, TARGET 64`. The segments say the same thing per
side: `CE n=64 net 4,638.59 win 100.00%` against `PE n=72 net -11,178.78 win 0.00%` - i.e. on
this deliberately naive threshold rule the costs eat the edge and one side is systematically
wrong, and the report states exactly that instead of an average. Two files are written and 136
rows reach the `trades` table; the verdict is `BELOW_GATE` for 4 captured days.
`--min-probability 0.9` produces an empty ledger with `decisions` at zero and no metric
pretending otherwise.

Stage 4.3 operator tests (71):

* `tests/unit/test_daily_pipeline.py` (38) — the step table is the documented order, no step
  depends on a later one, only the four build-from-ticks steps *need* ticks, and the engine/AI
  split of "one day" versus "all history" is asserted rather than assumed; `--skip` parsing
  (blanks, an unknown name refused); the **argv** of every step: the module, `--mode`, the day
  for the engine steps only, `--timeframes` for exactly `features`/`labels`/`validate`,
  `--timeframe` + repeated `--horizon` for the dataset, `--force` for `train` alone, one
  `--horizon` for the backtest, the resolved path flags everywhere, `--dry-run
  --no-run-artifacts` in a dry run; the flag validation against the shipped configuration
  (unknown timeframe, unknown horizon, a multi-horizon backtest refused with a way out,
  duplicate horizons collapsed); and the chain's *behaviour*: a failure stops it, the gate
  refusal skips the backtest instead of replaying an old model and exits 4,
  `--continue-on-error` presses on, an empty store is exit 2, a day without ticks is refused
  only when a step needs ticks, the pre-flight warnings (missing `index_history`, ticks on a
  calendar holiday), the dated log file's contents, the step named on stderr, and the JSON
  report's shape.
* `tests/unit/test_check_calendar.py` (15) — a provisional holiday inside the risk window
  exits 1 and is named; a verified one does not; a provisional date outside the window is a
  note, not a failure; `--within 0` only cares about today; the listing separates trading days,
  weekends and holidays (with the session window and the weekly/monthly expiry tags); a broken
  configuration is exit 2; `--require-trading-day` is the scheduler guard (exit 1 on a weekend
  or holiday, `verdict: CLOSED`); and a whole-year list whose provisional dates are all in the
  past reports no "next" one.
* `tests/unit/test_capture_run.py` (8) — the pre-session advisory: nothing to say when every
  holiday is confirmed, the date and its consequence (`NO_SESSION`) spelled out inside the
  window, "(today)" for today, and an out-of-window date left out; then the shared
  `calendar_risk` block in `status_report` (window, count, days away) proving every app's
  `--mode status` carries the same warning, JSON-friendly and additive.
* `tests/unit/test_market_calendar.py` (+6, 35 total) — `holiday_entry` and
  `holidays_between` (inclusive, ordered, filtered by `verified`, a backwards window refused),
  a provisional date closing the market in code, and `unverified_holiday_risk` using the
  supplied day and the shared default window (inclusive at the boundary).
* `tests/integration/test_daily_pipeline_e2e.py` (4) — the real chain, real child processes.
  On the *shipped* configuration: six steps succeed, `train` exits 4, the backtest is
  `SKIPPED`, and the assertions are made against the artefacts on disk (candles, the coverage
  ledger, non-NULL IVs from `enrich-greeks`, features, labels, the Parquet mirror, the
  validation report, no model, and a dated log naming the gate). Then a day with no ticks is
  refused before any step runs (exit 2, nothing on stdout), and finally the AI half on a
  walk-forward that fits the fixture: `dataset -> train --force -> backtest` exits 0 and
  leaves a model artefact, its metadata, the split definition, the Parquet ledger and the
  summary whose `model.horizon_minutes` and cost caveat are checked.

---

## 4. How to use it

```bash
# what a dataset would contain, at which horizons, split how (writes nothing)
uv run python -m apps.anjaneeya_ai.run --mode dataset --dry-run

# build the matrix for every configured horizon (5/10/15/30 in config/model.yaml)
uv run python -m apps.anjaneeya_ai.run --mode dataset

# a window / one horizon / one contract / machine-readable
uv run python -m apps.anjaneeya_ai.run --mode dataset --from 2026-09-01 --to 2026-09-30
uv run python -m apps.anjaneeya_ai.run --mode dataset --horizon 15
uv run python -m apps.anjaneeya_ai.run --mode dataset --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode dataset --json

# the documented order of operations: capture, then engine, then AI
uv run python -m apps.anjaneeya_engine.run --mode build-candles
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks
uv run python -m apps.anjaneeya_engine.run --mode build-features
uv run python -m apps.anjaneeya_engine.run --mode build-labels
uv run python -m apps.anjaneeya_ai.run     --mode dataset
```

What the run prints, per horizon: how many labelled rows came out of how many feature rows
(and why the rest did not), the class mix, the feature split (numeric/categorical, constant
here, missing from the frame), the days and contracts, the split verdict (the folds, or the
number of captured days still missing), and the artefacts written.

Reading a matrix back, with the split manifest beside it:

```python
import json, pandas as pd

frame = pd.read_parquet("data/datasets/1m_h15_2026-09-01_2026-09-30.parquet")
manifest = json.loads(open("data/datasets/1m_h15_2026-09-01_2026-09-30_splits.json").read())

x = frame[manifest["feature_columns"]]          # never contains a label
y = frame["label_code"]                         # classes[i] <-> i
classes = manifest["dataset"]["class_counts"]
for fold in manifest["split"]["splits"]:        # exact windows, per fold
    ...
```

```sql
-- how much of a day's label table actually ended up in a matrix?
SELECT label, count(*) AS rows
FROM labels
WHERE timestamp >= TIMESTAMPTZ '2026-09-29 09:15:00+05:30'
  AND timestamp <  TIMESTAMPTZ '2026-09-29 15:30:00+05:30'
  AND horizon_minutes = 15
GROUP BY 1 ORDER BY 2 DESC;
```

**Stage 4.3.** And the replay, end to end:

```bash
# what would be replayed: which model, which bars, which exits, which costs (writes nothing)
uv run python -m apps.anjaneeya_ai.run --mode backtest --dry-run

# replay the newest artefact for the configured dataset timeframe
uv run python -m apps.anjaneeya_ai.run --mode backtest

# a specific model / a stricter signal / one contract / a window / machine-readable
uv run python -m apps.anjaneeya_ai.run --mode backtest --model data/models/1m_h5_..._lightgbm.txt
uv run python -m apps.anjaneeya_ai.run --mode backtest --min-probability 0.45
uv run python -m apps.anjaneeya_ai.run --mode backtest --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode backtest --from 2026-09-01 --to 2026-09-30 --json

# replay without leaving evidence behind
uv run python -m apps.anjaneeya_ai.run --mode backtest --no-write

# the whole chain: matrix -> model -> ledger
uv run python -m apps.anjaneeya_ai.run --mode dataset  --horizon 5
uv run python -m apps.anjaneeya_ai.run --mode train    --horizon 5
uv run python -m apps.anjaneeya_ai.run --mode backtest --horizon 5
```

### The daily command, and the guard in front of the capture

Eight commands is seven too many for something that has to happen every evening, so Stage 4.3
also ships the operator half of the pipeline:

```bash
# every evening after the close: candles -> greeks -> features -> labels -> validate ->
# dataset -> train -> backtest, fail-fast, one dated log, one verdict
./scripts/run_daily_pipeline.sh                  # or run_daily_pipeline.ps1
./scripts/run_daily_pipeline.sh --dry-run --json # the plan, machine-readable, writes nothing
./scripts/run_daily_pipeline.sh --force          # train below the gate (smoke tests only)

# every morning: is the calendar safe? (0 = clear, 1 = a provisional holiday, 2 = config)
uv run python scripts/check_calendar.py --days 30
uv run python scripts/check_calendar.py --require-trading-day   # exits 1 on a holiday
```

Exit codes are chosen so a scheduler can tell the cases apart: `0` all good, `1` a step
failed, `2` configuration (unknown `--skip` step, no ticks for the day, locked database), `4`
only `--mode train` refused below spec 4.3's gate. After a gate refusal the runner marks the
backtest `SKIPPED` instead of replaying an older model as if it were today's. `--force` is
forwarded to `--mode train` alone.

`check_calendar.py` exists because a provisional holiday is the one failure that costs data
permanently: a date flagged `verified: false` that is really a trading day makes a capture
session exit `NO_SESSION`, and Zerodha serves no historical expired-option data, so the
session is gone. The script names every provisional date inside a risk window (default 10
days), prints the session window and the weekly/monthly expiries for each day, and exits `1`
when the risk is real. The capture app prints the same warning for anything within 7 days
before it starts, and `docs/runbook.md` carries the morning routine, the failure playbook and
the scheduling snippets.

The ledger is a Parquet file and the summary a JSON one, both written atomically, and the same
28 spec-23 columns also land in the stored `trades` table (validated through `TradeRecord` row
by row). Reading a ledger back - the segmented tables and the counters come with it:

```python
import json

import pandas as pd

ledger = pd.read_parquet("data/backtests/1m_h5_2026-09-01_2026-09-30_ledger.parquet")
print(ledger[["entry_timestamp", "option_type", "entry_price", "exit_price", "net_pnl", "costs"]])
print(ledger.groupby("option_type")["net_pnl"].sum())

summary = json.loads(open("data/backtests/1m_h5_2026-09-01_2026-09-30_summary.json").read())
print(summary["verdict"]["results_label"], summary["metrics"]["profit_factor"])
print(summary["counts"]["exit_reasons"], summary["costs"]["caveat"])
```

**Stage 4.2.** And the model, end to end:

```bash
# what would be trained, at which horizons, split how, gated how (writes nothing)
uv run python -m apps.anjaneeya_ai.run --mode train --dry-run

# train every configured horizon (one LightGBM model per horizon)
uv run python -m apps.anjaneeya_ai.run --mode train

# one horizon / one contract / a window / a specific seed / machine-readable
uv run python -m apps.anjaneeya_ai.run --mode train --horizon 15
uv run python -m apps.anjaneeya_ai.run --mode train --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode train --from 2026-09-01 --to 2026-09-30
uv run python -m apps.anjaneeya_ai.run --mode train --seed 7 --json

# score the final model on the untouched holdout (once), or train below the gate
uv run python -m apps.anjaneeya_ai.run --mode train --score-holdout
uv run python -m apps.anjaneeya_ai.run --mode train --force

# the order of operations: matrix (for the frozen artefact), then the model
uv run python -m apps.anjaneeya_ai.run --mode dataset --horizon 15
uv run python -m apps.anjaneeya_ai.run --mode train   --horizon 15
```

What the run prints, per horizon: how many rows the matrix has and over how many days, the
split, every fold's validation metrics *and* every baseline's on the same rows, the across-fold
aggregate, the comparison against the best baseline, the holdout if it was scored, and the
artefacts written. Reading a model back:

```python
import json

metadata = json.loads(open("data/models/1m_h15_..._lightgbm_metadata.json").read())
print(metadata["model"]["policy"], metadata["aggregates"]["log_loss"])
print(metadata["folds"][0]["comparison"]["log_loss_delta_vs_baseline"])

# LightGBM's own text: ANJANEYA is not needed to load it
import lightgbm as lgb

booster = lgb.Booster(model_file="data/models/1m_h15_..._lightgbm.txt")
probabilities = booster.predict(frame[metadata["model"]["feature_columns"]])
```

**Still to do in Phase 4:** `--mode report` (specs 24-25: the time-of-day × option-type and
time-of-day × cycle-phase performance matrices, the intraday path curves, holding-time and ATR
profiles, and the significance testing with a multiple-testing correction) - which is what the
segmented tables `--mode backtest` now produces already cover the *statistics* of, and what
the ledger was built to feed. **No schema change**: `schema_version` stays **1** - a dataset, a
model, a ledger and a summary are files, and the ledger's research columns live in the Parquet
copy rather than in the `trades` table.


---

## 5. Assumptions and open questions

* **The capture universe is a one-way door, and it is still at its Phase 1 default.**
  `config/option_universe.yaml` ships `strikes_offset: [-100, 0, 100]` and
  `expiries: [current_week, next_week]`, and those two lists bound the ticks *and* the
  option-chain snapshots (the poller polls the subscribed symbols, not the full chain).
  Zerodha serves no historical expired-option data, so a strike or an expiry that is not
  subscribed at capture time cannot be recovered for any past day. Widening now (`±200`, i.e.
  `[-200, -100, 0, 100, 200]`, plus `current_month`) costs rate-limit headroom and disk, and
  it is free to filter down at analysis time; not widening cannot be undone later. **This is
  an open research decision, deliberately not taken by the tooling**, and it is documented as
  a required pre-capture step in `docs/runbook.md` §1.4.
* **The opening gate and the first fold are 20 and 135 captured days respectively, and the
  second number is the one that surprises people.** `--mode train` refuses below
  `exploratory_min_days` (exit 4, which the pipeline reports as `GATE` and skips the
  backtest); but even with `--force` the shipped `walk_forward` windows
  (`train_window_days: 90`, `validate_window_days: 15`, `holdout_days: 30`) need 135 captured
  days before a single fold exists, and the run then reports `NOT_TRAINED` per horizon and
  writes nothing - while exiting `0`, because "not enough data for an honest split" is a
  finding, not a crash. The runbook states both numbers and the two legitimate responses
  (wait, or shrink the windows and record it as a decision).
* **The stored labels are assumed to be from the dataset's timeframe** (see §2's last
  paragraph). Everything else about a label row is checked; this one cannot be. The
  candidate fix is a `timeframe` column on `labels` (schema version 2) or a `label_version`
  convention that encodes it - both are schema/policy decisions, not bugs to paper over, and
  the run says which assumption it made.
* **One dataset per horizon, one target per dataset.** Spec 21 says the model outputs class
  probabilities; a multi-horizon model would need either a horizon dimension or a shared
  target design. Building one matrix per horizon keeps each artefact unambiguous and costs
  only disk.
* **One model per horizon, and each is scored on a *validation* window.** A fold metric says
  how well the model ranked the future it was shown. It does not include spreads, costs, fills
  or position sizing - spec 4.3 reserves "statistically meaningful" for a completed backtest
  (120 captured days), and every training run prints and stores that caveat. `--mode backtest`
  is where that claim can be earned.
* **The decision of *when* to trade does not exist yet.** The model predicts a distribution;
  nothing here turns a distribution into a position. That is deliberate (spec 21's targets are
  behaviour, not signals), and it means a training run can never be read as a trading result.
* **`nifty_regime_only` is only as good as the regime column.** It learns per-regime class
  frequencies, so it absorbs any predictive power the regime itself has. That is exactly why it
  is the sharpest of the four baselines: a model that cannot beat it has not added anything
  beyond the regime the feature row already knew.
* **Early stopping is off by default, and on purpose.** A researcher can turn it on
  (`training.early_stopping_rounds`), and the run then records that the number of trees was
  chosen on each fold's validation window - which makes those metrics optimistic by an unknown
  amount. Off keeps the fold metrics comparable between models and horizons.
* **A single-class training fold is skipped, not repaired.** Rebalancing a fold that happens to
  contain one class would be inventing data; the run reports the skip and the fold count so the
  reader knows how many folds the aggregate actually rests on.
* **Folds are whole days.** A fold boundary can never fall inside a session, so the embargo
  is trivially satisfied by the shipped policy (a 30-minute embargo across a session
  boundary is always clean). The guard is implemented and *counted*, but it only bites for a
  dataset whose rows run through a calendar day - which is why the splitter's unit tests use
  exactly such a fixture.
* **The gates are reported, not enforced, by `--mode dataset`.** Building a four-day matrix
  is legitimate exploratory work; refusing to *train* a model below
  `ml_training_min_days` is spec 4.3's rule and belongs to `--mode train`, which reads the
  same `GateStatus` - and `--mode backtest` reads it too, to *label* its result rather than
  to refuse it.
* **The decision of *when* to trade is still a threshold, and that is deliberately visible.**
  `backtest.signal` is a stand-in for spec 24: it enters when the directional class clears
  `min_probability`. Any number a backtest prints is therefore conditional on an arbitrary
  cut, and the report, the summary and a caveat all say so. The stage worth measuring with it
  is *costs* (does any rule survive friction at this bar size?), not the existence of an edge.
* **Costs are configuration, not truth.** Every statutory rate and the brokerage plan live in
  `config/model.yaml`'s `costs:` block, and they change. `COST_MODEL_CAVEAT` is printed on
  every run and stored in every summary so a quoted P&L cannot travel without it.
* **The captured bid/ask spread is an optional column the store does not yet write.** The
  engine reads `bid_ask_spread` from the context row when `execution.fill_at: bid_ask` and
  `costs.use_captured_spread` are both set, and *counts* the bars where it is missing (falling
  back to the bar open). The shipped feature contract has no such column, so the shipped
  configuration leaves `use_captured_spread: false` and charges slippage - the honest default
  until a capture stage writes spreads onto the feature rows.
* **One position per contract, sessions never held overnight, and a re-entry needs a later
  bar.** These are policy choices, not discoveries: an intraday research backtest on weekly
  options has no business holding a position across a session, and modelling one position per
  contract keeps the ledger readable. A portfolio-level backtest (many contracts at once, an
  account, leverage) is a different stage, and the ledger's shape would not carry it.
* **The fixture's losers are the fixture's business.** The four-day integration store is
  deliberately shaped so that a CE drifts up and a PE drifts down, which is what makes "a
  winner and a loser are both in the ledger" a property of the test rather than of the day.
  Nothing in those numbers is evidence about NIFTY; the *counters* are the evidence that the
  engine did what it says.


---

## 6. Stage 4.4 — the research report (spec 24-25)

**Status: complete.** `--mode report` answers all ten of spec 24's questions from the trade
ledger `--mode backtest` wrote - *losers included* - plus the stored NIFTY index bars for the
two questions that are about the market rather than the strategy. It writes one JSON document,
one Markdown report and one flat cells CSV under `data/reports/research/`, named after the
ledger it came from, and it **never writes to DuckDB** (read-only connection): a report cannot
change the evidence it describes. No schema change - `schema_version` stays **1** - and no new
configuration: the sample rule, the correction and the alpha are `model.evaluation`'s, the
seasonality horizon is `model.data_gates`'s and the ATR definition is `indicators.atr`'s.

### 6.1 The ten deliverables, and where each one comes from

| # | Spec 24 asks for | Carried by | Source |
| --- | --- | --- | --- |
| 1 | performance by time-of-day bucket x option side | `time_of_day_x_option_type` matrix | the ledger's closed trades |
| 2 | performance by time-of-day bucket x expiry-cycle phase | `time_of_day_x_cycle_phase` matrix | the ledger |
| 3 | performance by day of week | `day_of_week` segment table | the ledger |
| 4 | performance by days to expiry | `days_to_expiry` segment table | the ledger |
| 5 | regime x time-of-day, regime x cycle phase | `regime_x_time_of_day`, `regime_x_cycle_phase` | the ledger |
| 6 | holding-time distribution by bucket | `holding_time` (p5/p25/p50/p75/p95) | the ledger |
| 7 | volatility (ATR) profile by bucket and cycle phase | `volatility` table | **stored index bars** + `core.indicators.atr` |
| 8 | average cumulative intraday path per bucket | `intraday_paths` | **stored index bars** |
| 9 | significance, corrected for multiple testing | `significance` per matrix | z-test + Welch, BH/Bonferroni |
| 10 | PRELIMINARY labels below the configured sample | `preliminary_cells` | `evaluation.preliminary_min_samples` |

### 6.2 What was built

| Module | Responsibility |
| --- | --- |
| `core/research/segments.py` (new) | `PerformanceMatrix`/`MatrixCell` (spec 24.1, 24.2, 24.5), `build_matrix`/`build_matrices`, the single-axis cuts (24.3, 24.4) via the existing segment tables, `HoldingTimeProfile` (24.6), and `ledger_trades` - the Parquet -> `Trade` round trip that turns `NaN`/`NaT` back into `None` before a metric can see them. |
| `core/research/significance.py` (new) | `win_rate_p_value` (two-proportion z-test), `mean_pnl_p_value` (Welch), `adjust_p_values` (`none`/`bonferroni`/`benjamini_hochberg`), `significance_of_matrix` (each cell against the rest of the ledger, corrected as one family per metric), and the four verdicts `SIGNIFICANT` / `NOT_SIGNIFICANT` / `NOT_TESTABLE` / `INSUFFICIENT_SAMPLE`. |
| `core/research/profile.py` (new) | `build_bucket_profiles` (24.7) and `build_intraday_paths` (24.8) over the stored index bars, bucketed by **bar close** with the market calendar (the same convention `build-features` uses), one symbol per call, `OUT_OF_SESSION` bars counted rather than folded in. |
| `core/research/report.py` (new) | `ReportPolicy.from_config`, `ResearchReport` (the ten items as one object, `to_dict()`/`to_markdown()`), `build_research_report` and the caveats. `RESEARCH_ITEMS` is the spine: item name -> the attribute that answers it. |
| `apps/anjaneeya_ai/report_writer.py` (new) | Atomic writes of `<stem>_report.json`, `<stem>_report.md` and `<stem>_cells.csv` (temp file, then rename), with `cells_frame` joining every cell to its verdict. Failures are counted, never raised. |
| `apps/anjaneeya_ai/run.py` (extended) | `--mode report` plus `--ledger PATH`, `--no-write`, `--dry-run`, `--json`; the ledger is resolved from `data/backtests/` by modification time when `--ledger` is not given, the index bars are read read-only for the ledger's own window, and the artefacts are registered on the run manifest. |



### 6.3 Design decisions worth recording

* **A p-value is never printed for a cell that cannot carry one.** A cell - or the baseline it
  is compared against - below `evaluation.preliminary_min_samples` is `INSUFFICIENT_SAMPLE`
  with `null` p-values; a cell whose trades all won (no variance) or whose P&L never varied is
  `NOT_TESTABLE` with the reason recorded. The alternative (a p-value from six trades) is how a
  small sample becomes a finding.
* **The family is the tests that ran.** Each matrix is one family per metric - a win-rate
  comparison and a P&L comparison answer different questions, so pooling them would silently
  double the family size for both - and `None` (untested) cells are excluded from the
  denominator, so a matrix full of tiny cells cannot loosen the correction for the cells that
  *were* testable. The family size is reported next to every adjusted p-value.
* **A missing intersection is absent, not zero.** `PerformanceMatrix.cell()` returns `None` and
  `grid()` yields `None` where nothing traded; the Markdown prints a dash. A heatmap drawn from
  these tables can therefore never show a profit where there was no trade.
* **The market's profile is computed from the market's bars, over the ledger's own window.**
  Items 7 and 8 read the stored 1m index bars between the ledger's first entry and its last exit;
  a wider window would describe days the ledger never traded. Both bucket a bar by its **close**
  (`timestamp + timeframe`), the same convention `build-features` uses for
  `time_of_day_bucket`, so the profile and the stored feature rows agree about the boundaries.
* **"Not measured" and "measured as zero" are different words.** With no index bars, items 7
  and 8 are present in the document with `volatility_unavailable` / `intraday_paths_unavailable`
  and a caveat saying why; an empty intraday path is an absent key, not a flat curve.
* **The report is read-only, and it says what it is.** The connection is opened read-only, the
  labels table is never opened, and the first caveat in every report is the sentence that the
  ledger's entries come from a probability threshold rather than from spec 24's decision layer.
* **The three artefacts are one object rendered three ways.** `cells_frame` builds the CSV from
  the same `ResearchReport` the JSON and Markdown come from, so the three files cannot disagree;
  each write is independent, so a CSV failure cannot lose the JSON.


### 6.4 Verification performed

```bash
# the analysis core, on its own
uv run pytest tests/unit/test_research_segments.py tests/unit/test_research_significance.py \
             tests/unit/test_research_profile.py tests/unit/test_research_report.py -q
# 47 passed

# the mode end to end: a real Parquet ledger + a seeded database + the real main()
uv run pytest tests/integration/test_ai_report_pipeline.py -q
# 14 passed

# the whole suite, and the linter (re-run on 2026-09-30/10-01, after the Tuesday-expiry sweep)
uv run pytest tests/unit -q          # 1409 passed in 121.33 s (0:02:01)
uv run pytest tests/integration -q   # 240 passed in 882.02 s (0:14:42)
uv run ruff check .                  # clean
```

What the integration tests actually assert (each one runs `main()`, so it is about what
happened, not about what was intended):

* a stored ledger of 48 trades over 4 sessions produces 3 artefacts whose stem mirrors the
  ledger, and the JSON's `trades`/`days` match the ledger's rows;
* the ten items are all present, with `OPEN_915_930 | CE` carrying `sample = 8`,
  `total_net_pnl = 840.0` and `preliminary = false`, and the `EXPIRY_DAY` cells
  (one session, 4 trades) carrying `PRELIMINARY`;
* the significance of the CE/PE split is tested per matrix (`family_size = 6` for the first
  matrix) and every CE/PE cell clears the corrected level, while the win rate's `p` and the
  adjusted `p` both round-trip through the JSON;
* items 7 and 8 come out of the *stored index bars*: 4 sessions, 240 bars, 4 bucket x phase
  cells, and the paths start at `09:16` (the close of the 09:15 bar) as documented;
* the Markdown has a section per item plus `## Caveats`, and the CSV has one row per matrix
  cell with the verdict columns joined on;
* `--dry-run`, `--no-write` and `--json` each do what the README says, and **the stored tables
  are byte-for-byte unchanged** after a report (`candles`, `features`, `labels`,
  `index_history`, `trades` counts compared before and after);
* with no ledger at all the run exits `0` with `no trade ledger to analyse`, and with a Parquet
  that is not a ledger it exits `1` saying so.

A quiet consequence of the `PRELIMINARY` rule worth recording: the fixture's `EXPIRY_DAY`
cells are *reported* with their four trades and their flag rather than being dropped, which is
the behaviour spec 24.10 asks for - the reader sees the number and the reason to distrust it in
the same row.

### 6.5 Not verified / limitations (read these before quoting a number)

* **The report has never been produced from real captured trades**, because no real session has
  been captured yet (see §5). Every number in this stage's tests comes from a synthetic ledger
  whose winners and losers were *constructed*; the evidence that the stage works is the
  structure (which cells, which samples, which verdicts), not the P&L.
* **The significance machinery is a large-sample approximation.** The win-rate test is a
  two-proportion z-test and the P&L test is Welch's t-test, both normal-approximated; there is
  no exact (Fisher / permutation) alternative yet, and no bootstrap confidence intervals. With
  samples this small that distinction matters - the sample rule refuses the smallest cells, but
  it does not make the remaining ones exact.
* **`cells_frame` is one row per cell of the four matrices.** The single-axis cuts (weekday,
  days-to-expiry), the holding-time profile, the volatility profile and the intraday paths are
  in the JSON and the Markdown but **not** in the CSV yet.
* **Items 7 and 8 describe the index, not the option.** The ATR profile and the intraday path
  are NIFTY 50's; an option-side volatility profile would need the per-contract ATR the feature
  rows already carry, and a per-side intraday path would need the premium series. Neither is
  claimed here.
* **The intraday path has no significance test.** A path point is a mean return with its sample
  and its own `PRELIMINARY` flag, and nothing more; comparing two buckets' curves statistically
  is a later stage.
* **`profile_min_sessions` is a code constant, not configuration.** It is 2 sessions (one
  session is a shape, not a distribution) and it is deliberately conservative; promoting it to
  `config/model.yaml` is a one-line change if an operator wants a different rule.
* **A report never re-derives a trade.** If the ledger was produced by a buggy build, the report
  will faithfully describe the bug - by design: the ledger is the evidence, and re-running the
  backtest inside a report would make the two disagree.

### 6.6 How to use it

```bash
# the newest ledger under data/backtests/, the ten items, three artefacts
uv run python -m apps.anjaneeya_ai.run --mode report

# a specific ledger (e.g. one horizon), machine-readable, or without writing
uv run python -m apps.anjaneeya_ai.run --mode report --ledger data/backtests/1m_h5_2026-09-28_2026-10-01_ledger.parquet
uv run python -m apps.anjaneeya_ai.run --mode report --json
uv run python -m apps.anjaneeya_ai.run --mode report --dry-run
```

Exit codes: `0` ok (including "there is no ledger yet"), `1` the ledger is unreadable or is not
a trade ledger, `2` configuration.

* **`ledger_trades` validates the artefact it is handed.** A Parquet missing a trade-contract
  column, or carrying one the contract does not know, is refused (`exit 1`) instead of being
  read as a ledger of zeros - the failure mode a differently-named dataset dump would otherwise
  produce.



### 6.7 Example output (real, from the integration fixture)

The console summary (abridged - four matrices, the market items, the artefacts and the caveats):

```
ANJANEYA-AI research report (spec 24) - 48 closed trade(s) over 4 entry day(s)
  ledger        : 1m_h5_2026-09-28_2026-10-01_ledger.parquet
  window        : 2026-09-28 09:20:00+05:30 .. 2026-10-01 15:06:00+05:30
  samples       : a cell is PRELIMINARY below 5 trades
  significance  : benjamini_hochberg at alpha=0.05
  data gate     : INSUFFICIENT - too little data to conclude anything

  time_of_day_x_option_type:
    OPEN_915_930 | CE: 840.00 over 8 trade(s), win rate 1.000
    OPEN_915_930 | PE: -840.00 over 8 trade(s), win rate 0.000
    ...
    tested: 6 cell(s); clearing the corrected level: 6
  time_of_day_x_cycle_phase:
    OPEN_915_930 | EXPIRY_DAY: 0.00 over 4 trade(s), win rate 0.500 [PRELIMINARY]
    ...
    tested: 3 cell(s); clearing the corrected level: 0
  volatility    : NIFTY 50, 4 bucket x phase cell(s) over 4 session(s)
  intraday path : 2 bucket path(s)
  preliminary   : 5 label(s) across 2 table(s)
  wrote         : .../data/reports/research/1m_h5_2026-09-28_2026-10-01_report.json
  wrote         : .../data/reports/research/1m_h5_2026-09-28_2026-10-01_report.md
  wrote         : .../data/reports/research/1m_h5_2026-09-28_2026-10-01_cells.csv

  caveat: the ledger's entries come from a probability threshold, not from spec 24's decision
          layer (which does not exist yet): these are costs and shapes, not a claimed edge
  caveat: 19 cell(s) were tested, each matrix corrected as one benjamini_hochberg family at
          alpha=0.05; a cell that clears the level is still one cell of one ledger
  caveat: 4 captured day(s) against 250 for a seasonality claim: the weekday and cycle-phase
          tables are PRELIMINARY by the project's own accumulation rule, ...
```

and the same report as the Markdown reader sees it (items 1 and 2, then the PRELIMINARY list):

```markdown
## 1_time_of_day_x_option_type. spec 24.1 - performance by time-of-day bucket x option side

| time_of_day_bucket | CE | PE |
| --- | --- | --- |
| OPEN_915_930 | 840.00 (8) | -840.00 (8) |
| MORNING_930_1100 | 840.00 (8) | -840.00 (8) |
| CLOSE_1430_1515 | 840.00 (8) | -840.00 (8) |

## 2_time_of_day_x_cycle_phase. spec 24.2 - performance by time-of-day bucket x expiry-cycle phase

| time_of_day_bucket | EXPIRY_DAY | MID_CYCLE |
| --- | --- | --- |
| OPEN_915_930 | 0.00 (4) **PRELIMINARY** | 0.00 (12) |
| MORNING_930_1100 | 0.00 (4) **PRELIMINARY** | 0.00 (12) |
| CLOSE_1430_1515 | 0.00 (4) **PRELIMINARY** | 0.00 (12) |

## 10_preliminary_labels. spec 24.10 - PRELIMINARY labels wherever the sample is too small

* **time_of_day_x_cycle_phase** (3): OPEN_915_930 | EXPIRY_DAY, MORNING_930_1100 | EXPIRY_DAY,
  CLOSE_1430_1515 | EXPIRY_DAY
* **volatility_profile** (2): OPEN_915_930 | EXPIRY_DAY, MORNING_930_1100 | EXPIRY_DAY
```

Note what the numbers *are* here: the fixture's winners and losers were constructed, so the
`840.00 (8)` cells are evidence that the grouping, the sample counting and the PRELIMINARY rule
work - not evidence about NIFTY. The `EXPIRY_DAY` row is the more interesting one: one session
of four, four trades per bucket, reported with its flag rather than dropped.

