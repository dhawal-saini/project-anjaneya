# Phase 5 report — explainability, visualization and feature importance

**Status: Stages 5.1 (spec 26 — explainability), 5.2 (spec 27 — feature importance) and 5.3
(spec 28 — visualization) complete.**
`--mode explain` scores the stored
feature rows with one trained model artefact - exactly as `--mode backtest` does - and writes
down what the model was looking at for the newest `--rows` predictions: the configured blocks
of the stored feature row (NIFTY, OPTION, PRICE ACTION, CANDLE, SUPPORT / RESISTANCE, CONTEXT)
with the model's own class probabilities appended as the AI block, one JSON, one Markdown and
one flat CSV of *lines* under `data/reports/explanations/`, named after the model.
`--mode importance` measures - on the model's own training matrix - which features and which
configured *sources* actually carried the information: the model's native gain off the booster,
a permutation delta against the untouched-matrix baseline with a confidence interval per feature
and per source, and an optional SHAP attribution, one JSON, one Markdown and one flat CSV of
*features* under `data/reports/importance/`, also named after the model.
`--mode charts` draws spec 28's figures from what already exists - the newest ledger, the newest
importance artefact and the stored candles / features / index bars - and writes one image per chart
plus one index Markdown under `data/reports/charts/`: the NIFTY and option price charts (with their
MA lines, level/running overlays, breakout/rejection markers, structure labels, trend-score axis,
regime shading and the model's own entries), the time-of-day × cycle-phase and × day-of-week P&L
heatmaps, the CE-vs-PE curves, the regime-conditional bars and the feature-importance bars.
`schema_version` stays **1**: all three modes read the database read-only and write no feature row,
no label row and no trade.

Spec 28 is two sentences of instruction - *"Charts must be produced for validation"*, *"Visualization
is not optional"* - and a list of pictures. Three rules made the pictures evidence rather than
decoration: **the numbers are not recomputed** (the heatmaps and curves come from the ledger
`--mode report` reads and the bars from the artefact `--mode importance` wrote, so a chart cannot
disagree with the number beside it), **a value that is not there is left blank** (an empty heatmap
cell prints only its count and no colour; a feature the sample could not measure has no bar), and
**the sample travels with the picture** (every cell prints its trade count and a cell below
`evaluation.preliminary_min_samples` is hatched `PRELIMINARY`; the importance bars carry their
interval).

Spec 26 is three sentences long and unusually strict for its length: *every* prediction must
be explainable, the form must be plain and structured, and the values it lists are "examples
only. Do not hard-code." Four rules therefore became mechanical rather than aspirational:

* **the structure is configuration.** `model.explainability.sections` names stored feature
  *groups* (`core/features/feature_builder.FEATURE_GROUPS`) or individual feature *columns*.
  The loader checks every name against the feature contract, refuses a column listed in two
  blocks, refuses a block that names nothing, and refuses the reserved `identity`/`provenance`
  groups (both are already in the artefact: identity in the header, provenance in the audit
  block). `core/ml/explainer.py` re-applies the same invariants to a policy built by hand, so
  a test or a notebook cannot bypass them either.
* **the model's block is the model's.** The classes, their order and the model's own pick come
  from the loaded artefact's metadata - never a class list written in the code - and the JSON
  stores the *whole* distribution while the console shows the top
  `model.explainability.probabilities` of them.
* **unknown is not zero.** A `NULL` feature is rendered with
  `model.explainability.missing_text` (`unknown`), carries `known: false` and a `null` value,
  and a real `0` stays a measurement. A block the run could not fill (no model was scored) is
  present with its reason, so "this block is empty" and "we never looked" stay different
  statements.
* **a named column the row does not carry is a refusal**, not a blank - the same rule
  `ModelPredictor` applies to the model's own feature list, and for the same reason: a silently
  missing input turns an explanation into fiction.

An explanation is also not a result. Every artefact carries four caveats - it describes inputs
rather than causes; spec 24's decision layer still does not exist, so nothing here turns a
probability into a position; the input is the stored row, whose `fetch_timestamp`,
`source` and `quality_flag` sit in the audit block; and the model being explained is labelled
`EXPLORATORY`/`BELOW_GATE` by its own training run, a standing an explanation inherits rather
than improves. No data-accumulation gate applies (an explanation describes an input), and the
mode says so by *labelling* instead.

Spec 27 is a list of three measurements (native importance, permutation importance, "SHAP if
available") followed by a question and a rule: *which* inputs carry the information, and "do not
assume. Measure. Report with confidence intervals where practical." Four rules became mechanical
here too:

* **the measurement is configuration.** `model.importance` sets the repeats (at least 2), the
  scoring rule (`log_loss`, the trainer's own metric, or `accuracy`), the sample (`holdout` -
  spec 22's reserved days, the only rows the final model provably did not train on - or the
  whole `matrix`, which then carries an in-sample caveat), the row cap, the confidence level, the
  seed and the SHAP contract. Nothing about how a number is produced lives in Python.
* **the sources are configuration, and the map is complete.** `model.importance.topics` maps
  stored feature *groups* and columns onto spec 27's list (NIFTY regime, option MAs, candle
  behaviour, S/R, structure, IV, greeks, OI, option residual, time of day, cycle phase, time to
  expiry) plus four sources spec 27 does not name but whose columns exist (price action/levels,
  day of week, option side, moneyness/volume). Every name is checked against the feature contract
  at load, a column claimed by two sources is refused, a source that claims nothing is refused,
  and a feature no source claims is reported under `unmapped` rather than attributed to a source
  that never measured it.
* **native importance comes off the loaded model.** The gain numbers are read out of the booster
  the same run scores with (`ModelPredictor.booster.feature_importance`), never from a class list
  or a copy in a JSON beside it.
* **a feature the sample cannot measure is not measured, never 0.** An all-`NULL` column and a
  constant column both come back with `measured: false` and a reason ("every value of this
  column is NULL in the sample..." / "this column holds a single value..."), because a delta of
  `0` reads as "no information" when the truth is "these rows cannot say".

An importance ranking is also about one model, not about the market. Every artefact carries three
structural caveats - permutation splits credit between correlated features, an unmeasured feature
is named rather than scored, and which rows were measured (holdout or the whole matrix) - and one
that depends on the model: the artefact's own `results_label` (`EXPLORATORY`, `BELOW_GATE`) is
quoted, so a ranking of an unvalidated model says so. No data-accumulation gate applies, for the
same reason as `--mode explain`: a ranking describes a model, so it labels rather than refuses.

---

## 1. What was built

| Module | Responsibility |
| --- | --- |
| `core/ml/explainer.py` | Spec 26's structure and rendering: `ExplanationPolicy` (blocks, digits, how many probabilities to print, the word for a missing value), `explain_row`/`explain_frame`/`build_explanations`, the `Explanation`/`ExplanationSection`/`ExplanationEntry`/`ClassProbability` contract, `format_value` (the presentation rules), and the caveat constants. Pure: mappings and frames in, objects out, no I/O. |
| `apps/anjaneeya_ai/explain_writer.py` | The artefacts: `<stem>_explanations.json`, `<stem>_explanations.md` and `<stem>_entries.csv` (one row per explanation *line*), written atomically from one in-memory batch, with `model_stem` so an explanation always names the model it came from. |
| `apps/anjaneeya_ai/run.py` | `--mode explain` (spec 26): resolve the artefact (`--model`/newest/`--horizon`), resolve the timeframe, read the stored feature rows in the `--from/--to/--symbol` scope, score them, build the batch, write/prune/print, and register the artefacts on the run manifest. |
| `core/config_loader.py` | `ExplainabilityConfig` / `ExplainabilitySectionConfig`: the `explainability:` block, validated against the feature contract at load. |
| `config/model.yaml` | The shipped blocks, in spec 26's order, with the six presentation knobs. |
| `core/app_bootstrap.py` | `explain` added to `ANJANEYA-AI`'s `--mode` choices. |
| `core/ml/importance.py` | Spec 27's measurement: `ImportancePolicy` / `ImportanceTopic` (sources, repeats, metric, sample, cap, confidence, seed, SHAP contract), `native_importance` (gain off the booster), `select_sample` (holdout-or-matrix with a reason), `permutation_importance` (per-feature deltas + intervals, `_permuted_column` keeping a categorical's vocabulary), `shap_importance` (four statuses), `build_importance_report` (features ranked, sources aggregated as the per-repeat sum of their features, `unmapped` kept separate, the caveats attached), and `metric_score`. Pure: frames and a predictor in, objects out, no I/O. |
| `apps/anjaneeya_ai/importance_writer.py` | The artefacts: `<stem>_importance.json`, `<stem>_importance.md` and `<stem>_features.csv` (one row per feature), written atomically from one in-memory report, reusing `model_stem` so a ranking always names the model it measured. |
| `apps/anjaneeya_ai/run.py` | `--mode importance` (spec 27): resolve the artefact (`--model`/newest/`--horizon`), resolve the timeframe, build the model's own training matrix from the stored features + labels, select the sample, measure native/permutation/SHAP, aggregate the sources, write/prune/print, and register the artefacts on the run manifest. |
| `core/config_loader.py` | `ImportanceConfig` / `ImportanceTopicConfig`: the `importance:` block, validated against the feature contract at load. |
| `config/model.yaml` | The shipped `importance:` block: the measurement knobs and spec 27's sources (+4 unnamed ones) in spec order. |
| `core/visualization/charts.py` | Spec 28's figure builders: `ChartPolicy` (spec, size, resolution, format, colormaps, caps), the `PriceChartData`/`Heatmap`/`LevelLine`/`EventMarker`/`FeatureBar` contracts, `price_chart` (price, MA lines, level/running overlays, breakout/rejection/structure markers, an optional trend-score axis and regime shading), `heatmap`, `curve_chart`, `bar_chart`, `importance_bars`, and `CHART_KINDS`. Pure: data in, a `Figure` out, no I/O; matplotlib is imported lazily (headless `Agg`) so a plan costs nothing. |
| `core/visualization/sources.py` | Spec 28's loaders: `pnl_heatmap` / `time_of_day_cycle_heatmap` / `time_of_day_weekday_heatmap` (the ledger grouped on two axes, with samples and PRELIMINARY flags), `side_curves` (cumulative P&L per side), `regime_bars`, `feature_bars` (the stored importance JSON), and `option_price_data` / `nifty_price_data` (candles/features/index bars; the option MA stack recomputed with `core.indicators.moving_average`). Pure: frames and records in, the builders' dataclasses out. |
| `apps/anjaneeya_ai/chart_writer.py` | The artefacts: `<stem>_<chart>.<format>` (one image per figure) plus `<stem>_charts.md` (the index that names them), written atomically from one in-memory map of figures, reusing `model_stem`/`ledger_stem` so a chart names what it drew. |
| `apps/anjaneeya_ai/run.py` | `--mode charts` (spec 28): resolve the ledger (`--ledger`/newest), the importance artefact (`--importance`/newest) and the scoped candles/features/index bars, build each chart independently (a missing input skips exactly the charts that need it, with the reason), write the figures + index, print, and register them on the run manifest. `--only` narrows by family or exact name, `--metric` picks the importance column. |
| `core/config_loader.py` | `ChartsConfig`: the `charts:` block (spec, digits, dpi, size, format, colormaps, `top_features`, `max_points`, `max_markers`). |
| `config/model.yaml` | The shipped `charts:` block: the presentation knobs only - never what is drawn. |
| `core/app_bootstrap.py` | `charts` added to `ANJANEYA-AI`'s `--mode` choices. |

### Configuration added

```yaml
explainability:
  spec: "26"            # the master-spec section the structure comes from
  probabilities: 3      # how many of the model's class probabilities the console prints
  digits: 4             # decimals for every number (trailing zeros trimmed)
  rows: 5               # how many of the newest predictions --mode explain describes
  missing_text: unknown # what a NULL feature is rendered as - never 0
  sections:
    - {key: nifty, title: NIFTY, groups: [nifty]}
    - {key: option, title: OPTION, groups: [option]}
    - {key: price_action, title: PRICE ACTION, groups: [levels, level_events, structure]}
    - {key: candle, title: CANDLE, groups: [candle]}
    - {key: support_resistance, title: SUPPORT / RESISTANCE, groups: [support_resistance]}
    - {key: context, title: CONTEXT, groups: [context, relative]}
```

The shipped blocks resolve to **76 stored feature columns**; the AI block is appended after
them from the artefact's metadata.

The importance block (spec 27) in the same file:

```yaml
importance:
  spec: "27"           # the master-spec section; printed in every artefact
  digits: 4            # decimals for every number (trailing zeros trimmed)
  repeats: 5           # permutations per feature - at least 2, or there is no interval
  metric: log_loss     # log_loss (the trainer's metric) | accuracy
  confidence: 0.95     # two-sided Student's t over the repeats
  sample: holdout      # holdout (spec 22's reserved days) | matrix (labelled in-sample)
  max_rows: 2000       # the newest rows measured; the rest are counted, never folded away
  shap: auto           # auto (record why it could not run) | on (refuse) | off (never import)
  shap_rows: 500
  seed: null           # null = model.random_seed
  topics:              # spec 27's sources, then four it does not name but whose columns exist
    - {key: nifty_regime, title: NIFTY regime, groups: [nifty]}
    - {key: option_ma, title: option moving averages, groups: [option]}
    - {key: candle_behaviour, title: candle behaviour, groups: [candle]}
    - {key: support_resistance, title: support / resistance, groups: [support_resistance]}
    - {key: structure, title: market structure, groups: [structure]}
    - {key: price_action, title: "price action, levels (not named by spec 27)", groups: [levels, level_events]}
    - {key: iv, title: implied volatility, columns: [iv, iv_change]}
    - {key: greeks, title: greeks, columns: [delta, gamma, theta, vega]}
    - {key: oi, title: open interest, columns: [oi, oi_change]}
    - {key: option_residual, title: option residual (expected vs actual move), columns: [expected_option_move, actual_option_move, option_residual]}
    - {key: time_of_day, title: time of day, columns: [time_of_day_bucket]}
    - {key: cycle_phase, title: expiry-cycle phase, columns: [expiry_cycle_phase]}
    - {key: time_to_expiry, title: time to expiry, columns: [days_to_expiry]}
    - {key: day_of_week, title: "day of week (not named by spec 27)", columns: [day_of_week]}
    - {key: option_contract, title: "option side (not named by spec 27)", columns: [option_type]}
    - {key: option_context, title: "option context: moneyness and volume (not named by spec 27)", columns: [distance_from_atm, volume]}
```

The sixteen sources between them claim **every trainable stored feature** (the loader proves it,
and a test asserts it too), so `unmapped` is empty for the shipped configuration - it exists for
a contract that grows a column before the map catches up.

The charts block (spec 28) in the same file:

```yaml
charts:
  spec: "28"                # the master-spec section; printed beside every figure's title
  digits: 2                 # decimals for a value printed on a chart
  dpi: 120                  # raster resolution
  width: 14.0               # inches, for the wide charts (price, heatmap, curves)
  height: 7.0
  tall_height: 9.0          # the importance-bars chart
  format: png               # png | svg | pdf
  colormap: RdYlGn          # diverging - a P&L can be positive or negative
  sequential_colormap: viridis
  top_features: 20          # how many features the importance bars show
  max_points: 30000         # a dense series is decimated to fit
  max_markers: 60           # the newest event markers kept
```

### Tests created

* `tests/unit/test_ml_explainer.py` (33 tests) - the policy (shipped configuration versus the
  module's own defaults, a hand-built policy explaining exactly the column it names, the
  refusals, the reserved groups), `format_value` (missing/`0`/flags/timestamps/categories,
  decimals and trailing-zero trimming, numpy scalars), `explain_row` (blocks in order,
  identity, audit, the AI block's ranking and ties, a scored row's own pick, the refusals,
  text/Markdown), and the frame/batch level (newest-first ordering, `limit`, a mismatched
  scored frame, the batch document, an empty frame).
* `tests/unit/test_ml_importance.py` (50 tests) - the policy (shipped configuration and the
  module's own defaults both resolve to the same sixteen sources and every trainable feature is
  claimed exactly once), `metric_score`, `select_sample` (holdout with a reason / matrix with the
  in-sample caveat), `native_importance`, `permutation_importance` (the signal ranked first on a
  synthetic target, determinism at one seed, a categorical kept as the model's vocabulary, the
  interval widening with the repeats, a `NULL`/constant column `measured: false`), the four SHAP
  statuses (off / unavailable / a stubbed module / a broken one), and `build_importance_report`
  (ranked features, per-source sums equal to the per-repeat feature sums, `unmapped` kept apart,
  the three structural caveats present even in a hand-built report, the JSON/MD round-trip).
* `tests/unit/test_config_loader.py` (+9 tests) - the importance block's baseline values and the
  ways it can be wrong: an unknown group, an unknown column, a column claimed twice, a reserved
  group, the reserved `unmapped` key, a source that names nothing, and the nine numeric/enum
  knobs.
* `tests/integration/test_ai_explain_pipeline.py` (19 tests) - the real `main()` against a
  seeded database and a real artefact pair: the three artefacts named after the model, spec
  26's blocks with the AI block last, the model's whole distribution and its ranks, a `NULL`
  feature `unknown` in JSON *and* CSV, the audit block, newest-first `--rows`, the window and
  `--symbol` scopes, the refusals (no model, no rows, a timeframe or horizon the model never
  saw, a `--model` path that does not exist), the caveats in both renderings, `--dry-run`,
  `--no-write`, `--json`, the run manifest, and a before/after count of `features`, `labels`
  and `trades` proving the mode never writes to DuckDB.
* `tests/integration/test_ai_importance_pipeline.py` (19 tests) - the real `main()` against a
  seeded database and a real artefact pair: the three artefacts named after the model, all
  sixteen sources with their intervals, the sample named (spec 22's holdout), a ranking that
  finds the one column the fixture's class really is a function of, determinism across two runs,
  the refusals (no model, no rows, a bad timeframe or horizon, `--repeats 1`), `--dry-run`,
  `--no-write`, `--json`, the run manifest, and a before/after count of `features`, `labels` and
  `trades` proving the mode never writes to DuckDB.
* `tests/unit/test_visualization_charts.py` (35 tests) - the policy (the dataclass defaults and
  the shipped `charts:` block agree; `from_config` reads the six files; `to_dict` carries every
  knob), every builder (a figure from a handful of numbers, and a refusal - never a half-empty
  figure - for an empty price series, an empty grid, a one-point curve and an all-missing bar
  chart), and the loaders (a cell's value is the summed P&L with its sample and its PRELIMINARY
  flag; a configured order leads and an unknown label is appended; the weekday comes from the
  timestamp when the ledger predates the column; the CE/PE curves are cumulative and ordered by
  entry; the regime bars order by the regime sequence; the importance bars rank by |value|, skip
  unmeasured features and attach the interval only to the permutation metric; the option MA stack
  is recomputed from the contract's candles; the breakout/rejection markers fire on the breaks but
  not on the sustained `HELD_*` states; the NIFTY chart computes its running and previous-session
  high/low from the index bars itself).
* `tests/unit/test_config_loader.py` (+13 tests) - the charts block's baseline values, the
  dataclass defaults agreeing with the shipped baseline, and the eleven numeric/enum/string knobs
  that cannot be honoured (a negative `digits`, a `dpi` below 30, a zero size, an empty `format`,
  `colormap` or `sequential_colormap`, a zero `top_features`/`max_markers`, a `max_points` below
  2).
* `tests/integration/test_ai_charts_pipeline.py` (14 tests) - the real `main()` against a seeded
  database, a real ledger and a real importance artefact: one figure per chart plus the index, the
  index naming every figure, `--only` narrowing to a family or an exact chart, `--metric native`
  charting that column, a missing ledger (or a bad `--ledger`) or a missing importance artefact
  skipping exactly the charts that need it with the reason printed, a selector matching no chart
  exiting `0` with `no chart could be built`, `--dry-run`/`--no-write` writing nothing, `--json`
  printing the payload, the figures registered on the run manifest, and a before/after count of
  `features`, `labels` and `trades` proving the mode never writes to DuckDB.


---

## 2. Design decisions worth recording

* **The blocks are groups, not a hand-typed column list.** A `sections:` entry may name
  `groups` (resolved through `FEATURE_GROUPS`), individual `columns`, or both. Groups keep the
  configuration short and impossible to drift: a column added to the NIFTY group is explained
  the next time the mode runs, without editing YAML. The cost is that a *deliberately* narrow
  block has to name columns; both paths are validated identically.
* **A column may be explained exactly once, and the check is at load.** Two blocks printing the
  same feature is how an explanation contradicts itself, and it is the kind of mistake nobody
  notices in a 76-line block. The loader refuses it, and `ExplanationPolicy.__post_init__`
  refuses it again for a policy built without the loader.
* **The `identity` and `provenance` groups are reserved.** Every explanation already renders
  the row's identity in its header (`IDENTITY_FEATURE_COLUMNS`) and its provenance in the audit
  block (`PROVENANCE_COLUMNS`, spec 6.5), so a block repeating them would be the same number
  twice - and in the audit case, the number an *operator* needs first.
* **The AI block is appended, never configured.** Its class list, order and the model's own top
  pick come from the artefact's metadata (`ModelPredictor.classes`), and a configured section
  may not take the key `ai`. That is what "do not hard-code" has to mean here: the one block
  that is *about* the model cannot be a list of classes in a YAML file either.
* **The JSON stores the whole distribution; the console prints the top `probabilities`.** An
  artefact is for auditing, a console is for reading. The printed list is followed by
  `(top 3 of 4 class probabilities; the stored artefact carries every one of them)` so a reader
  who counts three classes is not misled about how many the model has.
* **`0` survives; `NULL` does not become it.** `format_value` treats `None`/`NaN`/`NaT`/`NA` as
  missing and everything else as a measurement, including `0.0` and `False`. Numbers are
  printed with **at most** `digits` decimals and trailing zeros trimmed, so a ratio is not
  dressed up as `0.5000` and a price is not padded with digits it never had.
* **A probability that is not a probability is refused.** `_distribution` checks `[0, 1]` and
  numeric-ness before ranking: a "probability" of 1.5 would otherwise become the model's top
  class and an explanation of nothing.
* **A scored frame is checked row by row, not trusted.** `explain_frame` compares length and
  the `(timestamp, tradingsymbol)` pair of every row against the feature frame. Probabilities
  attributed to the wrong bar would produce a confident, wrong explanation, and it is the one
  failure mode worth a per-row comparison.
* **The CSV is one row per *line*, not per prediction.** A 76-line explanation is a wide table
  nobody wants, and the question this file exists to answer is "which feature said what across
  the last twenty predictions" - which is a pivot, not a scroll.
* **The mode applies no data-accumulation gate.** Spec 4.3's gates govern *claims*; an
  explanation is a description of an input. Refusing to explain a prediction because there are
  four captured days would hide exactly the thing an operator needs to see. The model's own
  label (`EXPLORATORY`, `BELOW_GATE`) is carried in the caveats instead, so the explanation
  inherits the model's standing rather than laundering it.
* **A model can only be explained on its own timeframe.** `artefact.timeframe` disagreeing with
  `--timeframe` is a configuration error (`exit 2`), because the features come from the
  timeframe's rows: explaining a 5m model from 1m rows would describe inputs it never saw.
  A `--horizon` that disagrees with the artefact's own horizon is refused by the same check the
  backtest uses; a horizon with no artefact at all is `NO_MODEL` (`exit 0`), as in `--mode
  backtest`.
* **The console and the artefact come from one object.** `ExplanationBatch.to_text()` is a
  reader's view of what `to_markdown()`/`to_dict()` write; the mode prints the same lines it
  stores (plus the file paths), so a screenshot and a file cannot disagree.

Spec 27 added its own decisions:

* **Permutation, not a single attribution, is the headline.** Native gain is reported because
  spec 27 asks for it and because it is free, but it is computed on the training rows and is
  biased towards high-cardinality features. The permutation delta - shuffle one column, score
  again, measure how much worse the model got - is measured on *held-out* rows by default and
  carries an interval, which is what "report with confidence intervals where practical" asks for.
* **Permutation is repeated at a fixed seed, and the interval is Student's t over the repeats.**
  A single shuffle is one draw from a noisy distribution; `repeats` draws at one seed make the
  number reproducible *and* give it a spread. A source's interval is the sum of its features'
  per-repeat deltas (so it inherits the correlation between them), not the sum of their separate
  intervals - which would pretend the features were independent.
* **The sample defaults to spec 22's holdout, and it says so when it cannot.** `ImportanceSample`
  names the days it actually used and carries a reason; if the split reserves no holdout days the
  mode measures the whole matrix *and* attaches `IN_SAMPLE_CAVEAT`. Silently measuring on the
  training rows would inflate every number in the artefact.
* **A permuted categorical is rebuilt, not overwritten.** The single bug that cost a run while
  this stage was built: assigning an object array into a `pd.Categorical` column degrades it to
  `object`, and LightGBM then refuses the frame ("train and valid dataset categorical_feature do
  not match"). `_permuted_column` shuffles the *values* and rebuilds `pd.Categorical` with the
  model's own vocabulary (from the predictor), so the permuted frame has exactly the dtype and
  categories the booster was trained with. A test pins it.
* **SHAP has four statuses, and only `on` makes it fatal.** `off` never imports the package;
  `auto` records `unavailable` with the extra to install if it is absent and `failed` with the
  exception if it is present but breaks; `on` makes an absent package a configuration error. That
  keeps spec 27's "if available" honest: the optional extra never blocks the two measurements
  that *were* asked for unconditionally.
* **The console and the artefacts come from one `ImportanceReport`.** `to_text()` is a reader's
  view of `to_markdown()`/`to_dict()`; the sources are printed ranked by |delta| with their
  intervals and their unmeasured features named underneath, so a source that measured nothing is
  visible rather than absent.
* **`--dry-run` builds no matrix and imports no booster.** It resolves the artefact, the
  timeframe and the policy, prints the plan (which sources, which rows, which files) and stops -
  which is what makes it safe to run against a live database.

Design decisions specific to spec 28 (visualization):

* **The builders are pure and matplotlib is imported lazily.** `core/visualization/charts.py`
  takes plain data and returns a `Figure` - no file I/O, no database, no ANJANEYA import - so a
  builder is testable with a handful of numbers. `matplotlib` (and the headless `Agg` backend) is
  imported *inside* each builder, never at module import, which is what lets `--dry-run` print a
  plan without importing it at all and lets a scheduled run render with no display attached.
* **One price builder, two charts.** Spec 28 lists a NIFTY chart and an option chart, but they are
  the same picture with different series and overlays, so there is one `price_chart` and the
  optional arguments (`moving_averages`, `levels`, `running`, `markers`, `trend`, `regimes`) are
  exactly the spec's overlay vocabulary. A chart omits whatever the data does not carry; it never
  invents a flat zero line for an overlay that is absent.
* **The MA stack is recomputed, not guessed.** The stored feature row keeps the option-trend
  *derived* columns (alignment, slope, spread), not the `EMA8 high`/`EMA8 low`/`EMA20`/`EMA55`/
  `EMA200` lines themselves, so the chart rebuilds them from the contract's own candles with the
  very same `core.indicators.moving_average` the engine used - one definition, one result. The
  NIFTY chart draws the shortest configured MA combination (its `combination_3`), so the choice
  is configuration rather than a hard-coded period list.
* **A marker is an *event*.** The four level-state columns drive the breakout/rejection markers,
  but only the onset states (`CLOSED_ABOVE`, `CROSSED_ABOVE`, `FAILED_BREAKOUT`, `REJECTED`) mark;
  the sustained `HELD_*` states do not, or a level a price is still holding would be re-marked on
  every bar. The marker sits at the candle's own close for that timestamp, and a timestamp with no
  matching candle is skipped rather than placed at an invented price.
* **A heatmap cell carries its sample, always.** The grid holds the summed net P&L per cell, the
  trade count, and a PRELIMINARY flag computed from `evaluation.preliminary_min_samples`. The
  builder prints the count in every cell and hatches a preliminary one. A cell nothing traded in
  holds `None` and a zero count and is left blank - never a confident `0`, which is the same
  zero-versus-unknown rule the rest of the project enforces, now visible.
* **The weekday is derived when the ledger predates the column.** `day_of_week` is a research
  column the earliest ledgers do not carry, so the weekday axis uses it when present and the
  entry timestamp when it is not. A stale configured order never hides a label: labels outside it
  are appended sorted, exactly as the research matrices do.
* **The bars show what was measured.** `feature_bars` reads the stored importance JSON and keeps
  only features with a real value, so an all-`NULL`/constant feature (which spec 27 reports as
  *not measured*) never becomes a zero bar. The confidence interval is attached only when the
  charted metric is the permutation delta - a native-importance bar must not wear a permutation
  whisker.
* **Each chart is independent, and a missing input is a reason, not a failure.** A run with no
  ledger still draws the importance bars, the NIFTY chart and the option chart; it prints why the
  P&L charts were skipped. Only a run that can build *nothing* returns `NO_DATA` (exit `0`).
* **The mode recomputes nothing.** The heatmaps and the CE/PE curves come from the same ledger
  `--mode report` reads, and the bars come from the artefact `--mode importance` wrote, so a
  picture cannot disagree with the number beside it. That is the whole point of charting stored
  evidence rather than re-deriving it.
* **The index is part of the artefact.** Seven image files with no manifest is a directory, not a
  deliverable, so `chart_writer` writes `<stem>_charts.md` naming every figure, its file and its
  note (cells and preliminary count, sides, bars). A figure that fails to save is counted and
  named in the index, never swallowed.


---

## 3. How to use it

```bash
# what would be explained: which model, which rows, which blocks, which files
uv run python -m apps.anjaneeya_ai.run --mode explain --dry-run

# the newest model's newest predictions (default: model.explainability.rows = 5)
uv run python -m apps.anjaneeya_ai.run --mode explain

# how many / one contract / a window / a specific artefact / a specific horizon
uv run python -m apps.anjaneeya_ai.run --mode explain --rows 20
uv run python -m apps.anjaneeya_ai.run --mode explain --symbol NIFTY06OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode explain --from 2026-09-30 --to 2026-10-01
uv run python -m apps.anjaneeya_ai.run --mode explain --model data/models/1m_h5_..._lightgbm.txt
uv run python -m apps.anjaneeya_ai.run --mode explain --horizon 5

# machine-readable, or without leaving evidence
uv run python -m apps.anjaneeya_ai.run --mode explain --json > explanations.json
uv run python -m apps.anjaneeya_ai.run --mode explain --no-write
```

```
data/reports/explanations/<model stem>_explanations.json   # everything, all 4 classes
data/reports/explanations/<model stem>_explanations.md     # the blocks as a reader's document
data/reports/explanations/<model stem>_entries.csv         # one row per explanation line
```

Exit codes: `0` ok (including `NO_MODEL` - "nothing to explain yet" - and `NO_DATA` - "no
feature rows in scope"), `1` a crash or an unwritable artefact, `2` configuration (a `--model`
path that does not exist, an artefact whose metadata is missing, a timeframe or horizon the
model never saw, a row that does not carry a configured column).

`--mode importance` (spec 27):

```bash
# what would be measured: which model, which rows, which sources, which files
uv run python -m apps.anjaneeya_ai.run --mode importance --dry-run

# the newest model, measured on its own training matrix (default: model.importance.sample = holdout)
uv run python -m apps.anjaneeya_ai.run --mode importance

# a specific artefact / a window / one contract / more shuffles / a different metric
uv run python -m apps.anjaneeya_ai.run --mode importance --model data/models/1m_h5_..._lightgbm.txt
uv run python -m apps.anjaneeya_ai.run --mode importance --from 2026-09-01 --to 2026-10-01
uv run python -m apps.anjaneeya_ai.run --mode importance --symbol NIFTY06OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode importance --repeats 20
uv run python -m apps.anjaneeya_ai.run --mode importance --horizon 5

# machine-readable, or without leaving evidence
uv run python -m apps.anjaneeya_ai.run --mode importance --json > importance.json
uv run python -m apps.anjaneeya_ai.run --mode importance --no-write
```

```
data/reports/importance/<model stem>_importance.json   # everything, every feature and source
data/reports/importance/<model stem>_importance.md     # the ranking as a reader's document
data/reports/importance/<model stem>_features.csv      # one row per feature, ranked
```

Exit codes: `0` ok (including `NO_MODEL` - "nothing to measure yet" - and `NO_DATA` - "no
labelled rows to measure on"), `1` a crash or an unwritable artefact, `2` configuration (a
`--model` path that does not exist, an artefact whose metadata is missing, a timeframe or
horizon the model never saw, `--repeats` below 2).

`--mode charts` (spec 28):

```bash
# what would be drawn: which ledger, which importance artefact, which symbol, which figures
uv run python -m apps.anjaneeya_ai.run --mode charts --dry-run

# everything the stored data can support (the newest ledger + newest importance artefact)
uv run python -m apps.anjaneeya_ai.run --mode charts

# one family, one exact chart, a specific ledger, a specific importance artefact, another metric
uv run python -m apps.anjaneeya_ai.run --mode charts --only heatmap
uv run python -m apps.anjaneeya_ai.run --mode charts --only curves_ce_vs_pe
uv run python -m apps.anjaneeya_ai.run --mode charts --ledger data/backtests/1m_h5_..._ledger.parquet
uv run python -m apps.anjaneeya_ai.run --mode charts --importance data/reports/importance/..._importance.json
uv run python -m apps.anjaneeya_ai.run --mode charts --metric native

# the option contract to chart (default: the busiest one in scope), a window, machine-readable
uv run python -m apps.anjaneeya_ai.run --mode charts --symbol NIFTY06OCT2625000CE
uv run python -m apps.anjaneeya_ai.run --mode charts --from 2026-09-30 --to 2026-10-01
uv run python -m apps.anjaneeya_ai.run --mode charts --json > charts.json
uv run python -m apps.anjaneeya_ai.run --mode charts --no-write
```

```
data/reports/charts/<ledger stem>_heatmap_time_of_day_cycle_phase.png
data/reports/charts/<ledger stem>_heatmap_time_of_day_day_of_week.png
data/reports/charts/<ledger stem>_curves_ce_vs_pe.png
data/reports/charts/<ledger stem>_regime_performance.png
data/reports/charts/<ledger stem>_importance_bars.png
data/reports/charts/<ledger stem>_price_nifty.png
data/reports/charts/<ledger stem>_price_option.png
data/reports/charts/<ledger stem>_charts.md                 # the index that names them
```

Exit codes: `0` ok (including `NO_DATA` - "no chart could be built"), `1` a crash or an
unwritable figure, `2` configuration. A missing ledger, a missing importance artefact or missing
stored candles is **not** an error: the run prints the reason and skips exactly the charts that
need it.


---

## 4. Example output (real, from the integration fixture)

`--dry-run`, so the operator can see the plan without touching disk:

```
$ uv run python -m apps.anjaneeya_ai.run --mode explain --dry-run
ANJANEYA-AI explanations (spec 26) - dry run, nothing is written
  model      : 1m_h5_2026-09-28_2026-10-01_lightgbm.txt (lightgbm, horizon 5m)
  timeframe  : 1m
  window     : beginning .. newest
  scope      : every stored contract
  rows       : 480 feature row(s) in scope, the newest 5 of them explained
  blocks     : NIFTY, OPTION, PRICE ACTION, CANDLE, SUPPORT / RESISTANCE, CONTEXT, AI (76 stored feature(s) plus the model's class probabilities)
  policy     : 4 decimal(s), top 3 probability(ies), a NULL feature printed as 'unknown'
  pipeline   : read features -> score them -> build the blocks -> JSON/Markdown/CSV
  writes     : .../data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_explanations.json|.md and .../1m_h5_2026-09-28_2026-10-01_lightgbm_entries.csv
```

One prediction (`--rows 1`), abridged in the middle - the header, the AI block and the caveats
are verbatim:

```
$ uv run python -m apps.anjaneeya_ai.run --mode explain --rows 1
ANJANEYA-AI explanations (spec 26) - 1 prediction(s), 76 stored feature(s) of which 58 unknown at that instant
  model      : .../data/models/1m_h5_2026-09-28_2026-10-01_lightgbm.txt
  timeframe  : 1m 5m
  blocks     : NIFTY, OPTION, PRICE ACTION, CANDLE, SUPPORT / RESISTANCE, CONTEXT, AI

NIFTY06OCT2625000CE @ 2026-10-01T10:14:00+05:30
  prediction : UP_CONTINUATION p=0.804 (runner-up DOWN_CONTINUATION 0.196, margin 0.6079)
  NIFTY:
    nifty_fast_score            : unknown
    nifty_medium_score          : unknown
    nifty_long_score            : unknown
    nifty_composite_trend_score : 0.4833
    nifty_trend_acceleration    : unknown
    nifty_trend_persistence     : unknown
    nifty_regime                : BULLISH_DEVELOPING
  OPTION:
    option_trend_score          : 0.1
    option_atr                  : 1.59
    option_insufficient_history : unknown
    ...
  CANDLE:
    candle_body                             : 0.5
    candle_direction                        : BULLISH
    candle_inside_flag                      : false
    candle_range                            : unknown
    ...
  SUPPORT / RESISTANCE:
    support_strength           : unknown
    resistance_strength        : unknown
    support_touch_count        : 1
    support_rejection_count    : unknown
    ...
  CONTEXT:
    distance_from_atm    : -750
    days_to_expiry       : 2
    expiry_cycle_phase   : MID_CYCLE
    time_of_day_bucket   : MIDDAY
    iv                   : unknown
    delta                : unknown
    oi                   : 5000
    volume               : 100
    ...
  AI:
    UP_CONTINUATION   : 0.804  <- predicted
    DOWN_CONTINUATION : 0.196
    RANGE             : 0
    (top 3 of 4 class probabilities; the stored artefact carries every one of them)

  wrote      : .../data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_explanations.json
  wrote      : .../data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_explanations.md
  wrote      : .../data/reports/explanations/1m_h5_2026-09-28_2026-10-01_lightgbm_entries.csv
  caveat: an explanation quotes the stored feature row of that instant as-is, together with the probabilities one model produced from it; it is a description of the inputs, not a claim about why the market moved
  caveat: spec 24's decision layer does not exist yet: nothing here turns a probability distribution into a position, so an explanation is not a recommendation
  caveat: the input is the stored row: its fetch_timestamp, source and quality_flag are in the audit block, and a stale or low-quality row explains a stale or low-quality prediction
  caveat: the model being explained is labelled EXPLORATORY by its own training run: an explanation inherits the model's standing, it does not add to it
```

The Markdown artefact's AI block and caveat list:

```markdown
**AI**

- `UP_CONTINUATION`: 0.804 **(predicted)**
- `DOWN_CONTINUATION`: 0.196
- `RANGE`: 0
- _top 3 of 4 class probabilities shown; the stored artefact carries every one of them_

## Caveats

- an explanation quotes the stored feature row of that instant as-is, together with the
  probabilities one model produced from it; ...
```

The JSON's top-level keys and one explanation's envelope (the fixture's model is a 15-tree
LightGBM trained on the four-day fixture, so these probabilities are the fixture's, not a claim
about NIFTY):

```
keys        : caveats, config_hash, counts, explanations, model, policy, run_id, window
policy      : spec 26, probabilities 3, digits 4, rows 5, missing_text "unknown", 6 blocks
window      : 1m, horizon 5m, symbols [], from null, to null
counts      : predictions 1, entries 76, unknown 58
identity    : timestamp, underlying NIFTY, tradingsymbol NIFTY06OCT2625000CE,
              instrument_token 900001, expiry, strike 25000.0, option_type CE, timeframe 1m
audit       : source DERIVED, fetch_timestamp 2026-10-01T10:14:01+05:30,
              quality_flag OK, run_id anjaneeya_engine_20260929-153000_deadbeef
prediction  : UP_CONTINUATION 0.804, runner-up DOWN_CONTINUATION 0.196, margin 0.608
sections    : nifty, option, price_action, candle, support_resistance, context, ai
```

The CSV's first rows - one row per *line*, which is what makes "which feature said what" a pivot
table:

```
      tradingsymbol                 timestamp predicted_class  predicted_probability section                      column group    kind  known    text
NIFTY06OCT2625000CE 2026-10-01T10:14:00+05:30 UP_CONTINUATION                0.80397   nifty            nifty_fast_score nifty unknown  False unknown
NIFTY06OCT2625000CE 2026-10-01T10:14:00+05:30 UP_CONTINUATION                0.80397   nifty          nifty_medium_score nifty unknown  False unknown
NIFTY06OCT2625000CE 2026-10-01T10:14:00+05:30 UP_CONTINUATION                0.80397   nifty nifty_composite_trend_score nifty  number   True  0.4833
```


### `--mode importance` (spec 27), same fixture, newer stage

`--dry-run` - the plan, built without touching the matrix or the booster:

```
$ uv run python -m apps.anjaneeya_ai.run --mode importance --dry-run
ANJANEYA-AI feature importance (spec 27) - dry run, nothing is written
  model      : 1m_h5_2026-09-28_2026-10-01_lightgbm.txt (lightgbm, horizon 5m)
  timeframe  : 1m
  window     : beginning .. newest
  scope      : every stored contract
  matrix     : 248 labelled row(s) over 4 day(s), 77 feature(s)
  sample     : holdout - 62 of 62 row(s)
  measurement: native importance off the booster, then 5 permutation(s) per feature with log_loss at seed 42, 95% intervals
  sources    : nifty_regime, option_ma, candle_behaviour, support_resistance, structure, price_action, iv, greeks, oi, option_residual, time_of_day, cycle_phase, time_to_expiry, day_of_week, option_contract, option_context, unmapped
  SHAP       : auto (at most 500 row(s))
  writes     : .../data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_importance.json|.md and .../1m_h5_2026-09-28_2026-10-01_lightgbm_features.csv
```

The full measurement (`--repeats 3` to keep the example short), abridged in the middle - the
header, the sources and the caveats are verbatim. The fixture holds only four days and almost no
option context, so most features are `NULL` or constant on 62 holdout rows and the honest answer
is "not measured", which is exactly what the artefact says:

```
$ uv run python -m apps.anjaneeya_ai.run --mode importance --repeats 3
ANJANEYA-AI feature importance (spec 27) - 6 of 77 feature(s) measured over 62 row(s), 3 permutation(s) at seed 42
  model      : .../data/models/1m_h5_2026-09-28_2026-10-01_lightgbm.txt (lightgbm, horizon 5m, EXPLORATORY)
  sample     : holdout - 62 row(s) over 1 day(s), 2026-10-01T09:15:00+05:30 .. 2026-10-01T09:45:00+05:30
  metric     : log_loss (lower_is_better), baseline 10.4995
  sources    : 5 of 16 measured
    #1 nifty_regime (NIFTY regime): delta 0.0006 [-0.0011, 0.0024] over 2 feature(s)
    #2 candle_behaviour (candle behaviour): delta 0.0002 [-0.0001, 0.0005] over 1 feature(s)
    #3 option_ma (option moving averages): delta 0 [0, 0] over 1 feature(s)
    #4 time_of_day (time of day): delta 0 [0, 0] over 1 feature(s)
    #5 option_contract (option side (not named by spec 27)): delta -0.0222 [-0.2633, 0.2188] over 1 feature(s)
    -  iv: none of its 2 feature(s) could be measured in this sample: every value of this column is NULL in the sample, so the model is fed the same missing value either way
    -  greeks: none of its 4 feature(s) could be measured in this sample: every value of this column is NULL in the sample, so the model is fed the same missing value either way
    -  oi: none of its 2 feature(s) could be measured in this sample: this column holds a single value in the sample, so permuting it cannot change the score
    ...
  features   (top 3 of 6 measured)
    #1 nifty_composite_trend_score (nifty_regime): delta 0.0006 [-0.0011, 0.0024]
    #2 candle_direction (candle_behaviour): delta 0.0002 [-0.0001, 0.0005]
    #3 nifty_regime (nifty_regime): delta 0 [0, 0]
  unmeasured : 71 feature(s) the sample could not say anything about (all NULL or constant); each reason is in the artefact
  SHAP       : unavailable - shap is an optional extra of ANJANEYA (pip install -e '.[explain]'): native and permutation importance need no such package
  wrote      : .../data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_importance.json
  wrote      : .../data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_importance.md
  wrote      : .../data/reports/importance/1m_h5_2026-09-28_2026-10-01_lightgbm_features.csv
  caveat: importance ranks the information *this model* used from *these inputs*; it is not a claim that a feature moves a market, and a model labelled BELOW_GATE is ranked as the unvalidated model it is
  caveat: a permuted column is scored with all other columns held at their stored values: a feature that duplicates, or is a combination of, other features keeps less of the credit than its influence suggests, and one of a duplicated pair may take both shares
  caveat: the sample is spec 22's reserved holdout - the days the split kept out of the final model's training window - so these deltas are measured out-of-sample with respect to that split
  caveat: the model measured is labelled EXPLORATORY by its own training run: this ranking describes that model, not a validated one
  note: matrix: 248 labelled row(s) over 4 day(s), 77 feature(s)
  note: measurement: 62 row(s) of the holdout sample (62 available), 3 permutation(s) at seed 42, log_loss (baseline 10.4995)
  note: 71 feature(s) could not be measured in this sample and are reported as not measured (never as a zero): candle_body, candle_range, delta, gamma, iv, ...   # every one named
```

The JSON's top-level keys, policy and two feature envelopes (the fixture's model is a 15-tree
LightGBM trained on the four-day fixture, so these numbers describe the fixture, not NIFTY):

```
keys        : caveats, config_hash, counts, features, model, notes, policy, run_id, sample, shap, topics, window
policy      : spec 27, repeats 3, metric log_loss, confidence 0.95, sample holdout, max_rows 2000, shap auto, seed 42
window      : 1m, horizon 5m, symbols []
counts      : features 77, measured 6, unmeasured 71, shap_status unavailable
model       : 1m_h5_2026-09-28_2026-10-01_lightgbm.txt, results_label EXPLORATORY
sample      : holdout, 62 rows used of 62 available, 1 day, capped false
measured    : column nifty_regime, group nifty, topic nifty_regime, native 12, permutation mean 0.0, ci [-0.0001, 0.0001]
unmeasured  : column iv, measured false, reason "every value of this column is NULL in the sample, so the model is fed the same missing value either way"
```

The CSV is one row per *feature* (77 rows), ranked by permutation delta with the unmeasured ones
after; every row carries its `reason` (empty for a measured feature):

```
 rank                      column          topic group measured native permutation_mean permutation_std ci_low ci_high null_share constant reason
    1 nifty_composite_trend_score   nifty_regime nifty     True     88           0.0006          0.0008 -0.0011  0.0024        0.0    False
    2          candle_direction candle_behaviour candle     True     41           0.0002          0.0006 -0.0001  0.0005        0.0    False
                               gamma         greeks greeks    False     12              NaN             NaN    NaN     NaN        1.0     True every value of this column is NULL in the sample, ...
```



---

### `--mode charts` (spec 28), same fixture

The plan, so an operator knows which inputs exist and which charts that makes possible:

```
$ uv run python -m apps.anjaneeya_ai.run --mode charts --dry-run
ANJANEYA-AI chart plan (spec 28) - 1m
  format     : png at 120 dpi, 14.0x7.0 in (bars 9.0 in)
  ledger     : .../data/backtests/1m_h5_2026-09-28_2026-10-01_ledger.parquet (48 trade(s))
  importance : 1m_h5_2026-09-28_2026-10-01_lightgbm_importance.json
  symbol     : NIFTY06OCT2625000CE
  wrote      : nothing (--dry-run); the directory would be .../data/reports/charts
  would build:
    - heatmap_time_of_day_cycle_phase [heatmap] - net P&L by time-of-day bucket x cycle phase
    - heatmap_time_of_day_day_of_week [heatmap] - net P&L by time-of-day bucket x weekday
    - curves_ce_vs_pe [curves] - cumulative net P&L by option side
    - regime_performance [regime] - net P&L by NIFTY regime at entry
    - importance_bars [importance] - permutation importance per feature (with intervals)
    - price_nifty [price] - NIFTY price with MA lines, trend score and regime
    - price_option [price] - one option contract with its MA stack and overlays
```

The run itself:

```
$ uv run python -m apps.anjaneeya_ai.run --mode charts
ANJANEYA-AI charts (spec 28) - 7 figure(s), stem 1m_h5_2026-09-28_2026-10-01
  heatmap_time_of_day_cycle_phase: 12 cell(s), 12 preliminary
  heatmap_time_of_day_day_of_week: 12 cell(s), 12 preliminary
  curves_ce_vs_pe: 2 side(s) from 48 trade(s)
  regime_performance: 2 regime(s)
  importance_bars: 3 feature(s) from 1m_h5_2026-09-28_2026-10-01_lightgbm_importance.json
  price_nifty: 960 bar(s)
  price_option: 480 bar(s)
  wrote      : .../data/reports/charts/1m_h5_2026-09-28_2026-10-01_price_option.png
  wrote      : .../data/reports/charts/1m_h5_2026-09-28_2026-10-01_charts.md
```

When an input is missing, the mode says which charts it cost rather than failing:

```
ANJANEYA-AI charts (spec 28) - 3 figure(s), stem charts_anjaneeya_ai_20261001-133306_f4d280dc
  importance_bars: 3 feature(s) from ...
  price_nifty: 3271 bar(s)
  price_option: 480 bar(s)
  wrote      : .../data/reports/charts/..._price_option.png
  wrote      : .../data/reports/charts/..._charts.md
  skipped    : the P&L charts need a trade ledger under .../data/backtests - run --mode backtest first, or pass --ledger PATH
  skipped    : the importance bars were skipped: no importance artefact under .../data/reports/importance - run --mode importance first
```



## 5. Assumptions and open questions


* **"Explainable" here means *structured*, not *causal*.** Spec 26 asks for the inputs of a
  prediction in plain form, and that is what this stage produces: the stored values the model
  was fed plus the distribution it returned. Nothing here attributes a move to a feature or
  ranks features by influence - that is spec 27 (Stage 5.2, `--mode importance`), which measures
  the *model's* use of its inputs rather than the market's dependence on them. The caveat
  (`EXPLANATION_CAVEAT`) says so in every explanation artefact, and `IMPORTANCE_CAVEAT` says the
  same for a ranking.
* **The explanation quotes the feature row; it never recomputes it.** A feature row written
  by a buggy engine is explained as the buggy row. That is deliberate - the explanation's job
  is to say what the model saw - but it means an explanation is not a validation of the
  feature pipeline (spec 19's `--mode validate` is).
* **76 of the feature contract's columns are explained; identity and provenance deliberately
  are not.** The 8 identity columns are in every header and the 4 provenance columns in every
  audit block, so every column of the stored row is *present* in the artefact - but only the 76
  behaviour columns are *blocks*, and only numbers get a `kind`/`known` pair. `--mode importance`
  ranks the model's own feature list (77 trainable columns in the shipped contract), which is a
  superset of the 76 explained here.
* **The AI block is one model's distribution at one instant.** `--mode explain` scores every
  row in scope (the same call `--mode backtest` makes) but writes only the newest `rows` of
  them; the rest are discarded rather than stored. A per-row explanation of a whole session is
  a `--rows` value away, but the artefacts grow linearly, so the default stays small.
* **A model trained on one timeframe cannot be explained on another, and the check is the
  metadata's own `timeframe` field.** An artefact written before that field existed (none are
  shipped) would skip the check rather than fail - the mode explains it and the artefact records
  the timeframe that was used.
* **The feature row's `expiry` renders as whatever the store returns.** DuckDB returns a `DATE`
  for a real store (so `2026-10-06`), while a pandas-built fixture hands back a
  `Timestamp` (`2026-10-04T00:00:00`); the JSON is the store's value either way, and the
  fixture's oddity is a fixture, not a formatting rule.
* **Spec 26's "every prediction" is bounded by what is stored.** Rows whose `(timestamp,
  contract)` never reached the features table cannot be explained, and the mode says
  `NO_DATA` rather than inventing a row. The full pipeline (`--mode build-features`) is the
  only way to widen the set.
* **No gate, no score, no recommendation.** Nothing here creates a decision: the distribution
  is described, not acted on. Spec 24's decision layer is still the missing piece (master-prompt
  Phase 24), and it is the one thing that would let this stage's output be read as advice. It is
  named in the caveats precisely so it cannot be read that way by accident.

Spec 27's assumptions and open questions:

* **A permutation delta is "how much this model needs this column", not "how much this column
  moves the market".** Native gain is biased by cardinality and by how the trees were grown;
  permutation is biased by correlation (a feature that duplicates another keeps only part of the
  credit, and a model split across two copies may give one of them both). Both are reported,
  neither is the truth, and the caveats say which is which.
* **The sample is the honest part, and it is stated.** With the shipped configuration the deltas
  are measured on spec 22's holdout days only - the rows the final model was not trained on - so
  they are out-of-sample *with respect to that split*. When the split reserves none, the mode
  measures the whole matrix and says `IN_SAMPLE_CAVEAT`; it never silently measures in-sample.
* **The shipped sample is 2000 rows by default, because permutation cost grows with rows and
  repeats.** A permutation of a 77-column matrix at 5 repeats is ~385 full predict calls; the cap
  and the `--from/--to/--symbol` scope are the two dials, and the rows left out are counted in the
  `notes`, never folded away.
* **Confidence intervals are Student's t over the repeats, and a repeat count is not a sample
  size.** Five shuffles give a spread, not a population; the interval widens if `--repeats` is
  lowered, and `--repeats 1` is refused outright because it has no spread to report. This is the
  most literal reading of "where practical" that still keeps the number reproducible.
* **SHAP is genuinely optional.** When it is absent the artefact records `unavailable` with the
  extra to install (`pip install -e '.[explain]'`) and the run still succeeds; only
  `model.importance.shap: on` turns its absence into a configuration error, because then the
  operator explicitly asked for a measurement the environment cannot make. When it runs it is one
  `shap_values` per class, mean |value| per feature, on at most `shap_rows` newest rows.
* **The ranking is about one model artefact, and it says which.** The model's `results_label`
  (`EXPLORATORY`, `BELOW_GATE`) travels in the caveats, so a ranking of an unvalidated model is
  read as one. There is still no decision layer (master-prompt Phase 24): a ranking is not a
  filter, and nothing here drops or selects a feature for the next training run.
* **`unmapped` is a safety net, not a source.** The shipped map claims every trainable feature, so
  `unmapped` is empty; it exists so a column added to the contract before the YAML catches up is
  *reported* rather than attributed to a source that never measured it. The reserved `unmapped`
  key cannot be used as a user source.

Spec 28's assumptions and open questions:

* **A chart is a view of the evidence, not new evidence.** The heatmaps, curves and regime bars
  read the same ledger `--mode report` reads, and the importance bars read the artefact
  `--mode importance` wrote; the price charts read stored candles/features/index bars. Nothing is
  recomputed, so a chart cannot disagree with the number beside it - but a chart also inherits
  every caveat of the artefact behind it, which is why the importance chart says which model it
  ranks and the P&L charts are PRELIMINARY-flagged.
* **The CE-vs-PE curve is cumulative over the ledger's window, not one session.** A ledger can span
  many days, so "intraday P&L curve" is drawn as cumulative net P&L against entry time, one line
  per option side. A single-session curve would be a different (and more misleading) picture when
  the ledger has more than one day in it; the axis and the title say which one this is.
* **The price chart's overlays are what the stored row carries.** Support/resistance is drawn from
  the stored swings and previous-day levels (the contract's feature row keeps the *strengths*, not
  a support/resistance price per bar, so no invented line is drawn); the option MA lines are
  recomputed from the contract's own candles because the row keeps only the derived trend columns.
* **Matplotlib is a real dependency and the backend is chosen for a headless run.** The system
  already required `matplotlib` from Phase 16; the builders select `Agg` inside each call, so a
  scheduled run renders with no display and `--dry-run` never imports the library at all.
* **A figure has no schema version.** Spec 28's output is images plus a Markdown index under
  `data/reports/charts/` - it adds no table, no column and no row, so `schema_version` stays **1**
  and the mode is read-only against DuckDB, exactly like `--mode explain` and `--mode importance`.
* **There is still no decision layer.** A chart of a model's entries is not a signal, and spec 24's
  decision layer (master-prompt Phase 24) remains the one missing piece that would let any of this
  be read as advice. The price chart marks *where the model acted* on the stored ledger; it does
  not label those points as trades to take.


---

## 6. Master-prompt checklist (§33)

| Item | Where |
| --- | --- |
| **1. Files created** | `core/ml/explainer.py`, `apps/anjaneeya_ai/explain_writer.py`, `core/ml/importance.py`, `apps/anjaneeya_ai/importance_writer.py`, `core/visualization/charts.py`, `core/visualization/sources.py`, `apps/anjaneeya_ai/chart_writer.py`, `tests/unit/test_ml_explainer.py`, `tests/integration/test_ai_explain_pipeline.py`, `tests/unit/test_ml_importance.py`, `tests/integration/test_ai_importance_pipeline.py`, `tests/unit/test_visualization_charts.py`, `tests/integration/test_ai_charts_pipeline.py`, `docs/phase_reports/phase_05.md` |
| **2. Files modified** | `config/model.yaml` (the `explainability:`, `importance:` and `charts:` blocks), `core/config_loader.py` (`ExplainabilityConfig`, `ExplainabilitySectionConfig`, `ImportanceConfig`, `ImportanceTopicConfig`, `ChartsConfig`, the `ModelConfig` fields), `core/ml/__init__.py` (exports), `core/visualization/__init__.py` (exports), `core/app_bootstrap.py` (`explain`, `importance`, `charts` in `DEFAULT_MODES`), `apps/anjaneeya_ai/run.py` (imports, `MODE_PLAN`, `AVAILABLE_MODES`, `--rows`/`--repeats`/`--only`/`--metric`/`--importance`, `--no-write`/`--model` help, the `--mode explain`, `--mode importance` and `--mode charts` sections, dispatch, module docstring), `docs/runbook.md` (mode tables, the importance and charts paragraphs, the reports paths), `tests/unit/test_config_loader.py` (+11 explain, +9 importance, +13 charts), `README.md` (status table, the explain / importance / charts sections, layout, testing, honest caveats) |
| **3. What was implemented** | §1 (modules) and §2 (the design decisions behind them) |
| **4. Configuration added** | §1's `explainability:` block (6 knobs + 6 blocks), `importance:` block (9 knobs + 16 sources) and `charts:` block (12 presentation knobs), all validated at load |
| **5. Tests created** | §1's *Tests created*: 33 + 50 + 35 unit, 11 + 9 + 13 config, 19 + 19 + 14 integration = 203 tests |
| **6. Test results (actual output)** | `uv run ruff check .` clean. Targeted: `tests/unit/test_visualization_charts.py` 35 passed, `tests/unit/test_config_loader.py` + `tests/integration/test_app_entrypoints.py` 152 passed (30.46s together), `tests/integration/test_ai_charts_pipeline.py` 14 passed (183.30s). Full repository: `uv run pytest -q` - `1863 passed, 1 skipped, 1 failed in 1576.34s (0:26:16)`; the skip is the live-capture smoke test that needs real Zerodha credentials, and the one failure (`test_ai_backtest_pipeline.py::test_without_a_model_the_run_says_so_and_writes_nothing`, `_duckdb.OutOfMemoryException: Allocation failure` while seeding) is an environment memory flake - it passes in isolation in 13.34s, and it touches no code changed by spec 28. Plus the manual CLI runs captured in §4. |
| **7. Example output (small, real)** | §4 above: for explain, `--dry-run`, one full prediction, the Markdown AI block, the JSON envelope, the CSV head; for importance, the `--dry-run` plan, the source and feature rankings with intervals, the caveats and the JSON/CSV envelopes; for charts, the `--dry-run` plan of the seven figures, the run that writes them plus the index, and the run that skips the P&L charts with its reason - all from the integration fixture |
| **8. Known limitations** | §5 (*Assumptions and open questions*) and every caveat the artefacts carry |
| **9. Assumptions made** | §5, and the most load-bearing ones: an explanation describes inputs rather than causes; an importance ranking describes a model's use of its inputs rather than the market; a chart is a view of the stored evidence and inherits its caveats; and the stored feature row / training matrix / ledger is quoted as-is |





