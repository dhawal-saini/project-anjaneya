# Phase 3 report — ANJANEYA-ENGINE (Stages 3.2e, 3.3, 3.4a and 3.4b)

**Status: Stages 3.2, 3.3, 3.4a and 3.4b complete.** `--mode build-candles` reads stored ticks,
builds 1m bars, merges 3m/5m/15m from them, replaces the affected
`(day, timeframe, contract)` scopes, records a `data_coverage` verdict per stream and
mirrors the result to Parquet. `--mode enrich-greeks` solves each option bar's implied
volatility against the NIFTY level at the bar's *close instant*, derives
`delta`/`gamma`/`theta`/`vega` from it and `UPDATE`s exactly those columns (plus
`days_to_expiry`/`distance_from_atm`) onto the stored bars. `--mode build-features` turns
the candles into one row per `(timestamp, contract)` covering specs 8-16 - including
spec 16's `expected_option_move`/`option_residual`, which finally have a delta to work
with - and `--mode validate` re-reads the stored rows and runs spec 19's checks
(including a no-look-ahead recomputation) into a JSON report. `--mode universe` (stage
3.4a, spec 17) resolves the configured ATM anchor, strike ladder and expiries from
*stored* data and reports every selected contract with its candle/feature counts, and
`--mode build-labels` (stage 3.4b, spec 20) turns the same candles into the forward-looking
half of the dataset - `future_return`, the up/down excursions, `future_trend_change` and a
discretised class per horizon - in a **physically separate pass** that never reads a feature
row. None of them needs credentials or network, and all of them are covered end to end by the
offline suite.

Phase 3's engine scope is complete. **No schema change** — `schema_version` stays
**1**, so an existing database needs no migration: the greeks are columns the schema has
carried since Phase 1, and the universe is a *selection* that stores nothing.

Phase 4 (ANJANEYA-AI) continues in [`phase_04.md`](phase_04.md): its first stage builds the
leak-free training matrix out of these rows and cuts it with spec 22's purged walk-forward
splitter (`--mode dataset`); the second trains a LightGBM model per horizon on those folds,
scores it against spec 21's baselines and writes the model artefact (`--mode train`).

---

## 1. What was built

| Module | Responsibility |
| --- | --- |
| `core/storage/readers.py` (new) | The engine's read side: `read_ticks` (session window, symbol subset, quality filter), `read_candles`, and the discovery queries `trading_days_with_ticks` / `symbols_with_ticks` / `quality_counts`. No business logic in SQL. |
| `core/storage/coverage.py` (moved) | `StreamStats`, `CoverageTracker`, `build_coverage_row` — previously inside the capture app. Capture and engine must never disagree about "how much did we record?", so the vocabulary now lives in `core` and the capture module re-exports it (its tests were unchanged). |
| `apps/anjaneeya_engine/candles.py` (new) | Pure tick→candle logic: `build_1m_candles`, `aggregate_candles`, `build_timeframes`, `bucket_start`, `expected_bar_openings`, `missing_bar_summary`. Frames in, row objects out; no I/O. |
| `apps/anjaneeya_engine/storage_writer.py` (new) | `CandleWriter`: delete-then-append per scope, coverage upsert, atomic Parquet mirror (temp file + rename), failure counting. Borrows the caller's connection. |
| `apps/anjaneeya_engine/run.py` (extended) | `--mode build-candles` with `--date/--all/--symbol/--timeframes/--dry-run/--json`, a human and a JSON run summary, and exit codes 0/1/2. |

Supporting changes:

* `core/storage/__init__.py` — exports the readers and the coverage vocabulary.
* `core/config_loader.py` + `config/settings.yaml` — a validated `engine:` section
  (`mirror_parquet`), optional so a Phase-1 style settings file still loads.
* `apps/anjaneeya_engine/candles.py` — `_as_date`, because DuckDB hands a `DATE`
  column back as a pandas `Timestamp` and the row contract wants a `date`.
* `tests/support/engine.py` (new) — synthetic tick streams, a seeded database,
  the real `AppContext` and the CLI argument list.
* `tests/integration/test_app_entrypoints.py` — the engine's "unimplemented mode"
  now points at `build-features`, since `build-candles` is implemented.

---

## 2. Design decisions worth recording

**A bar's `timestamp` is its open time, floored in IST.** The 09:15 bar holds
`[09:15:00, 09:16:00)`. No session anchor, no offset: every consumer (features,
labels, joins) can then compare bar timestamps across timeframes without knowing
how the bar was produced.

**Higher timeframes are merged from the 1m bars, never rebuilt from ticks.**
Cheaper, and more importantly it makes the invariants structural: the 15m open is
*always* the first 1m open, so a feature computed on 15m data cannot disagree with
the same feature computed on 1m data.

**Nothing is fabricated.** A minute without a tick produces no bar. A gap is
reported through `data_coverage` (expected vs actual rows + the run-length of the
holes), never filled with a synthetic price or a zero volume. `expected_rows` is
exact for candles (375/125/75/25 on a normal day), so the ledger has a real
denominator — unlike the tick stream, where only timing gaps can be measured.

**Volume is a sum of cumulative increments, and the awkward cases are visible.**
Kite publishes cumulative day volume, so a bar's volume is the sum of the
increments observed inside it; the day's first tick has no predecessor (so the
09:15 bar starts where observation started), and a negative increment (a
mid-session counter reset on re-subscribe/roll) contributes zero and is *counted*
as `volume_resets` in the summary. A `NULL` volume column (the NIFTY index)
yields `volume = NULL`, not `0`.

**Delete-then-append, not `INSERT OR REPLACE`.** Coverage and index history are
upserts because a re-fetch replaces exactly the rows it is about. A candle
rebuild's correct row count is only known *after* the build, so an upsert would
leave a previous run's later bars in place when a day is rebuilt from fewer
ticks — stale rows that look real. The rebuild therefore deletes the
`(session, timeframe, contract)` scope and appends.

**`replace_candles([])` is meaningful, not a no-op.** Re-reading a scope and
finding nothing is how a rebuild clears rows that no longer have any input.

**Only `quality_flag = 'OK'` ticks build candles.** A `SUSPECT`/`STALE`/
`OUT_OF_SESSION` tick can never shape an OHLCV bar; what was left out is printed
per run (`excluded: SUSPECT=12 tick(s)`), and a contract whose ticks are *all*
flagged gets an explicit `MISSING_TICK` coverage row per timeframe instead of
silence.

**One DuckDB connection, owned by the CLI.** DuckDB allows a single read-write
process per file (a second in-process connection conflicts too), so the engine
opens one connection and the writer borrows it. The connection is closed *before*
`finalize_run` writes the manifest row — a test asserts that the manifest actually
lands, which is the behavioural guard for that ordering.

**A failure is local and loud.** One contract that cannot be built is reported in
the summary, logged, and turns the run into `PARTIAL` with exit code 1; every
other contract is still written. Its *stored* candles are left untouched
deliberately: a stale bar that the summary names is safer than deleting data that
cannot be rebuilt right now.

---

## 3. Verification performed

`uv run pytest -q` — **577 passed, 1 skipped** (the live capture smoke test skips
itself without credentials). `uv run ruff check .` — clean.

New tests (80):

* `tests/unit/test_engine_candles.py` (30) — open-time keying, IST flooring per
  timeframe, volume arithmetic across minutes, counter resets, `NULL` volume,
  out-of-order ticks, missing minutes producing no bar, naive/missing/wrong-typed
  input being rejected, merge rules (open/high/low/close/volume/OI/IV/metadata),
  `1m` aggregation being a no-op copy, session bar counts (375/125/75/25), and gap
  run-length reporting.
* `tests/unit/test_storage_readers.py` (13) — window/symbol/quality filtering,
  ordering, the dtypes that reach the builder, unknown-column rejection, IST day
  grouping (a 23:00 UTC tick belongs to the *next* IST date), `quality_counts`,
  `read_candles` filters, and that the coverage vocabulary is importable from
  `core.storage` while still re-exported by capture.
* `tests/unit/test_engine_writer.py` (16) — delete-then-append leaves no stale
  rows, an empty replacement clears a scope, other scopes untouched, scope
  counting, coverage upsert, mirror layout/schema/atomic overwrite, mirror
  disabled, and a mirror failure being counted + logged without raising.
* `tests/integration/test_engine_pipeline.py` (21) — the real CLI on seeded ticks:
  bar counts per timeframe, the first bar's OHLCV, a second run replacing instead
  of duplicating (no duplicate primary keys), coverage rows with the session
  expectation, Parquet contents, out-of-session ticks never reaching a candle,
  flagged-only contracts, a weekend `--date`, `--all` across two days, `--dry-run`
  writing nothing (and explaining itself without a database), `--json`, bad
  `--date`/`--timeframes`/disabled timeframe → exit 2, no ticks → exit 0, a
  patched per-contract failure → exit 1 `PARTIAL`, and the run manifest being
  recorded after the database is released.

One real bug was found by these tests and fixed: `quality_counts(symbols=...)` was
called with a `None` symbol list when `--symbol` was not given (`list(None)`).

Manual end-to-end check (synthetic ticks, scratch directory): 3 contracts × 20
minutes produced `1m=60, 3m=21, 5m=12, 15m=6`; `99` candles written and `99`
replaced on a second run with zero duplicates; `12` `GAP_FILLED` coverage rows
(`expected_rows=375`, `coverage_ratio=0.0533`, largest gap `21300s`); four Parquet
files with a `TIMESTAMP WITH TIME ZONE` timestamp column and no `.tmp` leftovers.

Also verified: the Phase 2 capture suite still passes **unchanged** after the
coverage refactor (`CoverageTracker`/`StreamStats`/`build_coverage_row` moved to
`core.storage.coverage` and are re-exported by the capture module).

---

## 4. Not verified (be explicit)

* **Nothing here has run on real captured ticks.** Every test uses synthetic
  ticks; live capture is still blocked on operator credentials (see the Phase 2
  report), so the engine has never seen a real session's volume/OI behaviour. The
  volume arithmetic is the part most likely to need adjusting against reality.
* **Kite's cumulative-volume semantics are taken from the payload contract, not
  from a measured session.** The reset rule (negative increment → 0 + counter) is
  a documented choice, not an observed one at the time of writing.
* **The IV model has never been checked against an exchange-published IV.** There is no
  such series in the database (the feed does not carry one), so the model is verified
  against its own algebra - put-call parity, finite-difference greeks, an IV round trip -
  and not against the market. A real session's IV surface is the first thing to eyeball
  after the first live capture.
* **Half-day and special sessions are handled by construction, not by an observed
  example.** `session_window(day)` drives both the bar count and the tick window,
  and the special-session list in `config/calendar.yaml` is still provisional.
* **Multi-day backfills have only been exercised on two synthetic days.** The
  per-day loop is straightforward, but a long `--all` run (hundreds of days,
  millions of ticks) is untested for wall-clock time and memory.
* **The Parquet mirror has never been written concurrently.** The rename is atomic
  on one volume; a mirror directory on a different volume (or a reader holding the
  file open on Windows) is untested.

---

## 5. How to use it

```bash
# what would be built (writes nothing)
uv run python -m apps.anjaneeya_engine.run --mode build-candles --dry-run

# the most recent day with ticks, or one specific day
uv run python -m apps.anjaneeya_engine.run --mode build-candles
uv run python -m apps.anjaneeya_engine.run --mode build-candles --date 2026-09-29

# narrowed, or everything that has ticks (machine-readable)
uv run python -m apps.anjaneeya_engine.run --mode build-candles --symbol NIFTY01OCT2625000CE --timeframes 1m,5m
uv run python -m apps.anjaneeya_engine.run --mode build-candles --all --json
```

```sql
-- how complete was a day, per contract and timeframe?
SELECT tradingsymbol, timeframe, actual_rows, expected_rows, coverage_ratio, quality_flag
FROM data_coverage WHERE trading_date = DATE '2026-09-29'
ORDER BY tradingsymbol, timeframe;

-- the afternoon's 15m closes
SELECT timestamp, close, volume FROM candles
WHERE timeframe = '15m' AND tradingsymbol = 'NIFTY01OCT2625000CE'
  AND timestamp >= TIMESTAMPTZ '2026-09-29 13:00:00+05:30'
ORDER BY timestamp;
```

---

## 6. What Stage 3.2 can rely on

* `read_candles(connection, start=..., end=..., symbols=..., timeframes=...)`
  returns exactly the stored contract, ordered `(symbol, timeframe, timestamp)`.
* Bars are complete, in order and honest: no synthetic rows, no hidden gaps, and
  no duplicate `(timestamp, timeframe, tradingsymbol)` - it is the primary key.
* The 1m layer is the single source for every higher timeframe, so an indicator
  written once against the 1m frame is correct for all of them.
* `iv` and the greeks are `NULL` until `--mode enrich-greeks` has run for a scope, and
  `enrich-greeks` never touches anything but those seven columns - so the enrichment is
  additive, exactly as this stage assumed it would be. (Note for a later reader: the
  assumption held, but the *alignment* it needs - "which NIFTY level belongs to this
  bar" - turned out to be a shared rule rather than a local one; see §8.)

---

## 7. Stage 3.2 progress (indicators and price-action engines)

Stage 3.2 is being landed in small, individually tested pieces. Four are in:

**7a. Indicator library (`core/indicators/`, 63 tests).** Pure, past-only,
`NaN`-when-undefined functions: SMA/EMA (EMA publishes only once `period`
observations exist), MA-stack alignment, true range and ATR (Wilder's recursive
mean with the classic SMA seed, and the simple variant), raw and ATR-normalised
slope, ATR-normalised spread and spread velocity, direction and trend
persistence, and the basic statistics (rolling mean/std, z-score, percentage
change, volatility, ATR-normalised momentum, running extremes). A new
`momentum.period` baseline was added to `config/indicators.yaml`.

**7b. Price-action engines (`core/price_action/`, 49 tests).** Candle shape and
relationships plus named patterns (spec 11); swings with an explicit
confirmation delay (spec 15) - a swing is attached to the bar that could first
*have known* it, never to an earlier one; market-structure state and swing
distances; today's running levels and the previous session's levels with
ATR-normalised distances (spec 12).

**7c. Level interaction and support/resistance (`core/price_action/`, 34 further
tests).** The spec 13 state machine over four level series -
`event_prev_day_high_state`, `event_prev_day_low_state`, `event_today_high_state`
and `event_today_low_state`, spelled as `core/storage/schema.sql` spells them -
plus its eleven per-bar event flags per level and a timestamped event log; and the
spec 14 candidate levels (previous-day OHLC, today's open, today's running
extremes, the `max_swings` most recent confirmed swings) with their interaction
counts and blended `support_strength`/`resistance_strength`.

Design points worth recording, because they are the ones a reader would otherwise
have to reverse-engineer:

* **Both spec 13 examples are tests.** `149 -> 151 -> 152 -> 151 -> 153` against a
  previous-day high of 150 reads `APPROACHING, CLOSED_ABOVE, HELD_ABOVE,
  HELD_ABOVE, RETESTED, HELD_ABOVE`; `149 -> 151 -> 148 -> 147` reads
  `APPROACHING, CLOSED_ABOVE, FAILED_BREAKOUT`, and the failure fires exactly once.
* **A break is only a failure if it never held.** `min_closes_to_hold` closes
  beyond the level make it `HELD_*`; losing a held level is `CLOSED_BELOW`/
  `CLOSED_ABOVE` (a genuine reversal), never a "failed breakout". This is the one
  rule that separates "the level rejected the move" from "the level broke and
  turned".
* **The level is the running extreme *before* the bar.** Compared against its own
  running high a bar could only ever be at or below the level, which would make
  the state meaningless; the prior value is the only one a bar can cross, so
  "this bar made a new high" is `CLOSED_ABOVE`/`HELD_ABOVE` and "it failed to" is
  `FAILED_BREAKOUT`.
* **A level's life is bounded.** Previous-day levels restart at the first bar of
  each session (yesterday's hold never leaks into today), while today's running
  level simply moves with price; the bar's own **open** stands in for the missing
  previous close on the first bar of a life. `NULL` state while the level or the
  ATR is unknown - never a fabricated `NOT_TESTED`.
* **The state column is a summary, the flags are the record.** Support/resistance
  counts are cumulative sums of those same flags, so a `touch_count` and the
  `TOUCHED` state can never disagree.
* **Strength normalisations are bounded and assumption-free.** `n / (n + 1)` for
  touches/rejections/holds, `1 / (1 + bars_since_touch)` for recency and
  `1 / (1 + |distance_atr|)` for proximity, blended with the configured weights
  (which must sum to 1, so the score is in `[0, 1]`). No saturation constant was
  invented.
* **The *nearest* level in each role becomes the feature, not the strongest.** The
  candidate at or below the close is the support being stood on and the one above
  is the resistance overhead (ties in distance go to the better-tested level), and
  its own counts are reported beside its strength. Ranking by score alone was
  measured and rejected: the oldest, most-touched swing level wins nearly every
  bar, so `support_strength` sat at ~0.9 permanently and carried no information
  about the current bar.
* **`reaction_strength` is past-only.** It is the largest excursion away from the
  level *since the level was first touched, using bars up to this one*. How far
  price travels after a signal is a label; labels live in `core.labels`, never in
  a feature.
* **Counts are cumulative over a level's life, and that is visible in the data.**
  On 60 synthetic days (16.5k bars) the winning candidate's touch count reaches
  the tens and the strength score settles around 0.87 ± 0.10; a recency-weighted
  count or a decayed score would be a *new, explicitly configured* column, not a
  silent change to these. Cost measured on the same frame: ~0.9 s for
  `level_events` plus `level_event_flags` and ~3.2 s for
  `support_resistance_features` over 27 candidates, so a full-history rebuild is a
  per-instrument, per-timeframe batch job rather than a per-bar one.

**7d. NIFTY and option trend engines (`core/nifty/`, `core/options/`, 99 tests).**
`core/nifty/trend_score.py` implements spec 8's per-average measurements (value,
which side of it price closed on, raw slope, ATR-normalised slope, distance in ATR)
as `ma_metrics`, and spec 9's collapse of them into `nifty_fast_score`,
`nifty_medium_score`, `nifty_long_score`, `nifty_composite_trend_score`,
`nifty_trend_acceleration` and `nifty_trend_persistence`. Each role scores the mean
of the evidence families *it can express* - position
`mean tanh((close - MA) / ATR)`, slope `mean tanh(normalized_slope)`, and stack
`ma_alignment` (a single-average role has no stack and simply does not vote on it) -
and the composite is the weight-renormalised mean of the roles that are defined.
`core/nifty/regime.py` maps that composite onto the seven spec 9 states using the
configured thresholds. `core/options/option_trend.py` measures **one contract's own
short stack** (spec 10 - the NIFTY combination never touches an option) and
publishes the ten `option_*` columns.

Design decisions worth recording:

* **`tanh` is the only scale choice in either score.** It maps an ATR-normalised
  quantity into `(-1, +1)` monotonically, so the sign is the direction and the
  magnitude is the conviction, and no "saturation constant" had to be invented,
  configured or explained.
* **The composite renormalises the weights** across the roles that exist at that
  bar. Without it spec 8's combination 3 (no medium average) could never produce a
  composite, and neither could combination 1's long role until SMA 2056 has 2056
  bars (~5.5 sessions of 1m bars). The tests caught the naive version of this: a
  plain weighted sum propagates one role's `NULL` into *every* bar, so the whole
  column stayed empty. `NaN` in a role now means "not counted", not "contagious".
* **`combination_1` is the combination the feature row carries.** Combinations 2
  and 3 are computed alongside it by `combination_scores` so the three can be
  compared over the same bars; promoting a different one changes the meaning of six
  persisted columns, so it is a feature-version decision, not a parameter tweak.
* **The option spread pair is named in the config, not inferred.**
  `option_ema20_ema200_spread` is a schema constant (spec 18), but the shipped
  option stack also carries EMA 8 low/high, so "fastest minus slowest" would have
  meant EMA 8 − EMA 200 while the column said otherwise. A new optional
  `option_trend.spread_pair: {fast, slow}` block in `config/indicators.yaml` names
  the two averages (EMA 20 close and EMA 200 close); the config loader validates
  that both exist and that `fast` is the shorter one, and the engine falls back to
  fastest-versus-slowest only when a config names no pair.
* **Insufficient history is three behaviours behind one flag.** `flag` (the default)
  publishes the score from the averages that exist, `skip` withholds every
  average-derived column, and `shorter_window` substitutes the longest window that
  exists (every bar so far), which is still flagged because a 40-bar "EMA 200" is an
  approximation. `option_insufficient_history` says the same thing under all three:
  fewer than `min_candles_for_full_set` bars seen, or some average still undefined.
  Each policy is tested on the same frame, so they cannot collapse into one another.
* **The engines take a resolved `IndicatorsConfig` plus a shared ATR series**, the
  same shape as the price-action engines (config section plus ATR): the caller
  computes ATR once per contract and timeframe, so every distance in the feature row
  is measured against one number and no engine silently recomputes it.
* **Regime boundaries are inclusive on the bullish side and exclusive on the
  bearish side**, which is what makes `±threshold` land on the two matching states
  (`+0.25` is `BULLISH` and `-0.25` is `BEARISH`, not one state and one neighbour).
  `NEUTRAL` is therefore a band of `|score| < bullish_developing`, and `NULL` - not
  `NEUTRAL` - is what a missing composite becomes.

The research tables stay out of storage on purpose: `ma_metrics` and
`combination_metrics` publish ~50 columns per bar (per-average values, alignments,
the fastest/slowest spread and its velocity, each role's score) for reports and
tests, while the feature row keeps six NIFTY numbers, one regime label and ten
option numbers. Storing every intermediate would multiply the table for numbers
that can be recomputed from the candles at any time.

Cost measured on 16.5k synthetic 1m bars (shipped configuration, one ATR shared):
`trend_features` 0.08 s, `regime_features` <0.01 s, `combination_metrics` 0.11 s,
`combination_scores` (all three combinations) 0.16 s, `option_metrics` 0.11 s per
contract. Unlike the level/S-R walk these are cheap enough to run per contract per
timeframe inside `build-features`; the one requirement they place on the builder is
that its **lookback window must be at least the longest configured period plus the
slope lookback** (SMA 2056 + k on the shipped config, i.e. ~2060 bars ≈ 5.5 sessions
of 1m data), otherwise `nifty_long_score` is `NULL` at the start of every window and
the composite is a two-role reading there. The composite is still published in that
case - deliberately, see the renormalisation note above - and the `NULL` long score
in the same row is what says so.

Conventions fixed here, because the rest of the feature work depends on them:

* every `distance_*` column is **in ATR units** (`(price - level) / atr`), and the
  raw level is stored in the same row, so nothing is lost;
* `candle_gap` is in price points, `candle_engulf_ratio` is signed
  (`+` bullish engulfing, `-` bearish), and
  `candle_close_position_within_prev_range` is deliberately **not clipped** - a
  close through the previous high is a real gap and must read `> 1`;
* `swing_label` is one of `HH_HL` / `LH_LL` / `LH_HL` / `HH_LL` (bullish /
  bearish / contracting range / expanding transition), and `NULL` until both a
  high pair and a low pair are confirmed;
* boolean feature columns are **nullable**: `False` means "did not happen" and
  `NULL` means "the level or the ATR was not known", which are different claims.

**Still to do in Stage 3.2:** nothing on the feature path. `--mode build-features`
and `--mode validate` are implemented, so the engine has no planned mode left;
Stage 3.2c-3.2e closed out the assembly work that Stages 3.2a-3.2d prepared.

**7e. The unified feature row and its validation (`core/features/`).**
`core/features/feature_builder.py` turns one contract's candle window into exactly
`core.constants.FEATURE_COLUMNS` - the NIFTY group, the option group, candle
geometry, levels, level-events, structure, support/resistance, the option context and
spec 16's relative behaviour - and `core/features/feature_validation.py` implements
spec 19's checks as a report. `apps/anjaneeya_engine/run.py` gained
`--mode build-features` (candles -> feature rows, coverage, Parquet mirror) and
`--mode validate` (stored rows -> a JSON report per day), plus `FeatureWriter` in
`apps/anjaneeya_engine/storage_writer.py` and three readers
(`read_features`, `read_index_history`, `first_candle_timestamp`). 89 new tests:
53 for the two new modules, 9 for `FeatureWriter`, 7 for the new readers and 20
driving the whole pipeline through the CLI.

Design decisions worth recording:

* **The feature row is one row per `(timestamp, contract)` - so a build names exactly
  one timeframe.** The `features` table's primary key has no timeframe in it (spec
  18's own wording: "one feature row per (timestamp, contract)"), and two timeframes
  of the same contract share every bar's open time. `--timeframes 1m,5m` is therefore
  a configuration error rather than a silent choice, and `FeatureWriter`'s rebuild
  scope is the whole **contract-day**: rebuilding a day at another resolution replaces
  that day instead of colliding with it on the primary key. Switching the stored
  timeframe is a rebuild, which is exactly what delete-then-append gives.
* **The NIFTY block is joined by bar *close* instant, never by open time.** A 15m
  index bar opening 09:15 is only knowable at 09:30, so it must not reach a 1m
  contract bar at 09:20 - the natural `merge_asof` on open time is a one-bar
  look-ahead that no single-engine test would catch. `close_moments` is the shared
  definition (open + timeframe), the index frame is resampled with the *same*
  wall-clock bucket rule as the candle builder (asserted equal to
  `apps.anjaneeya_engine.candles.bucket_start` in the tests), and a unit test pins the
  15m-index/1m-candle case down. That test caught the bug while it was being written.
* **The lookback window is a number, not a habit.** `required_lookback_bars` is the
  longest configured average + the slope lookback + 1 (2062 1m bars on the shipped
  config) and the CLI adds one full session on top, so the level state machine sees
  the previous session. A shorter window still builds rows - `nifty_long_score` is
  simply `NULL` there, which is what says so.
* **`option_insufficient_history` is measured against the contract's own birth.** The
  option engine counts bars since the start of the frame it is given, so the CLI starts
  each contract at `max(window_start, its first stored bar)` via
  `first_candle_timestamp` - otherwise the flag would describe the window instead of
  the contract. It is the one column the no-look-ahead probe exempts, and the check
  says so in its own detail line.
* **The NIFTY index series is captured, not derived.** `index_history` holds a daily
  backfill plus a minute-level top-up, so `build-features` reads what the timeframe
  needs and resamples the 1m series when a coarser one is absent. A missing index
  series is reported (warning + run summary + a coverage note) and only nulls the
  seven `nifty_*` columns: the contract-side features are still perfectly computable.
* **Feature coverage is reported once per day, keyed by the underlying.**
  `data_coverage` is keyed `(trading_date, tradingsymbol, timeframe)` and
  `build-candles` already owns the per-contract rows for that day and timeframe, so a
  contract-level feature verdict would overwrite the candle verdict. The feature
  stream reports `NIFTY` with `expected_rows` summed over the contracts built and a
  note saying how many.
* **Spec 19's checks are a report, and the no-look-ahead check is a recomputation.**
  Nulls, infinities, duplicate keys, broken OHLC, negative volume, contract identity,
  ordering, gaps and leakage; `FAIL` means "this row is wrong", `WARN` means "this row
  is honest but incomplete" (an all-`NULL` column, a session gap), and a check that
  could not run says `SKIPPED` *with the reason* rather than passing quietly. The
  leakage check rebuilds a sample of bars from a window truncated at that bar and
  compares every column - the only form of the check that can actually catch an engine
  reading forward. A monkeypatched future-peeking engine is one of the tests.
* **Everything unresolved stays `NULL`.** `expected_option_move` and
  `option_residual` need a delta (Stage 3.3) and are `NULL` beside a published
  `actual_option_move`; a six-bar session has no 14-bar ATR; a day with no index
  history has no regime. `feature_rows` converts `NaN` to `None` on the way into the
  row contract and deliberately leaves `inf` alone so validation can flag it.
* **`core/features/` asserts its own contract at import time.** The group constants
  must reconstruct `FEATURE_COLUMNS` exactly - same names, same order - so a column
  that is computed but never stored (or stored but never computed) fails the moment
  anything imports the builder.

Cost measured through the real CLI (2 contracts x 2400 1m bars, 750 published rows,
the NIFTY index seeded): **5.2 s end to end**, and the great majority of it is the
support/resistance walk, which is linear in candidates x bars. A 200-contract day at
1m is therefore a few minutes of post-session batch work - which is why the builder
runs the walk once per contract, once per timeframe, and hands the same level/swing/
candidate frames to the event state machine and the strength blend instead of letting
each engine recompute them. `nifty_long_score` is `NULL` for the first ~70 published
bars of that run and populated for the remaining 680: the warm-up boundary is visible
in the data rather than hidden by it.

---

## 8. Stage 3.3 — IV and greeks (spec 16)

| Module | Responsibility |
| --- | --- |
| `core/options/greeks.py` (filled in) | The model: Black-Scholes-Merton with a continuous yield, `GreeksModel` (the configured assumptions, `from_config` / `describe`), `black_scholes_price`, `black_scholes_greeks`, `implied_volatility` (a bracketed, vectorised bisection), `time_to_expiry_years`, `option_price_series` and `enrich_option_frame`. Frames in, frames out; no I/O, no config reads, no knowledge of candles. |
| `apps/anjaneeya_engine/enrichment.py` (new) | The one decision the model cannot make: **which NIFTY level belongs to which bar**. `spot_for_bars` resamples the index series with the feature builder's own bucket rule and joins it on **close instants**; `enrich_frame` assembles the model's inputs (close moments, spot, expiry moments from the calendar, the calendar's `days_to_expiry`), calls the model and returns an `EnrichmentResult` carrying the frame *and* the counters (`solved_iv`, `unsolved_iv`, `missing_spot`, `max_price_error`). |
| `apps/anjaneeya_engine/storage_writer.py` (extended) | `GreeksWriter.apply_greeks` + `GreeksWriteStats`: stages the seven columns in a temp table and `UPDATE`s exactly those columns of existing rows, keyed `(timestamp, timeframe, tradingsymbol)`. It never inserts, never deletes, and shares the Parquet mirror with `CandleWriter`. |
| `core/storage/readers.py` (extended) | `trading_days_with_candles` / `symbols_with_candles` (the bar-side twins of the tick-side discovery queries, so the enrichment picks its work from *candles*) and `CANDLE_READ_COLUMNS` now serving the whole stored row. |
| `core/config_loader.py` + `config/indicators.yaml` | The `greeks:` section: `risk_free_rate`, `dividend_yield`, `price_field`, `year_days`, `min_time_to_expiry_minutes`, the IV bracket and its tolerance. Strict, documented, and optional so a Phase-1 file still loads. |
| `core/features/feature_builder.py` | `_relative_block`: spec 16's `expected_option_move` / `actual_option_move` / `option_residual` are finally real numbers; `close_moments` returns canonical UTC instants. |
| `core/features/feature_validation.py` | Two new checks (`greeks_sanity`, `relative_identity`) and the "empty by design" reasons now name `--mode enrich-greeks` instead of a future stage. |
| `apps/anjaneeya_engine/run.py` | `--mode enrich-greeks` (`--date/--all/--symbol/--timeframes/--dry-run/--json`), a per-day result, a human and a JSON summary, exit codes 0/1/2. |

### 8.1 The model, documented (spec 16's requirement)

Black-Scholes-Merton, European, cash-settled, with a continuous dividend yield:

    d1 = (ln(S/K) + (r - q + vol^2/2) * T) / (vol * sqrt(T))
    call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
    put  = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

* **`S`** — the NIFTY index level at the bar's **close instant** (`close_moments`), never
  at its open time.
* **`r`** — `greeks.risk_free_rate`, shipped as **0.065** (the rounded India 10-year
  G-Sec yield). A *configured constant*, deliberately not a fetched curve: an IV is only
  comparable across runs if the rate that produced it is pinned. Every run prints it.
* **`q`** — `greeks.dividend_yield`, shipped as **0.0**: dividends are ignored. NIFTY
  options are in practice priced off the futures, so a zero-dividend *spot* model absorbs
  the basis into the IV instead of modelling it. A documented limitation, not an accident.
* **`T`** — ACT/365 from the bar's close instant to `calendar.session_end(expiry)` (so a
  half day is handled), **floored at one session** (`min_time_to_expiry_minutes: 375`).
  The floor matters: without it the last bars of an expiry day divide by ~zero and report
  an IV that is really float noise. `days_to_expiry` is *not* this number - it stays the
  calendar's own value, exactly as the feature row publishes it.
* **Units** (fixed in the module docstring, because a silent convention change would
  invalidate every model trained on the columns): `iv` annualised as a fraction, `delta`
  per index point, `gamma` per point squared, **`vega` per 1 vol point**, **`theta` per
  calendar day**.
* **The IV solve is bracketed before it starts.** A price at or below the model value at
  `volatility_floor` (0.5%) or at or above the value at `volatility_ceiling` (500%) has no
  IV under this model and comes back `NULL` - not the bracket's edge, which is what a bare
  Newton-Raphson returns. The bisection is written out (vectorised, deterministic, no
  scalar root finder), so a contract's whole window is one pass.
* **`price_field`** selects the price (`close` by default - the same price
  `actual_option_move` uses). A traded close is not a mid, and on a thin strike the IV
  inherits that; `hl2`/`ohlc4` exist for that reason and record themselves in the summary.

### 8.2 Design decisions worth recording

* **The greeks live on the candle, not in the feature row.** The schema has carried the
  columns since Phase 1 and the timeframe merger already copied them, so `enrich-greeks`
  writes them where every consumer (features, labels, charts, SQL) can see them without
  rebuilding features. `build-features` then *reads* them - it still prices nothing.
* **Enrichment is an `UPDATE`, not a rebuild.** A bar may only be replaced by the pass
  that builds it from ticks. `GreeksWriter` stages the new values in a temp table and
  updates exactly `iv, delta, gamma, theta, vega, days_to_expiry, distance_from_atm` by
  primary key - so it cannot rewrite an OHLCV it never computed, cannot insert a row, and
  cannot leave a stale greek behind (an unsolved bar is written as NULL). The bar's
  `run_id` is deliberately **not** touched: the manifest of the enrichment run is the
  record of which run computed its IVs.
* **The spot is the index level at the bar's close instant, resampled with the feature
  builder's own rule.** `spot_for_bars` reuses `aggregate_index_frame` + `close_moments`,
  so "the NIFTY level a 15m candle was priced against" and "the NIFTY level a 15m feature
  row sees" are the same number by construction rather than by coincidence. That reuse is
  what exposed the two bugs below.
* **Two bugs, both found by reuse rather than by a new test** (and both now pinned):
  1. `run.py::_index_frame` **relabelled** the 1m index rows it fell back to as the
     requested timeframe, which stopped `aggregate_index_frame` from bucketing them - so a
     15m build read the NIFTY level from the *start* of each bucket while computing the
     close instant from its end. The relabel is gone; the frame keeps its own label.
  2. With the relabel gone the aggregation ran for real, and `merge_asof` refused the
     join: DuckDB hands frames back with a **pytz** zone while `aggregate_index_frame`
     localised with **zoneinfo**, and pandas will not merge two timezone-aware columns
     whose `tzinfo` objects differ - even when both say `Asia/Kolkata`. `close_moments`
     now returns canonical `datetime64[ns, UTC]` instants (a close *instant* is an
     instant), which merge regardless of where the rows came from.
* **`expected_option_move` uses the delta of the bar *before* the move.** The move happens
  during `(t-1, t]`, so `delta[t-1] * (nifty_close[t] - nifty_close[t-1])` is the
  expectation that was knowable when it started. Using `delta[t]` would let the move
  partly define its own expectation and would mechanically shrink `|option_residual|`
  whenever the bar itself moved the delta (a strike crossing ATM is the extreme case). A
  unit test pins the difference between the two conventions.
* **A third bug, found by a finite-difference test rather than by re-reading the formula:**
  theta's decay term was divided by `sigma*sqrt(T)` instead of `sqrt(T)` - a factor of
  1/sigma, i.e. a 6x error at 15% vol. Every greek is now checked against a central
  finite difference of `black_scholes_price`, in the stored units (per vol *point*, per
  calendar *day*), which is the only test that could have caught it.
* **Nothing is invented.** No index history → no IV (and a warning naming the fix). A
  price outside the model's band → `NULL`. A missing bar → counted in `unmatched`,
  logged, never inserted. `days_to_expiry` and `distance_from_atm` are still written
  without the model, because neither needs one.
* **The mirror is refreshed after the updates**, never before, so `data/candles/*.parquet`
  cannot keep serving pre-enrichment bars that DuckDB no longer agrees with.

### 8.3 Verification performed

`uv run pytest -q` — **1023 passed, 1 skipped** (the live capture smoke test skips itself
without credentials; the suite was 912 + 1 before this stage). `uv run ruff check .` —
clean. `schema_version` stays **1**.

New tests (89 functions, 111 collected cases):

* `tests/unit/test_option_greeks.py` (28) — put-call parity across five strikes and five
  volatilities, every greek against a central finite difference, the delta/gamma/vega
  identities between calls and puts, deep-wing limits, `NaN` propagation for every missing
  input, scalar/vector agreement, unknown option types (raise for a scalar, `NULL` for a
  row), IV round trips in both directions, the no-arbitrage/bracket refusals, `T`'s
  ACT/365 arithmetic, its floor, NaT propagation, timezone rejection and
  timezone-independence, and the frame-level enrichment (columns, alignment, DTE
  passthrough, non-option rows, empty frames, misaligned inputs, `price_field`).
* `tests/unit/test_engine_enrichment.py` (12) — the close-instant join (including the
  15m-index/1m-candle case and a gap), the 1m→5m resample, order/index preservation, and
  the pass itself: recovered volatility, the reported price error, the context columns, a
  missing NIFTY level, a put, and an empty window.
* `tests/unit/test_engine_writer.py` (+8) — `apply_greeks` updates only the greek columns
  (OHLCV and `run_id` untouched), stores `NULL` for an unsolved bar, ignores a vanished
  bar, is scoped to one contract and timeframe, is idempotent, leaves no temp table,
  rejects a frame that is not the enrichment output, and refreshes the Parquet mirror.
* `tests/unit/test_feature_builder.py` (+6) — the expected move is
  `delta[t-1] * nifty_move`, the previous-bar rule, the `NULL` block without a delta or
  without an index, a real `iv_change`, and the row *after* an unsolved delta.
* `tests/unit/test_feature_validation.py` (+7) — `greeks_sanity` (skip/warn/pass/fail,
  including a delta with the wrong sign for its option type, and the schema's own
  rejection of a negative IV) and `relative_identity` (pass/fail/warn).
* `tests/unit/test_storage_readers.py` (+3) — the candle read serves the greeks, and the
  two candle-side discovery queries (IST day grouping, window/timeframe filters).
* `tests/unit/test_config_loader.py` (+4) — the shipped assumptions are pinned, the
  section defaults when absent, a bracket that cannot be searched is rejected, and a typo
  in the section is rejected.
* `tests/integration/test_greeks_pipeline.py` (18) — the real CLI: IV recovery for a call
  and a put, the model/coverage summary lines, idempotence, JSON, the refreshed mirror,
  `--symbol`/`--timeframes`/`--dry-run`/`--date`/`--json`, a weekend, a bad date, a
  database with no candles, no index history (all `NULL` + why), a price with no IV
  (counted), a patched per-contract failure (exit 1 `PARTIAL`, the other contract still
  enriched), the closed loop into `build-features` (`expected == delta * 4.0` against the
  fixture's one-point-per-minute index, `residual == actual - expected`), a `validate`
  report whose greek checks pass, and a rebuilt scope losing its greeks.

Manual end-to-end check (synthetic ticks, scratch directory; 2 contracts × 375 minutes ×
4 timeframes, the NIFTY index seeded): `build-candles` **4.2 s** (1200 bars written),
`enrich-greeks` **1.2-1.9 s** (**1200/1200** bars got an IV - all four timeframes),
`build-features` **4.6 s** (750 rows), `validate` **1.5 s**. In the feature rows `iv` is
present on all 750, `expected_option_move` on 748 (the two missing are each contract's
first published bar, which has no previous bar in the window), and `iv`, `delta` and
`option_residual` are mutually consistent on every row. The IV solve is therefore roughly
**1-1.6 ms per bar** - about a quarter of what the feature build that consumes it costs,
which is why the enrichment is a separate pass over a whole day.

### 8.4 How to use it

```bash
# which contracts, how many bars, and the model's assumptions (writes nothing)
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --dry-run

# price the most recent day that has candles (every configured timeframe)
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks

# one day / one contract / a subset of timeframes / machine-readable summary
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --date 2026-09-29
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks --symbol NIFTY01OCT2625000CE --json

# the documented order of operations
uv run python -m apps.anjaneeya_engine.run --mode build-candles
uv run python -m apps.anjaneeya_engine.run --mode enrich-greeks
uv run python -m apps.anjaneeya_engine.run --mode build-features
uv run python -m apps.anjaneeya_engine.run --mode validate
```

```sql
-- how much of a day's greeks are actually solved, per contract and timeframe?
SELECT tradingsymbol, timeframe, count(*) AS bars, count(iv) AS with_iv,
       round(avg(iv), 4) AS mean_iv, round(avg(delta), 3) AS mean_delta
FROM candles WHERE timestamp >= TIMESTAMPTZ '2026-09-29 09:15:00+05:30'
  AND timestamp <  TIMESTAMPTZ '2026-09-29 15:30:00+05:30'
GROUP BY 1, 2 ORDER BY 1, 2;

-- what did the model fail to explain?
SELECT tradingsymbol, timestamp, actual_option_move, expected_option_move, option_residual
FROM features WHERE option_residual IS NOT NULL
ORDER BY abs(option_residual) DESC LIMIT 20;
```

**Still to do in Phase 3:** nothing on the feature path - and, since Stage 3.4b, nothing on
the label path either: spec 16 is complete, and spec 20's remaining consumers (the ML
dataset builder, the walk-forward splitter and the model stages that read these columns)
belong to Phase 4.

---

## 9. Stage 3.4a — the option universe (spec 17)

Spec 17 is a *selection*: which expiries trade, which strikes sit around at-the-money,
and how the reference price becomes a strike. Until this stage that policy existed only
inside ANJANEYA-CAPTURE (as `atm_strike_for` / `select_expiries` /
`build_option_universe`), which meant the engine and the AI app had no way to ask what
the universe *was* for a given day, and `atm.reference: futures` was a config line the
code silently ignored (the ATM was always snapped from the spot price).

The policy now lives in `core/options/universe.py` as pure functions of
(configuration, one reference price, one trading day, the market calendar), the capture
app keeps only the half that needs the instrument dump, and the engine gained a mode
that answers spec 17's question against the **stored** data.

### 9.1 What was built

| Module | Responsibility |
| --- | --- |
| `core/options/universe.py` (implemented) | Spec 17's selection policy: `atm_strike_for` (ATM snapping), `strike_ladder` (the configurable offsets), `select_expiries` (+ `current_monthly_expiry`/`next_monthly_expiry`), `select_contracts`, `render_symbol`/`template_pattern` (the symbol cross-check), and `build_selection(...)` -> `OptionSelection` (the ATM anchor, the ladder, the expiries and one `ContractSpec` per `(expiry, strike, side)`). No tokens, no symbols, no I/O. |
| `core/options/__init__.py` | Re-exports the new vocabulary under `core.options`. |
| `apps/anjaneeya_capture/instruments.py` | Imports and **re-exports** the moved policy (capture's public surface is unchanged, its tests untouched), and now owns only the dump-dependent part: `resolve_atm_reference` (the instrument whose price places the ATM, per `atm.reference`), `futures_contract` (the configured futures month) and `build_option_universe` (attributes -> `instrument_token`, `missing` reporting). `OptionUniverse` gained `atm_anchor` / `atm_anchor_token` and reports the anchor in `describe()`. |
| `apps/anjaneeya_capture/session.py` | The live ATM anchor is now the **configured** reference: the stream's `reference_token` (and the intraday re-centre) follows `atm.reference` instead of always being the spot index. |
| `core/storage/readers.py` | `read_instruments` (the general instrument-master view, filterable by exchange/underlying/expiry/type), `read_option_instruments` (the CE/PE view, a thin wrapper) and the two count queries `candle_counts` / `feature_counts`. |
| `apps/anjaneeya_engine/run.py` (+ `core/app_bootstrap.py`) | `--mode universe`: resolve the configured selection for a day from stored data and report every selected contract with its symbol, candle/feature counts and status. `--price` places the ATM by hand; `--json` for machines; the database is opened **read-only**. |
| `config/option_universe.yaml` | The `atm` block now documents what `reference: spot` vs `futures` actually does, and that `nearest` rounds an exact midpoint up. |

Supporting changes: `tests/support/engine.py` gained the universe-side fixtures
(`option_instrument_rows`, `futures_instrument_row`, `weekly_option_symbol`,
`seed_instruments`, `seed_features`); `DEFAULT_MODES` lists `universe`.

### 9.2 Design decisions worth recording

**The policy is separated from the dump, because only one of them can be recomputed.**
A resolved selection is a *specification* (`(expiry, strike, option_type)`), and it is a
pure function of the config, one price and the calendar - so the same day can be
re-derived from the config alone, years later, without the instrument dump that named
its tokens. Capture is the only place that can answer "and is it listed?", so the split
is exactly: core says what *should* be there, capture says what *is*.

**`nearest` rounds a midpoint up, and that is now stated rather than implied.** The
inherited docstring claimed ties resolve downwards; the code is
`floor(price / interval + 0.5)`, which rounds `25025` *up* to `25050` on a 50-point
ladder. The behaviour is pinned in a test (changing it would move the ATM of a live
universe by one strike - a config decision, not a refactor) and the docstring now says
what the code does.

**A strike offset off the grid is a hard error.** `strikes_offset: [0, 25]` with a
50-point interval would name a strike the exchange never lists; that used to be a
"strike not listed" hole discovered at 09:15, and is now a configuration error.

**The ladder is a menu, and the ATM stays visible as an anchor.** A config may select
`[-200, -100, +100, +200]` (a strangle); the selection still carries `atm_strike`, and
`includes_atm` states whether the ladder contains it, so a reader never has to infer
"the ATM was selected" from a list that does not contain it.

**`atm.reference: futures` is now real, and its failure mode is loud.** Capture prices
the futures contract (through `resolve_atm_reference`, single-sourced so the *price* and
the *contract it describes* can never be different months), watches that instrument's
ticks, and re-centres on it; a master holding no futures contract raises
`UniverseResolutionError` instead of quietly snapping the spot price. The engine's
`--mode universe` reads the futures contract's stored candles for the same reason.

**`--mode universe` writes nothing.** The connection is opened read-only, no coverage
row is written, and a test asserts every table's row count is unchanged after a run. It
is a *report*: the honest answer to "which contracts should exist, and what do we hold?"
- including the contracts that should exist and do not (`NOT_LISTED`, with strike, side
and expiry named).

**A partially-covered universe is not a failed run.** The report's exit code is `0`
whenever a report was produced; the facts live in the per-contract statuses and the
`complete` flag. Exit `1` is reserved for a day that could not be placed at all (no
stored reference price and no `--price`), which is a run that produced nothing.

### 9.3 Two bugs the tests found (both silent in production)

1. **A `DATE` column never compares equal to a `date`.** `read_option_instruments`
   returns DuckDB's own dtype (`datetime64[us]`), so `(Timestamp, strike, side)` never
   matched `(date, strike, side)` from the calendar and *every* selected contract was
   reported `NOT_LISTED`. The mode now normalises through `_as_date` (the same trap
   `apps/anjaneeya_engine/candles.py` documents for the candle row contract), and the
   reader documents the dtype it returns.
2. **`pandas.assign` on an empty selection invents rows.** Normalising the expiry column
   *after* filtering (`frame[mask].assign(expiry=...)`) broadcasts the Series over a
   zero-row frame and fabricates rows - which turned "the master holds no futures
   contract" into a phantom contract with a `NaN` symbol. The column is normalised before
   filtering now, and the reason string is asserted in a test.

### 9.4 Verification performed

`uv run pytest -q` — **1072 passed, 1 skipped** (the live capture smoke test skips
itself without credentials). `uv run ruff check .` — clean.

New tests (44):

* `tests/unit/test_option_universe.py` (29) — ATM snapping for all three roundings
  including the pinned midpoint rule, the ladder's ordering/dedup/grid validation, every
  expiry selector plus the monthly-rollover helpers, contract ordering, symbol rendering
  and the template pattern, the shipped config's full selection (12 contracts), a ladder
  that excludes the ATM, the reference label the caller supplies, every unusable
  reference price, and the JSON-safe `describe()`.
* `tests/unit/test_engine_universe.py` (15) — the real CLI over a seeded database: the
  full menu with per-contract candle/feature counts, `NO_CANDLES` and `NO_FEATURES`
  reported separately, a strike the master does not list (named), `--price` moving the
  ATM, a rejected `--price` (exit 2), a day with no reference price (exit 1 + the reason),
  the index series falling back to candles, the JSON shape, `--dry-run`, the read-only
  guarantee (every table unchanged), the futures reference (the price comes from the
  futures contract, the ATM snaps from it, `next_month` resolution, no-futures
  reporting), and the readers themselves.
* `tests/unit/test_capture_instruments.py` (+5) — the spot anchor is the index, the
  futures anchor is the futures contract (with the ATM snapped from *its* price), a
  futures reference without a listed contract is a hard error, the futures selector
  resolves the months the config names, and `resolve_atm_reference` honours the label it
  is given.

Manual check (synthetic database, scratch directory): seeded with a full 375-bar session
for all 12 selected contracts plus the NIFTY series (4 500 candles, 4 500 feature rows),
`--mode universe` runs in **0.39 s** and reports each contract's counts and status; it is
a reporting mode, not on the hot path of a build, and `--json` round-trips.

### 9.5 How to use it

```bash
# the configured menu for the latest day with candles, checked against the database
uv run python -m apps.anjaneeya_engine.run --mode universe

# a specific day / a hand-placed ATM / machine-readable
uv run python -m apps.anjaneeya_engine.run --mode universe --date 2026-09-29
uv run python -m apps.anjaneeya_engine.run --mode universe --price 25160
uv run python -m apps.anjaneeya_engine.run --mode universe --json

# what it prints, per day
#   reference      : spot @ 25,014.00 (index_history:NIFTY 50 @ 2026-09-29 09:17:00+05:30)
#   ATM strike     : 25,000 (interval 50, rounding nearest)
#   strike ladder  : 24,900, 25,000, 25,100   offsets: -100, +0, +100
#   expiries       : 2026-10-01, 2026-10-08
#   selected       : 12 contract(s)
#   coverage       : NO_CANDLES=8, OK=4
#   complete       : no
#   missing        : CE 25100 not listed for 2026-10-01

# NOTE (2026-09-30): `weekly_expiry.weekday` was corrected from Thursday to Tuesday
# against that day's NFO instrument dump, so a session captured from now on resolves the
# Tuesday expiries (2026-09-29, 2026-10-06) instead. The block above is the Thursday-era
# console kept as the record of what the 2026-09-29 candles hold.
```

```sql
-- how much of the configured universe do we actually hold, per contract?
SELECT tradingsymbol, count(*) AS bars FROM candles
WHERE timeframe = '1m'
  AND timestamp >= TIMESTAMPTZ '2026-09-29 09:15:00+05:30'
  AND timestamp <  TIMESTAMPTZ '2026-09-29 15:30:00+05:30'
GROUP BY 1 ORDER BY 1;
```


---

## 10. Stage 3.4b — labels (spec 20)

Spec 20 asks for the half of the dataset that must never be produced in the same pass as
the other half: a **label** is a function of the candles *after* a moment, a **feature**
is a function of the candles *up to* it. Computing them together is how look-ahead bias
enters a research pipeline, so labels got their own module, their own mode, their own
table, their own coverage stream and their own Parquet mirror - and nothing on the label
path reads a feature row (`core.schemas.FeatureRow` structurally refuses the label
columns, so the separation is enforced by the row contract, not by discipline).

Every session bar of every contract with candles gets exactly one row per configured
horizon. The numbers are stated in units rather than left to convention:

| Column | Unit | Meaning |
| --- | --- | --- |
| `future_return` | relative | `close[t+h] / close[t] - 1` - the one number that needs no scale |
| `future_max_up_move` | price | the best level the market **reached** inside the window, floored at `0` |
| `future_max_down_move` | price | the worst level reached, capped at `0` (a barrier rule compares against these) |
| `future_trend_change` | threshold units | the normalised forward move *minus* the normalised move into `t`: `+1` means "the horizon added one whole threshold of trend" |
| `label` | class | `UP_CONTINUATION` / `DOWN_CONTINUATION` / `RANGE` / `REVERSAL` |

The threshold comes from `config/model.yaml`: `normalization: atr` in the shipped config
(`0.5 * ATR(14)`, Wilder), with `percent` (`percent_threshold * close[t]`) and `fixed`
(raw price units) also implemented. Every input to a threshold is read **at or before
`t`**; nothing about the forward window can influence the scale used to judge it.

### 10.1 What was built

| Module | Responsibility |
| --- | --- |
| `core/labels/label_builder.py` (new) | Pure frame→frame logic: `LabelPolicy.from_config` (the three normalisations, the horizons, the ATR that scales them), `build_label_frame` (one row per `(timestamp, contract, horizon)`), `label_rows` (validated `LabelRow`s, so a bad frame fails before anything is written) and `LabelBuildError`. No I/O, no feature access. |
| `core/labels/__init__.py` (new) | Re-exports `LabelPolicy`, `build_label_frame`, `label_rows`, `REQUIRED_CANDLE_COLUMNS`, `LabelBuildError`. |
| `apps/anjaneeya_engine/storage_writer.py` | `LabelWriter` / `LabelWriteStats`: the same delete-then-append model as `FeatureWriter` (scope = one contract-day at *every* horizon), coverage upsert, atomic Parquet mirror to `data/labels/<day>.parquet`. |
| `core/storage/readers.py` | `read_labels` (+ `LABEL_READ_COLUMNS`): an optional window / contract / horizon scope, ordered `(tradingsymbol, timestamp, horizon_minutes)` - the shape a dataset builder wants when it pivots horizons into columns. Unlabelled rows come back too. |
| `core/config_loader.py` + `config/model.yaml` | `labels.trend_lookback` is now a validated, documented setting (the prior move a continuation has to continue). |
| `apps/anjaneeya_engine/run.py` (+ `core/app_bootstrap.py`) | `--mode build-labels`: warm-up window, per-contract build, coverage verdict, mirror, human/JSON summary. `DEFAULT_MODES` lists `build-labels`. |


### 10.2 Design decisions worth recording

**A label needs its whole forward window inside one session.** `t + horizon <=
session_end`, and every minute of the window must actually be present in the frame. An
option's overnight gap is not a five-minute move, so the last `max(horizon)` bars of a
session carry `NULL` label fields - and the row still exists, because a row that says
"no label here" is a fact a dataset builder can skip, while a missing row is a fact it
has to guess about. The same rule applies to a hole in the middle of the window
(contiguity is checked against the *expected* next timestamp, not merely the next row).

**A continuation has to continue something.** The prior move is `close[t] -
close[t - trend_lookback]`, read from that *exact* bar (a hole in the window would
otherwise turn a 5-bar move into an arbitrarily long one). When the prior bar is missing,
no class is published at all - guessing a direction there would invent evidence. The
prior comparison is **strict** (`prior_move > 0` for up, `< 0` for down), so a prior bar
that is exactly flat cannot be "continued" and falls through to `REVERSAL`. The unit
tests found the code and its docstring disagreeing here (`>= 0` / `<= 0`); the code was
corrected to the documented rule rather than the docstring to the code, so
`UP_CONTINUATION` now means *"it was already going up"* without exception.

**The excursions are floored in their own direction.** `future_max_up_move` is
`max(high[t+1..t+h]) - close[t]` clipped at zero, `future_max_down_move` is
`min(low[...]) - close[t]` clipped at zero: an adverse excursion is `0`, not a negative
"favourite move", and the sign of the column always states its direction. The extremes
come from the bars' own high/low, because a wick is a real level.

**The ATR warm-up is a number, not an exception.** A threshold that does not exist yet
(an ATR-14 after three bars is not an ATR-14) leaves the row's label fields `NULL` while
its *forward* numbers are still published: the window is real even when the scale is not.
In the integration fixture that is 13 bars per horizon (the ATR's first value is the mean
of the first 14 true ranges), asserted exactly rather than approximately.

**The label stream spells its own coverage key: `1m:labels`.** `data_coverage` identifies
a stream by `(tradingsymbol, timeframe)` alone, and `build-candles` already owns
`(day, contract, 1m)`. `build-features` avoided that collision by keying by the
*underlying*; labels are per *contract* like candles, so they take the other route the
codebase already uses (the tick stream says `TICK`): a distinct timeframe spelling that
keeps the bar timeframe legible. Without it a label verdict would silently replace the
candle verdict - the only record of how many bars a contract had.

**The mode reads candles, never features.** `build-labels` is `symbols_with_candles` +
`read_candles` + the builder: a day can be labelled before its features exist, or
without them entirely, which is what makes the two passes independently re-runnable. A

### 10.3 Verification performed

`uv run pytest -q` — **1112 passed, 1 skipped** (the live capture smoke test skips itself
without credentials). `uv run ruff check .` — clean.

New tests (24 + 16):

* `tests/unit/test_labels.py` (24) — the output shape and column order, the row contract,
  the forward-window semantics (the next bars, a window that leaves the session, a gap
  inside the window, bars before the session computed but never published), the three
  threshold normalisations (percent scaling with the close, fixed price units, the ATR
  warm-up carrying NULLs), an unknown normalisation refused, the classifications
  (continuation with the prior trend, reversal against it both ways, range below the
  threshold, a flat prior as the documented boundary, no class without a prior move),
  `future_trend_change` in threshold units, the input validation (a horizon that is not a
  whole number of bars, a missing candle column named in the message, naive timestamps,
  duplicate timestamps, an empty frame returning the contract), and the policy itself
  (the shipped config's horizons/ATR/lookback, the documented defaults without an
  indicators config).
* `tests/integration/test_label_pipeline.py` (16) — the real CLI over a seeded database
  (ticks → `build-candles` → `build-labels`): one row per bar and horizon for both
  contracts, `label_version`/`source` on every row, `future_return` re-derived from the
  stored candles by SQL, the excursion signs, the continuation class the fixture's rising
  session must produce, an unwindowable bar and a warm-up bar both `NULL` (with the exact
  count per horizon), the label coverage verdict sitting *next to* the candle verdict,
  the Parquet mirror and the absence of `.tmp`, feature rows untouched, a second run
  replacing instead of duplicating, `--symbol`, `--dry-run` (plan printed, nothing
  written), a day with no candles, `--json` and the run manifest.

Manual check (synthetic ticks, scratch directory, 2 contracts x 60 one-minute bars):
`build-labels` runs in **1.4 s** and writes 480 rows (256 labelled), 2 coverage verdicts
and one Parquet file; the scratch run is what caught the `>= 0` / `<= 0` classification
mismatch, because a flat-prior bar was being called a continuation.

### 10.4 How to use it

```bash
# what labels would be built, at which horizons, from which warm-up (writes nothing)
uv run python -m apps.anjaneeya_engine.run --mode build-labels --dry-run

# label the most recent day that has candles (every contract, one timeframe)
uv run python -m apps.anjaneeya_engine.run --mode build-labels

# one day / one contract / machine-readable
uv run python -m apps.anjaneeya_engine.run --mode build-labels --date 2026-09-29
uv run python -m apps.anjaneeya_engine.run --mode build-labels --symbol NIFTY01OCT2625000CE
uv run python -m apps.anjaneeya_engine.run --mode build-labels --json

# the documented order of operations: labels are a separate pass over the same candles
uv run python -m apps.anjaneeya_engine.run --mode build-candles
uv run python -m apps.anjaneeya_engine.run --mode build-features
uv run python -m apps.anjaneeya_engine.run --mode build-labels
```

```sql
-- how much of a day's label table actually carries a class, per horizon?
SELECT horizon_minutes, count(*) AS rows, count(label) AS labelled,
       count(DISTINCT label) AS classes
FROM labels
WHERE timestamp >= TIMESTAMPTZ '2026-09-29 09:15:00+05:30'
  AND timestamp <  TIMESTAMPTZ '2026-09-29 15:30:00+05:30'
GROUP BY 1 ORDER BY 1;

-- the class mix, and the excursions that go with it
SELECT label, count(*) AS n, round(avg(future_return), 5) AS mean_return,
       round(avg(future_max_up_move), 3) AS mean_up,
       round(avg(future_max_down_move), 3) AS mean_down
FROM labels WHERE label IS NOT NULL GROUP BY 1 ORDER BY n DESC;
```

**Still to do in Phase 3:** nothing on the label path - spec 20's remaining consumers
(the ML dataset builder, the walk-forward splitter and the model stages that read these
columns) belong to Phase 4. **No schema change**: `schema_version` stays **1**, because
the `labels` table has been in `core/storage/schema.sql` since Phase 1.

test seeds a feature row, runs the label mode and asserts the feature row is unchanged -
and that the label build creates no feature row.

