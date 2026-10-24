"""ANJANEYA-AI - machine-learning, backtesting and research reporting app.

Responsibilities (master spec sections 21-25, 30, 34)

* build leak-free datasets from features + labels (``--mode dataset``),
* train and evaluate models with purged walk-forward validation (``--mode train``),
* run the event-driven backtest with the full Indian cost model (``--mode backtest``),
* produce research reports, including honest negative results (``--mode report``),
* explain every prediction in plain structured form (``--mode explain``, spec 26),
* rank where the model's information came from (``--mode importance``, spec 27),
* draw the figures that make the numbers checkable by eye (``--mode charts``, spec 28),
* report the data-accumulation state and the retraining readiness (``--mode accumulation``, spec 29),
* turn a model's probabilities into explicit decisions (``--mode decide``, spec 24),
* enforce the data-accumulation gates (20/60/120/250 trading days).

``--mode dataset`` is the first AI mode: it joins the stored feature rows to the forward
labels of the *same* instant (spec 21), cuts them with spec 22's purged walk-forward
splitter, and writes one Parquet matrix plus one split manifest per configured horizon.
It reads the database and writes only under ``data/datasets/`` - no credentials, no
network, nothing else touched.

``--mode train`` consumes that same matrix in memory: for every fold it fits a LightGBM
multiclass model on the fold's surviving rows, scores the fold's validation window, scores
spec 21's baselines on exactly the same rows, and finally refits one model on every row
outside the holdout. It refuses to run at all below spec 4.3's exploratory gate (exit code
4) unless the operator forces it, labels every result it produces, and writes only under
``data/models/``.

``--mode backtest`` replays the stored bars against one trained model. Signals are read at a
candle's close, fills happen at the next bar's open (plus the configured latency), and every
closed trade - **losers included** - is written to a ledger with spec 23's full cost model
charged on both legs. It is exploratory by construction and says so: the gate labels the
result instead of refusing it, the counters state how many signals were rejected and why,
and the signal rule itself is printed as the threshold stand-in it is (spec 24's decision
layer lives in ``--mode decide``, and this replay does *not* use it). It writes only under
``data/backtests/`` and space in the ``trades`` table; no feature row, no label row, and no
schema change.

``--mode report`` answers spec 24 - the project's primary research output. It reads a stored
backtest ledger (``data/backtests/<stem>_ledger.parquet``, or ``--ledger PATH``) and the
stored NIFTY index bars, and writes the ten deliverables of spec 24 (time-of-day x option
side, x cycle phase, weekday, days-to-expiry, the regime interactions, holding time per
bucket, the ATR/range profile, the average intraday path, significance corrected for
multiple testing, and every PRELIMINARY label) as one JSON document, one Markdown report and
one flat CSV of cells under ``data/reports/research/``. It never recomputes a trade and never
opens the labels table: the ledger is the evidence, and every number carries its sample.

``--mode explain`` answers spec 26. It scores the stored feature rows with one trained model
(exactly as ``--mode backtest`` does) and then writes down, for the newest ``--rows``
predictions, what the model was looking at: the configured blocks of the stored feature row
(``model.explainability``) with the model's own class probabilities appended as the AI block,
a ``NULL`` feature rendered as unknown rather than 0, and the row's provenance beside it. It
writes one JSON, one Markdown and one CSV of lines under ``data/reports/explanations/``,
prints the same blocks to the console, and says in every artefact what an explanation is not:
a description of inputs, not a claim about why the market moved, and not a recommendation
(the decision layer lives in ``--mode decide``, and an explanation is not it).

``--mode importance`` answers spec 27. It reads one model artefact's own gain importance off
the booster, measures permutation importance on a *named* sample of the stored matrix -
``importance.repeats`` shuffles per feature, log loss by default, Student's t confidence
intervals, one seed - and reports whether the predictive information comes from the NIFTY
regime, the option's moving averages, candle behaviour, S/R, structure, IV, the greeks, OI,
the option residual, time of day, cycle phase or time to expiry (``model.importance.topics``,
checked against the feature contract at load). SHAP is optional in the strongest sense: it is
not a dependency, and ``importance.shap`` decides whether a missing package is a recorded
reason, a configuration error, or never asked about. A feature the sample cannot measure is
reported as not measured, never as a zero; a feature no source claims is reported as unmapped;
and it writes one JSON, one Markdown and one CSV under ``data/reports/importance/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from apps.anjaneeya_ai.accumulation_writer import AccumulationWriter, accumulation_stem
from apps.anjaneeya_ai.backtest_writer import BacktestWriter
from apps.anjaneeya_ai.chart_writer import ChartWriter
from apps.anjaneeya_ai.dataset_writer import DatasetWriter
from apps.anjaneeya_ai.decision_writer import DecisionWriter, decisions_stem
from apps.anjaneeya_ai.explain_writer import ExplanationWriter, model_stem
from apps.anjaneeya_ai.importance_writer import ImportanceWriter
from apps.anjaneeya_ai.model_writer import ModelWriter
from apps.anjaneeya_ai.report_writer import ResearchWriter
from core.accumulation import (
    AccumulationInputs,
    AccumulationPolicy,
    AccumulationReport,
    ModelRecord,
    build_accumulation_report,
    render_lines,
)
from core.app_bootstrap import (
    APP_AI,
    EXIT_CONFIG_ERROR,
    EXIT_FAILED,
    EXIT_GATE_REFUSED,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    AppContext,
    bootstrap_app,
    finalize_run,
    parse_args,
    render_status,
    status_report,
)
from core.backtest import (
    COST_MODEL_CAVEAT,
    Backtester,
    BacktestError,
    BacktestResult,
    CostError,
    CostModel,
    CostPolicy,
    ExecutionPolicy,
    SignalPolicy,
    TradeMetrics,
    build_segment_tables,
    trading_days,
)
from core.config_loader import ConfigError
from core.constants import get_timezone
from core.decision import (
    DECISION_CAVEATS,
    DecisionError,
    DecisionPolicy,
    DecisionReport,
    DecisionRow,
    build_decision_report,
)
from core.decision import (
    render_lines as render_decision_lines,
)
from core.ml import (
    AI_SECTION_TITLE,
    DECISION_LAYER_CAVEAT,
    EXPLANATION_CAVEAT,
    FRESHNESS_CAVEAT,
    UNMAPPED_TOPIC_KEY,
    Dataset,
    DatasetError,
    DatasetPolicy,
    DatasetSplit,
    EvaluationError,
    EvaluationPolicy,
    ExplanationBatch,
    ExplanationError,
    ExplanationPolicy,
    ImportanceError,
    ImportancePolicy,
    ImportanceReport,
    ModelArtefact,
    ModelError,
    ModelPredictor,
    SplitPolicy,
    TrainingError,
    TrainingPolicy,
    TrainingReport,
    build_dataset,
    build_explanations,
    build_importance_report,
    data_gate_status,
    gate_verdict,
    native_importance,
    permutation_importance,
    purged_walk_forward_splits,
    select_sample,
    shap_importance,
    train_and_evaluate,
)
from core.research import (
    PRELIMINARY_LABEL,
    ReportPolicy,
    ResearchError,
    ResearchReport,
    bucket_order,
    build_research_report,
    ledger_trades,
)
from core.research.profile import CYCLE_PHASE_ORDER
from core.storage import (
    ACCUMULATION_TABLES,
    SchemaVersionError,
    connect,
    read_accumulation_inputs,
    read_candles,
    read_features,
    read_index_history,
    read_labels,
)
from core.visualization import (
    ChartError,
    ChartPolicy,
    EventMarker,
    bar_chart,
    curve_chart,
    feature_bars,
    heatmap,
    importance_bars,
    nifty_price_data,
    option_price_data,
    price_chart,
    regime_bars,
    side_curves,
    time_of_day_cycle_heatmap,
    time_of_day_weekday_heatmap,
)

DESCRIPTION = (
    "ANJANEYA-AI: build leak-free datasets, train and validate NIFTY50 option behaviour "
    "models, backtest them, and report what the data says (research only)."
)

#: What each mode does. Phase numbers follow the delivered roadmap.
MODE_PLAN: dict[str, str] = {
    "dataset": "build the leak-free training matrix and its purged walk-forward splits (spec 21-22)",
    "train": "train with purged walk-forward validation, baselines and calibration (spec 21-22)",
    "backtest": "event-driven backtest with the Indian cost model (spec 23)",
    "report": "answer spec 24 from a stored ledger: buckets x side/phase, interactions, "
    "holding time, the volatility profile, the average intraday path, corrected significance "
    "and every PRELIMINARY label (spec 24-25)",
    "explain": "explain the newest predictions in plain structured form: the configured blocks "
    "of the stored feature row plus the model's own class probabilities (spec 26)",
    "importance": "measure where one model's information came from: its own gain importance, a "
    "permutation measurement with confidence intervals per feature and per source, and an "
    "optional SHAP attribution (spec 27)",
    "charts": "draw spec 28's validation charts: the time-of-day x cycle-phase and x day-of-week "
    "P&L heatmaps, the CE-vs-PE curves, regime-conditional performance, the feature-importance "
    "bars and the NIFTY / option price charts with their overlays",
    "accumulation": "report the continuous data-accumulation state: what was captured, coverage "
    "and gaps, sample sizes and readiness for retraining, week by week (spec 29)",
    "decide": "the decision layer (spec 24): turn one model's class probabilities on the stored "
    "feature rows into explicit decisions - ENTER_CE / ENTER_PE / NO_TRADE plus the reason, the "
    "size and the exit plan - under data/predictions/ (a record of what a rule would do, never "
    "an order and never advice)",
    "status": "report configuration, calendar, database and data-gate state (available now)",
}

#: Modes that are implemented today. Printed by ``_run_planned_mode`` so an operator is
#: never told "not implemented" without being told what *is* available.
AVAILABLE_MODES = (
    "--mode status, --mode dataset, --mode train, --mode backtest, --mode report, "
    "--mode explain, --mode importance, --mode charts, --mode accumulation, --mode decide"
)


#: Printed on every ``--mode dataset`` run. Spec 20's ``labels`` table is keyed
#: ``(timestamp, tradingsymbol, horizon_minutes)`` - it has no ``timeframe`` column - so a
#: matrix built from 5m bars cannot be distinguished from one built from the 1m rows at the
#: same instants. Saying so out loud is the honest alternative to pretending the join
#: verified something it cannot.
LABELS_TIMEFRAME_CAVEAT = (
    "stored label rows carry no timeframe column, so this matrix assumes they were built "
    "at {timeframe} (--mode build-labels)"
)



def _configure_ai_parser(parser: argparse.ArgumentParser) -> None:
    """Add the AI-only flags (which window, timeframe, horizon and contracts)."""
    parser.add_argument(
        "--from",
        dest="from_date",
        default=None,
        metavar="YYYY-MM-DD",
        help="dataset: first trading day in scope, inclusive (IST). Default: the oldest "
        "feature row stored.",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default=None,
        metavar="YYYY-MM-DD",
        help="dataset: last trading day in scope, inclusive (IST). Default: the newest "
        "feature row stored.",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        metavar="1m",
        help="dataset: which candles a training row is built from (must be one of "
        "settings.market.timeframes). Default: model.dataset.timeframe.",
    )
    parser.add_argument(
        "--horizon",
        dest="horizons",
        action="append",
        type=int,
        default=None,
        metavar="MINUTES",
        help="dataset: build only this label horizon (repeatable). Default: every horizon "
        "in model.horizons_minutes.",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        metavar="TRADINGSYMBOL",
        help="dataset: restrict the matrix to one contract (repeatable). Default: every "
        "contract that has feature rows in scope.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="train: override model.random_seed for this run (the model metadata records it, "
        "so a reported number can always be re-derived).",
    )
    parser.add_argument(
        "--score-holdout",
        dest="score_holdout",
        action="store_true",
        help="train: score the final model on the untouched holdout, once. Off by default: "
        "spec 22's holdout is reserved for the reporting stage, and a holdout you look at "
        "twice is not a holdout.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="train: train even below spec 4.3's exploratory gate. The results are still "
        "labelled BELOW_GATE, and no model produced this way is a production model.",
    )
    parser.add_argument(
        "--no-write",
        dest="no_write",
        action="store_true",
        help="train: train and report without writing the model artefacts under data/models/. "
        "backtest: run the replay without writing the ledger, the summary or the trades table. "
        "explain: print the explanations without writing them under data/reports/explanations/. "
        "importance: measure and print without writing under data/reports/importance/. "
        "charts: draw and report the figures without writing them under data/reports/charts/. "
        "accumulation: report the accumulation state without writing under "
        "data/reports/accumulation/.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="backtest: the model artefact to replay (the .txt written by --mode train; its "
        "_metadata.json must sit beside it). explain: the model whose predictions are "
        "explained. importance: the model whose inputs are ranked. Default: the newest "
        "artefact under data/models/ for the requested --horizon.",
    )
    parser.add_argument(
        "--min-probability",
        dest="min_probability",
        type=float,
        default=None,
        metavar="P",
        help="backtest/decide: override backtest.signal.min_probability (backtest) or "
        "decision.min_probability (decide) for this run (the artefact records the value that "
        "was actually used, so the number stays reproducible).",
    )
    parser.add_argument(
        "--min-margin",
        dest="min_margin",
        type=float,
        default=None,
        metavar="M",
        help="decide: override decision.min_margin for this run - the smallest lead the "
        "directional class must have over the runner-up class. Default: decision.min_margin.",
    )
    parser.add_argument(
        "--max-decisions-per-day",
        dest="max_decisions_per_day",
        type=int,
        default=None,
        metavar="N",
        help="decide: override decision.max_decisions_per_day for this run - the most entries "
        "one trading day may carry (the budget goes to the earliest signals). Default: "
        "decision.max_decisions_per_day (null = no throttle).",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        metavar="PATH",
        help="report: the trade ledger Parquet to analyse (the artefact --mode backtest "
        "wrote). Default: the newest *_ledger.parquet under data/backtests/.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        metavar="N",
        help="explain: how many of the newest predictions to explain. Default: "
        "model.explainability.rows.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        metavar="N",
        help="importance: how many times each feature is permuted (at least 2: one repeat has "
        "no spread to put an interval on). Default: model.importance.repeats.",
    )
    parser.add_argument(
        "--only",
        dest="only",
        action="append",
        default=None,
        metavar="CHART",
        help="charts: draw only the named chart(s) (repeatable). One of the chart families "
        "(heatmap, curves, regime, importance, price) or a chart name such as "
        "heatmap_time_of_day_day_of_week. Default: every chart the stored data can support.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        metavar="COLUMN",
        help="charts: which column of the importance artefact the feature-importance bars show "
        "(permutation_mean, the default, or native).",
    )
    parser.add_argument(
        "--importance",
        default=None,
        metavar="PATH",
        help="charts: the importance JSON whose features the bars chart (the artefact --mode "
        "importance wrote). Default: the newest *_importance.json under data/reports/importance/.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=None,
        metavar="N",
        help="accumulation: how many weeks back the captured-day series is bucketed (the "
        "report's window). Default: model.accumulation.window_weeks.",
    )


# ---------------------------------------------------------------------------
# --mode dataset (specs 21-22): the leak-free training matrix and its splits
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HorizonResult:
    """What one configured horizon's dataset build did."""

    horizon: int
    status: str
    dataset: Dataset | None = None
    split: DatasetSplit | None = None
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        payload: dict[str, Any] = {
            "horizon_minutes": self.horizon,
            "status": self.status,
            "artifacts": list(self.artifacts),
            "error": self.error,
        }
        if self.dataset is not None:
            payload["dataset"] = self.dataset.describe()
        if self.split is not None:
            payload["split"] = self.split.to_dict()
        return payload



def _day_from(value: str | None, *, flag: str) -> date | None:
    """Parse an ISO date flag, naming the flag when it is malformed."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ConfigError(f"{flag} {value!r} is not a valid ISO date (expected YYYY-MM-DD)") from None



def _resolve_timeframe(context: AppContext, value: str | None) -> str:
    """The timeframe a dataset is built from - the configured default, or ``--timeframe``.

    A timeframe the market configuration does not enable cannot have feature rows, so
    asking for one is a configuration error rather than an empty dataset.
    """
    configured = list(context.config.settings.market.timeframes)
    chosen = value or context.config.model.dataset.timeframe
    if chosen not in configured:
        raise ConfigError(
            f"dataset timeframe {chosen!r} is not enabled in settings.market.timeframes "
            f"({', '.join(configured)}); enable it there rather than training on bars the "
            "platform does not build"
        )
    return chosen



def _resolve_horizons(context: AppContext, values: Sequence[int] | None) -> tuple[int, ...]:
    """``--horizon`` narrowed against ``model.horizons_minutes`` (never beyond it)."""
    configured = tuple(int(value) for value in context.config.model.horizons_minutes)
    if not values:
        return configured
    requested = tuple(dict.fromkeys(int(value) for value in values))
    unknown = [horizon for horizon in requested if horizon not in configured]
    if unknown:
        raise ConfigError(
            f"horizon(s) {unknown} are not in model.horizons_minutes ({list(configured)}); add "
            "them there and rebuild the labels rather than training on a horizon nobody built"
        )
    return requested



def _window(context: AppContext, args: argparse.Namespace) -> tuple[datetime | None, datetime | None]:
    """The read window ``[--from 00:00, --to+1 00:00)``, or open bounds when unset."""
    tz = get_timezone(context.config.timezone_name)
    first = _day_from(args.from_date, flag="--from")
    last = _day_from(args.to_date, flag="--to")
    if first is not None and last is not None and last < first:
        raise ConfigError(f"--to ({last.isoformat()}) precedes --from ({first.isoformat()})")
    start = datetime(first.year, first.month, first.day, tzinfo=tz) if first else None
    end = datetime(last.year, last.month, last.day, tzinfo=tz) + timedelta(days=1) if last else None
    return start, end



def _gate_status(context: AppContext, days: int) -> Any:
    """Spec 4.3's gate verdict for ``days`` days covered by the matrix."""
    gates = context.config.model.data_gates
    return data_gate_status(
        days,
        exploratory=gates.exploratory_min_days,
        ml_training=gates.ml_training_min_days,
        meaningful_backtest=gates.meaningful_backtest_min_days,
        seasonality=gates.seasonality_min_days,
    )



def _build_horizon(
    context: AppContext,
    *,
    writer: DatasetWriter,
    features: Any,
    labels: Any,
    policy: DatasetPolicy,
    horizon: int,
    symbols: Sequence[str] | None,
) -> HorizonResult:
    """Build, split and write one horizon's dataset (a failure stays local to it)."""
    try:
        dataset = build_dataset(features, labels, policy=policy, horizon=horizon, symbols=symbols)
        split = purged_walk_forward_splits(
            dataset,
            policy=SplitPolicy.from_config(
                context.config,
                horizon_minutes=horizon,
                timezone_name=context.config.timezone_name,
            ),
        )
    except DatasetError as exc:
        context.logger.error(f"dataset for horizon {horizon}m failed: {exc}")
        return HorizonResult(horizon=horizon, status="FAILED", error=str(exc))

    if not dataset.rows:
        context.logger.warning(
            f"dataset for horizon {horizon}m has no labelled rows "
            f"({dataset.counts.unlabelled_rows} unlabelled, "
            f"{dataset.counts.features_without_label} feature rows without a label row)"
        )
        return HorizonResult(horizon=horizon, status="NO_ROWS", dataset=dataset, split=split)

    paths = writer.write(dataset=dataset, split=split)
    for path in paths:
        context.run_context.register_artifact(path)
    context.logger.info(
        f"dataset for horizon {horizon}m: {dataset.rows} row(s), {len(dataset.feature_columns)} "
        f"feature(s), {split.describe()}"
    )
    return HorizonResult(
        horizon=horizon,
        status="SUCCESS",
        dataset=dataset,
        split=split,
        artifacts=[path.name for path in paths],
    )



def _render_dataset_summary(
    *,
    policy: DatasetPolicy,
    results: Sequence[HorizonResult],
    gate: Any,
    writer_stats: dict[str, int],
    as_json: bool,
) -> str:
    """The human (or JSON) report ``--mode dataset`` prints."""
    if as_json:
        return json.dumps(
            {
                "mode": "dataset",
                "policy": policy.describe(),
                "gate": gate.to_dict(),
                "caveats": [LABELS_TIMEFRAME_CAVEAT.format(timeframe=policy.timeframe)],
                "artifacts": writer_stats,
                "horizons": [result.to_dict() for result in results],
            },
            indent=2,
            default=str,
        )

    lines: list[str] = [
        f"ANJANEYA-AI dataset - timeframe {policy.timeframe}, {len(results)} horizon(s)",
        f"  gate           : {gate.describe()}",
        f"  labels         : {LABELS_TIMEFRAME_CAVEAT.format(timeframe=policy.timeframe)}",
    ]
    for result in results:
        lines.append("")
        lines.append(f"  horizon {result.horizon}m : {result.status}")
        if result.dataset is None:
            lines.append(f"    failure        : {result.error}")
            continue
        dataset = result.dataset
        counts = dataset.counts
        filtered = ""
        if counts.other_timeframe_rows or counts.other_symbol_rows:
            filtered = (
                f", {counts.other_timeframe_rows} at another timeframe, "
                f"{counts.other_symbol_rows} for another contract"
            )
        lines.append(
            f"    rows           : {dataset.rows} labelled row(s) of {counts.feature_rows} "
            f"feature row(s) ({counts.unlabelled_rows} without a class, "
            f"{counts.features_without_label} without a label row, "
            f"{counts.labels_without_feature} label row(s) without a feature{filtered})"
        )
        lines.append(
            "    classes        : "
            + ", ".join(f"{name} {count}" for name, count in dataset.class_counts().items())
        )
        lines.append(
            f"    features       : {len(dataset.feature_columns)} "
            f"({len(dataset.numeric_columns)} numeric, {len(dataset.categorical_columns)} "
            f"categorical; {len(dataset.constant_columns())} constant here, "
            f"{len(dataset.missing_feature_columns)} missing from the frame)"
        )
        lines.append(
            f"    days/contracts : {len(dataset.days)} day(s), {len(dataset.symbols)} "
            f"contract(s), label version {dataset.label_version}"
        )
        if result.split is not None:
            lines.append(f"    split          : {result.split.describe()}")
            for fold in result.split.folds:
                lines.append(
                    f"      fold {fold.index}: train {fold.train_rows} row(s) "
                    f"({fold.removed_rows} removed: {fold.embargoed_rows} embargoed, "
                    f"{fold.purged_rows} purged) -> validate {fold.validate_rows} row(s)"
                )
        if result.artifacts:
            lines.append(f"    artifacts      : {', '.join(result.artifacts)}")
    if writer_stats.get("datasets"):
        lines.append("")
        lines.append(
            f"  written        : {writer_stats['datasets']} dataset(s), "
            f"{writer_stats['rows']} row(s), {writer_stats['files']} file(s)"
        )
    return "\n".join(lines)



def _dataset_plan_lines(
    context: AppContext,
    *,
    policy: DatasetPolicy,
    horizons: Sequence[int],
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    feature_rows: int,
    label_rows: int,
    gate: Any,
) -> list[str]:
    """What ``--mode dataset`` would do (``--dry-run``), without writing anything."""
    start, end = window
    split = SplitPolicy.from_config(
        context.config,
        horizon_minutes=max(horizons),
        timezone_name=context.config.timezone_name,
    )
    scope = (
        f"{start.isoformat()} .. {end.isoformat()}" if start or end else "every stored feature row"
    )
    return [
        f"ANJANEYA-AI dataset dry-run - timeframe {policy.timeframe}, "
        f"horizon(s) {', '.join(f'{horizon}m' for horizon in horizons)}",
        f"  scope          : {scope}"
        + (f", contract(s) {', '.join(symbols)}" if symbols else ""),
        f"  rows in scope  : {feature_rows} feature row(s), {label_rows} label row(s)",
        f"  classes        : {', '.join(policy.classes)} (label version {policy.label_version})",
        f"  split policy   : train {split.train_window_days} / validate "
        f"{split.validate_window_days} / step {split.step_days} day(s), embargo "
        f"{split.embargo_minutes}m, holdout {split.holdout_days} day(s), purge={split.purge} "
        f"(needs {split.days_required} captured day(s))",
        f"  gate           : {gate.describe()}",
        f"  labels         : {LABELS_TIMEFRAME_CAVEAT.format(timeframe=policy.timeframe)}",
        f"  would write    : 2 file(s) per horizon that has rows, under "
        f"{context.config.data_dir('datasets')}",
    ]



def _run_dataset(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode dataset``: spec 21's matrix, cut by spec 22's splitter, written to disk."""
    policy = DatasetPolicy.from_config(
        context.config, timeframe=_resolve_timeframe(context, args.timeframe)
    )
    horizons = _resolve_horizons(context, args.horizons)
    window = _window(context, args)
    symbols = list(args.symbols) if args.symbols else None
    start, end = window

    status, exit_code = "SUCCESS", EXIT_OK
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        features = read_features(
            connection,
            start=start,
            end=end,
            symbols=symbols,
            timeframes=[policy.timeframe],
        )
        labels = read_labels(connection, start=start, end=end, symbols=symbols)
        if features.empty:
            print(
                "ANJANEYA-AI: no stored feature rows in scope - build them first "
                "(--mode build-features) and the labels beside them (--mode build-labels).",
                file=sys.stderr,
            )
            return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

        # The gates need the days the matrix covers, which is only known after a build, so
        # the dry run reports the same verdict the real run would - from the feature rows.
        tz = get_timezone(context.config.timezone_name)
        scope_days = features["timestamp"].dt.tz_convert(tz).dt.date.nunique()
        gate = _gate_status(context, int(scope_days))

        if context.dry_run:
            print(
                "\n".join(
                    _dataset_plan_lines(
                        context,
                        policy=policy,
                        horizons=horizons,
                        window=window,
                        symbols=symbols,
                        feature_rows=int(len(features)),
                        label_rows=int(len(labels)),
                        gate=gate,
                    )
                )
            )
            return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

        writer = DatasetWriter(
            root=context.config.data_dir("datasets"),
            run_id=context.run_context.run_id,
            config_hash=context.run_context.config_hash,
            compression=context.config.settings.data.parquet_compression,
            mirror=True,
            logger=context.logger,
        )
        results = [
            _build_horizon(
                context,
                writer=writer,
                features=features,
                labels=labels,
                policy=policy,
                horizon=horizon,
                symbols=symbols,
            )
            for horizon in horizons
        ]
        gate = _gate_status(
            context,
            max((len(result.dataset.days) for result in results if result.dataset), default=0),
        )
        if not gate.ready_for_training:
            context.logger.warning(f"data gate: {gate.describe()}")
            context.run_context.add_note(f"data gate: {gate.describe()}")
        context.run_context.add_note(
            LABELS_TIMEFRAME_CAVEAT.format(timeframe=policy.timeframe)
        )
        print(
            _render_dataset_summary(
                policy=policy,
                results=results,
                gate=gate,
                writer_stats=writer.stats.to_dict(),
                as_json=args.as_json,
            )
        )
        context.run_context.data_coverage = {
            "days": gate.days,
            "stage": gate.stage,
            "horizons": {
                str(result.horizon): (result.dataset.rows if result.dataset else 0)
                for result in results
            },
        }
        if any(result.status == "FAILED" for result in results):
            status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"dataset failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-AI dataset failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)


# ---------------------------------------------------------------------------
# --mode train (specs 21-22): the walk-forward run and the model it leaves behind
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainResult:
    """What one horizon's walk-forward run did."""

    horizon: int
    status: str
    dataset: Dataset | None = None
    report: TrainingReport | None = None
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        payload: dict[str, Any] = {
            "horizon_minutes": self.horizon,
            "status": self.status,
            "artifacts": list(self.artifacts),
            "error": self.error,
        }
        if self.dataset is not None:
            payload["dataset"] = self.dataset.describe()
        if self.report is not None:
            payload["report"] = self.report.to_dict()
        return payload



@dataclass(slots=True)
class Prepared:
    """A horizon's matrix and split, built *before* anything is trained or refused.

    The gate and the dry-run plan both need to know how many days the matrix actually
    covers, and only a build can answer that - so the build happens first, once, and the
    same in-memory matrix is then trained on. Nothing is written until a run is allowed.
    """

    horizon: int
    dataset: Dataset | None = None
    split: DatasetSplit | None = None
    error: str | None = None



def _prepare_horizon(
    context: AppContext,
    *,
    features: Any,
    labels: Any,
    dataset_policy: DatasetPolicy,
    horizon: int,
    symbols: Sequence[str] | None,
) -> Prepared:
    """Build one horizon's matrix and split, keeping a failure local to that horizon."""
    try:
        dataset = build_dataset(
            features, labels, policy=dataset_policy, horizon=horizon, symbols=symbols
        )
        split = purged_walk_forward_splits(
            dataset,
            policy=SplitPolicy.from_config(
                context.config,
                horizon_minutes=horizon,
                timezone_name=context.config.timezone_name,
            ),
        )
    except DatasetError as exc:
        context.logger.error(f"dataset for horizon {horizon}m failed: {exc}")
        return Prepared(horizon=horizon, error=str(exc))
    return Prepared(horizon=horizon, dataset=dataset, split=split)



def _train_prepared(
    context: AppContext,
    *,
    prepared: Prepared,
    writer: ModelWriter,
    namer: DatasetWriter,
    training_policy: TrainingPolicy,
    evaluation_policy: EvaluationPolicy,
    baseline_names: Sequence[str],
    include_holdout: bool,
    write_models: bool,
    verdict: Any,
) -> TrainResult:
    """Run the walk-forward for one horizon and write the model it produced."""
    dataset, split = prepared.dataset, prepared.split
    if dataset is None or split is None:
        return TrainResult(horizon=prepared.horizon, status="FAILED", error=prepared.error)
    if not dataset.rows:
        context.logger.warning(
            f"horizon {prepared.horizon}m has no labelled rows "
            f"({dataset.counts.unlabelled_rows} unlabelled, "
            f"{dataset.counts.features_without_label} feature rows without a label row)"
        )
        return TrainResult(horizon=prepared.horizon, status="NO_ROWS", dataset=dataset)

    report = train_and_evaluate(
        dataset,
        split=split,
        policy=training_policy,
        evaluation=evaluation_policy,
        baseline_names=baseline_names,
        include_holdout=include_holdout,
        verdict=verdict,
    )
    for result in report.folds:
        context.logger.info(f"horizon {prepared.horizon}m {result.describe()}")
    for note in report.notes:
        context.logger.warning(f"horizon {prepared.horizon}m: {note}")
    if not report.ok:
        context.logger.warning(f"nothing was trained for horizon {prepared.horizon}m: {split.describe()}")
        return TrainResult(
            horizon=prepared.horizon, status="NOT_TRAINED", dataset=dataset, report=report
        )

    dataset_name = namer.dataset_name(dataset=dataset)
    paths = writer.write(
        dataset_name=dataset_name,
        report=report,
        dataset_description=dataset.describe(),
        dataset_artefact=f"{dataset_name}.parquet",
        extra={
            "labels_timeframe_caveat": LABELS_TIMEFRAME_CAVEAT.format(
                timeframe=dataset.timeframe
            ),
            "categorical_columns": list(dataset.categorical_columns),
            "dataset_days": len(dataset.days),
        },
    )
    for path in paths:
        context.run_context.register_artifact(path)
    if not paths and write_models:
        context.logger.warning(
            f"horizon {prepared.horizon}m trained but no model artefact was written"
        )
    if paths:
        context.logger.info(
            f"horizon {prepared.horizon}m: {report.describe()} -> {paths[0].name}"
        )
    return TrainResult(
        horizon=prepared.horizon,
        status="SUCCESS",
        dataset=dataset,
        report=report,
        artifacts=[path.name for path in paths],
    )



def _metric_line(metrics: Any) -> str:
    """A metric summary in a fixed, readable order (only what was measured)."""
    parts: list[str] = []
    for name in ("log_loss", "brier_score", "accuracy", "expected_calibration_error"):
        value = getattr(metrics, name, None)
        if value is not None:
            parts.append(f"{name.replace('_', ' ')} {value:.4f}")
    return ", ".join(parts) or "nothing measured"



def _summary_head(
    *,
    dataset_policy: DatasetPolicy,
    training_policy: TrainingPolicy,
    baseline_names: Sequence[str],
    results: Sequence[TrainResult],
    gate: Any,
    verdict: Any,
    include_holdout: bool,
    horizon_count: int | None = None,
) -> list[str]:
    """The header block of the ``--mode train`` report: what ran, and what it may claim."""
    policy = training_policy
    count = len(results) if horizon_count is None else int(horizon_count)
    return [
        f"ANJANEYA-AI train - timeframe {dataset_policy.timeframe}, {count} horizon(s), "
        f"algorithm {policy.algorithm}",
        f"  gate           : {gate.describe()}",
        f"  verdict        : {verdict.describe()}",
        (
            "  training       : "
            f"n_estimators {policy.n_estimators}, learning_rate {policy.learning_rate}, "
            f"num_leaves {policy.num_leaves}, min_child_samples {policy.min_child_samples}, "
            f"class_weight {policy.class_weight}, seed {policy.random_seed}, "
            f"early_stopping {'on' if policy.early_stopping else 'off'}"
        ),
        (
            "  baselines      : "
            + ", ".join(baseline_names)
            + " (each fitted on the training fold, so both sides are out-of-sample)"
        ),
        (
            "  holdout        : scored once by this run"
            if include_holdout
            else "  holdout        : left untouched (--score-holdout scores it once)"
        ),
        f"  labels         : {LABELS_TIMEFRAME_CAVEAT.format(timeframe=dataset_policy.timeframe)}",
    ]



def _render_train_summary(
    *,
    dataset_policy: DatasetPolicy,
    training_policy: TrainingPolicy,
    evaluation_policy: EvaluationPolicy,
    baseline_names: Sequence[str],
    results: Sequence[TrainResult],
    gate: Any,
    verdict: Any,
    writer_stats: dict[str, int],
    include_holdout: bool,
    as_json: bool,
) -> str:
    """The human (or JSON) report ``--mode train`` prints."""
    if as_json:
        return json.dumps(
            {
                "mode": "train",
                "dataset_policy": dataset_policy.describe(),
                "training": training_policy.describe(),
                "evaluation": evaluation_policy.describe(),
                "baselines": list(baseline_names),
                "gate": gate.to_dict(),
                "verdict": verdict.to_dict(),
                "score_holdout": include_holdout,
                "caveats": [LABELS_TIMEFRAME_CAVEAT.format(timeframe=dataset_policy.timeframe)],
                "artifacts": writer_stats,
                "horizons": [result.to_dict() for result in results],
            },
            indent=2,
            default=str,
        )

    lines = _summary_head(
        dataset_policy=dataset_policy,
        training_policy=training_policy,
        baseline_names=baseline_names,
        results=results,
        gate=gate,
        verdict=verdict,
        include_holdout=include_holdout,
    )
    for result in results:
        lines.extend(_summary_horizon(result))
    if writer_stats.get("models"):
        lines.append("")
        lines.append(
            f"  written        : {writer_stats['models']} model(s), {writer_stats['folds']} "
            f"fold(s), {writer_stats['rows']} trained row(s), {writer_stats['files']} file(s) "
            "under data/models/"
        )
    if writer_stats.get("failures"):
        lines.append(f"  write failures : {writer_stats['failures']} (see the log)")
    return "\n".join(lines)



def _summary_horizon(result: TrainResult) -> list[str]:
    """One horizon's block: what the matrix was, what the folds measured, what was written."""
    lines: list[str] = ["", f"  horizon {result.horizon}m : {result.status}"]
    if result.error:
        lines.append(f"    failure        : {result.error}")
    if result.dataset is not None:
        dataset = result.dataset
        lines.append(
            f"    rows           : {dataset.rows} labelled row(s), {len(dataset.days)} day(s), "
            f"{len(dataset.symbols)} contract(s), {len(dataset.feature_columns)} feature(s) "
            f"({len(dataset.numeric_columns)} numeric, {len(dataset.categorical_columns)} "
            "categorical)"
        )
        lines.append(
            "    classes        : "
            + ", ".join(f"{name} {count}" for name, count in dataset.class_counts().items())
        )
    report = result.report
    if report is None:
        return lines
    lines.append(f"    split          : {report.split.describe()}")
    lines.append(
        f"    folds          : {len(report.trained_folds)} trained, "
        f"{len(report.skipped_folds)} skipped"
    )
    for fold in report.folds:
        lines.append(f"      {fold.describe()}")
    if not report.ok:
        for note in report.notes:
            lines.append(f"    note           : {note}")
        return lines

    first = True
    for aggregate in report.aggregates.values():
        label = "    model          : " if first else f"    {'':<14} : "
        lines.append(label + aggregate.describe())
        first = False
    best = report.best_baseline()
    for name, metrics in report.baseline_aggregates.items():
        aggregate = metrics.get("log_loss")
        if aggregate is not None:
            mark = " <- best" if best is not None and name == best[0] else ""
            lines.append(f"    baseline {name:<14}: {aggregate.describe()}{mark}")
    model_loss = report.model_log_loss()
    if best is not None and model_loss is not None:
        every_fold = all(
            fold.comparison is not None and fold.comparison.beats_all_baselines
            for fold in report.trained_folds
        )
        if every_fold:
            lines.append(
                "    comparison     : the model beats every baseline in every trained fold "
                f"[{report.results_label}]"
            )
        else:
            direction = "better" if model_loss < best[1] else "worse"
            lines.append(
                f"    comparison     : {best[0]} is the bar ({best[1]:.4f}); the model is "
                f"{direction} by {abs(best[1] - model_loss):.4f} on average - the per-fold "
                "numbers above are the evidence, not this line"
            )
    if report.holdout is not None:
        lines.append(
            f"    holdout        : {report.holdout.rows} row(s) scored once "
            f"[{report.holdout.label}] - {_metric_line(report.holdout.model)}"
        )
    if report.final_model is not None:
        top = ", ".join(
            f"{name} {value:.1f}" for name, value in report.final_model.top_features(3)
        )
        lines.append(
            f"    final model    : trained on {report.final_model.train_rows} row(s) outside the "
            f"holdout over {len(report.final_model.feature_columns)} feature(s); top importance "
            f"{top}"
        )
    if result.artifacts:
        lines.append(f"    artifacts      : {', '.join(result.artifacts)}")
    for note in report.notes:
        lines.append(f"    note           : {note}")
    return lines



def _train_plan_lines(
    context: AppContext,
    *,
    dataset_policy: DatasetPolicy,
    training_policy: TrainingPolicy,
    baseline_names: Sequence[str],
    prepared: Sequence[Prepared],
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    gate: Any,
    verdict: Any,
    include_holdout: bool,
    write_models: bool,
) -> list[str]:
    """What ``--mode train --dry-run`` prints instead of training (and writes nothing)."""
    start, end = window
    lines = _summary_head(
        dataset_policy=dataset_policy,
        training_policy=training_policy,
        baseline_names=baseline_names,
        results=[],
        gate=gate,
        verdict=verdict,
        include_holdout=include_holdout,
        horizon_count=len(prepared),
    )
    lines[0] = lines[0].replace("ANJANEYA-AI train -", "ANJANEYA-AI train (dry run) -")
    lines.insert(1, f"  run_id         : {context.run_context.run_id}")
    lines.append(
        "  window         : "
        + (
            f"{start.date().isoformat()} .. {end.date().isoformat()}"
            if start is not None and end is not None
            else "open bounds (every stored row)"
        )
    )
    lines.append(
        "  contracts      : " + (", ".join(symbols) if symbols else "every contract with rows")
    )
    lines.append(
        "  horizons       : " + ", ".join(f"{item.horizon}m" for item in prepared)
    )
    lines.append(
        "  writes to      : " + (str(context.config.data_dir("models")) if write_models else "nothing (--no-write)")
    )
    for item in prepared:
        lines.append("")
        lines.append(f"  horizon {item.horizon}m")
        if item.error:
            lines.append(f"    failure        : {item.error}")
            continue
        dataset, split = item.dataset, item.split
        if dataset is None or split is None:
            continue
        lines.append(
            f"    rows           : {dataset.rows} labelled row(s), {len(dataset.days)} day(s), "
            f"{len(dataset.feature_columns)} feature(s)"
        )
        lines.append(f"    split          : {split.describe()}")
        lines.append(
            "    would write    : "
            + (
                f"{dataset.timeframe}_h{dataset.horizon}_<days>_{training_policy.algorithm}.txt + "
                "_metadata.json"
                if write_models
                else "nothing (--no-write)"
            )
        )
    return lines



def _resolve_ai_policies(
    context: AppContext, args: argparse.Namespace
) -> tuple[TrainingPolicy, EvaluationPolicy]:
    """The training and evaluation policies, with a policy error reported as a config error.

    ``algorithm: xgboost`` or a probability metric nobody implemented is a *configuration*
    problem, not a crash: it exits 2 with a message that says what is supported, exactly as
    a malformed YAML value would.
    """
    try:
        training_policy = TrainingPolicy.from_config(context.config, seed=args.seed)
        evaluation_policy = EvaluationPolicy.from_config(context.config)
    except (TrainingError, EvaluationError) as exc:
        raise ConfigError(str(exc)) from None
    return training_policy, evaluation_policy



def _run_train(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode train``: build the matrices, enforce the gate, walk forward, write the models."""
    dataset_policy = DatasetPolicy.from_config(
        context.config, timeframe=_resolve_timeframe(context, args.timeframe)
    )
    horizons = _resolve_horizons(context, args.horizons)
    window = _window(context, args)
    symbols = list(args.symbols) if args.symbols else None
    training_policy, evaluation_policy = _resolve_ai_policies(context, args)
    baseline_names = tuple(str(name) for name in context.config.model.evaluation.baselines)
    include_holdout = bool(getattr(args, "score_holdout", False))
    write_models = not bool(getattr(args, "no_write", False))
    forced = bool(getattr(args, "force", False))
    start, end = window

    status, exit_code = "SUCCESS", EXIT_OK
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        features = read_features(
            connection,
            start=start,
            end=end,
            symbols=symbols,
            timeframes=[dataset_policy.timeframe],
        )
        labels = read_labels(connection, start=start, end=end, symbols=symbols)
        if features.empty:
            print(
                "ANJANEYA-AI: no stored feature rows in scope - build them first "
                "(--mode build-features) and the labels beside them (--mode build-labels).",
                file=sys.stderr,
            )
            return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

        prepared = [
            _prepare_horizon(
                context,
                features=features,
                labels=labels,
                dataset_policy=dataset_policy,
                horizon=horizon,
                symbols=symbols,
            )
            for horizon in horizons
        ]
        gate = _gate_status(
            context,
            max((len(item.dataset.days) for item in prepared if item.dataset), default=0),
        )
        verdict = gate_verdict(gate.stage, days=gate.days, forced=forced)
        namer = DatasetWriter(
            root=context.config.data_dir("datasets"),
            run_id=context.run_context.run_id,
            mirror=False,
        )
        if context.dry_run:
            print(
                "\n".join(
                    _train_plan_lines(
                        context,
                        dataset_policy=dataset_policy,
                        training_policy=training_policy,
                        baseline_names=baseline_names,
                        prepared=prepared,
                        window=window,
                        symbols=symbols,
                        gate=gate,
                        verdict=verdict,
                        include_holdout=include_holdout,
                        write_models=write_models,
                    )
                )
            )
            return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

        if verdict.refused:
            print(
                f"ANJANEYA-AI: refusing to train - {verdict.reason}.\n"
                f"  captured       : {gate.days} day(s) covered by the matrix\n"
                f"  gate           : {gate.describe()}\n"
                f"  exploratory    : needs {gate.gates['exploratory_min_days']} day(s)\n"
                f"  production     : needs {gate.gates['ml_training_min_days']} day(s) (spec 4.3)\n"
                "  to train anyway: --force (the results stay labelled BELOW_GATE)",
                file=sys.stderr,
            )
            context.logger.warning(f"refusing to train: {verdict.reason}")
            context.run_context.add_note(f"refused by the data gate: {verdict.reason}")
            context.run_context.data_coverage = {"days": gate.days, "stage": gate.stage}
            return finalize_run(context, status="REFUSED", exit_code=EXIT_GATE_REFUSED)

        writer = ModelWriter(
            root=context.config.data_dir("models"),
            run_id=context.run_context.run_id,
            config_hash=context.run_context.config_hash,
            mirror=write_models,
            logger=context.logger,
        )
        results = [
            _train_prepared(
                context,
                prepared=item,
                writer=writer,
                namer=namer,
                training_policy=training_policy,
                evaluation_policy=evaluation_policy,
                baseline_names=baseline_names,
                include_holdout=include_holdout,
                write_models=write_models,
                verdict=verdict,
            )
            for item in prepared
        ]
        _record_train_caveats(context, gate=gate, verdict=verdict, dataset_policy=dataset_policy)
        print(
            _render_train_summary(
                dataset_policy=dataset_policy,
                training_policy=training_policy,
                evaluation_policy=evaluation_policy,
                baseline_names=baseline_names,
                results=results,
                gate=gate,
                verdict=verdict,
                writer_stats=writer.stats.to_dict(),
                include_holdout=include_holdout,
                as_json=args.as_json,
            )
        )
        context.run_context.data_coverage = {
            "days": gate.days,
            "stage": gate.stage,
            "results_label": verdict.results_label,
            "horizons": {
                str(result.horizon): {
                    "trained_folds": len(result.report.trained_folds) if result.report else 0,
                    "artifacts": list(result.artifacts),
                }
                for result in results
            },
        }
        if any(result.status == "FAILED" for result in results):
            status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"training failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-AI train failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)



def _record_train_caveats(
    context: AppContext, *, gate: Any, verdict: Any, dataset_policy: DatasetPolicy
) -> None:
    """Attach the caveats the gate and the label contract impose on every training run."""
    if not gate.ready_for_training:
        context.logger.warning(f"data gate: {gate.describe()}")
        context.run_context.add_note(f"data gate: {gate.describe()}")
    context.run_context.add_note(verdict.describe())
    if verdict.note:
        context.logger.warning(verdict.note)
        context.run_context.add_note(verdict.note)
    context.run_context.add_note(
        LABELS_TIMEFRAME_CAVEAT.format(timeframe=dataset_policy.timeframe)
    )








# ---------------------------------------------------------------------------
# --mode backtest (spec 23): replay the stored bars against one model
# ---------------------------------------------------------------------------


def _model_candidates(root: Path, horizon: int | None = None) -> list[Path]:
    """Model booster files under ``root``, newest first (optionally a single horizon's)."""
    if not root.is_dir():
        return []
    pattern = f"*_h{horizon}_*_lightgbm.txt" if horizon is not None else "*_lightgbm.txt"
    paths = [path for path in root.glob(pattern) if path.is_file()]
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)



def _metadata_horizon(path: Path) -> int | None:
    """The horizon recorded in a model's metadata, or ``None`` when it cannot be read."""
    metadata = path.with_name(path.stem + "_metadata.json")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    block = payload.get("model") or {}
    horizon = block.get("horizon_minutes")
    return int(horizon) if horizon is not None else None



def _resolve_model_artefact(context: AppContext, args: argparse.Namespace) -> ModelArtefact | None:
    """The model a mode works with: ``--model``, or the newest artefact for the horizon.

    Returns ``None`` when there is no artefact at all (the caller then reports NO_MODEL with
    the place it looked). Anything *wrong* with an artefact that exists is a configuration
    error instead: a model whose metadata is missing or out of sync with its booster cannot be
    scored at all, and guessing would be worse than refusing. One run uses one model, and a
    model is one horizon - a backtest replays it, an explanation describes it, an importance
    measurement ranks it - so asking for several horizons at once is refused for all three.
    """
    if args.model:
        path = Path(args.model)
        if not path.is_file():
            raise ConfigError(f"--model {path} does not exist")
        try:
            artefact = ModelArtefact.load(path)
        except ModelError as exc:
            raise ConfigError(str(exc)) from None
        _check_horizon_match(artefact, args)
        return artefact

    requested = tuple(dict.fromkeys(int(value) for value in (args.horizons or ())))
    if len(requested) > 1:
        raise ConfigError(
            f"one run uses one model, and a model is one horizon: got --horizon "
            f"{list(requested)}. Run one backtest/explanation/importance per horizon."
        )
    root = context.config.data_dir("models")
    candidates = _model_candidates(root, requested[0] if requested else None)
    if not candidates:
        return None
    if not requested:
        available = sorted(
            {horizon for horizon in (_metadata_horizon(path) for path in candidates) if horizon}
        )
        if len(available) > 1:
            raise ConfigError(
                f"{root} holds model artefacts for several horizons ({available}); pass "
                "--horizon to say which model to use"
            )
    try:
        artefact = ModelArtefact.load(candidates[0])
    except ModelError as exc:
        raise ConfigError(str(exc)) from None
    _check_horizon_match(artefact, args)
    return artefact



def _check_horizon_match(artefact: ModelArtefact, args: argparse.Namespace) -> None:
    """A ``--horizon`` that disagrees with the artefact's own horizon is a config error."""
    requested = tuple(dict.fromkeys(int(value) for value in (args.horizons or ())))
    if requested and artefact.horizon is not None and requested[0] != int(artefact.horizon):
        raise ConfigError(
            f"the model {artefact.path.name} was trained for horizon {int(artefact.horizon)}m, "
            f"not --horizon {requested[0]}m; pass the matching horizon or point --model at "
            "another artefact"
        )



def _backtest_policies(
    context: AppContext, args: argparse.Namespace, *, timeframe: str
) -> tuple[SignalPolicy, ExecutionPolicy, CostPolicy]:
    """The signal, execution and cost policies, with a policy error reported as a config one."""
    try:
        signal = SignalPolicy.from_config(context.config, min_probability=args.min_probability)
        execution = ExecutionPolicy.from_config(context.config, timeframe=timeframe)
        costs = CostPolicy.from_config(context.config)
    except (BacktestError, CostError) as exc:
        raise ConfigError(str(exc)) from None
    return signal, execution, costs




def _money(value: float | None) -> str:
    """Rupees, two decimals, or ``n/a`` - never a zero standing in for "not measured"."""
    return "n/a" if value is None else f"{value:,.2f}"



def _ratio(value: float | None, *, percent: bool = False) -> str:
    """A ratio as a percentage (or a plain 4-decimal number), or ``n/a``."""
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"



def _exit_text(execution: ExecutionPolicy) -> str:
    """The exit policy in one phrase (each rule named, or off)."""
    parts = [
        f"stop {execution.stop_loss_atr:g}xATR" if execution.stop_loss_atr else "stop off",
        f"target {execution.target_atr:g}xATR" if execution.target_atr else "target off",
        (
            f"hold <= {execution.max_holding_minutes:g}m"
            if execution.max_holding_minutes
            else "hold unlimited"
        ),
    ]
    if execution.flat_at_session_end:
        parts.append("flat at session end")
    if execution.exit_on_opposite_signal:
        parts.append("opposite signal exits")
    parts.append(f"both-touched: {execution.intrabar_priority}")
    return ", ".join(parts)



def _segment_lines(segments: Sequence[Any]) -> list[str]:
    """One line per segment row, with its sample and its PRELIMINARY flag."""
    lines: list[str] = []
    for table in segments:
        if not table.rows:
            continue
        lines.append(
            f"  by {table.name:<18}: {table.description} "
            f"(PRELIMINARY below {table.preliminary_min_samples} trade(s))"
        )
        for row in table.rows:
            lines.append(
                f"    {row.label:<18} n={row.sample:<4} net {_money(row.metrics.total_net_pnl):>12} "
                f"win {_ratio(row.metrics.win_rate, percent=True):>7} "
                f"costs {_money(row.metrics.total_costs):>10}"
                + ("  [PRELIMINARY]" if row.preliminary else "")
            )
    return lines



def _render_backtest_summary(
    *, payload: Mapping[str, Any], lines: Sequence[str], as_json: bool
) -> str:
    """The JSON payload, or the human report."""
    if as_json:
        return json.dumps(payload, indent=2, default=str)
    return "\n".join(lines)




def _backtest_report_lines(
    *,
    result: BacktestResult,
    artefact: ModelArtefact,
    signal: SignalPolicy,
    execution: ExecutionPolicy,
    cost_policy: CostPolicy,
    metrics: TradeMetrics,
    segments: Sequence[Any],
    gate: Any,
    verdict: Any,
    days: int,
    writer_stats: Mapping[str, int],
    caveats: Sequence[str],
) -> list[str]:
    """The human ``--mode backtest`` report: what ran, what happened, what may not be claimed."""
    counts = result.counts
    directions = ", ".join(f"{name}<-{klass}" for name, klass in sorted(signal.directions.items()))
    window = (
        f"{result.data_start.date().isoformat()} .. {result.data_end.date().isoformat()}"
        if result.data_start is not None and result.data_end is not None
        else "open bounds (every stored bar)"
    )
    lines = [
        f"ANJANEYA-AI backtest - timeframe {result.timeframe}, "
        f"horizon {artefact.horizon if artefact.horizon is not None else '?'}m, "
        f"model {artefact.path.name}",
        f"  window         : {window} over {days} trading day(s)",
        f"  gate           : {gate.describe()}",
        f"  verdict        : {verdict.describe()}",
        (
            f"  signal         : {signal.mode}, p >= {signal.min_probability:.2f} of {directions}"
            f"{', argmax required' if signal.require_argmax else ''} "
            "- a threshold rule, not spec 24's decision layer"
        ),
        f"  exits          : {_exit_text(execution)}",
        (
            f"  fills          : {execution.fill_at} at the bar open, latency "
            f"{execution.latency_bars} bar(s) ({execution.latency_ms} ms); quantity "
            f"{execution.quantity} unit(s) = {execution.lots} x {execution.lot_size}"
        ),
        (
            "  costs          : "
            f"brokerage {cost_policy.brokerage_per_order:.2f}/order, "
            f"slippage {cost_policy.slippage_bps:.1f}bps, "
            f"STT {cost_policy.stt_rate * 100:.4f}% (sell), "
            f"exchange {cost_policy.exchange_txn_rate * 100:.4f}%, "
            f"SEBI {cost_policy.sebi_rate * 100:.6f}%, "
            f"GST {cost_policy.gst_rate * 100:.1f}% on charges, "
            f"stamp {cost_policy.stamp_duty_rate * 100:.4f}% (buy), "
            f"captured spread {'on' if cost_policy.use_captured_spread else 'off'}"
        ),
        (
            f"  bars           : {counts.bars} bar(s) over {counts.sessions} contract-session(s) "
            f"across {counts.contracts} contract(s); {counts.decision_bars} flat-bar decision "
            f"point(s), {counts.bars_without_features} without a feature row, "
            f"{counts.bars_without_probability} without a prediction"
        ),
        (
            f"  decisions      : {counts.decisions} signal(s) fired, {counts.entries} entered, "
            f"{counts.no_signal} rejected by the threshold, {counts.no_fill_bar} had no fill bar, "
            f"{counts.both_touched} bar(s) touched stop and target"
        ),
        (
            f"  trades         : {metrics.total_trades} closed ({metrics.winners} winner(s), "
            f"{metrics.losers} loser(s), {metrics.flat} scratch) - every loser is in the ledger"
        ),
        (
            f"  p&l            : gross {_money(metrics.total_gross_pnl)}, costs "
            f"{_money(metrics.total_costs)}, net {_money(metrics.total_net_pnl)}"
            + (
                f" (costs are {metrics.cost_ratio * 100:.1f}% of the gross edge)"
                if metrics.cost_ratio is not None
                else ""
            )
        ),
        (
            f"  rates          : win {_ratio(metrics.win_rate, percent=True)}, "
            f"avg win {_money(metrics.average_win)}, avg loss {_money(metrics.average_loss)}, "
            f"payoff {_ratio(metrics.payoff_ratio)}, profit factor {_ratio(metrics.profit_factor)}, "
            f"expectancy {_money(metrics.expectancy)}"
        ),
        (
            f"  drawdown       : {_money(metrics.max_drawdown)}"
            + (
                f" ({metrics.max_drawdown_pct * 100:.2f}% of peak equity)"
                if metrics.max_drawdown_pct is not None
                else ""
            )
        ),
        (
            f"  holding        : avg {_money(metrics.average_holding_minutes)} min, exposure "
            f"{_ratio(metrics.exposure, percent=True)} of bars"
        ),
        (
            "  distribution   : "
            + ", ".join(
                f"{name} {_money(value)}"
                for name, value in metrics.distribution.percentiles.items()
            )
            + f" (mean {_money(metrics.distribution.mean)}, "
            f"sd {_money(metrics.distribution.stdev)})"
        ),
        (
            f"  sharpe-like    : {_ratio(metrics.sharpe_like)} per trade, "
            f"{_ratio(metrics.annualised_sharpe_like)} annualised"
        ),
        "  exits          : "
        + (
            ", ".join(f"{name} {count}" for name, count in sorted(counts.exit_reasons.items()))
            or "none"
        ),
        (
            f"  spread fills   : {counts.spread_fills} order(s) crossed a captured spread, "
            f"{counts.spread_missing} bar(s) had none"
        ),
    ]
    lines.extend(_segment_lines(segments))
    lines.extend(f"  caveat         : {text}" for text in caveats)
    lines.append(
        f"  artifacts      : {writer_stats.get('files', 0)} file(s), "
        f"{writer_stats.get('trades', 0)} trade row(s), "
        f"{writer_stats.get('stored_rows', 0)} row(s) in the trades table, "
        f"{writer_stats.get('failures', 0)} failure(s)"
    )
    return lines




def _backtest_plan_lines(
    *,
    context: AppContext,
    artefact: ModelArtefact,
    signal: SignalPolicy,
    execution: ExecutionPolicy,
    cost_policy: CostPolicy,
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    bars: int,
    contracts: int,
    days: int,
    gate: Any,
    verdict: Any,
    write: bool,
) -> list[str]:
    """What ``--mode backtest --dry-run`` prints instead of replaying (and writes nothing)."""
    start, end = window
    return [
        f"ANJANEYA-AI backtest (dry run) - timeframe {execution.timeframe}, "
        f"horizon {artefact.horizon if artefact.horizon is not None else '?'}m, "
        f"model {artefact.path.name}",
        f"  run_id         : {context.run_context.run_id}",
        "  window         : "
        + (
            f"{start.date().isoformat()} .. {end.date().isoformat()}"
            if start is not None and end is not None
            else "open bounds (every stored bar)"
        ),
        f"  bars           : {bars} bar(s) over {days} trading day(s), {contracts} contract(s)",
        "  contracts      : " + (", ".join(symbols) if symbols else "every contract with bars"),
        f"  gate           : {gate.describe()}",
        f"  verdict        : {verdict.describe()}",
        f"  signal         : {signal.mode}, p >= {signal.min_probability:.2f} "
        f"({', '.join(f'{k}<-{v}' for k, v in sorted(signal.directions.items()))})",
        f"  exits          : {_exit_text(execution)}",
        f"  costs          : slippage {cost_policy.slippage_bps:.1f}bps, brokerage "
        f"{cost_policy.brokerage_per_order:.2f}/order, GST {cost_policy.gst_rate * 100:.1f}%",
        "  writes to      : "
        + (
            f"{context.config.data_dir('backtests')} + the trades table"
            if write
            else "nothing (--no-write)"
        ),
        f"  caveat         : {COST_MODEL_CAVEAT}",
    ]




def _backtest_caveats(
    *,
    artefact: ModelArtefact,
    signal: SignalPolicy,
    execution: ExecutionPolicy,
    timeframe: str,
) -> list[str]:
    """The sentences every backtest prints and stores (the model's own label included)."""
    caveats = [
        COST_MODEL_CAVEAT,
        f"the signal rule is {signal.mode} at p >= {signal.min_probability:.2f}, which is the "
        "stand-in for spec 24's decision layer: this measures costs, not an edge",
        "only the candles and features tables are read - the labels table is never touched, so "
        "no trade can be justified by its own future",
        "prices are session bars: the intrabar path is unknown, so a bar touching both stop and "
        f"target is resolved by {execution.intrabar_priority} and counted",
    ]
    if artefact.timeframe and artefact.timeframe != timeframe:
        caveats.append(
            f"the model was trained on {artefact.timeframe} bars but replays {timeframe} bars; "
            "the prediction horizon moves with the bar"
        )
    if artefact.results_label:
        caveats.append(f"the model artefact is itself labelled {artefact.results_label}")
    return caveats



def _backtest_payload(
    *,
    result: BacktestResult,
    artefact: ModelArtefact,
    signal: SignalPolicy,
    execution: ExecutionPolicy,
    cost_policy: CostPolicy,
    metrics: TradeMetrics,
    segments: Sequence[Any],
    gate: Any,
    verdict: Any,
    timeframe: str,
    horizon: int | None,
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    days: int,
    caveats: Sequence[str],
) -> dict[str, Any]:
    """Everything the summary file and the JSON report carry (the ledger stays in the ledger)."""
    start, end = window
    return {
        "mode": "backtest",
        "timeframe": timeframe,
        "horizon_minutes": horizon,
        "model": artefact.describe(),
        "signal": signal.describe(),
        "execution": execution.describe(),
        "costs": cost_policy.describe(),
        "window": {
            "from": start.isoformat() if start is not None else None,
            "to": end.isoformat() if end is not None else None,
            "trading_days": days,
            "symbols": list(symbols) if symbols else None,
        },
        "gate": gate.to_dict(),
        "verdict": verdict.to_dict(),
        "counts": result.counts.to_dict(),
        "metrics": metrics.to_dict(),
        "segments": [table.to_dict() for table in segments],
        "caveats": list(caveats),
    }


def _backtest_dry_run(
    *,
    context: AppContext,
    artefact: ModelArtefact,
    signal: SignalPolicy,
    execution: ExecutionPolicy,
    cost_policy: CostPolicy,
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    candles: Any,
    days: int,
    gate: Any,
    verdict: Any,
    write: bool,
) -> int:
    """``--dry-run``: print the plan (and write nothing at all)."""
    print(
        "\n".join(
            _backtest_plan_lines(
                context=context,
                artefact=artefact,
                signal=signal,
                execution=execution,
                cost_policy=cost_policy,
                window=window,
                symbols=symbols,
                bars=len(candles),
                contracts=int(candles["tradingsymbol"].nunique()),
                days=days,
                gate=gate,
                verdict=verdict,
                write=write,
            )
        )
    )
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)






def _run_backtest(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode backtest``: replay the stored bars against one model and write the ledger."""
    timeframe = _resolve_timeframe(context, args.timeframe)
    signal, execution, cost_policy = _backtest_policies(context, args, timeframe=timeframe)
    artefact = _resolve_model_artefact(context, args)
    if artefact is None:
        print(
            "ANJANEYA-AI: no model artefact to replay under "
            f"{context.config.data_dir('models')} - train one first (--mode train) or pass "
            "--model PATH.",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_MODEL", exit_code=EXIT_OK)
    horizon = int(artefact.horizon) if artefact.horizon is not None else None
    window = _window(context, args)
    start, end = window
    symbols = list(args.symbols) if args.symbols else None
    write = not bool(getattr(args, "no_write", False))
    predictor = ModelPredictor(artefact)
    caveats = _backtest_caveats(
        artefact=artefact, signal=signal, execution=execution, timeframe=timeframe
    )
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        candles = read_candles(
            connection, start=start, end=end, symbols=symbols, timeframes=[timeframe]
        )
        if candles.empty:
            print(
                f"ANJANEYA-AI: no stored {timeframe} candles in scope - build them first "
                "(--mode build-candles).",
                file=sys.stderr,
            )
            return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
        days = len(trading_days(candles, timezone_name=context.config.timezone_name))
        gate = _gate_status(context, days)
        verdict = gate_verdict(gate.stage, days=days, forced=True)
        features = read_features(
            connection, start=start, end=end, symbols=symbols, timeframes=[timeframe]
        )
        if context.dry_run:
            return _backtest_dry_run(
                context=context,
                artefact=artefact,
                signal=signal,
                execution=execution,
                cost_policy=cost_policy,
                window=window,
                symbols=symbols,
                candles=candles,
                days=days,
                gate=gate,
                verdict=verdict,
                write=write,
            )
        try:
            probabilities = predictor.predict_frame(features)
        except ModelError as exc:
            raise ConfigError(str(exc)) from None
        result = Backtester(
            cost_model=CostModel(cost_policy),
            execution=execution,
            signal=signal,
            run_id=context.run_context.run_id,
            logger=context.logger,
            timezone_name=context.config.timezone_name,
        ).run(candles=candles, features=features, probabilities=probabilities)
        metrics = TradeMetrics.from_trades(
            result.trades,
            bars=result.counts.bars,
            bars_in_position=result.counts.in_position_bars,
        )
        segments = build_segment_tables(
            result.trades,
            preliminary_min_samples=int(context.config.model.evaluation.preliminary_min_samples),
        )
        payload = _backtest_payload(
            result=result,
            artefact=artefact,
            signal=signal,
            execution=execution,
            cost_policy=cost_policy,
            metrics=metrics,
            segments=segments,
            gate=gate,
            verdict=verdict,
            timeframe=timeframe,
            horizon=horizon,
            window=window,
            symbols=symbols,
            days=days,
            caveats=caveats,
        )
        writer = BacktestWriter(
            root=context.config.data_dir("backtests"),
            run_id=context.run_context.run_id,
            config_hash=context.config.config_hash,
            mirror=write,
            logger=context.logger,
        )
        for path in writer.write(
            result=result,
            summary=payload,
            horizon=horizon,
            connection=connection,
            store_ledger=write,
        ):
            context.run_context.register_artifact(path)
        context.logger.info(
            f"backtest: {metrics.total_trades} trade(s) over {days} day(s), net "
            f"{metrics.total_net_pnl:.2f} after {metrics.total_costs:.2f} of costs "
            f"({verdict.results_label})"
        )
        context.run_context.add_note(f"backtest verdict: {verdict.describe()}")
        for caveat in caveats:
            context.run_context.add_note(caveat)
        lines = _backtest_report_lines(
            result=result,
            artefact=artefact,
            signal=signal,
            execution=execution,
            cost_policy=cost_policy,
            metrics=metrics,
            segments=segments,
            gate=gate,
            verdict=verdict,
            days=days,
            writer_stats=writer.stats.to_dict(),
            caveats=caveats,
        )
    finally:
        connection.close()
    print(_render_backtest_summary(payload=payload, lines=lines, as_json=args.as_json))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)



# ---------------------------------------------------------------------------
# --mode report (spec 24-25): the ten answers, from the ledger backtest wrote
# ---------------------------------------------------------------------------


def _newest_ledger(context: AppContext) -> Path | None:
    """The newest ``*_ledger.parquet`` under ``data/backtests``, or ``None``.

    Newest by modification time, not by name: an artefact's name sorts by timeframe and
    horizon, so two horizons of the same window would otherwise be picked by alphabet.
    """
    root = context.config.data_dir("backtests")
    if not root.exists():
        return None
    candidates = [path for path in root.glob("*_ledger.parquet") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)



def _read_ledger(context: AppContext, args: argparse.Namespace) -> Path | None:
    """``--ledger`` when it names a file, else the newest artefact under ``data/backtests``."""
    chosen = getattr(args, "ledger", None)
    if not chosen:
        return _newest_ledger(context)
    path = Path(chosen)
    return path if path.is_file() else None



def _index_bars(
    context: AppContext, *, start: Any, end: Any
) -> tuple[pd.DataFrame | None, str | None]:
    """The stored NIFTY index bars over the ledger's window, or the reason there are none.

    The bars answer spec 24.7 and 24.8 (the market's own profile), so their absence is
    reported rather than substituted: a missing database, an unreadable one or an empty range
    each come back as a sentence the report carries in its caveats.
    """
    symbol = (
        context.config.option_universe.reference_instruments.nifty_spot.tradingsymbol or "NIFTY"
    )
    if start is None:
        return None, (
            "the ledger holds no trades, so there is no window to profile (items 7 and 8 need "
            "days the ledger actually traded)"
        )
    if not context.db_path.exists():
        return None, f"there is no database at {context.db_path}, so no index bars were read"
    try:
        connection = connect(context.db_path, read_only=True)
    except Exception as exc:  # noqa: BLE001 - a busy database must not fail the whole report
        return None, f"the database could not be opened read-only ({type(exc).__name__}: {exc})"
    try:
        frame = read_index_history(
            connection, start=start, end=end, symbols=[symbol], timeframes=["1m"]
        )
    finally:
        connection.close()
    if frame.empty:
        return None, (
            f"no stored 1m index bars for {symbol!r} in the ledger's window "
            "(--mode backfill builds the index history)"
        )
    return frame, None



def _rate_text(value: Any) -> str:
    """A win rate for the human report, or ``n/a`` when there is no rate to print."""
    return "n/a" if value is None else f"{float(value):.3f}"



def _report_lines(
    *,
    report: ResearchReport,
    written: Sequence[Path],
    write: bool,
) -> list[str]:
    """The human ``--mode report`` output: what was read, what each item found, what not to claim."""
    policy = report.policy
    lines = [
        f"ANJANEYA-AI research report (spec 24) - {report.trades} closed trade(s) over "
        f"{report.days} entry day(s)",
        f"  ledger        : {report.ledger}",
        f"  window        : {report.first_entry} .. {report.last_exit}",
        f"  samples       : a cell is {PRELIMINARY_LABEL} below "
        f"{policy.preliminary_min_samples} trades",
        f"  significance  : {policy.multiple_testing_correction} at "
        f"alpha={policy.significance_level:g}",
    ]
    if report.gate:
        lines.append(f"  data gate     : {report.gate.get('stage')} - {report.gate.get('allowed')}")
    lines.append("")
    for matrix in report.matrices:
        lines.append(f"  {matrix.name}:")
        for row in matrix.rows():
            flag = f" [{PRELIMINARY_LABEL}]" if row["preliminary"] else ""
            lines.append(
                f"    {row['row']} | {row['column']}: {row['total_net_pnl']:.2f} over "
                f"{row['sample']} trade(s), win rate {_rate_text(row['win_rate'])}{flag}"
            )
        summary = next((item for item in report.significance if item.matrix == matrix.name), None)
        if summary is not None:
            lines.append(
                f"    tested: {summary.family_size} cell(s); clearing the corrected level: "
                f"{len(summary.significant_cells)}"
            )
    if report.volatility is not None:
        lines.append(
            f"  volatility    : {report.volatility.symbol}, {len(report.volatility.profiles)} "
            f"bucket x phase cell(s) over {report.volatility.sessions} session(s)"
        )
    else:
        lines.append(f"  volatility    : not measured - {report.volatility_unavailable}")
    if report.paths:
        lines.append(f"  intraday path : {len(report.paths)} bucket path(s)")
    else:
        lines.append(f"  intraday path : not measured - {report.paths_unavailable}")
    preliminary = report.preliminary_cells
    lines.append(
        f"  preliminary   : {sum(len(labels) for labels in preliminary.values())} label(s) "
        f"across {len(preliminary)} table(s)"
    )
    if write:
        lines.extend(f"  wrote         : {path}" for path in written)
    else:
        lines.append("  wrote         : nothing (--dry-run / --no-write)")
    lines.append("")
    lines.extend(f"  caveat: {caveat}" for caveat in report.caveats)
    lines.append("")
    lines.append("  the ten items in full: the JSON and Markdown artefacts beside the ledger")
    return lines





def _run_status(context: AppContext, *, as_json: bool) -> int:

    """Print the offline status report."""
    report = status_report(context)
    print(render_status(report, as_json=as_json))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


def _bar_days(context: AppContext, bars: pd.DataFrame) -> int:
    """How many exchange-local sessions the index bars cover (the gate's unit of account)."""
    tz = get_timezone(context.config.timezone_name)
    stamps = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"])).tz_convert(tz)
    return int(pd.Index(stamps.date).nunique())



def _journal_window(context: AppContext, trades: Sequence[Any]) -> tuple[Any, Any]:
    """The ``[start, end)`` read window the ledger's trades span, in exchange-local time.

    The report asks the market's own questions (items 7 and 8) about *the days the trades
    happened on*, so the window comes from the ledger rather than from ``--from/--to``: a
    profile computed over a wider window would describe days the ledger never traded.
    """
    tz = get_timezone(context.config.timezone_name)
    entries = [pd.Timestamp(trade.entry_timestamp) for trade in trades if trade.entry_timestamp]
    if not entries:
        return None, None
    exits = [pd.Timestamp(trade.exit_timestamp) for trade in trades if trade.exit_timestamp]
    first = min(entries).tz_convert(tz).date()
    last = max(exits).tz_convert(tz).date() if exits else first
    start = datetime(first.year, first.month, first.day, tzinfo=tz)
    end = datetime(last.year, last.month, last.day, tzinfo=tz) + timedelta(days=1)
    return start, end



def _run_report(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode report``: spec 24's ten answers over one stored ledger.

    The ledger is where every number comes from, plus the stored index bars for the two
    questions about the market itself. The mode reads the database read-only and writes three
    artefacts under ``data/reports/research/``; it never writes a feature row, a label row or
    a trade, so a report can never change the evidence it describes.
    """
    policy = ReportPolicy.from_config(context.config)
    path = _read_ledger(context, args)
    if path is None:
        chosen = getattr(args, "ledger", None)
        print(
            "ANJANEYA-AI: no trade ledger to analyse"
            + (f" at {chosen}" if chosen else f" under {context.config.data_dir('backtests')}")
            + " - run --mode backtest first (it writes the ledger this stage reads), or pass "
            "--ledger PATH.",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_LEDGER", exit_code=EXIT_OK)
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - a broken artefact is an operator problem
        print(
            f"ANJANEYA-AI: the ledger {path} could not be read ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        return finalize_run(context, status="FAILED", exit_code=EXIT_FAILED)
    try:
        trades = ledger_trades(frame)
    except ResearchError as exc:
        print(f"ANJANEYA-AI: the ledger {path.name} is not a trade ledger: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_FAILED)

    start, end = _journal_window(context, trades)
    bars, bars_reason = _index_bars(context, start=start, end=end)
    captured_days = (
        _bar_days(context, bars)
        if bars is not None
        else len({pd.Timestamp(trade.entry_timestamp).date() for trade in trades})
    )
    gate = _gate_status(context, captured_days)
    report = build_research_report(
        frame,
        policy=policy,
        calendar=context.calendar,
        bars=bars,
        ledger_name=path.name,
        gate=gate,
        captured_days=captured_days,
    )
    if bars_reason is not None:
        report.caveats.append(bars_reason)

    write = not bool(getattr(args, "no_write", False)) and not context.dry_run
    writer = ResearchWriter(
        root=context.config.data_dir("reports") / "research",
        run_id=context.run_context.run_id,
        config_hash=context.config.config_hash,
        mirror=write,
        logger=context.logger,
    )
    written = writer.write(report, stem=writer.stem(ledger=path.name))
    for written_path in written:
        context.run_context.register_artifact(written_path)
    context.run_context.add_note(
        f"research report: {report.trades} trade(s), {report.days} day(s)"
    )
    context.run_context.add_note(f"ledger analysed: {path.name}")
    if report.gate and report.gate.get("stage"):
        context.run_context.add_note(f"data gate: {report.gate['stage']}")
    for caveat in report.caveats:
        context.run_context.add_note(caveat)
    payload = {
        "run_id": context.run_context.run_id,
        "config_hash": context.config.config_hash,
        "ledger_path": str(path),
        "written": [str(item) for item in written],
        "writer": writer.stats.to_dict(),
        **report.to_dict(),
    }
    lines = _report_lines(report=report, written=written, write=write)
    print(json.dumps(payload, indent=2, default=str) if args.as_json else "\n".join(lines))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)



# ---------------------------------------------------------------------------
# --mode explain (spec 26): what the model was looking at, in plain form
# ---------------------------------------------------------------------------


def _explanation_policy(context: AppContext) -> ExplanationPolicy:
    """The explanation policy, with a policy error reported as a configuration one."""
    try:
        return ExplanationPolicy.from_config(context.config)
    except ExplanationError as exc:
        raise ConfigError(str(exc)) from None


def _explain_caveats(artefact: ModelArtefact) -> list[str]:
    """The honesty block every explanation artefact carries.

    The label of the model being explained is part of it: a model trained below spec 4.3's
    gate explains a prediction that was never fit for a claim, and an explanation must not
    launder that into a number a reader trusts.
    """
    caveats = [EXPLANATION_CAVEAT, DECISION_LAYER_CAVEAT, FRESHNESS_CAVEAT]
    if artefact.results_label:
        caveats.append(
            f"the model being explained is labelled {artefact.results_label} by its own "
            "training run: an explanation inherits the model's standing, it does not add to it"
        )
    return caveats


def _explain_count(args: argparse.Namespace, policy: ExplanationPolicy) -> int:
    """How many predictions to explain: ``--rows``, else the configured default."""
    requested = getattr(args, "rows", None)
    if requested is not None and int(requested) < 1:
        raise ConfigError(f"--rows must be at least 1, got {requested}")
    return int(requested) if requested is not None else int(policy.rows)


def _artefact_stem(artefact: ModelArtefact, run_id: str, *, fallback: str = "reports") -> str:
    """The stem an artefact set is written under: the model's own, else ``<fallback>_<run id>``."""
    return model_stem(str(artefact.path)) or f"{fallback}_{run_id}"


def _check_artefact_timeframe(artefact: ModelArtefact, timeframe: str, *, action: str) -> None:
    """A ``--timeframe`` the artefact never saw is a configuration error, for any mode.

    ``action`` completes the sentence with what the mode was about to do, so the operator reads
    the consequence rather than a rule number.
    """
    if artefact.timeframe and str(artefact.timeframe) != timeframe:
        raise ConfigError(
            f"the model {artefact.path.name} was trained on {artefact.timeframe} rows, not "
            f"--timeframe {timeframe}: {action}"
        )


def _horizon_text(horizon: int | None) -> str:
    """``5m`` or ``unknown`` - how every explanation line names the model's horizon."""
    return f"{int(horizon)}m" if horizon is not None else "unknown"


def _explain_plan_lines(
    *,
    context: AppContext,
    artefact: ModelArtefact,
    policy: ExplanationPolicy,
    timeframe: str,
    horizon: int | None,
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    rows_in_scope: int,
    to_explain: int,
    write: bool,
) -> list[str]:
    """``--dry-run``: what would be explained, from which rows, and where it would land."""
    start, end = window
    stem = _artefact_stem(artefact, context.run_context.run_id, fallback="explanations")
    root = context.config.data_dir("reports") / "explanations"
    return [
        f"ANJANEYA-AI explanations (spec {policy.spec}) - dry run, nothing is written",
        f"  model      : {artefact.path.name} ({artefact.algorithm}, "
        f"horizon {_horizon_text(horizon)})",
        f"  timeframe  : {timeframe}",
        f"  window     : {start or 'beginning'} .. {end or 'newest'}",
        f"  scope      : {', '.join(symbols) if symbols else 'every stored contract'}",
        f"  rows       : {rows_in_scope} feature row(s) in scope, the newest {to_explain} "
        "of them explained",
        f"  blocks     : {', '.join(block.title for block in policy.blocks)}, {AI_SECTION_TITLE} "
        f"({len(policy.columns)} stored feature(s) plus the model's class probabilities)",
        f"  policy     : {policy.digits} decimal(s), top {policy.probabilities} probability(ies), "
        f"a NULL feature printed as {policy.missing_text!r}",
        "  pipeline   : read features -> score them -> build the blocks -> JSON/Markdown/CSV",
        f"  writes     : {root / f'{stem}_explanations.json|.md'} and "
        f"{root / f'{stem}_entries.csv'}"
        if write
        else "  writes     : nothing (--no-write)",
    ]


def _explain_lines(
    *,
    batch: ExplanationBatch,
    written: Sequence[Path],
    write: bool,
) -> list[str]:
    """The console document: the header, every explained prediction, then the provenance."""
    lines = [
        f"ANJANEYA-AI explanations (spec {batch.policy.spec}) - {batch.rows} prediction(s), "
        f"{batch.entries} stored feature(s) of which {batch.unknown} unknown at that instant",
    ]
    if batch.model:
        lines.append(f"  model      : {batch.model.get('path')}")
    start, end = batch.window
    if start is not None or end is not None:
        lines.append(f"  window     : {start or 'beginning'} .. {end or 'newest'}")
    if batch.timeframe:
        lines.append(f"  timeframe  : {batch.timeframe} {_horizon_text(batch.horizon_minutes)}")
    if batch.symbols:
        lines.append(f"  contracts  : {', '.join(batch.symbols)}")
    lines.append(
        f"  blocks     : {', '.join(block.title for block in batch.policy.blocks)}, "
        f"{AI_SECTION_TITLE}"
    )
    for item in batch.explanations:
        lines.append("")
        lines.extend(item.to_text().splitlines())
    lines.append("")
    if write:
        for path in written:
            lines.append(f"  wrote      : {path}")
    else:
        lines.append("  wrote      : nothing (--no-write)")
    for caveat in batch.caveats:
        lines.append(f"  caveat: {caveat}")
    return lines


def _run_explain(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode explain``: spec 26 over the newest stored rows, with one model's probabilities.

    Read-only against everything that already exists: the mode reads stored feature rows,
    scores them with a loaded artefact, and writes explanation artefacts under
    ``data/reports/explanations/``. It writes no feature row, no label row and no trade, so an
    explanation can never change the evidence it describes. No data-accumulation gate applies
    - an explanation describes an input, it is not a result - and the model's own label travels
    with every artefact instead.
    """
    policy = _explanation_policy(context)
    timeframe = _resolve_timeframe(context, args.timeframe)
    artefact = _resolve_model_artefact(context, args)
    if artefact is None:
        print(
            "ANJANEYA-AI: no model artefact to explain under "
            f"{context.config.data_dir('models')} - train one first (--mode train) or pass "
            "--model PATH.",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_MODEL", exit_code=EXIT_OK)
    _check_artefact_timeframe(
        artefact,
        timeframe,
        action="explaining it from another timeframe would describe inputs it never saw",
    )
    horizon = int(artefact.horizon) if artefact.horizon is not None else None
    window = _window(context, args)
    symbols = list(args.symbols) if args.symbols else None
    to_explain = _explain_count(args, policy)
    write = not bool(getattr(args, "no_write", False))
    caveats = _explain_caveats(artefact)
    predictor = ModelPredictor(artefact)
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        features = read_features(
            connection, start=window[0], end=window[1], symbols=symbols, timeframes=[timeframe]
        )
        if features.empty:
            print(
                f"ANJANEYA-AI: no stored {timeframe} feature rows in scope - build them first "
                "(--mode build-features).",
                file=sys.stderr,
            )
            return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
        if context.dry_run:
            print(
                "\n".join(
                    _explain_plan_lines(
                        context=context,
                        artefact=artefact,
                        policy=policy,
                        timeframe=timeframe,
                        horizon=horizon,
                        window=window,
                        symbols=symbols,
                        rows_in_scope=int(len(features)),
                        to_explain=to_explain,
                        write=write,
                    )
                )
            )
            return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)
        try:
            predicted = predictor.predict_frame(features)
        except ModelError as exc:
            raise ConfigError(str(exc)) from None
    finally:
        connection.close()

    try:
        batch = build_explanations(
            features,
            policy=policy,
            predicted=predicted,
            model=artefact.describe(),
            limit=to_explain,
            window=window,
            symbols=tuple(symbols) if symbols else (),
            timeframe=timeframe,
            horizon_minutes=horizon,
            caveats=caveats,
        )
    except ExplanationError as exc:
        raise ConfigError(str(exc)) from None

    writer = ExplanationWriter(
        root=context.config.data_dir("reports") / "explanations",
        run_id=context.run_context.run_id,
        config_hash=context.config.config_hash,
        mirror=write,
        logger=context.logger,
    )
    written = writer.write(
        batch, stem=_artefact_stem(artefact, context.run_context.run_id, fallback="explanations")
    )
    for path in written:
        context.run_context.register_artifact(path)
    context.logger.info(
        f"explain: {batch.rows} prediction(s), {batch.entries} stored feature(s) "
        f"({batch.unknown} unknown) from {artefact.path.name}"
    )
    context.run_context.add_note(f"explained model: {artefact.path.name}")
    context.run_context.add_note(
        f"explanations: {batch.rows} prediction(s), {batch.entries} line(s), "
        f"{batch.unknown} unknown"
    )
    for caveat in caveats:
        context.run_context.add_note(caveat)
    payload = {
        "run_id": context.run_context.run_id,
        "config_hash": context.config.config_hash,
        "mode": "explain",
        "model_path": str(artefact.path),
        "written": [str(path) for path in written],
        "writer": writer.stats.to_dict(),
        **batch.to_dict(),
    }
    lines = _explain_lines(batch=batch, written=written, write=write)
    print(json.dumps(payload, indent=2, default=str) if args.as_json else "\n".join(lines))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


# ---------------------------------------------------------------------------
# --mode importance (spec 27): where one model's information came from
# ---------------------------------------------------------------------------


def _importance_policy(context: AppContext) -> ImportancePolicy:
    """The importance policy, with a policy error reported as a configuration one."""
    try:
        return ImportancePolicy.from_config(context.config)
    except ImportanceError as exc:
        raise ConfigError(str(exc)) from None


def _importance_repeats(args: argparse.Namespace, policy: ImportancePolicy) -> int:
    """How many permutations to run: ``--repeats``, else the configured default."""
    requested = getattr(args, "repeats", None)
    if requested is not None and int(requested) < 2:
        raise ConfigError(
            f"--repeats must be at least 2, got {requested}: one repeat has no spread to put "
            "an interval on"
        )
    return int(requested) if requested is not None else int(policy.repeats)


def _importance_caveats(artefact: ModelArtefact) -> list[str]:
    """The caveat the model's own standing adds to an importance measurement.

    The three structural caveats (importance describes a model, correlated features share
    credit, and which rows were measured) are added by ``core.ml.importance`` itself, so they
    are present even in a report a notebook builds by hand.
    """
    caveats: list[str] = []
    if artefact.results_label:
        caveats.append(
            f"the model measured is labelled {artefact.results_label} by its own training run: "
            "this ranking describes that model, not a validated one"
        )
    return caveats


def _importance_notes(
    *,
    policy: ImportancePolicy,
    dataset: Dataset,
    sample: Any,
    permutation: Any,
    shap: Any,
    split_reason: str | None = None,
) -> list[str]:
    """The measurement's own footnotes: what was measured, on what, and what could not be."""
    notes = [
        f"matrix: {dataset.rows} labelled row(s) over {len(dataset.days)} day(s), "
        f"{len(dataset.feature_columns)} feature(s)",
        f"measurement: {permutation.rows} row(s) of the {sample.name} sample "
        f"({sample.rows_available} available), {permutation.repeats} permutation(s) at seed "
        f"{permutation.seed}, {policy.metric} "
        f"(baseline {_text_number(permutation.baseline, policy.digits)})",
    ]
    if split_reason:
        notes.append(
            f"the walk-forward split reported: {split_reason} - a holdout is still the days "
            "the split reserved, but no fold window was built from these days"
        )
    if sample.capped:
        notes.append(
            f"the sample was capped at importance.max_rows={policy.max_rows}: "
            f"{sample.rows_available - sample.rows_used} older row(s) were left out, and the "
            "counts say so rather than the reader having to guess"
        )
    if sample.reason:
        notes.append(f"the sample fell back to the stored matrix because {sample.reason}")
    if permutation.blocked:
        notes.append(
            f"{len(permutation.blocked)} feature(s) could not be measured in this sample and "
            "are reported as not measured (never as a zero): "
            + ", ".join(sorted(permutation.blocked))
        )
    if shap is not None and not shap.measured:
        notes.append(f"SHAP: {shap.status} - {shap.reason}")
    return notes


def _importance_plan_lines(
    *,
    context: AppContext,
    artefact: ModelArtefact,
    policy: ImportancePolicy,
    timeframe: str,
    horizon: int,
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    dataset: Dataset,
    sample: Any,
    write: bool,
) -> list[str]:
    """``--dry-run``: what would be measured, on which rows, and where it would land."""
    start, end = window
    stem = _artefact_stem(artefact, context.run_context.run_id, fallback="importance")
    root = context.config.data_dir("reports") / "importance"
    return [
        f"ANJANEYA-AI feature importance (spec {policy.spec}) - dry run, nothing is written",
        f"  model      : {artefact.path.name} ({artefact.algorithm}, "
        f"horizon {_horizon_text(horizon)})",
        f"  timeframe  : {timeframe}",
        f"  window     : {start or 'beginning'} .. {end or 'newest'}",
        f"  scope      : {', '.join(symbols) if symbols else 'every stored contract'}",
        f"  matrix     : {dataset.rows} labelled row(s) over {len(dataset.days)} day(s), "
        f"{len(dataset.feature_columns)} feature(s)",
        f"  sample     : {sample.name} - {sample.rows_used} of {sample.rows_available} row(s)"
        f"{' (capped by importance.max_rows)' if sample.capped else ''}",
        f"  measurement: native importance off the booster, then {policy.repeats} permutation(s) "
        f"per feature with {policy.metric} at seed {policy.random_seed()}, "
        f"{int(policy.confidence * 100)}% intervals",
        f"  sources    : {', '.join(topic.key for topic in policy.topics)}, "
        f"{UNMAPPED_TOPIC_KEY}",
        f"  SHAP       : {policy.shap} (at most {policy.shap_rows} row(s))"
        if policy.shap != "off"
        else "  SHAP       : off (importance.shap)",
        f"  writes     : {root / f'{stem}_importance.json|.md'} and "
        f"{root / f'{stem}_features.csv'}"
        if write
        else "  writes     : nothing (--no-write)",
    ]


def _text_number(value: Any, digits: int) -> str:
    """``None`` as an em dash, else ``digits`` decimals with trailing zeros trimmed."""
    if value is None:
        return "-"
    text = f"{float(value):.{int(digits)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _importance_lines(
    *, report: ImportanceReport, written: Sequence[Path], write: bool, top: int = 5
) -> list[str]:
    """The console document: what was measured, the leading sources and features, the caveats."""
    digits = report.policy.digits
    lines = [
        f"ANJANEYA-AI feature importance (spec {report.policy.spec}) - "
        f"{report.counts['measured']} of {report.counts['features']} feature(s) measured over "
        f"{report.counts['rows']} row(s), {report.counts['repeats']} permutation(s) at seed "
        f"{report.measurement['seed']}",
        f"  model      : {report.model.get('path')} ({report.model.get('algorithm')}, "
        f"horizon {report.model.get('horizon_minutes')}m, "
        f"{report.model.get('results_label') or 'unlabelled'})",
        f"  sample     : {report.sample.get('name')} - {report.sample.get('rows_used')} row(s) "
        f"over {report.sample.get('days')} day(s), {report.sample.get('first_timestamp')} .. "
        f"{report.sample.get('last_timestamp')}",
        f"  metric     : {report.measurement['metric']} "
        f"({report.measurement['direction']}), baseline "
        f"{_text_number(report.measurement['baseline'], digits)}",
        f"  sources    : {report.counts['topics_measured']} of {report.counts['topics']} "
        "measured",
    ]
    if report.counts.get("unmapped_features"):
        lines.append(
            f"  unmapped   : {report.counts['unmapped_features']} feature(s) no configured "
            "source claims (reported, never attributed to a source that did not measure them)"
        )
    for topic in sorted(
        (item for item in report.topics if item.measured),
        key=lambda item: item.rank or len(report.topics),
    )[:top]:
        lines.append(
            f"    #{topic.rank} {topic.key} ({topic.title}): delta "
            f"{_text_number(topic.permutation_mean, digits)} ["
            f"{_text_number(topic.ci_low, digits)}, {_text_number(topic.ci_high, digits)}] "
            f"over {topic.measured_features} feature(s)"
        )
    for topic in report.topics:
        if not topic.measured:
            lines.append(f"    -  {topic.key}: {topic.reason}")
    for item in report.ranked(top):
        lines.append(
            f"    #{item.rank} {item.column} ({item.topic}): delta "
            f"{_text_number(item.permutation_mean, digits)} ["
            f"{_text_number(item.ci_low, digits)}, {_text_number(item.ci_high, digits)}]"
        )
    if report.counts.get("unmeasured"):
        lines.append(
            f"  unmeasured : {report.counts['unmeasured']} feature(s) the sample could not say "
            "anything about (all NULL or constant); each reason is in the artefact"
        )
    if report.shap is not None:
        lines.append(
            f"  SHAP       : {report.shap.get('status')}"
            + (f" - {report.shap.get('reason')}" if report.shap.get("reason") else "")
        )
    if write:
        for path in written:
            lines.append(f"  wrote      : {path}")
    else:
        lines.append("  wrote      : nothing (--no-write)")
    for caveat in report.caveats:
        lines.append(f"  caveat: {caveat}")
    for note in report.notes:
        lines.append(f"  note: {note}")
    return lines


def _run_importance(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode importance``: spec 27 over one artefact's own training matrix.

    Read-only against everything that already exists: the mode reads stored feature rows and
    labels, rebuilds the matrix the model was trained on (the same build ``--mode dataset``
    makes, policies and all) and measures the model - it never refits it and never touches its
    inputs. It writes one JSON, one Markdown and one CSV under ``data/reports/importance/`` and
    a row of notes on the run manifest. No data-accumulation gate applies: a ranking describes a
    model, so the model's own label travels in the caveats instead of a refusal.
    """
    policy = _importance_policy(context)
    repeats = _importance_repeats(args, policy)
    if repeats != policy.repeats:
        policy = replace(policy, repeats=repeats)
    timeframe = _resolve_timeframe(context, args.timeframe)
    artefact = _resolve_model_artefact(context, args)
    if artefact is None:
        print(
            "ANJANEYA-AI: no model artefact to measure under "
            f"{context.config.data_dir('models')} - train one first (--mode train) or pass "
            "--model PATH.",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_MODEL", exit_code=EXIT_OK)
    _check_artefact_timeframe(
        artefact,
        timeframe,
        action="ranking it from another timeframe would measure inputs it never saw",
    )
    if artefact.horizon is None:
        raise ConfigError(
            f"the model {artefact.path.name} records no label horizon, so the matrix it was "
            "trained on cannot be rebuilt; retrain it with --mode train, which records one"
        )
    horizon = int(artefact.horizon)
    window = _window(context, args)
    symbols = list(args.symbols) if args.symbols else None
    write = not bool(getattr(args, "no_write", False))
    caveats = _importance_caveats(artefact)
    dataset_policy = DatasetPolicy.from_config(context.config, timeframe=timeframe)
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        features = read_features(
            connection, start=window[0], end=window[1], symbols=symbols, timeframes=[timeframe]
        )
        if features.empty:
            print(
                f"ANJANEYA-AI: no stored {timeframe} feature rows in scope - build them first "
                "(--mode build-features).",
                file=sys.stderr,
            )
            return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
        labels = read_labels(connection, start=window[0], end=window[1], symbols=symbols)
    finally:
        connection.close()

    prepared = _prepare_horizon(
        context,
        features=features,
        labels=labels,
        dataset_policy=dataset_policy,
        horizon=horizon,
        symbols=symbols,
    )
    if prepared.dataset is None or prepared.error:
        print(
            f"ANJANEYA-AI: the training matrix for horizon {horizon}m could not be rebuilt: "
            f"{prepared.error}",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
    dataset, split = prepared.dataset, prepared.split
    if not dataset.rows:
        print(
            f"ANJANEYA-AI: no labelled rows for horizon {horizon}m in scope - build the labels "
            "(--mode build-labels).",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
    sample = select_sample(dataset, split, policy=policy)
    if not sample.rows:
        print("ANJANEYA-AI: no rows to measure - the sample came back empty.", file=sys.stderr)
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
    if context.dry_run:
        print(
            "\n".join(
                _importance_plan_lines(
                    context=context,
                    artefact=artefact,
                    policy=policy,
                    timeframe=timeframe,
                    horizon=horizon,
                    window=window,
                    symbols=symbols,
                    dataset=dataset,
                    sample=sample,
                    write=write,
                )
            )
        )
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    try:
        predictor = ModelPredictor(artefact)
        native = native_importance(artefact)
        permutation = permutation_importance(
            predictor, sample.frame, classes=dataset.classes, policy=policy
        )
        shap = shap_importance(predictor, sample.frame, policy=policy)
    except (ImportanceError, ModelError) as exc:
        raise ConfigError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"importance measurement failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-AI importance failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return finalize_run(context, status="FAILED", exit_code=EXIT_FAILED)

    report = build_importance_report(
        policy=policy,
        model=artefact.describe(),
        dataset=dataset.describe(),
        classes=dataset.classes,
        sample=sample,
        native=native,
        permutation=permutation,
        shap=shap,
        notes=_importance_notes(
            policy=policy,
            dataset=dataset,
            sample=sample,
            permutation=permutation,
            shap=shap,
            split_reason=getattr(split, "reason", None),
        ),
        caveats=caveats,
    )
    writer = ImportanceWriter(
        root=context.config.data_dir("reports") / "importance",
        run_id=context.run_context.run_id,
        config_hash=context.config.config_hash,
        mirror=write,
        logger=context.logger,
    )
    written = writer.write(
        report,
        stem=_artefact_stem(artefact, context.run_context.run_id, fallback="importance"),
    )
    for path in written:
        context.run_context.register_artifact(path)
    context.logger.info(
        f"importance: {report.counts['measured']} of {report.counts['features']} feature(s) "
        f"measured over {report.counts['rows']} row(s) from {artefact.path.name}"
    )
    context.run_context.add_note(f"measured model: {artefact.path.name}")
    context.run_context.add_note(
        f"importance: {report.counts['measured']} feature(s) measured, "
        f"{report.counts['topics_measured']} source(s) measured, "
        f"{report.counts['unmapped_features']} unmapped"
    )
    for caveat in report.caveats:
        context.run_context.add_note(caveat)
    for note in report.notes:
        context.run_context.add_note(note)
    payload = {
        "run_id": context.run_context.run_id,
        "config_hash": context.config.config_hash,
        "mode": "importance",
        "model_path": str(artefact.path),
        "written": [str(path) for path in written],
        "writer": writer.stats.to_dict(),
        **report.to_dict(),
    }
    lines = _importance_lines(report=report, written=written, write=write)
    print(json.dumps(payload, indent=2, default=str) if args.as_json else "\n".join(lines))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


# ---------------------------------------------------------------------------
# --mode charts (spec 28): the figures that make a result checkable by eye
# ---------------------------------------------------------------------------
#: One spec-28 chart -> (the family it belongs to, a one-line description). The order is the
#: order the index lists them, which is also roughly "market picture first, model picture last".
CHART_NAMES: dict[str, tuple[str, str]] = {
    "heatmap_time_of_day_cycle_phase": ("heatmap", "net P&L by time-of-day bucket x cycle phase"),
    "heatmap_time_of_day_day_of_week": ("heatmap", "net P&L by time-of-day bucket x weekday"),
    "curves_ce_vs_pe": ("curves", "cumulative net P&L by option side"),
    "regime_performance": ("regime", "net P&L by NIFTY regime at entry"),
    "importance_bars": ("importance", "permutation importance per feature (with intervals)"),
    "price_nifty": ("price", "NIFTY price with MA lines, trend score and regime"),
    "price_option": ("price", "one option contract with its MA stack and overlays"),
}

#: The order a regime bar chart plots its bars in (strongest down first).
REGIME_ORDER: tuple[str, ...] = (
    "STRONG_BEARISH",
    "BEARISH",
    "BEARISH_DEVELOPING",
    "NEUTRAL",
    "BULLISH_DEVELOPING",
    "BULLISH",
    "STRONG_BULLISH",
)


def _newest_importance_artefact(context: AppContext) -> Path | None:
    """The newest ``*_importance.json`` under ``data/reports/importance``, or ``None``."""
    root = context.config.data_dir("reports") / "importance"
    if not root.exists():
        return None
    candidates = [path for path in root.glob("*_importance.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_importance_payload(context: AppContext, args: argparse.Namespace) -> tuple[dict, str, str | None]:
    """``(payload, name, reason)`` for the importance artefact the bars chart reads.

    ``reason`` is ``None`` on success; otherwise the payload is empty and the reason says what
    is missing, so the mode skips exactly one chart rather than failing the run. ``--importance``
    wins when given, else the newest artefact is used.
    """
    chosen = getattr(args, "importance", None)
    path = Path(chosen) if chosen else _newest_importance_artefact(context)
    if path is None:
        root = context.config.data_dir("reports") / "importance"
        return {}, "", f"no importance artefact under {root} - run --mode importance first"
    if not path.is_file():
        return {}, path.name, f"the importance artefact {path} does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken artefact is an operator problem
        return {}, path.name, f"{path.name} could not be read as JSON ({type(exc).__name__}: {exc})"
    return payload, path.name, None


def _chart_symbol(features: pd.DataFrame, args: argparse.Namespace) -> str | None:
    """The contract the price chart draws: ``--symbol``, else the busiest one in scope.

    "Busiest" (the most feature rows over the window) is a deterministic, honest default: it
    names the contract the run actually has the most to show for, rather than the first row a
    query happened to return. A window with no feature rows has no symbol and the price charts
    are skipped with that reason.
    """
    requested = list(getattr(args, "symbols", None) or [])
    if requested:
        return requested[0]
    if features is None or features.empty or "tradingsymbol" not in features.columns:
        return None
    counts = features["tradingsymbol"].value_counts()
    return str(counts.index[0]) if len(counts) else None


def _selected(name: str, only: Sequence[str]) -> bool:
    """Whether a chart passes ``--only``: an exact name, or a family prefix, matches."""
    if not only:
        return True
    family = CHART_NAMES.get(name, (name, ""))[0]
    return any(choice in (name, family) for choice in only)


def _predictions_for(trades: Sequence[Any], symbol: str) -> list[EventMarker]:
    """The model's own predictions for one contract, as entry-point markers.

    A backtest ledger records ``ai_prediction`` per trade, so the price chart can show *where the
    model acted*: one marker per entry, at the entry price, labelled with the predicted class. A
    trade with no prediction is simply not marked - never marked as a blank.
    """
    markers: list[EventMarker] = []
    for trade in trades:
        if str(getattr(trade, "tradingsymbol", "")) != symbol:
            continue
        label = getattr(trade, "ai_prediction", None)
        if not label:
            continue
        price = getattr(trade, "entry_price", None)
        if price is None:
            continue
        markers.append(
            EventMarker(
                timestamp=getattr(trade, "entry_timestamp", None),
                value=float(price),
                label=str(label),
                kind="prediction",
            )
        )
    return markers


def _run_charts(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode charts``: spec 28's figures, drawn from the evidence already on disk.

    Spec 28 asks for charts "for validation" and says visualization is not optional. So this mode
    is read-only against everything that already exists - the trade ledger, the importance
    artefact, the stored candles/features/index bars - and writes only PNG/SVG files plus one
    index under ``data/reports/charts/``. It never recomputes a number: the heatmaps and curves
    come from the same ledger ``--mode report`` reads, and the bars come from the artefact
    ``--mode importance`` wrote, so a chart can never disagree with the number beside it.

    Each chart is attempted independently: an input that is missing (no ledger, no importance
    artefact, no stored candles) skips exactly the charts that need it, with the reason printed,
    rather than failing the run. ``--dry-run`` lists what would be built without importing
    matplotlib or writing anything.
    """
    policy = ChartPolicy.from_config(context.config)
    only = [str(choice) for choice in (getattr(args, "only", None) or []) if str(choice).strip()]
    metric = str(getattr(args, "metric", None) or "permutation_mean")
    timeframe = _resolve_timeframe(context, args.timeframe)
    min_samples = int(getattr(context.config.model.evaluation, "preliminary_min_samples", 1))
    buckets = bucket_order(context.calendar)

    figures: dict[str, Any] = {}
    notes: dict[str, str] = {}
    skipped: list[str] = []

    # -- the ledger (the heatmaps, the curves, the regime bars) --------------
    ledger_path = _read_ledger(context, args)
    trades: list[Any] = []
    if ledger_path is None:
        chosen = getattr(args, "ledger", None)
        skipped.append(
            "the P&L charts need a trade ledger"
            + (f" at {chosen}" if chosen else f" under {context.config.data_dir('backtests')}")
            + " - run --mode backtest first, or pass --ledger PATH"
        )
    else:
        try:
            trades = ledger_trades(pd.read_parquet(ledger_path))
        except Exception as exc:  # noqa: BLE001 - a broken ledger must not fail the whole run
            print(
                f"ANJANEYA-AI: the ledger {ledger_path} could not be read "
                f"({type(exc).__name__}: {exc}); the P&L charts will be skipped.",
                file=sys.stderr,
            )
            skipped.append(f"the ledger {ledger_path.name} could not be read")

    # -- the importance artefact (the bars chart) ----------------------------
    importance_payload, importance_name, importance_reason = _read_importance_payload(context, args)
    if importance_reason:
        skipped.append(f"the importance bars were skipped: {importance_reason}")

    # -- the database (the price charts) -------------------------------------
    window = _window(context, args)
    symbols = list(args.symbols) if args.symbols else None
    features = pd.DataFrame()
    candles = pd.DataFrame()
    bars = pd.DataFrame()
    symbol: str | None = None
    nifty_symbol = (
        context.config.option_universe.reference_instruments.nifty_spot.tradingsymbol or "NIFTY"
    )
    if context.db_path.exists():
        connection = connect(context.db_path, read_only=True)
        try:
            features = read_features(
                connection, start=window[0], end=window[1], symbols=symbols, timeframes=[timeframe]
            )
            symbol = _chart_symbol(features, args)
            if symbol:
                candles = read_candles(
                    connection,
                    start=window[0],
                    end=window[1],
                    symbols=[symbol],
                    timeframes=[timeframe],
                )
            bars = read_index_history(
                connection,
                start=window[0],
                end=window[1],
                symbols=[nifty_symbol],
                timeframes=[timeframe],
            )
        finally:
            connection.close()
    else:
        skipped.append(
            f"no database at {context.db_path} - the price charts need stored candles"
        )

    if context.dry_run:
        print(
            "\n".join(
                _chart_plan_lines(
                    context=context,
                    policy=policy,
                    only=only,
                    ledger_path=ledger_path,
                    importance_name=importance_name if not importance_reason else None,
                    symbol=symbol,
                    timeframe=timeframe,
                    trades=len(trades),
                )
            )
        )
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    def add(name: str, builder: Any) -> None:
        """Build one chart, or record why it could not be built (never raise, never pretend)."""
        if not _selected(name, only):
            return
        try:
            figure, note = builder()
        except ChartError as exc:
            skipped.append(f"{name}: {exc}")
            return
        figures[name] = figure
        notes[name] = note

    _build_research_charts(
        add,
        policy=policy,
        trades=trades,
        buckets=buckets,
        min_samples=min_samples,
        importance_payload=importance_payload,
        importance_name=importance_name,
        metric=metric,
    )
    _build_price_charts(
        add,
        policy=policy,
        features=features,
        candles=candles,
        bars=bars,
        trades=trades,
        symbol=symbol,
        nifty_symbol=nifty_symbol,
        timeframe=timeframe,
        option_mas=context.config.indicators.option_trend.moving_averages,
        nifty_mas=min(
            (
                context.config.indicators.nifty_moving_averages.combination_1,
                context.config.indicators.nifty_moving_averages.combination_2,
                context.config.indicators.nifty_moving_averages.combination_3,
            ),
            key=len,
        ),
    )

    if not figures:
        reason = "; ".join(skipped) if skipped else "there is no stored evidence to chart"
        print(f"ANJANEYA-AI: no chart could be built - {reason}", file=sys.stderr)
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    write = not bool(getattr(args, "no_write", False))
    writer = ChartWriter(
        root=context.config.data_dir("reports") / "charts",
        run_id=context.run_context.run_id,
        config_hash=context.config.config_hash,
        mirror=write,
        logger=context.logger,
        fmt=policy.format,
        dpi=policy.dpi,
    )
    stem = writer.stem(ledger=ledger_path.name if ledger_path is not None else None)
    written = writer.write(figures, stem=stem, notes=notes)
    for written_path in written:
        context.run_context.register_artifact(written_path)
    context.run_context.add_note(f"charts: {len(figures)} figure(s) for {stem}")
    for reason in skipped:
        context.run_context.add_note(f"chart skipped: {reason}")
    payload = {
        "run_id": context.run_context.run_id,
        "config_hash": context.config.config_hash,
        "stem": stem,
        "policy": policy.to_dict(),
        "charts": {name: notes.get(name, "") for name in figures},
        "skipped": list(skipped),
        "written": [str(path) for path in written],
        "writer": writer.stats.to_dict(),
    }
    lines = _chart_lines(
        figures=figures, notes=notes, skipped=skipped, written=written, write=write, stem=stem
    )
    print(json.dumps(payload, indent=2, default=str) if args.as_json else "\n".join(lines))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


def _heatmap_note(cells: Any) -> str:
    """A one-line note for the index: how many cells carry a value and how many are preliminary."""
    summary = cells.to_dict()
    return f"{summary['cells']} cell(s), {summary['preliminary_cells']} preliminary"


def _build_research_charts(
    add: Any,
    *,
    policy: ChartPolicy,
    trades: Sequence[Any],
    buckets: Sequence[str],
    min_samples: int,
    importance_payload: Mapping[str, Any],
    importance_name: str,
    metric: str,
) -> None:
    """Queue spec 28's research charts: the two heatmaps, the CE/PE curves, the regime bars and
    the feature-importance bars. Each needs only the ledger or the importance artefact, so a run
    with one but not the other still draws what it can.
    """
    if trades:
        def cycle_heatmap() -> tuple[Any, str]:
            cells = time_of_day_cycle_heatmap(
                trades,
                bucket_order=buckets,
                cycle_order=CYCLE_PHASE_ORDER,
                preliminary_min_samples=min_samples,
            )
            return heatmap(cells, policy=policy), _heatmap_note(cells)

        def weekday_heatmap() -> tuple[Any, str]:
            cells = time_of_day_weekday_heatmap(
                trades, bucket_order=buckets, preliminary_min_samples=min_samples
            )
            return heatmap(cells, policy=policy), _heatmap_note(cells)

        def curves() -> tuple[Any, str]:
            series = side_curves(trades)
            figure = curve_chart(
                series,
                title="CE vs PE cumulative net P&L",
                xlabel="entry time",
                ylabel="cumulative net P&L",
                policy=policy,
            )
            return figure, f"{len(series)} side(s) from {len(trades)} trade(s)"

        def regimes() -> tuple[Any, str]:
            labels, values, samples = regime_bars(trades, order=REGIME_ORDER)
            figure = bar_chart(
                labels,
                values,
                title="net P&L by NIFTY regime at entry",
                xlabel="regime",
                ylabel="net P&L",
                samples=samples,
                policy=policy,
            )
            return figure, f"{len(labels)} regime(s)"

        add("heatmap_time_of_day_cycle_phase", cycle_heatmap)
        add("heatmap_time_of_day_day_of_week", weekday_heatmap)
        add("curves_ce_vs_pe", curves)
        add("regime_performance", regimes)

    if importance_payload:
        def importance() -> tuple[Any, str]:
            items = feature_bars(importance_payload, metric=metric, limit=policy.top_features)
            figure = importance_bars(
                items,
                title=f"feature importance ({metric})",
                value_label=metric,
                policy=policy,
            )
            return figure, f"{len(items)} feature(s) from {importance_name}"

        add("importance_bars", importance)


def _build_price_charts(
    add: Any,
    *,
    policy: ChartPolicy,
    features: pd.DataFrame,
    candles: pd.DataFrame,
    bars: pd.DataFrame,
    trades: Sequence[Any],
    symbol: str | None,
    nifty_symbol: str,
    timeframe: str,
    option_mas: Sequence[Any],
    nifty_mas: Sequence[Any],
) -> None:
    """Queue spec 28's price charts: one option contract, and the NIFTY index.

    The contract chart draws the option's own MA stack (``EMA8 low``/``EMA8 high``/``EMA20``/
    ``EMA55``/``EMA200``) with its level, running and event overlays plus the model's own entries;
    the NIFTY chart draws the index with MA lines, the trend score and the regime shading. Either
    is skipped, with a reason, when its stored series is absent.
    """
    scoped: pd.DataFrame | None = None
    if symbol and not features.empty and "tradingsymbol" in features.columns:
        scoped = features[features["tradingsymbol"] == symbol]

    if symbol and not candles.empty:
        def option_chart() -> tuple[Any, str]:
            data = option_price_data(
                candles,
                scoped,
                symbol=symbol,
                timeframe=timeframe,
                moving_averages=option_mas,
                predictions=_predictions_for(trades, symbol),
            )
            return price_chart(data, policy=policy), f"{data.to_dict()['points']} bar(s)"

        add("price_option", option_chart)

    if not bars.empty:
        def nifty_chart() -> tuple[Any, str]:
            data = nifty_price_data(
                bars,
                features if not features.empty else None,
                symbol=nifty_symbol,
                timeframe=timeframe,
                moving_averages=nifty_mas,
            )
            return price_chart(data, policy=policy), f"{data.to_dict()['points']} bar(s)"

        add("price_nifty", nifty_chart)


def _chart_plan_lines(
    *,
    context: AppContext,
    policy: ChartPolicy,
    only: Sequence[str],
    ledger_path: Path | None,
    importance_name: str | None,
    symbol: str | None,
    timeframe: str,
    trades: int,
) -> list[str]:
    """The ``--dry-run`` plan: which inputs exist, and which charts that makes possible."""
    planned = [name for name in CHART_NAMES if _selected(name, only)]
    lines = [
        f"ANJANEYA-AI chart plan (spec 28) - {timeframe}",
        f"  format     : {policy.format} at {policy.dpi} dpi, {policy.width}x{policy.height} in "
        f"(bars {policy.tall_height} in)",
        f"  ledger     : {ledger_path if ledger_path else 'none - the P&L charts will be skipped'}"
        + (f" ({trades} trade(s))" if ledger_path else ""),
        f"  importance : {importance_name or 'none - the bars chart will be skipped'}",
        f"  symbol     : {symbol or 'none - the price charts will be skipped'}",
        f"  wrote      : nothing (--dry-run); the directory would be "
        f"{context.config.data_dir('reports') / 'charts'}",
        "  would build:",
    ]
    for name in planned:
        family, description = CHART_NAMES[name]
        lines.append(f"    - {name} [{family}] - {description}")
    return lines


def _chart_lines(
    *,
    figures: Mapping[str, Any],
    notes: Mapping[str, str],
    skipped: Sequence[str],
    written: Sequence[Path],
    write: bool,
    stem: str,
) -> list[str]:
    """The human ``--mode charts`` output: what was drawn, what was written, what was skipped."""
    lines = [f"ANJANEYA-AI charts (spec 28) - {len(figures)} figure(s), stem {stem}"]
    for name in CHART_NAMES:
        if name in figures:
            lines.append(f"  {name}: {notes.get(name, '')}")
    for name in figures:
        if name not in CHART_NAMES:
            lines.append(f"  {name}: {notes.get(name, '')}")
    if write:
        for path in written:
            lines.append(f"  wrote      : {path}")
    else:
        lines.append("  wrote      : nothing (--no-write)")
    for reason in skipped:
        lines.append(f"  skipped    : {reason}")
    return lines


# ---------------------------------------------------------------------------
# --mode accumulation (spec 29): the continuous data-accumulation report
# ---------------------------------------------------------------------------


def _accumulation_policy(context: AppContext) -> AccumulationPolicy:
    """Spec 29's rules, read from ``model.accumulation`` plus ``model.data_gates``."""
    return AccumulationPolicy.from_config(context.config)


def _newest_model_record(context: AppContext) -> ModelRecord | None:
    """The newest trained model artefact on disk, as the report describes it.

    The model lives on the filesystem, not in DuckDB, so it is resolved here rather than in
    the storage layer. ``_model_candidates`` already sorts newest-first by modification time
    and ``_metadata_horizon`` reads the horizon out of the artefact's own metadata, so the
    report and ``--mode backtest`` agree about which artefact is "the newest".
    """
    candidates = _model_candidates(context.config.data_dir("models"))
    if not candidates:
        return None
    path = candidates[0]
    try:
        created = datetime.fromtimestamp(
            path.stat().st_mtime, tz=get_timezone(context.config.timezone_name)
        )
    except OSError:  # pragma: no cover - a file that vanished between glob and stat
        created = None
    return ModelRecord(name=path.name, horizon=_metadata_horizon(path), created=created)


def _accumulation_window(context: AppContext, weeks: int) -> tuple[date, date]:
    """The report's inclusive ``[start, end]`` window, ending on today exchange-local time."""
    today = context.calendar.local_date(context.calendar.now())
    return today - timedelta(days=weeks * 7 - 1), today


def _accumulation_plan_lines(
    *,
    context: AppContext,
    policy: AccumulationPolicy,
    weeks: int,
    window: tuple[date, date],
    model: ModelRecord | None,
    write: bool,
) -> list[str]:
    """The ``--dry-run`` plan: which tables would be profiled, and what would be written."""
    trading_days = context.calendar.count_trading_days(window[0], window[1])
    destination = context.config.data_dir("reports") / "accumulation"
    lines = [
        f"ANJANEYA-AI accumulation plan (spec {policy.spec}) - as of {window[1].isoformat()}",
        f"  window     : {window[0].isoformat()} .. {window[1].isoformat()} "
        f"({weeks} week(s), {trading_days} trading day(s))",
        f"  stale after: {policy.stale_days} day(s)",
        f"  coverage   : a day is complete at or above {policy.healthy_coverage_ratio:g}",
        f"  retrain at : {policy.ml_training_min_days} day(s), "
        f"{policy.retrain_min_feature_rows} feature row(s), "
        f"{policy.retrain_min_label_rows} label row(s)",
        f"  newest model: {model.name if model else 'none - no model artefact yet'}",
        f"  would read : {', '.join(ACCUMULATION_TABLES)}",
    ]
    lines.append(
        "  wrote      : nothing (--dry-run); "
        + (f"the directory would be {destination}" if write else "--no-write requested")
    )
    return lines


def _accumulation_payload(
    *, context: AppContext, report: AccumulationReport, written: Sequence[Path], write: bool
) -> dict[str, Any]:
    """The JSON shape ``--json`` prints: the run's identity, then the report's document."""
    return {
        "run_id": context.run_context.run_id,
        "config_hash": context.config.config_hash,
        "mode": "accumulation",
        "written": [str(path) for path in written],
        "wrote": write,
        **report.to_dict(),
    }


def _run_accumulation(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode accumulation``: spec 29's weekly report, read from what is already stored.

    Master-prompt phase 25 asks for a recurring summary of "data captured, coverage, gaps,
    sample sizes, readiness for retraining". This mode measures those five facts and writes
    them under ``data/reports/accumulation/``. It is read-only against the database - it writes
    no tick, candle, feature, label or trade - so a report can never change the dataset it
    describes, and it applies no gate of its own: reporting the gate is the point.
    """
    policy = _accumulation_policy(context)
    weeks = max(1, int(args.weeks if args.weeks is not None else policy.window_weeks))
    window = _accumulation_window(context, weeks)
    trading_days = context.calendar.count_trading_days(window[0], window[1])
    write = not bool(getattr(args, "no_write", False))
    model = _newest_model_record(context)

    if context.dry_run:
        print(
            "\n".join(
                _accumulation_plan_lines(
                    context=context,
                    policy=policy,
                    weeks=weeks,
                    window=window,
                    model=model,
                    write=write,
                )
            )
        )
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)

    if context.db_path.exists():
        connection = connect(context.db_path, read_only=True)
        try:
            inputs = read_accumulation_inputs(connection, as_of=window[1], weeks=weeks)
        finally:
            connection.close()
    else:
        inputs = AccumulationInputs()
        inputs.unavailable.append(f"the database {context.db_path} does not exist yet")
    inputs.newest_model = model

    report = build_accumulation_report(
        inputs,
        policy,
        as_of=window[1],
        window_weeks=weeks,
        trading_days_in_window=trading_days,
    )

    writer = AccumulationWriter(
        root=context.config.data_dir("reports") / "accumulation",
        run_id=context.run_context.run_id,
        config_hash=context.config.config_hash,
        logger=context.logger,
    )
    written = writer.write(report, stem=accumulation_stem(report.as_of)) if write else []
    for path in written:
        context.run_context.register_artifact(path)

    context.logger.info(f"accumulation: {report.headline()}")
    context.run_context.add_note(f"accumulation: {report.headline()}")
    context.run_context.add_note(f"retrain: {report.retrain_reason}")
    for profile in report.tables:
        if profile.unavailable is not None:
            context.run_context.add_note(f"{profile.table} unavailable: {profile.unavailable}")

    lines = render_lines(report)
    if write:
        for path in written:
            lines.append(f"  wrote      : {path}")
    else:
        lines.append("  wrote      : nothing (--no-write)")
    payload = _accumulation_payload(context=context, report=report, written=written, write=write)
    print(json.dumps(payload, indent=2, default=str) if args.as_json else "\n".join(lines))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


# ---------------------------------------------------------------------------
# --mode decide (spec 24): the decision layer - what the policy would do
# ---------------------------------------------------------------------------


def _decision_policy(context: AppContext, args: argparse.Namespace) -> DecisionPolicy:
    """The decision policy, with a policy error reported as a configuration one."""
    try:
        return DecisionPolicy.from_config(
            context.config,
            min_probability=getattr(args, "min_probability", None),
            min_margin=getattr(args, "min_margin", None),
            max_decisions_per_day=getattr(args, "max_decisions_per_day", None),
        )
    except DecisionError as exc:
        raise ConfigError(str(exc)) from None


def _decision_caveats(artefact: ModelArtefact) -> list[str]:
    """The honesty block every decision record carries.

    The label of the model being decided on is part of it, exactly as it is for an explanation:
    a decision made from a model trained below spec 4.3's gate inherits that model's standing,
    and a record must not launder it into something a reader trusts.
    """
    caveats = list(DECISION_CAVEATS)
    if artefact.results_label:
        caveats.append(
            f"the model being decided on is labelled {artefact.results_label} by its own "
            "training run: a decision inherits the model's standing, it does not add to it"
        )
    return caveats


def _decision_rows(
    features: pd.DataFrame,
    predicted: pd.DataFrame,
    classes: Sequence[str],
) -> list[DecisionRow]:
    """One :class:`DecisionRow` per scored bar.

    ``ModelPredictor.predict_frame`` returns one row per input row, in the input's order, so
    the two frames are positionally aligned - the feature frame supplies the identity of each
    bar (which contract, which side, which timeframe) and the prediction frame the
    distribution. The counts are checked rather than assumed: a rotation between the two would
    attribute one bar's probabilities to another, which is the one failure a decision record
    could never recover from.
    """
    if len(features) != len(predicted):
        raise ConfigError(
            f"the model scored {len(predicted)} row(s) from {len(features)} stored row(s); a "
            "prediction cannot be attributed to a bar, so no decision can be recorded"
        )
    for column in ("timestamp", "tradingsymbol", "option_type"):
        if column not in features.columns:
            raise ConfigError(
                f"the stored feature rows carry no {column!r} column, so a decision cannot be "
                "attributed to a bar; build the feature rows with --mode build-features"
            )
    missing = [f"p_{name}" for name in classes if f"p_{name}" not in predicted.columns]
    if missing:
        raise ConfigError(
            f"the model emitted no {missing} column(s); the artefact and its classes disagree"
        )

    stamps = pd.to_datetime(predicted["timestamp"])
    symbols = predicted["tradingsymbol"].astype(str).to_numpy()
    option_types = features["option_type"].to_numpy()
    timeframes = (
        features["timeframe"].to_numpy()
        if "timeframe" in features.columns
        else [None] * len(features)
    )
    probability_columns = {
        name: predicted[f"p_{name}"].to_numpy(dtype="float64") for name in classes
    }
    rows: list[DecisionRow] = []
    for position in range(len(predicted)):
        # Stored timestamps are IST-aware (spec 6.6), so ``.date()`` is already the
        # exchange-local trading day the daily entry budget is counted against.
        moment = stamps.iloc[position].to_pydatetime()
        side = option_types[position]
        rows.append(
            DecisionRow(
                timestamp=moment,
                tradingsymbol=str(symbols[position]),
                option_type=None if side is None else str(side),
                probabilities={
                    name: float(values[position]) for name, values in probability_columns.items()
                },
                timeframe=None if timeframes[position] is None else str(timeframes[position]),
                trading_date=moment.date(),
            )
        )
    return rows


def _decision_plan_lines(
    *,
    context: AppContext,
    artefact: ModelArtefact,
    policy: DecisionPolicy,
    timeframe: str,
    horizon: int | None,
    window: tuple[datetime | None, datetime | None],
    symbols: Sequence[str] | None,
    rows_in_scope: int,
    write: bool,
) -> list[str]:
    """``--dry-run``: what would be decided, from what, and where it would land."""
    start, end = window
    stem = decisions_stem(model_stem(str(artefact.path)))
    root = context.config.data_dir("predictions")
    return [
        f"ANJANEYA-AI decision layer (spec {policy.spec}) - dry run, nothing is written",
        f"  model      : {artefact.path.name} ({artefact.algorithm}, "
        f"horizon {_horizon_text(horizon)})",
        f"  policy     : {policy.describe()}",
        f"  size       : {policy.lots} lot(s) x {policy.lot_size} = {policy.quantity} unit(s)",
        f"  exits      : stop {policy.stop_loss_atr} / target {policy.target_atr} ATR, "
        f"max {policy.max_holding_minutes} min"
        + (", flat at session end" if policy.flat_at_session_end else ""),
        f"  timeframe  : {timeframe}",
        f"  window     : {start or 'beginning'} .. {end or 'newest'}",
        f"  scope      : {', '.join(symbols) if symbols else 'every stored contract'}",
        f"  rows       : {rows_in_scope} stored feature row(s) would be decided",
        "  pipeline   : read features -> score them -> decide every row in time order -> "
        "JSON/Markdown/CSV",
        f"  writes     : {root / f'{stem}_decisions.json|.md|.csv'}"
        if write
        else "  writes     : nothing (--no-write)",
        "  note       : a decision is not an order and not advice; the exit plan is stated, not "
        "simulated (--mode backtest replays it against stored prices)",
    ]


def _decision_payload(
    *,
    context: AppContext,
    report: DecisionReport,
    written: Sequence[Path],
    write: bool,
) -> dict[str, Any]:
    """The JSON payload ``--json`` prints: the run's identity, then the record's document."""
    return {
        "mode": "decide",
        "run_id": context.run_context.run_id,
        "config_hash": context.config.config_hash,
        "write": write,
        "written": [str(path) for path in written],
        **report.to_dict(),
    }



def _run_decide(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode decide``: spec 24's decision layer, over what is already stored.

    Master-prompt phase 24 puts the decision layer "later, separate - only after features,
    model and backtest are validated". This mode is that separate step: it reads stored feature
    rows, scores them with one loaded model artefact, and writes what **one configured rule**
    would have done with each row - ``ENTER_CE`` / ``ENTER_PE`` / ``NO_TRADE``, the reason code,
    the size and the exit plan - to ``data/predictions/``.

    It is read-only against the database (it writes no tick, candle, feature, label or trade),
    it places no order, and it simulates no exit: the stop, the target, the maximum holding and
    the session-end flatten are *recorded* as the plan, and replaying them against stored prices
    with costs is ``--mode backtest``'s job. Like ``--mode backtest`` it never refuses on the
    gate; it labels the run (``BELOW_GATE``, ``EXPLORATORY``, ...) and prints the gate it fell
    short of, because a record of what a rule would do is exactly what an operator needs when
    there is not yet enough data to trust it.
    """
    policy = _decision_policy(context, args)
    timeframe = _resolve_timeframe(context, args.timeframe)
    artefact = _resolve_model_artefact(context, args)
    if artefact is None:
        print(
            "ANJANEYA-AI: no model artefact to decide with under "
            f"{context.config.data_dir('models')} - train one first (--mode train) or pass "
            "--model PATH.",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_MODEL", exit_code=EXIT_OK)
    _check_artefact_timeframe(
        artefact,
        timeframe,
        action="deciding on another timeframe's rows would decide on inputs it never saw",
    )
    horizon = int(artefact.horizon) if artefact.horizon is not None else None
    window = _window(context, args)
    symbols = list(args.symbols) if args.symbols else None
    write = not bool(getattr(args, "no_write", False))
    caveats = _decision_caveats(artefact)
    predictor = ModelPredictor(artefact)

    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        features = read_features(
            connection, start=window[0], end=window[1], symbols=symbols, timeframes=[timeframe]
        )
        if features.empty:
            print(
                f"ANJANEYA-AI: no stored {timeframe} feature rows in scope - build them first "
                "(--mode build-features).",
                file=sys.stderr,
            )
            return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)
        days = len(trading_days(features, timezone_name=context.config.timezone_name))
        gate = _gate_status(context, days)
        verdict = gate_verdict(gate.stage, days=days, forced=True)
        if context.dry_run:
            print(
                "\n".join(
                    _decision_plan_lines(
                        context=context,
                        artefact=artefact,
                        policy=policy,
                        timeframe=timeframe,
                        horizon=horizon,
                        window=window,
                        symbols=symbols,
                        rows_in_scope=int(len(features)),
                        write=write,
                    )
                )
            )
            return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)
        try:
            predicted = predictor.predict_frame(features)
        except ModelError as exc:
            raise ConfigError(str(exc)) from None
    finally:
        connection.close()

    rows = _decision_rows(features, predicted, artefact.classes)
    report = build_decision_report(
        rows,
        policy,
        as_of=max(row.trading_date for row in rows),
        modelled_classes=artefact.classes,
        timeframe=timeframe,
        horizon_minutes=horizon,
        model={"name": artefact.path.name, **artefact.describe()},
        gate=gate,
        verdict=verdict,
        window=window,
        symbols=tuple(symbols or ()),
        caveats=caveats,
    )

    writer = DecisionWriter(
        root=context.config.data_dir("predictions"),
        run_id=context.run_context.run_id,
        config_hash=context.config.config_hash,
        logger=context.logger,
    )
    stem = decisions_stem(model_stem(str(artefact.path)))
    written = writer.write(report, stem=stem) if write else []
    for path in written:
        context.run_context.register_artifact(path)

    context.logger.info(f"decide: {report.headline()} [{report.results_label}]")
    context.run_context.add_note(f"decide: {report.headline()}")
    context.run_context.add_note(f"decide verdict: {verdict.describe()}")
    for caveat in caveats:
        context.run_context.add_note(caveat)

    lines = render_decision_lines(report)
    if write:
        for path in written:
            lines.append(f"  wrote      : {path}")
    else:
        lines.append("  wrote      : nothing (--no-write)")
    payload = _decision_payload(context=context, report=report, written=written, write=write)
    print(json.dumps(payload, indent=2, default=str) if args.as_json else "\n".join(lines))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)



def _run_planned_mode(context: AppContext, mode: str) -> int:
    """Explain that a mode is not implemented yet, and exit with code 3."""
    plan = MODE_PLAN.get(mode, "planned in a later phase")
    context.logger.warning(f"mode {mode!r} is not implemented yet: {plan}")
    print(
        f"ANJANEYA-AI: mode '{mode}' is not implemented in this phase.\n"
        f"  planned work : {plan}\n"
        f"  available now: {AVAILABLE_MODES}  (offline, no credentials needed)\n"
        f"  see          : docs/phase_reports/ for the current phase report",
        file=sys.stderr,
    )
    return finalize_run(context, status="NOT_IMPLEMENTED", exit_code=EXIT_NOT_IMPLEMENTED)



def _dispatch(context: AppContext, args: argparse.Namespace) -> int:
    """Route a validated context to the mode that was asked for."""
    if context.mode == "status":
        return _run_status(context, as_json=args.as_json)
    if context.mode == "dataset":
        return _run_dataset(context, args)
    if context.mode == "train":
        return _run_train(context, args)
    if context.mode == "backtest":
        return _run_backtest(context, args)
    if context.mode == "report":
        return _run_report(context, args)
    if context.mode == "explain":
        return _run_explain(context, args)
    if context.mode == "importance":
        return _run_importance(context, args)
    if context.mode == "charts":
        return _run_charts(context, args)
    if context.mode == "accumulation":
        return _run_accumulation(context, args)
    if context.mode == "decide":
        return _run_decide(context, args)
    if context.dry_run:
        plan = MODE_PLAN.get(context.mode, "planned in a later phase")
        context.logger.info(f"dry-run: would execute {context.mode!r} -> {plan}")
        print(f"ANJANEYA-AI dry-run: {context.mode} -> {plan}")
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)
    return _run_planned_mode(context, context.mode)



def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 1 failed, 2 config, 3 planned)."""
    args = parse_args(
        argv,
        app_name=APP_AI,
        description=DESCRIPTION,
        configure_parser=_configure_ai_parser,
    )
    try:
        context = bootstrap_app(APP_AI, args)
    except (ConfigError, SchemaVersionError) as exc:
        print(f"ANJANEYA-AI configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the CLI
        print(f"ANJANEYA-AI failed during startup: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    try:
        return _dispatch(context, args)
    except ConfigError as exc:
        print(f"ANJANEYA-AI configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # noqa: BLE001 - a crash must still be an exit code
        print(
            f"ANJANEYA-AI failed during {context.mode!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

