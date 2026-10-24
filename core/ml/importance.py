"""Spec 27's feature importance: where the model's information actually came from.

Spec 27 asks for three things after training - **native importance**, **permutation
importance** and *optional* **SHAP** - and then for the question they exist to answer:
whether the predictive information comes from the NIFTY regime, the option's own moving
averages, candle behaviour, support/resistance, market structure, IV, the greeks, OI, the
option residual, time of day, expiry-cycle phase or time to expiry. Its last two sentences
are the rules: "Do not assume. Measure. Report with confidence intervals where practical."
This module is that sentence kept:

* **native importance is the model's own report, read off the artefact.** LightGBM's gain
  figures are recomputed from the loaded booster (not copied out of a metadata block), so a
  number here can be re-derived from the ``.txt`` a training run wrote and nothing in this
  module can quietly disagree with the model it describes.
* **permutation importance is measured here**, one column at a time, on a *named* sample of
  the stored matrix, with a proper scoring rule (log loss by default - accuracy is a
  documented alternative) and a fixed seed. :func:`permutation_importance` keeps the
  per-repeat deltas rather than only their mean, so a feature's interval and a *source's*
  interval are both arithmetic on repeated measurements instead of a judgement call.
* **SHAP is optional in the strongest sense.** The package is not a dependency of ANJANEYA
  and is not installed here; ``model.importance.shap`` decides what that means - ``auto``
  records why it did not run, ``on`` is a configuration error, ``off`` never imports it. A
  run that cannot measure SHAP still measures the other two: an absent *optional* measure is
  reported, never faked.
* **the sources are configuration.** ``model.importance.topics`` maps stored feature groups
  and columns onto the sources spec 27 names (checked against the feature contract at load,
  a column claimed twice refused). A feature no topic claims is reported under
  :data:`UNMAPPED_TOPIC_KEY` - never dropped, never silently attributed to a source that did
  not measure it, and never quietly counted in another topic's total.

What importance is **not**: a permuted column is scored with every other column held at its
stored value, so a feature that duplicates or combines others keeps less of the credit than
its influence suggests (:data:`CORRELATED_FEATURES_CAVEAT` says so in every artefact); and a
ranking of *this model's* inputs is not a claim that a feature moves a market. Two honesty
knobs are therefore structural rather than textual:

* a feature the sample cannot measure - every value ``NULL``, or a single constant value -
  is reported as **not measured** with the reason, never as a permutation delta of 0, which
  would read as "this feature carries no information" when the truth is "these rows cannot
  say";
* the sample is named. Spec 22's reserved holdout days come first (they are the only rows
  the *split* promises the final model did not train on); a dataset too short for a holdout
  falls back to the stored matrix with :data:`IN_SAMPLE_CAVEAT` attached, because a number
  measured in-sample is an honest number only if it is labelled one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.constants import FEATURE_COLUMNS
from core.features.feature_builder import FEATURE_GROUPS
from core.ml.dataset import CODE_COLUMN
from core.ml.evaluation import EvaluationError, accuracy, log_loss
from core.ml.trainer import IMPORTANCE_TYPE
from core.ml.walk_forward import holdout_frame

__all__ = [
    "CORRELATED_FEATURES_CAVEAT",
    "DEFAULT_TOPICS",
    "IMPORTANCE_CAVEAT",
    "IMPORTANCE_METRICS",
    "IN_SAMPLE_CAVEAT",
    "RESERVED_IMPORTANCE_GROUPS",
    "SAMPLE_SOURCES",
    "SHAP_FAILED",
    "SHAP_MEASURED",
    "SHAP_MODES",
    "SHAP_NOT_INSTALLED_HINT",
    "SHAP_SKIPPED",
    "SHAP_UNAVAILABLE",
    "UNMAPPED_TOPIC_KEY",
    "UNMAPPED_TOPIC_TITLE",
    "FeatureImportance",
    "ImportanceError",
    "ImportancePolicy",
    "ImportanceReport",
    "ImportanceSample",
    "ImportanceSource",
    "ImportanceTopic",
    "PermutationResult",
    "ShapResult",
    "TopicImportance",
    "build_importance_report",
    "measurement_caveat",
    "metric_score",
    "native_importance",
    "permutation_importance",
    "select_sample",
    "shap_importance",
]
#: Feature groups importance can never take as a *topic*: identity is how a row is found
#: (and is not a model input at all), and provenance describes the row's writer, not the
#: market - ranking either would answer a question nobody asked.
RESERVED_IMPORTANCE_GROUPS: frozenset[str] = frozenset({"identity", "provenance"})

#: The scoring rules a permutation delta may be measured with. Both are faithful scoring
#: rules over a class distribution; log loss is the default because it is the metric the
#: trainer already reports, so the two numbers are comparable.
IMPORTANCE_METRICS: tuple[str, ...] = ("log_loss", "accuracy")

#: ``model.importance.shap`` - whether the optional attribution is measured.
SHAP_MODES: tuple[str, ...] = ("auto", "on", "off")

#: Which rows permutation importance is measured on.
SAMPLE_SOURCES: tuple[str, ...] = ("holdout", "matrix")

#: The topic every feature no configured source claims is reported under. It is a *reporting*
#: bucket, never a claim about the market: spec 27's list is not exhaustive, and a feature
#: that is not one of its sources must be visible as exactly that.
UNMAPPED_TOPIC_KEY = "unmapped"
UNMAPPED_TOPIC_TITLE = "unmapped (no configured source claims these features)"

#: Printed on (and stored with) every importance artefact.
IMPORTANCE_CAVEAT = (
    "importance ranks the information *this model* used from *these inputs*; it is not a "
    "claim that a feature moves a market, and a model labelled BELOW_GATE is ranked as the "
    "unvalidated model it is"
)
#: A permuted column is scored with every other column held as stored.
CORRELATED_FEATURES_CAVEAT = (
    "a permuted column is scored with all other columns held at their stored values: a "
    "feature that duplicates, or is a combination of, other features keeps less of the "
    "credit than its influence suggests, and one of a duplicated pair may take both shares"
)
#: Attached when the rows measured are rows the final model was fitted on.
IN_SAMPLE_CAVEAT = (
    "the sample is the stored matrix itself, which the final model was fitted on: every "
    "number here is measured in-sample and is optimistic by an unknown amount ({reason})"
)
#: What to do about a missing optional dependency.
SHAP_NOT_INSTALLED_HINT = (
    "shap is an optional extra of ANJANEYA (pip install -e '.[explain]'): native and "
    "permutation importance need no such package"
)


class ImportanceError(RuntimeError):
    """Raised when importance cannot be measured honestly (no rows, an unknown metric, ...)."""


# ---------------------------------------------------------------------------
# The policy: which sources a run reports, and how it measures them
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImportanceSource:
    """One named source inside a topic: a stored feature column and the group it came from."""

    column: str
    #: ``None`` for a column the configuration named explicitly.
    group: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {"column": self.column, "group": self.group}


@dataclass(frozen=True, slots=True)
class ImportanceTopic:
    """A resolved topic: one source spec 27 names, once checked against the contract."""

    key: str
    title: str
    sources: tuple[ImportanceSource, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """The stored feature columns this topic claims, in claim order."""
        return tuple(source.column for source in self.sources)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "key": self.key,
            "title": self.title,
            "columns": list(self.columns),
            "sources": [source.to_dict() for source in self.sources],
        }


def _sources(groups: Sequence[str], columns: Sequence[str] = ()) -> tuple[ImportanceSource, ...]:
    """The sources of ``groups`` (in group order) followed by ``columns``, or an error.

    Every name is checked against the feature contract here rather than trusted: a hand-built
    policy (a test, a notebook) and the loaded configuration both pass through this function,
    so neither can claim a feature that does not exist.
    """
    items: list[ImportanceSource] = []
    for name in groups:
        if name in RESERVED_IMPORTANCE_GROUPS:
            raise ImportanceError(
                f"{name!r} is a reserved feature group: identity is how a row is found and "
                "provenance describes its writer, so neither is a source of market information"
            )
        if name not in FEATURE_GROUPS:
            raise ImportanceError(
                f"unknown feature group {name!r}; known groups: {sorted(FEATURE_GROUPS)}"
            )
        items.extend(ImportanceSource(column=column, group=name) for column in FEATURE_GROUPS[name])
    for name in columns:
        if name not in FEATURE_COLUMNS:
            raise ImportanceError(
                f"{name!r} is not a column of the stored feature row (core/constants.py)"
            )
        items.append(ImportanceSource(column=name, group=None))
    return tuple(items)
#: The default map from spec 27's own list of sources onto the stored feature contract. The
#: last four topics are columns the spec's list does not name (the option side, the day of
#: the week, moneyness/volume and the price-action levels): they exist so no model input is
#: left unattributed, and they say "not named by spec 27" in their titles rather than
#: borrowing a source they are not.
DEFAULT_TOPICS: tuple[ImportanceTopic, ...] = (
    ImportanceTopic("nifty_regime", "NIFTY regime", _sources(("nifty",))),
    ImportanceTopic("option_ma", "option moving averages", _sources(("option",))),
    ImportanceTopic("candle_behaviour", "candle behaviour", _sources(("candle",))),
    ImportanceTopic(
        "support_resistance", "support / resistance", _sources(("support_resistance",))
    ),
    ImportanceTopic("structure", "market structure", _sources(("structure",))),
    ImportanceTopic(
        "price_action",
        "price action, levels (not named by spec 27)",
        _sources(("levels", "level_events")),
    ),
    ImportanceTopic("iv", "implied volatility", _sources((), ("iv", "iv_change"))),
    ImportanceTopic("greeks", "greeks", _sources((), ("delta", "gamma", "theta", "vega"))),
    ImportanceTopic("oi", "open interest", _sources((), ("oi", "oi_change"))),
    ImportanceTopic(
        "option_residual",
        "option residual (expected vs actual move)",
        _sources((), ("expected_option_move", "actual_option_move", "option_residual")),
    ),
    ImportanceTopic("time_of_day", "time of day", _sources((), ("time_of_day_bucket",))),
    ImportanceTopic("cycle_phase", "expiry-cycle phase", _sources((), ("expiry_cycle_phase",))),
    ImportanceTopic("time_to_expiry", "time to expiry", _sources((), ("days_to_expiry",))),
    ImportanceTopic(
        "day_of_week", "day of week (not named by spec 27)", _sources((), ("day_of_week",))
    ),
    ImportanceTopic(
        "option_contract", "option side (not named by spec 27)", _sources((), ("option_type",))
    ),
    ImportanceTopic(
        "option_context",
        "option context: moneyness and volume (not named by spec 27)",
        _sources((), ("distance_from_atm", "volume")),
    ),
)
@dataclass(frozen=True, slots=True)
class ImportancePolicy:
    """Every rule an importance run applies, read from the configuration in one place.

    The defaults mirror ``config/model.yaml``'s ``importance:`` block, so a caller with no
    configuration still gets the shipped structure rather than a second, silently different
    one - and :meth:`__post_init__` re-checks the invariants the loader enforces, because a
    hand-built policy never passes through the loader.
    """

    #: The master-spec section the structure comes from (printed in every artefact).
    spec: str = "27"
    #: Decimals for every number printed (trailing zeros are trimmed).
    digits: int = 4
    #: How many times each column is permuted. Two is the floor: one repeat has no spread,
    #: and an interval is exactly what spec 27 asked for.
    repeats: int = 5
    #: The scoring rule a permutation delta is measured with (``log_loss`` | ``accuracy``).
    metric: str = "log_loss"
    #: Confidence level of every interval (two-sided, Student's t over the repeats).
    confidence: float = 0.95
    #: ``holdout`` (spec 22's reserved days) or ``matrix`` (every stored row).
    sample: str = "holdout"
    #: The most rows a measurement uses (the newest ones; the rest are counted, not hidden).
    max_rows: int = 2000
    #: What to do about SHAP: ``auto`` | ``on`` | ``off``.
    shap: str = "auto"
    #: The most rows SHAP is asked to explain.
    shap_rows: int = 500
    #: Seed for the permutations. ``None`` means "use the model's own seed".
    seed: int | None = None
    #: The sources spec 27's question is answered with.
    topics: tuple[ImportanceTopic, ...] = DEFAULT_TOPICS

    def __post_init__(self) -> None:
        if not self.topics:
            raise ImportanceError("a policy with no topics would report no source at all")
        keys = [topic.key for topic in self.topics]
        if len(set(keys)) != len(keys):
            raise ImportanceError(f"importance topic keys must be unique, got {keys}")
        if UNMAPPED_TOPIC_KEY in keys:
            raise ImportanceError(
                f"{UNMAPPED_TOPIC_KEY!r} is reserved for the features no configured source "
                "claims; rename the configured topic"
            )
        seen: dict[str, str] = {}
        for topic in self.topics:
            if not topic.sources:
                raise ImportanceError(f"importance topic {topic.key!r} claims no feature")
            for source in topic.sources:
                if source.group in RESERVED_IMPORTANCE_GROUPS:
                    raise ImportanceError(
                        f"importance topic {topic.key!r} claims {source.column!r} through the "
                        f"reserved feature group {source.group!r}"
                    )
                if source.column not in FEATURE_COLUMNS:
                    raise ImportanceError(
                        f"importance topic {topic.key!r} claims {source.column!r}, which is "
                        "not a column of the stored feature row (core/constants.py)"
                    )
                if source.column in seen:
                    raise ImportanceError(
                        f"feature column {source.column!r} is claimed twice (topics "
                        f"{seen[source.column]!r} and {topic.key!r}); one column, one source"
                    )
                seen[source.column] = topic.key
        if self.repeats < 2:
            raise ImportanceError(
                f"importance.repeats is {self.repeats}; at least 2 permutations are needed, "
                "because one repeat has no spread to put an interval on"
            )
        if self.metric not in IMPORTANCE_METRICS:
            raise ImportanceError(
                f"unknown importance metric {self.metric!r}; supported: {list(IMPORTANCE_METRICS)}"
            )
        if not 0.0 < float(self.confidence) < 1.0:
            raise ImportanceError(
                f"importance.confidence must be inside (0, 1), got {self.confidence}"
            )
        if self.digits < 0:
            raise ImportanceError("importance.digits must not be negative")
        if self.sample not in SAMPLE_SOURCES:
            raise ImportanceError(
                f"unknown importance sample {self.sample!r}; supported: {list(SAMPLE_SOURCES)}"
            )
        if self.max_rows < 1:
            raise ImportanceError("importance.max_rows must be at least 1")
        if self.shap not in SHAP_MODES:
            raise ImportanceError(
                f"unknown importance.shap mode {self.shap!r}; supported: {list(SHAP_MODES)}"
            )
        if self.shap_rows < 1:
            raise ImportanceError("importance.shap_rows must be at least 1")
        if self.seed is not None and int(self.seed) < 0:
            raise ImportanceError("importance.seed must not be negative")
    @classmethod
    def from_config(cls, config: Any) -> ImportancePolicy:
        """Read the policy out of a loaded :class:`core.config_loader.AnjaneyaConfig`.

        A configuration without the block (an older file, a hand-made object) falls back to
        the shipped defaults rather than failing: this is a reporting structure, and nothing
        downstream depends on it.
        """
        block = getattr(getattr(config, "model", None), "importance", None)
        if block is None:
            return cls()
        return cls(
            spec=str(block.spec),
            digits=int(block.digits),
            repeats=int(block.repeats),
            metric=str(block.metric),
            confidence=float(block.confidence),
            sample=str(block.sample),
            max_rows=int(block.max_rows),
            shap=str(block.shap),
            shap_rows=int(block.shap_rows),
            seed=None if block.seed is None else int(block.seed),
            topics=tuple(
                ImportanceTopic(
                    key=str(topic.key),
                    title=str(topic.title),
                    sources=_sources(tuple(topic.groups), tuple(topic.columns)),
                )
                for topic in block.topics
            ),
        )

    # -- what the policy claims ----------------------------------------------
    @property
    def columns(self) -> tuple[str, ...]:
        """Every stored feature column some topic claims, in topic order."""
        return tuple(column for topic in self.topics for column in topic.columns)

    def topic(self, key: str) -> ImportanceTopic | None:
        """The topic with ``key``, or ``None``."""
        for topic in self.topics:
            if topic.key == key:
                return topic
        return None

    def topic_for(self, column: str) -> str | None:
        """The key of the topic claiming ``column``, or ``None`` when nothing claims it."""
        for topic in self.topics:
            if column in topic.columns:
                return topic.key
        return None

    def unmapped(self, columns: Sequence[str]) -> tuple[str, ...]:
        """The columns of ``columns`` no topic claims - reported, never reattributed."""
        seen = set(self.columns)
        return tuple(name for name in columns if name not in seen)

    def random_seed(self, default: int = 42) -> int:
        """The seed the permutations use: ``importance.seed``, else the model's own."""
        return int(self.seed) if self.seed is not None else int(default)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every artefact so a number stays traceable."""
        return {
            "spec": self.spec,
            "digits": self.digits,
            "repeats": self.repeats,
            "metric": self.metric,
            "confidence": self.confidence,
            "sample": self.sample,
            "max_rows": self.max_rows,
            "shap": self.shap,
            "shap_rows": self.shap_rows,
            "seed": self.seed,
            "columns": list(self.columns),
            "topics": [topic.to_dict() for topic in self.topics],
        }
# ---------------------------------------------------------------------------
# The measurements
# ---------------------------------------------------------------------------
def metric_score(
    metric: str, outcomes: Any, probabilities: Any, *, classes: Sequence[str]
) -> float:
    """One scoring rule over a class distribution, dispatched by name.

    Both metrics come from :mod:`core.ml.evaluation`, the same module the trainer scores its
    folds with, so "log loss" means one thing across the project.
    """
    if metric == "log_loss":
        scorer = log_loss
    elif metric == "accuracy":
        scorer = accuracy
    else:
        raise ImportanceError(
            f"unknown importance metric {metric!r}; supported: {list(IMPORTANCE_METRICS)}"
        )
    try:
        return float(scorer(outcomes, probabilities, classes=classes))
    except EvaluationError as exc:
        raise ImportanceError(str(exc)) from None


def _penalty(metric: str, score: float) -> float:
    """The score oriented so that **larger means more information lost**.

    Permutation importance is only readable if a positive delta always means "permuting this
    column made the model worse". Log loss already points that way (bigger is worse); accuracy
    points the other way, so it is negated here rather than left for a reader to remember.
    """
    return -float(score) if metric == "accuracy" else float(score)


def native_importance(artefact: Any) -> dict[str, float]:
    """The model's own gain importance, recomputed from the loaded booster.

    The numbers come out of the booster rather than out of the metadata JSON beside it, so
    this module cannot report something the model itself does not say: metadata is a copy,
    the ``.txt`` is the artefact.
    """
    booster = getattr(artefact, "booster", None)
    columns = tuple(str(name) for name in getattr(artefact, "feature_columns", ()) or ())
    if booster is None or not columns:
        raise ImportanceError(
            "this artefact carries no booster or no feature list, so it has no native "
            "importance to report"
        )
    values = np.asarray(booster.feature_importance(importance_type=IMPORTANCE_TYPE), dtype="float64")
    numbers = [float(value) for value in values.ravel()]
    if len(numbers) != len(columns):
        raise ImportanceError(
            f"the booster reports {len(numbers)} importance value(s) for {len(columns)} "
            "feature column(s); the artefact and its metadata disagree, so no ranking here "
            "could be trusted"
        )
    return dict(zip(columns, numbers, strict=True))


def _interval(values: Any, *, confidence: float) -> tuple[float, float, float]:
    """``(mean, low, high)`` of repeated measurements, two-sided Student's t.

    Spec 27 asks for confidence intervals "where practical": this is the practical one. The
    repeats are the *same* measurement repeated with a different permutation, so their spread
    is the sampling noise of that measurement - which is what the interval is about. It is not
    a bootstrap over rows, and nothing here pretends otherwise.
    """
    numbers = np.asarray(values, dtype="float64").ravel()
    finite = numbers[np.isfinite(numbers)]
    if finite.size < 2:
        raise ImportanceError(
            "an interval needs at least 2 measured repeats; check importance.repeats"
        )
    mean = float(finite.mean())
    error = float(finite.std(ddof=1) / np.sqrt(finite.size))
    if error == 0.0:
        return mean, mean, mean
    half = float(stats.t.ppf(0.5 + float(confidence) / 2.0, finite.size - 1) * error)
    return mean, mean - half, mean + half


def _group_of(column: str) -> str | None:
    """The stored feature group ``column`` belongs to, or ``None`` when it is not grouped."""
    for name, columns in FEATURE_GROUPS.items():
        if column in columns:
            return name
    return None


def _outcome_codes(frame: pd.DataFrame, *, column: str) -> np.ndarray:
    """The observed class codes of ``frame`` (the dataset's own ``label_code`` column)."""
    if column not in frame.columns:
        raise ImportanceError(
            f"the sample carries no {column!r} column, so a permutation cannot be scored: "
            "pass the rows --mode dataset builds (features joined to their labels)"
        )
    values = frame[column].to_numpy()
    if values.size and not np.issubdtype(values.dtype, np.integer):
        try:
            values = values.astype("int64")
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ImportanceError(f"the {column!r} column is not integer class codes: {exc}") from None
    return values.astype("int64", copy=False)
# ---------------------------------------------------------------------------
# The sample: which rows a permutation is measured on
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ImportanceSample:
    """The rows a measurement used, and the honest reasons they are those rows."""

    #: ``holdout`` (spec 22's reserved days) or ``matrix`` (every stored row).
    name: str
    #: The rows, newest-last, positional index reset (so a permutation is reproducible).
    frame: pd.DataFrame
    #: How many rows the source had, before ``max_rows`` cut it down.
    rows_available: int
    #: How many rows a measurement actually uses.
    rows_used: int
    #: ``True`` when ``max_rows`` removed older rows.
    capped: bool
    #: Trading days the used rows cover, and their window.
    days: int = 0
    start: datetime | None = None
    end: datetime | None = None
    #: Why this sample was chosen (set when it is the fallback one).
    reason: str | None = None

    @property
    def rows(self) -> int:
        """The number of rows a measurement runs on."""
        return int(len(self.frame))

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape - every count that says what was measured, and no rows."""
        return {
            "name": self.name,
            "rows_available": self.rows_available,
            "rows_used": self.rows_used,
            "capped": self.capped,
            "days": self.days,
            "first_timestamp": self.start.isoformat() if self.start is not None else None,
            "last_timestamp": self.end.isoformat() if self.end is not None else None,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias of :meth:`describe`, so the artefact envelope reads uniformly."""
        return self.describe()


def select_sample(dataset: Any, split: Any = None, *, policy: ImportancePolicy) -> ImportanceSample:
    """Pick the rows permutation importance runs on, and say which kind they are.

    ``importance.sample: holdout`` means spec 22's reserved days - the only rows the *split*
    promises the final model was not fitted on - and falls back to the whole stored matrix
    with :attr:`ImportanceSample.reason` set, because "not enough days for a holdout" must not
    turn into "measured in-sample, quietly". ``importance.sample: matrix`` asks for the whole
    matrix on purpose, and the artefact then carries the in-sample caveat too.

    The newest ``max_rows`` rows win when the sample is too big: they are the rows closest to
    the conditions a prediction will be made in, and the count of dropped rows is reported
    rather than folded away.
    """
    frame = dataset.frame
    name, reason = "matrix", None
    if policy.sample == "holdout":
        reserved = holdout_frame(dataset, split) if split is not None else frame.iloc[0:0]
        if not reserved.empty:
            frame, name = reserved, "holdout"
        else:
            reason = str(getattr(split, "reason", None) or "the split built no holdout rows")
    available = int(len(frame))
    ordered = frame
    if "timestamp" in frame.columns and available:
        ordered = frame.sort_values("timestamp", kind="stable")
    capped = available > int(policy.max_rows)
    if capped:
        ordered = ordered.tail(int(policy.max_rows))
    used = int(len(ordered))
    start = end = None
    days = 0
    if used and "timestamp" in ordered.columns:
        stamps = pd.to_datetime(ordered["timestamp"])
        start, end = stamps.iloc[0], stamps.iloc[-1]
        days = len({stamp.date() for stamp in stamps})
    return ImportanceSample(
        name=name,
        frame=ordered.reset_index(drop=True),
        rows_available=available,
        rows_used=used,
        capped=capped,
        days=days,
        start=start,
        end=end,
        reason=reason,
    )


def measurement_caveat(sample: ImportanceSample) -> str:
    """The sentence an artefact carries about *which* rows were measured."""
    if sample.name == "holdout":
        return (
            "the sample is spec 22's reserved holdout - the days the split kept out of the "
            "final model's training window - so these deltas are measured out-of-sample with "
            "respect to that split"
        )
    return IN_SAMPLE_CAVEAT.format(
        reason=sample.reason or "the configuration asked for the stored matrix"
    )
# ---------------------------------------------------------------------------
# Permutation importance (measured here, at a seed, on a named sample)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PermutationResult:
    """Every column permuted ``repeats`` times over one sample, at one seed."""

    #: The model's feature columns, in the model's own order.
    columns: tuple[str, ...]
    repeats: int
    metric: str
    confidence: float
    #: The score of the untouched sample (the raw metric, in the metric's own direction).
    baseline: float
    #: Shape ``(len(columns), repeats)``: how much *worse* the model scored with that column's
    #: rows shuffled. Always oriented so a larger number means more information lost - which
    #: is why :func:`_penalty` exists.
    deltas: np.ndarray
    rows: int
    seed: int
    #: Column -> why it could not be measured at all. Its deltas are ``NaN``.
    blocked: dict[str, str] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """Which way the *raw* metric points (the deltas are always "worse is bigger")."""
        return "higher_is_better" if self.metric == "accuracy" else "lower_is_better"

    @property
    def measured_columns(self) -> tuple[str, ...]:
        """The columns a permutation could actually measure."""
        return tuple(name for name in self.columns if name not in self.blocked)

    def measured(self, column: str) -> bool:
        """Whether ``column`` has a measurement at all."""
        return column not in self.blocked

    def values(self, column: str) -> np.ndarray:
        """The per-repeat deltas of ``column`` (all ``NaN`` where it is blocked)."""
        if column not in self.columns:
            raise ImportanceError(f"{column!r} is not a feature of this measurement")
        return self.deltas[self.columns.index(column)]

    def summary(self, column: str) -> dict[str, Any]:
        """``{mean, std, ci_low, ci_high, measured, reason}`` for one column."""
        if column not in self.columns:
            raise ImportanceError(f"{column!r} is not a feature of this measurement")
        if column in self.blocked:
            return {
                "mean": None,
                "std": None,
                "ci_low": None,
                "ci_high": None,
                "measured": False,
                "reason": self.blocked[column],
            }
        numbers = self.values(column)
        mean, low, high = _interval(numbers, confidence=self.confidence)
        finite = numbers[np.isfinite(numbers)]
        return {
            "mean": mean,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "ci_low": low,
            "ci_high": high,
            "measured": True,
            "reason": None,
        }

    def ranking(self, *, limit: int | None = None) -> list[tuple[str, float]]:
        """The measured columns, largest delta first (a blocked column is not ranked)."""
        ranked = sorted(
            ((name, float(np.nanmean(self.values(name)))) for name in self.measured_columns),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[: int(limit)] if limit is not None else ranked

    def group_deltas(self, columns: Sequence[str]) -> np.ndarray | None:
        """The per-repeat *total* delta of ``columns``, or ``None`` when none is measured.

        Summing the repeats rather than the means is what lets a topic carry an interval: the
        spread of a sum of repeatedly measured quantities is measured, not guessed.
        """
        present = [name for name in columns if name in self.columns and name not in self.blocked]
        if not present:
            return None
        return np.nansum(np.vstack([self.values(name) for name in present]), axis=0)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape of the *measurement* (per-feature numbers live in the report)."""
        return {
            "metric": self.metric,
            "direction": self.direction,
            "repeats": self.repeats,
            "confidence": self.confidence,
            "baseline": self.baseline,
            "rows": self.rows,
            "seed": self.seed,
            "features": len(self.columns),
            "measured": len(self.measured_columns),
            "blocked": dict(sorted(self.blocked.items())),
        }
def permutation_importance(
    predictor: Any,
    frame: pd.DataFrame,
    *,
    classes: Sequence[str],
    policy: ImportancePolicy,
    seed: int | None = None,
) -> PermutationResult:
    """Score ``frame``, shuffle one feature at a time, and report how much that hurt.

    The matrix is built *once*, by the same :class:`core.ml.predictor.ModelPredictor` a
    backtest uses - training column order, training categorical vocabulary - and each
    permutation rewrites one column of a copy of it. Nothing is refitted, nothing is scaled and
    nothing is resampled: between the baseline and a repeat, the only thing that changes is
    which row each value of one column sits beside.
    """
    if frame.empty:
        raise ImportanceError("no rows to permute: an importance measurement needs a sample")
    names = tuple(str(name) for name in getattr(predictor, "feature_columns", ()) or ())
    if not names:
        raise ImportanceError("the predictor names no feature columns, so nothing can be ranked")
    try:
        matrix = predictor.matrix(frame)
    except ImportanceError:
        raise
    except Exception as exc:  # ModelError, TrainingError: re-raise in this module's words
        raise ImportanceError(str(exc)) from exc
    codes = _outcome_codes(frame, column=CODE_COLUMN)
    if codes.size != len(matrix):
        raise ImportanceError(
            f"the sample carries {codes.size} class code(s) for {len(matrix)} matrix row(s); "
            "the sample and its labels are out of step"
        )
    chosen_seed = policy.random_seed() if seed is None else int(seed)
    generator = np.random.default_rng(chosen_seed)
    probabilities = _score(predictor, matrix, classes=classes)
    baseline = metric_score(policy.metric, codes, probabilities, classes=classes)
    baseline_penalty = _penalty(policy.metric, baseline)

    deltas = np.full((len(names), int(policy.repeats)), np.nan, dtype="float64")
    blocked: dict[str, str] = {}
    for position, name in enumerate(names):
        column = matrix[name]
        if bool(column.isna().all()):
            blocked[name] = (
                "every value of this column is NULL in the sample, so the model is fed the "
                "same missing value either way"
            )
            continue
        if int(column.nunique(dropna=True)) <= 1:
            blocked[name] = (
                "this column holds a single value in the sample, so permuting it cannot "
                "change the score"
            )
            continue
        values = column.to_numpy()
        work = matrix.copy()
        for repeat in range(int(policy.repeats)):
            work[name] = _permuted_column(column, values, generator.permutation(len(values)))
            score = metric_score(
                policy.metric, codes, _score(predictor, work, classes=classes), classes=classes
            )
            deltas[position, repeat] = _penalty(policy.metric, score) - baseline_penalty
    return PermutationResult(
        columns=names,
        repeats=int(policy.repeats),
        metric=policy.metric,
        confidence=float(policy.confidence),
        baseline=float(baseline),
        deltas=deltas,
        rows=int(len(matrix)),
        seed=int(chosen_seed),
        blocked=blocked,
    )


def _permuted_column(column: pd.Series, values: np.ndarray, order: np.ndarray) -> pd.Series:
    """``column``'s values re-ordered by ``order``, **keeping the column's own dtype**.

    A categorical column has to be rebuilt as a categorical with the model's own vocabulary:
    assigning a shuffled object array into the frame would quietly downgrade the column to
    ``object``, and LightGBM would then refuse it ("categorical_feature do not match") or -
    worse - read the levels as strings in an order nobody chose. The permutation is therefore
    a permutation of the *stored levels*, and the encoding the model was trained with is
    untouched.
    """
    shuffled = values[order]
    if isinstance(column.dtype, pd.CategoricalDtype):
        return pd.Series(
            pd.Categorical(
                shuffled, categories=column.cat.categories, ordered=column.cat.ordered
            ),
            index=column.index,
        )
    return pd.Series(shuffled, index=column.index, dtype=column.dtype)


def _score(predictor: Any, matrix: pd.DataFrame, *, classes: Sequence[str]) -> np.ndarray:
    """The class distribution the model's own booster returns for ``matrix``.

    Calling the booster directly (rather than :meth:`ModelPredictor.predict`) is what makes a
    permuted matrix scoreable at all; the shape check is the same one the predictor applies, so
    a booster that disagrees with its metadata is refused rather than reshaped into plausibility.
    """
    booster = getattr(getattr(predictor, "artefact", None), "booster", None)
    if booster is None:  # pragma: no cover - a predictor without an artefact is not a predictor
        raise ImportanceError("the predictor carries no booster, so no column can be permuted")
    if matrix.empty:
        return np.zeros((0, len(classes)), dtype="float64")
    probabilities = np.asarray(booster.predict(matrix), dtype="float64")
    if probabilities.ndim == 1:  # pragma: no cover - multiclass returns (n, k)
        probabilities = probabilities.reshape(-1, len(classes))
    if probabilities.shape != (len(matrix), len(classes)):
        raise ImportanceError(
            f"the booster returned shape {probabilities.shape} for {len(matrix)} row(s) and "
            f"{len(classes)} class(es); the artefact and its metadata disagree"
        )
    return probabilities
# ---------------------------------------------------------------------------
# SHAP (optional: the package is not a dependency of ANJANEYA)
# ---------------------------------------------------------------------------
#: ``status`` values a :class:`ShapResult` can carry.
SHAP_MEASURED = "measured"
SHAP_UNAVAILABLE = "unavailable"
SHAP_SKIPPED = "skipped"
SHAP_FAILED = "failed"


@dataclass(slots=True)
class ShapResult:
    """What the optional attribution did, including the case where it did nothing."""

    #: ``measured`` | ``unavailable`` | ``skipped`` | ``failed``.
    status: str
    #: Why it is not ``measured`` - and, for a failure, the library's own words.
    reason: str | None = None
    rows: int = 0
    classes: int = 0
    #: Column -> mean ``|SHAP|`` over the explained rows and the model's classes.
    mean_absolute: dict[str, float] = field(default_factory=dict)
    #: Class -> column -> mean ``|SHAP|`` (the un-aggregated measurement).
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """Whether an attribution actually happened."""
        return self.status == SHAP_MEASURED

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape, ranked largest mean ``|SHAP|`` first."""
        ranked = sorted(self.mean_absolute.items(), key=lambda item: (-item[1], item[0]))
        return {
            "status": self.status,
            "reason": self.reason,
            "rows": self.rows,
            "classes": self.classes,
            "mean_absolute_shap": {name: float(value) for name, value in ranked},
            "per_class": {
                name: {column: float(value) for column, value in sorted(values.items())}
                for name, values in sorted(self.per_class.items())
            },
        }


def shap_importance(predictor: Any, frame: pd.DataFrame, *, policy: ImportancePolicy) -> ShapResult:
    """Measure spec 27's *optional* attribution, or say precisely why it did not run.

    ``importance.shap`` decides the contract: ``off`` never imports the package, ``auto``
    records an absent package as ``unavailable`` (with the extra to install), and ``on`` makes
    an absent package a configuration error - the operator asked for a measurement the
    environment cannot make, and only the operator can decide what to do about that. A failure
    *inside* SHAP is never fatal: it is a status on an optional measurement, and the native and
    permutation numbers are already in hand.
    """
    if policy.shap == "off":
        return ShapResult(status=SHAP_SKIPPED, reason="importance.shap is 'off'")
    try:
        import shap  # noqa: PLC0415 - optional dependency, imported only when asked for
    except ImportError:
        if policy.shap == "on":
            raise ImportanceError(SHAP_NOT_INSTALLED_HINT) from None
        return ShapResult(status=SHAP_UNAVAILABLE, reason=SHAP_NOT_INSTALLED_HINT)
    if frame.empty:
        return ShapResult(status=SHAP_SKIPPED, reason="no rows in scope to attribute")
    sample = frame
    if len(sample) > int(policy.shap_rows):
        if "timestamp" in sample.columns:
            sample = sample.sort_values("timestamp", kind="stable")
        sample = sample.tail(int(policy.shap_rows))
    try:
        matrix = predictor.matrix(sample)
        explainer = shap.TreeExplainer(getattr(predictor.artefact, "booster", None))
        values = explainer.shap_values(matrix)
    except Exception as exc:  # noqa: BLE001 - an optional measurement never fails a run
        return ShapResult(
            status=SHAP_FAILED, reason=f"{type(exc).__name__}: {exc}", rows=int(len(sample))
        )
    columns = tuple(str(name) for name in predictor.feature_columns)
    classes = tuple(str(name) for name in predictor.classes)
    try:
        per_class = _shap_per_class(values, columns=columns, classes=classes)
    except ImportanceError as exc:
        return ShapResult(
            status=SHAP_FAILED, reason=str(exc), rows=int(len(sample)), classes=len(classes)
        )
    mean_absolute = {
        column: float(np.mean([block[column] for block in per_class.values()]))
        for column in columns
    }
    return ShapResult(
        status=SHAP_MEASURED,
        rows=int(len(sample)),
        classes=len(classes),
        mean_absolute=mean_absolute,
        per_class=per_class,
    )
def _shap_per_class(
    values: Any, *, columns: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, float]]:
    """``shap``'s return value normalised to ``class -> column -> mean |SHAP|``.

    The library's layout differs between releases (a list of per-class arrays, or one
    three-dimensional array whose axes moved), so the shape is *inspected* rather than assumed
    - the feature axis and the class axis are found by their lengths, and anything that fits
    neither is refused. A guess here would attach the wrong numbers to the wrong features.
    """
    n_features, n_classes = len(columns), len(classes)
    if isinstance(values, (list, tuple)):
        arrays = [np.asarray(item, dtype="float64") for item in values]
    else:
        array = np.asarray(values, dtype="float64")
        if array.ndim == 2:
            arrays = [array]
        elif array.ndim == 3 and array.shape[-1] == n_features:
            arrays = [array[index] for index in range(array.shape[0])]
        elif array.ndim == 3 and array.shape[1] == n_features:
            arrays = [array[:, :, index] for index in range(array.shape[2])]
        else:
            raise ImportanceError(
                f"shap returned an array of shape {array.shape}, which matches neither a "
                f"(rows, {n_features}) nor a (rows, classes, {n_features}) attribution"
            )
    if len(arrays) not in {1, n_classes}:
        raise ImportanceError(
            f"shap returned {len(arrays)} attribution block(s) for {n_classes} class(es)"
        )
    per_class: dict[str, dict[str, float]] = {}
    for index, name in enumerate(classes):
        block = arrays[index] if len(arrays) > 1 else arrays[0]
        if block.ndim != 2 or block.shape[1] != n_features:
            raise ImportanceError(
                f"shap returned shape {block.shape} for class {name!r}; expected "
                f"(rows, {n_features})"
            )
        per_class[name] = {
            column: float(np.abs(block[:, position]).mean())
            for position, column in enumerate(columns)
        }
    return per_class
# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def _text(value: float | None, digits: int) -> str:
    """``None`` as an em dash (never 0), else ``digits`` decimals with zeros trimmed."""
    if value is None:
        return "-"
    text = f"{float(value):.{int(digits)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _interval_text(mean: float | None, low: float | None, high: float | None, digits: int) -> str:
    """``mean [low, high]`` - or ``-`` where nothing was measured."""
    if mean is None:
        return "-"
    if low is None or high is None:  # pragma: no cover - an interval always comes with a mean
        return _text(mean, digits)
    return f"{_text(mean, digits)} [{_text(low, digits)}, {_text(high, digits)}]"


@dataclass(slots=True)
class FeatureImportance:
    """One model feature: the model's own gain, the measured permutation delta, and the caveats."""

    column: str
    #: The topic key that claims this column, or :data:`UNMAPPED_TOPIC_KEY`.
    topic: str
    #: The stored feature group the column belongs to (``None`` when it is not grouped).
    group: str | None
    #: LightGBM's own gain for this feature (``None`` when the model never used it).
    native: float | None
    #: ``native`` as a share of the model's total gain.
    native_share: float | None
    #: Position by ``native`` (1 = the largest gain), or ``None``.
    native_rank: int | None
    #: Mean permutation delta over the repeats (larger = more information lost when shuffled).
    permutation_mean: float | None
    permutation_std: float | None
    ci_low: float | None
    ci_high: float | None
    #: Position by ``permutation_mean`` (1 = the largest delta), or ``None``.
    rank: int | None
    #: Share of the sample where this column is ``NULL`` (1.0 = never carried).
    null_share: float
    #: ``True`` when the sample holds a single value for this column.
    constant: bool
    #: ``False`` when the sample cannot measure this column at all, with ``reason`` saying why.
    measured: bool
    reason: str | None = None
    #: Mean ``|SHAP|`` for this feature, when the optional attribution ran.
    shap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "column": self.column,
            "topic": self.topic,
            "group": self.group,
            "native": self.native,
            "native_share": self.native_share,
            "native_rank": self.native_rank,
            "rank": self.rank,
            "permutation": {
                "mean": self.permutation_mean,
                "std": self.permutation_std,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "measured": self.measured,
                "reason": self.reason,
            },
            "null_share": self.null_share,
            "constant": self.constant,
            "shap": self.shap,
        }


@dataclass(slots=True)
class TopicImportance:
    """One source of spec 27's list: what it contributed, in the words of the measurement."""

    key: str
    title: str
    #: How many features of this model the topic claims.
    features: int
    #: How many of them the sample could actually measure.
    measured_features: int
    #: Sum of the native gains of the claimed features, over the model's total gain.
    native_share: float | None
    #: Sum of the claimed features' permutation deltas, with its own interval.
    permutation_mean: float | None
    ci_low: float | None
    ci_high: float | None
    #: Position by ``permutation_mean`` among measured topics (1 = the largest), or ``None``.
    rank: int | None
    #: ``False`` when no feature of this source could be measured here.
    measured: bool
    reason: str | None = None
    #: Sum of the claimed features' mean ``|SHAP|``, when the optional attribution ran.
    shap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "key": self.key,
            "title": self.title,
            "features": self.features,
            "measured_features": self.measured_features,
            "rank": self.rank,
            "native_share": self.native_share,
            "permutation": {
                "mean": self.permutation_mean,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "measured": self.measured,
                "reason": self.reason,
            },
            "shap": self.shap,
        }
@dataclass(slots=True)
class ImportanceReport:
    """One importance measurement, ready to be written as JSON, Markdown and CSV."""

    policy: ImportancePolicy
    #: The artefact (:meth:`core.ml.predictor.ModelArtefact.describe`).
    model: dict[str, Any]
    #: The matrix the sample came from (:meth:`core.ml.dataset.Dataset.describe`).
    dataset: dict[str, Any]
    #: The sample actually measured (:meth:`ImportanceSample.describe`).
    sample: dict[str, Any]
    classes: tuple[str, ...]
    features: tuple[FeatureImportance, ...]
    topics: tuple[TopicImportance, ...]
    #: The permutation measurement itself (:meth:`PermutationResult.to_dict`).
    measurement: dict[str, Any]
    shap: dict[str, Any] | None
    counts: dict[str, Any]
    notes: tuple[str, ...]
    caveats: tuple[str, ...]

    # -- what it says ---------------------------------------------------------
    @property
    def measured_features(self) -> tuple[FeatureImportance, ...]:
        """The features the sample could measure."""
        return tuple(item for item in self.features if item.measured)

    @property
    def top_topic(self) -> TopicImportance | None:
        """The measured source with the largest permutation delta, or ``None``."""
        ranked = [topic for topic in self.topics if topic.measured]
        if not ranked:
            return None
        return sorted(ranked, key=lambda item: (-float(item.permutation_mean or 0.0), item.key))[0]

    def ranked(self, limit: int | None = None) -> list[FeatureImportance]:
        """The measured features, largest permutation delta first."""
        ordered = sorted(
            self.measured_features, key=lambda item: (item.rank or len(self.features))
        )
        return ordered[: int(limit)] if limit is not None else ordered

    def topic(self, key: str) -> TopicImportance | None:
        """The topic with ``key``, or ``None``."""
        for item in self.topics:
            if item.key == key:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - the document the JSON artefact carries."""
        return {
            "spec": self.policy.spec,
            "policy": self.policy.to_dict(),
            "model": self.model,
            "dataset": self.dataset,
            "sample": self.sample,
            "classes": list(self.classes),
            "measurement": self.measurement,
            "shap": self.shap,
            "counts": self.counts,
            "topics": [topic.to_dict() for topic in self.topics],
            "features": [feature.to_dict() for feature in self.features],
            "caveats": list(self.caveats),
            "notes": list(self.notes),
        }
    def to_markdown(self) -> str:
        """The human-readable artefact: the sources first, then every feature, then the caveats."""
        digits = self.policy.digits
        lines = [f"# ANJANEYA feature importance (spec {self.policy.spec})", ""]
        lines.append(
            f"- model   : {self.model.get('path')} ({self.model.get('algorithm')}, horizon "
            f"{self.model.get('horizon_minutes')}m, "
            f"{self.model.get('results_label') or 'unlabelled'})"
        )
        lines.append(
            f"- metric  : {self.measurement.get('metric')} "
            f"({self.measurement.get('direction')}), baseline "
            f"{_text(self.measurement.get('baseline'), digits)}, "
            f"{self.measurement.get('repeats')} permutation(s) at seed "
            f"{self.measurement.get('seed')}"
        )
        lines.append(
            f"- sample  : {self.sample.get('name')} - {self.sample.get('rows_used')} row(s) of "
            f"{self.sample.get('days')} day(s), {self.sample.get('first_timestamp')} .. "
            f"{self.sample.get('last_timestamp')}"
            + (" (capped by importance.max_rows)" if self.sample.get("capped") else "")
        )
        if self.sample.get("reason"):
            lines.append(f"- fallback: {self.sample.get('reason')}")
        lines.append(
            f"- features: {self.counts.get('measured')} of {self.counts.get('features')} measured, "
            f"{self.counts.get('unmapped_features')} unmapped, SHAP "
            f"{self.counts.get('shap_status')}"
        )
        lines.append("")
        lines.append("## Sources (spec 27's question)")
        lines.append("")
        lines.append("| source | features | permutation delta (95% CI) | native share | mean \\|SHAP\\| |")
        lines.append("| --- | --- | --- | --- | --- |")
        for topic in sorted(
            self.topics, key=lambda item: (item.rank or len(self.topics), item.key)
        ):
            lines.append(
                f"| `{topic.key}` {topic.title} | {topic.measured_features}/{topic.features} | "
                f"{_interval_text(topic.permutation_mean, topic.ci_low, topic.ci_high, digits)} | "
                f"{_text(topic.native_share, digits)} | {_text(topic.shap, digits)} |"
            )
        absent = [topic for topic in self.topics if not topic.measured]
        if absent:
            lines.append("")
            lines.append("Sources this run could not measure:")
            for topic in absent:
                lines.append(f"- `{topic.key}`: {topic.reason}")
        lines.append("")
        lines.append("## Every feature")
        lines.append("")
        lines.append(
            "| # | feature | source | native | native share | permutation delta | 95% CI | NULL |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in self.ranked():
            lines.append(
                f"| {item.rank} | `{item.column}` | `{item.topic}` | {_text(item.native, digits)} | "
                f"{_text(item.native_share, digits)} | {_text(item.permutation_mean, digits)} | "
                f"[{_text(item.ci_low, digits)}, {_text(item.ci_high, digits)}] | "
                f"{_text(item.null_share, 2)} |"
            )
        unmeasured = [item for item in self.features if not item.measured]
        if unmeasured:
            lines.append("")
            lines.append("Features the sample could not measure (never silently zero):")
            for item in unmeasured:
                lines.append(f"- `{item.column}`: {item.reason}")
        lines.append("")
        lines.append("## SHAP (optional, spec 27)")
        lines.append("")
        if self.shap and self.shap.get("status") == "measured":
            lines.append(
                f"measured on {self.shap.get('rows')} row(s) over {self.shap.get('classes')} "
                "class(es); mean |SHAP| per feature is in the JSON and the CSV beside this file"
            )
        elif self.shap:
            lines.append(f"status: {self.shap.get('status')} - {self.shap.get('reason')}")
        else:
            lines.append("status: not requested")
        lines.append("")
        lines.append("## Caveats")
        lines.append("")
        lines.extend(f"- {caveat}" for caveat in self.caveats)
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        lines.append("")
        return "\n".join(lines)
def _caveats(sample: ImportanceSample, extra: Sequence[str]) -> tuple[str, ...]:
    """The three structural caveats, then whatever the caller adds (deduplicated, in order)."""
    items = [IMPORTANCE_CAVEAT, CORRELATED_FEATURES_CAVEAT, measurement_caveat(sample)]
    for caveat in extra:
        text = str(caveat)
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _topic_importances(
    *,
    policy: ImportancePolicy,
    features: Sequence[FeatureImportance],
    permutation: PermutationResult,
    names: Sequence[str],
) -> list[TopicImportance]:
    """Sum the per-repeat deltas per source, and rank the sources that could be measured."""
    by_column = {item.column: item for item in features}
    plan: list[tuple[str, str, tuple[str, ...]]] = [
        (topic.key, topic.title, tuple(name for name in topic.columns if name in by_column))
        for topic in policy.topics
    ]
    unmapped = tuple(name for name in names if by_column[name].topic == UNMAPPED_TOPIC_KEY)
    if unmapped:
        plan.append((UNMAPPED_TOPIC_KEY, UNMAPPED_TOPIC_TITLE, unmapped))

    total_native = float(sum(item.native or 0.0 for item in features))
    topics: list[TopicImportance] = []
    for key, title, claimed in plan:
        measured_names = [name for name in claimed if by_column[name].measured]
        native_sum = float(sum(by_column[name].native or 0.0 for name in claimed))
        deltas = permutation.group_deltas(claimed)
        mean = low = high = None
        if deltas is not None:
            mean, low, high = _interval(deltas, confidence=permutation.confidence)
        contributions = [
            by_column[name].shap for name in claimed if by_column[name].shap is not None
        ]
        reason = None
        if not claimed:
            reason = (
                "no feature of this source is in this model's matrix, so this run cannot "
                "measure it: the information is not here to measure"
            )
        elif not measured_names:
            reason = (
                f"none of its {len(claimed)} feature(s) could be measured in this sample: "
                f"{by_column[claimed[0]].reason}"
            )
        topics.append(
            TopicImportance(
                key=key,
                title=title,
                features=len(claimed),
                measured_features=len(measured_names),
                native_share=(native_sum / total_native) if (claimed and total_native > 0) else None,
                permutation_mean=mean,
                ci_low=low,
                ci_high=high,
                rank=None,
                measured=deltas is not None,
                reason=reason,
                shap=float(sum(contributions)) if contributions else None,
            )
        )
    ranked = sorted(
        (topic for topic in topics if topic.measured),
        key=lambda item: (-float(item.permutation_mean or 0.0), item.key),
    )
    for position, topic in enumerate(ranked, start=1):
        topic.rank = position
    return topics
def build_importance_report(
    *,
    policy: ImportancePolicy,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    classes: Sequence[str],
    sample: ImportanceSample,
    native: Mapping[str, float],
    permutation: PermutationResult,
    shap: ShapResult | None = None,
    notes: Sequence[str] = (),
    caveats: Sequence[str] = (),
) -> ImportanceReport:
    """Assemble spec 27's report out of measurements that have already been made.

    Pure: the permutation deltas, the model's own gain and (when it ran) the attributions arrive
    as arguments, and nothing here re-reads a file or re-scores a row. The three structural
    caveats are always present, whatever the caller passes - they are part of the answer, not
    decoration a caller may forget.
    """
    names = tuple(permutation.columns)
    native_values = {name: float(native[name]) for name in names if name in native}
    total_native = float(sum(native_values.values()))
    native_order = {
        name: position + 1
        for position, (name, _) in enumerate(
            sorted(native_values.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    permutation_order = {
        name: position + 1 for position, (name, _) in enumerate(permutation.ranking())
    }
    shap_values = dict(shap.mean_absolute) if shap is not None and shap.measured else {}

    features: list[FeatureImportance] = []
    for name in names:
        summary = permutation.summary(name)
        series = sample.frame[name] if name in sample.frame.columns else None
        value = native_values.get(name)
        features.append(
            FeatureImportance(
                column=name,
                topic=policy.topic_for(name) or UNMAPPED_TOPIC_KEY,
                group=_group_of(name),
                native=value,
                native_share=(value / total_native) if value is not None and total_native else None,
                native_rank=native_order.get(name),
                permutation_mean=summary["mean"],
                permutation_std=summary["std"],
                ci_low=summary["ci_low"],
                ci_high=summary["ci_high"],
                rank=permutation_order.get(name),
                null_share=float(series.isna().mean()) if series is not None else 1.0,
                constant=bool(series.nunique(dropna=True) <= 1) if series is not None else True,
                measured=bool(summary["measured"]),
                reason=summary["reason"],
                shap=shap_values.get(name),
            )
        )
    topics = _topic_importances(
        policy=policy, features=features, permutation=permutation, names=names
    )
    measured = [item for item in features if item.measured]
    counts: dict[str, Any] = {
        "features": len(features),
        "measured": len(measured),
        "unmeasured": len(features) - len(measured),
        "topics": len(topics),
        "topics_measured": sum(1 for topic in topics if topic.measured),
        "unmapped_features": sum(1 for item in features if item.topic == UNMAPPED_TOPIC_KEY),
        "rows": permutation.rows,
        "repeats": permutation.repeats,
        "sample_days": sample.days,
        "shap_status": shap.status if shap is not None else None,
        "shap_features": len(shap_values),
    }
    return ImportanceReport(
        policy=policy,
        model=dict(model),
        dataset=dict(dataset),
        sample=sample.describe(),
        classes=tuple(str(name) for name in classes),
        features=tuple(features),
        topics=tuple(topics),
        measurement=permutation.to_dict(),
        shap=shap.to_dict() if shap is not None else None,
        counts=counts,
        notes=tuple(str(note) for note in notes),
        caveats=_caveats(sample, caveats),
    )
