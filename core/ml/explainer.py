"""Spec 26's explainability: one prediction, described in plain structured form.

Spec 26 is short and unusually strict for its length: *every* prediction must be
explainable, the form must be plain and structured, and the values it lists are "examples
only. Do not hard-code." This module is that promise kept in code:

* the blocks of an explanation - NIFTY, OPTION, PRICE ACTION, CANDLE, S/R, CONTEXT and the
  model's own probabilities - are **configuration** (``model.explainability``), and each
  block names stored feature *groups* or individual feature *columns*. An explanation can
  therefore never describe a number the row it explains does not carry;
* the model's block is not a class list written in this file: it is whatever classes the
  loaded artefact's metadata declares, in the metadata's order, with the top
  ``explainability.probabilities`` of them shown (see :mod:`core.ml.predictor`);
* a stored feature that is ``NULL`` at that instant is reported **unknown**
  (``explainability.missing_text``), never 0 - "we do not know" and "it was zero" are
  different statements about a market;
* a **column the policy names but the row does not carry is a refusal**, not a blank -
  the same rule :class:`core.ml.predictor.ModelPredictor` applies to the model's own
  feature list, and for the same reason: a silently missing input turns an explanation
  into fiction.

What an explanation is *not*: it is a description of the inputs a model was fed at one
instant plus the probabilities it produced. It is not a claim about why the market moved,
and it is not a recommendation - spec 24's decision layer still does not exist, and
:data:`DECISION_LAYER_CAVEAT` travels with every artefact this module renders.

Identity (contract, expiry, strike, timestamp, timeframe) is carried by every explanation's
header and the row's provenance (spec 6.5) by its audit block, which is why the ``identity``
and ``provenance`` feature groups are reserved and can never be used as a section:
:data:`RESERVED_EXPLANATION_GROUPS`. The AI block is appended last, after every configured
section, under :data:`AI_SECTION_KEY`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from core.constants import FEATURE_COLUMNS, PROVENANCE_COLUMNS
from core.features.feature_builder import FEATURE_GROUPS, IDENTITY_FEATURE_COLUMNS

__all__ = [
    "AI_SECTION_KEY",
    "AI_SECTION_TITLE",
    "DECISION_LAYER_CAVEAT",
    "EXPLANATION_CAVEAT",
    "FRESHNESS_CAVEAT",
    "RESERVED_EXPLANATION_GROUPS",
    "ClassProbability",
    "Explanation",
    "ExplanationBatch",
    "ExplanationEntry",
    "ExplanationError",
    "ExplanationPolicy",
    "ExplanationSection",
    "build_explanations",
    "explain_frame",
    "explain_row",
    "format_value",
]

#: Feature groups an explanation can never take as a *section*: every explanation already
#: carries its contract identity in its header and the explained row's provenance in its
#: audit block, and a section repeating either would be the same number twice.
RESERVED_EXPLANATION_GROUPS: frozenset[str] = frozenset({"identity", "provenance"})

#: The identity columns every explanation's header is rendered from, in row order.
IDENTITY_COLUMNS: tuple[str, ...] = tuple(IDENTITY_FEATURE_COLUMNS)

#: The provenance columns every explanation carries from the row it explains (spec 6.5).
AUDIT_COLUMNS: tuple[str, ...] = tuple(PROVENANCE_COLUMNS)

#: The key and title of the model's own block, always rendered after the configured
#: sections. A configured section may not use the key (checked at load).
AI_SECTION_KEY = "ai"
AI_SECTION_TITLE = "AI"

#: Printed on (and stored with) every explanation. Spec 26 asks for structure, not
#: story-telling, and this is the sentence that keeps the two apart.
EXPLANATION_CAVEAT = (
    "an explanation quotes the stored feature row of that instant as-is, together with the "
    "probabilities one model produced from it; it is a description of the inputs, not a "
    "claim about why the market moved"
)
#: The second half of the same rule. The decision layer (spec 24, master-prompt phase 24) now
#: exists - as a separate, read-only research mode - and *this* is still not it: an explanation
#: describes an input, and `--mode decide` is what records a rule's decision.
DECISION_LAYER_CAVEAT = (
    "this is not the decision layer: spec 24's decision layer is a separate, read-only "
    "research step (--mode decide) that records what one configured rule would do with a "
    "distribution, and nothing here turns a probability into a position - so an explanation "
    "is not a recommendation"
)
#: An explanation is only as good as the row it quotes, and the row says when it was fetched.
FRESHNESS_CAVEAT = (
    "the input is the stored row: its fetch_timestamp, source and quality_flag are in the "
    "audit block, and a stale or low-quality row explains a stale or low-quality prediction"
)

#: The kinds an entry's value can be, as reported in JSON.
KIND_NUMBER = "number"
KIND_CATEGORY = "category"
KIND_FLAG = "flag"
KIND_TIMESTAMP = "timestamp"
KIND_UNKNOWN = "unknown"

#: How a boolean is rendered (a flag, not a number: a sum of flags means nothing).
FLAG_TRUE = "true"
FLAG_FALSE = "false"


class ExplanationError(RuntimeError):
    """Raised when a row cannot be explained honestly (a named column is missing, ...)."""


def _is_missing(value: Any) -> bool:
    """``True`` for ``None``, ``NaN``, ``NaT`` and pandas' ``NA`` - but never for ``0``."""
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - non-scalar inputs
        return False


def _json_value(value: Any) -> Any:
    """A JSON-serialisable version of a stored feature value (``None`` when unknown)."""
    if _is_missing(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):  # numpy scalar -> Python
        try:
            return item()
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(value)
    return value


def _number_text(value: int | float, digits: int) -> str:
    """A number with at most ``digits`` decimals, trailing zeros trimmed, never ``-0``."""
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):  # pragma: no cover - NaN is caught as missing
        return str(value)
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0", "-"} else text


def format_value(value: Any, *, digits: int, missing_text: str) -> tuple[str, str, bool]:
    """``(text, kind, known)`` for one stored feature value.

    The rules are the whole of spec 26's presentation policy, and all three are deliberate:

    * ``None``/``NaN``/``NaT`` -> ``missing_text``, kind ``unknown``. A zero-filled
      explanation would be a lie about a market, which is why the word is configuration
      rather than a hard-coded ``"n/a"``;
    * numbers are printed with **at most** ``digits`` decimals and their trailing zeros
      trimmed, so ``0.5`` is not dressed up as ``0.5000`` and a price is not padded with
      digits it never had;
    * ``True``/``False`` become ``true``/``false`` (a flag, not a number) and everything
      else - a categorical level, a timestamp - is rendered as the string it is.
    """
    if _is_missing(value):
        return missing_text, KIND_UNKNOWN, False
    if isinstance(value, bool):
        return (FLAG_TRUE if value else FLAG_FALSE), KIND_FLAG, True
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat(), KIND_TIMESTAMP, True
    if isinstance(value, (int, float)):
        return _number_text(value, digits), KIND_NUMBER, True
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            scalar = item()
        except (TypeError, ValueError):  # pragma: no cover - defensive
            scalar = value
        if isinstance(scalar, bool):
            return (FLAG_TRUE if scalar else FLAG_FALSE), KIND_FLAG, True
        if isinstance(scalar, (int, float)):
            return _number_text(scalar, digits), KIND_NUMBER, True
    return str(value), KIND_CATEGORY, True


@dataclass(frozen=True, slots=True)
class ExplanationEntry:
    """One stored feature, as the explanation renders it."""

    #: The stored feature column this line came from - the only name a reader can trace.
    column: str
    #: The feature group it belongs to (``None`` for a column named explicitly in config).
    group: str | None
    #: ``number`` | ``category`` | ``flag`` | ``timestamp`` | ``unknown``.
    kind: str
    #: The raw value (JSON-friendly), or ``None`` where the store holds nothing.
    value: Any
    #: The rendered value - what a human reads first.
    text: str
    #: ``False`` only where the stored value is ``NULL`` at that instant.
    known: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape: the line as the artefact carries it."""
        return {
            "column": self.column,
            "group": self.group,
            "kind": self.kind,
            "value": self.value,
            "text": self.text,
            "known": self.known,
        }


@dataclass(slots=True)
class ExplanationSection:
    """One block of an explanation - spec 26's NIFTY/OPTION/... list, plus the model's own.

    ``available`` is the honest half of the structure: a block that the run could not fill
    (no model was scored, or the model produced no distribution) is present with the reason
    it is empty, so "this block is empty" and "we never looked at it" cannot be confused.
    """

    key: str
    title: str
    entries: list[ExplanationEntry] = field(default_factory=list)
    #: ``feature_row`` for a configured block, ``model`` for the appended AI block.
    source: str = "feature_row"
    #: ``False`` when the block cannot be answered at all, with ``reason`` saying why.
    available: bool = True
    reason: str | None = None
    #: The class probabilities of the AI block, top rank first (empty for other blocks).
    probabilities: list[ClassProbability] = field(default_factory=list)

    @property
    def known(self) -> int:
        """How many lines carry a stored value."""
        return sum(1 for entry in self.entries if entry.known)

    @property
    def unknown(self) -> int:
        """How many lines are ``NULL`` in the store at this instant."""
        return sum(1 for entry in self.entries if not entry.known)

    def entry(self, column: str) -> ExplanationEntry | None:
        """The line for ``column``, or ``None`` when this block does not explain it."""
        for entry in self.entries:
            if entry.column == column:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "key": self.key,
            "title": self.title,
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "known": self.known,
            "unknown": self.unknown,
            "entries": [entry.to_dict() for entry in self.entries],
            "probabilities": [item.to_dict() for item in self.probabilities],
        }


@dataclass(frozen=True, slots=True)
class ClassProbability:
    """One class of the model's distribution, with its rank in that distribution."""

    name: str
    probability: float
    #: 0 = the model's top class.
    rank: int

    @property
    def top(self) -> bool:
        """``True`` for the model's own pick."""
        return self.rank == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {"class": self.name, "probability": self.probability, "rank": self.rank}


@dataclass(frozen=True, slots=True)
class ExplanationSource:
    """One named source inside a block: a stored feature column and the group it came from."""

    column: str
    #: ``None`` for a column the configuration named explicitly.
    group: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {"column": self.column, "group": self.group}


@dataclass(frozen=True, slots=True)
class ExplanationBlock:
    """A resolved block: what a configured section means once checked against the contract."""

    key: str
    title: str
    sources: tuple[ExplanationSource, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "key": self.key,
            "title": self.title,
            "columns": [source.column for source in self.sources],
        }


def _sources(
    groups: Sequence[str], columns: Sequence[str] = ()
) -> tuple[ExplanationSource, ...]:
    """The sources of ``groups`` (in group order) followed by ``columns``, or an error.

    Every name is checked against the feature contract here rather than trusted: this
    function is the single place a hand-built policy (a test, a notebook) and the loaded
    configuration both pass through, so neither can name a feature that does not exist.
    """
    items: list[ExplanationSource] = []
    for name in groups:
        if name in RESERVED_EXPLANATION_GROUPS:
            raise ExplanationError(
                f"{name!r} is a reserved feature group: every explanation carries the row's "
                "identity in its header and its provenance in its audit block, so neither "
                "can be a section"
            )
        if name not in FEATURE_GROUPS:
            raise ExplanationError(
                f"unknown feature group {name!r}; known groups: {sorted(FEATURE_GROUPS)}"
            )
        items.extend(ExplanationSource(column, name) for column in FEATURE_GROUPS[name])
    items.extend(ExplanationSource(column, None) for column in columns)
    return tuple(items)


#: The blocks a caller with no configuration gets: the same six the shipped
#: ``config/model.yaml`` declares, in the same order and with the same sources.
DEFAULT_BLOCKS: tuple[ExplanationBlock, ...] = (
    ExplanationBlock("nifty", "NIFTY", _sources(("nifty",))),
    ExplanationBlock("option", "OPTION", _sources(("option",))),
    ExplanationBlock(
        "price_action", "PRICE ACTION", _sources(("levels", "level_events", "structure"))
    ),
    ExplanationBlock("candle", "CANDLE", _sources(("candle",))),
    ExplanationBlock(
        "support_resistance", "SUPPORT / RESISTANCE", _sources(("support_resistance",))
    ),
    ExplanationBlock("context", "CONTEXT", _sources(("context", "relative"))),
)


@dataclass(frozen=True, slots=True)
class ExplanationPolicy:
    """Every rule an explanation applies, read from the configuration in one place.

    The defaults mirror ``config/model.yaml``'s ``explainability:`` block, so a caller that
    has no configuration still gets the shipped structure rather than a second, silently
    different one - and :meth:`__post_init__` re-checks the invariants the loader enforces,
    because a hand-built policy never passes through the loader.
    """

    #: The master-spec section the structure comes from (printed in every artefact).
    spec: str = "26"
    #: How many of the model's class probabilities an explanation prints.
    probabilities: int = 3
    #: Decimals for every number (trailing zeros are trimmed).
    digits: int = 4
    #: How many predictions the mode describes when ``--rows`` is not given.
    rows: int = 5
    #: Printed and stored where a stored feature is ``NULL`` at that instant.
    missing_text: str = "unknown"
    #: The configured blocks, in render order (the AI block is appended after them).
    blocks: tuple[ExplanationBlock, ...] = DEFAULT_BLOCKS

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ExplanationError("a policy with no blocks would explain nothing")
        keys = [block.key for block in self.blocks]
        if len(set(keys)) != len(keys):
            raise ExplanationError(f"block keys must be unique, got {keys}")
        if AI_SECTION_KEY in keys:
            raise ExplanationError(
                f"{AI_SECTION_KEY!r} is reserved for the model's own block, which is always "
                "appended last; rename the configured section"
            )
        seen: dict[str, str] = {}
        for block in self.blocks:
            if not block.sources:
                raise ExplanationError(f"block {block.key!r} explains nothing")
            for source in block.sources:
                if source.group in RESERVED_EXPLANATION_GROUPS:
                    raise ExplanationError(
                        f"block {block.key!r} explains {source.column!r} through the reserved "
                        f"feature group {source.group!r}: identity belongs in every "
                        "explanation's header and provenance in its audit block"
                    )
                if source.column not in FEATURE_COLUMNS:
                    raise ExplanationError(
                        f"block {block.key!r} explains {source.column!r}, which is not a "
                        "column of the stored feature row (core/constants.py)"
                    )
                if source.column in seen:
                    raise ExplanationError(
                        f"feature column {source.column!r} is explained twice (blocks "
                        f"{seen[source.column]!r} and {block.key!r}); one column, one line"
                    )
                seen[source.column] = block.key
        if self.probabilities < 1:
            raise ExplanationError("explainability.probabilities must be at least 1")
        if self.digits < 0:
            raise ExplanationError("explainability.digits must not be negative")
        if self.rows < 1:
            raise ExplanationError("explainability.rows must be at least 1")
        if not self.missing_text:
            raise ExplanationError(
                "explainability.missing_text must not be empty: a NULL feature has to be "
                "rendered as something"
            )

    @classmethod
    def from_config(cls, config: Any) -> ExplanationPolicy:
        """Read the policy out of a loaded :class:`core.config_loader.AnjaneyaConfig`.

        A configuration without the block (an older file, a hand-made object) falls back to
        the shipped defaults rather than failing: this structure is presentation, and unlike
        the model's own feature contract nothing downstream depends on it.
        """
        block = getattr(getattr(config, "model", None), "explainability", None)
        if block is None:
            return cls()
        return cls(
            spec=str(block.spec),
            probabilities=int(block.probabilities),
            digits=int(block.digits),
            rows=int(block.rows),
            missing_text=str(block.missing_text),
            blocks=tuple(
                ExplanationBlock(
                    key=str(section.key),
                    title=str(section.title),
                    sources=_sources(tuple(section.groups), tuple(section.columns)),
                )
                for section in block.sections
            ),
        )

    # -- what the policy explains --------------------------------------------
    @property
    def columns(self) -> tuple[str, ...]:
        """Every stored feature column the explanation renders, in block order."""
        return tuple(source.column for block in self.blocks for source in block.sources)

    def block(self, key: str) -> ExplanationBlock | None:
        """The block with ``key``, or ``None``."""
        for block in self.blocks:
            if block.key == key:
                return block
        return None

    def explains(self, column: str) -> bool:
        """Whether ``column`` appears in any block (the AI block is not a column set)."""
        return column in self.columns

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - stored in every artefact so a line stays traceable."""
        return {
            "spec": self.spec,
            "probabilities": self.probabilities,
            "digits": self.digits,
            "rows": self.rows,
            "missing_text": self.missing_text,
            "columns": list(self.columns),
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(slots=True)
class Explanation:
    """One prediction, explained: the row's blocks, the model's block, and the audit trail."""

    #: The policy that produced it (stored in the artefact, so a line stays reproducible).
    policy: ExplanationPolicy
    #: The explained row's contract identity, in ``IDENTITY_COLUMNS`` order.
    identity: dict[str, Any]
    sections: list[ExplanationSection] = field(default_factory=list)
    #: The explained row's provenance columns (spec 6.5) - which run wrote this input.
    audit: dict[str, Any] = field(default_factory=dict)
    #: The model's own pick and its numbers, when a model was scored.
    predicted_class: str | None = None
    predicted_probability: float | None = None
    runner_up_class: str | None = None
    runner_up_probability: float | None = None
    margin: float | None = None
    #: The artefact (:meth:`core.ml.predictor.ModelArtefact.describe`) behind the AI block.
    model: dict[str, Any] | None = None

    # -- what it says --------------------------------------------------------
    @property
    def label(self) -> str:
        """``NIFTY01OCT2625000CE @ 2026-09-30T14:45:00+05:30`` - how a heading names it."""
        symbol = self.identity.get("tradingsymbol") or "?"
        stamp = self.identity.get("timestamp") or "?"
        return f"{symbol} @ {stamp}"

    @property
    def entries(self) -> int:
        """How many stored features the explanation renders (every block, every line)."""
        return sum(len(section.entries) for section in self.sections)

    @property
    def unknown(self) -> int:
        """How many of those lines are ``NULL`` in the store at this instant."""
        return sum(section.unknown for section in self.sections)

    def section(self, key: str) -> ExplanationSection | None:
        """The block with ``key``, or ``None``."""
        for section in self.sections:
            if section.key == key:
                return section
        return None

    def entry(self, column: str) -> ExplanationEntry | None:
        """The line for a stored feature column, whichever block carries it."""
        for section in self.sections:
            found = section.entry(column)
            if found is not None:
                return found
        return None

    # -- the three shapes it is read in --------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape: identity, the prediction, every block, the audit block."""
        return {
            "identity": dict(self.identity),
            "prediction": {
                "class": self.predicted_class,
                "probability": self.predicted_probability,
                "runner_up_class": self.runner_up_class,
                "runner_up_probability": self.runner_up_probability,
                "margin": self.margin,
            },
            "model": self.model,
            "counts": {"entries": self.entries, "unknown": self.unknown},
            "sections": [section.to_dict() for section in self.sections],
            "audit": dict(self.audit),
        }

    def _prediction_line(self) -> str:
        """The one-line prediction summary, or the reason there is none."""
        digits = self.policy.digits
        if self.predicted_class is None:
            return "  prediction : not scored (no model produced a distribution for this row)"
        text = (
            f"  prediction : {self.predicted_class} "
            f"p={_number_text(self.predicted_probability or 0.0, digits)}"
        )
        if self.runner_up_class is not None:
            text += (
                f" (runner-up {self.runner_up_class} "
                f"{_number_text(self.runner_up_probability or 0.0, digits)}"
            )
            if self.margin is not None:
                text += f", margin {_number_text(self.margin, digits)}"
            text += ")"
        return text

    def _section_lines(self, section: ExplanationSection) -> list[str]:
        """The lines of one block, aligned, with the AI block's top probabilities last."""
        lines = [f"  {section.title}:"]
        if not section.available:
            lines.append(f"    (not answered: {section.reason})")
        if section.entries:
            width = max(len(entry.column) for entry in section.entries)
            for entry in section.entries:
                lines.append(f"    {entry.column:<{width}} : {entry.text}")
        shown = section.probabilities[: self.policy.probabilities]
        if shown:
            width = max(len(item.name) for item in shown)
            for item in shown:
                mark = "  <- predicted" if item.top else ""
                lines.append(
                    f"    {item.name:<{width}} : "
                    f"{_number_text(item.probability, self.policy.digits)}{mark}"
                )
            if len(section.probabilities) > len(shown):
                lines.append(
                    f"    (top {len(shown)} of {len(section.probabilities)} class "
                    "probabilities; the stored artefact carries every one of them)"
                )
        return lines

    def to_text(self) -> str:
        """The console block: contract, prediction, one line per stored feature."""
        lines = [self.label, self._prediction_line()]
        for section in self.sections:
            lines.extend(self._section_lines(section))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The same block as a Markdown subsection, for the reader who keeps the file."""
        lines = [f"### {self.label}", "", self._prediction_line().strip(), ""]
        for section in self.sections:
            lines.append(f"**{section.title}**")
            lines.append("")
            if not section.available:
                lines.append(f"_not answered: {section.reason}_")
                lines.append("")
            for entry in section.entries:
                lines.append(f"- `{entry.column}`: {entry.text}")
            shown = section.probabilities[: self.policy.probabilities]
            for item in shown:
                mark = " **(predicted)**" if item.top else ""
                lines.append(
                    f"- `{item.name}`: {_number_text(item.probability, self.policy.digits)}{mark}"
                )
            if len(section.probabilities) > len(shown):
                lines.append(
                    f"- _top {len(shown)} of {len(section.probabilities)} class probabilities "
                    "shown; the stored artefact carries every one of them_"
                )
            lines.append("")
        return "\n".join(lines)


@dataclass(slots=True)
class ExplanationBatch:
    """Every explanation one run produced, with the policy, the inputs and the caveats."""

    policy: ExplanationPolicy
    explanations: list[Explanation] = field(default_factory=list)
    #: The artefact's :meth:`~core.ml.predictor.ModelArtefact.describe` (or ``None``).
    model: dict[str, Any] | None = None
    #: The ``[start, end)`` window the explained rows came from, in exchange-local time.
    window: tuple[Any, Any] = (None, None)
    #: The contracts the explanations cover (empty = every contract in scope).
    symbols: tuple[str, ...] = ()
    timeframe: str | None = None
    horizon_minutes: int | None = None
    #: The honesty block: what an explanation does and does not claim.
    caveats: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        """How many predictions were explained."""
        return len(self.explanations)

    @property
    def entries(self) -> int:
        """How many stored features every explanation renders, summed."""
        return sum(item.entries for item in self.explanations)

    @property
    def unknown(self) -> int:
        """How many of those lines were ``NULL`` in the store."""
        return sum(item.unknown for item in self.explanations)

    def section_keys(self) -> tuple[str, ...]:
        """The block keys in render order (the AI block included exactly when it exists)."""
        if not self.explanations:
            return tuple(block.key for block in self.policy.blocks)
        return tuple(section.key for section in self.explanations[0].sections)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape - the whole artefact behind ``_explanations.json``."""
        start, end = self.window
        return {
            "policy": self.policy.to_dict(),
            "model": self.model,
            "window": {
                "from": str(start) if start is not None else None,
                "to": str(end) if end is not None else None,
                "symbols": list(self.symbols),
                "timeframe": self.timeframe,
                "horizon_minutes": self.horizon_minutes,
            },
            "counts": {
                "predictions": self.rows,
                "entries": self.entries,
                "unknown": self.unknown,
            },
            "caveats": list(self.caveats),
            "explanations": [item.to_dict() for item in self.explanations],
        }

    def to_text(self) -> str:
        """The console document: a header, then one block per explained prediction."""
        lines = [
            f"ANJANEYA-AI explanations (spec {self.policy.spec}) - {self.rows} prediction(s), "
            f"{self.entries} stored feature(s) of which {self.unknown} unknown"
        ]
        start, end = self.window
        if start is not None or end is not None:
            lines.append(f"  window     : {start or '?'} .. {end or '?'}")
        if self.timeframe:
            horizon = f" {self.horizon_minutes}m" if self.horizon_minutes else ""
            lines.append(f"  timeframe  : {self.timeframe}{horizon}")
        if self.model:
            lines.append(f"  model      : {self.model.get('path')}")
        if self.symbols:
            lines.append(f"  contracts  : {', '.join(self.symbols)}")
        for item in self.explanations:
            lines.append("")
            lines.extend(item.to_text().splitlines())
        if self.caveats:
            lines.append("")
            for caveat in self.caveats:
                lines.append(f"  caveat: {caveat}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The Markdown document, from the same object the JSON is written from."""
        lines = [
            f"# ANJANEYA-AI explanations (spec {self.policy.spec})",
            "",
            f"{self.rows} prediction(s), {self.entries} stored feature(s) explained, "
            f"{self.unknown} of them unknown at that instant.",
            "",
        ]
        start, end = self.window
        if start is not None or end is not None:
            lines.append(f"- window: `{start or '?'}` .. `{end or '?'}`")
        if self.timeframe:
            horizon = f" {self.horizon_minutes}m" if self.horizon_minutes else ""
            lines.append(f"- timeframe: `{self.timeframe}{horizon}`")
        if self.model:
            lines.append(f"- model: `{self.model.get('path')}`")
        lines.append(f"- policy: {len(self.policy.columns)} stored feature(s) in "
                     f"{len(self.policy.blocks)} block(s) plus the AI block")
        lines.append("")
        for item in self.explanations:
            lines.append(item.to_markdown())
        if self.caveats:
            lines.append("## Caveats")
            lines.append("")
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
            lines.append("")
        return "\n".join(lines)


#: The prefix of a class-probability column in the predictor's output frame.
PROBABILITY_PREFIX = "p_"

#: The predictor's own prediction columns, read when they are present.
PREDICTION_KEYS: tuple[str, ...] = (
    "predicted_class",
    "predicted_probability",
    "runner_up_class",
    "runner_up_probability",
    "margin",
)


def _as_row(value: Any, *, what: str) -> dict[str, Any]:
    """``value`` as a plain mapping (a pandas Series or a frame's record both work)."""
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    raise ExplanationError(
        f"a {what} must be a mapping or a pandas row, got {type(value).__name__}"
    )


def _identity_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """The explained row's contract identity, or a refusal naming what is missing.

    ``timestamp`` and ``tradingsymbol`` are required - the same two keys
    :meth:`core.ml.predictor.ModelPredictor.predict_frame` refuses to score without, and
    for the same reason: an explanation that cannot say *which* bar it describes is not an
    explanation. The rest of ``IDENTITY_COLUMNS`` is rendered when the row carries it.
    """
    for key in ("timestamp", "tradingsymbol"):
        if key not in row:
            raise ExplanationError(
                f"the row has no {key!r} column, so an explanation cannot be attributed to a "
                "bar; pass the stored feature rows (core/storage/readers.py)"
            )
    return {name: _json_value(row.get(name)) for name in IDENTITY_COLUMNS}


def _probabilities_of(values: Mapping[str, Any]) -> dict[str, float] | None:
    """The class distribution carried by ``values`` (the predictor's ``p_*`` columns).

    Returns ``None`` when the mapping carries no ``p_*`` column at all - the case where no
    model was scored - so the AI block can say so instead of printing an empty list.
    """
    items = {
        name[len(PROBABILITY_PREFIX) :]: value
        for name, value in values.items()
        if name.startswith(PROBABILITY_PREFIX)
    }
    return _distribution(items, source="the scored row")


def _ai_section(probabilities: Mapping[str, float] | None) -> ExplanationSection:
    """Spec 26's last line: the model's class probabilities, best first.

    The whole distribution is stored (an artefact should be auditable in full); the
    presentation policy decides how many of them are *printed*.
    """
    if not probabilities:
        return ExplanationSection(
            key=AI_SECTION_KEY,
            title=AI_SECTION_TITLE,
            source="model",
            available=False,
            reason="no model was scored for this row, so there is nothing to explain",
        )
    ranked = sorted(probabilities.items(), key=lambda item: -item[1])
    return ExplanationSection(
        key=AI_SECTION_KEY,
        title=AI_SECTION_TITLE,
        source="model",
        probabilities=[
            ClassProbability(name=name, probability=float(value), rank=rank)
            for rank, (name, value) in enumerate(ranked)
        ],
    )


def _fill_block(
    block: ExplanationBlock, values: Mapping[str, Any], *, policy: ExplanationPolicy
) -> ExplanationSection:
    """One configured block, rendered from ``values`` - or a refusal naming what is absent."""
    missing = [source.column for source in block.sources if source.column not in values]
    if missing:
        raise ExplanationError(
            f"the row does not carry {len(missing)} column(s) that block {block.key!r} "
            f"explains: {missing[:8]}{' ...' if len(missing) > 8 else ''} - explain the frame "
            "the model was actually scored on, or point model.explainability at columns the "
            "stored row has"
        )
    entries: list[ExplanationEntry] = []
    for source in block.sources:
        text, kind, known = format_value(
            values[source.column], digits=policy.digits, missing_text=policy.missing_text
        )
        entries.append(
            ExplanationEntry(
                column=source.column,
                group=source.group,
                kind=kind,
                value=_json_value(values[source.column]),
                text=text,
                known=known,
            )
        )
    return ExplanationSection(key=block.key, title=block.title, entries=entries)


def _records(value: Any, *, what: str) -> list[dict[str, Any]]:
    """``value`` as a list of plain rows (a pandas frame, or any iterable of rows)."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict) and hasattr(value, "columns"):  # a pandas DataFrame
        return [dict(record) for record in value.to_dict(orient="records")]
    return [_as_row(item, what=what) for item in value]


def _order(rows: Sequence[Mapping[str, Any]], *, newest_first: bool) -> list[int]:
    """Row indices, newest first, equal stamps in stored order, a missing stamp last."""
    stamps: list[pd.Timestamp] = []
    for row in rows:
        stamp = pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)
        # `sorted` is stable, so the `reverse=True` pass keeps equal stamps in stored order
        # (the same minute's CE and PE rows stay in the order the store returned them) and
        # an unparseable stamp sinks to the end instead of masquerading as the newest row.
        stamps.append(pd.Timestamp.min.tz_localize("UTC") if pd.isna(stamp) else stamp)
    return sorted(range(len(rows)), key=lambda index: stamps[index], reverse=newest_first)



def _distribution(values: Mapping[str, Any], *, source: str) -> dict[str, float] | None:
    """A class distribution from a ``{class: probability}`` mapping, or a refusal."""
    if not values:
        return None
    distribution: dict[str, float] = {}
    for name, value in values.items():
        if _is_missing(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ExplanationError(
                f"{source} holds {name}={value!r}, which is not a number"
            ) from None
        if not 0.0 <= number <= 1.0:
            raise ExplanationError(
                f"{source} holds {name}={number}, which is outside [0, 1] and is therefore "
                "not a probability"
            )
        distribution[str(name)] = number
    return distribution or None


def explain_row(
    row: Any,
    *,
    policy: ExplanationPolicy | None = None,
    probabilities: Mapping[str, float] | None = None,
    prediction: Any = None,
    model: Mapping[str, Any] | None = None,
) -> Explanation:
    """Explain one prediction: the configured blocks from ``row``, the AI block from the model.

    ``row`` is a stored feature row (a mapping or a pandas row). The model's block is taken
    from either ``prediction`` - a row of the scored frame, i.e. what
    :meth:`core.ml.predictor.ModelPredictor.predict_frame` returned, carrying both the
    ``p_<class>`` columns and its own top pick - or from ``probabilities``, a plain
    ``{class: probability}`` mapping. Passing both is refused rather than guessed at, and
    passing neither is honest and allowed: the AI block then says it was not scored.
    """
    policy = policy or ExplanationPolicy()
    if prediction is not None and probabilities is not None:
        raise ExplanationError(
            "pass either `prediction` (the scored row) or `probabilities` (a distribution), "
            "not both: two sources for one block is how a distribution gets contradicted"
        )
    values = _as_row(row, what="feature row")
    distribution: dict[str, float] | None = None
    top: dict[str, Any] = {}
    if prediction is not None:
        scored = _as_row(prediction, what="prediction")
        distribution = _probabilities_of(scored)
        top = {
            key: _json_value(scored.get(key))
            for key in PREDICTION_KEYS
            if not _is_missing(scored.get(key))
        }
    elif probabilities is not None:
        distribution = _distribution(probabilities, source="`probabilities`")

    identity = _identity_of(values)
    sections = [_fill_block(block, values, policy=policy) for block in policy.blocks]
    ai_section = _ai_section(distribution)
    sections.append(ai_section)

    predicted_class = top.get("predicted_class")
    predicted_probability = top.get("predicted_probability")
    runner_up_class = top.get("runner_up_class")
    runner_up_probability = top.get("runner_up_probability")
    margin = top.get("margin")
    if predicted_class is None and ai_section.probabilities:
        # No scored row was handed in, so the model's own ranking is the pick - which is
        # exactly what the predictor's `predicted_class` column holds.
        ranked = ai_section.probabilities
        predicted_class = ranked[0].name
        predicted_probability = ranked[0].probability
        if len(ranked) > 1:
            runner_up_class = ranked[1].name
            runner_up_probability = ranked[1].probability
    if (
        margin is None
        and isinstance(predicted_probability, float)
        and isinstance(runner_up_probability, float)
    ):
        margin = predicted_probability - runner_up_probability

    return Explanation(
        policy=policy,
        identity=identity,
        sections=sections,
        audit={name: _json_value(values.get(name)) for name in AUDIT_COLUMNS if name in values},
        predicted_class=predicted_class if predicted_class is None else str(predicted_class),
        predicted_probability=predicted_probability,
        runner_up_class=runner_up_class if runner_up_class is None else str(runner_up_class),
        runner_up_probability=runner_up_probability,
        margin=margin,
        model=dict(model) if model is not None else None,
    )


def explain_frame(
    frame: Any,
    *,
    policy: ExplanationPolicy | None = None,
    predicted: Any = None,
    model: Mapping[str, Any] | None = None,
    limit: int | None = None,
    newest_first: bool = True,
) -> list[Explanation]:
    """Explain the rows of ``frame``, newest first, optionally capped at ``limit`` rows.

    ``predicted`` is the frame :meth:`core.ml.predictor.ModelPredictor.predict_frame`
    returned for exactly this frame. It is checked row by row (length and the
    ``(timestamp, tradingsymbol)`` pair) rather than trusted: probabilities attributed to
    the wrong bar would produce a confident, wrong explanation, which is the one failure
    mode worth spending a comparison on. ``predicted=None`` is allowed - the AI block then
    says the row was not scored.
    """
    policy = policy or ExplanationPolicy()
    rows = _records(frame, what="feature row")
    scores: list[dict[str, Any]] | None = None
    if predicted is not None:
        scores = _records(predicted, what="scored row")
        if len(scores) != len(rows):
            raise ExplanationError(
                f"the scored frame has {len(scores)} row(s) and the feature frame {len(rows)}: "
                "the predictor returns exactly one scored row per input row"
            )
        for position, (left, right) in enumerate(zip(rows, scores, strict=True)):
            for key in ("timestamp", "tradingsymbol"):
                if left.get(key) != right.get(key):
                    raise ExplanationError(
                        f"scored row {position} belongs to {right.get(key)!r}, not to the "
                        f"feature frame's {key!r} {left.get(key)!r}; score the frame you are "
                        "explaining rather than joining two runs by eye"
                    )
    order = _order(rows, newest_first=newest_first)
    if limit is not None:
        if limit < 1:
            raise ExplanationError(f"limit must be at least 1, got {limit}")
        order = order[:limit]
    return [
        explain_row(
            rows[index],
            policy=policy,
            prediction=None if scores is None else scores[index],
            model=model,
        )
        for index in order
    ]


def build_explanations(
    frame: Any,
    *,
    policy: ExplanationPolicy | None = None,
    predicted: Any = None,
    model: Mapping[str, Any] | None = None,
    limit: int | None = None,
    newest_first: bool = True,
    window: tuple[Any, Any] = (None, None),
    symbols: Sequence[str] = (),
    timeframe: str | None = None,
    horizon_minutes: int | None = None,
    caveats: Sequence[str] = (),
) -> ExplanationBatch:
    """The batch an artefact is written from: the explanations plus what produced them."""
    resolved = policy or ExplanationPolicy()
    return ExplanationBatch(
        policy=resolved,
        explanations=explain_frame(
            frame,
            policy=resolved,
            predicted=predicted,
            model=model,
            limit=limit,
            newest_first=newest_first,
        ),
        model=dict(model) if model is not None else None,
        window=window,
        symbols=tuple(symbols),
        timeframe=timeframe,
        horizon_minutes=horizon_minutes,
        caveats=list(caveats),
    )










