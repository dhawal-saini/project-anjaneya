"""Tests for :mod:`core.ml.explainer` - spec 26's per-prediction explanations.

Three rules carry most of the weight here, and each has its own section below:

* **the structure is configuration.** A policy built from the shipped ``config/model.yaml``
  and the module's own defaults must resolve to the same blocks, and a hand-built policy that
  names one column must explain exactly that column - so "do not hard-code" is a test, not a
  promise.
* **unknown is not zero.** A ``NULL`` feature is rendered with the configured word and
  ``known=False``, and a column the policy names but the row does not carry is a refusal.
* **the model's block is the model's.** The AI block ranks whatever classes the scored row
  carries, ties keep their declared order, and a scored row's own top pick wins over the
  ranking rather than being recomputed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from core.constants import FEATURE_COLUMNS
from core.ml.explainer import (
    AI_SECTION_KEY,
    AI_SECTION_TITLE,
    DECISION_LAYER_CAVEAT,
    EXPLANATION_CAVEAT,
    FRESHNESS_CAVEAT,
    RESERVED_EXPLANATION_GROUPS,
    ClassProbability,
    ExplanationBlock,
    ExplanationError,
    ExplanationPolicy,
    ExplanationSource,
    build_explanations,
    explain_frame,
    explain_row,
    format_value,
)

#: The four label classes ``config/model.yaml`` ships, as a scored row would carry them.
CLASSES = ("UP_CONTINUATION", "DOWN_CONTINUATION", "RANGE", "REVERSAL")

#: A complete stored feature row: every column of the contract, most of them ``NULL``.
def row(**overrides):
    """One stored feature row with the values a real row would carry at that instant."""
    values = dict.fromkeys(FEATURE_COLUMNS)
    values.update(
        {
            "timestamp": pd.Timestamp("2026-10-01T09:20:00+05:30"),
            "underlying": "NIFTY",
            "tradingsymbol": "NIFTY06OCT2625000CE",
            "instrument_token": 12345,
            "expiry": date(2026, 10, 6),
            "strike": 25000.0,
            "option_type": "CE",
            "timeframe": "1m",
            "nifty_regime": "BULLISH",
            "nifty_composite_trend_score": 0.34214,
            "option_trend_score": -0.5,
            "option_insufficient_history": False,
            "candle_body": 0.5,
            "candle_body_ratio": 0.4,
            "support_touch_count": 1,
            "days_to_expiry": 2.0,
            "time_of_day_bucket": "OPEN_915_930",
            "day_of_week": 3,
            "volume": 100,
            "oi": 5000,
            "source": "DERIVED",
            "fetch_timestamp": pd.Timestamp("2026-10-01T09:20:59+05:30"),
            "quality_flag": "OK",
            "run_id": "run-1",
        }
    )
    values.update(overrides)
    return values


def scored_row(**overrides):
    """The predictor's output row for :func:`row`: the ``p_*`` columns and its own pick."""
    values = {
        "timestamp": row()["timestamp"],
        "tradingsymbol": row()["tradingsymbol"],
        "p_UP_CONTINUATION": 0.5,
        "p_DOWN_CONTINUATION": 0.1,
        "p_RANGE": 0.3,
        "p_REVERSAL": 0.1,
        "predicted_class": "UP_CONTINUATION",
        "predicted_probability": 0.5,
        "runner_up_class": "RANGE",
        "runner_up_probability": 0.3,
        "margin": 0.2,
    }
    values.update(overrides)
    return values


def distribution() -> dict[str, float]:
    """A plain ``{class: probability}`` mapping, as ``--mode explain``'s AI block sees one."""
    return {
        "UP_CONTINUATION": 0.5,
        "DOWN_CONTINUATION": 0.1,
        "RANGE": 0.3,
        "REVERSAL": 0.1,
    }


# ---------------------------------------------------------------------------
# The policy: structure is configuration, not code
# ---------------------------------------------------------------------------
def test_the_default_policy_is_the_shipped_configuration(config) -> None:
    """A caller with no configuration gets the documented one, block for block."""
    shipped = ExplanationPolicy.from_config(config)
    default = ExplanationPolicy()

    assert default.blocks == shipped.blocks
    assert default.digits == shipped.digits == 4
    assert default.probabilities == shipped.probabilities == 3
    assert default.rows == shipped.rows == 5
    assert default.missing_text == shipped.missing_text == "unknown"
    assert [block.key for block in shipped.blocks] == [
        "nifty",
        "option",
        "price_action",
        "candle",
        "support_resistance",
        "context",
    ]


def test_every_explained_column_is_a_stored_feature(config) -> None:
    """No block may name a column the contract does not carry."""
    policy = ExplanationPolicy.from_config(config)

    assert policy.columns
    assert set(policy.columns) <= set(FEATURE_COLUMNS)
    # the identity and provenance groups are the explanation's envelope, never a block
    assert not set(policy.columns) & {"tradingsymbol", "timestamp", "run_id", "source"}
    assert policy.explains("nifty_regime")
    assert not policy.explains("nifty_regime_typo")


def test_a_hand_built_policy_explains_exactly_what_it_names() -> None:
    """The "do not hard-code" rule: a policy is the whole structure."""
    policy = ExplanationPolicy(
        digits=2,
        blocks=(
            ExplanationBlock(
                key="mine",
                title="MINE",
                sources=(ExplanationSource(column="candle_body_ratio", group=None),),
            ),
        ),
    )

    explanation = explain_row(row(), policy=policy, probabilities=distribution())

    assert [section.key for section in explanation.sections] == ["mine", AI_SECTION_KEY]
    assert [entry.column for entry in explanation.sections[0].entries] == ["candle_body_ratio"]
    assert explanation.section("mine").entries[0].text == "0.4"
    assert explanation.section("mine").entries[0].group is None


def test_a_policy_that_would_render_nothing_or_twice_is_refused() -> None:
    block = ExplanationBlock("a", "A", (ExplanationSource("candle_body", "candle"),))
    same = ExplanationBlock("b", "B", (ExplanationSource("candle_body", "candle"),))

    with pytest.raises(ExplanationError, match="explained twice"):
        ExplanationPolicy(blocks=(block, same))
    with pytest.raises(ExplanationError, match="no blocks"):
        ExplanationPolicy(blocks=())
    with pytest.raises(ExplanationError, match="explains nothing"):
        ExplanationPolicy(blocks=(ExplanationBlock("a", "A", ()),))
    with pytest.raises(ExplanationError, match="reserved for the model"):
        ExplanationPolicy(blocks=(ExplanationBlock(AI_SECTION_KEY, "AI", (
            ExplanationSource("candle_body", "candle"),
        )),))
    with pytest.raises(ExplanationError, match="unique"):
        ExplanationPolicy(blocks=(block, block))


def test_the_numeric_knobs_are_validated() -> None:
    block = ExplanationBlock("a", "A", (ExplanationSource("candle_body", "candle"),))
    with pytest.raises(ExplanationError, match="probabilities"):
        ExplanationPolicy(probabilities=0, blocks=(block,))
    with pytest.raises(ExplanationError, match="digits"):
        ExplanationPolicy(digits=-1, blocks=(block,))
    with pytest.raises(ExplanationError, match="rows"):
        ExplanationPolicy(rows=0, blocks=(block,))
    with pytest.raises(ExplanationError, match="missing_text"):
        ExplanationPolicy(missing_text="", blocks=(block,))


def test_the_policy_describes_itself_for_the_artefact(config) -> None:
    described = ExplanationPolicy.from_config(config).to_dict()

    assert described["spec"] == "26"
    assert described["missing_text"] == "unknown"
    assert [block["key"] for block in described["blocks"]] == [
        "nifty",
        "option",
        "price_action",
        "candle",
        "support_resistance",
        "context",
    ]
    assert described["columns"] == list(ExplanationPolicy.from_config(config).columns)
    assert len(described["columns"]) == len(set(described["columns"]))


# ---------------------------------------------------------------------------
# Values: unknown, never zero
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, float("nan"), pd.NaT, pd.NA])
def test_a_missing_value_is_the_configured_word_and_not_a_zero(value) -> None:
    text, kind, known = format_value(value, digits=4, missing_text="unknown")

    assert (text, kind, known) == ("unknown", "unknown", False)
    assert text != "0"


def test_the_missing_word_comes_from_the_policy() -> None:
    text, _, known = format_value(None, digits=4, missing_text="n/a")

    assert (text, known) == ("n/a", False)


def test_numbers_are_printed_with_at_most_the_configured_decimals() -> None:
    assert format_value(0.342149, digits=4, missing_text="?")[0] == "0.3421"
    assert format_value(0.342149, digits=2, missing_text="?")[0] == "0.34"
    # trailing zeros are trimmed: 0.5 is not dressed up as 0.5000
    assert format_value(0.5, digits=4, missing_text="?")[0] == "0.5"
    assert format_value(25000.0, digits=4, missing_text="?")[0] == "25000"
    assert format_value(0.0, digits=4, missing_text="?")[0] == "0"
    assert format_value(-0.00001, digits=4, missing_text="?")[0] == "0"
    assert format_value(7, digits=4, missing_text="?")[0] == "7"
    assert format_value(np.float64(0.25), digits=4, missing_text="?")[0] == "0.25"
    assert format_value(np.int64(5000), digits=4, missing_text="?")[0] == "5000"


def test_zero_is_a_measurement_and_stays_known() -> None:
    text, kind, known = format_value(0.0, digits=4, missing_text="unknown")

    assert (text, kind, known) == ("0", "number", True)


def test_flags_timestamps_and_categories_keep_their_shape() -> None:
    assert format_value(False, digits=4, missing_text="?") == ("false", "flag", True)
    assert format_value(True, digits=4, missing_text="?") == ("true", "flag", True)
    assert format_value(np.bool_(True), digits=4, missing_text="?") == ("true", "flag", True)
    moment = pd.Timestamp("2026-10-01T09:20:00+05:30")
    assert format_value(moment, digits=4, missing_text="?") == (moment.isoformat(), "timestamp", True)
    assert format_value("BULLISH", digits=4, missing_text="?") == ("BULLISH", "category", True)
    assert format_value(date(2026, 10, 6), digits=4, missing_text="?") == (
        "2026-10-06",
        "timestamp",
        True,
    )
    naive = datetime(2026, 10, 1, 9, 20, tzinfo=UTC)
    assert format_value(naive, digits=4, missing_text="?")[1] == "timestamp"


# ---------------------------------------------------------------------------
# One prediction
# ---------------------------------------------------------------------------
def test_a_row_is_explained_block_by_block_and_the_model_goes_last(config) -> None:
    policy = ExplanationPolicy.from_config(config)

    explanation = explain_row(row(), policy=policy, prediction=scored_row())

    assert [section.key for section in explanation.sections] == [
        *(block.key for block in policy.blocks),
        AI_SECTION_KEY,
    ]
    assert explanation.identity["tradingsymbol"] == "NIFTY06OCT2625000CE"
    assert explanation.identity["option_type"] == "CE"
    assert explanation.entry("nifty_regime").text == "BULLISH"
    # the identity and provenance columns are the envelope, not a block
    assert explanation.entry("run_id") is None
    assert explanation.entries == len(policy.columns)


def test_the_audit_block_is_the_rows_own_provenance() -> None:
    explanation = explain_row(row(), probabilities=distribution())

    assert explanation.audit == {
        "source": "DERIVED",
        "fetch_timestamp": "2026-10-01T09:20:59+05:30",
        "quality_flag": "OK",
        "run_id": "run-1",
    }


def test_a_null_feature_is_unknown_in_the_row_and_in_the_json() -> None:
    explanation = explain_row(row(support_strength=None), probabilities=distribution())

    entry = explanation.entry("support_strength")
    assert entry.known is False
    assert entry.text == "unknown"
    assert entry.value is None
    assert entry.to_dict() == {
        "column": "support_strength",
        "group": "support_resistance",
        "kind": "unknown",
        "value": None,
        "text": "unknown",
        "known": False,
    }
    assert explanation.unknown >= 1


def test_a_column_the_row_does_not_carry_is_a_refusal_not_a_blank() -> None:
    incomplete = row()
    del incomplete["nifty_regime"]

    with pytest.raises(ExplanationError, match="does not carry 1 column"):
        explain_row(incomplete, probabilities=distribution())


def test_a_row_without_identity_cannot_be_explained() -> None:
    without_stamp = row()
    del without_stamp["timestamp"]

    with pytest.raises(ExplanationError, match="no 'timestamp' column"):
        explain_row(without_stamp)


def test_without_a_model_the_ai_block_says_so() -> None:
    explanation = explain_row(row())

    ai_section = explanation.section(AI_SECTION_KEY)
    assert ai_section.available is False
    assert ai_section.probabilities == []
    assert ai_section.reason == "no model was scored for this row, so there is nothing to explain"
    assert explanation.predicted_class is None
    assert "not scored" in explanation.to_text()
    assert "not answered" in explanation.to_markdown()


def test_the_model_block_ranks_the_distribution_and_finds_the_runner_up() -> None:
    explanation = explain_row(row(), probabilities=distribution())

    ranked = [item.name for item in explanation.section(AI_SECTION_KEY).probabilities]
    assert ranked == ["UP_CONTINUATION", "RANGE", "DOWN_CONTINUATION", "REVERSAL"]
    assert explanation.predicted_class == "UP_CONTINUATION"
    assert explanation.predicted_probability == pytest.approx(0.5)
    assert explanation.runner_up_class == "RANGE"
    assert explanation.margin == pytest.approx(0.2)


def test_a_tie_keeps_the_order_the_classes_came_in() -> None:
    tied = {"RANGE": 0.25, "UP_CONTINUATION": 0.25, "REVERSAL": 0.25, "DOWN_CONTINUATION": 0.25}

    explanation = explain_row(row(), probabilities=tied)

    assert [item.name for item in explanation.section(AI_SECTION_KEY).probabilities] == [
        "RANGE",
        "UP_CONTINUATION",
        "REVERSAL",
        "DOWN_CONTINUATION",
    ]
    assert explanation.predicted_class == "RANGE"


def test_a_scored_rows_own_pick_is_not_recomputed() -> None:
    """The model's metadata order and its pick are the model's business, not ours."""
    explanation = explain_row(
        row(),
        prediction=scored_row(
            predicted_class="REVERSAL",
            predicted_probability=0.4,
            runner_up_class="UP_CONTINUATION",
            runner_up_probability=0.35,
            margin=0.05,
        ),
    )

    assert explanation.predicted_class == "REVERSAL"
    assert explanation.runner_up_class == "UP_CONTINUATION"
    assert explanation.margin == pytest.approx(0.05)
    # the block still ranks the distribution it was handed
    assert explanation.section(AI_SECTION_KEY).probabilities[0].name == "UP_CONTINUATION"


def test_a_probability_that_is_not_a_probability_is_refused() -> None:
    with pytest.raises(ExplanationError, match="outside"):
        explain_row(row(), probabilities={"UP_CONTINUATION": 1.5})
    with pytest.raises(ExplanationError, match="not a number"):
        explain_row(row(), probabilities={"UP_CONTINUATION": "high"})


def test_two_sources_for_one_block_are_refused() -> None:
    with pytest.raises(ExplanationError, match="not both"):
        explain_row(row(), probabilities=distribution(), prediction=scored_row())


def test_the_text_and_markdown_forms_carry_the_same_lines() -> None:
    explanation = explain_row(row(), probabilities=distribution())

    text = explanation.to_text()
    markdown = explanation.to_markdown()
    assert text.splitlines()[0] == explanation.label
    assert "prediction : UP_CONTINUATION p=0.5" in text
    assert "nifty_regime" in text and "BULLISH" in text
    assert "support_strength" in text and "unknown" in text
    assert any(line.strip() == f"{AI_SECTION_TITLE}:" for line in text.splitlines())
    assert markdown.startswith(f"### {explanation.label}")
    assert "`nifty_regime`: BULLISH" in markdown
    assert "`UP_CONTINUATION`: 0.5 **(predicted)**" in markdown
    assert "(top 3 of 4 class probabilities" in text


# ---------------------------------------------------------------------------
# A frame of predictions, and the artefact that carries them
# ---------------------------------------------------------------------------
def frame() -> pd.DataFrame:
    """Three stored rows (09:20, 09:21, 09:22), ready to be scored."""
    moments = [
        pd.Timestamp("2026-10-01T09:20:00+05:30"),
        pd.Timestamp("2026-10-01T09:21:00+05:30"),
        pd.Timestamp("2026-10-01T09:22:00+05:30"),
    ]
    return pd.DataFrame(
        [row(timestamp=moment) for moment in moments], columns=list(FEATURE_COLUMNS)
    )


def scored(frame_: pd.DataFrame, *, top: str = "UP_CONTINUATION") -> pd.DataFrame:
    """The predictor's output for ``frame_``: one row per input row, in the same order."""
    records = []
    for position, record in enumerate(frame_.to_dict(orient="records")):
        probability = 0.5 + position / 100
        runner_up = 0.3 - position / 100
        records.append(
            {
                "timestamp": record["timestamp"],
                "tradingsymbol": record["tradingsymbol"],
                "p_UP_CONTINUATION": probability,
                "p_DOWN_CONTINUATION": 0.1,
                "p_RANGE": runner_up,
                "p_REVERSAL": 0.1,
                "predicted_class": top,
                "predicted_probability": probability,
                "runner_up_class": "RANGE",
                "runner_up_probability": runner_up,
                "margin": probability - runner_up,
            }
        )
    return pd.DataFrame(records)


def test_the_newest_rows_come_first_and_the_limit_caps_them() -> None:
    frame_ = frame()

    explanations = explain_frame(frame_, predicted=scored(frame_), limit=2)

    assert [item.identity["timestamp"] for item in explanations] == [
        "2026-10-01T09:22:00+05:30",
        "2026-10-01T09:21:00+05:30",
    ]
    assert len(explain_frame(frame_, predicted=scored(frame_))) == 3
    oldest_first = explain_frame(frame_, newest_first=False)
    assert [item.identity["timestamp"] for item in oldest_first] == [
        "2026-10-01T09:20:00+05:30",
        "2026-10-01T09:21:00+05:30",
        "2026-10-01T09:22:00+05:30",
    ]


def test_a_scored_frame_that_does_not_match_the_rows_is_refused() -> None:
    frame_ = frame()
    shifted = scored(frame_).iloc[1:].reset_index(drop=True)

    with pytest.raises(ExplanationError, match="one scored row per input row"):
        explain_frame(frame_, predicted=shifted)
    mismatched = scored(frame_)
    mismatched.loc[0, "tradingsymbol"] = "NIFTY06OCT2625000PE"
    with pytest.raises(ExplanationError, match="score the frame you are explaining"):
        explain_frame(frame_, predicted=mismatched)
    with pytest.raises(ExplanationError, match="limit must be at least 1"):
        explain_frame(frame_, limit=0)


def test_the_batch_carries_the_policy_the_window_and_the_caveats() -> None:
    frame_ = frame()
    batch = build_explanations(
        frame_,
        predicted=scored(frame_),
        model={"path": "model.txt", "classes": list(CLASSES)},
        window=("2026-10-01T09:20:00+05:30", "2026-10-01T09:23:00+05:30"),
        symbols=("NIFTY06OCT2625000CE",),
        timeframe="1m",
        horizon_minutes=5,
        caveats=(EXPLANATION_CAVEAT, DECISION_LAYER_CAVEAT, FRESHNESS_CAVEAT),
    )

    assert batch.rows == 3
    assert batch.entries == 3 * len(batch.policy.columns)
    assert batch.unknown > 0
    assert batch.section_keys()[-1] == AI_SECTION_KEY
    document = batch.to_dict()
    assert document["model"]["path"] == "model.txt"
    assert document["window"]["timeframe"] == "1m"
    assert document["window"]["horizon_minutes"] == 5
    assert document["window"]["symbols"] == ["NIFTY06OCT2625000CE"]
    assert document["counts"] == {
        "predictions": 3,
        "entries": 3 * len(batch.policy.columns),
        "unknown": batch.unknown,
    }
    assert document["caveats"] == [EXPLANATION_CAVEAT, DECISION_LAYER_CAVEAT, FRESHNESS_CAVEAT]
    assert len(document["explanations"]) == 3


def test_the_batch_renders_a_console_document_and_a_markdown_one() -> None:
    frame_ = frame()
    symbol = str(frame_.loc[0, "tradingsymbol"])
    batch = build_explanations(
        frame_,
        predicted=scored(frame_),
        model={"path": "model.txt"},
        timeframe="1m",
        horizon_minutes=5,
        caveats=(DECISION_LAYER_CAVEAT,),
    )

    text = batch.to_text()
    markdown = batch.to_markdown()
    assert text.splitlines()[0].startswith(
        "ANJANEYA-AI explanations (spec 26) - 3 prediction(s)"
    )
    assert "  timeframe  : 1m 5m" in text
    assert "  model      : model.txt" in text
    assert text.count("  prediction : UP_CONTINUATION") == 3
    assert f"  caveat: {DECISION_LAYER_CAVEAT}" in text
    assert markdown.startswith("# ANJANEYA-AI explanations (spec 26)")
    assert "## Caveats" in markdown
    assert markdown.count(f"### {symbol} @") == 3
    assert "plus the AI block" in markdown


def test_an_empty_frame_produces_an_empty_batch_rather_than_a_crash() -> None:
    empty = frame().iloc[0:0]

    batch = build_explanations(empty, timeframe="1m")

    assert batch.rows == 0
    assert batch.entries == 0
    assert batch.to_dict()["explanations"] == []
    assert "0 prediction(s)" in batch.to_text()


def test_the_sections_are_serialised_with_their_kinds_and_probability_ranks() -> None:
    frame_ = frame()
    batch = build_explanations(frame_, predicted=scored(frame_), limit=1)

    section = batch.explanations[0].to_dict()["sections"][-1]
    assert section["key"] == AI_SECTION_KEY
    assert section["source"] == "model"
    assert section["available"] is True
    assert [item["rank"] for item in section["probabilities"]] == list(range(len(CLASSES)))
    assert section["probabilities"][0]["class"] == "UP_CONTINUATION"
    assert isinstance(ClassProbability("RANGE", 0.3, 1).to_dict()["probability"], float)
    assert ClassProbability("RANGE", 0.3, 1).top is False
    assert ClassProbability("RANGE", 0.3, 0).top is True


def test_the_reserved_groups_are_a_contract_a_policy_cannot_use() -> None:
    assert frozenset({"identity", "provenance"}) == RESERVED_EXPLANATION_GROUPS
    with pytest.raises(ExplanationError, match="reserved feature group"):
        ExplanationPolicy(
            blocks=(
                ExplanationBlock(
                    "identity", "IDENTITY", (ExplanationSource("timestamp", "identity"),)
                ),
            )
        )






