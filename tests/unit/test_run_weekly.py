"""Tests for the ``--weekly`` cadence steps of ``scripts/run_daily_pipeline.py``.

Adding steps to the chain is exactly the change that could silently alter what a
plain ``run_daily_pipeline`` does, so the contract is pinned here: without
``--weekly`` the chain is byte-for-byte the eight daily steps, and the cadence
steps appear only when asked for, in the documented order, with the right argv.
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts import run_daily_pipeline as pipeline

WEEKLY_KEYS = ("report", "explain", "importance", "charts", "accumulation")


def test_weekly_step_keys_are_the_documented_order() -> None:
    assert pipeline.WEEKLY_STEP_KEYS == WEEKLY_KEYS
    # The daily chain itself is untouched.
    assert pipeline.STEP_KEYS == (
        "candles",
        "greeks",
        "features",
        "labels",
        "validate",
        "dataset",
        "train",
        "backtest",
    )


def test_select_steps_defaults_to_the_daily_chain_only() -> None:
    assert [step.key for step in pipeline.select_steps()] == list(pipeline.STEP_KEYS)


def test_select_steps_appends_the_weekly_steps_when_asked() -> None:
    keys = [step.key for step in pipeline.select_steps(weekly=True)]
    assert keys == [*pipeline.STEP_KEYS, *WEEKLY_KEYS]


def test_select_steps_can_skip_a_weekly_step() -> None:
    keys = [step.key for step in pipeline.select_steps(["charts"], weekly=True)]
    assert "charts" not in keys
    assert "report" in keys


def test_an_unknown_skip_still_points_at_every_known_step() -> None:
    with pytest.raises(pipeline.PipelineError) as excinfo:
        pipeline.select_steps(["nope"], weekly=True)
    assert "report" in str(excinfo.value)


def test_weekly_flag_parses() -> None:
    assert pipeline.parse_args(["--weekly"]).weekly is True
    assert pipeline.parse_args([]).weekly is False


def test_step_argv_for_a_weekly_step_targets_the_ai_app() -> None:
    options = pipeline.RunOptions(day=date(2026, 9, 29), horizons=(10,))
    report = pipeline.STEPS_BY_KEY["report"]
    argv = pipeline.step_argv(report, options)
    assert argv[1:4] == ["-m", "apps.anjaneeya_ai.run", "--mode"]
    assert argv[4] == "report"
    assert "--horizon" in argv and "10" in argv


def test_a_weekly_step_does_not_need_ticks() -> None:
    for key in WEEKLY_KEYS:
        assert pipeline.STEPS_BY_KEY[key].needs_ticks is False
