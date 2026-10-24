"""Tests for ``scripts/run_daily_pipeline.py`` - the one-command post-close chain.

Two things can go wrong with a pipeline script, and both are tested here:

1. **the order and the argv** - a step that runs against the wrong day, the wrong timeframe
   or with the wrong flags produces plausible-looking data that is quietly wrong. The child
   processes are replaced by a recorder, so every assertion is about the exact command line
   the runner would have executed;
2. **the verdict** - an operator (or a scheduler) has to be able to tell "clean", "a step
   broke", "bad configuration" and "the training gate refused" apart. The exit codes are
   pinned here, including the fact that a gate refusal skips the backtest instead of
   replaying a stale model.

The real chain - with the real child processes - is covered by
``tests/integration/test_daily_pipeline_e2e.py``; this file never spawns anything.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from scripts import run_daily_pipeline as pipeline
from tests.support.engine import DAY, seed_index_history, seed_ticks, session_ticks

TODAY = DAY.isoformat()


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def cli(tmp_path, *extra: str) -> list[str]:
    """Pipeline arguments rooted inside ``tmp_path``."""
    return [
        "--data-root",
        str(tmp_path / "data"),
        "--db-path",
        str(db_file(tmp_path)),
        *extra,
    ]


def rebase(rows, *, days: int):
    """Move seeded tick rows whole days forward (same wall-clock time)."""
    delta = timedelta(days=days)
    return [
        row.model_copy(
            update={
                "timestamp": row.timestamp + delta,
                "fetch_timestamp": row.fetch_timestamp + delta,
            }
        )
        for row in rows
    ]


@pytest.fixture
def recorder(monkeypatch):
    """Replace the child process with a recorder; tests script the exit codes by mode."""
    calls: list[list[str]] = []
    codes: dict[str, int] = {}

    def fake_run(argv, tee):
        calls.append(list(argv))
        mode = argv[argv.index("--mode") + 1]
        return codes.get(mode, pipeline.EXIT_OK)

    monkeypatch.setattr(pipeline, "_run_subprocess", fake_run)
    return calls, codes


def modes(calls: list[list[str]]) -> list[str]:
    """The ``--mode`` of every recorded call, in order."""
    return [call[call.index("--mode") + 1] for call in calls]


def options(tmp_path, *, timeframe=None, horizons=(), dry_run=False, force=False):
    """A :class:`RunOptions` rooted inside ``tmp_path``."""
    return pipeline.RunOptions(
        day=DAY,
        timeframe=timeframe,
        horizons=tuple(horizons),
        dry_run=dry_run,
        force=force,
        data_root=tmp_path / "data",
        db_path=db_file(tmp_path),
    )


# ---------------------------------------------------------------------------
# The step table
# ---------------------------------------------------------------------------
def test_the_chain_is_the_documented_order() -> None:
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
    assert [step.mode for step in pipeline.STEPS] == [
        "build-candles",
        "enrich-greeks",
        "build-features",
        "build-labels",
        "validate",
        "dataset",
        "train",
        "backtest",
    ]


def test_every_step_names_a_real_module_and_a_reason() -> None:
    for step in pipeline.STEPS:
        assert step.module in {"apps.anjaneeya_engine.run", "apps.anjaneeya_ai.run"}
        assert step.summary


def test_a_step_never_depends_on_a_later_one() -> None:
    """A dependency the chain cannot honour is a design bug, not a runtime surprise."""
    seen: list[str] = []
    for step in pipeline.STEPS:
        assert all(name in seen for name in step.depends_on), f"{step.key} depends on a later step"
        seen.append(step.key)


def test_only_the_engine_steps_read_one_day() -> None:
    """The engine builds the day that was captured; the AI steps read all history."""
    for step in pipeline.STEPS:
        assert step.whole_history == (step.app == pipeline.APP_AI), step.key


def test_only_the_tick_building_steps_need_ticks() -> None:
    """A features-only rerun must not require a capture session that never happened."""
    needs = {step.key for step in pipeline.STEPS if step.needs_ticks}
    assert needs == {"candles", "greeks", "features", "labels"}


def test_skip_drops_steps_and_keeps_the_order() -> None:
    steps = pipeline.select_steps(["validate", "greeks"])
    assert [step.key for step in steps] == [
        "candles",
        "features",
        "labels",
        "dataset",
        "train",
        "backtest",
    ]


def test_an_unknown_skip_name_is_a_configuration_error() -> None:
    with pytest.raises(pipeline.PipelineError, match="unknown --skip step"):
        pipeline.select_steps(["greekss"])


def test_parse_skip_ignores_blanks_and_spaces() -> None:
    assert pipeline.parse_skip("candles, , backtest ") == ["candles", "backtest"]
    assert pipeline.parse_skip("") == []
    assert pipeline.parse_skip(None) == []


# ---------------------------------------------------------------------------
# The command line handed to each step
# ---------------------------------------------------------------------------
def engine_argv(tmp_path, key: str, **kwargs):
    """The argv for one engine step."""
    step = pipeline.STEPS_BY_KEY[key]
    return pipeline.step_argv(step, options(tmp_path, **kwargs))


def ai_argv(tmp_path, key: str, **kwargs):
    """The argv for one AI step."""
    step = pipeline.STEPS_BY_KEY[key]
    return pipeline.step_argv(step, options(tmp_path, **kwargs))


def test_the_engine_steps_are_told_which_day_to_build(tmp_path) -> None:
    argv = engine_argv(tmp_path, "candles")
    assert argv[:4] == [
        pipeline.sys.executable,
        "-m",
        "apps.anjaneeya_engine.run",
        "--mode",
    ]
    assert argv[4] == "build-candles"
    assert argv[argv.index("--date") + 1] == TODAY
    assert argv[argv.index("--db-path") + 1] == str(db_file(tmp_path))
    assert argv[argv.index("--data-root") + 1] == str(tmp_path / "data")


def test_the_timeframe_is_only_passed_where_one_is_expected(tmp_path) -> None:
    """Candles and greeks build every configured timeframe: passing one would narrow them."""
    assert "--timeframes" not in engine_argv(tmp_path, "candles", timeframe="1m")
    assert "--timeframes" not in engine_argv(tmp_path, "greeks", timeframe="1m")
    for key in ("features", "labels", "validate"):
        argv = engine_argv(tmp_path, key, timeframe="1m")
        assert argv[argv.index("--timeframes") + 1] == "1m"
    assert "--timeframes" not in engine_argv(tmp_path, "features")


def test_the_ai_steps_read_all_history_but_keep_the_timeframe_and_horizon(tmp_path) -> None:
    argv = ai_argv(tmp_path, "dataset", timeframe="1m", horizons=(5, 10))
    assert "--date" not in argv
    assert argv[argv.index("--timeframe") + 1] == "1m"
    assert [argv[index + 1] for index, item in enumerate(argv) if item == "--horizon"] == ["5", "10"]


def test_train_gets_force_and_the_backtest_gets_exactly_one_horizon(tmp_path) -> None:
    train = ai_argv(tmp_path, "train", horizons=(5, 10), force=True)
    assert "--force" in train

    backtest = ai_argv(tmp_path, "backtest", horizons=(5,))
    assert backtest.count("--horizon") == 1
    assert backtest[backtest.index("--horizon") + 1] == "5"


def test_force_is_not_forwarded_to_the_other_steps(tmp_path) -> None:
    for key in ("candles", "features", "dataset", "backtest"):
        assert "--force" not in pipeline.step_argv(
            pipeline.STEPS_BY_KEY[key], options(tmp_path, force=True)
        )


def test_a_dry_run_is_write_free_everywhere(tmp_path) -> None:
    for step in pipeline.STEPS:
        argv = pipeline.step_argv(step, options(tmp_path, dry_run=True))
        assert "--dry-run" in argv
        assert "--no-run-artifacts" in argv


def test_a_normal_run_keeps_the_run_artefacts(tmp_path) -> None:
    argv = pipeline.step_argv(pipeline.STEPS_BY_KEY["candles"], options(tmp_path))
    assert "--no-run-artifacts" not in argv
    assert "--dry-run" not in argv


# ---------------------------------------------------------------------------
# Validating the flags against the configuration
# ---------------------------------------------------------------------------
def namespace(**overrides):
    """A parsed-namespace double with pipeline defaults."""
    base = {
        "config_dir": None,
        "data_root": None,
        "db_path": None,
        "date": DAY,
        "timeframe": None,
        "horizons": None,
        "skip": None,
        "force": False,
        "dry_run": False,
        "continue_on_error": False,
        "as_json": False,
        "log_file": None,
    }
    base.update(overrides)
    return pipeline.argparse.Namespace(**base)


def test_resolve_options_accepts_the_configured_values(config, tmp_path) -> None:
    resolved = pipeline.resolve_options(
        namespace(timeframe="1m", horizons=[5]), config, day=DAY
    )

    assert resolved.timeframe == "1m"
    assert resolved.horizons == (5,)
    assert resolved.day == DAY
    assert resolved.db_path == config.db_path


def test_an_unknown_timeframe_is_refused_before_anything_runs(config) -> None:
    with pytest.raises(pipeline.PipelineError, match="not one of the configured timeframes"):
        pipeline.resolve_options(namespace(timeframe="7m"), config, day=DAY)


def test_an_unknown_horizon_is_refused_before_anything_runs(config) -> None:
    with pytest.raises(pipeline.PipelineError, match="not in model.horizons_minutes"):
        pipeline.resolve_options(namespace(horizons=[7]), config, day=DAY)


def test_two_horizons_and_a_backtest_is_refused_with_a_way_out(config) -> None:
    with pytest.raises(pipeline.PipelineError, match="replays one model"):
        pipeline.resolve_options(namespace(horizons=[5, 10]), config, day=DAY)


def test_two_horizons_without_the_backtest_is_allowed(config) -> None:
    resolved = pipeline.resolve_options(
        namespace(horizons=[5, 10], skip="backtest"), config, day=DAY
    )
    assert resolved.horizons == (5, 10)


def test_duplicate_horizons_are_collapsed(config) -> None:
    resolved = pipeline.resolve_options(
        namespace(horizons=[5, 5, 10], skip="backtest"), config, day=DAY
    )
    assert resolved.horizons == (5, 10)



# ---------------------------------------------------------------------------
# The chain's behaviour
# ---------------------------------------------------------------------------
def seeded_store(tmp_path, *, day: date = DAY) -> None:
    """A store holding one synthetic session of ticks - and no index history."""
    seed_ticks(db_file(tmp_path), rebase(session_ticks(), days=(day - DAY).days))
    seed_index_history(db_file(tmp_path), [])


def test_the_first_failure_stops_the_chain(tmp_path, recorder) -> None:
    calls, codes = recorder
    codes["build-features"] = pipeline.EXIT_FAILED
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY, "--json"))

    assert exit_code == pipeline.EXIT_FAILED
    assert modes(calls) == ["build-candles", "enrich-greeks", "build-features"]


def test_continue_on_error_presses_on_and_still_fails(tmp_path, recorder) -> None:
    calls, codes = recorder
    codes["enrich-greeks"] = pipeline.EXIT_FAILED
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY, "--continue-on-error"))

    assert exit_code == pipeline.EXIT_FAILED
    assert modes(calls) == [
        "build-candles",
        "enrich-greeks",
        "build-features",
        "build-labels",
        "validate",
        "dataset",
        "train",
        "backtest",
    ]


def test_a_gate_refusal_skips_the_backtest_and_exits_four(tmp_path, recorder) -> None:
    """Replaying yesterday's model as if it were today's would be a lie."""
    calls, codes = recorder
    codes["train"] = pipeline.EXIT_GATE_REFUSED
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY, "--json"))

    assert exit_code == pipeline.EXIT_GATE_REFUSED
    assert "backtest" not in modes(calls)


def test_force_trains_and_the_backtest_runs(tmp_path, recorder) -> None:
    calls, codes = recorder
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY, "--force", "--horizon", "5"))

    assert exit_code == pipeline.EXIT_OK
    assert modes(calls)[-1] == "backtest"
    train = next(call for call in calls if "train" in call)
    assert train.count("--force") == 1


def test_an_empty_store_is_a_configuration_error_not_a_crash(tmp_path, recorder, capsys) -> None:
    """With no ticks anywhere the pipeline cannot even guess a day - and says how to fix it."""
    exit_code = pipeline.main(cli(tmp_path))

    assert exit_code == pipeline.EXIT_CONFIG_ERROR
    assert "the ticks table is empty" in capsys.readouterr().err


def test_a_day_without_ticks_is_refused_only_when_the_tick_steps_run(tmp_path, recorder, capsys) -> None:
    seeded_store(tmp_path)

    assert pipeline.main(cli(tmp_path, "--date", "2026-08-03")) == pipeline.EXIT_CONFIG_ERROR
    assert "no ticks are stored for 2026-08-03" in capsys.readouterr().err

    # validate reads features and the AI steps read all history: neither needs a capture.
    exit_code = pipeline.main(
        cli(tmp_path, "--date", "2026-08-03", "--skip", "candles,greeks,features,labels")
    )
    assert exit_code == pipeline.EXIT_OK



def test_the_preflight_warns_when_the_index_history_is_missing(tmp_path, recorder, capsys) -> None:
    """Without index history the IV columns stay NULL - the operator must hear it first."""
    seeded_store(tmp_path)

    pipeline.main(cli(tmp_path, "--date", TODAY, "--json"))

    report = json.loads(capsys.readouterr().out)
    warnings = report["preflight"]["warnings"]
    assert any("index_history holds no NIFTY rows" in warning for warning in warnings)
    assert report["preflight"]["ticks"] == 24
    assert report["preflight"]["tick_symbols"] == 2


def test_ticks_on_a_calendar_holiday_are_reported_as_a_contradiction(tmp_path, recorder, capsys) -> None:
    """Data on a day the calendar calls closed means one of the two is wrong."""
    holiday = date(2026, 10, 2)
    seeded_store(tmp_path, day=holiday)

    exit_code = pipeline.main(cli(tmp_path, "--date", holiday.isoformat(), "--json"))

    assert exit_code == pipeline.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["preflight"]["is_trading_day"] is False
    assert any("is the holiday" in warning for warning in report["preflight"]["warnings"])


def test_a_dry_run_writes_no_log_file_and_says_so(tmp_path, recorder, capsys) -> None:
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY, "--dry-run", "--json"))

    assert exit_code == pipeline.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["log_file"] is None
    assert report["verdict"] == "DRY-RUN - nothing was written"
    assert {row["status"] for row in report["results"]} == {"DRY-RUN"}
    assert not (tmp_path / "data" / "logs").exists()


def test_the_run_is_logged_to_a_dated_file(tmp_path, recorder, capsys) -> None:
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY))

    capsys.readouterr()
    log_path = tmp_path / "data" / "logs" / f"daily_pipeline_{TODAY}.log"
    assert exit_code == pipeline.EXIT_OK
    assert log_path.is_file()
    body = log_path.read_text(encoding="utf-8")
    assert body.startswith("ANJANEYA daily pipeline: day=2026-09-29 steps=8")
    assert "plan       : candles -> greeks" in body
    for position, step in enumerate(pipeline.STEP_KEYS, start=1):
        assert f"[{position}/8] {step}:" in body
    assert "verdict        : OK" in body


def test_a_failing_run_names_the_step_on_stderr(tmp_path, recorder, capsys) -> None:
    calls, codes = recorder
    codes["dataset"] = pipeline.EXIT_FAILED
    seeded_store(tmp_path)

    exit_code = pipeline.main(cli(tmp_path, "--date", TODAY))

    captured = capsys.readouterr()
    assert exit_code == pipeline.EXIT_FAILED
    assert "FAILED at dataset" in captured.err
    assert "first failure: dataset" in captured.err


def test_the_json_report_lists_the_planned_steps_and_the_day(tmp_path, recorder, capsys) -> None:
    seeded_store(tmp_path)

    pipeline.main(cli(tmp_path, "--date", TODAY, "--timeframe", "1m", "--horizon", "5", "--json"))

    report = json.loads(capsys.readouterr().out)
    assert report["day"] == TODAY
    assert report["timeframe"] == "1m"
    assert report["horizons"] == [5]
    assert report["planned_steps"] == list(pipeline.STEP_KEYS)
    assert report["exit_code"] == pipeline.EXIT_OK
    assert report["log_file"].endswith(f"daily_pipeline_{TODAY}.log")


# ---------------------------------------------------------------------------
# The verdict helpers
# ---------------------------------------------------------------------------
def result(key: str, status: str, code: int = pipeline.EXIT_OK) -> pipeline.StepResult:
    return pipeline.StepResult(key, status, code, 0.1)


def test_overall_exit_code_prefers_a_real_failure_over_the_gate() -> None:
    assert pipeline.overall_exit_code([]) == pipeline.EXIT_OK
    assert pipeline.overall_exit_code([result("candles", pipeline.STATUS_OK)]) == pipeline.EXIT_OK
    assert (
        pipeline.overall_exit_code([result("train", pipeline.STATUS_GATE, 4)]) == pipeline.EXIT_GATE_REFUSED
    )
    assert (
        pipeline.overall_exit_code(
            [result("train", pipeline.STATUS_GATE, 4), result("dataset", pipeline.STATUS_FAILED, 1)]
        )
        == pipeline.EXIT_FAILED
    )


def test_verdict_text_names_the_failing_step() -> None:
    assert pipeline.verdict_text([], pipeline.EXIT_OK) == "OK"
    assert "FAILED at dataset" in pipeline.verdict_text(
        [result("dataset", pipeline.STATUS_FAILED, 1)], pipeline.EXIT_FAILED
    )
    assert "BELOW_GATE" in pipeline.verdict_text(
        [result("train", pipeline.STATUS_GATE, 4)], pipeline.EXIT_GATE_REFUSED
    )
    assert "CONFIG_ERROR" in pipeline.verdict_text([], pipeline.EXIT_CONFIG_ERROR)


def test_verdict_text_reports_a_dry_run_as_nothing_written() -> None:
    verdict = pipeline.verdict_text(
        [result("candles", pipeline.STATUS_DRY_RUN), result("train", pipeline.STATUS_DRY_RUN)],
        pipeline.EXIT_OK,
    )
    assert verdict == "DRY-RUN - nothing was written"


def test_verdict_text_mentions_skipped_steps() -> None:
    verdict = pipeline.verdict_text(
        [result("candles", pipeline.STATUS_OK), result("backtest", pipeline.STATUS_SKIPPED)],
        pipeline.EXIT_OK,
    )
    assert verdict == "OK (with skipped steps)"


def test_the_rendered_report_has_a_row_per_step() -> None:
    report = pipeline.build_report(
        options=pipeline.RunOptions(day=DAY, db_path="db.duckdb"),
        inspection=pipeline.DayInspection(day=DAY, ticks=12, tick_symbols=2),
        steps=pipeline.STEPS,
        results=[result(step.key, pipeline.STATUS_OK) for step in pipeline.STEPS],
        log_path=None,
        exit_code=pipeline.EXIT_OK,
    )

    rendered = pipeline.render_report(report)
    assert "ANJANEYA daily pipeline - 2026-09-29" in rendered
    for step in pipeline.STEP_KEYS:
        assert f"  {step}" in rendered
    assert "verdict        : OK" in rendered

