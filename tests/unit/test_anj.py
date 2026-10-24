"""Tests for ``scripts/anj.py`` - the three-verb operator facade.

The facade's only job is to *delegate*, so that is what is pinned here: each verb
calls the scripts and apps that already own the behaviour, in the right order,
with the right arguments, and stops the moment a step fails. Every child is
replaced by a recorder, so nothing here touches the network, a broker or DuckDB.
"""

from __future__ import annotations

import pytest

from scripts import anj


class Recorder:
    """Stands in for :func:`scripts.anj._child`: records argv, scripts exit codes."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.codes: dict[str, int] = {}

    def __call__(self, argv: object) -> int:
        call = [str(item) for item in argv]  # type: ignore[union-attr]
        self.calls.append(call)
        joined = " ".join(call)
        for marker, code in self.codes.items():
            if marker in joined:
                return code
        return 0

    def step(self, index: int) -> str:
        """The recorded command at ``index``, as the facade would print it."""
        return " ".join(self.calls[index])


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replace the child runner and the ``.env`` loader; touch nothing real."""
    double = Recorder()
    monkeypatch.setattr(anj, "_child", double)
    monkeypatch.setattr(anj, "load_env_file", lambda **_kwargs: None)
    return double


def test_number_renders_years_compactly() -> None:
    assert anj._number(2.0) == "2"
    assert anj._number(0.25) == "0.25"


def test_history_runs_instruments_then_index_then_options(recorder: Recorder) -> None:
    assert anj.main(["history"]) == 0
    assert len(recorder.calls) == 3
    assert recorder.step(0) == "-m apps.anjaneeya_capture.run --mode instruments --force-refresh"
    assert recorder.step(1) == "-m apps.anjaneeya_capture.run --mode backfill --years 2"
    assert recorder.step(2) == "-m apps.anjaneeya_capture.run --mode backfill-options --years 0.25"


def test_history_renders_whole_years_without_a_decimal(recorder: Recorder) -> None:
    anj.main(["history", "--years", "3", "--option-years", "0.5"])
    assert recorder.step(1).endswith("--years 3")
    assert recorder.step(2).endswith("--years 0.5")


def test_history_skip_flags_leave_out_those_steps(recorder: Recorder) -> None:
    assert anj.main(["history", "--skip-instruments", "--skip-options"]) == 0
    assert len(recorder.calls) == 1
    assert recorder.step(0) == "-m apps.anjaneeya_capture.run --mode backfill --years 2"


def test_history_stops_at_the_first_failure(recorder: Recorder) -> None:
    recorder.codes["--mode instruments"] = 2
    assert anj.main(["history"]) == 2
    assert len(recorder.calls) == 1


def test_dry_run_is_forwarded_to_every_child(recorder: Recorder) -> None:
    assert anj.main(["history", "--dry-run"]) == 0
    assert all(call[-1] == "--dry-run" for call in recorder.calls)


def test_shared_overrides_are_forwarded(recorder: Recorder) -> None:
    anj.main(["learn", "--db-path", "db/x.duckdb"])
    assert recorder.step(0).endswith("--db-path db/x.duckdb")


def test_day_runs_the_session_then_the_postclose_chain(recorder: Recorder) -> None:
    assert anj.main(["day"]) == 0
    assert recorder.step(0) == "scripts/run_morning.py"
    assert recorder.step(1) == "scripts/run_daily_pipeline.py --skip train,backtest"
    assert recorder.step(2) == "-m apps.anjaneeya_ai.run --mode decide"


def test_day_never_trains(recorder: Recorder) -> None:
    """Training belongs to ``learn``; ``day`` must leave train and backtest out."""
    anj.main(["day"])
    chain = recorder.step(1)
    assert "--skip train,backtest" in chain
    assert "--force" not in chain


def test_day_without_capture_skips_the_session(recorder: Recorder) -> None:
    assert anj.main(["day", "--no-capture"]) == 0
    assert all("run_morning" not in recorder.step(i) for i in range(len(recorder.calls)))
    assert recorder.step(0) == "scripts/run_daily_pipeline.py --skip train,backtest"


def test_day_forwards_smoke_and_until_after_the_separator(recorder: Recorder) -> None:
    anj.main(["day", "--smoke", "--until", "15:30"])
    assert recorder.step(0) == "scripts/run_morning.py -- --smoke --until 15:30"


def test_day_skip_postclose_stops_after_the_session(recorder: Recorder) -> None:
    assert anj.main(["day", "--skip-postclose"]) == 0
    assert len(recorder.calls) == 1
    assert "run_morning" in recorder.step(0)


def test_day_forwards_the_date_to_the_chain(recorder: Recorder) -> None:
    anj.main(["day", "--no-capture", "--date", "2026-10-05"])
    assert recorder.step(0) == "scripts/run_daily_pipeline.py --skip train,backtest --date 2026-10-05"


def test_day_morning_failure_is_fatal(recorder: Recorder) -> None:
    recorder.codes["run_morning"] = 4
    assert anj.main(["day"]) == 4
    assert len(recorder.calls) == 1


def test_day_postclose_failure_is_fatal(recorder: Recorder) -> None:
    recorder.codes["run_daily_pipeline"] = 1
    assert anj.main(["day", "--no-capture"]) == 1


def test_day_missing_suggestion_is_not_fatal(recorder: Recorder, capsys: pytest.CaptureFixture[str]) -> None:
    """No model yet means no suggestion - that must not fail the day's math."""
    recorder.codes["--mode decide"] = 1
    assert anj.main(["day", "--no-capture"]) == 0
    assert len(recorder.calls) == 2
    assert "the math above is unaffected" in capsys.readouterr().err


def test_learn_trains_with_the_weekly_cadence(recorder: Recorder) -> None:
    assert anj.main(["learn", "--force"]) == 0
    assert len(recorder.calls) == 1
    assert recorder.step(0) == "scripts/run_daily_pipeline.py --weekly --force"


def test_learn_without_force_does_not_force_the_gate(recorder: Recorder) -> None:
    anj.main(["learn"])
    assert recorder.step(0) == "scripts/run_daily_pipeline.py --weekly"


def test_learn_explains_the_gate_instead_of_looking_like_a_crash(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder.codes["run_daily_pipeline"] = 4
    assert anj.main(["learn"]) == 4
    assert "accumulation gate" in capsys.readouterr().err


def test_learn_failure_propagates(recorder: Recorder) -> None:
    recorder.codes["run_daily_pipeline"] = 1
    assert anj.main(["learn"]) == 1


def test_a_verb_is_required() -> None:
    with pytest.raises(SystemExit):
        anj.parse_args([])
