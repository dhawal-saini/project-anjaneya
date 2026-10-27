"""Tests for ``scripts/anj_ui.py`` - the operator console's contract with the CLI.

The console owns no behaviour: it translates a switch into a ``--skip`` value, a
button into a command, and a status file into a card. That translation is what
is pinned here, because a checkbox that silently changes what runs is the one
bug a UI can introduce that a CLI cannot.

Nothing below starts Streamlit, touches the network, opens DuckDB or spawns a
process: the child runner, the process spawner and ``subprocess.run`` are all
replaced by recorders.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import anj_ui


class Spawner:
    """Stands in for :class:`subprocess.Popen`: records the call, starts nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> object:
        self.calls.append(([str(item) for item in argv], kwargs))
        # ``subprocess.Popen`` returns an object with a ``pid``; the console records it
        # to tell a live job from a status file whose process is gone.
        return SimpleNamespace(pid=os.getpid())

    @property
    def only(self) -> list[str]:
        """The single recorded command line."""
        assert len(self.calls) == 1, f"expected one spawn, got {len(self.calls)}"
        return self.calls[0][0]


@pytest.fixture
def spawner() -> Spawner:
    return Spawner()


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A throwaway data root, so no test can write into the real ``data/``."""
    root = tmp_path / "data"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# Switch -> --skip. The mapping, the default, and the switches that must not
# exist because the project cannot honour them.
# ---------------------------------------------------------------------------
def test_default_switches_run_the_whole_chain() -> None:
    """No switches given means everything runs - the safe default."""
    assert anj_ui.steps_for({}) == list(anj_ui.PIPELINE_ORDER)
    assert anj_ui.skip_argument({}) == ""


def test_missing_switch_counts_as_on() -> None:
    """A switch absent from the payload must not silently shrink a run."""
    assert anj_ui.skip_argument({"backtest": True}) == ""
    assert anj_ui.skip_argument({"ml": True, "backtest": False}) == "backtest"


def test_each_switch_controls_exactly_its_steps() -> None:
    assert anj_ui.skip_argument({"ml": False}) == "dataset,train"
    assert anj_ui.skip_argument({"backtest": False}) == "backtest"
    assert anj_ui.skip_argument({"research": False}) == "report,explain,importance,charts,accumulation"


def test_everything_off_skips_everything_optional() -> None:
    off = {name: False for name, _ in anj_ui.SWITCHED_STEPS}
    assert anj_ui.steps_for(off) == list(anj_ui.ALWAYS_ON)
    assert anj_ui.skip_argument(off) == "dataset,train,backtest,report,explain,importance,charts,accumulation"


def test_switches_never_skip_the_dataset_steps() -> None:
    """candles..validate are the dataset; no switch may reach them."""
    for name, _ in anj_ui.SWITCHED_STEPS:
        skipped = set(anj_ui.skip_argument({name: False}).split(","))
        assert not skipped & set(anj_ui.ALWAYS_ON), f"{name} reached a data step"


def test_unswitchable_work_is_documented_not_offered() -> None:
    """indicator / price_action / trading are explained, never mapped to a step."""
    assert set(anj_ui.UNSWITCHABLE) == {"indicator", "price_action", "trading"}
    mapped = {step for _, steps in anj_ui.SWITCHED_STEPS for step in steps}
    assert mapped <= set(anj_ui.PIPELINE_ORDER)
    assert set(anj_ui.ALWAYS_ON) | mapped == set(anj_ui.PIPELINE_ORDER)


# ---------------------------------------------------------------------------
# Command builders: the button must produce the command the operator would type.
# ---------------------------------------------------------------------------
def test_learn_with_all_switches_on_is_the_facade_verb() -> None:
    """Nothing to skip means no reason to bypass ``anj.py``."""
    assert anj_ui.learn_argv({}) == [anj_ui.FACADE, "learn"]


def test_learn_with_a_switch_off_uses_the_pipeline_owner() -> None:
    """``anj.py learn`` has no --skip, so the console calls the script it delegates to."""
    assert anj_ui.learn_argv({"backtest": False}) == [
        anj_ui.PIPELINE,
        "--weekly",
        "--skip",
        "backtest",
    ]


def test_learn_force_is_appended_and_never_dropped() -> None:
    assert anj_ui.learn_argv({}, force=True) == [anj_ui.FACADE, "learn", "--force"]
    assert anj_ui.learn_argv({"ml": False}, force=True)[-1] == "--force"


def test_common_flags_land_last() -> None:
    """argparse wants the shared options after the verb's own -- that order is pinned."""
    assert anj_ui.learn_argv({}, common=["--config-dir", "scratch"])[-2:] == ["--config-dir", "scratch"]
    assert anj_ui.day_argv(common=["--config-dir", "scratch"])[-2:] == ["--config-dir", "scratch"]
    assert anj_ui.common_flags() == []
    assert anj_ui.common_flags(config_dir="cfg") == ["--config-dir", "cfg"]


def test_day_never_carries_the_switches() -> None:
    """The pre-close policy belongs to the facade; the console must not reshape it."""
    assert anj_ui.day_argv() == [anj_ui.FACADE, "day"]
    assert anj_ui.day_argv(no_capture=True) == [anj_ui.FACADE, "day", "--no-capture"]
    assert "--skip" not in anj_ui.day_argv()


def test_history_numbers_are_rendered_the_way_argparse_expects() -> None:
    assert anj_ui.history_argv() == [
        anj_ui.FACADE,
        "history",
        "--years",
        "2",
        "--option-years",
        "0.25",
    ]
    assert anj_ui.history_argv(years=1.5, option_years=0.0)[3] == "1.5"


def test_history_flags_are_additive() -> None:
    plan = anj_ui.history_argv(skip_instruments=True, skip_options=True, dry_run=True)
    assert plan[-3:] == ["--skip-instruments", "--skip-options", "--dry-run"]
    assert "--dry-run" not in anj_ui.history_argv()


def test_login_argv_never_carries_a_secret() -> None:
    """The TOTP path reads .env; nothing secret may be typed into the command."""
    assert anj_ui.login_argv() == [anj_ui.LOGIN, "--json"]
    assert anj_ui.login_argv(force=True) == [anj_ui.LOGIN, "--json", "--force"]
    assert "--request-token" not in anj_ui.login_argv()


def test_morning_is_the_browser_login_owner_and_carries_no_secret() -> None:
    """One click at 09:00: `login_and_start.py` logs in, then captures.

    The console never handles the password or the request_token - the child opens
    the browser and catches the redirect itself.
    """
    argv = anj_ui.morning_argv()
    assert argv[:3] == [anj_ui.LOGIN_AND_START, "--timeout", "240"]
    assert "--request-token" not in argv
    assert anj_ui.morning_argv(common=["--config-dir", "cfg"])[-2:] == ["--config-dir", "cfg"]


def test_the_post_close_topup_is_the_index_mode_not_the_overnight_history() -> None:
    """The top-up must stay cheap: index history only, no option contracts."""
    argv = anj_ui.index_topup_argv()
    assert argv[:4] == ["-m", anj_ui.CAPTURE_APP, "--mode", "backfill"]
    assert "history" not in argv
    assert "--option-years" not in argv
    assert anj_ui.index_topup_argv(common=["--config-dir", "cfg"])[-2:] == ["--config-dir", "cfg"]


def test_the_playbook_exists_where_the_console_reads_it() -> None:
    """The Playbook tab renders this file; a rename must break a test, not the page."""
    assert anj_ui.PLAYBOOK.name == "daily-playbook.md"
    assert anj_ui.PLAYBOOK.exists(), f"missing {anj_ui.PLAYBOOK}"
    text = anj_ui.PLAYBOOK.read_text(encoding="utf-8")
    assert "## 1. Morning" in text
    assert "## 3. After the close" in text


def test_mask_hides_all_but_the_last_four() -> None:
    assert anj_ui.mask("abcdefghijkl") == "********ijkl"
    assert anj_ui.mask("abc") == "abc"
    assert anj_ui.mask(None) == "-"
    assert anj_ui.mask("") == "-"


# ---------------------------------------------------------------------------
# Preview SQL: whitelisted tables only, so no widget value can reach the query.
# ---------------------------------------------------------------------------
def test_preview_query_reads_the_first_or_last_rows() -> None:
    assert anj_ui.preview_query("candles") == "SELECT * FROM candles ORDER BY timestamp ASC LIMIT 10"
    assert anj_ui.preview_query("candles", last=True) == ("SELECT * FROM candles ORDER BY timestamp DESC LIMIT 10")
    assert anj_ui.preview_query("trades") == ("SELECT * FROM trades ORDER BY entry_timestamp ASC LIMIT 10")
    assert anj_ui.preview_query("ticks", limit=3).endswith("LIMIT 3")


@pytest.mark.parametrize("table", ["users", "candles; DROP TABLE candles", "", "ticks--"])
def test_preview_query_refuses_anything_not_whitelisted(table: str) -> None:
    with pytest.raises(ValueError, match="not a previewable table"):
        anj_ui.preview_query(table)


def test_table_helpers_agree_with_the_whitelist() -> None:
    assert anj_ui.table_names() == [name for name, _, _ in anj_ui.PREVIEW_TABLES]
    assert anj_ui.table_label("candles") == "Candles (after greeks)"
    assert anj_ui.table_label("nope") == "nope"


# ---------------------------------------------------------------------------
# The job registry: one writer, a status file, and a child that always reports.
# ---------------------------------------------------------------------------
def test_status_round_trips_through_disk(data_root: Path) -> None:
    path = anj_ui.status_file(data_root)
    assert anj_ui.read_status(path) is None
    status = anj_ui.JobStatus(name="history", state="ok", argv=("a", "b"), exit_code=0)
    anj_ui.write_status(path, status)
    assert anj_ui.read_status(path) == status
    assert path.parent == anj_ui.ui_dir(data_root)


def test_a_corrupt_status_file_reads_as_no_job(data_root: Path) -> None:
    path = anj_ui.status_file(data_root)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert anj_ui.read_status(path) is None


def test_start_job_spawns_a_detached_child_and_seeds_the_log(data_root: Path, spawner: Spawner) -> None:
    status = anj_ui.start_job("history", [anj_ui.FACADE, "history"], data_root=data_root, spawn=spawner)
    assert status.state == "starting"
    argv = spawner.only
    assert argv[1].endswith("anj_ui.py")
    assert argv[2:4] == ["--run-job", str(anj_ui.status_file(data_root))]
    assert argv[4:] == ["--", anj_ui.FACADE, "history"]
    assert spawner.calls[0][1]["stdout"] is subprocess.DEVNULL
    assert anj_ui.read_status(anj_ui.status_file(data_root)) == status
    assert Path(status.log_path or "").read_text(encoding="utf-8").startswith("=== history")


def test_start_job_refuses_a_second_writer(data_root: Path, spawner: Spawner) -> None:
    anj_ui.start_job("first", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner)
    with pytest.raises(anj_ui.JobBusy, match="still starting"):
        anj_ui.start_job("second", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner)
    assert len(spawner.calls) == 1


def test_a_status_file_whose_process_is_gone_does_not_block(
    data_root: Path, spawner: Spawner
) -> None:
    """2026-10-27: an orphaned 'token-force' card refused every button all morning.

    The console restarted while a job was live, the child died without writing a
    terminal state, and the file said 'running' forever. A pid that no longer exists
    must mean "not active".
    """
    path = anj_ui.status_file(data_root)
    anj_ui.write_status(
        path, anj_ui.JobStatus(name="token-force", state="running", pid=999_999_999)
    )
    assert anj_ui.read_status(path).active is False

    anj_ui.start_job("day", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner)
    assert len(spawner.calls) == 1


def test_liveness_decides_whether_a_status_is_active() -> None:
    assert anj_ui.pid_alive(os.getpid()) is True
    assert anj_ui.pid_alive(999_999_999) is False
    assert anj_ui.pid_alive(None) is False
    assert anj_ui.JobStatus(name="x", state="running", pid=os.getpid()).active is True
    assert anj_ui.JobStatus(name="x", state="running", pid=None).active is False
    assert anj_ui.JobStatus(name="x", state="ok", pid=os.getpid()).active is False


def test_start_job_proceeds_once_the_previous_job_finished(data_root: Path, spawner: Spawner) -> None:
    first = anj_ui.start_job("first", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner)
    anj_ui.write_status(anj_ui.status_file(data_root), replace(first, state="ok", exit_code=0))
    assert anj_ui.start_job("second", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner).name == "second"
    assert len(spawner.calls) == 2


def test_forgetting_a_stale_job_unblocks_the_console(data_root: Path, spawner: Spawner) -> None:
    """A killed child leaves 'running' behind; Forget rewrites it as abandoned."""
    stale = anj_ui.start_job("first", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner)
    path = anj_ui.status_file(data_root)
    anj_ui.write_status(path, replace(stale, state="abandoned", finished_at=anj_ui.now()))
    assert anj_ui.read_status(path).active is False
    assert anj_ui.start_job("again", [anj_ui.FACADE, "day"], data_root=data_root, spawn=spawner).name == "again"


@pytest.mark.parametrize("code,state", [(0, "ok"), (3, "failed")])
def test_run_job_always_records_a_terminal_state(
    monkeypatch: pytest.MonkeyPatch, data_root: Path, code: int, state: str
) -> None:
    status = anj_ui.JobStatus(
        name="demo",
        started_at="t0",
        log_path=str(anj_ui.ui_dir(data_root) / "demo.log"),
    )
    path = anj_ui.status_file(data_root)
    anj_ui.write_status(path, status)
    launched: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        launched.append(list(argv))
        return subprocess.CompletedProcess(argv, code)

    monkeypatch.setattr(anj_ui.subprocess, "run", fake_run)
    assert anj_ui.run_job(path, [anj_ui.FACADE, "learn"]) == code
    final = anj_ui.read_status(path)
    assert final is not None
    assert (final.state, final.exit_code) == (state, code)
    assert final.finished_at and final.started_at == "t0"
    assert launched[0][1:] == [anj_ui.FACADE, "learn"]
    log = Path(status.log_path or "").read_text(encoding="utf-8")
    assert f"exit {code}" in log


def test_run_job_without_a_status_file_fails_loudly(data_root: Path) -> None:
    assert anj_ui.run_job(anj_ui.status_file(data_root), [anj_ui.FACADE, "day"]) == 2


def test_run_job_cli_splits_the_status_path_from_the_command(monkeypatch: pytest.MonkeyPatch, data_root: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run(path: Path, argv: list[str]) -> int:
        seen.update(path=path, argv=list(argv))
        return 0

    monkeypatch.setattr(anj_ui, "run_job", fake_run)
    path = data_root / "job.json"
    assert anj_ui.run_job_cli(["--run-job", str(path), "--", anj_ui.FACADE, "day"]) == 0
    assert seen == {"path": path, "argv": [anj_ui.FACADE, "day"]}
    assert anj_ui.run_job_cli(["--run-job", str(path)]) == 2
    assert anj_ui.run_job_cli([]) == 2


def test_read_log_is_bounded_and_tolerates_missing_files(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    log.write_text("x" * 5000 + "TAIL", encoding="utf-8")
    tail = anj_ui.read_log(log, max_bytes=100)
    assert tail.endswith("TAIL") and len(tail) == 100
    assert anj_ui.read_log(tmp_path / "missing.log") == ""
    assert anj_ui.read_log(None) == ""


# ---------------------------------------------------------------------------
# Read-only views: reports, the token cache, and the one secret-bearing call.
# ---------------------------------------------------------------------------
def test_artefacts_are_newest_first_and_limited(tmp_path: Path) -> None:
    charts = tmp_path / "reports" / "charts"
    charts.mkdir(parents=True)
    old, new = charts / "old.md", charts / "new.md"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    (charts / "chart.png").write_text("png", encoding="utf-8")
    reports, images = anj_ui.artefacts(tmp_path)
    assert [path.name for path in reports] == ["new.md", "old.md"]
    assert [path.name for path in images] == ["chart.png"]
    assert len(anj_ui.artefacts(tmp_path, limit=1)[0]) == 1
    assert anj_ui.artefacts(tmp_path / "nothing-here") == ([], [])


def _config(tmp_path: Path, *, store: str = "./data/raw/.tokens/zerodha_token.json") -> SimpleNamespace:
    return SimpleNamespace(
        base_dir=str(tmp_path),
        settings=SimpleNamespace(data=SimpleNamespace(root="./data")),
        data_sources=SimpleNamespace(zerodha=SimpleNamespace(token_store=SimpleNamespace(path=store))),
    )


def test_config_paths_resolve_against_the_base_dir(tmp_path: Path) -> None:
    assert anj_ui.data_root_of(_config(tmp_path)) == tmp_path / "data"
    assert (
        anj_ui.token_store_path(_config(tmp_path)) == Path(tmp_path) / "data" / "raw" / ".tokens" / "zerodha_token.json"
    )
    absolute = str(tmp_path / "abs" / "token.json")
    assert anj_ui.token_store_path(_config(tmp_path, store=absolute)) == Path(absolute)


def test_token_summary_without_a_cached_token_says_so(tmp_path: Path) -> None:
    """No file is not an error; it is the first-run state the card explains."""
    assert anj_ui.token_summary(_config(tmp_path)) == {"present": False}


def test_exchange_returns_only_what_the_login_script_already_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "status": "ok",
        "api_key": "****abcd",
        "access_token": "****wxyz",
        "expires_at": "2026-10-24T18:00:00+05:30",
        "remaining_minutes": 700.0,
        "obtained_via": "request_token",
    }
    captured: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(anj_ui.subprocess, "run", fake_run)
    ok, detail = anj_ui.exchange_request_token(_config(tmp_path), "SECRET", common=["--config-dir", "x"])
    assert ok is True
    assert detail["access_token"] == "****wxyz"
    assert "SECRET" not in json.dumps(detail)
    assert captured[0][1:] == [anj_ui.LOGIN, "--json", "--request-token", "SECRET", "--config-dir", "x"]


def test_exchange_reports_a_failure_without_echoing_the_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="invalid request_token")

    monkeypatch.setattr(anj_ui.subprocess, "run", fake_run)
    ok, detail = anj_ui.exchange_request_token(_config(tmp_path), "SECRET")
    assert ok is False
    assert detail["exit_code"] == 1
    assert "invalid request_token" in str(detail["detail"])
    assert "SECRET" not in json.dumps(detail)
