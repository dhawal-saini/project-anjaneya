"""``--mode index-study`` end to end: seeded ``index_history`` -> artefacts on disk.

The study is the project's cheapest question (does the NIFTY signal separate forward
returns?), so what the test proves is that the answer is *reproducible*: a seeded
index series, the real ``main()``, and a JSON + Markdown pair that carries the caveats.
"""

from __future__ import annotations

import json
from datetime import timedelta

from apps.anjaneeya_engine.run import main as engine_main
from core.app_bootstrap import EXIT_OK
from core.storage import connect
from tests.support.engine import (
    SESSION_START,
    cli_args,
    make_index_row,
    seed_index_history,
)

#: Daily bars to seed (one per session).
DAILY_SESSIONS = 20
#: 1-minute bars: six full sessions, enough for the configured SMA to define itself.
INTRADAY_SESSIONS = 6
INTRADAY_BARS = 375


def db_file(tmp_path):
    return tmp_path / "db" / "anjaneeya.duckdb"


def reports_dir(tmp_path):
    return tmp_path / "data" / "reports" / "index_study"


def index_rows(*, timeframe: str, sessions: int, bars_per_session: int):
    """A slightly rising NIFTY series that still moves both ways bar to bar.

    The zig-zag matters: a monotonic seed makes every forward return positive, and a
    sample with no losses cannot be tested (``NOT_TESTABLE``), which would test the
    degenerate path instead of the one the study exists for.
    """
    rows = []
    for session in range(sessions):
        for bar in range(bars_per_session):
            index = session * bars_per_session + bar
            moment = SESSION_START + timedelta(days=session, minutes=bar)
            price = 25000.0 + index * 0.1 + (0.6 if index % 2 else -0.6)
            rows.append(
                make_index_row(
                    timestamp=moment,
                    timeframe=timeframe,
                    open=price - 0.5,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price,
                )
            )
    return rows


def seed(tmp_path, *, daily: int = DAILY_SESSIONS, intraday: int = INTRADAY_SESSIONS):
    seed_index_history(db_file(tmp_path), [])
    seed_index_history(
        db_file(tmp_path),
        index_rows(timeframe="1d", sessions=daily, bars_per_session=1)
        + (
            index_rows(timeframe="1m", sessions=intraday, bars_per_session=INTRADAY_BARS)
            if intraday
            else []
        ),
    )
    return tmp_path


def query(tmp_path, sql: str):
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def test_the_engine_writes_the_study_with_both_sections(tmp_path) -> None:
    seed(tmp_path)

    assert engine_main(cli_args(tmp_path, mode="index-study")) == EXIT_OK

    written = sorted(reports_dir(tmp_path).glob("index_study_*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["spec"] == "P1 - NIFTY index forward-return study"
    assert payload["symbol"] == "NIFTY 50"
    assert payload["daily"]["available"] is True
    assert payload["daily"]["bars"] == DAILY_SESSIONS
    assert payload["intraday"]["available"] is True
    assert payload["intraday"]["bars"] == INTRADAY_SESSIONS * INTRADAY_BARS
    assert payload["intraday"]["sessions"] == INTRADAY_SESSIONS
    # Every axis of the intraday section is present, in the order the report reads.
    assert [axis["axis"] for axis in payload["intraday"]["axes"][:4]] == [
        "regime",
        "score_band",
        "time_of_day",
        "level_proximity",
    ]
    assert payload["policy"]["daily_horizons"] == [1, 2, 5, 10]
    assert payload["caveats"]


def test_the_intraday_axes_actually_hold_rows(tmp_path) -> None:
    """A defined composite must produce buckets - an empty table would hide a bug."""
    seed(tmp_path)

    engine_main(cli_args(tmp_path, mode="index-study"))

    payload = json.loads(
        next(reports_dir(tmp_path).glob("index_study_*.json")).read_text(encoding="utf-8")
    )
    regime = payload["intraday"]["axes"][0]
    assert regime["baseline"]["sample"] > 0
    assert regime["rows"]
    assert all(row["sample"] > 0 for row in regime["rows"])
    # Thin buckets are refused rather than tested.
    assert all(
        (row["p_hit"] is not None) == (row["sample"] >= payload["policy"]["min_samples"])
        for row in regime["rows"]
    )


def test_the_markdown_is_written_beside_the_json(tmp_path) -> None:
    seed(tmp_path)

    engine_main(cli_args(tmp_path, mode="index-study"))

    markdown = next(reports_dir(tmp_path).glob("index_study_*.md")).read_text(encoding="utf-8")
    assert "# P1 - NIFTY index forward-return study" in markdown
    assert "## Caveats" in markdown
    assert "## Intraday" in markdown


def test_a_dry_run_reports_the_window_and_writes_nothing(tmp_path, capsys) -> None:
    seed(tmp_path)

    exit_code = engine_main(cli_args(tmp_path, mode="index-study", extra=["--dry-run"]))

    assert exit_code == EXIT_OK
    output = capsys.readouterr().out
    assert "index-study -> NIFTY 50" in output
    assert "dry-run reads only" in output
    assert not reports_dir(tmp_path).exists()


def test_an_empty_index_history_is_reported_rather_than_crashing(tmp_path, capsys) -> None:
    seed_index_history(db_file(tmp_path), [])

    exit_code = engine_main(cli_args(tmp_path, mode="index-study"))

    assert exit_code == EXIT_OK
    assert "no bars to study" in capsys.readouterr().err


def test_the_json_flag_prints_the_same_payload(tmp_path, capsys) -> None:
    seed(tmp_path, daily=3, intraday=0)

    engine_main(cli_args(tmp_path, mode="index-study", extra=["--json"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["daily"]["bars"] == 3
    assert payload["intraday"]["available"] is False


def test_the_run_manifest_records_the_study(tmp_path) -> None:
    seed(tmp_path)

    assert (
        engine_main(cli_args(tmp_path, mode="index-study", run_artifacts=True)) == EXIT_OK
    )

    assert (
        "anjaneeya_engine",
        "index-study",
        "SUCCESS",
    ) in query(tmp_path, "SELECT app_name, mode, status FROM run_manifest")
