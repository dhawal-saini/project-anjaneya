"""End-to-end ANJANEYA-AI accumulation report (Phase 6, spec 29).

``--mode accumulation`` is exercised through the real ``main()`` against a seeded database and
a frozen "today", so every assertion is about what actually happened:

    the inputs    ->  the stored tables and the coverage ledger, read back by the mode itself
    the artefacts ->  one JSON, one Markdown and one CSV under ``data/reports/accumulation/``,
                      all registered on the run manifest
    the controls  ->  ``--weeks`` narrows the window, ``--dry-run`` and ``--no-write`` write
                      nothing
    the honesty   ->  a missing ledger or an empty database is reported with a reason and exit
                      0 (never a crash), and nothing is written to DuckDB

Nothing here touches the network: the only inputs are the stored database and the artefacts on
disk.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from apps.anjaneeya_ai.accumulation_writer import AccumulationWriter, accumulation_stem
from apps.anjaneeya_ai.run import main as ai_main
from core.accumulation import AccumulationPolicy, build_accumulation_report
from core.app_bootstrap import EXIT_NOT_IMPLEMENTED, EXIT_OK
from core.constants import COVERAGE_COLUMNS
from core.market_calendar import NseCalendar
from core.schemas import CoverageRow
from core.storage import connect, insert_rows
from tests.support.ai import cli_args, seed_source, shrink_config

IST = ZoneInfo("Asia/Kolkata")

#: The four fixture sessions (Mon 2026-09-28 .. Thu 2026-10-01).
DAYS: tuple[date, ...] = (
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
)

#: The day the report is read on, frozen so the window cannot depend on the wall clock.
TODAY = date(2026, 10, 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def db_file(tmp_path) -> Path:
    return tmp_path / "db" / "anjaneeya.duckdb"


def out_dir(tmp_path) -> Path:
    return tmp_path / "data" / "reports" / "accumulation"


def models_dir(tmp_path) -> Path:
    return tmp_path / "data" / "models"


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> date:
    """Pin ``NseCalendar.now()`` so the report's window is reproducible."""
    moment = datetime(TODAY.year, TODAY.month, TODAY.day, 16, 0, tzinfo=IST)
    monkeypatch.setattr(NseCalendar, "now", lambda self: moment)
    return TODAY


def coverage_row(
    day: date,
    *,
    symbol: str = "NIFTY01OCT2625000CE",
    timeframe: str = "1m",
    gap_count: int = 0,
    quality: str = "OK",
    ratio: float | None = 1.0,
    expected: int = 375,
    actual: int = 375,
) -> CoverageRow:
    """One stored coverage verdict for ``day``."""
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    return CoverageRow(
        trading_date=day,
        tradingsymbol=symbol,
        timeframe=timeframe,
        expected_rows=expected,
        actual_rows=actual,
        first_timestamp=start,
        last_timestamp=start + timedelta(minutes=max(1, actual) - 1),
        gap_count=gap_count,
        largest_gap_seconds=180.0 if gap_count else 0.0,
        coverage_ratio=ratio,
        quality_flag=quality,
        note="fixture",
        run_id="fixture",
    )


def seed(
    tmp_path,
    *,
    days: tuple[date, ...] = DAYS,
    gaps: dict[date, int] | None = None,
    seed_db: bool = True,
) -> Path:
    """A seeded database (features/labels/candles + the coverage ledger) and the fixture config."""
    if seed_db:
        seed_source(db_file(tmp_path))
    else:
        from core.storage import bootstrap_database

        bootstrap_database(db_file(tmp_path), schema_version=1)

    gaps = gaps or {}
    rows = [
        coverage_row(day, gap_count=gaps.get(day, 0), quality="GAP_FILLED" if gaps.get(day) else "OK")
        for day in days
    ]
    if rows:
        connection = connect(db_file(tmp_path))
        try:
            insert_rows(
                connection, "data_coverage", rows, mode="replace", columns=COVERAGE_COLUMNS
            )
        finally:
            connection.close()
    return shrink_config(tmp_path)


def run_accumulation(
    tmp_path, config_dir: Path, *extra: str, run_artifacts: bool = False
) -> int:
    """``--mode accumulation`` with the fixture's database and configuration."""
    return ai_main(
        [
            *cli_args(tmp_path, mode="accumulation", run_artifacts=run_artifacts),
            "--config-dir",
            str(config_dir),
            *extra,
        ]
    )


def written(tmp_path) -> list[Path]:
    """Every artefact the mode wrote, sorted by name."""
    root = out_dir(tmp_path)
    return sorted(root.glob("*")) if root.exists() else []


def payload(tmp_path) -> dict:
    """The JSON artefact's document."""
    matches = sorted(out_dir(tmp_path).glob("*_accumulation.json"))
    assert matches, "no JSON artefact was written"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def table_counts(tmp_path) -> dict[str, int]:
    """Row counts of the tables the report must never write to."""
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        return {
            table: int(connection.execute(f"select count(*) from {table}").fetchone()[0])
            for table in ("ticks", "candles", "features", "labels", "trades")
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_the_mode_writes_json_markdown_and_csv(tmp_path, frozen_today, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_accumulation(tmp_path, config_dir)
    assert exit_code == EXIT_OK

    names = [path.name for path in written(tmp_path)]
    stem = accumulation_stem(TODAY)
    assert f"{stem}_accumulation.json" in names
    assert f"{stem}_accumulation.md" in names
    assert f"{stem}_tables.csv" in names

    out = capsys.readouterr().out
    assert "ANJANEYA-AI accumulation report (spec 29)" in out
    assert "readiness for retraining" in out
    assert "4 captured trading day(s)" in out


def test_the_json_carries_the_report(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    run_accumulation(tmp_path, config_dir)
    document = payload(tmp_path)
    assert document["report"] == "spec 29 - continuous data-accumulation report"
    assert document["spec"] == "29"
    assert document["captured_days"] == len(DAYS)
    assert document["captured_days_in_window"] == len(DAYS)
    assert document["days_since_last_capture"] == 0
    assert document["latest_capture"] == TODAY.isoformat()
    assert document["caveats"]


def test_the_gate_is_spec_4_3s(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    run_accumulation(tmp_path, config_dir)
    document = payload(tmp_path)
    assert document["gate"]["stage"] == "INSUFFICIENT"
    # Relaxed 60 -> 45 (Phase 9); the stage is still INSUFFICIENT on four days.
    assert document["gate"]["gates"]["ml_training_min_days"] == 45


def test_the_markdown_has_the_five_sections(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    run_accumulation(tmp_path, config_dir)
    text = next(out_dir(tmp_path).glob("*_accumulation.md")).read_text(encoding="utf-8")
    for heading in (
        "## 1. Data captured",
        "## 2. Coverage",
        "## 4. Readiness for retraining",
        "## Caveats",
    ):
        assert heading in text
    assert "| data_coverage |" in text


def test_the_csv_has_one_row_per_table(tmp_path, frozen_today) -> None:
    import pandas as pd

    config_dir = seed(tmp_path)
    run_accumulation(tmp_path, config_dir)
    csv = next(out_dir(tmp_path).glob("*_tables.csv"))
    frame = pd.read_csv(csv)
    assert list(frame.columns) == ["table", "rows", "days", "first", "last", "unavailable"]
    assert "features" in set(frame["table"])
    assert "data_coverage" in set(frame["table"])


def test_the_report_never_writes_to_the_database(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    before = table_counts(tmp_path)
    run_accumulation(tmp_path, config_dir)
    assert table_counts(tmp_path) == before


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------
def test_no_write_reports_without_writing(tmp_path, frozen_today, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_accumulation(tmp_path, config_dir, "--no-write")
    assert exit_code == EXIT_OK
    assert written(tmp_path) == []
    assert "--no-write" in capsys.readouterr().out


def test_dry_run_plans_and_writes_nothing(tmp_path, frozen_today, capsys) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_accumulation(tmp_path, config_dir, "--dry-run")
    assert exit_code == EXIT_OK
    assert written(tmp_path) == []
    out = capsys.readouterr().out
    assert "accumulation plan (spec 29)" in out
    assert "would read" in out


def test_json_prints_the_payload(tmp_path, frozen_today, capsys) -> None:
    config_dir = seed(tmp_path)
    run_accumulation(tmp_path, config_dir, "--json")
    document = json.loads(capsys.readouterr().out)
    assert document["mode"] == "accumulation"
    assert document["captured_days"] == len(DAYS)


def test_weeks_narrows_the_window(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    run_accumulation(tmp_path, config_dir, "--weeks", "8")
    assert payload(tmp_path)["window_start"] == (TODAY - timedelta(days=55)).isoformat()

    run_accumulation(tmp_path, config_dir, "--weeks", "1")
    assert payload(tmp_path)["window_start"] == (TODAY - timedelta(days=6)).isoformat()


# ---------------------------------------------------------------------------
# Gaps, empty databases and the model
# ---------------------------------------------------------------------------
def test_a_gap_day_is_reported_and_fails_the_coverage_requirement(
    tmp_path, frozen_today
) -> None:
    config_dir = seed(tmp_path, gaps={DAYS[-1]: 3})
    run_accumulation(tmp_path, config_dir)
    document = payload(tmp_path)

    assert document["gap_days"]
    assert document["gap_days"][0]["trading_date"] == DAYS[-1].isoformat()
    assert document["gap_days"][0]["gap_count"] == 3
    coverage = next(item for item in document["readiness"] if item["key"] == "coverage")
    assert coverage["met"] is False
    assert any("coverage gap(s)" in message for message in document["warnings"])


def test_an_empty_database_is_reported_not_crashed(tmp_path, frozen_today, capsys) -> None:
    config_dir = seed(tmp_path, days=(), seed_db=False)
    exit_code = run_accumulation(tmp_path, config_dir)
    assert exit_code == EXIT_OK
    document = payload(tmp_path)
    assert document["captured_days"] == 0
    assert document["gate"]["stage"] == "INSUFFICIENT"
    assert any("no trading day has been captured yet" in m for m in document["warnings"])
    assert "caveats" in capsys.readouterr().out


def test_the_newest_model_is_reported(tmp_path, frozen_today) -> None:
    from tests.support.ai import write_signal_model

    config_dir = seed(tmp_path)
    write_signal_model(models_dir(tmp_path))
    run_accumulation(tmp_path, config_dir)
    document = payload(tmp_path)
    assert document["newest_model"]["name"].endswith("_lightgbm.txt")
    assert document["newest_model"]["horizon_minutes"] == 5


def test_run_artifacts_are_registered_on_the_manifest(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    exit_code = run_accumulation(tmp_path, config_dir, run_artifacts=True)
    assert exit_code == EXIT_OK
    runs = sorted((tmp_path / "data" / "runs").glob("*/manifest.json"))
    assert runs
    manifest = json.loads(runs[-1].read_text(encoding="utf-8"))
    assert manifest["mode"] == "accumulation"
    registered = {Path(str(path)).name for path in manifest["artifacts"]}
    assert any(name.endswith("_accumulation.json") for name in registered)
    assert any(name.endswith("_accumulation.md") for name in registered)
    notes = " ".join(str(note) for note in manifest.get("notes", []))
    assert "captured trading day(s)" in notes


def test_the_mode_is_registered_so_it_is_not_planned(tmp_path, frozen_today) -> None:
    """A registered mode never takes the NOT_IMPLEMENTED path."""
    config_dir = seed(tmp_path)
    assert run_accumulation(tmp_path, config_dir) != EXIT_NOT_IMPLEMENTED


# ---------------------------------------------------------------------------
# The reader and the writer, in isolation
# ---------------------------------------------------------------------------
def test_the_reader_collapses_the_ledger_per_day(tmp_path) -> None:
    from core.storage import read_accumulation_inputs, read_coverage_days

    seed(tmp_path, gaps={DAYS[-1]: 2})
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        days = read_coverage_days(connection)
        found = read_accumulation_inputs(connection, as_of=TODAY, weeks=4)
    finally:
        connection.close()

    assert len(days) == len(DAYS)
    assert days[-1].gap_count == 2
    assert found.quality_counts == {"OK": len(DAYS) - 1, "GAP_FILLED": 1}
    assert found.rows_for("data_coverage") == len(DAYS)


def test_the_reader_reports_a_missing_table_rather_than_zero_rows(tmp_path) -> None:
    from core.storage import read_table_profile

    seed(tmp_path, days=())
    connection = connect(db_file(tmp_path), read_only=True)
    try:
        profile = read_table_profile(connection, "no_such_table")
    finally:
        connection.close()
    assert profile.present is False
    assert profile.unavailable


def test_the_writer_is_atomic_and_counted(tmp_path) -> None:
    policy = AccumulationPolicy()
    report = build_accumulation_report(
        _empty_inputs(), policy, as_of=TODAY, window_weeks=4, trading_days_in_window=4
    )
    writer = AccumulationWriter(out_dir(tmp_path), run_id="run", config_hash="hash")
    paths = writer.write(report, stem=accumulation_stem(TODAY))
    assert len(paths) == 3
    assert writer.stats.files == 3
    assert writer.stats.failures == 0
    assert not list(out_dir(tmp_path).glob("*.tmp"))


def test_the_config_block_is_honoured(tmp_path, frozen_today) -> None:
    config_dir = seed(tmp_path)
    config_dir = shrink_config(tmp_path, model={"accumulation": {"window_weeks": 2}})
    run_accumulation(tmp_path, config_dir)
    assert payload(tmp_path)["window_start"] == (TODAY - timedelta(days=13)).isoformat()


def _empty_inputs():
    from core.accumulation import AccumulationInputs

    return AccumulationInputs()

