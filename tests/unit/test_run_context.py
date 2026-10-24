"""Tests for :mod:`core.run_context` - reproducibility metadata (spec 36)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from core.run_context import (
    RunContext,
    collect_package_versions,
    create_run_context,
    get_git_commit,
)


def test_run_id_is_app_prefixed_sortable_and_unique(tmp_path: Path) -> None:
    first = create_run_context("anjaneeya_capture", artifacts_root=tmp_path)
    second = create_run_context("anjaneeya_capture", artifacts_root=tmp_path)

    assert first.run_id.startswith("anjaneeya_capture_")
    assert first.run_id != second.run_id
    # ``<app>_<YYYYmmdd-HHMMSS>_<8 hex>``
    stamp = first.run_id.split("_", 2)[2]
    datetime.strptime(stamp[:15], "%Y%m%d-%H%M%S")  # raises if malformed
    assert first.artifacts_dir == tmp_path / first.run_id


def test_started_at_is_utc_and_ist_views_agree() -> None:
    context = create_run_context("anjaneeya_engine")
    assert context.started_at.tzinfo is not None
    assert context.started_at.utcoffset() == timedelta(0)
    ist = context.started_at_ist
    assert ist.tzinfo is not None
    assert ist.utcoffset() == timedelta(hours=5, minutes=30)
    assert ist.replace(tzinfo=None) - context.started_at.replace(tzinfo=None) == timedelta(
        hours=5, minutes=30
    )


def test_context_records_versions_and_seed() -> None:
    context = create_run_context(
        "anjaneeya_ai",
        package_names=["pandas", "definitely-not-installed-package"],
        random_seed=42,
    )
    assert context.package_versions["pandas"] is not None
    assert context.package_versions["definitely-not-installed-package"] is None
    assert context.random_seed == 42
    assert context.python_version


def test_collect_package_versions_handles_missing_packages() -> None:
    versions = collect_package_versions(["pandas", "no-such-package-xyz"])
    assert set(versions) == {"pandas", "no-such-package-xyz"}
    assert versions["no-such-package-xyz"] is None


def test_get_git_commit_is_none_safe(tmp_path: Path) -> None:
    """A missing repository must annotate, never block, a research run."""
    assert get_git_commit(tmp_path) is None


def test_manifest_is_written_with_the_full_provenance(tmp_path: Path) -> None:
    context = create_run_context(
        "anjaneeya_capture",
        mode="status",
        config_hash="deadbeef",
        config_dir=tmp_path / "config",
        artifacts_root=tmp_path / "runs",
        random_seed=7,
        package_names=["pandas"],
    )
    context.add_note("holidays are provisional")
    context.register_artifact("data/candles/NIFTY_1m.parquet")
    context.data_coverage["trading_days"] = 3

    path = context.write_manifest(status="SUCCESS")
    assert path == context.artifacts_dir / "manifest.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == context.run_id
    assert payload["app_name"] == "anjaneeya_capture"
    assert payload["mode"] == "status"
    assert payload["status"] == "SUCCESS"
    assert payload["config_hash"] == "deadbeef"
    assert payload["random_seed"] == 7
    assert payload["notes"] == ["holidays are provisional"]
    assert payload["artifacts"] == ["data/candles/NIFTY_1m.parquet"]
    assert payload["data_coverage"] == {"trading_days": 3}
    assert payload["package_versions"]["pandas"] is not None
    assert payload["started_at"] and payload["started_at_ist"] and payload["finished_at"]


def test_manifest_can_record_a_failure(tmp_path: Path) -> None:
    """Failed runs must also leave a manifest - that is what enables post-mortems."""
    context = create_run_context("anjaneeya_engine", artifacts_root=tmp_path / "runs")
    path = context.write_manifest(status="FAILED")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"


def test_write_manifest_requires_an_artifacts_directory() -> None:
    context = RunContext(
        run_id="r1",
        app_name="anjaneeya_capture",
        started_at=datetime.now().astimezone(),
    )
    try:
        context.write_manifest()
    except ValueError as exc:
        assert "artifacts_dir" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("write_manifest should require an artifacts directory")
