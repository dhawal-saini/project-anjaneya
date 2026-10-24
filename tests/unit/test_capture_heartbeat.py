"""Tests for the capture heartbeat file."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from apps.anjaneeya_capture.heartbeat import (
    HEARTBEAT_FORMAT_VERSION,
    Heartbeat,
    HeartbeatState,
    heartbeat_age_seconds,
    read_heartbeat,
)
from tests.support.capture import RecordingLogger

NOW = datetime(2026, 9, 29, 9, 30, tzinfo=UTC)


def make_heartbeat(tmp_path: Path, *, clock=None, interval: float = 10.0) -> Heartbeat:
    return Heartbeat(
        tmp_path / "heartbeat" / "capture.json",
        interval_seconds=interval,
        stale_after_seconds=90.0,
        state=HeartbeatState(run_id="run-1"),
        logger=RecordingLogger(),
        clock=clock or (lambda: NOW),
    )


def test_publish_writes_a_complete_payload(tmp_path) -> None:
    heartbeat = make_heartbeat(tmp_path)
    heartbeat.update(
        status="RUNNING",
        connected=True,
        ticks_total=1234,
        newest_price=25012.4,
        expiries=["2026-10-01"],
    )
    path = heartbeat.publish(force=True)

    assert path is not None and path.is_file()
    payload = read_heartbeat(path)
    assert payload is not None
    assert payload["format_version"] == HEARTBEAT_FORMAT_VERSION
    assert payload["run_id"] == "run-1"
    assert payload["status"] == "RUNNING"
    assert payload["connected"] is True
    assert payload["ticks_total"] == 1234
    assert payload["newest_price"] == 25012.4
    assert payload["expiries"] == ["2026-10-01"]
    assert heartbeat.writes == 1


def test_publish_is_rate_limited_unless_forced(tmp_path) -> None:
    moments = [NOW]

    def clock() -> datetime:
        return moments[0]

    heartbeat = make_heartbeat(tmp_path, clock=clock, interval=10)
    assert heartbeat.publish() is not None  # the first write always happens
    moments[0] = NOW + timedelta(seconds=5)
    assert heartbeat.publish() is None
    moments[0] = NOW + timedelta(seconds=11)
    assert heartbeat.publish() is not None
    assert heartbeat.publish(force=True) is not None
    assert heartbeat.writes == 3


def test_publish_leaves_no_temp_files_behind(tmp_path) -> None:
    heartbeat = make_heartbeat(tmp_path)
    heartbeat.publish(force=True)
    assert list(heartbeat.path.parent.glob(".heartbeat-*")) == []


def test_update_rejects_unknown_fields(tmp_path) -> None:
    heartbeat = make_heartbeat(tmp_path)
    with pytest.raises(AttributeError, match="no field 'tick_total'"):
        heartbeat.update(tick_total=1)


def test_notes_are_deduplicated_and_bounded(tmp_path) -> None:
    heartbeat = make_heartbeat(tmp_path)
    for index in range(15):
        heartbeat.note(f"note {index}")
    heartbeat.note("note 14")
    assert len(heartbeat.state.notes) == 10
    assert heartbeat.state.notes[-1] == "note 14"


def test_staleness_and_age(tmp_path) -> None:
    moments = [NOW]

    def clock() -> datetime:
        return moments[0]

    heartbeat = make_heartbeat(tmp_path, clock=clock)
    assert heartbeat.is_stale() is True  # nothing written yet
    heartbeat.publish(force=True)
    assert heartbeat.age_seconds() == 0.0
    assert heartbeat.is_stale() is False
    moments[0] = NOW + timedelta(seconds=120)
    assert heartbeat.is_stale() is True
    assert heartbeat.age_seconds() == 120.0


def test_read_heartbeat_handles_missing_and_corrupt_files(tmp_path) -> None:
    assert read_heartbeat(tmp_path / "nope.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert read_heartbeat(corrupt) is None

    wrong_version = tmp_path / "old.json"
    wrong_version.write_text(json.dumps({"format_version": 0}), encoding="utf-8")
    assert read_heartbeat(wrong_version) is None

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"format_version": HEARTBEAT_FORMAT_VERSION}), encoding="utf-8")
    assert read_heartbeat(incomplete) is None


def test_heartbeat_age_seconds_is_timezone_tolerant() -> None:
    payload = {"written_at": NOW.isoformat()}
    assert heartbeat_age_seconds(payload, now=NOW + timedelta(seconds=30)) == 30.0
    naive = {"written_at": "2026-09-29T09:30:00"}
    assert heartbeat_age_seconds(naive, now=NOW) == 0.0
    assert heartbeat_age_seconds({"written_at": "rubbish"}, now=NOW) is None
    assert heartbeat_age_seconds({}, now=NOW) is None


def test_state_round_trips_through_a_payload(tmp_path) -> None:
    state = HeartbeatState(
        run_id="run-9",
        status="STOPPED",
        connected=False,
        ticks_total=42,
        newest_tick_timestamp=NOW,
        last_tick_at=NOW - timedelta(seconds=1),
        notes=["a"],
        expiries=["2026-10-01"],
        gap_count=2,
    )
    payload = state.to_payload(written_at=NOW)
    restored = HeartbeatState.from_payload(payload)
    assert restored.run_id == "run-9"
    assert restored.ticks_total == 42
    assert restored.newest_tick_timestamp == NOW
    assert restored.last_tick_at == NOW - timedelta(seconds=1)
    assert restored.notes == ["a"]
    assert restored.expiries == ["2026-10-01"]
    assert restored.gap_count == 2


def test_from_payload_ignores_unknown_keys() -> None:
    payload = {"format_version": 1, "run_id": "r", "app": "a", "future_field": 1}
    restored = HeartbeatState.from_payload(payload)
    assert restored.run_id == "r"
    assert not hasattr(restored, "future_field")
