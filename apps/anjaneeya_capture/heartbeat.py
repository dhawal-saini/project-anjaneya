"""Capture heartbeat: a small JSON file that says "capture is alive right now".

Why a file rather than a log line
---------------------------------
A log is a write-only stream; answering "is the capture process healthy *right
now*?" from logs means tailing and guessing. The heartbeat is a *state*: it is
rewritten atomically every ``interval_seconds`` with the counters an operator
actually asks about, and it goes stale after ``stale_after_seconds``.

It is also the only way to observe a running capture session: DuckDB allows a
single read-write process per file, so ``--mode status`` cannot open the database
while capture holds it. The heartbeat closes that gap.

Writing is atomic (temp file + ``os.replace``) and never raises into the capture
loop - a heartbeat problem must not cost a tick.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["HEARTBEAT_FORMAT_VERSION", "Heartbeat", "HeartbeatState", "heartbeat_age_seconds", "read_heartbeat"]

#: Bumped when the payload layout changes (readers must tolerate unknown keys).
HEARTBEAT_FORMAT_VERSION = 1

#: Fields a reader may rely on; new keys are always additive.
REQUIRED_FIELDS = ("format_version", "run_id", "app", "status", "written_at")


def _iso(value: datetime | None) -> str | None:
    """ISO-8601 with timezone, or ``None``."""
    return value.isoformat() if isinstance(value, datetime) else None


def _parse(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp tolerantly (``None`` on failure)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(slots=True)
class HeartbeatState:
    """Everything the heartbeat publishes about a running capture session."""

    run_id: str
    app: str = "anjaneeya_capture"
    status: str = "STARTING"
    mode: str | None = None
    session_date: str | None = None
    connected: bool = False
    reconnects: int = 0
    errors: int = 0
    ticks_total: int = 0
    snapshots_total: int = 0
    buffered_rows: int = 0
    ticks_per_minute: float = 0.0
    last_tick_at: datetime | None = None
    last_message_at: datetime | None = None
    last_flush_at: datetime | None = None
    newest_tick_timestamp: datetime | None = None
    newest_price: float | None = None
    poll_count: int = 0
    poll_failures: int = 0
    gap_count: int = 0
    largest_gap_seconds: float = 0.0
    subscribed_instruments: int = 0
    atm_strike: float | None = None
    expiries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self, *, written_at: datetime) -> dict[str, Any]:
        """Serialise the heartbeat (ISO-8601 for every timestamp)."""
        return {
            "format_version": HEARTBEAT_FORMAT_VERSION,
            "app": self.app,
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "written_at": written_at.isoformat(),
            "session_date": self.session_date,
            "connected": self.connected,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "ticks_total": self.ticks_total,
            "snapshots_total": self.snapshots_total,
            "buffered_rows": self.buffered_rows,
            "ticks_per_minute": round(self.ticks_per_minute, 3),
            "last_tick_at": _iso(self.last_tick_at),
            "last_message_at": _iso(self.last_message_at),
            "last_flush_at": _iso(self.last_flush_at),
            "newest_tick_timestamp": _iso(self.newest_tick_timestamp),
            "newest_price": self.newest_price,
            "poll_count": self.poll_count,
            "poll_failures": self.poll_failures,
            "gap_count": self.gap_count,
            "largest_gap_seconds": round(self.largest_gap_seconds, 3),
            "subscribed_instruments": self.subscribed_instruments,
            "atm_strike": self.atm_strike,
            "expiries": list(self.expiries),
            "notes": list(self.notes),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HeartbeatState:
        """Rebuild a state from a payload (unknown keys are ignored)."""
        state = cls(run_id=str(payload.get("run_id", "")), app=str(payload.get("app", "")))
        for key, value in payload.items():
            if key in {"format_version", "written_at", "run_id", "app"}:
                continue
            if key in {"expiries", "notes"}:
                setattr(state, key, list(value or []))
            elif key in {"last_tick_at", "last_message_at", "last_flush_at", "newest_tick_timestamp"}:
                setattr(state, key, _parse(value))
            elif hasattr(state, key):
                setattr(state, key, value)
        return state


class Heartbeat:
    """Publishes a :class:`HeartbeatState` to a JSON file at a fixed cadence."""

    def __init__(
        self,
        path: Path,
        *,
        interval_seconds: float,
        stale_after_seconds: float,
        state: HeartbeatState,
        logger: Any = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.state = state
        self.logger = logger
        self._clock = clock or (lambda: datetime.now(UTC))
        self._last_written: datetime | None = None
        self.writes = 0

    def update(self, **fields: Any) -> None:
        """Set state fields (an unknown name raises - typos must not be silent)."""
        for name, value in fields.items():
            if not hasattr(self.state, name):
                raise AttributeError(f"HeartbeatState has no field {name!r}")
            setattr(self.state, name, value)

    def note(self, message: str) -> None:
        """Attach a note, keeping only the most recent ones (bounded file size)."""
        if message not in self.state.notes:
            self.state.notes.append(message)
        del self.state.notes[:-10]

    def publish(self, *, force: bool = False, status: str | None = None) -> Path | None:
        """Write the heartbeat when due (or when ``force``); returns the path."""
        now = self._clock()
        if status is not None:
            self.state.status = status
        if not force and self._last_written is not None:
            elapsed = (now - self._last_written).total_seconds()
            if elapsed < self.interval_seconds:
                return None
        payload = self.state.to_payload(written_at=now)
        try:
            self._write_atomic(payload)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            if self.logger is not None:
                self.logger.warning(f"could not write the heartbeat {self.path}: {exc}")
            return None
        self._last_written = now
        self.writes += 1
        return self.path

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        """Write via a temp file + ``os.replace`` (readers never see a partial file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".heartbeat-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, default=str)
            os.replace(temp_name, self.path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def age_seconds(self) -> float | None:
        """Seconds since this object last wrote the file (``None`` if never)."""
        if self._last_written is None:
            return None
        return (self._clock() - self._last_written).total_seconds()

    def is_stale(self, *, now: datetime | None = None) -> bool:
        """True when the last write is older than ``stale_after_seconds``."""
        reference = now or self._clock()
        if self._last_written is None:
            return True
        return (reference - self._last_written).total_seconds() > self.stale_after_seconds


def read_heartbeat(path: Path) -> dict[str, Any] | None:
    """Read a heartbeat file, returning ``None`` when absent/incompatible.

    A *stale* heartbeat is still returned: callers decide (via
    :func:`heartbeat_age_seconds`) whether it is fresh enough to trust.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("format_version") != HEARTBEAT_FORMAT_VERSION:
        return None
    if any(field_name not in payload for field_name in REQUIRED_FIELDS):
        return None
    return payload


def heartbeat_age_seconds(payload: dict[str, Any], *, now: datetime | None = None) -> float | None:
    """Age of a heartbeat payload in seconds (``None`` when unparseable)."""
    written = _parse(payload.get("written_at"))
    if written is None:
        return None
    reference = now or datetime.now(UTC)
    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return (reference - written).total_seconds()

