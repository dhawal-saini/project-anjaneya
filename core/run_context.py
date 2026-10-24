"""Run context: reproducibility metadata for every ANJANEYA run (spec 36).

A run manifest answers "what exactly produced this output?" months later:

* ``run_id``            unambiguous identifier, written on every stored row,
* ``config_hash``       SHA-256 of the committed YAML configuration,
* ``git_commit``        which code revision was running,
* ``package_versions``  pandas/numpy/duckdb/... versions,
* ``random_seed``       the seed used for any stochastic step,
* ``data_coverage``     how much data the run actually saw (spec 4.3 gates).

Importing this module performs no I/O; nothing is written until
:meth:`RunContext.write_manifest` is called.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from core.constants import DEFAULT_TIMEZONE, get_timezone

__all__ = ["RunContext", "create_run_context", "get_git_commit", "collect_package_versions"]


def get_git_commit(base_dir: Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` when unavailable.

    A missing git repository (or missing git binary) is not an error: research
    output must never be blocked by version control, only annotated.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(base_dir) if base_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_package_versions(names: list[str]) -> dict[str, str | None]:
    """Look up installed versions for ``names``; missing packages map to None."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass(slots=True)
class RunContext:
    """Metadata describing one execution of one ANJANEYA app."""

    run_id: str
    app_name: str
    started_at: datetime
    mode: str | None = None
    config_hash: str | None = None
    config_dir: str | None = None
    git_commit: str | None = None
    random_seed: int | None = None
    timezone_name: str = DEFAULT_TIMEZONE
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    package_versions: dict[str, str | None] = field(default_factory=dict)
    artifacts_dir: Path | None = None
    artifacts: list[str] = field(default_factory=list)
    data_coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def started_at_ist(self) -> datetime:
        """Start time converted to the research timezone."""
        return self.started_at.astimezone(get_timezone(self.timezone_name))

    def add_note(self, text: str) -> None:
        """Attach a free-text note to the manifest (e.g. a caveat/warning)."""
        self.notes.append(text)

    def register_artifact(self, path: str | Path) -> None:
        """Record an artifact path (relative paths are stored as-is)."""
        self.artifacts.append(str(path))

    def to_manifest_dict(
        self, *, status: str, finished_at: datetime | None = None
    ) -> dict[str, Any]:
        """Build the manifest payload written to disk."""
        return {
            "run_id": self.run_id,
            "app_name": self.app_name,
            "mode": self.mode,
            "status": status,
            "started_at": self.started_at.isoformat(),
            "started_at_ist": self.started_at_ist.isoformat(),
            "finished_at": (finished_at or datetime.now(UTC)).isoformat(),
            "config_hash": self.config_hash,
            "config_dir": self.config_dir,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "platform": self.platform,
            "timezone": self.timezone_name,
            "random_seed": self.random_seed,
            "package_versions": self.package_versions,
            "data_coverage": self.data_coverage,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    def write_manifest(self, status: str = "SUCCESS") -> Path:
        """Write ``manifest.json`` into the run artifacts directory.

        The run directory is created on demand. Manifests are written even for
        failed runs (the caller passes ``status="FAILED"``), which is what makes
        post-mortems possible.
        """
        if self.artifacts_dir is None:
            raise ValueError("RunContext.artifacts_dir is not set; cannot write a manifest")
        directory = Path(self.artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.to_manifest_dict(status=status), indent=2, default=str),
            encoding="utf-8",
        )
        return manifest_path


def create_run_context(
    app_name: str,
    *,
    mode: str | None = None,
    config_hash: str | None = None,
    config_dir: Path | str | None = None,
    artifacts_root: Path | str | None = None,
    random_seed: int | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    package_names: list[str] | None = None,
    base_dir: Path | str | None = None,
) -> RunContext:
    """Create a :class:`RunContext` with a unique, sortable ``run_id``.

    ``run_id`` format: ``<app>_<YYYYmmdd-HHMMSS-IST>_<8 hex chars>``. The IST
    timestamp makes artifacts orderable; the random suffix keeps concurrent
    runs (capture + engine + AI) from colliding.
    """
    tz = get_timezone(timezone_name)
    started = datetime.now(UTC)
    stamp = started.astimezone(tz).strftime("%Y%m%d-%H%M%S")
    run_id = f"{app_name}_{stamp}_{uuid.uuid4().hex[:8]}"

    artifacts_dir = Path(artifacts_root) / run_id if artifacts_root else None

    return RunContext(
        run_id=run_id,
        app_name=app_name,
        started_at=started,
        mode=mode,
        config_hash=config_hash,
        config_dir=str(config_dir) if config_dir else None,
        git_commit=get_git_commit(Path(base_dir) if base_dir else None),
        random_seed=random_seed,
        timezone_name=timezone_name,
        package_versions=collect_package_versions(package_names or []),
        artifacts_dir=artifacts_dir,
    )
