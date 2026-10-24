"""Shared pytest fixtures and the project-wide network guard.

Two project rules are enforced *mechanically* here rather than by convention:

1. **No test may reach the network.** An autouse fixture wraps
   ``socket.socket.connect`` and ``socket.create_connection`` so that any
   *off-machine* connection raises :class:`NetworkAccessBlocked`. A test that
   accidentally called Kite would otherwise be slow, flaky, and could burn the
   real rate limit. Tests marked ``live`` (opt-in via ``ANJANEYA_RUN_LIVE=1``)
   skip the guard entirely.

   Loopback addresses are deliberately still allowed: importing ``kiteconnect``
   pulls in Twisted, which installs a reactor that opens a socket pair on
   ``127.0.0.1`` at import time. That is a local IPC detail, not network access.

2. **Tests never write into the repository's ``data/`` or ``db/``.**
   ``isolated_paths`` redirects ``ANJANEYA_DATA_ROOT`` and ``ANJANEYA_DB_PATH``
   to ``tmp_path`` for every test that uses the fixtures below.

3. **Tests never read a developer's ``.env``.** ``ANJANEYA_SKIP_DOTENV=1`` is set
   below for the whole session, so real credentials cannot leak into a test that
   happens to exercise an entry point. A test that *wants* to prove the loading
   behaviour passes an explicit mapping to ``core.env.load_env_file``, or removes
   the flag itself (``monkeypatch.delenv``) before calling a script's ``main``.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from core.config_loader import AnjaneyaConfig, load_config
from core.env import SKIP_ENV_FILE_VAR

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

#: Rule 3 above: the suite is hermetic even on a machine with a real ``.env``.
#: ``setdefault`` so an operator can still export the flag explicitly for a
#: debugging session, and so this never fights a deliberate override.
os.environ.setdefault(SKIP_ENV_FILE_VAR, "1")

#: Addresses that are considered local and therefore harmless in tests.
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test attempts real (off-machine) network access."""


def _is_loopback(address: Any) -> bool:
    if isinstance(address, tuple) and address:
        return str(address[0]) in LOOPBACK_HOSTS
    return str(address) in LOOPBACK_HOSTS


def _guarded_connect(original: Any) -> Any:
    def _connect(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback(address):
            return original(self, address, *args, **kwargs)
        raise NetworkAccessBlocked(
            f"Network access attempted during a test (target {address!r}). ANJANEYA tests must be "
            "fully offline; stub the HTTP/SDK call instead (see tests/unit/test_zerodha_auth.py)."
        )

    return _connect


def _guarded_create_connection(original: Any) -> Any:
    def _create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback(address):
            return original(address, *args, **kwargs)
        raise NetworkAccessBlocked(
            f"Network access attempted during a test (target {address!r}). ANJANEYA tests must be "
            "fully offline; stub the HTTP/SDK call instead."
        )

    return _create_connection


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast on off-machine socket use, unless the test is marked ``live``."""
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect(socket.socket.connect), raising=True)
    monkeypatch.setattr(
        socket,
        "create_connection",
        _guarded_create_connection(socket.create_connection),
        raising=True,
    )


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def config_dir() -> Path:
    """Absolute path of the shipped ``config/`` directory."""
    return CONFIG_DIR


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Path]]:
    """Redirect the data root and DuckDB path into ``tmp_path``."""
    data_root = tmp_path / "data"
    db_path = tmp_path / "db" / "anjaneeya.duckdb"
    monkeypatch.setenv("ANJANEYA_DATA_ROOT", str(data_root))
    monkeypatch.setenv("ANJANEYA_DB_PATH", str(db_path))
    # Keep the environment hermetic: no real credentials may leak into a test.
    for name in (
        "ZERODHA_API_KEY",
        "ZERODHA_API_SECRET",
        "ZERODHA_USER_ID",
        "ZERODHA_PASSWORD",
        "ZERODHA_TOTP_KEY",
        "ZERODHA_REDIRECT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield {"data_root": data_root, "db_path": db_path}


@pytest.fixture
def config(isolated_paths: dict[str, Path]) -> AnjaneyaConfig:
    """The real, shipped configuration, resolved with isolated paths."""
    return load_config(CONFIG_DIR)


@pytest.fixture
def live_enabled() -> bool:
    """True when the operator explicitly enabled live tests."""
    return os.environ.get("ANJANEYA_RUN_LIVE") == "1"

