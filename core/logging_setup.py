"""Centralised logging setup (loguru) for all ANJANEYA applications.

Design goals

* **One place** configures logging; apps never call ``loguru.add`` directly.
* **Secret hygiene**: configured secrets (API key, password, TOTP, access
  token) are redacted from every log record, so a stray ``repr()`` of a
  settings object can never leak credentials into a log file.
* **Structured sinks**: human-readable console/file sinks plus an optional
  JSON-lines sink for later analysis.
* **Idempotent**: calling :func:`setup_logging` twice does not duplicate sinks
  (which would otherwise double every log line).

Importing this module performs no I/O; sinks are only created when
:func:`setup_logging` is called explicitly.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "InterceptHandler",
    "SECRET_KEYS",
    "SECRET_SUFFIXES",
    "collect_secrets",
    "make_redaction_patcher",
    "setup_logging",
    "logger",
]

#: Record keys whose values are always replaced with ``"***"``.
SECRET_KEYS = frozenset(
    {
        "api_key",
        "api_secret",
        "password",
        "totp",
        "totp_key",
        "request_token",
        "access_token",
        "public_token",
        "enctoken",
        "authorization",
    }
)

#: Key-name suffixes that also mark a value as secret (e.g. ``zerodha_api_key``).
SECRET_SUFFIXES = ("api_key", "api_secret", "secret", "password", "passwd", "token", "totp")

_MASK = "***"
#: Handler ids registered by :func:`setup_logging`, so it can be re-run safely.
_ACTIVE_HANDLER_IDS: list[int] = []
_DEFAULT_HANDLER_REMOVED = False


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SECRET_KEYS or lowered.endswith(SECRET_SUFFIXES)


def _redact_mapping(data: Any) -> Any:
    if isinstance(data, Mapping):
        return {
            key: (_MASK if _is_secret_key(str(key)) else _redact_mapping(value))
            for key, value in data.items()
        }
    if isinstance(data, (list, tuple, set)):
        return type(data)(_redact_mapping(item) for item in data)
    return data


def collect_secrets(environ: Mapping[str, str], names: Iterable[str]) -> tuple[str, ...]:
    """Return the non-empty secret *values* for ``names``.

    Used to build a redaction filter that masks values even when they appear
    inside an unrelated log message (e.g. a URL carrying the access token).
    """
    found: list[str] = []
    for name in names:
        value = environ.get(name)
        if value and len(value) >= 6:  # ignore trivially short placeholders
            found.append(value)
    return tuple(found)


def make_redaction_patcher(secret_values: Iterable[str] = ()) -> Any:
    """Build a loguru record patcher that masks secret keys and values."""
    secrets = tuple(value for value in secret_values if value)

    def _patch(record: dict[str, Any]) -> None:
        extra = record.get("extra")
        if extra:
            record["extra"] = _redact_mapping(dict(extra))
        if not secrets:
            return
        message = str(record.get("message", ""))
        for secret in secrets:
            if secret in message:
                message = message.replace(secret, _MASK)
        record["message"] = message

    return _patch


class InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records into loguru (spec 6.5 traceability).

    ``kiteconnect``, ``urllib3`` and friends log through the stdlib; without
    this they would bypass the ANJANEYA sinks entirely.
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial routing
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    *,
    app_name: str,
    log_dir: Path | str,
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    rotation: str = "00:00",
    retention: str = "30 days",
    json_sink: bool = False,
    json_sink_name: str = "anjaneeya.jsonl",
    intercept_stdlib: bool = True,
    secret_values: Iterable[str] = (),
    quiet_loggers: Iterable[str] = ("urllib3", "kiteconnect", "asyncio"),
    enqueue: bool = False,
) -> Any:
    """Configure loguru and return the shared logger.

    Parameters
    ----------
    app_name:
        Used for the per-app text log file name (``<log_dir>/<app_name>.log``)
        and injected as ``extra["app"]`` into every record, so a sink format can
        always reference it (stdlib-routed records included).
    log_dir:
        Directory for file sinks. Created if missing.
    level:
        One of loguru's level names (``TRACE`` ... ``CRITICAL``).
    json_sink:
        When true an additional JSON-lines sink is written next to the text log.
    secret_values:
        Values that must be masked if they ever appear in a log message.
    quiet_loggers:
        Stdlib loggers forced to WARNING so third-party noise cannot swamp the
        ANJANEYA logs.
    enqueue:
        Hand records to a background thread (safer under concurrency, at the
        cost of slightly delayed writes). Off by default so log output is
        deterministic; the tick-capture app turns it on.

    Returns
    -------
    A loguru logger bound with ``app=<app_name>``, ready for immediate use.
    """
    global _DEFAULT_HANDLER_REMOVED

    level = level.upper()
    for handler_id in _ACTIVE_HANDLER_IDS:
        with contextlib.suppress(ValueError):  # handler already removed
            logger.remove(handler_id)
    _ACTIVE_HANDLER_IDS.clear()

    if not _DEFAULT_HANDLER_REMOVED:
        with contextlib.suppress(ValueError):  # default handler already gone
            logger.remove(0)
        _DEFAULT_HANDLER_REMOVED = True

    # ``extra`` provides a default ``app`` for records that were not created by
    # our bound logger (stdlib records routed through InterceptHandler would
    # otherwise crash the formatter with KeyError: 'app').
    logger.configure(extra={"app": app_name}, patcher=make_redaction_patcher(secret_values))

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[app]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[app]} | "
        "{name}:{function}:{line} | {message}"
    )

    if console:
        _ACTIVE_HANDLER_IDS.append(
            logger.add(
                sys.stderr,
                level=level,
                format=console_format,
                colorize=True,
                backtrace=False,
                diagnose=False,
                enqueue=False,
            )
        )

    log_path = Path(log_dir)
    if file:
        log_path.mkdir(parents=True, exist_ok=True)
        _ACTIVE_HANDLER_IDS.append(
            logger.add(
                log_path / f"{app_name}.log",
                level=level,
                format=file_format,
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=enqueue,
                backtrace=False,
                diagnose=False,
            )
        )
        if json_sink:
            _ACTIVE_HANDLER_IDS.append(
                logger.add(
                    log_path / json_sink_name,
                    level=level,
                    serialize=True,
                    rotation=rotation,
                    retention=retention,
                    encoding="utf-8",
                    enqueue=enqueue,
                )
            )

    if intercept_stdlib:
        root = logging.getLogger()
        root.handlers = [InterceptHandler()]
        root.setLevel(level)
        for name in quiet_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    return logger.bind(app=app_name)
