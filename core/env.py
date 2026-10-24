"""``.env`` loading - the single explicit place where ANJANEYA reads a dotenv file.

Credentials and runtime overrides live in the process environment, but the
ergonomic workflow is a ``.env`` file next to ``config/`` (copied from
``.env.example``). This module owns that file and nothing else:

* :mod:`core.config_loader` stays free of import-time I/O (a documented project
  rule: importing a ``core`` module must never read the disk), so the load is an
  explicit call;
* every entry point (``bootstrap_app``, ``scripts/daily_login.py``,
  ``scripts/health_check.py``) calls :func:`load_env_file` **before** the
  configuration is loaded, so that ``${VAR}`` placeholders in the YAML files
  resolve *and* the ``ZERODHA_*`` values reach the log-redaction list;
* YAML files still only ever hold variable **names**, never values.

Rules, in order of importance:

1. **A real environment variable always wins.** ``override=False`` is the
   default, so ``ZERODHA_API_KEY=... uv run python ...`` (or CI secrets) beat the
   file. ``.env`` fills gaps; it does not impose values.
2. **Blank means unset.** ``ZERODHA_PASSWORD=`` is ignored rather than written as
   an empty string, matching how credential presence is detected elsewhere
   (``bool(os.environ.get(name))``).
3. **Secrets are never logged.** Only the path and the *names* that were applied
   are ever exposed (:class:`EnvFileResult`).
4. **``ANJANEYA_SKIP_DOTENV=1`` disables loading entirely.** The test suite sets
   it so that a developer's real ``.env`` can never leak into a test run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

__all__ = [
    "ENV_FILE_NAME",
    "SKIP_ENV_FILE_VAR",
    "TRUTHY",
    "EnvFileResult",
    "default_env_path",
    "env_file_skipped",
    "load_env_file",
]

#: Name of the dotenv file, looked up next to the configuration directory.
ENV_FILE_NAME = ".env"

#: Set to a truthy value to make :func:`load_env_file` a no-op.
SKIP_ENV_FILE_VAR = "ANJANEYA_SKIP_DOTENV"

#: Values of :data:`SKIP_ENV_FILE_VAR` that mean "yes".
TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})


@dataclass(frozen=True, slots=True)
class EnvFileResult:
    """What :func:`load_env_file` did - never a secret value.

    ``names`` lists the variable *names* that were applied, i.e. those present in
    the file, non-blank, and not already set in the environment.
    """

    #: The path that was looked at (whether or not it exists).
    path: Path
    #: True when the file existed, was readable and was applied.
    loaded: bool = False
    #: True when loading was disabled by :data:`SKIP_ENV_FILE_VAR`.
    skipped: bool = False
    #: Variable names that were taken from the file.
    names: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Truthy only when values were actually applied."""
        return self.loaded

    def describe(self) -> str:
        """One-line, secret-free summary for logs and status output."""
        if self.skipped:
            return f"{self.path} (skipped: {SKIP_ENV_FILE_VAR}=1)"
        if not self.loaded:
            return f"{self.path} (not found)"
        return f"{self.path} (loaded: {', '.join(self.names)})"


def env_file_skipped(environ: Mapping[str, str] | None = None) -> bool:
    """True when ``ANJANEYA_SKIP_DOTENV`` asks us to ignore ``.env`` files."""
    env = os.environ if environ is None else environ
    return (env.get(SKIP_ENV_FILE_VAR) or "").strip().lower() in TRUTHY


def default_env_path(config_dir: str | Path | None = None) -> Path:
    """The ``.env`` a run looks for: the parent of ``config_dir`` (the repo root).

    When ``config_dir`` is omitted the shipped ``<repo>/config`` is assumed, which
    keeps the answer correct even when called from another working directory.
    """
    if config_dir is None:
        return Path(__file__).resolve().parent.parent / ENV_FILE_NAME
    return Path(config_dir).resolve().parent / ENV_FILE_NAME


def load_env_file(
    path: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> EnvFileResult:
    """Read a dotenv file into ``environ`` and report what happened.

    Parameters
    ----------
    path:
        Explicit file to read. When omitted, :func:`default_env_path` derives one
        from ``config_dir``.
    config_dir:
        Configuration directory the default path is derived from.
    environ:
        Mutable mapping to apply values to. Defaults to :data:`os.environ`; tests
        pass a plain dict to stay hermetic. The ``ANJANEYA_SKIP_DOTENV`` check
        reads this same mapping.
    override:
        When False (the default) existing non-empty variables are left untouched.

    Returns
    -------
    EnvFileResult
        Always returned (never ``None``): a missing file and a skipped file are
        ordinary outcomes, not errors - ANJANEYA also runs fully offline.

    Notes
    -----
    ``dotenv_values`` is used rather than ``load_dotenv`` so that the target
    mapping (and therefore the precedence and the blank handling above) is
    explicit and injectable instead of hard-wired to :data:`os.environ`.
    """
    env: MutableMapping[str, str] = os.environ if environ is None else environ
    target = Path(path).resolve() if path is not None else default_env_path(config_dir)

    if env_file_skipped(env):
        return EnvFileResult(path=target, skipped=True)
    if not target.is_file():
        return EnvFileResult(path=target)

    applied: list[str] = []
    for name, value in dotenv_values(target, encoding="utf-8").items():
        key = (name or "").strip()
        text = (value or "").strip()
        if not key or not text:
            continue
        if not override and (env.get(key) or "").strip():
            continue
        env[key] = text
        applied.append(key)
    return EnvFileResult(path=target, loaded=True, names=tuple(sorted(applied)))

