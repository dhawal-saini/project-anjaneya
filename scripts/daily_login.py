"""Kite Connect daily login - one command to make a capture session possible.

Zerodha invalidates the access token every day, so a morning capture run needs a
fresh one. This script performs the login *once*, stores the token in the
configured cache (``data/raw/.tokens/zerodha_token.json``) and verifies it against
``profile()``, so a later ``--mode capture`` never has to authenticate mid-session.

Two paths, in order of preference:

1. ``--request-token`` - paste the ``request_token`` from the Kite login redirect
   (works even when TOTP automation is blocked by the broker);
2. automated TOTP login using ``ZERODHA_USER_ID`` / ``ZERODHA_PASSWORD`` /
   ``ZERODHA_TOTP_KEY``.

Credentials are read from the environment, plus an optional ``.env`` file next to
the configuration directory (``core.env``). Real environment variables always win
over the file, and no secret is ever printed - only masked forms.

Usage::

    uv run python scripts/daily_login.py                     # cached or TOTP login
    uv run python scripts/daily_login.py --force             # ignore the cache
    uv run python scripts/daily_login.py --request-token abc # manual exchange
    uv run python scripts/daily_login.py --json

Exit codes: 0 = a usable token is stored, 2 = configuration/credentials problem,
1 = authentication failed.

The token value is never printed - only its masked form and its expiry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/daily_login.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.auth import (  # noqa: E402
    AuthenticationError,
    TokenStore,
    ZerodhaClient,
    ZerodhaCredentials,
    resolve_access_token,
)
from core.config_loader import ConfigError, load_config  # noqa: E402
from core.constants import get_timezone  # noqa: E402
from core.env import load_env_file  # noqa: E402
from core.logging_setup import collect_secrets, setup_logging  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG_ERROR = 2


def _mask(value: str | None) -> str:
    """Show only the tail of a secret."""
    if not value:
        return "<unset>"
    return f"...{value[-4:]}" if len(value) > 4 else "..."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/daily_login.py",
        description="Obtain and cache a fresh Zerodha Kite access token.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-dir", default=None, help="Directory holding the YAML configuration."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override settings.data.root for this run (useful for scratch runs and tests).",
    )
    parser.add_argument(
        "--request-token",
        default=None,
        help="Exchange this Kite request_token instead of performing a TOTP login.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore a valid cached token and log in again."
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the profile() call that proves the token works.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Machine-readable output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Log in, cache the token, verify it and report the outcome."""
    args = parse_args(argv)
    # `.env` next to the config directory is loaded first; a real environment
    # variable always wins over it (see core/env.py).
    env_file = load_env_file(config_dir=args.config_dir)
    if args.data_root:
        os.environ["ANJANEYA_DATA_ROOT"] = str(args.data_root)
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"ANJANEYA login: configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    zerodha = config.data_sources.zerodha
    config.ensure_directories()
    logger = setup_logging(
        app_name="daily_login",
        log_dir=config.log_dir,
        level=config.settings.logging.level,
        console=config.settings.logging.console,
        file=config.settings.logging.file,
        rotation=config.settings.logging.rotation,
        retention=config.settings.logging.retention,
        json_sink=config.settings.logging.json_sink,
        json_sink_name=config.settings.logging.json_sink_name,
        intercept_stdlib=config.settings.logging.intercept_stdlib,
        secret_values=collect_secrets(
            os.environ,
            [
                zerodha.api_key_env,
                zerodha.api_secret_env,
                zerodha.password_env,
                zerodha.totp_key_env,
            ],
        ),
    )
    now = datetime.now(get_timezone(config.timezone_name))

    # Say up front where the credentials came from: if `.env` was never found, that
    # is the single most common cause of a failing login.
    if not args.as_json:
        print(f"ANJANEYA Kite login (env file: {env_file.describe()})")

    try:
        credentials = ZerodhaCredentials.from_environ(zerodha)
    except AuthenticationError as exc:
        print(f"ANJANEYA login: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    store = TokenStore(
        config.token_store_path,
        timezone_name=config.timezone_name,
        validity_cutoff=zerodha.token_store.validity_cutoff,
    )
    cached_before = store.load()
    if args.force and cached_before is not None:
        store.clear()
        logger.warning("--force: the cached access token was cleared")

    try:
        token = resolve_access_token(
            zerodha,
            credentials=credentials,
            request_token=args.request_token,
            token_store=store,
            timezone_name=config.timezone_name,
            validity_cutoff=zerodha.token_store.validity_cutoff,
            now=now,
            refresh_on_startup=zerodha.token_store.refresh_on_startup,
            allow_totp_login=True,
        )
    except AuthenticationError as exc:
        print(f"ANJANEYA login failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - reported, never a traceback
        print(f"ANJANEYA login failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    store.save(token)
    reused = (
        cached_before is not None
        and not args.force
        and cached_before.access_token == token.access_token
    )

    profile: dict[str, object] | None = None
    if not args.no_verify:
        try:
            client = ZerodhaClient.from_config(config, token.access_token, logger=logger)
            profile = client.profile()
        except Exception as exc:  # noqa: BLE001 - a failed verification is worth reporting
            print(
                f"ANJANEYA login: token stored but profile() failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return EXIT_FAILED

    payload = {
        "status": "ok",
        "reused_cached_token": reused,
        "user_id": str(profile.get("user_id")) if profile else token.user_id,
        "api_key": _mask(token.api_key),
        "access_token": _mask(token.access_token),
        "login_time": token.login_time.isoformat(),
        "expires_at": token.expires_at.isoformat() if token.expires_at else None,
        "remaining_minutes": (
            round((token.expires_at - now).total_seconds() / 60, 1) if token.expires_at else None
        ),
        "obtained_via": token.obtained_via,
        "token_store": str(config.token_store_path),
        "verified": profile is not None,
        "totp_login_available": credentials.can_login_with_totp,
        "env_file": str(env_file.path),
        "env_file_loaded": env_file.loaded,
        "env_file_variables": list(env_file.names),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("ANJANEYA Kite login")
        print(f"  status          : {payload['status']}")
        print(f"  reused cache    : {payload['reused_cached_token']}")
        print(f"  obtained via    : {payload['obtained_via']}")
        print(f"  user id         : {payload['user_id']}")
        print(f"  api key         : {payload['api_key']}")
        print(f"  access token    : {payload['access_token']} (masked)")
        print(f"  expires at      : {payload['expires_at']}")
        print(f"  remaining       : {payload['remaining_minutes']} minutes")
        print(f"  token store     : {payload['token_store']}")
        print(f"  verified        : {payload['verified']}")
        if not credentials.can_login_with_totp and not args.request_token:
            print(
                "  note            : TOTP login is unavailable (set ZERODHA_USER_ID, "
                "ZERODHA_PASSWORD and ZERODHA_TOTP_KEY) - pass --request-token manually."
            )
    logger.info(f"daily login completed (verified={profile is not None})")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

