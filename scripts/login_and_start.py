"""One command: log in to Kite and start the capture session.

The runbook's morning routine is: log in, copy the ``request_token`` out of the
browser, exchange it, then start the capture. This script collapses those steps
into a single command:

1. **cached token** - a still-valid token in ``data/raw/.tokens/`` means no browser
   login is needed at all;
2. **login** - otherwise the Kite login URL is opened in the browser and the
   ``request_token`` is caught on the loopback redirect URL
   (:mod:`core.auth.browser_login`), then handed to :mod:`scripts.daily_login` so
   the exchange / verification / caching path exists exactly once (``--paste``
   keeps the manual copy/paste route as a fallback, ``--no-browser`` prints the
   URL instead of opening it);
3. **start** - only after ``daily_login`` verified the token with ``profile()``,
   ``apps.anjaneeya_capture.run --mode capture`` is started in a child process
   (exactly what ``scripts/start_capture.ps1`` does) and its exit code becomes
   this script's exit code.

A failed login therefore never produces a silent session that captured nothing:
the capture is not started unless the token was verified.

Usage::

    python scripts/login_and_start.py                          # browser login + capture
    python scripts/login_and_start.py --no-browser             # print the URL instead
    python scripts/login_and_start.py --paste                  # paste the redirect URL
    python scripts/login_and_start.py --no-start               # log in only
    python scripts/login_and_start.py -- --smoke --until 15:30  # capture flags
    python scripts/login_and_start.py --dry-run                # show the plan, do nothing

Exit codes: 0 = ok (when the capture runs, *its* exit code is propagated
unchanged), 1 = login failed, 2 = configuration/credentials problem,
3 = cancelled with Ctrl-C. Secrets are never printed - only masked forms, and the
``request_token`` never appears in the output at all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/login_and_start.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.auth import (  # noqa: E402
    AuthenticationError,
    TokenStore,
    ZerodhaCredentials,
    normalise_request_token,
    obtain_request_token,
)
from core.auth.browser_login import DEFAULT_REDIRECT_URL  # noqa: E402
from core.config_loader import ConfigError, load_config  # noqa: E402
from core.constants import get_timezone  # noqa: E402
from core.env import load_env_file  # noqa: E402
from scripts.daily_login import EXIT_OK  # noqa: E402

EXIT_LOGIN_FAILED = 1
EXIT_CONFIG_ERROR = 2
EXIT_CANCELLED = 3

#: How long the login waits for the browser before giving up.
DEFAULT_TIMEOUT_SECONDS = 300.0


def _mask(value: str | None) -> str:
    """Show only the tail of a secret."""
    if not value:
        return "<unset>"
    return f"...{value[-4:]}" if len(value) > 4 else "..."


def _port(value: str) -> int:
    """Argparse type for ``--port``: 1..65535 (0 could never match a registered URL)."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"{value!r} is not a port number") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port {port} is outside 1..65535")
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/login_and_start.py",
        description="Log in to Kite (browser-assisted) and start the ANJANEYA capture session.",
        epilog=(
            "Arguments after `--` are forwarded to the capture app, so every capture flag keeps "
            "working, e.g. `-- --smoke --until 15:30`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", default=None, help="Directory holding the YAML configuration.")
    parser.add_argument("--data-root", default=None, help="Override settings.data.root for this run.")
    parser.add_argument("--db-path", default=None, help="Override settings.data.db_path for this run.")
    parser.add_argument(
        "--redirect-url",
        default=None,
        help=(
            "Redirect URL registered for your Kite app; must be a loopback http URL such as "
            f"{DEFAULT_REDIRECT_URL} (default: ZERODHA_REDIRECT_URL, else {DEFAULT_REDIRECT_URL})."
        ),
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=None,
        help="Port to listen on; must be the port of the registered redirect URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for the browser to come back with the login redirect.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser; print the login URL for you to open.",
    )
    parser.add_argument(
        "--paste",
        action="store_true",
        help="Do not listen at all; paste the redirect URL (or the request_token) from the browser.",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Never log in: use the cached token, and fail if there is no usable one.",
    )
    parser.add_argument(
        "--force-login",
        action="store_true",
        help="Log in again even when the cached token is still valid.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the profile() call that proves the token works.",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Log in only; do not start the capture session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (login mode, redirect URL, capture command) without doing anything.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a JSON login summary instead of the human report.",
    )
    parser.add_argument(
        "capture_args",
        nargs=argparse.REMAINDER,
        help="Forwarded to the capture app; prefix them with `--`.",
    )
    return parser.parse_args(argv)


def start_capture(capture_args: list[str]) -> int:
    """Start ``apps.anjaneeya_capture.run --mode capture`` and wait for it.

    A child process (rather than an in-process call) keeps the long-running
    session's logging, signal handling and crash behaviour identical to
    ``scripts/start_capture.ps1``, and its exit code is returned unchanged so a
    scheduler sees the real outcome.
    """
    argv = [sys.executable, "-m", "apps.anjaneeya_capture.run", "--mode", "capture", *capture_args]
    print(f"==> ANJANEYA-CAPTURE session (mode=capture): {' '.join(argv[1:])}")
    completed = subprocess.run(argv, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def _login_argv(args: argparse.Namespace, request_token: str | None) -> list[str]:
    """Arguments for :mod:`scripts.daily_login` (the one login implementation)."""
    argv: list[str] = []
    if request_token:
        argv += ["--request-token", request_token]
    if args.config_dir:
        argv += ["--config-dir", str(args.config_dir)]
    if args.data_root:
        argv += ["--data-root", str(args.data_root)]
    if args.force_login:
        # Without --force, resolve_access_token prefers a cached token over the
        # request_token we just captured, which would silently ignore the login.
        argv.append("--force")
    if args.no_verify:
        argv.append("--no-verify")
    return argv


def _capture_argv(args: argparse.Namespace) -> list[str]:
    """Arguments forwarded to the capture app (its own flags come after `--`)."""
    argv: list[str] = []
    if args.config_dir:
        argv += ["--config-dir", str(args.config_dir)]
    if args.data_root:
        argv += ["--data-root", str(args.data_root)]
    if args.db_path:
        argv += ["--db-path", str(args.db_path)]
    return argv + list(args.capture_args or [])


def _obtain_token(args: argparse.Namespace, api_key: str, redirect_url: str) -> tuple[str, str]:
    """Return ``(request_token, login_label)``, from the browser or by pasting."""
    if args.paste:
        print("Paste the full redirect URL (or just the request_token) and press Enter:")
        return normalise_request_token(sys.stdin.readline()), "pasted request_token"
    token = obtain_request_token(
        api_key,
        redirect_url=redirect_url,
        timeout=args.timeout,
        open_browser=not args.no_browser,
        port=args.port,
        notify=lambda message: print(f"  {message}"),
    )
    return token, "browser login"


def run_login(login_args: list[str]) -> tuple[int, str, str]:
    """Run :mod:`scripts.daily_login` in a child process.

    Returns ``(exit_code, stdout, stderr)``. A child process keeps this script's
    own stdout clean for ``--json`` - and, more importantly, means the login's
    loguru console sink cannot capture *this* process's stdout.
    """
    argv = [sys.executable, "-m", "scripts.daily_login", *login_args]
    completed = subprocess.run(
        argv, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True
    )
    return int(completed.returncode), completed.stdout or "", completed.stderr or ""


def main(argv: list[str] | None = None) -> int:
    """Log in (verified) and, only then, start the capture session."""
    args = parse_args(argv)
    if args.capture_args and args.capture_args[0] == "--":
        args.capture_args = args.capture_args[1:]
    env_file = load_env_file(config_dir=args.config_dir)
    if args.data_root:
        os.environ["ANJANEYA_DATA_ROOT"] = str(args.data_root)
    if args.db_path:
        os.environ["ANJANEYA_DB_PATH"] = str(args.db_path)
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"ANJANEYA login: configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    zerodha = config.data_sources.zerodha
    try:
        credentials = ZerodhaCredentials.from_environ(zerodha)
    except AuthenticationError as exc:
        print(f"ANJANEYA login: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    redirect_url = (
        args.redirect_url
        or (os.environ.get(zerodha.redirect_url_env) or "").strip()
        or DEFAULT_REDIRECT_URL
    )
    store = TokenStore(
        config.token_store_path,
        timezone_name=config.timezone_name,
        validity_cutoff=zerodha.token_store.validity_cutoff,
    )
    cached = store.load_usable(
        api_key=credentials.api_key,
        now=datetime.now(get_timezone(config.timezone_name)),
        min_remaining_minutes=zerodha.token_store.min_remaining_minutes,
    )
    using_cache = cached is not None and not args.force_login
    capture_command = " ".join(["apps.anjaneeya_capture.run", "--mode", "capture", *_capture_argv(args)])

    if args.dry_run:
        print("ANJANEYA login + capture (dry run - nothing is opened, fetched or started)")
        print(f"  env file     : {env_file.describe()}")
        print(f"  api key      : {_mask(credentials.api_key)}")
        print(f"  cached token : {'usable' if using_cache else 'none or stale'}")
        print(f"  login        : {'cached token' if using_cache else 'paste' if args.paste else 'browser'}")
        print(f"  redirect url : {redirect_url}")
        print(f"  capture      : {'not started (--no-start)' if args.no_start else capture_command}")
        return EXIT_OK

    request_token: str | None = None
    login_label = "cached token"
    if not using_cache:
        if args.skip_login:
            print(
                "ANJANEYA login: no usable cached token and --skip-login was given. Drop "
                "--skip-login to log in (the login is what mints a new token).",
                file=sys.stderr,
            )
            return EXIT_LOGIN_FAILED
        try:
            request_token, login_label = _obtain_token(args, credentials.api_key, redirect_url)
        except AuthenticationError as exc:
            print(f"ANJANEYA login failed: {exc}", file=sys.stderr)
            return EXIT_LOGIN_FAILED
        except KeyboardInterrupt:
            print("\nANJANEYA login cancelled; nothing was cached.", file=sys.stderr)
            return EXIT_CANCELLED

    if not args.as_json:
        print(f"ANJANEYA login: {login_label} (env file: {env_file.describe()})")
    exit_code, out, err = run_login(_login_argv(args, request_token))
    if err:
        print(err, end="", file=sys.stderr)
    if exit_code != EXIT_OK:
        # daily_login explains the reason; echo its report and stop here, so a
        # scheduler never mistakes "capture started with a dead token" for success.
        print(out, end="", file=sys.stderr)
        print(
            f"ANJANEYA login failed (exit {exit_code}); the capture session was NOT started.",
            file=sys.stderr,
        )
        return EXIT_LOGIN_FAILED

    if args.as_json:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "login": login_label,
                    "env_file": str(env_file.path),
                    "env_file_loaded": env_file.loaded,
                    "redirect_url": redirect_url,
                    "token_store": str(config.token_store_path),
                    "capture_started": not args.no_start,
                    "capture_command": capture_command,
                },
                indent=2,
            )
        )
    elif out:
        print(out, end="")

    if args.no_start:
        return EXIT_OK
    return start_capture(_capture_argv(args))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
