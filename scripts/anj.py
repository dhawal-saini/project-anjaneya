"""ANJANEYA operator facade - the whole system behind three verbs.

The repository splits the workflow across three applications (capture, engine,
ai), a dozen modes and a fistful of wrapper scripts. That split is right for the
code and wrong for the operator, who only ever has three jobs:

    history   once, and occasionally after: refresh the instrument master,
              backfill the NIFTY index history, backfill the option history
    day       every trading day: calendar guard -> universe check -> login ->
              the live capture session (blocks until the close) -> the
              post-close chain (ticks -> candles -> greeks -> features ->
              labels -> validate -> dataset) -> the suggestion
    learn     end of day: train on everything captured so far, replay the
              backtest, and write the report / explain / importance / charts /
              accumulation artefacts

This is a facade over the scripts that already exist, not a replacement: every
step is delegated to the script or app that already owns it, so each behaviour
has exactly one implementation and there is no second thing to keep in sync.
Everything in ``scripts/`` remains available and unchanged.

Usage::

    uv run python scripts/anj.py history --years 2
    uv run python scripts/anj.py day                    # a normal trading day
    uv run python scripts/anj.py day --no-capture       # math only (holiday)
    uv run python scripts/anj.py learn --force

Exit codes (the vocabulary every app already shares):

    0  every step that ran succeeded - a holiday's clean NO_SESSION is a 0
    1  a step failed
    2  a configuration / credential problem
    4  ``train`` refused because the spec 4.3 accumulation gate is not open yet

Only ``history`` and the login touch the network; the capture session and the
pipeline read and write the local DuckDB file and the ``data/`` tree. Secrets
are never printed, and the ``request_token`` never appears in any output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/anj.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.app_bootstrap import (  # noqa: E402
    EXIT_CONFIG_ERROR,
    EXIT_GATE_REFUSED,
    EXIT_OK,
)
from core.env import load_env_file  # noqa: E402

#: The child entry points. The facade shells out to the real owners, exactly as
#: the PowerShell wrappers in ``scripts/`` do, so nothing is re-implemented here.
CAPTURE_APP = "apps.anjaneeya_capture.run"
AI_APP = "apps.anjaneeya_ai.run"
MORNING = "scripts/run_morning.py"
PIPELINE = "scripts/run_daily_pipeline.py"

#: How much history ``history`` fetches when no ``--years`` is given. Two years
#: of daily index closes is the window the regime / moving-average work needs;
#: option history is bounded by what Kite will still serve (about 60 days of
#: 1-minute bars), so a quarter is both safe and cheap.
DEFAULT_INDEX_YEARS = 2.0
DEFAULT_OPTION_YEARS = 0.25

#: Steps ``day`` leaves to ``learn``. A model trained before the close is a
#: model trained on an incomplete day, so the daily chain stops at the dataset
#: and training is the evening's own, separate, deliberate command.
DAY_SKIP_STEPS = "train,backtest"


def _number(value: float) -> str:
    """Render a year count as ``2`` rather than ``2.0``, but keep ``0.25``."""
    as_float = float(value)
    return str(int(as_float)) if as_float.is_integer() else str(as_float)


def _banner(text: str) -> None:
    """Print a step header so a long run stays readable."""
    print(f"\n=== {text} ===", flush=True)


def _child(argv: Sequence[str]) -> int:
    """Run a child command under this interpreter and return its exit code."""
    print(f"    $ {' '.join(argv)}", flush=True)
    return subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, check=False).returncode


def _fail(step: str, code: int) -> int:
    """Report a failed step on stderr and propagate its exit code unchanged."""
    print(f"ANJANEYA: step '{step}' failed with exit {code}; stopping.", file=sys.stderr)
    return code


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options every verb shares; each is forwarded to the child that owns it."""
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory holding the YAML configuration. (default: ./config)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override settings.data.root for this run. (default: None)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override the DuckDB file for this run. (default: None)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan every child step without writing data or calling the network.",
    )


def _common(args: argparse.Namespace) -> list[str]:
    """The forwarded form of the shared options."""
    argv: list[str] = []
    if args.config_dir:
        argv += ["--config-dir", args.config_dir]
    if args.data_root:
        argv += ["--data-root", args.data_root]
    if args.db_path:
        argv += ["--db-path", args.db_path]
    if args.dry_run:
        argv += ["--dry-run"]
    return argv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the facade's arguments - one of three verbs."""
    parser = argparse.ArgumentParser(
        prog="python scripts/anj.py",
        description="ANJANEYA in three commands: history, day, learn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run python scripts/anj.py history --years 2\n"
            "  uv run python scripts/anj.py day\n"
            "  uv run python scripts/anj.py day --no-capture\n"
            "  uv run python scripts/anj.py learn --force\n"
        ),
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{history,day,learn}")

    history = verbs.add_parser(
        "history",
        help="Once: instrument master + NIFTY index history + option history.",
        description=(
            "Refresh the instrument master, backfill the NIFTY index history and "
            "backfill the option contracts Kite still lists. Run this once when the "
            "database is empty, and again whenever the history looks short."
        ),
    )
    _add_common(history)
    history.add_argument(
        "--years",
        type=float,
        default=DEFAULT_INDEX_YEARS,
        help=f"Years of NIFTY index history to fetch. (default: {_number(DEFAULT_INDEX_YEARS)})",
    )
    history.add_argument(
        "--option-years",
        type=float,
        default=DEFAULT_OPTION_YEARS,
        help=(
            "Years of option history to attempt. Kite only serves about 60 days of "
            f"1-minute bars, so a quarter is enough. (default: {DEFAULT_OPTION_YEARS})"
        ),
    )
    history.add_argument(
        "--skip-instruments",
        action="store_true",
        help="Reuse the cached instrument master instead of re-downloading it.",
    )
    history.add_argument(
        "--skip-options",
        action="store_true",
        help="Skip the option backfill (index history only).",
    )

    day = verbs.add_parser(
        "day",
        help="Every trading day: capture the session, then the post-close math.",
        description=(
            "The daily loop. By default it runs the calendar guard, the universe "
            "check, the login and the live capture session (which blocks until the "
            "close), then the post-close chain up to the dataset, then the "
            "suggestion. Training is NOT part of this verb - run `learn` after the "
            "close. On a non-trading day the guard prints NO_SESSION and exits 0."
        ),
    )
    _add_common(day)
    day.add_argument(
        "--no-capture",
        action="store_true",
        help="Skip the live session entirely and run only the post-close math.",
    )
    day.add_argument(
        "--skip-postclose",
        action="store_true",
        help="Stop after the capture session (no engine/AI steps).",
    )
    day.add_argument(
        "--date",
        default=None,
        help="Trading day the post-close chain builds. (default: newest day with ticks)",
    )
    day.add_argument(
        "--smoke",
        action="store_true",
        help="Forward --smoke to the capture session (capture briefly, then stop).",
    )
    day.add_argument(
        "--until",
        default=None,
        help="Forward --until HH:MM to the capture session.",
    )

    learn = verbs.add_parser(
        "learn",
        help="End of day: train on everything, backtest and report.",
        description=(
            "The evening's deliberate run: dataset -> train -> backtest -> report -> "
            "explain -> importance -> charts -> accumulation. Exit 4 means train "
            "refused because the accumulation gate is not open yet; pass --force to "
            "train anyway on the history that exists."
        ),
    )
    _add_common(learn)
    learn.add_argument(
        "--date",
        default=None,
        help="Trading day the engine steps build. (default: newest day with ticks)",
    )
    learn.add_argument(
        "--force",
        action="store_true",
        help="Train even when the spec 4.3 accumulation gate is not open.",
    )
    return parser.parse_args(argv)


def cmd_history(args: argparse.Namespace) -> int:
    """Refresh the instrument master, then backfill index and option history."""
    common = _common(args)
    if not args.skip_instruments:
        _banner("instruments: refresh the Kite instrument master")
        code = _child(["-m", CAPTURE_APP, "--mode", "instruments", "--force-refresh", *common])
        if code != EXIT_OK:
            return _fail("instruments", code)

    _banner(f"index history: {_number(args.years)} year(s) of NIFTY daily bars")
    code = _child(["-m", CAPTURE_APP, "--mode", "backfill", "--years", _number(args.years), *common])
    if code != EXIT_OK:
        return _fail("backfill", code)

    if not args.skip_options:
        _banner(f"option history: {_number(args.option_years)} year(s) of listed contracts")
        code = _child(
            [
                "-m",
                CAPTURE_APP,
                "--mode",
                "backfill-options",
                "--years",
                _number(args.option_years),
                *common,
            ]
        )
        if code != EXIT_OK:
            return _fail("backfill-options", code)

    print("\nANJANEYA history: done.")
    return EXIT_OK


def cmd_day(args: argparse.Namespace) -> int:
    """Capture the session (unless skipped), then run the post-close math."""
    common = _common(args)
    if args.no_capture:
        print(
            "ANJANEYA day: --no-capture - the live session is skipped and only the "
            "post-close math runs (the right choice on a holiday or after the fact)."
        )
    else:
        _banner("morning: calendar guard -> universe -> login -> capture session")
        forwarding: list[str] = []
        if args.smoke:
            forwarding.append("--smoke")
        if args.until:
            forwarding += ["--until", args.until]
        argv = [MORNING, *common]
        if forwarding:
            argv += ["--", *forwarding]
        code = _child(argv)
        if code != EXIT_OK:
            return _fail("morning session", code)

    if args.skip_postclose:
        print("\nANJANEYA day: --skip-postclose - stopping after the session.")
        return EXIT_OK

    _banner("post-close: ticks -> candles -> greeks -> features -> labels -> dataset")
    argv = [PIPELINE, "--skip", DAY_SKIP_STEPS, *common]
    if args.date:
        argv += ["--date", args.date]
    code = _child(argv)
    if code != EXIT_OK:
        return _fail("post-close chain", code)

    _banner("suggestion: what the model says about the next session")
    code = _child(["-m", AI_APP, "--mode", "decide", *common])
    if code != EXIT_OK:
        print(
            f"ANJANEYA day: the suggestion step exited {code}. Until a model exists "
            "(`anj.py learn`) there is nothing to decide with; the math above is "
            "unaffected.",
            file=sys.stderr,
        )

    print("\nANJANEYA day: done.")
    return EXIT_OK


def cmd_learn(args: argparse.Namespace) -> int:
    """Train on everything captured so far, backtest it and write the reports."""
    common = _common(args)
    _banner("learn: dataset -> train -> backtest -> report -> explain -> importance")
    argv = [PIPELINE, "--weekly", *common]
    if args.date:
        argv += ["--date", args.date]
    if args.force:
        argv += ["--force"]
    code = _child(argv)
    if code == EXIT_GATE_REFUSED:
        print(
            "\nANJANEYA learn: train refused - the spec 4.3 accumulation gate is not "
            "open yet (it counts days with captured ticks). Pass --force to train on "
            "the history that does exist, or let the capture accumulate the days it "
            "wants.",
            file=sys.stderr,
        )
        return code
    if code != EXIT_OK:
        return _fail("learn", code)
    print("\nANJANEYA learn: done.")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: load ``.env`` once, then dispatch the verb."""
    args = parse_args(argv)
    load_env_file(config_dir=Path(args.config_dir) if args.config_dir else None)
    if args.verb == "history":
        return cmd_history(args)
    if args.verb == "day":
        return cmd_day(args)
    if args.verb == "learn":
        return cmd_learn(args)
    print(f"ANJANEYA: unknown verb {args.verb!r}", file=sys.stderr)
    return EXIT_CONFIG_ERROR


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
