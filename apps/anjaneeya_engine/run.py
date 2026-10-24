"""ANJANEYA-ENGINE - indicator, feature and label engineering application.

Responsibilities (master spec sections 7-20)

* build multi-timeframe candles from stored ticks (``--mode build-candles``),
* price them: IV and greeks from a documented Black-Scholes model
  (``--mode enrich-greeks``, spec 16),
* report the configured option universe - ATM anchor, strike ladder, expiries -
  against what the database actually holds (``--mode universe``, spec 17),
* evaluate the NIFTY trend-score combinations,
* compute option trend / candle behaviour / levels / structure / S-R engines,
* emit the unified feature row (``--mode build-features``, spec 18),
* validate those rows (``--mode validate``, spec 19),
* generate labels in a physically separate pass (``--mode build-labels``, spec 20).

``--mode build-candles`` reads the session's ticks out of DuckDB, builds 1m bars and
merges 3m/5m/15m from them, replaces the affected ``(session, timeframe, contract)``
scopes, records a ``data_coverage`` verdict per stream and mirrors the candles to
Parquet.

``--mode enrich-greeks`` solves each option bar's implied volatility from the configured
price field against the NIFTY level at the bar's **close instant**, computes
``delta``/``gamma``/``theta``/``vega`` from it, and ``UPDATE``s exactly those columns (and
``days_to_expiry``/``distance_from_atm``) onto the stored bars - it never rebuilds an
OHLCV row. A bar whose price has no IV under the configured bracket stays ``NULL``, and
the run prints how many did.

``--mode build-features`` reads those candles - the session plus the lookback window
the long NIFTY averages need - and writes one row per ``(timestamp, contract)``: the
NIFTY scores and regime, the option trend block, candle/level/structure/S-R features,
the option context and spec 16's relative behaviour. It replaces a whole *contract-day*
(the schema's key has no timeframe in it) and mirrors the rows to Parquet.

``--mode build-labels`` reads the same candles in a *separate* pass and writes one row
per ``(timestamp, contract, horizon)``: the forward return, the excursions, the trend
change and one class (``UP_CONTINUATION`` / ``DOWN_CONTINUATION`` / ``RANGE`` /
``REVERSAL``) whose threshold comes from ``config/model.yaml``. It never reads a feature
row and never writes one (spec 20's separation is structural, not a convention).

``--mode validate`` re-reads the stored feature rows and runs spec 19's checks over
them - including a no-look-ahead probe that recomputes a sample of bars from a window
truncated at that bar - then writes a JSON report per day.

No mode touches the network, and none of them ever writes to ``ticks``: derived data
can always be rebuilt, raw data cannot be re-created.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import partial
from typing import Any

from apps.anjaneeya_engine.candles import (
    CandleBuildError,
    build_1m_candles,
    build_timeframes,
    expected_bar_openings,
    missing_bar_summary,
)
from apps.anjaneeya_engine.enrichment import enrich_frame
from apps.anjaneeya_engine.storage_writer import (
    CandleWriter,
    CandleWriteStats,
    FeatureWriter,
    FeatureWriteStats,
    GreeksWriter,
    GreeksWriteStats,
    LabelWriter,
)
from core.app_bootstrap import (
    APP_ENGINE,
    EXIT_CONFIG_ERROR,
    EXIT_FAILED,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    AppContext,
    bootstrap_app,
    finalize_run,
    parse_args,
    render_status,
    status_report,
)
from core.config_loader import ConfigError
from core.constants import SUPPORTED_TIMEFRAMES, timeframe_to_minutes
from core.features import (
    FeatureBuildError,
    build_feature_frame,
    feature_rows,
    merge_reports,
    required_lookback_bars,
    validate_feature_frame,
)
from core.labels import LabelPolicy, build_label_frame, label_rows
from core.options import (
    GreeksModel,
    OptionSelection,
    build_selection,
    template_pattern,
)
from core.schemas import CoverageRow, QualityFlag
from core.storage import (
    SchemaVersionError,
    StreamStats,
    build_coverage_row,
    candle_counts,
    connect,
    feature_counts,
    first_candle_timestamp,
    quality_counts,
    read_candles,
    read_features,
    read_index_history,
    read_instruments,
    read_option_instruments,
    read_ticks,
    symbols_with_candles,
    symbols_with_ticks,
    trading_days_with_candles,
    trading_days_with_ticks,
)

DESCRIPTION = (
    "ANJANEYA-ENGINE: turn stored ticks into candles, indicators, features and labels "
    "(pure computation - no live trading, ever)."
)

#: Every engine mode is implemented as of Stage 3.2e, so this map is empty: it stays as
#: the one place a *future* mode is declared before it is written, and it is what
#: ``--dry-run`` / ``_run_planned_mode`` read to explain the work that is still ahead.
MODE_PLAN: dict[str, str] = {}

def _configure_engine_parser(parser: argparse.ArgumentParser) -> None:
    """Add the engine-only flags (which day, which contracts, which timeframes)."""
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Trading day to build (IST). Default: the most recent day with stored ticks.",
    )
    parser.add_argument(
        "--all",
        dest="all_days",
        action="store_true",
        help="Build every day that has stored ticks, oldest first.",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        metavar="TRADINGSYMBOL",
        help="Restrict the build to one contract (repeatable). Default: every contract "
        "with ticks in the session for the tick-driven modes, every contract with candles "
        "for build-labels.",
    )
    parser.add_argument(
        "--timeframes",
        default=None,
        metavar="1m,5m",
        help="Comma-separated subset of settings.market.timeframes. build-candles and "
        "enrich-greeks build/enrich every one of them; build-features, build-labels and "
        "validate need exactly one (the features and labels tables are keyed by timestamp "
        "+ contract, with no timeframe in them). "
        "Default: all configured timeframes for candles and greeks, the first configured one "
        "for features and labels.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        metavar="N",
        help="universe: the reference price used to place the ATM strike (the price of the "
        "instrument atm.reference names). Default: the last stored close of that "
        "instrument for the day.",
    )
    parser.add_argument(
        "--probes",
        type=int,
        default=3,
        metavar="N",
        help="validate: how many bars to recompute from a truncated window for the no-look-ahead "
        "check (0 skips it). Recomputing a bar costs about as much as building it.",
    )





#: A candle stream is keyed in the coverage ledger by its timeframe label
#: (``1m``, ``3m``, ...) - unlike ticks, it has a *known* expected row count.


def _resolve_timeframes(context: AppContext, value: str | None) -> list[str]:
    """Resolve ``--timeframes`` against the configured set (never beyond it)."""
    configured = list(context.config.settings.market.timeframes)
    if value is None:
        return configured
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested:
        raise ConfigError("--timeframes was empty; omit it to build every configured timeframe")
    unknown = [name for name in requested if name not in SUPPORTED_TIMEFRAMES]
    if unknown:
        raise ConfigError(
            f"Unsupported timeframe(s) {unknown}; supported: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    disabled = [name for name in requested if name not in configured]
    if disabled:
        raise ConfigError(
            f"Timeframe(s) {disabled} are not enabled in settings.market.timeframes "
            f"({', '.join(configured)}). Enable them there rather than building a timeframe "
            "the rest of the platform does not know about."
        )
    return [name for name in configured if name in requested]


def _day_from_args(args: argparse.Namespace) -> date | None:
    """``--date`` as a :class:`datetime.date`, or ``None`` when it was not given."""
    value = getattr(args, "date", None)
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ConfigError(
            f"--date {value!r} is not a valid ISO date (expected YYYY-MM-DD, IST)"
        ) from None


def _resolve_days(
    context: AppContext,
    connection: Any,
    args: argparse.Namespace,
    *,
    days_source: Any = None,
) -> list[date]:
    """Which trading days this run covers.

    With no flag the engine takes the most recent day that *has data*, which is what an
    operator wants after a capture session ("build what I just captured"). ``days_source``
    is the discovery query that defines "has data" - ticks for the build and feature
    modes, candles for the greeks enrichment, which works on stored bars and therefore
    does not need ticks to still be there.
    """
    explicit = _day_from_args(args)
    if explicit is not None:
        return [explicit]
    discover = days_source or trading_days_with_ticks
    days = discover(connection, timezone_name=context.config.timezone_name)
    if getattr(args, "all_days", False):
        return days
    return days[-1:]


def _coverage_row(
    context: AppContext,
    *,
    day: date,
    tradingsymbol: str,
    timeframe: str,
    bars: Sequence[Any],
    start: datetime,
    end: datetime,
) -> Any:
    """One ``data_coverage`` verdict for a contract's candle stream.

    ``expected_rows`` is the bar count a complete session would produce, so
    ``coverage_ratio`` and the gap count mean the same thing they do for ticks -
    this time with an exact denominator.
    """
    summary = missing_bar_summary(
        bars, session_start=start, session_end=end, minutes=timeframe_to_minutes(timeframe)
    )
    stats = StreamStats(
        tradingsymbol=tradingsymbol,
        timeframe=timeframe,
        rows=len(bars),
        first_timestamp=bars[0].timestamp if bars else None,
        last_timestamp=bars[-1].timestamp if bars else None,
        gap_count=summary.gap_count,
        largest_gap_seconds=summary.largest_gap_seconds,
        expected_rows=summary.expected_bars,
    )
    return build_coverage_row(
        trading_date=day, stats=stats, run_id=context.run_context.run_id
    )


@dataclass(slots=True)
class DayResult:
    """What one trading day's build did (the CLI summary is built from these)."""

    day: date
    status: str
    symbols: int = 0
    bars: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    volume_resets: int = 0
    empty_symbols: int = 0
    parquet: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "day": self.day.isoformat(),
            "status": self.status,
            "symbols": self.symbols,
            "bars": self.bars,
            "coverage": self.coverage,
            "excluded_ticks": self.excluded,
            "volume_resets": self.volume_resets,
            "contracts_without_ok_ticks": self.empty_symbols,
            "parquet": self.parquet,
            "failures": self.failures,
        }


def _build_day(
    context: AppContext,
    *,
    connection: Any,
    writer: CandleWriter,
    day: date,
    timeframes: Sequence[str],
    symbols: Sequence[str] | None,
) -> DayResult:
    """Build and store every timeframe of every contract for one trading day."""
    result = DayResult(day=day, status="SUCCESS")
    if not context.calendar.is_trading_day(day):
        reason = "weekend" if context.calendar.is_weekend(day) else "holiday"
        context.logger.info(f"{day.isoformat()} is not a trading day ({reason}); nothing to build")
        result.status = "NO_SESSION"
        return result

    start = context.calendar.session_start(day)
    end = context.calendar.session_end(day)
    requested = list(symbols) if symbols else None
    contracts = requested or symbols_with_ticks(connection, start=start, end=end)
    result.symbols = len(contracts)
    if not contracts:
        context.logger.warning(f"no ticks stored for {day.isoformat()}; nothing to build")
        result.status = "NO_DATA"
        return result
    result.excluded = {
        flag: count
        for flag, count in quality_counts(connection, start=start, end=end, symbols=requested).items()
        if flag != QualityFlag.OK.value
    }

    coverage_rows: list[Any] = []
    for symbol in contracts:
        frame = read_ticks(connection, start=start, end=end, symbols=[symbol])
        if frame.empty:
            # Every tick of this contract was flagged, or the contract only
            # appeared in another session: no bars, and an explicit MISSING_TICK
            # verdict per timeframe rather than silence.
            result.empty_symbols += 1
            coverage_rows.extend(
                _coverage_row(
                    context,
                    day=day,
                    tradingsymbol=symbol,
                    timeframe=timeframe,
                    bars=[],
                    start=start,
                    end=end,
                )
                for timeframe in timeframes
            )
            continue

        try:
            bars_1m, stats = build_1m_candles(
                frame,
                tradingsymbol=symbol,
                run_id=context.run_context.run_id,
                timezone_name=context.config.timezone_name,
            )
        except CandleBuildError as exc:
            # One unusable contract must not cost us the other two hundred. Its
            # *stored* candles are deliberately left untouched: a stale bar that
            # is visible in this summary (and exits non-zero) is safer than
            # deleting data we cannot rebuild right now.
            result.failures.append(f"{symbol}: {exc}")
            context.logger.error(f"{day.isoformat()} {symbol}: {exc}")
            continue

        result.volume_resets += stats.volume_resets
        for timeframe, rows in build_timeframes(bars_1m, timeframes=timeframes).items():
            written = writer.replace_candles(
                start=start,
                end=end,
                timeframe=timeframe,
                tradingsymbol=symbol,
                rows=rows,
            )
            result.bars[timeframe] = result.bars.get(timeframe, 0) + written
            coverage_rows.append(
                _coverage_row(
                    context,
                    day=day,
                    tradingsymbol=symbol,
                    timeframe=timeframe,
                    bars=rows,
                    start=start,
                    end=end,
                )
            )

    if coverage_rows:
        writer.write_coverage(coverage_rows)
    for row in coverage_rows:
        flag = row.quality_flag.value
        result.coverage[flag] = result.coverage.get(flag, 0) + 1
    if writer.parquet_enabled:
        result.parquet = [
            str(path)
            for path in writer.mirror_day(day=day, start=start, end=end, timeframes=timeframes)
        ]
    if result.failures:
        result.status = "PARTIAL"
    return result


def _plan_lines(
    context: AppContext,
    *,
    connection: Any,
    days: Sequence[date],
    timeframes: Sequence[str],
    symbols: Sequence[str] | None,
) -> list[str]:
    """Dry-run plan: what *would* be built, and how big a complete day is."""
    lines = [f"ANJANEYA-ENGINE dry-run: build-candles -> {len(days)} day(s)"]
    lines.append(f"  timeframes    : {', '.join(timeframes)}")
    lines.append(
        "  contracts     : "
        + (", ".join(symbols) if symbols else "every contract with ticks in the session")
    )
    for day in days:
        if not context.calendar.is_trading_day(day):
            lines.append(f"  {day.isoformat()}   : no session (weekend or holiday)")
            continue
        start = context.calendar.session_start(day)
        end = context.calendar.session_end(day)
        count = (
            len(symbols) if symbols else len(symbols_with_ticks(connection, start=start, end=end))
        )
        complete = ", ".join(
            f"{name}={len(expected_bar_openings(session_start=start, session_end=end, minutes=timeframe_to_minutes(name)))}"
            for name in timeframes
        )
        lines.append(
            f"  {day.isoformat()}   : {count} contract(s); a complete session = {complete} bar(s)"
        )
    lines.append("  dry-run writes no candles, no coverage rows and no Parquet")
    return lines


def _render_build_summary(
    *,
    results: Sequence[DayResult],
    timeframes: Sequence[str],
    stats: CandleWriteStats,
    as_json: bool,
) -> str:
    """Render the run summary as JSON or as an operator-friendly block."""
    failures = [message for result in results for message in result.failures]
    if as_json:
        return json.dumps(
            {
                "mode": "build-candles",
                "timeframes": list(timeframes),
                "days": [result.to_dict() for result in results],
                "candles_written": stats.candles,
                "candles_replaced": stats.deleted,
                "coverage_rows": stats.coverage,
                "parquet_files": stats.parquet_files,
                "errors": stats.failures,
                "failures": failures,
            },
            indent=2,
            default=str,
        )

    lines = [f"ANJANEYA-ENGINE build-candles - {len(results)} day(s)"]
    lines.append(f"  timeframes      : {', '.join(timeframes)}")
    for result in results:
        bars = ", ".join(f"{name}={count}" for name, count in result.bars.items()) or "no bars"
        lines.append(
            f"  {result.day.isoformat()} [{result.status}] {result.symbols} contract(s); {bars}"
        )
        if result.coverage:
            lines.append(
                "      coverage    : "
                + ", ".join(f"{name}={count}" for name, count in sorted(result.coverage.items()))
            )
        if result.excluded:
            lines.append(
                "      excluded    : "
                + ", ".join(f"{name}={count}" for name, count in sorted(result.excluded.items()))
                + " tick(s)"
            )
        if result.empty_symbols:
            lines.append(
                f"      empty       : {result.empty_symbols} contract(s) had no usable tick"
            )
    resets = sum(result.volume_resets for result in results)
    if resets:
        lines.append(f"  volume resets   : {resets} (cumulative volume counter restarted)")
    lines.append(f"  candles written : {stats.candles} ({stats.deleted} replaced)")
    lines.append(f"  coverage rows   : {stats.coverage}")
    lines.append(f"  parquet files   : {stats.parquet_files}")
    lines.append(f"  failures        : {len(failures)}")
    for message in failures:
        lines.append(f"      {message}")
    return "\n".join(lines)


def _run_build_candles(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode build-candles``: ticks -> 1m/3m/5m/15m bars, coverage, mirror."""
    timeframes = _resolve_timeframes(context, getattr(args, "timeframes", None))
    symbols = getattr(args, "symbols", None)
    if not context.db_path.exists():
        print(
            f"ANJANEYA-ENGINE: no DuckDB database at {context.db_path}; there is nothing to "
            "build from yet.\n"
            "  next step: capture a session first "
            "(python -m apps.anjaneeya_capture.run --mode capture)",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    status, exit_code = "SUCCESS", EXIT_OK
    # One connection for the whole run: DuckDB allows a single read-write process
    # per file, and the writer borrows this one instead of opening its own.
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        days = _resolve_days(context, connection, args)
        if not days:
            print(
                "ANJANEYA-ENGINE: no stored ticks found - capture a session first.",
                file=sys.stderr,
            )
            status = "NO_DATA"
        elif context.dry_run:
            print(
                "\n".join(
                    _plan_lines(
                        context,
                        connection=connection,
                        days=days,
                        timeframes=timeframes,
                        symbols=symbols,
                    )
                )
            )
        else:
            writer = CandleWriter(
                connection=connection,
                parquet_root=context.config.data_dir("candles"),
                run_id=context.run_context.run_id,
                mirror_parquet=context.config.settings.engine.mirror_parquet,
                parquet_compression=context.config.settings.data.parquet_compression,
                logger=context.logger,
            )
            results = [
                _build_day(
                    context,
                    connection=connection,
                    writer=writer,
                    day=day,
                    timeframes=timeframes,
                    symbols=symbols,
                )
                for day in days
            ]
            print(
                _render_build_summary(
                    results=results,
                    timeframes=timeframes,
                    stats=writer.stats,
                    as_json=args.as_json,
                )
            )
            if any(result.failures for result in results):
                status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"build-candles failed: {type(exc).__name__}: {exc}")
        print(
            f"ANJANEYA-ENGINE build-candles failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)


def _resolve_feature_timeframe(context: AppContext, value: str | None) -> str:
    """The one timeframe a feature build covers.

    ``features`` is keyed ``(timestamp, tradingsymbol)``: two timeframes of the same
    contract share every bar's open time, so they cannot both live in that table. A
    build therefore names exactly one timeframe - the first configured one by default -
    and ``--timeframes 1m,5m`` is a configuration error rather than a silent choice.
    """
    if value is None:
        configured = list(context.config.settings.market.timeframes)
        return configured[0]
    requested = _resolve_timeframes(context, value)
    if len(requested) != 1:
        raise ConfigError(
            "the features table is keyed (timestamp, tradingsymbol), so a build covers exactly "
            f"one timeframe; got {requested}. Pass --timeframes with a single label."
        )
    return requested[0]


def _feature_window(
    context: AppContext, *, day: date, timeframe: str
) -> tuple[datetime, datetime, datetime]:
    """``(window_start, session_start, session_end)`` for one day's feature build.

    The window starts ``required_lookback_bars + one session`` of bars before the
    session: the first term is what the long NIFTY averages need (2062 1m bars on the
    shipped config), the second guarantees the level state machine and the session
    levels see the previous session rather than starting blind.
    """
    session_start = context.calendar.session_start(day)
    session_end = context.calendar.session_end(day)
    minutes = timeframe_to_minutes(timeframe)
    session_bars = len(
        expected_bar_openings(session_start=session_start, session_end=session_end, minutes=minutes)
    )
    lookback = required_lookback_bars(context.config.indicators) + session_bars
    return session_start - timedelta(minutes=lookback * minutes), session_start, session_end


def _underlying_symbol(context: AppContext) -> str:
    """The NIFTY index ``tradingsymbol`` the ``index_history`` rows carry."""
    return context.config.option_universe.reference_instruments.nifty_spot.tradingsymbol or "NIFTY"


def _index_frame(
    context: AppContext, connection: Any, *, start: datetime, end: datetime, timeframe: str
) -> Any:
    """The NIFTY index window, at the *finest resolution actually stored*.

    ``index_history`` holds daily backfill plus a minute-level top-up, and an intraday
    build at 15m usually has no 15m index rows to read. Rather than silently mixing
    resolutions (or pretending 1m rows *are* 15m rows - which is what an earlier version
    did, and which left a coarse build reading the NIFTY level from the *start* of each
    bucket while its close instant was computed from the end), the finest intraday series
    is returned with its **own** timeframe label and the builder's bucket rule
    (:func:`core.features.feature_builder.aggregate_index_frame`) merges it. A missing
    index series is reported, never invented.
    """
    symbol = _underlying_symbol(context)
    frame = read_index_history(connection, start=start, end=end, symbols=[symbol])
    if frame.empty:
        return frame
    if timeframe not in {str(label) for label in frame["timeframe"].dropna().unique()}:
        preferred = frame[frame["timeframe"] == "1m"]
        if preferred.empty:
            context.logger.warning(
                f"index_history holds {sorted(set(frame['timeframe'].dropna()))} for {symbol} but no "
                f"1m rows, so the {timeframe} NIFTY block cannot be built from it; nifty_* stays NULL"
            )
            return frame.iloc[0:0]
        return preferred
    return frame[frame["timeframe"] == timeframe]


@dataclass(slots=True)
class FeatureDayResult:
    """What one trading day's feature build did (the CLI summary is built from these)."""

    day: date
    timeframe: str
    status: str
    contracts: int = 0
    rows: int = 0
    expected_rows: int = 0
    missing_rows: int = 0
    contracts_without_candles: int = 0
    index_rows: int = 0
    coverage: dict[str, int] = field(default_factory=dict)
    parquet: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "day": self.day.isoformat(),
            "timeframe": self.timeframe,
            "status": self.status,
            "contracts": self.contracts,
            "rows": self.rows,
            "expected_rows": self.expected_rows,
            "missing_rows": self.missing_rows,
            "contracts_without_candles": self.contracts_without_candles,
            "index_rows": self.index_rows,
            "coverage": self.coverage,
            "parquet": self.parquet,
            "failures": self.failures,
        }


def _feature_coverage_row(
    context: AppContext,
    *,
    day: date,
    timeframe: str,
    underlying: str,
    result: FeatureDayResult,
    first_timestamp: datetime | None,
    last_timestamp: datetime | None,
    gap_count: int,
    largest_gap_seconds: float,
) -> CoverageRow:
    """One ``data_coverage`` verdict for a day's feature stream.

    The stream is keyed by the **underlying**, not by a contract: ``data_coverage`` is
    keyed ``(trading_date, tradingsymbol, timeframe)`` and ``build-candles`` already
    owns the per-contract rows for the same day and timeframe. Writing contract-level
    feature coverage would silently overwrite the candle verdict - so the feature stream
    reports itself once per day, with ``expected_rows`` summed over the contracts that
    were built (the note says how many).
    """
    stats = StreamStats(
        tradingsymbol=underlying,
        timeframe=timeframe,
        rows=result.rows,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        gap_count=gap_count,
        largest_gap_seconds=largest_gap_seconds,
        expected_rows=result.expected_rows,
    )
    note = f"features over {result.contracts} contract(s)"
    if result.contracts_without_candles:
        note += f"; {result.contracts_without_candles} had no candles"
    if not result.index_rows:
        note += "; NIFTY index history missing, nifty_* is NULL"
    row = build_coverage_row(trading_date=day, stats=stats, run_id=context.run_context.run_id)
    return row.model_copy(update={"note": note})


def _build_feature_day(
    context: AppContext,
    *,
    connection: Any,
    writer: FeatureWriter,
    day: date,
    timeframe: str,
    symbols: Sequence[str] | None,
) -> FeatureDayResult:
    """Build, store and mirror one trading day's feature rows (one contract at a time)."""
    result = FeatureDayResult(day=day, timeframe=timeframe, status="SUCCESS")
    if not context.calendar.is_trading_day(day):
        reason = "weekend" if context.calendar.is_weekend(day) else "holiday"
        context.logger.info(f"{day.isoformat()} is not a trading day ({reason}); nothing to build")
        result.status = "NO_SESSION"
        return result

    window_start, session_start, session_end = _feature_window(context, day=day, timeframe=timeframe)
    minutes = timeframe_to_minutes(timeframe)
    session_bars = len(
        expected_bar_openings(session_start=session_start, session_end=session_end, minutes=minutes)
    )
    contracts = (
        list(symbols)
        if symbols
        else symbols_with_ticks(connection, start=session_start, end=session_end)
    )
    result.contracts = len(contracts)
    if not contracts:
        context.logger.warning(f"no ticks stored for {day.isoformat()}; nothing to build")
        result.status = "NO_DATA"
        return result

    index_frame = _index_frame(
        context, connection, start=window_start, end=session_end, timeframe=timeframe
    )
    result.index_rows = int(len(index_frame))
    if index_frame.empty:
        context.logger.warning(
            f"no NIFTY index history for {day.isoformat()} at {timeframe}; every nifty_* column "
            "will be NULL for this day (run capture's index-history mode to fill it)"
        )



    underlying = context.config.option_universe.underlying
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    gap_count = 0
    largest_gap = 0.0
    for position, symbol in enumerate(contracts, start=1):
        # Start at the contract's own first bar when that is later than the generic
        # lookback boundary: the option engine's insufficient-history flag counts bars
        # since the start of the frame, so a brand-new weekly contract must be measured
        # from its own birth, not from a window that predates it.
        first_bar = first_candle_timestamp(connection, tradingsymbol=symbol, timeframe=timeframe)
        start = first_bar if first_bar is not None and first_bar > window_start else window_start
        frame = read_candles(
            connection, start=start, end=session_end, symbols=[symbol], timeframes=[timeframe]
        )
        result.expected_rows += session_bars
        if frame.empty:
            result.contracts_without_candles += 1
            result.missing_rows += session_bars
            context.logger.warning(
                f"{day.isoformat()} {symbol}: no {timeframe} candles in the session; run "
                "--mode build-candles first"
            )
            continue
        try:
            features = build_feature_frame(
                candles=frame,
                index_candles=index_frame,
                calendar=context.calendar,
                config=context.config.indicators,
                timeframe=timeframe,
                underlying=underlying,
                run_id=context.run_context.run_id,
                session_start=session_start,
                session_end=session_end,
            )
            rows = feature_rows(features)
        except (FeatureBuildError, ValueError) as exc:
            # One unusable contract must not cost us the other two hundred. Its stored
            # rows are deliberately left untouched: a stale row that is visible in this
            # summary (and exits non-zero) beats deleting data we cannot rebuild now.
            result.failures.append(f"{symbol}: {exc}")
            result.missing_rows += session_bars
            context.logger.error(f"{day.isoformat()} {symbol}: {exc}")
            continue

        summary = missing_bar_summary(
            rows, session_start=session_start, session_end=session_end, minutes=minutes
        )
        result.rows += writer.replace_features(
            start=session_start,
            end=session_end,
            timeframe=timeframe,
            tradingsymbol=symbol,
            rows=rows,
        )
        result.missing_rows += summary.missing_bars
        gap_count += summary.gap_count
        largest_gap = max(largest_gap, summary.largest_gap_seconds)
        if rows:
            first_timestamp = rows[0].timestamp if first_timestamp is None else min(
                first_timestamp, rows[0].timestamp
            )
            last_timestamp = rows[-1].timestamp if last_timestamp is None else max(
                last_timestamp, rows[-1].timestamp
            )
        if position % 25 == 0:
            context.logger.info(
                f"{day.isoformat()} {timeframe}: {position}/{len(contracts)} contract(s) built"
            )

    coverage = _feature_coverage_row(
        context,
        day=day,
        timeframe=timeframe,
        underlying=underlying,
        result=result,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        gap_count=gap_count,
        largest_gap_seconds=largest_gap,
    )
    writer.write_coverage([coverage])
    result.coverage[coverage.quality_flag.value] = 1
    if writer.parquet_enabled:
        result.parquet = [
            str(path)
            for path in writer.mirror_day(
                day=day, start=session_start, end=session_end, timeframes=[timeframe]
            )
        ]
    if result.failures:
        result.status = "PARTIAL"
    return result



def _feature_plan_lines(
    context: AppContext,
    *,
    connection: Any,
    days: Sequence[date],
    timeframe: str,
    symbols: Sequence[str] | None,
) -> list[str]:
    """Dry-run plan: what *would* be built, and what it would be built from."""
    lines = [f"ANJANEYA-ENGINE dry-run: build-features -> {len(days)} day(s) at {timeframe}"]
    lines.append(
        "  contracts     : "
        + (", ".join(symbols) if symbols else "every contract with ticks in the session")
    )
    lines.append(
        f"  lookback      : {required_lookback_bars(context.config.indicators)} bar(s) of warm-up "
        "+ one session before each day"
    )
    for day in days:
        if not context.calendar.is_trading_day(day):
            lines.append(f"  {day.isoformat()}   : no session (weekend or holiday)")
            continue
        window_start, session_start, session_end = _feature_window(
            context, day=day, timeframe=timeframe
        )
        count = (
            len(symbols)
            if symbols
            else len(symbols_with_ticks(connection, start=session_start, end=session_end))
        )
        index_frame = _index_frame(
            context, connection, start=window_start, end=session_end, timeframe=timeframe
        )
        index_note = (
            f"{len(index_frame)} NIFTY index bar(s)"
            if len(index_frame)
            else "NO NIFTY index history (nifty_* would be NULL)"
        )
        lines.append(
            f"  {day.isoformat()}   : {count} contract(s); window opens "
            f"{window_start.isoformat()}; {index_note}"
        )
    lines.append("  dry-run writes no feature rows, no coverage rows and no Parquet")
    return lines


def _render_feature_summary(
    *,
    results: Sequence[FeatureDayResult],
    timeframe: str,
    stats: FeatureWriteStats,
    as_json: bool,
) -> str:
    """Render the build-features summary as JSON or as an operator-friendly block."""
    failures = [message for result in results for message in result.failures]
    if as_json:
        return json.dumps(
            {
                "mode": "build-features",
                "timeframe": timeframe,
                "days": [result.to_dict() for result in results],
                "rows_written": stats.rows,
                "rows_replaced": stats.deleted,
                "coverage_rows": stats.coverage,
                "parquet_files": stats.parquet_files,
                "errors": stats.failures,
                "failures": failures,
            },
            indent=2,
            default=str,
        )

    lines = [f"ANJANEYA-ENGINE build-features - {len(results)} day(s) at {timeframe}"]
    for result in results:
        lines.append(
            f"  {result.day.isoformat()} [{result.status}] {result.contracts} contract(s); "
            f"{result.rows}/{result.expected_rows} row(s)"
        )
        if result.missing_rows:
            lines.append(
                f"      missing     : {result.missing_rows} expected bar(s) have no feature row"
            )
        if result.contracts_without_candles:
            lines.append(
                f"      no candles  : {result.contracts_without_candles} contract(s) "
                "(run --mode build-candles first)"
            )
        lines.append(f"      NIFTY index : {result.index_rows} bar(s)")
        if result.coverage:
            lines.append(
                "      coverage    : "
                + ", ".join(f"{name}={count}" for name, count in sorted(result.coverage.items()))
            )
    lines.append(f"  rows written    : {stats.rows} ({stats.deleted} replaced)")
    lines.append(f"  coverage rows   : {stats.coverage}")
    lines.append(f"  parquet files   : {stats.parquet_files}")
    lines.append(f"  failures        : {len(failures)}")
    for message in failures:
        lines.append(f"      {message}")
    return "\n".join(lines)



def _run_build_features(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode build-features``: candles -> the unified feature row (spec 18)."""
    timeframe = _resolve_feature_timeframe(context, getattr(args, "timeframes", None))
    symbols = getattr(args, "symbols", None)
    if not context.db_path.exists():
        print(
            f"ANJANEYA-ENGINE: no DuckDB database at {context.db_path}; there is nothing to "
            "build from yet.\n"
            "  next step: capture a session and build its candles "
            "(python -m apps.anjaneeya_engine.run --mode build-candles)",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    status, exit_code = "SUCCESS", EXIT_OK
    # One connection for the whole run: DuckDB allows a single read-write process per
    # file, and the writer borrows this one instead of opening its own.
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        days = _resolve_days(context, connection, args)
        if not days:
            print(
                "ANJANEYA-ENGINE: no stored ticks found - capture a session first.",
                file=sys.stderr,
            )
            status = "NO_DATA"
        elif context.dry_run:
            print(
                "\n".join(
                    _feature_plan_lines(
                        context,
                        connection=connection,
                        days=days,
                        timeframe=timeframe,
                        symbols=symbols,
                    )
                )
            )
        else:
            writer = FeatureWriter(
                connection=connection,
                parquet_root=context.config.data_dir("features"),
                run_id=context.run_context.run_id,
                mirror_parquet=context.config.settings.engine.mirror_parquet,
                parquet_compression=context.config.settings.data.parquet_compression,
                logger=context.logger,
            )
            results = [
                _build_feature_day(
                    context,
                    connection=connection,
                    writer=writer,
                    day=day,
                    timeframe=timeframe,
                    symbols=symbols,
                )
                for day in days
            ]
            print(
                _render_feature_summary(
                    results=results,
                    timeframe=timeframe,
                    stats=writer.stats,
                    as_json=args.as_json,
                )
            )
            if any(result.failures for result in results):
                status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"build-features failed: {type(exc).__name__}: {exc}")
        print(
            f"ANJANEYA-ENGINE build-features failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)


def _greek_model(context: AppContext) -> GreeksModel:
    """The IV/greeks model this run uses (``indicators.greeks`` -> documented assumptions)."""
    return GreeksModel.from_config(context.config.indicators)


@dataclass(slots=True)
class GreekDayResult:
    """What one trading day's IV/greeks enrichment did."""

    day: date
    status: str
    timeframes: dict[str, int] = field(default_factory=dict)
    contracts: int = 0
    bars: int = 0
    solved_iv: int = 0
    unsolved_iv: int = 0
    missing_spot: int = 0
    max_price_error: float = float("nan")
    parquet: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "day": self.day.isoformat(),
            "status": self.status,
            "timeframes": self.timeframes,
            "contracts": self.contracts,
            "bars": self.bars,
            "solved_iv": self.solved_iv,
            "unsolved_iv": self.unsolved_iv,
            "missing_spot": self.missing_spot,
            "max_price_error": (
                None if not math.isfinite(self.max_price_error) else self.max_price_error
            ),
            "parquet": self.parquet,
            "failures": self.failures,
        }


def _enrich_greek_day(
    context: AppContext,
    *,
    connection: Any,
    writer: GreeksWriter,
    day: date,
    timeframes: Sequence[str],
    symbols: Sequence[str] | None,
    model: GreeksModel,
) -> GreekDayResult:
    """Enrich one trading day's stored candles, contract by contract, timeframe by timeframe.

    Only the *session* window is read: unlike a feature build, an IV needs no history - it
    is a function of this bar's price, the NIFTY level at its close instant and the time
    left to expiry. A failure is local: the contract is named in the summary, its greeks
    stay as they were, and the rest of the day is still enriched.
    """
    result = GreekDayResult(day=day, status="SUCCESS")
    if not context.calendar.is_trading_day(day):
        reason = "weekend" if context.calendar.is_weekend(day) else "holiday"
        context.logger.info(f"{day.isoformat()} is not a trading day ({reason}); nothing to enrich")
        result.status = "NO_SESSION"
        return result

    session_start = context.calendar.session_start(day)
    session_end = context.calendar.session_end(day)
    index_frames: dict[str, Any] = {}
    contracts: set[str] = set()
    for timeframe in timeframes:
        contracts_for_timeframe = (
            list(symbols)
            if symbols
            else symbols_with_candles(
                connection, start=session_start, end=session_end, timeframes=[timeframe]
            )
        )
        if not contracts_for_timeframe:
            context.logger.warning(
                f"{day.isoformat()} {timeframe}: no candles in the session; "
                "run --mode build-candles first"
            )
            result.timeframes[timeframe] = 0
            continue
        if timeframe not in index_frames:
            index_frames[timeframe] = _index_frame(
                context, connection, start=session_start, end=session_end, timeframe=timeframe
            )
            if index_frames[timeframe].empty:
                context.logger.warning(
                    f"no NIFTY index history for {day.isoformat()} at {timeframe}; every iv on this "
                    "timeframe stays NULL (run capture's index-history mode to fill it)"
                )
        index_frame = index_frames[timeframe]
        enriched_bars = 0
        for symbol in contracts_for_timeframe:
            frame = read_candles(
                connection,
                start=session_start,
                end=session_end,
                symbols=[symbol],
                timeframes=[timeframe],
            )
            if frame.empty:
                continue
            try:
                outcome = enrich_frame(
                    frame,
                    index_frame=index_frame,
                    timeframe=timeframe,
                    calendar=context.calendar,
                    model=model,
                    timezone_name=context.config.timezone_name,
                )
                writer.apply_greeks(
                    start=session_start,
                    end=session_end,
                    timeframe=timeframe,
                    tradingsymbol=symbol,
                    frame=outcome.frame,
                )
            except Exception as exc:  # noqa: BLE001 - one contract must not sink the day
                message = f"{day.isoformat()} {timeframe} {symbol}: {type(exc).__name__}: {exc}"
                context.logger.error(f"enrich-greeks failed: {message}")
                result.failures.append(message)
                continue
            contracts.add(symbol)
            result.bars += outcome.bars
            result.solved_iv += outcome.solved_iv
            result.unsolved_iv += outcome.unsolved_iv
            result.missing_spot += outcome.missing_spot
            if math.isfinite(outcome.max_price_error):
                current = result.max_price_error
                result.max_price_error = (
                    max(current, outcome.max_price_error)
                    if math.isfinite(current)
                    else outcome.max_price_error
                )
            enriched_bars += outcome.bars
        result.timeframes[timeframe] = enriched_bars

    result.contracts = len(contracts)
    if result.failures:
        result.status = "PARTIAL"
    elif not result.bars:
        result.status = "NO_DATA"
    if result.bars:
        # The mirror is refreshed *after* the UPDATEs, so Parquet cannot keep serving
        # pre-enrichment candles that DuckDB no longer agrees with.
        result.parquet = [
            str(path)
            for path in writer.mirror_day(
                day=day, start=session_start, end=session_end, timeframes=list(timeframes)
            )
        ]
    return result


def _greek_plan_lines(
    context: AppContext,
    *,
    connection: Any,
    days: Sequence[date],
    timeframes: Sequence[str],
    symbols: Sequence[str] | None,
    model: GreeksModel,
) -> list[str]:
    """What ``--mode enrich-greeks --dry-run`` would do (read-only)."""
    lines = [
        f"ANJANEYA-ENGINE dry-run: enrich-greeks -> {len(days)} day(s), timeframes "
        f"{', '.join(timeframes)}",
        f"  model      : {model.name}, r={model.risk_free_rate}, q={model.dividend_yield}, "
        f"price={model.price_field}, ACT/{model.year_days}, floor="
        f"{model.min_time_to_expiry_minutes:g}min",
        f"  iv bracket : {model.volatility_floor} .. {model.volatility_ceiling} "
        f"(tolerance {model.tolerance:g}, max {model.max_iterations} iterations)",
    ]
    for day in days:
        if not context.calendar.is_trading_day(day):
            lines.append(f"  {day.isoformat()}   : not a trading day; nothing to enrich")
            continue
        session_start = context.calendar.session_start(day)
        session_end = context.calendar.session_end(day)
        candles = read_candles(
            connection,
            start=session_start,
            end=session_end,
            symbols=symbols,
            timeframes=timeframes,
        )
        contract_count = int(candles["tradingsymbol"].nunique()) if not candles.empty else 0
        index_rows = len(
            _index_frame(
                context, connection, start=session_start, end=session_end, timeframe=timeframes[0]
            )
        )
        lines.append(
            f"  {day.isoformat()}   : {contract_count} contract(s), {len(candles)} bar(s), "
            f"NIFTY index {index_rows} row(s)"
        )
    lines.append("  dry-run writes no greeks and no Parquet")
    return lines


def _render_greek_summary(
    *,
    results: Sequence[GreekDayResult],
    stats: GreeksWriteStats,
    model: GreeksModel,
    as_json: bool,
) -> str:
    """Render the enrichment run summary as JSON or as an operator-friendly block."""
    failures = [message for result in results for message in result.failures]
    if as_json:
        return json.dumps(
            {
                "mode": "enrich-greeks",
                "model": model.describe(),
                "days": [result.to_dict() for result in results],
                "bars": stats.bars,
                "updated": stats.updated,
                "unmatched": stats.unmatched,
                "unsolved_iv": stats.unsolved_iv,
                "parquet_files": stats.parquet_files,
                "errors": stats.failures,
                "failures": failures,
            },
            indent=2,
            default=str,
        )

    lines = [f"ANJANEYA-ENGINE enrich-greeks - {len(results)} day(s)"]
    lines.append(
        f"  model           : {model.name}, r={model.risk_free_rate}, q={model.dividend_yield}, "
        f"price={model.price_field}"
    )
    for result in results:
        bars = (
            ", ".join(f"{name}={count}" for name, count in sorted(result.timeframes.items()))
            or "no bars"
        )
        lines.append(
            f"  {result.day.isoformat()} [{result.status}] {result.contracts} contract(s); {bars}"
        )
        iv_line = f"      iv          : {result.solved_iv} solved, {result.unsolved_iv} unsolved"
        if result.missing_spot:
            iv_line += f" ({result.missing_spot} without a NIFTY level at their close)"
        lines.append(iv_line)
        if math.isfinite(result.max_price_error):
            lines.append(f"      price error : max |price - BS(iv)| = {result.max_price_error:.3e}")
    lines.append(f"  bars updated    : {stats.updated} of {stats.bars}")
    if stats.unmatched:
        lines.append(f"  unmatched       : {stats.unmatched} enriched bar(s) had vanished")
    lines.append(f"  parquet files   : {stats.parquet_files}")
    lines.append(f"  failures        : {len(failures)}")
    for message in failures:
        lines.append(f"      {message}")
    return "\n".join(lines)


def _run_enrich_greeks(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode enrich-greeks``: spec 16's IV/greeks onto stored candles."""
    timeframes = _resolve_timeframes(context, getattr(args, "timeframes", None))
    symbols = getattr(args, "symbols", None)
    model = _greek_model(context)
    if not context.db_path.exists():
        print(
            f"ANJANEYA-ENGINE: no DuckDB database at {context.db_path}; there is nothing to "
            "enrich yet.\n"
            "  next step: capture a session and build its candles "
            "(python -m apps.anjaneeya_engine.run --mode build-candles)",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    status, exit_code = "SUCCESS", EXIT_OK
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        days = _resolve_days(
            context,
            connection,
            args,
            days_source=partial(trading_days_with_candles, timeframes=timeframes),
        )
        if not days:
            print(
                "ANJANEYA-ENGINE: no stored candles found - build them first "
                "(--mode build-candles).",
                file=sys.stderr,
            )
            status = "NO_DATA"
        elif context.dry_run:
            print(
                "\n".join(
                    _greek_plan_lines(
                        context,
                        connection=connection,
                        days=days,
                        timeframes=timeframes,
                        symbols=symbols,
                        model=model,
                    )
                )
            )
        else:
            writer = GreeksWriter(
                connection=connection,
                parquet_root=context.config.data_dir("candles"),
                run_id=context.run_context.run_id,
                mirror_parquet=context.config.settings.engine.mirror_parquet,
                parquet_compression=context.config.settings.data.parquet_compression,
                logger=context.logger,
            )
            results = [
                _enrich_greek_day(
                    context,
                    connection=connection,
                    writer=writer,
                    day=day,
                    timeframes=timeframes,
                    symbols=symbols,
                    model=model,
                )
                for day in days
            ]
            print(
                _render_greek_summary(
                    results=results, stats=writer.stats, model=model, as_json=args.as_json
                )
            )
            if any(result.failures for result in results):
                status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"enrich-greeks failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-ENGINE enrich-greeks failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)


def _validation_report_dir(context: AppContext) -> Any:
    """Where ``--mode validate`` writes its JSON reports."""
    return context.config.data_dir("reports") / "feature_validation"


def _write_validation_report(context: AppContext, *, day: date, timeframe: str, report: Any) -> Any:
    """Write one day's validation report (atomically: temp file, then rename)."""
    directory = _validation_report_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{day.isoformat()}_{timeframe}.json"
    temporary = target.with_name(target.name + ".tmp")
    payload = {
        "mode": "validate",
        "run_id": context.run_context.run_id,
        "config_hash": context.run_context.config_hash,
        "generated_at": datetime.now(tz=context.calendar.session_start(day).tzinfo).isoformat(),
        **report.to_dict(),
    }
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)
    context.run_context.register_artifact(target)
    return target


def _validate_day(
    context: AppContext,
    *,
    connection: Any,
    day: date,
    timeframe: str,
    symbols: Sequence[str] | None,
    probes: int,
) -> Any:
    """Validate one day's stored feature rows, contract by contract."""
    window_start, session_start, session_end = _feature_window(context, day=day, timeframe=timeframe)
    stored = read_features(
        connection,
        start=session_start,
        end=session_end,
        timeframes=[timeframe],
        symbols=symbols,
    )
    scope = {
        "day": day.isoformat(),
        "timeframe": timeframe,
        "underlying": context.config.option_universe.underlying,
    }
    if stored.empty:
        return validate_feature_frame(stored, timeframe=timeframe, scope=scope, probes=0)
    index_frame = _index_frame(
        context, connection, start=window_start, end=session_end, timeframe=timeframe
    )
    contract_reports = []
    for symbol in sorted({str(name) for name in stored["tradingsymbol"].unique()}):
        rows = stored[stored["tradingsymbol"] == symbol].reset_index(drop=True)
        first_bar = first_candle_timestamp(connection, tradingsymbol=symbol, timeframe=timeframe)
        start = first_bar if first_bar is not None and first_bar > window_start else window_start
        candles = read_candles(
            connection,
            start=start,
            end=session_end,
            symbols=[symbol],
            timeframes=[timeframe],
        )
        contract_reports.append(
            validate_feature_frame(
                rows,
                candles=candles,
                index_candles=index_frame,
                calendar=context.calendar,
                config=context.config.indicators,
                timeframe=timeframe,
                session_start=session_start,
                session_end=session_end,
                probes=probes,
                scope={**scope, "tradingsymbol": symbol},
            )
        )
    return merge_reports(contract_reports, scope={**scope, "contracts": len(contract_reports)})




def _run_validate(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode validate``: spec 19's checks over the *stored* feature rows."""
    timeframe = _resolve_feature_timeframe(context, getattr(args, "timeframes", None))
    symbols = getattr(args, "symbols", None)
    probes = int(getattr(args, "probes", 3))
    if not context.db_path.exists():
        print(
            f"ANJANEYA-ENGINE: no DuckDB database at {context.db_path}; there is nothing to "
            "validate yet.",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    status, exit_code = "SUCCESS", EXIT_OK
    # Validation never writes to the database: it reads the stored rows plus the inputs
    # they were built from, and writes only its report.
    connection = connect(context.db_path, read_only=True)
    try:
        days = _resolve_days(context, connection, args)
        if not days:
            print("ANJANEYA-ENGINE: no stored ticks found - nothing to validate.", file=sys.stderr)
            status = "NO_DATA"
        elif context.dry_run:
            lines = [f"ANJANEYA-ENGINE dry-run: validate -> {len(days)} day(s) at {timeframe}"]
            for day in days:
                _, session_start, session_end = _feature_window(context, day=day, timeframe=timeframe)
                stored = read_features(
                    connection,
                    start=session_start,
                    end=session_end,
                    timeframes=[timeframe],
                    symbols=symbols,
                )
                lines.append(
                    f"  {day.isoformat()}   : {len(stored)} stored row(s), {probes} "
                    "no-look-ahead probe(s)"
                )
            lines.append("  dry-run reads only; no validation report is written")
            print("\n".join(lines))
        else:
            reports: list[Any] = []
            printed: list[str] = []
            for day in days:
                report = _validate_day(
                    context,
                    connection=connection,
                    day=day,
                    timeframe=timeframe,
                    symbols=symbols,
                    probes=probes,
                )
                path = _write_validation_report(
                    context, day=day, timeframe=timeframe, report=report
                )
                context.logger.info(f"validation report written: {path}")
                reports.append(report)
                printed.append(f"  {day.isoformat()} [{report.status}] {report.summary_line()}")
                for check in report.checks:
                    if check.status == "FAIL":
                        printed.append(f"      FAIL {check.name}: {check.detail}")
                        printed.extend(f"          {sample}" for sample in check.samples[:3])
                    elif check.status == "WARN":
                        printed.append(f"      WARN {check.name}: {check.detail}")

            if args.as_json:
                print(
                    json.dumps(
                        {
                            "mode": "validate",
                            "timeframe": timeframe,
                            "days": [report.to_dict() for report in reports],
                        },
                        indent=2,
                        default=str,
                    )
                )
            else:
                print(f"ANJANEYA-ENGINE validate - {len(reports)} day(s) at {timeframe}")
                print("\n".join(printed))
            if any(not report.passed for report in reports):
                status, exit_code = "FAILED", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"validate failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-ENGINE validate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)



# ---------------------------------------------------------------------------
# --mode universe (spec 17): the configured menu, checked against the database
# ---------------------------------------------------------------------------
def _as_date(value: Any) -> date | None:
    """A ``DATE`` column's value normalised to :class:`datetime.date`.

    DuckDB hands a ``DATE`` back as a pandas ``Timestamp`` (``datetime64[us]``),
    which never compares equal to the ``date`` a calendar produced - the same trap
    ``apps.anjaneeya_engine.candles`` documents for the candle row contract. Every
    contract match in this mode goes through here.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@dataclass(slots=True)
class UniverseContractResult:
    """What the engine knows about one selected contract."""

    expiry: date
    strike: float
    option_type: str
    tradingsymbol: str | None = None
    instrument_token: int | None = None
    candles: int = 0
    features: int = 0
    symbol_matches_template: bool | None = None

    @property
    def status(self) -> str:
        """One word for the row: ``OK``, or the first thing that is missing."""
        if self.instrument_token is None:
            return "NOT_LISTED"
        if self.candles == 0:
            return "NO_CANDLES"
        if self.features == 0:
            return "NO_FEATURES"
        return "OK"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type,
            "tradingsymbol": self.tradingsymbol,
            "instrument_token": self.instrument_token,
            "candles": self.candles,
            "features": self.features,
            "symbol_matches_template": self.symbol_matches_template,
            "status": self.status,
        }


@dataclass(slots=True)
class UniverseDayResult:
    """One day's answer to spec 17: what the config selected, and what we hold."""

    day: date
    timeframe: str
    reference_label: str
    reference_price: float | None
    reference_source: str
    selection: OptionSelection | None
    contracts: list[UniverseContractResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """How many selected contracts are in each state (``OK``, ``NO_CANDLES``...)."""
        counts: dict[str, int] = {}
        for contract in self.contracts:
            counts[contract.status] = counts.get(contract.status, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def complete(self) -> bool:
        """True when every selected contract has both candles and features."""
        return bool(self.contracts) and all(contract.status == "OK" for contract in self.contracts)

    @property
    def symbol_drift(self) -> int:
        """Selected contracts whose stored symbol does not match the template."""
        return sum(1 for contract in self.contracts if contract.symbol_matches_template is False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "day": self.day.isoformat(),
            "timeframe": self.timeframe,
            "reference_label": self.reference_label,
            "reference_price": self.reference_price,
            "reference_source": self.reference_source,
            "selection": self.selection.describe() if self.selection else None,
            "counts": self.counts,
            "complete": self.complete,
            "symbol_drift": self.symbol_drift,
            "missing": self.missing,
            "contracts": [contract.to_dict() for contract in self.contracts],
        }


def _stored_futures_symbol(
    context: AppContext, frame: Any, *, day: date
) -> tuple[str | None, str]:
    """The stored futures contract that ``atm.reference: futures`` names.

    The engine has no instrument *dump* (that is capture's), so this applies the
    same rule capture's ``futures_contract`` applies - the nearest live expiry
    on/after the day, or the one after it for ``next_month`` - to the stored
    ``instruments`` table. Returns ``(tradingsymbol, reason)``; the reason is
    reported, never guessed.
    """
    if frame.empty:
        return None, "the instruments table is empty (capture has not stored a dump yet)"
    stored = frame[["tradingsymbol", "instrument_type", "expiry"]].copy()
    # Normalise *before* filtering: assigning a Series onto an empty selection
    # broadcasts it and invents rows (pandas), which would fabricate a futures
    # contract out of nothing.
    stored["expiry"] = stored["expiry"].map(_as_date)
    stored["instrument_type"] = stored["instrument_type"].astype(str).str.upper()
    futures = stored[(stored["instrument_type"] == "FUT") & stored["expiry"].notna()]
    futures = futures.sort_values("expiry")
    if futures.empty:
        return None, "the instruments table holds no futures row"
    live = futures[futures["expiry"] >= day]
    if live.empty:
        return None, f"no futures contract expiring on/after {day.isoformat()}"
    selector = context.config.option_universe.reference_instruments.nifty_futures.expiry_selector
    if selector == "next_month":
        if len(live) < 2:
            return None, "expiry_selector is next_month but only one contract is listed"
        row = live.iloc[1]
    else:
        row = live.iloc[0]
    return str(row["tradingsymbol"]), f"instruments:FUT expiry={row['expiry']}"


def _universe_reference_price(
    context: AppContext,
    connection: Any,
    *,
    day: date,
    explicit: float | None,
) -> tuple[float | None, str]:
    """``(price, source)`` for the ATM anchor, from ``--price`` or stored data.

    ``spot`` reads the NIFTY index rows - ``index_history`` first, candles as a
    fallback (a session captured through the WebSocket has both, an older day may
    have only one). ``futures`` reads the resolved futures contract's candles. A
    price that cannot be found is reported with its reason and never replaced by a
    guess: saying what is actually stored is the whole point of the mode.
    """
    if explicit is not None:
        return explicit, "--price (operator supplied)"
    label = context.config.option_universe.atm.reference
    session_start = context.calendar.session_start(day)
    session_end = context.calendar.session_end(day)

    if label == "futures":
        stored, reason = _stored_futures_symbol(
            context, read_instruments(connection), day=day
        )
        if stored is None:
            return None, reason
        frame = read_candles(connection, start=session_start, end=session_end, symbols=[stored])
        if frame.empty:
            return None, f"no stored candles for the futures contract {stored}"
        last = frame.sort_values("timestamp").iloc[-1]
        return float(last["close"]), f"candles:{stored} @ {last['timestamp']}"

    symbol = _underlying_symbol(context)
    index_frame = read_index_history(
        connection, start=session_start, end=session_end, symbols=[symbol]
    )
    if not index_frame.empty:
        last = index_frame.sort_values("timestamp").iloc[-1]
        return float(last["close"]), f"index_history:{symbol} @ {last['timestamp']}"
    candle_frame = read_candles(connection, start=session_start, end=session_end, symbols=[symbol])
    if not candle_frame.empty:
        last = candle_frame.sort_values("timestamp").iloc[-1]
        return float(last["close"]), f"candles:{symbol} @ {last['timestamp']}"
    return None, (
        f"no stored index_history or candle rows for {symbol} on {day.isoformat()}; "
        "pass --price to place the ATM by hand"
    )


def _universe_day(
    context: AppContext,
    connection: Any,
    *,
    day: date,
    timeframe: str,
    explicit_price: float | None,
) -> UniverseDayResult:
    """Resolve one day's configured universe and check it against the database."""
    label = context.config.option_universe.atm.reference
    price, source = _universe_reference_price(context, connection, day=day, explicit=explicit_price)
    result = UniverseDayResult(
        day=day,
        timeframe=timeframe,
        reference_label=label,
        reference_price=price,
        reference_source=source,
        selection=None,
    )
    if price is None:
        result.missing.append(source)
        return result

    selection = build_selection(
        universe_config=context.config.option_universe,
        reference_price=price,
        day=day,
        calendar=context.calendar,
        reference_label=label,
        created_at=context.calendar.session_start(day),
    )
    result.selection = selection
    session_start = context.calendar.session_start(day)
    session_end = context.calendar.session_end(day)

    instrument_rows = read_option_instruments(
        connection,
        exchange=selection.exchange,
        name=selection.underlying,
        expiries=selection.expiries,
    )
    known: dict[tuple[date, float, str], Any] = {}
    # A contract is matched by its attributes, never by a symbol string: the strike is
    # rounded to a fixed number of decimals because it is stored as a DOUBLE.
    for row in instrument_rows.itertuples(index=False):
        instrument_type = str(row.instrument_type).upper()
        expiry = _as_date(row.expiry)
        if expiry is None:
            continue
        known.setdefault((expiry, round(float(row.strike), 6), instrument_type), row)

    bars = candle_counts(connection, start=session_start, end=session_end, timeframe=timeframe)
    features = feature_counts(connection, start=session_start, end=session_end, timeframe=timeframe)

    templates = context.config.option_universe.symbol_templates
    pattern = template_pattern(templates.weekly)
    for spec in selection.contracts:
        key = (spec.expiry, round(spec.strike, 6), spec.option_type.value)
        row = known.get(key)
        symbol = str(row.tradingsymbol) if row is not None else None
        result.contracts.append(
            UniverseContractResult(
                expiry=spec.expiry,
                strike=spec.strike,
                option_type=spec.option_type.value,
                tradingsymbol=symbol,
                instrument_token=int(row.instrument_token) if row is not None else None,
                candles=bars.get(symbol, 0) if symbol else 0,
                features=features.get(symbol, 0) if symbol else 0,
                # A mismatch means the exchange's symbol shape moved away from the
                # configured template - worth stating, never a silent success.
                symbol_matches_template=(pattern.match(symbol) is not None) if symbol else None,
            )
        )
        if row is None:
            result.missing.append(
                f"{spec.option_type.value} {spec.strike:g} not listed for {spec.expiry.isoformat()}"
            )
    return result


def _render_universe_summary(
    results: Sequence[UniverseDayResult], *, rounding: str, as_json: bool
) -> str:
    """The human (or JSON) report ``--mode universe`` prints."""
    if as_json:
        return json.dumps(
            {"mode": "universe", "days": [result.to_dict() for result in results]},
            indent=2,
            default=str,
        )
    lines: list[str] = []
    for result in results:
        lines.append(
            f"ANJANEYA-ENGINE universe - {result.day.isoformat()} "
            f"(candle timeframe {result.timeframe})"
        )
        if result.selection is None:
            lines.append(f"  reference      : UNRESOLVED - {result.reference_source}")
            continue
        selection = result.selection
        lines.append(
            f"  reference      : {selection.reference_label} @ {result.reference_price:,.2f} "
            f"({result.reference_source})"
        )
        atm_note = "" if selection.includes_atm else "  (the ATM itself is NOT in the ladder)"
        lines.append(
            f"  ATM strike     : {selection.atm_strike:,.0f} "
            f"(interval {selection.strike_interval:g}, rounding {rounding}){atm_note}"
        )
        lines.append(
            "  strike ladder  : "
            + ", ".join(f"{strike:,.0f}" for strike in selection.strikes)
            + f"   offsets: {', '.join(f'{offset:+,.0f}' for offset in selection.offsets)}"
        )
        lines.append(
            "  expiries       : " + ", ".join(expiry.isoformat() for expiry in selection.expiries)
        )
        lines.append(f"  selected       : {len(result.contracts)} contract(s)")
        if result.contracts:
            lines.append("")
            lines.append(
                f"  {'expiry':<12} {'strike':>9} {'side':<4} {'tradingsymbol':<22} "
                f"{'candles':>7} {'features':>8}  status"
            )
            for contract in result.contracts:
                lines.append(
                    f"  {contract.expiry.isoformat():<12} {contract.strike:>9,.0f} "
                    f"{contract.option_type:<4} {contract.tradingsymbol or '-':<22} "
                    f"{contract.candles:>7} {contract.features:>8}  {contract.status}"
                )
        lines.append("")
        lines.append(
            "  coverage       : "
            + (", ".join(f"{name}={count}" for name, count in result.counts.items()) or "none")
        )
        lines.append(
            f"  complete       : {'yes' if result.complete else 'no'}"
            + (f" ({result.symbol_drift} symbol(s) off-template)" if result.symbol_drift else "")
        )
        for note in result.missing[:8]:
            lines.append(f"  missing        : {note}")
        if len(result.missing) > 8:
            lines.append(f"  missing        : ... and {len(result.missing) - 8} more")
    return "\n".join(lines)


def _universe_plan_lines(context: AppContext, *, days: Sequence[date], timeframe: str) -> list[str]:
    """What ``--mode universe`` would report (``--dry-run``), without reading data."""
    universe_config = context.config.option_universe
    return [
        f"ANJANEYA-ENGINE universe dry-run - {len(days)} day(s) at {timeframe}",
        f"  config         : ATM reference={universe_config.atm.reference!r}, rounding="
        f"{universe_config.atm.rounding!r}, offsets={list(universe_config.strikes_offset)}, "
        f"expiries={list(universe_config.expiries)}",
        "  would report   : the ATM anchor price and where it came from, the strike ladder,",
        "                   the expiries, and per contract its stored symbol, its candle and",
        "                   feature row counts, and whether the symbol still matches the",
        "                   configured template",
        "  writes         : nothing (the database is opened read-only)",
    ]


def _run_universe(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode universe``: spec 17's configured menu, checked against stored data."""
    timeframe = _resolve_feature_timeframe(context, getattr(args, "timeframes", None))
    explicit_price = getattr(args, "price", None)
    if explicit_price is not None and (not math.isfinite(explicit_price) or explicit_price <= 0):
        raise ConfigError(f"--price must be a positive number, got {explicit_price!r}")
    if not context.db_path.exists():
        print(
            f"ANJANEYA-ENGINE: no DuckDB database at {context.db_path}; there is nothing to "
            "report against yet.\n"
            "  next step: capture a session "
            "(python -m apps.anjaneeya_capture.run --mode capture)",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    status, exit_code = "SUCCESS", EXIT_OK
    # This mode only reads: opening the database read-only makes that structural
    # rather than a promise, and lets it run beside a capture process.
    connection = connect(context.db_path, read_only=True)
    try:
        days = _resolve_days(context, connection, args, days_source=trading_days_with_candles)
        if context.dry_run:
            print("\n".join(_universe_plan_lines(context, days=days, timeframe=timeframe)))
        elif not days:
            print(
                "ANJANEYA-ENGINE: no stored candles found - build a session first "
                "(--mode build-candles).",
                file=sys.stderr,
            )
            status = "NO_DATA"
        else:
            results = [
                _universe_day(
                    context,
                    connection,
                    day=day,
                    timeframe=timeframe,
                    explicit_price=explicit_price,
                )
                for day in days
            ]
            print(
                _render_universe_summary(
                    results,
                    rounding=context.config.option_universe.atm.rounding,
                    as_json=args.as_json,
                )
            )
            if any(result.selection is None for result in results):
                # The report was produced, but a day the operator asked about could not
                # be placed at all: that is a failed run, not a silent zero.
                status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"universe failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-ENGINE universe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)


# ---------------------------------------------------------------------------
# --mode build-labels (spec 20): the forward-looking pass, physically separate
# ---------------------------------------------------------------------------
#: One contract's line in the label summary.
@dataclass(slots=True)
class LabelContractResult:
    """What one contract-day's label build did."""

    tradingsymbol: str
    bars: int = 0
    rows: int = 0
    labelled: int = 0
    horizons: int = 0

    @property
    def labelled_share(self) -> float:
        """Share of rows carrying a class (``0.0`` when nothing was written)."""
        return (self.labelled / self.rows) if self.rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "tradingsymbol": self.tradingsymbol,
            "bars": self.bars,
            "rows": self.rows,
            "labelled": self.labelled,
            "horizons": self.horizons,
            "labelled_share": self.labelled_share,
        }


@dataclass(slots=True)
class LabelDayResult:
    """What one trading day's label build did (the CLI summary is built from these)."""

    day: date
    timeframe: str
    status: str
    contracts: list[LabelContractResult] = field(default_factory=list)
    rows: int = 0
    labelled: int = 0
    expected_rows: int = 0
    contracts_without_candles: int = 0
    coverage: dict[str, int] = field(default_factory=dict)
    parquet: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "day": self.day.isoformat(),
            "timeframe": self.timeframe,
            "status": self.status,
            "contracts": len(self.contracts),
            "rows": self.rows,
            "labelled": self.labelled,
            "expected_rows": self.expected_rows,
            "contracts_without_candles": self.contracts_without_candles,
            "coverage": self.coverage,
            "parquet": self.parquet,
            "failures": self.failures,
            "per_contract": [contract.to_dict() for contract in self.contracts],
        }


def _label_policy(context: AppContext) -> LabelPolicy:
    """The label policy this run uses (``model.yaml`` + the ATR it normalises by)."""
    return LabelPolicy.from_config(context.config.model, indicators=context.config.indicators)


def _label_warmup_bars(policy: LabelPolicy, timeframe: str) -> int:
    """Bars read *before* the session so the first session bars can be labelled.

    A threshold needs ``atr_period`` bars (when the policy normalises by ATR) and the
    prior move needs ``trend_lookback``; one extra bar makes the arithmetic inclusive.
    Reading more would not label anything extra - it would only cost time.
    """
    needed = max(policy.trend_lookback, policy.atr_period if policy.normalization == "atr" else 0)
    return needed + 1


def _label_coverage_timeframe(timeframe: str) -> str:
    """The label stream's spelling of a bar timeframe (``"1m:labels"``).

    ``data_coverage`` identifies a stream by ``(tradingsymbol, timeframe)`` alone, so a
    stream that is not the bar stream spells its own timeframe - the tick stream says
    ``"TICK"`` (see ANJANEYA-CAPTURE). Labels are keyed per *contract* like candles, so
    without its own spelling a label verdict would quietly overwrite the candle verdict
    for the same ``(day, contract, timeframe)`` - and the candle row is the only record
    of how many bars a contract actually had. Keeping the bar timeframe in the name
    (rather than dropping it) is what makes the row readable afterwards.
    """
    return f"{timeframe}:labels"


def _label_coverage_row(
    context: AppContext,
    *,
    day: date,
    timeframe: str,
    tradingsymbol: str,
    result: LabelContractResult,
    expected_rows: int,
) -> Any:
    """One ``data_coverage`` verdict for a contract's label stream.

    ``expected_rows`` is ``session bars x horizons`` - the number of rows a complete
    label build produces - so the coverage ratio answers "how much of this
    contract-day is described?", with the same vocabulary the candle and tick streams
    use. ``gap_count`` stays zero: a label row is never *missing* (every session bar
    gets one per horizon); what varies is whether it carries a class.
    """
    stats = StreamStats(
        tradingsymbol=tradingsymbol,
        timeframe=_label_coverage_timeframe(timeframe),
        rows=result.rows,
        first_timestamp=None,
        last_timestamp=None,
        gap_count=0,
        largest_gap_seconds=0.0,
        expected_rows=expected_rows,
    )
    return build_coverage_row(trading_date=day, stats=stats, run_id=context.run_context.run_id)


def _build_label_day(
    context: AppContext,
    connection: Any,
    *,
    writer: LabelWriter,
    day: date,
    timeframe: str,
    policy: LabelPolicy,
    symbols: Sequence[str] | None,
) -> LabelDayResult:
    """Build one day's labels, contract by contract.

    ``build-features`` is **not** consulted: labels are a function of the candles alone
    (spec 20's separation), so a day can be labelled before - or without - its feature
    rows existing.
    """
    session_start = context.calendar.session_start(day)
    session_end = context.calendar.session_end(day)
    minutes = timeframe_to_minutes(timeframe)
    warmup = _label_warmup_bars(policy, timeframe)
    window_start = session_start - timedelta(minutes=warmup * minutes)
    session_bars = len(
        expected_bar_openings(session_start=session_start, session_end=session_end, minutes=minutes)
    )
    expected = session_bars * len(policy.horizons)

    result = LabelDayResult(day=day, timeframe=timeframe, status="SUCCESS")
    if symbols:
        wanted = list(symbols)
    else:
        wanted = symbols_with_candles(
            connection, start=session_start, end=session_end, timeframes=[timeframe]
        )
    for symbol in sorted(wanted):
        try:
            candles = read_candles(
                connection,
                start=window_start,
                end=session_end,
                symbols=[symbol],
                timeframes=[timeframe],
            )
            if candles.empty:
                result.contracts_without_candles += 1
                result.failures.append(f"{symbol}: no {timeframe} candles for {day.isoformat()}")
                continue
            frame = build_label_frame(
                candles,
                policy=policy,
                timeframe=timeframe,
                run_id=context.run_context.run_id,
                session_start=session_start,
                session_end=session_end,
            )
            rows = label_rows(frame)
            contract = LabelContractResult(
                tradingsymbol=symbol,
                bars=len(candles),
                rows=len(rows),
                labelled=sum(1 for row in rows if row.label is not None),
                horizons=len(policy.horizons),
            )
            writer.replace_labels(
                start=session_start,
                end=session_end,
                timeframe=timeframe,
                tradingsymbol=symbol,
                rows=rows,
            )
            writer.write_coverage(
                [
                    _label_coverage_row(
                        context,
                        day=day,
                        timeframe=timeframe,
                        tradingsymbol=symbol,
                        result=contract,
                        expected_rows=expected,
                    )
                ]
            )
            result.contracts.append(contract)
            result.rows += contract.rows
            result.labelled += contract.labelled
            result.expected_rows += expected
        except Exception as exc:  # noqa: BLE001 - one contract must not sink the day
            result.failures.append(f"{symbol}: {type(exc).__name__}: {exc}")
            context.logger.error(f"labels for {symbol} {day.isoformat()} failed: {exc}")
    result.parquet = [
        path.name
        for path in writer.mirror_day(
            day=day, start=session_start, end=session_end, timeframes=[timeframe]
        )
    ]
    result.coverage = dict(writer.stats.describe())
    if result.failures:
        result.status = "PARTIAL"
    return result


def _render_label_summary(
    results: Sequence[LabelDayResult], *, policy: LabelPolicy, as_json: bool
) -> str:
    """The human (or JSON) report ``--mode build-labels`` prints."""
    if as_json:
        return json.dumps(
            {
                "mode": "build-labels",
                "policy": policy.describe(),
                "days": [result.to_dict() for result in results],
            },
            indent=2,
            default=str,
        )
    lines: list[str] = [
        f"ANJANEYA-ENGINE build-labels - {len(results)} day(s), policy "
        f"{policy.normalization} ({policy.version}), horizons "
        f"{', '.join(f'{horizon}m' for horizon in policy.horizons)}"
    ]
    for result in results:
        lines.append("")
        lines.append(f"  {result.day.isoformat()} @ {result.timeframe}: {result.status}")
        lines.append(
            f"    contracts      : {len(result.contracts)}"
            + (
                f" (+{result.contracts_without_candles} without candles)"
                if result.contracts_without_candles
                else ""
            )
        )
        lines.append(
            f"    rows           : {result.rows} of {result.expected_rows} expected "
            f"({result.labelled} labelled, "
            f"{result.rows - result.labelled} without a class)"
        )
        for contract in result.contracts[:12]:
            lines.append(
                f"      {contract.tradingsymbol:<22} {contract.bars:>4} bar(s) -> "
                f"{contract.rows:>5} row(s), {contract.labelled:>5} labelled "
                f"({contract.labelled_share:.0%})"
            )
        if len(result.contracts) > 12:
            lines.append(f"      ... and {len(result.contracts) - 12} more contract(s)")
        if result.parquet:
            lines.append(f"    parquet        : {', '.join(result.parquet)}")
        for note in result.failures[:8]:
            lines.append(f"    failure        : {note}")
        if len(result.failures) > 8:
            lines.append(f"    failure        : ... and {len(result.failures) - 8} more")
    return "\n".join(lines)


def _label_plan_lines(
    context: AppContext,
    *,
    days: Sequence[date],
    timeframe: str,
    policy: LabelPolicy,
    symbols: Sequence[str] | None,
) -> list[str]:
    """What ``--mode build-labels`` would do (``--dry-run``), without writing anything."""
    warmup = _label_warmup_bars(policy, timeframe)
    lines = [
        f"ANJANEYA-ENGINE build-labels dry-run - {len(days)} day(s) at {timeframe}",
        f"  policy         : normalization={policy.normalization!r} "
        f"(param {policy.normalization_param}, percent {policy.percent_threshold}, "
        f"fixed {policy.fixed_threshold}), trend_lookback={policy.trend_lookback}, "
        f"ATR({policy.atr_period}, {policy.atr_method}), version {policy.version}",
        f"  horizons       : {', '.join(f'{horizon}m' for horizon in policy.horizons)}",
        f"  warm-up        : {warmup} bar(s) read before each session (ATR + prior move)",
        f"  scope          : {', '.join(symbols) if symbols else 'every contract with candles'}",
        "  would replace  : one contract-day's labels at *every* horizon (delete, then append)",
        f"  writes         : the labels table, one data_coverage verdict per contract "
        f"spelled '{_label_coverage_timeframe(timeframe)}' (the candle verdict survives),",
        "                   the data/labels/<day>.parquet mirror, and nothing else - the",
        "                   features table is never touched (spec 20 keeps the two passes",
        "                   physically separate)",
    ]
    for day in days[:5]:
        session_start = context.calendar.session_start(day)
        session_end = context.calendar.session_end(day)
        bars = len(
            expected_bar_openings(
                session_start=session_start,
                session_end=session_end,
                minutes=timeframe_to_minutes(timeframe),
            )
        )
        unlabelable = min(policy.max_horizon // timeframe_to_minutes(timeframe), bars)
        lines.append(
            f"  {day.isoformat()}       : {bars} bar(s) x {len(policy.horizons)} horizon(s) = "
            f"{bars * len(policy.horizons)} row(s); the last {unlabelable} bar(s) cannot be "
            "labelled inside the session"
        )
    return lines


def _run_build_labels(context: AppContext, args: argparse.Namespace) -> int:
    """``--mode build-labels``: spec 20's forward-looking pass over stored candles."""
    timeframe = _resolve_feature_timeframe(context, getattr(args, "timeframes", None))
    symbols = getattr(args, "symbols", None)
    policy = _label_policy(context)
    if not context.db_path.exists():
        print(
            f"ANJANEYA-ENGINE: no DuckDB database at {context.db_path}; there is nothing to "
            "label yet.\n"
            "  next step: capture a session and build its candles "
            "(python -m apps.anjaneeya_engine.run --mode build-candles)",
            file=sys.stderr,
        )
        return finalize_run(context, status="NO_DATA", exit_code=EXIT_OK)

    status, exit_code = "SUCCESS", EXIT_OK
    connection = connect(context.db_path, read_only=context.dry_run)
    try:
        days = _resolve_days(context, connection, args, days_source=trading_days_with_candles)
        if context.dry_run:
            print(
                "\n".join(
                    _label_plan_lines(
                        context, days=days, timeframe=timeframe, policy=policy, symbols=symbols
                    )
                )
            )
        elif not days:
            print(
                "ANJANEYA-ENGINE: no stored candles found - build a session first "
                "(--mode build-candles).",
                file=sys.stderr,
            )
            status = "NO_DATA"
        else:
            writer = LabelWriter(
                connection=connection,
                parquet_root=context.config.data_dir("labels"),
                run_id=context.run_context.run_id,
                mirror_parquet=context.config.settings.engine.mirror_parquet,
                parquet_compression=context.config.settings.data.parquet_compression,
                logger=context.logger,
            )
            results = [
                _build_label_day(
                    context,
                    connection,
                    writer=writer,
                    day=day,
                    timeframe=timeframe,
                    policy=policy,
                    symbols=symbols,
                )
                for day in days
            ]
            print(_render_label_summary(results, policy=policy, as_json=args.as_json))
            if any(result.failures for result in results):
                status, exit_code = "PARTIAL", EXIT_FAILED
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - report it, then exit non-zero
        context.logger.error(f"build-labels failed: {type(exc).__name__}: {exc}")
        print(f"ANJANEYA-ENGINE build-labels failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        status, exit_code = "FAILED", EXIT_FAILED
    finally:
        connection.close()
    return finalize_run(context, status=status, exit_code=exit_code)


def _run_status(context: AppContext, *, as_json: bool) -> int:
    """Print the offline status report."""
    report = status_report(context)
    print(render_status(report, as_json=as_json))
    return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)


def _run_planned_mode(context: AppContext, mode: str) -> int:
    """Explain that a mode is not implemented yet, and exit with code 3."""
    plan = MODE_PLAN.get(mode, "planned in a later phase")
    context.logger.warning(f"mode {mode!r} is not implemented yet: {plan}")
    print(
        f"ANJANEYA-ENGINE: mode '{mode}' is not implemented in this phase.\n"
        f"  planned work : {plan}\n"
        f"  available now: --mode status  (offline report, no credentials needed)\n"
        f"  see          : docs/phase_reports/ for the current phase report",
        file=sys.stderr,
    )
    return finalize_run(context, status="NOT_IMPLEMENTED", exit_code=EXIT_NOT_IMPLEMENTED)


def _dispatch(context: AppContext, args: argparse.Namespace) -> int:
    """Route a validated context to the mode that was asked for."""
    if context.mode == "status":
        return _run_status(context, as_json=args.as_json)
    if context.mode == "build-candles":
        return _run_build_candles(context, args)
    if context.mode == "enrich-greeks":
        return _run_enrich_greeks(context, args)
    if context.mode == "universe":
        return _run_universe(context, args)
    if context.mode == "build-features":
        return _run_build_features(context, args)
    if context.mode == "build-labels":
        return _run_build_labels(context, args)
    if context.mode == "validate":
        return _run_validate(context, args)
    if context.dry_run:
        plan = MODE_PLAN.get(context.mode, "planned in a later phase")
        context.logger.info(f"dry-run: would execute {context.mode!r} -> {plan}")
        print(f"ANJANEYA-ENGINE dry-run: {context.mode} -> {plan}")
        return finalize_run(context, status="SUCCESS", exit_code=EXIT_OK)
    return _run_planned_mode(context, context.mode)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 1 failed, 2 config, 3 planned)."""
    args = parse_args(
        argv,
        app_name=APP_ENGINE,
        description=DESCRIPTION,
        configure_parser=_configure_engine_parser,
    )
    try:
        context = bootstrap_app(APP_ENGINE, args)
    except (ConfigError, SchemaVersionError) as exc:
        print(f"ANJANEYA-ENGINE configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the CLI
        print(f"ANJANEYA-ENGINE failed during startup: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    try:
        return _dispatch(context, args)
    except ConfigError as exc:
        print(f"ANJANEYA-ENGINE configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # noqa: BLE001 - a crash must still be an exit code
        print(
            f"ANJANEYA-ENGINE failed during {context.mode!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
