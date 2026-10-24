"""Feature validation (spec 19) - the report that stands behind every feature row.

Spec 19 asks for checks on missing values, infinite values, duplicate rows, invalid
OHLC, negative volume, invalid contract identity, timestamp ordering, gaps and -
the one that cannot be bolted on afterwards - **future leakage**. This module is
that report, as a pure function of frames: no I/O, no database, no config file
reads. The CLI (``--mode validate``) owns reading the stored rows and writing the
JSON report.

Design rules
------------
* **Nothing is silently dropped.** A check counts what it looked at, names what it
  found and quotes a few examples; it never deletes a row, never clips a value and
  never repairs anything. Repair is a *new build* of the affected scope.
* **``FAIL`` means "this row is wrong", ``WARN`` means "this row is honest but
  incomplete".** An all-``NULL`` column (the greeks before ``--mode enrich-greeks`` has
  run for that scope, ``nifty_long_score`` on a short lookback) is a ``WARN`` with the
  reason attached; an infinite value, a duplicate key, a broken OHLC bar or a delta with
  the wrong sign for its option type is a ``FAIL``, because no downstream research can
  interpret it.
* **Look-ahead is tested by recomputation, not by inspection.** For a sample of bars
  the whole contract block is recomputed from the window truncated at that bar, and
  the two values must agree. That is the only check that can actually catch an engine
  peeking forward: reading the code cannot.

Importing this module performs no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from core.constants import FEATURE_COLUMNS, PROVENANCE_COLUMNS, timeframe_to_minutes
from core.features.feature_builder import (
    IDENTITY_FEATURE_COLUMNS,
    REQUIRED_CANDLE_COLUMNS,
    build_feature_frame,
    close_moments,
    timeframe_of,
)
from core.schemas import OptionType

__all__ = [
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_SKIPPED",
    "STATUS_WARN",
    "CheckResult",
    "FeatureValidationReport",
    "merge_reports",
    "validate_feature_frame",
]

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED = "SKIPPED"

#: Columns that must never be ``NULL``: without them a row cannot even be identified
#: or traced back to the run that produced it.
ALWAYS_PRESENT_COLUMNS = (*IDENTITY_FEATURE_COLUMNS, *PROVENANCE_COLUMNS)

#: Columns that are legitimately all-``NULL`` today, with the reason. A missing-value
#: warning for one of these says *when* it stops being expected instead of looking
#: like a bug.
NULL_BY_DESIGN_COLUMNS: dict[str, str] = {
    "iv": "not enriched yet: run --mode enrich-greeks (or the price has no IV in the bracket)",
    "iv_change": "needs an IV: run --mode enrich-greeks",
    "delta": "not enriched yet: run --mode enrich-greeks",
    "gamma": "not enriched yet: run --mode enrich-greeks",
    "theta": "not enriched yet: run --mode enrich-greeks",
    "vega": "not enriched yet: run --mode enrich-greeks",
    "expected_option_move": "needs a previous-bar delta and a NIFTY move (--mode enrich-greeks)",
    "option_residual": "needs an expected move (--mode enrich-greeks)",
}

#: Absolute tolerance for the ``option_residual = actual - expected`` identity. It is
#: arithmetic on values that were stored as doubles, so agreement is expected to be
#: exact; the tolerance only absorbs the float round trip through DuckDB/Parquet.
RESIDUAL_TOLERANCE = 1e-6

#: The model's own range for each greek (spec 16). ``None`` means "unbounded on that
#: side"; a boundary value is allowed, so these are closed intervals.
_GREEKS_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "iv": (0.0, 10.0),
    "delta": (-1.0, 1.0),
    "gamma": (0.0, None),
}

#: Column groups excluded from the look-ahead comparison, with the reason.
LEAKAGE_EXEMPT_COLUMNS: dict[str, str] = {
    "option_insufficient_history": (
        "counts bars since the start of the loaded window, so a truncated probe "
        "legitimately changes it (the builder reads a contract from its first bar)"
    ),
}

#: Columns the leakage probe deliberately does not compare: identity and provenance
#: are inputs to the build, not outputs of an engine.
_LEAKAGE_IGNORED_COLUMNS = (*IDENTITY_FEATURE_COLUMNS, *PROVENANCE_COLUMNS)

#: Absolute tolerance for the recomputation comparison. The probe recomputes the same
#: arithmetic on the same inputs, so agreement is expected to be exact; the tolerance
#: only absorbs floating-point noise from a different (shorter) vector length.
PROBE_TOLERANCE = 1e-9


@dataclass(slots=True)
class CheckResult:
    """One spec 19 check and what it found."""

    name: str
    status: str = STATUS_PASS
    checked: int = 0
    issues: int = 0
    detail: str = ""
    samples: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        """True when this check failed (and the rows must not be trusted)."""
        return self.status == STATUS_FAIL

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "name": self.name,
            "status": self.status,
            "checked": self.checked,
            "issues": self.issues,
            "detail": self.detail,
            "samples": list(self.samples),
        }


@dataclass(slots=True)
class FeatureValidationReport:
    """Every check's verdict for one scope (a contract-day, or a whole day)."""

    scope: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    @property
    def failures(self) -> list[CheckResult]:
        """Checks that failed."""
        return [check for check in self.checks if check.status == STATUS_FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        """Checks that passed with a warning."""
        return [check for check in self.checks if check.status == STATUS_WARN]

    @property
    def status(self) -> str:
        """``FAIL`` > ``PASS_WITH_WARNINGS`` > ``PASS`` (``SKIPPED`` alone is a ``FAIL``)."""
        if self.failures:
            return STATUS_FAIL
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        if not self.checks or all(check.status == STATUS_SKIPPED for check in self.checks):
            return STATUS_FAIL
        return STATUS_PASS

    @property
    def passed(self) -> bool:
        """True when nothing blocks: no ``FAIL`` and at least one check actually ran."""
        return self.status != STATUS_FAIL

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape (deterministic: no timestamps, no paths)."""
        return {
            "scope": dict(self.scope),
            "status": self.status,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "checks": [check.to_dict() for check in self.checks],
        }

    def summary_line(self) -> str:
        """One greppable line: status plus the counts of each verdict."""
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        return f"{self.status}: {self.row_count} row(s); {rendered}"

# ---------------------------------------------------------------------------
# Individual checks. Each one is pure and returns its own verdict, so the report
# can name exactly what was looked at.
# ---------------------------------------------------------------------------
def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame's numeric columns only (booleans and labels are not numbers here)."""
    return frame.select_dtypes(include=["number"])


def _check_unknown_columns(frame: pd.DataFrame) -> CheckResult:
    """The frame may not carry a column the feature contract does not declare."""
    unknown = [str(column) for column in frame.columns if column not in FEATURE_COLUMNS]
    result = CheckResult(name="declared_columns", checked=len(frame.columns), issues=len(unknown))
    if unknown:
        result.status = STATUS_FAIL
        result.detail = (
            f"column(s) {unknown} are not part of FEATURE_COLUMNS; a feature table with "
            "undeclared columns cannot be reproduced from the contract"
        )
        result.samples = unknown[:10]
    else:
        result.detail = f"every column is declared ({len(frame.columns)} of {len(FEATURE_COLUMNS)})"
    return result


def _check_always_present(frame: pd.DataFrame) -> CheckResult:
    """Identity and provenance must never be ``NULL``."""
    present = [column for column in ALWAYS_PRESENT_COLUMNS if column in frame.columns]
    issues: list[str] = []
    count = 0
    for column in present:
        blank = int(frame[column].isna().sum())
        if blank:
            issues.append(f"{column}={blank}")
            count += blank
    result = CheckResult(name="identity_present", checked=len(present), issues=count)
    if issues:
        result.status = STATUS_FAIL
        result.detail = f"NULL in a column that must always be present: {', '.join(issues)}"
        result.samples = issues[:10]
    else:
        result.detail = f"{len(present)} identity/provenance column(s) fully populated"
    return result


def _check_duplicate_keys(frame: pd.DataFrame) -> CheckResult:
    """``(timestamp, tradingsymbol)`` is the primary key: a duplicate is a hard bug."""
    keys = [column for column in ("timestamp", "tradingsymbol") if column in frame.columns]
    if len(keys) != 2:
        return CheckResult(
            name="unique_keys",
            status=STATUS_SKIPPED,
            detail="the frame does not carry timestamp and tradingsymbol",
        )
    duplicated = frame.duplicated(subset=keys, keep=False)
    count = int(duplicated.sum())
    result = CheckResult(name="unique_keys", checked=len(frame), issues=count)
    if count:
        result.status = STATUS_FAIL
        result.detail = f"{count} row(s) duplicate (timestamp, tradingsymbol)"
        result.samples = [
            f"{row.timestamp} {row.tradingsymbol}"
            for row in frame[duplicated].head(5).itertuples()
        ]
    else:
        result.detail = "every (timestamp, tradingsymbol) appears exactly once"
    return result


def _check_timestamp_ordering(frame: pd.DataFrame) -> CheckResult:
    """Time must move forward: every engine reads "its own and earlier bars"."""
    if "timestamp" not in frame.columns or frame.empty:
        return CheckResult(name="timestamp_ordering", status=STATUS_SKIPPED, detail="no timestamps")
    stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    blanks = int(stamps.isna().sum())
    backwards = int((stamps.diff().dropna() <= timedelta(0)).sum())
    result = CheckResult(
        name="timestamp_ordering", checked=len(frame), issues=blanks + backwards
    )
    if blanks or backwards:
        result.status = STATUS_FAIL
        result.detail = (
            f"{blanks} unparseable timestamp(s) and {backwards} non-increasing step(s)"
        )
    else:
        result.detail = "timestamps strictly increase"
    return result


def _check_missing_values(frame: pd.DataFrame) -> CheckResult:
    """Per-column null census. Entirely-empty columns are a warning, with the reason."""
    counts = {str(column): int(frame[column].isna().sum()) for column in frame.columns}
    empty = [column for column, blank in counts.items() if blank == len(frame) and len(frame)]
    expected = [column for column in empty if column in NULL_BY_DESIGN_COLUMNS]
    unexpected = [column for column in empty if column not in NULL_BY_DESIGN_COLUMNS]
    result = CheckResult(
        name="missing_values",
        checked=len(frame) * len(frame.columns),
        issues=sum(counts.values()),
    )
    notes: list[str] = []
    if expected:
        notes.append(
            "empty by design: "
            + ", ".join(f"{column} ({NULL_BY_DESIGN_COLUMNS[column]})" for column in expected)
        )
    if unexpected:
        notes.append("empty for the whole scope: " + ", ".join(unexpected[:10]))
    result.detail = "; ".join(notes) if notes else "no column is empty for the whole scope"
    result.samples = [f"{column}={blank}" for column, blank in counts.items() if blank]
    result.status = STATUS_WARN if empty else STATUS_PASS
    return result


def _check_infinite_values(frame: pd.DataFrame) -> CheckResult:
    """``inf`` is never a measurement: it is a divide-by-zero that got away."""
    numeric = _numeric(frame)
    offenders: list[str] = []
    count = 0
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce").to_numpy(dtype="float64")
        blanks = int(np.isinf(values).sum())
        if blanks:
            offenders.append(f"{column}={blanks}")
            count += blanks
    result = CheckResult(name="infinite_values", checked=int(numeric.size), issues=count)
    if count:
        result.status = STATUS_FAIL
        result.detail = f"infinite value(s) in {', '.join(offenders)}"
        result.samples = offenders[:10]
    else:
        result.detail = f"no infinite value in {numeric.shape[1]} numeric column(s)"
    return result


def _check_ohlc(candles: pd.DataFrame | None) -> CheckResult:
    """The input bars a feature row was computed from must be sane OHLC."""
    if candles is None or candles.empty:
        return CheckResult(name="ohlc_sanity", status=STATUS_SKIPPED, detail="no candle input")
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        return CheckResult(
            name="ohlc_sanity",
            status=STATUS_SKIPPED,
            detail=f"the candle input is missing {missing}",
        )
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    open_ = pd.to_numeric(candles["open"], errors="coerce")
    close = pd.to_numeric(candles["close"], errors="coerce")
    broken = (high < low) | (open_ > high) | (open_ < low) | (close > high) | (close < low)
    count = int(broken.sum())
    result = CheckResult(name="ohlc_sanity", checked=len(candles), issues=count)
    if count:
        result.status = STATUS_FAIL
        result.detail = f"{count} bar(s) have an impossible OHLC relationship"
        result.samples = [
            f"{row.timestamp} o={row.open} h={row.high} l={row.low} c={row.close}"
            for row in candles[broken].head(5).itertuples()
        ]
    else:
        result.detail = "every bar satisfies low <= open/close <= high"
    return result


def _check_volume(candles: pd.DataFrame | None) -> CheckResult:
    """Negative volume or open interest cannot exist on a real contract."""
    if candles is None or candles.empty:
        return CheckResult(name="volume_sanity", status=STATUS_SKIPPED, detail="no candle input")
    columns = [column for column in ("volume", "oi") if column in candles.columns]
    if not columns:
        return CheckResult(
            name="volume_sanity", status=STATUS_SKIPPED, detail="the candle input carries no volume/oi"
        )
    offenders: list[str] = []
    count = 0
    for column in columns:
        values = pd.to_numeric(candles[column], errors="coerce")
        negative = int((values < 0).sum())
        if negative:
            offenders.append(f"{column}={negative}")
            count += negative
    result = CheckResult(name="volume_sanity", checked=len(candles) * len(columns), issues=count)
    if count:
        result.status = STATUS_FAIL
        result.detail = f"negative value(s) in {', '.join(offenders)}"
        result.samples = offenders
    else:
        result.detail = f"no negative value in {', '.join(columns)}"
    return result


def _check_greeks(frame: pd.DataFrame) -> CheckResult:
    """The stored greeks must sit inside the model's own range (spec 16 / 19).

    These are *not* market checks. An IV cannot be negative, a call's delta cannot be
    negative, a put's cannot be positive and gamma cannot be negative for a long option
    under the model this platform uses - so a violation means the numbers were computed
    wrongly, not that the market did something unusual. That is a ``FAIL``.

    When every greek column is ``NULL`` the check degrades to a ``WARN`` naming the mode
    that fills them: an unenriched feature table is incomplete, not corrupt.
    """
    columns = [name for name in ("iv", "delta", "gamma", "theta", "vega") if name in frame.columns]
    if not columns:
        return CheckResult(name="greeks_sanity", status=STATUS_SKIPPED, detail="no greek columns")
    values = {name: pd.to_numeric(frame[name], errors="coerce") for name in columns}
    present = {name: series for name, series in values.items() if bool(series.notna().any())}
    if not present:
        return CheckResult(
            name="greeks_sanity",
            status=STATUS_WARN,
            checked=len(frame),
            detail="every greek column is NULL for this scope: run --mode enrich-greeks",
        )

    issues: list[str] = []
    samples: list[str] = []
    count = 0
    for name, series in present.items():
        low, high = _GREEKS_BOUNDS.get(name, (None, None))
        outside = pd.Series(False, index=frame.index)
        if low is not None:
            outside |= series < low
        if high is not None:
            outside |= series > high
        bad = int(outside.sum())
        if bad:
            issues.append(f"{name} outside [{low}, {high}] in {bad} row(s)")
            count += bad
            samples.extend(str(value) for value in series[outside].head(3))

    typed = None
    if "delta" in present and "option_type" in frame.columns:
        types = frame["option_type"].astype("string").str.upper()
        delta = present["delta"]
        wrong_type = ((types == "CE") & (delta < -1e-9)) | ((types == "PE") & (delta > 1e-9))
        typed = int(wrong_type.sum())
        if typed:
            issues.append(f"delta has the wrong sign for the option_type in {typed} row(s)")
            count += typed
            samples.extend(
                f"{row.timestamp} {row.option_type} delta={row.delta}"
                for row in frame[wrong_type].head(3).itertuples()
            )

    result = CheckResult(
        name="greeks_sanity", checked=int(len(frame) * len(present)), issues=count
    )
    if count:
        result.status = STATUS_FAIL
        result.detail = "; ".join(issues)
        result.samples = samples[:10]
    else:
        result.detail = (
            f"{len(present)} greek column(s) inside their model range"
            + (f" and consistent with option_type ({typed} mismatch(es))" if typed is not None else "")
        )
    return result


def _check_relative_identity(frame: pd.DataFrame) -> CheckResult:
    """``option_residual`` must be exactly ``actual_option_move - expected_option_move``.

    The three relative columns are defined by that identity (spec 16), so a row that
    breaks it was either computed by a different rule or edited afterwards. Either way no
    downstream study can interpret it, which makes it a ``FAIL`` rather than a warning.
    """
    columns = ("actual_option_move", "expected_option_move", "option_residual")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        return CheckResult(
            name="relative_identity", status=STATUS_SKIPPED, detail=f"missing column(s) {missing}"
        )
    actual = pd.to_numeric(frame["actual_option_move"], errors="coerce")
    expected = pd.to_numeric(frame["expected_option_move"], errors="coerce")
    residual = pd.to_numeric(frame["option_residual"], errors="coerce")
    complete = actual.notna() & expected.notna() & residual.notna()
    checked = int(complete.sum())
    if not checked:
        return CheckResult(
            name="relative_identity",
            status=STATUS_WARN,
            checked=len(frame),
            detail=(
                "no row carries all three relative columns: expected move/residual need a "
                "previous-bar delta (run --mode enrich-greeks)"
            ),
        )
    broken = complete & ((actual - expected - residual).abs() > RESIDUAL_TOLERANCE)
    count = int(broken.sum())
    result = CheckResult(name="relative_identity", checked=checked, issues=count)
    if count:
        result.status = STATUS_FAIL
        result.detail = (
            f"{count} row(s) where option_residual != actual_option_move - expected_option_move"
        )
        result.samples = [
            f"{row.timestamp} {row.tradingsymbol}: actual={row.actual_option_move} "
            f"expected={row.expected_option_move} residual={row.option_residual}"
            for row in frame[broken].head(5).itertuples()
        ]
    else:
        result.detail = f"residual = actual - expected on all {checked} complete row(s)"
    return result


def _check_contract_identity(frame: pd.DataFrame) -> CheckResult:
    """Spec 6.4: a feature row must identify its contract, and it must be live."""
    checked = 0
    issues: list[str] = []
    count = 0

    if "strike" in frame.columns:
        checked += 1
        bad = int((pd.to_numeric(frame["strike"], errors="coerce") <= 0).sum())
        count += bad
        if bad:
            issues.append(f"strike<=0 in {bad} row(s)")

    if "option_type" in frame.columns:
        checked += 1
        allowed = {OptionType.CE.value, OptionType.PE.value}
        bad = int((~frame["option_type"].isin(allowed)).sum())
        count += bad
        if bad:
            issues.append(f"option_type outside {sorted(allowed)} in {bad} row(s)")

    if "instrument_token" in frame.columns:
        checked += 1
        bad = int((pd.to_numeric(frame["instrument_token"], errors="coerce") <= 0).sum())
        count += bad
        if bad:
            issues.append(f"instrument_token<=0 in {bad} row(s)")

    if "expiry" in frame.columns and "timestamp" in frame.columns:
        checked += 1
        expiry = pd.to_datetime(frame["expiry"], errors="coerce")
        stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        expired = (expiry.dt.tz_localize(None).dt.normalize() < stamps.dt.tz_localize(None).dt.normalize())
        bad = int((expiry.isna() | expired).sum())
        count += bad
        if bad:
            issues.append(f"missing or already-expired expiry in {bad} row(s)")

    result = CheckResult(name="contract_identity", checked=checked, issues=count)
    if count:
        result.status = STATUS_FAIL
        result.detail = "; ".join(issues)
        result.samples = issues[:10]
    else:
        result.detail = "strike, option_type, instrument_token and expiry all valid"
    return result


def _check_gaps(
    frame: pd.DataFrame, *, timeframe: str | None, session_start: datetime | None, session_end: datetime | None
) -> CheckResult:
    """Missing bars inside the session are a **data** fact, so they warn, never fail.

    A gap is reported, not filled: the coverage ledger already carries the per-stream
    verdict, and this check exists so a validation report cannot look clean while the
    session was only half built.
    """
    if timeframe is None or session_start is None or session_end is None or "timestamp" not in frame:
        return CheckResult(
            name="session_gaps",
            status=STATUS_SKIPPED,
            detail="needs timeframe plus session_start/session_end to know the expectation",
        )
    minutes = timeframe_to_minutes(timeframe)
    present = set(pd.to_datetime(frame["timestamp"]).tolist())
    moment = session_start
    expected = 0
    missing = 0
    while moment < session_end:
        expected += 1
        if moment not in present:
            missing += 1
        moment += timedelta(minutes=minutes)
    result = CheckResult(name="session_gaps", checked=expected, issues=missing)
    if missing:
        result.status = STATUS_WARN
        result.detail = (
            f"{missing} of {expected} expected {timeframe} bar(s) are missing from this scope; "
            "the bars are absent rather than fabricated (see data_coverage)"
        )
    else:
        result.detail = f"all {expected} expected {timeframe} bar(s) present"
    return result


def _is_missing(value: Any) -> bool:
    """True for ``None``, ``NaN``, ``NaT`` and pandas' nullable ``NA``."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic object dtype
        return False


def _same_value(left: Any, right: Any) -> bool:
    """Equality that treats ``None``/``NaN``/``NA`` as equal and compares numerics loosely."""
    if _is_missing(left) or _is_missing(right):
        return _is_missing(left) and _is_missing(right)
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        if np.isinf(left) or np.isinf(right):
            return bool(left == right)
        return bool(abs(float(left) - float(right)) <= PROBE_TOLERANCE)
    return bool(left == right)


def _probe_positions(count: int, positions: Sequence[int], probes: int) -> list[int]:
    """Up to ``probes`` evenly spread bar positions across the scope."""
    if not positions or probes <= 0:
        return []
    if len(positions) <= probes:
        return list(positions)
    step = len(positions) / probes
    picked = {positions[int(index * step)] for index in range(probes)}
    return sorted(picked)


def _check_look_ahead(
    frame: pd.DataFrame,
    *,
    candles: pd.DataFrame | None,
    index_candles: pd.DataFrame | None,
    calendar: Any,
    config: Any,
    timeframe: str | None,
    session_start: datetime | None,
    session_end: datetime | None,
    probes: int,
) -> CheckResult:
    """Recompute a sample of bars from a truncated window: the values must not move.

    This is the check that cannot be faked. If any engine reads a later bar - directly
    or through an index series joined on the wrong key - the recomputed value differs
    and the row is reported with both numbers.
    """
    if candles is None or candles.empty:
        return CheckResult(
            name="no_look_ahead", status=STATUS_SKIPPED, detail="no candle input to recompute from"
        )
    if config is None or calendar is None or timeframe is None:
        return CheckResult(
            name="no_look_ahead",
            status=STATUS_SKIPPED,
            detail="needs config, calendar and timeframe to recompute the row",
        )
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        return CheckResult(
            name="no_look_ahead", status=STATUS_SKIPPED, detail=f"candles are missing {missing}"
        )

    minutes = timeframe_to_minutes(timeframe)
    stamps = pd.to_datetime(candles["timestamp"])
    inside = [
        position
        for position, moment in enumerate(stamps)
        if (session_start is None or moment >= session_start)
        and (session_end is None or moment < session_end)
    ]
    stored_stamps = pd.to_datetime(frame["timestamp"]) if "timestamp" in frame.columns else None
    positions = [
        position
        for position in _probe_positions(len(candles), inside, probes)
        if stored_stamps is not None and bool((stored_stamps == stamps.iloc[position]).any())
    ]
    if not positions:
        return CheckResult(
            name="no_look_ahead",
            status=STATUS_SKIPPED,
            detail="no probed bar exists in the validated frame",
        )

    index_timeframe = (
        timeframe
        if index_candles is None or index_candles.empty
        else timeframe_of(index_candles, default=timeframe)
    )
    mismatches: list[str] = []
    for position in positions:
        moment = stamps.iloc[position]
        stored = frame.loc[stored_stamps == moment].iloc[0]
        if index_candles is None or len(index_candles) == 0:
            truncated_index = index_candles
        else:
            limit = moment + timedelta(minutes=minutes)
            truncated_index = index_candles[close_moments(index_candles, timeframe=index_timeframe) <= limit]
        try:
            rebuilt = build_feature_frame(
                candles=candles.iloc[: position + 1],
                index_candles=truncated_index,
                calendar=calendar,
                config=config,
                timeframe=timeframe,
                underlying=str(stored["underlying"]),
                run_id=str(stored["run_id"]),
                session_start=None,
                session_end=None,
            )
        except Exception as exc:  # noqa: BLE001 - validation reports, it never crashes
            mismatches.append(f"{moment}: recomputing the row raised {type(exc).__name__}: {exc}")
            continue
        if rebuilt.empty:
            mismatches.append(f"{moment}: the truncated window produced no row")
            continue
        fresh = rebuilt.iloc[-1]
        for column in FEATURE_COLUMNS:
            if column in _LEAKAGE_IGNORED_COLUMNS or column in LEAKAGE_EXEMPT_COLUMNS:
                continue
            if not _same_value(stored[column], fresh[column]):
                mismatches.append(f"{moment} {column}: stored={stored[column]!r} rebuilt={fresh[column]!r}")

    result = CheckResult(name="no_look_ahead", checked=len(positions), issues=len(mismatches))
    if mismatches:
        result.status = STATUS_FAIL
        result.detail = (
            f"{len(mismatches)} feature value(s) changed when the window was truncated at the "
            "probed bar: something reads a later row"
        )
        result.samples = mismatches[:10]
    else:
        exempt = ", ".join(sorted(LEAKAGE_EXEMPT_COLUMNS)) or "none"
        result.detail = (
            f"{len(positions)} bar(s) recomputed from a truncated window are identical "
            f"(exempt: {exempt})"
        )
    return result


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
_STATUS_RANK = {STATUS_PASS: 0, STATUS_SKIPPED: 1, STATUS_WARN: 2, STATUS_FAIL: 3}


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    candles: pd.DataFrame | None = None,
    index_candles: pd.DataFrame | None = None,
    calendar: Any = None,
    config: Any = None,
    timeframe: str | None = None,
    session_start: datetime | None = None,
    session_end: datetime | None = None,
    probes: int = 3,
    scope: dict[str, Any] | None = None,
) -> FeatureValidationReport:
    """Run every spec 19 check over one scope's feature rows.

    ``frame`` is what is being validated - the assembled frame for a fresh build, or
    the rows read back out of DuckDB. ``candles``/``index_candles``/``calendar``/
    ``config`` are the *inputs* the rows were computed from; they are optional in the
    sense that a missing one skips its check loudly (``SKIPPED``, with the reason) -
    never silently.
    """
    report = FeatureValidationReport(
        scope=dict(scope or {}), row_count=len(frame), column_count=len(frame.columns)
    )
    if frame.empty:
        report.checks = [
            CheckResult(
                name="rows_present",
                status=STATUS_FAIL,
                checked=0,
                issues=1,
                detail="the scope holds no feature rows; there is nothing to validate",
            )
        ]
        return report

    report.checks = [
        _check_unknown_columns(frame),
        _check_always_present(frame),
        _check_duplicate_keys(frame),
        _check_timestamp_ordering(frame),
        _check_missing_values(frame),
        _check_infinite_values(frame),
        _check_contract_identity(frame),
        _check_greeks(frame),
        _check_relative_identity(frame),
        _check_ohlc(candles),
        _check_volume(candles),
        _check_gaps(
            frame,
            timeframe=timeframe,
            session_start=session_start,
            session_end=session_end,
        ),
        _check_look_ahead(
            frame,
            candles=candles,
            index_candles=index_candles,
            calendar=calendar,
            config=config,
            timeframe=timeframe,
            session_start=session_start,
            session_end=session_end,
            probes=probes,
        ),
    ]
    return report


def merge_reports(
    reports: Iterable[FeatureValidationReport], *, scope: dict[str, Any] | None = None
) -> FeatureValidationReport:
    """Combine per-contract reports into one verdict for a day (or a whole run).

    Checks are matched by name: counters add up, samples are kept (bounded), and the
    status is the *worst* one seen - one contract with a duplicate key fails the day.
    """
    merged = FeatureValidationReport(scope=dict(scope or {}))
    combined: dict[str, CheckResult] = {}
    for report in reports:
        merged.row_count += report.row_count
        merged.column_count = max(merged.column_count, report.column_count)
        for check in report.checks:
            current = combined.get(check.name)
            if current is None:
                combined[check.name] = CheckResult(
                    name=check.name,
                    status=check.status,
                    checked=check.checked,
                    issues=check.issues,
                    detail=check.detail,
                    samples=list(check.samples),
                )
                continue
            current.checked += check.checked
            current.issues += check.issues
            if _STATUS_RANK.get(check.status, 0) > _STATUS_RANK.get(current.status, 0):
                current.status = check.status
                current.detail = check.detail
                current.samples = list(check.samples)
            elif check.detail and check.detail != current.detail and check.status == current.status:
                current.detail = f"{current.detail}; {check.detail}"
            if check.issues:
                current.samples = (current.samples + check.samples)[:10]
    merged.checks = [combined[name] for name in sorted(combined)]
    return merged
