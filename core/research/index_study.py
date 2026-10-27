"""P1 - does the NIFTY signal separate forward returns? (index study)

The question this answers, before any model is trained:

    Do the configured MA-combination regimes, the composite score, distance to the
    previous session's levels, time of day and day of week actually separate the
    **forward** returns of the NIFTY index - with a sample big enough to matter?

Why it exists
-------------
The option model needs option data. The *index* signal, however, does not: two years
of daily bars and the stored 1-minute history are enough to ask whether that signal
carries information at all. If it does not separate forward returns here, a model
built on top of it will fit noise; this study is the cheap test that says so first.

What it is not
--------------
* **Not a model.** No training, no probabilities, no artefact under ``data/models``.
* **Not a strategy.** Returns are gross: no costs, no slippage, no position sizing.
* **Not advice.** A row says "this bucket averaged +0.12 ATR over 5 days with n=340";
  it never says to buy anything.

Honesty rules (the same ones the rest of the project holds)
----------------------------------------------------------
1. **No test without a sample.** A cell below ``preliminary_min_samples`` is
   ``INSUFFICIENT_SAMPLE`` and carries no p-value.
2. **The correction is applied to the tests actually performed.** One axis (regime,
   score band, ...) is one family; the family size is printed next to every adjusted
   p-value, using the same helpers as the trade report.
3. **A missing bucket is absent, never zero.** A regime with no bars in the window
   does not appear as a row of zeros.
4. **Intraday returns never cross the close.** A bar whose horizon does not fit inside
   its own session has no label, exactly like :mod:`core.labels.label_builder`.
5. **The caveats travel with the numbers** (overlapping horizons, gappy 1m bars,
   regime persistence, gross returns).

Importing this module performs no I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.features.feature_builder import nifty_feature_block
from core.indicators.atr import atr as atr_series
from core.research.significance import (
    INSUFFICIENT_SAMPLE,
    NOT_SIGNIFICANT,
    SIGNIFICANT,
    adjust_p_values,
    mean_pnl_p_value,
    win_rate_p_value,
)
from core.storage import read_index_history

__all__ = [
    "DAILY_HORIZONS",
    "INDEX_STUDY_SPEC",
    "IndexStudyPolicy",
    "StudyAxis",
    "build_index_study",
    "forward_returns",
    "render_study_markdown",
    "study_stem",
    "study_wide_correction",
    "write_index_study",
]

#: Where the study's artefacts go, under ``data/reports``.
INDEX_STUDY_SPEC = "P1 - NIFTY index forward-return study"

#: Daily forward horizons, in trading days (close-to-close).
DAILY_HORIZONS: tuple[int, ...] = (1, 2, 5, 10)

#: The project's own intraday buckets (``config/calendar.yaml`` names them the same
#: way); edges are minutes after the 09:15 open, so the session clock is the only input.
INTRADAY_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("OPEN_915_930", 0, 15),
    ("MORNING_930_1100", 15, 105),
    ("MIDDAY_1100_1300", 105, 225),
    ("AFTERNOON_1300_1430", 225, 315),
    ("CLOSE_1430_1515", 315, 360),
)

#: Composite-score bands used as an axis (fixed and interpretable, not data-fitted).
SCORE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("< -0.50", -1.01, -0.50),
    ("-0.50 .. -0.10", -0.50, -0.10),
    ("-0.10 .. +0.10", -0.10, 0.10),
    ("+0.10 .. +0.50", 0.10, 0.50),
    (">= +0.50", 0.50, 1.01),
)

#: The MA combination whose regime the stored features carry; ``nifty_feature_block``
#: defaults to it, so the study and the feature table read the same score.
DEFAULT_COMBINATION = "combination_1"

#: Distance to the nearest previous-session level, in ATR units.
PROXIMITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("< 0.25 ATR", 0.0, 0.25),
    ("0.25 - 0.5 ATR", 0.25, 0.5),
    ("0.5 - 1.0 ATR", 0.5, 1.0),
    (">= 1.0 ATR", 1.0, 1.0e9),
)


@dataclass(frozen=True, slots=True)
class IndexStudyPolicy:
    """What the study measures (``config/model.yaml`` + ``config/indicators.yaml``)."""

    #: Which MA combination defines the regime (``combination_1`` by default).
    combination: str
    #: Multiple-testing correction, shared with the trade report's vocabulary.
    correction: str
    #: Below this, a cell is not tested at all.
    min_samples: int
    #: Forward horizons, in bars of the studied timeframe.
    intraday_horizons: tuple[int, ...]
    daily_horizons: tuple[int, ...]

    @classmethod
    def from_config(cls, config: Any) -> IndexStudyPolicy:
        """Build the policy from the **top-level** configuration (no magic numbers)."""
        model = config.model
        evaluation = getattr(model, "evaluation", None)
        return cls(
            combination=DEFAULT_COMBINATION,
            correction=str(
                getattr(evaluation, "multiple_testing_correction", "benjamini_hochberg") or "none"
            ),
            min_samples=int(getattr(evaluation, "preliminary_min_samples", 30)),
            intraday_horizons=tuple(int(item) for item in model.horizons_minutes),
            daily_horizons=DAILY_HORIZONS,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "combination": self.combination,
            "correction": self.correction,
            "min_samples": self.min_samples,
            "intraday_horizons": list(self.intraday_horizons),
            "daily_horizons": list(self.daily_horizons),
        }


@dataclass(slots=True)
class StudyAxis:
    """One axis (regime, score band, ...) at one horizon, with its tests."""

    name: str
    horizon: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    family_size: int = 0
    correction: str = "none"
    baseline: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "axis": self.name,
            "horizon": self.horizon,
            "correction": self.correction,
            "family_size": self.family_size,
            "baseline": self.baseline,
            "rows": self.rows,
        }


def study_stem(first: Any, last: Any) -> str:
    """Artefact stem: the studied window, so two runs never overwrite each other."""
    return f"index_study_{first}_{last}"


def index_frame(connection: Any, *, symbol: str, timeframe: str) -> pd.DataFrame:
    """The index's own bars at one timeframe, oldest first, with a trading date."""
    frame = read_index_history(connection, symbols=[symbol])
    if frame.empty or "timeframe" not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame["timeframe"] == timeframe]
    if frame.empty:
        return pd.DataFrame()
    working = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    working["trading_date"] = pd.to_datetime(working["timestamp"]).dt.date
    return working


def _band(values: pd.Series, bands: tuple[tuple[str, float, float], ...]) -> pd.Series:
    """Label values by half-open ``[low, high)`` bands; ``NaN`` stays ``None``."""
    labels = pd.Series([None] * len(values), index=values.index, dtype="object")
    for name, low, high in bands:
        mask = (values >= low) & (values < high)
        labels[mask] = name
    return labels


def _intraday_bucket(stamps: pd.Series) -> pd.Series:
    """The project's own intraday buckets, from the session clock (minutes after 09:15)."""
    parsed = pd.to_datetime(stamps)
    minutes = (parsed.dt.hour - 9) * 60 + (parsed.dt.minute - 15)
    labels = pd.Series([None] * len(stamps), index=stamps.index, dtype="object")
    for name, low, high in INTRADAY_BUCKETS:
        labels[(minutes >= low) & (minutes < high)] = name
    return labels


def _with_signal(frame: pd.DataFrame, *, indicators: Any, policy: IndexStudyPolicy) -> pd.DataFrame:
    """Attach the NIFTY block (composite/regime/…) plus ATR and the score band.

    ``indicators`` is ``config.indicators`` - the same object the feature builder is
    handed, so the study's regime is by construction the regime the rows carry.
    """
    block = nifty_feature_block(frame, config=indicators, combination=policy.combination)
    working = pd.concat([frame.reset_index(drop=True), block.reset_index(drop=True)], axis=1)
    working["atr"] = atr_series(
        working["high"],
        working["low"],
        working["close"],
        period=indicators.atr.period,
        method=indicators.atr.method,
    )
    working["score_band"] = _band(working["nifty_composite_trend_score"], SCORE_BANDS)
    return working


def forward_returns(
    frame: pd.DataFrame, horizons: tuple[int, ...], *, intraday: bool
) -> pd.DataFrame:
    """Forward simple returns (percent and ATR units) for each horizon.

    ``intraday`` computes each horizon **inside its own trading date**, so a bar whose
    window would cross the close has no return - the same rule the label builder
    applies to option bars. Daily horizons are close-to-close over ``h`` sessions.
    """
    close = frame["close"].astype(float)
    ranges = frame["atr"].astype(float)
    columns: dict[str, pd.Series] = {}
    for horizon in horizons:
        future = (
            frame.groupby("trading_date")["close"].shift(-horizon)
            if intraday
            else close.shift(-horizon)
        )
        columns[f"ret_{horizon}"] = (future / close - 1.0) * 100.0
        columns[f"atr_{horizon}"] = (future - close) / ranges
    return pd.DataFrame(columns, index=frame.index)


#: The regime vocabulary, bearish to bullish - the order the table reads in.
REGIME_ORDER: tuple[str, ...] = (
    "STRONG_BEARISH",
    "BEARISH",
    "BEARISH_DEVELOPING",
    "NEUTRAL",
    "BULLISH_DEVELOPING",
    "BULLISH",
    "STRONG_BULLISH",
)


def _verdict(sample: int, p_hit: float | None, p_mean: float | None, *, min_samples: int) -> str:
    """INSUFFICIENT_SAMPLE / SIGNIFICANT / NOT_SIGNIFICANT, from the corrected p-values."""
    if sample < min_samples:
        return INSUFFICIENT_SAMPLE
    candidates = [value for value in (p_hit, p_mean) if value is not None]
    if not candidates:
        return "NOT_TESTABLE"
    return SIGNIFICANT if min(candidates) < 0.05 else NOT_SIGNIFICANT


def _family(
    *,
    axis_name: str,
    labels: pd.Series,
    values: pd.Series,
    values_atr: pd.Series,
    horizon_label: str,
    policy: IndexStudyPolicy,
    order: tuple[str, ...] | None = None,
) -> StudyAxis:
    """Test every bucket of one axis against everything outside it.

    The mean test runs on **ATR-normalised** returns, so a cell measured in a quiet
    regime and one measured in a volatile regime are comparable; the percent figures
    are printed for the reader, not tested.
    """
    axis = StudyAxis(name=axis_name, horizon=horizon_label, correction=policy.correction)
    present = values.notna()
    usable = present & labels.notna()
    baseline = values[present]
    axis.baseline = {
        "sample": int(len(baseline)),
        "mean_pct": None if baseline.empty else float(baseline.mean()),
        "hit_rate": None if baseline.empty else float((baseline > 0).mean()),
    }

    candidates = (
        list(order)
        if order is not None
        else sorted(str(item) for item in labels.dropna().unique())
    )
    rows: list[dict[str, Any]] = []
    for name in candidates:
        cell_mask = usable & (labels == name)
        cell = values[cell_mask].dropna()
        if cell.empty:
            continue  # absent, never a row of zeros
        cell_atr = values_atr[cell_mask].dropna()
        rest_mask = present & (labels != name)
        rest = values[rest_mask].dropna()
        rest_atr = values_atr[rest_mask].dropna()
        wins = int((cell > 0).sum())
        losses = int((cell <= 0).sum())
        rest_wins = int((rest > 0).sum())
        rest_losses = int((rest <= 0).sum())
        rows.append(
            {
                "bucket": name,
                "sample": int(len(cell)),
                "hit_rate": wins / len(cell),
                "mean_pct": float(cell.mean()),
                "median_pct": float(cell.median()),
                "mean_atr": None if cell_atr.empty else float(cell_atr.mean()),
                "rest_sample": int(len(rest)),
                "rest_mean_pct": None if rest.empty else float(rest.mean()),
                "rest_hit_rate": None if rest.empty else float((rest > 0).mean()),
                "_p_hit": win_rate_p_value(wins, losses, rest_wins, rest_losses),
                "_p_mean": mean_pnl_p_value(cell_atr.tolist(), rest_atr.tolist()),
            }
        )

    tested = [row for row in rows if row["sample"] >= policy.min_samples]
    axis.family_size = len(tested)
    adjusted_hit = adjust_p_values([row["_p_hit"] for row in tested], policy.correction)
    adjusted_mean = adjust_p_values([row["_p_mean"] for row in tested], policy.correction)
    for row, p_hit, p_mean in zip(tested, adjusted_hit, adjusted_mean, strict=True):
        row["p_hit"] = p_hit
        row["p_mean"] = p_mean
        row["verdict"] = _verdict(
            row["sample"], p_hit, p_mean, min_samples=policy.min_samples
        )
    for row in rows:
        if row["sample"] < policy.min_samples:
            row["p_hit"] = None
            row["p_mean"] = None
            row["verdict"] = _verdict(
                row["sample"], None, None, min_samples=policy.min_samples
            )
        # The raw p-values are kept: the study-wide correction re-tests every cell as
        # one family, and the JSON should let a reader reproduce that.
        row["p_hit_raw"] = row.pop("_p_hit")
        row["p_mean_raw"] = row.pop("_p_mean")
    axis.rows = rows
    return axis


def study_wide_correction(sections: dict[str, dict], *, policy: IndexStudyPolicy) -> int:
    """Correct across *all* the study's tested cells, not within each axis.

    Every axis is small (4-7 buckets) but the study runs dozens of them; correcting
    family by family would still leave ~30 independent 5% chances. This pass is the
    headline verdict - it adds ``p_hit_study``/``p_mean_study``/``verdict_study`` to
    each tested row and returns the size of the family it corrected over.
    """
    rows = [
        row
        for section in sections.values()
        if section.get("available")
        for axis in section.get("axes", [])
        for row in axis.get("rows", [])
        if row.get("p_hit_raw") is not None
    ]
    if not rows:
        return 0
    hit = adjust_p_values([row["p_hit_raw"] for row in rows], policy.correction)
    mean = adjust_p_values([row["p_mean_raw"] for row in rows], policy.correction)
    for row, p_hit, p_mean in zip(rows, hit, mean, strict=True):
        row["p_hit_study"] = p_hit
        row["p_mean_study"] = p_mean
        row["verdict_study"] = _verdict(
            row["sample"], p_hit, p_mean, min_samples=policy.min_samples
        )
    return len(rows)


def _as_labels(values: pd.Series) -> pd.Series:
    """Enum/str/NaN → plain strings, so a bucket name compares reliably."""
    return values.map(
        lambda item: None if item is None or pd.isna(item) else str(getattr(item, "value", item))
    )


def _proximity(frame: pd.DataFrame) -> pd.Series:
    """Distance from the nearest **previous** session's high/low, in ATR units."""
    levels = (
        frame.groupby("trading_date").agg(hi=("high", "max"), lo=("low", "min")).reset_index()
    )
    levels["prev_hi"] = levels["hi"].shift(1)
    levels["prev_lo"] = levels["lo"].shift(1)
    merged = frame.merge(
        levels[["trading_date", "prev_hi", "prev_lo"]], on="trading_date", how="left"
    )
    up = (merged["prev_hi"] - merged["close"]).abs()
    down = (merged["close"] - merged["prev_lo"]).abs()
    nearest = pd.concat([up, down], axis=1).min(axis=1)
    return (nearest / merged["atr"]).reindex(frame.index)


@dataclass(slots=True)
class IndexStudy:
    """The whole study: two timeframes, their axes, and the caveats."""

    symbol: str
    first: str | None
    last: str | None
    policy: dict[str, Any]
    daily: dict[str, Any]
    intraday: dict[str, Any]
    caveats: list[str] = field(default_factory=list)
    #: How many cells the study-wide correction covered (0 when nothing was testable).
    study_family: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "spec": INDEX_STUDY_SPEC,
            "symbol": self.symbol,
            "window": {"first": self.first, "last": self.last},
            "policy": self.policy,
            "study_family": self.study_family,
            "daily": self.daily,
            "intraday": self.intraday,
            "caveats": list(self.caveats),
        }


def build_index_study(
    connection: Any, *, symbol: str, config: Any, policy: IndexStudyPolicy
) -> IndexStudy:
    """Run every axis at every horizon, on both stored timeframes."""
    daily_frame = index_frame(connection, symbol=symbol, timeframe="1d")
    intraday_frame = index_frame(connection, symbol=symbol, timeframe="1m")
    score_order = tuple(name for name, _low, _high in SCORE_BANDS)

    daily_section: dict[str, Any] = {"available": False, "reason": "no stored daily bars"}
    if not daily_frame.empty:
        working = _with_signal(daily_frame, indicators=config.indicators, policy=policy)
        returns = forward_returns(working, policy.daily_horizons, intraday=False)
        # Sample every h-th session: overlapping windows are not extra evidence, so `n`
        # counts independent observations rather than bars.
        positions = pd.Series(range(len(working)), index=working.index)
        axes: list[dict[str, Any]] = []
        for horizon in policy.daily_horizons:
            sample = positions % horizon == 0
            for axis_name, labels, order in (
                ("regime", _as_labels(working["nifty_regime"]), REGIME_ORDER),
                ("score_band", _as_labels(working["score_band"]), score_order),
            ):
                axes.append(
                    _family(
                        axis_name=axis_name,
                        labels=labels[sample],
                        values=returns[f"ret_{horizon}"][sample],
                        values_atr=returns[f"atr_{horizon}"][sample],
                        horizon_label=f"{horizon}d",
                        policy=policy,
                        order=order,
                    ).to_dict()
                )
        daily_section = {
            "available": True,
            "bars": int(len(working)),
            "first": str(working["trading_date"].iloc[0]),
            "last": str(working["trading_date"].iloc[-1]),
            "axes": axes,
        }

    intraday_section: dict[str, Any] = {"available": False, "reason": "no stored 1-minute bars"}
    if not intraday_frame.empty:
        working = _with_signal(intraday_frame, indicators=config.indicators, policy=policy)
        working["time_of_day"] = _intraday_bucket(working["timestamp"])
        working["level_proximity"] = _band(_proximity(working), PROXIMITY_BANDS)
        returns = forward_returns(working, policy.intraday_horizons, intraday=True)
        # Within each session, every h-th bar starts a window that does not overlap the
        # last one - the only sample whose z/Welch tests are not inflated by overlap.
        positions = working.groupby("trading_date").cumcount()
        axes = []
        for horizon in policy.intraday_horizons:
            sample = positions % horizon == 0
            for axis_name, labels, order in (
                ("regime", _as_labels(working["nifty_regime"]), REGIME_ORDER),
                ("score_band", _as_labels(working["score_band"]), score_order),
                (
                    "time_of_day",
                    _as_labels(working["time_of_day"]),
                    tuple(name for name, _low, _high in INTRADAY_BUCKETS),
                ),
                (
                    "level_proximity",
                    _as_labels(working["level_proximity"]),
                    tuple(name for name, _low, _high in PROXIMITY_BANDS),
                ),
            ):
                axes.append(
                    _family(
                        axis_name=axis_name,
                        labels=labels[sample],
                        values=returns[f"ret_{horizon}"][sample],
                        values_atr=returns[f"atr_{horizon}"][sample],
                        horizon_label=f"{horizon}m",
                        policy=policy,
                        order=order,
                    ).to_dict()
                )
        intraday_section = {
            "available": True,
            "bars": int(len(working)),
            "sessions": int(working["trading_date"].nunique()),
            "first": str(working["trading_date"].iloc[0]),
            "last": str(working["trading_date"].iloc[-1]),
            "axes": axes,
        }

    study_family = study_wide_correction(
        {"daily": daily_section, "intraday": intraday_section}, policy=policy
    )
    caveats = [
        "The p-values in each table are corrected within their own axis; the study-wide "
        "line above corrects across every test in the study and is the one to read.",
        "Forward returns are sampled every h-th bar (inside each session for intraday), so "
        "the tested windows do not overlap: `sample` counts independent windows, not bars. "
        "The section's bar count is the whole stored series.",
        "Returns are gross (no costs, no slippage, no sizing): this is a study of the "
        "index, not a strategy.",
        "Regimes persist, so a short intraday window holds only a handful of independent "
        "regime episodes even though it holds thousands of bars.",
        "Intraday bars come from index_history; a missing minute shortens the effective "
        "horizon of the bars before it.",
        "The mean test runs on ATR-normalised returns so quiet and volatile regimes are "
        "comparable; the percent columns are descriptive.",
    ]
    return IndexStudy(
        symbol=symbol,
        first=daily_section.get("first") or intraday_section.get("first"),
        last=daily_section.get("last") or intraday_section.get("last"),
        policy=policy.to_dict(),
        daily=daily_section,
        intraday=intraday_section,
        caveats=caveats,
        study_family=study_family,
    )


def _fmt(value: Any, digits: int = 4) -> str:
    """Render a cell: ``None`` is a dash, floats get a fixed precision."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_study_markdown(study: IndexStudy) -> str:
    """The study as a readable document (tables + the caveats under them)."""
    lines = [f"# {INDEX_STUDY_SPEC}", ""]
    lines.append(f"**Index:** {study.symbol} - **window:** {study.first} -> {study.last}")
    lines.append("")
    lines.append("## Policy")
    for key, value in study.policy.items():
        lines.append(f"- **{key}**: {value}")
    if study.study_family:
        survivors = [
            row
            for section in (study.daily, study.intraday)
            if section.get("available")
            for axis in section.get("axes", [])
            for row in axis.get("rows", [])
            if row.get("verdict_study") == SIGNIFICANT
        ]
        lines.append(
            f"- **study-wide family**: {study.study_family} tested cell(s), "
            f"{len(survivors)} survive correction"
        )
    lines.append("")
    for name in ("daily", "intraday"):
        section = study.daily if name == "daily" else study.intraday
        lines.append(f"## {name.capitalize()}")
        if not section.get("available"):
            lines.append(f"_not available: {section.get('reason')}_")
            lines.append("")
            continue
        header = f"{section['bars']} bar(s)"
        if "sessions" in section:
            header += f" over {section['sessions']} session(s)"
        lines.append(f"{header}, {section['first']} -> {section['last']}")
        lines.append("")
        for axis in section["axes"]:
            baseline = axis.get("baseline") or {}
            lines.append(f"### {axis['axis']} @ {axis['horizon']}")
            lines.append(
                f"_baseline_ n={baseline.get('sample')} | mean {_fmt(baseline.get('mean_pct'))}% | "
                f"hit {_fmt(baseline.get('hit_rate'))} | family={axis['family_size']} | "
                f"correction={axis['correction']}"
            )
            lines.append("")
            lines.append(
                "| bucket | n | hit rate | mean % | median % | mean ATR | rest n | "
                "rest mean % | p(hit) | p(mean) | verdict |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in axis["rows"]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row["bucket"]),
                            str(row["sample"]),
                            _fmt(row.get("hit_rate")),
                            _fmt(row.get("mean_pct")),
                            _fmt(row.get("median_pct")),
                            _fmt(row.get("mean_atr")),
                            str(row.get("rest_sample")),
                            _fmt(row.get("rest_mean_pct")),
                            _fmt(row.get("p_hit")),
                            _fmt(row.get("p_mean")),
                            str(row.get("verdict")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    lines.append("## Caveats")
    for caveat in study.caveats:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def write_index_study(study: IndexStudy, root: str | Path) -> list[Path]:
    """Write the JSON and Markdown artefacts; return the paths that landed."""
    directory = Path(root) / "index_study"
    directory.mkdir(parents=True, exist_ok=True)
    stem = study_stem(study.first or "unknown", study.last or "unknown")
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(json.dumps(study.to_dict(), indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(render_study_markdown(study), encoding="utf-8")
    return [json_path, markdown_path]
