"""Synthetic multi-day feature/label stores used by the ANJANEYA-AI tests.

The AI app has exactly two inputs - the ``features`` table and the ``labels`` table - and
one interesting property to test: that the matrix it builds out of them is *leak-free and
honestly counted*. So this module provides the fixtures that property needs:

* :func:`feature_rows` / :func:`label_rows` - several contract-days of rows with the shape
  the engine actually writes, including the parts that matter for a dataset: a tail of
  bars whose forward window did not fit inside the session (so their ``label`` is NULL), a
  class mix that uses all four configured classes, and a couple of NIFTY-only columns left
  NULL so the "missing value" reporting has something to report.
* :func:`seed_source` - that window, inserted into a bootstrapped DuckDB file, so the
  integration tests start from *stored* rows.
* :func:`make_ai_context` / :func:`cli_args` - the real :class:`AppContext` and the exact
  argument list ``apps.anjaneeya_ai.run.main`` would receive.
* :func:`shrink_config` / :func:`signal_frame` - the two things the *model* stage needs: a
  configuration whose walk-forward windows and tree sizes fit a four-day fixture, and a
  frame whose class is genuinely learnable from its features (so a trainer test can ask
  "did it learn?" and get a meaningful answer).

Nothing here writes into the repository's ``data/`` or ``db/``: callers pass ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from core.constants import CANDLE_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMNS
from core.ml import CODE_COLUMN, LABEL_DERIVED_COLUMNS, META_COLUMNS, TARGET_COLUMN
from core.schemas import FeatureRow, LabelRow
from core.storage import bootstrap_database, connect, insert_rows
from tests.support.engine import (
    CE_SYMBOL,
    CE_TOKEN,
    PE_SYMBOL,
    PE_TOKEN,
    RUN_ID,
    make_candle,
    make_feature,
)

#: Repository root and the shipped configuration directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

#: Walk-forward windows that fit the four-day fixture (one day train, one day validate).
ONE_DAY_FOLDS: dict[str, int] = {
    "train_window_days": 1,
    "validate_window_days": 1,
    "step_days": 1,
    "holdout_days": 1,
}

#: A tree small enough to train in a test, large enough to learn a real pattern.
#: The integration tests assert on *structure* (which folds ran, what was written), so the
#: tree stays deliberately tiny: twenty runs of a 30-tree model is minutes, not seconds.
FAST_TRAINING: dict[str, int] = {
    "n_estimators": 15,
    "num_leaves": 4,
    "min_child_samples": 5,
    "n_jobs": 1,
}

#: Feature columns of :func:`signal_frame`.
SIGNAL_FEATURES: tuple[str, ...] = ("signal_a", "signal_b", "noise", "nifty_regime")

#: The one categorical feature of :func:`signal_frame`.
SIGNAL_CATEGORICAL: tuple[str, ...] = ("nifty_regime",)

#: Exchange timezone for every synthetic timestamp.
IST = ZoneInfo("Asia/Kolkata")

#: Four consecutive trading days (Mon 2026-09-28 .. Thu 2026-10-01, the weekly expiry).
DAYS: tuple[date, ...] = (
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
)

#: The horizons ``config/model.yaml`` ships.
HORIZONS: tuple[int, ...] = (5, 10, 15, 30)

#: Bars per contract-day. More than the longest horizon on purpose: the last
#: ``max(HORIZONS)`` bars of a session have no forward window inside it, so their label is
#: NULL - which is what the "unlabelled rows are counted, not hidden" assertions need.
BARS = 60

#: The classes, in the order ``config/model.yaml`` lists them.
CLASSES: tuple[str, ...] = ("UP_CONTINUATION", "DOWN_CONTINUATION", "RANGE", "REVERSAL")

#: NIFTY regimes, cycled through the fixture so ``nifty_regime`` is a real categorical
#: feature with more than one level.
REGIMES: tuple[str, ...] = ("BULLISH", "BEARISH", "NEUTRAL", "BULLISH_DEVELOPING")

#: Label version written by the fixtures.
LABEL_VERSION = "v1"


def session_start(day: date) -> datetime:
    """The 09:15 IST session open of ``day``."""
    return datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)


def at(day: date, minute: int) -> datetime:
    """``minute`` minutes after the session open of ``day``."""
    return session_start(day) + timedelta(minutes=minute)


def feature_rows(
    days: Sequence[date] = DAYS,
    *,
    bars: int = BARS,
    timeframe: str = "1m",
    symbols: Sequence[tuple[str, int, str]] = (
        (CE_SYMBOL, CE_TOKEN, "CE"),
        (PE_SYMBOL, PE_TOKEN, "PE"),
    ),
) -> list[FeatureRow]:
    """One contract-day of feature rows per (day, contract), minute by minute.

    ``day_of_week`` and ``time_of_day_bucket`` are filled from the timestamp, so a
    timeframe override moves them too - exactly as the feature builder does it.
    """
    rows: list[FeatureRow] = []
    for day in days:
        for symbol, token, option_type in symbols:
            for minute in range(bars):
                moment = at(day, minute)
                rows.append(
                    make_feature(
                        timestamp=moment,
                        tradingsymbol=symbol,
                        instrument_token=token,
                        option_type=option_type,
                        timeframe=timeframe,
                        day_of_week=moment.weekday(),
                        nifty_regime=REGIMES[minute % 4],
                        nifty_composite_trend_score=(minute - bars / 2) / bars,
                        option_atr=1.0 + minute / 100,
                        candle_direction="BULLISH" if minute % 2 else "BEARISH",
                        time_of_day_bucket="OPEN_915_930" if minute < 15 else "MIDDAY",
                        expiry=day + timedelta(days=3),
                        fetch_timestamp=moment + timedelta(seconds=1),
                        run_id=RUN_ID,
                    )
                )
    return rows


def label_rows(
    days: Sequence[date] = DAYS,
    *,
    bars: int = BARS,
    horizons: Sequence[int] = HORIZONS,
    symbols: Sequence[tuple[str, int, str]] = (
        (CE_SYMBOL, CE_TOKEN, "CE"),
        (PE_SYMBOL, PE_TOKEN, "PE"),
    ),
) -> list[LabelRow]:
    """The forward labels of :func:`feature_rows`, one row per horizon.

    A bar keeps a class only when its forward window fits inside the session - the rule
    ``core/labels/label_builder.py`` enforces - so the last ``max(horizons)`` bars of each
    session come back with ``label=None`` while still carrying real forward numbers for
    the shorter horizons, exactly as the builder writes them.
    """
    longest = max(horizons)
    rows: list[LabelRow] = []
    for day in days:
        for symbol, token, option_type in symbols:
            for minute in range(bars):
                moment = at(day, minute)
                window_fits = minute + longest <= bars
                direction = 1 if minute % 2 else -1
                for horizon in horizons:
                    horizon_fits = minute + horizon <= bars
                    rows.append(
                        LabelRow(
                            timestamp=moment,
                            tradingsymbol=symbol,
                            instrument_token=token,
                            option_type=option_type,
                            expiry=day + timedelta(days=3),
                            strike=25000.0,
                            horizon_minutes=horizon,
                            future_return=(
                                None if not horizon_fits else direction * 0.004 * horizon / 5
                            ),
                            future_max_up_move=None if not horizon_fits else 0.5,
                            future_max_down_move=None if not horizon_fits else -0.4,
                            future_trend_change=None if not horizon_fits else float(direction),
                            label=CLASSES[minute % 4] if window_fits else None,
                            label_version=LABEL_VERSION,
                            fetch_timestamp=moment + timedelta(minutes=horizon + 1),
                            run_id=RUN_ID,
                        )
                    )
    return rows


def premium(symbol: str, minute: int) -> float:
    """A deterministic option premium: a ``CE`` contract drifts up, a ``PE`` contract down.

    The paths are what make the backtest fixtures *decidable*: with the shipped stop/target
    multiples, a long CE reaches its target and a long PE its stop, so "a winner and a loser
    are both in the ledger" is a property of the fixture rather than of luck.
    """
    drift = 0.5 if symbol.upper().endswith("CE") else -0.35
    return round(100.0 + drift * minute, 2)


def candle_rows(
    days: Sequence[date] = DAYS,
    *,
    bars: int = BARS,
    timeframe: str = "1m",
    symbols: Sequence[tuple[str, int, str]] = (
        (CE_SYMBOL, CE_TOKEN, "CE"),
        (PE_SYMBOL, PE_TOKEN, "PE"),
    ),
) -> list[Any]:
    """One contract-day of stored bars per (day, contract), minute by minute."""
    rows: list[Any] = []
    for day in days:
        for symbol, token, option_type in symbols:
            for minute in range(bars):
                opened = premium(symbol, minute)
                closed = premium(symbol, minute + 1)
                rows.append(
                    make_candle(
                        timestamp=at(day, minute),
                        timeframe=timeframe,
                        tradingsymbol=symbol,
                        instrument_token=token,
                        option_type=option_type,
                        expiry=day + timedelta(days=3),
                        open=opened,
                        high=max(opened, closed) + 0.25,
                        low=min(opened, closed) - 0.25,
                        close=closed,
                        volume=100,
                        oi=5000,
                        fetch_timestamp=at(day, minute) + timedelta(seconds=59),
                        run_id=RUN_ID,
                    )
                )
    return rows


def candle_frame(days: Sequence[date] = DAYS, **kwargs: Any) -> pd.DataFrame:
    """The stored bars of :func:`candle_rows`, as DuckDB would return them."""
    return frame(candle_rows(days, **kwargs), CANDLE_COLUMNS)


def signal_class_frame(days: Sequence[date] = DAYS, *, bars: int = BARS) -> pd.DataFrame:
    """The stored feature rows with a target that is *learnable* (it follows ``option_type``).

    The backtest integration tests need a model whose prediction is predictable: here
    ``UP_CONTINUATION`` follows a CE contract and ``DOWN_CONTINUATION`` a PE contract, so a
    model trained on this frame fires a signal on both sides - and, replayed against the
    premium path above, produces one winner (CE) and one loser (PE).
    """
    stored = feature_frame(days, bars=bars)
    classes = ("UP_CONTINUATION", "DOWN_CONTINUATION", "RANGE", "REVERSAL")
    codes = [0 if str(value).upper() == "CE" else 1 for value in stored["option_type"]]
    stored[CODE_COLUMN] = codes
    stored[TARGET_COLUMN] = [classes[code] for code in codes]
    return stored


def training_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """The feature columns ``core.ml.dataset.build_dataset`` would keep for ``frame``.

    Derived exactly the way the dataset builder derives them (the stored contract minus
    identity/provenance columns minus every label name), so a crafted model artefact in a
    test is fitted on the matrix the real pipeline would have produced.
    """
    return tuple(
        name
        for name in FEATURE_COLUMNS
        if name in frame.columns
        and name not in META_COLUMNS
        and name not in LABEL_DERIVED_COLUMNS
    )


def categorical_columns() -> tuple[str, ...]:
    """The shipped ``model.dataset.categorical_columns`` (what a real dataset declares)."""
    from core.config_loader import load_config

    return tuple(
        str(name) for name in load_config(DEFAULT_CONFIG_DIR).model.dataset.categorical_columns
    )


def write_signal_model(
    root: Path,
    *,
    horizon: int = 5,
    seed: int = 42,
    dataset_name: str = "1m_h5_2026-09-28_2026-10-01",
) -> Path:
    """Fit a small model on :func:`signal_class_frame` and write the artefact *pair*.

    Returns the booster path. The metadata is produced by ``TrainedModel.to_manifest()`` -
    the same method ``apps/anjaneeya_ai/model_writer.py`` uses - so ``--mode backtest``
    loads a real artefact (feature order, categorical vocabulary and all), not a stub.
    """
    from core.ml import TrainingPolicy, train_model
    from core.ml.trainer import TrainedModel  # noqa: F401 - documents the type being written

    frame_ = signal_class_frame()
    policy = TrainingPolicy(
        algorithm="lightgbm",
        random_seed=seed,
        n_estimators=15,
        num_leaves=4,
        min_child_samples=5,
        n_jobs=1,
    )
    model = train_model(
        frame_,
        policy=policy,
        classes=CLASSES,
        feature_columns=training_feature_columns(frame_),
        categorical_columns=categorical_columns(),
        horizon=horizon,
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dataset_name}_lightgbm.txt"
    model.booster.save_model(str(path))
    metadata = {
        "run_id": RUN_ID,
        "dataset_name": dataset_name,
        "dataset": {"timeframe": "1m", "horizon_minutes": horizon},
        "model": model.to_manifest(),
        "results_label": "EXPLORATORY",
    }
    path.with_name(path.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return path


def frame(rows: Sequence[Any], columns: Sequence[str]) -> pd.DataFrame:
    """Rows as a frame with tz-aware timestamps (what DuckDB hands back)."""
    if not rows:
        return pd.DataFrame(columns=list(columns))
    frame_ = pd.DataFrame([row.to_record(exclude_none=False) for row in rows])
    frame_["timestamp"] = pd.to_datetime(frame_["timestamp"])
    if "fetch_timestamp" in frame_.columns:
        frame_["fetch_timestamp"] = pd.to_datetime(frame_["fetch_timestamp"])
    return frame_


def feature_frame(days: Sequence[date] = DAYS, **kwargs: Any) -> pd.DataFrame:
    """The stored feature rows of :func:`feature_rows`, as DuckDB would return them."""
    return frame(feature_rows(days, **kwargs), FEATURE_COLUMNS)


def label_frame(days: Sequence[date] = DAYS, **kwargs: Any) -> pd.DataFrame:
    """The stored label rows of :func:`label_rows`, as DuckDB would return them."""
    return frame(label_rows(days, **kwargs), LABEL_COLUMNS)


def seed_source(
    db_path: Any,
    *,
    days: Sequence[date] = DAYS,
    bars: int = BARS,
    horizons: Sequence[int] = HORIZONS,
    timeframe: str = "1m",
    schema_version: int = 1,
    candles: bool = True,
) -> Any:
    """Bootstrap ``db_path`` and insert the fixture's feature, label and candle rows.

    The bars are inserted too, because ``--mode backtest`` replays *stored* prices: a
    backtest test that built its bars in memory would not be testing what the mode does.
    """
    bootstrap_database(db_path, schema_version=schema_version)
    rows = feature_rows(days, bars=bars, timeframe=timeframe)
    labels = label_rows(days, bars=bars, horizons=horizons)
    bars_rows = candle_rows(days, bars=bars, timeframe=timeframe) if candles else []
    connection = connect(db_path)
    try:
        insert_rows(connection, "features", rows, mode="append", columns=FEATURE_COLUMNS)
        insert_rows(connection, "labels", labels, mode="append", columns=LABEL_COLUMNS)
        if bars_rows:
            insert_rows(connection, "candles", bars_rows, mode="append", columns=CANDLE_COLUMNS)
    finally:
        connection.close()
    return db_path


def make_ai_context(
    config: Any,
    *,
    mode: str = "dataset",
    dry_run: bool = False,
    logger: Any = None,
    bootstrap: bool = True,
) -> Any:
    """Build the real :class:`AppContext` the AI CLI would build."""
    from core.app_bootstrap import APP_AI, AppContext
    from core.market_calendar import NseCalendar
    from core.run_context import create_run_context

    if bootstrap:
        bootstrap_database(config.db_path, schema_version=1)
    run_context = create_run_context(
        APP_AI,
        mode=mode,
        config_hash=config.config_hash,
        config_dir=config.config_dir,
        timezone_name=config.timezone_name,
        package_names=[],
    )
    return AppContext(
        app_name=APP_AI,
        mode=mode,
        config=config,
        run_context=run_context,
        calendar=NseCalendar(config.calendar),
        logger=logger if logger is not None else RecordingLogger(),
        dry_run=dry_run,
    )


def cli_args(
    tmp_path: Any,
    *,
    mode: str = "dataset",
    extra: Sequence[str] | None = None,
    run_artifacts: bool = False,
) -> list[str]:
    """Arguments for ``apps.anjaneeya_ai.run.main`` against ``tmp_path``."""
    args = [
        "--mode",
        mode,
        "--data-root",
        str(tmp_path / "data"),
        "--db-path",
        str(tmp_path / "db" / "anjaneeya.duckdb"),
    ]
    if not run_artifacts:
        args.append("--no-run-artifacts")
    return args + list(extra or [])


class RecordingLogger:
    """A logger double that records the level and message of every call.

    AI tests assert on *what the operator would have seen* (a horizon with no labelled
    rows, a data gate that is not yet open) rather than on log files.
    """

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, message: Any) -> None:
        self.records.append((level, str(message)))

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("DEBUG", message)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("INFO", message)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("WARNING", message)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("ERROR", message)

    def exception(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("ERROR", message)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._add("CRITICAL", message)

    def messages(self, level: str | None = None) -> list[str]:
        """Recorded messages, optionally filtered by level."""
        return [text for recorded, text in self.records if level is None or recorded == level]




# ---------------------------------------------------------------------------
# The model stage's fixtures
# ---------------------------------------------------------------------------
def signal_frame(rows: int = 600, *, seed: int = 0, noise: bool = True) -> pd.DataFrame:
    """A frame whose class is *learnable* from two features (for trainer tests).

    ``signal_a + signal_b`` decides the class through fixed thresholds, so a model that
    learns nothing is visibly worse than one that does - which is what makes an accuracy
    assertion meaningful rather than decorative. ``noise`` is an unrelated numeric column
    and ``regime`` an unrelated categorical, so neither can be an accidental shortcut.
    """
    generator = np.random.default_rng(seed)
    first = generator.normal(size=rows)
    second = generator.normal(size=rows)
    third = generator.normal(size=rows)
    code = np.digitize(first + second, (-0.6, 0.6, 1.8)).astype("int64")
    frame = pd.DataFrame(
        {
            "signal_a": first,
            "signal_b": second,
            "noise": third if noise else np.zeros(rows),
            "nifty_regime": np.where(third > 0, "BULLISH", "BEARISH"),
        }
    )
    frame[CODE_COLUMN] = code
    frame[TARGET_COLUMN] = [CLASSES[int(value)] for value in code]
    return frame


def shrink_config(
    tmp_path: Any,
    *,
    walk_forward: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    backtest: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
) -> Path:
    """Copy the shipped configuration into ``tmp_path/config`` and override sections.

    Returns the directory to hand to ``--config-dir``. Only ``model.yaml`` is edited: the
    point of the AI tests is what changes when a window, a tree size or a signal threshold
    does, and every other file should stay the shipped one.
    """
    target = Path(tmp_path) / "config"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    path = target / "model.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, values in (("walk_forward", walk_forward), ("training", training)):
        if values:
            payload[name].update(values)
    if backtest:
        for name, values in backtest.items():
            if isinstance(values, dict):
                payload["backtest"][name].update(values)
            else:
                payload["backtest"][name] = values
    if model:
        payload.update(model)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target

