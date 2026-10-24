"""Labels - the forward-looking half of the dataset (spec 20, Stage 3.4b).

Spec 20 asks for two things that pull in opposite directions, so both are stated
here rather than discovered later:

1. **Labels are generated in a physically separate pass.** Nothing in this module
   reads a feature row, and :class:`core.schemas.FeatureRow` structurally refuses
   the label fields (``future_return`` and friends). A label is a function of the
   *candles after* a moment; a feature is a function of the candles *up to* it, and
   mixing the two in one pass is how look-ahead bias gets in.
2. **The model is never trained on BUY/SELL.** It predicts future behaviour over
   configurable horizons - ``future_return``, ``future_max_up_move``,
   ``future_max_down_move``, ``future_trend_change`` - plus one discretised class
   (``UP_CONTINUATION`` / ``DOWN_CONTINUATION`` / ``RANGE`` / ``REVERSAL``) whose
   threshold comes from configuration.

What each number means (units are not negotiable)
-------------------------------------------------
* ``future_return`` - **relative**: ``close[t+h] / close[t] - 1``. Unit-free, and
  the one number that needs no scale to read.
* ``future_max_up_move`` / ``future_max_down_move`` - **price units**, measured
  from ``close[t]``: the best/worst level the market *reached* inside the forward
  window (``>= 0`` / ``<= 0``). An excursion that never goes in that direction is
  ``0``, not a negative "favourite move". A barrier or a take-profit rule is
  compared against these, which is why they stay in price units like the threshold
  (a threshold *is* a price move - see below).
* ``future_trend_change`` - **threshold units**: the normalised forward move minus
  the normalised move *into* the decision bar, so ``+1`` means "the horizon added
  one whole threshold of trend over what the trend already was".
* ``label`` - the discretised class (below).

The threshold, and where look-ahead would creep in
-------------------------------------------------
Spec 20 wants thresholds that are "configurable, ATR- or volatility-normalized".
:class:`LabelPolicy` implements all three spellings from ``config/model.yaml``:

* ``atr``     - ``normalization_param * ATR(period)`` **at bar t** (Wilder by
  default, the unit the rest of ANJANEYA measures things in),
* ``percent`` - ``percent_threshold * close[t]``,
* ``fixed``   - ``fixed_threshold`` price units.

Every input to a threshold is read at or before ``t``; nothing about the forward
window can influence the scale used to judge it. The classification rule is:

* ``|move| < threshold`` -> ``RANGE`` (the horizon did not clear the noise),
* ``move >= threshold`` with a **positive** prior move -> ``UP_CONTINUATION``,
* ``move <= -threshold`` with a **negative** prior move -> ``DOWN_CONTINUATION``,
* any other clearing move -> ``REVERSAL``.

The **prior move** is ``close[t] - close[t - trend_lookback bars]``: a continuation
has to continue something. When that prior bar is missing (the very start of the
data, or a hole in it) no class is published at all - guessing a direction there
would invent evidence - and when the prior is exactly flat a clearing move is
published as ``REVERSAL`` (with no established trend the four-class vocabulary has
no "initiation" to offer; the flat case is the documented boundary).

Sessions, and what a label cannot be
------------------------------------
A label needs its **whole forward window inside the same session**: ``t + horizon
<= session_end``, and every minute of that window must actually be present. An
option's overnight gap is not a five-minute move, so labels never cross the close,
and the last ``max(horizon)`` bars of a session carry ``NULL`` label fields rather
than a number that mixes two sessions. The holes are reported (a coverage row and
the run summary), never filled.

The output is one row per ``(timestamp, contract, horizon)`` - the shape of the
``labels`` table - for **every** session bar, including the ones whose label cannot
exist. A row that says "no label here" is a fact a dataset builder can skip; a
missing row is a fact it has to guess about.

Importing this module performs no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from core.constants import LABEL_COLUMNS, timeframe_to_minutes
from core.indicators.atr import atr
from core.schemas import LabelClass, LabelRow, QualityFlag, Source

__all__ = [
    "LABEL_HORIZON_COLUMN",
    "REQUIRED_CANDLE_COLUMNS",
    "LabelBuildError",
    "LabelPolicy",
    "build_label_frame",
    "label_rows",
]

#: The horizons a build covers live in this column (part of the table's key).
LABEL_HORIZON_COLUMN = "horizon_minutes"

#: Columns :func:`build_label_frame` needs on the candle frame. The greeks, the OI
#: and the NIFTY block are irrelevant to a label - which is the point of the pass.
REQUIRED_CANDLE_COLUMNS = ("timestamp", "tradingsymbol", "open", "high", "low", "close")

#: How a threshold is spelled in ``config/model.yaml``.
Normalization = Literal["atr", "percent", "fixed"]


class LabelBuildError(ValueError):
    """Raised when a label frame cannot be built from the inputs it was given."""


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    """Everything that decides *what counts as a move* (``config/model.yaml``).

    Built through :meth:`from_config`, so a label row can never come from a policy
    that does not exist in the configuration file - ``label_version`` is carried on
    every row precisely because changing any of these numbers changes what the
    labels *mean*.
    """

    horizons: tuple[int, ...]
    normalization: Normalization
    normalization_param: float
    percent_threshold: float
    fixed_threshold: float
    trend_lookback: int
    atr_period: int
    atr_method: str
    version: str

    @classmethod
    def from_config(cls, model: Any, *, indicators: Any = None) -> LabelPolicy:
        """Read the policy from the loaded configuration.

        ``model`` is ``config/model.yaml`` (the horizons and the label section).
        ``indicators`` is optional and only supplies the ATR period/method, so a
        caller without the indicators config still gets the documented defaults
        (ATR-14, Wilder) instead of an exception.
        """
        labels = model.labels
        atr_config = getattr(indicators, "atr", None)
        return cls(
            horizons=tuple(int(horizon) for horizon in model.horizons_minutes),
            normalization=labels.normalization,
            normalization_param=float(labels.normalization_param),
            percent_threshold=float(labels.percent_threshold),
            fixed_threshold=float(labels.fixed_threshold),
            trend_lookback=int(getattr(labels, "trend_lookback", 5)),
            atr_period=int(getattr(atr_config, "period", 14)),
            atr_method=str(getattr(atr_config, "method", "wilder")),
            version=str(labels.version),
        )

    @property
    def max_horizon(self) -> int:
        """The longest horizon - how many bars a session cannot label at its end."""
        return max(self.horizons)

    def bars_for(self, horizon: int, *, timeframe: str) -> int:
        """How many bars a horizon spans at this timeframe.

        A horizon that is not a whole number of bars (7 minutes on a 5m series) is a
        configuration error: rounding it would silently label a different horizon
        than the one the config names.
        """
        minutes = timeframe_to_minutes(timeframe)
        if horizon % minutes != 0:
            raise LabelBuildError(
                f"horizon {horizon}m is not a whole number of {timeframe} bars "
                f"({minutes}m each); label a timeframe the horizons divide, or change the "
                "horizons in config/model.yaml"
            )
        return horizon // minutes

    def describe(self) -> dict[str, Any]:
        """Log-safe summary - what these labels *mean* is a function of exactly this."""
        return {
            "horizons_minutes": list(self.horizons),
            "normalization": self.normalization,
            "normalization_param": self.normalization_param,
            "percent_threshold": self.percent_threshold,
            "fixed_threshold": self.fixed_threshold,
            "trend_lookback": self.trend_lookback,
            "atr_period": self.atr_period,
            "atr_method": self.atr_method,
            "version": self.version,
        }


def _forward_roll(series: pd.Series, bars: int, *, how: str) -> pd.Series:
    """The ``how`` (``max``/``min``) over the **next** ``bars`` bars, excluding this one.

    A backward-looking ``rolling`` cannot express "the window after me", so the series
    is reversed, rolled (which now looks *forward* in original order) and reversed
    back; the final ``shift(-1)`` moves the window off the current bar.
    """
    reversed_series = series.iloc[::-1]
    rolled = reversed_series.rolling(bars, min_periods=bars).agg(how)
    return rolled.iloc[::-1].shift(-1)


def _thresholds(working: pd.DataFrame, policy: LabelPolicy) -> pd.Series:
    """The price move that counts as a move, from the policy (no look-ahead)."""
    close = pd.to_numeric(working["close"], errors="coerce")
    if policy.normalization == "atr":
        measured = atr(
            working["high"],
            working["low"],
            close,
            period=policy.atr_period,
            method=policy.atr_method,
        )
        return policy.normalization_param * measured
    if policy.normalization == "percent":
        return policy.percent_threshold * close
    if policy.normalization == "fixed":
        return pd.Series(float(policy.fixed_threshold), index=working.index, dtype="float64")
    raise LabelBuildError(
        f"Unknown normalization {policy.normalization!r}; expected 'atr', 'percent' or 'fixed'"
    )


def _as_date(value: Any) -> Any:
    """``expiry`` as a ``datetime.date`` (a row contract may hand back a Timestamp)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date") and not isinstance(value, str):  # pragma: no cover - defensive
        return value.date()
    return value


def build_label_frame(
    candles: pd.DataFrame,
    *,
    policy: LabelPolicy,
    timeframe: str,
    run_id: str,
    session_start: datetime | None = None,
    session_end: datetime | None = None,
    source: Source = Source.DERIVED,
) -> pd.DataFrame:
    """One contract's candles -> one row per ``(timestamp, horizon)`` (spec 20).

    ``candles`` is that contract's candle window: the session being labelled **plus**
    the ``trend_lookback`` bars before it (the prior move of the session's first bars
    lives there). Rows outside ``[session_start, session_end)`` are used for
    computation but never published - the same rule the feature builder follows, and
    the reason the two tables line up on ``(timestamp, tradingsymbol)``.

    Every session bar gets a row for every horizon. A row whose forward window does
    not fit inside the session, is not contiguous in the frame, or has no threshold
    yet (ATR warm-up) carries ``NULL`` label fields - never a number computed from
    an incomplete window.
    """
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        raise LabelBuildError(
            f"the candle frame is missing column(s) {missing}; expected at least "
            f"{REQUIRED_CANDLE_COLUMNS}"
        )
    if candles.empty:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))

    working = candles.sort_values("timestamp", kind="stable").reset_index(drop=True)
    stamps = pd.to_datetime(working["timestamp"], errors="raise")
    if stamps.dt.tz is None:
        raise LabelBuildError(
            "candle timestamps are naive; ANJANEYA only stores timezone-aware data "
            "(see settings.time.forbid_naive_timestamps)"
        )
    if stamps.duplicated().any():
        raise LabelBuildError(
            "the candle window carries duplicate timestamps; refusing to build rows that "
            "would collide on the (timestamp, tradingsymbol, horizon_minutes) primary key"
        )
    working["timestamp"] = stamps
    minutes = timeframe_to_minutes(timeframe)
    close = pd.to_numeric(working["close"], errors="coerce")
    thresholds = _thresholds(working, policy)

    lookback = max(policy.trend_lookback, 1)
    prior_stamp = stamps.shift(lookback)
    # The prior bar has to be *that* bar, not merely the nth one in the frame: a hole
    # in the window would otherwise turn a 5-bar move into an arbitrarily long one.
    prior_ok = prior_stamp.notna() & (prior_stamp == stamps - timedelta(minutes=lookback * minutes))
    prior_move = close - close.shift(lookback)

    frames: list[pd.DataFrame] = []
    for horizon in policy.horizons:
        frames.append(
            _horizon_block(
                working,
                policy=policy,
                horizon=horizon,
                bars=policy.bars_for(horizon, timeframe=timeframe),
                minutes=minutes,
                close=close,
                thresholds=thresholds,
                prior_ok=prior_ok,
                prior_move=prior_move,
                session_end=session_end,
                run_id=run_id,
                source=source,
            )
        )

    assembled = pd.concat(frames, ignore_index=True)
    published = pd.Series(True, index=assembled.index)
    if session_start is not None:
        published &= assembled["timestamp"] >= session_start
    if session_end is not None:
        published &= assembled["timestamp"] < session_end
    return assembled[published].reset_index(drop=True)[list(LABEL_COLUMNS)]


def _horizon_block(
    working: pd.DataFrame,
    *,
    policy: LabelPolicy,
    horizon: int,
    bars: int,
    minutes: int,
    close: pd.Series,
    thresholds: pd.Series,
    prior_ok: pd.Series,
    prior_move: pd.Series,
    session_end: datetime | None,
    run_id: str,
    source: Source,
) -> pd.DataFrame:
    """Every bar's label fields for one horizon (``LABEL_COLUMNS``, minus the filter)."""
    stamps = working["timestamp"]
    window_end = stamps + timedelta(minutes=horizon)
    contiguous = stamps.shift(-bars).notna() & (stamps.shift(-bars) == window_end)
    if session_end is not None:
        # A label must live inside one session: the close is where the forward window
        # stops, so the day's last `horizon` minutes have no label rather than one
        # computed across the overnight gap.
        contiguous &= window_end <= session_end

    forward_close = close.shift(-bars)
    # The extremes come from the bars' own high/low, not their closes: an excursion is
    # what the market *reached*, and a wick is a real level.
    forward_high = _forward_roll(pd.to_numeric(working["high"], errors="coerce"), bars, how="max")
    forward_low = _forward_roll(pd.to_numeric(working["low"], errors="coerce"), bars, how="min")

    move = forward_close - close
    threshold_known = thresholds.notna() & (thresholds > 0)
    complete = contiguous & move.notna() & forward_high.notna() & forward_low.notna()
    scaled = complete & threshold_known
    normalised_move = move / thresholds

    # An excursion is only ever in one direction at a time: "the most favourable level
    # reached" floors at zero, so the sign of the column states that direction.
    up_excursion = (forward_high - close).clip(lower=0)
    down_excursion = (forward_low - close).clip(upper=0)

    classed = scaled & prior_ok & prior_move.notna()
    range_ = classed & (move.abs() < thresholds)
    # The prior move is compared **strictly**: a continuation has to continue something,
    # so a prior bar that is exactly flat (no established trend) cannot be continued -
    # it falls through to REVERSAL below, the documented flat boundary.
    up = classed & (move >= thresholds) & (prior_move > 0)
    down = classed & (move <= -thresholds) & (prior_move < 0)
    reversal = classed & ~range_ & ~up & ~down

    labels = np.select(
        [range_, up, down, reversal],
        [
            LabelClass.RANGE.value,
            LabelClass.UP_CONTINUATION.value,
            LabelClass.DOWN_CONTINUATION.value,
            LabelClass.REVERSAL.value,
        ],
        default=None,
    )

    block = pd.DataFrame(index=working.index)
    block["timestamp"] = stamps
    block["tradingsymbol"] = working["tradingsymbol"]
    block["instrument_token"] = _column(working, "instrument_token")
    block["expiry"] = [
        _as_date(value) for value in _column(working, "expiry", default=None)
    ]
    block["strike"] = _column(working, "strike")
    block["option_type"] = _column(working, "option_type")
    block[LABEL_HORIZON_COLUMN] = horizon
    block["future_return"] = (forward_close / close - 1.0).where(complete & (close > 0))
    block["future_max_up_move"] = up_excursion.where(complete)
    block["future_max_down_move"] = down_excursion.where(complete)
    block["future_trend_change"] = (normalised_move - prior_move / thresholds).where(classed)
    block["label"] = labels
    block["label_version"] = policy.version
    block["source"] = source.value
    block["fetch_timestamp"] = _column(working, "fetch_timestamp", default=None)
    block["quality_flag"] = [
        quality.value if isinstance(quality, QualityFlag) else str(quality)
        for quality in _column(working, "quality_flag", default=QualityFlag.OK.value)
    ]
    block["run_id"] = run_id
    return block


def _column(frame: pd.DataFrame, name: str, *, default: Any = None) -> Any:
    """A column when it exists, otherwise a constant (or ``None``) of the right length."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _clean(value: Any) -> Any:
    """One frame cell as a value the row contract accepts.

    ``NaN``/``NaT`` become ``None`` because pandas spells missing that way and the row
    contract says ``None``. ``inf`` is deliberately **not** sanitised: an infinite
    label is a bug, and it must fail loudly here rather than reach a model.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def label_rows(frame: pd.DataFrame) -> list[LabelRow]:
    """Turn an assembled label frame into validated :class:`core.schemas.LabelRow` objects.

    Validation happens *here*, before anything is written: a label row that violates
    the contract (an unknown column, a positive ``future_max_down_move``, a class the
    model was not configured to predict) fails while the database still holds the
    previous, consistent rows.
    """
    declared = set(LABEL_COLUMNS)
    extra = [column for column in frame.columns if column not in declared]
    if extra:
        raise LabelBuildError(
            f"the label frame declares column(s) {extra} that the row contract does not "
            f"have; expected exactly {list(LABEL_COLUMNS)}"
        )
    rows: list[LabelRow] = []
    for record in frame.to_dict("records"):
        rows.append(LabelRow(**{name: _clean(value) for name, value in record.items()}))
    return rows

