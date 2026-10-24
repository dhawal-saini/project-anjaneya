"""Input normalisation shared by every indicator.

Every indicator in this package takes a plain ``pandas.Series`` of prices and
returns a series *aligned to the input index*, with ``NaN`` wherever the value is
not yet defined. Two rules make that predictable across the whole library:

* **No look-ahead.** The value at position ``i`` depends only on positions
  ``<= i``. Nothing is centred, nothing is shifted backwards, nothing is
  rescaled with statistics of the full series. This is what lets the feature
  builder recompute an indicator over a lookback window and take only its last
  value without disagreeing with a full-history run; it is enforced by the
  prefix tests in ``tests/unit/test_indicators.py``.
* **No silent zeros.** An undefined indicator is ``NaN``, never ``0``. Zero is a
  real measurement (a flat slope, a zero spread) and must stay distinguishable
  from "not enough history yet" - the same distinction ``NULL`` versus ``0`` has
  in storage.

The helpers here are deliberately tiny: they validate, they coerce to a float
series, and they locate the first window in which an indicator becomes defined.
Anything else belongs to the indicator itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "as_float_series",
    "first_full_window",
    "nan_before",
    "require_pair",
    "require_period",
]

#: Numeric numpy dtype kinds an indicator input may have (int, uint, float).
_NUMERIC_KINDS = "iuf"


def as_float_series(values: Any, *, name: str = "values") -> pd.Series:
    """Return ``values`` as a float64 :class:`pandas.Series`.

    A Series keeps its index (that index is the alignment contract); a list,
    tuple or array is wrapped. Anything non-numeric raises instead of being
    coerced, because ``"12"`` or ``True`` silently turning into a price is
    exactly the kind of bug that survives to a research conclusion.
    """
    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, (list, tuple, np.ndarray)):
        series = pd.Series(values)
    else:
        raise TypeError(
            f"{name} must be a pandas Series, list, tuple or numpy array, "
            f"got {type(values).__name__}"
        )
    if series.empty:
        # An empty column carries no values to misinterpret, and pandas gives an
        # empty frame the ``object`` dtype - which is not a reason to refuse to
        # process "no bars".
        return series.astype("float64")
    if series.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"{name} must be numeric, got dtype {series.dtype!r}; "
            "cast it explicitly if that is really intended"
        )
    return series.astype("float64")


def require_period(period: Any, *, min_period: int = 2, name: str = "period") -> int:
    """Validate a lookback/period argument and return it as a plain ``int``.

    ``min_period`` is 2 by default because every indicator in this package is
    either a *change* (which needs two observations) or a window of at least two
    bars; a period of 1 is almost always a typo that would otherwise produce a
    plausible-looking identity.
    """
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(period).__name__}")
    if period < min_period:
        raise ValueError(f"{name} must be >= {min_period}, got {period}")
    return int(period)


def require_pair(left: pd.Series, right: pd.Series, *, names: tuple[str, str]) -> None:
    """Raise unless two series are position-aligned.

    Aligned series are the one assumption every two-series indicator makes, and
    a misalignment is invisible in the output (it just produces numbers). Since
    ANJANEYA builds every series from the same candle frame, requiring the same
    length *and* the same index costs nothing and removes the failure mode.
    """
    if len(left) != len(right):
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, "
            f"got {len(left)} and {len(right)}"
        )
    if not left.index.equals(right.index):
        raise ValueError(
            f"{names[0]} and {names[1]} must share one index; "
            "build both from the same candle frame"
        )


def first_full_window(values: pd.Series, period: int) -> int | None:
    """Index that closes the first run of ``period`` consecutive non-NaN values.

    Returns ``None`` when the series never contains such a run. Used to seed the
    recursive averages (Wilder's RMA) from real observations instead of assuming
    the head of the series is clean.
    """
    mask = values.notna().to_numpy()
    if period > mask.size:
        return None
    windows = np.convolve(
        mask.astype(np.int64), np.ones(period, dtype=np.int64), mode="valid"
    )
    hits = np.flatnonzero(windows == period)
    if hits.size == 0:
        return None
    return int(hits[0]) + period - 1


def nan_before(values: pd.Series, first_index: int | None) -> pd.Series:
    """Blank ``values`` *before* ``first_index``.

    ``first_index=None`` (nothing is ever defined) blanks the whole series. This
    is how a warm-up rule is expressed: compute the mathematically complete
    series, then refuse to publish the part that is not yet meaningful.
    """
    result = values.astype("float64").copy()
    if first_index is None:
        result.iloc[:] = np.nan
        return result
    if first_index > 0:
        result.iloc[:first_index] = np.nan
    return result
