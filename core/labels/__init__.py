"""Labels (spec 20) - the forward-looking half of the dataset.

* :mod:`~core.labels.label_builder` - the multi-horizon forward returns, excursions
  and trend change, plus the discretised class, built from the candles *after* each
  timestamp. Labels are deliberately a separate pass from features: a feature is a
  function of the past, a label of the future, and mixing the two is how look-ahead
  bias gets in.
"""

from __future__ import annotations

from core.labels.label_builder import (
    LABEL_HORIZON_COLUMN,
    REQUIRED_CANDLE_COLUMNS,
    LabelBuildError,
    LabelPolicy,
    build_label_frame,
    label_rows,
)

__all__ = [
    "LABEL_HORIZON_COLUMN",
    "REQUIRED_CANDLE_COLUMNS",
    "LabelBuildError",
    "LabelPolicy",
    "build_label_frame",
    "label_rows",
]
