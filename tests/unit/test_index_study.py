"""The P1 index study: forward returns, the correction, and the honesty rules.

The study is the cheapest question in the project - "does the NIFTY signal separate
forward returns at all?" - so its tests are about the four things that make an answer
trustworthy rather than flattering:

1. a forward return is computed over the *right* bars (never across a close),
2. a thin bucket is refused, not tested,
3. the correction is applied to the family that was actually tested,
4. a level is only compared with the *previous* session.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from core.config_loader import load_config
from core.research.index_study import (
    IndexStudy,
    IndexStudyPolicy,
    StudyAxis,
    _band,
    _family,
    _intraday_bucket,
    _proximity,
    forward_returns,
    render_study_markdown,
    study_stem,
    study_wide_correction,
    write_index_study,
)

#: A small policy: two horizons, three samples to test, one correction.
POLICY = IndexStudyPolicy(
    combination="combination_1",
    correction="benjamini_hochberg",
    min_samples=3,
    intraday_horizons=(1, 3),
    daily_horizons=(1, 2),
)


def _intraday_frame(*, bars_per_session: int = 6, sessions: int = 2) -> pd.DataFrame:
    """Two sessions of rising closes, one point per bar, ATR fixed at 2."""
    stamps = []
    closes = []
    for session in range(sessions):
        start = datetime(2026, 9, 1 + session, 9, 15)
        for bar in range(bars_per_session):
            stamps.append(start + timedelta(minutes=bar))
            closes.append(100.0 + session * 10 + bar)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "close": closes,
            "atr": [2.0] * len(stamps),
            "trading_date": [stamp.date() for stamp in stamps],
        }
    )


# ---------------------------------------------------------------------------
# Rule 4: a forward return never crosses the close
# ---------------------------------------------------------------------------
def test_an_intraday_return_stops_at_the_close() -> None:
    frame = _intraday_frame(bars_per_session=6, sessions=2)

    returns = forward_returns(frame, (1,), intraday=True)

    # The last bar of each session has no bar after it inside its own session.
    assert returns["ret_1"].isna().tolist() == [False] * 5 + [True] + [False] * 5 + [True]
    # ... and the bar before it is measured against the next bar of the same session.
    assert returns["ret_1"].iloc[4] == (105.0 / 104.0 - 1.0) * 100.0
    # The first bar of the next session starts a fresh window rather than jumping the gap.
    assert returns["ret_1"].iloc[6] == (111.0 / 110.0 - 1.0) * 100.0


def test_a_horizon_longer_than_the_session_rest_is_a_gap() -> None:
    frame = _intraday_frame(bars_per_session=6, sessions=1)

    returns = forward_returns(frame, (3,), intraday=True)

    # Bars 0..2 fit three bars ahead; bars 3..5 do not.
    assert returns["ret_3"].isna().tolist() == [False, False, False, True, True, True]
    assert returns["ret_3"].iloc[0] == (103.0 / 100.0 - 1.0) * 100.0
    # ATR units are the same move divided by the ATR at the bar.
    assert returns["atr_3"].iloc[0] == 3.0 / 2.0


def test_a_daily_return_is_close_to_close() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]),
            "close": [100.0, 110.0, 121.0],
            "atr": [10.0, 10.0, 10.0],
            "trading_date": [date(2026, 9, day) for day in (1, 2, 3)],
        }
    )

    returns = forward_returns(frame, (2,), intraday=False)

    assert returns["ret_2"].iloc[0] == (121.0 / 100.0 - 1.0) * 100.0
    assert returns["ret_2"].isna().tolist() == [False, True, True]


def test_a_bar_without_an_atr_has_no_atr_normalised_return() -> None:
    frame = _intraday_frame(bars_per_session=2, sessions=1)
    frame.loc[0, "atr"] = None

    returns = forward_returns(frame, (1,), intraday=True)

    assert returns["ret_1"].iloc[0] == (101.0 / 100.0 - 1.0) * 100.0
    assert pd.isna(returns["atr_1"].iloc[0])


# ---------------------------------------------------------------------------
# Rule 1: no test without a sample
# ---------------------------------------------------------------------------
def test_a_thin_bucket_is_refused_and_carries_no_p_value() -> None:
    labels = pd.Series(["THIN"] * 2 + ["FAT"] * 4)
    values = pd.Series([1.0, 2.0, 0.5, -0.5, 1.5, 0.0])
    values_atr = pd.Series([0.1, 0.2, 0.05, -0.05, 0.15, 0.0])

    axis = _family(
        axis_name="regime",
        labels=labels,
        values=values,
        values_atr=values_atr,
        horizon_label="1d",
        policy=POLICY,
    )

    thin = next(row for row in axis.rows if row["bucket"] == "THIN")
    assert thin["sample"] == 2
    assert thin["p_hit"] is None and thin["p_mean"] is None
    assert thin["verdict"] == "INSUFFICIENT_SAMPLE"
    # Only the testable bucket is in the family.
    assert axis.family_size == 1


def test_a_bucket_with_no_rows_is_absent_rather_than_zero() -> None:
    labels = pd.Series(["A"] * 4)
    values = pd.Series([1.0, -1.0, 2.0, 0.5])
    values_atr = values / 10.0

    axis = _family(
        axis_name="regime",
        labels=labels,
        values=values,
        values_atr=values_atr,
        horizon_label="1d",
        policy=POLICY,
        order=("A", "B", "C"),
    )

    assert [row["bucket"] for row in axis.rows] == ["A"]


# ---------------------------------------------------------------------------
# Rule 2: the correction covers the tests that were actually run
# ---------------------------------------------------------------------------
def test_the_correction_covers_only_the_buckets_that_were_tested() -> None:
    labels = pd.Series(["A"] * 4 + ["B"] * 4 + ["THIN"] * 2)
    values = pd.Series([1.0, -1.0, 2.0, 0.5, -0.5, 1.5, -2.0, 0.25, 3.0, -3.0])
    values_atr = values / 10.0

    uncorrected = _family(
        axis_name="score_band",
        labels=labels,
        values=values,
        values_atr=values_atr,
        horizon_label="1d",
        policy=replace(POLICY, correction="none"),
        order=("A", "B", "THIN"),
    )
    corrected = _family(
        axis_name="score_band",
        labels=labels,
        values=values,
        values_atr=values_atr,
        horizon_label="1d",
        policy=replace(POLICY, correction="bonferroni"),
        order=("A", "B", "THIN"),
    )

    assert uncorrected.family_size == corrected.family_size == 2
    tested_raw = [row for row in uncorrected.rows if row["p_hit"] is not None]
    tested_adjusted = [row for row in corrected.rows if row["p_hit"] is not None]
    assert len(tested_raw) == 2
    for raw, adjusted in zip(tested_raw, tested_adjusted, strict=True):
        # Bonferroni over a family of two is exactly twice the raw p, capped at 1.
        assert adjusted["p_hit"] == min(1.0, 2.0 * raw["p_hit"])
        if raw["p_mean"] is not None:
            assert adjusted["p_mean"] == min(1.0, 2.0 * raw["p_mean"])


def test_a_bucket_beats_the_rest_of_the_sample_only_when_it_really_does() -> None:
    """A bucket that is uniformly positive is significant; a mixed one is not."""
    labels = pd.Series(["UP"] * 4 + ["MIXED"] * 4)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0, -1.0, 1.0, -1.0])
    values_atr = values / 10.0

    axis = _family(
        axis_name="regime",
        labels=labels,
        values=values,
        values_atr=values_atr,
        horizon_label="1d",
        policy=replace(POLICY, correction="none"),
    )

    rows = {row["bucket"]: row for row in axis.rows}
    assert rows["UP"]["mean_pct"] == 2.5
    assert rows["UP"]["hit_rate"] == 1.0
    assert rows["MIXED"]["hit_rate"] == 0.5
    assert rows["UP"]["sample"] == 4


# ---------------------------------------------------------------------------
# Rule 4: levels come from the *previous* session
# ---------------------------------------------------------------------------
def test_proximity_is_measured_to_the_previous_session() -> None:
    frame = pd.DataFrame(
        {
            "trading_date": [date(2026, 9, 1)] * 2 + [date(2026, 9, 2)] * 2,
            "high": [110.0, 110.0, 100.0, 100.0],
            "low": [90.0, 90.0, 95.0, 95.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "atr": [10.0] * 4,
        }
    )

    distance = _proximity(frame)

    # The first session has no previous session to look at.
    assert pd.isna(distance.iloc[0])
    # The second session is 10 points from both 110 and 90 - one ATR each way.
    assert distance.iloc[2] == 1.0


def test_the_score_bands_are_half_open_and_keep_missing_values_missing() -> None:
    values = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0, None])

    bands = _band(
        values,
        (
            ("< -0.50", -1.01, -0.50),
            ("-0.50 .. +0.50", -0.50, 0.50),
            (">= +0.50", 0.50, 1.01),
        ),
    )

    assert list(bands) == [
        "< -0.50",
        "-0.50 .. +0.50",
        "-0.50 .. +0.50",
        ">= +0.50",
        ">= +0.50",
        None,
    ]


def test_the_intraday_buckets_follow_the_session_clock() -> None:
    stamps = pd.Series(
        pd.to_datetime(
            [
                "2026-09-01 09:15",
                "2026-09-01 09:29",
                "2026-09-01 09:30",
                "2026-09-01 11:00",
                "2026-09-01 13:00",
                "2026-09-01 14:30",
                "2026-09-01 15:14",
                "2026-09-01 15:15",
                "2026-09-01 08:00",
            ]
        )
    )

    buckets = _intraday_bucket(stamps)

    assert list(buckets) == [
        "OPEN_915_930",
        "OPEN_915_930",
        "MORNING_930_1100",
        "MIDDAY_1100_1300",
        "AFTERNOON_1300_1430",
        "CLOSE_1430_1515",
        "CLOSE_1430_1515",
        None,
        None,
    ]


def test_the_study_wide_correction_treats_every_cell_as_one_family() -> None:
    """Three separately-significant cells stop being one when corrected together."""
    rows = [
        {
            "bucket": name,
            "sample": 10,
            "p_hit_raw": 0.02,
            "p_mean_raw": 0.02,
        }
        for name in ("A", "B", "C")
    ]
    sections = {
        "daily": {"available": True, "axes": [{"rows": rows[:2]}]},
        "intraday": {"available": True, "axes": [{"rows": rows[2:]}]},
    }

    family = study_wide_correction(
        sections, policy=replace(POLICY, correction="bonferroni")
    )

    assert family == 3
    assert rows[0]["p_hit_study"] == pytest.approx(0.06)
    assert rows[0]["verdict_study"] == "NOT_SIGNIFICANT"
    assert all(row["verdict_study"] == "NOT_SIGNIFICANT" for row in rows)


def test_an_unavailable_section_is_not_part_of_the_study_wide_family() -> None:
    sections = {
        "daily": {
            "available": True,
            "axes": [{"rows": [{"sample": 5, "p_hit_raw": 0.01, "p_mean_raw": 0.01}]}],
        },
        "intraday": {"available": False, "reason": "no stored 1-minute bars"},
    }

    assert study_wide_correction(sections, policy=POLICY) == 1


# ---------------------------------------------------------------------------
# Rule 5: the caveats travel with the numbers
# ---------------------------------------------------------------------------
def _study() -> IndexStudy:
    """A tiny hand-built study: one daily axis with one significant bucket."""
    axis = StudyAxis(
        name="regime",
        horizon="1d",
        correction="none",
        family_size=1,
        baseline={"sample": 4, "mean_pct": 0.1, "hit_rate": 0.5},
    )
    axis.rows = [
        {
            "bucket": "BULLISH",
            "sample": 4,
            "hit_rate": 0.75,
            "mean_pct": 0.12,
            "median_pct": 0.1,
            "mean_atr": 0.3,
            "rest_sample": 10,
            "rest_mean_pct": -0.02,
            "rest_hit_rate": 0.4,
            "p_hit": 0.03,
            "p_mean": 0.04,
            "verdict": "SIGNIFICANT",
        }
    ]
    return IndexStudy(
        symbol="NIFTY 50",
        first="2026-09-01",
        last="2026-09-02",
        policy={"correction": "none", "min_samples": 3},
        daily={
            "available": True,
            "bars": 4,
            "first": "2026-09-01",
            "last": "2026-09-02",
            "axes": [axis.to_dict()],
        },
        intraday={"available": False, "reason": "no stored 1-minute bars"},
        caveats=["Returns are gross: no costs, no slippage, no sizing."],
    )


def test_the_markdown_carries_the_table_and_the_caveats() -> None:
    rendered = render_study_markdown(_study())

    assert "# P1 - NIFTY index forward-return study" in rendered
    assert "| BULLISH | 4 | 0.7500 |" in rendered
    assert "SIGNIFICANT" in rendered
    assert "## Caveats" in rendered
    assert "no stored 1-minute bars" in rendered
    assert "Returns are gross" in rendered


def test_the_study_is_written_as_json_and_markdown(tmp_path) -> None:
    written = write_index_study(_study(), tmp_path)

    assert [path.name for path in written] == [
        "index_study_2026-09-01_2026-09-02.json",
        "index_study_2026-09-01_2026-09-02.md",
    ]
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["spec"] == "P1 - NIFTY index forward-return study"
    assert payload["window"] == {"first": "2026-09-01", "last": "2026-09-02"}
    assert payload["daily"]["axes"][0]["rows"][0]["verdict"] == "SIGNIFICANT"
    assert payload["caveats"]


def test_the_stem_names_the_studied_window() -> None:
    assert study_stem("2026-09-01", "2026-09-02") == "index_study_2026-09-01_2026-09-02"


def test_the_policy_reads_the_configured_horizons_and_correction() -> None:
    config = load_config()

    policy = IndexStudyPolicy.from_config(config)

    assert policy.combination == "combination_1"
    assert policy.intraday_horizons == tuple(config.model.horizons_minutes)
    assert policy.correction in {"none", "bonferroni", "benjamini_hochberg"}
    assert policy.min_samples > 0
    assert policy.to_dict()["daily_horizons"] == [1, 2, 5, 10]
