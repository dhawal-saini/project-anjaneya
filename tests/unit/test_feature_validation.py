"""Tests for :mod:`core.features.feature_validation` (spec 19).

Every check has to be able to *fail*: a validation report that only ever says PASS is
worse than no report, because it looks like evidence. Each test therefore plants one
specific defect and asserts both the verdict and the reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    STATUS_WARN,
    CheckResult,
    FeatureValidationReport,
    build_feature_frame,
    merge_reports,
    validate_feature_frame,
)
from core.market_calendar import NseCalendar
from tests.support.engine import (
    CE_SYMBOL,
    feature_frame,
    feature_window,
    make_feature,
)


@pytest.fixture
def calendar(config) -> NseCalendar:
    return NseCalendar(config.calendar)


@pytest.fixture
def window() -> object:
    return feature_window(session_bars=12)


def built(config, calendar, window):
    """The assembled frame plus the inputs the checks recompute from."""
    frame = build_feature_frame(
        candles=window.candles,
        index_candles=window.index,
        calendar=calendar,
        config=config.indicators,
        timeframe="1m",
        underlying="NIFTY",
        run_id="run-1",
        session_start=window.session_start,
        session_end=window.session_end,
    )
    return frame


def validate(config, calendar, window, frame, **overrides):
    arguments = {
        "candles": window.candles,
        "index_candles": window.index,
        "calendar": calendar,
        "config": config.indicators,
        "timeframe": "1m",
        "session_start": window.session_start,
        "session_end": window.session_end,
        "probes": 2,
    }
    arguments.update(overrides)
    return validate_feature_frame(frame, **arguments)


def named(report, name):
    return next(check for check in report.checks if check.name == name)


# ---------------------------------------------------------------------------
# A clean build passes
# ---------------------------------------------------------------------------
def test_a_clean_build_passes_every_check(config, calendar, window) -> None:
    report = validate(config, calendar, window, built(config, calendar, window))

    assert report.status == "PASS_WITH_WARNINGS"  # the greeks are NULL by design
    assert [check.name for check in report.failures] == []
    assert named(report, "no_look_ahead").checked == 2
    assert named(report, "unique_keys").issues == 0
    assert named(report, "ohlc_sanity").issues == 0
    assert report.row_count == 12


def test_the_report_renders_as_json_shaped_data(config, calendar, window) -> None:
    report = validate(config, calendar, window, built(config, calendar, window))
    payload = report.to_dict()

    assert payload["status"] == report.status
    assert payload["row_count"] == 12
    assert {check["name"] for check in payload["checks"]} >= {
        "unique_keys",
        "ohlc_sanity",
        "no_look_ahead",
        "missing_values",
    }
    assert report.summary_line().startswith(report.status)


# ---------------------------------------------------------------------------
# The checks that must be able to fail
# ---------------------------------------------------------------------------
def test_an_empty_scope_fails_with_a_clear_reason(config, calendar, window) -> None:
    report = validate_feature_frame(
        feature_frame([]), timeframe="1m", scope={"day": "2026-09-29"}
    )

    assert report.status == STATUS_FAIL
    assert report.checks[0].name == "rows_present"
    assert "no feature rows" in report.checks[0].detail


def test_an_infinite_value_fails(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    frame.loc[0, "option_atr"] = np.inf
    report = validate(config, calendar, window, frame)

    check = named(report, "infinite_values")
    assert check.status == STATUS_FAIL
    assert "option_atr" in check.detail
    assert report.status == STATUS_FAIL


def test_a_duplicate_key_fails(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate(config, calendar, window, doubled)

    check = named(report, "unique_keys")
    assert check.status == STATUS_FAIL
    assert check.issues == 2
    assert CE_SYMBOL in check.samples[0]


def test_a_null_in_an_identity_column_fails(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    frame.loc[0, "run_id"] = None
    report = validate(config, calendar, window, frame)

    check = named(report, "identity_present")
    assert check.status == STATUS_FAIL
    assert "run_id" in check.detail


def test_an_undeclared_column_fails(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    frame["future_return"] = 0.0
    report = validate(config, calendar, window, frame)

    check = named(report, "declared_columns")
    assert check.status == STATUS_FAIL
    assert "future_return" in check.detail


def test_timestamps_must_increase(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    frame = frame.iloc[::-1].reset_index(drop=True)
    report = validate(config, calendar, window, frame)

    assert named(report, "timestamp_ordering").status == STATUS_FAIL


def test_a_broken_ohlc_bar_fails(config, calendar, window) -> None:
    candles = window.candles.copy()
    candles.loc[0, "high"] = candles.loc[0, "low"] - 1.0
    report = validate(config, calendar, window, built(config, calendar, window), candles=candles)

    check = named(report, "ohlc_sanity")
    assert check.status == STATUS_FAIL
    assert "impossible OHLC" in check.detail


def test_negative_volume_fails(config, calendar, window) -> None:
    candles = window.candles.copy()
    candles.loc[0, "volume"] = -5
    report = validate(config, calendar, window, built(config, calendar, window), candles=candles)

    check = named(report, "volume_sanity")
    assert check.status == STATUS_FAIL
    assert "volume=1" in check.detail


def test_an_expired_or_unidentified_contract_fails(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    frame.loc[0, "strike"] = -1.0
    frame.loc[1, "option_type"] = "XX"
    report = validate(config, calendar, window, frame)

    check = named(report, "contract_identity")
    assert check.status == STATUS_FAIL
    assert "strike" in check.detail and "option_type" in check.detail


def test_a_missing_bar_warns_rather_than_fails(config, calendar, window) -> None:
    """A gap is a data-coverage fact: it is reported, never fabricated."""
    frame = built(config, calendar, window).drop(index=[3]).reset_index(drop=True)
    report = validate(config, calendar, window, frame)

    check = named(report, "session_gaps")
    assert check.status == STATUS_WARN
    assert "1 of 12" in check.detail
    assert report.passed


def test_an_empty_column_warns_with_the_mode_that_fills_it(config, calendar, window) -> None:
    report = validate(config, calendar, window, built(config, calendar, window))

    check = named(report, "missing_values")
    assert check.status == STATUS_WARN
    assert "iv" in check.detail and "enrich-greeks" in check.detail


# ---------------------------------------------------------------------------
# The greeks and the relative block (spec 16)
# ---------------------------------------------------------------------------
def test_greeks_sanity_skips_when_the_frame_carries_no_greek_columns() -> None:
    """A frame without the columns is not a greek failure: the check says so."""
    frame = feature_frame([make_feature()]).drop(columns=["iv", "delta", "gamma", "theta", "vega"])
    check = named(validate_feature_frame(frame, timeframe="1m"), "greeks_sanity")

    assert check.status == STATUS_SKIPPED
    assert "no greek columns" in check.detail


def test_greeks_sanity_warns_when_every_greek_is_null(config, calendar, window) -> None:
    """An unenriched table is incomplete, not corrupt - and the mode is named."""
    check = named(validate(config, calendar, window, built(config, calendar, window)), "greeks_sanity")

    assert check.status == STATUS_WARN
    assert "enrich-greeks" in check.detail


def test_greeks_sanity_passes_for_enriched_rows() -> None:
    """A plausible row - call delta positive, put delta negative - passes."""
    frame = feature_frame(
        [
            make_feature(iv=0.14, delta=0.52, gamma=0.0003, theta=-12.0, vega=8.5),
            make_feature(
                option_type="PE", tradingsymbol="NIFTY01OCT2625000PE", iv=0.15, delta=-0.48
            ),
        ]
    )
    check = named(validate_feature_frame(frame, timeframe="1m"), "greeks_sanity")

    assert check.status == STATUS_PASS
    assert "inside their model range" in check.detail


def test_greeks_sanity_fails_on_an_impossible_value() -> None:
    """An outrageous IV, a gamma below zero or a delta outside [-1, 1] cannot be a market."""
    frame = feature_frame(
        [
            make_feature(iv=12.0, delta=1.4, gamma=-0.0001, theta=0.0, vega=0.0),
        ]
    )
    check = named(validate_feature_frame(frame, timeframe="1m"), "greeks_sanity")

    assert check.status == STATUS_FAIL
    assert "iv outside" in check.detail
    assert "delta outside" in check.detail
    assert "gamma outside" in check.detail


def test_the_row_contract_itself_rejects_a_negative_iv() -> None:
    """Layered defence: the schema refuses an impossible IV before validation ever sees it."""
    with pytest.raises(Exception, match="greater than or equal to 0"):
        make_feature(iv=-0.02)


def test_greeks_sanity_fails_when_a_delta_has_the_wrong_sign_for_its_type() -> None:
    """A call with a negative delta is a swapped option type, not a market."""
    frame = feature_frame(
        [
            make_feature(iv=0.14, delta=-0.5),  # CE with a put-like delta
            make_feature(option_type="PE", tradingsymbol="NIFTY01OCT2625000PE", iv=0.14, delta=0.5),
        ]
    )
    check = named(validate_feature_frame(frame, timeframe="1m"), "greeks_sanity")

    assert check.status == STATUS_FAIL
    assert "wrong sign" in check.detail
    assert len(check.samples) == 2


def test_relative_identity_passes_and_fails_on_the_definition() -> None:
    """``residual == actual - expected`` is checked, not assumed."""
    good = feature_frame(
        [
            make_feature(
                expected_option_move=1.5, actual_option_move=2.0, option_residual=0.5
            ),
        ]
    )
    assert named(validate_feature_frame(good, timeframe="1m"), "relative_identity").status == (
        STATUS_PASS
    )

    broken = feature_frame(
        [
            make_feature(
                expected_option_move=1.5, actual_option_move=2.0, option_residual=0.25
            ),
        ]
    )
    check = named(validate_feature_frame(broken, timeframe="1m"), "relative_identity")
    assert check.status == STATUS_FAIL
    assert "option_residual !=" in check.detail


def test_relative_identity_warns_when_no_row_is_complete(config, calendar, window) -> None:
    """Without a delta the three columns are NULL, and the warning says what fills them."""
    check = named(
        validate(config, calendar, window, built(config, calendar, window)), "relative_identity"
    )

    assert check.status == STATUS_WARN
    assert "enrich-greeks" in check.detail


def test_a_check_that_cannot_run_says_so_instead_of_passing_silently(config, calendar) -> None:
    frame = feature_frame([make_feature()])
    report = validate_feature_frame(frame, timeframe="1m")

    assert named(report, "ohlc_sanity").status == STATUS_SKIPPED
    assert named(report, "no_look_ahead").status == STATUS_SKIPPED
    assert named(report, "session_gaps").status == STATUS_SKIPPED


def test_an_empty_index_frame_is_reported_not_assumed(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    report = validate(config, calendar, window, frame, index_candles=None)

    assert named(report, "no_look_ahead").status in {STATUS_PASS, STATUS_FAIL}


def test_zero_probes_skips_the_recomputation(config, calendar, window) -> None:
    report = validate(config, calendar, window, built(config, calendar, window), probes=0)
    assert named(report, "no_look_ahead").status == STATUS_SKIPPED


def test_an_empty_feature_frame_is_reported_as_missing_rows(config, calendar, window) -> None:
    everything = built(config, calendar, window)
    empty = everything.iloc[0:0]
    report = validate_feature_frame(empty, timeframe="1m")
    assert report.checks[0].status == STATUS_FAIL
    assert report.checks[0].name == "rows_present"


# ---------------------------------------------------------------------------
# Look-ahead: the check that cannot be faked
# ---------------------------------------------------------------------------
def test_a_future_peeking_engine_is_caught(config, calendar, window, monkeypatch) -> None:
    """A feature that reads the *last* bar of the window must fail the probe."""
    frame = built(config, calendar, window)
    from core.features import feature_builder

    real = feature_builder.contract_feature_block

    def peeking(candles, **kwargs):
        block = real(candles, **kwargs)
        block["candle_body"] = float(candles["close"].iloc[-1])
        return block

    monkeypatch.setattr(feature_builder, "contract_feature_block", peeking)
    report = validate(config, calendar, window, frame)

    check = named(report, "no_look_ahead")
    assert check.status == STATUS_FAIL
    assert "candle_body" in check.samples[0]


def test_the_probe_names_the_column_it_exempts_and_why(config, calendar, window) -> None:
    report = validate(config, calendar, window, built(config, calendar, window))
    check = named(report, "no_look_ahead")

    assert "option_insufficient_history" in check.detail


# ---------------------------------------------------------------------------
# Merging contract reports
# ---------------------------------------------------------------------------
def test_merging_worst_status_wins(config, calendar, window) -> None:
    good = validate(config, calendar, window, built(config, calendar, window))
    bad_frame = built(config, calendar, window)
    bad_frame.loc[0, "option_atr"] = np.inf
    bad = validate(config, calendar, window, bad_frame)

    merged = merge_reports([good, bad], scope={"day": "2026-09-29"})

    assert merged.status == STATUS_FAIL
    assert merged.row_count == good.row_count + bad.row_count
    assert named(merged, "infinite_values").issues == 1
    assert merged.scope["day"] == "2026-09-29"


def test_merging_keeps_every_check_name(config, calendar, window) -> None:
    frame = built(config, calendar, window)
    one = validate(config, calendar, window, frame)
    merged = merge_reports([one, one])

    assert {check.name for check in merged.checks} == {check.name for check in one.checks}


def test_merging_nothing_is_a_failure_not_a_pass() -> None:
    assert merge_reports([]).status == STATUS_FAIL


def test_the_verdict_rules_are_worst_status_wins_with_warnings_allowed() -> None:
    passing = FeatureValidationReport(checks=[CheckResult(name="a", status=STATUS_PASS)])
    assert passing.status == STATUS_PASS and passing.passed

    warned = FeatureValidationReport(
        checks=[CheckResult(name="a", status=STATUS_PASS), CheckResult(name="b", status=STATUS_WARN)]
    )
    assert warned.status == "PASS_WITH_WARNINGS" and warned.passed

    failed = FeatureValidationReport(
        checks=[CheckResult(name="a", status=STATUS_WARN), CheckResult(name="b", status=STATUS_FAIL)]
    )
    assert failed.status == STATUS_FAIL and not failed.passed
    assert [check.name for check in failed.failures] == ["b"]

    nothing_ran = FeatureValidationReport(
        checks=[CheckResult(name="a", status=STATUS_SKIPPED), CheckResult(name="b", status=STATUS_SKIPPED)]
    )
    assert nothing_ran.status == STATUS_FAIL and not nothing_ran.passed
