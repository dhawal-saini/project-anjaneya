"""Unit tests for the purged walk-forward splitter (``core/ml/walk_forward.py``, spec 22).

Spec 22 is a rule about *boundaries*: no fold may train on a row whose label was read from
the window it is validated on, and the final holdout must never be touched. So the tests
mostly check the two guards - the embargo (time) and the purge (label window) - on a
fixture whose rows run through a whole calendar day, which is the only way a 30-minute
embargo can bite at all.

The other half is bookkeeping: which days a fold claims, the exact first/last timestamps of
each window, and the honest "why there are no folds yet" answer on a short dataset.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.ml import (
    Dataset,
    DatasetCounts,
    DatasetPolicy,
    SplitError,
    SplitPolicy,
    build_dataset,
    fold_frames,
    holdout_frame,
    purged_walk_forward_splits,
    usable_frame,
)
from tests.support.ai import (
    CLASSES,
    REGIMES,
    feature_frame,
    label_frame,
)
from tests.support.ai import (
    DAYS as SESSION_DAYS,
)

IST = ZoneInfo("Asia/Kolkata")

#: Eight consecutive days starting on a Monday, so "day 3" is unambiguous.
DAYS: tuple[date, ...] = tuple(date(2026, 9, 7) + timedelta(days=offset) for offset in range(8))

#: Rows per synthetic day: one per minute of the calendar day.
ROWS_PER_DAY = 1440

#: A minimal policy - the splitter only needs the class list and the timeframe.
POLICY = DatasetPolicy(
    horizons=(5, 10, 15, 30),
    classes=CLASSES,
    timeframe="1m",
    categorical_columns=("option_type",),
    label_version="v1",
)


def dense_dataset(*, days: tuple[date, ...] = DAYS, rows_per_day: int = ROWS_PER_DAY):
    """A dataset whose rows run 00:00-23:59 on each day.

    Real sessions do not span a calendar day, which is exactly why this fixture does: it
    puts a train window and the validation day that follows it 30 minutes apart, so the
    embargo has something to remove and the purge boundary is a real timestamp.
    """
    features, labels = [], []
    for day in days:
        midnight = datetime(day.year, day.month, day.day, tzinfo=IST)
        for minute in range(rows_per_day):
            moment = midnight + timedelta(minutes=minute)
            features.append(
                {
                    "timestamp": moment,
                    "tradingsymbol": "AAA",
                    "timeframe": "1m",
                    "option_type": "CE",
                    "nifty_regime": REGIMES[minute % 4],
                    "option_atr": 1.0 + minute / 1000,
                }
            )
            labels.append(
                {
                    "timestamp": moment,
                    "tradingsymbol": "AAA",
                    "horizon_minutes": 5,
                    "label": CLASSES[minute % 4],
                    "label_version": "v1",
                }
            )
    return build_dataset(pd.DataFrame(features), pd.DataFrame(labels), policy=POLICY, horizon=5)


def split_policy(**overrides) -> SplitPolicy:
    """A small, legible split policy (the shipped one needs 70 days)."""
    values = {
        "train_window_days": 2,
        "validate_window_days": 1,
        "step_days": 1,
        "embargo_minutes": 30,
        "holdout_days": 1,
        "horizon_minutes": 5,
    }
    values.update(overrides)
    return SplitPolicy(**values)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_split_policy_comes_from_the_shipped_config(config) -> None:
    policy = SplitPolicy.from_config(config, horizon_minutes=30, timezone_name=config.timezone_name)

    # Relaxed 90/15 -> 30/10 (Phase 9) so the first exploratory walk-forward can run
    # on a thinner accumulation; the embargo still covers the 30-minute horizon.
    assert policy.train_window_days == 30
    assert policy.validate_window_days == 10
    assert policy.step_days == 15
    assert policy.embargo_minutes == 30
    assert policy.holdout_days == 30
    assert policy.horizon_minutes == 30
    assert policy.purge is True
    assert policy.days_required == 70


def test_policy_describe_lists_every_choice() -> None:
    described = split_policy().describe()

    assert described["train_window_days"] == 2
    assert described["days_required"] == 4
    assert described["purge"] is True
    assert set(described) == {
        "train_window_days",
        "validate_window_days",
        "step_days",
        "embargo_minutes",
        "holdout_days",
        "horizon_minutes",
        "purge",
        "days_required",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embargo_minutes": 4}, "embargo_minutes"),
        ({"step_days": 5}, "step_days"),
        ({"train_window_days": 0}, "positive"),
        ({"holdout_days": 0}, "positive"),
    ],
)
def test_a_policy_that_cannot_be_leak_free_is_refused(overrides, message) -> None:
    with pytest.raises(SplitError, match=message):
        split_policy(**overrides)


def test_the_embargo_must_cover_the_longest_horizon() -> None:
    """Spec 22's rule, enforced by the splitter itself rather than only by the config."""
    with pytest.raises(SplitError, match="must be >= the label horizon"):
        SplitPolicy(
            train_window_days=2,
            validate_window_days=1,
            step_days=1,
            embargo_minutes=29,
            holdout_days=1,
            horizon_minutes=30,
        )


# ---------------------------------------------------------------------------
# The walk: whole days, in order, holdout removed first
# ---------------------------------------------------------------------------
def test_folds_walk_forward_day_by_day() -> None:
    built = dense_dataset(days=DAYS[:6])

    split = purged_walk_forward_splits(built, policy=split_policy())

    assert [fold.train_days for fold in split.folds] == [
        (DAYS[0], DAYS[1]),
        (DAYS[1], DAYS[2]),
        (DAYS[2], DAYS[3]),
    ]
    assert [fold.validate_days for fold in split.folds] == [
        (DAYS[2],),
        (DAYS[3],),
        (DAYS[4],),
    ]
    assert [fold.index for fold in split.folds] == [0, 1, 2]


def test_every_validation_window_follows_its_training_window() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:6]), policy=split_policy())

    for fold in split.folds:
        assert max(fold.train_days) < min(fold.validate_days)
        assert fold.validate_start is not None and fold.train_end is not None
        assert fold.validate_start > fold.train_end


def test_the_holdout_is_the_last_days_and_is_in_no_fold() -> None:
    dataset = dense_dataset()
    split = purged_walk_forward_splits(dataset, policy=split_policy(holdout_days=2))

    assert split.holdout.days == DAYS[-2:]
    assert split.holdout.rows == 2 * ROWS_PER_DAY
    assert split.holdout.start.date() == DAYS[-2]
    assert split.holdout.end.date() == DAYS[-1]
    for fold in split.folds:
        assert not set(fold.train_days) & set(split.holdout.days)
        assert not set(fold.validate_days) & set(split.holdout.days)


def test_a_step_larger_than_the_train_window_leaves_gaps() -> None:
    split = purged_walk_forward_splits(
        dense_dataset(), policy=split_policy(train_window_days=3, step_days=2, holdout_days=1)
    )

    assert [fold.train_days for fold in split.folds][:2] == [
        (DAYS[0], DAYS[1], DAYS[2]),
        (DAYS[2], DAYS[3], DAYS[4]),
    ]


def test_days_the_dataset_does_not_have_are_not_invented() -> None:
    days = (DAYS[0], DAYS[1], DAYS[3], DAYS[4])
    split = purged_walk_forward_splits(
        dense_dataset(days=days), policy=split_policy(holdout_days=1)
    )

    assert split.days == len(days)
    assert split.holdout.days == (DAYS[4],)
    assert split.folds[0].train_days == (DAYS[0], DAYS[1])
    assert split.folds[0].validate_days == (DAYS[3],)


def test_a_short_dataset_returns_no_folds_and_says_why() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:2]), policy=split_policy())

    assert split.ok is False
    assert split.folds == []
    assert "needs 4 captured day(s)" in split.reason
    assert "this dataset has 2" in split.reason
    assert "no folds" in split.describe()


def test_an_empty_dataset_says_there_is_nothing_to_split() -> None:
    empty = dense_dataset(days=DAYS[:1]).frame.iloc[0:0]

    built = Dataset(
        horizon=5,
        timeframe="1m",
        frame=empty,
        feature_columns=(),
        categorical_columns=(),
        classes=CLASSES,
        counts=DatasetCounts(),
    )

    split = purged_walk_forward_splits(built, policy=split_policy())

    assert split.ok is False
    assert "no labelled rows" in split.reason
    assert split.holdout is None


# ---------------------------------------------------------------------------
# The two guards on the boundary: embargo (time) and purge (label window)
# ---------------------------------------------------------------------------
def test_the_embargo_removes_the_tail_of_the_training_window() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:4]), policy=split_policy())

    fold = split.folds[0]
    validate_start = datetime(DAYS[2].year, DAYS[2].month, DAYS[2].day, tzinfo=IST)
    # 23:31-23:59 sit inside the last half hour of the last training day.
    assert fold.embargoed_rows == 29
    assert fold.removed_rows == 29
    assert fold.train_rows == 2 * ROWS_PER_DAY - 29
    assert fold.embargo_cutoff == validate_start - timedelta(minutes=30)
    # The row *on* the cut-off stays: its window [23:30, 23:35) ends before the next day.
    assert fold.train_end == validate_start - timedelta(minutes=30)
    # The purge is the same guarantee expressed through the label window; with an embargo
    # that covers the horizon, the purged rows are a subset of the embargoed ones.
    assert fold.purged_rows == 4


def test_the_purge_counts_the_rows_whose_label_window_reaches_the_validation_window() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:4]), policy=split_policy())

    fold = split.folds[0]
    validate_start = datetime(DAYS[2].year, DAYS[2].month, DAYS[2].day, tzinfo=IST)

    assert fold.purge_cutoff == validate_start - timedelta(minutes=5)
    # Every purged row is embargoed too (embargo >= horizon), which is the point of the
    # policy refusing an embargo shorter than the longest horizon.
    assert fold.purged_rows < fold.embargoed_rows == fold.removed_rows


def test_the_purge_flag_only_changes_what_is_reported() -> None:
    days = DAYS[:4]
    with_purge = purged_walk_forward_splits(dense_dataset(days=days), policy=split_policy())
    without = purged_walk_forward_splits(
        dense_dataset(days=days), policy=split_policy(purge=False)
    )

    assert without.folds[0].train_rows == with_purge.folds[0].train_rows
    assert without.folds[0].removed_rows == without.folds[0].embargoed_rows
    assert without.folds[0].purged_rows == with_purge.folds[0].purged_rows


def test_a_row_exactly_on_the_embargo_boundary_is_kept() -> None:
    """The cut-offs are strict: a label window that *ends* at the boundary does not leak."""
    split = purged_walk_forward_splits(
        dense_dataset(days=DAYS[:4]), policy=split_policy(embargo_minutes=5)
    )

    fold = split.folds[0]
    validate_start = datetime(DAYS[2].year, DAYS[2].month, DAYS[2].day, tzinfo=IST)

    # 23:56-23:59 are inside five minutes of the validation morning; 23:55 is not.
    assert fold.embargoed_rows == 4
    assert fold.purged_rows == 4
    assert fold.train_end == validate_start - timedelta(minutes=5)


def test_the_embargo_does_nothing_when_a_session_ends_far_from_the_next_day() -> None:
    """Day-separated sessions: the 30-minute embargo is trivially satisfied."""
    built = build_dataset(
        feature_frame(days=SESSION_DAYS[:4]),
        label_frame(days=SESSION_DAYS[:4]),
        policy=POLICY,
        horizon=5,
    )

    split = purged_walk_forward_splits(
        built,
        policy=split_policy(train_window_days=2, validate_window_days=1, holdout_days=1),
    )

    fold = split.folds[0]
    assert fold.embargoed_rows == 0
    assert fold.purged_rows == 0
    assert fold.removed_rows == 0


# ---------------------------------------------------------------------------
# The report: exact timestamps, counts that add up, deterministic output
# ---------------------------------------------------------------------------
def test_fold_timestamps_are_the_exact_window_bounds() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:4]), policy=split_policy())
    fold = split.folds[0]

    assert fold.train_start.date() == DAYS[0]
    assert fold.train_start.hour == 0 and fold.train_start.minute == 0
    assert fold.validate_start.date() == DAYS[2]
    assert fold.validate_end.date() == DAYS[2]
    assert fold.validate_end == fold.validate_start + timedelta(minutes=ROWS_PER_DAY - 1)


def test_fold_class_counts_add_up() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:4]), policy=split_policy())

    for fold in split.folds:
        assert sum(fold.train_class_counts.values()) == fold.train_rows
        assert sum(fold.validate_class_counts.values()) == fold.validate_rows
        assert set(fold.train_class_counts) == set(CLASSES)


def test_the_split_totals_are_the_sum_of_its_folds() -> None:
    dataset = dense_dataset(days=DAYS[:5])
    split = purged_walk_forward_splits(dataset, policy=split_policy())

    assert split.rows == dataset.rows
    assert split.days == len(DAYS[:5])
    assert split.train_rows == sum(fold.train_rows for fold in split.folds)
    assert split.validate_rows == sum(fold.validate_rows for fold in split.folds)
    assert split.horizon == dataset.horizon


def test_the_split_manifest_is_json_ready_and_documents_the_windows() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:4]), policy=split_policy())

    payload = json.loads(json.dumps(split.to_dict(), default=str))

    assert payload["horizon_minutes"] == 5
    assert payload["folds"] == 1
    assert payload["policy"]["embargo_minutes"] == 30
    assert payload["holdout"]["days"] == [DAYS[3].isoformat()]
    fold = payload["splits"][0]
    assert fold["train_days"] == [DAYS[0].isoformat(), DAYS[1].isoformat()]
    assert fold["validate_first_timestamp"].startswith(DAYS[2].isoformat())
    assert fold["embargo_cutoff"].endswith("+05:30")


def test_splitting_the_same_dataset_twice_gives_the_same_answer() -> None:
    dataset = dense_dataset(days=DAYS[:5])

    first = purged_walk_forward_splits(dataset, policy=split_policy()).to_dict()
    second = purged_walk_forward_splits(dataset, policy=split_policy()).to_dict()

    assert first == second


def test_split_describe_names_the_folds_and_the_holdout() -> None:
    split = purged_walk_forward_splits(dense_dataset(days=DAYS[:5]), policy=split_policy())

    assert split.describe() == "horizon 5m: 2 fold(s) over 5 day(s), holdout 1 day(s) untouched"


# ---------------------------------------------------------------------------
# Materialising a split: the frames the trainer is allowed to see
# ---------------------------------------------------------------------------
def test_fold_frames_hand_out_exactly_the_rows_the_fold_counted() -> None:
    dataset = dense_dataset(days=DAYS[:5])
    split = purged_walk_forward_splits(dataset, policy=split_policy())

    for fold in split.folds:
        train_frame, validate_frame = fold_frames(dataset, fold)
        assert len(train_frame) == fold.train_rows
        assert len(validate_frame) == fold.validate_rows
        # The guards are *in* the frames: nothing strictly inside the embargo of the
        # validation window survives (a row exactly on the cutoff is kept - the mask is
        # "timestamp > embargo_cutoff" - and the purge cutoff is respected too).
        if not train_frame.empty:
            assert train_frame["timestamp"].max() <= fold.embargo_cutoff
            assert train_frame["timestamp"].max() <= fold.purge_cutoff
        assert validate_frame["timestamp"].min() == fold.validate_start
        assert validate_frame["timestamp"].max() == fold.validate_end


def test_fold_frames_are_disjoint_on_the_key() -> None:
    dataset = dense_dataset(days=DAYS[:5])
    split = purged_walk_forward_splits(dataset, policy=split_policy())
    train_frame, validate_frame = fold_frames(dataset, split.folds[0])
    overlap = set(zip(train_frame["timestamp"], train_frame["tradingsymbol"], strict=True)) & set(
        zip(validate_frame["timestamp"], validate_frame["tradingsymbol"], strict=True)
    )
    assert overlap == set()


def test_fold_frames_agree_with_the_policy_without_the_purge() -> None:
    dataset = dense_dataset(days=DAYS[:5])
    with_purge = purged_walk_forward_splits(dataset, policy=split_policy())
    without = purged_walk_forward_splits(dataset, policy=split_policy(purge=False))

    assert with_purge.folds[0].purge is True
    assert without.folds[0].purge is False
    # The purge only ever removes *more*: without it, the embargo is the only guard.
    assert len(fold_frames(dataset, without.folds[0])[0]) >= len(
        fold_frames(dataset, with_purge.folds[0])[0]
    )
    assert without.folds[0].to_dict()["purge"] is False


def test_holdout_frame_is_the_untouched_window() -> None:
    dataset = dense_dataset(days=DAYS[:5])
    split = purged_walk_forward_splits(dataset, policy=split_policy())

    holdout = holdout_frame(dataset, split)
    assert len(holdout) == split.holdout.rows
    assert set(holdout["timestamp"].dt.date) == {DAYS[4]}


def test_usable_frame_is_everything_outside_the_holdout() -> None:
    dataset = dense_dataset(days=DAYS[:5])
    split = purged_walk_forward_splits(dataset, policy=split_policy())

    usable = usable_frame(dataset, split)
    assert len(usable) + split.holdout.rows == dataset.rows
    assert DAYS[4] not in set(usable["timestamp"].dt.date)
    # Every fold's training and validation rows come from the usable window.
    for fold in split.folds:
        train_frame, validate_frame = fold_frames(dataset, fold)
        assert set(train_frame["timestamp"]) <= set(usable["timestamp"])
        assert set(validate_frame["timestamp"]) <= set(usable["timestamp"])


def test_a_split_without_a_holdout_hands_out_the_whole_dataset() -> None:
    class Empty:
        """A split double whose holdout never got built (a short dataset)."""

        holdout = None

    dataset = dense_dataset(days=DAYS[:5])
    assert len(usable_frame(dataset, Empty())) == dataset.rows
    assert holdout_frame(dataset, Empty()).empty





