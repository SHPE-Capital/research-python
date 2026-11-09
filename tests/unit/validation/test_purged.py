from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.validation.purged import purged_walk_forward


def _meta(n: int = 120, hold_hours: int | np.ndarray = 30, freq: str = "6h") -> pd.DataFrame:
    decision = to_utc_ns(pd.Series(pd.date_range("2026-01-02T15:00Z", periods=n, freq=freq)))
    return pd.DataFrame({"decision_ts": decision, "label_end_ts": decision + pd.to_timedelta(hold_hours, unit="h")})


def test_rolling_window():
    meta = _meta(120, hold_hours=1)
    folds = list(purged_walk_forward(meta, n_folds=4, expanding=False))
    assert len(folds) == 4
    block = 120 // 5
    for k, (train, test) in enumerate(folds):
        assert list(test) == list(range(block * (k + 1), block * (k + 2)))
        assert list(train) == list(range(block * k, block * (k + 1)))


def test_expanding_trains_on_all_prior_blocks():
    meta = _meta(120, hold_hours=1)
    folds = list(purged_walk_forward(meta, n_folds=4))
    assert [len(t) for t, _ in folds] == [24, 48, 72, 96]


def test_purge_removes_overlapping_rows():
    meta = _meta(120, hold_hours=30)  # 5-row labels
    for train, test in purged_walk_forward(meta, n_folds=4):
        test_start = meta["decision_ts"].iloc[test].min()
        assert (meta["label_end_ts"].iloc[train] < test_start).all()
        assert train.max() == test.min() - 6  # 5 rows purged plus the boundary


def test_missing_label_end_raises():
    meta = _meta(50)
    meta.loc[10, "label_end_ts"] = pd.NaT
    with pytest.raises(ValueError):
        list(purged_walk_forward(meta, n_folds=3))


def test_same_session_labels_unpurged():
    meta = _meta(120, hold_hours=1)  # label ends before the next decision
    for train, test in purged_walk_forward(meta, n_folds=4):
        assert list(train) == list(range(test.min()))


def test_positions_follow_input_order_not_index():
    meta = _meta(60, hold_hours=1).iloc[::-1].reset_index(drop=True).set_axis(range(100, 160))
    folds = list(purged_walk_forward(meta, n_folds=2))
    for train, test in folds:
        assert meta["decision_ts"].iloc[train].max() < meta["decision_ts"].iloc[test].min()
        assert test.max() < 60


def test_embargo_drops_training_rows_after_test_block():
    # Hand-built case: rows after the test block would be dropped; walk-forward never yields them,
    # so just check a large embargo does not change walk-forward folds.
    meta = _meta(120, hold_hours=1)
    a = list(purged_walk_forward(meta, n_folds=4))
    b = list(purged_walk_forward(meta, n_folds=4, embargo=pd.Timedelta(days=30)))
    for (ta, _), (tb, _) in zip(a, b, strict=True):
        assert list(ta) == list(tb)


def test_deterministic():
    rng = np.random.default_rng(1)
    meta = _meta(200, hold_hours=rng.integers(1, 60, 200))
    a = list(purged_walk_forward(meta, n_folds=4, embargo=pd.Timedelta(hours=12)))
    b = list(purged_walk_forward(meta, n_folds=4, embargo=pd.Timedelta(hours=12)))
    for (t1, s1), (t2, s2) in zip(a, b, strict=True):
        assert (t1 == t2).all() and (s1 == s2).all()


def test_too_many_folds_raises():
    with pytest.raises(ValueError):
        list(purged_walk_forward(_meta(3), n_folds=5))
