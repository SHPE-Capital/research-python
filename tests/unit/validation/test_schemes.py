from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.pipeline.registry import FIT_SCHEME_REGISTRY, build_fit_scheme
from shpe_research.validation.schemes import (
    PurgedWalkForward,
    SingleScheme,
    WalkForward,
    parse_embargo,
)


def _meta(n_sessions: int = 60, per_day: int = 4, hold_days: float = 0.0, hold_hours: float = 1.0) -> pd.DataFrame:
    days = pd.date_range("2026-01-05", periods=n_sessions, tz="UTC", freq="D")
    decision = pd.DatetimeIndex([d + pd.Timedelta(hours=15 + i) for d in days for i in range(per_day)])
    hold = pd.Timedelta(days=hold_days) if hold_days else pd.Timedelta(hours=hold_hours)
    meta = pd.DataFrame({"decision_ts": to_utc_ns(pd.Series(decision))})
    meta["label_end_ts"] = meta["decision_ts"] + hold
    meta["era"] = meta["decision_ts"].dt.strftime("%Y-%m-%d")
    return meta


SCHEMES = [
    pytest.param(lambda **kw: PurgedWalkForward(n_folds=4, embargo="3 sessions", test_block=0.2, **kw), id="purged"),
    pytest.param(lambda **kw: WalkForward(n_folds=4, test_block=0.2, **kw), id="walk_forward"),
]


@pytest.mark.parametrize("make", SCHEMES)
def test_test_block_untouched(make):
    meta = _meta(hold_days=2)
    scheme = make()
    test_pos = set(np.flatnonzero(scheme.test_mask(meta)))
    assert test_pos  # a test block exists
    assert meta["era"].iloc[sorted(test_pos)].min() > meta["era"].iloc[0]
    folds = list(scheme.folds(meta))
    assert len(folds) == 4
    for fold in folds:
        assert not test_pos & set(fold.train_pos)
        assert not test_pos & set(fold.val_pos)


def test_purge_no_overlap():
    meta = _meta(hold_days=2)
    scheme = PurgedWalkForward(n_folds=4, embargo="3 sessions", test_block=0.2)
    for fold in scheme.folds(meta):
        val = meta.iloc[fold.val_pos]
        train = meta.iloc[fold.train_pos]
        assert len(train) > 0
        val_start, val_end = val["decision_ts"].min(), val["label_end_ts"].max()
        overlaps = (train["label_end_ts"] >= val_start) & (train["decision_ts"] <= val_end)
        assert not overlaps.any()
        assert fold.train_end < fold.val_start
        assert fold.val_start == val_start and fold.val_end == val_end


def test_walk_forward_does_not_purge():
    meta = _meta(hold_days=2)
    wf = list(WalkForward(n_folds=4, test_block=0.2).folds(meta))
    pw = list(PurgedWalkForward(n_folds=4, embargo="3 sessions", test_block=0.2).folds(meta))
    assert all(len(a.train_pos) > len(b.train_pos) for a, b in zip(wf, pw, strict=True))


def test_embargo_too_short_raises():
    meta = _meta(hold_days=3)
    meta.attrs["label_name"] = "fwd_return_3d"
    scheme = PurgedWalkForward(n_folds=4, embargo="1 hour", test_block=0.2)
    with pytest.raises(ValueError, match="fwd_return_3d"):
        list(scheme.folds(meta))


def test_era_boundaries():
    meta = _meta(hold_hours=1)
    era_starts = set(meta.groupby("era")["decision_ts"].min())
    for scheme in (PurgedWalkForward(n_folds=4, embargo="1 session", test_block=0.2), WalkForward(n_folds=4)):
        for fold in scheme.folds(meta):
            assert fold.val_start in era_starts
            val_eras = set(meta["era"].iloc[fold.val_pos])
            train_eras = set(meta["era"].iloc[fold.train_pos])
            assert not val_eras & train_eras  # no session is split between train and validation


def test_rolling_uses_previous_block_only():
    meta = _meta(hold_hours=1)
    expanding = list(WalkForward(n_folds=4, expanding=True).folds(meta))
    rolling = list(WalkForward(n_folds=4, expanding=False).folds(meta))
    assert len(rolling[-1].train_pos) < len(expanding[-1].train_pos)


def test_single_scheme_one_fold_and_test_block():
    meta = _meta(hold_hours=1)
    scheme = SingleScheme(test_block=0.25)
    (fold,) = list(scheme.folds(meta))
    test_pos = set(np.flatnonzero(scheme.test_mask(meta)))
    assert len(test_pos) == pytest.approx(0.25 * len(meta), abs=2 * 4)
    assert not test_pos & set(fold.train_pos) and not test_pos & set(fold.val_pos)
    assert meta["decision_ts"].iloc[fold.train_pos].max() < meta["decision_ts"].iloc[fold.val_pos].min()
    assert (scheme.assign(meta["decision_ts"]) == "test").sum() == len(test_pos)


def test_registered_and_buildable():
    spec = {"type": "purged_walk_forward", "n_folds": 3, "embargo": "1 session", "test_block": "20%"}
    import shpe_research.validation.schemes  # noqa: F401

    assert {"single", "walk_forward", "purged_walk_forward"} <= set(FIT_SCHEME_REGISTRY)
    scheme = build_fit_scheme(spec)
    assert isinstance(scheme, PurgedWalkForward) and scheme.test_block == pytest.approx(0.2)
    assert scheme.embargo == pd.Timedelta(days=1)


def test_parse_embargo_and_manifest():
    assert parse_embargo("2 sessions") == pd.Timedelta(days=2)
    assert parse_embargo("90min") == pd.Timedelta(minutes=90)
    assert parse_embargo(None) == pd.Timedelta(0)
    meta = _meta(hold_hours=1)
    scheme = PurgedWalkForward(n_folds=3, embargo="1 session", test_block=0.2)
    entries = scheme.manifest_folds(meta)
    assert [e["fold"] for e in entries] == [0, 1, 2]
    assert all(e["embargo"] == "1 days 00:00:00" and e["val_start"] and e["train_end"] for e in entries)
    assert scheme.test_scored is False


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        PurgedWalkForward(n_folds=0)
    with pytest.raises(ValueError):
        PurgedWalkForward(test_block=1.2)
    meta = _meta(hold_hours=1)
    meta.loc[3, "label_end_ts"] = pd.NaT
    with pytest.raises(ValueError):
        list(WalkForward(n_folds=3).folds(meta))
