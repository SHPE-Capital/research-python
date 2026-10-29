"""Acceptance tests for the validation tasks. Remove a test's skip marker in the PR that implements it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns


@pytest.mark.skip(reason="TODO(V1)")
def test_purged_walk_forward():
    from shpe_research.validation.purged import purged_walk_forward

    rng = np.random.default_rng(0)
    decision = to_utc_ns(pd.Series(pd.date_range("2026-01-02T15:00Z", periods=500, freq="6h")))
    meta = pd.DataFrame(
        {"decision_ts": decision, "label_end_ts": decision + pd.to_timedelta(rng.integers(1, 72, 500), unit="h")}
    )
    folds = list(purged_walk_forward(meta, n_folds=4, embargo=pd.Timedelta(hours=12)))
    assert len(folds) == 4
    for train, test in folds:
        assert len(train) and len(test)
        test_start, test_end = meta["decision_ts"].iloc[test].min(), meta["label_end_ts"].iloc[test].max()
        tr = meta.iloc[train]
        overlaps = (tr["label_end_ts"] >= test_start) & (tr["decision_ts"] <= test_end)
        assert not overlaps.any()
