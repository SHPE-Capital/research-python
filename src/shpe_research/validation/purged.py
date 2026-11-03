"""Purged walk-forward folds with an embargo (V1).

`split_by_day` is leak-free only because C0 labels end inside their session.
Labels that run longer (multi-day holds, exits carried overnight, daily-horizon
models) need purging. Rows are sorted by `decision_ts` and cut into
`n_folds + 1` contiguous blocks; fold k tests on block k + 1 and trains on the
blocks before it (expanding) or on block k only (rolling). Training rows whose
label interval overlaps the test interval are dropped, and so are rows that
start within `embargo` after the test block.

Reference: Lopez de Prado, Advances in Financial Machine Learning, ch. 7.
The `purged_walk_forward` fit scheme that wraps this is C1f.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def purged_walk_forward(
    meta: pd.DataFrame,
    n_folds: int = 5,
    embargo: pd.Timedelta | None = None,
    expanding: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(train_positions, test_positions)` per fold.

    Inputs: `meta` with UTC `decision_ts` (when the row is decided) and `label_end_ts`
    (when its label is known); `n_folds` walk-forward folds; `embargo` (None means
    zero) after each test block; `expanding` False trains on the single block before
    the test block. Positions index `meta` as passed, not its index labels, and each
    array is ascending.

    Timing: test_start is the earliest `decision_ts` in the test block and test_end its
    latest `label_end_ts`. A training row is dropped if `label_end_ts >= test_start` and
    `decision_ts <= test_end`, or if its `decision_ts` lies in `(test_end, test_end + embargo]`.
    Raises ValueError for missing `decision_ts` or `label_end_ts` (never treated as
    same-bar labels), `n_folds < 1`, or fewer rows than `n_folds + 1`.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    decision = pd.to_datetime(meta["decision_ts"], utc=True)
    label_end = pd.to_datetime(meta["label_end_ts"], utc=True)
    if decision.isna().any() or label_end.isna().any():
        raise ValueError("decision_ts and label_end_ts must not contain missing values")
    if len(meta) < n_folds + 1:
        raise ValueError(f"need at least {n_folds + 1} rows for {n_folds} folds, got {len(meta)}")
    pad = pd.Timedelta(0) if embargo is None else embargo
    dec = decision.to_numpy()
    end = label_end.to_numpy()
    order = np.argsort(dec, kind="stable")
    blocks = np.array_split(order, n_folds + 1)
    for k in range(n_folds):
        test = blocks[k + 1]
        cand = np.concatenate(blocks[: k + 1]) if expanding else blocks[k]
        test_start = dec[test].min()
        test_end = end[test].max()
        overlap = (end[cand] >= test_start) & (dec[cand] <= test_end)
        embargoed = (dec[cand] > test_end) & (dec[cand] <= test_end + pad.to_timedelta64())
        train = cand[~(overlap | embargoed)]
        yield np.sort(train), np.sort(test)
