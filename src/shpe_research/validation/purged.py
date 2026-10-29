"""TODO(V1): purged walk-forward folds with an embargo.  Level: intermediate.

split_by_day (C0) is leak-free only because C0 labels end inside their session.
Labels that run longer (multi-day holds, L2 exits near the close carried
overnight, daily-horizon models) need purging:

    - Walk forward: fold k trains on everything before its test block (expanding
      or rolling window, configurable) and tests on the next block of sessions.
    - Purge: drop training rows whose [decision_ts, label_end_ts] interval
      overlaps the test block.
    - Embargo: also drop training rows that start within `embargo` after the
      test block (for folds where training data follows the test block).

Reference: Lopez de Prado, Advances in Financial Machine Learning, ch. 7.

Done when:
    - tests/todo/test_validation_todo.py::test_purged_walk_forward passes: no
      training row's label interval overlaps its fold's test interval.
    - experiments/runner.py can use it via `split: {method: purged_walk_forward, ...}`.
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
    """(train_positions, test_positions) per fold, using meta.decision_ts and meta.label_end_ts.

    `embargo` defaults to zero.
    """
    raise NotImplementedError("TODO(V1): purged walk-forward splitter")
