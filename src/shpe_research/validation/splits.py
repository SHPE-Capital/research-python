"""Chronological splits (C0 minimal version; V1 adds purged walk-forward folds).

`split_by_day` assigns whole New York sessions to train, validation and test in
time order. Every symbol shares the same boundaries, and rows are never
shuffled across time. It is leak-free only when no label crosses a session
boundary (true for the C0 forward-return label). For labels that span days,
use V1's purged splitter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date


def split_by_day(
    decision_ts: pd.Series,
    train: float = 0.6,
    validation: float = 0.2,
    test: float = 0.2,
    gap_days: int = 0,
) -> pd.Series:
    """Split name per row; `gap_days` sessions after each boundary become "unassigned"."""
    if not np.isclose(train + validation + test, 1.0):
        raise ValueError("train + validation + test must equal 1")
    day = session_date(decision_ts)
    days = np.sort(day.unique())
    n = len(days)
    cut1 = int(round(n * train))
    cut2 = int(round(n * (train + validation)))
    names = np.empty(n, dtype=object)
    names[:cut1] = "train"
    names[cut1:cut2] = "validation"
    names[cut2:] = "test"
    for cut in (cut1, cut2):
        names[cut : min(cut + gap_days, n)] = "unassigned"
    lookup = pd.Series(names, index=pd.DatetimeIndex(days))
    return pd.Series(lookup.reindex(day.to_numpy()).to_numpy(), index=decision_ts.index).astype(str)
