"""Exercises for the leakage lab. Fill in each function; delete the ``raise``.

The starter raises ``NotImplementedError`` so ``notebook.py`` runs top to bottom with the exercises pending.
The filled-in version lives on the ``workshop-solutions`` branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_bar_lookahead(bars: pd.DataFrame) -> pd.Series:
    """Core 1. A one-bar look-ahead feature: the NEXT bar's close over this bar's close, minus 1.

    ``bars`` is one symbol's bars sorted by ts. Return one value per row, aligned to ``bars.index``.
    Question: how much does validation AUC rise, and does ``check_feature`` catch it?
    """
    # EXERCISE
    raise NotImplementedError


def day_zscore_expanding(bars: pd.DataFrame) -> pd.Series:
    """Core 2. The fixed ``DayZScore``: z-score of the close within its session using only closes up to now.

    Use an expanding mean and standard deviation that restart each New York session day. Rows with fewer than
    two closes (or zero spread) are NaN. ``bars`` is one symbol's bars sorted by ts; return a Series on ``bars.index``.
    """
    # EXERCISE
    raise NotImplementedError


def selection_gain(table: pd.DataFrame, n_configs: int, n_draws: int = 300, seed: int = 0) -> float:
    """Core 3. The snooping gain: how much better the test-best configuration looks than a typical one.

    ``table`` has one row per configuration with columns ``val_auc`` and ``test_auc`` measured on the same history.
    Draw ``n_configs`` rows at random (without replacement), pick the row with the highest ``test_auc``, and record
    its ``test_auc`` minus the mean ``test_auc`` of the whole table. Return the average over ``n_draws`` draws
    (use ``np.random.default_rng(seed)``). Question: how does it grow with ``n_configs``?
    """
    # EXERCISE
    raise NotImplementedError


__all__ = ["day_zscore_expanding", "one_bar_lookahead", "selection_gain"]
_ = np  # numpy is there for the solutions
