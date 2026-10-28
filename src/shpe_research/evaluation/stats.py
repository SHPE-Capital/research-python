"""TODO(E2): statistics that separate skill from luck.  Level: advanced.

    block_bootstrap(daily_values, block_length, n=10_000, seed)
        Confidence interval for a mean of daily values (returns, IC) that keeps
        consecutive days together, since trading days are not independent.
        Report sensitivity to block_length. For paired comparisons (model vs.
        baseline on the same dates), resample the DIFFERENCE.

    deflated_sharpe(sharpe, n_trials, n_obs, skew, kurtosis)
        Probability the observed Sharpe ratio beats the best of `n_trials`
        unskilled strategies (Bailey & Lopez de Prado 2014). n_trials comes from
        the experiment registry (M2), which is why every attempt is recorded.

    probability_of_backtest_overfitting(performance_matrix)
        Combinatorially symmetric cross-validation (Bailey et al. 2017) over the
        configurations x time-blocks matrix.

Done when:
    - tests/todo/test_eval_todo.py::test_stats passes, including a synthetic
      check: on pure-noise strategies, deflated_sharpe of the best of 100 is not significant.
"""

from __future__ import annotations

import numpy as np


def block_bootstrap(values: np.ndarray, block_length: int, n: int = 10_000, seed: int = 0) -> dict:
    raise NotImplementedError("TODO(E2): block bootstrap")


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    raise NotImplementedError("TODO(E2): deflated Sharpe ratio")


def probability_of_backtest_overfitting(performance: np.ndarray, n_splits: int = 16) -> float:
    raise NotImplementedError("TODO(E2): probability of backtest overfitting")
