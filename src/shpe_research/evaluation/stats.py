"""Statistics that separate skill from luck (E2).

    block_bootstrap(values, block_length, n, seed)
        Circular moving-block bootstrap CI for the mean of daily values.
    paired_block_bootstrap(a, b, block_length, n, seed)
        Same, on the difference ``a - b`` (model vs. baseline on the same dates).
    deflated_sharpe(sharpe, n_trials, n_obs, skew, kurtosis)
        Bailey & Lopez de Prado (2014): probability the observed Sharpe beats the
        best of ``n_trials`` unskilled strategies.
    probability_of_backtest_overfitting(performance, n_splits)
        Bailey, Borwein, Lopez de Prado & Zhu (2017): combinatorially symmetric
        cross-validation over a configurations x time matrix.

All inputs are realised, after-the-fact series; nothing here looks forward. The
caller is responsible for ``n_trials`` being the full count of configurations
tried, including exit-policy and sizing grids (AGENTS.md rule 3).
"""

from __future__ import annotations

from itertools import combinations
from math import comb, e

import numpy as np
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329


def _clean_1d(values: np.ndarray, name: str = "values") -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite values; drop or fill them explicitly")
    return x


def block_bootstrap(
    values: np.ndarray, block_length: int, n: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile CI for the mean of ``values`` from a circular moving-block bootstrap.

    Inputs: ``values`` is a 1-D array of per-day numbers (returns, IC) in time
    order with no missing values (raises ``ValueError`` otherwise); consecutive
    days are kept together in blocks of ``block_length`` (wrapping around the
    end) because trading days are serially dependent. ``n`` resamples, ``seed``
    fixes the RNG, ``alpha`` is the two-sided level.

    Output: dict with ``mean``, ``low``, ``high``, ``se``, ``block_length``,
    ``n``, ``alpha``. Run it at several ``block_length`` values and report the
    sensitivity. Timing: uses only the realised series passed in.
    """
    x = _clean_1d(values)
    t = x.size
    if not 1 <= block_length <= t:
        raise ValueError(f"block_length must be in [1, {t}], got {block_length}")
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    n_blocks = -(-t // block_length)
    starts = rng.integers(0, t, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % t
    idx = idx.reshape(n, -1)[:, :t]
    means = x[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(x.mean()),
        "low": float(low),
        "high": float(high),
        "se": float(means.std(ddof=1)) if n > 1 else float("nan"),
        "block_length": int(block_length),
        "n": int(n),
        "alpha": float(alpha),
    }


def paired_block_bootstrap(
    a: np.ndarray, b: np.ndarray, block_length: int, n: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Block-bootstrap CI for the mean of ``a - b`` (same dates, same order).

    Resamples the per-day DIFFERENCE, not the two series separately, so the
    pairing is preserved. ``a`` and ``b`` must have equal length. Output as in
    ``block_bootstrap``; ``low > 0`` means ``a`` beats ``b`` at level ``alpha``.
    """
    xa, xb = _clean_1d(a, "a"), _clean_1d(b, "b")
    if xa.shape != xb.shape:
        raise ValueError("a and b must have the same length")
    return block_bootstrap(xa - xb, block_length, n=n, seed=seed, alpha=alpha)


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 252,
    sharpe_variance: float | None = None,
) -> float:
    """Deflated Sharpe ratio: probability in [0, 1] that the true Sharpe is above zero
    after accounting for picking the best of ``n_trials`` configurations.

    Inputs: ``sharpe`` is the ANNUALISED Sharpe of the selected strategy (pass
    ``periods_per_year=1`` if it is already per observation); ``n_obs`` the number
    of return observations behind it; ``skew`` and ``kurtosis`` (raw, normal = 3)
    of those returns; ``n_trials`` every configuration tried. ``sharpe_variance``
    is the variance of the per-period Sharpe across trials; if omitted it is
    ``1 / (n_obs - 1)``, the sampling variance of a Sharpe estimate under no skill.

    The benchmark is ``SR0 = sqrt(V) * ((1-g) z(1-1/N) + g z(1-1/(N e)))`` (g is
    Euler's constant); the result is the probabilistic Sharpe ratio against
    ``SR0``. With ``n_trials=1`` it is the PSR against zero. Values below 0.95
    are not significant. Uses only the realised statistics passed in.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    sr = float(sharpe) / np.sqrt(periods_per_year)
    var = 1.0 / (n_obs - 1) if sharpe_variance is None else float(sharpe_variance)
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(var) * (
            (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + _EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * e))
        )
    denom_sq = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if denom_sq <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom_sq)))


def _sharpe_from_blocks(count: float, total: np.ndarray, total_sq: np.ndarray) -> np.ndarray:
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 1e-24, mean / np.sqrt(var), 0.0)


def probability_of_backtest_overfitting(performance: np.ndarray, n_splits: int = 16) -> float:
    """Probability of backtest overfitting via combinatorially symmetric CV.

    Inputs: ``performance`` is a (configurations x periods) matrix of per-period
    returns, periods in time order, all configurations on the same dates, no
    NaN. The columns are cut into ``n_splits`` (even) contiguous blocks; for each
    of the C(n_splits, n_splits/2) ways to pick half the blocks as in-sample, the
    in-sample Sharpe-best configuration is ranked among all configurations on the
    complementary out-of-sample blocks. PBO is the share of splits where that
    configuration lands at or below the out-of-sample median (logit <= 0).

    Output: float in [0, 1]; about 0.5 means selection is no better than chance,
    near 0 means the in-sample winner tends to stay good. Trailing columns that
    do not fill a block are dropped. Ties in in-sample Sharpe take the first
    configuration. Uses realised data only; it measures selection, not edge.
    """
    m = np.asarray(performance, dtype=float)
    if m.ndim != 2:
        raise ValueError("performance must be 2-D (configurations x periods)")
    n_cfg, n_per = m.shape
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even integer >= 2")
    if n_per < n_splits:
        raise ValueError("need at least n_splits periods")
    if not np.isfinite(m).all():
        raise ValueError("performance contains NaN or infinite values")
    size = n_per // n_splits
    blocks = m[:, : size * n_splits].reshape(n_cfg, n_splits, size)
    total = blocks.sum(axis=2)
    total_sq = (blocks**2).sum(axis=2)

    half = n_splits // 2
    logits = np.empty(comb(n_splits, half))
    all_blocks = set(range(n_splits))
    for k, is_idx in enumerate(combinations(range(n_splits), half)):
        is_list = list(is_idx)
        oos_list = sorted(all_blocks - set(is_idx))
        cnt = float(size * half)
        sr_is = _sharpe_from_blocks(cnt, total[:, is_list].sum(1), total_sq[:, is_list].sum(1))
        sr_oos = _sharpe_from_blocks(cnt, total[:, oos_list].sum(1), total_sq[:, oos_list].sum(1))
        best = int(np.argmax(sr_is))
        # relative OOS rank of the IS winner in (0, 1); average rank for ties
        below = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1)
        omega = (below + 1) / (n_cfg + 1)
        logits[k] = np.log(omega / (1 - omega))
    return float((logits <= 0).mean())
