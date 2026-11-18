"""Common score J for the momentum weekly calibration.

``J = mean(r) - downside_coeff * downside(r)`` over a window of daily returns,
where ``downside`` is the root of the (weighted) mean squared negative return.
All arithmetic is float64. Pure functions: no I/O and no clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ScoreTerms:
    """Components of the score for one return window.

    Attributes:
        mean: Weighted mean return, sum(w * r).
        downside: sqrt(sum(w * min(r, 0)**2)).
        j: ``mean - downside_coeff * downside``.
    """

    mean: float
    downside: float
    j: float


def normalize_weights(weights: ArrayLike | None, n: int) -> np.ndarray:
    """Return float64 weights of length ``n`` that sum to one.

    Args:
        weights: Non-negative finite weights, or None for uniform 1/n. Any
            positive scale is accepted and divided by its sum.
        n: Number of return observations the weights must match.

    Returns:
        A 1-D float64 array summing to 1. Pure computation; no timing
        dependence (the caller aligns weights to the return window).

    Raises:
        ValueError: If the length is not ``n``, any weight is non-finite or
            negative, or the sum is not positive.
    """
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"weights must be 1-D of length {n}, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def score_terms(returns: ArrayLike, weights: ArrayLike | None, *, downside_coeff: float) -> ScoreTerms:
    """Compute the mean, downside and J terms for a window of returns.

    Args:
        returns: 1-D, non-empty, finite daily returns for the scoring window
            (zero days count as observations). Timing is the caller's concern:
            pass only returns known at the calibration cut.
        weights: Per-observation weights (normalized internally) or None for
            uniform.
        downside_coeff: Penalty on the downside term, from
            ``ScoreSection.downside_coeff``. Keyword-only, no default.

    Returns:
        ScoreTerms with ``j = mean - downside_coeff * downside``.

    Raises:
        ValueError: If returns are not 1-D, are empty or non-finite, or the
            weights are invalid (see ``normalize_weights``).
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("returns must be a non-empty 1-D array")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    w = normalize_weights(weights, r.size)
    mean = float(np.sum(w * r))
    downside = float(np.sqrt(np.sum(w * np.minimum(r, 0.0) ** 2)))
    return ScoreTerms(mean=mean, downside=downside, j=mean - downside_coeff * downside)


def score_j(returns: ArrayLike, weights: ArrayLike | None, *, downside_coeff: float) -> float:
    """Return the scalar score J for a window of returns.

    Args:
        returns: 1-D, non-empty, finite daily returns (known at the cut).
        weights: Per-observation weights or None for uniform.
        downside_coeff: Downside penalty, keyword-only, from config.

    Returns:
        ``mean - downside_coeff * downside`` as a float.

    Raises:
        ValueError: On invalid returns or weights (see ``score_terms``).
    """
    return score_terms(returns, weights, downside_coeff=downside_coeff).j
