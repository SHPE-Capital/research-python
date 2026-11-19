"""Historical weighting policies for weekly momentum calibration.

The functions in this module only weight observations that the caller has
already restricted to the history available at the decision cutoff.  They do
not modify returns, fills, or account state.  Invalid or missing inputs raise
instead of being silently dropped, because dropping one side of an aligned
return/regime series can introduce look-ahead or selection bias.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

Policy = Literal["uniform", "recency", "similarity"]


def normalize_weights(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return non-negative finite weights normalized to sum to one."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("weights contain NaN or infinite values")
    if (values < 0).any():
        raise ValueError("weights must be non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return values / total


def effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    """Kish effective sample size of normalized or unnormalized weights."""
    normalized = normalize_weights(weights)
    return float(1.0 / np.square(normalized).sum())


def uniform_weights(n_observations: int) -> np.ndarray:
    """Equal weight for each historical observation."""
    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    return np.full(n_observations, 1.0 / n_observations, dtype=np.float64)


def _utc_index(values: Sequence[object] | pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid timestamps") from exc
    if len(index) == 0 or index.hasnans:
        raise ValueError(f"{name} must be non-empty and contain no missing timestamps")
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    return index


def recency_weights(
    observation_times: Sequence[object] | pd.DatetimeIndex,
    decision_cutoff: object,
    *,
    half_life_days: float = 63.0,
) -> np.ndarray:
    """Exponentially decay history by age at ``decision_cutoff``.

    Every observation must be known by the cutoff.  A value exactly one
    ``half_life_days`` older receives half the unnormalized weight.
    """
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be a positive finite number")
    times = _utc_index(observation_times, "observation_times")
    cutoff = pd.Timestamp(decision_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if (times > cutoff).any():
        raise ValueError("observation_times contain data after decision_cutoff")
    age_days = (cutoff - times).total_seconds().to_numpy(dtype=np.float64) / 86_400.0
    raw = np.exp2(-age_days / half_life_days)
    return normalize_weights(raw)


def similarity_weights(
    historical_regimes: Sequence[Sequence[float]] | np.ndarray,
    current_regime: Sequence[float] | np.ndarray,
    *,
    bandwidth: float | None = None,
    minimum_scale: float = 1e-12,
) -> np.ndarray:
    """Weight historical observations by similarity to the current regime.

    Regime columns are standardized using only ``historical_regimes``.  The
    Gaussian-kernel bandwidth defaults to the median positive historical
    distance, with a deterministic fallback of one for identical histories.
    """
    history = np.asarray(historical_regimes, dtype=np.float64)
    current = np.asarray(current_regime, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] == 0 or history.shape[1] == 0:
        raise ValueError("historical_regimes must be a non-empty 2-D array")
    if current.shape != (history.shape[1],):
        raise ValueError(f"current_regime must have shape ({history.shape[1]},)")
    if not np.isfinite(history).all() or not np.isfinite(current).all():
        raise ValueError("regime inputs contain NaN or infinite values")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive and finite")

    # Robust train-history-only scale. Constant columns carry no distance.
    q75, q25 = np.percentile(history, [75.0, 25.0], axis=0)
    scale = q75 - q25
    active = scale > minimum_scale
    if active.any():
        distances = np.sqrt(np.square((history[:, active] - current[active]) / scale[active]).sum(axis=1))
    else:
        distances = np.zeros(history.shape[0], dtype=np.float64)

    if bandwidth is None:
        positive = distances[distances > minimum_scale]
        width = float(np.median(positive)) if positive.size else 1.0
    else:
        width = float(bandwidth)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("bandwidth must be positive and finite")
    raw = np.exp(-0.5 * np.square(distances / width))
    return normalize_weights(raw)


def policy_weights(
    policy: Policy,
    *,
    n_observations: int,
    observation_times: Sequence[object] | pd.DatetimeIndex | None = None,
    decision_cutoff: object | None = None,
    half_life_days: float = 63.0,
    historical_regimes: Sequence[Sequence[float]] | np.ndarray | None = None,
    current_regime: Sequence[float] | np.ndarray | None = None,
    similarity_bandwidth: float | None = None,
) -> np.ndarray:
    """Dispatch one frozen policy and validate that its inputs align."""
    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    if policy == "uniform":
        return uniform_weights(n_observations)
    if policy == "recency":
        if observation_times is None or decision_cutoff is None:
            raise ValueError("recency needs observation_times and decision_cutoff")
        if len(observation_times) != n_observations:
            raise ValueError("observation_times length does not match n_observations")
        return recency_weights(observation_times, decision_cutoff, half_life_days=half_life_days)
    if policy == "similarity":
        if historical_regimes is None or current_regime is None:
            raise ValueError("similarity needs historical_regimes and current_regime")
        history = np.asarray(historical_regimes)
        if history.ndim < 1 or history.shape[0] != n_observations:
            raise ValueError("historical_regimes length does not match n_observations")
        return similarity_weights(history, current_regime, bandwidth=similarity_bandwidth)
    raise ValueError(f"unknown policy {policy!r}")
