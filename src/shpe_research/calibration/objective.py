"""Deterministic historical objective and acceptance diagnostics.

The account simulator owns fills and produces the raw daily return path.  This
module scores that unchanged path with weights computed from information known
at the decision cutoff.  Zero-return days remain in the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shpe_research.calibration.policies import effective_sample_size, normalize_weights


@dataclass(frozen=True)
class ObjectiveResult:
    """Weighted search statistics for one realized historical return path."""

    score: float
    weighted_mean: float
    weighted_volatility: float
    annualized_sharpe: float
    compounded_return: float
    max_drawdown: float
    active_fraction: float
    observations: int
    ess: float


@dataclass(frozen=True)
class AcceptanceRules:
    """Frozen minimum evidence and risk limits used after candidate search."""

    min_observations: int = 40
    min_ess: float = 20.0
    min_active_fraction: float = 0.02
    min_compounded_return: float = 0.0
    max_drawdown: float = 0.20


@dataclass(frozen=True)
class AcceptanceCheck:
    """One named acceptance result, suitable for artifact serialization."""

    name: str
    passed: bool
    value: float
    threshold: float


def _validated_returns(daily_returns: np.ndarray) -> np.ndarray:
    values = np.asarray(daily_returns, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("daily_returns must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("daily_returns contain NaN or infinite values")
    if (values <= -1).any():
        raise ValueError("daily_returns cannot be <= -1")
    return values


def score_returns(
    daily_returns: np.ndarray,
    weights: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> ObjectiveResult:
    """Score actual daily returns using a weighted annualized Sharpe ratio.

    Weights alter only the historical mean and volatility used for ranking.
    Compounded return, drawdown, and activity describe the original
    chronological path and are therefore deliberately unweighted.
    """
    values = _validated_returns(daily_returns)
    normalized = normalize_weights(weights)
    if normalized.shape != values.shape:
        raise ValueError("weights and daily_returns must have the same length")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite")

    mean = float(np.dot(normalized, values))
    variance = float(np.dot(normalized, np.square(values - mean)))
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = float(mean / max(volatility, 1e-12) * np.sqrt(periods_per_year))

    curve = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(curve)
    max_drawdown = float(np.max(1.0 - curve / peaks))
    compounded = float(curve[-1] - 1.0)
    active = float(np.count_nonzero(values) / values.size)
    return ObjectiveResult(
        score=sharpe,
        weighted_mean=mean,
        weighted_volatility=volatility,
        annualized_sharpe=sharpe,
        compounded_return=compounded,
        max_drawdown=max_drawdown,
        active_fraction=active,
        observations=int(values.size),
        ess=effective_sample_size(normalized),
    )


def acceptance_checks(
    result: ObjectiveResult, rules: AcceptanceRules | None = None
) -> tuple[AcceptanceCheck, ...]:
    """Evaluate frozen acceptance rules without changing candidate ranking."""
    rules = rules or AcceptanceRules()
    if rules.min_observations < 1:
        raise ValueError("min_observations must be positive")
    if rules.min_ess <= 0:
        raise ValueError("min_ess must be positive")
    if not 0 <= rules.min_active_fraction <= 1:
        raise ValueError("min_active_fraction must be in [0, 1]")
    if not 0 <= rules.max_drawdown <= 1:
        raise ValueError("max_drawdown must be in [0, 1]")
    checks = (
        AcceptanceCheck("observations", result.observations >= rules.min_observations,
                        float(result.observations), float(rules.min_observations)),
        AcceptanceCheck("effective_sample_size", result.ess >= rules.min_ess, result.ess, rules.min_ess),
        AcceptanceCheck("active_fraction", result.active_fraction >= rules.min_active_fraction,
                        result.active_fraction, rules.min_active_fraction),
        AcceptanceCheck("compounded_return", result.compounded_return >= rules.min_compounded_return,
                        result.compounded_return, rules.min_compounded_return),
        AcceptanceCheck("max_drawdown", result.max_drawdown <= rules.max_drawdown,
                        result.max_drawdown, rules.max_drawdown),
    )
    return checks


def accepted(result: ObjectiveResult, rules: AcceptanceRules | None = None) -> bool:
    """True only when every frozen acceptance check passes."""
    return all(check.passed for check in acceptance_checks(result, rules))
