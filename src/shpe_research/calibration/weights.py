"""Stage weighting policies (uniform, recency, similarity) with an ESS-targeted solver.

No function here accepts returns. Weights are computed from session counts or
descriptors known at the cutoff, then applied to the stage's daily returns.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from shpe_research.calibration.descriptor_scaling import RobustScaler, transform
from shpe_research.calibration.score import normalize_weights
from shpe_research.config.momentum import Policy, WeightsSection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightResult:
    """Weights for one stage plus how they were obtained.

    Attributes:
        weights: (n,) weights summing to 1.
        policy_requested: Policy the caller asked for.
        policy_used: Policy actually applied ("uniform" after a fallback).
        parameter: Half-life (sessions) or bandwidth; None for uniform.
        ess: Effective sample size of ``weights``.
        ess_target: ``ess_target_fraction * n``.
        target_reached: True if ESS is within tolerance of the target.
        fallback_reason: Why a fallback happened, else None.
    """

    weights: np.ndarray
    policy_requested: Policy
    policy_used: Policy
    parameter: float | None
    ess: float
    ess_target: float
    target_reached: bool
    fallback_reason: str | None


def effective_sample_size(w: np.ndarray) -> float:
    """Return ``1 / sum((w / sum(w))**2)``.

    Args:
        w: Non-negative weights with positive sum.

    Returns:
        The effective sample size. Pure computation, no timing dependence.
    """
    p = np.asarray(w, dtype=np.float64)
    p = p / p.sum()
    return float(1.0 / np.sum(p * p))


def mix_uniform(raw: np.ndarray, uniform_mix: float) -> np.ndarray:
    """Blend normalized raw weights with uniform weights.

    Args:
        raw: Non-negative raw weights (any positive scale).
        uniform_mix: Share of uniform weight in [0, 1].

    Returns:
        ``uniform_mix / n + (1 - uniform_mix) * raw / sum(raw)``, summing to 1.
        Pure computation; no timing dependence.
    """
    n = len(raw)
    return uniform_mix / n + (1.0 - uniform_mix) * normalize_weights(raw, n)


def _uniform_result(
    n: int, cfg: WeightsSection, requested: Policy, *, reached: bool, reason: str | None
) -> WeightResult:
    w = np.full(n, 1.0 / n, dtype=np.float64)
    return WeightResult(w, requested, "uniform", None, float(n), cfg.ess_target_fraction * n, reached, reason)


def uniform_weights(n: int, cfg: WeightsSection) -> WeightResult:
    """Uniform weights 1/n.

    Args:
        n: Number of sessions in the stage.
        cfg: Weights config (only the ESS target fraction is used for reporting).

    Returns:
        A WeightResult with ``parameter=None`` and ESS ``n``. No timing dependence.
    """
    return _uniform_result(n, cfg, "uniform", reached=True, reason=None)


def _recency_raw(n: int, half_life: float) -> np.ndarray:
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    return np.exp(-np.log(2.0) * ages / half_life)


def _similarity_raw(d2: np.ndarray, bandwidth: float) -> np.ndarray:
    # Shifting by the minimum is a constant factor, removed by normalization;
    # it only prevents underflow at small bandwidths.
    return np.exp(-(d2 - d2.min()) / (2.0 * bandwidth * bandwidth))


def _solve(
    raw_fn: Callable[[float], np.ndarray], n: int, scale: float, cfg: WeightsSection
) -> tuple[np.ndarray, float, float, bool]:
    """Bisect on log(parameter) so the mixed weights' ESS hits the target.

    Returns (weights, parameter, ess, target_reached).
    """
    target = cfg.ess_target_fraction * n
    tol = cfg.ess_tolerance * n
    lo, hi = cfg.solver_bracket[0] * scale, cfg.solver_bracket[1] * scale

    def mixed(p: float) -> np.ndarray:
        return mix_uniform(raw_fn(p), cfg.uniform_mix)

    w_lo, w_hi = mixed(lo), mixed(hi)
    ess_lo, ess_hi = effective_sample_size(w_lo), effective_sample_size(w_hi)
    bound: tuple[np.ndarray, float, float] | None = None
    if target > ess_hi:
        bound = (w_hi, hi, ess_hi)
    elif target < ess_lo:
        bound = (w_lo, lo, ess_lo)
    if bound is not None:
        w, p, e = bound
        logger.warning("ESS target %.4f is outside the solver bracket; using bound with ESS %.4f", target, e)
        return w, p, e, abs(e - target) <= tol

    a, b = np.log(lo), np.log(hi)
    w, p, e = w_lo, lo, ess_lo
    for _ in range(cfg.solver_max_iter):
        p = float(np.exp(0.5 * (a + b)))
        w = mixed(p)
        e = effective_sample_size(w)
        if abs(e - target) <= tol:
            break
        if e < target:
            a = np.log(p)
        else:
            b = np.log(p)
    return w, p, e, abs(e - target) <= tol


def recency_weights(n: int, cfg: WeightsSection) -> WeightResult:
    """Exponential recency weights solved to the ESS target.

    Args:
        n: Number of sessions in the stage. Ages run n-1 ... 0, so the stage's
            last session has age 0 (equivalent to measuring from the cutoff,
            since normalized exponential weights ignore a constant age offset).
        cfg: Weights config.

    Returns:
        A WeightResult whose parameter is the half-life in sessions. Depends only on
        session counts, not on any data after the cutoff.
    """
    w, p, e, ok = _solve(lambda hl: _recency_raw(n, hl), n, float(n), cfg)
    return WeightResult(w, "recency", "recency", p, e, cfg.ess_target_fraction * n, ok, None)


def similarity_weights(
    stage_rows: np.ndarray, current: np.ndarray, scaler: RobustScaler, cfg: WeightsSection
) -> WeightResult:
    """Gaussian-kernel similarity of each stage session's descriptors to the current ones.

    Args:
        stage_rows: (n, k) descriptor rows of the stage's sessions (known at each session).
        current: (k,) descriptor row at the cutoff.
        scaler: Robust scaler fit on search rows only.
        cfg: Weights config.

    Returns:
        A WeightResult whose parameter is the kernel bandwidth. Falls back to uniform
        (with ``fallback_reason``) when all descriptors are dropped, any value is
        non-finite, or no stage row differs from the current row.
    """
    rows = np.asarray(stage_rows, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    n = rows.shape[0]
    if not (np.all(np.isfinite(rows)) and np.all(np.isfinite(cur))):
        logger.warning("non-finite descriptors; similarity weights fall back to uniform")
        return _uniform_result(n, cfg, "similarity", reached=False, reason="non_finite_descriptors")
    if not scaler.kept.any():
        logger.warning("all descriptors have zero scale; similarity weights fall back to uniform")
        return _uniform_result(n, cfg, "similarity", reached=False, reason="all_descriptors_zero_scale")
    d2 = np.sum((transform(scaler, rows) - transform(scaler, cur[None, :])) ** 2, axis=1)
    positive = d2[d2 > 0]
    if positive.size == 0:
        logger.warning("all stage descriptors equal the current row; similarity weights fall back to uniform")
        return _uniform_result(n, cfg, "similarity", reached=False, reason="no_positive_distances")
    scale = float(np.median(np.sqrt(positive)))
    w, p, e, ok = _solve(lambda bw: _similarity_raw(d2, bw), n, scale, cfg)
    return WeightResult(w, "similarity", "similarity", p, e, cfg.ess_target_fraction * n, ok, None)


def stage_weights(
    policy: Policy,
    n: int,
    cfg: WeightsSection,
    *,
    stage_rows: np.ndarray | None = None,
    current: np.ndarray | None = None,
    scaler: RobustScaler | None = None,
) -> WeightResult:
    """Dispatch to the weighting policy for one stage.

    Args:
        policy: "uniform", "recency" or "similarity".
        n: Number of sessions in the stage.
        cfg: Weights config.
        stage_rows: (n, k) descriptors; required for "similarity".
        current: (k,) cutoff descriptors; required for "similarity".
        scaler: Search-fit scaler; required for "similarity".

    Returns:
        The WeightResult. Uses only cutoff-time information.

    Raises:
        ValueError: If similarity inputs are missing or the policy is unknown.
    """
    if policy == "uniform":
        return uniform_weights(n, cfg)
    if policy == "recency":
        return recency_weights(n, cfg)
    if policy == "similarity":
        if stage_rows is None or current is None or scaler is None:
            raise ValueError("similarity policy requires stage_rows, current and scaler")
        return similarity_weights(stage_rows, current, scaler, cfg)
    raise ValueError(f"unknown policy {policy!r}")
