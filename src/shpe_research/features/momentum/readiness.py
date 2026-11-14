# ruff: noqa: E741
"""B4: warm-up and readiness rules for the momentum components.

Two layers:

* Structural validity: first bar at which an indicator's inputs exist (ATR14 at
  t=14, ADX(a) at t=2a-1, breakout(n) at max(n, 14), EMA trend at
  max(slow-1+5, 14), VWAP band at 14 (ATR), RVOL once 15 prior sessions exist
  and the opening window is complete).
* Convergence warm-up (App. A): recursive indicators additionally need at
  least ``WARMUP_MULT`` (5) times their longest recursive length of history, so
  the seed's influence has decayed. ``convergence_report`` checks this
  empirically (short warm-up start vs a long reference run) and
  ``converged_warmup`` extends the multiple until it passes.

Readiness masks are indexed by absolute bar position t (0 = start of the
supplied history); ``first_valid_bars`` gives, per setting and symbol, the first
ready bar (-1 if never).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from shpe_research.features.momentum.breakout import BREAKOUT_N, preceding_high
from shpe_research.features.momentum.clock_rvol import RVOL_W, rvol_ratio
from shpe_research.features.momentum.ema_trend import EMA_FAST, EMA_SLOW, ema
from shpe_research.features.momentum.vwap_band import session_vwap
from shpe_research.features.momentum.wilder import (
    ADX_PERIODS,
    ATR_LENGTH,
    adx_di,
    atr,
)

WARMUP_MULT = 5


def first_valid_index(mask: np.ndarray) -> np.ndarray:
    """First True index per symbol, -1 if none."""
    m = np.asarray(mask, dtype=bool)
    first = np.argmax(m, axis=1)
    return np.where(m.any(axis=1), first, -1)


def readiness_masks(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_idx: np.ndarray,
    atr14: np.ndarray | None = None,
    warmup_mult: int = WARMUP_MULT,
) -> dict[str, np.ndarray]:
    """Ready mask (symbols, minutes) per component setting.

    Keys: ``atr14``, ``adx_a{a}``, ``breakout_n{n}``, ``ema_f{f}_s{s}``,
    ``vwap``, ``rvol_w{w}``. A bar is ready when the indicator values are finite
    and t >= warmup_mult * longest recursive length involved (ATR14 included
    for anything normalised by ATR).
    """
    n_t = np.shape(close)[1]
    t_idx = np.arange(n_t)[None, :]
    a14 = atr(high, low, close, ATR_LENGTH) if atr14 is None else atr14
    atr_ok = np.isfinite(a14) & (t_idx >= warmup_mult * ATR_LENGTH)
    out: dict[str, np.ndarray] = {"atr14": atr_ok}
    for a in ADX_PERIODS:
        adx, p, m = adx_di(high, low, close, a)
        out[f"adx_a{a}"] = np.isfinite(adx) & np.isfinite(p) & np.isfinite(m) & (t_idx >= warmup_mult * a)
    for n in BREAKOUT_N:
        out[f"breakout_n{n}"] = np.isfinite(preceding_high(high, n)) & atr_ok
    slow_ema = {s: ema(close, s) for s in EMA_SLOW}
    for s in EMA_SLOW:
        prev = np.full(slow_ema[s].shape, np.nan, dtype=slow_ema[s].dtype)
        prev[:, 5:] = slow_ema[s][:, :-5]
        for f in EMA_FAST:
            out[f"ema_f{f}_s{s}"] = (
                np.isfinite(ema(close, f)) & np.isfinite(prev) & atr_ok & (t_idx >= warmup_mult * s)
            )
    vw = session_vwap(high, low, close, volume, session_idx)
    out["vwap"] = np.isfinite(vw) & atr_ok
    for w in RVOL_W:
        out[f"rvol_w{w}"] = np.isfinite(rvol_ratio(volume, session_idx, w))
    return out


def first_valid_bars(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: first_valid_index(v) for k, v in masks.items()}


@dataclass(frozen=True)
class ConvergenceReport:
    name: str
    length: int
    warmup_bars: int
    max_abs_diff: float
    tolerance: float
    passed: bool


def convergence_report(
    name: str,
    compute: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int,
    warmup_bars: int,
    tolerance: float,
    relative_to: np.ndarray | None = None,
) -> ConvergenceReport:
    """Compare ``compute`` started at the last ``warmup_bars``-offset vs full history.

    The reference run uses the whole history (long warm-up). The short run
    starts at ``start = n_t - available`` chosen so that the short run has
    exactly ``warmup_bars`` bars of warm-up before the comparison region (the
    last half of the history). Differences are over bars after the short
    warm-up; if ``relative_to`` is given the difference is divided by it.
    """
    n_t = np.shape(close)[1]
    start = n_t // 2 - warmup_bars
    if start < 0:
        raise ValueError("history too short for requested warm-up")
    ref = compute(high, low, close)
    short = compute(high[:, start:], low[:, start:], close[:, start:])
    a, b = ref[:, n_t // 2 :], short[:, warmup_bars:]
    d = np.abs(a - b)
    if relative_to is not None:
        d = d / np.abs(relative_to[:, n_t // 2 :])
    worst = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
    return ConvergenceReport(name, length, warmup_bars, worst, tolerance, bool(worst <= tolerance))


def converged_warmup(
    name: str,
    compute: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int,
    tolerance: float,
    relative: bool = False,
    max_mult: int = 20,
) -> ConvergenceReport:
    """Smallest multiple (>= WARMUP_MULT) of ``length`` that converges."""
    ref = compute(high, low, close) if relative else None
    last = None
    for mult in range(WARMUP_MULT, max_mult + 1):
        try:
            last = convergence_report(
                name, compute, high, low, close, length, mult * length, tolerance, ref
            )
        except ValueError:
            break
        if last.passed:
            return last
    if last is None:
        raise ValueError("history too short for convergence check")
    return last


def convergence_reports(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, adx_tol: float = 0.5, ema_rel_tol: float = 1e-3
) -> list[ConvergenceReport]:
    """Convergence report for the longest settings: EMA 60 and ADX 21.

    ADX tolerance is in ADX points (0-100); EMA tolerance is relative.
    """
    return [
        converged_warmup("ema60", lambda h, l, c: ema(c, 60), high, low, close, 60, ema_rel_tol, relative=True),
        converged_warmup("adx21", lambda h, l, c: adx_di(h, l, c, 21)[0], high, low, close, 21, adx_tol),
    ]
