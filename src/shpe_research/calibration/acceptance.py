"""D5 acceptance floors and finalist selection for the weekly calibration.

Stage metrics are computed from a daily return window and its filled entries.
Floors come from ``AcceptanceSection`` (config), never from constants here.
Pure functions: no I/O and no clock.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from shpe_research.calibration.score import score_j
from shpe_research.config.momentum import AcceptanceSection
from shpe_research.strategies.momentum.breakout.grid import Config, config_id

Entry = tuple[int, str, bool]  # (session position in the window, asset_id, unresolved)


@dataclass(frozen=True)
class StageMetrics:
    """Window metrics gated by the acceptance rules.

    ``j`` and ``max_drawdown`` are NaN when ``invalid_accounting`` is True.
    """

    trades: int
    active_sessions: int
    j: float
    max_drawdown: float
    max_single_stock_share: float
    invalid_accounting: bool
    unresolved: int


@dataclass(frozen=True)
class Check:
    """One named acceptance check with its observed value and threshold."""

    name: str
    passed: bool
    value: float | None
    threshold: float | None


@dataclass(frozen=True)
class StageVerdict:
    """Outcome of one stage: pass only if every check passes."""

    passed: bool
    checks: tuple[Check, ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        """Names of failed checks, in check order."""
        return tuple(c.name for c in self.checks if not c.passed)


@dataclass(frozen=True)
class Finalist:
    """A validated candidate; ``report`` (report_bps) is carried but never selects."""

    config: Config
    config_id: str
    seed: int
    search_j: float
    validation: StageMetrics
    report: StageMetrics
    verdict: StageVerdict

    @property
    def n_filters(self) -> int:
        """Number of filter components beyond the breakout trigger."""
        return len(self.config.subset) - 1


@dataclass(frozen=True)
class Selection:
    """Deployment decision: a selected finalist or ``no_deployment`` with reasons."""

    status: Literal["selected", "no_deployment"]
    selected: Finalist | None
    finalists: tuple[Finalist, ...]
    reasons: tuple[str, ...]


def stage_metrics(
    daily_returns: np.ndarray,
    entries: Sequence[Entry],
    weights: np.ndarray | None,
    *,
    downside_coeff: float,
) -> StageMetrics:
    """Compute acceptance metrics for one stage window.

    Args:
        daily_returns: 1-D daily portfolio returns of the window (zero days included).
        entries: One ``(session_position, asset_id, unresolved)`` per filled entry.
        weights: Per-day score weights, or None for uniform.
        downside_coeff: Score downside penalty (``ScoreSection.downside_coeff``).

    Returns:
        StageMetrics. Drawdown is on the compounded curve from 1.0. Invalid
        accounting (non-finite return or return <= -1) sets J and drawdown to
        NaN so every threshold comparison fails. Timing: pass only data known
        at the calibration cut; nothing here reads a clock.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    trades = len(entries)
    counts = Counter(asset for _, asset, _ in entries)
    share = max(counts.values()) / trades if trades else 0.0
    invalid = bool(r.size == 0 or not np.all(np.isfinite(r)) or np.any(r <= -1.0))
    if invalid:
        j = dd = math.nan
    else:
        j = score_j(r, weights, downside_coeff=downside_coeff)
        curve = np.concatenate(([1.0], np.cumprod(1.0 + r)))
        dd = float(np.max(1.0 - curve / np.maximum.accumulate(curve)))
    return StageMetrics(
        trades=trades,
        active_sessions=len({pos for pos, _, _ in entries}),
        j=j,
        max_drawdown=dd,
        max_single_stock_share=share,
        invalid_accounting=invalid,
        unresolved=sum(1 for _, _, u in entries if u),
    )


def _share_check(m: StageMetrics, cfg: AcceptanceSection) -> Check:
    return Check(
        "max_single_stock_share",
        m.max_single_stock_share <= cfg.max_single_stock_share,
        m.max_single_stock_share,
        cfg.max_single_stock_share,
    )


def _tail_checks(m: StageMetrics) -> list[Check]:
    return [
        Check("valid_accounting", not m.invalid_accounting, None, None),
        Check("no_unresolved_data_errors", m.unresolved == 0, m.unresolved, 0),
    ]


def search_eligibility(m: StageMetrics, cfg: AcceptanceSection) -> StageVerdict:
    """Gate a search candidate on its search-window metrics.

    Args:
        m: Search-window StageMetrics (known at the cut).
        cfg: Floors from config.

    Returns:
        StageVerdict with checks in the pinned order; passes only if all pass.
    """
    checks = [
        Check("search_min_trades", m.trades >= cfg.search_min_trades, m.trades, cfg.search_min_trades),
        Check(
            "search_min_sessions",
            m.active_sessions >= cfg.search_min_sessions,
            m.active_sessions,
            cfg.search_min_sessions,
        ),
        _share_check(m, cfg),
        *_tail_checks(m),
    ]
    return StageVerdict(all(c.passed for c in checks), tuple(checks))


def validation_acceptance(m: StageMetrics, cfg: AcceptanceSection) -> StageVerdict:
    """Gate a validation finalist on its validation-window metrics at selection_bps.

    Args:
        m: Validation-window StageMetrics (known at the cut).
        cfg: Floors from config; J must be strictly above ``validation_min_j``.

    Returns:
        StageVerdict with checks in the pinned order; passes only if all pass.
    """
    checks = [
        Check("validation_min_trades", m.trades >= cfg.validation_min_trades, m.trades, cfg.validation_min_trades),
        Check(
            "validation_min_sessions",
            m.active_sessions >= cfg.validation_min_sessions,
            m.active_sessions,
            cfg.validation_min_sessions,
        ),
        Check("validation_min_j", m.j > cfg.validation_min_j, m.j, cfg.validation_min_j),
        Check("max_drawdown", m.max_drawdown <= cfg.max_drawdown, m.max_drawdown, cfg.max_drawdown),
        _share_check(m, cfg),
        *_tail_checks(m),
    ]
    return StageVerdict(all(c.passed for c in checks), tuple(checks))


def dedupe_finalists(
    per_seed: Sequence[Sequence[tuple[Config, int, float]]], max_finalists: int
) -> list[tuple[Config, int, float]]:
    """Merge per-seed finalist lists into unique configs, capped.

    Args:
        per_seed: For each seed in evaluation order, ``(config, seed, search_j)``.
        max_finalists: Cap on the number kept.

    Returns:
        First occurrence of each ``config_id`` (seeds walked in order); if more
        than the cap remain, the top by ``(-search_j, config_id)``, restored to
        evaluation order. Pure; uses only search-window scores.
    """
    seen: set[str] = set()
    kept: list[tuple[Config, int, float]] = []
    for seed_list in per_seed:
        for item in seed_list:
            cid = config_id(item[0])
            if cid not in seen:
                seen.add(cid)
                kept.append(item)
    if len(kept) > max_finalists:
        top = sorted(range(len(kept)), key=lambda i: (-kept[i][2], config_id(kept[i][0])))[:max_finalists]
        kept = [kept[i] for i in sorted(top)]
    return kept


def select_finalist(finalists: Sequence[Finalist], cfg: AcceptanceSection) -> Selection:
    """Pick the deployed finalist or ``no_deployment``.

    Args:
        finalists: Evaluated finalists (verdict already computed on validation
            metrics at selection_bps; report metrics never used here).
        cfg: Acceptance section (kept for signature symmetry with the stages).

    Returns:
        ``selected``: the passing finalist with max ``validation.j``, ties to
        fewer filters, then smaller ``config_id``. Otherwise ``no_deployment``
        with sorted ``"<check> x<count>"`` reasons over all finalists, or
        ``("no_feasible_finalists",)`` when there are none.
    """
    del cfg  # selection rules are fixed; floors were applied when verdicts were built
    fs = tuple(finalists)
    passing = [f for f in fs if f.verdict.passed]
    if passing:
        best = min(passing, key=lambda f: (-f.validation.j, f.n_filters, f.config_id))
        return Selection("selected", best, fs, ())
    if not fs:
        return Selection("no_deployment", None, fs, ("no_feasible_finalists",))
    counts = Counter(name for f in fs for name in f.verdict.reasons)
    return Selection("no_deployment", None, fs, tuple(sorted(f"{n} x{c}" for n, c in counts.items())))
