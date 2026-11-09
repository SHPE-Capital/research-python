"""OU helpers for the stat-arb family (SS3).

The fit itself lives in ``features/cross_asset/pair_math.py``; this module only
re-exports it and adds the half-life screen used by pair selection (SS1).
"""

from __future__ import annotations

import math

import pandas as pd

from shpe_research.features.cross_asset.pair_math import OUFit, fit_ou, spread_zscore

__all__ = ["OUFit", "fit_ou", "passes_half_life_screen", "spread_zscore"]


def passes_half_life_screen(
    spread: pd.Series, max_half_life_bars: float, min_half_life_bars: float = 0.0
) -> bool:
    """Return True if the spread's OU half-life is in ``[min, max]`` bars.

    Input: a spread known up to the screening time (use a trailing window).
    A non-reverting or degenerate fit (infinite or NaN half-life) fails.
    """
    hl = fit_ou(spread).half_life_bars
    return math.isfinite(hl) and min_half_life_bars <= hl <= max_half_life_bars
