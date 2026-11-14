"""C1 screening fill proxy and cost scenarios (scalar functions, Numba-portable).

The entry limit is the signal-bar close times ``1 + limit_markup`` (default 0.20%). The order fills at
the **next bar's open** only if that open is at or below the limit; otherwise it is canceled. A missing
next bar (NaN open, or none in the session) means no fill. The proxy is all-or-nothing: partial fills
are not modeled at this level (the quote replay, C6, keeps partials).

Costs are a per-side basis-point charge on traded notional, applied by the account as a cash
deduction on both entry and exit. 10 bps is the selection scenario; 5 and 20 are reported alongside.
This sits next to the platform ``FillModel`` (``trade/types.py``) and does not change it.
"""

from __future__ import annotations

FILLED, CANCEL_ABOVE_LIMIT, NO_NEXT_BAR = 0, 1, 2
OUTCOMES = ("filled", "entry_cancelled_above_limit", "no_next_bar")

COST_SCENARIOS_BPS: tuple[float, ...] = (5.0, 10.0, 20.0)
SELECTION_COST_BPS = 10.0


def entry_limit(signal_close: float, markup: float) -> float:
    """Buy limit for a signal on a bar closing at ``signal_close``."""
    return signal_close * (1.0 + markup)


def entry_fill(limit: float, next_open: float) -> tuple[int, float]:
    """Return ``(outcome_code, fill_price)``; the price is NaN unless filled."""
    if next_open != next_open:  # NaN: no next bar
        return NO_NEXT_BAR, float("nan")
    if next_open <= limit:
        return FILLED, next_open
    return CANCEL_ABOVE_LIMIT, float("nan")


def side_cost_rate(bps: float) -> float:
    """Per-side cost as a fraction of traded notional."""
    return bps / 1e4


def validate_cost_bps(bps: float, allowed: tuple[float, ...] = COST_SCENARIOS_BPS) -> float:
    """Raise unless ``bps`` is one of the named cost scenarios."""
    if bps not in allowed:
        raise ValueError(f"cost_bps {bps} is not a scenario in {allowed}")
    return bps
