"""C3 sizing and capacity for the funded account (scalar functions, Numba-portable).

quantity = floor(min of: per-stock % of equity / price, risk % of equity / stop distance, free cash
after reservations / unit cost, room under the total-exposure cap / price, liquidity cap). Zero means
no order and the binding cap is the rejection reason. There is no borrowing.

The liquidity cap is ``liquidity_cap_pct`` of the median share volume in the same minute across the
preceding sessions, with at least ``min_obs`` valid observations, otherwise zero.
"""

from __future__ import annotations

import math

import numpy as np

PER_STOCK, RISK, CASH, EXPOSURE, LIQUIDITY, INVALID = 0, 1, 2, 3, 4, 5
CAP_NAMES = ("per_stock", "risk", "cash", "exposure", "liquidity", "invalid_input")


def liquidity_cap_shares(buf: np.ndarray, n: int, cap_pct: float, min_obs: int) -> float:
    """Liquidity cap in shares from ``n`` valid volumes in ``buf[:n]`` (sorted in place)."""
    if n < min_obs or n <= 0:
        return 0.0
    buf[:n].sort()
    mid = n // 2
    med = buf[mid] if n % 2 else 0.5 * (buf[mid - 1] + buf[mid])
    return cap_pct * float(med)


def _floor(x: float) -> float:
    return math.floor(x + 1e-9)


def size_order(
    equity: float, price: float, stop_distance: float, cash_free: float, unit_cost: float,
    exposure_used: float, liq_shares: float, per_stock_pct: float, risk_pct: float,
    total_exposure_cap: float, size_multiplier: float = 1.0,
) -> tuple[int, int]:
    """Return ``(qty, binding_cap_code)``; ``qty`` is 0 when the binding cap leaves no whole share.

    ``price`` is the planned entry price, ``unit_cost`` the cash reserved per share (limit price plus
    entry cost), ``exposure_used`` the marked value of open positions plus pending notional.
    ``size_multiplier`` scales the two equity-proportional caps only (stream F seam).
    """
    if not (equity > 0 and price > 0 and stop_distance > 0 and unit_cost > 0) or not (
        math.isfinite(equity) and math.isfinite(price) and math.isfinite(stop_distance)
    ):
        return 0, INVALID
    caps = (
        size_multiplier * per_stock_pct * equity / price,
        size_multiplier * risk_pct * equity / stop_distance,
        cash_free / unit_cost,
        (total_exposure_cap * equity - exposure_used) / price,
        liq_shares,
    )
    best, code = caps[0], 0
    for i in range(1, 5):
        if caps[i] < best:
            best, code = caps[i], i
    q = _floor(best)
    return (int(q) if q > 0 else 0), code
