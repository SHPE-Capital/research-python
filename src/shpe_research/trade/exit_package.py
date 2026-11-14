"""C2 common trailing exit package for one long position.

`exit_scan` is a pure function over numpy arrays with scalar loops only (Numba-portable). Bars are
labelled by start time in integer minutes (`ts_min`). Order within a bar: gap through the stop at the
open, timed exit at the open (session / max hold), then low crossing the stop. The close of a bar
updates the stop for the next bar only. Prices returned are reference (mid) prices; `run_exit` applies
the adverse side cost.

Ambiguity flag: set when a stop is hit on a bar whose high also reached the trailing trigger
(entry + 1 ATR), i.e. the favorable and unfavorable moves cannot be ordered inside the bar. The less
favorable outcome (the stop fill, stop not yet raised) is taken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shpe_research.trade import costs as _costs
from shpe_research.trade.exits import initial_stop, step_stop
from shpe_research.trade.types import CostModel

STOP, MAX_HOLD, SESSION, DATA_END = 0, 1, 2, 3
REASONS = ("stop", "max_hold", "session_end", "data_end")
MAX_HOLD_MIN = 60
SESSION_BUFFER_MIN = 5


def session_exit_ts(close_ts_min: int, buffer_min: int = SESSION_BUFFER_MIN) -> int:
    """Session-exit time: actual close (early closes included) minus the buffer."""
    return close_ts_min - buffer_min


def exit_scan(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, ts_min: np.ndarray,
    entry_idx: int, entry_fill: float, atr: float, session_exit_min: int,
    max_hold_min: int = MAX_HOLD_MIN, stop_mult: float = 1.5, trail_start_mult: float = 1.0,
) -> tuple[int, float, int, bool]:
    """Return `(exit_idx, exit_ref_price, reason_code, ambiguous)`.

    The position is filled at the open of bar `entry_idx` at `entry_fill`; scanning starts on that bar.
    """
    n = open_.shape[0]
    stop = initial_stop(entry_fill, atr, stop_mult)
    high_close = entry_fill
    trailing = False
    t_hold = ts_min[entry_idx] + max_hold_min
    trigger = entry_fill + trail_start_mult * atr
    for i in range(entry_idx, n):
        if open_[i] <= stop:
            return i, open_[i], STOP, False
        if ts_min[i] >= session_exit_min:
            return i, open_[i], SESSION, False
        if ts_min[i] >= t_hold:
            return i, open_[i], MAX_HOLD, False
        if low[i] <= stop:
            return i, stop, STOP, bool(high[i] >= trigger)
        stop, high_close, trailing = step_stop(
            stop, high_close, trailing, entry_fill, atr, close[i], trail_start_mult, stop_mult
        )
    return n - 1, close[n - 1], DATA_END, False


@dataclass(frozen=True)
class ExitResult:
    exit_idx: int
    exit_price: float  # after adverse cost
    exit_ref_price: float
    reason: str
    ambiguous: bool
    early_close: bool


def run_exit(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, ts_min: np.ndarray,
    entry_idx: int, entry_fill: float, atr: float, close_ts_min: int, early_close: bool = False,
    costs: CostModel | None = None, **kw: float,
) -> ExitResult:
    """`exit_scan` plus adverse sell-side cost (spread and slippage) on the exit fill."""
    if not (atr > 0):
        raise ValueError("atr must be positive")
    if not 0 <= entry_idx < len(open_):
        raise ValueError("entry_idx out of range")
    idx, ref, code, amb = exit_scan(
        open_, high, low, close, ts_min, entry_idx, entry_fill, atr, session_exit_ts(close_ts_min), **kw
    )
    price = _costs.fill_price(costs or CostModel(), "sell", float(ref))
    return ExitResult(int(idx), price, float(ref), REASONS[code], amb, early_close)
