"""Exit rules: stop, target and trailing levels, and the exit_reason vocabulary (T1).

Everything here is pure arithmetic on numbers the simulator hands in. Conventions:

* ``sign`` is +1 for a long and -1 for a short.
* Order exits (price stop, target, trailing stop) are touch-checked inside a bar and
  fill from the trigger price, or from the open when the bar gapped through it.
* Levels that depend on bars (ATR, highest high since entry) use completed bars only:
  a level computed at the end of bar t applies from bar t+1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from shpe_research.trade.types import ExitPolicy

EXIT_REASONS: tuple[str, ...] = ("stop_loss", "take_profit", "exit", "max_hold_exit", "session_end", "unfilled")

# Lower number wins when several signal exits fall due at the same decision time.
SIGNAL_PRIORITY: dict[str, int] = {"stop_loss": 0, "max_hold_exit": 1, "exit": 2, "session_end": 3}


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Average true range over `window` bars, aligned to the bar it ends on.

    Element t uses bars up to and including t, so it is known at bar t's ``ts_available``.
    NaN until `window` bars exist. The first bar's true range is its high minus low.
    """
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.fmax(np.fmax(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))
    return pd.Series(tr).rolling(window, min_periods=window).mean().to_numpy()


def initial_stop(policy: ExitPolicy, sign: int, entry: float, candidate_stop: float, atr_at_entry: float) -> float:
    """Stop level at the entry fill, or NaN when the policy has none.

    `pct` takes `stop_pct` as a fraction of the entry fill price (0.01 = 1%); `atr` uses
    the ATR of the bars completed before the entry bar; `price` uses the candidate's level.
    """
    rule = policy.stop_rule
    if rule == "pct" and policy.stop_pct is not None:
        return entry * (1 - sign * policy.stop_pct)
    if rule == "atr" and policy.stop_atr_mult is not None and math.isfinite(atr_at_entry):
        return entry - sign * policy.stop_atr_mult * atr_at_entry
    if rule == "price":
        return candidate_stop
    return math.nan


def initial_target(policy: ExitPolicy, sign: int, entry: float, stop: float, candidate_target: float) -> float:
    """Target level at the entry fill, or NaN. `r_multiple` needs a stop (risk = |entry - stop|)."""
    rule = policy.target_rule
    if rule == "pct" and policy.target_pct is not None:
        return entry * (1 + sign * policy.target_pct)
    if rule == "price":
        return candidate_target
    if rule == "r_multiple" and policy.target_r is not None and math.isfinite(stop):
        return entry + sign * policy.target_r * abs(entry - stop)
    return math.nan


def stop_reference(sign: int, stop: float, open_: float, high: float, low: float) -> float | None:
    """Price a protective stop executes from inside a bar, or None if the bar never reached it.

    A long's stop is a sell stop (touched when low <= stop, never fills above the stop or the
    open); a short's is a buy stop (high >= stop, never below). Spread and slippage are added later.
    """
    if not math.isfinite(stop):
        return None
    if sign > 0:
        return min(open_, stop) if low <= stop else None
    return max(open_, stop) if high >= stop else None


def target_reference(sign: int, target: float, open_: float, high: float, low: float) -> float | None:
    """Price a target executes from inside a bar, or None: the target, or the open if it gapped past it."""
    if not math.isfinite(target):
        return None
    if sign > 0:
        return max(open_, target) if high >= target else None
    return min(open_, target) if low <= target else None


def entry_stop_reference(sign: int, level: float, open_: float, high: float, low: float) -> float | None:
    """Reference price of a stop *entry* order (buy stop for a long, sell stop for a short)."""
    if not math.isfinite(level):
        return None
    if sign > 0:
        return max(open_, level) if high >= level else None
    return min(open_, level) if low <= level else None


class StopState:
    """The protective stop of one open position as it ratchets with completed bars.

    ``stop`` is the level to test on the *next* bar. Call :meth:`update` once per completed
    bar (the entry bar first). The stop never loosens.
    """

    def __init__(self, policy: ExitPolicy, sign: int, entry: float, stop: float):
        self.policy = policy
        self.sign = sign
        self.entry = entry
        self.stop = stop
        self.risk = abs(entry - stop) if math.isfinite(stop) else math.nan
        self._extreme = -math.inf if sign > 0 else math.inf
        self._breakeven_done = False

    def _tighten(self, level: float) -> None:
        if not math.isfinite(self.stop):
            self.stop = level
        else:
            self.stop = max(self.stop, level) if self.sign > 0 else min(self.stop, level)

    def update(self, high: float, low: float, atr_value: float) -> None:
        """Fold in one completed bar: its high/low and the ATR ending on it."""
        sign = self.sign
        self._extreme = max(self._extreme, high) if sign > 0 else min(self._extreme, low)
        rule = self.policy.trail_rule
        if rule == "chandelier" and math.isfinite(atr_value):
            self._tighten(self._extreme - sign * self.policy.trail_atr_mult * atr_value)
        elif rule == "breakeven_1r" and not self._breakeven_done and math.isfinite(self.risk):
            favourable = (high - self.entry) if sign > 0 else (self.entry - low)
            if favourable >= self.risk:
                self._breakeven_done = True
                self._tighten(self.entry)
