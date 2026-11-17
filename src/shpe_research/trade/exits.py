"""Stop mechanics for one long position (T1 rebuild, task 0-E).

`StopState` holds the initial/trailing stop of a trade; `step_stop` is the same arithmetic as a plain
scalar function (no Python objects) so it can be ported to Numba unchanged.

Timing: a stop level computed from bar `i`'s close applies from bar `i + 1` onward. ATR is the
ATR14 value at entry and never changes.
"""

from __future__ import annotations

from dataclasses import dataclass


def initial_stop(entry: float, atr: float, stop_mult: float = 1.5) -> float:
    """Initial stop: `stop_mult` ATR below the filled entry."""
    return entry - stop_mult * atr


def step_stop(
    stop: float, high_close: float, trailing: bool, entry: float, atr: float, close: float,
    trail_start_mult: float = 1.0, stop_mult: float = 1.5,
) -> tuple[float, float, bool]:
    """Update `(stop, highest_close, trailing)` with a completed bar close; the result applies next bar.

    Trailing starts once a completed close >= entry + trail_start_mult ATR; the stop is then
    highest completed close since entry - stop_mult ATR and never moves down.
    """
    if close > high_close:
        high_close = close
    if not trailing and close >= entry + trail_start_mult * atr:
        trailing = True
    if trailing:
        cand = high_close - stop_mult * atr
        if cand > stop:
            stop = cand
    return stop, high_close, trailing


@dataclass
class StopState:
    """Mutable stop state of one long position."""

    entry: float
    atr: float
    stop: float
    high_close: float
    trailing: bool = False
    trail_start_mult: float = 1.0
    stop_mult: float = 1.5

    @classmethod
    def from_entry(
        cls, entry: float, atr: float, stop_mult: float = 1.5, trail_start_mult: float = 1.0
    ) -> StopState:
        if not (atr > 0) or not (entry > 0):
            raise ValueError("entry and atr must be positive")
        return cls(entry, atr, initial_stop(entry, atr, stop_mult), entry, False, trail_start_mult, stop_mult)

    def on_close(self, close: float) -> float:
        """Feed a completed close; returns the stop that applies from the next bar."""
        self.stop, self.high_close, self.trailing = step_stop(
            self.stop, self.high_close, self.trailing, self.entry, self.atr, close,
            self.trail_start_mult, self.stop_mult,
        )
        return self.stop
