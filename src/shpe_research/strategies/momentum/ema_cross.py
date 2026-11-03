"""EMA crossover baseline (SM2): the simple trend rule every momentum study compares against.

Rule: per symbol, a fast and a slow EMA of the close. A bullish cross (``fast[t] > slow[t]`` and
``fast[t-1] <= slow[t-1]``) opens a long; the opposite cross closes it (``direction="both"`` also opens a short
on the bearish cross, and the bullish cross closes it). Exits are signal exits: ``action = "exit"`` with
``exit_reason = "exit"``, filled by ``trade.simulate_trades`` at the next bar's open like any market order.
No stops, sizing or costs live here; they come from ``trade/``.

Timing: a cross is detected on a completed bar and the candidate's ``decision_ts`` is that bar's
``ts_available``. Only past closes enter any EMA, so truncating the bars never changes an earlier candidate.

Missing bars are skipped, not filled: the EMA steps over the bars that exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from shpe_research.contracts.base import UTC_NS
from shpe_research.contracts.candidates import CandidateFrame, validate_candidates
from shpe_research.events.samplers import session_clock
from shpe_research.strategies.registry import register_strategy

__all__ = ["EmaCross", "ema"]

_DIRECTIONS = ("long", "both")


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA seeded with the SMA of the first `period` values, as the platform's ``computeEMA``.

    Inputs: closes oldest first, integer period >= 1. Output: same length; NaN until `period` values exist,
    then ``out[period-1]`` is their mean and ``out[i] = v[i]*k + out[i-1]*(1-k)`` with ``k = 2/(period+1)``.
    The value at index i uses only ``values[:i+1]``. Fewer than `period` values gives all NaN.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    if period < 1 or len(v) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = v[:period].mean()
    for i in range(period, len(v)):
        out[i] = v[i] * k + out[i - 1] * (1 - k)
    return out


@register_strategy
@dataclass(frozen=True)
class EmaCross:
    """EMA crossover candidates (registered as ``ema_cross``). Defaults: fast 9, slow 21, long only.

    `carry=False` (default) restarts the EMAs and the position state every regular session, so there are no
    candidates in the first `slow` bars of a session; the simulator's session-end flatten closes what is open.
    `carry=True` runs the EMAs across sessions (overnight gaps included) with one position state per symbol.
    """

    strategy_type: ClassVar[str] = "ema_cross"
    fast: int = 9
    slow: int = 21
    direction: str = "long"
    carry: bool = False

    def __post_init__(self) -> None:
        if self.fast < 1 or self.slow <= self.fast:
            raise ValueError(f"need 1 <= fast < slow, got fast={self.fast}, slow={self.slow}")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {self.direction!r}")

    @property
    def strategy_id(self) -> str:
        """``ema_cross:<8-hex hash of the params>``, stable across runs."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return f"{self.strategy_type}:{hashlib.sha1(payload.encode()).hexdigest()[:8]}"

    def generate(self, bars: pd.DataFrame, context: pd.DataFrame | None = None) -> CandidateFrame:
        """Entry and exit rows for the regular-session `bars` (``context`` is unused).

        Known at ``decision_ts`` = ``ts_available`` of the bar whose close completed the cross. Rows carry
        ``trigger_value = fast - slow`` at that bar. A flip in ``both`` mode emits the exit row and the new
        entry row at the same ``decision_ts``. A bar with a NaN close is dropped. Returns an empty,
        valid frame when there is no cross.
        """
        reg, _ = session_clock(bars)
        reg = reg[reg["close"].notna()]
        group_keys = ["symbol"] if self.carry else ["symbol", "_day"]
        rows: list[tuple] = []
        position: dict[str, str | None] = {}
        for _, g in reg.groupby(group_keys, sort=True):
            symbol = g["symbol"].iloc[0]
            if not self.carry or symbol not in position:
                position[symbol] = None
            closes = g["close"].to_numpy()
            diff = ema(closes, self.fast) - ema(closes, self.slow)
            ts = g["ts_available"].to_numpy()
            for i in range(1, len(g)):
                a, b = diff[i - 1], diff[i]
                if np.isnan(a) or np.isnan(b):
                    continue
                held = position[symbol]
                if b > 0 >= a:  # bullish cross
                    if held == "short":
                        rows.append((symbol, ts[i], "short", "exit", "exit", b))
                    if held != "long":
                        rows.append((symbol, ts[i], "long", "enter", None, b))
                        position[symbol] = "long"
                elif b < 0 <= a:  # bearish cross
                    if held == "long":
                        rows.append((symbol, ts[i], "long", "exit", "exit", b))
                        position[symbol] = None
                    if self.direction == "both" and held != "short":
                        rows.append((symbol, ts[i], "short", "enter", None, b))
                        position[symbol] = "short"
        out = pd.DataFrame(
            rows, columns=["symbol", "decision_ts", "side", "action", "exit_reason", "trigger_value"]
        )
        out = out.astype({"symbol": str, "side": str, "action": str, "trigger_value": float, "exit_reason": object})
        out["exit_reason"] = out["exit_reason"].where(out["exit_reason"].notna(), None)  # None, never NaN
        out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True).astype(UTC_NS)
        out["strategy_id"] = self.strategy_id
        out["_exit"] = (out["action"] != "exit").astype(int)  # an exit sorts before a flip entry
        out = out.sort_values(["decision_ts", "symbol", "_exit"], kind="stable").drop(columns="_exit")
        out = out.reset_index(drop=True)
        validate_candidates(out)
        return out
