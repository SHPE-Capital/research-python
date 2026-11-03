"""Opening-range breakout event: `range_breakout` (C1b).

The generic event behind SM1 and W6. SM1 adds stops, targets and filters on top; this sampler only says
when the first breakout of a session happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from shpe_research.contracts.candidates import CandidateFrame
from shpe_research.events.samplers import Sampler, as_candidates, register_sampler, session_clock

_CONFIRM = ("close", "touch")
_SIDES = ("long", "short")


@register_sampler
@dataclass(frozen=True)
class RangeBreakout(Sampler):
    """First breakout of the opening range, one row per symbol and session at most.

    The range is every regular bar with `ts_available <= open + range_minutes` (09:30 to 09:30 + range). Its high
    H (low L for `side: short`) is fixed once that last bar completes; the check starts on the next bar. The
    decision is the first later bar whose close (`confirm: close`) or high/low (`confirm: touch`) is beyond H
    (below L), at that bar's `ts_available`. `entry_cutoff` stops the search: an int is minutes after the open, a
    "HH:MM" string a New York wall-clock time; a decision may not be later than it. `trigger_value` is H (or L).
    A session with no range bar yields no row; a missing minute is skipped, never filled.
    """

    sampler_type: ClassVar[str] = "range_breakout"
    range_minutes: int = 30
    entry_cutoff: int | str | None = None
    confirm: str = "close"
    side: str = "long"

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        if self.confirm not in _CONFIRM:
            raise ValueError(f"confirm must be one of {_CONFIRM}, got {self.confirm!r}")
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {self.side!r}")
        reg, daily = session_clock(bars)
        if daily:
            raise ValueError("range_breakout needs intraday bars")
        keys = ["symbol", "_day"]
        in_range = reg["_minute"] <= self.range_minutes
        long = self.side == "long"
        extreme = reg["high"] if long else reg["low"]
        level = extreme.where(in_range).groupby([reg[k] for k in keys]).transform("max" if long else "min")
        price = reg["close"] if self.confirm == "close" else extreme
        beyond = price > level if long else price < level
        cond = beyond & ~in_range & level.notna() & self._before_cutoff(reg)
        first = cond & (cond.groupby([reg[k] for k in keys]).cumsum() == 1)
        hit = reg[first].assign(trigger_value=level[first]).rename(columns={"ts_available": "decision_ts"})
        return as_candidates(hit[["symbol", "decision_ts", "trigger_value"]], self, self.side)

    def _before_cutoff(self, reg: pd.DataFrame) -> pd.Series:
        cutoff = self.entry_cutoff
        if cutoff is None:
            return pd.Series(True, index=reg.index)
        if isinstance(cutoff, str):
            hours, minutes = (int(part) for part in cutoff.split(":"))
            local = reg["ts_available"].dt.tz_convert("America/New_York")
            return (local.dt.hour * 60 + local.dt.minute) <= hours * 60 + minutes
        return reg["_minute"] <= int(cutoff)
