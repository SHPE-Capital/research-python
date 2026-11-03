"""Samplers on a precomputed column: `threshold` and `cusum` (C1b).

Both read a column already on the bars frame (a feature joined at the bar's `ts_available`, a return), so
the value on a row is what was known when that bar completed. The decision is that bar's `ts_available`.
Rows with a missing value are skipped; only regular-session bars are read, per symbol in time order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from shpe_research.contracts.candidates import CandidateFrame
from shpe_research.events.samplers import Sampler, as_candidates, register_sampler, session_clock


def _column(bars: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name not in bars.columns:
        raise ValueError(f"bars have no column {name!r}; compute it before sampling")
    reg, _ = session_clock(bars)
    return reg, reg[name].astype(float)


@register_sampler
@dataclass(frozen=True)
class Threshold(Sampler):
    """Rows where `column` crosses `upper` (up through it) or `lower` (down through it).

    The decision is the first bar after the cross: the first bar whose value is beyond the level when the
    previous valid bar of the same symbol was not. A missing value resets nothing and never crosses. At least
    one of `upper` and `lower` is required. `trigger_value` holds the column value at the decision.
    """

    sampler_type: ClassVar[str] = "threshold"
    column: str = ""
    upper: float | None = None
    lower: float | None = None
    side: str = "long"

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        if self.upper is None and self.lower is None:
            raise ValueError("threshold needs `upper`, `lower` or both")
        reg, value = _column(bars, self.column)
        valid = reg.assign(_v=value).dropna(subset=["_v"])
        prev = valid.groupby("symbol")["_v"].shift(1)
        cross = pd.Series(False, index=valid.index)
        if self.upper is not None:
            cross |= (valid["_v"] > self.upper) & (prev <= self.upper)
        if self.lower is not None:
            cross |= (valid["_v"] < self.lower) & (prev >= self.lower)
        hit = valid[cross].rename(columns={"ts_available": "decision_ts", "_v": "trigger_value"})
        return as_candidates(hit[["symbol", "decision_ts", "trigger_value"]], self, self.side)


@register_sampler
@dataclass(frozen=True)
class Cusum(Sampler):
    """Symmetric CUSUM filter (López de Prado, AFML ch. 2) over the values of `column`.

    Per symbol, in time order: `s_pos = max(0, s_pos + x)` and `s_neg = min(0, s_neg + x)`. When `s_pos > h` or
    `s_neg < -h` the bar is an event and both sums reset to zero. `column` holds the per-bar increments (for
    example log returns), known at the bar's `ts_available`. `trigger_value` is the signed sum that fired.
    Missing values are skipped.
    """

    sampler_type: ClassVar[str] = "cusum"
    column: str = ""
    h: float = 0.0
    side: str = "long"

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        if self.h <= 0:
            raise ValueError("cusum needs h > 0")
        reg, value = _column(bars, self.column)
        reg = reg.assign(_v=value)
        picked: list[pd.DataFrame] = []
        for _, part in reg.groupby("symbol", sort=False):
            values = part["_v"].to_numpy()
            fired: list[int] = []
            fired_value: list[float] = []
            s_pos = s_neg = 0.0
            for i, x in enumerate(values):
                if np.isnan(x):
                    continue
                s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
                if s_pos > self.h or s_neg < -self.h:
                    fired.append(i)
                    fired_value.append(s_pos if s_pos > self.h else s_neg)
                    s_pos = s_neg = 0.0
            picked.append(part.iloc[fired].assign(trigger_value=fired_value))
        hit = pd.concat(picked) if picked else reg.iloc[0:0].assign(trigger_value=[])
        hit = hit.rename(columns={"ts_available": "decision_ts"})
        return as_candidates(hit[["symbol", "decision_ts", "trigger_value"]], self, self.side)
