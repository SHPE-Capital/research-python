"""Forward-return label (C0 minimal version; L1 generalizes it).

For a decision at a bar's ts_available, the label is the log return from that
bar's close to the last close available by decision + horizon, within the same
session. Labels never cross the session close, which keeps day-level splits
free of overlap. Missing data gives NaN, never a zero return:

    - the horizon runs past the session close  -> NaN
    - no bar printed during the horizon        -> NaN

The entry at the decision bar's close is an idealization (no spread, no delay);
L2's exit-policy labeler models fills.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shpe_research.data.calendar import get_calendar, session_date


@dataclass(frozen=True)
class ForwardReturn:
    horizon_minutes: int

    @property
    def name(self) -> str:
        return f"fwd_return_{self.horizon_minutes}m"

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """decision_ts, label_end_ts and the forward log return for each bar of one symbol."""
        horizon = pd.Timedelta(minutes=self.horizon_minutes)
        decision = bars["ts_available"]
        day = session_date(decision - pd.Timedelta(minutes=1))
        closes = get_calendar().schedule(day.min(), day.max())["close"].reindex(day.to_numpy()).to_numpy()
        end = decision + horizon
        target = pd.DataFrame({"t": end, "day": day})
        history = pd.DataFrame({"t": decision, "day": day, "exit": bars["close"], "exit_ts": decision})
        matched = pd.merge_asof(target, history, on="t", by="day", direction="backward")
        exit_price = matched["exit"].to_numpy()
        valid = (end.to_numpy() <= closes) & (matched["exit_ts"].to_numpy() > decision.to_numpy())
        value = np.where(valid, np.log(exit_price / bars["close"].to_numpy()), np.nan)
        return pd.DataFrame(
            {"decision_ts": decision.to_numpy(), "label_end_ts": end.to_numpy(), self.name: value},
            index=bars.index,
        )


def compute_labels(bars: pd.DataFrame, label: ForwardReturn) -> pd.DataFrame:
    """Labels for every symbol: symbol, decision_ts, label_end_ts and the label column."""
    frames = []
    for symbol, group in bars.groupby("symbol", sort=True):
        out = label.compute(group.sort_values("ts"))
        out.insert(0, "symbol", symbol)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)
