"""Forward-return labels at several horizons (L1), with cost and volatility variants.

Generalizes labels/forward_return.py: one return column and one label_end_ts column per horizon, same
semantics per horizon (same session only, NaN when nothing prints or the horizon passes the close).
Also used by E1 to measure how fast a signal's IC decays with horizon.

TODO(C1a): experiments/runner.py wiring (`label: {type: ..., horizon_minutes: [...]}`) is owned by C1a.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from shpe_research.labels.forward_return import ForwardReturn


def forward_returns(
    bars: pd.DataFrame,
    horizons: Sequence[int] = (5, 15, 30, 60),
    vol_normalized: bool = False,
    cost_bps: float = 0.0,
    vol_window: int = 15,
) -> pd.DataFrame:
    """Forward log returns at several horizons for every symbol.

    Output has the index of ``bars`` and columns ``symbol``, ``decision_ts`` (= ``ts_available``), and for each
    horizon ``h``: ``fwd_return_{h}m`` and ``label_end_ts_{h}m`` (= decision_ts + h minutes). Each return column
    equals ``ForwardReturn(h)`` (same session only; NaN if nothing prints or the horizon passes the close).

    ``cost_bps`` subtracts a round-trip cost (bps of price, as a log-return approximation) from every
    non-NaN return, so a positive sign means positive after costs.
    ``vol_normalized`` divides the (cost-adjusted) return by ``RealizedVol(vol_window)`` x sqrt(h), where the
    volatility is the per-minute estimate on the decision bar (known at ``ts_available``). Rows without a
    positive finite volatility (warm-up, gaps) are NaN, never zero.
    """
    from shpe_research.features.volatility.realized_vol import RealizedVol

    if not horizons:
        raise ValueError("horizons must not be empty")
    cost = cost_bps / 1e4
    labels = [ForwardReturn(int(h)) for h in horizons]
    vol_feature = RealizedVol(vol_window)
    frames = []
    for symbol, group in bars.groupby("symbol", sort=True):
        group = group.sort_values("ts")
        out = pd.DataFrame({"symbol": symbol, "decision_ts": group["ts_available"].to_numpy()}, index=group.index)
        vol = vol_feature.compute(group) if vol_normalized else None
        for label in labels:
            res = label.compute(group)
            value = res[label.name] - cost
            if vol is not None:
                scale = vol * np.sqrt(label.horizon_minutes)
                value = (value / scale.where(scale > 0)).astype("float64")
            out[label.name] = value
            out[f"label_end_ts_{label.horizon_minutes}m"] = res["label_end_ts"]
        frames.append(out)
    return pd.concat(frames).reindex(bars.index)
