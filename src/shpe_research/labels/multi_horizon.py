"""TODO(L1): forward-return labels at several horizons, with variants.  Level: beginner.

labels/forward_return.py (C0) handles one horizon. Generalize it:

    forward_returns(bars, horizons=(5, 15, 30, 60)) -> one column per horizon,
        each with the C0 semantics (same session only, NaN when nothing prints
        or the horizon passes the close), plus label_end_ts per horizon.
    vol_normalized=True   divide by a volatility estimate known at the decision
                          (use F3's RealizedVol once it lands).
    cost_bps=x            subtract a round-trip cost so "positive" means
                          "positive after costs" for sign targets.

Also used by E1 to measure how fast a signal's IC decays with horizon.

Done when:
    - tests/todo/test_labels_todo.py::test_multi_horizon passes.
    - experiments/runner.py accepts `label: {type: forward_return, horizon_minutes: [...]}`
      or a separate label type, documented in configs/experiments/README.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def forward_returns(
    bars: pd.DataFrame,
    horizons: Sequence[int] = (5, 15, 30, 60),
    vol_normalized: bool = False,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    raise NotImplementedError("TODO(L1): multi-horizon forward returns")
