"""TODO(E1): trading-oriented evaluation added to every report.  Level: intermediate.

evaluation/report.py measures prediction and ranking quality. Add what decides
whether a signal is worth trading:

    ic_decay        mean daily IC of the same score against forward returns at
                    several horizons (needs L1). How fast does the edge fade?
    turnover        how much the score-ranked portfolio changes between decisions.
    cost_adjusted   P&L of a simple rule (long top decile / short bottom decile,
                    or "take when score > threshold") after a cost in bps per
                    side; report the cost at which the edge disappears.
    by_group        the IC and P&L split by symbol, month and (when present) the
                    universe cell from D4, to show whether a few names drive the result.

Done when:
    - tests/todo/test_eval_todo.py::test_trading_metrics passes on fixture runs.
    - run_experiment writes these into report.json and report.md for every run.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def trading_metrics(predictions: pd.DataFrame, cost_bps: Sequence[float] = (0, 2, 5, 10)) -> dict:
    raise NotImplementedError("TODO(E1): turnover, IC decay and cost-adjusted P&L")
