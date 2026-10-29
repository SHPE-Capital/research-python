"""TODO(L2): exit-policy (triple-barrier) labels with the platform's bar fill rules.  Level: intermediate.

For each candidate entry, simulate one trade under a FIXED exit policy and
record its net return. This is the label meta-labeling models learn (the ORB
plan's y = [q(exit fill - entry fill) - fees] / (q * entry fill)).

Policy parameters (chosen on validation data, never on test):
    stop        price level or distance (e.g. opening-range midpoint, or k * volatility)
    target      optional profit-taking level
    max_minutes time exit (ORB plan: 30 minutes after the entry fill)
    side        long (ORB) or short

Fill rules must match the trading platform's bar model
(trading-platform/docs/architecture/backtest-execution.md):
    - entry fills at the NEXT bar's open after the decision, +/- half spread and slippage bps
    - a stop triggers when a regular-session bar reaches it and fills at the stop,
      or at that bar's open if the bar gapped through it (never better)
    - when stop and target are both touched inside one bar, assume the stop (conservative)
    - no fill is invented when there is no next bar; such candidates get an explicit
      status, never a zero return

Output per candidate: entry_ts, entry_price, exit_ts, exit_price, exit_reason
(stop | target | time | session_end | unfilled), net_return, label_end_ts (= exit_ts).

Done when:
    - tests/todo/test_labels_todo.py::test_exit_policy passes (hand-computed trades,
      including a gap through the stop and a same-bar stop/target).
    - Reconciled against the platform's backtest on 5 sessions (record the comparison
      in studies/r1-orb-selectors/).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExitPolicy:
    max_minutes: int = 30
    stop_distance: float | None = None
    target_distance: float | None = None
    side: str = "long"
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    commission_per_share: float = 0.0


def simulate_exits(bars: pd.DataFrame, candidates: pd.DataFrame, policy: ExitPolicy) -> pd.DataFrame:
    """One row per candidate (symbol, decision_ts, optional stop/target levels) with its trade outcome."""
    raise NotImplementedError("TODO(L2): exit-policy labels")
