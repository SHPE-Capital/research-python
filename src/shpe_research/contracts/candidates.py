"""CandidateFrame contract (C1a names; C1b validator).

A CandidateFrame is a plain DataFrame, one row per decision a strategy wants
to take, known at ``decision_ts`` and not before:

    symbol               string   Traded symbol (leg 1 for pairs)
    decision_ts          utc ns   Equal to the ts_available of the bar the decision used
    side                 string   "long" | "short" (pairs: long = buy leg 1, sell leg 2)
    strategy_id          string   "<strategy_type>:<config hash>"
    action               string   optional, "enter" (default) | "exit"
    exit_reason          string   on exit rows, from the T1 vocabulary
    trigger_value, zscore, stretch    float   optional, what the generator knew at decision_ts
    counterpart_symbol, hedge_ratio   string, float   pairs only
    stop_price, target_price          float   optional price levels for order exits
    entry_price          float    optional trigger level for a stop entry
    qty                  int      optional strategy-native quantity
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from shpe_research.contracts.base import ContractError, check_columns

CandidateFrame = pd.DataFrame

CANDIDATE_COLUMNS: dict[str, str] = {
    "symbol": "str",
    "decision_ts": "utc",
    "side": "str",
    "strategy_id": "str",
}

OPTIONAL_CANDIDATE_COLUMNS: dict[str, str] = {
    "action": "str",
    "exit_reason": "str",
    "trigger_value": "float",
    "zscore": "float",
    "stretch": "float",
    "counterpart_symbol": "str",
    "hedge_ratio": "float",
    "stop_price": "float",
    "target_price": "float",
    "entry_price": "float",
    "qty": "nullable_int",
}


SIDES = ("long", "short")
ACTIONS = ("enter", "exit")
_KEY = ["symbol", "decision_ts", "strategy_id", "action"]


def validate_candidates(candidates: Any) -> None:
    """Raise ContractError("candidates", problems) listing every way `candidates` breaks the schema.

    Checks: the four required columns exist with the right dtypes (`decision_ts` must be UTC nanoseconds);
    optional columns, when present, have their dtype; `side` in {long, short}; `action` in {enter, exit}
    (a missing `action` column means every row is "enter"); every exit row has a non-empty `exit_reason`; and
    no two rows share (symbol, decision_ts, strategy_id, action). Timing (decision_ts is a bar's
    ts_available) is the generator's duty and is not checkable without the bars.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise ContractError("candidates", [f"expected a DataFrame, got {type(candidates).__name__}"])
    df = candidates
    problems = check_columns(df, CANDIDATE_COLUMNS)
    problems += check_columns(df, {k: v for k, v in OPTIONAL_CANDIDATE_COLUMNS.items() if k in df.columns})
    if "side" in df.columns:
        bad = sorted(set(df["side"].dropna().astype(str)) - set(SIDES))
        if bad or df["side"].isna().any():
            problems.append(f"'side' must be one of {SIDES}, got {bad or 'missing values'}")
    action = df["action"] if "action" in df.columns else pd.Series("enter", index=df.index)
    bad = sorted(set(action.dropna().astype(str)) - set(ACTIONS))
    if bad or action.isna().any():
        problems.append(f"'action' must be one of {ACTIONS}, got {bad or 'missing values'}")
    exits = action == "exit"
    if exits.any():
        reason = df["exit_reason"] if "exit_reason" in df.columns else pd.Series(None, index=df.index)
        if (reason[exits].isna() | (reason[exits].astype(str).str.strip() == "")).any():
            problems.append("'exit_reason' is required on every row with action 'exit'")
    if all(k in df.columns for k in ("symbol", "decision_ts", "strategy_id")):
        keyed = df.assign(action=action)[_KEY]
        if keyed.duplicated().any():
            problems.append(f"duplicate rows on {_KEY} ('decision_ts' must be unique per symbol, strategy, action)")
    if problems:
        raise ContractError("candidates", problems)
