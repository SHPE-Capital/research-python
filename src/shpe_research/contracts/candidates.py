"""CandidateFrame contract (C1a stub; C1b fills the validator).

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


def validate_candidates(candidates: Any) -> None:
    """Raise ContractError listing every way `candidates` breaks the schema (C1b)."""
    raise NotImplementedError("validate_candidates is filled by C1b")
