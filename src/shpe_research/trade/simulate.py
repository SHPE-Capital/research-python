"""Trade simulator (C1a stub; T1 implements it)."""

from __future__ import annotations

import pandas as pd

from shpe_research.contracts.candidates import CandidateFrame
from shpe_research.trade.types import CostModel, ExitPolicy, FillModel, RiskLimits, SizingRule, TradeLedger


def simulate_trades(
    bars: pd.DataFrame,
    candidates: CandidateFrame,
    policy: ExitPolicy,
    fill: FillModel = FillModel(),  # noqa: B008  (frozen dataclass default)
    costs: CostModel | None = None,
    sizing: SizingRule | None = None,
    risk: RiskLimits | None = None,
) -> TradeLedger:
    """Fill candidates on `bars` under `policy` and return the trade ledger."""
    raise NotImplementedError("TODO(T1): trade simulator")
