from __future__ import annotations

import subprocess
import sys

import pytest

from shpe_research.contracts import candidates
from shpe_research.contracts.base import ContractError


def test_schema_names():
    assert set(candidates.CANDIDATE_COLUMNS) == {"symbol", "decision_ts", "side", "strategy_id"}
    assert set(candidates.OPTIONAL_CANDIDATE_COLUMNS) == {
        "action", "exit_reason", "trigger_value", "zscore", "stretch", "counterpart_symbol",
        "hedge_ratio", "stop_price", "target_price", "entry_price", "qty",
    }


def test_validator_rejects_non_frames():
    with pytest.raises(ContractError):
        candidates.validate_candidates(None)


def test_trade_does_not_import_events():
    code = (
        "import sys, shpe_research.trade, shpe_research.trade.types, shpe_research.trade.simulate;"
        "sys.exit(1 if 'shpe_research.events' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_simulate_trades_is_a_t1_stub():
    from shpe_research.trade.simulate import simulate_trades
    from shpe_research.trade.types import ExitPolicy, FillModel

    assert FillModel().half_spread_bps == 2.0
    with pytest.raises(NotImplementedError, match=r"TODO\(T1\)"):
        simulate_trades(None, None, ExitPolicy())
