import numpy as np
import pytest

from shpe_research.trade.account_types import (
    AccountConfig,
    MarketPanel,
    SignalMask,
    simulate_account,
)


def _panel(n_sym=2, n_min=4, **over):
    arr = np.ones((n_sym, n_min), dtype=np.float32)
    kw = dict(
        asset_ids=tuple(f"A{i}" for i in range(n_sym)),
        open=arr, high=arr, low=arr, close=arr, volume=arr,
        session_index=np.array([0, 0, 1, 1]),
        eligible=np.ones((n_sym, n_min), dtype=bool),
    )  # fmt: skip
    kw.update(over)
    return MarketPanel(**kw)


def test_defaults_reproduce_guide():
    c = AccountConfig()
    assert (c.starting_cash, c.max_positions, c.per_stock_pct, c.risk_pct) == (100_000.0, 10, 0.08, 0.0025)
    assert c.ranking == "dollar_volume_20d" and c.entry_score_threshold is None and c.size_multiplier is None


def test_simulate_account_delegates_to_engine():
    p = _panel()
    m = SignalMask(np.zeros((2, 4), dtype=bool), np.ones((2, 4), dtype=np.float32))
    r = simulate_account(p, m, AccountConfig())
    assert r.daily_returns.tolist() == [0.0, 0.0] and r.ledger == ()


def test_panel_validates_shapes_and_ids():
    with pytest.raises(ValueError, match="shape"):
        _panel(close=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        _panel(asset_ids=("A", "A"))
    with pytest.raises(ValueError, match="non-decreasing"):
        _panel(session_index=np.array([1, 0, 1, 1]))
    with pytest.raises(ValueError, match="shapes differ"):
        SignalMask(np.zeros((2, 4), dtype=bool), np.ones((2, 3)))
