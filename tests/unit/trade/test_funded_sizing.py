import numpy as np
import pytest

from shpe_research.trade import funded_sizing as fs

KW = dict(per_stock_pct=0.08, risk_pct=0.0025, total_exposure_cap=0.80)
# Worked example (assumed inputs, the guide text is not in this repo): equity 100k, price 50, ATR 1.0
BASE = dict(equity=100_000.0, price=50.0, stop_distance=1.5, cash_free=100_000.0, unit_cost=50.0,
            exposure_used=0.0, liq_shares=1e9)


def size(**over):
    return fs.size_order(**{**BASE, **over}, **KW)


def test_worked_example_160_shares():
    # 8% cap 160, risk cap 166.7, cash 2000, exposure 1600, liquidity huge
    assert size() == (160, fs.PER_STOCK)


def test_each_cap_binds_in_turn():
    assert size(stop_distance=3.0) == (83, fs.RISK)            # 250 / 3
    assert size(cash_free=2_500.0) == (50, fs.CASH)            # 2500 / 50
    assert size(exposure_used=79_000.0) == (20, fs.EXPOSURE)   # (80k - 79k) / 50
    assert size(liq_shares=12.7) == (12, fs.LIQUIDITY)


def test_zero_quantity_reports_binding_cap():
    assert size(cash_free=40.0) == (0, fs.CASH)
    assert size(exposure_used=80_000.0) == (0, fs.EXPOSURE)
    assert size(liq_shares=0.0) == (0, fs.LIQUIDITY)


def test_negative_room_is_zero_not_negative():
    assert size(exposure_used=90_000.0)[0] == 0


def test_invalid_inputs_give_zero():
    assert size(price=float("nan")) == (0, fs.INVALID)
    assert size(stop_distance=0.0) == (0, fs.INVALID)
    assert size(equity=-5.0) == (0, fs.INVALID)


def test_size_multiplier_scales_equity_caps_only():
    assert fs.size_order(**BASE, **KW, size_multiplier=0.5) == (80, fs.PER_STOCK)
    assert fs.size_order(**{**BASE, "cash_free": 2_500.0}, **KW, size_multiplier=2.0)[0] == 50


def test_liquidity_cap_median_and_min_obs():
    buf = np.zeros(20)
    buf[:5] = [500.0, 100.0, 300.0, 200.0, 400.0]
    assert fs.liquidity_cap_shares(buf, 5, 0.01, 5) == pytest.approx(3.0)       # median 300
    buf[:4] = [100.0, 300.0, 200.0, 400.0]
    assert fs.liquidity_cap_shares(buf, 4, 0.01, 4) == pytest.approx(2.5)       # mean of 200, 300
    assert fs.liquidity_cap_shares(buf, 14, 0.01, 15) == 0.0                    # below min observations
    assert fs.liquidity_cap_shares(buf, 0, 0.01, 0) == 0.0
