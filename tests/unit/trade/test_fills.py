"""T1 acceptance: platform bar fill rules (BACKTEST_ENGINE_VERSION 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.trade.simulate import simulate_trades
from shpe_research.trade.types import ExitPolicy, FillModel

ZERO = FillModel(half_spread_bps=0, slippage_bps=0, commission_per_share=0)
FLAT = [100.0, 100.5, 99.5, 100.0]


def _bars(rows, start="2026-10-01T13:30Z", volume=1000):
    ts = to_utc_ns(pd.date_range(start, periods=len(rows), freq="1min"))
    cols = ["open", "high", "low", "close", *(["volume"] if len(rows[0]) == 5 else [])]
    df = pd.DataFrame(rows, columns=cols)
    if "volume" not in df:
        df["volume"] = volume
    return df.assign(symbol="AAA", ts=ts, ts_available=ts + pd.Timedelta(minutes=1), vwap=df["close"],
                     trade_count=pd.array([10] * len(df), dtype="Int64"), session="regular")


def _cands(bars, side="long", at=0, **extra):
    return pd.DataFrame({"symbol": ["AAA"], "decision_ts": [bars["ts_available"].iloc[at]], "side": [side],
                         "strategy_id": ["t:1"], **{k: [v] for k, v in extra.items()}})


def test_defaults_match_platform():
    f = FillModel()
    assert (f.half_spread_bps, f.slippage_bps, f.commission_per_share) == (2.0, 5.0, 0.005)
    assert f.volume_participation_cap == 0.1
    assert f.allow_partial_fills is True


@pytest.mark.parametrize("side", ["long", "short"])
def test_market_price_formula(side):
    bars = _bars([FLAT, [101.0, 101.5, 100.5, 101.0], FLAT, [102.0, 102.5, 101.5, 102.0]])
    out = simulate_trades(bars, _cands(bars, side), ExitPolicy()).iloc[0]
    hs, sl = 2e-4, 5e-4
    if side == "long":
        assert out["entry_price"] == pytest.approx(101.0 * (1 + hs) * (1 + sl), abs=1e-12)
        assert out["exit_price"] == pytest.approx(102.0 * (1 - hs) * (1 - sl), abs=1e-12)
    else:
        assert out["entry_price"] == pytest.approx(101.0 * (1 - hs) * (1 - sl), abs=1e-12)
        assert out["exit_price"] == pytest.approx(102.0 * (1 + hs) * (1 + sl), abs=1e-12)
    assert out["exit_reason"] == "session_end"
    assert out["cost"] > 0
    assert out["net_return"] < out["gross_return"]


def test_zero_volume_unfilled():
    bars = _bars([[*FLAT, 1000], [100.0, 100.5, 99.5, 100.0, 0], [*FLAT, 1000], [*FLAT, 1000]])
    out = simulate_trades(bars, _cands(bars), ExitPolicy())
    assert list(out["exit_reason"]) == ["unfilled"]
    assert pd.isna(out["entry_ts"].iloc[0])


def test_no_next_bar_unfilled():
    bars = _bars([FLAT, FLAT])
    out = simulate_trades(bars, _cands(bars, at=1), ExitPolicy())
    assert list(out["exit_reason"]) == ["unfilled"]


def test_partial_fill_cap():
    bars = _bars([FLAT, FLAT, FLAT, FLAT])
    out = simulate_trades(bars, _cands(bars, qty=500), ExitPolicy(), fill=ZERO).iloc[0]
    assert out["qty"] == 100  # 0.1 x 1000
    assert bool(out["partial_fill"])
    full = simulate_trades(bars, _cands(bars, qty=50), ExitPolicy(), fill=ZERO).iloc[0]
    assert full["qty"] == 50
    assert not bool(full["partial_fill"])


def test_entry_bar_excluded():
    # the stop (99) is touched only inside the entry bar; later bars stay above it
    bars = _bars([FLAT, [100.0, 100.5, 98.0, 100.0], FLAT, FLAT])
    out = simulate_trades(bars, _cands(bars, stop_price=99.0), ExitPolicy(stop_rule="price"), fill=ZERO).iloc[0]
    assert out["exit_reason"] == "session_end"


def test_buy_stop_entry_gap():
    bars = _bars([FLAT, [102.0, 103.0, 101.5, 102.5], FLAT, FLAT])
    out = simulate_trades(bars, _cands(bars, entry_price=101.0), ExitPolicy(entry_order="stop")).iloc[0]
    assert out["entry_ts"] == bars["ts"].iloc[1]
    assert out["entry_price"] == pytest.approx(102.0 * (1 + 2e-4) * (1 + 5e-4), abs=1e-12)


def test_buy_stop_entry_not_triggered():
    bars = _bars([FLAT, FLAT, FLAT, FLAT])
    out = simulate_trades(bars, _cands(bars, entry_price=105.0), ExitPolicy(entry_order="stop"))
    assert list(out["exit_reason"]) == ["unfilled"]


def test_stop_first_same_bar():
    bars = _bars([FLAT, FLAT, [100.5, 103.0, 98.5, 100.0], FLAT])
    pol = ExitPolicy(stop_rule="price", target_rule="price")
    out = simulate_trades(bars, _cands(bars, stop_price=99.0, target_price=102.0), pol, fill=ZERO).iloc[0]
    assert out["exit_reason"] == "stop_loss"
    assert out["exit_price"] == pytest.approx(99.0)


def test_target_fill_and_short_side():
    bars = _bars([FLAT, FLAT, [100.0, 100.2, 97.5, 98.0], FLAT])
    pol = ExitPolicy(target_rule="price")
    out = simulate_trades(bars, _cands(bars, "short", target_price=98.0), pol, fill=ZERO).iloc[0]
    assert out["exit_reason"] == "take_profit"
    assert out["exit_price"] == pytest.approx(98.0)
    assert out["gross_return"] == pytest.approx(0.02)


def test_early_close_flag():
    pol = ExitPolicy(stop_rule="price")
    early = _bars([FLAT] * 4, start="2026-11-27T14:30Z")  # 13:00 close
    normal = _bars([FLAT] * 4, start="2026-11-25T14:30Z")
    assert bool(simulate_trades(early, _cands(early, stop_price=90.0), pol).iloc[0]["early_close"])
    assert not bool(simulate_trades(normal, _cands(normal, stop_price=90.0), pol).iloc[0]["early_close"])
