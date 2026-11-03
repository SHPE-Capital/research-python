"""T1 acceptance: exit priority, signal vs order exits, trailing stops and the ledger schema."""

from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.trade.simulate import simulate_trades
from shpe_research.trade.types import TRADE_LEDGER_COLUMNS, ExitPolicy, FillModel

ZERO = FillModel(half_spread_bps=0, slippage_bps=0, commission_per_share=0)
FLAT = [100.0, 100.5, 99.5, 100.0]


def _bars(rows, start="2026-10-01T13:30Z"):
    ts = to_utc_ns(pd.date_range(start, periods=len(rows), freq="1min"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    return df.assign(symbol="AAA", ts=ts, ts_available=ts + pd.Timedelta(minutes=1), volume=1000,
                     vwap=df["close"], trade_count=pd.array([10] * len(df), dtype="Int64"), session="regular")


def _cands(bars, at=0, side="long", **extra):
    return pd.DataFrame({"symbol": ["AAA"], "decision_ts": [bars["ts_available"].iloc[at]], "side": [side],
                         "strategy_id": ["t:1"], **{k: [v] for k, v in extra.items()}})


def test_signal_exit_next_open():
    bars = _bars([FLAT, FLAT, [100.0, 100.5, 99.5, 100.0], [103.0, 104.0, 102.0, 104.0],
                  [105.0, 105.5, 104.5, 105.0], [106.0, 106.5, 105.5, 106.0]])
    cands = pd.concat([_cands(bars), _cands(bars, at=2, action="exit")], ignore_index=True)
    out = simulate_trades(bars, cands, ExitPolicy(), fill=ZERO)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["exit_reason"] == "exit"
    assert row["exit_ts"] == bars["ts"].iloc[3]
    assert row["exit_price"] == pytest.approx(103.0)  # next open, not the decision bar's close (100.0)


def test_priority():
    gap = [98.0, 98.5, 97.0, 98.0]
    bars = _bars([FLAT, FLAT, FLAT, gap, FLAT, FLAT])
    cands = pd.concat([_cands(bars, stop_price=99.0), _cands(bars, at=2, action="exit")], ignore_index=True)
    pol = ExitPolicy(stop_rule="price", max_hold_bars=2)  # max hold also falls due at bar 3
    assert simulate_trades(bars, cands, pol, fill=ZERO).iloc[0]["exit_reason"] == "stop_loss"
    held = simulate_trades(bars, cands, ExitPolicy(max_hold_bars=2), fill=ZERO)
    assert held.iloc[0]["exit_reason"] == "max_hold_exit"
    assert simulate_trades(bars, cands, ExitPolicy(), fill=ZERO).iloc[0]["exit_reason"] == "exit"


def test_chandelier_completed_bars_only():
    bars = _bars([
        [100.0, 100.5, 99.5, 100.0],
        [100.0, 100.5, 99.5, 100.0],  # entry at 100.0; ATR(2) after this bar = 1.0
        [100.0, 101.5, 99.8, 101.0],  # TR 1.7, ATR 1.35, highest high 101.5
        [101.0, 105.0, 100.1, 104.0],  # trail from completed bars: 101.5 - 1.35 = 100.15; own high 105 is ignored
        [104.0, 104.5, 103.5, 104.0],
    ])
    pol = ExitPolicy(trail_rule="chandelier", trail_atr_mult=1.0, atr_window=2)
    row = simulate_trades(bars, _cands(bars), pol, fill=ZERO).iloc[0]
    assert row["exit_reason"] == "stop_loss"
    assert row["exit_price"] == pytest.approx(100.15)
    assert row["exit_ts"] == bars["ts_available"].iloc[3]


def test_breakeven_1r():
    bars = _bars([FLAT, FLAT, [100.0, 101.2, 99.6, 101.0], [101.0, 101.5, 99.9, 100.5], FLAT])
    pol = ExitPolicy(stop_rule="price", trail_rule="breakeven_1r")
    row = simulate_trades(bars, _cands(bars, stop_price=99.0), pol, fill=ZERO).iloc[0]
    # bar 2 moved +1.2 >= 1R (1.0): the stop moves to entry (100.0) and bar 3 (low 99.9) hits it
    assert row["exit_reason"] == "stop_loss"
    assert row["exit_price"] == pytest.approx(100.0)


def test_r_multiple_target():
    bars = _bars([FLAT, FLAT, [100.5, 102.5, 100.0, 102.0], FLAT])
    pol = ExitPolicy(stop_rule="price", target_rule="r_multiple", target_r=2.0)
    row = simulate_trades(bars, _cands(bars, stop_price=99.0), pol, fill=ZERO).iloc[0]
    assert row["exit_reason"] == "take_profit"
    assert row["exit_price"] == pytest.approx(102.0)  # entry 100.0 + 2 x 1.0 risk


def test_session_end_off():
    day1 = _bars([FLAT] * 4, start="2026-10-01T13:30Z")
    day2 = _bars([FLAT] * 4, start="2026-10-02T13:30Z")
    bars = pd.concat([day1, day2], ignore_index=True)
    on = simulate_trades(bars, _cands(bars), ExitPolicy(max_hold_bars=5), fill=ZERO).iloc[0]
    assert on["exit_reason"] == "session_end"
    assert on["exit_ts"] == day1["ts"].iloc[-1]
    off = simulate_trades(bars, _cands(bars), ExitPolicy(max_hold_bars=5, session_end=False), fill=ZERO).iloc[0]
    assert off["exit_reason"] == "max_hold_exit"
    assert off["exit_ts"] > day1["ts"].iloc[-1]


def test_open_position_at_data_end_is_not_invented():
    bars = _bars([FLAT] * 4)
    out = simulate_trades(bars, _cands(bars), ExitPolicy(session_end=False), fill=ZERO)
    assert out.empty
    assert out.attrs["open_at_end"] == 1


def test_one_position_and_flip():
    bars = _bars([FLAT] * 8)
    same = pd.concat([_cands(bars, at=0), _cands(bars, at=2)], ignore_index=True)
    assert len(simulate_trades(bars, same, ExitPolicy(), fill=ZERO)) == 1
    flip = pd.concat([_cands(bars, at=0), _cands(bars, at=2, side="short")], ignore_index=True)
    out = simulate_trades(bars, flip, ExitPolicy(), fill=ZERO)
    assert list(out["side"]) == ["long", "short"]
    assert out["exit_reason"].iloc[0] == "exit"
    assert out["exit_ts"].iloc[0] == out["entry_ts"].iloc[1] == bars["ts"].iloc[3]


def test_flat_by():
    bars = _bars([FLAT] * 20)  # 09:30 .. 09:49 New York
    row = simulate_trades(bars, _cands(bars), ExitPolicy(flat_by="09:40", session_end=False), fill=ZERO).iloc[0]
    assert row["exit_ts"] == bars["ts"].iloc[10]


def test_no_lookahead():
    bars = synthetic_bars(symbols=["AAA"], start="2026-10-01", end="2026-10-09")
    ts = bars["ts"].dt.tz_convert("America/New_York")
    decisions = bars.loc[(ts.dt.hour == 10) & (ts.dt.minute == 0), "ts_available"].tolist()
    cands = pd.DataFrame({"symbol": "AAA", "decision_ts": decisions, "side": "long", "strategy_id": "t:1"})
    pol = ExitPolicy(stop_rule="atr", stop_atr_mult=1.0, target_rule="pct", target_pct=0.001, max_hold_bars=45,
                     trail_rule="chandelier", atr_window=10)
    full = simulate_trades(bars, cands, pol).reset_index(drop=True)
    assert len(full) >= 3
    for i, row in full.iterrows():
        cut = bars[bars["ts"] <= row["exit_ts"]]
        again = simulate_trades(cut, cands.iloc[[i]], pol).iloc[0]
        pd.testing.assert_series_equal(again, row, check_names=False)


def test_ledger_schema():
    bars = _bars([FLAT] * 5)
    out = simulate_trades(bars, _cands(bars), ExitPolicy())
    assert tuple(out.columns[: len(TRADE_LEDGER_COLUMNS)]) == TRADE_LEDGER_COLUMNS
    assert (out["label_end_ts"] == out["exit_ts"]).all()
    assert str(out["exit_ts"].dtype) == "datetime64[ns, UTC]"
    empty = simulate_trades(bars, _cands(bars).iloc[0:0], ExitPolicy())
    assert tuple(empty.columns[: len(TRADE_LEDGER_COLUMNS)]) == TRADE_LEDGER_COLUMNS
