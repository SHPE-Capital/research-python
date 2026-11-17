import numpy as np
import pytest

from shpe_research.trade.exit_package import run_exit
from shpe_research.trade.types import CostModel

ZERO = CostModel(half_spread_bps=0.0, slippage_bps=0.0, commission_per_share=0.0)


def bars(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.arange(len(rows))


def run(rows, close_ts=390, costs=ZERO, **kw):
    o, h, lo, c, t = bars(rows)
    return run_exit(o, h, lo, c, t, 0, 100.0, 2.0, close_ts, costs=costs, **kw)  # initial stop 97


def test_initial_stop_cross():
    r = run([(100, 101, 99, 100), (100, 100.5, 96, 98)])
    assert (r.reason, r.exit_idx, r.exit_ref_price, r.ambiguous) == ("stop", 1, 97.0, False)


def test_gap_through_stop_fills_at_open_with_cost():
    cm = CostModel(half_spread_bps=10.0, slippage_bps=0.0, commission_per_share=0.0)
    r = run([(100, 101, 99, 100), (95, 96, 94, 95)], costs=cm)
    assert r.reason == "stop" and r.exit_ref_price == 95.0
    assert r.exit_price == pytest.approx(95.0 * (1 - 0.001))


def test_trailing_activation_applies_next_bar():
    # bar0 close 103 -> trailing, stop 103 - 3 = 100 from bar1; bar1 low 99.5 crosses it
    r = run([(100, 103.5, 99.5, 103), (103, 104, 99.5, 103.5)])
    assert (r.reason, r.exit_idx, r.exit_ref_price) == ("stop", 1, 100.0)


def test_new_stop_not_applied_in_same_bar():
    # bar0 low 97.5 > initial stop 97, so no exit on bar0 even though the raised stop (100) would hit
    r = run([(100, 103.5, 97.5, 103), (103, 104, 102, 103)], close_ts=1000)
    assert r.reason == "data_end" and r.exit_idx == 1


def test_ambiguous_flag():
    r = run([(100, 102.5, 96.5, 100)])
    assert r.reason == "stop" and r.ambiguous and r.exit_ref_price == 97.0


def test_max_hold_60_min():
    r = run([(100, 100.5, 99.5, 100)] * 70, close_ts=1000)
    assert (r.reason, r.exit_idx, r.exit_ref_price) == ("max_hold", 60, 100.0)


def test_session_exit_close_minus_5():
    r = run([(100, 100.5, 99.5, 100)] * 30, close_ts=20)
    assert (r.reason, r.exit_idx) == ("session_end", 15)


def test_early_close_uses_actual_close():
    r = run([(100, 100.5, 99.5, 100)] * 30, close_ts=10, early_close=True)
    assert (r.reason, r.exit_idx, r.early_close) == ("session_end", 5, True)


def test_data_end():
    r = run([(100, 100.5, 99.5, 100.2)] * 3, close_ts=1000)
    assert (r.reason, r.exit_idx, r.exit_ref_price) == ("data_end", 2, 100.2)


def test_invalid_atr():
    o, h, lo, c, t = bars([(1, 1, 1, 1)])
    with pytest.raises(ValueError):
        run_exit(o, h, lo, c, t, 0, 1.0, 0.0, 10)
