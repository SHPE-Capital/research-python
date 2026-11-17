"""C7 correctness scenarios for the funded account engine (hand-worked, small panels).

Panels: flat 50.00 bars unless overridden, sessions of 20 minute bars (session exit at offset 15, so
an entry must fill at offset <= 14), 16 flat warm-up sessions of huge volume for the 20-session
liquidity and ranking history. Trade sessions are numbered from 0 after the warm-up. Costs are zero
unless a test says otherwise. Partial fills are not modeled by the screening proxy (all or nothing).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shpe_research.trade.account import simulate_account
from shpe_research.trade.account_types import AccountConfig, ExitPackage, MarketPanel, SignalMask
from shpe_research.trade.exit_package import exit_scan

WARM = 16
L = 20
BASE = 50.0
HUGE = 1e9


def build(ids, lengths=(L,), bars=None, sigs=None, base=None, vol=None, warm_len=L):
    """Panel and mask. bars: {(id, sess, m): (o, h, l, c)} (NaN marks a missing bar); sigs: {(id, sess, m): atr}."""
    bars, sigs, base, vol = bars or {}, sigs or {}, base or {}, vol or {}
    all_len = [warm_len] * WARM + list(lengths)
    n = sum(all_len)
    sess = np.repeat(np.arange(len(all_len)), all_len)
    offs = np.concatenate([np.arange(k) for k in all_len])
    shp = (len(ids), n)
    o, h, lo, c = (np.zeros(shp) for _ in range(4))
    v = np.zeros(shp)
    sig = np.zeros(shp, dtype=bool)
    atr = np.full(shp, 1.0)
    for i, a in enumerate(ids):
        p = base.get(a, BASE)
        o[i] = h[i] = lo[i] = c[i] = p
        v[i] = vol.get(a, HUGE)
    for (a, s, m), (bo, bh, bl, bc) in bars.items():
        i, t = ids.index(a), int(np.flatnonzero((sess == s + WARM) & (offs == m))[0])
        o[i, t], h[i, t], lo[i, t], c[i, t] = bo, bh, bl, bc
    for (a, s, m), at in sigs.items():
        i, t = ids.index(a), int(np.flatnonzero((sess == s + WARM) & (offs == m))[0])
        sig[i, t], atr[i, t] = True, at
    panel = MarketPanel(tuple(ids), o, h, lo, c, v, sess, np.ones(shp, dtype=bool))
    return panel, SignalMask(sig, atr)


def run(panel, mask, **kw):
    kw.setdefault("cost_bps", 0.0)
    return simulate_account(panel, mask, AccountConfig(**kw), warmup_sessions=WARM)


def flat(p, n=1):
    return (p, p, p, p)


# ---- one stock, hand-worked ---------------------------------------------------------------

def test_one_stock_hand_worked_three_sessions():
    # Session 0: signal on bar 2 (close 50, ATR 1) -> limit 50.10, fills bar 3 open 50.05, 160 shares
    # (8% of 100k / 50 = 160 binds; risk cap 250 / 1.5 = 166). Stop 50.05 - 1.5 = 48.55.
    bars = {
        ("A", 0, 3): (50.05, 50.5, 50.0, 50.4),   # close < trigger 51.05, stop stays 48.55
        ("A", 0, 4): (50.4, 51.2, 50.3, 51.1),    # close 51.1 >= 51.05: trailing, stop 49.60 from bar 5
        ("A", 0, 5): (51.0, 51.5, 50.9, 51.4),    # stop 49.90 from bar 6
        ("A", 0, 6): (51.4, 51.5, 49.8, 50.0),    # low 49.8 <= 49.90: exit at 49.90
    }
    sigs = {("A", 0, 2): 1.0, ("A", 0, 8): 1.0}   # second signal same session: ignored, already entered
    r = run(*build(["A"], lengths=(L, L, L), bars=bars, sigs=sigs))
    (e,) = r.ledger
    assert (e.asset_id, e.qty, e.entry_price, e.exit_price, e.exit_reason) == ("A", 160, 50.05, 49.9, "stop")
    assert e.net_pnl == pytest.approx(160 * (49.9 - 50.05))
    assert r.daily_returns.tolist() == pytest.approx([160 * (49.9 - 50.05) / 100_000, 0.0, 0.0])
    # bar 6 high (51.5) also reached the trigger, so the C2 ambiguity flag is set once
    assert r.rejections == () and r.ambiguous_bars == 1 and r.unresolved_positions == 0
    assert r.summary["trades"] == 1 and r.summary["active_sessions"] == 1


def test_costs_are_deducted_per_side():
    bars = {("A", 0, 3): (50.0, 50.0, 50.0, 50.0)}
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}), cost_bps=10.0)
    (e,) = r.ledger
    assert e.exit_reason == "session_end" and e.qty == 160
    cost = e.qty * 50.0 * 0.001 * 2
    assert e.net_pnl == pytest.approx(-cost)
    assert r.daily_returns[0] == pytest.approx(-cost / 100_000)


def test_replaying_same_signal_minutes_places_one_order():
    sigs = {("A", 0, m): 1.0 for m in (2, 3, 4, 5)}  # signal stays true: pending and entered blocks re-entry
    r = run(*build(["A"], sigs=sigs))
    assert len(r.ledger) == 1 and r.rejections == ()


def test_replay_identical_twice():
    bars = {("A", 0, 3): (50.05, 50.5, 50.0, 50.4), ("B", 0, 4): (49.0, 50.0, 48.0, 49.0)}
    sigs = {("A", 0, 2): 1.0, ("B", 0, 2): 1.0, ("B", 0, 3): 1.0}
    panel, mask = build(["A", "B"], bars=bars, sigs=sigs)
    a, b = run(panel, mask), run(panel, mask)
    assert a.daily_returns.tobytes() == b.daily_returns.tobytes()
    assert a.ledger == b.ledger and a.rejections == b.rejections and a.summary == b.summary


# ---- competition, reservation, sizing ------------------------------------------------------

BIG = dict(per_stock_pct=0.6, total_exposure_cap=1.0, risk_pct=1.0)


def test_competing_signals_rank_by_dollar_volume_then_cash_limits_second():
    # B has more volume so it ranks first despite the later id; it takes 60%; A gets the remaining cash.
    sigs = {("A", 0, 2): 1.0, ("B", 0, 2): 1.0}
    r = run(*build(["A", "B"], sigs=sigs, vol={"A": 1e8, "B": 1e9}), **BIG)
    by = {e.asset_id: e.qty for e in r.ledger}
    assert by["B"] == 1200
    assert by["A"] == math.floor((100_000 - 1200 * 50.1) / 50.1)  # cash left after B's reservation


def test_cash_reserved_at_limit_for_pending_order_not_fill_price():
    # Both fill at 50.00 (below the 50.10 limit); the second order must still be sized on cash after the
    # first order's reservation at the limit (796), not after its fill (798).
    sigs = {("A", 0, 2): 1.0, ("B", 0, 2): 1.0}
    r = run(*build(["A", "B"], sigs=sigs, vol={"A": 1e8, "B": 1e9}), **BIG)
    assert {e.asset_id: e.qty for e in r.ledger} == {"B": 1200, "A": 796}


def test_equal_dollar_volume_breaks_tie_by_asset_id():
    sigs = {("A", 0, 2): 1.0, ("B", 0, 2): 1.0}
    r = run(*build(["B", "A"], sigs=sigs), max_positions=1)  # panel row order must not matter
    assert [e.asset_id for e in r.ledger] == ["A"]
    assert [(x.asset_id, x.reason) for x in r.rejections] == [("B", "max_positions")]


def test_integer_sizing_floors_to_whole_shares():
    r = run(*build(["A"], sigs={("A", 0, 2): 0.5}, base={"A": 33.33}))
    (e,) = r.ledger
    assert isinstance(e.qty, int) and e.qty == math.floor(8000 / 33.33) == 240


def test_proxy_fills_fully_or_not_at_all():
    # Partial fills are unsupported in proxy mode: the filled quantity is always the sized quantity.
    sigs = {("A", 0, 2): 1.0, ("B", 0, 2): 1.0}
    bars = {("B", 0, 3): (50.5, 50.5, 50.5, 50.5)}  # B opens above its limit: canceled whole
    r = run(*build(["A", "B"], sigs=sigs, bars=bars))
    assert [(e.asset_id, e.qty) for e in r.ledger] == [("A", 160)]


# ---- rejections and cancellations ----------------------------------------------------------

def test_rejected_order_size_zero_by_liquidity_is_logged():
    # 1% of 100 shares of same-minute volume is 1 share, so sizing gives 1; with 50 shares it gives 0.
    r = run(*build(["A"], sigs={("A", 0, 2): 1.0}, vol={"A": 50.0}))
    assert r.ledger == () and [(x.asset_id, x.minute, x.reason) for x in r.rejections] == [
        ("A", (WARM) * L + 2, "size_zero:liquidity")
    ]
    assert r.daily_returns[0] == 0.0


def test_liquidity_needs_min_observations():
    # only 10 warm-up sessions of history < 15 valid observations
    panel, mask = build(["A"], sigs={("A", 0, 2): 1.0})
    r = simulate_account(panel, mask, AccountConfig(cost_bps=0.0), warmup_sessions=WARM)
    assert len(r.ledger) == 1
    cfg = AccountConfig(cost_bps=0.0, liquidity_min_obs=17)  # 16 sessions of history available
    r2 = simulate_account(panel, mask, cfg, warmup_sessions=WARM)
    assert r2.ledger == () and r2.rejections[0].reason == "size_zero:liquidity"


def test_entry_cancelled_above_limit_returns_cash_and_logs():
    bars = {("A", 0, 3): (50.2, 50.2, 50.2, 50.2)}  # open 50.20 > limit 50.10
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    assert r.ledger == () and r.rejections[0].reason == "entry_cancelled_above_limit"
    assert r.daily_returns[0] == 0.0


def test_open_equal_to_limit_fills():
    bars = {("A", 0, 3): (50.1, 50.1, 50.1, 50.1)}
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    assert r.ledger[0].entry_price == 50.1 or r.ledger[0].entry_price == pytest.approx(50.1)


def test_missing_next_bar_means_no_fill():
    nan = float("nan")
    bars = {("A", 0, 3): (nan, nan, nan, nan)}
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    assert r.ledger == () and r.rejections[0].reason == "no_next_bar"


def test_canceled_entry_does_not_count_as_entered():
    bars = {("A", 0, 3): (50.2, 50.2, 50.2, 50.2)}
    sigs = {("A", 0, 2): 1.0, ("A", 0, 5): 1.0}  # second signal fills normally at bar 6
    r = run(*build(["A"], bars=bars, sigs=sigs))
    assert len(r.ledger) == 1 and r.ledger[0].entry_minute == WARM * L + 6


# ---- stops, gaps, trailing, ambiguity ------------------------------------------------------

def test_stop_replacement_applies_from_next_bar_only():
    # Entry bar 3 at 50.00 (ATR 1: stop 48.5, trigger 51.0). Bar 3 closes 51.2: trailing, new stop 49.7
    # from bar 4. Bar 3's own low (49.9) is above 48.5 -> no exit although it is below the new stop.
    bars = {
        ("A", 0, 3): (50.0, 51.3, 49.9, 51.2),
        ("A", 0, 4): (51.2, 51.3, 49.6, 50.0),  # low 49.6 <= 49.7: exit at the new stop
    }
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    (e,) = r.ledger
    assert (e.exit_minute, e.exit_reason) == (WARM * L + 4, "stop") and e.exit_price == pytest.approx(49.7)


def test_gap_through_stop_fills_at_lower_open():
    bars = {("A", 0, 3): (50.0, 50.0, 50.0, 50.0), ("A", 0, 4): (45.0, 46.0, 44.0, 45.0)}
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    (e,) = r.ledger
    assert e.exit_reason == "stop" and e.exit_price == 45.0
    assert e.net_pnl == pytest.approx(160 * -5.0)


def test_gap_exit_pays_adverse_cost():
    bars = {("A", 0, 3): (50.0, 50.0, 50.0, 50.0), ("A", 0, 4): (45.0, 46.0, 44.0, 45.0)}
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}), cost_bps=20.0)
    (e,) = r.ledger
    assert e.exit_price == 45.0  # reference price; cost is a cash deduction
    assert e.net_pnl == pytest.approx(e.qty * -5.0 - e.qty * 50.0 * 0.002 - e.qty * 45.0 * 0.002)


def test_same_bar_stop_and_trigger_is_flagged_and_takes_unfavorable_outcome():
    bars = {("A", 0, 3): (50.0, 51.5, 48.0, 50.0)}  # high reached trigger 51 and low hit the stop 48.5
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    assert r.ambiguous_bars == 1 and r.ledger[0].exit_price == pytest.approx(48.5)


def test_max_hold_exit_at_open():
    r = run(*build(["A"], sigs={("A", 0, 2): 1.0}), exit=ExitPackage(max_hold_minutes=3))
    (e,) = r.ledger
    assert (e.entry_minute, e.exit_minute, e.exit_reason) == (WARM * L + 3, WARM * L + 6, "max_hold")


def test_engine_matches_exit_scan_on_random_walks():
    rng = np.random.default_rng(7)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = L
        c = 50 + np.cumsum(rng.normal(0, 0.5, n))
        o = np.concatenate(([50.0], c[:-1])) + rng.normal(0, 0.1, n)
        h = np.maximum(o, c) + rng.uniform(0, 0.4, n)
        lo = np.minimum(o, c) - rng.uniform(0, 0.4, n)
        o[3], h[3], lo[3], c[3] = 50.0, max(50.0, h[3]), min(50.0, lo[3]), c[3]
        bars = {("A", 0, m): (o[m], h[m], lo[m], c[m]) for m in range(n)}
        bars[("A", 0, 3)] = (50.0, max(50.0, h[3], c[3]), min(50.0, lo[3], c[3]), c[3])
        r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
        o3 = bars[("A", 0, 3)]
        if not r.ledger:
            continue  # signal bar 2 close != 50, entry may cancel above the limit
        e = r.ledger[0]
        a = np.array([bars[("A", 0, m)] for m in range(n)])
        idx, ref, code, _ = exit_scan(a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.arange(n),
                                      e.entry_minute - WARM * L, e.entry_price, 1.0, 15)
        assert (e.exit_minute - WARM * L, e.exit_price) == (idx, pytest.approx(ref)), (seed, o3)


# ---- daily loss shutdown, early close, session exit ---------------------------------------

def test_daily_loss_halt_blocks_entries_liquidates_and_resets_next_session():
    cfg = dict(per_stock_pct=0.5, risk_pct=0.05)
    bars = {
        ("A", 0, 3): (50.0, 50.0, 50.0, 50.0),
        ("A", 0, 4): (50.0, 50.0, 48.9, 48.9),    # unrealized -1.1% of 100k -> halt after this bar
        ("A", 0, 5): (48.8, 48.9, 48.7, 48.8),    # liquidated at this open
        ("B", 0, 6): (50.0, 50.0, 50.0, 50.0),
    }
    sigs = {
        ("A", 0, 2): 2.0,
        ("B", 0, 4): 2.0,                          # same minute as the halt: must not be entered
        ("A", 1, 2): 2.0,                          # next session trades again
    }
    r = run(*build(["A", "B"], lengths=(L, L), bars=bars, sigs=sigs), **cfg)
    assert [(e.asset_id, e.exit_reason, e.exit_price) for e in r.ledger][0] == ("A", "halt", 48.8)
    assert r.ledger[0].qty == 1000
    assert r.daily_returns[0] == pytest.approx(1000 * (48.8 - 50.0) / 100_000)
    assert all(e.asset_id != "B" for e in r.ledger)
    assert len(r.ledger) == 2 and r.ledger[1].entry_minute > (WARM + 1) * L  # session 1 trades after the reset


def test_halt_includes_unrealized_but_not_before_threshold():
    bars = {("A", 0, 3): (50.0, 50.0, 50.0, 50.0), ("A", 0, 4): (50.0, 50.0, 49.2, 49.2)}  # -0.8%: no halt
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 2.0}), per_stock_pct=0.5, risk_pct=0.05)
    assert r.ledger[0].exit_reason == "session_end"


def test_session_exit_at_close_minus_five_and_early_close():
    r = run(*build(["A"], sigs={("A", 0, 2): 1.0}))
    assert r.ledger[0].exit_minute == WARM * L + 15 and r.ledger[0].exit_reason == "session_end"
    # early close: 12-minute session exits at offset 7; a signal at 6 would fill at 7 and is not submitted
    r = run(*build(["A"], lengths=(12,), sigs={("A", 0, 2): 1.0, }))
    assert r.ledger[0].exit_minute == WARM * L + 7
    r = run(*build(["A"], lengths=(12,), sigs={("A", 0, 6): 1.0}))
    assert r.ledger == () and r.rejections == ()


def test_zero_trade_days_still_return_zero():
    r = run(*build(["A"], lengths=(L, L, L)))
    assert r.daily_returns.shape == (3,) and not r.daily_returns.any()


def test_unexitable_position_closed_at_last_mark_and_flagged():
    nan = float("nan")
    bars = {("A", 0, m): (nan, nan, nan, nan) for m in range(4, L)}
    r = run(*build(["A"], bars=bars, sigs={("A", 0, 2): 1.0}))
    assert r.unresolved_positions == 1 and r.ledger[0].exit_reason == "unresolved"
    assert r.ledger[0].exit_price == 50.0


# ---- reference equity ------------------------------------------------------------------------

def test_reference_equity_makes_session_return_independent_of_other_sessions():
    win = {("A", 0, 3): (50.0, 53.0, 50.0, 53.0), ("A", 0, 4): (53.0, 53.0, 48.0, 48.0)}
    sigs0 = {("A", 0, 2): 1.0, ("A", 1, 2): 1.0}
    sigs1 = {("A", 0, 2): 1.0}
    big = dict(per_stock_pct=0.5, risk_pct=0.05)
    two = run(*build(["A"], lengths=(L, L), bars=win, sigs=sigs0), search_replay_reference_equity=100_000.0, **big)
    one = run(*build(["A"], lengths=(L,), sigs=sigs1), search_replay_reference_equity=100_000.0, **big)
    assert two.daily_returns[1] == one.daily_returns[0]
    carry = run(*build(["A"], lengths=(L, L), bars=win, sigs=sigs0), **big)
    assert carry.daily_returns[0] == two.daily_returns[0]
    assert carry.ledger[1].qty != two.ledger[1].qty or carry.daily_returns[1] != two.daily_returns[1]


def test_unknown_ranking_and_bad_mask_rejected():
    panel, mask = build(["A"])
    with pytest.raises(ValueError, match="ranking"):
        simulate_account(panel, mask, AccountConfig(ranking="nope"))
    with pytest.raises(ValueError, match="entry_score"):
        simulate_account(panel, mask, AccountConfig(entry_score_threshold=0.5))


def test_entry_score_threshold_filters_candidates():
    panel, mask = build(["A", "B"], sigs={("A", 0, 2): 1.0, ("B", 0, 2): 1.0})
    score = np.zeros(mask.signal.shape)
    score[1] = 1.0
    r = simulate_account(panel, mask, AccountConfig(cost_bps=0.0, entry_score_threshold=0.5),
                         warmup_sessions=WARM, entry_score=score)
    assert [e.asset_id for e in r.ledger] == ["B"]
