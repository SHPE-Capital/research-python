"""C5: the Numba kernel against the pure-Python C4 oracle on random panels and configs.

Tolerance: market data are float32 (or float64) and promoted to float64 on load, accumulators are float64,
so the kernel does the same arithmetic as the oracle: ledger rows, rejections, counters and trade counts
must match exactly and daily returns to ``rtol=1e-12``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from shpe_research.trade.account import simulate_account
from shpe_research.trade.account_fast import prepare_panel, simulate_account_fast
from shpe_research.trade.account_types import AccountConfig, ExitPackage, FillModel, MarketPanel, SignalMask

WARM = 6
RTOL = 1e-12


def make_case(seed: int, dtype=np.float32) -> tuple[MarketPanel, SignalMask, AccountConfig, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_sym = int(rng.integers(3, 25))
    lens = rng.choice([30, 45, 60], size=WARM + int(rng.integers(3, 7)))
    n = int(lens.sum())
    shp = (n_sym, n)
    vol_mag = rng.choice([0.0005, 0.002, 0.006])
    ret = rng.normal(0.0, vol_mag, shp)
    close = (rng.uniform(5, 300, (n_sym, 1)) * np.exp(np.cumsum(ret, axis=1))).astype(dtype)
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1) * (1 + rng.normal(0, vol_mag / 2, shp)).astype(dtype)
    wick = np.abs(rng.normal(0, vol_mag, shp)).astype(dtype)
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = rng.lognormal(rng.uniform(6, 11), 0.6, shp).astype(dtype)
    miss = rng.random(shp) < rng.choice([0.0, 0.003, 0.02])
    for a in (open_, high, low, close):
        a[miss] = np.nan
    volume[miss] = 0
    sig = rng.random(shp) < rng.choice([0.002, 0.01, 0.05])
    atr = (close * rng.uniform(0.002, 0.02)).astype(dtype)
    atr[rng.random(shp) < 0.01] = np.nan
    elig = rng.random(shp) < 0.9
    panel = MarketPanel(
        tuple(f"S{i}" for i in rng.permutation(n_sym)), open_, high, low, close, volume,
        np.repeat(np.arange(len(lens)), lens), elig,
    )
    thr = float(rng.uniform(0.2, 0.7)) if rng.random() < 0.3 else None
    ref = 50_000.0 if rng.random() < 0.2 else None
    cfg = AccountConfig(
        max_positions=int(rng.integers(1, 8)),
        per_stock_pct=float(rng.uniform(0.03, 0.3)),
        risk_pct=float(rng.uniform(0.001, 0.01)),
        total_exposure_cap=float(rng.uniform(0.3, 1.0)),
        liquidity_cap_pct=float(rng.choice([0.0005, 0.01, 0.05])),
        liquidity_min_obs=int(rng.integers(3, 6)),
        liquidity_lookback=int(rng.choice([5, 20])),
        daily_halt_pct=float(rng.choice([0.002, 0.01, 0.05])),
        exit=ExitPackage(
            initial_stop_atr=float(rng.uniform(0.7, 3)), trail_activation_atr=float(rng.uniform(0.5, 2)),
            trail_distance_atr=float(rng.uniform(0.7, 3)), max_hold_minutes=int(rng.integers(3, 40)),
            session_exit_minutes_before_close=int(rng.integers(1, 6)),
        ),
        cost_bps=float(rng.choice([0.0, 5.0, 20.0])),
        fill=FillModel(limit_markup=float(rng.choice([0.0, 0.002, 0.01]))),
        entry_score_threshold=thr,
        size_multiplier=float(rng.uniform(0.5, 1.5)) if rng.random() < 0.3 else None,
        search_replay_reference_equity=ref,
    )
    score = rng.random(shp).astype(dtype)
    return panel, SignalMask(sig, atr), cfg, score


def assert_same(a, b, ledger: bool) -> None:
    np.testing.assert_allclose(a.daily_returns, b.daily_returns, rtol=RTOL, atol=1e-15)
    assert a.ambiguous_bars == b.ambiguous_bars
    assert a.unresolved_positions == b.unresolved_positions
    assert a.summary["trades"] == b.summary["trades"]
    assert a.summary["rejections"] == b.summary["rejections"]
    assert a.summary["active_sessions"] == b.summary["active_sessions"]
    for k in ("max_drawdown", "max_stock_share", "max_date_share"):
        assert a.summary[k] == pytest.approx(b.summary[k], rel=1e-9, abs=1e-12)
    if ledger:
        assert len(a.ledger) == len(b.ledger)
        for x, y in zip(a.ledger, b.ledger, strict=True):
            assert (x.asset_id, x.entry_minute, x.exit_minute, x.qty, x.exit_reason) == (
                y.asset_id, y.entry_minute, y.exit_minute, y.qty, y.exit_reason)
            assert x.entry_price == pytest.approx(y.entry_price, rel=1e-12)
            assert x.exit_price == pytest.approx(y.exit_price, rel=1e-12)
            assert x.net_pnl == pytest.approx(y.net_pnl, rel=1e-9, abs=1e-6)
        assert a.rejections == b.rejections


def test_parity_random_float32_panels() -> None:
    trades = rejects = 0
    reasons: set[str] = set()
    for seed in range(60):
        panel, mask, cfg, score = make_case(seed)
        kw = {"warmup_sessions": WARM, "entry_score": score}
        want = simulate_account(panel, mask, cfg, **kw)
        got = simulate_account_fast(panel, mask, cfg, return_ledger=True, **kw)
        assert_same(want, got, ledger=True)
        # counters-only path agrees with the full path
        lean = simulate_account_fast(prepare_panel(panel, cfg.ranking, cfg.liquidity_lookback), mask, cfg, **kw)
        assert lean.ledger == () and lean.rejections == ()
        assert_same(want, lean, ledger=False)
        trades += len(want.ledger)
        rejects += len(want.rejections)
        reasons |= {e.exit_reason for e in want.ledger}
    assert trades > 200 and rejects > 200, (trades, rejects)  # the random cases exercise the engine
    assert {"stop", "max_hold", "session_end"} <= reasons, reasons


def test_parity_float64_panels() -> None:
    for seed in range(100, 110):
        panel, mask, cfg, score = make_case(seed, np.float64)
        want = simulate_account(panel, mask, cfg, warmup_sessions=WARM, entry_score=score)
        got = simulate_account_fast(panel, mask, cfg, warmup_sessions=WARM, entry_score=score, return_ledger=True)
        assert_same(want, got, ledger=True)


def test_parity_includes_halt_and_unresolved_paths() -> None:
    seen_halt = seen_unres = False
    for seed in range(200, 260):
        panel, mask, cfg, score = make_case(seed)
        want = simulate_account(panel, mask, cfg, warmup_sessions=WARM, entry_score=score)
        got = simulate_account_fast(panel, mask, cfg, warmup_sessions=WARM, entry_score=score, return_ledger=True)
        assert_same(want, got, ledger=True)
        seen_halt |= any(e.exit_reason == "halt" for e in want.ledger)
        seen_unres |= want.unresolved_positions > 0
    assert seen_halt and seen_unres


def test_threads_share_one_prepared_panel() -> None:
    cases = [make_case(s) for s in range(300, 308)]
    panel = cases[0][0]
    prep = prepare_panel(panel)
    masks = [SignalMask(np.random.default_rng(i).random(panel.close.shape) < 0.02, cases[0][1].atr) for i in range(8)]
    cfg = AccountConfig(liquidity_min_obs=3)
    serial = [simulate_account_fast(prep, m, cfg, warmup_sessions=WARM).daily_returns for m in masks]
    with ThreadPoolExecutor(4) as ex:
        par = list(ex.map(lambda m: simulate_account_fast(prep, m, cfg, warmup_sessions=WARM).daily_returns, masks))
    for a, b in zip(serial, par, strict=True):
        np.testing.assert_array_equal(a, b)


def test_validation_errors_match_oracle() -> None:
    panel, mask, cfg, _ = make_case(1)
    bad_mask = SignalMask(mask.signal[:, :-1], mask.atr[:, :-1])
    with pytest.raises(ValueError, match="mask shape"):
        simulate_account_fast(panel, bad_mask, cfg)
    with pytest.raises(ValueError, match="entry_score"):
        simulate_account_fast(panel, mask, AccountConfig(entry_score_threshold=0.5))
    with pytest.raises(ValueError, match="warmup"):
        simulate_account_fast(panel, mask, AccountConfig(), warmup_sessions=999)
    prep = prepare_panel(panel, lookback=20)
    with pytest.raises(ValueError, match="lookback"):
        simulate_account_fast(prep, mask, AccountConfig(liquidity_lookback=5))
    with pytest.raises(ValueError, match="positive"):
        simulate_account_fast(panel, mask, AccountConfig(max_positions=0))
