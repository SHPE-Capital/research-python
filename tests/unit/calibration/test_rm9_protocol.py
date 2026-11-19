"""Counterexamples for time boundaries, IPO membership, actual quote costs and conserved work."""

from __future__ import annotations

from dataclasses import replace

import httpx
import numpy as np
import pandas as pd
import pytest

from shpe_research.calibration import rm9_ops
from shpe_research.calibration.rm9_ops import resources, single_writer, workloads
from shpe_research.calibration.rm9_spec import spec
from shpe_research.calibration.rm9_study import weights
from shpe_research.data.alpaca import AlpacaDataClient, Credentials
from shpe_research.data.rm9_market import Market, checked_master
from shpe_research.data.rm9_universe import build_daily, ranked_pool
from shpe_research.trade.account_fast import prepare_panel, simulate_account_fast
from shpe_research.trade.account_types import AccountConfig, ExitPackage, MarketPanel, SignalMask


def test_market_selects_by_past_only_and_protects_identity(tmp_path):
    p = spec("demo")
    m = Market(tmp_path, p)
    master, _ = m.asset_master()
    data = build_daily(m, master)
    day = pd.Timestamp("2023-01-09")
    selected, _ = ranked_pool(day, data, p, master)
    assert {r["id"] for b in selected for r in b} == {f"demo-{i:02}" for i in range(9)}
    later, _ = ranked_pool(pd.Timestamp("2023-03-27"), data, p, master)
    assert "demo-09" in [r["id"] for b in later for r in b] and "demo-08" not in [r["id"] for b in later for r in b]
    for matrix in data[2].values():
        matrix[data[0] >= day] = 999999
    assert ranked_pool(day, data, p, master)[0] == selected
    bad = pd.concat([master, master.iloc[:1].assign(asset_id="distinct-owner")])
    with pytest.raises(ValueError, match="overlapping"):
        checked_master(bad)


def test_queried_period_is_raw_sip_paged_literal_alias(tmp_path):
    seen = []

    def reply(req):
        q = req.url.params
        assert q["asof"] == "-" and q["feed"] == "sip" and q["adjustment"] == "raw"
        j = int(q.get("page_token", 0))
        seen.append(j)
        return httpx.Response(
            200,
            json={
                "bars": [dict(t=f"2023-01-09T14:3{j}:00Z", o=10, h=11, l=9, c=10, v=450)],
                "next_page_token": "1" if j == 0 else None,
            },
        )

    m = Market(tmp_path, spec("pilot"))
    m.client = AlpacaDataClient(Credentials("fixture", "fixture"), transport=httpx.MockTransport(reply))
    f = m.frame("AAA", "2023-01-09", "2023-01-09", "minute")
    assert seen == [0, 1] and len(f) == 2
    m.client.close()
    offline = Market(tmp_path, spec("pilot"), offline=True)
    np.testing.assert_array_equal(f, offline.frame("AAA", "2023-01-09", "2023-01-09", "minute"))


def test_cash_quote_limit_and_loss_reconcile_hand_calculation():
    # Next open 100.10, markup permits 100.20; a wider modeled ask would cancel it.
    o = np.array([[100.0, 100.1, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]])
    c = o.copy()
    panel = MarketPanel(
        ("a",), o, o + 0.01, o - 0.01, c, np.full_like(o, 900000), np.zeros(8, int), np.ones(o.shape, bool)
    )
    sig = np.zeros(o.shape, bool)
    sig[0, 0] = True
    mask = SignalMask(sig, np.full_like(o, 5.0))
    prep = prepare_panel(panel)
    account = AccountConfig(
        cost_bps=0, liquidity_min_obs=1, exit=ExitPackage(max_hold_minutes=1, session_exit_minutes_before_close=1)
    )
    cube = np.zeros((9, 3, 2, 2))
    cube[..., 0] = 4.0
    cube[..., 1] = 10000.0
    q = (cube, np.array([0]), np.full(o.shape, 1e6, dtype="f4"), 0.0)
    audit = {}
    result = simulate_account_fast(prep, mask, account, quote_inputs=q, diagnostics=audit, return_ledger=True)
    lot = result.ledger[0]
    assert lot.entry_minute == 1 and lot.exit_minute == 2 and lot.qty == 33 and lot.exit_reason == "max_hold"
    assert lot.entry_price == pytest.approx(100.1 * 1.0004)
    assert lot.exit_price == pytest.approx(99 * 0.9996)
    assert audit["costs"] == pytest.approx([33 * (100.1 + 99) * 0.0004])
    assert result.daily_returns[0] * 100000 == pytest.approx(lot.net_pnl)
    cube[..., 0] = 25
    assert not simulate_account_fast(prep, mask, account, quote_inputs=q, return_ledger=True).ledger
    # Even a print below the old limit cannot fill if the executable modeled price exceeds it.
    with pytest.raises(ValueError, match="overlapping"):
        simulate_account_fast(prep, mask, replace(account, cost_bps=10), quote_inputs=q)


def test_early_close_gap_exit_and_missing_open():
    a = np.array([[50.0, 50.0, 45.0, 46.0, 46.0, 46.0]])
    panel = MarketPanel(
        ("a",), a, a + 0.05, a - 0.02, a, np.full(a.shape, 10000), np.zeros(6, int), np.ones(a.shape, bool)
    )
    sig = np.zeros(a.shape, bool)
    sig[0, 0] = True
    cfg = AccountConfig(cost_bps=0, exit=ExitPackage(session_exit_minutes_before_close=2))
    cube = np.zeros((9, 3, 2, 2))
    cube[..., 1] = 1000.0
    q = (cube, np.array([0]), np.full(a.shape, 1e5, dtype="f4"), 0.0)
    r = simulate_account_fast(panel, SignalMask(sig, np.full_like(a, 1.0)), cfg, quote_inputs=q, return_ledger=True)
    assert (r.ledger[0].exit_reason, r.ledger[0].exit_price) == ("stop", 45.0)
    a[0, 1] = np.nan
    r = simulate_account_fast(panel, SignalMask(sig, np.full_like(a, 1.0)), cfg, quote_inputs=q, return_ledger=True)
    assert not r.ledger and any(x.reason == "no_next_bar" for x in r.rejections)


def test_history_weights_cannot_see_the_outcome_and_cost_work_is_conserved():
    # The "regime" field is a pre-open descriptor calculated from prior days; forward P&L is never an input.
    plans = [
        {"days": [str(d.date())], "asof": str((d - pd.Timedelta(days=1)).date()), "regime": [i % 3, i]}
        for i, d in enumerate(pd.date_range("2021-01-01", periods=30))
    ]
    p = spec("demo")
    before = weights(p, plans, 20, 16)
    for r in plans[21:]:
        r["regime"] = [1e12, -1e12]
    for k, v in weights(p, plans, 20, 16).items():
        np.testing.assert_array_equal(before[k], v)
        assert np.isfinite(v).all() and np.isclose(v.sum(), 1)
    assert before["recency"][-1] > before["recency"][0]
    assert 1 / (before["similarity"] ** 2).sum() >= 0.60 * 16 - 1e-6
    work = workloads(12, 4, 3, 18, 2)
    assert work["initial_cells"] == 72
    assert work["trial_count"] == 26 and work["initial_cells"] + work["later_reference_cells"] == 286


def test_concurrent_writers_refused_and_release(tmp_path):
    path = tmp_path / "lock"
    with single_writer(path), pytest.raises(RuntimeError, match="another writer"), single_writer(path):
        pytest.fail("second writer entered")
    with single_writer(path):
        assert path.exists()


def test_resource_probe_passes_string_path_to_psutil(monkeypatch, tmp_path):
    """The Windows psutil backend rejects pathlib paths even though the POSIX backend accepts them."""
    seen = []
    original = rm9_ops.psutil.disk_usage

    def checked(path):
        seen.append(path)
        assert isinstance(path, str)
        return original(path)

    monkeypatch.setattr(rm9_ops.psutil, "disk_usage", checked)
    result = resources(tmp_path)
    assert seen == [str(tmp_path)]
    assert result["free_disk_bytes"] > 0
