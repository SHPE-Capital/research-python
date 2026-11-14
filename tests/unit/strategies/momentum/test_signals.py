"""B5: signal masks vs an independent per-config reference, cache reuse, and timing."""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from shpe_research.features.momentum.breakout import breakout_signal
from shpe_research.features.momentum.clock_rvol import rvol_signal
from shpe_research.features.momentum.ema_trend import ema_trend_from_close
from shpe_research.features.momentum.readiness import readiness_masks
from shpe_research.features.momentum.vwap_band import session_vwap, vwap_band_signal
from shpe_research.features.momentum.wilder import adx_di, adx_signal, atr
from shpe_research.strategies.momentum.breakout.grid import Config, enumerate_configs
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade.account_types import MarketPanel

BARS = 78


def make_panel(n_sym: int, n_sess: int, seed: int = 0) -> MarketPanel:
    rng = np.random.default_rng(seed)
    t = n_sess * BARS
    ret = rng.normal(0.0003, 0.002, (n_sym, t))
    close = (100 * np.exp(np.cumsum(ret, axis=1))).astype(np.float32)
    spread = (np.abs(rng.normal(0, 0.002, (n_sym, t))) * close).astype(np.float32)
    high = close + spread
    low = close - spread
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    volume = rng.lognormal(8, 0.5, (n_sym, t)).astype(np.float32)
    miss = rng.random((n_sym, t)) < 0.002  # a few missing bars
    for a in (high, low, close, open_, volume):
        a[miss] = np.nan
    return MarketPanel(
        asset_ids=tuple(f"S{i}" for i in range(n_sym)),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        session_index=np.repeat(np.arange(n_sess), BARS),
        eligible=np.ones((n_sym, t), dtype=bool),
    )


def reference_mask(panel: MarketPanel, cfg: Config, rd: dict[str, np.ndarray]) -> np.ndarray:
    """Per-config recomputation of every indicator and signal from the batch functions.

    Only the readiness masks ``rd`` (a per-panel quantity, ~1.7 s to build) are passed in.
    """
    p, subset = cfg.as_dict(), set(cfg.subset)
    h, lo, c, v, s = panel.high, panel.low, panel.close, panel.volume, panel.session_index
    a14 = atr(h, lo, c, 14)
    n = int(p["breakout.n"])
    m = breakout_signal(c, h, a14, n, p["breakout.b"]) & rd[f"breakout_n{n}"]
    if "adx" in subset:
        a = int(p["adx.period"])
        m = m & adx_signal(*adx_di(h, lo, c, a), p["adx.q"]) & rd[f"adx_a{a}"]
    if "ema" in subset:
        f, sl = int(p["ema.fast"]), int(p["ema.slow"])
        m = m & ema_trend_from_close(c, a14, f, sl, p["ema.d"]) & rd[f"ema_f{f}_s{sl}"]
    if "rvol" in subset:
        w = int(p["rvol.w"])
        m = m & rvol_signal(v, s, w, p["rvol.v"]) & rd[f"rvol_w{w}"]
    if "vwap" in subset:
        vw = session_vwap(h, lo, c, v, s)
        m = m & vwap_band_signal(c, vw, a14, p["vwap.e"]) & rd["vwap"]
    return m & rd["atr14"]


@pytest.fixture(scope="module")
def panel() -> MarketPanel:
    return make_panel(30, 20)


def test_masks_match_reference_and_cache_reuses(panel: MarketPanel) -> None:
    sample = random.Random(1).sample(list(enumerate_configs()), 1000)
    cache = IndicatorCache(panel)
    rd = readiness_masks(panel.high, panel.low, panel.close, panel.volume, panel.session_index)
    total = 0
    for cfg in sample:
        m = signal_mask(panel, cfg, cache)
        assert m.signal.dtype == np.bool_
        assert m.atr is cache.atr14()
        np.testing.assert_array_equal(m.signal, reference_mask(panel, cfg, rd))
        total += int(m.signal.sum())
    assert total > 0, "synthetic panel should produce some signals"
    # Distinct computations are bounded by the small set of distinct settings.
    assert cache.hits["sig_breakout"] > 900
    assert cache.misses["sig_breakout"] <= 12
    assert cache.misses["atr"] == 1
    assert cache.misses["readiness"] == 1
    assert cache.misses["adx"] <= 3
    assert cache.misses["ema"] <= 6
    assert cache.misses["rvol"] <= 3
    assert cache.misses["vwap"] == 1
    assert cache.misses["prec_high"] <= 4
    # 1,000 configs x >= 2 components vs distinct arrays actually computed.
    assert cache.distinct_arrays * 10 < len(sample) * 2


def test_numeric_indicators_retained(panel: MarketPanel) -> None:
    cache = IndicatorCache(panel)
    cfg = next(c for c in enumerate_configs() if "ema" in c.subset)
    signal_mask(panel, cfg, cache)
    prm = cfg.as_dict()
    s = int(prm["ema.slow"])
    assert cache.indicator("ema", int(prm["ema.fast"])).shape == panel.close.shape
    assert cache.indicator("ema", s) is cache.ema(s)
    assert cache.indicator("atr").dtype == np.float32


def test_rejects_foreign_cache_and_missing_breakout(panel: MarketPanel) -> None:
    other = make_panel(3, 2)
    cfg = next(iter(enumerate_configs()))
    with pytest.raises(ValueError, match="different panel"):
        signal_mask(panel, cfg, IndicatorCache(other))
    bad = Config.make("v1", ["vwap"], {"vwap.e": 1})
    with pytest.raises(ValueError, match="breakout"):
        signal_mask(panel, bad)


def test_timing_report(capsys: pytest.CaptureFixture[str]) -> None:
    big = make_panel(300, 63, seed=3)
    cache = IndicatorCache(big)
    cfgs = random.Random(2).sample(list(enumerate_configs()), 200)
    t0 = time.perf_counter()
    for cfg in cfgs:
        signal_mask(big, cfg, cache)
    cold_total = time.perf_counter() - t0
    t1 = time.perf_counter()
    for cfg in cfgs:
        signal_mask(big, cfg, cache)
    warm = (time.perf_counter() - t1) / len(cfgs)
    with capsys.disabled():
        print(
            f"\n[B5 timing 300x63] warm per-config mask: {warm * 1e3:.2f} ms; cold total {cold_total:.2f}s "
            f"over {cache.distinct_arrays} distinct arrays "
            f"({cold_total / cache.distinct_arrays * 1e3:.1f} ms/array avg)"
        )
        for kind in sorted(cache.misses):
            print(f"   misses {kind}: {cache.misses[kind]}")
    assert warm < cold_total
