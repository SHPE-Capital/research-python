"""B5: entry signal mask for one breakout configuration.

mask = breakout AND each selected confirmation AND the readiness masks of the active components.
All arrays come from an ``IndicatorCache`` so configs sharing a setting share the computation.
"""

from __future__ import annotations

import numpy as np

from shpe_research.strategies.momentum.breakout.grid import Config
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.trade.account_types import MarketPanel, SignalMask


def signal_mask(panel: MarketPanel, config: Config, cache: IndicatorCache | None = None) -> SignalMask:
    """Entry mask (bool, symbol x minute) with ATR14 attached.

    Pass a shared ``cache`` (built for the same ``panel``) to reuse arrays across configs. The cache
    keeps the numeric indicator values (``cache.indicator(...)``) for logging.
    """
    if cache is None:
        cache = IndicatorCache(panel)
    elif cache.panel is not panel:
        raise ValueError("cache was built for a different panel")
    p = config.as_dict()
    subset = set(config.subset)
    if "breakout" not in subset:
        raise ValueError("config has no breakout component")
    ready = cache.readiness()
    n = int(p["breakout.n"])
    sig = cache.breakout(n, p["breakout.b"]) & ready["atr14"] & ready[f"breakout_n{n}"]
    if "adx" in subset:
        a = int(p["adx.period"])
        sig &= cache.adx_sig(a, p["adx.q"]) & ready[f"adx_a{a}"]
    if "ema" in subset:
        f, s = int(p["ema.fast"]), int(p["ema.slow"])
        sig &= cache.ema_sig(f, s, p["ema.d"]) & ready[f"ema_f{f}_s{s}"]
    if "rvol" in subset:
        w = int(p["rvol.w"])
        sig &= cache.rvol_sig(w, p["rvol.v"]) & ready[f"rvol_w{w}"]
    if "vwap" in subset:
        sig &= cache.vwap_sig(p["vwap.e"]) & ready["vwap"]
    return SignalMask(signal=np.asarray(sig, dtype=np.bool_), atr=cache.atr14())
