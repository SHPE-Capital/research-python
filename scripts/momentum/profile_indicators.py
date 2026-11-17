"""Profile the momentum indicator math (NumPy reference vs Numba kernels).

    uv run python scripts/momentum/profile_indicators.py --symbols 300 --sessions 5 --warmup 0
    uv run python scripts/momentum/profile_indicators.py --symbols 300 --sessions 5 --warmup 20 --mode both
    uv run python scripts/momentum/profile_indicators.py jit

``--warmup`` adds history sessions before the week (RVOL needs 20 prior sessions, so a bare 5-session panel
has no usable RVOL windows). ``--mode`` selects the path through ``SHPE_MOMENTUM_NUMBA`` (numpy | numba | both).
Each timing is the minimum over ``--repeats`` runs; the Numba path is warmed up first so compile time is
excluded (``jit`` measures it separately: cold cache vs on-disk cache, each in a fresh interpreter).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import SyntheticPanel, session_grid
from shpe_research.features.momentum import readiness
from shpe_research.features.momentum.breakout import preceding_high
from shpe_research.features.momentum.clock_rvol import rvol_ratio
from shpe_research.features.momentum.ema_trend import ema
from shpe_research.features.momentum.vwap_band import session_vwap
from shpe_research.features.momentum.wilder import ATR_LENGTH, adx_di, atr, seeded_recursive
from shpe_research.strategies.momentum.breakout.grid import enumerate_configs
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade.account_types import MarketPanel

ENV = "SHPE_MOMENTUM_NUMBA"
KERNEL_CASES = ("seeded_recursive (wilder 14)", "preceding_high(60)", "session_vwap", "rvol_ratio(w=10)")


def build_panel(n_symbols: int, sessions: int, seed: int = 0) -> MarketPanel:
    cal = get_calendar().sessions("2024-01-02", "2024-12-31")[:sessions]
    gen = SyntheticPanel(n_symbols, cal, seed)
    ts, local = session_grid(cal)
    a = gen.source()(cal, ts, local)
    return MarketPanel(
        tuple(gen.asset_ids), a["open"], a["high"], a["low"], a["close"], a["volume"], local, a["eligible"]
    )


def best(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def cases(p: MarketPanel, cfg) -> dict:
    return {
        "seeded_recursive (wilder 14)": lambda: seeded_recursive(p.close, 14, 1 / 14),
        "atr14": lambda: atr(p.high, p.low, p.close, ATR_LENGTH),
        "adx_di(14)": lambda: adx_di(p.high, p.low, p.close, 14),
        "ema(60)": lambda: ema(p.close, 60),
        "preceding_high(60)": lambda: preceding_high(p.high, 60),
        "session_vwap": lambda: session_vwap(p.high, p.low, p.close, p.volume, p.session_index),
        "rvol_ratio(w=10)": lambda: rvol_ratio(p.volume, p.session_index, 10),
        "readiness_masks": lambda: readiness.readiness_masks(p.high, p.low, p.close, p.volume, p.session_index),
        "cold IndicatorCache, 1 config": lambda: signal_mask(p, cfg, IndicatorCache(p)),
    }


def run_mode(mode: str, p: MarketPanel, cfg, repeats: int) -> dict[str, float]:
    os.environ[ENV] = "1" if mode == "numba" else "0"
    out = {}
    for name, fn in cases(p, cfg).items():
        fn()  # warm (JIT compile for numba, page cache for numpy)
        out[name] = best(fn, repeats)
    return out


def pick_config():
    full = {"adx", "breakout", "ema", "rvol"}
    return next(c for c in enumerate_configs() if full == set(c.subset))


def cmd_profile(a) -> None:
    p = build_panel(a.symbols, a.sessions + a.warmup)
    cfg = pick_config()
    print(f"N={a.symbols} bars={p.close.shape[1]} ({a.warmup}+{a.sessions} sessions) dtype={p.close.dtype}")
    print(f"config={cfg.as_dict()}")
    modes = ["numpy", "numba"] if a.mode == "both" else [a.mode]
    res = {m: run_mode(m, p, cfg, a.repeats) for m in modes}
    print(f"{'case':36s}" + "".join(f"{m + ' s':>12s}" for m in modes) + ("   speedup" if len(modes) == 2 else ""))
    for name in res[modes[0]]:
        row = f"{name:36s}" + "".join(f"{res[m][name]:12.4f}" for m in modes)
        if len(modes) == 2:
            row += f"   {res['numpy'][name] / res['numba'][name]:7.1f}x"
        print(row)


def cmd_jit(a) -> None:
    """Cold compile (cache wiped) vs on-disk cache, each in a fresh interpreter."""
    import shpe_research.features.momentum as pkg

    for f in Path(pkg.__file__).parent.glob("__pycache__/_kernels*.nb*"):
        f.unlink()
    for label in ("cold (cache wiped)", "with on-disk cache"):
        r = subprocess.run([sys.executable, __file__, "jit-child"], capture_output=True, text=True, check=True)
        print(f"{label}: {r.stdout.strip()}")


def cmd_jit_child(a) -> None:
    os.environ[ENV] = "1"
    p = build_panel(8, 25)
    cfg = pick_config()
    t0 = time.perf_counter()
    import shpe_research.features.momentum._kernels  # noqa: F401  (numba import)

    t1 = time.perf_counter()
    c = cases(p, cfg)
    for name in KERNEL_CASES:
        c[name]()
    t2 = time.perf_counter()
    print(f"numba import {t1 - t0:.2f}s, first call of the 4 kernel families {t2 - t1:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", default="profile", choices=["profile", "jit", "jit-child"])
    ap.add_argument("--symbols", type=int, default=300)
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--mode", choices=["numpy", "numba", "both"], default="both")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    {"profile": cmd_profile, "jit": cmd_jit, "jit-child": cmd_jit_child}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
