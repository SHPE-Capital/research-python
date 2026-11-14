# ruff: noqa: B023, E501
"""C5: profile the pure-Python account oracle and time the Numba kernel.

    uv run python scripts/momentum/profile_account.py profile --symbols 300
    uv run python scripts/momentum/profile_account.py timing --symbols 300 500 1000 --threads 1 8 16
    uv run python scripts/momentum/profile_account.py jit

Cases are a synthetic panel (``data.panel.SyntheticPanel``) of ``warmup + sessions`` regular sessions (20
history sessions that produce no return, then 63 replayed) with signal masks from random breakout configs
via ``signal_mask``. Every timing is repeated and reported as min / median over repeats.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import SyntheticPanel, session_grid
from shpe_research.strategies.momentum.breakout.grid import enumerate_configs
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade import account_fast
from shpe_research.trade.account import simulate_account
from shpe_research.trade.account_fast import prepare_panel, simulate_account_fast
from shpe_research.trade.account_types import AccountConfig, MarketPanel, SignalMask

WARMUP = 20


def build_panel(n_symbols: int, sessions: int, seed: int = 0) -> MarketPanel:
    """Synthetic in-memory panel of ``WARMUP + sessions`` sessions (float32, symbol x minute)."""
    cal = get_calendar().sessions("2024-01-02", "2024-12-31")[: WARMUP + sessions]
    gen = SyntheticPanel(n_symbols, cal, seed)
    ts, local = session_grid(cal)
    a = gen.source()(cal, ts, local)
    return MarketPanel(
        tuple(gen.asset_ids), a["open"], a["high"], a["low"], a["close"], a["volume"], local, a["eligible"]
    )


def build_masks(panel: MarketPanel, n_cfg: int, seed: int = 1) -> list[SignalMask]:
    """Signal masks of ``n_cfg`` random grid configs (shared indicator cache)."""
    cfgs = random.Random(seed).sample(list(enumerate_configs()), n_cfg)
    cache = IndicatorCache(panel)
    return [signal_mask(panel, c, cache) for c in cfgs]


def sweep(fn, repeats: int) -> tuple[float, float]:
    """(min, median) wall seconds of ``fn()`` over ``repeats`` runs."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def cmd_profile(a) -> None:
    panel = build_panel(a.symbols, a.sessions)
    masks = build_masks(panel, a.configs)
    cfg = AccountConfig()
    sig = [int((m.signal & panel.eligible).sum()) for m in masks]
    print(f"N={a.symbols} sessions={a.sessions} candidates={a.configs} mean signals/candidate={np.mean(sig):.0f}")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for m in masks:
        simulate_account(panel, m, cfg, warmup_sessions=WARMUP)
    pr.disable()
    print(f"oracle total {time.perf_counter() - t0:.2f}s (cProfile overhead included)")
    pstats.Stats(pr).sort_stats("tottime").print_stats(12)


def cmd_timing(a) -> None:
    cfg = AccountConfig()
    for n in a.symbols:
        panel = build_panel(n, a.sessions)
        masks = build_masks(panel, a.configs)
        prep = prepare_panel(panel)
        simulate_account_fast(prep, masks[0], cfg, warmup_sessions=WARMUP)  # warm JIT
        sigs = np.mean([int((m.signal & panel.eligible).sum()) for m in masks])
        trades = np.mean([simulate_account_fast(prep, m, cfg, warmup_sessions=WARMUP).summary["trades"] for m in masks])
        k = len(masks)
        fmin, fmed = sweep(lambda: [simulate_account_fast(prep, m, cfg, warmup_sessions=WARMUP) for m in masks], a.repeats)
        omin, omed = sweep(lambda: [simulate_account(panel, m, cfg, warmup_sessions=WARMUP) for m in masks[: a.oracle]], a.repeats)
        ok = a.oracle
        print(f"N={n}: signals/cand={sigs:.0f} trades/cand={trades:.0f}")
        print(f"  fast 1 thread  per candidate  min {fmin / k:.4f}s  median {fmed / k:.4f}s")
        print(f"  oracle         per candidate  min {omin / ok:.3f}s  median {omed / ok:.3f}s  "
              f"(speedup median {omed / ok / (fmed / k):.0f}x)")
        work = masks * (a.threads_work // k)
        base = None
        for th in a.threads:
            with ThreadPoolExecutor(th) as ex:
                wmin, wmed = sweep(
                    lambda: list(ex.map(lambda m: simulate_account_fast(prep, m, cfg, warmup_sessions=WARMUP), work)),
                    a.repeats,
                )
            base = base or wmed
            print(f"  threads={th:2d}: {len(work)} candidates min {wmin:.3f}s median {wmed:.3f}s "
                  f"-> {wmed / len(work):.4f}s/cand, speedup vs 1 thread {base / wmed:.2f}x")


def cmd_jit(a) -> None:
    """Cold compile (cache wiped), then cached load, each in a fresh interpreter."""
    cache_dir = Path(account_fast.__file__).parent / "__pycache__"
    for f in cache_dir.glob("account_fast*.nb*"):
        f.unlink()
    for label in ("cold (cache wiped)", "with on-disk cache"):
        r = subprocess.run([sys.executable, __file__, "jit-child"], capture_output=True, text=True, check=True)
        print(f"{label}: {r.stdout.strip()}")


def cmd_jit_child(a) -> None:
    panel = build_panel(20, 3)
    cfg = AccountConfig()
    mask = SignalMask(np.zeros(panel.close.shape, bool), np.asarray(panel.close))
    t0 = time.perf_counter()
    simulate_account_fast(panel, mask, cfg, warmup_sessions=1, return_ledger=False)
    t1 = time.perf_counter()
    simulate_account_fast(panel, mask, cfg, warmup_sessions=1, return_ledger=True)  # other flag, same signature
    t2 = time.perf_counter()
    print(f"first call {t1 - t0:.2f}s, second call (ledger on) {t2 - t1:.3f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["profile", "timing", "jit", "jit-child"])
    ap.add_argument("--symbols", type=int, nargs="+", default=[300])
    ap.add_argument("--sessions", type=int, default=63)
    ap.add_argument("--configs", type=int, default=8, help="distinct candidates (signal masks)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--oracle", type=int, default=2, help="candidates timed on the pure-Python oracle")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--threads-work", type=int, default=64, help="candidates per thread-scaling run")
    a = ap.parse_args()
    if a.cmd == "profile":
        a.symbols = a.symbols[0]
        cmd_profile(a)
    else:
        {"timing": cmd_timing, "jit": cmd_jit, "jit-child": cmd_jit_child}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
