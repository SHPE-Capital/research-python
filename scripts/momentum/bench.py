"""E1 benchmark harness and compute projection (momentum plan, task E1) on the REAL pieces.

Uses the real account engine (`trade.account.simulate_account`, or the Numba `trade.account_fast` kernel with
`--kernel fast`, imported lazily), the B5 signal masks / `IndicatorCache`, the A6 chunked panel
(`data.panel`) and the 0-B config grid. Panels are synthetic (`generate_synthetic_panel`) unless `--panel-dir`
points at a real chunk root for steps 3-4.

Unit of work: one *candidate-week* = one config replayed through the account over one week chunk plus a
20-session history prefix (the D6 week-chunk unit; sessions restart from a fixed reference equity, so chunks are
independent and cacheable). A *candidate-window* = the same config over every week chunk of the history window.

Steps: 1 one stock, determinism; 2 30 stocks x 20 sessions load + indicator profile; 3 calibration cell
(early + late window, cold/warm); 4 consecutive weekly cycles with a (config, chunk) result cache; plus the scale
matrix (N x years x threads) and the full-run projection with and without the week-chunk cache.

Run natively (not in a CPU-capped container) with nothing else on the machine. `--smoke` finishes in about a
minute; it is for tests, its numbers mean nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import generate_synthetic_panel, list_chunks, open_panel_chunk, read_meta
from shpe_research.strategies.momentum.breakout.grid import Config, config_id, enumerate_configs
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade.account_types import AccountConfig, MarketPanel

OUT_DIR = Path("studies/momentum/rm9-weekly-calibration/bench")
WEEKEND_TARGET_HOURS = 24.0
RERUN_RESERVE = 2.0
WARMUP_SESSIONS = 20
REFERENCE_EQUITY = 100_000.0
FIRST_MONDAY = pd.Timestamp("2020-01-06")
MINUTES_PER_SESSION = 390  # (prefix weeks = ceil(warmup sessions / 5), see ``prefix_weeks``)
SESSIONS_PER_WEEK = 5

# Rough byte costs per (symbol x minute) cell, used ONLY to decide whether a cell fits the budget.
EST_PANEL_B = 21  # 5 float32 fields + bool eligible
EST_CACHE_B = 60  # indicator + signal arrays held by IndicatorCache for one week
EST_WORKER_B = 110  # float64 copies, transposed mask and ranking inside the account loop
EST_BASE_BYTES = 500e6

DEFAULT_PARAMS: dict[str, Any] = {
    "weeks": 52,  # walk-forward cutoffs in the full run
    "policies": 3,
    "seeds": 3,
    "budget": 2048,
    "carry_over": 0.5,  # fraction of a cutoff's candidates already evaluated at the previous cutoff
    "policies_share_replays": True,  # weighting policies reuse the same replayed paths (D6)
}


# ------------------------------------------------------------------------------------------ machine / memory

def machine_profile(scratch: Path | None = None) -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(scratch or Path.cwd())
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cpu": platform.processor(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "scratch_free_gb": round(disk.free / 1e9, 1),
    }


def _rss_bytes(proc: psutil.Process, children: bool) -> tuple[int, int]:
    """(own rss, summed children rss); tolerant of processes that vanish mid-sample."""
    own = proc.memory_info().rss
    kids = 0
    if children:
        for c in proc.children(recursive=True):
            try:
                kids += c.memory_info().rss
            except psutil.Error:
                pass
    return own, kids


class RssMonitor:
    """Samples RSS (parent and children) on a thread; the peak is read after ``stop``."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.baseline = 0
        self.peak_total = 0
        self.peak_children = 0

    def _sample(self) -> None:
        own, kids = _rss_bytes(self._proc, True)
        self.peak_total = max(self.peak_total, own + kids)
        self.peak_children = max(self.peak_children, kids)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> RssMonitor:
        self.baseline = self._proc.memory_info().rss
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def rss_report(mon: RssMonitor, workers: int, pool: str) -> dict:
    """Peak RAM total, and per worker (children for processes; increment over baseline / workers for threads)."""
    if pool == "process":
        per = mon.peak_children / max(workers, 1)
    else:
        per = max(mon.peak_total - mon.baseline, 0) / max(workers, 1)
    return {
        "peak_total_mb": round(mon.peak_total / 1e6, 1),
        "baseline_mb": round(mon.baseline / 1e6, 1),
        "peak_per_worker_mb": round(per / 1e6, 1),
    }


# ------------------------------------------------------------------------------------------ kernel selection

@dataclass(frozen=True)
class Kernel:
    name: str
    fn: Callable[..., Any]
    releases_gil: bool


def resolve_kernel(name: str) -> Kernel:
    """``python``: the pure-Python oracle. ``fast``: ``trade.account_fast``, imported lazily (needs w3-fast)."""
    if name == "python":
        from shpe_research.trade.account import simulate_account

        return Kernel("python", simulate_account, False)
    if name == "fast":
        try:
            import importlib

            mod = importlib.import_module("shpe_research.trade.account_fast")
        except ImportError as e:
            raise SystemExit(f"--kernel fast needs shpe_research.trade.account_fast (not merged yet): {e}") from e
        fn = getattr(mod, "simulate_account_fast", None) or mod.simulate_account
        return Kernel("fast", fn, bool(getattr(mod, "RELEASES_GIL", True)))
    raise ValueError(f"unknown kernel {name!r}")


def choose_pool(kernel: Kernel, workers: int, override: str) -> str:
    if workers <= 1:
        return "serial"
    if override != "auto":
        return override
    return "thread" if kernel.releases_gil else "process"


# ------------------------------------------------------------------------------------------ panel helpers

def prefix_weeks(warmup: int) -> int:
    return max(1, math.ceil(warmup / SESSIONS_PER_WEEK))


def make_panel(root: Path, n_symbols: int, n_weeks: int, seed: int = 0) -> tuple[list[Path], float]:
    """Synthetic week chunks (``n_weeks`` ISO weeks from a fixed Monday); returns (paths, build seconds)."""
    end = FIRST_MONDAY + pd.Timedelta(weeks=n_weeks) - pd.Timedelta(days=1)
    sessions = get_calendar().sessions(FIRST_MONDAY, end)
    t = time.perf_counter()
    paths = generate_synthetic_panel(root, n_symbols, sessions, seed=seed)
    return paths, time.perf_counter() - t


def dir_bytes(paths: Sequence[Path]) -> int:
    return sum(f.stat().st_size for p in paths for f in Path(p).iterdir())


def build_week_panel(paths: Sequence[Path], k: int, warmup: int = WARMUP_SESSIONS) -> tuple[MarketPanel, int]:
    """Chunk ``k`` with up to ``warmup`` sessions of earlier chunks prepended. Returns (panel, warmup sessions)."""
    cur = open_panel_chunk(paths[k])
    cur_first = int(np.asarray(cur.session_index)[0])
    pieces: list[MarketPanel] = []
    have = 0
    j = k - 1
    while have < warmup and j >= 0:
        pieces.append(open_panel_chunk(paths[j]))
        have += len(np.unique(np.asarray(pieces[-1].session_index)))
        j -= 1
    if not pieces:
        return cur, 0
    pieces.reverse()
    parts = [*pieces, cur]
    sidx = np.concatenate([np.asarray(p.session_index) for p in parts])
    first_keep = max(cur_first - warmup, int(sidx[0]))
    lo = int(np.searchsorted(sidx, first_keep, side="left"))
    cat = {f: np.concatenate([np.asarray(getattr(p, f)) for p in parts], axis=1)[:, lo:]
           for f in ("open", "high", "low", "close", "volume", "eligible")}
    panel = MarketPanel(asset_ids=cur.asset_ids, session_index=sidx[lo:], **cat)
    return panel, cur_first - first_keep


_ALL_CONFIGS: list[Config] | None = None


def all_configs() -> list[Config]:
    global _ALL_CONFIGS
    if _ALL_CONFIGS is None:
        _ALL_CONFIGS = list(enumerate_configs())
    return _ALL_CONFIGS


def sample_configs(n: int, seed: int) -> list[Config]:
    allc = all_configs()
    if n > len(allc):
        raise ValueError(f"asked for {n} configs, grid has {len(allc)}")
    idx = np.random.default_rng(seed).choice(len(allc), size=n, replace=False)
    return [allc[int(i)] for i in idx]


def account_cfg() -> AccountConfig:
    return AccountConfig(search_replay_reference_equity=REFERENCE_EQUITY)


# ------------------------------------------------------------------------------------------ replay workers

class WeekStore:
    """Per-process store of the most recent week panels and their IndicatorCaches (thread-safe build)."""

    def __init__(self, paths: Sequence[str], warmup: int = WARMUP_SESSIONS, keep: int = 2) -> None:
        self.paths = [Path(p) for p in paths]
        self.warmup = warmup
        self.keep = keep
        self.lock = threading.Lock()
        self.items: dict[int, tuple[MarketPanel, int, IndicatorCache]] = {}
        self.build_s = 0.0
        self.hits: Counter[str] = Counter()  # from evicted caches
        self.misses: Counter[str] = Counter()

    def get(self, k: int) -> tuple[MarketPanel, int, IndicatorCache]:
        with self.lock:
            if k not in self.items:
                t = time.perf_counter()
                panel, warm = build_week_panel(self.paths, k, self.warmup)
                self.build_s += time.perf_counter() - t
                self.items[k] = (panel, warm, IndicatorCache(panel))
                while len(self.items) > self.keep:
                    old = self.items.pop(next(iter(self.items)))[2]
                    self.hits.update(old.hits)
                    self.misses.update(old.misses)
            return self.items[k]

    def totals(self) -> tuple[int, int]:
        h, m = Counter(self.hits), Counter(self.misses)
        for _, _, c in self.items.values():
            h.update(c.hits)
            m.update(c.misses)
        return sum(h.values()), sum(m.values())

    def hits_by_kind(self) -> dict[str, tuple[int, int]]:
        h, m = Counter(self.hits), Counter(self.misses)
        for _, _, c in self.items.values():
            h.update(c.hits)
            m.update(c.misses)
        return {k: (h[k], m[k]) for k in sorted(set(h) | set(m))}


_CTX: dict[str, Any] = {}


def _init_worker(paths: list[str], kernel_name: str, warmup: int) -> None:
    _CTX["store"] = WeekStore(paths, warmup)
    _CTX["kernel"] = resolve_kernel(kernel_name)
    _CTX["cfg"] = account_cfg()


def replay_task(week: int, configs: tuple[Config, ...], ctx: dict[str, Any] | None = None) -> dict:
    """Replay ``configs`` on week chunk ``week``; returns daily returns, trade counts and timing."""
    c = ctx if ctx is not None else _CTX
    store: WeekStore = c["store"]
    panel, warm, cache = store.get(week)
    h0, m0 = sum(cache.hits.values()), sum(cache.misses.values())
    t_mask = t_sim = 0.0
    out = []
    for cfg in configs:
        t = time.perf_counter()
        mask = signal_mask(panel, cfg, cache)
        t_mask += time.perf_counter() - t
        t = time.perf_counter()
        res = c["kernel"].fn(panel, mask, c["cfg"], warmup_sessions=warm)
        t_sim += time.perf_counter() - t
        out.append((np.asarray(res.daily_returns, dtype=np.float64), len(res.ledger)))
    return {
        "week": week, "results": out, "mask_s": t_mask, "sim_s": t_sim,
        "hits": sum(cache.hits.values()) - h0, "misses": sum(cache.misses.values()) - m0,
        "pid_rss": psutil.Process().memory_info().rss,
    }


class Runner:
    """Runs (week, configs) tasks serially, on threads sharing one WeekStore, or on processes (memmapped panel)."""

    def __init__(self, paths: Sequence[Path], kernel: Kernel, workers: int, pool: str,
                 warmup: int = WARMUP_SESSIONS) -> None:
        self.paths = [str(p) for p in paths]
        self.kernel = kernel
        self.workers = max(workers, 1)
        self.pool = pool
        self.ctx: dict[str, Any] = {}
        self.ex: Any = None
        if pool in ("serial", "thread"):
            self.ctx = {"store": WeekStore(self.paths, warmup), "kernel": kernel, "cfg": account_cfg()}
        if pool == "thread":
            self.ex = ThreadPoolExecutor(self.workers)
        elif pool == "process":
            self.ex = ProcessPoolExecutor(self.workers, initializer=_init_worker,
                                          initargs=(self.paths, kernel.name, warmup))

    def close(self) -> None:
        if self.ex is not None:
            self.ex.shutdown()

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def slices(self, weeks: Sequence[int], configs: Sequence[Config]) -> list[tuple[int, tuple[Config, ...]]]:
        step = max(1, math.ceil(len(configs) / self.workers))
        return [(w, tuple(configs[i:i + step])) for w in weeks for i in range(0, len(configs), step)]

    def run(self, tasks: list[tuple[int, tuple[Config, ...]]]) -> tuple[list[dict], float]:
        t = time.perf_counter()
        if self.pool == "serial":
            outs = [replay_task(w, cs, self.ctx) for w, cs in tasks]
        elif self.pool == "thread":
            outs = list(self.ex.map(lambda a: replay_task(a[0], a[1], self.ctx), tasks))
        else:
            outs = list(self.ex.map(replay_task, [a[0] for a in tasks], [a[1] for a in tasks]))
        return outs, time.perf_counter() - t


def pass_stats(outs: list[dict], wall: float, n_cands: int, n_weeks: int) -> dict:
    hits = sum(o["hits"] for o in outs)
    misses = sum(o["misses"] for o in outs)
    h = hashlib.sha256()
    for o in sorted(outs, key=lambda o: o["week"]):
        for dr, nt in o["results"]:
            h.update(dr.tobytes())
            h.update(np.int64(nt).tobytes())
    cw = max(n_cands * n_weeks, 1)
    return {
        "candidates": n_cands, "weeks": n_weeks, "wall_s": round(wall, 4),
        "s_per_candidate_week": wall / cw,
        "mask_cpu_s": round(sum(o["mask_s"] for o in outs), 4),
        "sim_cpu_s": round(sum(o["sim_s"] for o in outs), 4),
        "cache_hit_rate": round(hits / (hits + misses), 4) if hits + misses else 0.0,
        "trades": int(sum(nt for o in outs for _, nt in o["results"])),
        "digest": h.hexdigest()[:16],
    }


def run_pass(runner: Runner, weeks: Sequence[int], configs: Sequence[Config]) -> dict:
    outs, wall = runner.run(runner.slices(weeks, configs))
    return pass_stats(outs, wall, len(configs), len(weeks))


# ------------------------------------------------------------------------------------------ budget gate

def estimate_cell(n: int, years: float, workers: int, pool: str) -> dict:
    sessions = years * 252 + WARMUP_SESSIONS
    minutes = sessions * MINUTES_PER_SESSION
    disk = n * minutes * EST_PANEL_B + minutes * 12
    week_cells = n * (WARMUP_SESSIONS + SESSIONS_PER_WEEK) * MINUTES_PER_SESSION
    if pool == "process":
        ram = EST_BASE_BYTES * (1 + workers) + workers * week_cells * (EST_PANEL_B + EST_CACHE_B + EST_WORKER_B)
    else:
        ram = EST_BASE_BYTES + week_cells * (2 * (EST_PANEL_B + EST_CACHE_B) + workers * EST_WORKER_B)
    return {"est_disk_bytes": int(disk), "est_ram_bytes": int(ram),
            "est_panel_bytes_total": int(n * minutes * EST_PANEL_B)}


def skip_reason(est: dict, ram_gb: float, disk_gb: float, free_disk: float) -> str | None:
    if est["est_ram_bytes"] > ram_gb * 1e9:
        return f"estimated RAM {est['est_ram_bytes'] / 1e9:.1f} GB exceeds budget {ram_gb} GB"
    if est["est_disk_bytes"] > disk_gb * 1e9:
        return f"estimated disk {est['est_disk_bytes'] / 1e9:.1f} GB exceeds budget {disk_gb} GB"
    if est["est_disk_bytes"] > free_disk:
        return f"estimated disk {est['est_disk_bytes'] / 1e9:.1f} GB exceeds free space {free_disk / 1e9:.1f} GB"
    return None


# ------------------------------------------------------------------------------------------ projection

def project(per_cw_s: float, *, window_weeks: int, params: dict[str, Any]) -> dict:
    """Full-run projection from seconds per candidate-week (wall, at the measured thread count).

    Without the cache every (cutoff, policy, seed, budget candidate) replays the whole window. With the D6
    week-chunk cache a (config, chunk) is replayed once: the first cutoff replays whole windows, later cutoffs
    replay one new chunk for carried-over candidates and a whole window for new ones; policies share replays
    when ``policies_share_replays``.
    """
    K, P, S, B = int(params["weeks"]), int(params["policies"]), int(params["seeds"]), int(params["budget"])
    carry = float(params["carry_over"])
    W = window_weeks
    cands = S * B
    pol_cached = 1 if params["policies_share_replays"] else P
    nocache_cw = K * P * cands * W
    cached_first = pol_cached * cands * W
    cached_step = pol_cached * cands * (1 + (1 - carry) * (W - 1))
    cached_cw = cached_first + (K - 1) * cached_step

    def hours(cw: float) -> float:
        return cw * per_cw_s * RERUN_RESERVE / 3600

    weekend_nocache = P * cands * W * per_cw_s / 3600
    weekend_cached = cached_step * per_cw_s / 3600
    return {
        "per_candidate_week_s": per_cw_s, "window_weeks": W,
        "per_candidate_window_s": per_cw_s * W,
        "with_cache": {"candidate_weeks": cached_cw, "full_run_hours_2x_reserve": round(hours(cached_cw), 2),
                       "weekend_job_hours": round(weekend_cached, 3),
                       "weekend_vs_24h": round(weekend_cached / WEEKEND_TARGET_HOURS, 4)},
        "without_cache": {"candidate_weeks": nocache_cw, "full_run_hours_2x_reserve": round(hours(nocache_cw), 2),
                          "weekend_job_hours": round(weekend_nocache, 3),
                          "weekend_vs_24h": round(weekend_nocache / WEEKEND_TARGET_HOURS, 4)},
        "assumptions": {**params, "rerun_reserve": RERUN_RESERVE, "weekend_target_hours": WEEKEND_TARGET_HOURS},
    }


# ------------------------------------------------------------------------------------------ steps

def step1(root: Path, kernel: Kernel, seed: int, warmup: int, n_cands: int = 8) -> dict:
    pw = prefix_weeks(warmup)
    paths, _ = make_panel(root / "s1", 1, pw + 1, seed)
    k = len(paths) - 1
    cfgs = sample_configs(n_cands, seed)
    outs = []
    for _ in range(2):  # second run starts from fresh caches
        with Runner(paths, kernel, 1, "serial", warmup) as r:
            o, _ = r.run([(k, tuple(cfgs))])
        outs.append(o[0]["results"])
    same = all(a[0].tobytes() == b[0].tobytes() and a[1] == b[1] for a, b in zip(*outs, strict=True))
    if not same:
        raise AssertionError("account replay is not byte-identical on rerun")
    return {"name": "1 one stock, few sessions (correctness)", "candidates": len(cfgs), "deterministic": True,
            "trades": int(sum(nt for _, nt in outs[0])), "sessions_returned": int(len(outs[0][0][0]))}


def step2(root: Path, kernel: Kernel, seed: int, n_cands: int, warmup: int, n_session_weeks: int) -> dict:
    n, pw = 30, prefix_weeks(warmup)
    n_weeks = pw + n_session_weeks  # default 4 measured weeks = 20 sessions
    paths, build_s = make_panel(root / "s2", n, n_weeks, seed)
    nbytes = dir_bytes(paths)
    t = time.perf_counter()
    for p in paths:  # full (non-mmap) read of every array = raw npy read throughput (OS page cache is warm)
        for f in Path(p).glob("*.npy"):
            np.load(f)
    read_s = time.perf_counter() - t
    t = time.perf_counter()
    for p in paths:
        open_panel_chunk(p)
    mmap_open_s = time.perf_counter() - t
    cfgs = sample_configs(n_cands, seed)
    weeks = list(range(pw, n_weeks))
    with Runner(paths, kernel, 1, "serial", warmup) as r:
        store: WeekStore = r.ctx["store"]
        store.keep = n_weeks  # keep every week's cache so the warm pass hits
        cold = run_pass(r, weeks, cfgs)
        cold_kinds = {k: list(v) for k, v in store.hits_by_kind().items()}
        warm = run_pass(r, weeks, cfgs)
        warm_kinds = {k: list(v) for k, v in store.hits_by_kind().items()}
        build_week_s = store.build_s
    return {
        "name": "2 30 stocks x 20 sessions (load + indicator profile)", "symbols": n, "sessions": len(weeks) * 5,
        "panel_build_s": round(build_s, 3), "disk_mb": round(nbytes / 1e6, 2),
        "npy_read_mb_s": round(nbytes / 1e6 / read_s, 1) if read_s else None,
        "mmap_open_s": round(mmap_open_s, 4), "week_panel_with_prefix_build_s": round(build_week_s, 3),
        "indicator_hits_misses_cold_pass": cold_kinds, "indicator_hits_misses_after_warm_pass": warm_kinds,
        "replay_cold": cold, "replay_warm": warm,
        "deterministic_cold_vs_warm": cold["digest"] == warm["digest"],
    }


def step3(paths: Sequence[Path], kernel: Kernel, args: argparse.Namespace, n_symbols: int) -> dict:
    """Early and late history windows of ``window_weeks``, ``calib_candidates`` configs, cold then warm cache."""
    n_chunks = len(paths)
    W = args.window_weeks
    cfgs = sample_configs(args.calib_candidates, args.seed)
    pw = prefix_weeks(args.warmup)
    windows = {"early": list(range(pw, pw + W)), "late": list(range(n_chunks - W, n_chunks))}
    pool = choose_pool(kernel, args.calib_workers, args.pool)
    cells: dict[str, Any] = {}
    for label, weeks in windows.items():
        with Runner(paths, kernel, args.calib_workers, pool, args.warmup) as r, RssMonitor() as mon:
            cold = run_pass(r, weeks, cfgs)
            warm = run_pass(r, weeks, cfgs)
        cells[label] = {"weeks": [weeks[0], weeks[-1]], "cold": cold, "warm": warm,
                        "s_per_candidate_window_cold": cold["s_per_candidate_week"] * W,
                        "s_per_candidate_window_warm": warm["s_per_candidate_week"] * W,
                        "deterministic_cold_vs_warm": cold["digest"] == warm["digest"] or pool == "process",
                        "rss": rss_report(mon, args.calib_workers, pool)}
    return {"name": f"3 calibration cell N={n_symbols}, {W}-week windows, {len(cfgs)} candidates", "pool": pool,
            "workers": args.calib_workers, "symbols": n_symbols, "windows": cells}


def step4(paths: Sequence[Path], kernel: Kernel, args: argparse.Namespace) -> dict:
    """Consecutive weekly cycles with a (config id, chunk) result cache; reports new chunks computed per week."""
    W = args.window_weeks_step4
    cfgs = sample_configs(args.cycle_candidates, args.seed + 1)
    cache: dict[tuple[str, int], np.ndarray] = {}
    pool = choose_pool(kernel, args.calib_workers, args.pool)
    cycles = []
    pw = prefix_weeks(args.warmup)
    with Runner(paths, kernel, args.calib_workers, pool, args.warmup) as r:
        for c in range(args.cycles):
            weeks = list(range(pw + c, pw + c + W))
            need = [(w, cf) for w in weeks for cf in cfgs if (config_id(cf), w) not in cache]
            by_week: dict[int, list[Config]] = {}
            for w, cf in need:
                by_week.setdefault(w, []).append(cf)
            wall = 0.0
            for w, cs in by_week.items():
                tasks = r.slices([w], cs)
                outs, dt = r.run(tasks)
                wall += dt
                for (_, tcs), o in zip(tasks, outs, strict=True):
                    for cf, (dr, _) in zip(tcs, o["results"], strict=True):
                        cache[(config_id(cf), w)] = dr
            total = len(weeks) * len(cfgs)
            cycles.append({"cycle": c, "window_weeks": [weeks[0], weeks[-1]], "chunks_new": len(need),
                           "chunks_new_per_candidate": round(len(need) / len(cfgs), 3),
                           "chunks_reused": total - len(need), "reuse_fraction": round(1 - len(need) / total, 4),
                           "wall_s": round(wall, 3)})
    return {"name": "4 consecutive weekly cycles (week-chunk cache)", "pool": pool, "candidates": len(cfgs),
            "window_weeks": W, "cycles": cycles}


def matrix_cell(n: int, years: float, args: argparse.Namespace, kernel: Kernel, root: Path) -> dict:
    n_window = max(round(years * 52), 1)
    pw = prefix_weeks(args.warmup)
    n_weeks_total = n_window + pw
    cell: dict[str, Any] = {"n": n, "years": years, "window_weeks": n_window}
    free = shutil.disk_usage(root).free
    # panel disk is shared by all thread counts; gate each thread count on RAM, the cell on disk
    est0 = estimate_cell(n, years, 1, "serial")
    why = skip_reason({**est0, "est_ram_bytes": 0}, args.ram_budget_gb, args.disk_budget_gb, free)
    cell["estimate"] = est0
    if why:
        cell["skipped"] = why
        return cell
    paths, build_s = make_panel(root / f"m{n}_{years}", n, n_weeks_total, args.seed)
    cell["build_s"] = round(build_s, 2)
    cell["disk_bytes"] = dir_bytes(paths)
    cell["panel_bytes"] = cell["disk_bytes"]  # chunks are plain npy: in-RAM size equals on-disk size
    cell["week_chunk_bytes"] = cell["disk_bytes"] // n_weeks_total
    sample_w = min(args.matrix_weeks, n_window)
    weeks = list(range(pw, pw + sample_w))
    cfgs = sample_configs(args.matrix_candidates, args.seed)
    cell["sample"] = {"candidates": len(cfgs), "weeks": sample_w}
    cell["threads"] = {}
    digests = set()
    for th in args.threads:
        pool = choose_pool(kernel, th, args.pool)
        est = estimate_cell(n, years, th, pool)
        why = skip_reason(est, args.ram_budget_gb, args.disk_budget_gb, free)
        if why:
            cell["threads"][str(th)] = {"pool": pool, "skipped": why, "est_ram_bytes": est["est_ram_bytes"]}
            continue
        with Runner(paths, kernel, th, pool, args.warmup) as r, RssMonitor() as mon:
            st = run_pass(r, weeks, cfgs)
        digests.add(st["digest"])
        st.update(pool=pool, rss=rss_report(mon, th, pool), est_ram_bytes=est["est_ram_bytes"],
                  s_per_candidate_window=st["s_per_candidate_week"] * n_window)
        cell["threads"][str(th)] = st
    cell["identical_results_across_threads"] = len(digests) <= 1
    ran = {t: v for t, v in cell["threads"].items() if "skipped" not in v}
    if ran:
        best = min(ran, key=lambda t: ran[t]["s_per_candidate_week"])
        cell["best_threads"] = int(best)
        cell["projection"] = project(ran[best]["s_per_candidate_week"], window_weeks=n_window, params=args.params)
    if not args.keep_panels:
        shutil.rmtree(root / f"m{n}_{years}", ignore_errors=True)
    return cell


# ------------------------------------------------------------------------------------------ report

def _fmt_ms(v: float) -> str:
    return f"{v * 1e3:.1f}"


def render_markdown(result: dict) -> str:
    L = ["# E1 benchmark (real account engine, signal masks, panel loader)", ""]
    meta = result["meta"]
    L += [f"- kernel: `{meta['kernel']}`; matrix pool rule: {meta['pool_rule']}; smoke: {meta['smoke']}",
          f"- budgets: RAM {meta['ram_budget_gb']} GB, disk {meta['disk_budget_gb']} GB; reference equity "
          f"{REFERENCE_EQUITY:.0f}; panel: {meta['panel_source']}",
          f"- machine: {result['profile']['cpu']}, {result['profile']['cores_logical']} threads, "
          f"{result['profile']['ram_total_gb']} GB RAM", ""]
    s1, s2, s3, s4 = (result["steps"].get(k) for k in ("1", "2", "3", "4"))
    if s1:
        L += [f"**Step 1** deterministic rerun: {s1['deterministic']} ({s1['candidates']} candidates, "
              f"{s1['trades']} trades)", ""]
    if s2:
        c, w = s2["replay_cold"], s2["replay_warm"]
        L += [f"**Step 2** 30 x {s2['sessions']} sessions: build {s2['panel_build_s']} s, disk {s2['disk_mb']} MB, "
              f"npy read {s2['npy_read_mb_s']} MB/s, mmap open {s2['mmap_open_s']} s; replay cold "
              f"{_fmt_ms(c['s_per_candidate_week'])} / warm {_fmt_ms(w['s_per_candidate_week'])} ms per "
              f"candidate-week; indicator-cache hit rate cold pass {c['cache_hit_rate']}, warm pass "
              f"{w['cache_hit_rate']}; mask CPU cold {c['mask_cpu_s']} s vs warm {w['mask_cpu_s']} s", ""]
    if s3:
        L += [f"## {s3['name']} ({s3['workers']} worker(s), pool {s3['pool']})", "",
              "| window | s/cand-window cold | warm | hit rate cold/warm | peak RAM total MB | per worker MB |",
              "|---|---|---|---|---|---|"]
        for lab, v in s3["windows"].items():
            L.append(f"| {lab} | {v['s_per_candidate_window_cold']:.3f} | {v['s_per_candidate_window_warm']:.3f} | "
                     f"{v['cold']['cache_hit_rate']}/{v['warm']['cache_hit_rate']} | {v['rss']['peak_total_mb']} | "
                     f"{v['rss']['peak_per_worker_mb']} |")
        L.append("")
    if s4:
        L += [f"## {s4['name']}", "", "| cycle | new chunks | reused | reuse | wall s |", "|---|---|---|---|---|"]
        L += [f"| {c['cycle']} | {c['chunks_new']} | {c['chunks_reused']} | {c['reuse_fraction']:.1%} | "
              f"{c['wall_s']} |" for c in s4["cycles"]]
        L.append("")
    cells = result.get("matrix", [])
    if cells:
        ths = sorted({int(t) for c in cells for t in c.get("threads", {})})
        L += ["## Scale matrix (seconds per candidate-window; window = full history, extrapolated from a sample)", "",
              "| N | years | build s | panel GB | " + " | ".join(f"{t}T s/cw (RAM total/worker MB)" for t in ths)
              + " | full run h cache / no cache | weekend h cache / no cache |",
              "|---|---|---|---|" + "---|" * len(ths) + "---|---|"]
        for c in cells:
            if "skipped" in c:
                L.append(f"| {c['n']} | {c['years']} | SKIPPED: {c['skipped']} |" + " |" * (len(ths) + 3))
                continue
            cols = []
            for t in ths:
                v = c["threads"].get(str(t))
                if v is None:
                    cols.append("-")
                elif "skipped" in v:
                    cols.append(f"skip: {v['skipped']}")
                else:
                    cols.append(f"{v['s_per_candidate_window']:.2f} ({v['rss']['peak_total_mb']}/"
                                f"{v['rss']['peak_per_worker_mb']}, {v['pool']})")
            p = c.get("projection")
            proj = (f"{p['with_cache']['full_run_hours_2x_reserve']} / "
                    f"{p['without_cache']['full_run_hours_2x_reserve']}"
                    if p else "-")
            wk = (f"{p['with_cache']['weekend_job_hours']} / {p['without_cache']['weekend_job_hours']}" if p else "-")
            L.append(f"| {c['n']} | {c['years']} | {c['build_s']} | {c['panel_bytes'] / 1e9:.2f} | "
                     + " | ".join(cols) + f" | {proj} | {wk} |")
        L += ["", f"Projection parameters: `{json.dumps(result['params'])}`; 2x rerun reserve; 24 h weekend target. "
                  "With cache: first cutoff replays whole windows, later cutoffs one new chunk per carried-over "
                  "candidate (D6). Rates use the fastest measured thread count per cell.", ""]
    return "\n".join(L)


def parse_list(s: str, typ: Callable[[str], Any]) -> list[Any]:
    return [typ(x) for x in s.split(",") if x.strip()]


def load_params(raw: str | None) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    if raw:
        text = Path(raw).read_text(encoding="utf-8") if Path(raw).is_file() else raw
        params.update(json.loads(text))
    return params


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--kernel", choices=["python", "fast"], default="python")
    ap.add_argument("--pool", choices=["auto", "thread", "process"], default="auto",
                    help="auto: threads if the kernel releases the GIL, else processes over memmapped panels")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes; completes in well under 2 minutes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-list", default="300,500,1000")
    ap.add_argument("--years-list", default="2,5,10")
    ap.add_argument("--threads", default="1,8,16")
    ap.add_argument("--matrix-candidates", type=int, default=32)
    ap.add_argument("--matrix-weeks", type=int, default=4, help="sampled weeks per matrix cell (extrapolated)")
    ap.add_argument("--calib-n", type=int, default=300, help="N for step 3-4 (synthetic panel)")
    ap.add_argument("--window-weeks", type=int, default=52, help="step 3 history window length in weeks")
    ap.add_argument("--calib-candidates", type=int, default=256)
    ap.add_argument("--calib-workers", type=int, default=1)
    ap.add_argument("--window-weeks-step4", type=int, default=26)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--cycle-candidates", type=int, default=64)
    ap.add_argument("--step2-candidates", type=int, default=64)
    ap.add_argument("--step2-weeks", type=int, default=4, help="measured weeks in step 2 (4 = 20 sessions)")
    ap.add_argument("--step1-candidates", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=WARMUP_SESSIONS, help="history-prefix sessions per week chunk")
    ap.add_argument("--ram-budget-gb", type=float, default=11.0)
    ap.add_argument("--disk-budget-gb", type=float, default=30.0)
    ap.add_argument("--params-json", default=None, help="JSON file or inline JSON overriding projection params")
    ap.add_argument("--panel-dir", type=Path, default=None, help="existing real chunk root for steps 3-4")
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--keep-panels", action="store_true")
    ap.add_argument("--work-dir", type=Path, default=None, help="scratch for generated panels (default: temp)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    return ap


def apply_smoke(args: argparse.Namespace) -> None:
    args.n_list, args.years_list, args.threads = "12", "0.05", "1,2"
    args.matrix_candidates, args.matrix_weeks = 4, 1
    args.calib_n, args.window_weeks, args.calib_candidates = 12, 1, 4
    args.window_weeks_step4, args.cycles, args.cycle_candidates = 2, 3, 4
    args.step2_candidates, args.step2_weeks, args.step1_candidates, args.warmup = 4, 1, 2, 5
    args.params = {**args.params, "weeks": 4, "budget": 16}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.params = load_params(args.params_json)
    if args.smoke:
        apply_smoke(args)
    args.threads = parse_list(args.threads, int) if isinstance(args.threads, str) else args.threads
    kernel = resolve_kernel(args.kernel)
    t_start = time.perf_counter()
    tmp = tempfile.TemporaryDirectory(prefix="e1bench_", dir=args.work_dir) if not args.work_dir else None
    root = Path(tmp.name) if tmp else Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        steps: dict[str, Any] = {"1": step1(root, kernel, args.seed, args.warmup, args.step1_candidates),
                                 "2": step2(root, kernel, args.seed, args.step2_candidates, args.warmup,
                                            args.step2_weeks)}
        if args.panel_dir:
            paths = list_chunks(args.panel_dir)
            n_sym = read_meta(paths[0])["n_symbols"]
            source = f"real chunks at {args.panel_dir}"
        else:
            n_weeks = prefix_weeks(args.warmup) + 2 * args.window_weeks + max(args.cycles, 1) + args.window_weeks_step4
            paths, _ = make_panel(root / "calib", args.calib_n, n_weeks, args.seed)
            n_sym, source = args.calib_n, "synthetic (generate_synthetic_panel)"
        steps["3"] = step3(paths, kernel, args, n_sym)
        steps["4"] = step4(paths, kernel, args)
        matrix: list[dict] = []
        if not args.skip_matrix:
            for n in parse_list(args.n_list, int):
                for years in parse_list(args.years_list, float):
                    matrix.append(matrix_cell(n, years, args, kernel, root))
        result = {
            "meta": {"kernel": kernel.name, "kernel_releases_gil": kernel.releases_gil, "smoke": args.smoke,
                     "pool_rule": "threads (kernel releases GIL)" if kernel.releases_gil
                     else "processes with panels memmapped read-only (kernel holds the GIL)",
                     "ram_budget_gb": args.ram_budget_gb, "disk_budget_gb": args.disk_budget_gb,
                     "panel_source": source, "total_wall_s": round(time.perf_counter() - t_start, 1)},
            "profile": machine_profile(root), "params": args.params, "steps": steps, "matrix": matrix,
        }
    finally:
        if tmp:
            tmp.cleanup()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "bench_smoke" if args.smoke else "bench"
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md = render_markdown(result)
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
