"""E1 benchmark harness and compute projection (momentum plan, task E1).

Runs the benchmark sequence on this machine and writes a machine profile, measured numbers and a compute
projection to `studies/momentum/rm9-weekly-calibration/bench/`.

IMPORTANT: the C4 account simulator and the A6 panel do not exist yet, and `trade.simulate.simulate_trades`
is a stub. Steps 1-2 therefore time a *proxy* replay kernel (a vectorised opening-range breakout over
synthetic bars). Its seconds-per-candidate are provisional and must be re-measured on C4. Steps 3-4 report
BLOCKED until `shpe_research.trade.account` exists. Run natively (`uv run python scripts/momentum/bench.py`),
not in a CPU-capped container.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from shpe_research.data.fixtures import synthetic_bars

OUT_DIR = Path("studies/momentum/rm9-weekly-calibration/bench")
WEEKEND_TARGET_HOURS = 24.0
RERUN_RESERVE = 2.0
SYMBOL_POOL = 300


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


def peak_rss_mb() -> float:
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)  # peak_wset is Windows-only


def make_symbols(n: int) -> list[str]:
    return [f"S{i:03d}" for i in range(n)]


def sessions_of(bars: pd.DataFrame) -> pd.Series:
    return bars["ts"].dt.tz_convert("America/New_York").dt.date


@dataclass
class OpeningRange:
    high: float
    low: float
    atr: float
    day: np.ndarray  # close path after the opening range


class RangeCache:
    """Caches per-(symbol, session, or_minutes) opening-range features; counts hits for the hit-rate metric."""

    def __init__(self) -> None:
        self.store: dict[tuple, OpeningRange] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple, build) -> OpeningRange:
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        self.store[key] = value = build()
        return value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def split_sessions(bars: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """[(symbol, session, high, low, close)] arrays, one entry per symbol-session."""
    day = sessions_of(bars)
    out = []
    for (sym, d), g in bars.groupby([bars["symbol"], day], sort=False):
        out.append((sym, str(d), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()))
    return out


def replay_candidate(units, params: tuple[int, float, float], cache: RangeCache) -> tuple[float, int]:
    """PROXY replay of one candidate (or_minutes, stop_atr, target_atr) over all symbol-sessions.

    Long-only breakout of the opening range; exits at stop/target (in range-ATR units) or the close.
    Returns (sum of returns, trades). Deterministic. Not the C4 simulator.
    """
    or_min, stop_k, target_k = params
    total, trades = 0.0, 0
    for sym, d, hi, lo, cl in units:
        if len(cl) <= or_min + 1:
            continue

        def build(hi=hi, lo=lo, cl=cl):
            rng_hi, rng_lo = hi[:or_min].max(), lo[:or_min].min()
            atr = float(np.mean(hi[:or_min] - lo[:or_min])) or 1e-9
            return OpeningRange(float(rng_hi), float(rng_lo), atr, cl[or_min:])

        r = cache.get((sym, d, or_min), build)
        path = r.day
        brk = np.flatnonzero(path > r.high)
        if brk.size == 0:
            continue
        entry = path[brk[0]]
        after = path[brk[0] + 1 :]
        pnl = (after - entry) if after.size else np.zeros(1)
        hit = np.flatnonzero((pnl <= -stop_k * r.atr) | (pnl >= target_k * r.atr))
        exit_pnl = pnl[hit[0]] if hit.size else pnl[-1]
        total += float(exit_pnl / entry)
        trades += 1
    return total, trades


def candidate_grid(n: int) -> list[tuple[int, float, float]]:
    ors = [5, 10, 15, 20, 30, 45, 60, 90]
    stops = [0.5, 1.0, 1.5, 2.0]
    targets = [1.0, 2.0, 3.0, 4.0, 6.0]
    grid = [(o, s, t) for o in ors for s in stops for t in targets]  # 160
    while len(grid) < n:
        grid += [(o, s * 1.1, t * 1.1) for o, s, t in grid[: n - len(grid)]]
    return grid[:n]


@dataclass
class StepResult:
    name: str
    status: str  # OK | BLOCKED
    detail: dict = field(default_factory=dict)


def timed_panel(n_sym: int, start: str, end: str, scratch: Path) -> tuple[pd.DataFrame, dict]:
    t = time.perf_counter()
    bars = synthetic_bars(symbols=make_symbols(n_sym), start=start, end=end)
    gen_s = time.perf_counter() - t
    path = scratch / f"panel_{n_sym}.parquet"
    t = time.perf_counter()
    bars.to_parquet(path)
    write_s = time.perf_counter() - t
    size_mb = path.stat().st_size / 1e6
    t = time.perf_counter()
    bars = pd.read_parquet(path)
    read_s = time.perf_counter() - t
    return bars, {
        "rows": len(bars),
        "generate_s": round(gen_s, 3),
        "parquet_mb": round(size_mb, 2),
        "write_mb_s": round(size_mb / write_s, 1) if write_s else None,
        "read_s": round(read_s, 3),
        "read_mb_s": round(size_mb / read_s, 1) if read_s else None,
    }


def measure_replay(units, n_candidates: int) -> dict:
    grid = candidate_grid(n_candidates)
    cache = RangeCache()
    t = time.perf_counter()
    cold_out = [replay_candidate(units, g, cache) for g in grid]
    cold = time.perf_counter() - t
    cold_hit = cache.hit_rate
    t = time.perf_counter()
    warm_out = [replay_candidate(units, g, cache) for g in grid]  # same candidates, fully cached
    warm = time.perf_counter() - t
    assert cold_out == warm_out, "replay is not deterministic"
    return {
        "candidates": n_candidates,
        "symbol_sessions": len(units),
        "cold_s_per_candidate": cold / n_candidates,
        "warm_s_per_candidate": warm / n_candidates,
        "cache_hit_rate_cold_pass": round(cold_hit, 3),
        "cache_hit_rate_overall": round(cache.hit_rate, 3),
        "trades_total": int(sum(t for _, t in cold_out)),
    }


def step1(scratch: Path) -> StepResult:
    bars, io = timed_panel(1, "2026-03-02", "2026-03-06", scratch)
    units = split_sessions(bars)
    m = measure_replay(units, 16)
    return StepResult("1 one stock, few sessions (correctness)", "OK", {"io": io, **m, "deterministic": True})


def step2(scratch: Path, n_sym: int, n_candidates: int) -> StepResult:
    bars, io = timed_panel(n_sym, "2026-03-02", "2026-03-27", scratch)  # 20 sessions
    t = time.perf_counter()
    units = split_sessions(bars)
    split_s = time.perf_counter() - t
    m = measure_replay(units, n_candidates)
    return StepResult(
        "2 30 stocks x 20 sessions (load + indicator profile)",
        "OK",
        {"io": io, "session_split_s": round(split_s, 3), **m},
    )


def steps_3_4() -> list[StepResult]:
    try:
        import shpe_research.trade.account  # noqa: F401
        ready = True
    except ImportError:
        ready = False
    reason = "C4 account simulator (shpe_research.trade.account) and A6 panel not implemented"
    out = []
    for name in ("3 300-stock early+late calibration (>=256 candidates, finalists, account replay)",
                 "4 consecutive weekly cycles (incremental reuse)"):
        out.append(StepResult(name, "BLOCKED" if not ready else "NOT_RUN", {"reason": reason}))
    return out


def project(step2_result: StepResult, *, history_sessions: int, candidates: int, weeks: int, policies: int,
            seeds: int, n_symbols: int = SYMBOL_POOL) -> dict:
    d = step2_result.detail
    out = {}
    for label, per_cand in (("cold", d["cold_s_per_candidate"]), ("warm", d["warm_s_per_candidate"])):
        per_unit = per_cand / d["symbol_sessions"]
        calib_s = per_unit * n_symbols * history_sessions * candidates  # one weekly calibration
        total_s = calib_s * weeks * policies * seeds * RERUN_RESERVE
        out[label] = {
            "weekend_job_hours": round(calib_s / 3600, 3),
            "weekend_job_vs_24h_target": round(calib_s / 3600 / WEEKEND_TARGET_HOURS, 4),
            "full_run_hours_with_2x_reserve": round(total_s / 3600, 2),
        }
    out["assumptions"] = {
        "n_symbols": n_symbols, "history_sessions": history_sessions, "candidates": candidates,
        "weeks": weeks, "policies": policies, "seeds": seeds, "rerun_reserve": RERUN_RESERVE,
        "note": "weeks/policies/seeds/history are PLACEHOLDERS until 0-F fixes scope; proxy kernel, not C4",
    }
    return out


def render_markdown(profile: dict, steps: list[StepResult], proj: dict, peak_mb: float) -> str:
    lines = ["# E1 benchmark (PROVISIONAL: proxy replay kernel on synthetic bars)", "", "## Machine", ""]
    lines += [f"- {k}: {v}" for k, v in profile.items()]
    lines += ["", f"Peak process RSS: {peak_mb} MB", "", "## Steps", "",
              "| step | status | key numbers |", "|---|---|---|"]
    for s in steps:
        if s.status == "OK":
            d = s.detail
            nums = (f"cold {d['cold_s_per_candidate']*1e3:.2f} ms/cand, "
                    f"warm {d['warm_s_per_candidate']*1e3:.2f} ms/cand, "
                    f"hit rate {d['cache_hit_rate_overall']}, read {d['io']['read_mb_s']} MB/s, "
                    f"{d['symbol_sessions']} symbol-sessions")
        else:
            nums = s.detail["reason"]
        lines.append(f"| {s.name} | {s.status} | {nums} |")
    lines += ["", "## Projection (300 symbols, 2x rerun reserve)", "",
              "| cache | weekend job (h) | vs 24h | full run (h) |", "|---|---|---|---|"]
    for k in ("cold", "warm"):
        p = proj[k]
        lines.append(f"| {k} | {p['weekend_job_hours']} | {p['weekend_job_vs_24h_target']:.2%} | "
                     f"{p['full_run_hours_with_2x_reserve']} |")
    lines += ["", f"Assumptions: `{json.dumps(proj['assumptions'])}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument("--symbols", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--history-sessions", type=int, default=252)
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--policies", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="e1bench_") as tmp:
        scratch = Path(tmp)
        profile = machine_profile(scratch)
        steps = [step1(scratch), step2(scratch, args.symbols, args.candidates), *steps_3_4()]
    peak = peak_rss_mb()
    proj = project(steps[1], history_sessions=args.history_sessions, candidates=args.candidates,
                   weeks=args.weeks, policies=args.policies, seeds=args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(
        json.dumps({"profile": profile, "peak_rss_mb": peak, "steps": [asdict(s) for s in steps], "projection": proj},
                   indent=2, default=str), encoding="utf-8")
    md = render_markdown(profile, steps, proj, peak)
    (args.out / "bench.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
