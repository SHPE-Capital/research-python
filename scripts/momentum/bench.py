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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.panel import generate_synthetic_panel

OUT_DIR = Path("studies/momentum/rm9-weekly-calibration/bench")
WEEKEND_TARGET_HOURS = 24.0
RERUN_RESERVE = 2.0
SYMBOL_POOL = 300
PROTOCOL_PATH = Path("configs/strategies/momentum/protocol.yaml")  # 0-G; absent until it lands, see load_protocol
SESSIONS_PER_YEAR = 252
MINUTES_PER_SESSION = 390
MATRIX_N = (300, 500, 1000)
MATRIX_YEARS = (2, 5, 10)
MATRIX_THREADS = (1, 8, 16)


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


def load_protocol(path: Path = PROTOCOL_PATH) -> dict | None:
    """Read N / years / search budget from the 0-G protocol config; None when 0-G has not landed.

    Assumed keys (0-G schema not available here): universe_size, history_years, search_budget (candidates).
    """
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def measure_build(scratch: Path, n_sym: int, n_sessions: int) -> dict:
    """Time the A6 synthetic week-chunk build and measure its bytes on disk (the D6 cache's cost and size)."""
    sessions = pd.bdate_range("2026-03-02", periods=n_sessions)
    out = scratch / "a6_panel"
    t = time.perf_counter()
    generate_synthetic_panel(out, n_sym, sessions)
    build_s = time.perf_counter() - t
    units = n_sym * n_sessions
    return {"symbol_sessions": units, "build_s_per_symbol_session": build_s / units,
            "panel_bytes_per_symbol_session": dir_bytes(out) / units}


def _thread_worker(args: tuple) -> tuple[float, float]:
    """Replay a slice of the candidate grid in a fresh cache; return (seconds, process peak RSS MB)."""
    units, grid = args
    cache = RangeCache()
    t = time.perf_counter()
    for g in grid:
        replay_candidate(units, g, cache)
    return time.perf_counter() - t, peak_rss_mb()


def _noop(_: int) -> int:
    return 0


def measure_threads(units, n_candidates: int, threads: tuple[int, ...]) -> dict[int, dict]:
    """Wall time and per-worker RSS of the stand-in replay split over spawned workers.

    A worker count is SKIPPED_RAM when workers x measured worker RSS would exceed 80% of available RAM.
    """
    grid = candidate_grid(n_candidates)
    out: dict[int, dict] = {}
    worker_mb = peak_rss_mb()  # upper bound until a 1-worker run measures the real footprint
    for w in sorted(threads):
        if w * worker_mb * 1e6 > 0.8 * psutil.virtual_memory().available:
            out[w] = {"status": "SKIPPED_RAM"}
            continue
        slices = [(units, grid[i::w]) for i in range(w)]
        with ProcessPoolExecutor(max_workers=w) as ex:
            list(ex.map(_noop, range(w)))  # start the workers before timing
            t = time.perf_counter()
            res = list(ex.map(_thread_worker, slices))
            wall = time.perf_counter() - t
        out[w] = {"status": "OK", "wall_s": wall, "worker_peak_rss_mb": max(r[1] for r in res)}
        worker_mb = out[w]["worker_peak_rss_mb"]
    base = out.get(1, {}).get("wall_s")
    for r in out.values():
        if r["status"] == "OK" and base:
            r["speedup"] = base / r["wall_s"]
    return out


def scale_matrix(*, ns, years, threads, replay: dict, build: dict, parquet_bytes_per_unit: float,
                 thread_runs: dict[int, dict], candidates: int, weeks: int, policies: int, seeds: int,
                 pull_rows_per_s: float | None = None) -> list[dict]:
    """Project every (N, years, threads) cell from per-symbol-session stand-in costs.

    Without the D6 cache each weekly calibration rebuilds the whole panel and replays cold; with it only the
    newest week chunk is built and replayed cold, older sessions hit the cache. Linear extrapolation: a floor.
    """
    cold = replay["cold_s_per_candidate"] / replay["symbol_sessions"]
    warm = replay["warm_s_per_candidate"] / replay["symbol_sessions"]
    reps = weeks * policies * seeds * RERUN_RESERVE
    rows = []
    for n in ns:
        for y in years:
            units = n * y * SESSIONS_PER_YEAR
            new_units = n * 5  # one week chunk
            per_unit_bytes = build["panel_bytes_per_symbol_session"]
            for t in threads:
                run = thread_runs.get(t, {})
                if run.get("status") != "OK" or "speedup" not in run:
                    rows.append({"n": n, "years": y, "threads": t, "status": run.get("status", "NOT_MEASURED")})
                    continue
                sp = run["speedup"]
                build_s = build["build_s_per_symbol_session"]
                no_cache = units * build_s + units * candidates * cold / sp
                cache = new_units * build_s + (units * warm + new_units * (cold - warm)) * candidates / sp
                worker_mb = run["worker_peak_rss_mb"] + new_units * per_unit_bytes / 1e6  # one week chunk mapped
                rows.append({
                    "n": n, "years": y, "threads": t, "status": "OK",
                    "symbol_sessions": units,
                    "panel_gb": units * per_unit_bytes / 1e9,
                    "parquet_gb": units * parquet_bytes_per_unit / 1e9,
                    "pull_hours": units * MINUTES_PER_SESSION / pull_rows_per_s / 3600 if pull_rows_per_s else None,
                    "ram_per_worker_mb": worker_mb,
                    "ram_total_gb": worker_mb * t / 1e3,
                    "weekly_h_no_cache": no_cache / 3600,
                    "weekly_h_d6_cache": cache / 3600,
                    "full_run_h_no_cache": no_cache * reps / 3600,
                    "full_run_h_d6_cache": cache * reps / 3600,
                })
    return rows


def render_matrix(rows: list[dict], protocol: dict | None) -> str:
    src = "protocol config (0-G)" if protocol else "PLACEHOLDERS (0-G protocol config not present)"
    lines = ["", "## Scale matrix (STAND-IN replay: a FLOOR, not a C4 measurement)", "",
             f"N / years / search budget source: {src}. Linear extrapolation from per-symbol-session costs of the "
             "proxy kernel and the A6 synthetic builder; real C4 replay will be slower. Pull time needs "
             "`--pull-rows-per-s` (no data was pulled); otherwise n/a.", "",
             "| N | years | threads | panel GB | parquet GB | pull h | RAM/worker MB | RAM total GB | "
             "weekly h no cache | weekly h D6 | full run h no cache | full run h D6 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "OK":
            lines.append(f"| {r['n']} | {r['years']} | {r['threads']} | {r['status']} |" + " |" * 8)
            continue
        pull = f"{r['pull_hours']:.1f}" if r["pull_hours"] is not None else "n/a"
        lines.append(f"| {r['n']} | {r['years']} | {r['threads']} | {r['panel_gb']:.1f} | {r['parquet_gb']:.1f} | "
                     f"{pull} | {r['ram_per_worker_mb']:.0f} | {r['ram_total_gb']:.2f} | "
                     f"{r['weekly_h_no_cache']:.2f} | {r['weekly_h_d6_cache']:.2f} | "
                     f"{r['full_run_h_no_cache']:.1f} | {r['full_run_h_d6_cache']:.1f} |")
    lines.append("")
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
    ap.add_argument("--matrix", action="store_true", help="also run the scale matrix and thread sweep")
    ap.add_argument("--matrix-n", type=int, nargs="+", default=list(MATRIX_N))
    ap.add_argument("--matrix-years", type=int, nargs="+", default=list(MATRIX_YEARS))
    ap.add_argument("--matrix-threads", type=int, nargs="+", default=list(MATRIX_THREADS))
    ap.add_argument("--thread-candidates", type=int, default=1024)
    ap.add_argument("--pull-rows-per-s", type=float, default=None, help="measured archive pull rate; no pulls here")
    ap.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if protocol:  # 0-G values replace the CLI defaults
        args.candidates = protocol.get("search_budget", args.candidates)
        args.matrix_n = [protocol.get("universe_size", n) for n in args.matrix_n[:1]]
        args.matrix_years = [protocol.get("history_years", y) for y in args.matrix_years[:1]]

    with tempfile.TemporaryDirectory(prefix="e1bench_") as tmp:
        scratch = Path(tmp)
        profile = machine_profile(scratch)
        steps = [step1(scratch), step2(scratch, args.symbols, args.candidates), *steps_3_4()]
    peak = peak_rss_mb()
    proj = project(steps[1], history_sessions=args.history_sessions, candidates=args.candidates,
                   weeks=args.weeks, policies=args.policies, seeds=args.seeds)
    matrix, md_extra = None, ""
    if args.matrix:
        bars = synthetic_bars(symbols=make_symbols(args.symbols), start="2026-03-02", end="2026-03-27")
        thread_runs = measure_threads(split_sessions(bars), args.thread_candidates, tuple(args.matrix_threads))
        with tempfile.TemporaryDirectory(prefix="e1matrix_") as tmp:
            build = measure_build(Path(tmp), 20, 10)
        d = steps[1].detail
        rows = scale_matrix(ns=args.matrix_n, years=args.matrix_years, threads=args.matrix_threads, replay=d,
                            build=build, parquet_bytes_per_unit=d["io"]["parquet_mb"] * 1e6 / d["symbol_sessions"],
                            thread_runs=thread_runs, candidates=args.candidates, weeks=args.weeks,
                            policies=args.policies, seeds=args.seeds, pull_rows_per_s=args.pull_rows_per_s)
        md_extra = render_matrix(rows, protocol) + (
            "\nMeasured thread sweep (stand-in): `" + json.dumps(thread_runs) + "`\n")
        matrix = {"rows": rows, "thread_runs": thread_runs, "build": build, "protocol": protocol}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(
        json.dumps({"profile": profile, "peak_rss_mb": peak, "steps": [asdict(s) for s in steps], "projection": proj,
                    "matrix": matrix}, indent=2, default=str), encoding="utf-8")
    md = render_markdown(profile, steps, proj, peak) + md_extra
    (args.out / "bench.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
