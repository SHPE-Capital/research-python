# ruff: noqa: E402 -- set the numerical thread limits before importing NumPy or sklearn.
"""Synthetic width benchmark using the production features and account. Never a market result."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

from shpe_research.calibration.rm9_engine import Week, compile_week
from shpe_research.calibration.rm9_ops import resources, single_writer
from shpe_research.calibration.rm9_search import Grid
from shpe_research.calibration.rm9_spec import source_id, spec
from shpe_research.data.rm9_io import write


def bench(root, n, workers):
    root.mkdir(parents=True, exist_ok=True)
    p = spec("full")
    machine = resources(root)
    if not 1 <= n <= 1000 or not 1 <= workers <= machine["cores"] or machine["available_bytes"] < n * 1950 * 45:
        raise ValueError("check requested width, allocated CPUs and memory")
    days = pd.bdate_range("2023-01-09", periods=5)
    ids = [f"S{i:04d}" for i in range(n)]
    groups = dict.fromkeys(ids, pd.DataFrame({"start": ["2010-01-01"], "end": ["2099-01-01"]}))
    plan = {
        "days": [str(d.date()) for d in days],
        "asof": "2023-01-06",
        "members": [dict(id=key, cell=i % 9, post_split_block=[]) for i, key in enumerate(ids)],
    }

    def block(lo, hi, ds, ts, si):
        a = np.empty((hi - lo, len(ts), 5), dtype="f4")
        for j in range(lo, hi):
            rng = np.random.default_rng(j)
            ret = rng.normal(0.00002, 0.00065, len(ts))
            c = (15 + j % 150) * np.exp(np.cumsum(ret))
            o = np.r_[c[0], c[:-1]]
            a[j - lo] = np.column_stack(
                (o, np.maximum(o, c) * 1.00008, np.minimum(o, c) * 0.99992, c, rng.uniform(12000, 200000, len(ts)))
            )
        return a

    tab = np.ones((9, 3, 2, 2))
    tab[..., 0] = 3
    tab[..., 1] = 700
    grid = Grid()
    cfgs = [grid.configs[i] for i, _ in grid.broad(32, np.random.default_rng(81))]
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        where = Path(tmp) / "width"
        meta = compile_week(None, groups, plan, where, p, block=min(24, max(1, n)), fixture_block=block)
        wk = Week(where, {"cube": tab.tolist(), "impact_bps": 2.0}, p)
        wk.simulate(cfgs[0])
        timings = []
        for k in sorted({1, workers}):
            t = time.perf_counter()
            with ThreadPoolExecutor(k) as ex:
                results = list(ex.map(wk.simulate, cfgs))
            elapsed = time.perf_counter() - t
            timings.append(
                {
                    "workers": k,
                    "candidates": len(cfgs),
                    "seconds": elapsed,
                    "candidate_weeks_per_second": len(cfgs) / elapsed,
                    "simulated_round_trips": sum(sum(r["trades"]) for r in results),
                }
            )
        del wk
    return {
        "test": "synthetic_width_only",
        "not_a_cluster_forecast": True,
        "source": source_id(),
        "machine": machine,
        "stocks": n,
        "target_minutes": meta["minutes"],
        "warmup_sessions": p["warmup"],
        "target_sessions": len(days),
        "prepared_bytes": meta["bytes"],
        "compile_seconds_including_fixture": meta["compile_seconds"],
        "timings": timings,
    }


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--root", type=Path, default=Path("runs/rm9-width"))
    args = parser.parse_args()
    with single_writer(args.root / "width.lock"):
        result = bench(args.root, args.stocks, args.workers)
        write(args.root / f"width-{args.stocks}.json", result)
        print(json.dumps(result, indent=2))
