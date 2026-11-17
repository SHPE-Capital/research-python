"""Generate a synthetic week-chunked market panel on disk (momentum plan A6) and report cost.

    uv run python scripts/momentum/make_synth_panel.py --symbols 1000 --start 2024-01-02 --end 2024-12-31 \
        --out D:/panels/synth1000 [--measure-load]

Writes one directory per ISO week under --out, one week in RAM at a time. Reports build time, bytes per
stock-week and per stock-year, and the generator's peak RSS. With --measure-load it opens every chunk in a
fresh child process (memory-mapped, no copy) and reports load time and the RSS growth of that "worker".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import generate_synthetic_panel, list_chunks, open_panel_chunk


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def peak_rss_mb() -> float:
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)


def worker_load(out: str) -> None:
    """Child process body: map every chunk, touch only one column per chunk, print RSS growth."""
    proc = psutil.Process()
    before = proc.memory_info().rss
    t0 = time.perf_counter()
    panels = [open_panel_chunk(p) for p in list_chunks(out)]
    load_s = time.perf_counter() - t0
    after = proc.memory_info().rss
    touched = float(sum(np.nansum(p.close[:, 0]) for p in panels))  # touch one column per chunk
    mapped = sum(a.nbytes for p in panels for a in (p.open, p.high, p.low, p.close, p.volume, p.eligible))
    print(json.dumps({"chunks": len(panels), "load_seconds": round(load_s, 3), "mapped_mb": round(mapped / 1e6, 1),
                      "rss_growth_mb_after_open": round((after - before) / 1e6, 1),
                      "rss_growth_mb_after_touch": round((proc.memory_info().rss - before) / 1e6, 1),
                      "touch_checksum": touched}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", type=int, default=300)
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-load", action="store_true")
    ap.add_argument("--worker", action="store_true", help="internal: child load measurement")
    a = ap.parse_args()
    if a.worker:
        worker_load(a.out)
        return 0

    sessions = get_calendar().sessions(a.start, a.end)
    t0 = time.perf_counter()
    paths = generate_synthetic_panel(a.out, a.symbols, sessions, a.seed)
    build_s = time.perf_counter() - t0
    total = sum(dir_bytes(p) for p in paths)
    n_weeks = len(paths)
    report = {
        "symbols": a.symbols, "sessions": len(sessions), "weeks": n_weeks,
        "build_seconds": round(build_s, 1), "total_gb": round(total / 1e9, 3),
        "bytes_per_stock_week_full": round(total / (a.symbols * n_weeks)),
        "bytes_per_stock_year": round(total / a.symbols / (len(sessions) / 252)),
        "largest_chunk_mb": round(max(dir_bytes(p) for p in paths) / 1e6, 1),
        "generator_peak_rss_mb": peak_rss_mb(),
    }
    print(json.dumps(report, indent=2))
    if a.measure_load:
        cmd = [sys.executable, __file__, "--worker", "--out", a.out]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(r.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
