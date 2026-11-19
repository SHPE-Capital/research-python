"""Run lock, allocated resources, durable science inputs, and elapsed-compute deadlines."""

from __future__ import annotations

import contextlib
import os
import platform
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from shpe_research.calibration.rm9_engine import Week, compile_week
from shpe_research.calibration.rm9_spec import source_id
from shpe_research.data.rm9_io import digest, file_hash, pin, read, write
from shpe_research.data.rm9_market import Market, data_id


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextlib.contextmanager
def single_writer(path):
    """An OS lock is released on process death; never guess ownership from a stale PID file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt

            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise RuntimeError(f"another writer owns {path}") from e
        else:
            import fcntl

            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise RuntimeError(f"another writer owns {path}") from e
        yield


def resources(path):
    proc = psutil.Process()
    cores = os.cpu_count() or 1
    if hasattr(proc, "cpu_affinity"):
        cores = min(cores, len(proc.cpu_affinity()))
    if os.getenv("SLURM_CPUS_PER_TASK"):
        cores = min(cores, int(os.environ["SLURM_CPUS_PER_TASK"]))
    ram = psutil.virtual_memory().available
    c = Path("/sys/fs/cgroup")
    if (c / "cpu.max").exists():
        q, period = (c / "cpu.max").read_text().split()
        if q != "max":
            cores = min(cores, max(1, int(q) // int(period)))
    if (c / "memory.max").exists():
        limit = (c / "memory.max").read_text().strip()
        if limit != "max":
            ram = min(ram, int(limit) - int((c / "memory.current").read_text().strip()))
    for var, mult in (("SLURM_MEM_PER_NODE", 1), ("SLURM_MEM_PER_CPU", cores)):
        if os.getenv(var) and int(os.environ[var]) > 0:
            ram = min(ram, int(os.environ[var]) * mult * 1024**2)
    return {
        "cores": max(1, cores),
        "available_bytes": max(0, ram),
        # psutil's Windows backend expects a string path (it rejects
        # pathlib.Path on some supported Windows releases).
        "free_disk_bytes": psutil.disk_usage(str(path)).free,
        "host": platform.node(),
        "python": platform.python_version(),
        "slurm_job": os.getenv("SLURM_JOB_ID"),
    }


class Stopping(Exception):
    pass


class Clock:
    def __init__(self, root, hours, minutes=None, *, diagnostic=False):
        self.root = Path(root)
        path = self.root / "clock.json"
        previous = read(path) if path.exists() and not diagnostic else {}
        self.base = previous.get("active_seconds", 0.0)
        self.peak_rss_mb = previous.get("sampled_peak_rss_mb", 0.0)
        self.diagnostic = diagnostic
        self.start = time.monotonic()
        self.cap = hours * 3600
        self.job = minutes * 60 if minutes is not None else float("inf")
        self.reason = ""
        self.state = {"stage": "starting"}
        self.closed = threading.Event()
        self.lock = threading.RLock()
        self.handlers = {}
        for key in ("SIGINT", "SIGTERM", "SIGUSR1"):
            sig = getattr(signal, key, None)
            if sig is not None:
                self.handlers[sig] = signal.signal(sig, self.handle)
        self.thread = threading.Thread(target=self._pulse, daemon=True)
        self.thread.start()

    def handle(self, sig, frame):
        self.reason = f"signal {sig}"

    @property
    def spent(self):
        return self.base + time.monotonic() - self.start

    def check(self):
        if not self.diagnostic and self.spent >= self.cap:
            raise Stopping("TOTAL_COMPUTE_CAP: incomplete; no final comparison. A new budget is a new study.")
        if time.monotonic() - self.start >= self.job or self.reason:
            raise Stopping(self.reason or "job slice complete; resume the identical manifest")

    def status(self, stage, **kw):
        with self.lock:
            self.state = dict(stage=stage, **kw)
            self.flush()

    def flush(self):
        import json

        if self.diagnostic:
            return
        with self.lock:
            t = self.spent
            rss = psutil.Process().memory_info().rss / 1e6
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
            row = {
                **self.state,
                "utc": now(),
                "active_seconds": round(t, 3),
                "remaining_seconds": round(max(0, self.cap - t), 3),
                "rss_mb": round(rss, 1),
                "sampled_peak_rss_mb": round(self.peak_rss_mb, 1),
            }
            write(self.root / "clock.json", {"active_seconds": t, "sampled_peak_rss_mb": self.peak_rss_mb})
            write(self.root / "progress.json", row)
            with (self.root / "progress.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

    def _pulse(self):
        while not self.closed.wait(15):
            self.flush()

    def close(self):
        self.closed.set()
        self.thread.join()
        self.flush()
        for sig, fn in self.handlers.items():
            signal.signal(sig, fn)


def load_archive(root, p, deep=False):
    root = Path(root)
    a = read(root / "archive.json")
    if a["spec"] != data_id(p) or a["raw_inventory"] != file_hash(root / "inventory.json"):
        raise ValueError("archive spec or raw inventory mismatch")
    if a["source"]["sha256"] != file_hash(root / "instruments.csv") or a["cost_hash"] != file_hash(
        root / "quote-costs.json"
    ):
        raise ValueError("cost or membership source changed")
    plans = []
    for i, sha in enumerate(a["schedule_hashes"]):
        path = root / "schedule" / f"{i:04}.json"
        if sha != file_hash(path):
            raise ValueError("weekly membership checksum mismatch")
        plan = read(path)
        if not plan["asof"] < plan["days"][0] or len(plan["members"]) > p["universe"]:
            raise ValueError("invalid weekly selection")
        plans.append(plan)
    from shpe_research.data.calendar import get_calendar
    from shpe_research.data.panel import week_chunks

    calendar = [[str(x.date()) for x in w] for w in week_chunks(get_calendar().sessions(p["start"], p["end"]))]
    if [x["days"] for x in plans] != calendar or len(plans) != a["weeks"]:
        raise ValueError("incomplete or non-contiguous exchange calendar")
    qt = read(root / "quote-costs.json")
    if qt["latest_sample"] >= p["start"] or qt["source"] != p["mode"]:
        raise ValueError("quote costs not frozen before first search week")
    if deep:
        for side, h in read(root / "inventory.json").items():
            marker = root / side
            raw = root / (side.removesuffix(".sha.json") + ".parquet")
            if read(marker)["sha256"] != h or file_hash(raw) != h:
                # Filename may contain a year.month or historical symbol; suffix removal only affects basename.
                raise ValueError(f"raw checksum mismatch {side}")
    return a, plans, qt


class Runtime:
    def __init__(self, archive, run, p, clock):
        self.root = Path(run)
        self.root.mkdir(exist_ok=True, parents=True)
        self.p, self.clock = p, clock
        self.a, self.plans, self.qt = load_archive(archive, p)
        self.market = Market(archive, p, offline=True)
        # Verify the sealed inventory, not just the mutable adjacent sidecars, on first raw access.
        self.market.expected = read(Path(archive) / "inventory.json")
        self.groups = dict(tuple(self.market.asset_master()[0].groupby("asset_id")))
        self.identity = {"source": source_id(), "archive": digest(self.a), "spec": digest(p)}
        pin(self.root / "inputs.json", self.identity)
        self.maps = {}
        self.verified = set()
        self.block = max(1, min(24, int(resources(self.root)["available_bytes"] * 0.15 / (10000 * 700))))

    def week(self, w):
        self.clock.check()
        if w in self.maps:
            return self.maps[w]
        self.maps.clear()
        path = self.root / "prepared" / f"w{w:04}"
        meta = compile_week(
            self.market, self.groups, self.plans[w], path, self.p, self.block, should_stop=self.clock.check
        )
        if meta["selection"] != digest(self.plans[w]):
            raise ValueError("compiled artifact belongs to another selection")
        if w not in self.verified:
            for name, h in meta["hashes"].items():
                if file_hash(path / f"{name}.npy") != h:
                    raise ValueError(f"compiled artifact corrupt: {path}/{name}")
            self.verified.add(w)
        self.maps[w] = Week(path, self.qt, self.p)
        return self.maps[w]


def workloads(weeks, h, v, initial, new):
    """Every candidate has an expanding reference path through T-2. Funded base + 2x per policy-week."""
    decisions = weeks - h - v
    n = initial + new * (decisions - 1)
    return {
        "trial_count": n,
        "forward_weeks": decisions,
        "initial_cells": initial * h,
        "later_reference_cells": n * (weeks - 1) - initial * h,
        "max_funded_cells": 6 * decisions,
    }


def storage_probe(root):
    """Time a scratch-file read and durable SQLite transactions on the actual output filesystem."""
    import sqlite3
    import tempfile

    buff = os.urandom(1024 * 1024)
    size = 16 * len(buff)
    with tempfile.TemporaryDirectory(dir=root) as d:
        path = Path(d) / "io.bin"
        with path.open("w+b") as f:
            for _ in range(16):
                f.write(buff)
            f.flush()
            os.fsync(f.fileno())
            if hasattr(os, "posix_fadvise"):
                with contextlib.suppress(OSError):
                    os.posix_fadvise(f.fileno(), 0, size, os.POSIX_FADV_DONTNEED)
            f.seek(0)
            t = time.monotonic()
            while f.read(len(buff)):
                pass
            speed = size / max(time.monotonic() - t, 1e-6)
        conn = sqlite3.connect(Path(d) / "speed.db")
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("create table bench (return_vector blob)")
            times = []
            for _ in range(12):
                t = time.monotonic()
                with conn:
                    conn.executemany("insert into bench values(?)", [(b"0" * 90,)] * 8)
                times.append(time.monotonic() - t)
        finally:
            conn.close()
    return {"read_bytes_per_second": speed, "read_bytes": size, "commit_seconds_p90": float(np.quantile(times, 0.9))}


def benchmark(rt, grid, limit=None):
    from concurrent.futures import ThreadPoolExecutor

    p, clock = rt.p, rt.clock
    info = resources(rt.root)
    n = len(rt.plans)
    test_weeks = list(dict.fromkeys([0, p["history"] // 2, p["history"] - 1]))
    # Sample configurations from every family, do not inspect or export returns.
    cfgs = [grid.configs[i] for i, _ in grid.broad(16, np.random.default_rng(19))]
    workers = min(info["cores"], limit or 32, 32)
    est_bytes = p["universe"] * 1950 * 38
    if info["available_bytes"] < max(96 * 1024**2, est_bytes * 2):
        raise RuntimeError("insufficient allocated RAM for a full-width week")
    workers = max(1, min(workers, int(info["available_bytes"] * 0.40 / max(12_000_000, p["universe"] * 1950 * 11))))
    choices = sorted({1, min(2, workers), min(4, workers), min(8, workers), workers})
    measurements = []
    compile_s = []
    for w in test_weeks:
        clock.status("benchmark", week=w, history_only=True)
        wk = rt.week(w)
        # Compile JIT and page in representative bars; timing rows separately include cold open cost below.
        wk.simulate(cfgs[0])
        scale = max(1, p["universe"] / wk.meta["members"]) * max(1, 1950 / wk.meta["minutes"])
        compile_s.append(wk.meta["compile_seconds"] * scale)
        start = time.perf_counter()
        Week(rt.root / "prepared" / f"w{w:04}", rt.qt, p).simulate(cfgs[-1])
        one = (time.perf_counter() - start) * scale
        for k in choices:
            clock.check()
            start = time.perf_counter()
            with ThreadPoolExecutor(k) as ex:
                list(ex.map(wk.simulate, cfgs))
            measurements.append(
                {
                    "week": w,
                    "workers": k,
                    "seconds_per_cell": (time.perf_counter() - start) / len(cfgs) * scale,
                    "single_seconds_with_open": one,
                    "samples": len(cfgs),
                    "width_scale": scale,
                }
            )
    per = {k: max(x["seconds_per_cell"] for x in measurements if x["workers"] == k) for k in choices}
    k = min(per, key=per.get)
    margin = p["benchmark_margin"]
    batch = per[k] * margin
    one = max(x["single_seconds_with_open"] for x in measurements) * margin
    # Forecast cold feature work plus storage: use the slowest sampled week; include observed benchmark elapsed.
    overhead = max(compile_s) * max(0, n - len(test_weeks)) * margin + clock.spent
    disk_gb = est_bytes * n / 1e9
    disk_probe = storage_probe(rt.root)
    profiles = []
    for i in p["initial_ladder"]:
        for new in p["weekly_ladder"]:
            work = workloads(n, p["history"], p["validation"], i, new)
            if work["trial_count"] > len(grid.configs):
                continue
            initial = work["initial_cells"] * batch
            # New cohorts are small; use serial single + open for backfill to avoid assuming ideal thread scaling.
            late = work["later_reference_cells"] * max(batch, one / min(k, new)) + work["max_funded_cells"] * one

            # Old cache pages may be evicted. Allow a full prepared-week read per visit, plus durable writes.
            # Count unique batch visits: initial discovery round length 8 after a 14-config seed.
            def rounds(count, first=False):
                return 1 + (max(0, count - 14) + 7) // 8 if first else (count + 7) // 8

            h, v, w = p["history"], p["validation"], work["forward_weeks"]
            initial_visits = h * rounds(i, True)
            # Each new search cohort backfills the expanding old history; validation is visited again per weekend.
            later_visits = w * (v + 2) + rounds(new) * ((w - 1) * h + (w - 1) * w / 2)
            read_s = margin * est_bytes / disk_probe["read_bytes_per_second"]
            write_s = margin * disk_probe["commit_seconds_p90"]
            initial += initial_visits * read_s + write_s * work["initial_cells"] / min(8, i)
            late += later_visits * read_s + write_s * (later_visits + work["later_reference_cells"] / max(8, 2 * k))
            # Count discovery/refits AND SQL selection overhead with an empirical in-process representative fit.
            profiles.append(dict(initial=i, new=new, work=work, initial_seconds=initial, later_seconds=late))
    for x in profiles:
        clock.check()
        expected = x["work"]["trial_count"]
        # Allocate RAM for dense scores AND temporary weighting/learning matrices.
        ram_need = expected * sum(len(r["days"]) for r in rt.plans) * 40 + est_bytes * 2 + est_bytes * k * 0.4
        t = time.perf_counter()
        rng = np.random.default_rng(91)
        pilot_n = min(expected, 256)
        observed = rng.choice(len(grid.configs), pilot_n, replace=False).tolist()
        grid.propose(min(4, x["new"]), observed, rng.normal(size=pilot_n), p, 7, 1)
        fit = (time.perf_counter() - t) * max(1, expected / pilot_n) * margin
        x["later_seconds"] += fit * (x["work"]["forward_weeks"] * (1 + (x["new"] - 1) // 8) + 1)
        x["initial_seconds"] += fit * (1 + (x["initial"] - 1) // 8)
        capacity = p["cap_hours"] * 3600
        cache_need = x["work"]["trial_count"] * (n - 1) * 180
        x["feasible"] = (
            x["initial_seconds"] <= capacity * p["search_fraction"]
            and x["later_seconds"] <= capacity * p["weekly_fraction"]
            and overhead <= capacity * p["reserve_fraction"]
            and ram_need < info["available_bytes"] * 0.82
            and info["free_disk_bytes"] > disk_gb * 1e9 * 1.15 + cache_need
        )
    feasible = [r for r in profiles if r["feasible"]]
    chosen = max(feasible, key=lambda x: (x["new"], x["initial"])) if feasible else None
    return {
        "status": "PASS" if chosen else "NO_FIT",
        "inputs": rt.identity,
        "machine": info,
        "workers": k,
        "trials": profiles,
        "selected": chosen,
        "timings": measurements,
        "storage_probe": disk_probe,
        "estimated_prepared_gb": disk_gb,
        "overhead_seconds": overhead,
        "compute_seconds_cap": p["cap_hours"] * 3600,
        "note": "Host estimates; true archive/cache I/O and scheduler vary. Only initial-search weeks were timed.",
    }
