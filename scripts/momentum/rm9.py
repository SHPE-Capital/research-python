# ruff: noqa: E402 -- set the numerical thread limits before importing NumPy or sklearn.
"""Entry point for Christopher's RM9 research handoff. No broker order API is used."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# One math thread per independent candidate. Set before NumPy/sklearn/Numba load.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import pandas as pd
from dotenv import load_dotenv

from shpe_research.calibration.rm9_ops import (
    Clock,
    Runtime,
    Stopping,
    benchmark,
    load_archive,
    resources,
    single_writer,
)
from shpe_research.calibration.rm9_report import compare_ledgers, report
from shpe_research.calibration.rm9_search import Grid
from shpe_research.calibration.rm9_spec import REPO, spec
from shpe_research.calibration.rm9_study import drive
from shpe_research.data.rm9_io import digest, file_hash, pin, read, write
from shpe_research.data.rm9_universe import stage


def emit(row):
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def status(root, code, detail, **kw):
    write(root / "status.json", dict(status=code, detail=detail, **kw))
    emit(dict(status=code, detail=detail, **kw))


def pinned_profile(root):
    profile = read(root / "machine-profile.json")
    if profile["seal"] != digest({k: v for k, v in profile.items() if k != "seal"}):
        raise ValueError("machine profile changed since benchmark")
    return profile


def git_state():
    try:
        return {
            "git_head": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_head": None}


def process(args, cmd, arc, root, p):
    if cmd == "status":
        found = False
        for name in ("status", "progress", "verification"):
            if (root / f"{name}.json").exists():
                emit({name: read(root / f"{name}.json")})
                found = True
        if not found:
            emit({"status": "NOT_STARTED", "job": str(root)})
        return 0
    if cmd == "audit-data":
        a, plans, q = load_archive(arc, p, deep=args.deep)
        emit(
            {
                "status": "PASS",
                "source": a["source"],
                "coverage": a["coverage"],
                "weeks": len(plans),
                "quote_strata": len(q["coverage"]),
                "latest_quote": q["latest_sample"],
                "deep_byte_check": args.deep,
            }
        )
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with single_writer(root / "writer.lock"):
        if cmd == "verify":
            a, _, _ = load_archive(arc, p, deep=args.deep)
            prof = pinned_profile(root)
            from shpe_research.calibration.rm9_spec import source_id

            if prof["inputs"] != {"source": source_id(), "spec": digest(p), "archive": digest(a)}:
                raise ValueError("source or data identity mismatch")
            if read(root / "status.json")["status"] != "COMPLETE":
                raise ValueError("cannot verify an incomplete run")
            old = read(root / "report.json")
            output_seal = read(root / "output-seal.json")
            for name, h in output_seal.items():
                if file_hash(root / name) != h:
                    raise ValueError(f"failed output integrity: {name}")
            new = report(root, p, prof, check_only=True)
            if new["scientific_digest"] != old["scientific_digest"] or new != old:
                raise ValueError("recomputed report differs")
            if args.deep:
                from shpe_research.calibration.rm9_engine import NAMES

                folders = sorted((root / "prepared").glob("w[0-9][0-9][0-9][0-9]"))
                required = {f"w{i:04}" for i in range(a["weeks"] - 1)}
                if not required.issubset({f.name for f in folders}):
                    raise ValueError("missing a required compiled week")
                for folder in folders:
                    meta = read(folder / "meta.json")
                    i = int(folder.name[1:])
                    if meta["selection"] != digest(read(arc / "schedule" / f"{i:04}.json")) or set(
                        meta["hashes"]
                    ) != set(NAMES):
                        raise ValueError(f"compiled metadata or week selection mismatch: {folder}")
                    for name, h in meta["hashes"].items():
                        if file_hash(folder / f"{name}.npy") != h:
                            raise ValueError(f"damaged prepared array {folder}/{name}")
            verdict = {
                "status": "PASS",
                "checks": [
                    "frozen inputs",
                    "complete schedule",
                    "actual trial counts",
                    "common archive",
                    "date and equity continuity",
                    "gross/net cost reconciliation",
                    "report regeneration",
                ],
                "deep_byte_check": args.deep,
                "scientific_digest": new["scientific_digest"],
            }
            write(root / "verification.json", verdict)
            emit(verdict)
            return 0
        if cmd == "run" and (root / "status.json").exists() and read(root / "status.json")["status"] == "COMPLETE":
            return process_completed(arc, root, p)
        clock = Clock(root, p["cap_hours"], args.job_minutes, diagnostic=(cmd == "ledger"))
        try:
            clock.check()
            rt = Runtime(arc, root, p, clock)
            pin(root / "archive-source.json", rt.a["source"])
            if cmd == "ledger":
                prof = pinned_profile(root)
                if read(root / "status.json")["status"] != "COMPLETE" or prof["inputs"] != rt.identity:
                    raise ValueError("ledger export needs a completed matching run")
                grid = Grid()
                with sqlite3.connect(root / "archive.sqlite3") as db:
                    row = db.execute("select body from decision where cut=?", (args.week,)).fetchone()
                if not row:
                    raise ValueError("no decision at that week")
                r = json.loads(row[0])[args.policy]
                f = next((x for x in r["finalists"] if r["selected"] and x["id"] == r["selected"]["config_id"]), None)
                ledger = (
                    rt.week(args.week).simulate(grid.configs[f["idx"]], r["start_equity"], ledger=True)["ledger"]
                    if f
                    else []
                )
                file = root / f"ledger-{args.week}-{args.policy}.csv"
                pd.DataFrame(
                    ledger,
                    columns=[
                        "asset_id",
                        "entry_utc",
                        "exit_utc",
                        "qty",
                        "exit_reason",
                        "entry_price",
                        "exit_price",
                        "net_pnl",
                    ],
                ).to_csv(file, index=False)
                emit({"status": "EXPORTED", "file": str(file), "round_trips": len(ledger)})
                return 0
            grid = Grid()
            if not (root / "machine-profile.json").exists():
                prof = benchmark(rt, grid, args.workers)
                prof["seal"] = digest(prof)
                write(root / "machine-profile.json", prof)
            else:
                prof = pinned_profile(root)
            if prof["inputs"] != rt.identity:
                raise ValueError("benchmark belongs to different source or inputs")
            if prof["status"] != "PASS":
                status(
                    root, "NO_FIT", "The frozen workload ladder does not fit. Inspect machine-profile.json; do not run."
                )
                return 3
            if prof["workers"] > resources(root)["cores"]:
                raise ValueError("resumed allocation has fewer CPU cores than the frozen worker count")
            pin(
                root / "run-manifest.json",
                {
                    "schema": "rm9.run.2",
                    "inputs": rt.identity,
                    "profile": prof["seal"],
                    "protocol": p,
                    "selected_budget": prof["selected"],
                },
            )
            if not (root / "provenance.json").exists():
                write(
                    root / "provenance.json",
                    dict(**git_state(), python=platform.python_version(), machine=prof["machine"]),
                )
            if cmd == "benchmark":
                # A redundant benchmark invocation must never downgrade a completed study to readiness.
                if not (root / "status.json").exists():
                    status(
                        root,
                        "READY",
                        "Frozen benchmark passed. Inspect the profile, then run.",
                        workers=prof["workers"],
                        initial=prof["selected"]["initial"],
                        weekly=prof["selected"]["new"],
                    )
                return 0
            status(root, "RUNNING", "Resuming complete transactions in the common research archive.")
            drive(rt, grid, prof, args.pause_after)
            clock.check()
            clock.status("reconcile")
            done = report(root, p, prof)
            clock.check()
            write(
                root / "platform-costs.json",
                {
                    "model": "limit_markup_sampled_sip_v2",
                    "quote_table": rt.qt,
                    "archive_cost_sha256": rt.a["cost_hash"],
                    "source_tree_sha256": rt.identity["source"],
                    "supported_by_club_platform": "requires independent ledger agreement",
                },
            )
            files = [
                "report.json",
                "results.html",
                "daily.csv",
                "weekly.csv",
                "trials.csv",
                "platform-costs.json",
                *(str(f.relative_to(root).as_posix()) for f in sorted((root / "weekly-configs").glob("*.json"))),
            ]
            write(root / "output-seal.json", {name: file_hash(root / name) for name in files})
            clock.status(
                "complete", forward_weeks=done["forward_weeks"], actual_cells=done["actual_reference_config_weeks"]
            )
            status(
                root,
                "COMPLETE",
                "Research paths and costs reconciled.",
                mode=p["mode"],
                weeks=done["forward_weeks"],
                scientific_digest=done["scientific_digest"],
            )
            return 0
        except Stopping as e:
            code = "BUDGET_EXHAUSTED" if "TOTAL_COMPUTE_CAP" in str(e) else "PAUSED"
            if cmd != "ledger":
                status(root, code, str(e))
            return 4 if code == "BUDGET_EXHAUSTED" else 75
        except Exception as e:
            if cmd != "ledger":
                status(root, "FAILED", f"{type(e).__name__}: {e}")
            raise
        finally:
            clock.close()


def process_completed(arc, root, p):
    from shpe_research.calibration.rm9_spec import source_id

    a, _, _ = load_archive(arc, p)
    prof = pinned_profile(root)
    if prof["inputs"] != {"source": source_id(), "archive": digest(a), "spec": digest(p)}:
        raise ValueError("completed run belongs to different inputs")
    emit({"status": "COMPLETE", "detail": "Already complete. Use verify to recheck independently."})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "demo",
            "probe",
            "plan",
            "stage",
            "audit-data",
            "benchmark",
            "run",
            "verify",
            "status",
            "ledger",
            "compare-ledgers",
        ],
    )
    parser.add_argument("--mode", choices=["demo", "pilot", "full"], default="demo")
    parser.add_argument("--root", type=Path, help="default runs/rm9-<mode>; archive/ and job/ inside")
    parser.add_argument(
        "--archive", type=Path, help="external staged data path; share read-only across independent runs"
    )
    parser.add_argument("--master", type=Path, help="dated instruments CSV; format in handoff")
    parser.add_argument(
        "--approximate-master", action="store_true", help="permit approximate present active+inactive directory"
    )
    parser.add_argument("--workers", type=int, help="upper bound for benchmark CPU workers")
    parser.add_argument("--pull-workers", type=int, default=2)
    parser.add_argument(
        "--job-minutes", type=float, help="checkpoint after this job slice; never extends total compute budget"
    )
    parser.add_argument(
        "--pause-after", type=int, help="checkpoint after N new weekends; restart same run without flag"
    )
    parser.add_argument("--deep", action="store_true", help="rehash every staged and compiled byte")
    parser.add_argument("--week", type=int)
    parser.add_argument("--policy", choices=["uniform", "recency", "similarity"], default="uniform")
    parser.add_argument("--research-ledger", type=Path)
    parser.add_argument("--platform-ledger", type=Path)
    args = parser.parse_args(argv)
    if args.command == "demo" and args.mode != "demo":
        parser.error("demo can never run a real profile")
    for v in (args.workers, args.pull_workers, args.job_minutes, args.pause_after):
        if v is not None and v <= 0:
            parser.error("limits must be positive")
    if args.master and args.approximate_master:
        parser.error("choose one candidate source")
    p = spec(args.mode)
    base = (args.root or Path("runs") / f"rm9-{args.mode}").resolve()
    arc = (args.archive or base / "archive").resolve()
    root = base / "job"
    if args.command == "compare-ledgers":
        if not args.research_ledger or not args.platform_ledger:
            parser.error("supply both ledger CSV paths")
        r = compare_ledgers(args.research_ledger, args.platform_ledger)
        emit(r)
        return 0 if r["status"] == "PASS" else 5
    if args.command == "probe":
        from shpe_research.data.alpaca import AlpacaDataClient
        from shpe_research.data.calendar import get_calendar

        load_dotenv(REPO / ".env", override=False)
        d = get_calendar().previous_sessions(p["start"], 1)[0]
        a, b = get_calendar().session_bounds(d)
        with AlpacaDataClient() as client:
            bars = client.bars("SPY", a, b, "1Min", "raw", "sip")
            quotes = client.ticks(
                "quotes", "SPY", a + pd.Timedelta(minutes=9), a + pd.Timedelta(minutes=9, seconds=10), "sip"
            )
            good = sum(1 for x in quotes if 0 < float(x.get("bp", 0)) <= float(x.get("ap", 0)))
            obj = {
                "status": "PASS" if bars and good else "FAIL",
                "date": str(d.date()),
                "symbol": "SPY",
                "feed": "sip",
                "minute_bars": len(bars),
                "valid_two_sided_quotes": good,
                "requests": client.requests_sent,
                "orders": 0,
            }
        write(base / "probe.json", obj)
        emit(obj)
        return 0 if obj["status"] == "PASS" else 3
    if args.command == "plan":
        emit({"protocol": p, "archive": str(arc), "job": str(root), "network_calls": 0})
        return 0
    if args.command in ("demo", "stage"):
        load_dotenv(REPO / ".env", override=False)
        with single_writer(arc / "staging.lock"):
            if (arc / "archive.json").exists():
                seal, _, _ = load_archive(arc, p)
                if args.master and seal["source"].get("input_sha256") != file_hash(args.master):
                    raise ValueError("different dated master supplied for an already sealed archive")
            else:
                t = time.perf_counter()
                a = stage(arc, p, args.master, args.approximate_master, args.pull_workers, emit)
                emit(
                    {
                        "stage": "raw_sealed",
                        "weeks": a["weeks"],
                        "outside_compute_seconds": round(time.perf_counter() - t, 2),
                    }
                )
        if args.command == "stage":
            return 0
    if args.command == "ledger" and args.week is None:
        parser.error("supply --week (zero-based archive index)")
    if args.command == "demo":
        code = process(args, "run", arc, root, p)
        if code == 0:
            return process(args, "verify", arc, root, p)
        return code
    return process(args, args.command, arc, root, p)


if __name__ == "__main__":
    sys.exit(main())
