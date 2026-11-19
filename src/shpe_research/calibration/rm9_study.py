"""Chronological, resumable three-policy comparison over one shared archive."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
from threadpoolctl import threadpool_limits

from shpe_research.calibration.objective import AcceptanceRules, acceptance_checks, score_returns
from shpe_research.calibration.policies import effective_sample_size, recency_weights, uniform_weights
from shpe_research.calibration.rm9_engine import account
from shpe_research.calibration.rm9_ops import Stopping
from shpe_research.calibration.rm9_search import Bank, weighted_scores
from shpe_research.contracts.weekly_config import WeeklyConfig
from shpe_research.data.calendar import get_calendar
from shpe_research.data.rm9_io import digest, read, write
from shpe_research.strategies.momentum.breakout.grid import config_id

POLICIES = ("uniform", "recency", "similarity")


def weights(p, plans, k, h):
    dates = [d for w in plans[:h] for d in w["days"]]
    uni = uniform_weights(len(dates))
    rec = recency_weights(dates, plans[k]["asof"], half_life_days=p["half_life"])
    history = np.array([w["regime"] for w in plans[:h] for _ in w["days"]])
    current = np.array(plans[k]["regime"])
    iqr = np.diff(np.quantile(history, [0.25, 0.75], axis=0), axis=0)[0]
    keep = iqr > 1e-12
    dist = np.sum(((history[:, keep] - current[keep]) / iqr[keep]) ** 2, axis=1)

    def mix(bw):
        v = np.exp(-0.5 * (dist - dist.min()) / (bw * bw))
        return (1 - p["similarity_mix"]) * v / v.sum() + p["similarity_mix"] * uni

    lo, hi = 1e-7, max(1.0, float(np.sqrt(dist.max())))
    for _ in range(38):
        mid = (lo + hi) / 2
        if effective_sample_size(mix(mid)) < p["similarity_ess"] * len(dates):
            lo = mid
        else:
            hi = mid
    return {"uniform": uni, "recency": rec, "similarity": mix(hi)}


def fiat(cfg):
    if cfg is None:
        return None
    return {
        "config_id": config_id(cfg),
        "subset": list(cfg.subset),
        "grid_version": cfg.grid_version,
        "params": cfg.as_dict(),
    }


def select(bank, grid, plans, p, k):
    """Only data strictly before k are accessible here. All policies use identical unweighted validation."""
    h = k - p["validation"]
    x, y = bank.offset[h], bank.offset[k]
    n = len(bank.idx)
    wts = weights(p, plans, k, h)
    rules = AcceptanceRules(**p["accept"])
    out = {}
    ids = np.array([config_id(grid.configs[i]) for i in bank.idx])
    for name, w in wts.items():
        scores = weighted_scores(bank.r[:n, :x], w)
        slots = np.lexsort((ids, -scores))[: p["finalists"]]
        finalists = []
        for s in slots:
            v = bank.r[s, x:y]
            val = score_returns(v, uniform_weights(len(v)))
            checks = [asdict(c) for c in acceptance_checks(val, rules)]
            count = int(bank.trade[s, x:y].sum())
            checks.append(
                {
                    "name": "validation_trades",
                    "passed": count >= p["min_validation_trades"],
                    "value": count,
                    "threshold": p["min_validation_trades"],
                }
            )
            finalists.append(
                {
                    "idx": bank.idx[s],
                    "id": str(ids[s]),
                    "search_score": float(scores[s]),
                    "validation": val.score,
                    "trades": count,
                    "checks": checks,
                    "pass": all(c["passed"] for c in checks),
                }
            )
        passed = sorted((f for f in finalists if f["pass"]), key=lambda f: (-f["validation"], f["id"]))
        out[name] = {
            "idx": passed[0]["idx"] if passed else None,
            "ess": effective_sample_size(w),
            "finalists": finalists,
        }
    return out


def empty(cash, n):
    return {
        "r": [0.0] * n,
        "gross": [0.0] * n,
        "trades": [0] * n,
        "costs": [0.0] * n,
        "reject": {},
        "ambiguous": 0,
        "stock_concentration": 0.0,
        "ending_equity": cash,
    }


def export_week(rt, k, rows, prof):
    p, plans = rt.p, rt.plans
    v = k - p["validation"]
    folder = rt.root / "weekly-configs"
    commit = read(rt.root / "provenance.json").get("git_head") or "source-" + rt.identity["source"]
    for name, r in rows.items():
        picked = r["selected"]
        a = asdict(account(p))
        # Fail closed on an old platform which only implements limit_markup + flat bps.
        a["fill"]["kind"] = "limit_markup_sampled_sip_v2"
        cfg = WeeklyConfig.model_validate(
            {
                "run_id": prof["seal"],
                "code_commit": commit,
                "data_version": rt.identity["archive"],
                "protocol_hash": rt.identity["spec"],
                "policy": name,
                "decision_cutoff": get_calendar().session_bounds(plans[k]["asof"])[1].isoformat(),
                "search_range": {"start": plans[0]["days"][0], "end": plans[v - 1]["days"][-1]},
                "validation_range": {"start": plans[v]["days"][0], "end": plans[k - 1]["days"][-1]},
                "activation": {"start": plans[k]["days"][0], "end": plans[k]["days"][-1]},
                "universe": [{"asset_id": a["id"], "symbol": a["symbol"]} for a in plans[k]["members"]],
                "selected": picked,
                "no_deployment": None if picked else {"reason": "no finalist passed validation"},
                "acceptance": {
                    "passed": bool(picked),
                    "checks": next(
                        (f["checks"] for f in r["finalists"] if picked and f["id"] == picked["config_id"]), []
                    ),
                },
                "ess": r["ess"],
                "seeds": [p["seed"]],
                "account": a,
                "evaluations": r["evaluations"],
                "runtime": r["runtime"],
            }
        )
        write(folder / f"{k:04}-{name}.json", cfg.model_dump(mode="json"))
    # The sidecar always records the exact source tree, including uncommitted changes.


def drive(rt, grid, prof, pause_after=None):
    import time

    import psutil

    plans = rt.plans
    full = prof["selected"]
    count = full["work"]["trial_count"]
    bank = Bank(rt.root / "archive.sqlite3", count, plans)
    try:
        _loop(rt, grid, prof, bank, pause_after, time, psutil)
    finally:
        bank.db.close()


def _loop(rt, grid, prof, bank, pause_after, time, psutil):
    p, clock, plans = rt.p, rt.clock, rt.plans
    full = prof["selected"]
    h, v = p["history"], p["validation"]
    first = h + v
    cap = full["work"]["trial_count"]
    prev = None
    fresh = 0
    committed_cells = int(bank.done.sum())
    expected_cells = full["work"]["initial_cells"] + full["work"]["later_reference_cells"]
    start_cells = committed_cells
    timer = time.perf_counter()
    with ThreadPoolExecutor(prof["workers"]) as pool, threadpool_limits(limits=1):
        for k in range(first, len(plans)):
            stored = bank.db.execute("select body from decision where cut=?", (k,)).fetchone()
            if stored:
                prev = json.loads(stored[0])
                export_week(rt, k, prev, prof)  # regenerate any file lost after the atomic SQLite commit
                continue
            mark = time.perf_counter()
            hist = k - v
            phase = "primary" if k >= len(plans) - p["primary"] else "development"
            clock.status(
                "weekend",
                forward_week=k,
                first_forward=first,
                total_weeks=len(plans),
                completed_forward=k - first,
                period=phase,
                archive_configs=len(bank.idx),
            )
            prior = [s for (s,) in bank.db.execute("select slot from trial where cut<? order by slot", (k,))]

            def fill(slots, lo, hi):
                nonlocal committed_cells
                for week in range(lo, hi):
                    clock.check()
                    missing = [s for s in slots if not bank.done[s, week]]
                    if not missing:
                        continue
                    wk = rt.week(week)
                    for off in range(0, len(missing), max(8, prof["workers"] * 2)):
                        clock.check()
                        sel = missing[off : off + max(8, prof["workers"] * 2)]
                        records = pool.map(wk.simulate, [grid.configs[bank.idx[s]] for s in sel])
                        bank.store(week, zip(sel, records, strict=True))
                        committed_cells += len(sel)
                        rate = (committed_cells - start_cells) / max(0.001, time.perf_counter() - timer)
                        with clock.lock:
                            clock.state.update(
                                replay_week=week,
                                archive_configs=len(bank.idx),
                                target_configs=cap,
                                completed_cells=committed_cells,
                                total_cells=expected_cells,
                                cells_per_second=round(rate, 3),
                                reference_eta_seconds=round((expected_cells - committed_cells) / max(0.0001, rate)),
                            )

            fill(prior, 0, hist)
            target = full["initial"] if k == first else full["new"]
            slots = prior.copy()
            built = 0
            num = 0
            while built < target:
                clock.check()
                n = min(14 if k == first and num == 0 else 8, target - built)

                def propose(n=n, num=num, slots=tuple(slots), hist=hist, k=k):
                    scores = (
                        weighted_scores(bank.r[list(slots), : bank.offset[hist]], uniform_weights(bank.offset[hist]))
                        if slots
                        else np.empty(0)
                    )
                    return grid.propose(n, [bank.idx[s] for s in slots], scores, p, k, num)

                cohort = bank.cohort(grid, k, num, propose)
                if len(cohort) != n:
                    raise ValueError("partial proposal cohort")
                fill(cohort, 0, hist)
                slots.extend(cohort)
                built += len(cohort)
                num += 1
            fill(slots, hist, k)
            choices = select(bank, grid, plans, p, k)
            weekly = {}
            for name, pick in choices.items():
                # No call to the activation week's price data until all three decisions are finished.
                idx = pick["idx"]
                cfg = grid.configs[idx] if idx is not None else None
                base = prev[name]["base"]["ending_equity"] if prev else p["account"]["starting_cash"]
                stress = prev[name]["stress"]["ending_equity"] if prev else p["account"]["starting_cash"]
                if cfg:
                    wk = rt.week(k)
                    b = wk.simulate(cfg, base)
                    st = wk.simulate(cfg, stress, 2.0)
                else:
                    b, st = empty(base, len(plans[k]["days"])), empty(stress, len(plans[k]["days"]))
                work = len(slots) * k
                prev_work = len(prior) * (k - 1)
                weekly[name] = {
                    "phase": phase,
                    "dates": plans[k]["days"],
                    "selected": fiat(cfg),
                    "ess": pick["ess"],
                    "finalists": pick["finalists"],
                    "base": b,
                    "stress": st,
                    "start_equity": base,
                    "stress_start_equity": stress,
                    "archive_configs": len(slots),
                    "archive_snapshot": digest([bank.idx[s] for s in slots]),
                    "evaluations": {"logical": work, "actual": work - prev_work},
                    "runtime": {
                        "seconds": time.perf_counter() - mark,
                        "peak_memory_mb": max(clock.peak_rss_mb, psutil.Process().memory_info().rss / 1e6),
                    },
                }
            with bank.db:
                bank.db.execute("insert into decision values(?,?)", (k, json.dumps(weekly, allow_nan=False)))
            export_week(rt, k, weekly, prof)
            prev = weekly
            fresh += 1
            clock.status(
                "week_complete",
                forward_week=k,
                completed_forward=k - first + 1,
                total_forward=len(plans) - first,
                archive_configs=len(slots),
                policy_trades={s: sum(r["base"]["trades"]) for s, r in weekly.items()},
            )
            if pause_after and fresh >= pause_after and k + 1 < len(plans):
                raise Stopping("requested demo checkpoint; run the same command again to resume")
