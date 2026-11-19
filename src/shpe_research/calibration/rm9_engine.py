"""Build each week's target data once, persist bit-packed filters, then share read-only maps across candidates."""

from __future__ import annotations

import itertools
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import session_grid
from shpe_research.data.rm9_io import digest, file_hash, read, write
from shpe_research.data.rm9_market import FIELDS
from shpe_research.data.rm9_universe import values
from shpe_research.strategies.momentum.breakout.grid import DEFAULT_DOMAIN
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.trade.account_fast import PreparedPanel, simulate_account_fast
from shpe_research.trade.account_types import AccountConfig, ExitPackage, FillModel, MarketPanel, SignalMask

KEYS = [
    (name, *vals) for name, params in DEFAULT_DOMAIN.components for vals in itertools.product(*(v for _, v in params))
]
KEY_INDEX = {k: i for i, k in enumerate(KEYS)}
NAMES = ("o", "h", "l", "c", "v", "atr", "liq", "eligible", "rank", "bits", "session_index", "cells")


def component(cache, k, ready):
    name, *a = k
    if name == "breakout":
        return cache.breakout(*a) & ready[f"breakout_n{a[0]}"] & ready["atr14"]
    if name == "adx":
        return cache.adx_sig(*a) & ready[f"adx_a{a[0]}"]
    if name == "ema":
        return cache.ema_sig(*a) & ready[f"ema_f{a[0]}_s{a[1]}"]
    if name == "rvol":
        return cache.rvol_sig(*a) & ready[f"rvol_w{a[0]}"]
    return cache.vwap_sig(*a) & ready["vwap"]


def account(p):
    a = dict(p["account"])
    a["fill"] = FillModel(**a["fill"])
    a["exit"] = ExitPackage(**a["exit"])
    return AccountConfig(**a, cost_bps=0)


def compile_week(m, groups, plan, dest, p, block=24, progress=None, should_stop=None, fixture_block=None):
    """Compile one immutable week in symbol blocks. Every target minute uses only earlier observations."""
    dest = Path(dest)
    if (dest / "meta.json").exists():
        return read(dest / "meta.json")
    start = time.monotonic()
    tmp = dest.with_name(dest.name + ".building")
    tmp.mkdir(parents=True, exist_ok=True)
    days = pd.DatetimeIndex(plan["days"])
    warm = get_calendar().previous_sessions(days[0], p["warmup"])
    all_days = warm.append(days)
    ts, sidx = session_grid(all_days)
    starts = np.flatnonzero(np.r_[True, np.diff(sidx) != 0])
    lens = np.diff(np.r_[starts, len(ts)])
    prefix = starts[len(warm)]
    n = len(plan["members"])
    t = int(len(ts) - prefix)
    ids = [r["id"] for r in plan["members"]]
    shape = (n, t)
    layouts = {
        **{k: (shape, "f4") for k in (*FIELDS, "atr", "liq")},
        "eligible": (shape, bool),
        "bits": ((len(KEYS), n, (t + 7) // 8), "u1"),
        "rank": ((len(days), n), "i4"),
        "session_index": ((t,), "i4"),
        "cells": ((n,), "i4"),
    }
    needed = sum(int(np.prod(s)) * np.dtype(dt).itemsize for s, dt in layouts.values())
    if shutil.disk_usage(tmp).free < needed * 1.1:
        raise OSError("insufficient disk for next prepared week")
    maps = {
        k: np.lib.format.open_memmap(tmp / f"{k}.npy", mode="w+", dtype=dt, shape=sh) for k, (sh, dt) in layouts.items()
    }
    maps["session_index"][:] = sidx[prefix:] - len(warm)
    maps["cells"][:] = [r["cell"] for r in plan["members"]]
    daydv = np.zeros((n, len(all_days)), dtype=np.float64)
    observed = expected = 0
    for lo in range(0, n, block):
        if should_stop and should_stop():
            raise TimeoutError("interrupted while compiling a week; restart from its last complete unit")
        hi = min(n, lo + block)
        a = (
            np.stack([values(m, groups[x], all_days, ts) for x in ids[lo:hi]])
            if fixture_block is None
            else fixture_block(lo, hi, all_days, ts, sidx)
        )
        mats = [np.ascontiguousarray(a[:, :, i]) for i in range(5)]
        oo, hh, ll, cc, vv = mats
        eligible = np.isfinite(cc) & (vv > 0)
        for j, r in enumerate(plan["members"][lo:hi]):
            listing = groups[r["id"]]
            for d, day in enumerate(days, len(warm)):
                date = str(day.date())
                sl = slice(starts[d], starts[d] + lens[d])
                if ((listing.start <= date) & (listing.end >= date)).any():
                    expected += int(lens[d])
                    observed += int(eligible[j, sl].sum())
                if date in r["post_split_block"]:
                    eligible[j, sl] = False
                # A discontinuity apparent at the first regular print is observable, regardless of its cause.
                previous = cc[j, starts[d - 1] : starts[d]]
                previous = previous[np.isfinite(previous)]
                if (
                    len(previous)
                    and np.isfinite(oo[j, sl.start])
                    and abs(np.log(oo[j, sl.start] / previous[-1])) > 0.30
                ):
                    eligible[j, sl] = False
        panel = MarketPanel(tuple(ids[lo:hi]), *mats, sidx, eligible)
        cache = IndicatorCache(panel)
        ready = cache.readiness()
        for index, key in enumerate(KEYS):
            sig = component(cache, key, ready)[:, prefix:]
            maps["bits"][index, lo:hi] = np.packbits(sig, axis=1)
        maps["atr"][lo:hi] = cache.atr14()[:, prefix:]
        for k, mat in zip(FIELDS, mats, strict=True):
            maps[k][lo:hi] = mat[:, prefix:]
        maps["eligible"][lo:hi] = eligible[:, prefix:]
        daydv[lo:hi] = np.add.reduceat(np.nan_to_num(cc.astype("f8") * vv), starts, axis=1)
        # Previous-session volume at this clock minute. Missing/short-session observations do not count.
        for d in range(len(warm), len(all_days)):
            length = lens[d]
            hist = np.full((d - max(0, d - p["account"]["liquidity_lookback"]), hi - lo, length), np.inf, "f4")
            for j, h in enumerate(range(max(0, d - p["account"]["liquidity_lookback"]), d)):
                ln = min(length, lens[h])
                x = vv[:, starts[h] : starts[h] + ln]
                hist[j, :, :ln] = np.where(np.isfinite(x), x, np.inf)
            count = np.isfinite(hist).sum(axis=0)
            hist.sort(axis=0)
            low = np.take_along_axis(hist, np.maximum((count - 1) // 2, 0)[None], axis=0)[0]
            high = np.take_along_axis(hist, (count // 2)[None], axis=0)[0]
            target = slice(starts[d] - prefix, starts[d] - prefix + length)
            maps["liq"][lo:hi, target] = np.where(count >= p["account"]["liquidity_min_obs"], (low + high) * 0.5, 0.0)
        del cache, ready
        if progress:
            progress(hi, n)
    for j, d in enumerate(range(len(warm), len(all_days))):
        score = np.median(daydv[:, d - p["account"]["liquidity_lookback"] : d], axis=1)
        order = np.lexsort((np.array(ids), -score))
        maps["rank"][j, order] = np.arange(n)
    coverage = observed / max(expected, 1)
    if coverage < p["min_dataset_coverage"]:
        raise ValueError(
            f"DATA_QUALITY_STOP {days[0].date()}: {coverage:.1%} coverage, check raw archive. Do not skip week."
        )
    for arr in maps.values():
        arr.flush()
    del arr, maps
    meta = {
        "ids": ids,
        "days": plan["days"],
        "asof": plan["asof"],
        "members": n,
        "observed_fraction": coverage,
        "minutes": t,
        "bytes": needed,
        "compile_seconds": time.monotonic() - start,
        "selection": digest(plan),
        "hashes": {name: file_hash(tmp / f"{name}.npy") for name in NAMES},
    }
    write(tmp / "meta.json", meta)
    os.replace(tmp, dest)
    return meta


class Week:
    """Read-only maps of one compiled target week; all workers use the same prepared data."""

    def __init__(self, path, table, p):
        self.meta = read(path / "meta.json")
        a = {k: np.load(path / f"{k}.npy", mmap_mode="r", allow_pickle=False) for k in NAMES}
        self.a = a
        panel = MarketPanel(tuple(self.meta["ids"]), *(a[k] for k in FIELDS), a["session_index"], a["eligible"])
        starts = np.flatnonzero(np.r_[True, np.diff(panel.session_index) != 0])
        lens = np.diff(np.r_[starts, panel.n_minutes])
        self.cfg = account(p)
        self.prepared = PreparedPanel(
            panel,
            *(a[k] for k in FIELDS),
            panel.eligible,
            starts,
            lens,
            a["rank"],
            self.cfg.ranking,
            self.cfg.liquidity_lookback,
        )
        self.quote = (np.array(table["cube"], dtype="f8"), a["cells"], a["liq"], table["impact_bps"])

    def simulate(self, cfg, starting=None, stress=1.0, *, ledger=False):
        params = cfg.as_dict()
        keys = [(n, *(params[f"{n}.{key}"] for key, _ in DEFAULT_DOMAIN.params_of(n))) for n in cfg.subset]
        packed = self.a["bits"][KEY_INDEX[keys[0]]].copy()
        for key in keys[1:]:
            np.bitwise_and(packed, self.a["bits"][KEY_INDEX[key]], out=packed)
        sig = np.unpackbits(packed, axis=1, count=self.prepared.panel.n_minutes).view(np.bool_)
        acct = replace(
            self.cfg,
            starting_cash=self.cfg.starting_cash if starting is None else starting,
            search_replay_reference_equity=self.cfg.starting_cash if starting is None else None,
        )
        quote = self.quote
        if stress != 1.0:
            cube = quote[0].copy()
            cube[..., 0] *= stress
            quote = (cube, *quote[1:3], quote[3] * stress)
        diag = {}
        result = simulate_account_fast(
            self.prepared,
            SignalMask(sig, self.a["atr"]),
            acct,
            quote_inputs=quote,
            diagnostics=diag,
            return_ledger=ledger,
        )
        if result.unresolved_positions:
            raise ValueError(
                f"UNRESOLVED_EXIT {self.meta['days'][0]}: no executable exit; inspect data, no fabricated P&L"
            )
        r = result.daily_returns
        balances = (
            acct.starting_cash * np.r_[1.0, np.cumprod(1 + r)[:-1]] if starting is not None else acct.starting_cash
        )
        extra = {}
        if ledger:
            ts, _ = session_grid(pd.DatetimeIndex(self.meta["days"]))
            extra["ledger"] = [
                {
                    "asset_id": e.asset_id,
                    "entry_utc": pd.Timestamp(ts[e.entry_minute], tz="UTC").isoformat(),
                    "exit_utc": pd.Timestamp(ts[e.exit_minute], tz="UTC").isoformat(),
                    "qty": e.qty,
                    "exit_reason": e.exit_reason,
                    "entry_price": float(e.entry_price),
                    "exit_price": float(e.exit_price),
                    "net_pnl": float(e.net_pnl),
                }
                for e in result.ledger
            ]
        return {
            **extra,
            "r": r.tolist(),
            "gross": (r + np.array(diag["costs"]) / balances).tolist(),
            "trades": diag["daily_trades"],
            "costs": diag["costs"],
            "reject": diag["rejections"],
            "ambiguous": result.ambiguous_bars,
            "stock_concentration": result.summary["max_stock_share"],
            "ending_equity": float(acct.starting_cash * np.prod(1 + r)),
        }
