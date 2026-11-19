"""Weekly PIT membership and quote calibration. Never use next week's prices to choose its members."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import session_grid, week_chunks
from shpe_research.data.rm9_io import file_hash, read, write
from shpe_research.data.rm9_market import FIELDS, Market, calendar_dates, data_id, ordinal


def build_daily(market, master, jobs=1, emit=None):
    """Dense daily × asset matrices, one column per stable identity, intervals inclusive."""
    dates = calendar_dates(market.p)
    groups = list(master.groupby("asset_id", sort=True))
    ids = [a for a, _ in groups]
    data = {k: np.full((len(dates), len(ids)), np.nan) for k in ("close", "dollar", "adj", "split")}

    def one(item):
        j, (_, group) = item
        for row in group.itertuples():
            lo, hi = max(pd.Timestamp(row.start), dates[0]), min(pd.Timestamp(row.end), dates[-1])
            if lo > hi:
                continue
            bars = market.frame(row.symbol, lo, hi, "daily")
            adj = market.frame(row.symbol, lo, hi, "split")
            if not bars.index.equals(adj.index):
                raise ValueError(f"raw and adjusted daily indexes differ for {row.symbol}")
            if not len(bars):
                continue
            ix = dates.get_indexer(bars.index.tz_convert("America/New_York").tz_localize(None).normalize())
            ok = ix >= 0
            ix, bars, adj = ix[ok], bars.iloc[ok], adj.iloc[ok]
            data["close"][ix, j] = bars.c
            data["dollar"][ix, j] = bars.c * bars.v
            data["adj"][ix, j] = adj.c
            data["split"][ix, j] = adj.c / bars.c
        return j

    with ThreadPoolExecutor(jobs) as pool:
        for done, _ in enumerate(pool.map(one, enumerate(groups)), 1):
            if emit and (done % 100 == 0 or done == len(groups)):
                emit(
                    {
                        "stage": "daily_acquisition",
                        "done": done,
                        "total": len(groups),
                        "http_requests": market.client.requests_sent if market.client else 0,
                    }
                )
    return dates, ids, data


def ranked_pool(day, daily, p, master):
    """Order candidates by nine stratified cells, with stable ID-based priority. All metrics stop before day."""
    dates, ids, data = daily
    n = p["screen_days"]
    end = dates.searchsorted(day)
    if end < n:
        raise ValueError("insufficient pre-cutoff sessions")
    sl = slice(end - n, end)
    c, v, adj = (data[k][sl] for k in ("close", "dollar", "adj"))
    # Listing information as of previous close; no future week status in ranking.
    cutoff = str(dates[end - 1].date())
    live = master.loc[(master.start <= cutoff) & (master.end >= cutoff)].set_index("asset_id")
    good = np.isfinite(c).all(axis=0) & np.isfinite(adj).all(axis=0) & np.isfinite(v).all(axis=0)
    positions = np.flatnonzero(good & np.isin(ids, live.index))
    med = np.median(c[:, positions], axis=0)
    dv = np.median(v[:, positions], axis=0)
    vol = np.std(np.diff(np.log(adj[:, positions]), axis=0), axis=0, ddof=1)
    ix = np.flatnonzero((med >= p["min_price"]) & (dv >= p["min_dollar_volume"]))
    df = pd.DataFrame(
        [
            {
                "id": ids[positions[i]],
                "price": float(med[i]),
                "dv": float(dv[i]),
                "vol": float(vol[i]),
                "symbol": str(live.loc[ids[positions[i]], "symbol"]),
                "lot": int(live.loc[ids[positions[i]], "lot"]),
            }
            for i in ix
        ]
    )
    bins = [[] for _ in range(9)]
    if len(df):
        for lb, part in enumerate(np.array_split(df.sort_values(["dv", "id"]).index.to_numpy(), 3)):
            local = df.loc[part].sort_values(["vol", "id"]).index.to_numpy()
            for vbin, indices in enumerate(np.array_split(local, 3)):
                cell = 3 * lb + vbin
                for row in df.loc[indices].to_dict("records"):
                    bins[cell].append(dict(row, cell=cell))
                bins[cell].sort(key=lambda r: (ordinal(f"{p['seed']}:{r['id']}"), r["id"]))
    stats = {
        "candidates": len(ids),
        "valid_complete_daily_window": int(len(positions)),
        "passed_daily_screens": len(df),
    }
    return bins, stats


def values(market, group, days, ts):
    """Locate each date's alias. No backward price fill. Output columns o,h,l,c,v on the exchange minute grid."""
    arr = np.full((len(ts), 5), np.nan, dtype="f4")
    for row in group.itertuples():
        lo = max(days[0], pd.Timestamp(row.start))
        hi = min(days[-1], pd.Timestamp(row.end))
        if lo > hi:
            continue
        f = market.frame(row.symbol, lo, hi, "minute")
        if f.empty:
            continue
        raw = f.index.asi8
        offsets = np.searchsorted(ts, raw)
        ok = (offsets < len(ts)) & (ts[np.minimum(offsets, len(ts) - 1)] == raw)
        arr[offsets[ok]] = f[FIELDS].iloc[ok].to_numpy("f4")
    valid = np.isfinite(arr[:, 3]) & (arr[:, 4] > 0)
    arr[~valid] = np.nan
    return arr


def pick_week(market, master, groups, daily, days):
    p = market.p
    bins, stats = ranked_pool(days[0], daily, p, master)
    warm = get_calendar().previous_sessions(days[0], p["warmup"])
    ts, _ = session_grid(warm)
    members = []
    rejects = []
    # Round robin redistributes undersubscribed cells. Rotate which cell gets the one remainder.
    offset = ordinal(str(days[0].date()) + str(p["seed"])) % 9
    ordered = [iter(bins[(j + offset) % 9]) for j in range(9)]
    for _ in range(max(map(len, bins), default=0) + 1):
        added = 0
        for group in ordered:
            for row in group:
                a = values(market, groups[row["id"]], warm, ts)
                coverage = float(np.isfinite(a[:, 3]).mean())
                if coverage < p["min_prior_minute_coverage"]:
                    rejects.append([row["id"], "prior_minute_coverage", coverage])
                    continue
                members.append(dict(row, prior_coverage=round(coverage, 6)))
                added += 1
                break
            if len(members) == p["universe"]:
                break
        if len(members) == p["universe"] or not added:
            break
    if not members:
        raise ValueError(f"no eligible stocks at {days[0].date()}; inspect master, feed and coverage")
    # Known split discontinuities: no entries for 20 sessions from first day on new terms.
    # Schedule eligibility never uses these future target-week flags; today's factor flag is used only on that day.
    dates, ids, data = daily
    by_id = {x: j for j, x in enumerate(ids)}
    for r in members:
        col = by_id[r["id"]]
        hist = dates.get_indexer(warm.append(days))
        f = data["split"][hist, col]
        ix = np.flatnonzero(np.abs(np.diff(np.log(f))) > 0.005) + 1
        # Today's close factor cannot block today's trade. Act next session; the separate open-gap guard is causal.
        r["post_split_block"] = [
            str(day.date()) for i, day in enumerate(days, len(warm)) if np.any((ix < i) & (i - ix <= p["warmup"]))
        ]
    # Broad prior-day descriptors for similarity, identical for all candidates and policies.
    pos = dates.get_loc(warm[-1])
    adj = data["adj"][pos - 5 : pos + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.log(adj[-1] / adj[0])
    selected = np.array([by_id[x["id"]] for x in members])
    ret = ret[selected]
    ret = ret[np.isfinite(ret)]
    if not len(ret):
        raise ValueError("no prior regime observations")
    desc = [
        float(np.median(ret)),
        float(np.std(ret)),
        float(np.mean(ret > 0)),
        float(np.log1p(np.median([r["dv"] for r in members]))),
    ]
    return {
        "days": [str(x.date()) for x in days],
        "asof": str(warm[-1].date()),
        "members": sorted(members, key=lambda r: r["id"]),
        "stage_counts": stats,
        "rejected_prior_minutes": rejects,
        "shortfall": p["universe"] - len(members),
        "regime": desc,
    }


def quote_table(m, master, daily):
    """Choose whole sampled windows BEFORE archive start; equalize window contributions; fail on coverage holes."""
    p = m.p
    pool = get_calendar().previous_sessions(p["start"], p["quote_days"])
    rows = []
    for day in pool:
        # Rank regimes on prior daily cross-sectional volatility; stratification only in the pre-study sample.
        i = daily[0].get_loc(day)
        prev = daily[2]["adj"][i - 2 : i]
        r = np.log(prev[-1] / prev[0])
        ok = np.isfinite(r)
        if ok.any():
            rows.append((day, float(np.median(np.abs(r[ok])))))
    if len(rows) < 3 * p["quote_draws_per_regime"]:
        raise ValueError("insufficient distinct pre-study quote days")
    obs = []
    visits = []
    rng = np.random.default_rng(p["seed"])
    for regime, indices in enumerate(np.array_split(np.argsort([x[1] for x in rows], kind="stable"), 3)):
        for ix in sorted(rng.choice(indices, p["quote_draws_per_regime"], replace=False)):
            day = rows[ix][0]
            cells, _ = ranked_pool(day, daily, p, master)
            op, cl = get_calendar().session_bounds(day)
            for cell, choices in enumerate(cells):
                take = min(len(choices), p["quote_symbol_samples"])
                for ind in rng.choice(len(choices), take, replace=False):
                    r = choices[ind]
                    for b, left in enumerate(
                        (op + pd.Timedelta(minutes=8), op + (cl - op) / 2, cl - pd.Timedelta(minutes=12))
                    ):
                        f = m.quotes(r["symbol"], left)
                        if len(f):
                            f = f[(f.bp > 0) & (f.ap >= f.bp) & (f.bs > 0) & (f["as"] > 0)].copy()
                        metrics = []
                        if len(f) > 1:
                            f["t"] = pd.to_datetime(f.t, utc=True, format="mixed").astype("datetime64[ns, UTC]")
                            f = f.sort_values("t").drop_duplicates("t", keep="last")
                            times = f.t.array.asi8
                            # One observation per clock second. Use latest available quote, at most two seconds old.
                            grid = left.value + np.arange(60, dtype=np.int64) * 10**9
                            idx = np.searchsorted(times, grid, side="right") - 1
                            good = (idx >= 0) & (grid - times[np.maximum(idx, 0)] <= 2_000_000_000)
                            paired = good[:-1] & good[1:]
                            prev = f.iloc[idx[:-1][paired]]
                            after = f.iloc[idx[1:][paired]]
                            if len(after) >= 5:
                                mid = (prev.ap.to_numpy() + prev.bp.to_numpy()) / 2
                                lot = r["lot"] if str(day.date()) < "2025-11-03" else 1
                                for px, sz, sign in ((after.ap, after["as"], 1), (after.bp, after.bs, -1)):
                                    adverse = np.maximum((px.to_numpy() / mid - 1) * sign * 10000, 0)
                                    metrics.append(
                                        [
                                            float(np.quantile(adverse, p["quote_p"])),
                                            float(np.quantile(sz.to_numpy() * lot, 0.25)),
                                        ]
                                    )
                        visits.append([str(day.date()), int(regime), r["symbol"], cell, b, bool(metrics)])
                        if metrics:
                            obs.append((cell, b, regime, metrics))
    cube = np.zeros((9, 3, 2, 2))
    counts = []
    for cell, b in itertools.product(range(9), range(3)):
        sample = [x for x in obs if x[0] == cell and x[1] == b]
        regimes = sorted({int(x[2]) for x in sample})
        if len(sample) < p["quote_min_windows"] or len(regimes) < 3:
            raise ValueError(
                f"insufficient distinct quote coverage: cell={cell} time={b}, windows={len(sample)}, regimes={regimes}"
            )
        vals = np.asarray([x[3] for x in sample])
        cube[cell, b, :, 0] = np.maximum(np.quantile(vals[:, :, 0], p["quote_p"], axis=0), p["turnover_floor_bps"])
        cube[cell, b, :, 1] = np.maximum(np.quantile(vals[:, :, 1], 0.25, axis=0), 1)
        counts.append([cell, b, len(sample), regimes])
    return {
        "before": p["start"],
        "latest_sample": max(x[0] for x in visits),
        "cube": cube.tolist(),
        "impact_bps": p["impact_bps"],
        "sample_windows": visits,
        "coverage": counts,
        "source": p["mode"],
    }


def stage(root, p, master_csv=None, approximate=False, jobs=1, emit=print):
    """Acquire resumable symbol periods; freeze complete membership schedule, raw inventory and pre-study cost table."""
    root = Path(root)
    m = Market(root, p)
    master, source = m.asset_master(master_csv, approximate)
    groups = dict(tuple(master.groupby("asset_id", sort=True)))
    m.api() if not m.demo else None
    daily = build_daily(m, master, jobs, emit)
    emit({"stage": "daily_ready", "ids": len(groups), "sessions": len(daily[0])})
    table = root / "quote-costs.json"
    if not table.exists():
        write(table, quote_table(m, master, daily))
    weeks = week_chunks(get_calendar().sessions(p["start"], p["end"]))
    hashes = []
    widths = []
    for j, days in enumerate(weeks):
        path = root / "schedule" / f"{j:04}.json"
        if path.exists():
            item = read(path)
        else:
            item = pick_week(m, master, groups, daily, days)
            # Stage target raw periods too: compute nodes never need API keys or network.
            for r in item["members"]:
                for row in groups[r["id"]].itertuples():
                    lo, hi = max(days[0], pd.Timestamp(row.start)), min(days[-1], pd.Timestamp(row.end))
                    if lo <= hi:
                        m.frame(row.symbol, lo, hi, "minute")
            write(path, item)
        hashes.append(file_hash(path))
        widths.append(len(item["members"]))
        if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(weeks):
            emit({"stage": "membership_ready", "weeks": j + 1, "total": len(weeks), "stocks": len(item["members"])})
    # Use digests already taken at unit write/read. Offline deep verification rehashes transferred bytes.
    inventory = {str(x.relative_to(root).as_posix()): read(x)["sha256"] for x in sorted(root.rglob("*.sha.json"))}
    write(root / "inventory.json", inventory)
    if p["mode"] == "full" and max(widths) < p["universe"]:
        raise ValueError("full archive never reached 1000 stocks: incomplete candidate/coverage source")
    output = {
        "coverage": {
            "min": min(widths),
            "max": max(widths),
            "median": float(np.median(widths)),
            "shortfall_weeks": sum(w < p["universe"] for w in widths),
        },
        "source": source,
        "spec": data_id(p),
        "weeks": len(weeks),
        "schedule_hashes": hashes,
        "cost_hash": file_hash(table),
        "raw_inventory": file_hash(root / "inventory.json"),
    }
    pinpath = root / "archive.json"
    if pinpath.exists() and read(pinpath) != output:
        raise ValueError("sealed archive changed; start a new folder")
    write(pinpath, output)
    if m.client:
        m.client.close()
    return output
