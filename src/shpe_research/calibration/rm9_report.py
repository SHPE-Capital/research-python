"""Reconcile persisted portfolios and issue a machine-readable technical verdict separate from financial findings."""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.calibration.rm9_study import POLICIES
from shpe_research.data.rm9_io import digest, file_hash, read, write
from shpe_research.evaluation.stats import deflated_sharpe, paired_block_bootstrap


def stats(rows, trials):
    r = rows.net.to_numpy()
    curve = np.r_[1.0, np.cumprod(1 + r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sr = float(r.mean() / sd * np.sqrt(252)) if sd > 1e-12 else None
    ds = None
    if sr is not None and len(r) > 3:
        z = (r - r.mean()) / r.std()
        val = deflated_sharpe(sr, max(1, trials), len(r), float((z**3).mean()), float((z**4).mean()))
        ds = float(val) if np.isfinite(val) else None
    return {
        "days": len(r),
        "weeks": int(rows.week.nunique()),
        "net_return": float(curve[-1] - 1),
        "gross_same_trades_return": float(np.prod(1 + rows.gross) - 1),
        "stress_2x_return": float(np.prod(1 + rows.stress_net) - 1),
        "max_drawdown": float(np.max(1 - curve / np.maximum.accumulate(curve))),
        "mean_daily_net": float(r.mean()),
        "sharpe": sr,
        "deflated_sharpe_diagnostic": ds,
        "cost_dollars": float(rows.cost.sum()),
        "trades": int(rows.trades.sum()),
        "active_days": int((rows.trades > 0).sum()),
        "worst_day": float(r.min()),
        "cash_weeks": int(rows.loc[rows.selected == "CASH", "week"].nunique()),
        "max_week_stock_abs_pnl_share": float(rows.week_stock_share.max()),
    }


def report(root, p, prof, *, check_only=False):
    root = Path(root)
    db = sqlite3.connect((root / "archive.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = [(k, json.loads(s)) for k, s in db.execute("select cut,body from decision order by cut")]
        trials = pd.read_sql_query("select * from trial order by slot", db)
        cells = db.execute("select count(*) from cell").fetchone()[0]
    finally:
        db.close()
    work = prof["selected"]["work"]
    if (
        [k for k, _ in rows]
        != list(range(p["history"] + p["validation"], p["history"] + p["validation"] + work["forward_weeks"]))
        or len(trials) != work["trial_count"]
        or cells != work["initial_cells"] + work["later_reference_cells"]
    ):
        raise ValueError("INCOMPLETE: expected weekends, actual trial count or cached reference paths do not reconcile")
    from shpe_research.data.calendar import get_calendar
    from shpe_research.data.panel import week_chunks

    weeks = week_chunks(get_calendar().sessions(p["start"], p["end"]))
    daily = []
    weekly_summary = []
    friction = {
        phase: {name: {"rejections": {}, "ambiguous_stop_bars": 0} for name in POLICIES}
        for phase in ("development", "primary")
    }
    equity = {name: p["account"]["starting_cash"] for name in POLICIES}
    stress_equity = equity.copy()
    for k, weekly in rows:
        if set(weekly) != set(POLICIES) or len({row["archive_snapshot"] for row in weekly.values()}) != 1:
            raise ValueError("mismatched policies or archive")
        for policy, row in weekly.items():
            b, s = row["base"], row["stress"]
            expected_phase = "primary" if k >= len(weeks) - p["primary"] else "development"
            if row["dates"] != [str(x.date()) for x in weeks[k]] or row["phase"] != expected_phase:
                raise ValueError("forward dates or phase differ from the exchange schedule")
            if not np.isclose(row["stress_start_equity"], stress_equity[policy], rtol=1e-12):
                raise ValueError("stress account discontinuity")
            costs = friction[expected_phase][policy]
            for reason, n_reject in b["reject"].items():
                costs["rejections"][reason] = costs["rejections"].get(reason, 0) + n_reject
            costs["ambiguous_stop_bars"] += b["ambiguous"]
            if not np.isclose(row["start_equity"], equity[policy], rtol=1e-12):
                raise ValueError("weekly cash discontinuity")
            sel = row["selected"]["config_id"] if row["selected"] else "CASH"
            dated = zip(row["dates"], b["r"], b["gross"], b["trades"], b["costs"], s["r"], strict=True)
            week_net_start = equity[policy]
            week_stress_start = stress_equity[policy]
            week_cost = 0.0
            week_trades = 0
            for day, r, g, t, c, stress in dated:
                if not np.isclose((g - r) * equity[policy], c, atol=1e-6):
                    raise ValueError("unreconciled transaction cost")
                equity[policy] *= 1 + r
                stress_equity[policy] *= 1 + stress
                week_cost += c
                week_trades += t
                daily.append(
                    dict(
                        date=day,
                        phase=row["phase"],
                        week=k,
                        policy=policy,
                        selected=sel,
                        trades=t,
                        cost=c,
                        net=r,
                        gross=g,
                        stress_net=stress,
                        equity=equity[policy],
                        week_stock_share=b["stock_concentration"],
                    )
                )
            if not np.isclose(equity[policy], b["ending_equity"], rtol=1e-10) or not np.isclose(
                stress_equity[policy], s["ending_equity"], rtol=1e-10
            ):
                raise ValueError("ending equity does not match daily path")
            weekly_summary.append(
                {
                    "week": k,
                    "phase": expected_phase,
                    "policy": policy,
                    "selected_config_id": sel,
                    "start_equity": week_net_start,
                    "ending_equity": equity[policy],
                    "net_return": equity[policy] / week_net_start - 1,
                    "stress_start_equity": week_stress_start,
                    "stress_ending_equity": stress_equity[policy],
                    "stress_return": stress_equity[policy] / week_stress_start - 1,
                    "gross_same_trades_return": float(np.prod(1 + np.asarray(b["gross"])) - 1),
                    "cost_dollars": week_cost,
                    "trades": week_trades,
                    "ambiguous_stop_bars": b["ambiguous"],
                    "stock_concentration": b["stock_concentration"],
                    "cash_week": bool(row["selected"] is None),
                }
            )
    df = pd.DataFrame(daily)
    import hashlib

    payloads = {
        "daily.csv": df.to_csv(index=False, lineterminator="\n"),
        "weekly.csv": pd.DataFrame(weekly_summary).to_csv(index=False, lineterminator="\n"),
        "trials.csv": trials.to_csv(index=False, lineterminator="\n"),
    }
    sums = {}
    for filename, txt in payloads.items():
        blob = txt.encode("utf-8")
        sums[filename] = hashlib.sha256(blob).hexdigest()
        if check_only:
            if file_hash(root / filename) != sums[filename]:
                raise ValueError(f"damaged or changed report file: {filename}")
        else:
            (root / filename).write_bytes(blob)
    stats_by = {}
    diffs = {}
    for phase, part in df.groupby("phase"):
        stats_by[phase] = {k: stats(v, len(trials)) for k, v in part.groupby("policy")}
        wide = part.pivot(index="date", columns="policy", values="net").sort_index()
        if wide.isna().any().any():
            raise ValueError("mismatched forward dates")
        diffs[phase] = {}
        for policy in ("similarity", "recency"):
            diffs[phase][policy + "_minus_uniform"] = {
                str(b): paired_block_bootstrap(
                    wide[policy].to_numpy(),
                    wide.uniform.to_numpy(),
                    min(b, len(wide)),
                    n=p["bootstrap_n"],
                    seed=p["seed"],
                    alpha=0.025,
                )
                for b in (5, 10)
            }
    plans = [read(root / "weekly-configs" / f"{k:04}-uniform.json")["universe"] for k, _ in rows]
    supported = [
        name
        for name in ("recency", "similarity")
        if all(v["low"] > 0 for v in diffs["primary"][name + "_minus_uniform"].values())
    ]
    result = {
        "execution_diagnostics": friction,
        "weight_comparison": {
            "period": "primary",
            "screen": "Both 5- and 10-session paired intervals must have a positive lower end versus uniform.",
            "policies_passing_screen": supported,
            "reading": (
                "synthetic only"
                if p["mode"] == "demo"
                else "No demonstrated improvement over uniform."
                if not supported
                else "Evidence versus uniform only; inspect risk, costs and independent platform parity."
            ),
        },
        "technical_status": "PASS",
        "mode": p["mode"],
        "inputs": prof["inputs"],
        "source": read(root / "archive-source.json"),
        "trial_count": len(trials),
        "origins": trials.origin.value_counts().to_dict(),
        "actual_reference_config_weeks": cells,
        "forward_weeks": len(rows),
        "policy_week_decisions": 3 * len(rows),
        "unique_forward_assets": len({m["asset_id"] for w in plans for m in w}),
        "periods": stats_by,
        "paired_mean_daily_differences": diffs,
        "scientific_digest": digest(daily),
        "dates": [min(df.date), max(df.date)],
        "files": sums,
        "weekly_rows": len(weekly_summary),
        "interpretation": "Synthetic software demonstration; no market inference."
        if p["mode"] == "demo"
        else "Paper research only. A time-held-out comparison does not establish a live trading edge.",
        "platform_parity": "pending independent platform replay",
    }
    page = report_page(result).encode("utf-8")
    if check_only:
        if (root / "results.html").read_bytes() != page:
            raise ValueError("regenerated human-readable results differ")
    else:
        (root / "results.html").write_bytes(page)
        write(root / "report.json", result)
    return result


def report_page(result):
    """Render the reconciled report offline, without web assets or access to raw market data."""

    def e(value):
        return html.escape(str(value))

    def pct(x):
        return f"{x:.2%}"

    phase_tables = []
    for phase in ("primary", "development"):
        rows = []
        for name in POLICIES:
            a = result["periods"][phase][name]
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{e(v)}</td>"
                    for v in (
                        name,
                        pct(a["net_return"]),
                        pct(a["gross_same_trades_return"]),
                        pct(a["stress_2x_return"]),
                        pct(a["max_drawdown"]),
                        f"${a['cost_dollars']:,.2f}",
                        a["trades"],
                        f"{a['active_days']}/{a['days']}",
                        f"{a['cash_weeks']}/{a['weeks']}",
                    )
                )
                + "</tr>"
            )
        headings = (
            "History rule",
            "Net",
            "Gross¹",
            "2× costs",
            "Max drawdown",
            "Modeled costs",
            "Trades",
            "Active days",
            "Cash weeks",
        )
        phase_tables.append(
            f"<h2>{phase.title()} period</h2><div><table><thead><tr>"
            + "".join(f"<th>{e(v)}</th>" for v in headings)
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    comparisons = []
    for key, spans in result["paired_mean_daily_differences"]["primary"].items():
        segments = []
        for n in ("5", "10"):
            lo, hi = (spans[n][k] * 10000 for k in ("low", "high"))
            segments.append(f"{n}-day blocks: [{lo:+.2f}, {hi:+.2f}]")
        comparisons.append(f"<li><strong>{e(key.replace('_', ' '))}</strong> · {e('; '.join(segments))} bps/day</li>")
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SHPE Capital | Weekly momentum results</title><style>
body{font:16px/1.5 system-ui,sans-serif;background:#f6f8fa;color:#172a3b;margin:0}
main{max-width:1050px;margin:auto;padding:3vw} h1{font-size:2.1em;margin:.2em 0} h2{font-size:1.25em}
header{border-top:5px solid #126176;padding:1.2em 0}.note{background:#e4f1f4;padding:1em}
div{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}
th,td{padding:.5em .6em;border-bottom:1px solid #d2dde2;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:#125262}code{overflow-wrap:anywhere}a{color:#0a5870}
</style><main><header><b>SHPE CAPITAL / RM9</b><h1>Weekly momentum results</h1>"""
        + f"<p>{e(result['mode'].upper())} · {e(' through '.join(result['dates']))}</p></header>"
        + f"<p class='note'><strong>Technical result: reconciled.</strong> {e(result['interpretation'])}</p>"
        + f"<p>{result['forward_weeks']} forward weeks · {result['policy_week_decisions']:,} policy decisions · "
        + f"{result['trial_count']:,} actually tested settings · {result['actual_reference_config_weeks']:,} "
        + "historical configuration-week replays.</p>"
        + "".join(phase_tables)
        + "<p>¹ Gross adds modeled costs back on the same accepted trades. The stress portfolio is replayed and "
        + "can take different trades. Cash weeks are weeks in which no finalist qualified.</p>"
        + "<h2>How much better than equal weighting?</h2><p>Paired intervals on mean daily net return in the "
        + "primary period. An interval crossing zero does not separate that method from uniform.</p><ul>"
        + "".join(comparisons)
        + "</ul><p>"
        + e(result["weight_comparison"]["reading"])
        + "</p>"
        + "<h2>Evidence</h2><p><a href='report.json'>Full report</a> · <a href='daily.csv'>Daily account</a> · "
        + "<a href='weekly.csv'>Weekly decisions</a> · <a href='trials.csv'>Candidate register</a> · "
        + "<a href='verification.json'>Verification</a> · "
        + "<a href='machine-profile.json'>Machine benchmark</a></p>"
        + f"<p>Platform parity: {e(result['platform_parity'])}.<br>Daily outcome fingerprint: "
        + f"<code>{e(result['scientific_digest'])}</code></p></main></html>\n"
    )


def compare_ledgers(research_csv, platform_csv):
    keys = ["asset_id", "entry_utc"]
    exact = ["exit_utc", "qty", "exit_reason"]
    numeric = ["entry_price", "exit_price", "net_pnl"]
    frames = [pd.read_csv(p, dtype={"asset_id": str}) for p in (research_csv, platform_csv)]
    for f in frames:
        if set([*keys, *exact, *numeric]) - set(f) or f[keys].duplicated().any():
            raise ValueError("ledger columns missing or duplicate round trips")
        for k in ("entry_utc", "exit_utc"):
            f[k] = pd.to_datetime(f[k], utc=True, format="mixed")
    a, b = (f.set_index(keys) for f in frames)
    common = a.index.intersection(b.index)
    mismatch = []
    for key in common:
        x, y = a.loc[key], b.loc[key]
        fields = [k for k in exact if x[k] != y[k]]
        fields.extend(
            k
            for k in numeric
            if not np.isclose(x[k], y[k], rtol=0 if k == "net_pnl" else 0.00005, atol=0.01 if k == "net_pnl" else 1e-8)
        )
        if fields:
            mismatch.append([str(key), fields])
    result = {
        "matched": len(common) - len(mismatch),
        "research_only": list(map(str, a.index.difference(b.index))),
        "platform_only": list(map(str, b.index.difference(a.index))),
        "mismatches": mismatch,
    }
    result["status"] = (
        "PASS"
        if len(common) and not any(result[k] for k in ("research_only", "platform_only", "mismatches"))
        else "FAIL"
    )
    return result
