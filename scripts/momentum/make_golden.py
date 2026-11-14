"""B6: generate the momentum golden parity fixtures (format: docs/contracts/momentum_golden.md).

    uv run python scripts/momentum/make_golden.py            # (re)write tests/fixtures/momentum_golden/
    uv run python scripts/momentum/make_golden.py --check    # exit 1 if the checked-in files differ

Scenarios are synthetic, small and fully deterministic (``random.Random`` seeded by strings, prices on a
0.01 grid, integer volumes). Indicators, signals, the ledger and the order decisions come from the real
code: B1-B5 (``features/momentum``, ``strategies/momentum/breakout``) and C3/C4 (``shpe_research.trade``).
The NumPy reference path is used (``SHPE_MOMENTUM_NUMBA=0``) so the output does not depend on Numba.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from shpe_research.contracts.weekly_config import WeeklyConfig, to_json
from shpe_research.features.momentum import wilder
from shpe_research.strategies.momentum.breakout.grid import Config, config_id, is_valid
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade import account, fill_proxy, funded_sizing
from shpe_research.trade.account_types import AccountConfig, MarketPanel

FORMAT_VERSION = "momentum_golden.v1"
OUT_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "momentum_golden"
SESSION_OPEN_UTC = (14, 30)  # 09:30 America/New_York in January (EST, UTC-5)
BASE_DATE = date(2026, 1, 5)

# Per-indicator comparison tolerance: |actual - expected| <= abs + rel * |expected|. Everything is float64.
TOL_DEFAULT = {"abs": 1e-9, "rel": 1e-9}
TOL_ADX = {"abs": 1e-7, "rel": 1e-9}  # points on a 0..100 scale


@dataclass(frozen=True)
class Spec:
    name: str
    description: str
    seed: str
    symbols: tuple[tuple[str, str, float, int], ...]  # (asset_id, symbol, base price, base 1-minute volume)
    session_lengths: tuple[int, ...]
    config: Config
    account: AccountConfig
    warmup_sessions: int
    drift: float = 0.0004
    noise: float = 0.0015
    pops: tuple[tuple[int, int, tuple[int, ...], float, float], ...] = ()  # (session, minute, symbols, ret, vol x)
    missing: tuple[tuple[int, int, int], ...] = ()  # (symbol index, session, minute)
    clones: dict[int, int] = field(default_factory=dict)  # symbol index -> source symbol index (identical bars)
    notes: tuple[str, ...] = ()


def _cfg(subset: list[str], params: dict[str, float]) -> Config:
    cfg = Config.make("v1", subset, params)
    assert is_valid(cfg), cfg
    return cfg


SCENARIOS: tuple[Spec, ...] = (
    Spec(
        name="s1_trend_gap",
        description="Breakout + ADX + EMA trend; early-close session, one missing minute, warm-up boundaries.",
        seed="s1",
        symbols=(("a-0001", "AAA", 100.0, 40_000), ("a-0002", "BBB", 60.0, 90_000),
                 ("a-0003", "CCC", 150.0, 25_000), ("a-0004", "DDD", 35.0, 150_000)),
        session_lengths=(36, 36, 36, 36, 36, 24, 36, 36),
        config=_cfg(["adx", "breakout", "ema"], {
            "breakout.n": 10, "breakout.b": 0.10, "adx.period": 7, "adx.q": 15,
            "ema.fast": 5, "ema.slow": 20, "ema.d": 0}),
        account=AccountConfig(liquidity_lookback=5, liquidity_min_obs=3),
        warmup_sessions=3,
        drift=0.0010,
        pops=((3, 4, (0, 1, 2), 0.0040, 4.0), (4, 3, (0, 3), 0.0040, 4.0), (5, 3, (1, 2, 3), 0.0040, 4.0),
              (6, 5, (0, 1), 0.0040, 4.0), (7, 4, (0, 1, 2, 3), 0.0040, 4.0)),
        missing=((2, 4, 12),),
        notes=("ATR14 ready at bar 70 (session 1, minute 34); ADX7 at bar 35; EMA20 and its 5-bar lag at bar 100.",
               "Session 5 is an early close (24 minutes instead of 36).",
               "Symbol a-0003 has no bar at session 4, minute 12: every recursive indicator restarts there."),
    ),
    Spec(
        name="s2_rvol_vwap",
        description="Breakout + relative volume + VWAP band over 20 short sessions; early close; missing minute.",
        seed="s2",
        symbols=(("a-0010", "RRR", 80.0, 30_000), ("a-0011", "SSS", 45.0, 60_000), ("a-0012", "TTT", 210.0, 12_000)),
        session_lengths=(12,) * 17 + (8,) + (12, 12),
        config=_cfg(["breakout", "rvol", "vwap"], {
            "breakout.n": 10, "breakout.b": 0, "rvol.w": 3, "rvol.v": 1.5, "vwap.e": 3}),
        account=AccountConfig(liquidity_lookback=5, liquidity_min_obs=3),
        warmup_sessions=15,
        drift=0.0002,
        pops=tuple((s, 2, (0, 1, 2), 0.0040, 4.0) for s in (15, 16, 18, 19)) + ((17, 1, (0, 1), 0.0040, 4.0),),
        missing=((1, 16, 4),),
        notes=("RVOL needs 15 prior sessions: first valid at session 15 (bar 180).",
               "Session 17 is an early close (8 minutes instead of 12): later clock windows have fewer samples.",
               "Symbol a-0011 has no bar at session 16, minute 4: its RVOL windows touching it are invalid."),
    ),
    Spec(
        name="s3_rank_size",
        description="Breakout + VWAP; simultaneous signals, a tie on dollar volume, max_positions=3.",
        seed="s3",
        symbols=(("a-0020", "EEE", 50.0, 50_000), ("a-0021", "FFF", 50.0, 50_000), ("a-0022", "GGG", 120.0, 20_000),
                 ("a-0023", "HHH", 25.0, 200_000), ("a-0024", "III", 300.0, 8_000), ("a-0025", "JJJ", 70.0, 45_000)),
        session_lengths=(30,) * 6,
        config=_cfg(["breakout", "vwap"], {"breakout.n": 10, "breakout.b": 0.10, "vwap.e": 3}),
        account=AccountConfig(max_positions=3, liquidity_lookback=5, liquidity_min_obs=2),
        warmup_sessions=2,
        drift=0.0002,
        pops=((2, 4, (0, 1, 2, 3, 4, 5), 0.0045, 4.0), (3, 5, (0, 1, 2, 3, 4, 5), 0.0045, 4.0),
              (4, 6, (0, 2, 3, 5), 0.0045, 4.0), (5, 4, (1, 2, 3, 4), 0.0045, 4.0)),
        clones={1: 0},
        notes=("a-0021 is a clone of a-0020 (identical bars): equal dollar volume, so the asset-id tie-break "
               "orders them.", "The account allows 3 positions, so decisions with state 'busy' reject by "
               "max_positions."),
    ),
)

PROFILES = ("fresh", "busy", "tight_cash", "entered")


# --- synthetic bars ---------------------------------------------------------------------------------------


def _build_panel(spec: Spec) -> tuple[MarketPanel, list[date]]:
    lens = spec.session_lengths
    n_min, n_sym = sum(lens), len(spec.symbols)
    arr = {k: np.full((n_sym, n_min), np.nan) for k in ("open", "high", "low", "close", "volume")}
    pops: dict[tuple[int, int, int], tuple[float, float]] = {}
    for s, m, syms, ret, vmul in spec.pops:
        for k in syms:
            pops[(k, s, m)] = (ret, vmul)
    for k, (_, sym, price0, vol0) in enumerate(spec.symbols):
        if k in spec.clones:
            continue
        rng = random.Random(f"{spec.seed}-{sym}")
        prev = price0
        t = 0
        for s, length in enumerate(lens):
            for m in range(length):
                ret = rng.gauss(spec.drift, spec.noise)
                if m == 0:
                    ret += rng.gauss(0.0, spec.noise)
                ret2, vmul = pops.get((k, s, m), (0.0, 1.0))
                o = round(prev * (1.0 + 0.3 * ret), 2)
                c = round(prev * (1.0 + ret + ret2), 2)
                h = round(max(o, c) * (1.0 + abs(rng.gauss(0.0, 0.0007))), 2)
                lo = round(min(o, c) * (1.0 - abs(rng.gauss(0.0, 0.0007))), 2)
                v = int(vol0 * (0.6 + 0.8 * rng.random()) * vmul)
                arr["open"][k, t], arr["high"][k, t] = o, max(h, o, c)
                arr["low"][k, t], arr["close"][k, t], arr["volume"][k, t] = min(lo, o, c), c, float(v)
                prev = c
                t += 1
    for k, src in spec.clones.items():
        for key in arr:
            arr[key][k] = arr[key][src]
    for k, s, m in spec.missing:
        t = sum(lens[:s]) + m
        for key in arr:
            arr[key][k, t] = np.nan
    session_index = np.repeat(np.arange(len(lens)), lens)
    panel = MarketPanel(
        asset_ids=tuple(a for a, *_ in spec.symbols),
        session_index=session_index,
        eligible=np.ones((n_sym, n_min), dtype=np.bool_),
        **arr,
    )
    dates: list[date] = []
    d = BASE_DATE
    for _ in lens:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        dates.append(d)
        d += timedelta(days=1)
    return panel, dates


# --- helpers ----------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _numpy_reference() -> Iterator[None]:
    old = os.environ.get(wilder.NUMBA_ENV)
    os.environ[wilder.NUMBA_ENV] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(wilder.NUMBA_ENV, None)
        else:
            os.environ[wilder.NUMBA_ENV] = old


def _num(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"


def _ts(day: date, minute: int) -> str:
    base = datetime(day.year, day.month, day.day, *SESSION_OPEN_UTC, tzinfo=UTC)
    return (base + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_starts(panel: MarketPanel) -> tuple[np.ndarray, np.ndarray]:
    si = np.asarray(panel.session_index)
    starts = np.flatnonzero(np.concatenate(([True], si[1:] != si[:-1])))
    lens = np.diff(np.concatenate((starts, [panel.n_minutes])))
    return starts, lens


# --- indicator columns ------------------------------------------------------------------------------------


def _indicator_columns(cfg: Config, cache: IndicatorCache) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Ordered ``name -> array`` plus ``name -> {kind, tol}`` for the active components."""
    p, subset = cfg.as_dict(), set(cfg.subset)
    ready = cache.readiness()
    cols: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    def num(name: str, arr: np.ndarray, tol: dict) -> None:
        cols[name], meta[name] = arr, {"kind": "number", "abs": tol["abs"], "rel": tol["rel"]}

    def flag(name: str, arr: np.ndarray) -> None:
        cols[name], meta[name] = np.asarray(arr, dtype=bool), {"kind": "bool"}

    n = int(p["breakout.n"])
    num("atr14", cache.atr14(), TOL_DEFAULT)
    flag("ready_atr14", ready["atr14"])
    num(f"prec_high_n{n}", cache.prec_high(n), TOL_DEFAULT)
    flag("sig_breakout", cache.breakout(n, p["breakout.b"]))
    flag(f"ready_breakout_n{n}", ready[f"breakout_n{n}"])
    if "adx" in subset:
        a = int(p["adx.period"])
        adx, pdi, mdi = cache.adx(a)
        num(f"adx_a{a}", adx, TOL_ADX)
        num(f"pdi_a{a}", pdi, TOL_ADX)
        num(f"mdi_a{a}", mdi, TOL_ADX)
        flag("sig_adx", cache.adx_sig(a, p["adx.q"]))
        flag(f"ready_adx_a{a}", ready[f"adx_a{a}"])
    if "ema" in subset:
        f, s = int(p["ema.fast"]), int(p["ema.slow"])
        num(f"ema_{f}", cache.ema(f), TOL_DEFAULT)
        num(f"ema_{s}", cache.ema(s), TOL_DEFAULT)
        flag("sig_ema", cache.ema_sig(f, s, p["ema.d"]))
        flag(f"ready_ema_f{f}_s{s}", ready[f"ema_f{f}_s{s}"])
    if "rvol" in subset:
        w = int(p["rvol.w"])
        num(f"rvol_w{w}", cache.rvol(w), TOL_DEFAULT)
        flag("sig_rvol", cache.rvol_sig(w, p["rvol.v"]))
        flag(f"ready_rvol_w{w}", ready[f"rvol_w{w}"])
    if "vwap" in subset:
        num("vwap", cache.vwap(), TOL_DEFAULT)
        flag("sig_vwap", cache.vwap_sig(p["vwap.e"]))
        flag("ready_vwap", ready["vwap"])
    return cols, meta


# --- ranked and sized orders (C3 sizing + C4 ranking, scripted account state) ---------------------------


def _scripted_state(profile: str, cfg: AccountConfig, candidates: list[str]) -> dict:
    if profile == "fresh":
        return {"equity": cfg.starting_cash, "cash_free": cfg.starting_cash, "exposure_used": 0.0,
                "open_positions": 0, "pending_orders": 0, "entered_today": []}
    if profile == "busy":
        return {"equity": 104250.5, "cash_free": 38000.0, "exposure_used": 57500.0,
                "open_positions": 2, "pending_orders": 0, "entered_today": []}
    if profile == "tight_cash":
        return {"equity": 100000.0, "cash_free": 1500.0, "exposure_used": 20000.0,
                "open_positions": 1, "pending_orders": 0, "entered_today": []}
    return {"equity": 99000.0, "cash_free": 99000.0, "exposure_used": 0.0,
            "open_positions": 0, "pending_orders": 0, "entered_today": candidates[:1]}


def decide_orders(
    panel: MarketPanel, atr: np.ndarray, cfg: AccountConfig, t: int, signal_ids: list[str], state: dict,
    rank_pos: np.ndarray, starts: np.ndarray, lens: np.ndarray,
) -> dict:
    """What C4 steps 4-6 do at bar ``t``, for a scripted account state (mirrors ``account.simulate_account``).

    Candidates are the signals minus ``entered_today``; ordered by ``rank_pos`` (session ranking row);
    each is allocated against ``max_positions`` then sized by ``funded_sizing.size_order``.
    """
    s = int(panel.session_index[t])
    m = t - int(starts[s])
    ids = list(panel.asset_ids)
    cand = [a for a in signal_ids if a not in state["entered_today"]]
    cand.sort(key=lambda a: int(rank_pos[ids.index(a)]))
    cb = fill_proxy.side_cost_rate(cfg.cost_bps)
    mult = 1.0 if cfg.size_multiplier is None else float(cfg.size_multiplier)
    n_pend, reserved, pend_notional = int(state["pending_orders"]), 0.0, 0.0
    orders, rejections = [], []
    buf = np.empty(max(cfg.liquidity_lookback, 1))
    for a in cand:
        k = ids.index(a)
        if int(state["open_positions"]) + n_pend >= cfg.max_positions:
            rejections.append({"asset_id": a, "reason": "max_positions"})
            continue
        price, av = float(panel.close[k, t]), float(atr[k, t])
        if not (0.0 < price < math.inf and 0.0 < av < math.inf):
            rejections.append({"asset_id": a, "reason": "bad_data"})
            continue
        limit = fill_proxy.entry_limit(price, cfg.fill.limit_markup)
        nv = 0
        for q in range(1, cfg.liquidity_lookback + 1):
            ps = s - q
            if ps < 0:
                break
            if m < lens[ps]:
                v = panel.volume[k, starts[ps] + m]
                if v == v:
                    buf[nv] = v
                    nv += 1
        liq = funded_sizing.liquidity_cap_shares(buf, nv, cfg.liquidity_cap_pct, cfg.liquidity_min_obs)
        unit = limit * (1.0 + cb)
        stop_distance = cfg.exit.initial_stop_atr * av
        qty, cap = funded_sizing.size_order(
            state["equity"], price, stop_distance, state["cash_free"] - reserved, unit,
            state["exposure_used"] + pend_notional, liq, cfg.per_stock_pct, cfg.risk_pct,
            cfg.total_exposure_cap, mult,
        )
        if qty <= 0:
            rejections.append({"asset_id": a, "reason": f"size_zero:{funded_sizing.CAP_NAMES[cap]}"})
            continue
        reserved += qty * unit
        pend_notional += qty * limit
        n_pend += 1
        orders.append({
            "asset_id": a, "rank": len(orders) + len(rejections), "qty": int(qty), "signal_close": price,
            "atr": av, "stop_distance": stop_distance, "limit_price": limit, "unit_cost": unit,
            "reserved": qty * unit, "liquidity_shares": liq, "binding_cap": funded_sizing.CAP_NAMES[cap],
        })
    return {"candidates": cand, "orders": orders, "rejections": rejections}


def _order_decisions(
    panel: MarketPanel, mask_signal: np.ndarray, atr: np.ndarray, cfg: AccountConfig, warmup_sessions: int
) -> list[dict]:
    starts, lens = _session_starts(panel)
    ids = list(panel.asset_ids)
    id_rank = np.empty(len(ids), dtype=np.int64)
    id_rank[np.argsort(np.asarray(ids, dtype=object), kind="stable")] = np.arange(len(ids))
    rk = account.RANKINGS[cfg.ranking](
        np.asarray(panel.close, dtype=np.float64), np.asarray(panel.volume, dtype=np.float64),
        starts, id_rank, cfg.liquidity_lookback,
    )
    dollar_vol = np.add.reduceat(np.nan_to_num(panel.close * panel.volume), starts, axis=1)
    exit_buf = cfg.exit.session_exit_minutes_before_close
    out: list[dict] = []
    for t in range(panel.n_minutes):
        s = int(panel.session_index[t])
        m = t - int(starts[s])
        if s < warmup_sessions or m + 1 >= int(lens[s]) - exit_buf:
            continue
        sig = [ids[k] for k in range(len(ids)) if mask_signal[k, t]]
        if not sig:
            continue
        profile = "fresh" if not out else PROFILES[len(out) % len(PROFILES)]
        state = _scripted_state(profile, cfg, sorted(sig, key=lambda a: int(rk[s, ids.index(a)])))
        res = decide_orders(panel, atr, cfg, t, sig, state, rk[s], starts, lens)
        scores = {
            a: _num(np.median(dollar_vol[ids.index(a), max(0, s - cfg.liquidity_lookback):s]) if s > 0 else 0.0)
            for a in res["candidates"]
        }
        out.append({"decision": len(out), "i": t, "session": s, "minute": m, "profile": profile, "state": state,
                    "signals": sig, "dollar_volume_median": scores, **res})
    return out


# --- scenario files ---------------------------------------------------------------------------------------


def _weekly_config(spec: Spec, dates: list[date]) -> WeeklyConfig:
    cfg = spec.config
    return WeeklyConfig.model_validate({
        "schema_version": "weekly_config.v1", "run_id": f"golden-{spec.name}", "code_commit": "0000000",
        "data_version": "synthetic", "protocol_hash": None, "policy": "uniform",
        "decision_cutoff": "2025-12-31T21:00:00Z",
        "search_range": {"start": "2025-01-02", "end": "2025-12-12"},
        "validation_range": {"start": "2025-12-15", "end": "2025-12-31"},
        "activation": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "universe": [{"asset_id": a, "symbol": sym} for a, sym, _, _ in spec.symbols],
        "selected": {"config_id": config_id(cfg), "grid_version": cfg.grid_version, "subset": list(cfg.subset),
                     "params": cfg.as_dict()},
        "no_deployment": None,
        "acceptance": {"passed": True, "checks": []},
        "ess": 1.0, "evaluations": {"logical": 0, "actual": 0}, "seeds": [0],
        "account": asdict(spec.account), "runtime": {"seconds": 0.0, "peak_memory_mb": 0.0},
    })


def build_scenario(spec: Spec) -> dict[str, str]:
    """All files of one scenario as ``filename -> text``."""
    with _numpy_reference():
        panel, dates = _build_panel(spec)
        cache = IndicatorCache(panel)
        mask = signal_mask(panel, spec.config, cache)
        cols, meta = _indicator_columns(spec.config, cache)
        result = account.simulate_account(panel, mask, spec.account, warmup_sessions=spec.warmup_sessions)
        decisions = _order_decisions(panel, mask.signal, mask.atr, spec.account, spec.warmup_sessions)
    starts, lens = _session_starts(panel)
    ids = list(panel.asset_ids)
    sym_of = {a: s for a, s, _, _ in spec.symbols}

    bars, inds, sigs = [], [], []
    for t in range(panel.n_minutes):
        s = int(panel.session_index[t])
        m = t - int(starts[s])
        for k, a in enumerate(ids):
            if math.isfinite(panel.close[k, t]):
                bars.append(_line({
                    "i": t, "session": s, "minute": m, "ts": _ts(dates[s], m), "asset_id": a,
                    "o": float(panel.open[k, t]), "h": float(panel.high[k, t]), "l": float(panel.low[k, t]),
                    "c": float(panel.close[k, t]), "v": int(panel.volume[k, t]),
                }))
            row: dict = {"i": t, "asset_id": a}
            for name, arr in cols.items():
                row[name] = bool(arr[k, t]) if meta[name]["kind"] == "bool" else _num(arr[k, t])
            row["signal"] = bool(mask.signal[k, t])
            inds.append(_line(row))
            if mask.signal[k, t]:
                sigs.append(_line({"i": t, "session": s, "minute": m, "asset_id": a, "atr": float(mask.atr[k, t])}))
    orders = [_line(d) for d in decisions]
    ledger = [_line({"asset_id": e.asset_id, "entry_i": e.entry_minute, "exit_i": e.exit_minute, "qty": e.qty,
                     "entry_price": e.entry_price, "exit_price": e.exit_price, "exit_reason": e.exit_reason,
                     "net_pnl": e.net_pnl}) for e in result.ledger]
    if not sigs or not any(d["orders"] for d in decisions):
        raise RuntimeError(f"{spec.name}: scenario produced no signals or no orders; retune it")

    manifest = {
        "format_version": FORMAT_VERSION, "scenario": spec.name, "description": spec.description,
        "notes": list(spec.notes),
        "symbols": [{"asset_id": a, "symbol": sym_of[a]} for a in ids],
        "sessions": [{"session": s, "date": dates[s].isoformat(), "start_i": int(starts[s]), "n_minutes": int(lens[s]),
                      "early_close": int(lens[s]) != max(spec.session_lengths)} for s in range(len(lens))],
        "n_bars": int(panel.n_minutes), "warmup_sessions": spec.warmup_sessions,
        "missing_bars": [{"asset_id": ids[k], "i": int(starts[s]) + m, "session": s, "minute": m}
                         for k, s, m in spec.missing],
        "indicator_columns": meta,
        "counts": {"bars": len(bars), "indicator_rows": len(inds), "signals": len(sigs),
                   "decisions": len(orders), "ledger": len(ledger)},
        "files": {"weekly_config": "weekly_config.json", "bars": "bars.jsonl", "indicators": "indicators.jsonl",
                  "signals": "signals.jsonl", "orders": "orders.jsonl", "ledger": "ledger.jsonl"},
    }
    return {
        "manifest.json": json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        "weekly_config.json": to_json(_weekly_config(spec, dates)),
        "bars.jsonl": "".join(bars),
        "indicators.jsonl": "".join(inds),
        "signals.jsonl": "".join(sigs),
        "orders.jsonl": "".join(orders),
        "ledger.jsonl": "".join(ledger),
    }


def build_all() -> dict[str, str]:
    """``relative path -> text`` for every scenario."""
    out: dict[str, str] = {}
    for spec in SCENARIOS:
        for name, text in build_scenario(spec).items():
            out[f"{spec.name}/{name}"] = text
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing")
    args = ap.parse_args(argv)
    files = build_all()
    if args.check:
        stale = [p for p, text in files.items()
                 if not (args.out / p).exists() or (args.out / p).read_bytes() != text.encode("utf-8")]
        for p in stale:
            print(f"stale: {p}", file=sys.stderr)
        return 1 if stale else 0
    for p, text in files.items():
        dest = args.out / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(text.encode("utf-8"))
    print(f"wrote {len(files)} files under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


