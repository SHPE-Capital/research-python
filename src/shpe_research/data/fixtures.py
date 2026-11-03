"""Deterministic synthetic bars for tests, the workshop and offline development.

Regular-session minute bars on real NYSE sessions (holidays and early closes
included) that satisfy the bars contract, so every layer can be built and
tested before anyone downloads market data.

The price process has a small, known amount of short-horizon mean reversion
(`signal_strength`): each minute's return leans against the previous five
minutes' returns. A correct pipeline should find it; a pipeline that finds a
much stronger signal is probably leaking future data. Real markets are not
this simple, so never draw conclusions about strategies from fixture results.

Planted effects (C1o)
---------------------
Every effect is an opt-in keyword-only option and defaults to off; with none of
them set the output is byte-identical to the original fixture for any seed.
Effects only add to the base draws (they use their own random streams, all
derived from `seed`), they compose, and the truth is returned in
`bars.attrs["planted"]` so a test can compare an estimate with what was
planted. Effects apply to the `symbols` you pass; the extra symbols an option
creates (`null_symbols`, pair legs, the `orb_rvol` context symbol) are plain
random walks otherwise. An option that needs a minimum number of sessions
extends `end` to reach it (the minimum is listed per option).

`momentum` (min 30 sessions)
    `{breakout_prob, drift_bps, drift_minutes, range_minutes=30, max_delay_minutes=60}`.
    On a random `breakout_prob` share of symbol-sessions the first close above
    the opening-range high H happens within `max_delay_minutes` after the range
    (forced if the path would not get there) and the price then drifts up by
    `drift_bps` over the next `drift_minutes` bars. `planted["momentum"]["events"]`
    lists `symbol, ts, session` of each such bar (the decision bar: its close is
    the first close above H, known at `ts_available`).

`orb_rvol` (min 60 sessions)
    `{rvol_threshold, drift_bps, drift_minutes, context="SPY", breakout_prob=0.5,
    high_share=0.5, range_minutes=30, max_delay_minutes=60}`. Same breakout
    mechanics, but the post-breakout drift is planted only when the session's
    relative volume is at or above `rvol_threshold`. Relative volume is set by
    scaling the whole session's volume (so cumulative volume through the
    breakout bar is about `rvol` times a normal day); the planted value and
    class are in `planted["orb_rvol"]["events"]` (`symbol, ts, session, rvol,
    above, drift_applied`). Adds the `context` symbol (a random walk).
    If `momentum` is also set, a session's two breakouts may coincide and the
    later-applied one (`orb_rvol`) can move the price before the first one's bar.

`reversal` (any span)
    `{ar: [phi_1, ...], scale=1.0}`. 1-minute returns follow r_t = sum_k phi_k
    r_{t-k} + e_t, with e_t the base innovation times `scale`. Replaces the
    default `signal_strength` reversion for the `symbols`. Negative coefficients
    are mean reversion. The AR polynomial must be stationary.

`cointegrated_pairs` (min 20 sessions)
    A list of `{beta, half_life_bars, spread_sd}` or a mapping
    `{"pairs": [...], "trap": True}`; a list entry `{"trap": True}` also adds the
    trap. Pair k adds symbols `P{k}A` and `P{k}B`. B is a random walk and
    `log A - beta * log B` is an OU process with the given half-life in bars
    (`spread_sd` is its stationary standard deviation in log points; defaults 0.01
    on minute bars). The spread is defined in log prices, so the hedge ratio is
    estimated by regressing log A on log B. The trap adds `TRAPA` and `TRAPB`:
    returns correlated at `corr` (default 0.8) but with an independent random-walk
    component, so they are not cointegrated.

`vol_clustering` (min 126 sessions: 22 of HAR history plus a test block)
    `{persistence, alpha=0.1, shared=False}`. Daily variance h_d follows
    GARCH(1,1) with alpha + beta = `persistence` and unit mean; it scales that
    day's minute returns and overnight gap. `planted["vol_clustering"]` holds
    the daily variance paths. Not applied to extra symbols.

`factors` (min 20 sessions)
    `{market, sectors}`: `market` is the 1-bar standard deviation of the market
    factor or `{sigma, beta_range=(0.5, 1.5)}`, `sectors` a count or `{n, sigma}`
    (defaults: 3 sectors, market sigma 0.5 and sector sigma 0.3 of a typical
    idiosyncratic bar sigma). Returns gain beta_i * market + sector factor.
    `planted["beta"]` (Series by symbol), `planted["sector"]` (names `SEC1...`)
    and `planted["factors"]` (`market_return`, `sector_return`, indexed by `ts`).

`xs_alpha` (min 100 sessions)
    `{ic, contaminate: {beta, sector}}`. `planted["xs_alpha"]["score"]` is a
    sessions x symbols frame; the score of session d (known at its close) has
    cross-sectional Pearson correlation exactly `ic` with the residual return of
    session d + 1 (close-to-close, factor part removed, demeaned across symbols).
    Rank IC is slightly below `ic`. The returns themselves are not modified: the
    score is built from them. `contaminate` mixes `beta` times the standardised
    beta and `sector` times a random daily sector score into `score`; the
    uncontaminated version is `score_clean`. Needs `factors` if `contaminate` is set.
    The last session's score is random.

`null_symbols` (any span)
    An int (names `N01`, `N02`, ...) or a list of names. Pure random walks, no
    effect of any kind.

`daily` (3 or more years, 50 or more symbols)
    `{years=3, n_symbols=50, xs_momentum=False}`. Daily bars on real sessions
    from `start` for `years` (`end`, `symbols` and `signal_strength` are ignored; if the
    calendar ends before `start` + `years`, the last `years` years of the calendar are used),
    symbols `S001...`. `ts` is the session open and `ts_available` the session
    close. `xs_momentum` adds a slowly varying per-symbol drift (half-life 100
    sessions, standard deviation 0.12% a day), so past 12-1 returns rank future
    returns. `momentum` and `orb_rvol` need intraday bars and raise in this mode;
    the other options compose, with bar-level parameters read as daily ones.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from shpe_research.contracts.bars import validate_bars
from shpe_research.contracts.base import UTC_NS
from shpe_research.data.calendar import get_calendar

DEFAULT_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")

_MIN_SESSIONS = {
    "momentum": 30,
    "orb_rvol": 60,
    "cointegrated_pairs": 20,
    "vol_clustering": 126,
    "factors": 20,
    "xs_alpha": 100,
}
_INTRADAY_ONLY = ("momentum", "orb_rvol")
_DAILY_MIN_YEARS = 3
_DAILY_MIN_SYMBOLS = 50


@dataclass(frozen=True)
class _Grid:
    """The bar clock: one entry per bar, grouped into sessions."""

    ts: pd.DatetimeIndex
    ts_available: pd.DatetimeIndex
    starts: np.ndarray  # index of each session's first bar
    lengths: np.ndarray
    session_dates: pd.DatetimeIndex  # tz-naive session dates
    first_of_session: np.ndarray
    u_shape: np.ndarray
    daily: bool

    @property
    def n(self) -> int:
        return len(self.ts)

    @property
    def sigma_range(self) -> tuple[float, float]:
        return (0.010, 0.022) if self.daily else (0.0006, 0.0015)

    @property
    def gap_sd(self) -> float:
        return 0.004 if self.daily else 0.006

    @property
    def unit(self) -> float:
        """A typical idiosyncratic standard deviation of one bar's return."""
        return 0.015 if self.daily else 0.001

    @property
    def spread_sd(self) -> float:
        return 0.05 if self.daily else 0.01

    def per_bar(self, per_session: np.ndarray) -> np.ndarray:
        return np.repeat(per_session, self.lengths)


@dataclass
class _Path:
    """One symbol's random draws plus the return path built from them."""

    symbol: str
    sigma: float
    noise: np.ndarray  # white innovations, sd sigma
    gaps: np.ndarray  # overnight gaps (log), zero except on a session's first bar
    p0: float
    wick: np.ndarray  # (2, n) absolute wicks (log)
    vol_level: float
    vol_noise: np.ndarray
    returns: np.ndarray | None = None
    vol_mult: np.ndarray | None = None
    factor_part: np.ndarray | None = None


def _as_dict(value: Any, key: str | None, option: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if key is not None and value is not None and not isinstance(value, (list, tuple)):
        return {key: value}
    raise TypeError(f"option {option!r} must be a mapping, got {type(value).__name__}")


def _effect_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def _make_grid(start: str, end: str, options: Mapping[str, Any]) -> _Grid:
    cal = get_calendar()
    if "daily" in options:
        cfg = options["daily"]
        years = int(cfg.get("years", _DAILY_MIN_YEARS))
        stop = pd.Timestamp(start) + pd.DateOffset(years=years)
        if stop > cal.last_session:  # the calendar ends: use the last `years` years it has instead
            stop = cal.last_session
            start = stop - pd.DateOffset(years=years)
        schedule = cal.schedule(start, stop)
    else:
        schedule = cal.schedule(start, end)
        needed = max((_MIN_SESSIONS[k] for k in options if k in _MIN_SESSIONS), default=0)
        if 0 < len(schedule) < needed:
            longer = cal.schedule(start, pd.Timestamp(start) + pd.Timedelta(days=int(needed * 1.6) + 14))
            schedule = longer.iloc[:needed]
    if schedule.empty:
        raise ValueError(f"no sessions between {start} and {end}")
    dates = schedule.index.tz_localize(None) if schedule.index.tz is not None else schedule.index
    if "daily" in options:
        n = len(schedule)
        ts = pd.DatetimeIndex(schedule["open"]).astype(UTC_NS)
        avail = pd.DatetimeIndex(schedule["close"]).astype(UTC_NS)
        return _Grid(ts, avail, np.arange(n), np.ones(n, dtype=int), dates, np.ones(n, dtype=bool),
                     np.full(n, 390.0), True)
    minute_starts = [
        pd.date_range(row.open, row.close, freq="1min", inclusive="left") for row in schedule.itertuples()
    ]
    ts = minute_starts[0].append(minute_starts[1:])
    lengths = np.array([len(m) for m in minute_starts])
    first = np.zeros(len(ts), dtype=bool)
    first[np.r_[0, np.cumsum(lengths)[:-1]]] = True
    minute_index = np.concatenate([np.arange(k) for k in lengths])
    u_shape = 1 + 2.5 * ((minute_index - 195) / 195) ** 2
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    return _Grid(ts, ts + pd.Timedelta(minutes=1), starts, lengths, dates, first, u_shape, False)


def _draw_symbol(rng: np.random.Generator, grid: _Grid, symbol: str) -> _Path:
    """The base random draws for one symbol. The order of draws is part of the fixture's contract."""
    n = grid.n
    sigma = rng.uniform(*grid.sigma_range)
    noise = rng.normal(0, sigma, n)
    gaps = np.where(grid.first_of_session, rng.normal(0, grid.gap_sd, n), 0.0)
    p0 = rng.uniform(20, 300)
    wick = np.abs(rng.normal(0, sigma / 2, (2, n)))
    vol_level = rng.uniform(2_000, 40_000)
    vol_noise = rng.lognormal(0, 0.5, n)
    return _Path(symbol, sigma, noise, gaps, p0, wick, vol_level, vol_noise)


def _base_returns(path: _Path, grid: _Grid, signal_strength: float) -> np.ndarray:
    if grid.daily:
        return path.noise.copy()
    # r_t = -(k/5) * (r_{t-1} + ... + r_{t-5}) + noise
    a = np.r_[1.0, np.full(5, signal_strength / 5)]
    return lfilter([1.0], a, path.noise)


def _frame(grid: _Grid, path: _Path) -> pd.DataFrame:
    returns, gaps = path.returns, path.gaps
    close = path.p0 * np.exp(np.cumsum(returns + gaps))
    open_ = np.r_[close[0] * np.exp(-returns[0]), close[:-1]] * np.exp(gaps)
    high = np.maximum(open_, close) * np.exp(path.wick[0])
    low = np.minimum(open_, close) * np.exp(-path.wick[1])
    level = path.vol_level * grid.u_shape * path.vol_noise
    if path.vol_mult is not None:
        level = level * path.vol_mult
    volume = level.astype("int64") + 1
    return pd.DataFrame(
        {
            "symbol": path.symbol,
            "ts": grid.ts,
            "ts_available": grid.ts_available,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": (open_ + high + low + close) / 4,
            "trade_count": pd.array(np.maximum(1, volume // 50), dtype="Int64"),
        }
    )


def _walk(rng: np.random.Generator, grid: _Grid, symbol: str) -> _Path:
    """A plain random walk: no signal_strength reversion, no planted effect."""
    path = _draw_symbol(rng, grid, symbol)
    path.returns = path.noise.copy()
    return path


# --------------------------------------------------------------------------- effects on the return path


def _plant_vol_clustering(grid, paths, cfg, rng, planted) -> None:
    """GARCH(1,1) daily variance (unit mean) scaling each day's innovations and gap. Runs before returns exist."""
    persistence = float(cfg["persistence"])
    if not 0 <= persistence < 1:
        raise ValueError("vol_clustering persistence must be in [0, 1)")
    alpha = min(float(cfg.get("alpha", 0.1)), persistence / 2)
    beta, omega = persistence - alpha, 1.0 - persistence
    n_days, burn = len(grid.starts), 200

    def variance_path() -> np.ndarray:
        z = rng.standard_normal(n_days + burn)
        h = np.ones(n_days + burn)
        for d in range(1, n_days + burn):
            h[d] = omega + (beta + alpha * z[d - 1] ** 2) * h[d - 1]
        return h[burn:]

    shared = variance_path() if cfg.get("shared", False) else None
    variance = {}
    for path in paths:
        h = shared if shared is not None else variance_path()
        scale = grid.per_bar(np.sqrt(h))
        path.noise = path.noise * scale
        path.gaps = path.gaps * scale
        variance[path.symbol] = h
    planted["vol_clustering"] = {
        "persistence": persistence,
        "alpha": alpha,
        "daily_variance": pd.DataFrame(variance, index=grid.session_dates),
    }


def _plant_reversal(grid, paths, cfg, planted) -> None:
    ar = list(cfg["ar"])
    scale = float(cfg.get("scale", 1.0))
    poly = np.r_[1.0, -np.asarray(ar, dtype=float)]
    if np.any(np.abs(np.roots(poly[::-1][::-1])) >= 1.0):  # roots of z^p - phi_1 z^(p-1) - ... must be inside
        raise ValueError(f"reversal ar={ar} is not stationary")
    for path in paths:
        path.returns = lfilter([1.0], poly, path.noise * scale)
    planted["reversal"] = {"ar": ar, "scale": scale}


def _plant_factors(grid, paths, cfg, rng, planted) -> None:
    market = _as_dict(cfg.get("market", {}), "sigma", "factors.market")
    sectors = _as_dict(cfg.get("sectors", 3), "n", "factors.sectors")
    sigma_m = float(market.get("sigma", 0.5 * grid.unit))
    lo, hi = market.get("beta_range", (0.5, 1.5))
    n_sectors = int(sectors.get("n", 3))
    sigma_s = float(sectors.get("sigma", 0.3 * grid.unit))
    m = rng.normal(0, sigma_m, grid.n)
    sec = rng.normal(0, sigma_s, (n_sectors, grid.n))
    betas = rng.uniform(lo, hi, len(paths))
    assign = np.arange(len(paths)) % n_sectors
    rng.shuffle(assign)
    names = [f"SEC{k + 1}" for k in range(n_sectors)]
    for path, b, k in zip(paths, betas, assign, strict=True):
        part = b * m + sec[k]
        path.returns = path.returns + part
        path.factor_part = part
    symbols = [p.symbol for p in paths]
    planted["beta"] = pd.Series(betas, index=symbols, name="beta")
    planted["sector"] = pd.Series([names[k] for k in assign], index=symbols, name="sector")
    planted["factors"] = {
        "market_return": pd.Series(m, index=grid.ts, name="market"),
        "sector_return": pd.DataFrame(sec.T, index=grid.ts, columns=names),
    }


def _plant_xs_momentum(grid, paths, rng, planted, half_life: float = 100.0, drift_sd: float = 0.0012) -> None:
    phi = 0.5 ** (1.0 / half_life)
    n = grid.n
    mu = np.empty((len(paths), n))
    mu[:, 0] = rng.normal(0, drift_sd, len(paths))
    shocks = rng.normal(0, drift_sd * np.sqrt(1 - phi**2), (len(paths), n))
    for t in range(1, n):
        mu[:, t] = phi * mu[:, t - 1] + shocks[:, t]
    for path, m in zip(paths, mu, strict=True):
        path.returns = path.returns + m
    planted["daily"]["xs_momentum"] = {"half_life_sessions": half_life, "drift_sd": drift_sd}
    planted["daily"]["drift"] = pd.DataFrame(mu.T, index=grid.ts, columns=[p.symbol for p in paths])


def _plant_breakouts(grid, path, rng, *, prob, range_minutes, max_delay, drift_minutes, drift_bps, on_event=None):
    """Force or find a first close above the opening-range high and add a post-breakout drift. Returns events."""
    margin = 5e-4
    drift = drift_bps / 1e4 / drift_minutes
    events = []
    for s, (a, length) in enumerate(zip(grid.starts, grid.lengths, strict=True)):
        chosen, delay = rng.random() < prob, int(rng.integers(0, max_delay))
        if not chosen or length < range_minutes + max_delay + 2:
            continue
        sl = slice(a, a + length)
        local = np.cumsum(path.returns[sl] + path.gaps[sl])
        high_log = np.maximum(local - path.returns[sl], local) + path.wick[0][sl]
        h = high_log[:range_minutes].max()
        k0 = range_minutes + delay
        crossed = np.flatnonzero(local[range_minutes:k0] > h)
        if crossed.size:
            k = range_minutes + int(crossed[0])
        else:
            k = k0
            jump = h + margin - local[k0]
            if jump > 0:
                path.returns[a + k0] += jump
        verdict = on_event(a, length, k) if on_event is not None else {"drift_applied": True}
        if verdict["drift_applied"]:
            end = min(k + drift_minutes, length - 1)
            path.returns[a + k + 1 : a + end + 1] += drift
        events.append({"symbol": path.symbol, "ts": grid.ts[a + k], "session": grid.session_dates[s], **verdict})
    return events


def _plant_momentum(grid, paths, cfg, rng, planted) -> None:
    p = {"breakout_prob": 0.3, "drift_bps": 30.0, "drift_minutes": 30, "range_minutes": 30, "max_delay_minutes": 60}
    p.update(cfg)
    events = []
    for path in paths:
        events += _plant_breakouts(
            grid, path, rng, prob=p["breakout_prob"], range_minutes=int(p["range_minutes"]),
            max_delay=int(p["max_delay_minutes"]), drift_minutes=int(p["drift_minutes"]), drift_bps=p["drift_bps"],
        )
    p["events"] = pd.DataFrame(events, columns=["symbol", "ts", "session", "drift_applied"])
    p["events"] = p["events"].drop(columns="drift_applied")
    planted["momentum"] = p


def _plant_orb_rvol(grid, paths, cfg, rng, planted) -> None:
    p = {"drift_bps": 30.0, "drift_minutes": 30, "context": "SPY", "breakout_prob": 0.5, "high_share": 0.5,
         "range_minutes": 30, "max_delay_minutes": 60}
    p.update(cfg)
    thr = float(p["rvol_threshold"])
    events = []
    for path in paths:
        if path.vol_mult is None:
            path.vol_mult = np.ones(grid.n)

        def classify(a, length, k, path=path):
            above = bool(rng.random() < p["high_share"])
            rvol = thr * (rng.uniform(1.3, 2.5) if above else rng.uniform(0.3, 0.8))
            path.vol_mult[a : a + length] = rvol
            return {"rvol": rvol, "above": above, "drift_applied": above}

        events += _plant_breakouts(
            grid, path, rng, prob=p["breakout_prob"], range_minutes=int(p["range_minutes"]),
            max_delay=int(p["max_delay_minutes"]), drift_minutes=int(p["drift_minutes"]), drift_bps=p["drift_bps"],
            on_event=classify,
        )
    p["events"] = pd.DataFrame(events, columns=["symbol", "ts", "session", "rvol", "above", "drift_applied"])
    planted["orb_rvol"] = p


# --------------------------------------------------------------------------- extra symbols


def _null_names(value: Any) -> list[str]:
    if isinstance(value, bool):
        raise TypeError("null_symbols must be an int or a list of names")
    if isinstance(value, int):
        width = max(2, len(str(value)))
        return [f"N{i + 1:0{width}d}" for i in range(value)]
    return [str(s) for s in value]


def _pair_specs(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    trap: dict[str, Any] | None = None
    if isinstance(value, Mapping):
        pairs = [dict(x) for x in value.get("pairs", [])]
        if value.get("trap"):
            trap = dict(value["trap"]) if isinstance(value["trap"], Mapping) else {}
    else:
        pairs = []
        for x in value:
            x = dict(x)
            if x.pop("trap", False):
                trap = x
            else:
                pairs.append(x)
    return pairs, trap


def _plant_pairs(grid, value, rng, planted) -> list[_Path]:
    """Extra symbols: OU-spread pairs and the correlated-but-not-cointegrated trap."""
    pairs, trap = _pair_specs(value)
    extras: list[_Path] = []
    out: dict[str, Any] = {"pairs": []}
    for k, spec in enumerate(pairs, start=1):
        beta = float(spec.get("beta", 1.0))
        hl = float(spec.get("half_life_bars", 60))
        if hl <= 0:
            raise ValueError("half_life_bars must be positive")
        sd = float(spec.get("spread_sd", grid.spread_sd))
        phi = 0.5 ** (1.0 / hl)
        leg_b = _walk(rng, grid, f"P{k}B")
        leg_a = _draw_symbol(rng, grid, f"P{k}A")
        x0 = rng.normal(0, sd)
        x = lfilter([1.0], [1.0, -phi], rng.normal(0, sd * np.sqrt(1 - phi**2), grid.n), zi=[phi * x0])[0]
        leg_a.returns = beta * leg_b.returns + np.diff(np.r_[x0, x])
        leg_a.gaps = beta * leg_b.gaps
        intercept = float(np.log(leg_a.p0) - beta * np.log(leg_b.p0) - x0)
        extras += [leg_a, leg_b]
        out["pairs"].append({"symbols": (leg_a.symbol, leg_b.symbol), "beta": beta, "half_life_bars": hl,
                             "spread_sd": sd, "intercept": intercept})
    if trap is not None:
        corr = float(trap.get("corr", 0.8))
        leg_a, leg_b = _walk(rng, grid, "TRAPA"), _walk(rng, grid, "TRAPB")
        loading = corr * leg_b.sigma / leg_a.sigma
        mix = np.sqrt(1 - corr**2)
        leg_b.returns = loading * leg_a.noise + mix * leg_b.noise
        leg_b.gaps = loading * leg_a.gaps + mix * leg_b.gaps
        extras += [leg_a, leg_b]
        out["trap"] = {"symbols": (leg_a.symbol, leg_b.symbol), "corr": corr, "loading": float(loading)}
    planted["cointegrated_pairs"] = out
    return extras


# --------------------------------------------------------------------------- xs_alpha (built from the final returns)


def _plant_xs_alpha(grid, paths, cfg, rng, planted) -> None:
    ic = float(cfg["ic"])
    if not -1 < ic < 1:
        raise ValueError("xs_alpha ic must be in (-1, 1)")
    n_sym = len(paths)
    if n_sym < 3:
        raise ValueError("xs_alpha needs at least 3 symbols")
    total = np.column_stack([np.add.reduceat(p.returns + p.gaps, grid.starts) for p in paths])
    factor = np.column_stack(
        [np.add.reduceat(p.factor_part, grid.starts) if p.factor_part is not None else np.zeros(len(grid.starts))
         for p in paths]
    )
    resid = total - factor
    resid = resid - resid.mean(axis=1, keepdims=True)
    n_days = len(grid.starts)
    score = np.empty((n_days, n_sym))
    for d in range(n_days):
        w = rng.standard_normal(n_sym)
        w -= w.mean()
        if d == n_days - 1:
            score[d] = w / w.std()
            continue
        e = resid[d + 1]
        x = (e - e.mean()) / e.std()
        w -= (w @ x) / (x @ x) * x
        w /= w.std()
        score[d] = ic * x + np.sqrt(1 - ic**2) * w
    symbols = [p.symbol for p in paths]
    clean = pd.DataFrame(score, index=grid.session_dates, columns=symbols)
    result = clean
    contaminate = cfg.get("contaminate")
    if contaminate:
        if "beta" not in planted:
            raise ValueError("xs_alpha contaminate needs the factors option")
        beta = planted["beta"].reindex(symbols).to_numpy()
        beta_z = (beta - beta.mean()) / beta.std()
        names = sorted(set(planted["sector"]))
        member = np.array([names.index(s) for s in planted["sector"].reindex(symbols)])
        sector_score = rng.standard_normal((n_days, len(names)))[:, member]
        w_beta, w_sector = float(contaminate.get("beta", 0.0)), float(contaminate.get("sector", 0.0))
        result = clean + w_beta * beta_z + w_sector * sector_score
    planted["xs_alpha"] = {"ic": ic, "score": result, "score_clean": clean, "contaminate": contaminate}


# --------------------------------------------------------------------------- entry point


_OPTION_ORDER = ("vol_clustering", "reversal", "factors", "momentum", "orb_rvol")


def synthetic_bars(
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    start: str = "2026-03-02",
    end: str = "2026-08-28",
    seed: int = 7,
    signal_strength: float = 0.3,
    *,
    momentum: Mapping[str, Any] | None = None,
    orb_rvol: Mapping[str, Any] | None = None,
    reversal: Mapping[str, Any] | None = None,
    cointegrated_pairs: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    vol_clustering: Mapping[str, Any] | None = None,
    factors: Mapping[str, Any] | None = None,
    xs_alpha: Mapping[str, Any] | None = None,
    null_symbols: int | Sequence[str] | None = None,
    daily: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Contract-valid 1-minute regular-session bars for `symbols` over [start, end] sessions.

    With no keyword-only option the output is the original fixture, byte for byte. The options plant known
    effects (see the module docstring for each shape, minimum span and `attrs` entry); `bars.attrs["planted"]`
    holds the truth. Bar `ts` is the bar's start and `ts_available` its end; every planted quantity is built so
    that a decision at `ts_available` can in principle see it, and the unobservable truth (scores, drifts,
    events) lives only in `attrs`, never in the bars. All randomness derives from `seed`.
    """
    raw = {"momentum": momentum, "orb_rvol": orb_rvol, "reversal": reversal, "cointegrated_pairs": cointegrated_pairs,
           "vol_clustering": vol_clustering, "factors": factors, "xs_alpha": xs_alpha, "null_symbols": null_symbols,
           "daily": daily}
    options = {k: v for k, v in raw.items() if v is not None}
    for name in ("momentum", "reversal", "orb_rvol", "vol_clustering", "factors", "xs_alpha", "daily"):
        if name in options:
            options[name] = _as_dict(options[name], None, name)
    if "reversal" in options and "ar" not in options["reversal"]:
        raise ValueError("reversal needs 'ar', a list of AR coefficients")
    if "vol_clustering" in options and "persistence" not in options["vol_clustering"]:
        raise ValueError("vol_clustering needs 'persistence'")
    if "orb_rvol" in options and "rvol_threshold" not in options["orb_rvol"]:
        raise ValueError("orb_rvol needs 'rvol_threshold'")
    if "xs_alpha" in options and "ic" not in options["xs_alpha"]:
        raise ValueError("xs_alpha needs 'ic'")
    if "daily" in options:
        d = options["daily"]
        if int(d.get("years", _DAILY_MIN_YEARS)) < _DAILY_MIN_YEARS:
            raise ValueError(f"daily needs at least {_DAILY_MIN_YEARS} years")
        if int(d.get("n_symbols", _DAILY_MIN_SYMBOLS)) < _DAILY_MIN_SYMBOLS:
            raise ValueError(f"daily needs at least {_DAILY_MIN_SYMBOLS} symbols")
        bad = [k for k in _INTRADAY_ONLY if k in options]
        if bad:
            raise ValueError(f"{bad} need intraday bars and cannot be combined with daily")
        symbols = [f"S{i + 1:03d}" for i in range(int(d.get("n_symbols", _DAILY_MIN_SYMBOLS)))]

    grid = _make_grid(start, end, options)
    rng = np.random.default_rng(seed)
    base = [_draw_symbol(rng, grid, s) for s in symbols]
    if len({p.symbol for p in base}) != len(base):
        raise ValueError("duplicate symbols")
    planted: dict[str, Any] = {}
    if "daily" in options:
        planted["daily"] = {"years": int(options["daily"].get("years", _DAILY_MIN_YEARS)), "symbols": list(symbols)}

    if "vol_clustering" in options:
        _plant_vol_clustering(grid, base, options["vol_clustering"], _effect_rng(seed, "vol_clustering"), planted)
    for path in base:
        path.returns = _base_returns(path, grid, signal_strength)
    if "reversal" in options:
        _plant_reversal(grid, base, options["reversal"], planted)
    if "factors" in options:
        _plant_factors(grid, base, options["factors"], _effect_rng(seed, "factors"), planted)
    if options.get("daily", {}).get("xs_momentum"):
        _plant_xs_momentum(grid, base, _effect_rng(seed, "xs_momentum"), planted)
    if "momentum" in options:
        _plant_momentum(grid, base, options["momentum"], _effect_rng(seed, "momentum"), planted)
    if "orb_rvol" in options:
        _plant_orb_rvol(grid, base, options["orb_rvol"], _effect_rng(seed, "orb_rvol"), planted)

    taken = {p.symbol for p in base}
    extras: list[_Path] = []

    def add(path: _Path) -> None:
        if path.symbol in taken:
            raise ValueError(f"symbol {path.symbol!r} is already in the fixture")
        taken.add(path.symbol)
        extras.append(path)

    if "orb_rvol" in options:
        context = str(options["orb_rvol"].get("context", "SPY"))
        if context not in taken:
            add(_walk(_effect_rng(seed, "context"), grid, context))
    if "null_symbols" in options:
        names = _null_names(options["null_symbols"])
        nrng = _effect_rng(seed, "null_symbols")
        for name in names:
            add(_walk(nrng, grid, name))
        planted["null_symbols"] = names
    if "cointegrated_pairs" in options:
        for path in _plant_pairs(grid, options["cointegrated_pairs"], _effect_rng(seed, "cointegrated_pairs"), planted):
            add(path)
    if "xs_alpha" in options:
        _plant_xs_alpha(grid, base, options["xs_alpha"], _effect_rng(seed, "xs_alpha"), planted)

    bars = pd.concat([_frame(grid, p) for p in [*base, *extras]], ignore_index=True)
    bars["ts"] = bars["ts"].astype(UTC_NS)
    bars["ts_available"] = bars["ts_available"].astype(UTC_NS)
    bars["session"] = "regular"
    bars = bars.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
    validate_bars(bars)
    bars.attrs["planted"] = planted
    return bars
