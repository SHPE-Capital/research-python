"""Portfolio construction (T9): scores -> dollar-neutral weights -> trades -> net portfolio returns.

Conventions
-----------
* Scores, vol and weights are `pd.Series` indexed by a two-level `MultiIndex` `(decision_ts, symbol)`.
  A score at `decision_ts` uses only information known at that time (the caller's responsibility, as for any
  feature); nothing here looks across timestamps except `portfolio_returns`, which only moves forward in time.
* Weights are fractions of equity. A weight set at `decision_ts` is filled at the OPEN of the symbol's first bar
  with `ts >= decision_ts` (the next bar after the bars the decision saw) and held until the next decision's fill.
* Missing data never raises and never becomes a position: a symbol with a missing score (or vol) gets weight 0
  and is listed in the result's `attrs`; a date that cannot form a book is all zeros and is listed too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from shpe_research.trade.types import CostModel
from shpe_research.transforms.neutralize import Neutralize

__all__ = ["neutralize_weights", "portfolio_returns", "rebalance", "scores_to_weights"]

SCHEMES = ("top_k", "rank", "zscore", "vol_scaled")
_Z_CLIP = 3.0
_EPS = 1e-12


def _check_index(s: pd.Series, name: str) -> None:
    idx = s.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        raise ValueError(f"{name} must be indexed by (decision_ts, symbol)")
    if idx.has_duplicates:
        raise ValueError(f"{name} has duplicate (decision_ts, symbol) rows")


def _two_sided(w: np.ndarray, gross: float) -> np.ndarray | None:
    """Scale longs and shorts separately so each side has gross/2; None if a side is empty."""
    pos, neg = w > _EPS, w < -_EPS
    if not pos.any() or not neg.any():
        return None
    out = np.zeros_like(w)
    out[pos] = w[pos] / w[pos].sum() * gross / 2
    out[neg] = w[neg] / -w[neg].sum() * gross / 2
    return out


def _base_weights(s: np.ndarray, scheme: str, k: int | None) -> np.ndarray | None:
    """Unscaled raw weights from the finite scores `s` of one date; None if the date cannot form a book."""
    n = len(s)
    if scheme == "top_k":
        if k is None or n < 2 * k:
            return None
        order = np.argsort(-s, kind="stable")
        raw = np.zeros(n)
        raw[order[:k]] = 1.0
        raw[order[-k:]] = -1.0
        return raw
    if n < 2:
        return None
    if scheme == "rank":
        return rankdata(s) - (n + 1) / 2
    sd = s.std()
    if sd < _EPS:
        return None
    return np.clip((s - s.mean()) / sd, -_Z_CLIP, _Z_CLIP)


def scores_to_weights(
    scores: pd.Series, scheme: str, gross: float, top_k: int | None = None, vol: pd.Series | None = None,
    base: str = "zscore",
) -> pd.Series:
    """Dollar-neutral weights per `(decision_ts, symbol)` from cross-sectional scores.

    Inputs: `scores` indexed by `(decision_ts, symbol)`; `scheme` one of `top_k` (equal long top k, equal short
    bottom k; needs `top_k` and at least 2k scored names), `rank` (linear in centred rank), `zscore` (proportional
    to the z-score, clipped at +-3 then re-centred) or `vol_scaled` (`base` scheme, default `zscore`, divided by
    `vol` and rescaled; `vol` required, finite and positive). Output: a float Series on the input index with
    sum(w) = 0 and sum(|w|) = `gross` on every date that can form a book (`vol_scaled` scales each side to
    gross/2 so neutrality survives the division).

    Timing: each date uses only its own rows. Missing: a symbol whose score (or vol) is not finite gets weight 0;
    they are listed in `result.attrs["missing_score"]` / `["missing_vol"]` (MultiIndex). A date with too few names,
    zero dispersion or an empty side is all zeros and listed in `result.attrs["skipped_periods"]` (DatetimeIndex).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if gross <= 0:
        raise ValueError("gross must be positive")
    if scheme == "top_k" and (top_k is None or top_k < 1):
        raise ValueError("top_k scheme needs top_k >= 1")
    inner = base if scheme == "vol_scaled" else scheme
    if inner not in ("top_k", "rank", "zscore"):
        raise ValueError(f"base must be top_k, rank or zscore, got {base!r}")
    if inner == "top_k" and (top_k is None or top_k < 1):
        raise ValueError("base top_k needs top_k >= 1")
    _check_index(scores, "scores")
    values = scores.to_numpy(dtype="float64")
    if scheme == "vol_scaled":
        if vol is None:
            raise ValueError("vol_scaled needs a vol Series")
        _check_index(vol, "vol")
        vol_values = vol.reindex(scores.index).to_numpy(dtype="float64")
        usable_vol = np.isfinite(vol_values) & (vol_values > 0)
    else:
        vol_values = None
        usable_vol = np.ones(len(values), dtype=bool)
    finite = np.isfinite(values)
    out = np.zeros(len(values))
    skipped = []
    for ts, pos in scores.groupby(level=0, sort=False).indices.items():
        pos = np.asarray(pos)
        ok = pos[finite[pos] & usable_vol[pos]]
        raw = _base_weights(values[ok], inner, top_k) if len(ok) else None
        w = None
        if raw is not None:
            if vol_values is not None:
                w = _two_sided(raw / vol_values[ok], gross)
            else:
                centred = raw - raw.mean() if inner == "zscore" else raw
                total = np.abs(centred).sum()
                w = centred / total * gross if total > _EPS else None
        if w is None:
            skipped.append(ts)
        else:
            out[ok] = w
    result = pd.Series(out, index=scores.index, name="weight")
    result.attrs["missing_score"] = scores.index[~finite]
    result.attrs["missing_vol"] = scores.index[finite & ~usable_vol]
    result.attrs["skipped_periods"] = pd.DatetimeIndex(skipped, name="decision_ts")
    return result


def neutralize_weights(
    weights: pd.Series, exposures: pd.DataFrame, proportion: float, gross: float | None = None,
) -> pd.Series:
    """Remove `proportion` of the exposures (beta, sector dummies, ...) from weights, then rescale to gross.

    Calls C1d's `neutralize` (a per-`decision_ts` OLS of the weights on the exposures plus an intercept, so the
    weight sum is preserved) and rescales each date to `gross` (default: that date's gross before neutralizing).
    `exposures` is indexed like `weights`, one column per exposure, known at `decision_ts` (rolling beta on past
    data, point-in-time sector). With `proportion=1` the book exposure sum(w * exposure) is zero on each date;
    `0` returns the input. Missing: names with a missing score (from `weights.attrs["missing_score"]`) or any
    missing exposure are left out of the fit and get weight 0; a date with too few names for the regression is all
    zeros. Clip or apply risk limits AFTER this and recheck the exposures.
    """
    _check_index(weights, "weights")
    exposures = exposures.reindex(weights.index)
    if exposures.shape[1] == 0:
        raise ValueError("exposures needs at least one column")
    cols = [f"_exp{i}" for i in range(exposures.shape[1])]
    frame = pd.DataFrame(exposures.to_numpy(dtype="float64"), columns=cols, index=weights.index)
    frame["_w"] = weights.to_numpy(dtype="float64")
    missing = weights.attrs.get("missing_score")
    if missing is not None and len(missing):
        frame.loc[frame.index.isin(missing), "_w"] = np.nan
    frame["decision_ts"] = weights.index.get_level_values(0)
    flat = frame.reset_index(drop=True)
    resid = Neutralize(column="_w", exposures=cols, proportion=proportion, min_count=len(cols) + 2).compute(flat)
    resid.index = weights.index
    resid = resid.fillna(0.0)
    before = weights.abs().groupby(level=0, sort=False).sum()
    after = resid.abs().groupby(level=0, sort=False).sum()
    target = before if gross is None else pd.Series(gross, index=before.index)
    scale = (target / after.where(after > _EPS)).fillna(0.0)
    out = resid * scale.reindex(weights.index.get_level_values(0)).to_numpy()
    out.name = "weight"
    out.attrs.update(weights.attrs)
    return out


def rebalance(
    target: pd.Series, current: pd.Series, no_trade_band: float = 0.0, partial_adjustment: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Move current weights toward target; returns `(new_weights, trades)` per symbol (new = current + trades).

    Inputs are weights by symbol for one decision time (current = the book as drifted to now, not the previous
    target). A symbol is skipped (trade 0) when `|target - current| < no_trade_band`; otherwise it trades
    `partial_adjustment * (target - current)`, with `partial_adjustment` in (0, 1]. A symbol missing from either
    input counts as 0. Turnover of the step is `trades.abs().sum() / 2`. Skipping and partial moves leave the
    book short of exactly neutral or at less than target gross; that is the price of trading less.
    """
    if not 0 < partial_adjustment <= 1:
        raise ValueError("partial_adjustment must be in (0, 1]")
    if no_trade_band < 0:
        raise ValueError("no_trade_band must be >= 0")
    symbols = target.index.union(current.index, sort=False)
    t = target.reindex(symbols).fillna(0.0)
    c = current.reindex(symbols).fillna(0.0)
    gap = t - c
    trades = (partial_adjustment * gap).where(gap.abs() >= no_trade_band, 0.0)
    return c + trades, trades


def _ns(x: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(x)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    return idx.as_unit("ns").asi8


def portfolio_returns(
    weights: pd.Series, bars: pd.DataFrame, cost_model: CostModel | None = None, equity: float = 100_000.0,
) -> pd.DataFrame:
    """Per-period gross return, cost, net return and turnover of a weight book.

    `weights` (fractions of equity, indexed `(decision_ts, symbol)`) are the positions wanted after each
    rebalance; `bars` has `symbol`, `ts`, `open`. Weights set at `decision_ts` are filled at the open of each
    symbol's first bar with `ts >= decision_ts` and earn the open-to-open return to the next decision's fill, so
    the bar a decision was made on never prices the trade. Rows are indexed by `decision_ts`, one per decision
    except the last (its holding period is not observed).

    Holdings drift with returns between decisions, so turnover = sum(|w_target - w_drifted|) / 2 against the
    drifted book, starting from cash (the opening build is charged). Cost is T5's `CostModel.cost` on every
    traded notional (`equity` converts weights to shares for the per-share commission), charged to the period the
    trade opens, as a fraction of equity. Prices are mid (open), so spread and slippage are charged once, in `cost`.
    Missing: a symbol with no bar at or after a fill time cannot be priced; its return is 0, it is not traded,
    and it is counted in `n_missing_price`.
    """
    model = cost_model or CostModel()
    _check_index(weights, "weights")
    for col in ("symbol", "ts", "open"):
        if col not in bars.columns:
            raise ValueError(f"bars needs a {col!r} column")
    wide = weights.fillna(0.0).unstack("symbol").sort_index().fillna(0.0)
    decisions, symbols = wide.index, list(wide.columns)
    n = len(decisions)
    d_ns = _ns(decisions)
    prices = np.full((n, len(symbols)), np.nan)
    for j, sym in enumerate(symbols):
        b = bars[bars["symbol"] == sym].sort_values("ts")
        if b.empty:
            continue
        ts_ns = _ns(b["ts"])
        pos = np.searchsorted(ts_ns, d_ns, side="left")
        have = pos < len(ts_ns)
        prices[have, j] = b["open"].to_numpy(dtype="float64")[pos[have]]
    w = wide.to_numpy(dtype="float64")
    cols = ("gross_return", "cost", "net_return", "turnover", "n_missing_price")
    if n < 2:
        return pd.DataFrame(columns=list(cols), index=pd.DatetimeIndex([], name="decision_ts", tz=decisions.tz))
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = prices[1:] / prices[:-1] - 1.0
    priced = np.isfinite(prices)
    rows = []
    drifted = np.zeros(len(symbols))
    for k in range(n - 1):
        wk = np.where(priced[k], w[k], 0.0)  # an unpriceable name is not held
        delta = wk - drifted
        r = np.where(np.isfinite(ret[k]) & priced[k], ret[k], 0.0)
        cost_usd = 0.0
        for j in np.flatnonzero(np.abs(delta) > _EPS):
            side = "buy" if delta[j] > 0 else "sell"
            cost_usd += model.cost(side, float(prices[k, j]), abs(delta[j]) * equity / float(prices[k, j]))[0]
        gross_ret = float(wk @ r)
        cost = cost_usd / equity
        rows.append((gross_ret, cost, gross_ret - cost, 0.5 * float(np.abs(delta).sum()),
                     int(((w[k] != 0) & ~(priced[k] & np.isfinite(ret[k]))).sum())))
        drifted = wk * (1.0 + r) / (1.0 + gross_ret)
    return pd.DataFrame(rows, index=decisions[:-1], columns=list(cols))

