"""Pair selection on a formation window (SS1).

`select_pairs(prices, formation, method, **params)` picks candidate pairs from a wide price panel
using only prices inside the formation window, and reports how many pairs it tested.

Inputs and timing. `prices` is a DataFrame of prices (rows indexed by the time each value is KNOWN,
i.e. `ts_available`; columns are symbols). Only rows with index in `[start, end]` are read. A
date-only string `end` (for example "2026-04-10") means the end of that day. Anything after `end`
is never looked at, so selection cannot see the trading window. Spreads are built on log prices,
`log leg1 - beta * log leg2`, with `leg1` the earlier column of `prices`. Engle-Granger is not
symmetric; this fixed orientation is the convention and only one test is counted per pair.

Missing data. A pair uses the rows where both legs have a price. A pair with fewer than
`MIN_PAIR_OBSERVATIONS` such rows is still counted in `n_tested` but gets NaN statistics and fails
every filter. A symbol with fewer than `MIN_PAIR_OBSERVATIONS` prices in the window is dropped
before pairs are formed (so it is not counted).

Methods (the `score` column is the ranking key; rows come out best first):

    all_pairs    every pair (optionally within `sectors`); score = Engle-Granger p-value, ascending
    distance     every pair; score = sum of squared differences of prices normalised to 1 at the
                 first formation price (Gatev et al. 2006), ascending
    correlation  every pair; score = return correlation, descending
    pca_cluster  pairs inside an OPTICS cluster of PCA return loadings; score = EG p-value

`pvalue_max` defaults to 0.05 for every method except `distance`, which is a pure price-distance
baseline and does not filter on cointegration unless you ask. `max_pairs` is applied last.

Trials: `n_tested` is the number of pairs evaluated. Record it with the experiment's trial count.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS

from shpe_research.features.cross_asset.pair_math import engle_granger, engle_granger_detail, fit_ou
from shpe_research.strategies.statarb.hedge import hedge_ratio

__all__ = [
    "MIN_PAIR_OBSERVATIONS",
    "PAIRSET_COLUMNS",
    "SELECTION_METHODS",
    "PairSet",
    "pca_clusters",
    "prices_from_bars",
    "select_pairs",
]

#: A PairSet is a DataFrame with `PAIRSET_COLUMNS` and `attrs` `n_tested`, `method`, `formation`
#: (and `clusters` for `pca_cluster`: a Series of cluster label by symbol, -1 for noise).
PairSet = pd.DataFrame
PAIRSET_COLUMNS = ["leg1", "leg2", "score", "beta", "half_life_bars", "coint_pvalue", "cluster"]
SELECTION_METHODS = ("all_pairs", "distance", "correlation", "pca_cluster")
MIN_PAIR_OBSERVATIONS = 30
_DEFAULT_PVALUE_MAX = {"all_pairs": 0.05, "distance": 1.0, "correlation": 0.05, "pca_cluster": 0.05}


def prices_from_bars(bars: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """Wide close-price panel from contract bars, indexed by `ts_available` (the time each close is known).

    With `freq` (a pandas offset such as "30min" or "1D") the last close in each interval is kept and
    stamped at the interval's right edge, so a value is never stamped before it is known.
    Symbols missing in an interval give NaN.
    """
    wide = bars.pivot(index="ts_available", columns="symbol", values="close").sort_index()
    wide.columns.name = None
    if freq is not None:
        wide = wide.resample(freq, label="right", closed="right").last().dropna(how="all")
    return wide


def _window(prices: pd.DataFrame, formation: tuple[Any, Any]) -> pd.DataFrame:
    """Rows of `prices` inside the formation window, keeping symbols with enough prices."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices needs a DatetimeIndex")
    tz = prices.index.tz

    def conv(x: Any, *, is_end: bool) -> pd.Timestamp:
        date_only = isinstance(x, str) and ":" not in x and "T" not in x
        ts = pd.Timestamp(x)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        if date_only and is_end:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
        return ts

    start, end = conv(formation[0], is_end=False), conv(formation[1], is_end=True)
    if start > end:
        raise ValueError("formation start is after its end")
    win = prices.loc[(prices.index >= start) & (prices.index <= end)]
    win = win.loc[:, win.count() >= MIN_PAIR_OBSERVATIONS]
    if win.shape[1] < 2:
        raise ValueError("fewer than 2 symbols have enough prices inside the formation window")
    return win


def _pair_stats(
    log1: pd.Series, log2: pd.Series, hedge: str | None, hedge_params: Mapping[str, Any]
) -> tuple[float, float, float]:
    """(beta, half_life_bars, Engle-Granger p-value) for one pair on formation-window log prices."""
    nan = float("nan")
    both = pd.concat([log1, log2], axis=1).dropna()
    if len(both) < MIN_PAIR_OBSERVATIONS:
        return nan, nan, nan
    y, x = both.iloc[:, 0], both.iloc[:, 1]
    eg = engle_granger_detail(y, x)
    if eg is None:
        return nan, nan, nan
    pvalue = engle_granger(y, x)[1]
    beta = eg.beta
    if hedge is not None:
        beta = float(hedge_ratio(y, x, hedge, **hedge_params).iloc[-1])
    return beta, fit_ou(y - beta * x).half_life_bars, pvalue


def pca_clusters(
    prices: pd.DataFrame,
    formation: tuple[Any, Any],
    n_components: int = 5,
    min_samples: int = 3,
    drop_market: bool = True,
) -> pd.Series:
    """Cluster label by symbol from OPTICS on PCA loadings of standardised formation-window log returns.

    Timing: formation-window rows only. Returns are standardised with formation-window means and
    standard deviations; the loadings are the first `n_components` right singular vectors (one row per
    symbol). With `drop_market` the first component, the market factor on which every symbol loads and
    which only differs by beta, is left out of the clustering features. Each row is normalised to unit
    length before OPTICS (`min_samples`). Label -1 is noise. Rows with any missing return are dropped.
    """
    win = _window(prices, formation)
    rets = np.log(win.where(win > 0)).diff().iloc[1:].dropna(axis=0, how="any")
    if len(rets) < MIN_PAIR_OBSERVATIONS:
        raise ValueError("too few complete return rows in the formation window for PCA")
    std = rets.std(ddof=1).replace(0, np.nan)
    z = ((rets - rets.mean()) / std).dropna(axis=1)
    k = max(1, min(n_components, z.shape[1] - 1, len(z) - 1))
    _, _, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    load = vt[1 if drop_market and k > 1 else 0 : k].T
    norms = np.linalg.norm(load, axis=1, keepdims=True)
    load = load / np.where(norms == 0, 1.0, norms)
    ms = max(2, min(min_samples, len(z.columns)))
    labels = OPTICS(min_samples=ms).fit(load).labels_
    return pd.Series(labels, index=z.columns, name="cluster")


def select_pairs(
    prices: pd.DataFrame,
    formation: tuple[Any, Any],
    method: str = "all_pairs",
    *,
    max_pairs: int | None = None,
    min_half_life_bars: float = 0.0,
    max_half_life_bars: float = math.inf,
    pvalue_max: float | None = None,
    hedge: str | None = None,
    sectors: Mapping[str, str] | None = None,
    n_components: int = 5,
    min_samples: int = 3,
    drop_market: bool = True,
    **hedge_params: Any,
) -> PairSet:
    """Choose candidate pairs from `prices` using only rows inside `formation = (start, end)`.

    Inputs: a wide price panel (see the module docstring for timing and missing data), the window, a
    method from `SELECTION_METHODS`, and filters. `hedge` is an SS2 `HEDGE_REGISTRY` key; left as None
    the beta is the formation-window OLS slope of log leg1 on log leg2 (the one the Engle-Granger test
    uses). With a key, beta is that estimator's last value on the formation window and `hedge_params`
    are passed to it. The p-value is always the Engle-Granger p-value of the OLS pair. `sectors`
    (symbol -> sector) limits `all_pairs` to same-sector pairs. `n_components` and `min_samples` are
    for `pca_cluster` (`drop_market` as in `pca_clusters`).

    Output: a PairSet (columns `PAIRSET_COLUMNS`, best first) with `attrs` `n_tested` (every pair
    evaluated, before the p-value, half-life and `max_pairs` filters), `method`, `formation` (the
    window as given), and `clusters` for `pca_cluster`. `cluster` is NA except for `pca_cluster`.
    A pair passes the half-life screen if `min <= half_life_bars <= max`; NaN statistics fail it.
    `pvalue_max=1.0` turns the p-value filter off (pairs with an undefined test are still dropped).
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {SELECTION_METHODS}")
    win = _window(prices, formation)
    logp = np.log(win.where(win > 0))
    symbols = list(win.columns)
    clusters: pd.Series | None = None

    if method == "pca_cluster":
        clusters = pca_clusters(prices, formation, n_components, min_samples, drop_market)
        pairs = [
            (a, b)
            for a, b in itertools.combinations(symbols, 2)
            if a in clusters.index and b in clusters.index and clusters[a] == clusters[b] and clusters[a] >= 0
        ]
    else:
        pairs = list(itertools.combinations(symbols, 2))
        if method == "all_pairs" and sectors is not None:
            pairs = [(a, b) for a, b in pairs if a in sectors and b in sectors and sectors[a] == sectors[b]]

    norm = win / win.apply(lambda c: c.dropna().iloc[0]) if method == "distance" else None
    rets = logp.diff() if method == "correlation" else None

    rows = []
    for a, b in pairs:
        beta, hl, pval = _pair_stats(logp[a], logp[b], hedge, hedge_params)
        if method in ("all_pairs", "pca_cluster"):
            score = pval
        elif method == "distance":
            d = (norm[a] - norm[b]).dropna()
            score = float((d**2).sum()) if len(d) >= MIN_PAIR_OBSERVATIONS else float("nan")
        else:
            score = float(rets[a].corr(rets[b]))
        cluster = int(clusters[a]) if clusters is not None else pd.NA
        rows.append((a, b, score, beta, hl, pval, cluster))

    out = pd.DataFrame(rows, columns=PAIRSET_COLUMNS)
    out["cluster"] = out["cluster"].astype("Int64")
    n_tested = len(out)
    pmax = _DEFAULT_PVALUE_MAX[method] if pvalue_max is None else pvalue_max
    keep = out["half_life_bars"].between(min_half_life_bars, max_half_life_bars) & out["score"].notna()
    keep &= out["coint_pvalue"] <= pmax if pmax < 1.0 else out["coint_pvalue"].notna()
    out = out[keep]
    out = out.sort_values(
        ["score", "leg1", "leg2"], ascending=[method != "correlation", True, True], kind="stable"
    )
    if max_pairs is not None:
        out = out.head(max_pairs)
    out = out.reset_index(drop=True)
    out.attrs.update({"n_tested": n_tested, "method": method, "formation": tuple(formation)})
    if clusters is not None:
        out.attrs["clusters"] = clusters
    return out
