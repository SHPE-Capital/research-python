"""F8: market features against a benchmark context symbol (SPY by default).

Inputs: one traded symbol's bars and the benchmark's bars (`context={"SPY": ...}`, a context symbol that is
read but never traded). Prices are as supplied in the bars (split-adjustment is the data layer's concern).

Timing. A row is decided at its `ts_available`. The benchmark value on a row is the benchmark bar of the SAME
session whose `ts_available` is the latest one <= the row's (`align_context`, backward only). That bar must
also be fresh: if it is a full bar interval or more older than the row (a missing benchmark minute), the value
is NaN. Nothing is forward-filled across a gap and nothing is carried over from the previous session.

Features (all float64, NaN where unknown):

* `MarketReturn`   -> `market_return_<benchmark>`: benchmark close / benchmark session open - 1.
* `MarketBeta`     -> `beta_<benchmark>_<w>` (output="beta") and `resid_return_<benchmark>_<w>`
                      (output="resid_return"). Beta is the OLS slope of the symbol's 1-bar returns on the
                      benchmark's over the previous `window_bars` rows, EXCLUDING the current row; the residual
                      is the current 1-bar return minus beta times the benchmark's. 1-bar returns are
                      simple close-to-close changes within a session (the first bar of a session has none).
                      A returned pair counts only if both returns are finite. Warm-up (fewer than
                      max(3, window_bars // 2) usable pairs) is NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.features.base import Feature, align_context, register_feature

_DEFAULT_BAR = pd.Timedelta(minutes=1)


def _seconds(ts: pd.Series) -> np.ndarray:
    return ((pd.DatetimeIndex(ts).tz_convert("UTC") - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(
        dtype=float
    )


def _prepare(ctx: pd.DataFrame, column: str) -> pd.DataFrame:
    ctx = ctx.sort_values("ts", kind="stable")
    if "session" in ctx.columns:
        ctx = ctx[ctx["session"] == "regular"]
    return ctx.reset_index(drop=True).assign(_avail_s=lambda d: _seconds(d["ts_available"]))


def align_context_exact(bars: pd.DataFrame, ctx: pd.DataFrame, column: str, session_open: bool = False) -> pd.Series:
    """`ctx[column]` (or the context's session open) at each row of `bars`, NaN when the context bar is missing.

    Uses `align_context` (latest context bar of the same session with `ts_available <=` the row's), then
    drops values whose bar is a full bar interval or more older than the row, so a missing context minute
    gives NaN instead of the previous minute's value. The bar interval is the median spacing of
    `bars.ts_available` (one minute if it cannot be measured). With `session_open=True` the value is the
    open of the context's first bar of that session, which is known from that bar's `ts_available` on.
    """
    ctx = _prepare(ctx, column)
    if session_open:
        ctx["_value"] = ctx.groupby(session_date(ctx["ts"]).to_numpy())["open"].transform("first")
        column = "_value"
    values = align_context(bars, ctx, column)
    seen = align_context(bars, ctx, "_avail_s")
    gaps = np.diff(bars["ts_available"].to_numpy())
    tol = pd.Timedelta(np.median(gaps)) if len(gaps) else _DEFAULT_BAR
    if tol <= pd.Timedelta(0):
        tol = _DEFAULT_BAR
    age = _seconds(bars["ts_available"]) - seen.to_numpy()
    return values.where(~(age >= tol.total_seconds()))


@register_feature
class MarketReturn(Feature):
    """Benchmark return since its session open: latest available benchmark close / session open - 1.

    Output `market_return_<benchmark>`. Known at the row's `ts_available`; NaN if the benchmark has no fresh
    bar for that row. No warm-up beyond the benchmark's first bar of the session.
    """

    feature_type = "market_return"
    requires = ("ts_available",)

    def __init__(self, benchmark: str = "SPY"):
        self.benchmark = benchmark
        self.context_symbols = (benchmark,)

    @property
    def name(self) -> str:
        return f"market_return_{self.benchmark.lower()}"

    def compute(self, bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.Series:
        if not context or self.benchmark not in context:
            raise ValueError(f"{self.name} needs context bars for {self.benchmark}")
        ctx = context[self.benchmark]
        close = align_context_exact(bars, ctx, "close")
        open_ = align_context_exact(bars, ctx, "open", session_open=True)
        return (close / open_ - 1.0).astype("float64")


@register_feature
class MarketBeta(Feature):
    """Rolling beta to the benchmark, or the residual return after removing it (see the module notes).

    `output="beta"` -> `beta_<benchmark>_<window_bars>`; `output="resid_return"` ->
    `resid_return_<benchmark>_<window_bars>`. Beta at a row uses the previous `window_bars` rows only; the
    residual uses beta and the current row's returns, all known at the row's `ts_available`.
    """

    feature_type = "market_beta"
    requires = ("close", "ts_available")
    OUTPUTS = ("beta", "resid_return")

    def __init__(self, benchmark: str = "SPY", window_bars: int = 390, output: str = "beta"):
        if output not in self.OUTPUTS:
            raise ValueError(f"output must be one of {self.OUTPUTS}, got {output!r}")
        if int(window_bars) < 5:
            raise ValueError("window_bars must be at least 5")
        self.benchmark = benchmark
        self.window_bars = int(window_bars)
        self.output = output
        self.context_symbols = (benchmark,)

    @property
    def name(self) -> str:
        prefix = "beta" if self.output == "beta" else "resid_return"
        return f"{prefix}_{self.benchmark.lower()}_{self.window_bars}"

    def compute(self, bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.Series:
        if not context or self.benchmark not in context:
            raise ValueError(f"{self.name} needs context bars for {self.benchmark}")
        sessions = session_date(bars["ts_available"]).to_numpy()
        bench = align_context_exact(bars, context[self.benchmark], "close")
        rs = bars["close"].astype(float).groupby(sessions).pct_change(fill_method=None)
        rb = bench.groupby(sessions).pct_change(fill_method=None)
        valid = rs.notna() & rb.notna()
        x = rb.where(valid, 0.0)
        y = rs.where(valid, 0.0)
        w = self.window_bars

        def past(s: pd.Series) -> pd.Series:
            return s.rolling(w, min_periods=1).sum().shift(1)

        n, sx, sy = past(valid.astype(float)), past(x), past(y)
        sxy, sxx = past(x * y), past(x * x)
        var = sxx - sx * sx / n
        cov = sxy - sx * sy / n
        enough = (n >= max(3, w // 2)) & (var > 1e-18)
        beta = (cov / var).where(enough)
        if self.output == "beta":
            return beta.astype("float64")
        return (rs - beta * rb).astype("float64")
