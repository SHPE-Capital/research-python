"""Sizing rules (T2): turn candidates into whole-share quantities.

Every rule floors to whole shares, like the platform's ``fixedNotionalSizer``
(``floor(maxPositionSizeUsd / estimatedPrice)``). A result <= 0, or any input
that is missing or non-finite, gives 0, which the simulator reports as
``unfilled`` with ``skip_reason = size_zero``.

Timing: ``price`` is the close of the decision bar, i.e. the last bar of the
symbol with ``ts_available <= decision_ts``. Candidate columns used (stop
price, vol, probability, native qty) must be known at ``decision_ts``.

Rule parameters:

- ``native``: none; uses the candidate's own ``qty`` column.
- ``fixed_qty``: ``qty``.
- ``fixed_notional``: ``notional_usd``.
- ``risk_per_trade``: ``risk_usd`` or ``risk_pct`` (of equity); candidate ``stop_price``.
- ``inverse_vol``: ``target_risk_usd``, ``vol_column`` (return vol per bar, as a fraction).
- ``prob_to_size``: ``prob_column``, ``min_prob``, ``max_notional_usd``; linear from 0 at
  ``min_prob`` to the maximum at probability 1.
- ``kelly_fractional``: ``fraction``, ``prob_column``, ``payoff_ratio``, ``cap_notional_usd``;
  Kelly f = p - (1 - p) / b, notional = fraction * f * equity, capped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from shpe_research.trade.types import SizingRule

#: Rules the platform can run today (``qtyPerTrade``, ``fixed_notional``/``tradeNotionalUsd``, strategy-native qty).
EXPORTABLE_RULES: frozenset[str] = frozenset({"fixed_qty", "fixed_notional", "native"})

_REQUIRED: dict[str, tuple[str, ...]] = {
    "native": (),
    "fixed_qty": ("qty",),
    "fixed_notional": ("notional_usd",),
    "risk_per_trade": (),
    "inverse_vol": ("target_risk_usd", "vol_column"),
    "prob_to_size": ("prob_column", "min_prob", "max_notional_usd"),
    "kelly_fractional": ("fraction", "prob_column", "payoff_ratio", "cap_notional_usd"),
}


def validate_rule(rule: str, params: dict[str, Any]) -> None:
    """Raise ValueError for an unknown rule or a missing parameter."""
    if rule not in _REQUIRED:
        raise ValueError(f"unknown sizing rule {rule!r}; choose from {sorted(_REQUIRED)}")
    missing = [p for p in _REQUIRED[rule] if p not in params]
    if rule == "risk_per_trade" and "risk_usd" not in params and "risk_pct" not in params:
        missing.append("risk_usd or risk_pct")
    if missing:
        raise ValueError(f"sizing rule {rule!r} is missing parameters: {missing}")


def _entry_price(candidates: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """Close of the decision bar per candidate; NaN if the symbol has no bar available by decision_ts."""
    out = pd.Series(np.nan, index=candidates.index, dtype="float64")
    if candidates.empty or bars.empty:
        return out
    for sym, grp in candidates.groupby("symbol", sort=False):
        b = bars.loc[bars["symbol"] == sym, ["ts_available", "close"]].sort_values("ts_available")
        if b.empty:
            continue
        # positional join keeps duplicate candidate indexes safe
        left = pd.DataFrame({"decision_ts": pd.to_datetime(grp["decision_ts"], utc=True), "pos": np.arange(len(grp))})
        m = pd.merge_asof(left.sort_values("decision_ts"), b, left_on="decision_ts", right_on="ts_available")
        vals = np.full(len(grp), np.nan)
        vals[m["pos"].to_numpy()] = m["close"].to_numpy()
        out.loc[grp.index] = vals
    return out


def _col(candidates: pd.DataFrame, name: str) -> pd.Series:
    if name not in candidates.columns:
        raise ValueError(f"candidates have no column {name!r} required by the sizing rule")
    return pd.to_numeric(candidates[name], errors="coerce").astype("float64")


def _to_shares(raw: pd.Series) -> pd.Series:
    """Floor to whole shares; NaN, inf and values <= 0 become 0."""
    v = raw.where(np.isfinite(raw), 0.0)
    return np.floor(v.clip(lower=0.0)).astype("int64")


def size_candidates(rule: SizingRule, candidates: pd.DataFrame, bars: pd.DataFrame, equity: float) -> pd.Series:
    """Whole-share quantity per candidate (int64, same index); 0 means no trade.

    Inputs: candidates with ``symbol`` and ``decision_ts`` (plus the rule's columns), contract
    ``bars``, and account ``equity`` in USD. Price is the decision-bar close. A missing bar or a
    missing or non-finite input gives 0.
    """
    p = rule.params
    name = rule.rule
    if name == "native":
        if "qty" not in candidates.columns:
            return pd.Series(0, index=candidates.index, dtype="int64")
        return _to_shares(_col(candidates, "qty"))
    if name == "fixed_qty":
        return pd.Series(int(max(np.floor(p["qty"]), 0)), index=candidates.index, dtype="int64")

    price = _entry_price(candidates, bars)
    price = price.where(price > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "fixed_notional":
            raw = p["notional_usd"] / price
        elif name == "risk_per_trade":
            risk = p["risk_usd"] if "risk_usd" in p else p["risk_pct"] * equity
            raw = risk / (price - _col(candidates, "stop_price")).abs()
        elif name == "inverse_vol":
            raw = p["target_risk_usd"] / (price * _col(candidates, p["vol_column"]))
        elif name == "prob_to_size":
            frac = (_col(candidates, p["prob_column"]) - p["min_prob"]) / (1.0 - p["min_prob"])
            raw = frac.clip(lower=0.0, upper=1.0) * p["max_notional_usd"] / price
        else:  # kelly_fractional
            pr = _col(candidates, p["prob_column"])
            kelly = pr - (1.0 - pr) / p["payoff_ratio"]
            notional = (p["fraction"] * kelly.clip(lower=0.0) * equity).clip(upper=p["cap_notional_usd"])
            raw = notional / price
    return _to_shares(raw)
