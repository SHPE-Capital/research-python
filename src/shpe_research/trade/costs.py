"""Transaction-cost arithmetic for `CostModel` (T5).

Convention: the platform bar model puts half-spread and slippage into the *fill price*
(buy: mid x (1 + h) x (1 + s); sell: mid x (1 - h) x (1 - s)). `cost` reports those two terms for
attribution and adds commission and the optional impact term. Because the fill price already carries
spread and slippage, a ledger's `net_return` must subtract only the commission (and impact, which the
platform fill does not model) from the fill-to-fill return; `round_trip` does exactly that.

Timing: nothing here looks at data over time. Each function takes the prices of one fill.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from shpe_research.trade.types import CostModel, FillModel

_BUY = {"buy", "long"}
_SELL = {"sell", "short"}
_TCA_FIELDS = ("half_spread_bps", "slippage_bps", "commission_per_share", "impact_coef", "impact_exponent")


def _is_buy(side: str) -> bool:
    s = str(side).lower()
    if s in _BUY:
        return True
    if s in _SELL:
        return False
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def fill_price(model: CostModel, side: str, mid: float) -> float:
    """Platform fill price for a market order on reference price `mid` (spread and slippage, no commission)."""
    h, s = model.half_spread_bps / 1e4, model.slippage_bps / 1e4
    return mid * (1 + h) * (1 + s) if _is_buy(side) else mid * (1 - h) * (1 - s)


def cost(model: CostModel, side: str, price: float, qty: float, adv: float | None = None) -> tuple[float, float]:
    """Cost of one fill: `(cost_usd, cost_bps)`.

    `price` is the reference (mid / next-open) price before spread and slippage; `qty` is shares;
    `adv` is average daily volume in shares, known at decision time (optional). `cost_usd` = spread +
    slippage (exactly the gap between fill price and `price`, times `qty`) + commission + impact.
    Impact in bps = `impact_coef * (qty / adv) ** impact_exponent` when `adv` is given and positive.
    `cost_bps` is `cost_usd` over notional `price * qty`. Zero quantity costs nothing.
    """
    _is_buy(side)
    if qty <= 0 or price <= 0:
        return 0.0, 0.0
    notional = price * qty
    slip_usd = abs(fill_price(model, side, price) - price) * qty
    impact_bps = model.impact_coef * (qty / adv) ** model.impact_exponent if adv else 0.0
    usd = slip_usd + model.commission_per_share * qty + notional * impact_bps / 1e4
    return usd, usd / notional * 1e4


def round_trip(
    side: str, entry_mid: float, exit_mid: float, qty: float,
    costs: CostModel | None = None, fill: FillModel | None = None, adv: float | None = None,
) -> dict[str, float]:
    """Price a round trip from mid prices; each cost is charged exactly once.

    `side` is the position side (`long` / `short`). Entry and exit fill at the platform fill price
    (spread and slippage inside the price). `net_usd` = fill-to-fill P&L minus commission and impact on
    both legs. Returns entry_fill, exit_fill, gross_return and net_return (on entry notional
    `entry_mid * qty`), gross_usd, cost_usd (all costs, both legs) and net_usd.
    If `fill` is given it must agree with `costs` on spread, slippage and commission.
    """
    from shpe_research.trade.types import CostModel as _CostModel

    cm = costs or _CostModel()
    if fill is not None and (fill.half_spread_bps, fill.slippage_bps, fill.commission_per_share) != (
        cm.half_spread_bps, cm.slippage_bps, cm.commission_per_share
    ):
        raise ValueError("FillModel and CostModel disagree; spread/slippage would be charged twice or not at all")
    long = _is_buy(side)
    sign = 1.0 if long else -1.0
    in_side, out_side = ("buy", "sell") if long else ("sell", "buy")
    entry_fill, exit_fill = fill_price(cm, in_side, entry_mid), fill_price(cm, out_side, exit_mid)
    in_usd, _ = cost(cm, in_side, entry_mid, qty, adv)
    out_usd, _ = cost(cm, out_side, exit_mid, qty, adv)
    cost_usd = in_usd + out_usd
    gross_usd = sign * (exit_mid - entry_mid) * qty
    extra = cost_usd - (abs(entry_fill - entry_mid) + abs(exit_fill - exit_mid)) * qty  # commission + impact
    net_usd = sign * (exit_fill - entry_fill) * qty - extra
    base = entry_mid * qty
    return {
        "entry_fill": entry_fill, "exit_fill": exit_fill, "gross_usd": gross_usd, "cost_usd": cost_usd,
        "net_usd": net_usd, "gross_return": gross_usd / base, "net_return": net_usd / base,
    }


def from_fill(fill: FillModel) -> CostModel:
    """CostModel with the fill model's spread, slippage and commission (source `platform_default`)."""
    from shpe_research.trade.types import CostModel as _CostModel

    return _CostModel(
        half_spread_bps=fill.half_spread_bps, slippage_bps=fill.slippage_bps,
        commission_per_share=fill.commission_per_share,
    )


def from_tca(path: str | Path) -> CostModel:
    """Load a calibrated CostModel from an R5 result table (`.csv` or `.parquet`).

    The table has exactly one row and a `run_id` column; any of `half_spread_bps`, `slippage_bps`,
    `commission_per_share`, `impact_coef`, `impact_exponent` present are used, absent ones keep the
    platform default. `source` becomes `calibrated:<run_id>`. R5 has not reported, so this layout is
    provisional; a per-liquidity-cell table must be reduced to one row by the caller.
    """
    from shpe_research.trade.types import CostModel as _CostModel

    p = Path(path)
    table = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if len(table) != 1:
        raise ValueError(f"TCA table must have exactly one row, got {len(table)}")
    row = table.iloc[0]
    if "run_id" not in table.columns or pd.isna(row["run_id"]) or not str(row["run_id"]):
        raise ValueError("TCA table needs a run_id to label the model as calibrated")
    vals = {f: float(row[f]) for f in _TCA_FIELDS if f in table.columns and not pd.isna(row[f])}
    return _CostModel(**vals, source=f"calibrated:{row['run_id']}")
