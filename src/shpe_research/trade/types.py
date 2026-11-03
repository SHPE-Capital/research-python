"""Trade-layer types (C1a stubs with final names and fields; T1, T2, T4 and T5 fill the behaviour).

Defaults of FillModel and CostModel equal the platform defaults: 2 bps
half-spread, 5 bps slippage, $0.005 per share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

TradeLedger = pd.DataFrame

TRADE_LEDGER_COLUMNS: tuple[str, ...] = (
    "symbol", "strategy_id", "side", "decision_ts", "entry_ts", "entry_price", "qty", "exit_ts",
    "exit_price", "exit_reason", "gross_return", "cost", "net_return", "label_end_ts",
)
OPTIONAL_LEDGER_COLUMNS: tuple[str, ...] = ("mfe", "mae", "partial_fill", "early_close")


@dataclass(frozen=True)
class FillModel:
    """Platform bar fill model (BACKTEST_ENGINE_VERSION 2)."""

    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    volume_participation_cap: float = 0.1
    allow_partial_fills: bool = True
    price_source: str = "next_open"
    engine_version: int = 2


@dataclass(frozen=True)
class ExitPolicy:
    """How a trade is closed."""

    stop_rule: Literal["none", "pct", "atr", "price"] = "none"
    stop_pct: float | None = None
    stop_atr_mult: float | None = None
    atr_window: int = 14
    target_rule: Literal["none", "pct", "price", "r_multiple"] = "none"
    target_pct: float | None = None
    target_r: float | None = None
    trail_rule: Literal["none", "chandelier", "breakeven_1r"] = "none"
    trail_atr_mult: float = 3.0
    max_hold_bars: int | None = None
    session_end: bool = True
    flat_by: str | None = None
    use_signal_exits: bool = True
    entry_order: Literal["market", "stop"] = "market"


@dataclass(frozen=True, init=False)
class SizingRule:
    """Quantity rule (T2): ``SizingRule(rule, **params)``.

    Rules: native, fixed_qty, fixed_notional, risk_per_trade, inverse_vol, prob_to_size,
    kelly_fractional (params per rule in ``trade/sizing.py``). Validated at construction.
    ``size(candidates, bars, equity)`` returns whole-share quantities (see ``trade.sizing``).
    A ``params={...}`` dict keyword is also accepted and merged with the keyword params.
    """

    rule: str
    params: dict[str, Any]

    def __init__(self, rule: str = "native", params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        from shpe_research.trade.sizing import validate_rule

        merged = {**(params or {}), **kwargs}
        validate_rule(rule, merged)
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "params", merged)

    def __hash__(self) -> int:
        return hash((self.rule, tuple(sorted((k, str(v)) for k, v in self.params.items()))))

    @property
    def exportable(self) -> bool:
        """True if the platform can run this rule today (fixed_qty, fixed_notional, native)."""
        from shpe_research.trade.sizing import EXPORTABLE_RULES

        return self.rule in EXPORTABLE_RULES

    def size(self, candidates: pd.DataFrame, bars: pd.DataFrame, equity: float) -> pd.Series:
        """Whole-share quantity per candidate row (int64, index of `candidates`); 0 means no trade."""
        from shpe_research.trade.sizing import size_candidates

        return size_candidates(self, candidates, bars, equity)


@dataclass(frozen=True)
class RiskLimits:
    """Risk limits (T4): snake_case of the platform's RiskConfig and StrategyRiskBudget."""

    max_position_size_usd: float | None = None
    max_notional_exposure_usd: float | None = None
    order_cooldown_ms: int | None = None
    stale_quote_threshold_ms: int | None = None
    allow_short_selling: bool = True
    max_intraday_drawdown_pct: float | None = None
    max_concentration_pct: float | None = None
    cash_reserve_pct: float | None = None
    max_gross_exposure_pct: float | None = None
    max_net_exposure_pct: float | None = None
    strategy_max_capital_pct: float | None = None
    strategy_max_order_notional_pct: float | None = None
    strategy_max_open_orders: int | None = None
    max_concurrent_same_side: int | None = None
    initial_equity: float | None = None


@dataclass(frozen=True)
class CostModel:
    """Transaction costs (T5)."""

    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    impact_coef: float = 0.0
    impact_exponent: float = 0.5
    source: str = "platform_default"
