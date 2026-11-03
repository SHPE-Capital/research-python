"""Risk checks that mirror the trading platform's RiskEngine (T4).

Source: trading-platform `backend/src/core/risk/riskEngine.ts`, `types/risk.ts`,
`docs/architecture/risk-layer.md`. Checks run in the platform's order and reject
with its reason codes; the first failure wins. A limit set to ``None`` is off.

Timing: an order is checked at its ``ts`` (the decision time, when every input is
known). Portfolio checks run once per bar through `check_portfolio`.

Research deviations, all listed here so parity can account for them:

* ``KILL_SWITCH`` is never engaged by configuration; only a drawdown breach
  engages it (as in the platform). ``STALE_QUOTE`` never fires on historical bars.
* ``MAX_CONCURRENT_SAME_SIDE`` is research-only and not exportable.
* ``reset_session`` clears the drawdown halt on a new session; the platform
  keeps the kill switch until it is cleared by hand.
* The strategy-budget group is checked in the packet's order
  (``STRATEGY_BUDGET``, ``STRATEGY_ORDER_NOTIONAL``, ``MAX_OPEN_ORDERS``); the
  platform's ``checkStrategyBudget`` tests ``MAX_OPEN_ORDERS`` first. They differ
  only when several of the three fail on the same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from shpe_research.data.calendar import NY, get_calendar
from shpe_research.trade.types import RiskLimits

REASON_ORDER: tuple[str, ...] = (
    "KILL_SWITCH", "MARKET_CLOSED", "STALE_QUOTE", "ORDER_COOLDOWN", "MAX_POSITION_SIZE",
    "MAX_NOTIONAL_EXPOSURE", "SHORT_SELLING_DISALLOWED", "CASH_RESERVE", "INTRADAY_DRAWDOWN",
    "CONCENTRATION_LIMIT", "STRATEGY_BUDGET", "STRATEGY_ORDER_NOTIONAL", "MAX_OPEN_ORDERS",
    "MAX_CONCURRENT_SAME_SIDE",
)

#: `RiskLimits` field -> platform name (`RiskConfig` or `StrategyRiskBudget`).
PLATFORM_NAMES: dict[str, str] = {
    "max_position_size_usd": "maxPositionSizeUsd",
    "max_notional_exposure_usd": "maxNotionalExposureUsd",
    "order_cooldown_ms": "orderCooldownMs",
    "stale_quote_threshold_ms": "staleQuoteThresholdMs",
    "allow_short_selling": "allowShortSelling",
    "max_intraday_drawdown_pct": "maxIntradayDrawdownPct",
    "max_concentration_pct": "maxConcentrationPct",
    "cash_reserve_pct": "cashReservePct",
    "max_gross_exposure_pct": "maxGrossExposurePct",
    "max_net_exposure_pct": "maxNetExposurePct",
    "strategy_max_capital_pct": "StrategyRiskBudget.maxCapitalPct",
    "strategy_max_order_notional_pct": "StrategyRiskBudget.maxOrderNotionalPct",
    "strategy_max_open_orders": "StrategyRiskBudget.maxOpenOrders",
}
#: Fields with no platform name (T7 refuses or maps them).
RESEARCH_ONLY_FIELDS: tuple[str, ...] = ("max_concurrent_same_side", "initial_equity")

#: Platform DEFAULT_RISK_CONFIG buffers for worst-case market-order prices (basis points).
GAP_BUFFER_BPS = 20.0
SPREAD_BUFFER_BPS = 5.0


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of one check: ``accepted`` and, on rejection, the platform ``reason`` code."""

    accepted: bool
    reason: str | None = None
    detail: str = ""


@dataclass
class Position:
    """An open position: signed ``qty`` (negative = short) valued at ``current_price`` (USD per share)."""

    symbol: str
    qty: float
    current_price: float
    strategy_id: str | None = None


@dataclass
class RiskOrder:
    """An order to check. ``ts`` is the decision time (UTC); ``reference_price`` is known at ``ts``.

    ``limit_price`` wins over ``reference_price``; with neither, the existing position's price is used,
    and with none of them the price-based checks are skipped, as in the platform.
    """

    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    ts: pd.Timestamp
    strategy_id: str = ""
    reference_price: float | None = None
    limit_price: float | None = None
    reference_ts: pd.Timestamp | None = None


@dataclass
class PortfolioState:
    """Portfolio as the risk engine sees it. Mutated by `check_order` / `check_portfolio`.

    ``session_start_equity`` is set on first use; ``kill_switch_active`` is set by a drawdown breach.
    ``last_order_ts`` keys are ``(strategy_id, symbol)``.
    """

    equity: float
    cash: float
    positions: list[Position] = field(default_factory=list)
    initial_capital: float | None = None
    session_start_equity: float | None = None
    kill_switch_active: bool = False
    last_order_ts: dict[tuple[str, str], pd.Timestamp] = field(default_factory=dict)
    strategy_reserved: dict[str, float] = field(default_factory=dict)
    strategy_open_orders: dict[str, int] = field(default_factory=dict)

    def position(self, symbol: str) -> Position | None:
        """The open position in ``symbol``, or None."""
        return next((p for p in self.positions if p.symbol == symbol), None)


def reset_session(portfolio: PortfolioState) -> None:
    """Start a new session: forget the drawdown baseline and clear the drawdown halt (research only)."""
    portfolio.session_start_equity = None
    portfolio.kill_switch_active = False


def _reject(code: str, detail: str = "") -> RiskDecision:
    return RiskDecision(False, code, detail)


def _init_session_equity(portfolio: PortfolioState) -> None:
    if portfolio.session_start_equity is None:
        eq = portfolio.equity if portfolio.equity > 0 else portfolio.initial_capital
        portfolio.session_start_equity = eq


def _reference_price(order: RiskOrder, portfolio: PortfolioState) -> float | None:
    if order.limit_price is not None and order.limit_price > 0:
        return order.limit_price
    if order.reference_price is not None and order.reference_price > 0:
        return order.reference_price
    pos = portfolio.position(order.symbol)
    if pos is not None and pos.current_price > 0:
        return pos.current_price
    return None


def _market_open(ts: pd.Timestamp) -> bool:
    """True when ``ts`` is inside the regular session (early closes from the calendar)."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    day = ts.tz_convert(NY).tz_localize(None).normalize()
    cal = get_calendar()
    if not cal.is_session(day):
        return False
    open_, close = cal.session_bounds(day)
    return bool(open_ <= ts < close)


def _delta(order: RiskOrder) -> float:
    return order.qty if order.side == "buy" else -order.qty


def _worst_case_price(order: RiskOrder, raw: float) -> float:
    bps = (GAP_BUFFER_BPS + SPREAD_BUFFER_BPS) / 10_000
    return raw * (1 + bps) if order.side == "buy" else raw * (1 - bps)


def check_order(
    order: RiskOrder, portfolio: PortfolioState, limits: RiskLimits, *, record: bool = True
) -> RiskDecision:
    """Accept or reject one order; the first failing check in `REASON_ORDER` wins.

    Inputs: the order at its decision time, the portfolio as known then, and the limits.
    Side effects, as in the platform: the drawdown baseline is initialised, a drawdown breach engages the
    kill switch, and an accepted order starts its (strategy, symbol) cooldown unless ``record`` is False.
    A non-positive quantity is rejected as ``INVALID_QTY`` (the platform's pre-check).
    """
    if order.qty <= 0:
        return _reject("INVALID_QTY", f"Order qty must be > 0 (got {order.qty})")
    _init_session_equity(portfolio)
    raw = _reference_price(order, portfolio)
    pos = portfolio.position(order.symbol)
    held = pos.qty if pos is not None else 0.0
    new_qty = held + _delta(order)

    def kill_switch():
        return _reject("KILL_SWITCH", "Global kill switch is active") if portfolio.kill_switch_active else None

    def market_closed():
        return None if _market_open(order.ts) else _reject("MARKET_CLOSED", "Outside the regular session")

    def stale_quote():
        return None  # historical bars: quote age is not meaningful

    def cooldown():
        cd = limits.order_cooldown_ms
        last = portfolio.last_order_ts.get((order.strategy_id, order.symbol))
        if not cd or cd <= 0 or last is None:
            return None
        elapsed_ms = (pd.Timestamp(order.ts) - last) / pd.Timedelta(milliseconds=1)
        if elapsed_ms < cd:
            return _reject("ORDER_COOLDOWN", f"{cd - elapsed_ms:.0f}ms remaining")
        return None

    def max_position():
        if limits.max_position_size_usd is None or raw is None:
            return None
        if abs(new_qty * raw) > limits.max_position_size_usd:
            return _reject("MAX_POSITION_SIZE", f"new notional {abs(new_qty * raw):.2f}")
        return None

    def max_notional():
        if limits.max_notional_exposure_usd is None or raw is None:
            return None
        others = sum(abs(p.qty * p.current_price) for p in portfolio.positions if p.symbol != order.symbol)
        total = others + abs(new_qty * raw)
        if total > limits.max_notional_exposure_usd:
            return _reject("MAX_NOTIONAL_EXPOSURE", f"new total {total:.2f}")
        return None

    def short_selling():
        if limits.allow_short_selling or order.side != "sell":
            return None
        if order.qty > held:
            return _reject("SHORT_SELLING_DISALLOWED", f"sell {order.qty} > held {held}")
        return None

    def cash_reserve():
        if limits.cash_reserve_pct is None:
            return None
        base = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        floor = base * limits.cash_reserve_pct
        if order.side == "sell":
            if order.qty > held and portfolio.cash <= floor:
                return _reject("CASH_RESERVE", "short open with cash at or below the reserve floor")
            return None
        if raw is None:
            return None
        cost = order.qty * _worst_case_price(order, raw)
        available = min(portfolio.cash, base) - floor
        if cost > available:
            return _reject("CASH_RESERVE", f"cost {cost:.2f} > available {available:.2f}")
        return None

    def drawdown():
        lim = limits.max_intraday_drawdown_pct
        start = portfolio.session_start_equity
        if lim is None or start is None or start <= 0:
            return None
        dd = (start - portfolio.equity) / start
        if dd >= lim:
            portfolio.kill_switch_active = True
            return _reject("INTRADAY_DRAWDOWN", f"drawdown {dd:.4f} >= {lim}")
        return None

    def concentration():
        lim = limits.max_concentration_pct
        if lim is None or portfolio.equity <= 0 or raw is None:
            return None
        pct = abs(new_qty * raw) / portfolio.equity
        if pct > lim:
            return _reject("CONCENTRATION_LIMIT", f"{pct:.4f} > {lim}")
        return None

    def worst_notional() -> float | None:
        return None if raw is None else order.qty * _worst_case_price(order, raw)

    def strategy_budget():
        cap = limits.strategy_max_capital_pct
        wn = worst_notional()
        if cap is None or wn is None:
            return None
        projected = portfolio.strategy_reserved.get(order.strategy_id, 0.0) + wn
        if projected > cap * portfolio.equity:
            return _reject("STRATEGY_BUDGET", f"projected {projected:.2f} > {cap * portfolio.equity:.2f}")
        return None

    def strategy_order_notional():
        lim = limits.strategy_max_order_notional_pct
        wn = worst_notional()
        if lim is None or wn is None:
            return None
        if wn > lim * portfolio.equity:
            return _reject("STRATEGY_ORDER_NOTIONAL", f"{wn:.2f} > {lim * portfolio.equity:.2f}")
        return None

    def max_open_orders():
        lim = limits.strategy_max_open_orders
        n = portfolio.strategy_open_orders.get(order.strategy_id, 0)
        if lim is not None and n >= lim:
            return _reject("MAX_OPEN_ORDERS", f"{n} open, limit {lim}")
        return None

    def same_side():
        cap = limits.max_concurrent_same_side
        if cap is None:
            return None
        direction = 1 if order.side == "buy" else -1
        if held != 0:
            return None  # adds to, reduces or closes an existing position: no new position opens
        n = sum(
            1 for p in portfolio.positions
            if p.symbol != order.symbol and p.qty * direction > 0 and p.strategy_id in (None, order.strategy_id)
        )
        if n >= cap:
            return _reject("MAX_CONCURRENT_SAME_SIDE", f"{n} open same-side positions, cap {cap}")
        return None

    checks = (
        kill_switch, market_closed, stale_quote, cooldown, max_position, max_notional, short_selling,
        cash_reserve, drawdown, concentration, strategy_budget, strategy_order_notional, max_open_orders,
        same_side,
    )
    for run in checks:
        failure = run()
        if failure is not None:
            return failure
    if record:
        portfolio.last_order_ts[(order.strategy_id, order.symbol)] = pd.Timestamp(order.ts)
    return RiskDecision(True)


def check_portfolio(portfolio: PortfolioState, limits: RiskLimits, ts: pd.Timestamp) -> list[RiskDecision]:
    """Portfolio-level checks, run once per bar at ``ts``; returns one rejected decision per violation.

    Codes: ``MAX_GROSS_EXPOSURE``, ``MAX_NET_EXPOSURE``, ``INTRADAY_DRAWDOWN``. A drawdown breach engages
    the kill switch, so every later `check_order` returns ``KILL_SWITCH`` until `reset_session`.
    Returns an empty list when equity is not positive, as in the platform. ``ts`` is for the caller's log.
    """
    if portfolio.session_start_equity is None and portfolio.equity > 0:
        portfolio.session_start_equity = portfolio.equity
    if portfolio.equity <= 0:
        return []
    out: list[RiskDecision] = []
    if limits.max_gross_exposure_pct is not None:
        pct = sum(abs(p.qty * p.current_price) for p in portfolio.positions) / portfolio.equity
        if pct > limits.max_gross_exposure_pct:
            out.append(_reject("MAX_GROSS_EXPOSURE", f"gross {pct:.4f} at {ts}"))
    if limits.max_net_exposure_pct is not None:
        pct = abs(sum(p.qty * p.current_price for p in portfolio.positions)) / portfolio.equity
        if pct > limits.max_net_exposure_pct:
            out.append(_reject("MAX_NET_EXPOSURE", f"net {pct:.4f} at {ts}"))
    start = portfolio.session_start_equity
    if limits.max_intraday_drawdown_pct is not None and start:
        dd = (start - portfolio.equity) / start
        if dd >= limits.max_intraday_drawdown_pct:
            portfolio.kill_switch_active = True
            out.append(_reject("INTRADAY_DRAWDOWN", f"drawdown {dd:.4f} at {ts}"))
    return out
