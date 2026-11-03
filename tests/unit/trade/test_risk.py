"""T4 acceptance tests: platform-ordered risk checks with platform reason codes."""

from __future__ import annotations

import dataclasses

import pandas as pd

from shpe_research.trade import risk
from shpe_research.trade.risk import (
    PLATFORM_NAMES,
    REASON_ORDER,
    PortfolioState,
    Position,
    RiskOrder,
    check_order,
    check_portfolio,
)
from shpe_research.trade.types import RiskLimits

# Wednesday 2024-01-03, 10:00 New York = 15:00 UTC (regular session)
OPEN_TS = pd.Timestamp("2024-01-03 15:00", tz="UTC")


def order(symbol="AAA", side="buy", qty=10, ts=OPEN_TS, price=100.0, **kw) -> RiskOrder:
    return RiskOrder(symbol=symbol, side=side, qty=qty, ts=ts, reference_price=price, strategy_id="s", **kw)


def book(positions=(), equity=100_000.0, cash=100_000.0, **kw) -> PortfolioState:
    return PortfolioState(equity=equity, cash=cash, positions=list(positions), **kw)


def test_reason_order():
    # Fails cooldown, position size and short selling; cooldown comes first in the list.
    limits = RiskLimits(order_cooldown_ms=60_000, max_position_size_usd=500.0, allow_short_selling=False)
    pf = book()
    first = check_order(order(side="buy", qty=1), pf, limits)
    assert first.accepted
    second = check_order(order(side="sell", qty=10), pf, limits)
    assert not second.accepted and second.reason == "ORDER_COOLDOWN"
    # Cooldown is per strategy and symbol: another symbol fails position size next
    third = check_order(order(symbol="BBB", side="sell", qty=10), pf, limits)
    assert third.reason == "MAX_POSITION_SIZE"
    # With position size relaxed, short selling is next
    fourth = check_order(order(symbol="CCC", side="sell", qty=10), book(), dataclasses.replace(
        limits, max_position_size_usd=None))
    assert fourth.reason == "SHORT_SELLING_DISALLOWED"
    assert REASON_ORDER.index("ORDER_COOLDOWN") < REASON_ORDER.index("MAX_POSITION_SIZE") < REASON_ORDER.index(
        "SHORT_SELLING_DISALLOWED")


def test_reason_order_matches_packet():
    assert REASON_ORDER == (
        "KILL_SWITCH", "MARKET_CLOSED", "STALE_QUOTE", "ORDER_COOLDOWN", "MAX_POSITION_SIZE",
        "MAX_NOTIONAL_EXPOSURE", "SHORT_SELLING_DISALLOWED", "CASH_RESERVE", "INTRADAY_DRAWDOWN",
        "CONCENTRATION_LIMIT", "STRATEGY_BUDGET", "STRATEGY_ORDER_NOTIONAL", "MAX_OPEN_ORDERS",
        "MAX_CONCURRENT_SAME_SIDE",
    )


def test_kill_switch_and_stale_pass():
    # Neither engaged nor stale on historical bars, even with an ancient quote and a tiny threshold
    limits = RiskLimits(stale_quote_threshold_ms=1)
    old = OPEN_TS - pd.Timedelta(days=30)
    d = check_order(order(reference_ts=old), book(), limits)
    assert d.accepted and d.reason is None


def test_market_closed_uses_session_and_early_close():
    pre = pd.Timestamp("2024-01-03 13:00", tz="UTC")  # 08:00 NY
    assert check_order(order(ts=pre), book(), RiskLimits()).reason == "MARKET_CLOSED"
    # 2023-11-24 is an early close (13:00 NY); 14:00 NY = 19:00 UTC is closed
    late = pd.Timestamp("2023-11-24 19:00", tz="UTC")
    assert check_order(order(ts=late), book(), RiskLimits()).reason == "MARKET_CLOSED"
    # The 15:59 NY bar is closed on an ordinary day only after 16:00
    assert check_order(order(ts=pd.Timestamp("2024-01-03 20:59", tz="UTC")), book(), RiskLimits()).accepted
    assert check_order(order(ts=pd.Timestamp("2024-01-03 21:00", tz="UTC")), book(), RiskLimits()).reason == (
        "MARKET_CLOSED")


def test_short_disallowed():
    limits = RiskLimits(allow_short_selling=False)
    assert check_order(order(side="sell", qty=5), book(), limits).reason == "SHORT_SELLING_DISALLOWED"
    held = book([Position("AAA", qty=10, current_price=100.0)])
    assert check_order(order(side="sell", qty=10), held, limits).accepted  # closing a long is fine


def test_cooldown():
    limits = RiskLimits(order_cooldown_ms=60_000)
    pf = book()
    assert check_order(order(), pf, limits).accepted
    soon = order(ts=OPEN_TS + pd.Timedelta(seconds=30))
    assert check_order(soon, pf, limits).reason == "ORDER_COOLDOWN"
    later = order(ts=OPEN_TS + pd.Timedelta(seconds=60))
    assert check_order(later, pf, limits).accepted


def test_rejected_order_does_not_start_cooldown():
    limits = RiskLimits(order_cooldown_ms=60_000, max_position_size_usd=100.0)
    pf = book()
    assert check_order(order(qty=5), pf, limits).reason == "MAX_POSITION_SIZE"
    assert check_order(order(qty=1), pf, limits).accepted


def test_position_notional_concentration_cash_and_budget_checks():
    pf = book([Position("BBB", qty=100, current_price=100.0)])
    assert check_order(order(qty=10), pf, RiskLimits(max_notional_exposure_usd=10_500.0)).reason == (
        "MAX_NOTIONAL_EXPOSURE")
    assert check_order(order(qty=10), book(cash=500.0), RiskLimits(cash_reserve_pct=0.0)).reason == "CASH_RESERVE"
    assert check_order(order(qty=400), book(), RiskLimits(max_concentration_pct=0.3)).reason == (
        "CONCENTRATION_LIMIT")
    limits = RiskLimits(strategy_max_capital_pct=0.1)
    assert check_order(order(qty=200), book(), limits).reason == "STRATEGY_BUDGET"
    assert check_order(order(qty=100), book(strategy_reserved={"s": 5_000.0}), limits).reason == "STRATEGY_BUDGET"
    limits = RiskLimits(strategy_max_capital_pct=1.0, strategy_max_order_notional_pct=0.01)
    assert check_order(order(qty=11), book(), limits).reason == "STRATEGY_ORDER_NOTIONAL"
    limits = RiskLimits(strategy_max_capital_pct=1.0, strategy_max_open_orders=2)
    assert check_order(order(), book(strategy_open_orders={"s": 2}), limits).reason == "MAX_OPEN_ORDERS"


def test_max_concurrent_same_side():
    limits = RiskLimits(max_concurrent_same_side=2)
    pf = book([Position("BBB", 10, 100.0), Position("CCC", 10, 100.0), Position("DDD", -10, 100.0)])
    assert check_order(order(symbol="EEE", side="buy"), pf, limits).reason == "MAX_CONCURRENT_SAME_SIDE"
    assert check_order(order(symbol="EEE", side="sell"), pf, limits).accepted  # only one short open
    assert check_order(order(symbol="BBB", side="buy"), pf, limits).accepted  # adds to an existing long
    assert check_order(order(symbol="BBB", side="sell", qty=10), pf, limits).accepted  # closes


def test_portfolio_check_per_bar():
    limits = RiskLimits(max_intraday_drawdown_pct=0.05)
    pf = book(equity=100_000.0)
    assert check_portfolio(pf, limits, OPEN_TS) == []
    assert pf.session_start_equity == 100_000.0
    assert check_order(order(), pf, limits).accepted
    # Bar t: equity falls 6%
    pf.equity = 94_000.0
    violations = check_portfolio(pf, limits, OPEN_TS + pd.Timedelta(minutes=1))
    assert [v.reason for v in violations] == ["INTRADAY_DRAWDOWN"]
    # Orders from bar t on are blocked, even if equity recovers within the session
    pf.equity = 99_000.0
    blocked = check_order(order(ts=OPEN_TS + pd.Timedelta(minutes=1)), pf, limits)
    assert not blocked.accepted and blocked.reason == "KILL_SWITCH"
    risk.reset_session(pf)
    assert check_order(order(ts=OPEN_TS + pd.Timedelta(days=1)), pf, limits).accepted


def test_portfolio_gross_and_net():
    pf = book([Position("A", 600, 100.0), Position("B", -400, 100.0)], equity=100_000.0)
    out = check_portfolio(pf, RiskLimits(max_gross_exposure_pct=0.9, max_net_exposure_pct=0.1), OPEN_TS)
    assert [v.reason for v in out] == ["MAX_GROSS_EXPOSURE", "MAX_NET_EXPOSURE"]
    assert check_portfolio(pf, RiskLimits(max_gross_exposure_pct=1.0, max_net_exposure_pct=0.2), OPEN_TS) == []


def test_missing_reference_price_skips_price_checks():
    o = RiskOrder(symbol="AAA", side="buy", qty=10, ts=OPEN_TS)
    assert check_order(o, book(), RiskLimits(max_position_size_usd=1.0)).accepted


def test_names_match_platform():
    mapping = PLATFORM_NAMES
    fields = {f.name for f in dataclasses.fields(RiskLimits)}
    research_only = {"max_concurrent_same_side", "initial_equity"}
    assert set(mapping) == fields - research_only
    assert mapping["max_position_size_usd"] == "maxPositionSizeUsd"
    assert mapping["strategy_max_capital_pct"] == "StrategyRiskBudget.maxCapitalPct"
    assert set(risk.RESEARCH_ONLY_FIELDS) == research_only
