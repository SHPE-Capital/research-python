"""C4 chronological multi-stock funded-account engine (pure-Python correctness oracle).

The hot loop runs over plain numpy arrays and scalars only (fixed-size position and order slots,
scratch buffers, integer reason codes) so a Numba port can follow line by line. Python objects are
built only after the loop, from the ledger and rejection records.

Time model. ``MarketPanel`` has one column per minute; bar ``t`` is labelled by its start. A session's
minute offset ``m`` is the column's offset from the session's first column, so a short (early-close)
session simply has fewer columns. The session exit is at offset ``length - exit_buffer`` (close - 5).

Per completed bar ``t`` (spec order):
  1. pending entries from bar t-1 meet bar t's open (C1 proxy) and fill or cancel;
  2. every open position is stepped on bar t (C2 order: halt exit at the open, gap through the stop at
     the open, session or max-hold exit at the open, low crossing the stop, then the close updates the
     stop for the next bar); exits pay the per-side cost;
  3. mark equity; a fall of ``daily_halt_pct`` from start-of-day equity halts the day: no new entries,
     remaining positions are liquidated at the next bar's open;
  4. candidates are the eligible signals at bar t for stocks not yet entered today (and not pending);
  5. ranked by descending 20-session median daily dollar volume, stable asset id tie-break;
  6. allocated within ``max_positions`` (open plus pending), sized by C3 with cash reserved at the
     limit price, and submitted; they meet the fill proxy on bar t+1.

Assumptions: history for the liquidity and ranking statistics comes from sessions earlier in the panel
(``warmup_sessions`` sessions are history only and produce no daily return); NaN prices mark missing
bars; an order is only submitted when its fill bar is before the session exit; a position that cannot
exit (missing bars) is closed at its last valid close at session end and flagged unresolved; marked
equity does not deduct the estimated exit cost.
"""

from __future__ import annotations

import numpy as np

from shpe_research.trade import fill_proxy, funded_sizing
from shpe_research.trade.account_types import (
    AccountConfig,
    AccountResult,
    LedgerEntry,
    MarketPanel,
    Rejection,
    SignalMask,
)
from shpe_research.trade.exits import initial_stop, step_stop

EXIT_REASONS = ("stop", "max_hold", "session_end", "halt", "unresolved")
X_STOP, X_MAX_HOLD, X_SESSION, X_HALT, X_UNRESOLVED = 0, 1, 2, 3, 4
REJECT_REASONS = (
    "max_positions", "entry_cancelled_above_limit", "no_next_bar", "bad_data",
    *(f"size_zero:{n}" for n in funded_sizing.CAP_NAMES),
)
R_MAX_POS, R_CANCEL, R_NO_BAR, R_BAD, R_SIZE0 = 0, 1, 2, 3, 4


def rank_dollar_volume_20d(
    close: np.ndarray, volume: np.ndarray, starts: np.ndarray, id_rank: np.ndarray, lookback: int
) -> np.ndarray:
    """Priority position per ``(session, symbol)``, 0 = first: descending median daily dollar volume
    over the previous ``lookback`` sessions, ties broken by ascending asset id."""
    n_sym = close.shape[0]
    dv = np.add.reduceat(np.nan_to_num(close * volume), starts, axis=1)  # (n_sym, S)
    n_sess = len(starts)
    rk = np.empty((n_sess, n_sym), dtype=np.int32)
    for s in range(n_sess):
        score = np.median(dv[:, max(0, s - lookback):s], axis=1) if s > 0 else np.zeros(n_sym)
        order = np.lexsort((id_rank, -score))
        rk[s, order] = np.arange(n_sym, dtype=np.int32)
    return rk


RANKINGS = {"dollar_volume_20d": rank_dollar_volume_20d}


def simulate_account(
    panel: MarketPanel, mask: SignalMask, cfg: AccountConfig, *,
    warmup_sessions: int = 0, entry_score: np.ndarray | None = None,
) -> AccountResult:
    """Run the funded account over the panel; one daily return per session after the warm-up."""
    n_sym, n_min = panel.n_symbols, panel.n_minutes
    if mask.signal.shape != (n_sym, n_min):
        raise ValueError(f"mask shape {mask.signal.shape} != panel shape {(n_sym, n_min)}")
    if cfg.fill.kind != "limit_markup":
        raise ValueError(f"unsupported fill model {cfg.fill.kind!r}")
    if cfg.ranking not in RANKINGS:
        raise ValueError(f"unknown ranking {cfg.ranking!r}; choose from {sorted(RANKINGS)}")
    if cfg.entry_score_threshold is not None and (entry_score is None or entry_score.shape != (n_sym, n_min)):
        raise ValueError("entry_score_threshold needs an entry_score array of panel shape")
    if cfg.exit.max_hold_minutes <= 0 or cfg.max_positions <= 0:
        raise ValueError("max_hold_minutes and max_positions must be positive")

    open_, high_, low_, close_, vol_ = (
        np.asarray(a, dtype=np.float64) for a in (panel.open, panel.high, panel.low, panel.close, panel.volume)
    )
    atr_ = np.asarray(mask.atr, dtype=np.float64)
    sigT = np.ascontiguousarray((mask.signal & panel.eligible).T)  # (minute, symbol)
    sig_count = sigT.sum(axis=1)

    si = np.asarray(panel.session_index)
    if n_min:
        starts = np.flatnonzero(np.concatenate(([True], si[1:] != si[:-1])))
    else:
        starts = np.zeros(0, dtype=np.int64)
    n_sess = len(starts)
    lens = np.diff(np.concatenate((starts, [n_min]))).astype(np.int64)
    if not 0 <= warmup_sessions <= n_sess:
        raise ValueError("warmup_sessions out of range")

    ids = panel.asset_ids
    id_rank = np.empty(n_sym, dtype=np.int64)
    id_rank[np.argsort(np.asarray(ids, dtype=object), kind="stable")] = np.arange(n_sym)
    rk = RANKINGS[cfg.ranking](close_, vol_, starts, id_rank, cfg.liquidity_lookback)

    max_pos = cfg.max_positions
    cb = fill_proxy.side_cost_rate(cfg.cost_bps)
    markup = cfg.fill.limit_markup
    ex = cfg.exit
    mult = 1.0 if cfg.size_multiplier is None else float(cfg.size_multiplier)
    thr = cfg.entry_score_threshold
    ref = cfg.search_replay_reference_equity
    lookback, min_obs = cfg.liquidity_lookback, cfg.liquidity_min_obs

    # fixed-size state (positions and pending orders)
    p_sym = np.full(max_pos, -1, dtype=np.int64)
    p_qty = np.zeros(max_pos)
    p_entry = np.zeros(max_pos)
    p_atr = np.zeros(max_pos)
    p_stop = np.zeros(max_pos)
    p_hc = np.zeros(max_pos)
    p_trail = np.zeros(max_pos, dtype=np.bool_)
    p_trig = np.zeros(max_pos)
    p_thold = np.zeros(max_pos, dtype=np.int64)
    p_last = np.zeros(max_pos)
    p_cost = np.zeros(max_pos)
    p_ecol = np.zeros(max_pos, dtype=np.int64)
    o_sym = np.zeros(max_pos, dtype=np.int64)
    o_qty = np.zeros(max_pos)
    o_limit = np.zeros(max_pos)
    o_res = np.zeros(max_pos)
    o_atr = np.zeros(max_pos)
    entered = np.full(n_sym, -1, dtype=np.int64)  # session stamp of the last filled entry
    pend = np.zeros(n_sym, dtype=np.bool_)
    buf = np.empty(max(lookback, 1), dtype=np.float64)

    led: list[tuple[int, int, int, int, float, float, int, float]] = []
    rej: list[tuple[int, int, int]] = []
    ambiguous = 0
    unresolved = 0
    daily = np.zeros(n_sess - warmup_sessions)
    sess_trades = np.zeros(n_sess - warmup_sessions, dtype=np.int64)

    cash = float(cfg.starting_cash)
    for s in range(warmup_sessions, n_sess):
        c0, length = int(starts[s]), int(lens[s])
        exit_m = length - ex.session_exit_minutes_before_close
        if ref is not None:
            cash = float(ref)
        cash_start = cash
        halt_level = cash_start * (1.0 - cfg.daily_halt_pct)
        halted = False
        n_open = 0
        n_pend = 0
        reserved = 0.0
        for m in range(length):
            t = c0 + m
            # 1. pending entries meet this bar's open
            for k in range(n_pend):
                sym = int(o_sym[k])
                pend[sym] = False
                code, px = fill_proxy.entry_fill(o_limit[k], open_[sym, t])
                reserved -= o_res[k]
                if code != fill_proxy.FILLED:
                    rej.append((sym, t, R_CANCEL if code == fill_proxy.CANCEL_ABOVE_LIMIT else R_NO_BAR))
                    continue
                j = 0
                while p_sym[j] >= 0:
                    j += 1
                qty = o_qty[k]
                fee = qty * px * cb
                cash -= qty * px + fee
                a = o_atr[k]
                p_sym[j], p_qty[j], p_entry[j], p_atr[j] = sym, qty, px, a
                p_stop[j] = initial_stop(px, a, ex.initial_stop_atr)
                p_hc[j], p_trail[j] = px, False
                p_trig[j] = px + ex.trail_activation_atr * a
                p_thold[j] = m + ex.max_hold_minutes
                p_last[j], p_cost[j], p_ecol[j] = px, fee, t
                entered[sym] = s
                n_open += 1
            n_pend = 0
            pend_notional = 0.0

            # 2. exits
            pos_value = 0.0
            if n_open:
                for j in range(max_pos):
                    sym = int(p_sym[j])
                    if sym < 0:
                        continue
                    po = open_[sym, t]
                    reason = -1
                    xpx = 0.0
                    if po == po:
                        if halted:
                            reason, xpx = X_HALT, po
                        elif po <= p_stop[j]:
                            reason, xpx = X_STOP, po
                        elif m >= exit_m:
                            reason, xpx = X_SESSION, po
                        elif m >= p_thold[j]:
                            reason, xpx = X_MAX_HOLD, po
                        elif low_[sym, t] <= p_stop[j]:
                            reason, xpx = X_STOP, p_stop[j]
                            if high_[sym, t] >= p_trig[j]:
                                ambiguous += 1
                        else:
                            pc = close_[sym, t]
                            if pc == pc:
                                p_stop[j], p_hc[j], p_trail[j] = step_stop(
                                    p_stop[j], p_hc[j], p_trail[j], p_entry[j], p_atr[j], pc,
                                    ex.trail_activation_atr, ex.trail_distance_atr,
                                )
                                p_last[j] = pc
                    if reason >= 0:
                        qty = p_qty[j]
                        cash += qty * xpx * (1.0 - cb)
                        led.append((sym, int(p_ecol[j]), t, int(qty), p_entry[j], xpx, reason,
                                    qty * (xpx - p_entry[j]) - p_cost[j] - qty * xpx * cb))
                        sess_trades[s - warmup_sessions] += 1
                        p_sym[j] = -1
                        n_open -= 1
                    else:
                        pos_value += p_qty[j] * p_last[j]

            # 3. mark and daily halt
            equity = cash + pos_value
            if not halted and equity <= halt_level:
                halted = True

            # 4-6. candidates, ranking, allocation, submission
            if halted or m + 1 >= exit_m or sig_count[t] == 0:
                continue
            idx = np.flatnonzero(sigT[t])
            idx = idx[(entered[idx] != s) & ~pend[idx]]
            if thr is not None and idx.size:
                idx = idx[entry_score[idx, t] >= thr]  # type: ignore[index]
            if idx.size == 0:
                continue
            if idx.size > 1:
                idx = idx[np.argsort(rk[s, idx], kind="stable")]
            for ci in range(idx.size):
                sym = int(idx[ci])
                if n_open + n_pend >= max_pos:
                    rej.append((sym, t, R_MAX_POS))
                    continue
                price = close_[sym, t]
                a = atr_[sym, t]
                if not (0.0 < price < np.inf and 0.0 < a < np.inf):
                    rej.append((sym, t, R_BAD))
                    continue
                limit = fill_proxy.entry_limit(price, markup)
                nv = 0
                for q in range(1, lookback + 1):
                    ps = s - q
                    if ps < 0:
                        break
                    if m < lens[ps]:
                        v = vol_[sym, starts[ps] + m]
                        if v == v:
                            buf[nv] = v
                            nv += 1
                liq = funded_sizing.liquidity_cap_shares(buf, nv, cfg.liquidity_cap_pct, min_obs)
                unit = limit * (1.0 + cb)
                qty_i, cap = funded_sizing.size_order(
                    equity, price, ex.initial_stop_atr * a, cash - reserved, unit,
                    pos_value + pend_notional, liq, cfg.per_stock_pct, cfg.risk_pct,
                    cfg.total_exposure_cap, mult,
                )
                if qty_i <= 0:
                    rej.append((sym, t, R_SIZE0 + cap))
                    continue
                o_sym[n_pend], o_qty[n_pend], o_limit[n_pend], o_atr[n_pend] = sym, qty_i, limit, a
                o_res[n_pend] = qty_i * unit
                reserved += qty_i * unit
                pend_notional += qty_i * limit
                pend[sym] = True
                n_pend += 1

        # session end: anything still open could not exit (missing bars); close at last valid mark
        for j in range(max_pos):
            sym = int(p_sym[j])
            if sym >= 0:
                qty = p_qty[j]
                xpx = p_last[j]
                cash += qty * xpx * (1.0 - cb)
                led.append((sym, int(p_ecol[j]), c0 + length - 1, int(qty), p_entry[j], xpx, X_UNRESOLVED,
                            qty * (xpx - p_entry[j]) - p_cost[j] - qty * xpx * cb))
                sess_trades[s - warmup_sessions] += 1
                unresolved += 1
                p_sym[j] = -1
        for k in range(n_pend):
            pend[int(o_sym[k])] = False
        daily[s - warmup_sessions] = (cash - cash_start) / cash_start

    return _result(ids, daily, led, rej, ambiguous, unresolved, sess_trades)


def _result(ids, daily, led, rej, ambiguous, unresolved, sess_trades) -> AccountResult:
    ledger = tuple(LedgerEntry(ids[a], b, c, d, e, f, EXIT_REASONS[g], h) for a, b, c, d, e, f, g, h in led)
    rejections = tuple(Rejection(ids[a], b, REJECT_REASONS[c]) for a, b, c in rej)
    mdd = 0.0
    if len(daily):
        curve = np.concatenate(([1.0], np.cumprod(1.0 + daily)))
        mdd = float(np.max(1.0 - curve / np.maximum.accumulate(curve)))
    by_stock: dict[str, float] = {}
    for e in ledger:
        by_stock[e.asset_id] = by_stock.get(e.asset_id, 0.0) + abs(e.net_pnl)
    gross = sum(by_stock.values())
    tot = float(np.abs(daily).sum())
    summary = {
        "trades": float(len(ledger)),
        "active_sessions": float(int((sess_trades > 0).sum())),
        "max_drawdown": mdd,
        "max_stock_share": (max(by_stock.values()) / gross) if gross > 0 else 0.0,
        "max_date_share": float(np.abs(daily).max() / tot) if tot > 0 else 0.0,
        "rejections": float(len(rej)),
    }
    return AccountResult(daily, ledger, rejections, ambiguous, unresolved, summary)
