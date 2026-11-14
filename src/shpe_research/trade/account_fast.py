"""C5 Numba port of the C4 funded-account engine (``trade.account`` stays the correctness oracle).

The kernel is a line-by-line port of the oracle's hot loop, ``@njit(cache=True, nogil=True)``, so a
thread pool scales across independent candidates that share one read-only panel.

Numeric policy. Market data stay in the panel's own dtype (float32 for real panels) and are promoted to
float64 on load, exactly like the oracle's ``np.asarray(..., float64)``. Every price, cash and equity
accumulator is float64, so the daily returns match the oracle to float rounding (the parity tests assert
``rtol=1e-12``) and trade counts, ledger rows and rejection reasons match exactly. Only storage is
float32; float32 accumulators were not needed and were not used.

Layout. Panel arrays stay symbol-major ``(n_symbols, n_minutes)`` as produced by A6. The per-candidate
signal mask is bucketed by minute inside the kernel (one sequential pass, only the replayed sessions),
which replaces the oracle's per-call transpose. Per-call allocations are small fixed-size workspaces
(positions, scratch) plus the event list (one int per signal); ledger and rejection records are
allocated only when ``return_ledger=True``, otherwise rejections are only counted by reason.

``prepare_panel`` hoists the per-panel work (session starts, id ranks, liquidity ranking) so many
candidates reuse it; ``simulate_account_fast`` accepts either a ``MarketPanel`` or a ``PreparedPanel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from shpe_research.trade import fill_proxy
from shpe_research.trade.account import (
    EXIT_REASONS,
    R_BAD,
    R_CANCEL,
    R_MAX_POS,
    R_NO_BAR,
    R_SIZE0,
    RANKINGS,
    REJECT_REASONS,
    X_HALT,
    X_MAX_HOLD,
    X_SESSION,
    X_STOP,
    X_UNRESOLVED,
)
from shpe_research.trade.account_types import (
    AccountConfig,
    AccountResult,
    LedgerEntry,
    MarketPanel,
    Rejection,
    SignalMask,
)

N_REJECT = len(REJECT_REASONS)
_SORT_INSERTION_MAX = 32


@dataclass(frozen=True)
class PreparedPanel:
    """Per-panel quantities shared by every candidate replayed on the panel (read-only)."""

    panel: MarketPanel
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    eligible: np.ndarray
    starts: np.ndarray
    lens: np.ndarray
    rank: np.ndarray  # (n_sessions, n_symbols) int32, 0 = first priority
    ranking: str
    lookback: int


def prepare_panel(panel: MarketPanel, ranking: str = "dollar_volume_20d", lookback: int = 20) -> PreparedPanel:
    """Hoist the candidate-independent work; contiguous views of the panel arrays (no copy if already so)."""
    if ranking not in RANKINGS:
        raise ValueError(f"unknown ranking {ranking!r}; choose from {sorted(RANKINGS)}")
    n_sym, n_min = panel.n_symbols, panel.n_minutes
    si = np.asarray(panel.session_index)
    if n_min:
        starts = np.flatnonzero(np.concatenate(([True], si[1:] != si[:-1]))).astype(np.int64)
    else:
        starts = np.zeros(0, dtype=np.int64)
    lens = np.diff(np.concatenate((starts, [n_min]))).astype(np.int64)
    id_rank = np.empty(n_sym, dtype=np.int64)
    id_rank[np.argsort(np.asarray(panel.asset_ids, dtype=object), kind="stable")] = np.arange(n_sym)
    close64 = np.asarray(panel.close, dtype=np.float64)
    vol64 = np.asarray(panel.volume, dtype=np.float64)
    rk = RANKINGS[ranking](close64, vol64, starts, id_rank, lookback)
    c = np.ascontiguousarray
    return PreparedPanel(
        panel, c(panel.open), c(panel.high), c(panel.low), c(panel.close), c(panel.volume), c(panel.eligible),
        starts, lens, np.ascontiguousarray(rk), ranking, lookback,
    )


# ---------------------------------------------------------------------------------------- kernel


@njit(cache=True, nogil=True)
def _bucket_events(sig, elig, c_lo, n_min):
    """Signals (signal AND eligible) from column ``c_lo`` on, grouped by minute, ascending symbol within."""
    n_sym = sig.shape[0]
    off = np.zeros(n_min + 1, dtype=np.int64)
    for i in range(n_sym):
        for t in range(c_lo, n_min):
            if sig[i, t] and elig[i, t]:
                off[t + 1] += 1
    for t in range(n_min):
        off[t + 1] += off[t]
    ev = np.empty(off[n_min], dtype=np.int32)
    fill = off[:n_min].copy()
    for i in range(n_sym):
        for t in range(c_lo, n_min):
            if sig[i, t] and elig[i, t]:
                ev[fill[t]] = i
                fill[t] += 1
    return off, ev


@njit(cache=True, nogil=True)
def _floor(x):
    return math.floor(x + 1e-9)


@njit(cache=True, nogil=True)
def _size_order(
    equity, price, stop_distance, cash_free, unit_cost, exposure_used, liq_shares,
    per_stock_pct, risk_pct, total_exposure_cap, size_multiplier,
):
    """Port of ``funded_sizing.size_order``: returns ``(qty, binding_cap_code)``."""
    if not (equity > 0 and price > 0 and stop_distance > 0 and unit_cost > 0) or not (
        math.isfinite(equity) and math.isfinite(price) and math.isfinite(stop_distance)
    ):
        return 0, 5
    best = size_multiplier * per_stock_pct * equity / price
    code = 0
    c = size_multiplier * risk_pct * equity / stop_distance
    if c < best:
        best, code = c, 1
    c = cash_free / unit_cost
    if c < best:
        best, code = c, 2
    c = (total_exposure_cap * equity - exposure_used) / price
    if c < best:
        best, code = c, 3
    if liq_shares < best:
        best, code = liq_shares, 4
    q = _floor(best)
    return (int(q) if q > 0 else 0), code


@njit(cache=True, nogil=True)
def _liquidity_cap(buf, n, cap_pct, min_obs):
    """Port of ``funded_sizing.liquidity_cap_shares`` (sorts ``buf[:n]`` in place)."""
    if n < min_obs or n <= 0:
        return 0.0
    buf[:n].sort()
    mid = n // 2
    med = buf[mid] if n % 2 else 0.5 * (buf[mid - 1] + buf[mid])
    return cap_pct * med


@njit(cache=True, nogil=True)
def _sort_candidates(keys, syms, n):
    """Sort ``syms[:n]`` by ascending ``keys[:n]`` (keys are distinct ranks) in place."""
    if n <= _SORT_INSERTION_MAX:
        for i in range(1, n):
            k, s = keys[i], syms[i]
            j = i - 1
            while j >= 0 and keys[j] > k:
                keys[j + 1], syms[j + 1] = keys[j], syms[j]
                j -= 1
            keys[j + 1], syms[j + 1] = k, s
    else:
        order = np.argsort(keys[:n])
        tmp = syms[:n].copy()
        for i in range(n):
            syms[i] = tmp[order[i]]


@njit(cache=True, nogil=True)
def _kernel(
    open_, high_, low_, close_, vol_, atr_, ev_off, ev_sym, entry_score, use_thr, thr,
    starts, lens, rk, warmup, max_pos, lookback, min_obs, cb, markup,
    stop_atr, trail_act, trail_dist, max_hold, exit_buf,
    starting_cash, ref_equity, use_ref, halt_pct, per_stock_pct, risk_pct, exposure_cap, liq_pct, mult,
    record, led_i, led_f, rej_i,
    daily, sess_trades, counters, rej_counts, by_stock,
):
    """Hot loop. ``counters`` = [ambiguous, unresolved, n_ledger, n_rej_records]."""
    n_sym = open_.shape[0]
    n_sess = starts.shape[0]
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
    entered = np.full(n_sym, -1, dtype=np.int64)
    pend = np.zeros(n_sym, dtype=np.bool_)
    buf = np.empty(max(lookback, 1), dtype=np.float64)
    ck = np.empty(n_sym, dtype=np.int64)  # candidate rank keys
    cs = np.empty(n_sym, dtype=np.int64)  # candidate symbols
    n_led = 0
    n_rej = 0
    ambiguous = 0
    unresolved = 0

    cash = starting_cash
    for s in range(warmup, n_sess):
        c0 = starts[s]
        length = lens[s]
        exit_m = length - exit_buf
        if use_ref:
            cash = ref_equity
        cash_start = cash
        halt_level = cash_start * (1.0 - halt_pct)
        halted = False
        n_open = 0
        n_pend = 0
        reserved = 0.0
        di = s - warmup
        for m in range(length):
            t = c0 + m
            # 1. pending entries meet this bar's open
            for k in range(n_pend):
                sym = o_sym[k]
                pend[sym] = False
                reserved -= o_res[k]
                nxt = np.float64(open_[sym, t])
                if nxt != nxt:
                    rej_counts[R_NO_BAR] += 1
                    if record:
                        rej_i[n_rej, 0], rej_i[n_rej, 1], rej_i[n_rej, 2] = sym, t, R_NO_BAR
                        n_rej += 1
                    continue
                if nxt > o_limit[k]:
                    rej_counts[R_CANCEL] += 1
                    if record:
                        rej_i[n_rej, 0], rej_i[n_rej, 1], rej_i[n_rej, 2] = sym, t, R_CANCEL
                        n_rej += 1
                    continue
                px = nxt
                j = 0
                while p_sym[j] >= 0:
                    j += 1
                qty = o_qty[k]
                fee = qty * px * cb
                cash -= qty * px + fee
                a = o_atr[k]
                p_sym[j], p_qty[j], p_entry[j], p_atr[j] = sym, qty, px, a
                p_stop[j] = px - stop_atr * a
                p_hc[j] = px
                p_trail[j] = False
                p_trig[j] = px + trail_act * a
                p_thold[j] = m + max_hold
                p_last[j], p_cost[j], p_ecol[j] = px, fee, t
                entered[sym] = s
                n_open += 1
            n_pend = 0
            pend_notional = 0.0

            # 2. exits
            pos_value = 0.0
            if n_open:
                for j in range(max_pos):
                    sym = p_sym[j]
                    if sym < 0:
                        continue
                    po = np.float64(open_[sym, t])
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
                        elif np.float64(low_[sym, t]) <= p_stop[j]:
                            reason, xpx = X_STOP, p_stop[j]
                            if np.float64(high_[sym, t]) >= p_trig[j]:
                                ambiguous += 1
                        else:
                            pc = np.float64(close_[sym, t])
                            if pc == pc:
                                if pc > p_hc[j]:
                                    p_hc[j] = pc
                                if (not p_trail[j]) and pc >= p_entry[j] + trail_act * p_atr[j]:
                                    p_trail[j] = True
                                if p_trail[j]:
                                    cand = p_hc[j] - trail_dist * p_atr[j]
                                    if cand > p_stop[j]:
                                        p_stop[j] = cand
                                p_last[j] = pc
                    if reason >= 0:
                        qty = p_qty[j]
                        cash += qty * xpx * (1.0 - cb)
                        pnl = qty * (xpx - p_entry[j]) - p_cost[j] - qty * xpx * cb
                        by_stock[sym] += abs(pnl)
                        if record:
                            led_i[n_led, 0], led_i[n_led, 1], led_i[n_led, 2] = sym, p_ecol[j], t
                            led_i[n_led, 3], led_i[n_led, 4] = int(qty), reason
                            led_f[n_led, 0], led_f[n_led, 1], led_f[n_led, 2] = p_entry[j], xpx, pnl
                        n_led += 1
                        sess_trades[di] += 1
                        p_sym[j] = -1
                        n_open -= 1
                    else:
                        pos_value += p_qty[j] * p_last[j]

            # 3. mark and daily halt
            equity = cash + pos_value
            if (not halted) and equity <= halt_level:
                halted = True

            # 4-6. candidates, ranking, allocation, submission
            if halted or m + 1 >= exit_m:
                continue
            lo_e, hi_e = ev_off[t], ev_off[t + 1]
            if hi_e == lo_e:
                continue
            nc = 0
            for e in range(lo_e, hi_e):
                sym = ev_sym[e]
                if entered[sym] == s or pend[sym]:
                    continue
                if use_thr and not (entry_score[sym, t] >= thr):
                    continue
                ck[nc] = rk[s, sym]
                cs[nc] = sym
                nc += 1
            if nc == 0:
                continue
            if nc > 1:
                _sort_candidates(ck, cs, nc)
            for ci in range(nc):
                sym = cs[ci]
                if n_open + n_pend >= max_pos:
                    if record:
                        for cj in range(ci, nc):
                            rej_i[n_rej, 0], rej_i[n_rej, 1], rej_i[n_rej, 2] = cs[cj], t, R_MAX_POS
                            n_rej += 1
                    rej_counts[R_MAX_POS] += nc - ci
                    break
                price = np.float64(close_[sym, t])
                a = np.float64(atr_[sym, t])
                if not (0.0 < price < np.inf and 0.0 < a < np.inf):
                    rej_counts[R_BAD] += 1
                    if record:
                        rej_i[n_rej, 0], rej_i[n_rej, 1], rej_i[n_rej, 2] = sym, t, R_BAD
                        n_rej += 1
                    continue
                limit = price * (1.0 + markup)
                nv = 0
                for q in range(1, lookback + 1):
                    ps = s - q
                    if ps < 0:
                        break
                    if m < lens[ps]:
                        v = np.float64(vol_[sym, starts[ps] + m])
                        if v == v:
                            buf[nv] = v
                            nv += 1
                liq = _liquidity_cap(buf, nv, liq_pct, min_obs)
                unit = limit * (1.0 + cb)
                qty_i, cap = _size_order(
                    equity, price, stop_atr * a, cash - reserved, unit,
                    pos_value + pend_notional, liq, per_stock_pct, risk_pct, exposure_cap, mult,
                )
                if qty_i <= 0:
                    rej_counts[R_SIZE0 + cap] += 1
                    if record:
                        rej_i[n_rej, 0], rej_i[n_rej, 1], rej_i[n_rej, 2] = sym, t, R_SIZE0 + cap
                        n_rej += 1
                    continue
                o_sym[n_pend], o_qty[n_pend], o_limit[n_pend], o_atr[n_pend] = sym, qty_i, limit, a
                o_res[n_pend] = qty_i * unit
                reserved += qty_i * unit
                pend_notional += qty_i * limit
                pend[sym] = True
                n_pend += 1

        # session end: anything still open could not exit (missing bars); close at last valid mark
        for j in range(max_pos):
            sym = p_sym[j]
            if sym >= 0:
                qty = p_qty[j]
                xpx = p_last[j]
                cash += qty * xpx * (1.0 - cb)
                pnl = qty * (xpx - p_entry[j]) - p_cost[j] - qty * xpx * cb
                by_stock[sym] += abs(pnl)
                if record:
                    led_i[n_led, 0], led_i[n_led, 1], led_i[n_led, 2] = sym, p_ecol[j], c0 + length - 1
                    led_i[n_led, 3], led_i[n_led, 4] = int(qty), X_UNRESOLVED
                    led_f[n_led, 0], led_f[n_led, 1], led_f[n_led, 2] = p_entry[j], xpx, pnl
                n_led += 1
                sess_trades[di] += 1
                unresolved += 1
                p_sym[j] = -1
        for k in range(n_pend):
            pend[o_sym[k]] = False
        daily[di] = (cash - cash_start) / cash_start

    counters[0], counters[1], counters[2], counters[3] = ambiguous, unresolved, n_led, n_rej


# ---------------------------------------------------------------------------------------- wrapper


def _validate(prep: PreparedPanel, mask: SignalMask, cfg: AccountConfig, warmup: int, entry_score) -> None:
    n_sym, n_min = prep.panel.n_symbols, prep.panel.n_minutes
    if mask.signal.shape != (n_sym, n_min):
        raise ValueError(f"mask shape {mask.signal.shape} != panel shape {(n_sym, n_min)}")
    if cfg.fill.kind != "limit_markup":
        raise ValueError(f"unsupported fill model {cfg.fill.kind!r}")
    if cfg.ranking != prep.ranking:
        raise ValueError(f"cfg.ranking {cfg.ranking!r} != prepared ranking {prep.ranking!r}")
    if cfg.liquidity_lookback != prep.lookback:
        raise ValueError(f"cfg.liquidity_lookback {cfg.liquidity_lookback} != prepared lookback {prep.lookback}")
    if cfg.entry_score_threshold is not None and (entry_score is None or entry_score.shape != (n_sym, n_min)):
        raise ValueError("entry_score_threshold needs an entry_score array of panel shape")
    if cfg.exit.max_hold_minutes <= 0 or cfg.max_positions <= 0:
        raise ValueError("max_hold_minutes and max_positions must be positive")
    if not 0 <= warmup <= len(prep.starts):
        raise ValueError("warmup_sessions out of range")


def simulate_account_fast(
    panel: MarketPanel | PreparedPanel, mask: SignalMask, cfg: AccountConfig, *,
    warmup_sessions: int = 0, entry_score: np.ndarray | None = None, return_ledger: bool = False,
) -> AccountResult:
    """Numba replay with the oracle's inputs and outputs.

    Daily returns, trade and rejection counts, ``ambiguous_bars``, ``unresolved_positions`` and the
    ``summary`` are always produced. ``ledger`` and ``rejections`` are populated only with
    ``return_ledger=True``; otherwise they are empty tuples (``summary["trades"]`` and
    ``summary["rejections"]`` still carry the counts) and ``rejection_counts`` is not exposed.
    Safe to call from many threads on one ``PreparedPanel``.
    """
    prep = panel if isinstance(panel, PreparedPanel) else prepare_panel(panel, cfg.ranking, cfg.liquidity_lookback)
    _validate(prep, mask, cfg, warmup_sessions, entry_score)
    n_sym, n_min = prep.panel.n_symbols, prep.panel.n_minutes
    n_sess = len(prep.starts)
    n_out = n_sess - warmup_sessions
    ex = cfg.exit

    sig = np.ascontiguousarray(mask.signal)
    atr = np.ascontiguousarray(mask.atr)
    c_lo = int(prep.starts[warmup_sessions]) if warmup_sessions < n_sess else n_min
    ev_off, ev_sym = _bucket_events(sig, prep.eligible, c_lo, n_min)
    n_ev = int(ev_off[n_min])
    thr = cfg.entry_score_threshold
    es = np.ascontiguousarray(entry_score) if thr is not None else np.zeros((1, 1), dtype=np.float32)

    cap_led = n_ev if return_ledger else 0  # at most one fill per signal event
    led_i = np.empty((cap_led, 5), dtype=np.int64)
    led_f = np.empty((cap_led, 3), dtype=np.float64)
    rej_i = np.empty((2 * n_ev if return_ledger else 0, 3), dtype=np.int64)
    daily = np.zeros(n_out)
    sess_trades = np.zeros(n_out, dtype=np.int64)
    counters = np.zeros(4, dtype=np.int64)
    rej_counts = np.zeros(N_REJECT, dtype=np.int64)
    by_stock = np.zeros(n_sym)

    _kernel(
        prep.open, prep.high, prep.low, prep.close, prep.volume, atr, ev_off, ev_sym, es, thr is not None,
        float(thr) if thr is not None else 0.0,
        prep.starts, prep.lens, prep.rank, warmup_sessions, cfg.max_positions, cfg.liquidity_lookback,
        cfg.liquidity_min_obs, fill_proxy.side_cost_rate(cfg.cost_bps), cfg.fill.limit_markup,
        ex.initial_stop_atr, ex.trail_activation_atr, ex.trail_distance_atr, ex.max_hold_minutes,
        ex.session_exit_minutes_before_close,
        float(cfg.starting_cash), float(cfg.search_replay_reference_equity or 0.0),
        cfg.search_replay_reference_equity is not None, cfg.daily_halt_pct, cfg.per_stock_pct, cfg.risk_pct,
        cfg.total_exposure_cap, cfg.liquidity_cap_pct,
        1.0 if cfg.size_multiplier is None else float(cfg.size_multiplier),
        return_ledger, led_i, led_f, rej_i,
        daily, sess_trades, counters, rej_counts, by_stock,
    )
    ambiguous, unresolved, n_led, n_rej = (int(x) for x in counters)
    ids = prep.panel.asset_ids
    ledger: tuple[LedgerEntry, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    if return_ledger:
        ledger = tuple(
            LedgerEntry(ids[r[0]], r[1], r[2], r[3], f[0], f[1], EXIT_REASONS[r[4]], f[2])
            for r, f in zip(led_i[:n_led].tolist(), led_f[:n_led].tolist(), strict=True)
        )
        rejections = tuple(Rejection(ids[r[0]], r[1], REJECT_REASONS[r[2]]) for r in rej_i[:n_rej].tolist())

    mdd = 0.0
    if n_out:
        curve = np.concatenate(([1.0], np.cumprod(1.0 + daily)))
        mdd = float(np.max(1.0 - curve / np.maximum.accumulate(curve)))
    gross = float(by_stock.sum())
    tot = float(np.abs(daily).sum())
    summary = {
        "trades": float(n_led),
        "active_sessions": float(int((sess_trades > 0).sum())),
        "max_drawdown": mdd,
        "max_stock_share": float(by_stock.max()) / gross if gross > 0 else 0.0,
        "max_date_share": float(np.abs(daily).max() / tot) if tot > 0 else 0.0,
        "rejections": float(int(rej_counts.sum())),
    }
    return AccountResult(daily, ledger, rejections, ambiguous, unresolved, summary)


__all__ = ["PreparedPanel", "prepare_panel", "simulate_account_fast"]
