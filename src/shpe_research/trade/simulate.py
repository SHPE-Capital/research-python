"""Trade simulator: candidates + bars + ExitPolicy -> TradeLedger, on the platform's bar fill rules (T1).

Mirrors trading-platform ``core/execution/fillModel.ts`` (BACKTEST_ENGINE_VERSION 2):

* A candidate decided at ``decision_ts`` fills at the open of the first bar with ``ts >= decision_ts``
  (a missing minute means the next bar that exists). Buys pay ``open * (1+hs) * (1+slip)``, sells
  ``open * (1-hs) * (1-slip)``. Zero volume bars give no fill; volume is capped at a fraction of the bar.
* Stops and targets are checked on bars *after* the entry bar. A stop gapped through fills from the open.
  When stop and target are both touched in one bar, the stop wins.
* Signal exits (exit rows, ``max_hold_bars``, ``flat_by``, ``session_end``) are decided at a bar's end and fill
  at the next bar's open. Order exits (stop, target, trailing) fill intrabar.

Only regular-session bars are simulated. Open positions at the end of the data are not closed by
invention: they are left out of the ledger and counted in ``ledger.attrs["open_at_end"]``.

Timestamps: ``entry_ts`` and the ``exit_ts`` of a signal exit are the fill bar's ``ts`` (its open).
An order exit depends on the whole bar, so its ``exit_ts`` is that bar's ``ts_available``; this keeps
``label_end_ts == exit_ts`` safe for purged validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shpe_research.contracts.base import UTC_NS
from shpe_research.contracts.candidates import CandidateFrame
from shpe_research.data.calendar import NY, get_calendar
from shpe_research.trade import exit_policy as ex
from shpe_research.trade.types import (
    OPTIONAL_LEDGER_COLUMNS,
    TRADE_LEDGER_COLUMNS,
    CostModel,
    ExitPolicy,
    FillModel,
    RiskLimits,
    SizingRule,
    TradeLedger,
)

_NAT = np.iinfo("int64").min
_TS_COLUMNS = ("decision_ts", "entry_ts", "exit_ts", "label_end_ts")


def _ns(values) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8


@dataclass
class _Costs:
    half_spread_bps: float
    slippage_bps: float
    commission_per_share: float

    def price(self, reference: float, buy: bool) -> float:
        hs, sl = self.half_spread_bps / 1e4, self.slippage_bps / 1e4
        return reference * (1 + hs) * (1 + sl) if buy else reference * (1 - hs) * (1 - sl)


class _SymbolBars:
    """Regular-session bars of one symbol as arrays, plus calendar facts per bar."""

    def __init__(self, df: pd.DataFrame, closes: dict[int, int], early: set[int]):
        self.ts = _ns(df["ts"])
        self.ts_av = _ns(df["ts_available"])
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)  # noqa: E741
        self.c = df["close"].to_numpy(float)
        self.v = df["volume"].to_numpy(float)
        self.n = len(df)
        local = pd.DatetimeIndex(df["ts"]).tz_convert(NY).tz_localize(None).normalize()
        self.day = local.values.astype("datetime64[D]").astype("int64")
        self.close_ns = np.array([closes.get(int(d), _NAT) for d in self.day], dtype="int64")
        self.early = np.array([int(d) in early for d in self.day], dtype=bool)
        self._atr: dict[int, np.ndarray] = {}

    def atr(self, window: int) -> np.ndarray:
        if window not in self._atr:
            self._atr[window] = ex.atr(self.h, self.l, self.c, window)
        return self._atr[window]


def _entry_candidate_fields(row: dict) -> tuple[int, float, float, float]:
    """(qty, stop_price, target_price, entry_price) of a candidate; qty defaults to 1, levels to NaN."""
    qty = _float(row.get("qty"))
    qty = 1 if not math.isfinite(qty) or qty <= 0 else int(qty)
    return (qty, *(_float(row.get(k)) for k in ("stop_price", "target_price", "entry_price")))


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _unfilled(row: dict, strategy_id: str) -> dict:
    return {"symbol": row["symbol"], "strategy_id": strategy_id, "side": row["side"], "decision_ts": row["_dec"],
            "entry_ts": _NAT, "entry_price": math.nan, "qty": None, "exit_ts": _NAT, "exit_price": math.nan,
            "exit_reason": "unfilled", "gross_return": math.nan, "cost": math.nan, "net_return": math.nan,
            "label_end_ts": _NAT, "mfe": math.nan, "mae": math.nan, "partial_fill": False, "early_close": False}


@dataclass
class _Trade:
    record: dict
    status: str  # "closed" | "open" | "unfilled"
    i0: int = -1
    release: int = 0  # first bar index at which a new entry may fill


def _simulate_one(
    sb: _SymbolBars, row: dict, strategy_id: str, policy: ExitPolicy, fill: FillModel, costs: _Costs,
    extra_signals: list[tuple[int, str]], exit_rows: tuple[np.ndarray, list[str]],
) -> _Trade:
    """Simulate one entry candidate to its exit. `extra_signals` are (fill bar index, reason) exits to honour."""
    sign = 1 if row["side"] == "long" else -1
    buy_entry = sign > 0
    dec = row["_dec"]
    qty, cand_stop, cand_target, cand_level = _entry_candidate_fields(row)
    i1 = int(np.searchsorted(sb.ts, dec, side="left"))
    if i1 >= sb.n:
        return _Trade(_unfilled(row, strategy_id), "unfilled")

    # --- entry fill -------------------------------------------------------------------------------
    i0, ref_entry = -1, math.nan
    if policy.entry_order == "stop":
        j = i1
        while j < sb.n and sb.day[j] == sb.day[i1]:  # DAY order: expires with the session
            if sb.v[j] > 0:
                hit = ex.entry_stop_reference(sign, cand_level, sb.o[j], sb.h[j], sb.l[j])
                if hit is not None:
                    i0, ref_entry = j, hit
                    break
            j += 1
    elif sb.v[i1] > 0:
        i0, ref_entry = i1, sb.o[i1]
    if i0 < 0:
        return _Trade(_unfilled(row, strategy_id), "unfilled")

    partial = False
    frac = fill.volume_participation_cap
    cap = qty if frac >= 1 else max(int(math.floor(sb.v[i0] * frac)), 1)
    if qty > cap:
        if not fill.allow_partial_fills:
            return _Trade(_unfilled(row, strategy_id), "unfilled")
        qty, partial = cap, True
    entry_fill = costs.price(ref_entry, buy_entry)

    # --- levels -----------------------------------------------------------------------------------
    atr = sb.atr(policy.atr_window)
    atr_at_entry = atr[i0 - 1] if i0 >= 1 else math.nan
    stop = ex.initial_stop(policy, sign, entry_fill, cand_stop, atr_at_entry)
    target = ex.initial_target(policy, sign, entry_fill, stop, cand_target)
    state = ex.StopState(policy, sign, entry_fill, stop)
    state.update(sb.h[i0], sb.l[i0], atr[i0])
    resting = (
        math.isfinite(stop) or math.isfinite(target) or policy.trail_rule != "none" or policy.entry_order == "stop"
    )

    # --- signal exits armed by index --------------------------------------------------------------
    armed: dict[int, str] = {}

    def arm(idx: int, reason: str) -> None:
        idx = max(idx, i0 + 1)
        if idx not in armed or ex.SIGNAL_PRIORITY[reason] < ex.SIGNAL_PRIORITY[armed[idx]]:
            armed[idx] = reason

    for idx, reason in extra_signals:
        arm(idx, reason)
    ex_dec, ex_reasons = exit_rows
    first = int(np.searchsorted(ex_dec, dec, side="right")) if policy.use_signal_exits else len(ex_dec)
    if first < len(ex_dec):
        first_idx = max(int(np.searchsorted(sb.ts, ex_dec[first], side="left")), i0 + 1)
        k = first
        while k < len(ex_dec) and max(int(np.searchsorted(sb.ts, ex_dec[k], side="left")), i0 + 1) == first_idx:
            arm(first_idx, ex_reasons[k])
            k += 1
    if policy.max_hold_bars is not None:
        arm(i0 + policy.max_hold_bars, "max_hold_exit")
    if policy.flat_by is not None:
        flat = pd.Timestamp(f"{pd.Timestamp(sb.ts[i0], tz='UTC').tz_convert(NY).date()} {policy.flat_by}")
        arm(int(np.searchsorted(sb.ts, flat.tz_localize(NY).tz_convert("UTC").value, side="left")), "exit")
    pending_idx = sorted(armed)

    # --- walk forward -----------------------------------------------------------------------------
    for j in range(i0 + 1, sb.n):
        tradable = sb.v[j] > 0
        due = [armed[i] for i in pending_idx if i <= j]
        if policy.session_end and tradable and (
            sb.ts_av[j] >= sb.close_ns[j] or j == sb.n - 1 or sb.day[j + 1] != sb.day[j]
        ):
            due.append("session_end")
        if tradable:
            stop_ref = ex.stop_reference(sign, state.stop, sb.o[j], sb.h[j], sb.l[j])
            if stop_ref is not None:
                return _close(sb, row, strategy_id, costs, i0, j, sign, qty, partial, ref_entry, entry_fill,
                              stop_ref, "stop_loss", True, resting)
            if due:
                reason = min(due, key=ex.SIGNAL_PRIORITY.__getitem__)
                return _close(sb, row, strategy_id, costs, i0, j, sign, qty, partial, ref_entry, entry_fill,
                              sb.o[j], reason, False, resting)
            tgt_ref = ex.target_reference(sign, target, sb.o[j], sb.h[j], sb.l[j])
            if tgt_ref is not None:
                return _close(sb, row, strategy_id, costs, i0, j, sign, qty, partial, ref_entry, entry_fill,
                              tgt_ref, "take_profit", True, resting)
        state.update(sb.h[j], sb.l[j], atr[j])
    return _Trade({}, "open", i0=i0, release=sb.n + 1)


def _close(sb, row, strategy_id, costs, i0, j, sign, qty, partial, ref_entry, entry_fill, ref_exit, reason, intrabar,
           resting) -> _Trade:
    exit_fill = costs.price(ref_exit, buy=sign < 0)
    gross = sign * (ref_exit / ref_entry - 1)
    cost = qty * (abs(entry_fill - ref_entry) + abs(exit_fill - ref_exit) + 2 * costs.commission_per_share)
    last = j + 1 if intrabar else j  # bars whose range counts towards MFE/MAE
    hi, lo = sb.h[i0:last].max(), sb.l[i0:last].min()
    if sign > 0:
        mfe, mae = hi / entry_fill - 1, lo / entry_fill - 1
    else:
        mfe, mae = 1 - lo / entry_fill, 1 - hi / entry_fill
    exit_ts = int(sb.ts_av[j] if intrabar else sb.ts[j])
    record = {
        "symbol": row["symbol"], "strategy_id": strategy_id, "side": row["side"], "decision_ts": row["_dec"],
        "entry_ts": int(sb.ts[i0]), "entry_price": entry_fill, "qty": qty, "exit_ts": exit_ts,
        "exit_price": exit_fill, "exit_reason": reason, "gross_return": gross, "cost": cost,
        "net_return": gross - cost / (qty * ref_entry), "label_end_ts": exit_ts, "mfe": mfe, "mae": mae,
        "partial_fill": partial, "early_close": bool(resting and sb.early[i0 : j + 1].any()),
    }
    return _Trade(record, "closed", i0=i0, release=j + 1 if intrabar else j)


def simulate_trades(
    bars: pd.DataFrame,
    candidates: CandidateFrame,
    policy: ExitPolicy,
    fill: FillModel = FillModel(),  # noqa: B008  (frozen dataclass default)
    costs: CostModel | None = None,
    sizing: SizingRule | None = None,
    risk: RiskLimits | None = None,
) -> TradeLedger:
    """Fill candidates on `bars` under `policy` and return the trade ledger.

    Inputs: contract `bars` (only ``session == "regular"`` rows are used); `candidates` with ``symbol``,
    ``decision_ts``, ``side`` and optionally ``strategy_id``, ``action`` (``enter`` default, or ``exit``),
    ``exit_reason``, ``qty``, ``stop_price``, ``target_price`` and ``entry_price`` (the trigger of a stop
    entry). Percent rules take fractions (0.01 = 1%).

    Timing: a candidate is known at ``decision_ts`` and fills no earlier than the first bar with
    ``ts >= decision_ts``; nothing after a trade's ``exit_ts`` influences it. Missing data: no next bar or
    a zero-volume entry bar gives a row with ``exit_reason == "unfilled"``; a missing minute shifts the
    fill to the next existing bar.

    One position per ``(symbol, strategy_id)``: an entry while open is ignored, except an opposite-side
    entry, which closes the position at its own fill bar (``exit``) and then opens. Returns one row per
    attempted trade, ordered by ``decision_ts``, with `TRADE_LEDGER_COLUMNS` then the optional columns.

    ``costs`` (when given) supplies the spread, slippage and commission instead of `fill`; the impact
    term is not applied yet (T5). ``sizing`` and ``risk`` are accepted but not applied until T2 and T4
    land; ``qty`` comes from the candidate, else 1.
    """
    cost = _Costs(*(
        (costs.half_spread_bps, costs.slippage_bps, costs.commission_per_share)
        if costs is not None
        else (fill.half_spread_bps, fill.slippage_bps, fill.commission_per_share)
    ))
    cands = candidates.reset_index(drop=True).copy()
    if "strategy_id" not in cands:
        cands["strategy_id"] = ""
    if "action" not in cands:
        cands["action"] = "enter"
    cands["action"] = cands["action"].fillna("enter")
    cands["_dec"] = _ns(cands["decision_ts"]) if len(cands) else np.array([], dtype="int64")
    cands = cands.sort_values("_dec", kind="mergesort")

    reg = bars[bars["session"] == "regular"]
    reg = reg[reg["symbol"].isin(set(cands["symbol"]))]
    closes: dict[int, int] = {}
    early: set[int] = set()
    if len(reg):
        days = pd.DatetimeIndex(reg["ts"]).tz_convert(NY).tz_localize(None).normalize()
        sched = get_calendar().schedule(days.min(), days.max())
        for d, r in sched.iterrows():
            key = int(np.datetime64(d, "D").astype("int64"))
            closes[key] = int(pd.Timestamp(r["close"]).as_unit("ns").value)
            if bool(r["early_close"]):
                early.add(key)

    records: list[dict] = []
    open_at_end = 0
    for (symbol, strategy_id), grp in cands.groupby(["symbol", "strategy_id"], sort=False):
        sym_bars = reg[reg["symbol"] == symbol].sort_values("ts", kind="mergesort")
        enters = grp[grp["action"] != "exit"].to_dict("records")
        exits_df = grp[grp["action"] == "exit"]
        given = exits_df["exit_reason"].tolist() if "exit_reason" in exits_df else [None] * len(exits_df)
        ex_rows = (exits_df["_dec"].to_numpy(), [r if r in ex.SIGNAL_PRIORITY else "exit" for r in given])
        if not len(sym_bars):
            records.extend(_unfilled(r, strategy_id) for r in enters)
            continue
        sb = _SymbolBars(sym_bars, closes, early)
        last: _Trade | None = None
        last_row: dict = {}
        last_extra: list[tuple[int, str]] = []
        group_records: list[dict | None] = []  # one slot per attempted trade; None while still open at data end
        for row in enters:
            row["symbol"] = symbol
            i1 = int(np.searchsorted(sb.ts, row["_dec"], side="left"))
            if last is not None and last.status != "unfilled" and i1 < last.release:
                if row["side"] == last_row["side"] or i1 <= last.i0:
                    continue  # one open position per (symbol, strategy_id): same-side entries are ignored
                last_extra = [*last_extra, (i1, "exit")]  # opposite side: close at this fill bar, then open
                last = _simulate_one(sb, last_row, strategy_id, policy, fill, cost, last_extra, ex_rows)
                group_records[-1] = last.record if last.status == "closed" else None
                if i1 < last.release:
                    continue
            trade = _simulate_one(sb, row, strategy_id, policy, fill, cost, [], ex_rows)
            group_records.append(trade.record if trade.status != "open" else None)
            last, last_row, last_extra = trade, row, []
        open_at_end += int(last is not None and last.status == "open")
        records.extend(r for r in group_records if r is not None)

    ledger = _to_ledger(records)
    ledger.attrs["open_at_end"] = open_at_end
    return ledger


def _to_ledger(records: list[dict]) -> TradeLedger:
    columns = [*TRADE_LEDGER_COLUMNS, *OPTIONAL_LEDGER_COLUMNS]
    df = pd.DataFrame.from_records(records, columns=columns)
    for col in _TS_COLUMNS:
        values = df[col].to_numpy(dtype="int64") if len(df) else np.array([], dtype="int64")
        df[col] = pd.Series(pd.to_datetime(values, unit="ns", utc=True), index=df.index).astype(UTC_NS)
    df["qty"] = pd.array(df["qty"].tolist(), dtype="Int64")
    for col in ("partial_fill", "early_close"):
        df[col] = df[col].astype(bool)
    for col in ("entry_price", "exit_price", "gross_return", "cost", "net_return", "mfe", "mae"):
        df[col] = df[col].astype(float)
    df["symbol"] = df["symbol"].astype(str)
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["side"] = df["side"].astype(str)
    df["exit_reason"] = df["exit_reason"].astype(str)
    if len(df):
        df = df.sort_values("decision_ts", kind="mergesort")
    return df.reset_index(drop=True)
