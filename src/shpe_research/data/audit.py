"""D2: audit stored bars before anyone trusts them.

    uv run shpe data audit --symbols SPY,KO --from 2026-10-01 --to 2026-10-27

Per symbol and New York session it reports:

    coverage           regular-session bars / expected minutes (early closes included).
                       Minutes without trades have no bar, so illiquid names score below 1
                       legitimately; this is information, not an error.
    morning_coverage   the same over 09:30-11:00, the window D4's universe screen uses.
    duplicates         repeated (symbol, ts) rows                                   [error]
    invalid_ohlc       high below open/close/low, low above open/close, prices <= 0,
                       negative volume                                               [error]
    off_session        bars on non-session days or outside 04:00-20:00 New York      [error]
    missing_day        a session in range with no regular bars (halt, delisting, bad
                       symbol, or a gap in the download)                             [warning]
    possible_split     overnight |log(open / previous close)| above a threshold;
                       D6 confirms these against corporate actions                   [warning]
    low_coverage       coverage below `low_coverage` (default 0.5)                   [warning]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.contracts.bars import timeframe_delta
from shpe_research.data.calendar import NY, TradingCalendar, get_calendar, session_date

ERROR_KINDS = ("duplicates", "invalid_ohlc", "off_session")
MORNING = ("09:30", "11:00")
# report column -> per-bar flag summed per session
_COUNTS = {
    "regular_bars": "regular",
    "morning_bars": "morning",
    "pre_bars": "pre",
    "post_bars": "post",
    "duplicates": "duplicate",
    "invalid_ohlc": "invalid",
    "off_session": "off",
}


@dataclass
class AuditReport:
    days: pd.DataFrame  # one row per symbol and session
    symbols: pd.DataFrame  # one row per symbol
    issues: pd.DataFrame  # symbol, date, kind, severity, detail

    @property
    def errors(self) -> pd.DataFrame:
        return self.issues[self.issues["severity"] == "error"]

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbols": self.symbols.to_dict("records"),
            "issues": self.issues.astype({"date": str}).to_dict("records"),
            "days": self.days.astype({"date": str}).to_dict("records"),
        }
        (out / "audit.json").write_text(json.dumps(payload, indent=1, default=str))
        (out / "audit.md").write_text(self.to_markdown())
        return out

    def to_markdown(self) -> str:
        lines = [
            "# Data audit",
            "",
            "| symbol | sessions | missing days | coverage | morning coverage | errors | warnings |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.symbols.itertuples():
            lines.append(f"| {r.symbol} | {r.sessions} | {r.missing_days} | {r.coverage:.3f} | "
                         f"{r.morning_coverage:.3f} | {r.errors} | {r.warnings} |")
        lines += ["", f"## Issues ({len(self.issues)})", ""]
        if self.issues.empty:
            lines.append("None.")
        else:
            lines += ["| symbol | date | kind | severity | detail |", "|---|---|---|---|---|"]
            for r in self.issues.itertuples():
                lines.append(f"| {r.symbol} | {pd.Timestamp(r.date).date()} | {r.kind} | {r.severity} | {r.detail} |")
        return "\n".join(lines) + "\n"


def _expected(schedule: pd.DataFrame, step: pd.Timedelta) -> pd.DataFrame:
    """Expected regular and morning bar counts per session."""
    days = schedule.index
    morning_start = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {MORNING[0]}") for d in days]).tz_localize(NY)
    morning_end = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {MORNING[1]}") for d in days]).tz_localize(NY)
    m_start = np.maximum(schedule["open"].to_numpy(), morning_start.tz_convert("UTC").to_numpy())
    m_end = np.minimum(schedule["close"].to_numpy(), morning_end.tz_convert("UTC").to_numpy())
    return pd.DataFrame(
        {
            "expected": ((schedule["close"] - schedule["open"]) / step).to_numpy(),
            "morning_expected": np.maximum(0, (m_end - m_start) / step),
            "early_close": schedule["early_close"].to_numpy(),
        },
        index=days,
    )


def audit_bars(
    bars: pd.DataFrame,
    start=None,
    end=None,
    timeframe: str = "1Min",
    split_threshold: float = 0.25,
    low_coverage: float = 0.5,
    cal: TradingCalendar | None = None,
) -> AuditReport:
    """Audit intraday contract bars over New York sessions start..end (default: the data's own range)."""
    cal = cal or get_calendar()
    step = timeframe_delta(timeframe)
    date = session_date(bars["ts"])
    start = pd.Timestamp(start) if start is not None else date.min()
    end = pd.Timestamp(end) if end is not None else date.max()
    expected = _expected(cal.schedule(start, end), step)
    ny = bars["ts"].dt.tz_convert(NY)
    hhmm = ny.dt.strftime("%H:%M")
    frame = bars.assign(
        date=date,
        regular=bars["session"] == "regular",
        pre=bars["session"] == "pre",
        post=bars["session"] == "post",
        morning=(bars["session"] == "regular") & (hhmm >= MORNING[0]) & (hhmm < MORNING[1]),
        duplicate=bars.duplicated(["symbol", "ts"]),
        invalid=(
            (bars["high"] < bars[["open", "close", "low"]].max(axis=1))
            | (bars["low"] > bars[["open", "close"]].min(axis=1))
            | (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (bars["volume"] < 0)
        ),
        off=(bars["session"] == "off") | ~date.isin(expected.index),
    )
    issues: list[dict] = []
    day_rows = []
    for symbol in sorted(bars["symbol"].unique()):
        sym = frame[frame["symbol"] == symbol]
        counts = sym.groupby("date")[list(_COUNTS.values())].sum().rename(columns={v: k for k, v in _COUNTS.items()})
        days = expected.join(counts, how="left").fillna(dict.fromkeys(_COUNTS, 0))
        days["coverage"] = days["regular_bars"] / days["expected"]
        has_morning = days["morning_expected"] > 0
        days["morning_coverage"] = (days["morning_bars"] / days["morning_expected"]).where(has_morning)
        reg = sym[sym["regular"]].sort_values("ts")
        first_open = reg.groupby("date")["open"].first()
        last_close = reg.groupby("date")["close"].last()
        days["overnight_gap"] = np.log(first_open / last_close.shift(1)).reindex(days.index)
        days.insert(0, "symbol", symbol)
        days.index.name = "date"
        day_rows.append(days.reset_index())
        issues += _session_issues(symbol, days, split_threshold, low_coverage)
        off_days = sym.loc[sym["off"] & ~sym["date"].isin(expected.index)].groupby("date").size()
        issues += [_issue(symbol, d, "off_session", "error", f"{n} bars on a non-session day")
                   for d, n in off_days.items()]

    days_df = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame()
    issues_df = pd.DataFrame(issues, columns=["symbol", "date", "kind", "severity", "detail"])
    return AuditReport(days=days_df, symbols=_summarize(days_df, issues_df), issues=issues_df)


def _issue(symbol: str, date, kind: str, severity: str, detail: str) -> dict:
    return {"symbol": symbol, "date": date, "kind": kind, "severity": severity, "detail": detail}


def _session_issues(symbol: str, days: pd.DataFrame, split_threshold: float, low_coverage: float) -> list[dict]:
    issues = []
    for d, r in days.iterrows():
        issues += [_issue(symbol, d, kind, "error", f"{int(r[kind])} rows") for kind in ERROR_KINDS if r[kind] > 0]
        if r["regular_bars"] == 0:
            issues.append(_issue(symbol, d, "missing_day", "warning", "no regular-session bars"))
        elif r["coverage"] < low_coverage:
            issues.append(_issue(symbol, d, "low_coverage", "warning", f"coverage {r['coverage']:.2f}"))
        if abs(r["overnight_gap"]) > split_threshold:
            gap = f"overnight log gap {r['overnight_gap']:+.3f}"
            issues.append(_issue(symbol, d, "possible_split", "warning", gap))
    return issues


def _summarize(days: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    if days.empty:
        return pd.DataFrame(columns=["symbol", "sessions", "missing_days", "coverage", "morning_coverage", "errors",
                                     "warnings"])
    g = days.groupby("symbol")
    summary = pd.DataFrame(
        {
            "sessions": g.size(),
            "missing_days": g["regular_bars"].apply(lambda s: int((s == 0).sum())),
            "coverage": g["regular_bars"].sum() / g["expected"].sum(),
            "morning_coverage": g["morning_bars"].sum() / g["morning_expected"].sum(),
        }
    )
    severity = issues.groupby(["symbol", "severity"]).size().unstack(fill_value=0) if not issues.empty else None
    for level, column in (("error", "errors"), ("warning", "warnings")):
        summary[column] = (
            severity[level].reindex(summary.index, fill_value=0) if severity is not None and level in severity else 0
        )
    return summary.reset_index()
