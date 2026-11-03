"""D6: corporate actions (splits, symbol changes) from Alpaca, stored beside the raw bars.

D2's audit flags overnight gaps that look like splits; this module confirms them.

    fetch_corporate_actions   GET https://data.alpaca.markets/v1/corporate-actions for forward_split,
                              reverse_split and name_change, follow next_page_token, and store the rows as
                              Parquet under <data_dir>/corporate_actions/<SYMBOL>.parquet.
    load_corporate_actions    read stored rows back.
    split_factor              factor to divide raw prices by so they are comparable across a window.
    explain_split_gaps        relabel the audit's `possible_split` issues as explained / unexplained.

Raw prices in the bar store are never rewritten. Frames have columns
type, symbol, new_symbol, ratio, ex_date (see `COLUMNS`). `ratio` is new shares per old share (4.0 for a
4-for-1 forward split, 0.1 for a 1-for-10 reverse split) and is NaN for name changes. `ex_date` is the first
New York session that trades on the new terms (Alpaca's `ex_date`, or `process_date` for name changes). It is
announced ahead of time, but treat a fetched table as knowledge from the fetch date, not from `ex_date`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from shpe_research.config import data_dir
from shpe_research.data.alpaca import AlpacaDataClient
from shpe_research.data.audit import AuditReport, _summarize

CORPORATE_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
TYPES = ("forward_split", "reverse_split", "name_change")
COLUMNS = ["type", "symbol", "new_symbol", "ratio", "ex_date"]
_KEYS = {"forward_splits": "forward_split", "reverse_splits": "reverse_split", "name_changes": "name_change"}
PAGE_LIMIT = 1000


def _dir(root: str | Path | None) -> Path:
    return (Path(root) if root else data_dir()) / "corporate_actions"


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "type": pd.Series(dtype="object"), "symbol": pd.Series(dtype="object"),
        "new_symbol": pd.Series(dtype="object"), "ratio": pd.Series(dtype="float64"),
        "ex_date": pd.Series(dtype="datetime64[ns]"),
    })


def _normalize(page: dict) -> list[dict]:
    rows = []
    for key, kind in _KEYS.items():
        for r in (page.get("corporate_actions") or {}).get(key) or []:
            if kind == "name_change":
                rows.append({"type": kind, "symbol": r["old_symbol"], "new_symbol": r["new_symbol"],
                             "ratio": np.nan, "ex_date": r["process_date"]})
            else:
                rows.append({"type": kind, "symbol": r["symbol"], "new_symbol": None,
                             "ratio": float(r["new_rate"]) / float(r["old_rate"]),
                             "ex_date": r.get("ex_date") or r["process_date"]})
    return rows


def fetch_corporate_actions(
    symbols: list[str],
    start: str,
    end: str,
    client: AlpacaDataClient | None = None,
    root: str | Path | None = None,
    store: bool = True,
) -> pd.DataFrame:
    """Splits and name changes for `symbols` with ex-dates in [start, end] (dates, inclusive).

    Uses `client` (default: one built from ALPACA_DATA_KEY / ALPACA_DATA_SECRET) and follows every page.
    With `store`, rows are merged into <root or data_dir>/corporate_actions/<SYMBOL>.parquet; every requested
    symbol gets a file even when Alpaca returns nothing, so `split_factor` can tell "no splits" from "never
    fetched". Name changes are filed under the old symbol. Returns the fetched rows, sorted by ex_date.
    """
    own = client is None
    client = client or AlpacaDataClient()
    try:
        params = {"symbols": ",".join(symbols), "types": ",".join(TYPES), "start": str(start), "end": str(end),
                  "limit": PAGE_LIMIT}
        rows: list[dict] = []
        token = None
        while True:
            # The shared client's base URL is /v2; an absolute URL overrides it for this /v1 endpoint.
            page = client._get(CORPORATE_ACTIONS_URL, {**params, **({"page_token": token} if token else {})})
            rows += _normalize(page)
            token = page.get("next_page_token")
            if not token:
                break
    finally:
        if own:
            client.close()
    frame = pd.DataFrame(rows, columns=COLUMNS) if rows else _empty()
    frame["ex_date"] = pd.to_datetime(frame["ex_date"]).astype("datetime64[ns]")
    frame = frame.sort_values(["ex_date", "symbol", "type"]).reset_index(drop=True)
    if store:
        out = _dir(root)
        out.mkdir(parents=True, exist_ok=True)
        for symbol in sorted(set(symbols) | set(frame["symbol"])):
            path = out / f"{symbol}.parquet"
            old = pq.read_table(path).to_pandas() if path.exists() else _empty()
            merged = pd.concat([old, frame[frame["symbol"] == symbol]], ignore_index=True)
            merged = merged.drop_duplicates(["type", "symbol", "ex_date"], keep="last")
            merged = merged.sort_values("ex_date").reset_index(drop=True)
            tmp = path.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), tmp)
            tmp.replace(path)
    return frame


def load_corporate_actions(symbols: list[str], root: str | Path | None = None) -> pd.DataFrame:
    """Stored actions for `symbols`. Raises ValueError naming symbols that were never fetched."""
    folder = _dir(root)
    missing = [s for s in symbols if not (folder / f"{s}.parquet").exists()]
    if missing:
        raise ValueError(f"no stored corporate actions for {missing}; run fetch_corporate_actions first")
    frames = [pq.read_table(folder / f"{s}.parquet").to_pandas() for s in symbols]
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def split_factor(
    symbol: str, start, end, actions: pd.DataFrame | None = None, root: str | Path | None = None
) -> float:
    """Product of the split ratios whose ex-date lies in (start, end] (New York dates); 1.0 if none.

    Divide a raw price observed on `start` by this factor to put it on the same footing as raw prices on
    `end` (multiply volumes by it). A 4-for-1 split between the dates gives 4.0; a 1-for-10 reverse split
    gives 0.1. Name changes are ignored. Reads the stored table unless `actions` is given; raises ValueError
    if `symbol` was never fetched, rather than silently assuming "no splits".
    """
    if actions is None:
        actions = load_corporate_actions([symbol], root)
    s, e = pd.Timestamp(start).tz_localize(None).normalize(), pd.Timestamp(end).tz_localize(None).normalize()
    hit = actions[(actions["symbol"] == symbol) & actions["type"].isin(("forward_split", "reverse_split"))
                  & (actions["ex_date"] > s) & (actions["ex_date"] <= e)]
    return float(np.prod(hit["ratio"])) if len(hit) else 1.0


def explain_split_gaps(report: AuditReport, actions: pd.DataFrame, tolerance: float = 0.15) -> AuditReport:
    """Copy of `report` whose `possible_split` issues say whether a stored split explains them.

    A gap on session `d` is explained when a split of that symbol has ex_date == d and the overnight log gap
    is within `tolerance` of -log(ratio) (the remainder is ordinary overnight movement). Explained issues get
    severity "info" and detail "explained: ..."; others keep severity "warning" with detail "unexplained: ...".
    Summary counts are recomputed, and `to_markdown` shows the details.
    """
    splits = actions[actions["type"].isin(("forward_split", "reverse_split"))]
    issues = report.issues.copy()
    for i, r in issues[issues["kind"] == "possible_split"].iterrows():
        day = pd.Timestamp(r["date"]).tz_localize(None).normalize()
        gap = float(r["detail"].rsplit(" ", 1)[1])
        match = splits[(splits["symbol"] == r["symbol"]) & (splits["ex_date"] == day)
                       & (np.abs(gap + np.log(splits["ratio"])) <= tolerance)]
        if len(match):
            issues.at[i, "severity"] = "info"
            issues.at[i, "detail"] = f"explained: {r['detail']}; split ratio {match['ratio'].iloc[0]:g} ex {day.date()}"
        else:
            issues.at[i, "detail"] = f"unexplained: {r['detail']}; no matching split"
    return AuditReport(days=report.days, symbols=_summarize(report.days, issues), issues=issues)
