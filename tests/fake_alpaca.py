"""An in-memory stand-in for Alpaca's market-data REST API, served through httpx.MockTransport."""

from __future__ import annotations

import json
from collections import deque
from urllib.parse import parse_qs

import httpx
import pandas as pd

from shpe_research.data.alpaca import AlpacaDataClient, Credentials
from shpe_research.data.calendar import session_date
from shpe_research.data.ratelimit import RateLimiter


def _iso(ts: pd.Series) -> list[str]:
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.rstrip("0").str.rstrip(".").add("Z").tolist()


def raw_bars(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"t": _iso(bars["ts"]), "o": bars["open"], "h": bars["high"], "l": bars["low"], "c": bars["close"],
         "v": bars["volume"], "vw": bars["vwap"], "n": bars["trade_count"].astype(int), "_ts": bars["ts"]}
    )


def raw_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily bars aggregated from minute bars, stamped at New York midnight like Alpaca's."""
    day = session_date(bars["ts"])
    g = bars.groupby(day)
    daily = pd.DataFrame({"o": g["open"].first(), "h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last(),
                          "v": g["volume"].sum(), "vw": g["vwap"].mean(), "n": g["trade_count"].sum().astype(int)})
    ts = pd.Series(daily.index.tz_localize("America/New_York").tz_convert("UTC")).astype("datetime64[ns, UTC]")
    daily = daily.reset_index(drop=True)
    daily.insert(0, "t", _iso(ts))
    daily["_ts"] = ts
    return daily


class FakeAlpaca:
    """Serves /v2/stocks/{symbol}/{bars|quotes|trades} from frames keyed by (kind, timeframe, symbol).

    Like Alpaca, `end` is inclusive and pages are linked with next_page_token.
    `script` holds status codes to return (in order) before serving normally.
    """

    def __init__(self):
        self.data: dict[tuple[str, str, str], pd.DataFrame] = {}
        self.requests: list[tuple[str, dict]] = []
        self.script: deque[int] = deque()

    def add(self, kind: str, symbol: str, raw: pd.DataFrame, timeframe: str = "") -> None:
        self.data[(kind, timeframe, symbol)] = raw.sort_values("_ts").reset_index(drop=True)

    def client(self, page_limit: int = 500) -> AlpacaDataClient:
        return AlpacaDataClient(
            Credentials("key", "secret"), limiter=RateLimiter(10**9), transport=httpx.MockTransport(self.handle),
            page_limit=page_limit, sleep=lambda s: None,
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        self.requests.append((request.url.path, params))
        if self.script:
            return httpx.Response(self.script.popleft(), text="scripted failure")
        _, _, _, symbol, kind = request.url.path.split("/")
        frame = self.data.get((kind, params.get("timeframe", ""), symbol))
        if frame is None:
            return httpx.Response(200, json={kind: None, "next_page_token": None})
        start, end = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
        rows = frame[(frame["_ts"] >= start) & (frame["_ts"] <= end)]
        offset, limit = int(params.get("page_token", 0)), int(params["limit"])
        page = rows.iloc[offset : offset + limit].drop(columns="_ts")
        token = str(offset + limit) if offset + limit < len(rows) else None
        body = {kind: json.loads(page.to_json(orient="records")), "next_page_token": token}
        return httpx.Response(200, json=body)
