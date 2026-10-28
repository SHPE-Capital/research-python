"""Alpaca market-data REST client: bars, quotes and trades for one symbol and window.

Mirrors the trading platform's adapters (backend/src/adapters/alpaca/rest.ts,
historicalData.ts and core/backtest/backtestLoader.ts):

    - every page is followed through next_page_token
    - Alpaca's `end` is inclusive; records at or after `end` are dropped so
      windows are half-open [start, end) and adjacent windows never overlap
    - 429 and 5xx (and network errors) are retried, honoring Retry-After,
      otherwise with exponential backoff; other errors are raised with
      Alpaca's message
    - all requests share one process-wide RateLimiter

Records come back raw (Alpaca's field names, RFC-3339 nanosecond timestamps)
so callers normalize them into the contracts.

Free plan: SIP history excludes the most recent 15 minutes (requests reaching
into it get 403). Callers clip windows with `sip_safe_end`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from shpe_research.config import requests_per_minute
from shpe_research.data.ratelimit import RateLimiter

log = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets/v2"
PAGE_LIMIT = 10_000
SIP_DELAY = pd.Timedelta(minutes=16)
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class AlpacaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Credentials:
    key: str
    secret: str

    @classmethod
    def from_env(cls) -> Credentials:
        key = os.environ.get("ALPACA_DATA_KEY", "").strip()
        secret = os.environ.get("ALPACA_DATA_SECRET", "").strip()
        if not key or not secret:
            raise AlpacaError(
                "ALPACA_DATA_KEY and ALPACA_DATA_SECRET are not set. Copy .env.example to .env and add your "
                "personal Alpaca keys (a paper account's keys work), or work from fixtures or a snapshot instead."
            )
        return cls(key, secret)


def sip_safe_end(end: pd.Timestamp, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """`end`, pulled back so a SIP request never reaches the last 15 minutes the free plan withholds."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return min(end, now - SIP_DELAY)


_shared_limiter: RateLimiter | None = None
_shared_lock = threading.Lock()


def shared_limiter() -> RateLimiter:
    global _shared_limiter
    with _shared_lock:
        if _shared_limiter is None:
            _shared_limiter = RateLimiter(requests_per_minute())
        return _shared_limiter


class AlpacaDataClient:
    def __init__(
        self,
        credentials: Credentials | None = None,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        base_url: str = DATA_URL,
        timeout: float = 30.0,
        max_attempts: int = 4,
        page_limit: int = PAGE_LIMIT,
        sleep: Callable[[float], None] = time.sleep,
    ):
        creds = credentials or Credentials.from_env()
        self._limiter = limiter or shared_limiter()
        self._max_attempts = max_attempts
        self._page_limit = page_limit
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            headers={"APCA-API-KEY-ID": creds.key, "APCA-API-SECRET-KEY": creds.secret},
            timeout=timeout,
            transport=transport,
        )
        self.requests_sent = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AlpacaDataClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ public

    def bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str = "1Min",
        adjustment: str = "raw",
        feed: str = "sip",
    ) -> list[dict[str, Any]]:
        """Raw bars with start <= t < end, ascending."""
        params = {"timeframe": timeframe, "adjustment": adjustment, "feed": feed}
        return self._records(f"/stocks/{symbol}/bars", "bars", start, end, params)

    def ticks(self, kind: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp, feed: str = "sip") -> list[dict]:
        """Raw quotes or trades (`kind`) with start <= t < end, ascending."""
        if kind not in ("quotes", "trades"):
            raise ValueError("kind must be 'quotes' or 'trades'")
        return self._records(f"/stocks/{symbol}/{kind}", kind, start, end, {"feed": feed})

    # ------------------------------------------------------------ internals

    def _records(self, path: str, key: str, start: pd.Timestamp, end: pd.Timestamp, params: dict) -> list[dict]:
        records: list[dict[str, Any]] = []
        page_token: str | None = None
        base = {**params, "start": _iso(start), "end": _iso(end), "limit": self._page_limit}
        while True:
            page = self._get(path, {**base, **({"page_token": page_token} if page_token else {})})
            records.extend(page.get(key) or [])
            page_token = page.get("next_page_token")
            if not page_token:
                break
        # Alpaca's end is inclusive; records are ascending, so the excess is at the tail.
        while records and pd.Timestamp(records[-1]["t"]) >= end:
            records.pop()
        return records

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(1, self._max_attempts + 1):
            self._limiter.acquire()
            self.requests_sent += 1
            try:
                res = self._http.get(path, params=params)
            except httpx.TransportError as err:
                if attempt == self._max_attempts:
                    raise AlpacaError(f"GET {path} failed after {attempt} attempts: {err}") from err
                self._sleep(0.5 * 2 ** (attempt - 1))
                continue
            if res.status_code == 200:
                return res.json()
            if res.status_code in _RETRY_STATUSES and attempt < self._max_attempts:
                self._sleep(_retry_delay(res, attempt))
                continue
            raise AlpacaError(_explain(path, res), status=res.status_code)
        raise AssertionError("unreachable")


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _retry_delay(res: httpx.Response, attempt: int) -> float:
    try:
        retry_after = float(res.headers.get("retry-after", ""))
    except ValueError:
        retry_after = 0.0
    return retry_after if retry_after > 0 else 0.5 * 2 ** (attempt - 1)


def _explain(path: str, res: httpx.Response) -> str:
    body = res.text[:300]
    if res.status_code == 401:
        return f"Alpaca rejected the keys for GET {path} (401). Check ALPACA_DATA_KEY and ALPACA_DATA_SECRET. {body}"
    if res.status_code == 403:
        return (
            f"Alpaca refused GET {path} (403). On the free plan, SIP data from the last 15 minutes is not "
            f"available and some feeds need a subscription. {body}"
        )
    return f"Alpaca GET {path} failed ({res.status_code}): {body}"
