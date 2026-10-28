"""Process-wide request spacing so every concurrent download shares one Alpaca budget.

The budget belongs to the account, not to a caller (the trading platform's
utils/rateLimiter.ts makes the same choice). Requests are spaced evenly, with
no bursts, which keeps a rolling one-minute window under the limit.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        """Block until this caller may send its request."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
