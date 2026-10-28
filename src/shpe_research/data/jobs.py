"""Bounded-concurrency job runner shared by the downloaders."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

log = logging.getLogger(__name__)


def run_units[U](
    units: Sequence[U],
    fetch: Callable[[U], Any],
    workers: int,
    label: str,
    key: Callable[[U], str] = str,
) -> tuple[dict[U, Any], dict[str, str]]:
    """Run `fetch` on every unit with up to `workers` threads, logging progress and an ETA.

    One failing unit never stops the rest. Returns (results by unit, first error line by unit key).
    """
    results: dict[U, Any] = {}
    failures: dict[str, str] = {}
    if not units:
        return results, failures
    started = time.monotonic()
    total = len(units)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, unit): unit for unit in units}
        for done, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                results[unit] = future.result()
            except Exception as err:  # noqa: BLE001
                failures[key(unit)] = (str(err).splitlines() or [type(err).__name__])[0]
                log.warning("%s: failed %s: %s", label, key(unit), err)
            if done == total or done % max(1, total // 20) == 0:
                elapsed = time.monotonic() - started
                log.info(
                    "%s: %d/%d (%.0f%%), elapsed %.0fs, eta %.0fs",
                    label, done, total, 100 * done / total, elapsed, elapsed / done * (total - done),
                )
    return results, failures
