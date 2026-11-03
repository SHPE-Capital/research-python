"""Pass bands for workshop tests: a metric must sit in ``[low, high]`` for every seed."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Band:
    """An inclusive acceptance interval for one metric, checked across several seeds.

    Bands are set from the spread over several seeds, never from one lucky seed.
    """

    metric: str
    low: float
    high: float
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Band {self.metric!r}: low {self.low} > high {self.high}")
        if not self.seeds:
            raise ValueError(f"Band {self.metric!r}: needs at least one seed")

    def contains(self, value: float) -> bool:
        """True when ``value`` is a number within ``[low, high]`` (NaN is never inside)."""
        return not math.isnan(value) and self.low <= value <= self.high


def assert_in_band(fn: Callable[[int], float], band: Band) -> list[float]:
    """Run ``fn(seed)`` for every seed in ``band`` and require each result inside the band.

    Every seed is evaluated even after a failure. On failure the ``AssertionError``
    lists the value for every seed, so the spread is visible. Returns the values in
    seed order when all pass.
    """
    values = [float(fn(seed)) for seed in band.seeds]
    if all(band.contains(v) for v in values):
        return values
    lines = [
        f"  seed {seed}: {value}  [{'ok' if band.contains(value) else 'OUTSIDE'}]"
        for seed, value in zip(band.seeds, values, strict=True)
    ]
    raise AssertionError(f"{band.metric} not in [{band.low}, {band.high}] for every seed:\n" + "\n".join(lines))
