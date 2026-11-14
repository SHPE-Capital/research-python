"""Breakout configuration grid (task 0-B): versioned domain, Config, stable config_id.

The domain is data (``GridDomain``). Widening it is a new grid version, never an in-place edit; the
version is part of every config ID. ``DEFAULT_DOMAIN`` is the Appendix A grid (69,732 configs over
14 component subsets): breakout is always active, together with one to three of ADX, EMA trend,
relative volume and VWAP band.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

Number = int | float


@dataclass(frozen=True)
class GridDomain:
    """A versioned grid: per-component parameter domains and the subset rule."""

    version: str
    # component -> ((param, values), ...); order defines the canonical component/param order.
    components: tuple[tuple[str, tuple[tuple[str, tuple[Number, ...]], ...]], ...]
    required: tuple[str, ...] = ("breakout",)
    min_components: int = 2
    max_components: int = 4

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.components)

    def params_of(self, component: str) -> tuple[tuple[str, tuple[Number, ...]], ...]:
        for name, params in self.components:
            if name == component:
                return params
        raise KeyError(component)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> GridDomain:
        """Build from protocol-config data: ``{"version", "components": {c: {p: [..]}}, ...}``."""
        comps = tuple(
            (name, tuple((p, tuple(vals)) for p, vals in params.items())) for name, params in m["components"].items()
        )
        return cls(
            version=str(m["version"]),
            components=comps,
            required=tuple(m.get("required", ("breakout",))),
            min_components=int(m.get("min_components", 2)),
            max_components=int(m.get("max_components", 4)),
        )


DEFAULT_DOMAIN = GridDomain.from_mapping(
    {
        "version": "v1",
        "components": {
            "breakout": {"n": [10, 20, 40, 60], "b": [0, 0.10, 0.25]},
            "adx": {"period": [7, 14, 21], "q": [15, 20, 25, 30]},
            "ema": {"fast": [5, 8, 12], "slow": [20, 35, 60], "d": [0, 0.10, 0.25]},
            "rvol": {"w": [3, 5, 10], "v": [1, 1.5, 2]},
            "vwap": {"e": [1, 2, 3]},
        },
        "required": ["breakout"],
        "min_components": 2,
        "max_components": 4,
    }
)


@dataclass(frozen=True)
class Config:
    """One configuration: active component subset plus parameters (flat ``component.param`` keys).

    ``params`` is a sorted tuple of pairs so equality and hashing ignore dict ordering.
    """

    grid_version: str
    subset: tuple[str, ...]
    params: tuple[tuple[str, Number], ...]

    @classmethod
    def make(cls, grid_version: str, subset: Sequence[str], params: Mapping[str, Number]) -> Config:
        return cls(grid_version, tuple(sorted(subset)), tuple(sorted(params.items())))

    def as_dict(self) -> dict[str, Number]:
        return dict(self.params)


def config_id(config: Config) -> str:
    """Stable 16-hex-char ID: canonical JSON (sorted keys) of version, subset and params, SHA-256."""
    payload = {
        "v": config.grid_version,
        "subset": sorted(config.subset),
        "params": {k: _canon(v) for k, v in sorted(config.params)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _canon(v: Number) -> Number:
    """Collapse 1.0 to 1 so an int and an equal float hash identically."""
    return int(v) if float(v) == int(v) else float(v)


def _subsets(domain: GridDomain) -> Iterator[tuple[str, ...]]:
    names = domain.component_names
    optional = [n for n in names if n not in domain.required]
    for k in range(domain.min_components, domain.max_components + 1):
        extra = k - len(domain.required)
        if extra < 0 or extra > len(optional):
            continue
        for chosen in combinations(optional, extra):
            yield tuple(sorted((*domain.required, *chosen)))


def enumerate_configs(domain: GridDomain = DEFAULT_DOMAIN) -> Iterator[Config]:
    """Yield every valid config, subsets in canonical order, parameters in domain order."""
    for subset in _subsets(domain):
        keys: list[str] = []
        value_lists: list[tuple[Number, ...]] = []
        for comp in domain.component_names:
            if comp in subset:
                for p, vals in domain.params_of(comp):
                    keys.append(f"{comp}.{p}")
                    value_lists.append(vals)
        for combo in product(*value_lists):
            cfg = Config.make(domain.version, subset, dict(zip(keys, combo, strict=True)))
            if is_valid(cfg, domain):
                yield cfg


def is_valid(config: Config, domain: GridDomain = DEFAULT_DOMAIN) -> bool:
    """True if the config belongs to ``domain``: version, subset rule, exact params, values, EMA slow > fast."""
    if config.grid_version != domain.version:
        return False
    subset = set(config.subset)
    if len(subset) != len(config.subset) or not subset <= set(domain.component_names):
        return False
    if not set(domain.required) <= subset or not domain.min_components <= len(subset) <= domain.max_components:
        return False
    params = config.as_dict()
    expected: dict[str, tuple[Number, ...]] = {}
    for comp in domain.component_names:
        if comp in subset:
            for p, vals in domain.params_of(comp):
                expected[f"{comp}.{p}"] = vals
    if len(params) != len(config.params) or set(params) != set(expected):
        return False
    if any(params[k] not in expected[k] for k in expected):
        return False
    return not ("ema" in subset and params["ema.slow"] <= params["ema.fast"])


INACTIVE = -1.0


def feature_vector(config: Config, domain: GridDomain = DEFAULT_DOMAIN) -> list[float]:
    """Fixed-length encoding: per component an active flag, then its params (``INACTIVE`` if off).

    Domain values are all >= 0, so -1 is an unambiguous inactive marker.
    """
    params = config.as_dict()
    vec: list[float] = []
    for comp in domain.component_names:
        active = comp in config.subset
        vec.append(1.0 if active else 0.0)
        for p, _ in domain.params_of(comp):
            vec.append(float(params[f"{comp}.{p}"]) if active else INACTIVE)
    return vec


def from_feature_vector(vec: Sequence[float], domain: GridDomain = DEFAULT_DOMAIN) -> Config:
    """Inverse of ``feature_vector``."""
    i = 0
    subset: list[str] = []
    params: dict[str, Number] = {}
    for comp in domain.component_names:
        active = vec[i] == 1.0
        i += 1
        if active:
            subset.append(comp)
        for p, vals in domain.params_of(comp):
            if active:
                params[f"{comp}.{p}"] = next(v for v in vals if float(v) == vec[i])
            i += 1
    if i != len(vec):
        raise ValueError(f"vector length {len(vec)} != expected {i}")
    return Config.make(domain.version, subset, params)
