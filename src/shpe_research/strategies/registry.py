"""Strategy registry: same pattern as features/base.py."""

from __future__ import annotations

from typing import Any

STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(cls: type) -> type:
    if cls.strategy_type in STRATEGY_REGISTRY:
        raise ValueError(f"strategy_type {cls.strategy_type!r} is already registered")
    STRATEGY_REGISTRY[cls.strategy_type] = cls
    return cls


def build_strategy(spec: dict[str, Any]) -> Any:
    """A strategy from a config entry like {"type": "orb", ...}."""
    import shpe_research.strategies  # noqa: F401  (importing the package registers the built-ins)

    params = dict(spec)
    key = params.pop("type")
    try:
        cls = STRATEGY_REGISTRY[key]
    except KeyError:
        raise ValueError(f"unknown strategy {key!r}; registered: {sorted(STRATEGY_REGISTRY)}") from None
    return cls(**params)
