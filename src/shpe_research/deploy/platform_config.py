"""Export a frozen StrategySpec to a platform config (C1a stub; T7 implements it)."""

from __future__ import annotations

from typing import Any


def export_platform_config(spec: Any) -> Any:
    """A `PlatformExport` (strategy type and params, BacktestConfig fragment) for `spec`.

    Refuses unsupported strategy types by name.
    """
    raise NotImplementedError("TODO(T7): export_platform_config")
