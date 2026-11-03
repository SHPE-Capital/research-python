"""`StrategySpec`: a frozen, validated strategy definition (C1h; T7 exports it, T8 fills in behaviour).

Field names are snake_case versions of the trading platform's config names so that
`export_platform_config` never has to rename or drop one. The nested trade blocks
are mappings here; T1, T2, T4 and T5 own the types they are built into.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrategySpec(BaseModel):
    """One strategy configuration: entry rule params plus the trade-layer blocks.

    Nothing is evaluated here, so there is no decision-time data; the spec only names parameters.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["strategy"] = "strategy"
    strategy_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    filters: list[dict[str, Any]] = Field(default_factory=list)
    exit_policy: dict[str, Any] = Field(default_factory=dict)
    sizing: dict[str, Any] = Field(default_factory=dict)
    risk_limits: dict[str, Any] = Field(default_factory=dict)
    cost_model: dict[str, Any] = Field(default_factory=dict)
    fill_model: dict[str, Any] = Field(default_factory=dict)
    universe: dict[str, Any] | list[str] | None = None
    version: int = Field(1, ge=1)
