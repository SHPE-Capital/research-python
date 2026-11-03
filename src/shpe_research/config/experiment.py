"""Typed experiment config (C1h, ADR 0008).

Every model forbids unknown keys, so `fetaures:` fails at load with the dotted
path instead of silently using a default. Nothing here reads data or touches a
clock: it only types and validates. Component blocks are `{type: str, **params}`;
when the registered class declares a pydantic `Params` model the params are
validated against it, otherwise they pass through for the builder.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(BaseModel):
    """A registry component: `type` plus its params, either inline or under `params:`.

    Subclasses name the registry to check against. An unregistered `type` is not an
    error here (the stage builders raise and list the registered keys).
    """

    model_config = ConfigDict(extra="allow")

    type: str
    registry: ClassVar[tuple[str, str] | None] = None  # (module, attribute) of the registry dict

    def params(self) -> dict[str, Any]:
        """The component's params: the `params:` mapping if it is the only extra key, else the inline keys."""
        extra = dict(self.model_extra or {})
        inner = extra.get("params")
        return dict(inner) if isinstance(inner, dict) and set(extra) == {"params"} else extra

    @model_validator(mode="after")
    def _check_params(self) -> ComponentSpec:
        if self.registry is None:
            return self
        module, attr = self.registry
        try:
            cls = getattr(importlib.import_module(module), attr).get(self.type)
        except ImportError:
            return self
        params_model = getattr(cls, "Params", None)
        if isinstance(params_model, type) and issubclass(params_model, BaseModel):
            try:
                params_model.model_validate(self.params())
            except ValidationError as exc:
                detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                raise ValueError(f"invalid params for {self.type!r}: {detail}") from None
        return self


class FeatureSpec(ComponentSpec):
    registry: ClassVar[tuple[str, str] | None] = ("shpe_research.features.base", "FEATURE_REGISTRY")


class ModelSpec(ComponentSpec):
    registry: ClassVar[tuple[str, str] | None] = ("shpe_research.contracts.model", "MODEL_REGISTRY")


class SamplerSpec(ComponentSpec):
    registry: ClassVar[tuple[str, str] | None] = ("shpe_research.pipeline.registry", "SAMPLER_REGISTRY")


class LabelSpec(ComponentSpec):
    registry: ClassVar[tuple[str, str] | None] = ("shpe_research.pipeline.registry", "LABEL_REGISTRY")


class FitSchemeSpec(ComponentSpec):
    registry: ClassVar[tuple[str, str] | None] = ("shpe_research.pipeline.registry", "FIT_SCHEME_REGISTRY")


class ReportSpec(ComponentSpec):
    registry: ClassVar[tuple[str, str] | None] = ("shpe_research.pipeline.registry", "REPORT_SECTION_REGISTRY")


class DataConfig(_Strict):
    """Where bars come from (see configs/experiments/README.md for the per-source keys)."""

    source: Literal["fixture", "parquet", "store", "snapshot"] = "fixture"
    symbols: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    frequency: str = "1min"
    context_symbols: list[str] = Field(default_factory=list)
    # source-specific keys
    path: str | None = None
    role: Literal["all", "dev", "held_out", "context"] | None = None
    root: str | None = None
    timeframe: str | None = None
    feed: str | None = None
    adjustment: str | None = None
    seed: int | None = None
    signal_strength: float | None = None


class TurnoverConfig(_Strict):
    """Turnover control for portfolio rebalances (T9)."""

    no_trade_band: float = Field(0.0, ge=0)
    partial_adjustment: float = Field(1.0, gt=0, le=1)


class PortfolioConfig(_Strict):
    """Portfolio construction block (T9): scores to weights, neutralization and turnover."""

    scheme: Literal["top_k", "rank", "zscore", "vol_scaled"] = "top_k"
    top_k: int | None = Field(None, gt=0)
    neutralize: list[str] = Field(default_factory=list)
    neutralize_proportion: float = Field(1.0, ge=0, le=1)
    gross: float = Field(1.0, gt=0)
    turnover: TurnoverConfig = Field(default_factory=TurnoverConfig)


class ExperimentConfig(_Strict):
    """One experiment: the twelve pipeline stages' settings, validated before any data loads.

    Legacy spellings still load: `label` (for `labels`), `session` (for `session_filter`)
    and `split` (for `fit_scheme`, with a DeprecationWarning).
    """

    kind: Literal["experiment"] = "experiment"
    name: str
    seed: int = 0
    data: DataConfig = Field(default_factory=DataConfig)
    universe: dict[str, Any] | None = None
    session_filter: Literal["regular", "pre", "post", "all"] = "regular"
    features: list[FeatureSpec] = Field(default_factory=list)
    transforms: list[ComponentSpec] = Field(default_factory=list)
    decisions: SamplerSpec | None = None
    labels: LabelSpec | None = None
    fit_scheme: FitSchemeSpec | None = None
    model: ModelSpec
    evaluate: dict[str, Any] | None = None
    report: list[ReportSpec] = Field(default_factory=list)
    portfolio: PortfolioConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for old, new in (("label", "labels"), ("session", "session_filter")):
            if old in data:
                if new in data:
                    raise ValueError(f"give either {old!r} or {new!r}, not both")
                data[new] = data.pop(old)
        if "split" in data:
            if "fit_scheme" in data:
                raise ValueError("give either 'split' or 'fit_scheme', not both")
            warnings.warn(
                "config key 'split' is deprecated; use 'fit_scheme' (with type: single)",
                DeprecationWarning,
                stacklevel=2,
            )
            split = data.pop("split")
            data["fit_scheme"] = {"type": "single", **split} if isinstance(split, dict) else split
        return data
