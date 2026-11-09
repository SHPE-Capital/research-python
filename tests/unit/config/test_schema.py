"""C1h acceptance tests: typed, unknown-key-rejecting configs."""

from __future__ import annotations

import warnings

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from shpe_research.config import REPO_ROOT, ExperimentConfig, StrategySpec, SweepConfig, load_config
from shpe_research.features.base import FEATURE_REGISTRY

BASE = {
    "name": "t",
    "data": {"source": "fixture", "symbols": ["AAA"]},
    "features": [{"type": "past_return", "window": 5}],
    "labels": {"type": "forward_return", "horizon_minutes": 30},
    "model": {"type": "logistic", "params": {"C": 1.0}},
}


def test_existing_configs_validate():
    paths = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
    assert paths
    for path in paths:
        assert isinstance(load_config(path), ExperimentConfig), path


def test_unknown_key_rejected(tmp_path):
    bad = {**BASE, "fetaures": []}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError, match="fetaures"):
        load_config(path)
    with pytest.raises(ValidationError, match=r"data\.symbl"):
        ExperimentConfig.model_validate({**BASE, "data": {"symbl": ["A"]}})


def test_component_params_validated(monkeypatch):
    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        window: int

    class Dummy:
        feature_type = "dummy_c1h"

    Dummy.Params = Params
    monkeypatch.setitem(FEATURE_REGISTRY, "dummy_c1h", Dummy)
    ok = ExperimentConfig.model_validate({**BASE, "features": [{"type": "dummy_c1h", "window": 3}]})
    assert ok.features[0].type == "dummy_c1h"
    with pytest.raises(ValidationError, match="window"):
        ExperimentConfig.model_validate({**BASE, "features": [{"type": "dummy_c1h", "window": "x"}]})


def test_split_alias_warns():
    raw = {k: v for k, v in BASE.items()}
    raw["split"] = {"train": 0.5, "validation": 0.25, "test": 0.25}
    with pytest.warns(DeprecationWarning, match="fit_scheme"):
        cfg = ExperimentConfig.model_validate(raw)
    assert cfg.fit_scheme is not None and cfg.fit_scheme.type == "single"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**raw, "fit_scheme": {"type": "single"}})


def test_portfolio_block():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = ExperimentConfig.model_validate(
            {
                **BASE,
                "portfolio": {
                    "scheme": "top_k", "top_k": 10, "neutralize": ["sector"], "neutralize_proportion": 1.0,
                    "gross": 1.0, "turnover": {"no_trade_band": 0.01, "partial_adjustment": 0.5},
                },
            }
        )
    assert cfg.portfolio.turnover.no_trade_band == 0.01
    with pytest.raises(ValidationError, match="scheme"):
        ExperimentConfig.model_validate({**BASE, "portfolio": {"scheme": "bogus"}})


def test_strategy_and_sweep_kinds(tmp_path):
    spec = {"kind": "strategy", "strategy_type": "orb", "params": {"range_minutes": 15}, "version": 1}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(spec))
    assert isinstance(load_config(p), StrategySpec)
    with pytest.raises(ValidationError, match="exit_polcy"):
        StrategySpec.model_validate({**spec, "exit_polcy": {}})
    sweep = {"kind": "sweep", "base": BASE, "grid": {"seed": [1, 2]}, "metric": "ic", "direction": "maximize"}
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump(sweep))
    assert isinstance(load_config(p), SweepConfig)
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({**sweep, "optuna": {"n_trials": 3}})  # grid and optuna are exclusive
