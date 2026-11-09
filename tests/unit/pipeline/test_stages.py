from __future__ import annotations

from pathlib import Path

import pandas as pd

from shpe_research.pipeline import stages
from tests.conftest import HELLO

ORDER = ["load", "universe", "session_filter", "features", "transforms", "decisions", "labels", "join",
         "fit_scheme", "model", "evaluate", "record"]

SMALL = {"source": "fixture", "symbols": ["AAA", "BBB"], "start": "2026-05-01", "end": "2026-05-29"}


def test_stage_names_documented_order():
    assert list(stages.STAGE_ORDER) == ORDER


def test_stage_order(monkeypatch, tmp_path: Path):
    from shpe_research.experiments.runner import ExperimentConfig

    calls: list[str] = []
    for name in ORDER:
        real = getattr(stages, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(stages, name, spy)
    cfg = ExperimentConfig.from_yaml(HELLO)
    cfg.data = dict(SMALL)
    stages.run_pipeline(cfg, tmp_path)
    assert calls == ORDER


def test_runner_wrapper_compat(tmp_path: Path):
    from shpe_research.contracts.dataset import Dataset
    from shpe_research.experiments import runner

    assert runner.EVAL_SPLITS == ("train", "validation", "test")
    cfg = runner.ExperimentConfig.from_yaml(HELLO)
    cfg.data = dict(SMALL)
    bars = runner.load_bars(cfg.data)
    assert isinstance(runner.build_dataset(cfg, bars), Dataset)
    result = runner.run_experiment(cfg, out_root=tmp_path)
    assert isinstance(result, runner.RunResult)
    assert isinstance(result.run_dir, Path)
    for name in ("manifest.json", "config.yaml", "predictions.parquet", "report.json", "report.md"):
        assert (result.run_dir / name).exists(), name
    assert not (result.run_dir / "ledger.parquet").exists()
    assert not (result.run_dir / "candidates.parquet").exists()


def test_unknown_registry_key_fails_before_data_loads(monkeypatch, tmp_path: Path):
    import pytest

    from shpe_research.experiments.runner import ExperimentConfig

    def boom(*args, **kwargs):
        raise AssertionError("data must not load")

    monkeypatch.setattr(stages, "load", boom)
    cfg = ExperimentConfig.from_yaml(HELLO)
    cfg.label = {"type": "nope", "horizon_minutes": 30}
    with pytest.raises(ValueError, match="forward_return"):
        stages.run_pipeline(cfg, tmp_path)


def test_join_drops_missing_labels_instead_of_filling():
    ts = pd.date_range("2026-05-01T14:00Z", periods=3, freq="15min")
    feats = pd.DataFrame({"symbol": "AAA", "decision_ts": ts, "f": [1.0, 2.0, 3.0]})
    labels = pd.DataFrame({"symbol": "AAA", "decision_ts": ts, "label_end_ts": ts + pd.Timedelta(minutes=30),
                           "lab": [0.1, None, 0.3]})
    out = stages.join(feats, feats[["symbol", "decision_ts"]], labels, ["f"], "lab")
    assert len(out) == 2
