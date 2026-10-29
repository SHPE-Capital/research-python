"""One config file -> one recorded run (C0; M2 adds sweeps and an experiment registry).

    bars -> session filter -> features -> label -> sampled decisions -> Dataset
         -> chronological split -> fit on train -> score validation and test
         -> runs/<timestamp>-<name>/{config.yaml, model/, predictions.parquet, report.json, report.md}

See configs/experiments/hello_model.yaml for the config shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from shpe_research.config import runs_dir
from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import Model, build_model
from shpe_research.evaluation.report import decile_table, evaluate, write_report
from shpe_research.experiments.sources import load_bars
from shpe_research.features.base import build_feature, compute_features
from shpe_research.features.returns import minutes_since_open
from shpe_research.labels.forward_return import ForwardReturn, compute_labels
from shpe_research.validation.splits import split_by_day

log = logging.getLogger(__name__)

EVAL_SPLITS = ("train", "validation", "test")


@dataclass
class ExperimentConfig:
    name: str
    data: dict[str, Any]
    features: list[dict[str, Any]]
    label: dict[str, Any]
    model: dict[str, Any]
    split: dict[str, Any] = field(default_factory=dict)
    session: str = "regular"
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RunResult:
    run_dir: Path
    metrics: dict[str, dict[str, Any]]
    model: Model


def build_dataset(cfg: ExperimentConfig, bars: pd.DataFrame) -> Dataset:
    """Features and labels joined at each sampled decision, split chronologically by session."""
    if cfg.session != "all":
        bars = bars[bars["session"] == cfg.session]
    features = [build_feature(spec) for spec in cfg.features]
    feats = compute_features(bars, features).rename(columns={"ts_available": "decision_ts"}).drop(columns="ts")

    label_cfg = dict(cfg.label)
    if label_cfg.pop("type", "forward_return") != "forward_return":
        raise ValueError("C0 supports only the forward_return label; exit-policy labels are task L2")
    label = ForwardReturn(int(label_cfg["horizon_minutes"]))
    labels = compute_labels(bars, label)

    rows = feats.merge(labels, on=["symbol", "decision_ts"], how="inner", validate="one_to_one")
    every = int(label_cfg.get("sample_every_minutes", 1))
    rows = rows[(minutes_since_open(rows["decision_ts"]) % every) == 0]
    feature_names = [f.name for f in features]
    rows = rows.dropna(subset=[*feature_names, label.name]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("no rows left after joining features and labels; check dates, session and horizon")

    target = label_cfg.get("target", "sign")
    y = (rows[label.name] > 0).astype(int) if target == "sign" else rows[label.name]
    meta = rows[["symbol", "decision_ts", "label_end_ts"]].assign(weight=1.0, outcome=rows[label.name])
    split_cfg = {"train": 0.6, "validation": 0.2, "test": 0.2, "gap_days": 0, **cfg.split}
    meta["split"] = split_by_day(rows["decision_ts"], **split_cfg)
    label_name = f"{label.name}_sign" if target == "sign" else label.name
    return Dataset(X=rows[feature_names], y=y.rename(label_name), meta=meta, label_name=label_name)


def run_experiment(cfg: ExperimentConfig, out_root: str | Path | None = None) -> RunResult:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root or runs_dir()) / f"{stamp}-{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    bars = load_bars(cfg.data)
    log.info("loaded %d bars for %d symbols", len(bars), bars["symbol"].nunique())
    dataset = build_dataset(cfg, bars)
    log.info("dataset: %d rows; splits %s", len(dataset), dataset.meta["split"].value_counts().to_dict())

    model_cfg = dict(cfg.model)
    model = build_model(model_cfg.pop("type"), seed=cfg.seed, **model_cfg.get("params", {}))
    model.fit(dataset.subset("train"))

    predictions = dataset.meta.assign(y=dataset.y, score=model.predict(dataset.X))
    metrics, deciles = {}, {}
    for split in EVAL_SPLITS:
        part = predictions[predictions["split"] == split]
        if part.empty:
            continue
        metrics[split] = evaluate(part, model.task)
        deciles[split] = decile_table(part)
    model.manifest.metrics = {k: v for k, v in metrics.items() if k != "test"}
    model.save(run_dir / "model")
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    write_report(run_dir, cfg.name, metrics, deciles)
    log.info("run written to %s", run_dir)
    return RunResult(run_dir=run_dir, metrics=metrics, model=model)
