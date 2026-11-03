"""The twelve pipeline stages, in order (design/pipeline.md).

    load -> universe -> session_filter -> features -> transforms -> decisions
         -> labels -> join -> fit_scheme -> model -> evaluate -> record

Each stage takes and returns plain DataFrames or the Dataset contract.
`run_pipeline` resolves every registry key first, so a typo fails before any
data loads, then calls the stages in order. No stage reads data from a later
stage: labels run after decisions, and join drops rows with a missing label
rather than filling them.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from shpe_research.config import runs_dir
from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import Model, build_model
from shpe_research.contracts.version import CONTRACT_VERSION
from shpe_research.data.calendar import session_date
from shpe_research.features.base import Feature, build_feature, compute_features
from shpe_research.pipeline.registry import (
    build_fit_scheme,
    build_label,
    build_report_section,
    build_sampler,
)

log = logging.getLogger(__name__)

STAGE_ORDER = (
    "load", "universe", "session_filter", "features", "transforms", "decisions",
    "labels", "join", "fit_scheme", "model", "evaluate", "record",
)  # fmt: skip
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


@dataclass
class Plan:
    """Every registry object a run needs, built before any data loads."""

    features: list[Feature]
    sampler: Any
    label: Any
    target: str
    scheme: Any
    sections: list[Any]
    keys: dict[str, Any]


def plan(cfg: ExperimentConfig) -> Plan:
    """Resolve the config's registry keys; raises ValueError listing the valid keys when one is unknown."""
    label_spec = dict(cfg.label)
    label_spec.setdefault("type", "forward_return")
    target = label_spec.pop("target", "sign")
    sample_every = int(label_spec.pop("sample_every_minutes", 1))
    sampler_spec = {"type": "every_n_minutes", "minutes": sample_every}
    scheme_spec = {"type": "single", "train": 0.6, "validation": 0.2, "test": 0.2, "gap_days": 0, **cfg.split}
    section_specs = [{"type": "signal"}]
    return Plan(
        features=[build_feature(spec) for spec in cfg.features],
        sampler=build_sampler(dict(sampler_spec)),
        label=build_label(dict(label_spec)),
        target=target,
        scheme=build_fit_scheme(dict(scheme_spec)),
        sections=[build_report_section(dict(spec)) for spec in section_specs],
        keys={
            "decisions": sampler_spec,
            "labels": {**label_spec, "target": target},
            "fit_scheme": scheme_spec,
            "evaluate": section_specs,
        },
    )


def load(data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Bars for the config's `data` section."""
    from shpe_research.experiments.sources import load_bars

    bars = load_bars(data_cfg)
    log.info("loaded %d bars for %d symbols", len(bars), bars["symbol"].nunique())
    return bars


def universe(bars: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """The symbols to research. C1a keeps every loaded symbol; the dev/held-out split lives in the source."""
    return bars


def session_filter(bars: pd.DataFrame, session: str) -> pd.DataFrame:
    """Bars of one session ("regular", ...) or all of them."""
    return bars if session == "all" else bars[bars["session"] == session]


def features(bars: pd.DataFrame, feature_list: list[Feature]) -> pd.DataFrame:
    """symbol, decision_ts (= the bar's ts_available) and one column per feature."""
    return compute_features(bars, feature_list).rename(columns={"ts_available": "decision_ts"}).drop(columns="ts")


def transforms(feats: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional transforms of the feature frame (identity until C1d)."""
    return feats


def decisions(feats: pd.DataFrame, sampler: Any) -> pd.DataFrame:
    """The candidate decisions (symbol, decision_ts) the sampler selects from the feature rows."""
    return feats.loc[sampler.mask(feats["decision_ts"]), ["symbol", "decision_ts"]]


def labels(bars: pd.DataFrame, label: Any) -> pd.DataFrame:
    """symbol, decision_ts, label_end_ts and the label column, computed from the bars alone."""
    return label.compute(bars)


def join(
    feats: pd.DataFrame,
    candidates: pd.DataFrame,
    label_frame: pd.DataFrame,
    feature_names: list[str],
    label_name: str,
) -> pd.DataFrame:
    """Features and labels at each candidate decision; rows with a missing value are dropped, never filled."""
    keys = ["symbol", "decision_ts"]
    rows = candidates[keys].merge(feats, on=keys, how="inner", validate="one_to_one")
    rows = rows.merge(label_frame, on=keys, how="inner", validate="one_to_one")
    rows = rows.dropna(subset=[*feature_names, label_name]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("no rows left after joining features and labels; check dates, session and horizon")
    return rows


def fit_scheme(rows: pd.DataFrame, feature_names: list[str], label: Any, target: str, scheme: Any) -> Dataset:
    """The Dataset contract: X, y and meta, with each row assigned to a split and an era."""
    outcome = rows[label.name]
    y = (outcome > 0).astype(int) if target == "sign" else outcome
    meta = rows[["symbol", "decision_ts", "label_end_ts"]].assign(weight=1.0, outcome=outcome)
    meta["split"] = scheme.assign(rows["decision_ts"])
    meta["era"] = session_date(rows["decision_ts"]).dt.strftime("%Y-%m-%d")
    label_name = f"{label.name}_sign" if target == "sign" else label.name
    dataset = Dataset(X=rows[feature_names], y=y.rename(label_name), meta=meta, label_name=label_name)
    log.info("dataset: %d rows; splits %s", len(dataset), dataset.meta["split"].value_counts().to_dict())
    return dataset


def model(dataset: Dataset, model_cfg: dict[str, Any], seed: int) -> Model:
    """The configured model, fitted on the train split."""
    spec = dict(model_cfg)
    fitted = build_model(spec.pop("type"), seed=seed, **spec.get("params", {}))
    fitted.fit(dataset.subset("train"))
    return fitted


def evaluate(
    dataset: Dataset, fitted: Model, sections: list[Any]
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, pd.DataFrame]]:
    """Predictions for every row, then each report section's metrics and decile tables."""
    predictions = dataset.meta.assign(y=dataset.y, score=fitted.predict(dataset.X))
    metrics: dict[str, dict] = {}
    deciles: dict[str, pd.DataFrame] = {}
    for section in sections:
        section_metrics, section_deciles = section.evaluate(predictions, fitted.task)
        metrics.update(section_metrics)
        deciles.update(section_deciles)
    return predictions, metrics, deciles


def record(
    run_dir: Path,
    cfg: ExperimentConfig,
    resolved: Plan,
    dataset: Dataset,
    fitted: Model,
    predictions: pd.DataFrame,
    metrics: dict[str, dict],
    deciles: dict[str, pd.DataFrame],
) -> RunResult:
    """Write the run directory: config, manifest, model, predictions and report."""
    from shpe_research.evaluation.report import write_report

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    # Test metrics stay out of the model manifest so tuning against them is visible in the report only.
    fitted.manifest.metrics = {k: v for k, v in metrics.items() if k != "test"}
    fitted.save(run_dir / "model")
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    write_report(run_dir, cfg.name, metrics, deciles)
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(run_dir, cfg, resolved, dataset, metrics), indent=2, default=str)
    )
    log.info("run written to %s", run_dir)
    return RunResult(run_dir=run_dir, metrics=metrics, model=fitted)


def _manifest(
    run_dir: Path, cfg: ExperimentConfig, resolved: Plan, dataset: Dataset, metrics: dict[str, dict]
) -> dict[str, Any]:
    meta = dataset.meta
    fold: dict[str, Any] = {"fold": 0, "embargo_days": int(resolved.keys["fit_scheme"].get("gap_days", 0))}
    for name in EVAL_SPLITS:
        part = meta.loc[meta["split"] == name, "era"]
        fold[name] = [part.min(), part.max()] if len(part) else None
    return {
        "run_id": run_dir.name,
        "name": cfg.name,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "contract_version": CONTRACT_VERSION,
        "seed": cfg.seed,
        "snapshot_checksum": _snapshot_checksum(cfg.data),
        "registry_keys": {
            "features": cfg.features,
            "model": cfg.model,
            "session": cfg.session,
            **resolved.keys,
        },
        "folds": [fold],
        "trial_number": None,
        "test_scored": "test" in metrics,
    }


def _snapshot_checksum(data_cfg: dict[str, Any]) -> str | None:
    if data_cfg.get("source") != "snapshot":
        return None
    from shpe_research.data.snapshot import load_snapshot

    return load_snapshot(data_cfg["path"]).fingerprint()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def run_pipeline(cfg: ExperimentConfig, out_root: str | Path | None = None) -> RunResult:
    """One config -> one recorded run, through the twelve stages in order."""
    resolved = plan(cfg)  # unknown registry keys fail here, before any data loads
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root or runs_dir()) / f"{stamp}-{cfg.name}"
    feature_names = [f.name for f in resolved.features]

    bars = load(cfg.data)
    bars = universe(bars, cfg)
    bars = session_filter(bars, cfg.session)
    feature_frame = features(bars, resolved.features)
    feature_frame = transforms(feature_frame)
    candidates = decisions(feature_frame, resolved.sampler)
    label_frame = labels(bars, resolved.label)
    rows = join(feature_frame, candidates, label_frame, feature_names, resolved.label.name)
    dataset = fit_scheme(rows, feature_names, resolved.label, resolved.target, resolved.scheme)
    fitted = model(dataset, cfg.model, cfg.seed)
    predictions, metrics, deciles = evaluate(dataset, fitted, resolved.sections)
    return record(run_dir, cfg, resolved, dataset, fitted, predictions, metrics, deciles)

