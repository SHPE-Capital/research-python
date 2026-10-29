"""Model contract: the interface every model implements, and the manifest it ships with.

    fit(dataset) -> self        learn from a Dataset (callers pass the train rows)
    predict(X) -> np.ndarray    one score per row; for classification, the
                                probability of the positive class; for
                                regression, the predicted label value
    save(directory) / load_model(directory)

Any preprocessing a model needs (scaling, imputation) is fitted inside fit()
on the rows it is given, so applying the saved model never sees statistics
from data it was not trained on.

The manifest records everything needed to reproduce or deploy the model:
feature order, label, training window, seeds, parameters, metrics, library
versions, contract version and the research repo commit. The platform reads
the same manifest when it loads an exported model (task M3).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.version import CONTRACT_VERSION

Task = Literal["classification", "regression"]

MANIFEST_FILE = "manifest.json"

_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "pyarrow", "lightgbm", "torch")


def package_versions() -> dict[str, str]:
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@dataclass
class ModelManifest:
    name: str
    model_type: str
    task: Task
    feature_names: list[str]
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    train_start: str | None = None
    train_end: str | None = None
    n_train: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION
    git_commit: str | None = None
    package_versions: dict[str, str] = field(default_factory=package_versions)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def write(self, directory: Path) -> Path:
        path = Path(directory) / MANIFEST_FILE
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @classmethod
    def read(cls, directory: Path) -> ModelManifest:
        return cls(**json.loads((Path(directory) / MANIFEST_FILE).read_text()))


MODEL_REGISTRY: dict[str, type[Model]] = {}


def register_model(cls: type[Model]) -> type[Model]:
    """Class decorator: makes a model loadable by its `model_type` and buildable from configs."""
    if cls.model_type in MODEL_REGISTRY:
        raise ValueError(f"model_type {cls.model_type!r} is already registered")
    MODEL_REGISTRY[cls.model_type] = cls
    return cls


class Model(ABC):
    """Base class for every model. Subclasses set `model_type` and `task`."""

    model_type: ClassVar[str]
    task: ClassVar[Task]

    def __init__(self, name: str | None = None, seed: int = 0, **params: Any):
        self.name = name or self.model_type
        self.seed = seed
        self.params = params
        self.manifest: ModelManifest | None = None

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series, weight: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def _save_state(self, directory: Path) -> None: ...

    @abstractmethod
    def _load_state(self, directory: Path) -> None: ...

    def fit(self, dataset: Dataset) -> Model:
        from shpe_research.config import git_commit

        self._fit(dataset.X, dataset.y, dataset.meta["weight"])
        decisions = dataset.meta["decision_ts"]
        self.manifest = ModelManifest(
            name=self.name,
            model_type=self.model_type,
            task=self.task,
            feature_names=dataset.feature_names,
            label=dataset.label_name,
            params=self.params,
            seeds=[self.seed],
            train_start=decisions.min().isoformat() if len(decisions) else None,
            train_end=decisions.max().isoformat() if len(decisions) else None,
            n_train=len(dataset),
            git_commit=git_commit(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.manifest is None:
            raise RuntimeError(f"{self.name}: predict() before fit() or load")
        missing = [c for c in self.manifest.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: missing features {missing}")
        return np.asarray(self._predict(X[self.manifest.feature_names]), dtype=float)

    def save(self, directory: str | Path) -> Path:
        if self.manifest is None:
            raise RuntimeError(f"{self.name}: save() before fit()")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_state(directory)
        self.manifest.write(directory)
        return directory


def build_model(model_type: str, **params: Any) -> Model:
    """A fresh, unfitted model of a registered type (as named in experiment configs)."""
    import shpe_research.models  # noqa: F401  (importing the package registers the built-in models)

    try:
        cls = MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(f"unknown model type {model_type!r}; registered: {sorted(MODEL_REGISTRY)}") from None
    return cls(**params)


def load_model(directory: str | Path) -> Model:
    """A saved model, ready to predict."""
    manifest = ModelManifest.read(Path(directory))
    model = build_model(manifest.model_type, name=manifest.name, seed=manifest.seeds[0] if manifest.seeds else 0,
                        **manifest.params)
    model._load_state(Path(directory))
    model.manifest = manifest
    return model
