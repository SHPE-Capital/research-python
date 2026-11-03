"""Acceptance tests for the C1j `shpe new` scaffolding framework."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from shpe_research.cli import main
from shpe_research.scaffold.new import ScaffoldError, list_kinds, scaffold


def _import(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_feature_scaffold_passes_check(tmp_path):
    from shpe_research.features.base import FEATURE_REGISTRY
    from shpe_research.testing.checks import check_feature

    written = scaffold("feature", "returns/scaffold_demo", root=tmp_path)
    module_path = tmp_path / "src/shpe_research/features/returns/scaffold_demo.py"
    assert module_path in written
    assert (tmp_path / "tests/unit/features/returns/test_scaffold_demo.py").exists()
    module = _import(module_path, "_scaffolded_feature")
    try:
        check_feature(module.ScaffoldDemo())
    finally:
        FEATURE_REGISTRY.pop("scaffold_demo", None)


def test_model_scaffold_passes_check(tmp_path):
    from shpe_research.contracts.model import MODEL_REGISTRY
    from shpe_research.testing.checks import check_model
    from shpe_research.testing.fixtures import synthetic_xy

    scaffold("model", "scaffold_demo", root=tmp_path)
    module_path = tmp_path / "src/shpe_research/models/scaffold_demo.py"
    assert (tmp_path / "tests/unit/models/test_scaffold_demo.py").exists()
    module = _import(module_path, "_scaffolded_model")
    try:
        X, y = synthetic_xy("classification")
        check_model(module.ScaffoldDemoModel(name="demo", seed=7), X, y)
    finally:
        MODEL_REGISTRY.pop("scaffold_demo", None)


def test_study_scaffold_layout(tmp_path):
    scaffold("study", "momentum/r9-demo-study", root=tmp_path)
    folder = tmp_path / "studies/momentum/r9-demo-study"
    for name in ("README.md", "config.yaml", "notes.md"):
        assert (folder / name).is_file()
    assert "{{" not in (folder / "README.md").read_text()
    assert "r9" in (folder / "README.md").read_text().lower()


def test_refuses_overwrite(tmp_path, capsys):
    args = ["new", "study", "reversals/r9-demo", "--root", str(tmp_path)]
    assert main(args) == 0
    target = tmp_path / "studies/reversals/r9-demo/notes.md"
    target.write_text("edited")
    assert main(args) != 0
    assert target.read_text() == "edited"
    with pytest.raises(ScaffoldError):
        scaffold("study", "reversals/r9-demo", root=tmp_path)


def test_unknown_kind_lists_kinds(tmp_path, capsys):
    assert main(["new", "widget", "x", "--root", str(tmp_path)]) != 0
    err = capsys.readouterr().err
    for kind in ("feature", "model", "study"):
        assert kind in err
    with pytest.raises(ScaffoldError, match="feature"):
        scaffold("widget", "x", root=tmp_path)
    assert {"feature", "model", "study"} <= set(list_kinds())


def test_rejects_bad_subpackage_and_team(tmp_path):
    with pytest.raises(ScaffoldError, match="subpackage"):
        scaffold("feature", "nonsense/demo", root=tmp_path)
    with pytest.raises(ScaffoldError, match="team"):
        scaffold("study", "quant/r9-demo", root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_dry_run_prints_paths_and_writes_nothing(tmp_path, capsys):
    assert main(["new", "feature", "returns/demo", "--dry-run", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "features" in out and "demo.py" in out
    assert not list(tmp_path.iterdir())
