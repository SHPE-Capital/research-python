"""Import-layer enforcement (C1l): `lint-imports` must pass on the tree and catch upward imports."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_RUNNER = "from importlinter.cli import lint_imports_command as main; main()"


def _lint(cwd: Path, pythonpath: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(pythonpath), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, "--config", str(cwd / "pyproject.toml")],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_lint_imports_passes():
    result = _lint(ROOT, ROOT / "src")
    assert result.returncode == 0, result.stdout + result.stderr


def test_violation_detected(tmp_path):
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(
        ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    bad = tmp_path / "src" / "shpe_research" / "features" / "_bad_upward_import.py"
    bad.write_text("from shpe_research import models  # noqa: F401\n")
    result = _lint(tmp_path, tmp_path / "src")
    assert result.returncode != 0, result.stdout
    assert "_bad_upward_import" in result.stdout


def test_trade_must_not_import_events(tmp_path):
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(
        ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    bad = tmp_path / "src" / "shpe_research" / "trade" / "_bad_events_import.py"
    bad.write_text("from shpe_research import events  # noqa: F401\n")
    result = _lint(tmp_path, tmp_path / "src")
    assert result.returncode != 0, result.stdout
    assert "_bad_events_import" in result.stdout
