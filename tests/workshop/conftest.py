"""Harness for workshop example tests.

``run_notebook`` runs an example's ``notebook.py`` as a plain script in a scratch copy of
its folder. When a solution module is given (argument or ``SHPE_WORKSHOP_SOLUTIONS``) the
copy's ``exercises.py`` is replaced by it, so CI on the ``workshop-solutions`` branch
checks the full path while ``main`` checks the starter. Notebooks need no code for this.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSHOP_DIR = REPO_ROOT / "workshop"
TIME_BUDGET_SECONDS = 300  # each example test must finish in under 5 minutes on a laptop


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "workshop: a workshop example test (runs on the fixture, offline)")
    config.addinivalue_line("markers", "slow: takes more than 60 seconds")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test under tests/workshop carries the ``workshop`` mark."""
    for item in items:
        if "tests/workshop/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.workshop)


def _solution_file(solutions: Path | str | None) -> Path | None:
    chosen = solutions if solutions is not None else os.environ.get("SHPE_WORKSHOP_SOLUTIONS")
    if not chosen:
        return None
    path = Path(chosen)
    if path.is_dir():
        path = path / "exercises.py"
    if not path.is_file():
        raise FileNotFoundError(f"solution module not found: {path}")
    return path


def run_notebook(
    example: Path | str,
    *,
    solutions: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = TIME_BUDGET_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run ``<example>/notebook.py`` as a script and return the completed process.

    ``example`` is a folder path, or a slug such as ``"01_leakage_lab"`` under ``workshop/``.
    The folder is copied to a scratch directory; if a solution module is given, it replaces
    ``exercises.py`` in the copy (the original is never touched). ``SHPE_WORKSHOP_REAL_DATA``
    is cleared unless ``env`` sets it. The repo root is on ``PYTHONPATH`` so ``workshop._common``
    imports. Raises ``subprocess.TimeoutExpired`` over ``timeout`` seconds.
    """
    folder = Path(example)
    if not folder.is_absolute() and not folder.exists():
        folder = WORKSHOP_DIR / folder
    if not (folder / "notebook.py").is_file():
        raise FileNotFoundError(f"no notebook.py in {folder}")
    solution = _solution_file(solutions)

    child_env = {k: v for k, v in os.environ.items() if k != "SHPE_WORKSHOP_REAL_DATA"}
    child_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), child_env.get("PYTHONPATH")]))
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env.update(env or {})

    with tempfile.TemporaryDirectory(prefix="shpe_workshop_") as scratch:
        work = Path(scratch) / folder.name
        shutil.copytree(folder, work, ignore=shutil.ignore_patterns("__pycache__"))
        if solution is not None:
            shutil.copyfile(solution, work / "exercises.py")
        return subprocess.run(
            [sys.executable, str(work / "notebook.py")],
            cwd=work,
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
