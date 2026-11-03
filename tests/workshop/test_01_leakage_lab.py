"""W1 acceptance tests: the leakage lab (workshop/01_leakage_lab).

The notebook defines the leaky helpers (``CenteredMean``, ``DayZScore``, ``NextReturn``) so the library never ships a
leaky feature. The tests execute the notebook's setup cells (everything before its ``imported by tests`` guard) into a
namespace and use those helpers. Planted effect: the fixture's default short-horizon reversion (``signal_strength``
0.3), which gives a validation AUC just above 0.5 on the clean config; bands below were measured over seeds 0-4.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from shpe_research.features.base import FEATURE_REGISTRY
from shpe_research.testing.checks import check_feature
from tests.workshop.bands import Band, assert_in_band
from tests.workshop.conftest import REPO_ROOT, run_notebook

pytestmark = pytest.mark.workshop

LAB = REPO_ROOT / "workshop" / "01_leakage_lab"
SEEDS = (0, 1, 2)
LEAKS = ("centered_mean", "day_zscore", "next_return")


@pytest.fixture(scope="module")
def lab() -> dict:
    """The notebook's helper namespace (the walkthrough itself is not run)."""
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(LAB))
    sys.modules.pop("exercises", None)
    solution = os.environ.get("SHPE_WORKSHOP_SOLUTIONS")
    if solution:  # the solutions branch's CI: preload the solution module as `exercises`
        solution_path = Path(solution) / "exercises.py" if Path(solution).is_dir() else Path(solution)
        spec = importlib.util.spec_from_file_location("exercises", solution_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["exercises"] = module
        spec.loader.exec_module(module)
    try:
        namespace: dict = {"__name__": "w1_helpers", "__file__": str(LAB / "notebook.py")}
        try:
            source = (LAB / "notebook.py").read_text(encoding="utf-8")
            exec(compile(source, str(LAB / "notebook.py"), "exec"), namespace)
        except SystemExit:  # the notebook stops itself after its helper cells when imported
            pass
        assert "_helpers_only" in namespace, "the notebook must stop after its helpers when imported"
        return namespace
    finally:
        sys.path.remove(str(LAB))
        sys.path.remove(str(REPO_ROOT))


def test_files_exist():
    for rel in ("README.md", "notebook.py", "exercises.py", "configs/clean.yaml", "configs/leaky_feature.yaml"):
        assert (LAB / rel).is_file(), rel


def test_clean_auc_band(lab):
    """Clean validation AUC is slightly above 0.5 on every seed (the planted reversion, no more)."""
    clean = lab["load_config"]("clean")
    assert_in_band(lambda seed: lab["validation_auc"](clean, seed), Band("clean AUC", 0.505, 0.56, SEEDS))


def test_leaks_inflate(lab):
    """Each leaky config's validation AUC exceeds the clean one in every seed."""
    for seed in SEEDS:
        clean = lab["validation_auc"](lab["load_config"]("clean"), seed)
        leaky = {name: lab["validation_auc"](lab["leaky_config"](name), seed) for name in LEAKS}
        for name, auc in leaky.items():
            assert auc > clean, f"seed {seed}: {name} AUC {auc:.4f} does not exceed clean {clean:.4f} ({leaky})"
        assert leaky["next_return"] > 0.7, f"seed {seed}: the label-in-features leak should be glaring: {leaky}"


def test_leaky_yaml_swaps_in_the_helper(lab):
    clean, leaky = lab["load_config"]("clean"), lab["load_config"]("leaky_feature")
    assert [f for f in leaky.features if f not in clean.features] == [{"type": "centered_mean", "window": 11}]
    assert "centered_mean" in FEATURE_REGISTRY


@pytest.mark.parametrize("name", LEAKS)
def test_checks_catch_leaks(lab, name):
    """check_feature (and so assert_no_lookahead) fails on every leaky helper."""
    feature = lab["make_feature"](name, fixed=False)
    with pytest.raises(AssertionError, match="future data|uses future"):
        check_feature(feature)


@pytest.mark.parametrize("name", LEAKS)
def test_fix_passes(lab, name):
    """The fixed versions pass check_feature. DayZScore's fix is an exercise: skipped while it is pending."""
    try:
        feature = lab["make_feature"](name, fixed=True)
        check_feature(feature)
    except NotImplementedError:
        pytest.skip(f"{name}: the fix is an exercise and is still pending in this starter")


def test_survivorship_bias(lab):
    """Dropping the worst 20% of names over the window makes buy-the-losers look better (fixed seed)."""
    result = lab["survivorship_demo"](seed=7, n_symbols=60)
    assert result["survivors"]["gross"] > result["all"]["gross"]
    assert result["survivors"]["net"] > result["all"]["net"]
    assert result["n_survivors"] == 48


def test_snooping_grows_with_trials(lab):
    table = lab["snooping_table"](seed=7)
    assert len(table) == 124
    try:
        gains = {n: lab["selection_gain"](table, n) for n in (10, 100)}
    except NotImplementedError:
        pytest.skip("exercise 3 is pending in this starter")
    assert gains[100] > gains[10] > 0


def test_notebook_runs_top_to_bottom():
    """CI check: the starter notebook runs on the fixture, offline, inside the time budget."""
    done = run_notebook("01_leakage_lab", timeout=290)
    assert done.returncode == 0, done.stderr[-3000:]
    if "SHPE_WORKSHOP_SOLUTIONS" not in os.environ:
        assert "EXERCISE pending" in done.stdout, "the starter should say which exercises are pending"
    else:
        assert "EXERCISE pending" not in done.stdout, "a solutions run should have no pending exercise"
    assert "trials counted: 40" in done.stdout
    assert "real-data cell skipped" in done.stdout


def test_exercises_are_stubs_on_main():
    text = (LAB / "exercises.py").read_text(encoding="utf-8")
    if "SHPE_WORKSHOP_SOLUTIONS" in os.environ:
        pytest.skip("solutions run")
    assert text.count("# EXERCISE") >= 3 and "NotImplementedError" in text
    assert not Path(LAB / "solutions.py").exists()
