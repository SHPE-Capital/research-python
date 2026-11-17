from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from shpe_research.contracts import weekly_config as wc
from shpe_research.strategies.momentum.breakout.grid import Config, config_id, is_valid
from shpe_research.trade.account_types import AccountConfig

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "tests" / "fixtures" / "weekly_config"
SCHEMA = ROOT / "docs" / "contracts" / "weekly_config.v1.schema.json"


def _load(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["selected", "no_deployment"])
def test_fixture_round_trips_byte_identical(name):
    text = (FIX / f"{name}.json").read_text(encoding="utf-8")
    assert wc.to_json(wc.from_json(text)) == text


def test_selected_config_id_matches_grid():
    sel = wc.from_json((FIX / "selected.json").read_text(encoding="utf-8")).selected
    cfg = Config.make(sel.grid_version, sel.subset, sel.params)
    assert is_valid(cfg)
    assert config_id(cfg) == sel.config_id


def test_no_deployment_fixture_has_no_selection_but_full_account():
    m = wc.from_json((FIX / "no_deployment.json").read_text(encoding="utf-8"))
    assert m.selected is None and m.no_deployment.reason
    assert m.account.exit.max_hold_minutes == 60


def test_account_settings_mirror_account_config():
    acct = _load("selected")["account"]
    expected = dataclasses.asdict(AccountConfig())
    assert acct == expected  # same field names and (default) values, nested exit and fill included


def test_extra_keys_rejected():
    d = _load("selected")
    d["surprise"] = 1
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)
    d = _load("selected")
    d["account"]["exit"]["extra"] = 1
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_schema_version_is_explicit():
    d = _load("selected")
    d["schema_version"] = "weekly_config.v2"
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_account_settings_have_no_defaults():
    d = _load("selected")
    del d["account"]["exit"]
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_exactly_one_outcome():
    d = _load("selected")
    d["no_deployment"] = {"reason": "x"}
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)
    d = _load("selected")
    del d["selected"]
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_params_must_cover_active_components():
    d = _load("selected")
    d["selected"]["subset"] = ["breakout", "rvol"]
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_validation_must_follow_search():
    d = _load("selected")
    d["validation_range"]["start"] = d["search_range"]["end"]
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_decision_cutoff_needs_timezone():
    d = _load("selected")
    d["decision_cutoff"] = "2026-11-20T21:00:00"
    with pytest.raises(ValidationError):
        wc.WeeklyConfig.model_validate(d)


def test_checked_in_schema_is_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_weekly_config_schema.py"), "--check"])
    assert r.returncode == 0, "run: uv run python scripts/export_weekly_config_schema.py"
    assert json.loads(SCHEMA.read_text(encoding="utf-8"))["title"] == "WeeklyConfig"
