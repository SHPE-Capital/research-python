"""Public Windows/Linux command, with no keys/network: exercise forced restart, evidence and tamper detection."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

from shpe_research.calibration.rm9_spec import REPO

CLI = REPO / "scripts/momentum/rm9.py"


def invoke(*args, expected=0):
    env = os.environ.copy()
    env.pop("ALPACA_DATA_KEY", None)
    env.pop("ALPACA_DATA_SECRET", None)
    cmd = [sys.executable, str(CLI), *map(str, args)]
    res = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True, timeout=360)
    assert res.returncode == expected, res.stdout[-1800:] + res.stderr[-2500:]
    return res


def test_laptop_job_resume_matches_uninterrupted_and_detects_damage(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    invoke("demo", "--root", a, "--workers", 1, "--pause-after", 1, expected=75)
    job = a / "job"

    def get(p):
        return json.loads(p.read_text())

    assert get(job / "status.json")["status"] == "PAUSED" and not (job / "report.json").exists()
    seconds = get(job / "clock.json")["active_seconds"]
    invoke("demo", "--root", a)
    assert get(job / "clock.json")["active_seconds"] > seconds
    invoke("demo", "--root", b, "--archive", a / "archive", "--workers", 1)
    r = get(job / "report.json")
    assert r["scientific_digest"] == get(b / "job" / "report.json")["scientific_digest"]
    assert (job / "trials.csv").read_bytes() == (b / "job" / "trials.csv").read_bytes()
    assert r["actual_reference_config_weeks"] == 286 and r["policy_week_decisions"] == 15
    assert r["trial_count"] == 26 and sum(v for k, v in r["origins"].items() if "surrogate" in k) > 0
    assert len(r["periods"]) == 2 and r["periods"]["primary"]["uniform"]["cost_dollars"] > 0
    with sqlite3.connect(job / "archive.sqlite3") as db:
        decisions = [json.loads(x[0]) for x in db.execute("select body from decision order by cut")]
    assert len({v["archive_snapshot"] for v in decisions[-1].values()}) == 1
    assert all(len(s) == 3 for s in decisions)
    invoke("verify", "--mode", "demo", "--root", a, "--deep")
    frozen = job / "weekly-configs" / "0007-uniform.json"
    frozen.write_text("{}")
    assert "integrity" in invoke("verify", "--mode", "demo", "--root", a, expected=1).stderr
