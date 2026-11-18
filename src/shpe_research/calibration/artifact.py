"""D7 0-A artifact: build the weekly configuration and write a calibration's run directory.

``build_artifact`` turns one calibration's plan, selection and counters into a ``WeeklyConfig``
(``contracts/weekly_config.py``). The 0-A schema has no field for the resolved protocol, so the run
directory carries it as sidecars (``protocol.resolved.yaml``, ``overrides.json``) linked by
``protocol_hash``; full per-finalist detail goes to ``finalists.json``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from shpe_research.calibration.acceptance import Check, Finalist, Selection
from shpe_research.calibration.window import UniverseWeek, WindowPlan
from shpe_research.config import git_commit
from shpe_research.config.momentum import Policy, ResolvedProtocol, dump_protocol_yaml
from shpe_research.contracts.weekly_config import (
    AcceptanceCheck,
    AcceptanceResults,
    AccountSettings,
    DateRange,
    Evaluations,
    NoDeployment,
    Runtime,
    SelectedConfig,
    UniverseMember,
    WeeklyConfig,
    to_json,
)
from shpe_research.data.calendar import get_calendar
from shpe_research.trade.account_types import AccountConfig

logger = logging.getLogger(__name__)

NO_GIT_COMMIT = "0000000"  # the schema needs >= 7 characters; flagged in the log
_BYTES_PER_MB = 1e6
_NO_FINALISTS = "no_feasible_finalists"


def account_settings(cfg: AccountConfig) -> AccountSettings:
    """Mirror an ``AccountConfig`` field for field (nested exit and fill included) as 0-A settings.

    Input: the live account. Output: ``AccountSettings``. Pure; no timing dependence.
    """
    return AccountSettings.model_validate(dataclasses.asdict(cfg))


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (10^6 bytes).

    Windows: ``GetProcessMemoryInfo`` ``PeakWorkingSetSize`` via ctypes. POSIX: ``resource``
    ``ru_maxrss`` (KiB on Linux, bytes on macOS). Child processes are not included. Diagnostic only
    (a cache-state field of the artifact).
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / _BYTES_PER_MB
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (peak if sys.platform == "darwin" else peak * 1024) / _BYTES_PER_MB


def decision_cutoff_ts(cutoff: date) -> datetime:
    """Return the close of the ``cutoff`` session as a tz-aware UTC datetime (exchange calendar).

    Input: the cutoff session. Output: the instant the last data the calibration may use is known.
    """
    _, close = get_calendar().session_bounds(cutoff)
    return close.tz_convert("UTC").to_pydatetime().astimezone(UTC)


def _finite_or_none(value: float | int | None) -> float | None:
    """Float value, or None for None/NaN/inf (JSON has no NaN; None reads as "not available")."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _acceptance_check(c: Check) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=c.name, passed=c.passed, value=_finite_or_none(c.value), threshold=_finite_or_none(c.threshold)
    )


def _j_key(f: Finalist) -> tuple[float, str]:
    j = f.validation.j
    return (-(j if math.isfinite(j) else -math.inf), f.config_id)


def _acceptance(selection: Selection, report_bps: float) -> AcceptanceResults:
    """Checks of the selected finalist (+ report J), or of the best finalist by validation J."""
    if selection.selected is not None:
        f = selection.selected
        report = AcceptanceCheck(
            name=f"report_j_at_{int(report_bps)}bps", passed=True, value=_finite_or_none(f.report.j), threshold=None
        )
        return AcceptanceResults(passed=True, checks=[*map(_acceptance_check, f.verdict.checks), report])
    if not selection.finalists:
        return AcceptanceResults(
            passed=False, checks=[AcceptanceCheck(name=_NO_FINALISTS, passed=False, value=0.0, threshold=1.0)]
        )
    best = min(selection.finalists, key=_j_key)
    return AcceptanceResults(passed=False, checks=[_acceptance_check(c) for c in best.verdict.checks])


def _code_commit() -> str:
    commit = git_commit()
    if commit is None:
        logger.warning("not in a git checkout; artifact code_commit is %s (unreproducible)", NO_GIT_COMMIT)
        return NO_GIT_COMMIT
    return commit


def build_artifact(
    *,
    run_id: str,
    rp: ResolvedProtocol,
    data_version: str,
    policy: Policy,
    plan: WindowPlan,
    universe: UniverseWeek,
    selection: Selection,
    search_reasons: Sequence[str],
    search_ess: float,
    logical: int,
    actual: int,
    seeds: Sequence[int],
    live_account: AccountConfig,
    runtime_s: float,
    peak_mb: float,
) -> WeeklyConfig:
    """Build the 0-A weekly configuration for one calibration.

    Inputs: the run id, the resolved protocol, the data snapshot version, the policy, the window plan,
    the activation week's universe, the D5 selection, the search-stage rejection reasons
    (``"search:<check> x<count>"`` over every distinct evaluated config), the realized ESS of the
    search weights, logical evaluations (sum over seeds), actual replays (sum of
    ``ReplayResponse.replays`` over every search and validation replay of this calibration), the
    seeds, the live account, the wall time and the peak memory.

    Output: a ``WeeklyConfig``. ``acceptance.passed`` is ``selection.status == "selected"``;
    ``acceptance.checks`` holds the selected finalist's validation checks plus the report-only
    ``report_j_at_<report_bps>bps`` check, or for ``no_deployment`` the checks of the finalist with
    the highest validation J (one ``no_feasible_finalists`` check when none exists), and
    ``no_deployment.reason = "; ".join(selection.reasons + search_reasons)``. NaN check values are
    written as null.

    Timing: the decision cutoff is the close of ``plan.cutoff``; the activation week is after it.
    """
    selected = no_deployment = None
    if selection.status == "selected":
        if selection.selected is None:
            raise ValueError("a 'selected' selection needs its selected finalist")
        cfg = selection.selected.config
        selected = SelectedConfig(
            config_id=selection.selected.config_id,
            grid_version=cfg.grid_version,
            subset=list(cfg.subset),
            params=cfg.as_dict(),
        )
    else:
        no_deployment = NoDeployment(reason="; ".join((*selection.reasons, *search_reasons)))
    return WeeklyConfig(
        run_id=run_id,
        code_commit=_code_commit(),
        data_version=data_version,
        protocol_hash=rp.hash,
        policy=policy,
        decision_cutoff=decision_cutoff_ts(plan.cutoff),
        search_range=DateRange(start=plan.search_sessions[0], end=plan.search_sessions[-1]),
        validation_range=DateRange(start=plan.validation_sessions[0], end=plan.validation_sessions[-1]),
        activation=DateRange(start=plan.activation_sessions[0], end=plan.activation_sessions[-1]),
        universe=[UniverseMember(asset_id=a, symbol=s) for a, s in universe.members],
        selected=selected,
        no_deployment=no_deployment,
        acceptance=_acceptance(selection, rp.protocol.acceptance.report_bps),
        ess=float(search_ess),
        evaluations=Evaluations(logical=logical, actual=actual),
        seeds=[int(s) for s in seeds],
        account=account_settings(live_account),
        runtime=Runtime(seconds=float(runtime_s), peak_memory_mb=float(peak_mb)),
    )


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, tuples, dates and non-finite floats (-> None) for JSON."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return _finite_or_none(float(obj))
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _finalist_dict(f: Finalist) -> dict[str, Any]:
    return {
        "config_id": f.config_id,
        "grid_version": f.config.grid_version,
        "subset": list(f.config.subset),
        "params": f.config.as_dict(),
        "seed": f.seed,
        "search_j": f.search_j,
        "n_filters": f.n_filters,
        "passed": f.verdict.passed,
        "checks": [dataclasses.asdict(c) for c in f.verdict.checks],
        "validation": dataclasses.asdict(f.validation),
        "report": dataclasses.asdict(f.report),
    }


def finalists_document(selection: Selection) -> dict[str, Any]:
    """The ``finalists.json`` body: status, selected id, reasons and every finalist's metrics and checks."""
    return {
        "status": selection.status,
        "selected": selection.selected.config_id if selection.selected is not None else None,
        "reasons": list(selection.reasons),
        "finalists": [_finalist_dict(f) for f in selection.finalists],
    }


def _dump(obj: Any) -> str:
    return json.dumps(_json_safe(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_run_dir(
    out_dir: Path,
    artifact: WeeklyConfig | None,
    *,
    audit: Iterable[str],
    finalists: Selection,
    rp: ResolvedProtocol,
    extra: Mapping[str, Any],
) -> Path:
    """Write a calibration's run directory atomically and return ``out_dir``.

    Inputs: the target directory, the artifact (None for a run that is not a protocol result), the
    audit lines, the selection, the resolved protocol and the ``run.json`` body. Files:
    ``weekly_config.json`` (``to_json``; absent when ``artifact`` is None), ``audit.jsonl``,
    ``finalists.json``, ``protocol.resolved.yaml`` (``dump_protocol_yaml``), ``overrides.json``
    (source, ``--set`` overrides and protocol hash) and ``run.json``. Everything is written to a
    sibling ``<name>.tmp`` directory, then any existing ``out_dir`` is removed and the tmp directory
    renamed into place (``os.replace`` onto a non-empty directory fails on Windows). No timing
    dependence.
    """
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if artifact is not None:
        (tmp / "weekly_config.json").write_text(to_json(artifact), encoding="utf-8")
    (tmp / "audit.jsonl").write_text("".join(f"{line}\n" for line in audit), encoding="utf-8")
    (tmp / "finalists.json").write_text(_dump(finalists_document(finalists)), encoding="utf-8")
    (tmp / "protocol.resolved.yaml").write_text(dump_protocol_yaml(rp.protocol), encoding="utf-8")
    overrides = {"source": rp.source, "overrides": list(rp.overrides), "protocol_hash": rp.hash}
    (tmp / "overrides.json").write_text(_dump(overrides), encoding="utf-8")
    (tmp / "run.json").write_text(_dump(extra), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return out_dir
