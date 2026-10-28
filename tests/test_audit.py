from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shpe_research.cli import main
from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.audit import audit_bars
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.store import BarUnit, DataStore, period_window


@pytest.fixture(scope="module")
def clean():
    return synthetic_bars(symbols=["AAA", "BBB"], start="2026-11-23", end="2026-11-30")


def test_clean_data_has_full_coverage_and_no_issues(clean):
    report = audit_bars(clean)
    assert report.issues.empty
    assert (report.symbols["coverage"] == 1.0).all()
    assert (report.symbols["morning_coverage"] == 1.0).all()
    early = report.days[(report.days["symbol"] == "AAA") & (report.days["date"] == "2026-11-27")].iloc[0]
    assert early["expected"] == 210
    assert bool(early["early_close"])


def test_each_problem_is_reported(clean):
    bars = clean.copy()
    aaa = bars["symbol"] == "AAA"
    day = bars["ts"].dt.tz_convert("America/New_York").dt.date.astype(str)
    bars = bars[~(aaa & (day == "2026-11-24"))]  # AAA missing a whole session
    after = (bars["symbol"] == "BBB") & (bars["ts"] >= pd.Timestamp("2026-11-27T00:00Z"))
    bars.loc[after, ["open", "high", "low", "close", "vwap"]] /= 2  # BBB 2-for-1 split
    bars.loc[bars.index[5], "high"] = bars.loc[bars.index[5], "close"] - 1  # invalid OHLC
    sat = to_utc_ns(pd.Series(["2026-11-28T15:00Z"], index=bars.index[:1]))
    saturday = bars.iloc[[0]].assign(ts=sat, ts_available=sat + pd.Timedelta(minutes=1))
    bars = pd.concat([bars, bars.iloc[[10]], saturday]).sort_values(["symbol", "ts"], kind="stable")

    report = audit_bars(bars, start="2026-11-23", end="2026-11-30")
    kinds = set(zip(report.issues["symbol"], report.issues["kind"], strict=True))
    assert ("AAA", "missing_day") in kinds
    assert ("AAA", "invalid_ohlc") in kinds
    assert ("AAA", "duplicates") in kinds
    assert ("AAA", "off_session") in kinds
    assert ("BBB", "possible_split") in kinds
    assert set(report.errors["kind"]) == {"invalid_ohlc", "duplicates", "off_session"}
    aaa_summary = report.symbols.set_index("symbol").loc["AAA"]
    assert aaa_summary["missing_days"] == 1
    assert aaa_summary["errors"] >= 3


def test_illiquid_minutes_lower_coverage_without_errors(clean):
    sparse = clean[clean.index % 3 != 0]
    report = audit_bars(sparse)
    assert report.errors.empty
    assert report.symbols["coverage"].iloc[0] == pytest.approx(2 / 3, abs=0.01)


def test_report_files_and_cli(clean, tmp_path: Path, monkeypatch, capsys):
    store = DataStore(tmp_path)
    nov = clean[clean["symbol"] == "AAA"]
    unit = BarUnit("sip", "1Min", "raw", "AAA", "2026-11")
    start, end = period_window("2026-11")
    store.write_bar_unit(unit, nov.reset_index(drop=True), start, end, complete=True)
    store.bars_manifest.flush()

    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    out = tmp_path / "audit"
    assert main(["data", "audit", "--symbols", "AAA", "--from", "2026-11-23", "--to", "2026-11-30",
                 "--out", str(out)]) == 0
    assert (out / "audit.md").exists()
    assert (out / "audit.json").exists()
    assert "| AAA | 5 | 0 | 1.000 | 1.000 | 0 | 0 |" in capsys.readouterr().out
