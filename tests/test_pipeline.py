from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shpe_research.cli import main
from shpe_research.contracts.bars import validate_bars
from shpe_research.contracts.model import load_model
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.experiments.runner import ExperimentConfig, run_experiment
from shpe_research.features import MinuteOfSession, PastReturn, assert_no_lookahead, build_feature
from shpe_research.labels import ForwardReturn
from shpe_research.validation import split_by_day

SMALL = {"source": "fixture", "symbols": ["AAA", "BBB", "CCC", "DDD"], "start": "2026-04-01", "end": "2026-06-30"}


@pytest.fixture(scope="module")
def two_days():
    return synthetic_bars(symbols=["AAA"], start="2026-11-25", end="2026-11-27")


def test_fixture_is_valid_deterministic_and_calendar_aware(two_days):
    validate_bars(two_days)
    per_day = two_days.groupby(two_days["ts"].dt.tz_convert("America/New_York").dt.date).size()
    assert per_day.tolist() == [390, 210]  # Thanksgiving skipped, early close on the 27th
    again = synthetic_bars(symbols=["AAA"], start="2026-11-25", end="2026-11-27")
    pd.testing.assert_frame_equal(two_days, again)


def test_past_return_values_and_causality(two_days):
    feature = build_feature({"type": "past_return", "window": 5})
    assert isinstance(feature, PastReturn)
    assert feature.name == "past_return_5m"
    values = feature.compute(two_days)
    close = two_days["close"]
    assert values.iloc[:5].isna().all()  # first five minutes of the session have no 5-minute history
    assert values.iloc[5] == pytest.approx(np.log(close.iloc[5] / close.iloc[0]))
    assert values.iloc[390:395].isna().all()  # history does not carry across sessions
    assert_no_lookahead(feature, two_days)
    assert_no_lookahead(MinuteOfSession(), two_days)


def test_past_return_uses_time_not_rows(two_days):
    gappy = two_days.drop(index=[2, 3]).reset_index(drop=True)  # two minutes without trades
    values = PastReturn(5).compute(gappy)
    # Row 4 is the 09:36 bar (available 09:37); its reference is the bar available at 09:32 (row 1).
    assert values.iloc[4] == pytest.approx(np.log(gappy["close"].iloc[4] / gappy["close"].iloc[1]))


def test_forward_return_stays_inside_the_session(two_days):
    label = ForwardReturn(30).compute(two_days)
    col = label["fwd_return_30m"]
    close = two_days["close"]
    assert col.iloc[0] == pytest.approx(np.log(close.iloc[30] / close.iloc[0]))
    assert (label["label_end_ts"] - label["decision_ts"] == pd.Timedelta(minutes=30)).all()
    assert col.iloc[360:390].isna().all()  # horizon would pass the 16:00 close
    assert col.iloc[389 + 180] == pytest.approx(np.log(close.iloc[389 + 210] / close.iloc[389 + 180]))
    assert col.iloc[390 + 180 :].isna().all()  # early close at 13:00


def test_forward_return_is_nan_when_nothing_prints(two_days):
    sparse = two_days.drop(index=range(1, 40)).reset_index(drop=True)
    assert np.isnan(ForwardReturn(30).compute(sparse)["fwd_return_30m"].iloc[0])


def test_split_by_day_is_chronological_with_gap():
    ts = synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-04-30")["ts_available"]
    splits = split_by_day(ts, 0.5, 0.25, 0.25, gap_days=1)
    per_day = splits.groupby(ts.dt.date).first()
    runs = [name for name, _ in itertools.groupby(per_day)]
    assert runs == ["train", "unassigned", "validation", "unassigned", "test"]


def _small_config(**overrides) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(Path(__file__).parents[1] / "configs/experiments/hello_model.yaml")
    cfg.data = {**SMALL, **overrides}
    return cfg


def test_hello_model_end_to_end(tmp_path: Path):
    result = run_experiment(_small_config(signal_strength=0.5), out_root=tmp_path)
    for name in ("config.yaml", "predictions.parquet", "report.json", "report.md", "model/manifest.json"):
        assert (result.run_dir / name).exists(), name
    assert result.metrics["validation"]["auc"] > 0.51
    assert result.metrics["validation"]["ic_mean"] > 0
    loaded = load_model(result.run_dir / "model")
    assert loaded.manifest.feature_names == ["past_return_5m", "past_return_30m", "minute_of_day"]
    preds = pd.read_parquet(result.run_dir / "predictions.parquet")
    assert set(preds["split"]) == {"train", "validation", "test"}
    assert "test" not in loaded.manifest.metrics  # the test split is reported, never used for selection


def test_no_signal_gives_no_edge(tmp_path: Path):
    """A pipeline that finds an edge in pure noise is leaking future data."""
    result = run_experiment(_small_config(signal_strength=0.0), out_root=tmp_path)
    assert abs(result.metrics["validation"]["auc"] - 0.5) < 0.03


def test_cli_run_and_fixtures(tmp_path: Path, capsys):
    cfg = _small_config(symbols=["AAA", "BBB"], start="2026-05-01", end="2026-05-29").to_dict()
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(path), "--out", str(tmp_path / "runs")]) == 0
    assert "| auc |" in capsys.readouterr().out
    out = tmp_path / "fx.parquet"
    argv = ["fixtures", "--out", str(out), "--symbols", "AAA", "--start", "2026-05-01", "--end", "2026-05-05"]
    assert main(argv) == 0
    assert out.exists()
