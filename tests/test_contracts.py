from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from shpe_research.contracts import ContractError
from shpe_research.contracts.bars import (
    BarMetadata,
    bars_to_table,
    empty_bars,
    table_to_bars,
    validate_bars,
)
from shpe_research.contracts.base import to_utc_ns
from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import Model, load_model, register_model
from shpe_research.contracts.ticks import validate_quotes, validate_trades
from shpe_research.features.base import Feature, assert_no_lookahead

from .conftest import make_bars

# ---------------------------------------------------------------- bars


def test_valid_bars_pass(bars):
    validate_bars(bars)
    validate_bars(empty_bars())


def test_bars_problems_are_all_reported(bars):
    broken = pd.concat([bars, bars.iloc[[0]]]).reset_index(drop=True)
    broken.loc[1, "ts_available"] = broken.loc[1, "ts"]
    broken.loc[2, "session"] = "lunch"
    with pytest.raises(ContractError) as err:
        validate_bars(broken)
    text = str(err.value)
    for expected in ("duplicate", "sorted", "ts_available", "lunch"):
        assert expected in text


def test_bars_require_nanosecond_utc(bars):
    with pytest.raises(ContractError, match="ns, UTC"):
        validate_bars(bars.assign(ts=bars["ts"].astype("datetime64[us, UTC]")))
    with pytest.raises(ContractError, match="missing column 'vwap'"):
        validate_bars(bars.drop(columns="vwap"))


def test_bars_parquet_round_trip_keeps_dtypes_and_metadata(bars, tmp_path: Path):
    meta = BarMetadata(feed="sip", timeframe="1Min", adjustment="raw")
    pq.write_table(bars_to_table(bars, meta), tmp_path / "bars.parquet")
    table = pq.read_table(tmp_path / "bars.parquet")
    back = table_to_bars(table)
    validate_bars(back)
    pd.testing.assert_frame_equal(back, bars, check_dtype=False)
    assert BarMetadata.from_schema_metadata(table.schema.metadata) == meta


# ---------------------------------------------------------------- ticks


def test_quotes_and_trades_contracts():
    ts = to_utc_ns(pd.Series(["2026-10-01T13:30:00.000000001Z", "2026-10-01T13:30:01Z"]))
    quotes = pd.DataFrame(
        {
            "symbol": "AAA", "ts": ts, "bid_price": [10.0, 10.01], "ask_price": [10.02, 10.03],
            "bid_size": [3, 4], "ask_size": [2, 1], "bid_exchange": "V", "ask_exchange": "Q",
            "tape": "A", "conditions": [["R"], []],
        }
    )
    validate_quotes(quotes)
    with pytest.raises(ContractError, match="crossed"):
        validate_quotes(quotes.assign(ask_price=[9.0, 10.03]))
    trades = pd.DataFrame(
        {
            "symbol": "AAA", "ts": ts, "price": [10.01, 10.02], "size": [100, 5], "exchange": "V",
            "tape": "A", "conditions": [["@"], ["@", "I"]], "trade_id": ["1", "2"],
        }
    )
    validate_trades(trades)
    with pytest.raises(ContractError, match="positive"):
        validate_trades(trades.assign(size=[0, 5]))


# ---------------------------------------------------------------- dataset


def _dataset(n: int = 10, **overrides) -> Dataset:
    decision = to_utc_ns(pd.date_range("2026-10-01T14:00Z", periods=n, freq="15min"))
    parts = {
        "X": pd.DataFrame({"f1": np.arange(n, dtype=float), "f2": np.ones(n)}),
        "y": pd.Series(np.arange(n) % 2),
        "meta": pd.DataFrame(
            {
                "symbol": "AAA",
                "decision_ts": decision,
                "label_end_ts": decision + pd.Timedelta(minutes=30),
                "weight": 1.0,
                "split": ["train"] * (n // 2) + ["validation"] * (n - n // 2),
            }
        ),
        "label_name": "fwd_sign_30m",
    }
    parts.update(overrides)
    return Dataset(**parts)


def test_dataset_valid_and_subsets():
    ds = _dataset()
    assert ds.feature_names == ["f1", "f2"]
    assert len(ds.subset("train")) == 5
    relabeled = ds.with_splits(pd.Series("test", index=ds.meta.index))
    assert set(relabeled.meta["split"]) == {"test"}


def test_dataset_rejects_label_ending_before_decision():
    meta = _dataset().meta
    meta.loc[3, "label_end_ts"] = meta.loc[3, "decision_ts"]
    with pytest.raises(ContractError, match="label_end_ts"):
        _dataset(meta=meta)


def test_dataset_rejects_missing_labels_and_nan_features():
    with pytest.raises(ContractError, match="no label"):
        _dataset(y=pd.Series([np.nan] + [1.0] * 9))
    X = _dataset().X
    X.loc[0, "f1"] = np.nan
    with pytest.raises(ContractError, match="NaN features"):
        _dataset(X=X)
    assert len(_dataset(X=X, allow_nan_features=True)) == 10


# ---------------------------------------------------------------- model


@register_model
class _MeanModel(Model):
    model_type = "test_mean"
    task = "regression"

    def _fit(self, X, y, weight):
        self.mean_ = float(np.average(y, weights=weight))

    def _predict(self, X):
        return np.full(len(X), self.mean_)

    def _save_state(self, directory):
        (directory / "state.txt").write_text(str(self.mean_))

    def _load_state(self, directory):
        self.mean_ = float((directory / "state.txt").read_text())


def test_model_contract_round_trip(tmp_path: Path):
    ds = _dataset()
    model = _MeanModel(seed=3).fit(ds)
    assert model.manifest.feature_names == ["f1", "f2"]
    assert model.manifest.n_train == 10
    model.save(tmp_path / "m")
    loaded = load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(ds.X[["f2", "f1"]]), model.predict(ds.X))
    with pytest.raises(ValueError, match="missing features"):
        loaded.predict(ds.X[["f1"]])


# ---------------------------------------------------------------- features


class _TrailingMean(Feature):
    feature_type = "test_trailing_mean"

    @property
    def name(self):
        return "trailing_mean_5"

    def compute(self, bars):
        return bars["close"].rolling(5).mean()


class _CenteredMean(_TrailingMean):
    def compute(self, bars):
        return bars["close"].rolling(5, center=True).mean()


def test_assert_no_lookahead_accepts_causal_and_rejects_future_use():
    bars = make_bars(n=120)
    assert_no_lookahead(_TrailingMean(), bars)
    with pytest.raises(AssertionError, match="uses future data"):
        assert_no_lookahead(_CenteredMean(), bars)
