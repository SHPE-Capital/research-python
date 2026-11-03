"""Acceptance tests for the C1k check kits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import shpe_research.features  # noqa: F401  (registers the built-in features)
import shpe_research.models  # noqa: F401  (registers the built-in models)
from shpe_research.contracts.model import MODEL_REGISTRY
from shpe_research.features.base import FEATURE_REGISTRY, Feature
from shpe_research.testing.checks import check_feature, check_label, check_model, check_strategy
from shpe_research.testing.fixtures import check_bars, synthetic_bars, synthetic_xy

_FEATURE_PARAMS = {"past_return": {"window": 5}, "relative_volume": {"lookback_sessions": 2}}


def _build_feature(key: str) -> Feature:
    return FEATURE_REGISTRY[key](**_FEATURE_PARAMS.get(key, {}))


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return check_bars()


@pytest.mark.parametrize("key", sorted(FEATURE_REGISTRY))
def test_check_feature_passes_builtins(key, bars):
    check_feature(_build_feature(key), bars)


def test_check_feature_default_bars():
    check_feature(_build_feature("minute_of_day"))


def test_check_fixtures_reexports_synthetic_bars():
    from shpe_research.data.fixtures import synthetic_bars as original

    assert synthetic_bars is original


class _CenteredMean(Feature):
    feature_type = "centered_mean_test"

    @property
    def name(self) -> str:
        return "centered_mean"

    def compute(self, bars):
        return bars["close"].rolling(5, center=True, min_periods=1).mean()


def test_check_feature_catches_centered_mean(bars):
    with pytest.raises(AssertionError, match="centered_mean"):
        check_feature(_CenteredMean(), bars)


class _WrongIndex(Feature):
    feature_type = "wrong_index_test"

    @property
    def name(self) -> str:
        return "wrong_index"

    def compute(self, bars):
        return pd.Series(1.0, index=bars.index + 1)


def test_check_feature_catches_misaligned_index(bars):
    with pytest.raises(AssertionError, match="aligned"):
        check_feature(_WrongIndex(), bars)


class _AlwaysNaN(Feature):
    feature_type = "always_nan_test"

    @property
    def name(self) -> str:
        return "always_nan"

    def compute(self, bars):
        return pd.Series(np.nan, index=bars.index)


def test_check_feature_catches_nan_after_warmup(bars):
    with pytest.raises(AssertionError, match="NaN"):
        check_feature(_AlwaysNaN(), bars)


@pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
def test_check_model_passes_builtins(key):
    cls = MODEL_REGISTRY[key]
    X, y = synthetic_xy(cls.task)
    check_model(cls(), X, y)


def test_check_model_catches_unseeded_model(monkeypatch):
    from shpe_research.models.logistic import LogisticModel

    class Noisy(LogisticModel):
        model_type = "noisy_test"

        def _predict(self, X):
            return np.random.default_rng().uniform(size=len(X))

    monkeypatch.setitem(MODEL_REGISTRY, Noisy.model_type, Noisy)
    X, y = synthetic_xy("classification")
    with pytest.raises(AssertionError, match="seed"):
        check_model(Noisy(), X, y)


def test_check_model_catches_wrong_task(monkeypatch):
    from shpe_research.models.logistic import LogisticModel

    class Mislabelled(LogisticModel):
        model_type = "mislabelled_test"
        task = "regression"  # but predicts probabilities of a 0/1 label

        def _predict(self, X):
            return super()._predict(X) * 5.0

    monkeypatch.setitem(MODEL_REGISTRY, Mislabelled.model_type, Mislabelled)
    X, y = synthetic_xy("classification")
    with pytest.raises(AssertionError, match="task"):
        check_model(Mislabelled(), X, y)


class _Fwd:
    """Forward return over `h` rows; optionally truncates the horizon at the data end."""

    def __init__(self, truncate: bool, h: int = 10):
        self.truncate, self.h = truncate, h
        self.name = "fwd_test"

    def compute(self, bars):
        bars = bars.reset_index(drop=True)
        n = len(bars)
        idx = np.arange(n) + self.h
        ok = idx < n
        if self.truncate:
            idx = np.minimum(idx, n - 1)
        close = bars["close"].to_numpy()
        ts = bars["ts_available"]
        value = np.full(n, np.nan)
        has_value = np.ones(n, dtype=bool) if self.truncate else ok  # a truncating label scores every row
        value[has_value] = np.log(close[idx[has_value]] / close[has_value])
        end = ts + pd.Timedelta(minutes=self.h)  # rows whose horizon runs past the data keep this nominal end
        end[ok] = ts.iloc[idx[ok]].to_numpy()
        return pd.DataFrame({"decision_ts": ts, "label_end_ts": end, "fwd_test": value})


def test_check_label_passes_clean_label(bars):
    check_label(_Fwd(truncate=False), bars, bars["ts_available"])


def test_check_label_catches_truncation(bars):
    with pytest.raises(AssertionError, match="truncat"):
        check_label(_Fwd(truncate=True), bars, bars["ts_available"])


def test_check_label_catches_end_before_decision(bars):
    class Backwards(_Fwd):
        def compute(self, bars):
            out = super().compute(bars)
            out["label_end_ts"] = out["decision_ts"]
            return out

    with pytest.raises(AssertionError, match="label_end_ts"):
        check_label(Backwards(truncate=False), bars, bars["ts_available"])


class _EveryTenth:
    strategy_type = "every_tenth_test"
    shift = pd.Timedelta(0)

    def generate(self, bars, context=None):
        sel = bars.iloc[::10]
        return pd.DataFrame(
            {
                "symbol": sel["symbol"].to_numpy(),
                "decision_ts": (sel["ts_available"] + self.shift).to_numpy(),
                "side": "long",
                "strategy_id": "every_tenth_test:0",
            }
        )


def test_check_strategy_passes_clean_generator(bars):
    check_strategy(_EveryTenth(), bars)


def test_check_strategy_catches_future_ts(bars):
    class Future(_EveryTenth):
        shift = pd.Timedelta(seconds=30)  # a timestamp that is no bar's ts_available

    with pytest.raises(AssertionError, match="ts_available"):
        check_strategy(Future(), bars)


def test_check_strategy_catches_lookahead(bars):
    class Peek(_EveryTenth):
        def generate(self, bars, context=None):
            out = super().generate(bars.iloc[:0]).iloc[:0]
            # one candidate per bar, kept only if the NEXT bar closes higher: reads one bar past the decision
            close = bars["close"].to_numpy()
            keep = np.r_[close[1:] > close[:-1], False]
            sel = bars[keep]
            return pd.DataFrame({"symbol": sel["symbol"].to_numpy(), "decision_ts": sel["ts_available"].to_numpy(),
                                 "side": "long", "strategy_id": "peek:0"}).reindex(columns=out.columns.union(
                ["symbol", "decision_ts", "side", "strategy_id"], sort=False))

    with pytest.raises(AssertionError, match="truncat|future"):
        check_strategy(Peek(), bars)


def test_check_strategy_catches_invalid_frame(bars):
    class Bad(_EveryTenth):
        def generate(self, bars, context=None):
            return super().generate(bars).drop(columns=["side"])

    with pytest.raises(AssertionError, match="validate_candidates"):
        check_strategy(Bad(), bars)
