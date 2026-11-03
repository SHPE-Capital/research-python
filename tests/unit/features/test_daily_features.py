"""C1e: daily features (past_return_days, momentum_12_1, har_rv) and the daily pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.resample import resample_daily
from shpe_research.events.samplers import SessionClose
from shpe_research.features import FEATURE_REGISTRY, assert_no_lookahead, build_feature, compute_features
from shpe_research.features.daily import HarRv, Momentum121, PastReturnDays


def _daily(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    n = len(close)
    local = pd.bdate_range("2026-01-05", periods=n) + pd.Timedelta(hours=9, minutes=30)
    ts = to_utc_ns(pd.Series(local.tz_localize("America/New_York").tz_convert("UTC")))
    c = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": ts,
            "ts_available": ts + pd.Timedelta(hours=6, minutes=30),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1000,
            "vwap": c,
            "trade_count": pd.array([10] * n, dtype="Int64"),
            "session": "regular",
        }
    )


@pytest.fixture(scope="module")
def vol_minutes():
    return synthetic_bars(symbols=["AAA"], start="2026-03-02", end="2026-12-31", vol_clustering={"persistence": 0.95})


def test_registered():
    for key in ("past_return_days", "momentum_12_1", "har_rv"):
        assert key in FEATURE_REGISTRY
    assert isinstance(build_feature({"type": "past_return_days", "days": 5}), PastReturnDays)
    assert isinstance(build_feature({"type": "momentum_12_1"}), Momentum121)
    assert isinstance(build_feature({"type": "har_rv"}), HarRv)


def test_past_return_days_hand():
    bars = _daily([100, 110, 121, 133.1, 146.41])
    values = PastReturnDays(days=2).compute(bars)
    assert values.iloc[:2].isna().all()
    assert values.iloc[2] == pytest.approx(121 / 100 - 1)
    assert values.iloc[4] == pytest.approx(146.41 / 121 - 1)


def test_momentum_12_1_hand():
    close = [float(c) for c in range(100, 110)]  # 100..109 over ten sessions
    feature = Momentum121(skip_days=2, lookback_days=6)
    values = feature.compute(_daily(close))
    assert values.iloc[:6].isna().all()
    assert values.iloc[6] == pytest.approx(104 / 100 - 1)  # close_{6-2} / close_{6-6} - 1
    assert values.iloc[9] == pytest.approx(107 / 103 - 1)
    # the latest `skip_days` closes never enter: changing them changes nothing
    changed = list(close)
    changed[-1], changed[-2] = 500.0, 600.0
    assert feature.compute(_daily(changed)).iloc[9] == pytest.approx(values.iloc[9])


def test_momentum_defaults_and_validation():
    f = Momentum121()
    assert (f.skip_days, f.lookback_days) == (21, 252)
    assert f.name == "momentum_12_1"
    with pytest.raises(ValueError):
        Momentum121(skip_days=10, lookback_days=5)


def test_har_rv_hand():
    minute = synthetic_bars(symbols=["AAA"], start="2026-03-02", end="2026-03-04")
    daily = resample_daily(minute)
    out = HarRv(bar_minutes=5).compute_frame(daily, {"minute_bars": minute})
    assert list(out.columns) == ["rv_d", "rv_w", "rv_m"]
    first = minute[minute["ts"] < pd.Timestamp("2026-03-03T00:00:00Z")]
    # 5-minute closes: the close of the minute bar available at open + 5k minutes; the first return starts at the open
    closes = first["close"].to_numpy()[4::5]
    ref = np.r_[first["open"].iloc[0], closes[:-1]]
    assert out["rv_d"].iloc[0] == pytest.approx(float(np.sum(np.log(closes / ref) ** 2)))
    assert out[["rv_w", "rv_m"]].iloc[:3].isna().all().all()  # too few sessions for the 5 and 22 session means


def test_har_rv_planted(vol_minutes):
    daily = resample_daily(vol_minutes)
    out = HarRv(bar_minutes=5).compute_frame(daily, {"minute_bars": vol_minutes})
    rv = out["rv_d"].dropna()
    assert len(rv) >= 126
    assert rv.autocorr(lag=1) > 0.5
    # weekly and monthly components are trailing means of rv_d over sessions <= d
    assert out["rv_w"].iloc[10] == pytest.approx(out["rv_d"].iloc[6:11].mean())
    assert out["rv_m"].iloc[30] == pytest.approx(out["rv_d"].iloc[9:31].mean())


def test_har_rv_via_compute_features(vol_minutes):
    daily = resample_daily(vol_minutes)
    features = [HarRv(column=c) for c in ("rv_d", "rv_w", "rv_m")]
    frame = compute_features(daily, features, context={"minute_bars": vol_minutes})
    assert {"rv_d", "rv_w", "rv_m"} <= set(frame.columns)
    with pytest.raises(ValueError, match="minute_bars"):
        compute_features(daily, features)


class _WithMinutes:
    """Binds the minute bars so assert_no_lookahead (which passes no context) can drive har_rv."""

    def __init__(self, feature: HarRv, minute: pd.DataFrame):
        self.feature, self.minute, self.name = feature, minute, feature.name

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return self.feature.compute(bars, {"minute_bars": self.minute})


@pytest.mark.parametrize("key", ["past_return_days", "momentum_12_1", "har_rv"])
def test_no_lookahead(key, vol_minutes):
    daily = resample_daily(vol_minutes)
    if key == "har_rv":
        for column in ("rv_d", "rv_w", "rv_m"):
            assert_no_lookahead(_WithMinutes(HarRv(column=column), vol_minutes), daily)  # type: ignore[arg-type]
        return
    params = {"days": 5} if key == "past_return_days" else {"skip_days": 5, "lookback_days": 40}
    assert_no_lookahead(build_feature({"type": key, **params}), daily)


def test_har_rv_ignores_minute_bars_after_the_decision(vol_minutes):
    """Removing later daily and minute bars leaves earlier rv values unchanged."""
    daily = resample_daily(vol_minutes)
    cut = daily["ts_available"].iloc[60]
    full = HarRv().compute_frame(daily, {"minute_bars": vol_minutes})
    early = HarRv().compute_frame(
        daily[daily["ts_available"] <= cut], {"minute_bars": vol_minutes[vol_minutes["ts_available"] <= cut]}
    )
    np.testing.assert_allclose(early.to_numpy(), full.iloc[: len(early)].to_numpy(), equal_nan=True)


def test_daily_features_and_session_close_decisions_on_daily_fixture():
    """The part of the daily pipeline this packet owns: daily fixture bars -> features -> session_close decisions."""
    bars = synthetic_bars(daily={"years": 3, "n_symbols": 50})
    features = [PastReturnDays(days=5), Momentum121(skip_days=5, lookback_days=40)]
    frame = compute_features(bars, features)
    decisions = SessionClose().sample(bars)
    assert len(decisions) == len(bars)  # one decision per symbol and session, at the session close
    merged = decisions.merge(
        frame.rename(columns={"ts_available": "decision_ts"}), on=["symbol", "decision_ts"], validate="one_to_one"
    )
    assert len(merged) == len(bars)
    assert merged["past_return_5d"].notna().sum() > 0.9 * len(merged)
    assert merged["momentum_40_5d"].notna().sum() > 0.8 * len(merged)


@pytest.mark.skip(
    reason="needs C1h/C1a follow-up: pipeline.stages (old dataclass config, minute-only ForwardReturn, "
    "every_n_minutes forced, era = date) and experiments.sources are not owned by C1e"
)
def test_daily_pipeline_runs():
    """`frequency: daily` + `session_close` + `forward_return` end to end on the C1o daily fixture."""
