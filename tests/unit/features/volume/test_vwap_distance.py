from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.base import assert_no_lookahead, build_feature, compute_features
from shpe_research.features.volume.vwap_distance import VwapDistance


def _bars(rows):
    """rows: (ts, high, low, close, vwap, volume); ts_available = ts + 1 minute."""
    df = pd.DataFrame(rows, columns=["ts", "high", "low", "close", "vwap", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["ts_available"] = df["ts"] + pd.Timedelta(minutes=1)
    df.insert(0, "symbol", "AAA")
    return df


@pytest.fixture
def bars():
    return _bars(
        [
            ("2026-04-01 13:30", 10.5, 9.5, 10.0, 10.0, 100.0),
            ("2026-04-01 13:31", 11.5, 10.5, 11.0, 11.0, 300.0),
            # missing minute 13:32 stays missing; this bar has no vwap -> (h+l+c)/3 fallback
            ("2026-04-01 13:33", 12.5, 11.5, 12.0, np.nan, 100.0),
            ("2026-04-01 13:34", 12.5, 11.5, 12.0, 12.0, 0.0),
            # next session resets
            ("2026-04-02 13:30", 20.5, 19.5, 20.0, 19.0, 0.0),
            ("2026-04-02 13:31", 20.5, 19.5, 20.0, 19.0, 50.0),
        ]
    )


def test_hand_computed_with_fallback_and_session_reset(bars):
    values = VwapDistance().compute(bars)
    fallback = (12.5 + 11.5 + 12.0) / 3
    v2 = (10 * 100 + 11 * 300 + fallback * 100) / 500
    expected = [
        10.0 / 10.0 - 1,
        11.0 / ((1000 + 3300) / 400) - 1,
        12.0 / v2 - 1,
        12.0 / v2 - 1,  # zero-volume bar keeps cumulative VWAP
        np.nan,  # zero cumulative volume at session start
        20.0 / 19.0 - 1,  # prior session does not leak in
    ]
    np.testing.assert_allclose(values.to_numpy(), expected, rtol=1e-12, equal_nan=True)
    assert values.dtype == "float64"
    assert values.index.equals(bars.index)


def test_registered_and_lookahead(bars):
    assert isinstance(build_feature({"type": "vwap_distance"}), VwapDistance)
    assert_no_lookahead(VwapDistance(), bars, cutoffs=[1, 2, 3, 4, 5])
    out = compute_features(bars, [VwapDistance()])
    assert "vwap_distance" in out.columns
