"""Independent controls for the packed signals and the weekend information boundary."""

from itertools import islice
from types import SimpleNamespace

import numpy as np
import pandas as pd

from shpe_research.calibration.rm9_engine import KEY_INDEX, compile_week
from shpe_research.calibration.rm9_spec import spec
from shpe_research.calibration.rm9_study import select
from shpe_research.strategies.momentum.breakout.grid import DEFAULT_DOMAIN, enumerate_configs
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade.account_types import MarketPanel


def test_packed_week_matches_existing_signal_engine_on_all_four_filter_families(tmp_path):
    p = spec("demo")
    ids = ["x", "y"]
    plans = {
        "asof": "2023-01-06",
        "days": ["2023-01-09", "2023-01-10"],
        "members": [dict(id=i, cell=0, post_split_block=[]) for i in ids],
    }
    groups = {i: pd.DataFrame([dict(start="2010-01-01", end="2099-01-01")]) for i in ids}
    captures = []

    def loader(lo, hi, days, ts, si):
        src = np.empty((hi - lo, len(ts), 5), dtype="f4")
        for j in range(hi - lo):
            t = np.arange(len(ts))
            c = 20 * np.exp(np.cumsum(0.00015 + 0.0007 * np.sin(t / 12 + j)))
            o = np.r_[c[0], c[:-1]]
            src[j] = np.column_stack(
                [o, np.maximum(o, c) * 1.0001, np.minimum(o, c) * 0.9999, c, 50000 + 9000 * np.cos(t / 9)]
            )
        captures.append((src, si))
        return src

    path = tmp_path / "week"
    meta = compile_week(None, groups, plans, path, p, block=2, fixture_block=loader)
    src, si = captures[0]
    full = MarketPanel(tuple(ids), *(src[:, :, k] for k in range(5)), si, np.ones(src.shape[:2], bool))
    oracle = IndicatorCache(full)
    bits = np.load(path / "bits.npy")
    atr = np.load(path / "atr.npy")
    checked, any_buy = set(), False
    for cfg in enumerate_configs():
        if cfg.subset in checked:
            continue
        checked.add(cfg.subset)
        ref = signal_mask(full, cfg, oracle)
        vals = cfg.as_dict()
        ix = [
            KEY_INDEX[(name, *(vals[f"{name}.{k}"] for k, _ in DEFAULT_DOMAIN.params_of(name)))] for name in cfg.subset
        ]
        out = np.unpackbits(np.bitwise_and.reduce(bits[ix], axis=0), axis=1, count=meta["minutes"])
        np.testing.assert_array_equal(out, ref.signal[:, -meta["minutes"] :])
        any_buy |= bool(out.any())
        np.testing.assert_allclose(atr, ref.atr[:, -meta["minutes"] :], rtol=1e-6, equal_nan=True)
    assert len(checked) == 14 and any_buy


def test_no_future_pnl_can_change_weekend_selection_and_all_fail_means_cash():
    p = spec("demo")
    days = pd.bdate_range("2023-01-09", periods=35)
    plans = [
        {
            "days": [str(d.date()) for d in days[i : i + 5]],
            "asof": str((days[i] - pd.Timedelta(days=1)).date()),
            "regime": [float(i), 0],
        }
        for i in range(0, 35, 5)
    ]
    vals = (np.sin(np.arange(35)) * 0.0009 + 0.001)[None, :]
    returns = np.concatenate((vals, vals * 0.3 - 0.001))
    g = SimpleNamespace(configs=list(islice(enumerate_configs(), 2)))
    bank = SimpleNamespace(offset=np.arange(0, 36, 5), idx=[0, 1], r=returns, trade=np.ones((2, 35), int))
    first = select(bank, g, plans, p, 6)
    assert all(x["idx"] == 0 for x in first.values())
    returns[:, 30:] = -0.99
    assert first == select(bank, g, plans, p, 6)
    returns[:, 15:30] = -0.02
    assert all(x["idx"] is None for x in select(bank, g, plans, p, 6).values())
