from collections import Counter

from shpe_research.strategies.momentum.breakout.grid import (
    Config,
    GridDomain,
    config_id,
    enumerate_configs,
    feature_vector,
    from_feature_vector,
    is_valid,
)


def test_default_grid_count_and_subsets():
    configs = list(enumerate_configs())
    assert len(configs) == 69_732
    assert len(Counter(c.subset for c in configs)) == 14
    assert all("breakout" in c.subset for c in configs)


def test_ids_unique_and_stable():
    configs = list(enumerate_configs())
    assert len({config_id(c) for c in configs}) == len(configs)
    # Pinned value: guards cross-session / cross-platform stability.
    c = Config.make("v1", ["breakout"], {"breakout.n": 10, "breakout.b": 0})
    assert config_id(c) == config_id(Config.make("v1", ["breakout"], {"breakout.b": 0.0, "breakout.n": 10}))
    assert config_id(c) == "f890f19577080f6f"


def test_id_independent_of_order_and_includes_version():
    a = Config.make("v1", ["vwap", "breakout"], {"vwap.e": 1, "breakout.n": 10, "breakout.b": 0.1})
    b = Config.make("v1", ["breakout", "vwap"], {"breakout.b": 0.1, "breakout.n": 10, "vwap.e": 1})
    assert a == b and config_id(a) == config_id(b)
    v2 = Config.make("v2", ["breakout", "vwap"], a.as_dict())
    assert config_id(v2) != config_id(a)


def test_is_valid_rejects_bad_configs():
    good = Config.make("v1", ["breakout", "vwap"], {"breakout.n": 10, "breakout.b": 0, "vwap.e": 2})
    assert is_valid(good)
    bad_value = Config.make("v1", ["breakout", "vwap"], {"breakout.n": 11, "breakout.b": 0, "vwap.e": 2})
    no_anchor = Config.make("v1", ["vwap", "rvol"], {"vwap.e": 2, "rvol.w": 3, "rvol.v": 1})
    missing = Config.make("v1", ["breakout", "vwap"], {"breakout.n": 10, "breakout.b": 0})
    wrong_ver = Config.make("v9", ["breakout", "vwap"], good.as_dict())
    five = Config.make(
        "v1",
        ["breakout", "adx", "ema", "rvol", "vwap"],
        {
            "breakout.n": 10, "breakout.b": 0, "adx.period": 7, "adx.q": 15, "ema.fast": 5, "ema.slow": 20,
            "ema.d": 0, "rvol.w": 3, "rvol.v": 1, "vwap.e": 1,
        },
    )  # fmt: skip
    for c in (bad_value, no_anchor, missing, wrong_ver, five):
        assert not is_valid(c)


def test_feature_vector_round_trip_with_inactive_markers():
    cfgs = list(enumerate_configs())[::997]
    for c in cfgs:
        vec = feature_vector(c)
        assert from_feature_vector(vec) == c
    c = Config.make("v1", ["breakout", "vwap"], {"breakout.n": 10, "breakout.b": 0, "vwap.e": 2})
    vec = feature_vector(c)
    assert len(vec) == 5 + 10
    assert vec[:6] == [1.0, 10.0, 0.0, 0.0, -1.0, -1.0]  # breakout on, adx off
    assert vec.count(-1.0) == 2 + 3 + 2  # adx(2) + ema(3) + rvol(2)


def test_custom_domain_from_mapping():
    dom = GridDomain.from_mapping(
        {
            "version": "t1",
            "components": {"breakout": {"n": [5, 10]}, "vwap": {"e": [1, 2, 3]}, "rvol": {"w": [3], "v": [1, 2]}},
            "required": ["breakout"],
            "min_components": 2,
            "max_components": 3,
        }
    )
    configs = list(enumerate_configs(dom))
    assert len(configs) == 2 * 3 + 2 * 2 + 2 * 3 * 2
    assert all(is_valid(c, dom) for c in configs)
    assert not is_valid(configs[0])  # default domain rejects it


def test_ema_slow_must_exceed_fast():
    dom = GridDomain.from_mapping(
        {
            "version": "t2",
            "components": {"breakout": {"n": [5]}, "ema": {"fast": [5, 20], "slow": [20]}},
            "min_components": 2,
            "max_components": 2,
        }
    )
    assert [c.as_dict()["ema.fast"] for c in enumerate_configs(dom)] == [5]
