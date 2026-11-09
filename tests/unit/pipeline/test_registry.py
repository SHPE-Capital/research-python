from __future__ import annotations

import pytest

from shpe_research.pipeline import registry as reg
from shpe_research.strategies import registry as strat_reg

CASES = [
    (reg.register_sampler, "sampler_type", reg.SAMPLER_REGISTRY),
    (reg.register_label, "label_type", reg.LABEL_REGISTRY),
    (reg.register_fit_scheme, "scheme_type", reg.FIT_SCHEME_REGISTRY),
    (reg.register_report_section, "section_type", reg.REPORT_SECTION_REGISTRY),
    (strat_reg.register_strategy, "strategy_type", strat_reg.STRATEGY_REGISTRY),
]


@pytest.mark.parametrize(("decorator", "key_attr", "registry"), CASES)
def test_duplicate_key_raises(decorator, key_attr, registry):
    key = "_dup_test_key"
    first = type("First", (), {key_attr: key})
    second = type("Second", (), {key_attr: key})
    try:
        decorator(first)
        with pytest.raises(ValueError, match="already registered"):
            decorator(second)
    finally:
        registry.pop(key, None)


def test_unknown_key_lists_registered():
    with pytest.raises(ValueError, match="forward_return"):
        reg.build_label({"type": "nope"})
    with pytest.raises(ValueError, match="every_n_minutes"):
        reg.build_sampler({"type": "nope"})
    with pytest.raises(ValueError, match="single"):
        reg.build_fit_scheme({"type": "nope"})
    with pytest.raises(ValueError, match="signal"):
        reg.build_report_section({"type": "nope"})
    with pytest.raises(ValueError, match="unknown strategy"):
        strat_reg.build_strategy({"type": "nope"})


def test_builtins_registered_on_import():
    # The builders import the owning packages, so the built-ins resolve with no explicit module import.
    assert reg.build_sampler({"type": "every_n_minutes", "minutes": 15}).minutes == 15
    assert reg.build_label({"type": "forward_return", "horizon_minutes": 30}).horizon_minutes == 30
    assert reg.build_fit_scheme({"type": "single"}).scheme_type == "single"
    assert reg.build_report_section({"type": "signal"}).section_type == "signal"
