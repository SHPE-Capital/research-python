import math

import pytest

from shpe_research.trade import fill_proxy as fp


def test_limit_is_close_times_markup():
    assert fp.entry_limit(100.0, 0.0020) == pytest.approx(100.2)


def test_fills_at_next_open_when_at_or_below_limit():
    assert fp.entry_fill(100.2, 100.05) == (fp.FILLED, 100.05)
    assert fp.entry_fill(100.2, 100.2) == (fp.FILLED, 100.2)  # equal fills


def test_cancel_above_limit():
    code, px = fp.entry_fill(100.2, 100.21)
    assert code == fp.CANCEL_ABOVE_LIMIT and math.isnan(px)


def test_missing_next_bar_no_fill():
    code, px = fp.entry_fill(100.2, float("nan"))
    assert code == fp.NO_NEXT_BAR and math.isnan(px)


@pytest.mark.parametrize("bps", fp.COST_SCENARIOS_BPS)
def test_cost_scenarios(bps):
    assert fp.validate_cost_bps(bps) == bps
    assert fp.side_cost_rate(bps) * 1e4 == pytest.approx(bps)


def test_cost_scenario_values_and_selection_and_invalid():
    assert fp.COST_SCENARIOS_BPS == (5.0, 10.0, 20.0) and fp.SELECTION_COST_BPS == 10.0
    with pytest.raises(ValueError):
        fp.validate_cost_bps(7.0)
