"""T5 acceptance tests: CostModel (defaults, no double charge, impact, TCA loading)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from shpe_research.trade.costs import round_trip
from shpe_research.trade.types import CostModel, FillModel


def test_default_equals_platform():
    assert CostModel() == CostModel.from_fill(FillModel())
    fill = FillModel(half_spread_bps=1.0, slippage_bps=3.0, commission_per_share=0.01)
    cm = CostModel.from_fill(fill)
    assert (cm.half_spread_bps, cm.slippage_bps, cm.commission_per_share) == (1.0, 3.0, 0.01)
    assert cm.source == "platform_default"


@pytest.mark.parametrize("side", ["long", "short"])
def test_no_double_charge(side):
    """Fill prices already carry spread and slippage; net = mid-price gross minus each cost once."""
    cm, fill = CostModel(), FillModel()
    entry_mid, exit_mid, qty = 100.0, 101.0, 200
    rt = round_trip(side, entry_mid, exit_mid, qty, costs=cm, fill=fill)
    sign = 1.0 if side == "long" else -1.0
    gross_usd = sign * (exit_mid - entry_mid) * qty
    # spread + slippage charged once per leg, commission once per leg, nothing else
    entry_leg, _ = cm.cost("buy" if side == "long" else "sell", entry_mid, qty)
    exit_leg, _ = cm.cost("sell" if side == "long" else "buy", exit_mid, qty)
    spread_slip = sum(
        qty * mid * m
        for mid, m in (
            (entry_mid, rt["entry_fill"] / entry_mid - 1 if side == "long" else 1 - rt["entry_fill"] / entry_mid),
            (exit_mid, 1 - rt["exit_fill"] / exit_mid if side == "long" else rt["exit_fill"] / exit_mid - 1),
        )
    )
    commission = 2 * cm.commission_per_share * qty
    assert entry_leg + exit_leg == pytest.approx(spread_slip + commission)
    net_usd = sign * (rt["exit_fill"] - rt["entry_fill"]) * qty - commission  # fills + commission only
    assert net_usd == pytest.approx(gross_usd - spread_slip - commission)
    assert rt["net_usd"] == pytest.approx(net_usd)
    assert rt["cost_usd"] == pytest.approx(spread_slip + commission)
    assert rt["net_return"] == pytest.approx(net_usd / (entry_mid * qty))
    assert rt["gross_return"] == pytest.approx(gross_usd / (entry_mid * qty))
    assert rt["gross_return"] - rt["net_return"] == pytest.approx(rt["cost_usd"] / (entry_mid * qty))


def test_impact_monotone():
    cm = CostModel(impact_coef=10.0)
    costs = [cm.cost("buy", 50.0, q, adv=100_000)[1] for q in (100, 1_000, 10_000, 50_000)]
    assert all(b > a for a, b in zip(costs, costs[1:], strict=False))
    # without adv, no impact term
    assert cm.cost("buy", 50.0, 1_000)[1] == CostModel().cost("buy", 50.0, 1_000)[1]
    # impact term in bps = coef * (qty/adv) ** exponent
    base = CostModel().cost("buy", 50.0, 10_000, adv=100_000)[1]
    assert costs[2] - base == pytest.approx(10.0 * 0.1**0.5)


def test_cost_units():
    usd, bps = CostModel().cost("buy", 100.0, 100)
    notional = 100.0 * 100
    assert usd == pytest.approx(notional * bps / 1e4)
    assert usd == pytest.approx(notional * ((1.0002) * (1.0005) - 1) + 0.5)
    with pytest.raises(ValueError):
        CostModel().cost("hold", 100.0, 1)
    assert CostModel().cost("buy", 100.0, 0) == (0.0, 0.0)


def test_from_tca_roundtrip(tmp_path):
    table = pd.DataFrame(
        [{"run_id": "r5-2026-q1", "half_spread_bps": 1.5, "slippage_bps": 3.5, "commission_per_share": 0.0035,
          "impact_coef": 4.0, "impact_exponent": 0.6}]
    )
    path = tmp_path / "tca.csv"
    table.to_csv(path, index=False)
    cm = CostModel.from_tca(path)
    assert cm == CostModel(1.5, 3.5, 0.0035, 4.0, 0.6, "calibrated:r5-2026-q1")
    pq = tmp_path / "tca.parquet"
    table.to_parquet(pq)
    assert CostModel.from_tca(pq) == cm


def test_from_tca_validates(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"half_spread_bps": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        CostModel.from_tca(bad)  # no run_id, so it cannot be labelled calibrated
    many = tmp_path / "many.csv"
    pd.DataFrame({"run_id": ["a", "b"], "half_spread_bps": [1.0, 2.0]}).to_csv(many, index=False)
    with pytest.raises(ValueError):
        CostModel.from_tca(many)
    assert {f.name for f in dataclasses.fields(CostModel)} >= {"source", "impact_coef"}
