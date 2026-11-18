"""Tests for the 0-G momentum protocol config (stream D slice)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from shpe_research.config.momentum import (
    DEFAULT_PROTOCOL_PATH,
    HASH_EXCLUDED_FIELDS,
    FrozenProtocolError,
    MomentumProtocol,
    dump_protocol_yaml,
    load_protocol,
    protocol_hash,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_match_guide() -> None:
    p = MomentumProtocol()
    assert p.search.budget == 2048
    assert p.search.seeds == [1, 2, 3]
    assert p.search.evals_split == (32, 16, 16)
    assert p.surrogate.max_iter == 200
    assert p.weights.uniform_mix == 0.20
    assert p.weights.ess_target_fraction == 0.60
    assert p.score.downside_coeff == 0.5
    assert p.validation.sessions == 63
    assert p.acceptance.search_min_trades == 100
    assert p.acceptance.max_finalists == 36
    assert p.acceptance.report_bps == 20.0
    assert p.cache.warmup_sessions == 20
    assert p.walkforward.reserved_start is None
    assert p.walkforward.starting_cash == 100_000.0


def test_default_yaml_round_trips() -> None:
    assert load_protocol(DEFAULT_PROTOCOL_PATH).protocol == MomentumProtocol()
    again = MomentumProtocol.model_validate(yaml.safe_load(dump_protocol_yaml(MomentumProtocol())))
    assert again == MomentumProtocol()


def test_misspelled_key_fails(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="budgte"):
        load_protocol(_write(tmp_path, "search: {budgte: 10}\n"))
    with pytest.raises(ValidationError, match="serach"):
        load_protocol(_write(tmp_path, "serach: {budget: 10}\n"))


def _leaves(model: BaseModel, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    out: list[tuple[tuple[Any, ...], Any]] = []
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            out += _leaves(value, (*path, name))
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            for i, item in enumerate(value):
                out += _leaves(item, (*path, name, i))
        else:
            out.append(((*path, name), value))
    return out


def _changed(value: Any) -> Any:
    if value is None:
        return None  # replaced by caller
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 0.01
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return (_changed(value[0]), *value[1:])
    if isinstance(value, list):
        return [*value, "x"] if value and isinstance(value[0], str) else [*value, 99]
    if isinstance(value, dict):
        return {**value, "cost_bps": 20.0}
    raise AssertionError(type(value))


def _replace(model: Any, path: tuple[Any, ...], value: Any) -> Any:
    head, *rest = path
    if isinstance(head, int):
        items = list(model)
        items[head] = _replace(items[head], tuple(rest), value) if rest else value
        return items
    child = getattr(model, head)
    new = _replace(child, tuple(rest), value) if rest else value
    return model.model_copy(update={head: new})


def test_every_result_field_changes_hash() -> None:
    """Every field changes the hash except the documented operational ones, which never do."""
    base = MomentumProtocol()
    base_hash = protocol_hash(base)
    leaves = _leaves(base)
    assert len(leaves) >= 45
    excluded = set(HASH_EXCLUDED_FIELDS)
    assert excluded <= {".".join(map(str, path)) for path, _ in leaves}
    for path, value in leaves:
        if value is None:
            new = date(2030, 1, 1) if path[-1] == "reserved_start" else "x"
            if path[-1] in ("budget",):
                new = 4096
            elif path[-1] == "seeds":
                new = [7]
            elif path[-1] == "max_wall_hours":
                new = 1.0
        else:
            new = _changed(value)
        changed = _replace(base, path, new)
        if ".".join(map(str, path)) in excluded:
            assert protocol_hash(changed) == base_hash, path
        else:
            assert protocol_hash(changed) != base_hash, path


def test_hash_is_stable() -> None:
    h = protocol_hash(MomentumProtocol())
    assert h == protocol_hash(MomentumProtocol())
    assert len(h) == 64
    int(h, 16)


@pytest.mark.parametrize(
    ("text", "check"),
    [
        (
            "search: {budget: 8192, seeds: [1,2,3,4,5]}\n",
            lambda p: p.search.budget == 8192 and p.search.seeds == [1, 2, 3, 4, 5],
        ),
        ("search: {finalists_per_seed: 24}\n", lambda p: p.search.finalists_per_seed == 24),
        ("validation: {sessions: 126}\n", lambda p: p.validation.sessions == 126),
        (
            "acceptance: {validation_min_trades: 50, validation_min_sessions: 30}\n",
            lambda p: p.acceptance.validation_min_trades == 50 and p.acceptance.validation_min_sessions == 30,
        ),
        ("compute: {max_wall_hours: null}\n", lambda p: p.compute.max_wall_hours is None),
        (
            "walkforward: {reserved_start: 2026-06-01}\n",
            lambda p: p.walkforward.reserved_start == date(2026, 6, 1),
        ),
    ],
)
def test_yaml_only_changes(tmp_path: Path, text: str, check: Any) -> None:
    assert check(load_protocol(_write(tmp_path, text)).protocol)


def test_overrides_applied_and_recorded() -> None:
    r = load_protocol(None, ["search.budget=4096", "acceptance.max_drawdown=0.2"])
    assert r.protocol.search.budget == 4096
    assert r.protocol.acceptance.max_drawdown == 0.2
    assert r.overrides == ("search.budget=4096", "acceptance.max_drawdown=0.2")
    assert r.hash != protocol_hash(MomentumProtocol())
    assert r.hash == protocol_hash(r.protocol)
    assert r.source == "<defaults>"


def test_overrides_validated_once() -> None:
    ov = [
        "search.budget=24",
        "search.initial_samples=12",
        "search.evals_per_round=6",
        "search.evals_split=[2,2,2]",
    ]
    fwd = load_protocol(None, ov).protocol
    assert fwd.search.budget == 24
    assert load_protocol(None, ov[::-1]).protocol == fwd


def test_override_unknown_path_fails() -> None:
    with pytest.raises(ValueError, match=r"search\.nope"):
        load_protocol(None, ["search.nope=1"])


def test_overrides_refused_when_frozen(tmp_path: Path) -> None:
    with pytest.raises(FrozenProtocolError):
        load_protocol(_write(tmp_path, "frozen: true\n"), ["search.budget=4096"])
    with pytest.raises(FrozenProtocolError):
        load_protocol(None, ["frozen=true"])


def test_validators() -> None:
    p = MomentumProtocol()
    s = p.search
    with pytest.raises(ValidationError):
        s.model_validate({**s.model_dump(), "evals_split": (32, 16, 15)})
    with pytest.raises(ValidationError):
        s.model_validate({**s.model_dump(), "budget": 100})
    with pytest.raises(ValidationError):
        s.model_validate({**s.model_dump(), "seeds": [1, 1]})
    with pytest.raises(ValidationError):
        s.model_validate({**s.model_dump(), "strategies": [{"id": "a"}, {"id": "a"}]})


def test_operational_fields_stay_out_of_the_hash() -> None:
    base = load_protocol(None)
    assert set(HASH_EXCLUDED_FIELDS) == {
        "compute.workers",
        "compute.threads_per_worker",
        "compute.scratch_dir",
        "compute.max_wall_hours",
        "cache.path",
        "cache.busy_timeout_s",
    }
    same = ["compute.workers=8", "compute.max_wall_hours=2.0", "cache.path=/scratch/c.sqlite"]
    assert load_protocol(None, same).hash == base.hash
    changed = ("search.budget=4096", "acceptance.max_drawdown=0.2", "weights.uniform_mix=0.3", "compute.kernel=oracle")
    for item in changed:
        assert load_protocol(None, [item]).hash != base.hash, item


def test_descriptor_lookback_needs_two() -> None:
    with pytest.raises(ValidationError, match="descriptor_lookback"):
        load_protocol(None, ["weights.descriptor_lookback=1"])
    assert load_protocol(None, ["weights.descriptor_lookback=2"]).protocol.weights.descriptor_lookback == 2


def test_walkforward_policies_non_empty_and_unique() -> None:
    with pytest.raises(ValidationError, match="walkforward.policies"):
        load_protocol(None, ["walkforward.policies=[]"])
    with pytest.raises(ValidationError, match="walkforward.policies"):
        load_protocol(None, ["walkforward.policies=[uniform, uniform]"])
