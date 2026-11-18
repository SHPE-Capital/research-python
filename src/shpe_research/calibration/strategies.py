"""D7/D9 strategy specs: what the calibration searches and how its account differs from the base.

A ``StrategySpec`` names a strategy family's grid domain, its signal rule and its account
overrides. The weekly calibration replays every candidate through the same account engine; the
strategy namespace (``strategy_id/grid_version/config_id``) keys only the cache alias table and the
audit, while the artifact keeps the 0-B ``config_id``. Task 11 (D9) extends this module with the
registry built from the protocol's ``search.strategies``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from shpe_research.strategies.momentum.breakout import signals
from shpe_research.strategies.momentum.breakout.grid import DEFAULT_DOMAIN, Config, GridDomain, config_id
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.trade.account_types import AccountConfig, MarketPanel, SignalMask

OverrideValue = float | int | str


def _coerce(current: object, value: OverrideValue, key: str) -> object:
    """Match ``value`` to the type of the field it replaces so equal settings hash equally."""
    if isinstance(value, bool) or isinstance(current, bool):
        raise TypeError(f"account override {key!r}: booleans are not supported")
    if isinstance(current, float) and isinstance(value, int):
        return float(value)
    if isinstance(current, int) and isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"account override {key!r} needs an integer, got {value!r}")
        return int(value)
    if current is not None and not isinstance(value, type(current)):
        raise TypeError(f"account override {key!r} needs {type(current).__name__}, got {value!r}")
    return value


def _replace_path(obj: object, path: list[str], value: OverrideValue, key: str) -> object:
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise ValueError(f"account override {key!r}: {type(obj).__name__} has no field {path[0]!r}")
    names = {f.name for f in dataclasses.fields(obj)}
    head = path[0]
    if head not in names:
        raise ValueError(f"account override {key!r}: unknown field {head!r} on {type(obj).__name__}")
    current = getattr(obj, head)
    new = _coerce(current, value, key) if len(path) == 1 else _replace_path(current, path[1:], value, key)
    return dataclasses.replace(obj, **{head: new})


@dataclass(frozen=True)
class StrategySpec:
    """One searchable strategy family.

    Attributes:
        strategy_id: Namespace of the strategy (``[a-z0-9_]+``).
        domain: The versioned grid the search draws configs from.
        signal_mask: ``(panel, config, indicator_cache) -> SignalMask``; must be a module-level
            function (spawn workers unpickle it) whose ATR array is the cache's ATR14.
        account_overrides: ``(field, value)`` pairs applied to the base account; nested fields use
            dotted names such as ``"exit.initial_stop_atr"``.
        budget: Per-strategy search budget, or None for the protocol's.
        seeds: Per-strategy seeds, or None for the protocol's.
    """

    strategy_id: str
    domain: GridDomain
    signal_mask: Callable[[MarketPanel, Config, IndicatorCache], SignalMask]
    account_overrides: tuple[tuple[str, OverrideValue], ...] = ()
    budget: int | None = None
    seeds: tuple[int, ...] | None = None

    @property
    def grid_version(self) -> str:
        """Version of ``domain``; part of every namespaced id."""
        return self.domain.version

    def namespaced_id(self, config: Config) -> str:
        """Return ``f"{strategy_id}/{grid_version}/{config_id(config)}"``.

        Input: a config of this strategy's grid. Output: the alias id used by the cache's alias table
        and the audit (never the artifact's ``selected.config_id``). Pure; no timing dependence.
        """
        return f"{self.strategy_id}/{self.grid_version}/{config_id(config)}"

    def account_config(self, base: AccountConfig) -> AccountConfig:
        """Return ``base`` with this strategy's overrides applied (``dataclasses.replace``, nested included).

        Input: the base replay account. Output: a new ``AccountConfig``; an int given for a float field
        is stored as float (and an integral float for an int field as int) so equal settings give equal
        ``account_hash`` values. Raises ValueError for an unknown field and TypeError for a value of the
        wrong type. Pure; no timing dependence.
        """
        out: object = base
        for key, value in self.account_overrides:
            out = _replace_path(out, key.split("."), value, key)
        assert isinstance(out, AccountConfig)
        return out


BREAKOUT = StrategySpec("breakout", DEFAULT_DOMAIN, signals.signal_mask)
