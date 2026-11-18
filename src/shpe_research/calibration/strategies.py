"""D7/D9 strategy specs: what the calibration searches and how its account differs from the base.

A ``StrategySpec`` names a strategy family's grid domain, its signal rule and its account
overrides. The weekly calibration replays every candidate through the same account engine; the
strategy namespace (``strategy_id/grid_version/config_id``) keys only the cache alias table and the
audit, while the artifact keeps the 0-B ``config_id``. D9 adds the registry: ``STRATEGY_BASES`` names
the strategy families, and ``load_strategies`` builds one spec per ``search.strategies`` entry.

A spec's ``signal_modules`` (the signal function's own module plus any ``code_modules``) are hashed
into its cache context's code version by ``ChunkReplayer.context``, so editing a strategy's signal
code invalidates its alias rows like editing the breakout signals does.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shpe_research.config.momentum import ResolvedProtocol
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
        code_modules: Further dotted modules whose source decides the mask (helpers the signal
            function calls); ``signal_mask``'s own module is always included (``signal_modules``).
    """

    strategy_id: str
    domain: GridDomain
    signal_mask: Callable[[MarketPanel, Config, IndicatorCache], SignalMask]
    account_overrides: tuple[tuple[str, OverrideValue], ...] = ()
    budget: int | None = None
    seeds: tuple[int, ...] | None = None
    code_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require ``signal_mask`` to be a module-level function, found by name in its module.

        Spawn workers unpickle it by reference and ``signal_modules`` hashes its module's source, so
        partials, callable objects, lambdas and closures are refused with TypeError.
        """
        fn = self.signal_mask
        module = sys.modules.get(getattr(fn, "__module__", None) or "")
        name = getattr(fn, "__name__", None)
        if not inspect.isfunction(fn) or module is None or name is None or getattr(module, name, None) is not fn:
            raise TypeError(
                f"strategy {self.strategy_id!r}: signal_mask must be a module-level function, got {fn!r}"
            )

    @property
    def signal_modules(self) -> tuple[str, ...]:
        """``signal_mask.__module__`` followed by ``code_modules``, deduplicated in order.

        Output: the modules whose source decides this strategy's masks; ``ChunkReplayer.context`` adds
        those not already in the replay code to the context's code version. Pure.
        """
        return tuple(dict.fromkeys((self.signal_mask.__module__, *self.code_modules)))

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


STRATEGY_BASES: Mapping[str, StrategySpec] = MappingProxyType({"breakout": BREAKOUT})


def load_strategies(rp: ResolvedProtocol) -> tuple[StrategySpec, ...]:
    """Build one ``StrategySpec`` per ``search.strategies`` entry, in protocol order.

    Input: the resolved protocol. Output: specs whose ``strategy_id`` is the entry's ``id``, whose
    domain, signal rule and code modules come from ``STRATEGY_BASES[entry.base]``, whose account
    overrides are the base's followed by the entry's (sorted by field name), and whose ``budget`` and
    ``seeds`` are the entry's (else the base's; None means the protocol's). Raises KeyError naming the
    known bases for an unknown ``base``, ValueError for an unknown ``AccountConfig`` field and
    TypeError for a value of the wrong type (each override is applied to ``AccountConfig()`` here, so
    a bad entry fails before any replay). Pure; no timing dependence.
    """
    specs = []
    for entry in rp.protocol.search.strategies:
        if entry.base not in STRATEGY_BASES:
            raise KeyError(f"strategy {entry.id!r}: unknown base {entry.base!r}; known bases: {sorted(STRATEGY_BASES)}")
        base = STRATEGY_BASES[entry.base]
        spec = dataclasses.replace(
            base,
            strategy_id=entry.id,
            account_overrides=base.account_overrides + tuple(sorted(entry.account_overrides.items())),
            budget=entry.budget if entry.budget is not None else base.budget,
            seeds=tuple(entry.seeds) if entry.seeds is not None else base.seeds,
        )
        spec.account_config(AccountConfig())
        specs.append(spec)
    return tuple(specs)
