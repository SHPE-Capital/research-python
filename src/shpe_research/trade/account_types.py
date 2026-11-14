"""Account simulator I/O contract (task 0-C): dataclasses and the ``simulate_account`` stub.

Types only. The simulator itself is MO-C4. Defaults reproduce the momentum guide; the optional
fields (ranking name, entry-score threshold, size multiplier, search-replay reference equity) are
the seams for stream F and are inert at their defaults.

Array layout (``MarketPanel``): one week, shape ``(n_symbols, n_minutes)``, C-contiguous, so a
panel can be written with ``np.save`` and re-opened with ``np.load(mmap_mode="r")``. Prices and
volume are float32-friendly; the contract does not force a dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExitPackage:
    """ATR-based exit rules (config keys ``exit.*``)."""

    initial_stop_atr: float = 1.5
    trail_activation_atr: float = 1.0
    trail_distance_atr: float = 1.5
    max_hold_minutes: int = 60
    session_exit_minutes_before_close: int = 5


@dataclass(frozen=True)
class FillModel:
    """Entry fill model (config key ``fill.limit_markup``)."""

    kind: str = "limit_markup"
    limit_markup: float = 0.0020


@dataclass(frozen=True)
class AccountConfig:
    """Account rules. Built from the 0-G protocol config; defaults equal the guide."""

    starting_cash: float = 100_000.0
    max_positions: int = 10
    per_stock_pct: float = 0.08
    risk_pct: float = 0.0025
    total_exposure_cap: float = 0.80
    liquidity_cap_pct: float = 0.01
    liquidity_lookback: int = 20
    liquidity_min_obs: int = 15
    daily_halt_pct: float = 0.01
    exit: ExitPackage = field(default_factory=ExitPackage)
    cost_bps: float = 10.0
    fill: FillModel = field(default_factory=FillModel)
    # Seams for stream F; defaults reproduce the guide exactly.
    ranking: str = "dollar_volume_20d"
    entry_score_threshold: float | None = None
    size_multiplier: float | None = None
    search_replay_reference_equity: float | None = None


@dataclass(frozen=True)
class MarketPanel:
    """One week of market data, symbol x minute, memory-map friendly.

    ``asset_ids`` are stable identifiers (row i of every array is ``asset_ids[i]``).
    ``session_index`` maps each minute column to a session number (non-decreasing, int).
    ``eligible`` is the weekly universe mask (symbol x minute, bool); NaN prices mark missing bars.
    """

    asset_ids: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    session_index: np.ndarray
    eligible: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.asset_ids), len(self.session_index))
        for name in ("open", "high", "low", "close", "volume", "eligible"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
        if self.eligible.dtype != np.bool_:
            raise ValueError("eligible must be a bool array")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique")
        if self.session_index.ndim != 1 or (np.diff(self.session_index) < 0).any():
            raise ValueError("session_index must be 1-D and non-decreasing")

    @property
    def n_symbols(self) -> int:
        return len(self.asset_ids)

    @property
    def n_minutes(self) -> int:
        return len(self.session_index)


@dataclass(frozen=True)
class SignalMask:
    """Entry signals for one config: bool symbol x minute plus ATR at each minute."""

    signal: np.ndarray
    atr: np.ndarray

    def __post_init__(self) -> None:
        if self.signal.dtype != np.bool_:
            raise ValueError("signal must be a bool array")
        if self.signal.shape != self.atr.shape:
            raise ValueError(f"signal {self.signal.shape} and atr {self.atr.shape} shapes differ")


@dataclass(frozen=True)
class LedgerEntry:
    """One closed (or force-flagged) trade."""

    asset_id: str
    entry_minute: int
    exit_minute: int
    qty: int
    entry_price: float
    exit_price: float
    exit_reason: str
    net_pnl: float


@dataclass(frozen=True)
class Rejection:
    """A signal that did not become a position, with the reason."""

    asset_id: str
    minute: int
    reason: str


@dataclass(frozen=True)
class AccountResult:
    """Simulator output. ``daily_returns`` has one net return per session, zero-trade days included."""

    daily_returns: np.ndarray
    ledger: tuple[LedgerEntry, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    ambiguous_bars: int = 0
    unresolved_positions: int = 0
    summary: dict[str, float] = field(default_factory=dict)


def simulate_account(panel: MarketPanel, mask: SignalMask, cfg: AccountConfig) -> AccountResult:
    """Run the account over one week. Stub: implemented in MO-C4."""
    raise NotImplementedError("TODO(MO-C4)")
