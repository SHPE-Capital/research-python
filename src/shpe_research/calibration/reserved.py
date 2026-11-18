"""Reserved-period guard: nothing reads on or after ``walkforward.reserved_start`` until the study is frozen.

The reserved period opens only when the protocol is frozen (``frozen: true`` in the YAML) and the
caller supplies the expected protocol hash, which must equal the resolved protocol's hash. This is
the "test data is opened once" rule applied to the weekly calibration and its walk-forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from shpe_research.config.momentum import ResolvedProtocol


class ReservedPeriodError(RuntimeError):
    """A calibration or replay would read the reserved period without the frozen-protocol unlock."""


@dataclass(frozen=True)
class ReservedGuard:
    """The reserved-period rule for one resolved protocol.

    Attributes:
        reserved_start: First reserved session date, or None while 0-F has not fixed it.
        frozen: The protocol's ``frozen`` flag.
        protocol_hash: Hash of the resolved protocol.
        expected_hash: Hash the caller expects (``--protocol-hash``), or None.
    """

    reserved_start: date | None
    frozen: bool
    protocol_hash: str
    expected_hash: str | None

    @classmethod
    def from_protocol(cls, rp: ResolvedProtocol, expected_hash: str | None) -> ReservedGuard:
        """Build the guard from a resolved protocol and the caller's expected hash (None = no unlock)."""
        return cls(rp.protocol.walkforward.reserved_start, rp.protocol.frozen, rp.hash, expected_hash)

    def unlocked(self) -> bool:
        """True only if the protocol is frozen and ``expected_hash`` equals its hash."""
        return self.frozen and self.expected_hash is not None and self.expected_hash == self.protocol_hash

    def require_set(self) -> None:
        """Raise ReservedPeriodError if ``reserved_start`` is unset (callers that must not run without it)."""
        if self.reserved_start is None:
            raise ReservedPeriodError("walkforward.reserved_start is not set; declare it before any evaluation")

    def allows_calibration(self, cutoff: date) -> bool:
        """True if ``check_calibration(cutoff)`` would pass.

        Input: a decision cutoff. Output: False only when ``reserved_start`` is set, the cutoff is on
        or after it and the guard is not unlocked. Pure; reads no data.
        """
        return self.reserved_start is None or cutoff < self.reserved_start or self.unlocked()

    def check_calibration(self, cutoff: date) -> None:
        """Refuse a calibration whose decision cutoff is on or after ``reserved_start`` unless unlocked.

        Input: the cutoff session (every session the calibration reads is on or before it). Raises
        ReservedPeriodError; does nothing when ``reserved_start`` is None.
        """
        if not self.allows_calibration(cutoff):
            raise ReservedPeriodError(
                f"cutoff {cutoff} is in the reserved period (from {self.reserved_start}); "
                "freeze the protocol and pass its hash to read it"
            )

    def check_replay(self, sessions: Sequence[date]) -> None:
        """Refuse replaying any session on or after ``reserved_start`` unless unlocked.

        Input: the sessions about to be read. Raises ReservedPeriodError; does nothing when
        ``reserved_start`` is None.
        """
        if self.reserved_start is None or self.unlocked():
            return
        late = [s for s in sessions if s >= self.reserved_start]
        if late:
            raise ReservedPeriodError(
                f"session {min(late)} is in the reserved period (from {self.reserved_start}); "
                "freeze the protocol and pass its hash to read it"
            )
