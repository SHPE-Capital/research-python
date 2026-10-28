"""Versioned contracts every layer agrees on. See docs/contracts.md.

A change that breaks a contract bumps CONTRACT_VERSION, which is stamped on
stored data, snapshots and model manifests so mismatches are caught on load.
"""

from shpe_research.contracts.base import ContractError
from shpe_research.contracts.version import CONTRACT_VERSION

__all__ = ["CONTRACT_VERSION", "ContractError"]
