"""Scaffolding for `shpe new <kind> <target>` (C1j). Library code never imports this package."""

from shpe_research.scaffold.new import ScaffoldError, list_kinds, scaffold

__all__ = ["ScaffoldError", "list_kinds", "scaffold"]
