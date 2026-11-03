"""Shared helpers for workshop notebooks: display, plotting and the real-data guard.

Nothing here computes research results; library code belongs in ``src/shpe_research``.
"""

from workshop._common.display import show_table
from workshop._common.guard import real_data_enabled, require_real_data

__all__ = ["real_data_enabled", "require_real_data", "show_table"]
