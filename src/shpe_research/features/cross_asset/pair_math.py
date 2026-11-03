"""Shared pair estimators: hedge ratios (SS2) and Ornstein-Uhlenbeck fits (SS3).

They live at the feature level so that both `features/cross_asset/pair_spread.py`
(F7) and `strategies/statarb/` can import them without an upward import.
SS2 and SS3 each append only under their own section header below and never
edit the other's; whoever merges second keeps both blocks.
"""

# --- Hedge ratios (SS2) -------------------------------------------------------


# --- Ornstein-Uhlenbeck fit and half-life (SS3) -------------------------------
