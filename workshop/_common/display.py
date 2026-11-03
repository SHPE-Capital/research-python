"""Display helpers for notebooks. Pure presentation, no research logic."""

from __future__ import annotations

import pandas as pd


def show_table(frame: pd.DataFrame, title: str | None = None, digits: int = 4) -> None:
    """Print ``frame`` rounded to ``digits`` places, with an optional title.

    Works the same as a script and in a notebook cell (plain text, no Jupyter dependency).
    """
    if title:
        print(f"\n{title}\n{'-' * len(title)}")
    print(frame.round(digits).to_string())
