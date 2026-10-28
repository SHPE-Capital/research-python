"""TODO(D6): corporate actions (splits, symbol changes).  Level: beginner-intermediate.

D2's audit flags overnight gaps that look like splits; this task confirms them.

    - Fetch splits and symbol changes from Alpaca's corporate actions endpoint
      (GET https://data.alpaca.markets/v1/corporate-actions with types
      forward_split, reverse_split, name_change) using data/alpaca.py's client.
    - Store them as Parquet under <data_dir>/corporate_actions/.
    - Let audit.py mark a flagged gap "explained" when a split matches it.
    - Provide split_factor(symbol, start, end) so features that compare prices
      across days can adjust raw bars consistently (raw prices stay untouched in
      the store).

Done when:
    - tests/todo/test_data_todo.py::test_corporate_actions passes with a mocked response.
    - The audit report lists explained vs. unexplained gaps.
"""

from __future__ import annotations


def fetch_corporate_actions(symbols: list[str], start: str, end: str):
    raise NotImplementedError("TODO(D6): corporate actions")
