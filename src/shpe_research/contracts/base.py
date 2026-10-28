"""Shared validation helpers for the contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

UTC_NS = "datetime64[ns, UTC]"


def to_utc_ns(values) -> pd.Series:
    """Any timestamps (ISO/RFC-3339 strings of mixed precision, datetimes, other units) as UTC nanoseconds."""
    return pd.to_datetime(values, utc=True, format="ISO8601").astype(UTC_NS)


class ContractError(ValueError):
    """Data that does not satisfy a contract. Lists every problem found, not just the first."""

    def __init__(self, contract: str, problems: Iterable[str]):
        self.contract = contract
        self.problems = list(problems)
        super().__init__(f"{contract} contract violated:\n  - " + "\n  - ".join(self.problems))


# Nanosecond unit is part of the contract: pandas 3 infers microseconds from
# strings, and joins (merge_asof) between mismatched units fail.
_KINDS: dict[str, tuple[Callable[[pd.Series], bool], str]] = {
    "utc": (lambda c: str(c.dtype) == UTC_NS, UTC_NS),
    "float": (pd.api.types.is_float_dtype, "float"),
    "int": (
        lambda c: pd.api.types.is_integer_dtype(c) and not pd.api.types.is_extension_array_dtype(c),
        "int64",
    ),
    "nullable_int": (pd.api.types.is_integer_dtype, "a (nullable) integer"),
    "str": (lambda c: pd.api.types.is_string_dtype(c) or pd.api.types.is_object_dtype(c), "string"),
    "list": (pd.api.types.is_object_dtype, "lists (object)"),
    "any": (lambda c: True, "anything"),
}


def check_columns(df: pd.DataFrame, expected: dict[str, str]) -> list[str]:
    """Problems with missing columns or wrong dtypes.

    `expected` maps column name to a kind: "utc" (UTC nanosecond datetime),
    "float", "int", "nullable_int", "str", "list" or "any".
    """
    problems: list[str] = []
    for name, kind in expected.items():
        if name not in df.columns:
            problems.append(f"missing column {name!r}")
            continue
        accepts, description = _KINDS[kind]
        if not accepts(df[name]):
            problems.append(f"{name!r} must be {description}, got {df[name].dtype}")
    return problems
