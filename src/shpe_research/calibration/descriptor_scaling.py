"""Robust (median / IQR) scaling of market descriptors for similarity weighting.

The scaler is fit on search-stage rows only; validation rows are transformed with
the search-fit statistics and never influence them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustScaler:
    """Median/IQR statistics fit on search rows.

    Attributes:
        medians: (k,) median of every descriptor.
        iqrs: (k,) 75th minus 25th percentile (linear interpolation).
        kept: (k,) bool, True where ``iqr > iqr_floor``; other descriptors are dropped.
        n_fit_rows: Number of search rows the statistics were fit on.
    """

    medians: np.ndarray
    iqrs: np.ndarray
    kept: np.ndarray
    n_fit_rows: int


def fit_robust_scaler(search_rows: np.ndarray, *, iqr_floor: float) -> RobustScaler:
    """Fit a robust scaler on search-stage descriptor rows only.

    Args:
        search_rows: (n, k) descriptor rows from the search stage. Validation
            rows must never be passed here.
        iqr_floor: Descriptors with ``IQR <= iqr_floor`` are dropped.

    Returns:
        The fitted scaler. Pure computation over the given rows; timing is
        whatever the caller used to select the search rows (<= cutoff).

    Raises:
        ValueError: If the rows are not 2-D with at least one row, or any value is non-finite.
    """
    rows = np.asarray(search_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError(f"search_rows must be 2-D with at least one row, got shape {rows.shape}")
    if not np.all(np.isfinite(rows)):
        raise ValueError("search_rows must be finite")
    q25, q50, q75 = np.percentile(rows, [25, 50, 75], axis=0)
    iqrs = q75 - q25
    return RobustScaler(medians=q50, iqrs=iqrs, kept=iqrs > iqr_floor, n_fit_rows=int(rows.shape[0]))


def transform(scaler: RobustScaler, rows: np.ndarray) -> np.ndarray:
    """Scale rows with the fitted statistics, keeping only non-dropped descriptors.

    Args:
        scaler: Scaler fit on search rows.
        rows: (n, k) descriptor rows (search, validation or the current row).

    Returns:
        (n, kept.sum()) array ``(rows[:, kept] - median) / iqr``. Uses only the
        scaler's stored statistics, so validation rows never alter them.
    """
    x = np.asarray(rows, dtype=np.float64)
    k = scaler.kept
    return (x[:, k] - scaler.medians[k]) / scaler.iqrs[k]
