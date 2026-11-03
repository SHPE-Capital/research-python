"""neutralize: remove exposures (beta, sector dummies) from a score by a per-timestamp regression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shpe_research.transforms.base import TIME_KEY, Transform, register_transform


@register_transform
@dataclass(frozen=True)
class Neutralize(Transform):
    """x - proportion * (exposures @ slopes), the slopes from an OLS of x on the exposures plus an intercept.

    The regression is fitted separately at each `decision_ts` on that timestamp's rows with no NaN in x or
    any exposure, so nothing is learned across time. Exposures are centred within the timestamp before the
    fit, which makes the removed part (the fitted values without the intercept) unique even when sector
    dummies are collinear with the intercept (a least-squares solve handles the rank deficiency). With
    `proportion=1` the exposures' cross-sectional correlation with the output is zero; `0` returns x.
    Rows with NaN in x or an exposure, and timestamps with fewer than `min_count` complete rows, give NaN.
    """

    transform_type = "neutralize"
    column: str
    exposures: list[str] = field(default_factory=list)
    proportion: float = 1.0
    min_count: int = 10

    def __post_init__(self) -> None:
        if not self.exposures:
            raise ValueError("neutralize needs at least one exposure column")
        if not 0.0 <= self.proportion <= 1.0:
            raise ValueError("proportion must be in [0, 1]")

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        data = frame[[self.column, *self.exposures]].astype("float64")
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        complete = data.notna().all(axis=1)
        for idx in frame[complete].groupby(TIME_KEY, sort=False).groups.values():
            if len(idx) < self.min_count:
                continue
            block = data.loc[idx]
            x = block[self.column].to_numpy()
            exposure = block[self.exposures].to_numpy()
            centred = exposure - exposure.mean(axis=0)
            slopes, *_ = np.linalg.lstsq(centred, x - x.mean(), rcond=None)
            out.loc[idx] = x - self.proportion * (centred @ slopes)
        return out
