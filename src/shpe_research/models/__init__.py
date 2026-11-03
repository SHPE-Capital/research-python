"""Model implementations. Importing this package registers every built-in model."""

from shpe_research.models.logistic import LogisticModel
from shpe_research.models.random_forest import RandomForestModel, RandomForestRegressorModel

__all__ = ["LogisticModel", "RandomForestModel", "RandomForestRegressorModel"]

try:  # optional `ml` extra: a missing lightgbm only disables the LightGBM models
    import lightgbm  # noqa: F401
except ImportError:
    pass
else:
    from shpe_research.models.gradient_boosting import LightGBMModel, LightGBMRegressionModel

    __all__ += ["LightGBMModel", "LightGBMRegressionModel"]
