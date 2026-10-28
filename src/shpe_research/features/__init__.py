"""Point-in-time features. Importing this package registers every built-in feature."""

from shpe_research.features.base import (
    FEATURE_REGISTRY,
    Feature,
    assert_no_lookahead,
    build_feature,
    compute_features,
    register_feature,
)

__all__ = [
    "FEATURE_REGISTRY",
    "Feature",
    "assert_no_lookahead",
    "build_feature",
    "compute_features",
    "register_feature",
]
