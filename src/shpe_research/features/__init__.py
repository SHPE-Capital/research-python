"""Point-in-time features. Importing this package registers every built-in feature."""

from shpe_research.features.base import (
    FEATURE_REGISTRY,
    Feature,
    assert_no_lookahead,
    build_feature,
    compute_features,
    register_feature,
)
from shpe_research.features.returns.past_return import PastReturn
from shpe_research.features.session.minute_of_session import MinuteOfSession
from shpe_research.features.session.opening_range import OpeningRangeBreakoutDistance, OpeningRangeWidth

__all__ = [
    "FEATURE_REGISTRY",
    "Feature",
    "MinuteOfSession",
    "OpeningRangeBreakoutDistance",
    "OpeningRangeWidth",
    "PastReturn",
    "assert_no_lookahead",
    "build_feature",
    "compute_features",
    "register_feature",
]
