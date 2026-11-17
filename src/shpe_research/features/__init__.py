"""Point-in-time features. Importing this package registers every built-in feature."""

from shpe_research.features.base import (
    FEATURE_REGISTRY,
    Feature,
    assert_no_lookahead,
    build_feature,
    compute_features,
    register_feature,
)
from shpe_research.features.daily.har import HarRv
from shpe_research.features.daily.returns import Momentum121, PastReturnDays
from shpe_research.features.microstructure.order_flow import SignedVolumeRatio
from shpe_research.features.regime.regime import RegimeFeature
from shpe_research.features.returns.past_return import PastReturn
from shpe_research.features.session.minute_of_session import MinuteOfSession
from shpe_research.features.session.opening_range import OpeningRangeBreakoutDistance, OpeningRangeWidth
from shpe_research.features.volatility.realized_vol import RealizedVol
from shpe_research.features.volume.relative_volume import RelativeVolume
from shpe_research.features.volume.vwap_distance import VwapDistance

__all__ = [
    "FEATURE_REGISTRY",
    "Feature",
    "HarRv",
    "MinuteOfSession",
    "Momentum121",
    "OpeningRangeBreakoutDistance",
    "OpeningRangeWidth",
    "PastReturn",
    "PastReturnDays",
    "RelativeVolume",
    "RealizedVol",
    "RegimeFeature",
    "SignedVolumeRatio",
    "VwapDistance",
    "assert_no_lookahead",
    "build_feature",
    "compute_features",
    "register_feature",
]
