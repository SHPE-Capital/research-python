"""Cross-sectional transforms (C1d). Importing this package registers the built-in transforms."""

from shpe_research.transforms.base import (
    TRANSFORM_REGISTRY,
    Transform,
    apply_transforms,
    build_transform,
    register_transform,
)
from shpe_research.transforms.cross_sectional import Demean, Rank, Winsorize, ZScore
from shpe_research.transforms.neutralize import Neutralize

__all__ = [
    "TRANSFORM_REGISTRY",
    "Demean",
    "Neutralize",
    "Rank",
    "Transform",
    "Winsorize",
    "ZScore",
    "apply_transforms",
    "build_transform",
    "register_transform",
]
