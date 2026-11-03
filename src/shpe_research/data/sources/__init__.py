"""Pluggable data sources (C1g): `fixture`, `alpaca`, `store`, `snapshot`, `file`, `fred`, `edgar`."""

from shpe_research.data.sources.base import SOURCE_REGISTRY, Source, build_source, register_source

__all__ = ["SOURCE_REGISTRY", "Source", "build_source", "register_source"]
