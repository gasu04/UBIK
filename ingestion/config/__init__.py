"""UBIK ingestion configuration package. Public API: ingestion_config."""

from .ingestion_config import (
    EndpointConfig,
    GateThresholds,
    IngestionConfig,
    PathsConfig,
    load_config,
)

__all__ = [
    "EndpointConfig",
    "GateThresholds",
    "IngestionConfig",
    "PathsConfig",
    "load_config",
]
