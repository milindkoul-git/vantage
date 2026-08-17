"""Configuration: typed schema plus a YAML/CLI loader."""

from vantage.config.loader import default_config_path, load_config
from vantage.config.schema import (
    AppConfig,
    Backpressure,
    DetectionConfig,
    DisplayConfig,
    IngestConfig,
    IngestMode,
    ReconnectConfig,
    SourceConfig,
    VantageConfig,
)

__all__ = [
    "AppConfig",
    "Backpressure",
    "DetectionConfig",
    "DisplayConfig",
    "IngestConfig",
    "IngestMode",
    "ReconnectConfig",
    "SourceConfig",
    "VantageConfig",
    "default_config_path",
    "load_config",
]
