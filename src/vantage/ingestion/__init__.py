"""Video ingestion: sources, pacing, backpressure and the capture pipeline.

The public surface every later phase depends on:

- :class:`~vantage.ingestion.base.FrameSource` - where pixels come from.
- :class:`~vantage.core.frame.Frame` - what crosses the boundary.
- :class:`~vantage.ingestion.pipeline.IngestionPipeline` - how frames are
  delivered, paced, buffered and measured.

A perception stage should only ever need the middle one.
"""

from vantage.ingestion.base import FrameSource, SourceInfo, SourceKind, SourceState
from vantage.ingestion.pipeline import IngestionPipeline, PipelineStats
from vantage.ingestion.registry import create_source, describe_schemes

__all__ = [
    "FrameSource",
    "IngestionPipeline",
    "PipelineStats",
    "SourceInfo",
    "SourceKind",
    "SourceState",
    "create_source",
    "describe_schemes",
]
