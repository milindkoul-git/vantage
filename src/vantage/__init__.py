"""Vantage - a modular intelligent video analytics platform.

Phase 1 provides the video ingestion subsystem: sources, framing, pacing,
backpressure, metrics and a diagnostic viewer. Perception stages (detection,
tracking, pose, events) attach to the ingestion pipeline in later phases
without modifying anything in :mod:`vantage.ingestion`.
"""

from vantage.core.frame import Frame

__version__ = "0.1.0"
__all__ = ["Frame", "__version__"]
