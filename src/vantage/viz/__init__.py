"""Diagnostic visualisation.

Strictly a *diagnostic* surface: it renders what ingestion measured and nothing
else. It contains no analysis logic, and the pipeline runs identically with the
display disabled - which is how tests and headless deployments run it.

The Phase 9 dashboard is a separate concern and will consume observations over
an API rather than extending this window.
"""

from vantage.viz.hud import HudRenderer
from vantage.viz.overlay import class_color, draw_detections
from vantage.viz.window import FrameSink, NullSink, WindowSink

__all__ = [
    "FrameSink",
    "HudRenderer",
    "NullSink",
    "WindowSink",
    "class_color",
    "draw_detections",
]
