"""Perception: turning pixels into structured observations.

Phase 2 populates this package with object detection. The contracts here are
deliberately wider than detection alone, because Phase 3 tracking and Phase 4
pose will attach to the same records rather than inventing parallel ones.

The layering separates three things that change for different reasons:

``ModelAdapter``
    How a *model family* wants its input shaped and its output decoded.
    Swapping YOLOX for RT-DETR touches only an adapter.

``InferenceBackend``
    How a *runtime* executes a graph. Swapping ONNX Runtime for OpenVINO
    touches only a backend, and never the detections that come out.

``DetectionEngine``
    Composes one of each and produces :class:`DetectionResult`. This is the
    only type the rest of the platform sees.
"""

from vantage.perception.contracts import BoundingBox, Detection, DetectionResult
from vantage.perception.engine import DetectionEngine, EngineInfo, build_engine

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionEngine",
    "DetectionResult",
    "EngineInfo",
    "build_engine",
]
