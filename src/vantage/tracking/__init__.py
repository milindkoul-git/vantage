"""Multi-object tracking: turning per-frame detections into persistent objects.

Importing this package pulls in numpy and the perception contracts, but no
inference runtime and no model - tracking is testable, and tunable, with no
hardware at all.
"""

from vantage.tracking.base import Tracker
from vantage.tracking.bytetrack import ByteTracker, TrackerParams
from vantage.tracking.contracts import (
    Track,
    TrackingResult,
    TrackState,
    empty_tracking_result,
)
from vantage.tracking.kalman import MotionNoise

__all__ = [
    "ByteTracker",
    "MotionNoise",
    "Track",
    "TrackState",
    "Tracker",
    "TrackerParams",
    "TrackingResult",
    "empty_tracking_result",
]
