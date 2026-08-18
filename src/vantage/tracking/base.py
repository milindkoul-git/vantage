"""The tracker seam.

One Protocol, so the association strategy is replaceable without any consumer
knowing which one is running. ByteTrack is the Phase 3 implementation, but the
choice is not permanent: an appearance-based tracker, a global/offline tracker
for batch analysis, or a cheaper centroid tracker for very constrained hardware
all fit behind this interface. Keeping the seam explicit now is what makes that
a configuration change later rather than a rewrite.

The interface is deliberately narrow. A tracker consumes detections and time,
and produces tracks. It does not see pixels, which means:

* No appearance model can be added by accident. Anything that extracted visual
  features would need to change this signature, which is a review-visible act
  rather than a quiet import - and appearance features on people are the first
  step toward re-identification, which this phase must not perform.
* Tracking is testable with no images, no model and no camera, which is what
  makes the evaluation harness in :mod:`vantage.tracking.evaluation` possible.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from vantage.core.frame import Frame
from vantage.perception.contracts import Detection, DetectionResult
from vantage.tracking.contracts import TrackingResult


@runtime_checkable
class Tracker(Protocol):
    """Maintains object identity across time."""

    def update(self, result: DetectionResult, *, frame: Frame | None = None) -> TrackingResult:
        """Advance the tracker by one step and return the current tracks.

        Args:
            result: Detections for this step, in original-frame pixels.
            frame: The frame they came from, when available. Used only for its
                timestamps - the elapsed time since the previous step drives the
                motion model - never for its pixels.
        """
        ...

    def reset(self) -> None:
        """Drop all state. Used when a source reconnects and continuity is lost."""
        ...

    @property
    def active_tracks(self) -> Sequence[object]:
        """Tracks currently maintained, including unpublished tentative ones."""
        ...


def detections_of(result: DetectionResult) -> list[Detection]:
    """Detections as a plain list, the form the association code wants."""
    return list(result.detections)
