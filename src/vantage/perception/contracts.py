"""What perception produces.

These are the records every later phase consumes, so a few conventions are
fixed here once and relied on everywhere:

Coordinates
    Always pixels in the **original frame**, never in the model's letterboxed
    input space and never normalised. A consumer must never need to know that
    the detector resized anything. Undoing the letterbox is the adapter's job
    and it happens before a :class:`Detection` exists.

No pixels
    :class:`DetectionResult` references its frame by id and index rather than
    holding the image. Detections outlive frames - Phase 8 will store them -
    and a result that pinned a 6 MB buffer would quietly exhaust memory on a
    machine with 3 GB free.

No identity
    A :class:`Detection` has a class, not an entity. Persistent anonymous
    identity (``person_17``) is assigned by tracking in Phase 3, and identity
    resolution is a separate optional subsystem after that. Detection stays
    ignorant of both.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in original-frame pixel coordinates, ``x1 <= x2``."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"box corners are inverted: ({self.x1}, {self.y1})-({self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """``(x, y, width, height)`` with the origin at the top-left corner."""
        return self.x1, self.y1, self.width, self.height

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the object meets the ground plane.

        The right anchor for zone membership and trajectory work in Phase 6 -
        a person's box centre drifts with posture, their feet do not.
        """
        return (self.x1 + self.x2) / 2.0, self.y2

    def to_int(self) -> tuple[int, int, int, int]:
        """Rounded ``(x1, y1, x2, y2)``, for drawing and cropping."""
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )

    def clipped(self, width: int, height: int) -> BoundingBox:
        """Clamp to a frame of ``width`` x ``height``.

        Detectors routinely predict boxes that run past the frame edge for
        partially visible objects; cropping or drawing those unclipped raises
        errors far from the cause.
        """
        return BoundingBox(
            x1=min(max(self.x1, 0.0), float(width)),
            y1=min(max(self.y1, 0.0), float(height)),
            x2=min(max(self.x2, 0.0), float(width)),
            y2=min(max(self.y2, 0.0), float(height)),
        )

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with ``other``; ``0.0`` when disjoint."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    """One object found in one frame."""

    box: BoundingBox
    class_id: int
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"class_id must be non-negative, got {self.class_id}")

    def describe(self) -> str:
        x1, y1, x2, y2 = self.box.to_int()
        return f"{self.label}@{self.confidence:.2f} [{x1},{y1},{x2},{y2}]"


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything one detector pass produced, plus how long it took."""

    detections: tuple[Detection, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    frame_size: tuple[int, int]
    """``(width, height)`` of the frame the boxes are expressed in."""

    model: str = "unknown"
    backend: str = "unknown"
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    @property
    def total_ms(self) -> float:
        """Wall time for the whole pass. The number that governs achievable FPS."""
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def of_class(self, *labels: str) -> tuple[Detection, ...]:
        wanted = {label.lower() for label in labels}
        return tuple(d for d in self.detections if d.label.lower() in wanted)

    def above(self, confidence: float) -> tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.confidence >= confidence)

    def counts(self) -> dict[str, int]:
        """Detections per label, for HUD summaries and event rules later."""
        tally: dict[str, int] = {}
        for detection in self.detections:
            tally[detection.label] = tally.get(detection.label, 0) + 1
        return tally

    def describe(self) -> str:
        if not self.detections:
            return f"{self.source_id}#{self.frame_index}: nothing detected"
        summary = ", ".join(
            f"{count}x {label}" for label, count in sorted(self.counts().items())
        )
        return f"{self.source_id}#{self.frame_index}: {summary} ({self.total_ms:.1f} ms)"


EMPTY_RESULT_METADATA: dict[str, object] = {}


def empty_result(
    source_id: str, frame_index: int, capture_wall: float, frame_size: tuple[int, int], **kwargs
) -> DetectionResult:
    """A result with no detections.

    Used when detection is skipped for a frame (see ``detection.interval``).
    Explicitly empty beats ``None``: consumers keep one code path, and "we
    looked and found nothing" stays distinguishable from "we did not look" via
    :attr:`DetectionResult.metadata`.
    """
    return DetectionResult(
        detections=(),
        source_id=source_id,
        frame_index=frame_index,
        capture_wall=capture_wall,
        frame_size=frame_size,
        **kwargs,
    )
