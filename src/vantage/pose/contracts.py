"""Pose contracts: what a pose estimator produces, and what it deliberately does not.

A :class:`Pose` is 17 body landmarks attached to a track, in original-frame
pixel coordinates. It carries the tracker's anonymous ``entity_id`` rather than
any new notion of who the person is, so the whole privacy stance of Phase 3
carries forward unchanged: pose refines *what an entity is doing*, never *who
it is*.

What this is not
----------------
The first five COCO keypoints are head landmarks - a nose, two eyes, two ears -
and it is worth being precise about why that is not face recognition. They are
five ``(x, y)`` coordinates. No crop is retained, no texture is sampled, no
descriptor is computed, and nothing here is comparable between two frames of two
different people. A face-recognition system produces an embedding that is
matched against a gallery; this produces five points that say which way a head
is turned. For deployments that would rather not carry even that,
``pose.include_face_keypoints: false`` removes them before a
:class:`Pose` is ever constructed.

Coordinates are floats in the original frame, matching
:class:`~vantage.perception.contracts.BoundingBox`, so nothing downstream needs
to know that the estimator worked on a 192x256 crop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from vantage.perception.contracts import BoundingBox
from vantage.perception.labels import COCO_KEYPOINTS

KEYPOINT_NAMES: tuple[str, ...] = COCO_KEYPOINTS
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

FACE_KEYPOINTS: tuple[int, ...] = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
"""Head landmarks, separated out so they can be dropped as a unit."""

SKELETON: tuple[tuple[int, int], ...] = (
    (LEFT_ANKLE, LEFT_KNEE),
    (LEFT_KNEE, LEFT_HIP),
    (RIGHT_ANKLE, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_EYE, RIGHT_EYE),
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_EAR, LEFT_SHOULDER),
    (RIGHT_EAR, RIGHT_SHOULDER),
)
"""Bones, as index pairs. Drawing order only - nothing infers structure from it."""


class Posture(str, Enum):
    """Coarse body configuration.

    Deliberately coarse. These four are separable from 17 points by geometry
    alone with no training data and no per-camera calibration, which is what
    makes them honest to ship. Finer distinctions - leaning, reaching, kneeling
    versus crouching - need either temporal context or a learned classifier, and
    both belong to Phase 5 rather than being faked here.
    """

    STANDING = "standing"
    SITTING = "sitting"
    CROUCHING = "crouching"
    LYING = "lying"
    UNKNOWN = "unknown"
    """Too few of the joints the rules need were visible. An explicit outcome,
    never a silent fallback to the most likely-looking answer."""


@dataclass(frozen=True, slots=True)
class Keypoint:
    """One landmark in original-frame pixels."""

    x: float
    y: float
    confidence: float
    """The model's own score for this joint. Low means occluded, out of frame,
    or guessed - the estimator always emits all 17 points, so this is the only
    thing separating a located joint from a placeholder."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"keypoint confidence must be in [0, 1], got {self.confidence}")

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Pose:
    """A skeleton bound to one tracked entity."""

    keypoints: tuple[Keypoint, ...]
    track_id: int
    entity_id: str
    box: BoundingBox
    """The person box the estimate was cropped from - the tracker's box, which
    may be a prediction rather than an observation."""

    posture: Posture = Posture.UNKNOWN
    posture_confidence: float = 0.0
    posture_reason: str = ""
    """Why the classifier concluded what it did - and, for
    :attr:`Posture.UNKNOWN`, which joints it needed and did not get. Without
    this, "unknown" is indistinguishable from "broken", and the most common
    real case (a desk webcam that never sees anyone's legs) looks like a
    failure when it is the correct answer."""

    model: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.keypoints) not in (len(KEYPOINT_NAMES), len(KEYPOINT_NAMES) - len(FACE_KEYPOINTS)):
            raise ValueError(
                f"expected {len(KEYPOINT_NAMES)} keypoints (or "
                f"{len(KEYPOINT_NAMES) - len(FACE_KEYPOINTS)} with face landmarks "
                f"dropped), got {len(self.keypoints)}"
            )

    def __len__(self) -> int:
        return len(self.keypoints)

    def __iter__(self) -> Iterator[Keypoint]:
        return iter(self.keypoints)

    @property
    def has_face_keypoints(self) -> bool:
        return len(self.keypoints) == len(KEYPOINT_NAMES)

    def keypoint(self, index: int) -> Keypoint | None:
        """A landmark by index, or ``None`` when face points were dropped.

        Returning ``None`` rather than a zero-confidence placeholder keeps
        "this deployment does not collect head landmarks" distinguishable from
        "the head was not visible".
        """
        if not self.has_face_keypoints:
            if index in FACE_KEYPOINTS:
                return None
            index -= len(FACE_KEYPOINTS)
        return self.keypoints[index]

    def visible(self, threshold: float) -> tuple[int, ...]:
        """Indices scoring at or above ``threshold``, in full COCO numbering."""
        offset = 0 if self.has_face_keypoints else len(FACE_KEYPOINTS)
        return tuple(
            i + offset for i, kp in enumerate(self.keypoints) if kp.confidence >= threshold
        )

    @property
    def confidence(self) -> float:
        """Mean landmark score - a rough "is this a person at all" signal."""
        if not self.keypoints:
            return 0.0
        return sum(kp.confidence for kp in self.keypoints) / len(self.keypoints)

    def describe(self) -> str:
        return f"{self.entity_id} {self.posture.value} ({self.posture_confidence:.2f})"


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Every pose found in one frame.

    Mirrors :class:`~vantage.tracking.contracts.TrackingResult`: it references
    its frame by id and index rather than holding pixels, so it can be logged,
    queued or stored without keeping the image alive.
    """

    poses: tuple[Pose, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    frame_size: tuple[int, int]

    model: str = "unknown"
    backend: str = "unknown"
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    people_seen: int = 0
    """Person tracks offered to the estimator, before any budget was applied.
    With :attr:`poses` this makes a skipped person visible rather than silent."""

    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self) -> Iterator[Pose]:
        return iter(self.poses)

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def skipped(self) -> int:
        return max(0, self.people_seen - len(self.poses))

    def by_track(self) -> dict[int, Pose]:
        return {pose.track_id: pose for pose in self.poses}

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for pose in self.poses:
            tally[pose.posture.value] = tally.get(pose.posture.value, 0) + 1
        return tally

    def describe(self) -> str:
        if not self.poses:
            return "no poses"
        summary = ", ".join(f"{n}x {name}" for name, n in sorted(self.counts().items()))
        skipped = f", {self.skipped} over budget" if self.skipped else ""
        return f"{len(self.poses)} poses ({summary}){skipped} in {self.total_ms:.1f} ms"


def empty_pose_result(
    source_id: str, frame_index: int, capture_wall: float, frame_size: tuple[int, int], **kwargs
) -> PoseResult:
    """A result with no poses, for frames where estimation did not run."""
    return PoseResult(
        poses=(),
        source_id=source_id,
        frame_index=frame_index,
        capture_wall=capture_wall,
        frame_size=frame_size,
        **kwargs,
    )
