"""The pose engine: tracks in, skeletons out.

Sits downstream of tracking rather than beside detection, and that ordering is
the design. Running pose on raw detections would mean re-estimating an
unidentified body every frame; running it on tracks means each skeleton arrives
already attached to a stable anonymous ``entity_id``, so a later phase can ask
what *this* entity has been doing rather than what some person-shaped box in
frame 412 was doing.

Cost is linear in people, so the budget is explicit
---------------------------------------------------
One pass per person, measured at 3.5 ms on the iGPU and 8.9 ms on this CPU. Four
people is 14 ms of a 33 ms frame; twelve would blow the budget entirely. Rather
than let frame rate collapse silently as a room fills,
:attr:`PoseEngine.max_persons` caps the work and :attr:`PoseResult.people_seen`
records how many were offered, so a skipped person is visible in the result and
on the HUD instead of being quietly dropped.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import ConfigError
from vantage.core.frame import Frame
from vantage.core.logging import get_logger
from vantage.perception.backends.base import InferenceBackend
from vantage.pose.adapter import RTMPoseAdapter
from vantage.pose.contracts import Pose, PoseResult
from vantage.pose.posture import classify
from vantage.tracking.contracts import Track, TrackingResult

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PoseEngineInfo:
    """What the engine resolved to, for the HUD, logs and reports."""

    model: str
    backend: str
    device: str
    input_size: tuple[int, int]
    precision: str
    license: str
    num_keypoints: int

    def describe(self) -> str:
        height, width = self.input_size
        return (
            f"{self.model} ({width}x{height}, {self.num_keypoints} keypoints, "
            f"{self.license}) on {self.backend}/{self.device}"
        )


class PoseEngine:
    """Estimates body pose for the people a tracker is following."""

    def __init__(
        self,
        adapter: RTMPoseAdapter,
        backend: InferenceBackend,
        *,
        model_name: str = "unknown",
        license: str = "unknown",
        min_keypoint_confidence: float = 0.3,
        max_persons: int = 6,
        person_labels: Sequence[str] = ("person",),
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        if not 0.0 <= min_keypoint_confidence < 1.0:
            raise ConfigError(
                f"pose.min_keypoint_confidence must be in [0, 1), got {min_keypoint_confidence}"
            )
        if max_persons < 1:
            raise ConfigError(f"pose.max_persons must be >= 1, got {max_persons}")
        if not person_labels:
            raise ConfigError("pose.classes must name at least one label")

        self._adapter = adapter
        self._backend = backend
        self._min_keypoint_confidence = min_keypoint_confidence
        self._max_persons = max_persons
        self._labels = tuple(label.strip().lower() for label in person_labels if label.strip())
        self._clock = clock
        self._closed = False
        self._info = PoseEngineInfo(
            model=model_name,
            backend=backend.info.name,
            device=backend.info.device,
            input_size=adapter.input_size,
            precision=backend.info.precision,
            license=license,
            num_keypoints=len(adapter.labels),
        )

    @property
    def info(self) -> PoseEngineInfo:
        return self._info

    @property
    def max_persons(self) -> int:
        return self._max_persons

    @property
    def min_keypoint_confidence(self) -> float:
        return self._min_keypoint_confidence

    # -- estimation -----------------------------------------------------

    def estimate(self, frame: Frame, tracking: TrackingResult) -> PoseResult:
        """Estimate pose for every person track this frame can afford."""
        if self._closed:
            raise RuntimeError("pose engine has been closed")

        candidates = self._select(tracking)
        preprocess_ms = inference_ms = postprocess_ms = 0.0
        poses: list[Pose] = []

        for track in candidates[: self._max_persons]:
            start = self._clock.monotonic()
            prepared = self._adapter.preprocess(frame.image, track.box)
            after_pre = self._clock.monotonic()

            outputs = self._backend.run(prepared.tensor)
            after_infer = self._clock.monotonic()

            keypoints = self._adapter.postprocess(outputs, prepared)
            pose = Pose(
                keypoints=tuple(keypoints),
                track_id=track.track_id,
                entity_id=track.entity_id,
                box=track.box,
                model=self._info.model,
            )
            estimate = classify(pose, self._min_keypoint_confidence)
            poses.append(
                Pose(
                    keypoints=pose.keypoints,
                    track_id=pose.track_id,
                    entity_id=pose.entity_id,
                    box=pose.box,
                    posture=estimate.posture,
                    posture_confidence=estimate.confidence,
                    posture_reason=estimate.reason,
                    model=pose.model,
                )
            )
            after_post = self._clock.monotonic()

            preprocess_ms += (after_pre - start) * 1000.0
            inference_ms += (after_infer - after_pre) * 1000.0
            postprocess_ms += (after_post - after_infer) * 1000.0

        return PoseResult(
            poses=tuple(poses),
            source_id=frame.source_id,
            frame_index=frame.index,
            capture_wall=frame.capture_wall,
            frame_size=frame.resolution,
            model=self._info.model,
            backend=f"{self._info.backend}/{self._info.device}",
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            people_seen=len(candidates),
        )

    def _select(self, tracking: TrackingResult) -> list[Track]:
        """People worth estimating, largest first.

        Two filters, both load-bearing:

        * **Observed this step only.** A coasting track's box is a motion
          prediction for something currently hidden. Cropping to it yields
          whatever is actually there - a wall, the occluder - and the model
          would dutifully return 17 landmarks for it. Predicted boxes are good
          enough to preserve identity and not good enough to read a body from.
        * **Largest first**, when the budget bites. Box area is a direct proxy
          for how many pixels each joint gets, so under pressure the estimates
          kept are the ones most likely to be right, rather than whichever
          tracks happen to sort first by id.
        """
        people = [
            track
            for track in tracking.observed
            if track.label.lower() in self._labels and track.is_confirmed
        ]
        people.sort(key=lambda t: t.box.area, reverse=True)
        return people

    def estimate_image(self, image: np.ndarray, boxes: Sequence) -> list[Pose]:
        """Estimate pose for bare boxes on a bare image. For tests and offline use."""
        poses: list[Pose] = []
        for index, box in enumerate(boxes):
            prepared = self._adapter.preprocess(image, box)
            outputs = self._backend.run(prepared.tensor)
            keypoints = tuple(self._adapter.postprocess(outputs, prepared))
            draft = Pose(
                keypoints=keypoints,
                track_id=index,
                entity_id=f"person_{index}",
                box=box,
                model=self._info.model,
            )
            estimate = classify(draft, self._min_keypoint_confidence)
            poses.append(
                Pose(
                    keypoints=keypoints,
                    track_id=index,
                    entity_id=draft.entity_id,
                    box=box,
                    posture=estimate.posture,
                    posture_confidence=estimate.confidence,
                    posture_reason=estimate.reason,
                    model=draft.model,
                )
            )
        return poses

    def warmup(self, iterations: int = 2) -> float:
        """Force graph compilation before the first real frame. See DetectionEngine."""
        from vantage.perception.contracts import BoundingBox

        height, width = self._adapter.input_size
        blank = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        box = BoundingBox(0.0, 0.0, float(width), float(height))
        started = time.perf_counter()
        for _ in range(max(0, iterations)):
            prepared = self._adapter.preprocess(blank, box)
            self._adapter.postprocess(self._backend.run(prepared.tensor), prepared)
        elapsed = (time.perf_counter() - started) * 1000.0
        log.debug(
            "pose engine warmed up",
            extra={"vantage_fields": {"iterations": iterations, "ms": round(elapsed, 1)}},
        )
        return elapsed

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> PoseEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
