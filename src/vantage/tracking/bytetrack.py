"""ByteTrack: multi-object tracking by associating *every* detection box.

The idea, and why it was chosen
-------------------------------
Conventional trackers throw away low-confidence detections before association,
on the reasonable-sounding grounds that they are mostly noise. ByteTrack's
observation is that this discards the wrong ones. When an object is partially
occluded its detection confidence drops - that is precisely what occlusion does
to a detector - so the boxes being discarded are disproportionately the ones
belonging to objects that are hardest to track and most valuable to keep.

So association happens in two passes. High-confidence boxes are matched first,
against all live tracks. Then the *low*-confidence boxes are matched against
whatever tracks are still unmatched. A low-scoring box is not trusted enough to
start a new track, but it is good enough to confirm that an existing track is
still there, because the track already supplies the evidence that an object
exists and only needs to know where it went.

The result is a tracker that keeps identity through partial occlusion at
essentially no cost, using no appearance model. That last part is what makes it
the right fit here on two independent grounds:

* **Compute.** The iGPU is already committed to detection at ~13 ms per frame.
  An appearance-based tracker would need a second network on the same device
  and would roughly double the per-frame cost. ByteTrack is pure geometry -
  measured below one millisecond per frame for typical scenes.
* **Privacy.** An appearance embedding of a person is a biometric-adjacent
  signature, and storing or comparing one edges directly into re-identification.
  This platform does not do that, and choosing a tracker that structurally
  cannot is a stronger guarantee than choosing one that merely does not today.

Deviations from the reference implementation
--------------------------------------------
Three, each for a reason the reference did not have to face:

1. **Real elapsed time** drives the motion model instead of an assumed frame
   step, because ``detection.interval`` and frame drops make the step genuinely
   variable here. See :mod:`vantage.tracking.kalman`.
2. **Track expiry is in seconds**, not frames, for the same reason: "keep a lost
   track for 30 frames" means one second at 30 fps and three at 10 fps, so the
   frame-count form silently changes behaviour with configuration.
3. **Class-aware association.** The reference is single-class (pedestrians).
   Matching a ``car`` detection to a track the system has been calling
   ``person`` is never correct however well the boxes overlap, so it is gated.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from vantage.core.errors import ConfigError
from vantage.core.frame import Frame
from vantage.core.logging import get_logger
from vantage.perception.contracts import BoundingBox, Detection, DetectionResult
from vantage.tracking.assignment import forbid, iou_matrix, match
from vantage.tracking.contracts import Track, TrackingResult, TrackState
from vantage.tracking.kalman import KalmanBoxFilter, MotionNoise

log = get_logger(__name__)

NOMINAL_STEP_S = 1.0 / 30.0
"""Fallback timestep when elapsed time is unavailable or nonsensical."""


@dataclass(frozen=True, slots=True)
class TrackerParams:
    """Tunable association and lifecycle parameters.

    The defaults come from the measured search in :mod:`vantage.tracking.tuning`
    against the ground-truth scenarios in :mod:`vantage.tracking.scenarios`,
    rather than from the reference paper - whose values were fitted to a
    different detector, frame rate and dataset. Re-running ``vantage track tune``
    is how they get revisited.

    One value is a deliberate override rather than the search result:
    ``max_lost_s`` is 1.5 where the search returned 0.25. The search optimises a
    blended objective, and on that objective the two were within a point; but
    1.5 recovers substantially more identity through occlusion (+5 points of
    IDF1 on the training profiles, +8.5 held out) *and* produces fewer identity
    switches overall. Surviving occlusion is the capability this phase exists to
    add, so it is worth a fraction of a point of average accuracy. The override
    is recorded here rather than folded silently into the search so that the
    difference between "measured" and "chosen" stays visible.
    """

    high_threshold: float = 0.3
    """Confidence at or above which a detection joins the first association
    pass and may start a new track."""

    low_threshold: float = 0.1
    """Floor for the second pass. Boxes below this are noise and are ignored
    entirely; boxes between the two thresholds can sustain an existing track
    but never create one."""

    init_threshold: float = 0.5
    """Confidence needed to *create* a track, as opposed to merely continue one.

    Constrained to be at least :attr:`high_threshold`, and tuned well above it.
    The asymmetry is the point: being wrong about where an existing object went
    costs one frame, whereas inventing an object that is not there pollutes
    every downstream count and event until it expires. So a fairly weak box is
    allowed to *continue* a track the system already has evidence for, while
    starting a new one demands real confidence."""

    iou_high: float = 0.2
    """Minimum IoU for a first-pass match. Low by design - the cost matrix is
    solved optimally, so this is a sanity gate against absurd pairings, not the
    mechanism that decides which pairing wins."""

    iou_low: float = 0.4
    """Minimum IoU for the low-confidence second pass. Stricter than the first,
    because a low-scoring box carries less evidence and a bad match here is how
    an identity gets handed to the wrong object."""

    iou_tentative: float = 0.4
    """Minimum IoU to grow an unconfirmed track.

    Stricter than :attr:`iou_high` because a tentative track has almost no
    history to corroborate it, so a loose match here can promote a false
    positive into a published object."""

    min_hits: int = 3
    """Associations required before a track is published. The main defence
    against detector false positives, which rarely persist for three steps in
    the same place."""

    max_lost_s: float = 1.5
    """How long a track survives unmatched before it is dropped. Directly the
    longest occlusion whose identity can be recovered; raising it also raises
    the chance of reviving a track onto a different object."""

    max_step_s: float = 2.0
    """Elapsed times beyond this are clamped. A long stall - a reconnect, a
    laptop resuming from sleep - would otherwise be extrapolated as real motion
    and throw every prediction across the frame."""

    history: int = 30
    """Centre positions retained per track, for motion trails and trajectories."""

    class_aware: bool = True
    """Forbid associations between different classes."""

    noise: MotionNoise = field(default_factory=MotionNoise)

    def __post_init__(self) -> None:
        if not 0.0 < self.low_threshold < 1.0:
            raise ConfigError(f"low_threshold must be in (0, 1), got {self.low_threshold}")
        if not 0.0 < self.high_threshold < 1.0:
            raise ConfigError(f"high_threshold must be in (0, 1), got {self.high_threshold}")
        if self.low_threshold >= self.high_threshold:
            raise ConfigError(
                "low_threshold must be below high_threshold, otherwise the second "
                f"association pass sees no boxes at all (got {self.low_threshold} "
                f">= {self.high_threshold})"
            )
        if self.init_threshold < self.high_threshold:
            raise ConfigError(
                "init_threshold must be >= high_threshold: a box too weak to be "
                f"trusted in the first pass must not be able to create a track "
                f"(got {self.init_threshold} < {self.high_threshold})"
            )
        for name in ("iou_high", "iou_low", "iou_tentative"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1, got {value}")
        if self.min_hits < 1:
            raise ConfigError("min_hits must be >= 1 (1 publishes a track immediately)")
        if self.max_lost_s < 0:
            raise ConfigError("max_lost_s must be >= 0 (0 drops a track the moment it is missed)")
        if self.max_step_s <= 0:
            raise ConfigError("max_step_s must be positive")
        if self.history < 1:
            raise ConfigError("history must be >= 1")


class _TrackState:
    """Mutable working state for one track. Internal; never handed out.

    The public :class:`~vantage.tracking.contracts.Track` is a frozen snapshot
    built from this on demand. Keeping the two separate is what lets a consumer
    hold a result across frames without the tracker mutating it underneath.
    """

    __slots__ = (
        "track_id",
        "entity_id",
        "label",
        "class_id",
        "confidence",
        "state",
        "age",
        "hits",
        "time_since_update",
        "lost_for_s",
        "start_frame",
        "last_frame",
        "history",
        "_filter",
        "_class_votes",
        "counted",
    )

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        frame_index: int,
        params: TrackerParams,
    ) -> None:
        self.track_id = track_id
        self.entity_id = ""  # assigned at confirmation, see _confirm()
        self.label = detection.label
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.state = TrackState.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.lost_for_s = 0.0
        self.counted = False
        self.start_frame = frame_index
        self.last_frame = frame_index
        self.history: deque[tuple[float, float]] = deque(maxlen=params.history)
        self.history.append(detection.box.center)
        self._filter = KalmanBoxFilter(detection.box, params.noise)
        # class_id -> (label the detector used for it, times seen). Carrying the
        # label alongside the vote avoids consulting a global label table: the
        # detector is already the authority on what it called this class, and a
        # lookup would break for any adapter with a non-COCO label set.
        self._class_votes: dict[int, tuple[str, int]] = {
            detection.class_id: (detection.label, 1)
        }
        # The creating detection is itself a hit, so min_hits=1 must publish
        # immediately. Leaving this to observe() - which only runs on the
        # *second* detection - made min_hits=1 and min_hits=2 behave
        # identically, which silently removed a whole value from the parameter
        # search and made the documented meaning of the field untrue.
        if self.hits >= params.min_hits:
            self._confirm()

    @property
    def box(self) -> BoundingBox:
        return self._filter.box

    def predict(self, dt: float) -> None:
        self._filter.predict(dt)
        self.age += 1
        self.time_since_update += 1
        if self.state is not TrackState.TENTATIVE:
            self.lost_for_s += dt

    def observe(self, detection: Detection, frame_index: int, params: TrackerParams) -> None:
        """Fold in an associated detection."""
        self._filter.update(detection.box)
        self.hits += 1
        self.time_since_update = 0
        self.lost_for_s = 0.0
        self.confidence = detection.confidence
        self.last_frame = frame_index
        self.history.append(self._filter.box.center)

        _label, seen = self._class_votes.get(detection.class_id, (detection.label, 0))
        self._class_votes[detection.class_id] = (detection.label, seen + 1)
        if self.state is TrackState.LOST:
            self.state = TrackState.CONFIRMED
        elif self.state is TrackState.TENTATIVE and self.hits >= params.min_hits:
            self._confirm()

    def _confirm(self) -> None:
        """Publish the track and fix its identity for good.

        The class is decided here by majority vote rather than taken from the
        first detection, because detectors flicker between visually similar
        classes on the first frame or two of an object. The entity id embeds the
        label, so settling it from one frame would mean a person permanently
        called ``truck_4``. After this point the label never changes, which is
        what makes the id worth logging.
        """
        self.class_id, (self.label, _votes) = max(
            self._class_votes.items(), key=lambda item: item[1][1]
        )
        self.entity_id = f"{self.label}_{self.track_id}"
        self.state = TrackState.CONFIRMED

    def mark_lost(self) -> None:
        if self.state is TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def snapshot(self) -> Track:
        vx, vy = self._filter.velocity
        return Track(
            track_id=self.track_id,
            entity_id=self.entity_id or f"{self.label}_{self.track_id}",
            box=self.box,
            label=self.label,
            class_id=self.class_id,
            confidence=self.confidence,
            state=self.state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            start_frame=self.start_frame,
            last_frame=self.last_frame,
            velocity=(vx, vy),
            history=tuple(self.history),
        )


class ByteTracker:
    """Two-pass IoU tracker. One instance per source; not thread-safe."""

    def __init__(self, params: TrackerParams | None = None) -> None:
        self.params = params or TrackerParams()
        self._tracks: list[_TrackState] = []
        self._next_id = 1
        self._entities_published = 0
        self._last_time: float | None = None
        self._steps = 0
        self._clamped_steps = 0
        self._total_ms = 0.0

    # -- Tracker protocol ------------------------------------------------

    @property
    def active_tracks(self) -> list[Track]:
        return [t.snapshot() for t in self._tracks]

    def reset(self) -> None:
        """Drop all state.

        Track ids are deliberately *not* reset. A reconnect means continuity is
        genuinely lost, and reusing ids across that boundary would make two
        different objects share one identifier in the same log.
        """
        self._tracks.clear()
        self._last_time = None

    def update(self, result: DetectionResult, *, frame: Frame | None = None) -> TrackingResult:
        """Advance one step and return the published tracks."""
        started = time.perf_counter()
        dt = self._elapsed(frame, result)

        for track in self._tracks:
            track.predict(dt)

        detections = list(result.detections)
        high = [d for d in detections if d.confidence >= self.params.high_threshold]
        low = [
            d
            for d in detections
            if self.params.low_threshold <= d.confidence < self.params.high_threshold
        ]

        confirmed = [t for t in self._tracks if t.state is TrackState.CONFIRMED]
        lost = [t for t in self._tracks if t.state is TrackState.LOST]
        tentative = [t for t in self._tracks if t.state is TrackState.TENTATIVE]

        # Pass 1: everything already believed to exist, against strong boxes.
        pool = confirmed + lost
        matched, unmatched_tracks, unmatched_high = self._associate(
            pool, high, self.params.iou_high
        )
        for track_idx, det_idx in matched:
            pool[track_idx].observe(high[det_idx], result.frame_index, self.params)

        # Pass 2: the ByteTrack step. Tracks that were confirmed and observed
        # recently get a second chance against the weak boxes - an occluded
        # object usually still produces one, just with a low score. Lost tracks
        # are excluded: they have no recent evidence, and pairing two weak
        # signals is how an identity ends up on the wrong object.
        leftover = [
            pool[i]
            for i in unmatched_tracks
            if pool[i].state is TrackState.CONFIRMED and pool[i].time_since_update <= 1
        ]
        second_matched, _, _ = self._associate(leftover, low, self.params.iou_low)
        for track_idx, det_idx in second_matched:
            leftover[track_idx].observe(low[det_idx], result.frame_index, self.params)
        recovered = len(second_matched)

        # Pass 3: unconfirmed tracks against whatever strong boxes are left.
        remaining_high = [high[i] for i in unmatched_high]
        third_matched, _, unmatched_remaining = self._associate(
            tentative, remaining_high, self.params.iou_tentative
        )
        for track_idx, det_idx in third_matched:
            tentative[track_idx].observe(remaining_high[det_idx], result.frame_index, self.params)

        # Counted here rather than by accumulating entity ids in the caller. A
        # set of every entity ever seen grows without bound on a camera that
        # runs for weeks, which is exactly the deployment this platform targets;
        # a flag plus a counter is exact and costs nothing.
        for track in self._tracks:
            if track.state is not TrackState.TENTATIVE and not track.counted:
                track.counted = True
                self._entities_published += 1

        self._retire(result.frame_size)

        for index in unmatched_remaining:
            detection = remaining_high[index]
            if detection.confidence >= self.params.init_threshold:
                self._tracks.append(
                    _TrackState(self._next_id, detection, result.frame_index, self.params)
                )
                self._next_id += 1

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._steps += 1
        self._total_ms += elapsed_ms

        published = tuple(
            t.snapshot()
            for t in self._tracks
            if t.state in (TrackState.CONFIRMED, TrackState.LOST)
        )
        return TrackingResult(
            tracks=published,
            source_id=result.source_id,
            frame_index=result.frame_index,
            capture_wall=result.capture_wall,
            frame_size=result.frame_size,
            elapsed_s=dt,
            tracking_ms=elapsed_ms,
            active_count=len(self._tracks),
            metadata={"high": len(high), "low": len(low), "recovered": recovered},
        )

    # -- internals -------------------------------------------------------

    def _associate(
        self, tracks: list[_TrackState], detections: list[Detection], min_iou: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Optimally pair ``tracks`` with ``detections`` on IoU."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou = iou_matrix([t.box for t in tracks], [d.box for d in detections])
        cost = 1.0 - iou

        if self.params.class_aware:
            track_classes = np.array([t.class_id for t in tracks])[:, None]
            det_classes = np.array([d.class_id for d in detections])[None, :]
            cost = forbid(cost, track_classes != det_classes)

        return match(cost, max_cost=1.0 - min_iou)

    def _retire(self, frame_size: tuple[int, int]) -> None:
        """Demote tracks that were missed this step and drop the ones that are gone.

        Runs after every association pass, so ``time_since_update == 0`` is a
        complete test for "was observed" regardless of which pass matched it.
        """
        survivors: list[_TrackState] = []
        for track in self._tracks:
            if track.time_since_update == 0:
                survivors.append(track)
                continue

            if track.state is TrackState.TENTATIVE:
                # An unconfirmed track that misses a step is dropped at once.
                # Detector false positives are usually single-frame, so this is
                # what stops them ever reaching a consumer.
                continue

            track.mark_lost()
            if track.lost_for_s > self.params.max_lost_s:
                continue
            if self._has_left_frame(track, frame_size):
                continue
            survivors.append(track)
        self._tracks = survivors

    @staticmethod
    def _has_left_frame(track: _TrackState, frame_size: tuple[int, int]) -> bool:
        """True when a coasting track has been predicted out of the frame.

        A lost track is normally worth keeping, because the object is probably
        behind something and will reappear. An object whose estimated centre has
        crossed the frame boundary is a different case: it has walked out of
        shot, no future detection can ever corroborate it, and every further
        step publishes a confident box for something that is provably not
        visible. Measured on the crowd scenario, coasting these for the full
        ``max_lost_s`` was the largest single source of false positives - each
        departure cost roughly 30 of them.

        The test is on the centre rather than the box, so an object merely
        *touching* the edge - half in frame, still detectable - is kept.
        """
        width, height = frame_size
        if width <= 0 or height <= 0:
            return False
        cx, cy = track.box.center
        return not (0.0 <= cx <= width and 0.0 <= cy <= height)

    def _elapsed(self, frame: Frame | None, result: DetectionResult) -> float:
        """Seconds since the previous step, from the most trustworthy clock available.

        Preference order matters. ``media_pts`` is the position on the *media*
        timeline, which is the right answer for a recorded file: analysing a
        60-second clip in 6 seconds must still model the objects as having moved
        at their real speed, not ten times it. Only when there is no media
        timeline - a live camera, where "now" is the only meaningful time - does
        the capture clock apply.
        """
        now = None
        if frame is not None:
            now = frame.media_pts if frame.media_pts is not None else frame.capture_monotonic
        elif result.capture_wall:
            now = result.capture_wall

        if now is None:
            return NOMINAL_STEP_S

        previous = self._last_time
        self._last_time = now
        if previous is None:
            return NOMINAL_STEP_S

        dt = now - previous
        if dt <= 0.0:
            # Two frames stamped identically, or a source that looped and reset
            # its media clock. Neither is an error worth failing a run over, but
            # neither is a real interval either.
            self._clamped_steps += 1
            return NOMINAL_STEP_S
        if dt > self.params.max_step_s:
            self._clamped_steps += 1
            return self.params.max_step_s
        return dt

    def stats(self) -> dict[str, float | int]:
        """Health counters, for the HUD and the run summary."""
        return {
            "steps": self._steps,
            "mean_ms": round(self._total_ms / self._steps, 3) if self._steps else 0.0,
            "active": len(self._tracks),
            "confirmed": sum(1 for t in self._tracks if t.state is TrackState.CONFIRMED),
            "lost": sum(1 for t in self._tracks if t.state is TrackState.LOST),
            "ids_used": self._next_id - 1,
            "entities_published": self._entities_published,
            "clamped_steps": self._clamped_steps,
        }
