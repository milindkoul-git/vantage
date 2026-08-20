"""What tracking produces.

A :class:`Track` is the platform's first *persistent* object. Everything before
it described a single instant: a :class:`~vantage.core.frame.Frame` is one
moment of pixels, a :class:`~vantage.perception.contracts.Detection` is one
object in one frame. A track is the same object across time, and every later
phase - dwell time, trajectories, activity, events - is a question about a
track rather than about a detection.

Three conventions are fixed here and relied on everywhere after:

Anonymous identity
    :attr:`Track.entity_id` is a label plus a counter (``person_17``). It is
    stable for the lifetime of one track within one run and it means exactly
    one thing: "the system believes these observations are the same object".
    It carries no personal information, is not derived from appearance, and is
    not comparable across runs or across cameras. Resolving an entity to a real
    identity is a separate, optional, later subsystem; nothing here anticipates
    it beyond leaving the seam clean.

Coordinates
    Pixels in the original frame, exactly as
    :class:`~vantage.perception.contracts.Detection` defines them. A track's
    box is the filtered estimate, which is *not* the same as the last detection
    box - it is smoothed, and on steps where the object was not detected it is
    predicted.

Estimates are labelled as such
    :attr:`Track.time_since_update` says how many tracker steps have passed
    since real evidence arrived. Zero means this box was confirmed by a
    detection this step; greater than zero means it is extrapolation. Consumers
    that must not act on a guess check that field rather than assuming.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from vantage.perception.contracts import BoundingBox


class TrackState(str, Enum):
    """Lifecycle of a single track.

    The tentative stage is what keeps a detector's false positives out of the
    output: a spurious box appears for one frame and dies without ever being
    published, because a track must be corroborated on several frames before it
    is confirmed.
    """

    TENTATIVE = "tentative"
    """Seen, but not yet enough times to be believed. Not published."""

    CONFIRMED = "confirmed"
    """Corroborated across frames. This is what consumers see."""

    LOST = "lost"
    """Was confirmed, currently unmatched. Kept alive so the identity survives a
    brief occlusion, and still published (flagged as coasting) so a consumer can
    decide for itself whether a prediction is good enough."""

    REMOVED = "removed"
    """Terminal. Retained only long enough for the caller to observe the end."""


@dataclass(frozen=True, slots=True)
class Track:
    """One object's state at one instant, plus its recent history.

    Frozen, like every other record that crosses a stage boundary: the mutable
    working state lives inside the tracker, and what comes out is a snapshot a
    consumer can hold, queue or store without it changing underneath.
    """

    track_id: int
    """Unique within a run. Never reused, so an id in a log always refers to
    exactly one object."""

    entity_id: str
    """Anonymous stable identifier, ``label_id`` (e.g. ``person_17``)."""

    box: BoundingBox
    """Current estimated box, in original-frame pixels."""

    label: str
    class_id: int

    confidence: float
    """Confidence of the most recent supporting detection. Retains the last real
    value while coasting rather than decaying, so it never implies the detector
    said something it did not."""

    state: TrackState
    age: int
    """Tracker steps since this track was created."""

    hits: int
    """Steps on which a detection was successfully associated."""

    time_since_update: int
    """Steps since the last association. ``0`` means observed this step; any
    higher value means the box is predicted, not measured."""

    start_frame: int
    last_frame: int
    """Source frame indices bounding the track's observed lifetime."""

    velocity: tuple[float, float] = (0.0, 0.0)
    """Estimated centre motion in pixels per **second**, not per frame: frames
    do not arrive at a fixed rate once ``detection.interval`` or backpressure is
    in play, so a per-frame figure would not be comparable between runs."""

    history: tuple[tuple[float, float], ...] = ()
    """Recent centre positions, oldest first. Bounded by the tracker - enough
    for a motion trail and for the trajectory work later, not a full archive."""

    @property
    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED

    @property
    def is_coasting(self) -> bool:
        """True when this box is a prediction rather than an observation."""
        return self.time_since_update > 0

    @property
    def center(self) -> tuple[float, float]:
        return self.box.center

    @property
    def duration_frames(self) -> int:
        """Source frames spanned, first observation to last."""
        return max(0, self.last_frame - self.start_frame)

    def describe(self) -> str:
        x1, y1, x2, y2 = self.box.to_int()
        suffix = f" coasting({self.time_since_update})" if self.is_coasting else ""
        return f"{self.entity_id}@{self.confidence:.2f} [{x1},{y1},{x2},{y2}]{suffix}"


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Everything one tracker step produced.

    Holds no pixels, for the same reason
    :class:`~vantage.perception.contracts.DetectionResult` does not: tracks
    outlive frames, and a result that pinned an image buffer would leak memory
    at exactly the rate the camera produces frames.
    """

    tracks: tuple[Track, ...]
    """Published tracks: confirmed and lost. Tentative tracks are never published."""

    source_id: str
    frame_index: int
    capture_wall: float
    frame_size: tuple[int, int]

    elapsed_s: float = 0.0
    """Real time since the previous tracker step. The motion model uses this
    rather than assuming a fixed frame interval."""

    tracking_ms: float = 0.0

    active_count: int = 0
    """Tracks maintained internally, including tentative ones. Larger than
    ``len(tracks)``; the gap is a useful health signal, since a large persistent
    gap means the detector is producing noise that never confirms."""

    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self.tracks)

    @property
    def confirmed(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.state is TrackState.CONFIRMED)

    @property
    def observed(self) -> tuple[Track, ...]:
        """Tracks backed by a detection this step, excluding pure predictions."""
        return tuple(t for t in self.tracks if t.time_since_update == 0)

    def of_class(self, *labels: str) -> tuple[Track, ...]:
        wanted = {label.lower() for label in labels}
        return tuple(t for t in self.tracks if t.label.lower() in wanted)

    def counts(self) -> dict[str, int]:
        """Tracks per label, for HUD summaries and for the event rules later."""
        tally: dict[str, int] = {}
        for track in self.tracks:
            tally[track.label] = tally.get(track.label, 0) + 1
        return tally

    def describe(self) -> str:
        if not self.tracks:
            return f"{self.source_id}#{self.frame_index}: no tracks"
        summary = ", ".join(
            f"{count}x {label}" for label, count in sorted(self.counts().items())
        )
        coasting = sum(1 for t in self.tracks if t.is_coasting)
        extra = f", {coasting} coasting" if coasting else ""
        return (
            f"{self.source_id}#{self.frame_index}: {summary}{extra} ({self.tracking_ms:.1f} ms)"
        )


def empty_tracking_result(
    source_id: str, frame_index: int, capture_wall: float, frame_size: tuple[int, int]
) -> TrackingResult:
    """A result with no tracks.

    Explicitly empty beats ``None`` for the same reason it does in perception:
    consumers keep one code path, and "nothing is being tracked" stays
    distinguishable from "tracking did not run" via the caller's own state.
    """
    return TrackingResult(
        tracks=(),
        source_id=source_id,
        frame_index=frame_index,
        capture_wall=capture_wall,
        frame_size=frame_size,
    )
