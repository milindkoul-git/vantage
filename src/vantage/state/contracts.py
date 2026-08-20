"""Entity state contracts: what an object is doing, independent of what it is.

Pose answers "what shape is this person in". State answers a question that
applies to a person, a car or a mug equally: is it moving, how fast, which way,
and for how long has that been true. Both are Phase 4 because both turn a
per-frame observation into something with duration, which is the prerequisite
for the activity recognition and event rules that follow.

Speed is measured in **entity heights per second**, never pixels. A person
walking at the far end of a corridor covers a handful of pixels per second and
the same person a metre from the lens covers hundreds; both are walking. Box
height is the only scale reference available without camera calibration, and
dividing by it makes one threshold work across the frame. It is imperfect - a
person walking directly at the camera grows rather than translates, and reads as
slower than they are - and that limitation is real rather than hidden.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class MotionState(str, Enum):
    """Whether an entity is going anywhere."""

    MOVING = "moving"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"
    """Not yet observed for long enough to say. A track's first frames have no
    velocity estimate worth the name, and calling that "stationary" would invent
    a stillness nobody measured."""


@dataclass(frozen=True, slots=True)
class EntityState:
    """The current motion state of one tracked entity."""

    track_id: int
    entity_id: str
    label: str

    motion: MotionState
    speed: float
    """Heights per second, smoothed by the tracker's motion model."""

    dwell_s: float
    """How long :attr:`motion` has held. Reset on every state change, which is
    what makes "stationary for 40 seconds" expressible at all."""

    bearing_deg: float | None
    """Direction of travel in degrees clockwise from up, or ``None`` when
    stationary. Image-plane, not ground-plane: without calibration there is no
    world direction to report, and pretending otherwise would be a fabrication
    a later phase would build on."""

    distance: float
    """Total path length since the entity appeared, in heights. Path length, not
    displacement, so pacing back and forth accumulates rather than cancelling."""

    age_s: float
    observed: bool
    """Whether this update was backed by a detection. A coasting entity keeps
    its last state rather than being re-measured from a predicted box."""

    posture: str | None = None
    """Filled in from pose when it is running, so one record answers both
    halves of the question. ``None`` means pose was not available, never that
    the person had no posture."""

    def describe(self) -> str:
        detail = f"{self.motion.value} {self.dwell_s:.0f}s"
        if self.motion is MotionState.MOVING:
            detail += f" @ {self.speed:.2f} h/s"
        if self.posture:
            detail += f", {self.posture}"
        return f"{self.entity_id}: {detail}"

    def to_observation(self, camera_id: str, wall_time: float) -> dict[str, object]:
        """Render as a structured observation record.

        The shape the spec sketched for Phase 8 storage, emitted now so the
        later phase inherits a producer rather than a redesign. Deliberately a
        plain dict of primitives: no numpy, no dataclasses, nothing that would
        need a custom encoder to reach a database or a socket.

        ``identity`` is present and always ``None``. That is the seam the
        identity layer would later fill, and writing it now keeps the field from
        being retrofitted into records that already exist.
        """
        observations: list[dict[str, object]] = [
            {"type": self.motion.value, "confidence": round(min(1.0, self.dwell_s), 3)}
        ]
        if self.posture:
            observations.append({"type": self.posture, "confidence": None})
        return {
            "timestamp": datetime.fromtimestamp(wall_time, tz=UTC).isoformat(),
            "camera_id": camera_id,
            "entity_id": self.entity_id,
            "entity_type": self.label,
            "identity": None,
            "motion": {
                "state": self.motion.value,
                "speed_heights_per_s": round(self.speed, 4),
                "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 1),
                "dwell_s": round(self.dwell_s, 2),
                "distance_heights": round(self.distance, 3),
            },
            "observations": observations,
        }


@dataclass(frozen=True, slots=True)
class StateResult:
    """Every entity state for one frame."""

    states: tuple[EntityState, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    elapsed_s: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Iterator[EntityState]:
        return iter(self.states)

    def by_track(self) -> dict[int, EntityState]:
        return {state.track_id: state for state in self.states}

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for state in self.states:
            tally[state.motion.value] = tally.get(state.motion.value, 0) + 1
        return tally

    def moving(self) -> tuple[EntityState, ...]:
        return tuple(s for s in self.states if s.motion is MotionState.MOVING)

    def describe(self) -> str:
        if not self.states:
            return "no entities"
        return ", ".join(f"{n} {name}" for name, n in sorted(self.counts().items()))
