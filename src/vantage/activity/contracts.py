"""Activity contracts: what an entity has been doing, over time.

Phase 4 answers two instantaneous questions - what shape is this person in, and
is this entity moving. Activity is the first thing in the platform that cannot
be answered from a single frame at all: ``sitting_down`` is not a posture, it is
a *change* of posture, and ``loitering`` is indistinguishable from ``standing``
until you know how long it has been going on.

Several at once, deliberately
-----------------------------
An entity carries a *list* of observations rather than one label, because the
real answers overlap: someone can be walking and holding an arm up at the same
time, and forcing a single winner would throw away one of two true statements.
This is also the shape the spec sketched for stored observations, so the record
this produces is the record Phase 8 will keep.

Why the vocabulary is small
---------------------------
Every activity here is derivable from signals the platform already measures, and
each one was kept only because it is reliably determinable. The off-the-shelf
alternative was surveyed and rejected on the evidence: no skeleton-action model
exists as a permissively licensed ONNX export with real provenance, and the
video classifiers that do exist label a *frame* rather than an entity, from
vocabularies like Kinetics-400 - ``abseiling``, ``zumba``, ``shredding paper``.
A model that confidently reports the wrong kind of thing is worse than a short
list of things that are actually true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterator


class Activity(str, Enum):
    """What an entity is doing.

    Split into three groups by what they are derived from, which is also what
    determines when they are available at all:

    * **Locomotion** - from motion state alone. Works for any entity, with no
      pose model and no person classifier.
    * **Dwell** - from how long a state has held.
    * **Posture-derived** - needs pose, and needs the joints the posture rules
      require. On a camera that never sees anyone's legs these simply never
      fire, which is correct rather than a fault.
    """

    IDLE = "idle"
    """Present and doing nothing else detectable. An explicit observation rather
    than an empty list, so "nothing is happening" stays distinguishable from
    "the recogniser produced nothing"."""

    WALKING = "walking"
    RUNNING = "running"
    LOITERING = "loitering"
    SITTING_DOWN = "sitting_down"
    STANDING_UP = "standing_up"
    FALLING = "falling"
    ARM_RAISED = "arm_raised"

    @property
    def needs_pose(self) -> bool:
        return self in _POSTURE_DERIVED

    @property
    def is_transient(self) -> bool:
        """Whether this describes a moment rather than a continuing state.

        Transient activities are held for a short window after they occur so
        that a consumer sampling at a lower rate cannot miss them entirely.
        """
        return self in _TRANSIENT


_POSTURE_DERIVED = frozenset(
    {Activity.SITTING_DOWN, Activity.STANDING_UP, Activity.FALLING, Activity.ARM_RAISED}
)
_TRANSIENT = frozenset({Activity.SITTING_DOWN, Activity.STANDING_UP, Activity.FALLING})


@dataclass(frozen=True, slots=True)
class ActivityObservation:
    """One thing an entity is doing, with how long and on what grounds."""

    activity: Activity
    confidence: float
    duration_s: float
    """How long this has held. Zero on the frame it is first reported."""

    evidence: str
    """What the recogniser actually measured, in words - "0.62 h/s sustained
    over 1.2 s". Every rule states its grounds, so a surprising observation can
    be argued with instead of merely believed."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"activity confidence must be in [0, 1], got {self.confidence}"
            )

    def describe(self) -> str:
        return f"{self.activity.value} ({self.confidence:.2f}, {self.duration_s:.1f}s)"


@dataclass(frozen=True, slots=True)
class EntityActivity:
    """Everything one entity is doing this frame."""

    track_id: int
    entity_id: str
    label: str
    observations: tuple[ActivityObservation, ...]

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[ActivityObservation]:
        return iter(self.observations)

    @property
    def activities(self) -> tuple[Activity, ...]:
        return tuple(o.activity for o in self.observations)

    def has(self, activity: Activity) -> bool:
        return any(o.activity is activity for o in self.observations)

    def get(self, activity: Activity) -> ActivityObservation | None:
        return next((o for o in self.observations if o.activity is activity), None)

    @property
    def primary(self) -> ActivityObservation | None:
        """The observation most worth showing when there is room for one.

        Transient events win over continuing states regardless of confidence: a
        fall matters more than the fact that the person had been walking, and a
        display that ranked purely on confidence would bury it.
        """
        if not self.observations:
            return None
        return max(
            self.observations,
            key=lambda o: (o.activity.is_transient, o.activity is not Activity.IDLE, o.confidence),
        )

    def describe(self) -> str:
        if not self.observations:
            return f"{self.entity_id}: nothing"
        return f"{self.entity_id}: " + ", ".join(o.describe() for o in self.observations)


@dataclass(frozen=True, slots=True)
class ActivityResult:
    """Every entity's activities for one frame."""

    entities: tuple[EntityActivity, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    elapsed_s: float = 0.0
    pose_available: bool = False
    """Whether pose was running. Without it the posture-derived activities
    cannot fire, and a consumer needs to know the difference between "did not
    happen" and "could not be seen"."""

    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self) -> Iterator[EntityActivity]:
        return iter(self.entities)

    def by_track(self) -> dict[int, EntityActivity]:
        return {entity.track_id: entity for entity in self.entities}

    def counts(self) -> dict[str, int]:
        """How many entities are doing each thing. Sums above the entity count,
        because activities overlap."""
        tally: dict[str, int] = {}
        for entity in self.entities:
            for observation in entity:
                key = observation.activity.value
                tally[key] = tally.get(key, 0) + 1
        return tally

    def of(self, activity: Activity) -> tuple[EntityActivity, ...]:
        return tuple(entity for entity in self.entities if entity.has(activity))

    def notable(self) -> tuple[EntityActivity, ...]:
        """Entities doing something other than existing quietly."""
        return tuple(
            entity
            for entity in self.entities
            if any(o.activity is not Activity.IDLE for o in entity)
        )

    def describe(self) -> str:
        if not self.entities:
            return "no entities"
        counts = self.counts()
        if not counts:
            return f"{len(self.entities)} entities, nothing recognised"
        return ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))


def to_observation_record(
    entity: EntityActivity,
    camera_id: str,
    wall_time: float,
    motion: dict[str, object] | None = None,
) -> dict[str, object]:
    """Render one entity's activities as the structured record the spec sketched.

    Plain primitives only, so it reaches a database or a socket without a custom
    encoder. ``identity`` is present and always ``None``: the seam the identity
    layer would later fill, written now so that phase inherits a producer rather
    than a schema migration.
    """
    record: dict[str, object] = {
        "timestamp": datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat(),
        "camera_id": camera_id,
        "entity_id": entity.entity_id,
        "entity_type": entity.label,
        "identity": None,
        "observations": [
            {
                "type": observation.activity.value,
                "confidence": round(observation.confidence, 3),
                "duration_s": round(observation.duration_s, 2),
                "evidence": observation.evidence,
            }
            for observation in entity
        ],
    }
    if motion is not None:
        record["motion"] = motion
    return record
