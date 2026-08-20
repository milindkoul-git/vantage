"""Spatial contracts: where entities are, and how they stand to each other.

Phases 3 to 5 describe entities one at a time - this one is the first that is
about *pairs* and *places*. It produces the two things the spec sketched for the
event engine to consume::

    Person #17  --approached-->  Person #21
    Object #5   --moved from-->  Zone A --to--> Zone B

The measurement problem, stated before anything is claimed
----------------------------------------------------------
A camera gives no depth. Two people on opposite sides of a room can have boxes
that touch, or even overlap, and nothing in a single view distinguishes that
from two people standing together. Every spatial claim here rests on one
assumption, which is worth naming rather than burying:

    **Entities share a common ground plane, and the bottom edge of a box is
    where the entity meets it.**

That makes ``bottom_center`` the anchor for everything - a person's box centre
drifts up and down as they change posture, their feet do not - and it makes
image-plane distance a usable proxy for ground distance for entities at similar
depth. It is an approximation, and the failures it produces are documented in
:mod:`vantage.spatial` rather than left to be discovered.

Distances are in **entity heights**, never pixels, for the reason Phase 4
established: a metre at the far end of a corridor is a handful of pixels and a
metre near the lens is hundreds.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class Relation(str, Enum):
    """How one entity stands to another."""

    NEAR = "near"
    """Within the proximity threshold, sustained. Symmetric."""

    APPROACHING = "approaching"
    """Separation shrinking steadily. Symmetric - two entities closing on each
    other is one fact, not two."""

    RECEDING = "receding"
    INTERACTING = "interacting_with"
    """A person and an object, close for long enough that incidental passing is
    excluded. Directed: the person is the subject. See
    :attr:`RelationObservation.evidence` for which test was actually met -
    reach-confirmed by a wrist landmark, or proximity alone."""

    @property
    def is_symmetric(self) -> bool:
        return self is not Relation.INTERACTING


class ZoneEvent(str, Enum):
    """A change in zone membership. Both are moments, not states."""

    ENTERED = "entered"
    EXITED = "exited"


@dataclass(frozen=True, slots=True)
class Zone:
    """A named region of the frame, as a polygon in normalised coordinates.

    Normalised to ``[0, 1]`` on purpose: a zone drawn against a 1920x1080 stream
    still means the same part of the scene when the camera is reconfigured to
    1280x720, or when a file source of a different size is replayed. Pixel
    coordinates would silently point somewhere else.
    """

    name: str
    points: tuple[tuple[float, float], ...]
    kind: str = "area"
    """Free-form label - ``entrance``, ``till``, ``restricted``. Carried through
    to the observation record so a later phase can attach policy to it without
    this module knowing what any of them mean."""

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(
                f"zone {self.name!r} needs at least 3 points, got {len(self.points)}"
            )
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"zone {self.name!r} has a point outside [0, 1]: ({x}, {y}). "
                    "Zone coordinates are normalised so they survive a change of "
                    "resolution."
                )

    def contains(self, point: tuple[float, float], frame_size: tuple[int, int]) -> bool:
        """Whether a pixel-space point falls inside, by ray casting.

        Ray casting rather than a geometry dependency: it is fifteen lines, it
        handles concave polygons, and the platform already declined SciPy for
        the assignment solver on the same grounds.
        """
        width, height = frame_size
        if width <= 0 or height <= 0:
            return False
        x = point[0] / width
        y = point[1] / height

        inside = False
        count = len(self.points)
        for index in range(count):
            x1, y1 = self.points[index]
            x2, y2 = self.points[(index + 1) % count]
            # Half-open vertical test, so a point level with a vertex is counted
            # once rather than twice or not at all.
            if (y1 > y) != (y2 > y):
                crossing = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                if x < crossing:
                    inside = not inside
        return inside

    def centroid(self) -> tuple[float, float]:
        """Average of the vertices, in normalised coordinates. For labelling."""
        return (
            sum(p[0] for p in self.points) / len(self.points),
            sum(p[1] for p in self.points) / len(self.points),
        )


@dataclass(frozen=True, slots=True)
class ZoneOccupancy:
    """One entity's presence in one zone."""

    zone: str
    kind: str
    dwell_s: float
    event: ZoneEvent | None = None
    """Set on the frame the entity crossed the boundary, and held briefly
    afterwards so a slow consumer cannot miss the crossing entirely."""


@dataclass(frozen=True, slots=True)
class RelationObservation:
    """One relation between two entities."""

    relation: Relation
    subject_id: str
    object_id: str
    subject_track: int
    object_track: int
    distance: float
    """Ground-plane separation in entity heights, under the common-ground
    assumption."""

    confidence: float
    duration_s: float
    evidence: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"relation confidence must be in [0, 1], got {self.confidence}")

    @property
    def key(self) -> tuple[str, int, int]:
        """Identity of this relation, with symmetric pairs ordered.

        Without the ordering, ``near(a, b)`` and ``near(b, a)`` are two
        different edges describing one fact, and every count of "how many
        entities are near each other" doubles.
        """
        if self.relation.is_symmetric:
            low, high = sorted((self.subject_track, self.object_track))
            return (self.relation.value, low, high)
        return (self.relation.value, self.subject_track, self.object_track)

    def describe(self) -> str:
        return (
            f"{self.subject_id} {self.relation.value} {self.object_id} "
            f"({self.distance:.2f}h, {self.confidence:.2f})"
        )


@dataclass(frozen=True, slots=True)
class EntitySpatial:
    """Where one entity is."""

    track_id: int
    entity_id: str
    label: str
    zones: tuple[ZoneOccupancy, ...] = ()
    """Zones the entity is in, plus any it has just left.

    A list rather than one name because overlapping zones are allowed - a till
    can sit inside a shop floor. Entries marked :attr:`ZoneEvent.EXITED` are
    zones the entity is **no longer** in, carried for a moment so the crossing
    is visible to a consumer that samples slowly; :meth:`zone_names` and
    :meth:`in_zone` exclude them, so "where is this entity" never reports a
    place it has left.
    """

    ground_point: tuple[float, float] = (0.0, 0.0)
    """Where the entity meets the ground, in pixels."""

    @property
    def occupied(self) -> tuple[ZoneOccupancy, ...]:
        """Only the zones the entity is currently inside."""
        return tuple(z for z in self.zones if z.event is not ZoneEvent.EXITED)

    @property
    def zone_names(self) -> tuple[str, ...]:
        return tuple(z.zone for z in self.occupied)

    def in_zone(self, name: str) -> bool:
        return any(z.zone == name for z in self.occupied)


@dataclass(frozen=True, slots=True)
class SpatialResult:
    """Zones and relations for one frame - the scene graph, flattened."""

    entities: tuple[EntitySpatial, ...]
    relations: tuple[RelationObservation, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    elapsed_s: float = 0.0
    zones_defined: int = 0
    pose_available: bool = False
    state_available: bool = False
    """Whether motion state was supplied. Without it, interaction is only
    claimed on a confirmed reach - see :mod:`vantage.spatial.analyzer`."""

    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self) -> Iterator[EntitySpatial]:
        return iter(self.entities)

    def by_track(self) -> dict[int, EntitySpatial]:
        return {entity.track_id: entity for entity in self.entities}

    def of(self, relation: Relation) -> tuple[RelationObservation, ...]:
        return tuple(r for r in self.relations if r.relation is relation)

    def occupancy(self) -> dict[str, int]:
        """How many entities are in each zone."""
        tally: dict[str, int] = {}
        for entity in self.entities:
            for zone in entity.occupied:
                tally[zone.zone] = tally.get(zone.zone, 0) + 1
        return tally

    def crossings(self) -> tuple[tuple[EntitySpatial, ZoneOccupancy], ...]:
        """Boundary crossings on this frame - the transient half of zones.

        Every occupancy returned has a non-``None`` ``event``; that is the
        filter. Callers still have to narrow it for a type checker, which is
        worth the small friction: the alternative is a separate type whose only
        difference is a field that cannot be None.
        """
        return tuple(
            (entity, zone)
            for entity in self.entities
            for zone in entity.zones
            if zone.event is not None
        )

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for relation in self.relations:
            key = relation.relation.value
            tally[key] = tally.get(key, 0) + 1
        return tally

    def describe(self) -> str:
        parts = []
        occupancy = self.occupancy()
        if occupancy:
            parts.append(
                "zones: " + ", ".join(f"{n} in {name}" for name, n in sorted(occupancy.items()))
            )
        counts = self.counts()
        if counts:
            parts.append(
                "relations: " + ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
            )
        return "; ".join(parts) if parts else "nothing spatial"


def to_scene_record(
    result: SpatialResult, camera_id: str, wall_time: float
) -> dict[str, object]:
    """Render the frame's scene graph as a structured record.

    Nodes and edges rather than prose, because the event engine of the next
    phase needs to query it - "which entities are in the restricted zone" is a
    filter over nodes, and "who approached whom" is a filter over edges.

    Plain primitives only, and ``identity`` present and always ``None`` on every
    node: the same seam the earlier phases left, kept consistent so a single
    identity resolver can fill all of them.
    """
    return {
        "timestamp": datetime.fromtimestamp(wall_time, tz=UTC).isoformat(),
        "camera_id": camera_id,
        "nodes": [
            {
                "entity_id": entity.entity_id,
                "entity_type": entity.label,
                "identity": None,
                "zones": [
                    {
                        "zone": zone.zone,
                        "kind": zone.kind,
                        "dwell_s": round(zone.dwell_s, 2),
                        "event": zone.event.value if zone.event else None,
                    }
                    for zone in entity.zones
                ],
            }
            for entity in result.entities
        ],
        "edges": [
            {
                "subject": relation.subject_id,
                "relation": relation.relation.value,
                "object": relation.object_id,
                "distance_heights": round(relation.distance, 3),
                "confidence": round(relation.confidence, 3),
                "duration_s": round(relation.duration_s, 2),
                "evidence": relation.evidence,
            }
            for relation in result.relations
        ],
    }
