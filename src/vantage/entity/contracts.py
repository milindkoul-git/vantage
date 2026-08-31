"""Entity Contracts: Canonical entity data models and immutable snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vantage.perception.contracts import BoundingBox


class IdentityLevel(str, Enum):
    """Hierarchy level of an entity's identity certainty."""

    LOCAL_TRACK = "local_track"
    """Identified only within a single camera tracking session (e.g. cam_01:track_17)."""

    GLOBAL_ASSOCIATED = "global_associated"
    """Associated across cameras/viewpoints via spatial-temporal-appearance fusion (e.g. global_person_4)."""

    NAMED_CONFIRMED = "named_confirmed"
    """Cryptographically or biometrically matched against an enrolled identity with consent."""


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    """Structured biometric or visual identity evidence attached to an entity."""

    global_id: str
    local_track_id: int
    camera_id: str
    level: IdentityLevel = IdentityLevel.LOCAL_TRACK
    name: str | None = None
    similarity: float = 0.0
    margin: float = 0.0
    source: str = "tracker"  # 'tracker' | 'reid' | 'face_yunet_sface' | 'manual'
    enrolled_at: float | None = None
    is_consented: bool = True
    evidence_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def is_named(self) -> bool:
        return self.name is not None and self.name.lower() not in ("unknown", "", "none")

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "local_track_id": self.local_track_id,
            "camera_id": self.camera_id,
            "level": self.level.value,
            "name": self.name,
            "similarity": round(self.similarity, 3),
            "margin": round(self.margin, 3),
            "source": self.source,
            "is_named": self.is_named,
        }


@dataclass(frozen=True, slots=True)
class SpatialPresence:
    """Current and historical spatial positioning for an entity."""

    camera_id: str
    recent_cameras: tuple[str, ...]
    image_box: BoundingBox
    normalized_foot_point: tuple[float, float]  # (x_norm, y_norm) in [0, 1]
    world_position: tuple[float, float, float] | None = None  # (x, y, z) in meters if projected
    first_seen_wall: float = 0.0
    last_seen_wall: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_seen_wall - self.first_seen_wall)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "recent_cameras": list(self.recent_cameras),
            "image_box": {
                "x1": round(self.image_box.x1, 1),
                "y1": round(self.image_box.y1, 1),
                "x2": round(self.image_box.x2, 1),
                "y2": round(self.image_box.y2, 1),
            },
            "normalized_foot_point": (
                round(
                    getattr(
                        self.normalized_foot_point,
                        "x",
                        self.normalized_foot_point[0]
                        if isinstance(self.normalized_foot_point, (tuple, list))
                        else 0.5,
                    ),
                    4,
                ),
                round(
                    getattr(
                        self.normalized_foot_point,
                        "y",
                        self.normalized_foot_point[1]
                        if isinstance(self.normalized_foot_point, (tuple, list))
                        else 1.0,
                    ),
                    4,
                ),
            ),
            "world_position": (
                [round(c, 2) for c in self.world_position]
                if self.world_position is not None
                else None
            ),
            "first_seen": round(self.first_seen_wall, 2),
            "last_seen": round(self.last_seen_wall, 2),
            "duration_s": round(self.duration_s, 2),
        }


@dataclass(frozen=True, slots=True)
class TemporalKinematics:
    """Physical motion, speed, posture, and kinematic state."""

    speed_h_s: float
    motion_state: str  # 'stationary' | 'walking' | 'running' | 'moving'
    posture: str  # 'standing' | 'sitting' | 'crouching' | 'lying' | 'unknown'
    bearing_deg: float | None = None
    state_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speed_h_s": round(self.speed_h_s, 3),
            "motion_state": self.motion_state,
            "posture": self.posture,
            "bearing_deg": round(self.bearing_deg, 1) if self.bearing_deg is not None else None,
            "state_confidence": round(self.state_confidence, 2),
        }


@dataclass(frozen=True, slots=True)
class ActivityContext:
    """Action and temporal activity state."""

    current_activities: tuple[str, ...]
    primary_activity: str
    confidence: float
    evidence_summary: str
    activity_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_activities": list(self.current_activities),
            "primary_activity": self.primary_activity,
            "confidence": round(self.confidence, 2),
            "evidence_summary": self.evidence_summary,
            "recent_history": list(self.activity_history[-5:]),
        }


@dataclass(frozen=True, slots=True)
class SpatialContext:
    """Zone membership, spatial proximity, and geofence state."""

    current_zones: tuple[str, ...]
    recent_zones: tuple[str, ...]
    spatial_relations: tuple[str, ...]
    nearby_entities: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_zones": list(self.current_zones),
            "recent_zones": list(self.recent_zones),
            "spatial_relations": list(self.spatial_relations),
            "nearby_entities": list(self.nearby_entities),
        }


@dataclass(frozen=True, slots=True)
class JourneyContext:
    """Multi-camera journey timeline and transition history."""

    current_camera: str
    camera_sequence: tuple[str, ...]
    sighting_count: int
    journey_state: str  # 'active' | 'in_transit' | 'departed'
    first_camera: str
    last_transition_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_camera": self.current_camera,
            "camera_sequence": list(self.camera_sequence),
            "sighting_count": self.sighting_count,
            "journey_state": self.journey_state,
            "first_camera": self.first_camera,
            "last_transition_time": (
                round(self.last_transition_time, 2) if self.last_transition_time else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RelationshipContext:
    """Interactive relationships, persistent associations, and proximity graph edges."""

    related_entities: tuple[str, ...]
    relationship_types: tuple[str, ...]
    interaction_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    active_relationships: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "related_entities": list(self.related_entities),
            "relationship_types": list(self.relationship_types),
            "recent_interactions": list(self.interaction_history[-5:]),
            "active_relationships": list(self.active_relationships),
        }


@dataclass(frozen=True, slots=True)
class EventContext:
    """Active security alerts and recent incident history for an entity."""

    active_events: tuple[dict[str, Any], ...]
    recent_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_events": list(self.active_events),
            "recent_events": list(self.recent_events[-10:]),
        }


@dataclass(frozen=True, slots=True)
class TemporalBehaviorContext:
    """Deterministic spatio-temporal behavioral patterns (e.g. collapse, pacing, high-energy)."""

    behaviors: tuple[str, ...]
    primary_behavior: str
    confidence: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "behaviors": list(self.behaviors),
            "primary_behavior": self.primary_behavior,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class EntitySnapshot:
    """Immutable, thread-safe canonical snapshot of an evolving entity.

    Created at a point in time by EntityContext. Zero lock contention when read.
    """

    global_id: str
    label: str
    identity: IdentityEvidence
    spatial: SpatialPresence
    kinematics: TemporalKinematics
    activity: ActivityContext
    zones: SpatialContext
    journey: JourneyContext
    relationships: RelationshipContext
    events: EventContext
    timestamp: float
    behavior: TemporalBehaviorContext | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to JSON-serializable dictionary."""
        res = {
            "global_id": self.global_id,
            "label": self.label,
            "timestamp": round(self.timestamp, 3),
            "identity": self.identity.to_dict(),
            "spatial": self.spatial.to_dict(),
            "kinematics": self.kinematics.to_dict(),
            "activity": self.activity.to_dict(),
            "zones": self.zones.to_dict(),
            "journey": self.journey.to_dict(),
            "relationships": self.relationships.to_dict(),
            "events": self.events.to_dict(),
        }
        if self.behavior is not None:
            res["behavior"] = self.behavior.to_dict()
        return res
