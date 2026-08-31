"""Entity Context: Mutable, thread-safe aggregator for a single evolving entity."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from vantage.entity.contracts import (
    ActivityContext,
    EntitySnapshot,
    EventContext,
    IdentityEvidence,
    IdentityLevel,
    JourneyContext,
    RelationshipContext,
    SpatialContext,
    SpatialPresence,
    TemporalKinematics,
)
from vantage.perception.contracts import BoundingBox


class EntityContext:
    """Aggregates evolving knowledge about a single physical entity across subsystems.

    Follows the Aggregator pattern: subsystems compute; EntityContext maintains
    temporal continuity and produces immutable `EntitySnapshot` objects on demand.
    """

    def __init__(
        self,
        global_id: str,
        label: str,
        initial_camera: str,
        initial_track_id: int,
        initial_box: BoundingBox,
        wall_time: float,
    ) -> None:
        self._lock = threading.Lock()
        self.global_id = global_id
        self.label = label
        self.first_seen_wall = wall_time
        self.last_seen_wall = wall_time

        # Identity
        self.identity_level = IdentityLevel.LOCAL_TRACK
        self.named_identity: str | None = None
        self.identity_similarity: float = 0.0
        self.identity_margin: float = 0.0
        self.identity_source: str = "tracker"
        self.identity_history: deque[dict[str, Any]] = deque(maxlen=20)

        # Spatial Presence
        self.current_camera = initial_camera
        self.recent_cameras: deque[str] = deque([initial_camera], maxlen=10)
        self.current_box = initial_box
        self.normalized_foot_point: tuple[float, float] = (0.5, 1.0)
        self.world_position: tuple[float, float, float] | None = None

        # Kinematics
        self.speed_h_s: float = 0.0
        self.motion_state: str = "stationary"
        self.posture: str = "standing"
        self.bearing_deg: float | None = None
        self.state_confidence: float = 1.0

        # Activity
        self.current_activities: list[str] = ["idle"]
        self.primary_activity: str = "idle"
        self.activity_confidence: float = 1.0
        self.activity_evidence: str = "Initial sighting"
        self.activity_history: deque[dict[str, Any]] = deque(maxlen=20)

        # Spatial / Zones
        self.current_zones: set[str] = set()
        self.recent_zones: deque[str] = deque(maxlen=10)
        self.spatial_relations: list[str] = []
        self.nearby_entities: list[str] = []

        # Journey
        self.camera_sequence: list[str] = [initial_camera]
        self.sighting_count: int = 1
        self.journey_state: str = "active"
        self.last_transition_time: float | None = None

        # Relationships
        self.related_entities: list[str] = []
        self.relationship_types: list[str] = []
        self.active_relationships_data: list[dict[str, Any]] = []
        self.interaction_history: deque[dict[str, Any]] = deque(maxlen=20)

        # Events
        self.active_events: list[dict[str, Any]] = []
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=50)

        # Behavioral Interpretation (Phase 17)
        self.current_behaviors: list[str] = []
        self.primary_behavior: str = "nominal"
        self.behavior_confidence: float = 1.0
        self.behavior_evidence: str = ""

        self.last_track_id = initial_track_id

    def update_spatial(
        self,
        camera_id: str,
        box: BoundingBox,
        foot_point: tuple[float, float],
        wall_time: float,
        world_position: tuple[float, float, float] | None = None,
    ) -> None:
        """Update entity bounding box, camera, and ground-plane position."""
        with self._lock:
            self.last_seen_wall = wall_time
            self.current_box = box
            self.normalized_foot_point = foot_point
            if world_position is not None:
                self.world_position = world_position

            if camera_id != self.current_camera:
                self.last_transition_time = wall_time
                self.current_camera = camera_id
                self.recent_cameras.append(camera_id)
                self.camera_sequence.append(camera_id)
                self.identity_level = IdentityLevel.GLOBAL_ASSOCIATED

            self.sighting_count += 1
            self.journey_state = "active"

    def update_kinematics(
        self,
        speed_h_s: float,
        motion_state: str,
        posture: str = "standing",
        bearing_deg: float | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Update movement speed, motion state, and posture."""
        with self._lock:
            self.speed_h_s = speed_h_s
            self.motion_state = motion_state
            self.posture = posture
            self.bearing_deg = bearing_deg
            self.state_confidence = confidence

    def update_activity(
        self,
        activities: list[str] | tuple[str, ...],
        primary: str,
        confidence: float,
        evidence: str,
        wall_time: float,
    ) -> None:
        """Update current activities and append to activity history."""
        with self._lock:
            self.current_activities = list(activities)
            self.primary_activity = primary
            self.activity_confidence = confidence
            self.activity_evidence = evidence
            self.activity_history.append(
                {
                    "timestamp": wall_time,
                    "primary": primary,
                    "activities": list(activities),
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )

    def update_zones(
        self,
        zones: set[str] | list[str],
        relations: list[str] | None = None,
        nearby: list[str] | None = None,
    ) -> None:
        """Update active geofence zones and proximity relations."""
        with self._lock:
            new_zones = set(zones)
            for z in new_zones:
                if z not in self.current_zones:
                    self.recent_zones.append(z)
            self.current_zones = new_zones
            if relations is not None:
                self.spatial_relations = list(relations)
            if nearby is not None:
                self.nearby_entities = list(nearby)

    def attach_identity(
        self,
        name: str,
        similarity: float,
        margin: float = 0.0,
        source: str = "face_yunet_sface",
        wall_time: float | None = None,
    ) -> None:
        """Attach verified biometric or external identity evidence."""
        with self._lock:
            if name.lower() not in ("unknown", "", "none"):
                self.named_identity = name
                self.identity_similarity = similarity
                self.identity_margin = margin
                self.identity_source = source
                self.identity_level = IdentityLevel.NAMED_CONFIRMED
                self.identity_history.append(
                    {
                        "timestamp": wall_time or time.time(),
                        "name": name,
                        "similarity": similarity,
                        "margin": margin,
                        "source": source,
                    }
                )

    def add_event(self, event_dict: dict[str, Any]) -> None:
        """Record an active or recent security event on this entity."""
        with self._lock:
            self.active_events.append(event_dict)
            self.recent_events.append(event_dict)
            # Keep only active events from the last 60 seconds
            t_now = event_dict.get("timestamp", time.time())
            self.active_events = [
                e for e in self.active_events if t_now - e.get("timestamp", 0) < 60.0
            ]

    def to_snapshot(self) -> EntitySnapshot:
        """Generate an immutable point-in-time snapshot with minimal lock overhead."""
        with self._lock:
            identity_evidence = IdentityEvidence(
                global_id=self.global_id,
                local_track_id=self.last_track_id,
                camera_id=self.current_camera,
                level=self.identity_level,
                name=self.named_identity,
                similarity=self.identity_similarity,
                margin=self.identity_margin,
                source=self.identity_source,
                evidence_history=tuple(self.identity_history),
            )

            spatial = SpatialPresence(
                camera_id=self.current_camera,
                recent_cameras=tuple(self.recent_cameras),
                image_box=self.current_box,
                normalized_foot_point=self.normalized_foot_point,
                world_position=self.world_position,
                first_seen_wall=self.first_seen_wall,
                last_seen_wall=self.last_seen_wall,
            )

            kinematics = TemporalKinematics(
                speed_h_s=self.speed_h_s,
                motion_state=self.motion_state,
                posture=self.posture,
                bearing_deg=self.bearing_deg,
                state_confidence=self.state_confidence,
            )

            activity = ActivityContext(
                current_activities=tuple(self.current_activities),
                primary_activity=self.primary_activity,
                confidence=self.activity_confidence,
                evidence_summary=self.activity_evidence,
                activity_history=tuple(self.activity_history),
            )

            zones = SpatialContext(
                current_zones=tuple(self.current_zones),
                recent_zones=tuple(self.recent_zones),
                spatial_relations=tuple(self.spatial_relations),
                nearby_entities=tuple(self.nearby_entities),
            )

            journey = JourneyContext(
                current_camera=self.current_camera,
                camera_sequence=tuple(self.camera_sequence),
                sighting_count=self.sighting_count,
                journey_state=self.journey_state,
                first_camera=self.camera_sequence[0]
                if self.camera_sequence
                else self.current_camera,
                last_transition_time=self.last_transition_time,
            )

            relationships = RelationshipContext(
                related_entities=tuple(self.related_entities),
                relationship_types=tuple(self.relationship_types),
                interaction_history=tuple(self.interaction_history),
                active_relationships=tuple(self.active_relationships_data),
            )

            events = EventContext(
                active_events=tuple(self.active_events),
                recent_events=tuple(self.recent_events),
            )

            behavior_ctx = None
            if hasattr(self, "current_behaviors") and self.current_behaviors:
                from vantage.entity.contracts import TemporalBehaviorContext

                behavior_ctx = TemporalBehaviorContext(
                    behaviors=tuple(self.current_behaviors),
                    primary_behavior=self.primary_behavior,
                    confidence=self.behavior_confidence,
                    evidence=self.behavior_evidence,
                )

            return EntitySnapshot(
                global_id=self.global_id,
                label=self.label,
                identity=identity_evidence,
                spatial=spatial,
                kinematics=kinematics,
                activity=activity,
                zones=zones,
                journey=journey,
                relationships=relationships,
                events=events,
                timestamp=self.last_seen_wall,
                behavior=behavior_ctx,
            )

    def update_behavior(
        self,
        behaviors: list[str] | tuple[str, ...],
        primary: str,
        confidence: float = 1.0,
        evidence: str = "",
    ) -> None:
        """Update deterministic temporal behavioral interpretation on this entity."""
        with self._lock:
            self.current_behaviors = list(behaviors)
            self.primary_behavior = primary
            self.behavior_confidence = confidence
            self.behavior_evidence = evidence

    def update_persistent_relationships(
        self,
        related_entities: Sequence[str],
        relationship_types: Sequence[str],
        active_relationships: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Update long-horizon persistent relationship associations."""
        with self._lock:
            self.related_entities = list(related_entities)
            self.relationship_types = list(relationship_types)
            if active_relationships is not None:
                self.active_relationships_data = list(active_relationships)
