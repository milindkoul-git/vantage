"""Transient Scene Graph & Group Intelligence.

Evaluates multi-entity spatial relationships, collective dynamics, and ownership-verified
unattended objects over short time horizons.

Preserves architectural boundaries:
- Phase 17: Transient scene relationships & collective dynamics
- Phase 18: Persistent multi-session cross-camera relationships
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vantage.events.contracts import EventCandidate
from vantage.perception.contracts import BoundingBox, Detection
from vantage.scene.window import SceneTemporalWindow
from vantage.tracking.contracts import Track


@dataclass(frozen=True, slots=True)
class TransientInteractionEdge:
    """One instantaneous spatial interaction edge between two entities or entity and object."""

    source_id: str
    target_id: str
    relation: str  # 'approaching' | 'near' | 'trailing' | 'interacting_with'
    distance_norm: float
    confidence: float
    evidence: str


@dataclass(frozen=True, slots=True)
class CollectiveSceneBehavior:
    """A collective multi-entity scene-level phenomenon."""

    behavior_type: str  # 'group_convergence' | 'group_dispersion' | 'high_crowd_density'
    entity_ids: tuple[str, ...]
    centroid: tuple[float, float]
    confidence: float
    evidence: str


@dataclass
class _UnattendedObjectState:
    """Lifecycle tracking for an object with verified prior human association."""

    object_id: str
    label: str
    box: BoundingBox
    initial_owner_id: str
    associated_time: float
    confidence: float = 1.0
    source: str = "hoi_fusion"
    unattended_start_time: float | None = None
    last_seen_time: float = 0.0
    alert_emitted: bool = False
    third_party_interactor: str | None = None


@dataclass(frozen=True, slots=True)
class SceneGraphSnapshot:
    """Immutable point-in-time snapshot of the transient scene graph."""

    camera_id: str
    timestamp: float
    entity_count: int
    active_edges: tuple[TransientInteractionEdge, ...]
    collective_behaviors: tuple[CollectiveSceneBehavior, ...]
    unattended_objects: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": round(self.timestamp, 2),
            "entity_count": self.entity_count,
            "active_edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation,
                    "distance": round(e.distance_norm, 3),
                    "confidence": round(e.confidence, 2),
                    "evidence": e.evidence,
                }
                for e in self.active_edges
            ],
            "collective_behaviors": [
                {
                    "type": b.behavior_type,
                    "entities": list(b.entity_ids),
                    "centroid": [round(c, 3) for c in b.centroid],
                    "confidence": round(b.confidence, 2),
                    "evidence": b.evidence,
                }
                for b in self.collective_behaviors
            ],
            "unattended_objects": list(self.unattended_objects),
        }


class TransientSceneGraph:
    """Evaluates transient multi-entity spatial interactions, group dynamics, and object lifecycles."""

    def __init__(
        self,
        camera_id: str,
        *,
        proximity_threshold_norm: float = 0.15,
        crowd_density_threshold: int = 4,
        unattended_dwell_s: float = 25.0,
        adaptive_perspective: bool = True,
    ) -> None:
        self.camera_id = camera_id
        self.proximity_threshold_norm = proximity_threshold_norm
        self.crowd_density_threshold = crowd_density_threshold
        self.unattended_dwell_s = unattended_dwell_s
        self.adaptive_perspective = adaptive_perspective

        self.scene_window = SceneTemporalWindow(max_samples=30, max_span_s=3.0)
        self._tracked_objects: dict[str, _UnattendedObjectState] = {}
        self._last_snapshot: SceneGraphSnapshot | None = None

    def register_ownership(
        self,
        object_id: str,
        label: str,
        box: BoundingBox,
        owner_id: str,
        now: float,
        confidence: float = 1.0,
        source: str = "hoi_fusion",
    ) -> None:
        """Register verified human-object ownership (e.g. from HOIFusionEngine)."""
        self._tracked_objects[object_id] = _UnattendedObjectState(
            object_id=object_id,
            label=label,
            box=box,
            initial_owner_id=owner_id,
            associated_time=now,
            confidence=confidence,
            source=source,
            last_seen_time=now,
        )

    def update(
        self,
        tracks: Sequence[Track],
        raw_detections: Sequence[Detection] | None,
        now: float,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> tuple[SceneGraphSnapshot, list[EventCandidate]]:
        """Evaluate scene interactions, collective behaviors, and unattended objects for one frame."""
        w_f = max(1, frame_width)
        h_f = max(1, frame_height)

        # 1. Compute normalized foot points and normalized heights for active entity tracks
        entity_positions: list[
            tuple[str, float, float, float]
        ] = []  # (entity_id, fx, fy, norm_height)
        for t in tracks:
            fx = (t.box.x1 + t.box.x2) / (2.0 * w_f)
            fy = t.box.y2 / h_f
            nh = t.box.height / h_f
            entity_positions.append((t.entity_id, fx, fy, nh))

        # 2. Update SceneTemporalWindow
        self.scene_window.add(
            timestamp=now,
            camera_id=self.camera_id,
            entities=[(e[0], e[1], e[2]) for e in entity_positions],
        )
        convergence = self.scene_window.extract_convergence()

        # 3. Build transient pairwise interaction edges (perspective adaptive)
        edges: list[TransientInteractionEdge] = []
        n_ents = len(entity_positions)
        for i in range(n_ents):
            id_a, ax, ay, ha = entity_positions[i]
            for j in range(i + 1, n_ents):
                id_b, bx, by, hb = entity_positions[j]
                dist = math.hypot(ax - bx, ay - by)

                # Adaptive threshold: scale by mean entity height relative to baseline 0.20
                if self.adaptive_perspective:
                    mean_h = max(0.05, (ha + hb) / 2.0)
                    pair_thresh = max(
                        0.08, min(0.35, self.proximity_threshold_norm * (mean_h / 0.18))
                    )
                else:
                    pair_thresh = self.proximity_threshold_norm

                if dist <= pair_thresh:
                    edges.append(
                        TransientInteractionEdge(
                            source_id=id_a,
                            target_id=id_b,
                            relation="near",
                            distance_norm=dist,
                            confidence=0.85,
                            evidence=f"spatial separation {dist:.3f} <= adaptive threshold {pair_thresh:.3f}",
                        )
                    )

        # 4. Detect Collective Behaviors
        collective: list[CollectiveSceneBehavior] = []
        candidates: list[EventCandidate] = []

        all_ids = tuple(e[0] for e in entity_positions)
        last_sample = self.scene_window.samples[-1] if self.scene_window.samples else None
        centroid = last_sample.centroid if last_sample else (0.5, 0.5)

        # 4a. Group Convergence (3+ entities closing distance rapidly)
        if convergence.is_converging:
            cb = CollectiveSceneBehavior(
                behavior_type="group_convergence",
                entity_ids=all_ids,
                centroid=centroid,
                confidence=0.88,
                evidence=f"{convergence.entity_count} entities converging at spread rate {convergence.spread_rate:.2f}/s",
            )
            collective.append(cb)
            candidates.append(
                EventCandidate(
                    rule="group_convergence",
                    severity="notice",
                    summary=f"Rapid group convergence ({convergence.entity_count} entities) in {self.camera_id}",
                    camera_id=self.camera_id,
                    wall_time=now,
                    evidence={
                        "entity_count": convergence.entity_count,
                        "spread_rate": convergence.spread_rate,
                    },
                )
            )

        # 4b. Group Dispersion (3+ entities scattering rapidly)
        if convergence.is_dispersing:
            cb = CollectiveSceneBehavior(
                behavior_type="group_dispersion",
                entity_ids=all_ids,
                centroid=centroid,
                confidence=0.85,
                evidence=f"{convergence.entity_count} entities dispersing at spread rate {convergence.spread_rate:.2f}/s",
            )
            collective.append(cb)
            candidates.append(
                EventCandidate(
                    rule="group_dispersion",
                    severity="notice",
                    summary=f"Sudden group dispersion ({convergence.entity_count} entities) in {self.camera_id}",
                    camera_id=self.camera_id,
                    wall_time=now,
                    evidence={
                        "entity_count": convergence.entity_count,
                        "spread_rate": convergence.spread_rate,
                    },
                )
            )

        # 4c. Local Crowd Density
        if (
            n_ents >= self.crowd_density_threshold
            and last_sample
            and last_sample.spread_radius < 0.20
        ):
            cb = CollectiveSceneBehavior(
                behavior_type="high_crowd_density",
                entity_ids=all_ids,
                centroid=centroid,
                confidence=0.90,
                evidence=f"{n_ents} entities concentrated within radius {last_sample.spread_radius:.2f}",
            )
            collective.append(cb)

        # 5. Evaluate Ownership-Verified Unattended Objects & Third-Party Interaction
        unattended_info: list[dict[str, Any]] = []
        for obj_id, ostate in list(self._tracked_objects.items()):
            # Purge stale object entries exceeding 300s
            if now - ostate.associated_time > 300.0 and (now - ostate.last_seen_time > 120.0):
                del self._tracked_objects[obj_id]
                continue

            obj_cx = (ostate.box.x1 + ostate.box.x2) / (2.0 * w_f)
            obj_cy = (ostate.box.y1 + ostate.box.y2) / (2.0 * h_f)

            # Find distance to initial owner
            owner_pos = next(
                (pos for pos in entity_positions if pos[0] == ostate.initial_owner_id), None
            )
            if owner_pos is not None:
                owner_dist = math.hypot(obj_cx - owner_pos[1], obj_cy - owner_pos[2])
            else:
                owner_dist = 1.0  # Owner departed scene

            # Check if a 3rd party (non-owner) is actively interacting with the object
            third_party = next(
                (
                    pos
                    for pos in entity_positions
                    if pos[0] != ostate.initial_owner_id
                    and math.hypot(obj_cx - pos[1], obj_cy - pos[2]) <= 0.10
                ),
                None,
            )

            if third_party is not None:
                ostate.third_party_interactor = third_party[0]
                ostate.unattended_start_time = None  # Interacted with, not unattended
            elif owner_dist > 0.25:
                ostate.third_party_interactor = None
                if ostate.unattended_start_time is None:
                    ostate.unattended_start_time = now
                dwell = now - ostate.unattended_start_time
                unattended_info.append(
                    {
                        "object_id": obj_id,
                        "label": ostate.label,
                        "owner_id": ostate.initial_owner_id,
                        "source": ostate.source,
                        "confidence": round(ostate.confidence, 2),
                        "unattended_dwell_s": round(dwell, 1),
                        "owner_distance_norm": round(owner_dist, 3),
                    }
                )

                if dwell >= self.unattended_dwell_s and not ostate.alert_emitted:
                    ostate.alert_emitted = True
                    candidates.append(
                        EventCandidate(
                            rule="unattended_object_dwell",
                            severity="alert",
                            summary=f"Unattended {ostate.label} left by {ostate.initial_owner_id} ({dwell:.1f}s dwell)",
                            entity_id=ostate.initial_owner_id,
                            camera_id=self.camera_id,
                            wall_time=now,
                            evidence={
                                "object_id": obj_id,
                                "label": ostate.label,
                                "dwell_s": round(dwell, 1),
                                "owner_distance": round(owner_dist, 3),
                                "ownership_source": ostate.source,
                                "ownership_confidence": ostate.confidence,
                            },
                        )
                    )
            else:
                # Owner is close; reset unattended dwell
                ostate.unattended_start_time = None
                ostate.third_party_interactor = None

        snapshot = SceneGraphSnapshot(
            camera_id=self.camera_id,
            timestamp=now,
            entity_count=n_ents,
            active_edges=tuple(edges),
            collective_behaviors=tuple(collective),
            unattended_objects=tuple(unattended_info),
        )
        self._last_snapshot = snapshot
        return snapshot, candidates

    @property
    def last_snapshot(self) -> SceneGraphSnapshot | None:
        return self._last_snapshot
