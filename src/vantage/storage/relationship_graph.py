"""Persistent relationship graph storage and accumulation.

Tracks entity-to-entity and entity-to-zone interaction edges across time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vantage.spatial.contracts import RelationObservation


@dataclass(slots=True)
class RelationshipRecord:
    """A persistent relationship edge."""

    id: int | None
    camera_id: str
    entity_a: str
    entity_b_or_zone: str
    relation_type: str
    first_seen: float
    last_seen: float
    occurrence_count: int
    max_confidence_tier: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "entity_a": self.entity_a,
            "entity_b_or_zone": self.entity_b_or_zone,
            "relation_type": self.relation_type,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "occurrence_count": self.occurrence_count,
            "max_confidence_tier": self.max_confidence_tier,
            "evidence": self.evidence,
        }


class RelationshipGraphAccumulator:
    """Accumulates frame spatial relations and prepares batch database records."""

    def __init__(self, min_confidence: float = 0.4) -> None:
        self._min_confidence = min_confidence
        # Key: (camera_id, entity_a, entity_b_or_zone, relation_type)
        self._edges: dict[tuple[str, str, str, str], RelationshipRecord] = {}

    def observe(
        self,
        camera_id: str,
        timestamp: float,
        relations: list[RelationObservation],
    ) -> None:
        """Observe spatial relations from a frame."""
        for rel in relations:
            if rel.confidence < self._min_confidence:
                continue

            subj = rel.subject_id
            obj = rel.object_id
            rel_type = rel.relation.value
            key = (camera_id, subj, obj, rel_type)

            if key not in self._edges:
                self._edges[key] = RelationshipRecord(
                    id=None,
                    camera_id=camera_id,
                    entity_a=subj,
                    entity_b_or_zone=obj,
                    relation_type=rel_type,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    occurrence_count=1,
                    max_confidence_tier=rel.confidence,
                    evidence={"last_distance": rel.distance, "why": rel.evidence},
                )
            else:
                edge = self._edges[key]
                edge.last_seen = timestamp
                edge.occurrence_count += 1
                if rel.confidence > edge.max_confidence_tier:
                    edge.max_confidence_tier = rel.confidence
                    edge.evidence = {"last_distance": rel.distance, "why": rel.evidence}

    def flush_records(self) -> list[dict[str, Any]]:
        """Extract records ready for database write and reset buffer."""
        records = [
            {
                "camera_id": e.camera_id,
                "entity_a": e.entity_a,
                "entity_b_or_zone": e.entity_b_or_zone,
                "relation_type": e.relation_type,
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
                "occurrence_count": e.occurrence_count,
                "max_confidence_tier": round(e.max_confidence_tier, 3),
                "evidence": json.dumps(e.evidence),
            }
            for e in self._edges.values()
        ]
        self._edges.clear()
        return records
