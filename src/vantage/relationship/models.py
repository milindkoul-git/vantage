"""Canonical Data Models and Evidence Contracts for Persistent Entity Relationships."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProximityBasis(str, Enum):
    """Measurement frame for spatial proximity observations."""

    WORLD_SPACE = "world_space"  # Calibrated metric ground-plane coordinates (meters)
    NORMALIZED_IMAGE_SPACE = (
        "normalized_image_space"  # Normalized image-plane frame coordinates
    )


class RelationshipSignalType(str, Enum):
    """Fundamental observational evidence primitives."""

    CO_OCCURRENCE = "co_occurrence"
    RECURRENT_PROXIMITY = "recurrent_proximity"
    LAGGED_TRAJECTORY_ALIGNMENT = "lagged_trajectory_alignment"
    REPEATED_GROUP_CO_CLUSTERING = "repeated_group_co_clustering"
    SHARED_ZONE_PRESENCE = "shared_zone_presence"


class DerivedRelationshipPattern(str, Enum):
    """Higher-level derived relationship patterns supported by observable multi-signal evidence."""

    FOLLOWING_PATTERN_CANDIDATE = "following_pattern_candidate"
    FREQUENT_CO_TRAVELER = "frequent_co_traveler"
    PERSISTENT_CLUSTER_ASSOCIATE = "persistent_cluster_associate"
    RECURRENT_INTERACTION_PAIR = "recurrent_interaction_pair"


@dataclass(frozen=True, slots=True)
class RelationshipScoreBreakdown:
    """Explainable attribution breakdown of contributing evidence signals."""

    co_occurrence_contribution: float
    proximity_contribution: float
    following_contribution: float
    duration_contribution: float
    total_raw_score: float
    active_decayed_score: float
    decay_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "co_occurrence_contribution": round(self.co_occurrence_contribution, 3),
            "proximity_contribution": round(self.proximity_contribution, 3),
            "following_contribution": round(self.following_contribution, 3),
            "duration_contribution": round(self.duration_contribution, 3),
            "total_raw_score": round(self.total_raw_score, 3),
            "active_decayed_score": round(self.active_decayed_score, 3),
            "decay_factor": round(self.decay_factor, 3),
        }


@dataclass(frozen=True, slots=True)
class RelationshipSignal:
    """One discrete observational evidence signal between two entities."""

    signal_type: RelationshipSignalType
    timestamp: float
    camera_id: str
    zone_id: str | None
    strength: float
    duration_s: float
    proximity_basis: ProximityBasis
    distance_metric: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "timestamp": round(self.timestamp, 2),
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "strength": round(self.strength, 3),
            "duration_s": round(self.duration_s, 2),
            "proximity_basis": self.proximity_basis.value,
            "distance_metric": round(self.distance_metric, 3),
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class EntityRelationship:
    """Durable, explainable relationship state between two canonical entities."""

    entity_a: str
    entity_b: str
    active_strength: float
    historical_score: float
    score_breakdown: RelationshipScoreBreakdown
    primary_derived_pattern: DerivedRelationshipPattern | None
    first_observed: float
    last_observed: float
    co_occurrence_count: int
    proximity_count: int
    following_count: int
    total_interaction_duration_s: float
    signals: list[RelationshipSignal] = field(default_factory=list)
    evidence_summary: str = ""

    def __post_init__(self) -> None:
        # Guarantee canonical undirected ordering: entity_a <= entity_b
        if self.entity_a > self.entity_b:
            self.entity_a, self.entity_b = self.entity_b, self.entity_a

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.entity_a, self.entity_b)

    def other_entity(self, entity_id: str) -> str:
        """Given one entity in the pair, return the counterpart entity ID."""
        if entity_id == self.entity_a:
            return self.entity_b
        if entity_id == self.entity_b:
            return self.entity_a
        raise ValueError(
            f"{entity_id!r} is not part of relationship ({self.entity_a}, {self.entity_b})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "active_strength": round(self.active_strength, 3),
            "historical_score": round(self.historical_score, 3),
            "score_breakdown": self.score_breakdown.to_dict(),
            "primary_derived_pattern": (
                self.primary_derived_pattern.value if self.primary_derived_pattern else None
            ),
            "first_observed": round(self.first_observed, 2),
            "last_observed": round(self.last_observed, 2),
            "co_occurrence_count": self.co_occurrence_count,
            "proximity_count": self.proximity_count,
            "following_count": self.following_count,
            "total_interaction_duration_s": round(self.total_interaction_duration_s, 2),
            "evidence_summary": self.evidence_summary,
            "recent_signals": [s.to_dict() for s in self.signals[-10:]],
        }
