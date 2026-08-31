"""Data Models, State Enums, and Evidence Contracts for Situational Incidents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vantage.core.logging import get_logger

log = get_logger(__name__)


def decode_dossier(row: dict[str, Any]) -> dict[str, Any]:
    """Unpack the stored dossier of one incident row.

    A stored incident keeps its full dossier - timeline, evidence, severity
    breakdown - as JSON in ``dossier_json``; the flat columns beside it exist
    only for indexing. Callers want the dossier, so a row whose JSON will not
    parse is a real fault: it is logged and the flat row returned, rather than
    quietly handing back an empty dict that reads as "this incident has no
    evidence".
    """
    raw = row.get("dossier_json") or "{}"
    try:
        dossier = json.loads(raw)
    except (TypeError, ValueError) as exc:
        log.warning(
            "incident dossier could not be decoded; falling back to the flat row",
            extra={
                "vantage_fields": {"incident_id": row.get("incident_id"), "error": str(exc)}
            },
        )
        return dict(row)
    if not isinstance(dossier, dict) or not dossier:
        return dict(row)
    return dossier


class IncidentState(str, Enum):
    """Lifecycle state machine for a situational incident."""

    ACTIVE = "active"  # Correlated events currently arriving (< 60s)
    QUIESCENT = "quiescent"  # Inactivity >= 60s, but still in continuation window (< 300s)
    RESOLVED = (
        "resolved"  # Inactive >= 300s (Vantage is no longer observing correlated activity)
    )
    EXPIRED = "expired"  # Archived past retention policy


class IncidentCorrelationDecision(str, Enum):
    """Three-way decision band for event-to-incident correlation."""

    ATTACH = "attach"  # High confidence: attach automatically
    CORRELATION_CANDIDATE = (
        "correlation_candidate"  # Ambiguous: record candidate link with evidence
    )
    NEW_INCIDENT = "new_incident"  # Low confidence: spawn new independent incident


@dataclass(frozen=True, slots=True)
class IncidentCorrelationBreakdown:
    """Explainable attribution breakdown detailing positive evidence and negative penalties."""

    entity_overlap_score: float
    temporal_proximity_score: float
    spatial_zone_score: float
    relationship_score: float
    behavior_scene_score: float
    continuity_penalty: float
    positive_score: float
    total_correlation_score: float
    decision: IncidentCorrelationDecision
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_overlap_score": round(self.entity_overlap_score, 3),
            "temporal_proximity_score": round(self.temporal_proximity_score, 3),
            "spatial_zone_score": round(self.spatial_zone_score, 3),
            "relationship_score": round(self.relationship_score, 3),
            "behavior_scene_score": round(self.behavior_scene_score, 3),
            "continuity_penalty": round(self.continuity_penalty, 3),
            "positive_score": round(self.positive_score, 3),
            "total_correlation_score": round(self.total_correlation_score, 3),
            "decision": self.decision.value,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class IncidentSeverityBreakdown:
    """Explainable determination of incident severity anchored in constituent events."""

    highest_event_severity: str
    corroborating_event_count: int
    involved_entity_count: int
    restricted_zone_factor: float
    escalation_factor: float
    final_severity: str
    severity_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "highest_event_severity": self.highest_event_severity,
            "corroborating_event_count": self.corroborating_event_count,
            "involved_entity_count": self.involved_entity_count,
            "restricted_zone_factor": round(self.restricted_zone_factor, 2),
            "escalation_factor": round(self.escalation_factor, 2),
            "final_severity": self.final_severity,
            "severity_score": round(self.severity_score, 3),
        }


@dataclass(frozen=True, slots=True)
class IncidentTimelineEntry:
    """One chronological, provenance-backed observation entry in an incident timeline."""

    entry_id: str
    timestamp: float
    event_id: str | int | None
    event_type: str
    camera_id: str
    entities: tuple[str, ...]
    objects: tuple[str, ...]
    zone: str | None
    summary: str
    evidence_ref: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": round(self.timestamp, 2),
            "event_id": self.event_id,
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "entities": list(self.entities),
            "objects": list(self.objects),
            "zone": self.zone,
            "summary": self.summary,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(slots=True)
class CanonicalIncident:
    """Durable, evolving incident state aggregating events, entities, objects, and timeline dossiers."""

    incident_id: str
    title: str
    state: IncidentState
    severity: str
    severity_breakdown: IncidentSeverityBreakdown
    first_seen: float
    last_seen: float
    cameras: set[str] = field(default_factory=set)
    zones: set[str] = field(default_factory=set)
    involved_entities: set[str] = field(default_factory=set)
    involved_objects: set[str] = field(default_factory=set)
    timeline: list[IncidentTimelineEntry] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    relationship_links: list[dict[str, Any]] = field(default_factory=list)
    correlation_candidates: list[dict[str, Any]] = field(default_factory=list)
    merge_candidates: list[str] = field(default_factory=list)
    evidence_dossier: dict[str, Any] = field(default_factory=dict)
    correlation_breakdown: dict[str, Any] | None = None
    """Why the most recent event was attached to this incident.

    The correlator already computes a full factor split - entity overlap,
    temporal proximity, spatial continuity, relationship and behaviour, less a
    continuity penalty - and until now it was discarded the moment the decision
    was made. Keeping the winning breakdown is what lets a dossier answer "why
    is this one incident and not two" with the numbers that actually decided it.
    ``None`` on an incident that was spawned rather than attached to."""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "state": self.state.value,
            "severity": self.severity,
            "severity_breakdown": self.severity_breakdown.to_dict(),
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "duration_s": round(self.duration_s, 2),
            "cameras": sorted(self.cameras),
            "zones": sorted(self.zones),
            "involved_entities": sorted(self.involved_entities),
            "involved_objects": sorted(self.involved_objects),
            "event_count": self.event_count,
            "timeline": [t.to_dict() for t in self.timeline],
            "relationship_links": self.relationship_links,
            "correlation_candidates": self.correlation_candidates,
            "merge_candidates": self.merge_candidates,
            "evidence_dossier": self.evidence_dossier,
            "correlation_breakdown": self.correlation_breakdown,
        }
