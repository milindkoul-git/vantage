"""Unit tests for IncidentCorrelator multi-factor scoring and negative penalties."""

from __future__ import annotations

from vantage.incident.correlator import IncidentCorrelator
from vantage.incident.models import (
    CanonicalIncident,
    IncidentCorrelationDecision,
    IncidentSeverityBreakdown,
    IncidentState,
    IncidentTimelineEntry,
)


def _create_sample_incident(
    inc_id: str = "inc_01", entity: str = "person_a", now: float = 100.0
) -> CanonicalIncident:
    tle = IncidentTimelineEntry(
        entry_id="tle_01",
        timestamp=now,
        event_id="ev_01",
        event_type="exclusion_breach",
        camera_id="cam_01",
        entities=(entity,),
        objects=(),
        zone="vault",
        summary=f"{entity} entered vault",
    )
    sev_b = IncidentSeverityBreakdown("alert", 1, 1, 0.2, 0.0, "alert", 0.85)
    return CanonicalIncident(
        incident_id=inc_id,
        title=f"Incident {inc_id}",
        state=IncidentState.ACTIVE,
        severity="alert",
        severity_breakdown=sev_b,
        first_seen=now,
        last_seen=now,
        cameras={"cam_01"},
        zones={"vault"},
        involved_entities={entity},
        timeline=[tle],
        events=[
            {
                "rule": "exclusion_breach",
                "entity_id": entity,
                "camera_id": "cam_01",
                "timestamp": now,
            }
        ],
    )


def test_high_confidence_attach_decision() -> None:
    correlator = IncidentCorrelator()
    inc = _create_sample_incident(now=100.0)

    # Event 10s later with same entity in same vault
    event = {
        "id": 102,
        "rule": "unattended_object_dwell",
        "entity_id": "person_a",
        "camera_id": "cam_01",
        "zone": "vault",
        "timestamp": 110.0,
        "summary": "person_a left backpack",
    }

    breakdown = correlator.evaluate_correlation(event, inc, now=110.0)
    assert breakdown.decision == IncidentCorrelationDecision.ATTACH
    assert breakdown.total_correlation_score >= 0.65
    assert "shared entity 'person_a'" in breakdown.explanation


def test_ambiguous_correlation_candidate() -> None:
    correlator = IncidentCorrelator()
    inc = _create_sample_incident(now=100.0)

    # Unrelated entity enters the same camera 15s later
    event = {
        "id": 103,
        "rule": "loitering",
        "entity_id": "person_stranger",
        "camera_id": "cam_01",
        "zone": None,
        "timestamp": 115.0,
    }

    breakdown = correlator.evaluate_correlation(event, inc, now=115.0)
    # Zone/camera overlap without entity overlap is supporting only -> ambiguous or new incident
    assert breakdown.decision in (
        IncidentCorrelationDecision.CORRELATION_CANDIDATE,
        IncidentCorrelationDecision.NEW_INCIDENT,
    )
    assert breakdown.total_correlation_score < 0.65


def test_negative_penalty_impossible_camera_jump() -> None:
    correlator = IncidentCorrelator()
    inc = _create_sample_incident(now=100.0)

    # Same entity appears on cam_04 just 0.5s later (physically impossible jump)
    event = {
        "id": 104,
        "rule": "loitering",
        "entity_id": "person_a",
        "camera_id": "cam_04",
        "timestamp": 100.5,
    }

    breakdown = correlator.evaluate_correlation(event, inc, now=100.5)
    assert breakdown.continuity_penalty >= 0.40
    assert "physically implausible camera transition" in breakdown.explanation
