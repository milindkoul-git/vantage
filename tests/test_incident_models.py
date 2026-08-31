"""Unit tests for Phase 19 Incident Models, Decisions, and Explainable Severities."""

from __future__ import annotations

import re

import pytest

from vantage.incident.config import IncidentCorrelatorConfig
from vantage.incident.models import (
    CanonicalIncident,
    IncidentSeverityBreakdown,
    IncidentState,
    IncidentTimelineEntry,
)


def test_incident_models_serialization() -> None:
    tle = IncidentTimelineEntry(
        entry_id="tle_01",
        timestamp=100.0,
        event_id="ev_101",
        event_type="exclusion_breach",
        camera_id="cam_01",
        entities=("person_1",),
        objects=(),
        zone="restricted_vault",
        summary="Person 1 entered restricted vault",
        evidence_ref={"clip_url": "/clips/ev_101.mp4"},
    )
    tle_dict = tle.to_dict()
    assert tle_dict["entry_id"] == "tle_01"
    assert tle_dict["event_type"] == "exclusion_breach"

    sev_breakdown = IncidentSeverityBreakdown(
        highest_event_severity="alert",
        corroborating_event_count=3,
        involved_entity_count=2,
        restricted_zone_factor=0.20,
        escalation_factor=0.10,
        final_severity="alert",
        severity_score=0.95,
    )

    inc = CanonicalIncident(
        incident_id="inc_test_01",
        title="Restricted Area Breach on cam_01",
        state=IncidentState.ACTIVE,
        severity="alert",
        severity_breakdown=sev_breakdown,
        first_seen=100.0,
        last_seen=125.0,
        cameras={"cam_01"},
        zones={"restricted_vault"},
        involved_entities={"person_1"},
        timeline=[tle],
        events=[{"rule": "exclusion_breach", "severity": "alert"}],
    )

    d = inc.to_dict()
    assert d["incident_id"] == "inc_test_01"
    assert d["state"] == "active"
    assert d["severity"] == "alert"
    assert d["duration_s"] == 25.0
    assert len(d["timeline"]) == 1


def test_correlator_config_validation() -> None:
    # Valid config
    cfg = IncidentCorrelatorConfig()
    assert cfg.attach_threshold == 0.65
    assert cfg.candidate_threshold == 0.35

    # Invalid weights sum
    with pytest.raises(
        ValueError, match=re.escape("positive correlation weights must sum to 1.0")
    ):
        IncidentCorrelatorConfig(entity_overlap_weight=0.90)

    # Invalid thresholds
    with pytest.raises(ValueError, match="invalid decision thresholds"):
        IncidentCorrelatorConfig(candidate_threshold=0.80, attach_threshold=0.60)
