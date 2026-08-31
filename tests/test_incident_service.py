"""Unit tests for IncidentService lifecycle, escalation, and SQLite persistence."""

from __future__ import annotations

import pytest

from vantage.incident.models import IncidentState
from vantage.incident.service import IncidentService
from vantage.storage.sqlite_store import SqliteStore


def test_service_lifecycle_and_escalation() -> None:
    service = IncidentService()

    # 1. Ingest initial NOTICE event
    ev1 = {
        "id": 1,
        "rule": "loitering",
        "severity": "notice",
        "entity_id": "person_1",
        "camera_id": "cam_01",
        "timestamp": 100.0,
        "summary": "Person 1 loitering",
    }
    inc1, _dec1, cands1 = service.ingest_event(ev1, now=100.0)
    assert inc1.state == IncidentState.ACTIVE
    assert inc1.severity == "notice"
    assert len(cands1) == 0

    # 2. Ingest ALERT event for same entity 10s later -> triggers incident_escalation
    ev2 = {
        "id": 2,
        "rule": "exclusion_breach",
        "severity": "alert",
        "entity_id": "person_1",
        "camera_id": "cam_01",
        "zone": "restricted_vault",
        "timestamp": 110.0,
        "summary": "Person 1 entered vault",
    }
    inc2, _dec2, cands2 = service.ingest_event(ev2, now=110.0)
    assert inc2.incident_id == inc1.incident_id
    assert inc2.severity == "alert"
    assert any(c.rule == "incident_escalation" for c in cands2)

    # 3. Advance time to 200s (90s after last event -> QUIESCENT)
    service._advance_lifecycle(200.0)
    assert inc2.state == IncidentState.QUIESCENT

    # 4. Advance time to 500s (390s after last event -> RESOLVED)
    service._advance_lifecycle(500.0)
    assert inc2.state == IncidentState.RESOLVED


def test_service_sqlite_persistence_and_hydration(tmp_path: pytest.TempPathFactory) -> None:
    db_file = tmp_path / "test_incidents.db"
    store = SqliteStore(str(db_file))

    service1 = IncidentService(store=store)
    ev = {
        "id": 10,
        "rule": "sudden_collapse",
        "severity": "alert",
        "entity_id": "person_victim",
        "camera_id": "cam_02",
        "timestamp": 100.0,
        "summary": "Person collapsed",
    }
    inc, _, _ = service1.ingest_event(ev, now=100.0)
    service1.persist_to_store()

    # Create new service instance and verify hydration
    service2 = IncidentService(store=store)
    hydrated = service2.get_incident(inc.incident_id)
    assert hydrated is not None
    assert hydrated.incident_id == inc.incident_id
    assert "person_victim" in hydrated.involved_entities
    assert hydrated.severity == "alert"

    store.close()
