"""Adversarial & Stress Robustness Test Suite for Phase 19 Incident Intelligence."""

from __future__ import annotations

import concurrent.futures

import pytest

from vantage.incident.models import IncidentCorrelationDecision, IncidentState
from vantage.incident.service import IncidentService
from vantage.storage.sqlite_store import SqliteStore


# 1. False Correlation Resistance (Two unrelated people in the same room)
def test_false_correlation_resistance_same_room() -> None:
    service = IncidentService()

    ev1 = {
        "id": 1,
        "rule": "exclusion_breach",
        "severity": "alert",
        "entity_id": "person_alpha",
        "camera_id": "cam_01",
        "zone": "lobby_restricted",
        "timestamp": 100.0,
        "summary": "Alpha in lobby",
    }
    inc1, _, _ = service.ingest_event(ev1, now=100.0)

    # 30s later, completely unrelated person_beta has an unrelated loitering event in same room
    ev2 = {
        "id": 2,
        "rule": "loitering",
        "severity": "notice",
        "entity_id": "person_beta",
        "camera_id": "cam_01",
        "zone": "lobby_restricted",
        "timestamp": 130.0,
        "summary": "Beta loitering",
    }
    inc2, decision, _ = service.ingest_event(ev2, now=130.0)

    # Zone coincidence alone MUST NOT attach person_beta into person_alpha's incident
    assert inc2.incident_id != inc1.incident_id
    assert decision != IncidentCorrelationDecision.ATTACH


# 2. Same-Entity Separation Across 4-Hour Gap
def test_same_entity_temporal_separation() -> None:
    service = IncidentService()

    # Morning incident at 09:00 (t=0)
    ev_morning = {
        "id": 10,
        "rule": "exclusion_breach",
        "severity": "alert",
        "entity_id": "person_x",
        "camera_id": "cam_01",
        "timestamp": 1000.0,
        "summary": "Person X morning breach",
    }
    inc_morning, _, _ = service.ingest_event(ev_morning, now=1000.0)

    # Afternoon event 4 hours later (14400s gap)
    ev_afternoon = {
        "id": 20,
        "rule": "loitering",
        "severity": "notice",
        "entity_id": "person_x",
        "camera_id": "cam_01",
        "timestamp": 1000.0 + 14400.0,
        "summary": "Person X afternoon loiter",
    }
    inc_afternoon, decision, _ = service.ingest_event(ev_afternoon, now=1000.0 + 14400.0)

    # Morning incident must be resolved, afternoon must be a new incident
    assert inc_morning.state == IncidentState.RESOLVED
    assert inc_afternoon.incident_id != inc_morning.incident_id
    assert decision == IncidentCorrelationDecision.NEW_INCIDENT


# 3. Multi-Camera Continuation vs Impossible Jump
def test_multi_camera_continuation_and_jump_penalty() -> None:
    service = IncidentService()

    ev1 = {
        "id": 31,
        "rule": "exclusion_breach",
        "severity": "alert",
        "entity_id": "person_runner",
        "camera_id": "cam_01",
        "timestamp": 100.0,
        "summary": "Runner on cam 1",
    }
    inc1, _, _ = service.ingest_event(ev1, now=100.0)

    # Valid handover 15s later to adjacent cam_02
    ev2 = {
        "id": 32,
        "rule": "loitering",
        "severity": "notice",
        "entity_id": "person_runner",
        "camera_id": "cam_02",
        "timestamp": 115.0,
        "summary": "Runner on cam 2",
    }
    inc2, dec2, _ = service.ingest_event(ev2, now=115.0)
    assert inc2.incident_id == inc1.incident_id
    assert dec2 == IncidentCorrelationDecision.ATTACH

    # Impossible jump 0.5s later to disjoint cam_09 -> penalized and separated
    ev3 = {
        "id": 33,
        "rule": "loitering",
        "entity_id": "person_runner",
        "camera_id": "cam_09",
        "timestamp": 115.5,
        "summary": "Runner impossible jump to cam 9",
    }
    _inc3, dec3, _ = service.ingest_event(ev3, now=115.5)
    assert dec3 != IncidentCorrelationDecision.ATTACH


# 4. Concurrency Stress Across 8 Camera Worker Threads
def test_multi_camera_concurrent_incident_ingestion(tmp_path: pytest.TempPathFactory) -> None:
    db_file = tmp_path / "test_concurrent_incidents.db"
    store = SqliteStore(str(db_file))
    service = IncidentService(store=store)

    def worker_stream(cam_idx: int) -> None:
        cam_id = f"cam_{cam_idx:02d}"
        for step in range(25):
            t = 100.0 + cam_idx * 5.0 + step * 1.5
            ev = {
                "id": cam_idx * 100 + step,
                "rule": "loitering" if step % 2 == 0 else "exclusion_breach",
                "severity": "alert" if step % 2 != 0 else "notice",
                "entity_id": f"worker_person_{cam_idx}",
                "camera_id": cam_id,
                "timestamp": t,
                "summary": f"Worker entity on {cam_id}",
            }
            service.ingest_event(ev, now=t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker_stream, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Verify 8 distinct clean incidents created with strictly ordered timelines
    incidents = service.get_incidents()
    assert len(incidents) >= 8

    for inc in incidents:
        # Check timeline chronological ordering
        for k in range(1, len(inc.timeline)):
            assert inc.timeline[k].timestamp >= inc.timeline[k - 1].timestamp

    store.close()
