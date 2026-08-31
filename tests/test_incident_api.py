"""Unit tests for Phase 19 Dashboard API incident routes and search."""

from __future__ import annotations

import pytest

from vantage.dashboard.api import DashboardApi
from vantage.incident.service import IncidentService
from vantage.search.semantic import IncidentSearch
from vantage.storage.sqlite_store import SqliteStore


def test_dashboard_api_incidents(tmp_path: pytest.TempPathFactory) -> None:
    db_file = tmp_path / "test_api_incidents.db"
    store = SqliteStore(str(db_file))

    class MockPipeline:
        def __init__(self, store: SqliteStore) -> None:
            self.incident_service = IncidentService(store=store)

    pipeline = MockPipeline(store)
    ev = {
        "id": 101,
        "rule": "exclusion_breach",
        "severity": "alert",
        "entity_id": "person_intruder",
        "camera_id": "cam_01",
        "zone": "vault",
        "timestamp": 100.0,
        "summary": "Intruder in vault",
    }
    inc, _, _ = pipeline.incident_service.ingest_event(ev, now=100.0)

    api = DashboardApi(store=store, pipeline=pipeline)

    # 1. GET /api/incidents
    res_list = api.handle("incidents", {})
    assert res_list["available"] is True
    assert res_list["count"] == 1
    assert res_list["incidents"][0]["incident_id"] == inc.incident_id

    # 2. GET /api/incident?id=...
    res_det = api.handle("incident", {"id": inc.incident_id})
    assert res_det["found"] is True
    assert res_det["incident"]["incident_id"] == inc.incident_id

    # 3. GET /api/incident/timeline?id=...
    res_time = api.handle("incident/timeline", {"id": inc.incident_id})
    assert res_time["available"] is True
    assert res_time["count"] == 1
    assert res_time["timeline"][0]["event_type"] == "exclusion_breach"

    # 4. GET /api/incident/dossier?id=...
    res_dos = api.handle("incident/dossier", {"id": inc.incident_id})
    assert res_dos["available"] is True
    assert res_dos["incident_id"] == inc.incident_id
    assert "severity_breakdown" in res_dos

    # 5. IncidentSearch.search_incidents
    pipeline.incident_service.persist_to_store()
    search = IncidentSearch(store=store)
    search_res = search.search_incidents("intruder in vault")
    assert search_res["total"] >= 1

    store.close()
