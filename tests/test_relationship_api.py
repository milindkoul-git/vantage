"""Unit tests for Phase 18 Dashboard API relationship endpoints."""

from __future__ import annotations

import pytest

from vantage.dashboard.api import DashboardApi
from vantage.relationship.service import RelationshipService
from vantage.scene.graph import SceneGraphSnapshot, TransientInteractionEdge
from vantage.storage.sqlite_store import SqliteStore


def test_dashboard_api_relationships_and_graph(tmp_path: pytest.TempPathFactory) -> None:
    db_file = tmp_path / "test_api_relationships.db"
    store = SqliteStore(str(db_file))

    # Mock pipeline with relationship service
    class MockPipeline:
        def __init__(self, store: SqliteStore) -> None:
            self.relationship_service = RelationshipService(store=store)
            self.relationship_tracker = self.relationship_service.tracker

    pipeline = MockPipeline(store)

    edge = TransientInteractionEdge("p1", "p2", "near", 0.06, 0.9, "near")
    snap = SceneGraphSnapshot("cam_01", 100.0, 2, (edge,), (), ())
    entities = [("p1", 0.4, 0.5, 0.5, 0.0), ("p2", 0.45, 0.5, 0.5, 0.0)]

    for i in range(5):
        pipeline.relationship_tracker.process_frame(
            "cam_01", entities, snap, None, now=100.0 + i
        )

    api = DashboardApi(store=store, pipeline=pipeline)

    # 1. GET /api/relationships
    res_all = api.handle("relationships", {})
    assert res_all["available"] is True
    assert res_all["count"] == 1
    rel = res_all["relationships"][0]
    assert rel["entity_a"] == "p1"
    assert rel["entity_b"] == "p2"
    assert "score_breakdown" in rel

    # 2. GET /api/relationships?entity_id=p1
    res_ent = api.handle("relationships", {"entity_id": "p1"})
    assert res_ent["count"] == 1
    assert res_ent["relationships"][0]["entity_b"] == "p2"

    # 3. GET /api/relationships/graph
    res_graph = api.handle("relationships/graph", {})
    assert res_graph["available"] is True
    graph = res_graph["graph"]
    assert graph["total_nodes"] == 2
    assert graph["total_edges"] == 1
    assert graph["edges"][0]["source"] == "p1"
    assert graph["edges"][0]["target"] == "p2"

    store.close()
