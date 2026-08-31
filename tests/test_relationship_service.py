"""Unit tests for RelationshipService SQLite persistence and graph export."""

from __future__ import annotations

import pytest

from vantage.relationship.service import RelationshipService
from vantage.scene.graph import SceneGraphSnapshot, TransientInteractionEdge
from vantage.storage.sqlite_store import SqliteStore


def test_service_persistence_and_hydration(tmp_path: pytest.TempPathFactory) -> None:
    db_file = tmp_path / "test_relationships.db"
    store = SqliteStore(str(db_file))

    service1 = RelationshipService(store=store)

    edge = TransientInteractionEdge("p1", "p2", "near", 0.05, 0.9, "near")
    snap = SceneGraphSnapshot("cam_01", 10.0, 2, (edge,), (), ())
    entities = [("p1", 0.5, 0.5, 0.5, 0.0), ("p2", 0.52, 0.5, 0.5, 0.0)]

    for i in range(5):
        service1.tracker.process_frame("cam_01", entities, snap, None, now=10.0 + i)

    # Persist to disk
    persisted_count = service1.persist_to_store()
    assert persisted_count > 0

    # Create new service instance and verify hydration
    service2 = RelationshipService(store=store)
    rel = service2.tracker.get_relationship("p1", "p2")
    assert rel is not None
    assert rel.co_occurrence_count >= 5

    # Graph snapshot
    graph = service2.get_graph_snapshot()
    assert graph["total_nodes"] == 2
    assert graph["total_edges"] == 1
    assert graph["edges"][0]["source"] == "p1"
    assert graph["edges"][0]["target"] == "p2"

    store.close()
