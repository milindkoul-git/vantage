"""Tests for persistent relationship graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.spatial.contracts import Relation, RelationObservation
from vantage.storage.relationship_graph import (
    RelationshipGraphAccumulator,
)
from vantage.storage.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    created = SqliteStore(tmp_path / "rel_test.db")
    yield created
    created.close()


def test_relationship_accumulator_and_store(store: SqliteStore) -> None:
    accumulator = RelationshipGraphAccumulator(min_confidence=0.4)

    # 1. Feed frame relations
    rel1 = RelationObservation(
        relation=Relation.NEAR,
        subject_id="person_1",
        subject_track=1,
        object_id="desk_1",
        object_track=2,
        confidence=0.85,
        distance=0.4,
        duration_s=1.0,
        evidence="wrist landmark inside object box",
    )
    accumulator.observe(camera_id="cam0", timestamp=100.0, relations=[rel1])

    # 2. Feed second observation (updates last_seen and occurrence_count)
    rel2 = RelationObservation(
        relation=Relation.NEAR,
        subject_id="person_1",
        subject_track=1,
        object_id="desk_1",
        object_track=2,
        confidence=0.88,
        distance=0.35,
        duration_s=2.0,
        evidence="wrist landmark inside object box",
    )
    accumulator.observe(camera_id="cam0", timestamp=101.0, relations=[rel2])

    records = accumulator.flush_records()
    assert len(records) == 1
    assert records[0]["entity_a"] == "person_1"
    assert records[0]["entity_b_or_zone"] == "desk_1"
    assert records[0]["occurrence_count"] == 2
    assert records[0]["first_seen"] == 100.0
    assert records[0]["last_seen"] == 101.0
    assert records[0]["max_confidence_tier"] == 0.88

    # 3. Write to SQLite store and query
    written = store.write_relationships(records)
    assert written == 1

    edges = store.relationships(entity_id="person_1")
    assert len(edges) == 1
    assert edges[0]["entity_b_or_zone"] == "desk_1"
    assert edges[0]["max_confidence_tier"] == 0.88

    # Query non-matching entity
    empty = store.relationships(entity_id="person_99")
    assert len(empty) == 0
