"""Tests for entity timeline projection (temporal scene memory)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.storage.entity_timeline import build_entity_timeline
from vantage.storage.schema import wrap_list
from vantage.storage.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    created = SqliteStore(tmp_path / "timeline_test.db")
    yield created
    created.close()


def test_entity_timeline_projection(store: SqliteStore) -> None:
    # 1. Insert sequence of observations for person_1
    obs = [
        {
            "timestamp": 100.0,
            "camera_id": "cam0",
            "entity_id": "person_1",
            "identity": None,
            "entity_type": "person",
            "motion": "stationary",
            "speed": 0.02,
            "posture": "sitting",
            "zones": wrap_list(["lobby"]),
            "activities": wrap_list(["idle"]),
            "frame_index": 1,
            "elapsed_s": 0.0,
        },
        {
            "timestamp": 101.0,
            "camera_id": "cam0",
            "entity_id": "person_1",
            "identity": None,
            "entity_type": "person",
            "motion": "stationary",
            "speed": 0.01,
            "posture": "sitting",
            "zones": wrap_list(["lobby"]),
            "activities": wrap_list(["idle"]),
            "frame_index": 30,
            "elapsed_s": 1.0,
        },
        {
            "timestamp": 102.0,
            "camera_id": "cam0",
            "entity_id": "person_1",
            "identity": "alice",
            "entity_type": "person",
            "motion": "moving",
            "speed": 0.5,
            "posture": "standing",
            "zones": wrap_list(["hallway"]),
            "activities": wrap_list(["walking"]),
            "frame_index": 60,
            "elapsed_s": 2.0,
        },
    ]
    store.write_observations(obs)

    # 2. Insert event
    events = [
        {
            "timestamp": 102.0,
            "camera_id": "cam0",
            "rule": "zone_entry",
            "severity": "info",
            "summary": "person_1 entered hallway",
            "entity_id": "person_1",
            "identity": "alice",
            "related_id": None,
            "zone": "hallway",
            "frame_index": 60,
            "elapsed_s": 2.0,
            "evidence": {"zone": "hallway"},
        }
    ]
    store.write_events(events)

    # 3. Build timeline
    timeline = build_entity_timeline(store, "person_1")
    assert timeline is not None
    assert timeline.entity_id == "person_1"
    assert timeline.identity == "alice"
    assert timeline.first_seen == 100.0
    assert timeline.last_seen == 102.0
    assert len(timeline.segments) == 2  # sitting in lobby -> walking in hallway
    assert len(timeline.events) == 1
    assert timeline.events[0].rule == "zone_entry"

    d = timeline.to_dict()
    assert d["entity_id"] == "person_1"
    assert len(d["segments"]) == 2
    assert "alice" in d["summary"]


def test_entity_timeline_not_found(store: SqliteStore) -> None:
    timeline = build_entity_timeline(store, "non_existent_entity")
    assert timeline is None
