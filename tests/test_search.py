"""Tests for Natural Language Semantic Incident Search."""

from __future__ import annotations

import time
from pathlib import Path

from vantage.search.semantic import SemanticEventSearch
from vantage.storage.sqlite_store import SqliteStore


def test_semantic_search_intent_parsing(tmp_path: Path) -> None:
    db_path = tmp_path / "test_events.db"
    store = SqliteStore(db_path)

    # Insert test events
    now = time.time()
    store.write_events(
        [
            {
                "id": "ev_1",
                "timestamp": now - 100,
                "camera_id": "cam_04_doorway",
                "rule": "tailgating",
                "severity": "alert",
                "summary": "Tailgating breach detected at CAM_04_DOORWAY",
                "entity_id": "global_person_5",
                "identity": "GLOBAL_PERSON_5",
                "zone": "CAM_04_DOORWAY",
                "frame_index": 100,
                "elapsed_s": 10.0,
                "evidence": {"gap_seconds": 0.8},
            },
            {
                "id": "ev_2",
                "timestamp": now - 50,
                "camera_id": "cam_03_corridor",
                "rule": "loitering",
                "severity": "notice",
                "summary": "GLOBAL_PERSON_2 loitering in CAM_03_CORRIDOR",
                "entity_id": "global_person_2",
                "identity": "GLOBAL_PERSON_2",
                "zone": "CAM_03_CORRIDOR",
                "frame_index": 200,
                "elapsed_s": 20.0,
                "evidence": {"dwell_s": 25.0},
            },
        ]
    )

    search_engine = SemanticEventSearch(store=store)

    # Query 1: search for tailgating
    res1 = search_engine.search("tailgating in doorway")
    assert res1["total"] >= 1
    assert res1["results"][0]["rule"] == "tailgating"
    assert res1["results"][0]["severity"] == "alert"

    # Query 2: search for loitering in corridor
    res2 = search_engine.search("who was loitering in the corridor?")
    assert res2["total"] >= 1
    assert res2["results"][0]["rule"] == "loitering"
    assert res2["results"][0]["entity_id"] == "global_person_2"
