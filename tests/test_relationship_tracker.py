"""Unit tests for PersistentRelationshipTracker with candidate pair gating and event candidate emission."""

from __future__ import annotations

from vantage.relationship.tracker import PersistentRelationshipTracker
from vantage.scene.graph import SceneGraphSnapshot, TransientInteractionEdge


def test_tracker_candidate_gating_and_milestones() -> None:
    tracker = PersistentRelationshipTracker()

    edge = TransientInteractionEdge(
        source_id="person_1",
        target_id="person_2",
        relation="near",
        distance_norm=0.08,
        confidence=0.90,
        evidence="test proximity",
    )
    snap = SceneGraphSnapshot(
        camera_id="cam_01",
        timestamp=100.0,
        entity_count=2,
        active_edges=(edge,),
        collective_behaviors=(),
        unattended_objects=(),
    )

    entities = [
        ("person_1", 0.4, 0.5, 0.5, 0.0),
        ("person_2", 0.45, 0.5, 0.5, 0.0),
    ]

    # Process 6 frames of recurrent proximity
    all_cands = []
    for i in range(6):
        t = 100.0 + i * 1.0
        cands = tracker.process_frame(
            camera_id="cam_01",
            active_entities=entities,
            scene_graph=snap,
            entity_trajectories=None,
            now=t,
        )
        all_cands.extend(cands)

    rel = tracker.get_relationship("person_1", "person_2")
    assert rel is not None
    assert rel.proximity_count >= 5
    assert rel.active_strength > 0.30

    # Verify event candidate was raised once proximity threshold (5x) was reached
    assert any(c.rule == "recurring_proximity" for c in all_cands)

    # Top associates
    associates = tracker.get_top_associates("person_1")
    assert associates == ["person_2"]
