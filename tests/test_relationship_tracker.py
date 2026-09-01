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


class TestProximityGating:
    """Which pairs are even considered, which used to be exactly backwards.

    The gate read `if len(entities) <= 5: pair everything`, and above that
    nothing was ever paired on a single-camera run - Gate A needs a scene graph
    that pipeline does not build, and Gate B can only match pairs that already
    exist, so nothing could ever seed. Measured on real clips: a corridor with
    one or two people produced 34 associations, all between fragments of one
    person left by an id switch, and a street with 24 people in frame at once
    produced none at all.
    """

    @staticmethod
    def run(tracker: PersistentRelationshipTracker, entities, frames: int = 4):
        for i in range(frames):
            tracker.process_frame(
                camera_id="cam_01",
                active_entities=entities,
                scene_graph=None,
                entity_trajectories=None,
                now=100.0 + i,
            )
        return tracker.get_all_relationships()

    def test_two_people_standing_together_are_paired(self) -> None:
        tracker = PersistentRelationshipTracker()
        together = [("person_1", 0.40, 0.50, 0.1, None), ("person_2", 0.45, 0.50, 0.1, None)]
        assert len(self.run(tracker, together)) == 1

    def test_two_people_at_opposite_ends_are_not(self) -> None:
        tracker = PersistentRelationshipTracker()
        apart = [("person_1", 0.05, 0.05, 0.1, None), ("person_2", 0.95, 0.95, 0.1, None)]
        assert self.run(tracker, apart) == []

    def test_a_crowd_still_pairs_the_people_who_are_together(self) -> None:
        """The case the old rule switched off entirely.

        Twelve entities, two of them side by side. Under `len <= 5` this frame
        produced nothing at all, for ever.
        """
        crowd = [(f"person_{i}", 0.05 + i * 0.08, 0.9, 0.1, None) for i in range(10)]
        crowd += [("person_a", 0.40, 0.20, 0.1, None), ("person_b", 0.42, 0.21, 0.1, None)]
        tracker = PersistentRelationshipTracker()
        pairs = {(rel.entity_a, rel.entity_b) for rel in self.run(tracker, crowd)}
        assert ("person_a", "person_b") in pairs

    def test_a_crowd_does_not_pair_everyone_with_everyone(self) -> None:
        """Twelve entities is 66 pairs; only the near ones should be scored."""
        crowd = [(f"person_{i}", 0.05 + i * 0.08, 0.9, 0.1, None) for i in range(10)]
        crowd += [("person_a", 0.40, 0.20, 0.1, None), ("person_b", 0.42, 0.21, 0.1, None)]
        tracker = PersistentRelationshipTracker()
        assert len(self.run(tracker, crowd)) < 66

    def test_the_gate_is_configurable(self) -> None:
        """A camera down a long corridor needs a tighter one than a doorway."""
        apart = [("person_1", 0.20, 0.50, 0.1, None), ("person_2", 0.60, 0.50, 0.1, None)]
        assert self.run(PersistentRelationshipTracker(proximity_gate=0.1), apart) == []
        assert len(self.run(PersistentRelationshipTracker(proximity_gate=0.5), apart)) == 1

    def test_an_impossible_gate_is_refused(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="proximity_gate"):
            PersistentRelationshipTracker(proximity_gate=0.0)

    def test_the_work_one_frame_can_ask_for_is_bounded(self) -> None:
        """A packed concourse must not become the frame budget."""
        packed = [(f"person_{i}", 0.5, 0.5, 0.1, None) for i in range(60)]
        tracker = PersistentRelationshipTracker(max_candidate_pairs=50)
        assert len(self.run(tracker, packed, frames=1)) <= 50
