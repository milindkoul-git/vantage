"""Adversarial & Stress Robustness Test Suite for Phase 18 Relationship Intelligence."""

from __future__ import annotations

import concurrent.futures

from vantage.events.contracts import EventCandidate
from vantage.events.engine import EventEngine
from vantage.relationship.config import RelationshipScoringConfig
from vantage.relationship.following import FollowingPatternDetector
from vantage.relationship.scoring import RelationshipScorer
from vantage.relationship.tracker import PersistentRelationshipTracker
from vantage.scene.graph import SceneGraphSnapshot, TransientInteractionEdge


# 1. False Relationship Resistance in Crowded Scenes
def test_false_relationships_from_crowded_scenes() -> None:
    tracker = PersistentRelationshipTracker(max_relationships=1000)

    # 30 entities in frame, but only 2 pairs have interaction edges
    entities = [(f"person_{k}", 0.1 * (k % 10), 0.1 * (k // 10), 0.2, 0.0) for k in range(30)]
    edge1 = TransientInteractionEdge("person_1", "person_2", "near", 0.05, 0.9, "near")
    edge2 = TransientInteractionEdge("person_5", "person_6", "near", 0.05, 0.9, "near")

    snap = SceneGraphSnapshot(
        camera_id="cam_crowd",
        timestamp=100.0,
        entity_count=30,
        active_edges=(edge1, edge2),
        collective_behaviors=(),
        unattended_objects=(),
    )

    # Process single frame
    tracker.process_frame("cam_crowd", entities, snap, None, now=100.0)

    # Only gated pairs should be tracked, preventing 30*29/2 = 435 pair explosion
    all_rels = tracker.get_all_relationships()
    assert len(all_rels) <= 5


# 2. Trajectory Crossing vs Following Pattern
def test_following_vs_side_by_side_walking() -> None:
    detector = FollowingPatternDetector()

    # Two people walking side-by-side simultaneously (lag = 0.0s)
    traj_a = [(10.0 + i * 0.5, 0.1 + (i / 20.0) * 0.7, 0.5, 90.0) for i in range(20)]
    traj_b = [(10.0 + i * 0.5, 0.1 + (i / 20.0) * 0.7, 0.55, 90.0) for i in range(20)]

    is_fol, _sig = detector.evaluate_trajectories(
        entity_a="person_a",
        traj_a=traj_a,
        entity_b="person_b",
        traj_b=traj_b,
        camera_id="cam_01",
        now=20.0,
    )

    # Side-by-side walking (lag < 0.5s) is NOT following
    assert is_fol is False


# 3. Time-Decay Resilience over Long Horizons
def test_exponential_decay_over_long_horizon() -> None:
    scorer = RelationshipScorer(RelationshipScoringConfig(half_life_s=3600.0))

    # Strong historical relationship
    b_now, _, _ = scorer.evaluate(
        co_occurrence_count=20,
        proximity_count=10,
        following_count=4,
        total_duration_s=300.0,
        last_observed=1000.0,
        now=1000.0,
    )
    raw_hist = b_now.total_raw_score
    assert raw_hist > 0.80

    # 24 hours later (86400s)
    b_24h, _, _ = scorer.evaluate(
        co_occurrence_count=20,
        proximity_count=10,
        following_count=4,
        total_duration_s=300.0,
        last_observed=1000.0,
        now=1000.0 + 86400.0,
    )

    # Historical score is intact, active decayed score is practically 0
    assert b_24h.total_raw_score == raw_hist
    assert b_24h.active_decayed_score < 0.01


# 4. Multi-Camera Concurrency Stress
def test_multi_camera_concurrent_updates() -> None:
    tracker = PersistentRelationshipTracker()

    def worker_thread(cam_idx: int) -> None:
        cam_id = f"cam_{cam_idx}"
        for step in range(50):
            t = 100.0 + cam_idx * 10 + step * 0.1
            p_a = f"global_person_{cam_idx}"
            p_b = f"global_person_{cam_idx + 1}"
            edge = TransientInteractionEdge(p_a, p_b, "near", 0.05, 0.9, "near")
            snap = SceneGraphSnapshot(cam_id, t, 2, (edge,), (), ())
            entities = [(p_a, 0.4, 0.4, 0.5, 0.0), (p_b, 0.42, 0.42, 0.5, 0.0)]

            tracker.process_frame(cam_id, entities, snap, None, now=t)
            _ = tracker.get_all_relationships(now=t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker_thread, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    rels = tracker.get_all_relationships()
    assert len(rels) >= 8


# 5. Cooldown Suppression for Recurrent Relationships
def test_event_engine_relationship_cooldown() -> None:
    engine = EventEngine()

    cand1 = EventCandidate(
        rule="following_pattern",
        severity="notice",
        summary="Following pattern observed between p1 and p2",
        entity_id="p2",
        camera_id="cam_01",
        wall_time=100.0,
    )

    ev1 = engine.evaluate_candidate(cand1)
    assert ev1 is not None

    # Next 20 frames within cooldown (45s) must be suppressed
    for i in range(1, 20):
        c_i = EventCandidate(
            rule="following_pattern",
            severity="notice",
            summary="Following pattern observed between p1 and p2",
            entity_id="p2",
            camera_id="cam_01",
            wall_time=100.0 + i * 1.0,
        )
        assert engine.evaluate_candidate(c_i) is None

    # After cooldown expires at t=150.0s
    cand_after = EventCandidate(
        rule="following_pattern",
        severity="notice",
        summary="Following pattern observed between p1 and p2",
        entity_id="p2",
        camera_id="cam_01",
        wall_time=150.0,
    )
    assert engine.evaluate_candidate(cand_after) is not None
