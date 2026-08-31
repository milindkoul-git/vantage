"""Adversarial & Stress Robustness Test Suite for Phase 17 Scene Intelligence."""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from vantage.activity.contracts import Activity
from vantage.activity.learned import FeatureBasedTemporalRecognizer
from vantage.entity.manager import EntityContextManager
from vantage.events.contracts import EventCandidate
from vantage.events.engine import EventEngine
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    Keypoint,
    Pose,
    Posture,
)
from vantage.scene.graph import TransientSceneGraph
from vantage.scene.window import EntityTemporalWindow
from vantage.search.semantic import IncidentSearch
from vantage.state.contracts import EntityState, MotionState
from vantage.storage.sqlite_store import SqliteStore
from vantage.tracking.contracts import Track, TrackState


# 1. Variable FPS & Temporal Gaps
def test_variable_fps_and_temporal_gaps() -> None:
    win = EntityTemporalWindow(max_samples=60, max_span_s=5.0)

    # High FPS jitter (60 FPS, dt = 0.016s)
    box = BoundingBox(100.0, 100.0, 150.0, 250.0)
    for i in range(30):
        t = 10.0 + i * 0.016
        speed = 0.5 + (0.05 if i % 2 == 0 else -0.05)
        win.add(
            timestamp=t,
            box=box,
            frame_width=1920,
            frame_height=1080,
            speed=speed,
            bearing_deg=0.0,
            posture=Posture.STANDING,
        )

    kin1 = win.extract_kinematics()
    # Ensure dt clamping prevents artificial 1000 h/s^2 spikes
    assert kin1.max_acceleration < 10.0

    # Introduce sudden 5.0-second temporal gap (camera blackout)
    t_resume = 15.0
    win.add(
        timestamp=t_resume,
        box=box,
        frame_width=1920,
        frame_height=1080,
        speed=2.0,
        bearing_deg=0.0,
        posture=Posture.STANDING,
    )
    kin2 = win.extract_kinematics()
    # Gap is gracefully handled and acceleration over the 5s gap is not treated as a single instantaneous spike
    assert kin2.max_acceleration < 10.0


# 2. Missing & Unstable Pose Landmarks
def test_missing_and_unstable_pose_landmarks() -> None:
    win = EntityTemporalWindow(max_samples=30, max_span_s=3.0)
    box = BoundingBox(100.0, 100.0, 150.0, 300.0)

    # Frame 1: Low confidence / missing keypoints (all confidence = 0.0)
    kpts_bad = [Keypoint(0.0, 0.0, 0.0) for _ in range(17)]
    pose_bad = Pose(keypoints=tuple(kpts_bad), track_id=1, entity_id="p1", box=box)
    win.add(
        timestamp=10.0,
        box=box,
        frame_width=1920,
        frame_height=1080,
        speed=0.0,
        bearing_deg=None,
        posture=Posture.UNKNOWN,
        pose=pose_bad,
    )

    # Frame 2: Another missing pose frame
    win.add(
        timestamp=10.5,
        box=box,
        frame_width=1920,
        frame_height=1080,
        speed=0.0,
        bearing_deg=None,
        posture=Posture.UNKNOWN,
        pose=pose_bad,
    )

    skel = win.extract_skeletal()
    # Hips missing -> hip drop rate must be exactly 0.0, NOT NaN or artificial drop
    assert skel.hip_drop_rate == 0.0
    assert skel.wrist_velocity == 0.0
    assert skel.is_prone is False


# 3. False-Positive Behavior Scenarios
def test_false_positive_behavior_resistance() -> None:
    rec = FeatureBasedTemporalRecognizer()

    # FP Case A: Controlled quick sitting down (posture becomes SITTING, not LYING)
    box_sitting = BoundingBox(100.0, 100.0, 150.0, 220.0)
    kpts_sit = [Keypoint(125.0, 120.0 + idx * 6, 0.9) for idx in range(17)]
    pose_sit = Pose(
        keypoints=tuple(kpts_sit),
        track_id=1,
        entity_id="p1",
        box=box_sitting,
        posture=Posture.SITTING,
    )

    st_sit = EntityState(
        track_id=1,
        entity_id="p1",
        label="person",
        motion=MotionState.STATIONARY,
        speed=0.2,
        dwell_s=1.0,
        bearing_deg=None,
        distance=0.1,
        age_s=2.0,
        observed=True,
        posture="sitting",
    )
    act_sit = rec.observe(st_sit, pose_sit, now=10.5)
    acts = {o.activity for o in act_sit.observations}
    assert Activity.SUDDEN_COLLAPSE not in acts
    assert Activity.FALLING not in acts

    # FP Case B: Stationary bounding box noise / jitter (tiny path length)
    for i in range(10):
        st_jitter = EntityState(
            track_id=2,
            entity_id="p2",
            label="person",
            motion=MotionState.STATIONARY,
            speed=0.02,
            dwell_s=1.0 + i * 0.1,
            bearing_deg=float((i * 180) % 360),
            distance=0.01,
            age_s=1.0 + i * 0.1,
            observed=True,
            posture="standing",
        )
        act_jitter = rec.observe(st_jitter, None, now=20.0 + i * 0.1)

    acts_jitter = {o.activity for o in act_jitter.observations}
    assert Activity.ERRATIC_PACING not in acts_jitter


# 4. Ownership Confidence & Source Attribution
def test_ownership_confidence_and_source() -> None:
    sg = TransientSceneGraph(camera_id="cam_01", unattended_dwell_s=5.0)
    box = BoundingBox(100.0, 100.0, 140.0, 140.0)

    # Register ownership with explicit confidence and source
    sg.register_ownership(
        object_id="luggage_99",
        label="suitcase",
        box=box,
        owner_id="person_42",
        now=100.0,
        confidence=0.95,
        source="hoi_wrist_proximity",
    )

    # 1. Owner leaves scene at t=101 (timer begins)
    _, _ = sg.update(tracks=[], raw_detections=None, now=101.0)

    # 2. Dwell threshold (5s) exceeded at t=107
    _, cands = sg.update(tracks=[], raw_detections=None, now=107.0)
    assert len(cands) == 1
    cand = cands[0]
    assert cand.rule == "unattended_object_dwell"
    assert cand.evidence["ownership_source"] == "hoi_wrist_proximity"
    assert cand.evidence["ownership_confidence"] == 0.95


# 5. Third-Party Interference on Unattended Objects
def test_third_party_interaction_prevents_unattended_alert() -> None:
    sg = TransientSceneGraph(camera_id="cam_01", unattended_dwell_s=5.0)
    box = BoundingBox(500.0, 500.0, 540.0, 540.0)

    sg.register_ownership(
        object_id="bag_1", label="backpack", box=box, owner_id="person_1", now=100.0
    )

    # Owner person_1 is far away (at x=1800), but person_2 is right next to the bag (at x=510, y=520)
    tracks = [
        Track(
            1,
            "person_1",
            BoundingBox(1800.0, 100.0, 1850.0, 200.0),
            "person",
            0,
            0.9,
            TrackState.CONFIRMED,
            1,
            1,
            0,
            1,
            1,
        ),
        Track(
            2,
            "person_2",
            BoundingBox(510.0, 480.0, 560.0, 580.0),
            "person",
            0,
            0.9,
            TrackState.CONFIRMED,
            1,
            1,
            0,
            1,
            1,
        ),
    ]

    _, cands = sg.update(tracks=tracks, raw_detections=None, now=110.0)
    # Third party presence prevents unattended alert
    assert len(cands) == 0


# 6. Perspective-Adaptive Proximity
def test_perspective_adaptive_proximity() -> None:
    sg = TransientSceneGraph(camera_id="cam_01", adaptive_perspective=True)

    # Two large entities in the foreground (height = 400px in 1080p -> norm_h = 0.37)
    # Distance = 200px (norm_dist = 0.10)
    t_near = [
        Track(
            1,
            "p1",
            BoundingBox(100.0, 600.0, 250.0, 1000.0),
            "person",
            0,
            0.9,
            TrackState.CONFIRMED,
            1,
            1,
            0,
            1,
            1,
        ),
        Track(
            2,
            "p2",
            BoundingBox(300.0, 600.0, 450.0, 1000.0),
            "person",
            0,
            0.9,
            TrackState.CONFIRMED,
            1,
            1,
            0,
            1,
            1,
        ),
    ]
    snap_near, _ = sg.update(tracks=t_near, raw_detections=None, now=1.0)
    assert len(snap_near.active_edges) == 1


# 7. Scene Graph Auto-Cleanup & Expiration
def test_scene_graph_cleanup_and_expiration() -> None:
    sg = TransientSceneGraph(camera_id="cam_01")
    box = BoundingBox(100.0, 100.0, 150.0, 150.0)

    # Register object at t=100
    sg.register_ownership(
        object_id="old_bag", label="backpack", box=box, owner_id="p1", now=100.0
    )
    assert "old_bag" in sg._tracked_objects

    # Update at t=500 (>300s later)
    sg.update(tracks=[], raw_detections=None, now=500.0)
    assert "old_bag" not in sg._tracked_objects


# 8. Persistent Condition Cooldown Suppression
def test_persistent_condition_cooldown_suppression() -> None:
    engine = EventEngine()

    cand = EventCandidate(
        rule="sudden_collapse",
        severity="alert",
        summary="PERSON_1 sudden collapse",
        entity_id="person_1",
        camera_id="cam_01",
        wall_time=100.0,
    )

    # First event fires
    ev1 = engine.evaluate_candidate(cand)
    assert ev1 is not None

    # Next 100 frames over 10 seconds: all suppressed by cooldown (20s)
    for i in range(1, 100):
        t = 100.0 + i * 0.1
        c_i = EventCandidate(
            rule="sudden_collapse",
            severity="alert",
            summary="PERSON_1 sudden collapse",
            entity_id="person_1",
            camera_id="cam_01",
            wall_time=t,
        )
        assert engine.evaluate_candidate(c_i) is None

    # After cooldown expires at t=125.0s, candidate can fire again
    c_after = EventCandidate(
        rule="sudden_collapse",
        severity="alert",
        summary="PERSON_1 sudden collapse",
        entity_id="person_1",
        camera_id="cam_01",
        wall_time=125.0,
    )
    ev2 = engine.evaluate_candidate(c_after)
    assert ev2 is not None


# 9. Search Ranking Edge Cases & Deterministic Tie-Breaking
def test_search_ranking_edge_cases(tmp_path: pytest.TempPathFactory) -> None:
    db_file = tmp_path / "test_search.db"
    store = SqliteStore(str(db_file))

    # Seed events with identical scores
    t_now = time.time()
    store.write_events(
        [
            {
                "timestamp": t_now - 10,
                "camera_id": "cam_01",
                "rule": "tailgating",
                "severity": "notice",
                "summary": "Tailgating incident A",
                "entity_id": "p1",
                "identity": "P1",
                "related_id": None,
                "zone": "CAM_01",
                "frame_index": 100,
                "elapsed_s": 10.0,
                "evidence": {},
            },
            {
                "timestamp": t_now - 5,
                "camera_id": "cam_01",
                "rule": "tailgating",
                "severity": "notice",
                "summary": "Tailgating incident B",
                "entity_id": "p2",
                "identity": "P2",
                "related_id": None,
                "zone": "CAM_01",
                "frame_index": 150,
                "elapsed_s": 15.0,
                "evidence": {},
            },
        ]
    )

    search = IncidentSearch(store)

    # A: Empty / whitespace query
    res_empty = search.search("   ")
    assert res_empty["total"] == 0

    # B: Punctuation only
    res_punct = search.search("??? !!!")
    assert res_punct["total"] == 0

    # C: Out of vocabulary query
    res_oov = search.search("extraterrestrial spacecraft landing")
    assert res_oov["total"] == 0

    # D: Query matching both records with tie-breaking on timestamp DESC
    res_tie = search.search("tailgating")
    assert res_tie["total"] == 2
    # ID 2 is newer (t_now - 5) than ID 1 (t_now - 10)
    assert res_tie["results"][0]["id"] == 2
    assert res_tie["results"][1]["id"] == 1

    store.close()


# 10. Concurrency Stress Around Entity Context Updates
def test_concurrency_stress_entity_context_updates() -> None:
    em = EntityContextManager()
    box = BoundingBox(10.0, 10.0, 50.0, 100.0)

    # Spawn 8 threads hammering the same entity
    def worker_fn(thread_id: int) -> None:
        for i in range(50):
            t = 100.0 + thread_id * 10 + i * 0.1
            ctx = em.get_or_create(
                global_id="concurrent_person_1",
                label="person",
                camera_id=f"cam_0{thread_id % 4 + 1}",
                track_id=100 + thread_id,
                box=box,
                wall_time=t,
            )
            ctx.update_spatial(
                camera_id=f"cam_0{thread_id % 4 + 1}",
                box=box,
                foot_point=(0.5, 0.9),
                wall_time=t,
            )
            ctx.update_kinematics(speed_h_s=0.5, motion_state="walking", posture="standing")
            ctx.update_behavior(["walking"], primary="walking", confidence=0.9, evidence="test")
            ctx.add_event({"id": f"ev_{thread_id}_{i}", "timestamp": t, "rule": "test_rule"})
            snap = ctx.to_snapshot()
            assert snap.global_id == "concurrent_person_1"
            assert snap.behavior is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker_fn, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Verify no exceptions raised

    snap_final = em.get_snapshot("concurrent_person_1")
    assert snap_final is not None
    assert snap_final.global_id == "concurrent_person_1"
