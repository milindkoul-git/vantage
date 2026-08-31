"""Unit Tests for TransientSceneGraph, Group Dynamics, and Unattended Object Tracking."""

from __future__ import annotations

from vantage.perception.contracts import BoundingBox
from vantage.scene.graph import (
    SceneGraphSnapshot,
    TransientSceneGraph,
)
from vantage.tracking.contracts import Track, TrackState


def test_transient_scene_graph_group_convergence_and_edges() -> None:
    sg = TransientSceneGraph(camera_id="cam_01", proximity_threshold_norm=0.20)

    # 1. First frame: 3 entities approaching each other
    t1 = [
        Track(
            1,
            "person_1",
            BoundingBox(100.0, 100.0, 150.0, 250.0),
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
            BoundingBox(160.0, 100.0, 210.0, 250.0),
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
            3,
            "person_3",
            BoundingBox(130.0, 150.0, 180.0, 300.0),
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

    snap, _candidates = sg.update(
        tracks=t1, raw_detections=None, now=10.0, frame_width=1920, frame_height=1080
    )

    assert isinstance(snap, SceneGraphSnapshot)
    assert snap.entity_count == 3
    assert len(snap.active_edges) >= 1

    # Verify JSON serialization
    snap_dict = snap.to_dict()
    assert snap_dict["camera_id"] == "cam_01"
    assert snap_dict["entity_count"] == 3


def test_transient_scene_graph_unattended_object_lifecycle() -> None:
    sg = TransientSceneGraph(camera_id="cam_01", unattended_dwell_s=10.0)

    bag_box = BoundingBox(200.0, 400.0, 240.0, 460.0)

    # 1. Register ownership link: person_1 is initially holding/carrying backpack_1
    sg.register_ownership(
        object_id="backpack_1", label="backpack", box=bag_box, owner_id="person_1", now=100.0
    )

    # 2. Frame where owner is close: no alert
    owner_track_close = [
        Track(
            1,
            "person_1",
            BoundingBox(210.0, 380.0, 260.0, 500.0),
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
    _, cands1 = sg.update(tracks=owner_track_close, raw_detections=None, now=101.0)
    assert len(cands1) == 0

    # 3. Owner departs (far away at x=1800): unattended timer starts
    owner_track_far = [
        Track(
            1,
            "person_1",
            BoundingBox(1800.0, 380.0, 1850.0, 500.0),
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
    _, cands2 = sg.update(tracks=owner_track_far, raw_detections=None, now=105.0)
    assert len(cands2) == 0  # Only 4s elapsed, threshold is 10s

    # 4. Unattended threshold exceeded (12s later) -> alert candidate generated
    _, cands3 = sg.update(tracks=owner_track_far, raw_detections=None, now=116.0)
    assert len(cands3) == 1
    ev_cand = cands3[0]
    assert ev_cand.rule == "unattended_object_dwell"
    assert ev_cand.severity == "alert"
    assert ev_cand.entity_id == "person_1"
