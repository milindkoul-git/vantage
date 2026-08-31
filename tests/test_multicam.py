"""Tests for Multi-Camera Pipeline, Cross-Camera Re-ID and Facility Journeys."""

from __future__ import annotations

import numpy as np
import pytest

from vantage.core.frame import Frame
from vantage.multicam.journey import FacilityJourneyTracker
from vantage.multicam.reid import CrossCameraReIDTracker, VisualDescriptorExtractor
from vantage.perception.contracts import BoundingBox
from vantage.tracking.contracts import Track, TrackState


def _make_track(
    track_id: int, x1: float, y1: float, x2: float, y2: float, label: str = "person"
) -> Track:
    return Track(
        track_id=track_id,
        entity_id=f"{label}_{track_id}",
        box=BoundingBox(x1, y1, x2, y2),
        label=label,
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=10,
        hits=10,
        time_since_update=0,
        start_frame=0,
        last_frame=10,
    )


def test_visual_descriptor_extraction() -> None:
    extractor = VisualDescriptorExtractor()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Paint top red (torso) and bottom blue (legs)
    img[100:200, 100:200] = [0, 0, 255]
    img[200:300, 100:200] = [255, 0, 0]

    box = BoundingBox(100.0, 100.0, 200.0, 300.0)
    emb = extractor.extract(img, box)
    assert len(emb.vector) == 128

    # Similarity to self is 1.0
    sim = emb.cosine_similarity(emb)
    assert pytest.approx(sim, abs=1e-3) == 1.0


def test_simultaneous_different_people_get_distinct_ids() -> None:
    """People simultaneously present in different cameras MUST receive distinct global IDs."""
    tracker = CrossCameraReIDTracker(
        min_reid_similarity=0.80, allow_overlapping=False, min_transit_time_s=2.0
    )

    # Frame on Camera 1 at t=100.0
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img1[100:300, 100:200] = [0, 120, 255]
    frame1 = Frame(
        image=img1, index=1, source_id="cam_lobby", capture_monotonic=1.0, capture_wall=100.0
    )
    track1 = _make_track(1, 100.0, 100.0, 200.0, 300.0)
    mapping1 = tracker.update_camera("cam_lobby", frame1, [track1])
    gid1 = mapping1[1]

    # Simultaneous frame on Camera 2 at t=100.1 (Person active on Cam 1 cannot be on Cam 2!)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2[120:320, 150:250] = [0, 120, 255]
    frame2 = Frame(
        image=img2, index=1, source_id="cam_corridor", capture_monotonic=1.1, capture_wall=100.1
    )
    track2 = _make_track(1, 150.0, 120.0, 250.0, 320.0)
    mapping2 = tracker.update_camera("cam_corridor", frame2, [track2])
    gid2 = mapping2[1]

    # Co-presence constraint forces them to be distinct entities!
    assert gid1 != gid2


def test_cross_camera_reid_handover_after_transit() -> None:
    """Same person appearing on Camera 2 after leaving Camera 1 is correctly re-identified."""
    tracker = CrossCameraReIDTracker(
        min_reid_similarity=0.80, allow_overlapping=False, min_transit_time_s=2.0
    )

    # Person seen on Camera 1 at t=100.0
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img1[100:300, 100:200] = [0, 120, 255]  # Distinct orange clothing
    frame1 = Frame(
        image=img1, index=1, source_id="cam_lobby", capture_monotonic=1.0, capture_wall=100.0
    )
    track1 = _make_track(1, 100.0, 100.0, 200.0, 300.0)
    mapping1 = tracker.update_camera("cam_lobby", frame1, [track1])
    gid1 = mapping1[1]

    # Person leaves Camera 1 and appears on Camera 2 4.0 seconds later (t=104.0)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2[100:300, 100:200] = [0, 120, 255]  # Same clothing
    frame2 = Frame(
        image=img2, index=2, source_id="cam_corridor", capture_monotonic=5.0, capture_wall=104.0
    )
    track2 = _make_track(7, 100.0, 100.0, 200.0, 300.0)
    mapping2 = tracker.update_camera("cam_corridor", frame2, [track2])
    gid2 = mapping2[7]

    # Re-ID correctly matches after physical transit delay!
    assert gid1 == gid2


def test_facility_journey_timeline() -> None:
    journey_mgr = FacilityJourneyTracker()
    box = BoundingBox(10.0, 10.0, 50.0, 100.0)

    # Sighting on Cam 1 (Lobby)
    journey_mgr.record_sighting(
        global_id="global_person_1",
        label="person",
        camera_id="cam_lobby",
        wall_time=100.0,
        box=box,
        activity="walking",
        posture="standing",
    )
    # Sighting on Cam 2 (Corridor)
    journey_mgr.record_sighting(
        global_id="global_person_1",
        label="person",
        camera_id="cam_corridor",
        wall_time=115.0,
        box=box,
        activity="walking",
        posture="standing",
    )
    # Sighting on Cam 3 (Exit)
    journey_mgr.record_sighting(
        global_id="global_person_1",
        label="person",
        camera_id="cam_exit",
        wall_time=140.0,
        box=box,
        activity="walking",
        posture="standing",
    )

    journey = journey_mgr.get_journey("global_person_1")
    assert journey is not None
    assert len(journey.legs) == 3
    assert journey.cameras_traversed == ["cam_lobby", "cam_corridor", "cam_exit"]
    assert journey.total_duration_s == 40.0
