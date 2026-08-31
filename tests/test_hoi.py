"""Tests for Human-Object Interaction (HOI) fusion engine."""

from __future__ import annotations

from vantage.activity.hoi import HOIFusionEngine
from vantage.perception.contracts import BoundingBox, Detection
from vantage.pose.contracts import (
    LEFT_SHOULDER,
    NOSE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Keypoint,
    Pose,
    Posture,
)
from vantage.tracking.contracts import Track, TrackState


def _make_person_track(x1: float, y1: float, x2: float, y2: float) -> Track:
    return Track(
        track_id=1,
        entity_id="person_1",
        box=BoundingBox(x1, y1, x2, y2),
        label="person",
        class_id=0,
        confidence=0.95,
        state=TrackState.CONFIRMED,
        age=10,
        hits=10,
        time_since_update=0,
        start_frame=0,
        last_frame=10,
        velocity=(0.0, 0.0),
        history=(),
    )


def _make_pose(kpts_dict: dict[int, tuple[float, float]]) -> Pose:
    kpts = []
    for i in range(17):
        if i in kpts_dict:
            x, y = kpts_dict[i]
            kpts.append(Keypoint(x=x, y=y, confidence=0.9))
        else:
            kpts.append(Keypoint(x=0.0, y=0.0, confidence=0.0))
    return Pose(
        keypoints=tuple(kpts),
        track_id=1,
        entity_id="person_1",
        box=BoundingBox(100.0, 100.0, 200.0, 400.0),
        posture=Posture.STANDING,
    )


def test_hoi_talking_on_phone() -> None:
    engine = HOIFusionEngine()
    track = _make_person_track(100.0, 100.0, 200.0, 400.0)
    # Head at y=120, wrist at y=130
    pose = _make_pose(
        {
            NOSE: (150.0, 120.0),
            LEFT_SHOULDER: (130.0, 160.0),
            RIGHT_SHOULDER: (170.0, 160.0),
            RIGHT_WRIST: (175.0, 130.0),
        }
    )
    phone_det = Detection(
        box=BoundingBox(165.0, 115.0, 185.0, 145.0),
        confidence=0.90,
        class_id=6,
        label="cell_phone",
    )
    interactions = engine.analyze(track.box, pose, [phone_det])
    assert len(interactions) == 1
    assert interactions[0].verb == "talking_on_phone"
    assert interactions[0].target_class == "cell_phone"


def test_hoi_carrying_backpack() -> None:
    engine = HOIFusionEngine()
    track = _make_person_track(100.0, 100.0, 200.0, 400.0)
    # Backpack overlapping torso
    backpack_det = Detection(
        box=BoundingBox(110.0, 150.0, 180.0, 250.0),
        confidence=0.88,
        class_id=8,
        label="backpack",
    )
    interactions = engine.analyze(track.box, None, [backpack_det])
    assert len(interactions) == 1
    assert interactions[0].verb == "carrying_baggage"


def test_hoi_riding_bicycle() -> None:
    engine = HOIFusionEngine()
    track = _make_person_track(100.0, 100.0, 200.0, 350.0)
    bike_det = Detection(
        box=BoundingBox(80.0, 250.0, 240.0, 400.0),
        confidence=0.92,
        class_id=1,
        label="bicycle",
    )
    interactions = engine.analyze(track.box, None, [bike_det])
    assert len(interactions) == 1
    assert interactions[0].verb == "riding_vehicle"


def test_hoi_holding_bottle() -> None:
    engine = HOIFusionEngine()
    track = _make_person_track(100.0, 100.0, 200.0, 400.0)
    pose = _make_pose(
        {
            RIGHT_WRIST: (180.0, 250.0),
        }
    )
    bottle_det = Detection(
        box=BoundingBox(175.0, 240.0, 195.0, 280.0),
        confidence=0.85,
        class_id=7,
        label="bottle",
    )
    interactions = engine.analyze(track.box, pose, [bottle_det])
    assert len(interactions) == 1
    assert interactions[0].verb == "holding_bottle"
