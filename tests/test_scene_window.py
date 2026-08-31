"""Unit Tests for Temporal Observation Windows (EntityTemporalWindow & SceneTemporalWindow)."""

from __future__ import annotations

from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    LEFT_HIP,
    RIGHT_HIP,
    Keypoint,
    Pose,
    Posture,
)
from vantage.scene.window import (
    ConvergenceDynamics,
    EntityTemporalWindow,
    KinematicFeatures,
    SceneTemporalWindow,
    SkeletalDynamics,
)


def test_entity_temporal_window_kinematics() -> None:
    win = EntityTemporalWindow(max_samples=60, max_span_s=5.0)

    # Simulate multi-directional erratic pacing trajectory
    for i in range(20):
        t = 100.0 + i * 0.2  # 4 seconds span
        # Multi-directional bearings: 0, 90, 180, 270 deg
        bearing = float((i * 90) % 360)
        speed = 0.5 + (0.3 if i % 2 == 0 else -0.3)
        # Shift foot point slightly in bounded area
        x_shift = 100.0 + (i % 4) * 10.0
        box = BoundingBox(x_shift, 100.0, x_shift + 50.0, 250.0)
        win.add(
            timestamp=t,
            box=box,
            frame_width=1920,
            frame_height=1080,
            speed=speed,
            bearing_deg=bearing,
            posture=Posture.STANDING,
        )

    kin = win.extract_kinematics()
    assert isinstance(kin, KinematicFeatures)
    assert kin.sample_count == 20
    assert kin.duration_s > 3.5
    assert kin.mean_speed > 0.3
    assert kin.directional_entropy > 0.4  # High directional changes
    assert kin.pacing_ratio < 0.6  # Trapped / pacing back and forth


def test_entity_temporal_window_skeletal_collapse() -> None:
    win = EntityTemporalWindow(max_samples=30, max_span_s=3.0)

    # Frame 1: Standing pose
    box1 = BoundingBox(100.0, 100.0, 150.0, 300.0)
    kpts1 = [Keypoint(125.0, 120.0 + idx * 10, 0.9) for idx in range(17)]
    pose1 = Pose(keypoints=tuple(kpts1), track_id=1, entity_id="person_1", box=box1)
    win.add(
        timestamp=10.0,
        box=box1,
        frame_width=1920,
        frame_height=1080,
        speed=0.1,
        bearing_deg=0.0,
        posture=Posture.STANDING,
        pose=pose1,
    )

    # Frame 2: Rapid downward vertical hip drop (Falling)
    box2 = BoundingBox(100.0, 200.0, 150.0, 350.0)
    kpts2 = list(kpts1)
    kpts2[LEFT_HIP] = Keypoint(125.0, 280.0, 0.9)
    kpts2[RIGHT_HIP] = Keypoint(130.0, 280.0, 0.9)
    pose2 = Pose(keypoints=tuple(kpts2), track_id=1, entity_id="person_1", box=box2)
    win.add(
        timestamp=10.5,
        box=box2,
        frame_width=1920,
        frame_height=1080,
        speed=0.8,
        bearing_deg=0.0,
        posture=Posture.LYING,
        pose=pose2,
    )

    skel = win.extract_skeletal()
    assert isinstance(skel, SkeletalDynamics)
    assert skel.hip_drop_rate > 0.40  # Fast descent
    assert skel.is_prone is True


def test_scene_temporal_window_convergence_and_dispersion() -> None:
    win = SceneTemporalWindow(max_samples=20, max_span_s=2.0)

    # 1. Simulate 4 entities converging toward centroid (0.5, 0.5)
    for i in range(10):
        t = 50.0 + i * 0.1
        factor = 1.0 - (i * 0.08)  # Shrinking spread
        entities = [
            ("p1", 0.5 - 0.4 * factor, 0.5 - 0.4 * factor),
            ("p2", 0.5 + 0.4 * factor, 0.5 - 0.4 * factor),
            ("p3", 0.5 - 0.4 * factor, 0.5 + 0.4 * factor),
            ("p4", 0.5 + 0.4 * factor, 0.5 + 0.4 * factor),
        ]
        win.add(timestamp=t, camera_id="cam_01", entities=entities)

    conv = win.extract_convergence()
    assert isinstance(conv, ConvergenceDynamics)
    assert conv.entity_count == 4
    assert conv.spread_rate < -0.05  # Negative spread rate = converging
    assert conv.is_converging is True
    assert conv.is_dispersing is False
