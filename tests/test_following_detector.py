"""Unit tests for FollowingPatternDetector using lagged trajectory alignment."""

from __future__ import annotations

from vantage.relationship.following import FollowingPatternDetector
from vantage.relationship.models import RelationshipSignalType


def test_following_detector_lagged_trajectory_success() -> None:
    detector = FollowingPatternDetector()

    # Create Leader Trajectory moving from (0.1, 0.1) to (0.8, 0.8) along bearing 45.0 deg
    traj_a: list[tuple[float, float, float, float | None]] = []
    for i in range(20):
        t = 10.0 + i * 0.5
        pos = 0.1 + (i / 20.0) * 0.7
        traj_a.append((t, pos, pos, 45.0))

    # Create Follower Trajectory lagging exactly 2.0 seconds behind
    traj_b: list[tuple[float, float, float, float | None]] = []
    for i in range(20):
        t = 12.0 + i * 0.5
        pos = 0.1 + (i / 20.0) * 0.7
        traj_b.append((t, pos, pos, 45.0))

    is_fol, sig = detector.evaluate_trajectories(
        entity_a="leader_1",
        traj_a=traj_a,
        entity_b="follower_2",
        traj_b=traj_b,
        camera_id="cam_01",
        now=22.0,
    )

    assert is_fol is True
    assert sig is not None
    assert sig.signal_type == RelationshipSignalType.LAGGED_TRAJECTORY_ALIGNMENT
    assert sig.evidence["follower_id"] == "follower_2"
    assert abs(sig.evidence["lag_s"] - 2.0) <= 0.25
    assert sig.evidence["mean_trajectory_error"] < 0.05


def test_following_detector_crossing_paths_rejected() -> None:
    detector = FollowingPatternDetector()

    # Leader moving left-to-right (bearing 90 deg)
    traj_a = [(10.0 + i * 0.5, 0.1 + (i / 20.0) * 0.8, 0.5, 90.0) for i in range(20)]

    # Person B moving top-to-bottom crossing at the center (bearing 180 deg)
    traj_b = [(10.0 + i * 0.5, 0.5, 0.1 + (i / 20.0) * 0.8, 180.0) for i in range(20)]

    is_fol, sig = detector.evaluate_trajectories(
        entity_a="person_a",
        traj_a=traj_a,
        entity_b="person_b",
        traj_b=traj_b,
        camera_id="cam_01",
        now=20.0,
    )

    assert is_fol is False
    assert sig is None
