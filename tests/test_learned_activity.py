"""Tests for the LearnedActionClassifier spatio-temporal activity recognizer."""

from __future__ import annotations

from vantage.activity.base import Recognizer
from vantage.activity.contracts import Activity
from vantage.activity.learned import LearnedActionClassifier
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Keypoint,
    Pose,
    Posture,
)
from vantage.state.contracts import EntityState, MotionState


def _make_pose(
    kpts_dict: dict[int, tuple[float, float]], posture: Posture = Posture.STANDING
) -> Pose:
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
        entity_id="E1",
        box=BoundingBox(0.0, 0.0, 100.0, 200.0),
        posture=posture,
        posture_confidence=0.9,
    )


def test_learned_classifier_implements_protocol() -> None:
    classifier = LearnedActionClassifier()
    assert isinstance(classifier, Recognizer)


def test_learned_classifier_idle_when_stationary() -> None:
    classifier = LearnedActionClassifier()
    state = EntityState(
        track_id=1,
        entity_id="E1",
        label="person",
        motion=MotionState.STATIONARY,
        speed=0.0,
        dwell_s=2.0,
        bearing_deg=None,
        distance=0.0,
        age_s=2.0,
        observed=True,
    )
    res = classifier.observe(state, None, now=2.0)
    assert len(res.observations) == 1
    assert res.observations[0].activity is Activity.IDLE


def test_learned_classifier_walking_and_running() -> None:
    classifier = LearnedActionClassifier()
    # Walk
    state_walk = EntityState(
        track_id=1,
        entity_id="E1",
        label="person",
        motion=MotionState.MOVING,
        speed=0.6,
        dwell_s=1.0,
        bearing_deg=90.0,
        distance=0.6,
        age_s=1.0,
        observed=True,
    )
    res = classifier.observe(state_walk, None, now=1.0)
    assert any(obs.activity is Activity.WALKING for obs in res.observations)

    # Run for a sustained window
    state_run = EntityState(
        track_id=2,
        entity_id="E2",
        label="person",
        motion=MotionState.MOVING,
        speed=1.8,
        dwell_s=2.0,
        bearing_deg=90.0,
        distance=2.4,
        age_s=2.0,
        observed=True,
    )
    res2 = classifier.observe(state_run, None, now=2.0)
    assert any(obs.activity is Activity.RUNNING for obs in res2.observations)


def test_learned_classifier_arm_raised() -> None:
    classifier = LearnedActionClassifier()
    state = EntityState(
        track_id=1,
        entity_id="E1",
        label="person",
        motion=MotionState.STATIONARY,
        speed=0.0,
        dwell_s=1.0,
        bearing_deg=None,
        distance=0.0,
        age_s=1.0,
        observed=True,
    )
    # Wrist (y=20) above nose (y=50)
    pose = _make_pose(
        {
            LEFT_SHOULDER: (40.0, 60.0),
            RIGHT_SHOULDER: (60.0, 60.0),
            NOSE: (50.0, 50.0),
            LEFT_WRIST: (40.0, 20.0),
            RIGHT_WRIST: (60.0, 80.0),
        }
    )
    res = classifier.observe(state, pose, now=1.0)
    assert any(obs.activity is Activity.ARM_RAISED for obs in res.observations)


def test_learned_classifier_fall_detection() -> None:
    classifier = LearnedActionClassifier()
    state = EntityState(
        track_id=1,
        entity_id="E1",
        label="person",
        motion=MotionState.MOVING,
        speed=1.2,
        dwell_s=0.5,
        bearing_deg=180.0,
        distance=0.5,
        age_s=0.5,
        observed=True,
    )
    # Frame 1: Standing high (hip at y=80)
    p1 = _make_pose({LEFT_HIP: (50.0, 80.0), RIGHT_HIP: (60.0, 80.0)}, Posture.STANDING)
    classifier.observe(state, p1, now=0.0)

    # Frame 2: Dropping (hip at y=120)
    p2 = _make_pose({LEFT_HIP: (50.0, 120.0), RIGHT_HIP: (60.0, 120.0)}, Posture.STANDING)
    classifier.observe(state, p2, now=0.1)

    # Frame 3: Lying down low (hip at y=180)
    p3 = _make_pose({LEFT_HIP: (50.0, 180.0), RIGHT_HIP: (60.0, 180.0)}, Posture.LYING)
    res = classifier.observe(state, p3, now=0.2)

    assert any(obs.activity is Activity.FALLING for obs in res.observations)


def test_learned_classifier_forget_and_reset() -> None:
    classifier = LearnedActionClassifier()
    state = EntityState(
        track_id=1,
        entity_id="E1",
        label="person",
        motion=MotionState.STATIONARY,
        speed=0.0,
        dwell_s=1.0,
        bearing_deg=None,
        distance=0.0,
        age_s=1.0,
        observed=True,
    )
    classifier.observe(state, None, now=1.0)
    assert 1 in classifier._history

    # Forget entity 1
    classifier.forget(set())
    assert 1 not in classifier._history

    classifier.observe(state, None, now=2.0)
    assert 1 in classifier._history
    classifier.reset()
    assert len(classifier._history) == 0
