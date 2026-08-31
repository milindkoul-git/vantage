"""Unit Tests for Deterministic Spatio-Temporal Behavior Recognition."""

from __future__ import annotations

import pytest

from vantage.activity.contracts import Activity, EntityActivity
from vantage.activity.learned import (
    FeatureBasedTemporalRecognizer,
    OptionalModelTemporalRecognizer,
    TemporalBehaviorRecognizer,
)
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    LEFT_HIP,
    RIGHT_HIP,
    Keypoint,
    Pose,
    Posture,
)
from vantage.state.contracts import EntityState, MotionState


def test_feature_based_temporal_recognizer_sudden_collapse() -> None:
    rec = FeatureBasedTemporalRecognizer(transient_hold_s=2.0)
    box = BoundingBox(10.0, 10.0, 50.0, 150.0)

    # Frame 1: Standing
    kpts1 = [Keypoint(30.0, 20.0 + idx * 8, 0.9) for idx in range(17)]
    pose1 = Pose(keypoints=tuple(kpts1), track_id=1, entity_id="person_1", box=box)
    st1 = EntityState(
        track_id=1,
        entity_id="person_1",
        label="person",
        motion=MotionState.STATIONARY,
        speed=0.0,
        dwell_s=1.0,
        bearing_deg=None,
        distance=0.0,
        age_s=1.0,
        observed=True,
        posture="standing",
    )
    rec.observe(st1, pose1, now=10.0)

    # Frame 2: Sudden drop to prone
    kpts2 = list(kpts1)
    kpts2[LEFT_HIP] = Keypoint(30.0, 140.0, 0.9)
    kpts2[RIGHT_HIP] = Keypoint(35.0, 140.0, 0.9)
    pose2 = Pose(
        keypoints=tuple(kpts2), track_id=1, entity_id="person_1", box=box, posture=Posture.LYING
    )
    st2 = EntityState(
        track_id=1,
        entity_id="person_1",
        label="person",
        motion=MotionState.MOVING,
        speed=0.6,
        dwell_s=1.5,
        bearing_deg=0.0,
        distance=0.5,
        age_s=1.5,
        observed=True,
        posture="lying",
    )
    act = rec.observe(st2, pose2, now=10.5)

    assert isinstance(act, EntityActivity)
    act_names = {o.activity for o in act.observations}
    assert Activity.SUDDEN_COLLAPSE in act_names
    assert Activity.FALLING in act_names


def test_feature_based_temporal_recognizer_erratic_pacing() -> None:
    rec = FeatureBasedTemporalRecognizer(pacing_window_s=3.0)

    # Alternate multi-directional pacing continuously in place
    for i in range(15):
        t = 100.0 + i * 0.3
        bearing = float((i * 90) % 360)
        st = EntityState(
            track_id=2,
            entity_id="person_2",
            label="person",
            motion=MotionState.MOVING,
            speed=0.5,
            dwell_s=t - 100.0,
            bearing_deg=bearing,
            distance=float(i * 0.2),
            age_s=t - 100.0,
            observed=True,
            posture="standing",
        )
        act = rec.observe(st, None, now=t)

    act_names = {o.activity for o in act.observations}
    assert Activity.ERRATIC_PACING in act_names


def test_temporal_behavior_recognizer_wrapper_and_model_seam() -> None:
    wrapper = TemporalBehaviorRecognizer()
    st = EntityState(
        track_id=3,
        entity_id="person_3",
        label="person",
        motion=MotionState.MOVING,
        speed=1.5,
        dwell_s=2.0,
        bearing_deg=90.0,
        distance=3.0,
        age_s=2.0,
        observed=True,
        posture="standing",
    )
    act = wrapper.observe(st, None, now=50.0)
    assert isinstance(act, EntityActivity)
    assert Activity.RUNNING in {o.activity for o in act.observations}

    # Model seam raises NotImplementedError when invoked without model
    with pytest.raises(NotImplementedError):
        OptionalModelTemporalRecognizer()
