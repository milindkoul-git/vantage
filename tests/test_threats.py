"""Tests for Advanced Threat Detection Engine."""

from __future__ import annotations

from vantage.events.threats import ThreatDetectionEngine
from vantage.state.contracts import EntityState, MotionState


def _make_state(
    entity_id: str, speed: float = 0.5, bearing_deg: float | None = 180.0
) -> EntityState:
    return EntityState(
        track_id=1,
        entity_id=entity_id,
        label="person",
        motion=MotionState.MOVING,
        speed=speed,
        dwell_s=5.0,
        bearing_deg=bearing_deg,
        distance=10.0,
        age_s=5.0,
        observed=True,
    )


def test_tailgating_detection() -> None:
    engine = ThreatDetectionEngine(tailgating_gap_s=1.8)

    st1 = _make_state("person_1", speed=0.4)
    st2 = _make_state("person_2", speed=0.4)

    # Person 1 crosses doorway at t=100.0
    threats1 = engine.evaluate_threats("cam_04_doorway", [st1], wall_time=100.0)
    assert len(threats1) == 0

    # Person 2 follows Person 1 through doorway 0.9s later at t=100.9 (Tailgating breach!)
    threats2 = engine.evaluate_threats("cam_04_doorway", [st2], wall_time=100.9)
    assert len(threats2) == 1
    assert threats2[0]["rule"] == "tailgating"
    assert threats2[0]["severity"] == "alert"
