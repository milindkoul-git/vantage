"""Unit and Integration Tests for Unified Event Architecture (Candidate -> Policy -> Emission)."""

from __future__ import annotations

from vantage.events.contracts import Event, EventCandidate, Severity
from vantage.events.engine import EventEngine


def test_event_candidate_evaluation_and_cooldown() -> None:
    engine = EventEngine()
    now = 1000.0

    # 1. First candidate event fires successfully
    cand1 = EventCandidate(
        rule="tailgating",
        severity="notice",
        summary="Tailgating detected at Turnstile A",
        entity_id="global_person_1",
        camera_id="cam_01",
        wall_time=now,
        evidence={"gap_s": 0.4},
    )
    ev1 = engine.evaluate_candidate(cand1)
    assert ev1 is not None
    assert isinstance(ev1, Event)
    assert ev1.rule == "tailgating"
    assert ev1.severity is Severity.NOTICE
    assert ev1.entity_id == "global_person_1"
    assert engine.raised == 1
    assert engine.suppressed == 0

    # 2. Duplicate candidate within cooldown (25s) is suppressed
    cand2 = EventCandidate(
        rule="tailgating",
        severity="notice",
        summary="Tailgating detected again",
        entity_id="global_person_1",
        camera_id="cam_01",
        wall_time=now + 5.0,  # only 5 seconds later
    )
    ev2 = engine.evaluate_candidate(cand2)
    assert ev2 is None
    assert engine.raised == 1
    assert engine.suppressed == 1

    # 3. Different entity with same rule fires without suppression
    cand3 = EventCandidate(
        rule="tailgating",
        severity="notice",
        summary="Tailgating detected for person 2",
        entity_id="global_person_2",
        camera_id="cam_01",
        wall_time=now + 6.0,
    )
    ev3 = engine.evaluate_candidate(cand3)
    assert ev3 is not None
    assert ev3.entity_id == "global_person_2"
    assert engine.raised == 2

    # 4. Same entity after cooldown expires fires successfully
    cand4 = EventCandidate(
        rule="tailgating",
        severity="notice",
        summary="Tailgating detected after 30s",
        entity_id="global_person_1",
        camera_id="cam_01",
        wall_time=now + 30.0,
    )
    ev4 = engine.evaluate_candidate(cand4)
    assert ev4 is not None
    assert engine.raised == 3


def test_evaluate_candidates_batch() -> None:
    engine = EventEngine()
    now = 500.0

    candidates = [
        EventCandidate(
            rule="exclusion_breach",
            severity="alert",
            summary="Entered Server Room",
            entity_id="global_person_1",
            wall_time=now,
        ),
        EventCandidate(
            rule="exclusion_breach",
            severity="alert",
            summary="Entered Server Room Duplicate",
            entity_id="global_person_1",
            wall_time=now + 1.0,
        ),
        EventCandidate(
            rule="wrong_way_direction",
            severity="notice",
            summary="Wrong way in Corridor",
            entity_id="global_person_1",
            wall_time=now + 2.0,
        ),
    ]

    events = engine.evaluate_candidates(candidates)
    assert len(events) == 2  # 1 breach + 1 wrong way (duplicate breach suppressed)
    assert events[0].rule == "exclusion_breach"
    assert events[1].rule == "wrong_way_direction"
