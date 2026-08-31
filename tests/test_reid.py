"""Tests for external AppearanceMatcher."""

from __future__ import annotations

import numpy as np

from vantage.perception.contracts import BoundingBox
from vantage.tracking.reid import (
    AppearanceMatcher,
    cosine_similarity,
    extract_appearance_descriptor,
)


def test_descriptor_and_similarity() -> None:
    # Build two solid color images: blue and red
    blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_img[:, :] = (255, 0, 0)  # BGR

    red_img = np.zeros((100, 100, 3), dtype=np.uint8)
    red_img[:, :] = (0, 0, 255)

    box = BoundingBox(10, 10, 90, 90)
    desc_blue = extract_appearance_descriptor(blue_img, box)
    desc_blue_2 = extract_appearance_descriptor(blue_img, box)
    desc_red = extract_appearance_descriptor(red_img, box)

    # Identical images have similarity ~1.0
    sim_same = cosine_similarity(desc_blue, desc_blue_2)
    assert sim_same > 0.99

    # Different colors have lower similarity
    sim_diff = cosine_similarity(desc_blue, desc_red)
    assert sim_diff < 0.20


def test_appearance_matcher_lifecycle() -> None:
    matcher = AppearanceMatcher(threshold=0.70, margin=0.10, max_lost_time_s=5.0)

    # 1. Person 1 is lost at t=10.0
    desc_p1 = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 60, dtype=np.float32)
    matcher.record_lost(
        "person_1", track_id=1, label="person", timestamp=10.0, descriptor=desc_p1
    )

    # 2. Candidate new track appears at t=12.0 with similar appearance
    desc_new = np.array([0.95, 0.05, 0.0, 0.0] + [0.0] * 60, dtype=np.float32)
    match = matcher.match_candidate(
        new_track_id=5, label="person", current_time=12.0, candidate_descriptor=desc_new
    )

    assert match is not None
    assert match.new_track_id == 5
    assert match.matched_entity_id == "person_1"
    assert match.similarity > 0.90

    # 3. Subsequent match returns None because person_1 is already claimed
    match_again = matcher.match_candidate(
        new_track_id=6, label="person", current_time=12.5, candidate_descriptor=desc_new
    )
    assert match_again is None


def test_appearance_matcher_expiry() -> None:
    matcher = AppearanceMatcher(threshold=0.70, max_lost_time_s=3.0)

    desc_p1 = np.ones(64, dtype=np.float32)
    matcher.record_lost(
        "person_1", track_id=1, label="person", timestamp=10.0, descriptor=desc_p1
    )

    # Attempt to match past expiry at t=15.0 (> 3.0s window)
    match = matcher.match_candidate(
        new_track_id=5, label="person", current_time=15.0, candidate_descriptor=desc_p1
    )
    assert match is None
