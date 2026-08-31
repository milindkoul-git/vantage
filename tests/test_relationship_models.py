"""Unit tests for Phase 18 relationship models and evidence contracts."""

from __future__ import annotations

import pytest

from vantage.relationship.models import (
    DerivedRelationshipPattern,
    EntityRelationship,
    ProximityBasis,
    RelationshipScoreBreakdown,
    RelationshipSignal,
    RelationshipSignalType,
)


def test_relationship_models_and_immutability() -> None:
    signal = RelationshipSignal(
        signal_type=RelationshipSignalType.RECURRENT_PROXIMITY,
        timestamp=100.5,
        camera_id="cam_01",
        zone_id="lobby",
        strength=0.85,
        duration_s=12.5,
        proximity_basis=ProximityBasis.NORMALIZED_IMAGE_SPACE,
        distance_metric=0.08,
        evidence={"separation_norm": 0.08},
    )

    assert signal.signal_type == RelationshipSignalType.RECURRENT_PROXIMITY
    assert signal.proximity_basis == ProximityBasis.NORMALIZED_IMAGE_SPACE

    d = signal.to_dict()
    assert d["signal_type"] == "recurrent_proximity"
    assert d["proximity_basis"] == "normalized_image_space"


def test_entity_relationship_canonical_ordering() -> None:
    breakdown = RelationshipScoreBreakdown(
        co_occurrence_contribution=0.10,
        proximity_contribution=0.20,
        following_contribution=0.0,
        duration_contribution=0.05,
        total_raw_score=0.35,
        active_decayed_score=0.35,
        decay_factor=1.0,
    )

    # Instantiate with inverted order (person_b, person_a)
    rel = EntityRelationship(
        entity_a="person_b",
        entity_b="person_a",
        active_strength=0.35,
        historical_score=0.35,
        score_breakdown=breakdown,
        primary_derived_pattern=DerivedRelationshipPattern.FREQUENT_CO_TRAVELER,
        first_observed=10.0,
        last_observed=20.0,
        co_occurrence_count=5,
        proximity_count=2,
        following_count=0,
        total_interaction_duration_s=15.0,
    )

    # Enforces canonical sorted undirected pair
    assert rel.entity_a == "person_a"
    assert rel.entity_b == "person_b"
    assert rel.pair_key == ("person_a", "person_b")
    assert rel.other_entity("person_a") == "person_b"
    assert rel.other_entity("person_b") == "person_a"

    with pytest.raises(ValueError):
        rel.other_entity("person_c")
