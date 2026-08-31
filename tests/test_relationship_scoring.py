"""Unit tests for Phase 18 relationship scoring, explainability breakdowns, and exponential time decay."""

from __future__ import annotations

from vantage.relationship.config import RelationshipScoringConfig
from vantage.relationship.models import DerivedRelationshipPattern
from vantage.relationship.scoring import RelationshipScorer


def test_scoring_single_coincidence_resistance() -> None:
    scorer = RelationshipScorer()

    # 1. Single accidental co-occurrence: score must remain low (< 0.10)
    breakdown, pattern, summary = scorer.evaluate(
        co_occurrence_count=1,
        proximity_count=0,
        following_count=0,
        total_duration_s=0.0,
        last_observed=100.0,
        now=100.0,
    )

    assert breakdown.total_raw_score < 0.05
    assert breakdown.active_decayed_score < 0.05
    assert pattern is None
    assert "co-appeared 1x" in summary


def test_scoring_explainability_and_pattern_promotion() -> None:
    scorer = RelationshipScorer()

    # Multiple recurrent proximities, following alignments, and long duration
    breakdown, pattern, summary = scorer.evaluate(
        co_occurrence_count=10,
        proximity_count=6,
        following_count=3,
        total_duration_s=120.0,
        last_observed=100.0,
        now=100.0,
    )

    # Verify score attribution sums correctly
    expected_sum = (
        breakdown.co_occurrence_contribution
        + breakdown.proximity_contribution
        + breakdown.following_contribution
        + breakdown.duration_contribution
    )
    assert abs(breakdown.total_raw_score - expected_sum) < 1e-4
    assert breakdown.total_raw_score > 0.60
    assert pattern == DerivedRelationshipPattern.FOLLOWING_PATTERN_CANDIDATE
    assert "following alignment 3x" in summary


def test_exponential_recency_decay_preserves_historical_score() -> None:
    cfg = RelationshipScoringConfig(half_life_s=3600.0)  # 1 hour half-life
    scorer = RelationshipScorer(cfg)

    # Initial interaction at t=1000
    b_init, _, _ = scorer.evaluate(
        co_occurrence_count=8,
        proximity_count=4,
        following_count=0,
        total_duration_s=60.0,
        last_observed=1000.0,
        now=1000.0,
    )
    init_score = b_init.total_raw_score

    # Evaluate 1 hour later (t=4600): active score should be halved, while raw historical remains identical
    b_1hr, _, _ = scorer.evaluate(
        co_occurrence_count=8,
        proximity_count=4,
        following_count=0,
        total_duration_s=60.0,
        last_observed=1000.0,
        now=4600.0,
    )

    assert b_1hr.total_raw_score == init_score
    assert abs(b_1hr.active_decayed_score - (init_score * 0.5)) < 0.01
    assert abs(b_1hr.decay_factor - 0.5) < 0.01
