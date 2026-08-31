"""Explainable Relationship Scoring, Contribution Attribution, and Exponential Time-Decay."""

from __future__ import annotations

import math

from vantage.relationship.config import RelationshipScoringConfig
from vantage.relationship.models import (
    DerivedRelationshipPattern,
    RelationshipScoreBreakdown,
)


class RelationshipScorer:
    """Computes explainable, evidence-backed relationship scores with exponential recency decay."""

    def __init__(self, config: RelationshipScoringConfig | None = None) -> None:
        self.config = config or RelationshipScoringConfig()

    def evaluate(
        self,
        co_occurrence_count: int,
        proximity_count: int,
        following_count: int,
        total_duration_s: float,
        last_observed: float,
        now: float,
    ) -> tuple[RelationshipScoreBreakdown, DerivedRelationshipPattern | None, str]:
        """Compute score breakdown, decayed active score, primary pattern, and summary."""
        # 1. Non-linear asymptotic saturation curves for each signal
        c_co = self.config.co_occurrence_weight * (
            1.0 - math.exp(-max(0, co_occurrence_count) / self.config.co_occurrence_scale)
        )
        c_prox = self.config.proximity_weight * (
            1.0 - math.exp(-max(0, proximity_count) / self.config.proximity_scale)
        )
        c_follow = self.config.following_weight * (
            1.0 - math.exp(-max(0, following_count) / self.config.following_scale)
        )
        c_dur = self.config.duration_weight * (
            1.0 - math.exp(-max(0.0, total_duration_s) / self.config.duration_scale_s)
        )

        total_raw = min(1.0, c_co + c_prox + c_follow + c_dur)

        # 2. Exponential recency decay (preserves historical score while decaying active strength)
        elapsed = max(0.0, now - last_observed)
        decay_factor = math.exp(-math.log(2.0) * elapsed / self.config.half_life_s)
        active_score = total_raw * decay_factor

        breakdown = RelationshipScoreBreakdown(
            co_occurrence_contribution=c_co,
            proximity_contribution=c_prox,
            following_contribution=c_follow,
            duration_contribution=c_dur,
            total_raw_score=total_raw,
            active_decayed_score=active_score,
            decay_factor=decay_factor,
        )

        # 3. Derive primary relationship pattern from observable signals
        pattern = None
        if following_count >= 2:
            pattern = DerivedRelationshipPattern.FOLLOWING_PATTERN_CANDIDATE
        elif proximity_count >= 4 and total_duration_s >= 45.0:
            pattern = DerivedRelationshipPattern.RECURRENT_INTERACTION_PAIR
        elif co_occurrence_count >= 5 and proximity_count >= 2:
            pattern = DerivedRelationshipPattern.FREQUENT_CO_TRAVELER
        elif co_occurrence_count >= 3:
            pattern = DerivedRelationshipPattern.PERSISTENT_CLUSTER_ASSOCIATE

        # 4. Formulate human-auditable evidence summary
        summary_parts = []
        if co_occurrence_count > 0:
            summary_parts.append(f"co-appeared {co_occurrence_count}x")
        if proximity_count > 0:
            summary_parts.append(f"recurrent proximity {proximity_count}x")
        if following_count > 0:
            summary_parts.append(f"following alignment {following_count}x")
        if total_duration_s > 0:
            summary_parts.append(f"{total_duration_s:.1f}s joint duration")

        summary = ", ".join(summary_parts) if summary_parts else "initial observation"

        return breakdown, pattern, summary
