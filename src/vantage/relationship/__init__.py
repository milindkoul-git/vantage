"""Persistent Entity Relationship & Long-Horizon Intelligence Subsystem.

Provides long-horizon entity association, observational relationship modeling,
configurable evidence-backed scoring with exponential time decay, and following-pattern detection.
"""

from __future__ import annotations

from vantage.relationship.config import FollowingDetectorConfig, RelationshipScoringConfig
from vantage.relationship.following import FollowingPatternDetector
from vantage.relationship.models import (
    DerivedRelationshipPattern,
    EntityRelationship,
    ProximityBasis,
    RelationshipScoreBreakdown,
    RelationshipSignal,
    RelationshipSignalType,
)
from vantage.relationship.scoring import RelationshipScorer
from vantage.relationship.service import RelationshipService
from vantage.relationship.tracker import PersistentRelationshipTracker

__all__ = [
    "DerivedRelationshipPattern",
    "EntityRelationship",
    "FollowingDetectorConfig",
    "FollowingPatternDetector",
    "PersistentRelationshipTracker",
    "ProximityBasis",
    "RelationshipScoreBreakdown",
    "RelationshipScorer",
    "RelationshipScoringConfig",
    "RelationshipService",
    "RelationshipSignal",
    "RelationshipSignalType",
]
