"""Configuration parameters for Relationship Scoring and Following-Pattern Detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelationshipScoringConfig:
    """Configurable weights, saturation scales, and time-decay half-life for relationship scoring."""

    co_occurrence_weight: float = 0.20
    proximity_weight: float = 0.35
    following_weight: float = 0.30
    duration_weight: float = 0.15

    half_life_s: float = 3600.0  # 1 hour half-life for active score recency decay
    co_occurrence_scale: float = 8.0  # N_co for ~63% saturation
    proximity_scale: float = 4.0  # N_prox for ~63% saturation
    following_scale: float = 2.0  # N_follow for ~63% saturation
    duration_scale_s: float = 60.0  # Interaction seconds for ~63% saturation

    def __post_init__(self) -> None:
        total_w = (
            self.co_occurrence_weight
            + self.proximity_weight
            + self.following_weight
            + self.duration_weight
        )
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"relationship scoring weights must sum to 1.0, got {total_w:.4f}")
        if self.half_life_s <= 0:
            raise ValueError(f"half_life_s must be strictly positive, got {self.half_life_s}")


@dataclass(frozen=True, slots=True)
class FollowingDetectorConfig:
    """Configurable thresholds for lagged trajectory alignment and following-pattern detection."""

    min_lag_s: float = 0.5
    max_lag_s: float = 6.0
    max_trajectory_error: float = 0.12  # Maximum normalized Euclidean error
    min_heading_alignment: float = (
        0.70  # Fraction of sampled points matching heading within 35 deg
    )
    min_path_length: float = 0.30  # Minimum path length in entity heights
    min_evidence_count: int = 4  # Minimum aligned trajectory points

    def __post_init__(self) -> None:
        if self.min_lag_s <= 0 or self.max_lag_s <= self.min_lag_s:
            raise ValueError(
                f"invalid lag range: min_lag_s={self.min_lag_s}, max_lag_s={self.max_lag_s}"
            )
        if not 0.0 < self.min_heading_alignment <= 1.0:
            raise ValueError(
                f"min_heading_alignment must be in (0, 1], got {self.min_heading_alignment}"
            )
