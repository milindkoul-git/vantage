"""Configuration Parameters for Incident Correlation, Attribution Weights, and Lifecycle Timeouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncidentCorrelatorConfig:
    """Configurable weights, decision thresholds, and timeouts for multi-event incident reasoning."""

    # Decision thresholds
    attach_threshold: float = 0.65  # High confidence: attach automatically
    candidate_threshold: float = 0.35  # Ambiguous: record correlation candidate
    merge_threshold: float = 0.70  # Threshold to suggest merging two active incidents

    # Timeouts (seconds)
    temporal_window_s: float = 120.0  # Max temporal window for immediate proximity
    quiescent_timeout_s: float = 60.0  # Time without events to transition to QUIESCENT
    resolution_timeout_s: float = 300.0  # Total inactivity before transitioning to RESOLVED

    # Positive Attribution Weights (sum to 1.0)
    entity_overlap_weight: float = 0.35
    temporal_proximity_weight: float = 0.20
    spatial_zone_weight: float = 0.15  # Supporting evidence only, never dominates
    relationship_weight: float = 0.15
    behavior_scene_weight: float = 0.15

    # Negative Continuity Penalties
    impossible_speed_penalty: float = 0.40  # Physically impossible cross-camera transition
    degraded_identity_penalty: float = 0.20  # Coasting or low-confidence identity
    temporal_gap_penalty: float = 0.25  # Extended unexplained temporal gap (>180s)

    def __post_init__(self) -> None:
        total_w = (
            self.entity_overlap_weight
            + self.temporal_proximity_weight
            + self.spatial_zone_weight
            + self.relationship_weight
            + self.behavior_scene_weight
        )
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"positive correlation weights must sum to 1.0, got {total_w:.4f}")
        if not (0.0 < self.candidate_threshold < self.attach_threshold <= 1.0):
            raise ValueError(
                f"invalid decision thresholds: candidate={self.candidate_threshold}, attach={self.attach_threshold}"
            )
        if (
            self.quiescent_timeout_s <= 0
            or self.resolution_timeout_s <= self.quiescent_timeout_s
        ):
            raise ValueError(
                f"invalid timeouts: quiescent={self.quiescent_timeout_s}, resolution={self.resolution_timeout_s}"
            )
