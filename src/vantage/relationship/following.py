"""Observational Following-Pattern Detector using Lagged Trajectory Alignment and Heading Consistency."""

from __future__ import annotations

import math
from collections.abc import Sequence

from vantage.relationship.config import FollowingDetectorConfig
from vantage.relationship.models import (
    ProximityBasis,
    RelationshipSignal,
    RelationshipSignalType,
)


class FollowingPatternDetector:
    """Detects delayed trajectory alignment and directional following between entity pairs."""

    def __init__(self, config: FollowingDetectorConfig | None = None) -> None:
        self.config = config or FollowingDetectorConfig()

    def evaluate_trajectories(
        self,
        entity_a: str,
        traj_a: Sequence[
            tuple[float, float, float, float | None]
        ],  # (timestamp, x, y, bearing_deg)
        entity_b: str,
        traj_b: Sequence[tuple[float, float, float, float | None]],
        camera_id: str,
        now: float,
        proximity_basis: ProximityBasis = ProximityBasis.NORMALIZED_IMAGE_SPACE,
    ) -> tuple[bool, RelationshipSignal | None]:
        """Evaluate if Entity B is observationally following Entity A (or vice versa)."""
        if (
            len(traj_a) < self.config.min_evidence_count
            or len(traj_b) < self.config.min_evidence_count
        ):
            return False, None

        # Check for B following A: search optimal lag tau where B(t) matches A(t - tau)
        best_lag, best_error, heading_alignment, path_b = self._search_optimal_lag(
            traj_a, traj_b
        )

        if (
            best_lag is not None
            and best_error <= self.config.max_trajectory_error
            and heading_alignment >= self.config.min_heading_alignment
            and path_b >= self.config.min_path_length
        ):
            strength = max(0.5, 1.0 - (best_error / self.config.max_trajectory_error))
            signal = RelationshipSignal(
                signal_type=RelationshipSignalType.LAGGED_TRAJECTORY_ALIGNMENT,
                timestamp=now,
                camera_id=camera_id,
                zone_id=None,
                strength=round(strength, 3),
                duration_s=round(traj_b[-1][0] - traj_b[0][0], 2),
                proximity_basis=proximity_basis,
                distance_metric=round(best_error, 4),
                evidence={
                    "follower_id": entity_b,
                    "target_id": entity_a,
                    "lag_s": round(best_lag, 2),
                    "mean_trajectory_error": round(best_error, 4),
                    "heading_alignment_rate": round(heading_alignment, 3),
                    "path_length": round(path_b, 3),
                },
            )
            return True, signal

        return False, None

    def _search_optimal_lag(
        self,
        traj_lead: Sequence[tuple[float, float, float, float | None]],
        traj_follow: Sequence[tuple[float, float, float, float | None]],
    ) -> tuple[float | None, float, float, float]:
        """Find the temporal lag tau in [min_lag_s, max_lag_s] that minimizes trajectory separation."""
        # Calculate total path length of follower
        path_length = 0.0
        for k in range(1, len(traj_follow)):
            path_length += math.hypot(
                traj_follow[k][1] - traj_follow[k - 1][1],
                traj_follow[k][2] - traj_follow[k - 1][2],
            )

        if path_length < self.config.min_path_length:
            return None, 1.0, 0.0, path_length

        # Step 1. Compute contemporaneous separation at tau = 0 (simultaneous co-presence)
        contemporaneous_errors: list[float] = []
        for t_f, fx, fy, _ in traj_follow:
            nearest_lead = min(traj_lead, key=lambda s: abs(s[0] - t_f))
            if abs(nearest_lead[0] - t_f) <= 0.40:
                contemporaneous_errors.append(
                    math.hypot(fx - nearest_lead[1], fy - nearest_lead[2])
                )
        mean_contemp_err = (
            sum(contemporaneous_errors) / len(contemporaneous_errors)
            if contemporaneous_errors
            else 0.0
        )

        best_lag = None
        min_error = 1e9
        best_heading_alignment = 0.0

        # Step 2. Step through candidate lags in 0.25s increments
        lag_steps = int((self.config.max_lag_s - self.config.min_lag_s) / 0.25) + 1
        for step in range(lag_steps):
            tau = self.config.min_lag_s + step * 0.25
            errors: list[float] = []
            heading_matches = 0
            heading_samples = 0

            for t_f, fx, fy, f_bearing in traj_follow:
                target_t = t_f - tau
                # Find nearest sample in leader trajectory
                nearest_lead = min(traj_lead, key=lambda s: abs(s[0] - target_t))
                if abs(nearest_lead[0] - target_t) > 0.40:
                    continue  # Gap too wide

                err = math.hypot(fx - nearest_lead[1], fy - nearest_lead[2])
                errors.append(err)

                if f_bearing is not None and nearest_lead[3] is not None:
                    heading_samples += 1
                    diff = abs(f_bearing - nearest_lead[3]) % 360.0
                    delta_angle = min(diff, 360.0 - diff)
                    if delta_angle <= 35.0:
                        heading_matches += 1

            if len(errors) >= self.config.min_evidence_count:
                mean_err = sum(errors) / len(errors)
                align_rate = (heading_matches / heading_samples) if heading_samples > 0 else 0.8
                if mean_err < min_error:
                    min_error = mean_err
                    best_lag = tau
                    best_heading_alignment = align_rate

        # Guard: If simultaneous separation is tiny (e.g. < 0.07) and lag does not significantly improve error,
        # entities are walking side-by-side or abreast, NOT following.
        if mean_contemp_err < 0.07 and min_error >= (mean_contemp_err * 0.80):
            return None, 1.0, 0.0, path_length

        return best_lag, min_error, best_heading_alignment, path_length
