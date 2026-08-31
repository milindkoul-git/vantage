"""Spatial Coordinate Projections & Metric Space Mappings.

Formalizes the distinction between:
1. Camera/Image Space [0, W] x [0, H]
2. Normalized Camera Space [0, 1] x [0, 1]
3. Configured Sector Facility Projection (e.g. 40m x 24m)
4. Calibrated Homography / Camera Matrix Projection (future)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SpatialProjection(Protocol):
    """Protocol for projecting normalized camera coordinates into world/facility coordinates."""

    def project(self, x_norm: float, y_norm: float) -> tuple[float, float, float]:
        """Project (x_norm, y_norm) to (x_world, y_world, z_world) in meters."""
        ...


@dataclass(frozen=True, slots=True)
class ConfiguredSectorProjection:
    """Projects normalized camera ground foot points into a configured facility sector bounding box.

    Truthful representation: maps camera ground plane [0, 1] into configured metric sector bounds
    [x_min, z_min] -> [x_max, z_max] without claiming optical homography calibration.
    """

    x_min: float
    x_max: float
    z_min: float
    z_max: float
    ground_y: float = 0.0

    def project(self, x_norm: float, y_norm: float) -> tuple[float, float, float]:
        """Map normalized camera foot point (x_norm, y_norm) into sector (X, 0, Z)."""
        x_clamped = max(0.0, min(1.0, x_norm))
        y_clamped = max(0.0, min(1.0, y_norm))
        world_x = self.x_min + x_clamped * (self.x_max - self.x_min)
        world_z = self.z_min + y_clamped * (self.z_max - self.z_min)
        return (round(world_x, 2), round(self.ground_y, 2), round(world_z, 2))


class CalibratedProjection:
    """Future-proof seam for 3x3 planar homography or 3x4 PnP camera projection matrices."""

    def __init__(self, homography_matrix: list[list[float]] | None = None) -> None:
        if homography_matrix is None:
            raise NotImplementedError("Calibrated projection matrix required for homography.")
        self.homography = homography_matrix

    def project(self, x_norm: float, y_norm: float) -> tuple[float, float, float]:
        raise NotImplementedError(
            "Optical homography projection is deferred to future calibration phase."
        )
