"""Mapping a camera's ground plane onto facility coordinates.

Three coordinate spaces are in play, and the distinction matters because
conflating them is how a dashboard comes to report metres it never measured:

1. Camera/image space, ``[0, W] x [0, H]`` pixels.
2. Normalised camera space, ``[0, 1] x [0, 1]``.
3. Facility space, metres on a ground plane.

What is here is the honest third step: an affine map from a camera's normalised
ground plane onto a rectangle of the facility that the operator has declared
that camera covers. It is a *configured* correspondence, not a measured one.
Two people the same distance apart in metres will not be the same distance apart
in this space unless the camera happens to be looking straight down.

A calibrated projection - a planar homography from four surveyed correspondences,
or a full 3x4 camera matrix - would remove that caveat. It is not implemented,
and there is deliberately no placeholder class for it: a constructor that raises
``NotImplementedError`` reads, from a call site, exactly like a feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SpatialProjection(Protocol):
    """Projects normalised camera coordinates into facility coordinates."""

    def project(self, x_norm: float, y_norm: float) -> tuple[float, float, float]:
        """``(x_norm, y_norm)`` in ``[0, 1]`` to ``(x, y, z)`` in metres."""
        ...


@dataclass(frozen=True, slots=True)
class ConfiguredSectorProjection:
    """Maps a camera's ground plane onto a declared rectangle of the facility.

    The rectangle is configuration: it says "this camera watches that part of the
    floor", which someone decided when the camera was mounted. Within it the map
    is linear, so the top of the frame lands on ``z_min`` and the bottom on
    ``z_max`` regardless of the perspective foreshortening a real lens produces.

    That is a coarse model, and it is stated as one. It is enough to put two
    people who are in different rooms in different places, which is what the
    facility view is for; it is not enough to measure the distance between them.
    """

    x_min: float
    x_max: float
    z_min: float
    z_max: float
    ground_y: float = 0.0

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.z_max <= self.z_min:
            raise ValueError(
                f"sector bounds are inverted or empty: "
                f"({self.x_min}, {self.z_min})-({self.x_max}, {self.z_max})"
            )

    def project(self, x_norm: float, y_norm: float) -> tuple[float, float, float]:
        """Map a normalised foot point into the sector.

        Clamped rather than extrapolated: a foot point slightly outside the frame
        is a tracking artefact, and continuing the line would place an entity
        outside the building.
        """
        x_clamped = max(0.0, min(1.0, x_norm))
        y_clamped = max(0.0, min(1.0, y_norm))
        world_x = self.x_min + x_clamped * (self.x_max - self.x_min)
        world_z = self.z_min + y_clamped * (self.z_max - self.z_min)
        return (round(world_x, 2), round(self.ground_y, 2), round(world_z, 2))
