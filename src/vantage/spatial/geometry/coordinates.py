"""Coordinate Systems & Entity Ground Plane Geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vantage.perception.contracts import BoundingBox


class CoordinateSpace(str, Enum):
    """Explicit coordinate reference spaces for spatial video analytics."""

    CAMERA_IMAGE_NORMALIZED = "camera_image_normalized"  # [0.0, 1.0] image plane
    FLOORPLAN_RADAR_PERCENT = "floorplan_radar_percent"  # [0.0, 100.0] 2D facility floorplan
    WORLD_GROUND_PLANE = "world_ground_plane"  # Real-world metric coords (meters)
    DIGITAL_TWIN_3D = "digital_twin_3d"  # 3D WebGL / Spatial coordinate frame


@dataclass(frozen=True, slots=True)
class Point2D:
    """A 2D point representation in continuous space."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: Point2D) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


def get_entity_foot_point(
    box: BoundingBox,
    frame_width: int = 1,
    frame_height: int = 1,
    normalize: bool = True,
) -> Point2D:
    """Calculate the ground-plane foot contact point for an entity's bounding box.

    Design Rationale:
    -----------------
    In surveillance and spatial analytics, an entity's center point (box.y + height/2)
    floats in mid-air (torso height) and varies with body posture, perspective foreshortening,
    and camera pitch.

    The bottom-center of the bounding box:
        foot_x = (box.x1 + box.x2) / 2.0
        foot_y = box.y2
    approximates where the person's feet touch the physical ground plane.

    When testing geofence polygon inclusion (such as exclusion doorways, till queues,
    or hazard zones), the foot contact point provides the mathematically robust anchor
    for ground-plane containment.
    """
    w = max(1, frame_width) if normalize else 1.0
    h = max(1, frame_height) if normalize else 1.0

    bx, by = box.bottom_center
    foot_x = bx / w
    foot_y = by / h

    if normalize:
        foot_x = max(0.0, min(1.0, float(foot_x)))
        foot_y = max(0.0, min(1.0, float(foot_y)))

    return Point2D(x=foot_x, y=foot_y)
