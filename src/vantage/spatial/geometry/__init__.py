"""Spatial 2D/3D Geometry & Coordinate Transformations."""

from vantage.spatial.geometry.coordinates import (
    CoordinateSpace,
    Point2D,
    get_entity_foot_point,
)
from vantage.spatial.geometry.polygon import Polygon

__all__ = [
    "CoordinateSpace",
    "Point2D",
    "Polygon",
    "get_entity_foot_point",
]
