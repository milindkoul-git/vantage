"""2D Polygon Representation & Robust Ray-Casting Point-in-Polygon Engine."""

from __future__ import annotations

from collections.abc import Sequence

from vantage.spatial.geometry.coordinates import Point2D

_EPSILON = 1e-9


class Polygon:
    """Immutable 2D Polygon supporting arbitrary convex and concave boundaries."""

    __slots__ = ("_area", "_max_x", "_max_y", "_min_x", "_min_y", "_vertices")

    def __init__(self, vertices: Sequence[Point2D | tuple[float, float] | list[float]]) -> None:
        pts: list[Point2D] = []
        for v in vertices:
            if isinstance(v, Point2D):
                pts.append(v)
            elif isinstance(v, (tuple, list)) and len(v) >= 2:
                pts.append(Point2D(x=float(v[0]), y=float(v[1])))
            else:
                raise ValueError(f"Invalid vertex representation: {v}")

        if len(pts) < 3:
            raise ValueError(f"A polygon requires at least 3 vertices, got {len(pts)}")

        self._vertices: tuple[Point2D, ...] = tuple(pts)

        xs = [p.x for p in self._vertices]
        ys = [p.y for p in self._vertices]
        self._min_x: float = min(xs)
        self._max_x: float = max(xs)
        self._min_y: float = min(ys)
        self._max_y: float = max(ys)

        # Precompute Shoelace area
        n = len(self._vertices)
        area2 = 0.0
        for i in range(n):
            j = (i + 1) % n
            area2 += self._vertices[i].x * self._vertices[j].y
            area2 -= self._vertices[j].x * self._vertices[i].y
        self._area: float = abs(area2) / 2.0

    @property
    def vertices(self) -> tuple[Point2D, ...]:
        return self._vertices

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        return (self._min_x, self._min_y, self._max_x, self._max_y)

    @property
    def area(self) -> float:
        return self._area

    def contains_point(self, point: Point2D | tuple[float, float]) -> bool:
        """Evaluate if a point is strictly inside, on an edge, or on a vertex of the polygon.

        Deterministic Boundary Policy:
        ------------------------------
        - Strictly inside polygon: True
        - On polygon vertex: True
        - On polygon edge: True
        - Outside polygon: False

        Uses ray-casting algorithm casting a horizontal ray to the positive X direction
        with explicit edge-collinearity and vertex collision handling.
        """
        px = point.x if isinstance(point, Point2D) else float(point[0])
        py = point.y if isinstance(point, Point2D) else float(point[1])

        # 1. Fast bounding box rejection
        if px < self._min_x - _EPSILON or px > self._max_x + _EPSILON:
            return False
        if py < self._min_y - _EPSILON or py > self._max_y + _EPSILON:
            return False

        n = len(self._vertices)
        inside = False

        for i in range(n):
            p1 = self._vertices[i]
            p2 = self._vertices[(i + 1) % n]

            # 2. Check vertex collision
            if abs(px - p1.x) < _EPSILON and abs(py - p1.y) < _EPSILON:
                return True

            # 3. Check segment boundary collision (point on edge)
            if self._is_point_on_segment(px, py, p1.x, p1.y, p2.x, p2.y):
                return True

            # 4. Ray-Casting intersection
            # Ensure p1 is lower Y and p2 is higher Y for crossing test
            if (p1.y > py) != (p2.y > py):
                # Calculate X-intersection of ray with edge p1-p2
                x_intersection = (p2.x - p1.x) * (py - p1.y) / (p2.y - p1.y) + p1.x
                if px < x_intersection:
                    inside = not inside

        return inside

    @staticmethod
    def _is_point_on_segment(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> bool:
        """Check if (px, py) lies on the segment (x1, y1)-(x2, y2)."""
        # Cross product to check collinearity
        cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross) > _EPSILON:
            return False

        # Dot product / bounding box check to ensure point is between endpoints
        return bool(
            min(x1, x2) - _EPSILON <= px <= max(x1, x2) + _EPSILON
            and min(y1, y2) - _EPSILON <= py <= max(y1, y2) + _EPSILON
        )

    def is_valid(self) -> tuple[bool, str]:
        """Verify non-degenerate geometry (area > 0 and no self-intersection)."""
        if len(self._vertices) < 3:
            return False, "Polygon requires at least 3 vertices"
        if self._area < _EPSILON:
            return False, "Polygon has zero or degenerate area"

        # Check self-intersection among non-adjacent edges
        n = len(self._vertices)
        for i in range(n):
            a1 = self._vertices[i]
            a2 = self._vertices[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Adjacent edges at wrap-around
                b1 = self._vertices[j]
                b2 = self._vertices[(j + 1) % n]
                if self._segments_intersect(a1, a2, b1, b2):
                    return False, f"Polygon is self-intersecting between edges {i} and {j}"

        return True, "Valid"

    @staticmethod
    def _segments_intersect(p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D) -> bool:
        """Check if segment p1-p2 strictly intersects segment p3-p4."""

        def ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
            return (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)

        d1 = ccw(p1, p2, p3)
        d2 = ccw(p1, p2, p4)
        d3 = ccw(p3, p4, p1)
        d4 = ccw(p3, p4, p2)

        return bool(
            ((d1 > _EPSILON and d2 < -_EPSILON) or (d1 < -_EPSILON and d2 > _EPSILON))
            and ((d3 > _EPSILON and d4 < -_EPSILON) or (d3 < -_EPSILON and d4 > _EPSILON))
        )

    def to_list(self) -> list[tuple[float, float]]:
        return [p.to_tuple() for p in self._vertices]

    @classmethod
    def from_list(cls, coords: Sequence[Sequence[float]]) -> Polygon:
        return cls([Point2D(x=float(c[0]), y=float(c[1])) for c in coords])
