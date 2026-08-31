"""2D Facility Digital Twin Radar & Ground-Plane Homography Projection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vantage.perception.contracts import BoundingBox


@dataclass
class CameraFrustum:
    """Represents a camera's field of view on the top-down 2D floorplan."""

    camera_id: str
    zone_name: str
    zone_rect: tuple[float, float, float, float]  # (x1, y1, x2, y2) in 0-100% floorplan coords
    camera_pos: tuple[float, float]  # (x, y) origin of camera mount
    fov_angle_deg: float = 65.0


@dataclass
class RadarEntityDot:
    """A single entity rendered on the 2D floorplan radar."""

    global_id: str
    camera_id: str
    label: str
    x: float  # 0-100% floorplan coordinate
    y: float  # 0-100% floorplan coordinate
    motion: str
    speed: float
    activity: str
    last_updated: float


class FacilityRadarMap:
    """Projects multi-camera detections onto a unified 2D facility floorplan."""

    def __init__(self) -> None:
        # Default 4-zone facility architectural layout
        self._frustums: dict[str, CameraFrustum] = {
            "cam_01_view_a": CameraFrustum(
                "cam_01_view_a", "NORTH_CORRIDOR_A", (10.0, 10.0, 48.0, 48.0), (10.0, 10.0)
            ),
            "cam_02_view_b": CameraFrustum(
                "cam_02_view_b", "NORTH_CORRIDOR_B", (10.0, 10.0, 48.0, 48.0), (48.0, 10.0)
            ),
            "cam_01_retail": CameraFrustum(
                "cam_01_retail", "RETAIL_SHOWROOM", (10.0, 10.0, 48.0, 48.0), (10.0, 10.0)
            ),
            "cam_02_crosswalk": CameraFrustum(
                "cam_02_crosswalk", "OUTDOOR_CROSSWALK", (52.0, 10.0, 90.0, 48.0), (52.0, 10.0)
            ),
            "cam_03_corridor": CameraFrustum(
                "cam_03_corridor", "MAIN_CORRIDOR", (10.0, 52.0, 48.0, 90.0), (10.0, 52.0)
            ),
            "cam_04_doorway": CameraFrustum(
                "cam_04_doorway", "ENTRY_DOORWAY", (52.0, 52.0, 90.0, 90.0), (52.0, 52.0)
            ),
        }
        self._active_dots: dict[str, RadarEntityDot] = {}
        self._trail_history: dict[
            str, list[tuple[float, float, float]]
        ] = {}  # global_id -> list of (x, y, t)

    def register_camera(
        self, camera_id: str, zone_name: str, zone_rect: tuple[float, float, float, float]
    ) -> None:
        self._frustums[camera_id] = CameraFrustum(
            camera_id=camera_id,
            zone_name=zone_name,
            zone_rect=zone_rect,
            camera_pos=(zone_rect[0], zone_rect[1]),
        )

    def project_entity(
        self,
        camera_id: str,
        global_id: str,
        label: str,
        box: BoundingBox,
        frame_width: int,
        frame_height: int,
        motion: str = "walking",
        speed: float = 0.0,
        activity: str = "idle",
        wall_time: float | None = None,
    ) -> tuple[float, float]:
        """Project bounding box bottom-center into 2D floorplan percentage coordinates (0-100%)."""
        now = wall_time or time.time()
        frustum = self._frustums.get(camera_id)
        if not frustum:
            # Default fallback zone
            frustum = CameraFrustum(
                camera_id, camera_id.upper(), (20.0, 20.0, 80.0, 80.0), (20.0, 20.0)
            )

        # Bottom center in camera image
        bx = (box.x1 + box.x2) / 2.0
        by = box.y2  # feet on ground plane

        norm_u = max(0.0, min(1.0, bx / max(1, frame_width)))
        norm_v = max(0.0, min(1.0, by / max(1, frame_height)))

        # Perspective ground-plane mapping within zone bounding rect
        zx1, zy1, zx2, zy2 = frustum.zone_rect
        zw = zx2 - zx1
        zh = zy2 - zy1

        # Projected floor coordinates
        floor_x = round(zx1 + norm_u * zw, 1)
        floor_y = round(zy1 + norm_v * zh, 1)

        dot = RadarEntityDot(
            global_id=global_id,
            camera_id=camera_id,
            label=label,
            x=floor_x,
            y=floor_y,
            motion=motion,
            speed=speed,
            activity=activity,
            last_updated=now,
        )
        self._active_dots[global_id] = dot

        # Record trail
        trails = self._trail_history.setdefault(global_id, [])
        trails.append((floor_x, floor_y, now))
        # Keep last 20 seconds of trail
        self._trail_history[global_id] = [p for p in trails if now - p[2] <= 20.0]

        return floor_x, floor_y

    def get_radar_state(self, active_window_s: float = 5.0) -> dict[str, Any]:
        """Return the current 2D radar snapshot for frontend rendering."""
        now = time.time()
        live_dots = []
        for gid, dot in list(self._active_dots.items()):
            if now - dot.last_updated <= active_window_s:
                trail = [(p[0], p[1]) for p in self._trail_history.get(gid, [])]
                live_dots.append(
                    {
                        "id": dot.global_id,
                        "label": dot.label,
                        "camera": dot.camera_id,
                        "x": dot.x,
                        "y": dot.y,
                        "motion": dot.motion,
                        "speed": dot.speed,
                        "activity": dot.activity,
                        "trail": trail[-15:],  # Last 15 waypoints
                    }
                )
            else:
                self._active_dots.pop(gid, None)

        zones = [
            {
                "camera_id": f.camera_id,
                "name": f.zone_name,
                "rect": list(f.zone_rect),
                "origin": list(f.camera_pos),
            }
            for f in self._frustums.values()
        ]

        return {
            "timestamp": now,
            "zones": zones,
            "entities": live_dots,
            "active_count": len(live_dots),
        }
