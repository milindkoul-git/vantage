"""3D Spatial Facility Twin: Digital Mesh, Camera Frustums, and Metric World Coordinates."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vantage.spatial.geometry.coordinates import Point2D

if TYPE_CHECKING:
    from vantage.events.zone_registry import ZoneRegistry


@dataclass(frozen=True, slots=True)
class CameraMount3D:
    """3D physical position, orientation, and viewing frustum for a surveillance camera."""

    camera_id: str
    name: str
    x: float  # meters
    y: float  # meters (height above ground)
    z: float  # meters
    yaw_deg: float  # heading angle (0 = North/+Z, 90 = East/+X)
    pitch_deg: float  # downward tilt angle (e.g. -30°)
    fov_deg: float  # horizontal field of view
    range_m: float  # effective sensing range in meters
    color: str = "#00e5ff"

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "position": [self.x, self.y, self.z],
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "fov_deg": self.fov_deg,
            "range_m": self.range_m,
            "color": self.color,
        }


class FacilitySpatialTwin:
    """Manages the 3D metric coordinate system, facility layout, camera frustums, and entities."""

    def __init__(self, zone_registry: ZoneRegistry | None = None) -> None:
        self.zone_registry = zone_registry
        # Overall facility metric bounds (40m width x 24m depth x 4.5m ceiling)
        self.width_m: float = 40.0
        self.depth_m: float = 24.0
        self.height_m: float = 4.5

        # 3D Physical Camera Mounts across facility sectors
        self.camera_mounts: dict[str, CameraMount3D] = {
            "cam_01_retail": CameraMount3D(
                camera_id="cam_01_retail",
                name="Retail Showroom",
                x=2.0,
                y=3.4,
                z=2.0,
                yaw_deg=45.0,
                pitch_deg=-28.0,
                fov_deg=72.0,
                range_m=16.0,
                color="#00e5ff",
            ),
            "cam_02_crosswalk": CameraMount3D(
                camera_id="cam_02_crosswalk",
                name="Crosswalk Traffic",
                x=38.0,
                y=3.6,
                z=2.0,
                yaw_deg=-135.0,
                pitch_deg=-25.0,
                fov_deg=78.0,
                range_m=18.0,
                color="#ffb700",
            ),
            "cam_03_corridor": CameraMount3D(
                camera_id="cam_03_corridor",
                name="Corridor Walkway",
                x=2.0,
                y=3.2,
                z=22.0,
                yaw_deg=15.0,
                pitch_deg=-22.0,
                fov_deg=65.0,
                range_m=20.0,
                color="#00ffc8",
            ),
            "cam_04_doorway": CameraMount3D(
                camera_id="cam_04_doorway",
                name="Pedestrians Entry",
                x=38.0,
                y=3.2,
                z=22.0,
                yaw_deg=-145.0,
                pitch_deg=-30.0,
                fov_deg=68.0,
                range_m=15.0,
                color="#af52de",
            ),
        }

        # Facility Architecture: Rooms & Structural Partitions
        self.rooms = [
            {
                "id": "sector_retail",
                "name": "Retail Showroom Floor",
                "bounds": [0.0, 0.0, 19.0, 11.5],  # x1, z1, x2, z2
                "floor_color": "#161b26",
                "wall_color": "#20293a",
            },
            {
                "id": "sector_plaza",
                "name": "Plaza & Crosswalk",
                "bounds": [21.0, 0.0, 40.0, 11.5],
                "floor_color": "#131822",
                "wall_color": "#1c2436",
            },
            {
                "id": "sector_corridor",
                "name": "Secure Transit Corridor",
                "bounds": [0.0, 12.5, 21.0, 24.0],
                "floor_color": "#111620",
                "wall_color": "#1a2233",
            },
            {
                "id": "sector_doorway",
                "name": "Main Entrance Vestibule",
                "bounds": [21.0, 12.5, 40.0, 24.0],
                "floor_color": "#141a24",
                "wall_color": "#1e2738",
            },
        ]

        # Architectural walls with doorways (segments: [x1, z1, x2, z2, height])
        self.walls = [
            # Outer perimeter
            [0.0, 0.0, 40.0, 0.0, 3.8],
            [40.0, 0.0, 40.0, 24.0, 3.8],
            [40.0, 24.0, 0.0, 24.0, 3.8],
            [0.0, 24.0, 0.0, 0.0, 3.8],
            # Center dividing wall with corridor passage opening
            [0.0, 12.0, 16.0, 12.0, 3.2],
            [22.0, 12.0, 40.0, 12.0, 3.2],
            # Vertical sector dividing wall with doorway opening
            [20.0, 0.0, 20.0, 8.0, 3.2],
            [20.0, 15.0, 20.0, 24.0, 3.2],
        ]

        # Entity 3D Positions & Motion Trails: global_id -> list of [x, y, z, wall_time]
        self._entity_3d_trails: dict[str, list[tuple[float, float, float, float]]] = {}
        self._entity_3d_state: dict[str, dict[str, Any]] = {}

    def project_camera_to_3d(
        self,
        camera_id: str,
        norm_x: float,
        norm_y: float,
    ) -> tuple[float, float, float]:
        """Transform normalized camera image coordinates [0, 1] into 3D world space (meters)."""
        mount = self.camera_mounts.get(camera_id)
        if not mount:
            # Fallback center projection
            return (self.width_m * norm_x, 0.0, self.depth_m * norm_y)

        # Sector boundary anchors
        if camera_id == "cam_01_retail":
            wx = 1.0 + norm_x * 17.5
            wz = 1.0 + norm_y * 10.0
        elif camera_id == "cam_02_crosswalk":
            wx = 21.5 + (1.0 - norm_x) * 17.5
            wz = 1.0 + norm_y * 10.0
        elif camera_id == "cam_03_corridor":
            wx = 1.0 + norm_x * 19.5
            wz = 13.0 + norm_y * 10.0
        elif camera_id == "cam_04_doorway":
            wx = 21.5 + (1.0 - norm_x) * 17.5
            wz = 13.0 + norm_y * 10.0
        else:
            wx = self.width_m * norm_x
            wz = self.depth_m * norm_y

        return (round(wx, 2), 0.0, round(wz, 2))

    def update_entity_3d(
        self,
        global_id: str,
        camera_id: str,
        label: str,
        foot_point: Point2D,
        speed: float,
        bearing_deg: float | None,
        motion: str,
        posture: str,
        wall_time: float,
    ) -> None:
        """Update live 3D entity tracking with trajectory trail."""
        wx, _wy, wz = self.project_camera_to_3d(camera_id, foot_point.x, foot_point.y)

        # Compute velocity vector in 3D world coordinates
        vel_x = 0.0
        vel_z = 0.0
        if bearing_deg is not None and speed > 0.1:
            rad = math.radians(bearing_deg)
            vel_x = round(math.sin(rad) * speed * 1.5, 2)
            vel_z = round(math.cos(rad) * speed * 1.5, 2)

        self._entity_3d_state[global_id] = {
            "entity_id": global_id,
            "label": label,
            "camera_id": camera_id,
            "position": [wx, 0.0, wz],
            "velocity": [vel_x, 0.0, vel_z],
            "speed": round(speed, 2),
            "bearing_deg": bearing_deg,
            "motion": motion,
            "posture": posture,
            "last_seen": wall_time,
        }

        # Update Trail
        if global_id not in self._entity_3d_trails:
            self._entity_3d_trails[global_id] = []
        trail = self._entity_3d_trails[global_id]
        trail.append((wx, 0.05, wz, wall_time))
        # Keep last 25 trail points (max 15s history)
        if len(trail) > 25:
            trail.pop(0)

    def get_digital_twin_state(
        self, live_occupancies: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Export comprehensive 3D scene graph state for Three.js WebGL renderer."""
        now = time.time()
        # Clean stale entities (> 4.0s without update)
        active_entities = [
            st for gid, st in self._entity_3d_state.items() if now - st["last_seen"] < 4.0
        ]

        # 3D Geofence Extrusions (from ZoneRegistry)
        extruded_zones = []
        if self.zone_registry:
            snapshot = self.zone_registry.get_snapshot()
            for z in snapshot.list_all_zones():
                # Convert 2D normalized camera polygon vertices into 3D metric ground plane vertices
                poly_3d = []
                for pt in z.polygon.vertices:
                    wx, _, wz = self.project_camera_to_3d(z.camera_id, pt.x, pt.y)
                    poly_3d.append([wx, wz])

                occ = (live_occupancies or {}).get(z.zone_id, 0)
                extruded_zones.append(
                    {
                        "zone_id": z.zone_id,
                        "name": z.name,
                        "camera_id": z.camera_id,
                        "zone_type": z.zone_type.value,
                        "polygon_3d": poly_3d,
                        "height_m": 2.8,
                        "color": z.color,
                        "severity": z.severity,
                        "occupancy": occ,
                    }
                )

        # Format trails
        trails_payload = {}
        for ent in active_entities:
            gid = ent["entity_id"]
            if gid in self._entity_3d_trails:
                trails_payload[gid] = [
                    [p[0], p[1], p[2]] for p in self._entity_3d_trails[gid] if now - p[3] < 12.0
                ]

        return {
            "facility": {
                "width_m": self.width_m,
                "depth_m": self.depth_m,
                "height_m": self.height_m,
                "rooms": self.rooms,
                "walls": self.walls,
            },
            "cameras": [c.to_dict() for c in self.camera_mounts.values()],
            "zones": extruded_zones,
            "entities": active_entities,
            "trails": trails_payload,
            "timestamp": now,
        }
