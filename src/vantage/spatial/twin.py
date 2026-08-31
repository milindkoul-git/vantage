"""3D Spatial Facility Twin: Digital Mesh, Camera Frustums, and Metric World Coordinates."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vantage.spatial.geometry.coordinates import Point2D
from vantage.spatial.projection import ConfiguredSectorProjection

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
    """The facility as metric geometry: sectors, camera mounts, and where people are.

    The layout is derived from the cameras it is given. Pass the ids of the
    cameras actually running and it lays them out as a grid of sectors across the
    configured floor area, one sector per camera, with a mount above each looking
    into its own sector. Pass nothing and it is empty, and the dashboard reports
    that there is no facility model rather than drawing one.

    That is a change of kind from what this used to do. It shipped a specific
    fictional building - four sectors named "Retail Showroom Floor", "Plaza &
    Crosswalk", "Secure Transit Corridor" and "Main Entrance Vestibule", eight
    interior wall segments, and four camera mounts keyed to the ids of a demo
    recording - and produced it regardless of which cameras were connected.
    ``project_camera_to_3d`` then branched on those same four id strings, so a
    real deployment fell through to a generic mapping while the view showed a
    building nobody had.

    What it produces now is honest but coarse, and the coarseness is stated
    rather than hidden: a sector grid is a declaration about which part of the
    floor each camera watches, not a survey of the building, and the projection
    inside each sector is affine rather than a calibrated homography. Rooms are
    named after their camera because that is the only name anyone has given.
    """

    #: Sensible default extent when the caller does not know the building's size.
    #: Chosen so that a grid of sectors is legible at the twin's default camera
    #: distance; it is a canvas, not a measurement, and is overridable.
    DEFAULT_WIDTH_M = 40.0
    DEFAULT_DEPTH_M = 24.0
    DEFAULT_HEIGHT_M = 4.5

    def __init__(
        self,
        camera_ids: Sequence[str] = (),
        *,
        zone_registry: ZoneRegistry | None = None,
        width_m: float = DEFAULT_WIDTH_M,
        depth_m: float = DEFAULT_DEPTH_M,
        height_m: float = DEFAULT_HEIGHT_M,
    ) -> None:
        if width_m <= 0 or depth_m <= 0 or height_m <= 0:
            raise ValueError("facility dimensions must be positive")

        self.zone_registry = zone_registry
        self.width_m = width_m
        self.depth_m = depth_m
        self.height_m = height_m

        self.camera_mounts: dict[str, CameraMount3D] = {}
        self.sectors: dict[str, ConfiguredSectorProjection] = {}
        self.rooms: list[dict[str, Any]] = []
        self.walls: list[list[float]] = []

        for camera_id in camera_ids:
            self.add_camera(camera_id)

        # Entity 3D positions and motion trails: global_id -> [x, y, z, wall_time]
        self._entity_3d_trails: dict[str, list[tuple[float, float, float, float]]] = {}
        self._entity_3d_state: dict[str, dict[str, Any]] = {}

    # -- layout -----------------------------------------------------------

    def add_camera(self, camera_id: str, name: str | None = None) -> CameraMount3D:
        """Give a camera a sector of the floor and a mount above it.

        Adding a camera re-lays-out every sector, because the grid's shape
        depends on how many there are. A camera attached mid-run therefore moves
        the others, which is the honest consequence of the layout being derived
        rather than surveyed: nothing here knows where the cameras really are.
        """
        if camera_id in self.camera_mounts:
            return self.camera_mounts[camera_id]

        ordered = [*self.camera_mounts, camera_id]
        names = {cid: mount.name for cid, mount in self.camera_mounts.items()}
        names[camera_id] = name or camera_id.replace("_", " ").title()

        self.camera_mounts.clear()
        self.sectors.clear()
        self.rooms.clear()

        columns = math.ceil(math.sqrt(len(ordered)))
        rows = math.ceil(len(ordered) / columns)
        sector_w = self.width_m / columns
        sector_d = self.depth_m / rows
        # A margin so adjacent sectors read as separate floors rather than one.
        margin = min(sector_w, sector_d) * 0.04

        for index, cid in enumerate(ordered):
            column, row = index % columns, index // columns
            x_min = column * sector_w + margin
            x_max = (column + 1) * sector_w - margin
            z_min = row * sector_d + margin
            z_max = (row + 1) * sector_d - margin

            self.sectors[cid] = ConfiguredSectorProjection(
                x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max
            )
            self.rooms.append(
                {
                    "id": f"sector_{cid}",
                    "name": names[cid],
                    "bounds": [
                        round(x_min, 2),
                        round(z_min, 2),
                        round(x_max, 2),
                        round(z_max, 2),
                    ],
                    "floor_color": "#1C1916",
                    "wall_color": "#2E2820",
                }
            )
            # Mounted at the near edge of its own sector and aimed at the far
            # edge, with the pitch, range and field of view that geometry
            # implies. Derived rather than declared: these numbers describe the
            # sector assignment, and the frustum drawn from them lands on the
            # floor it claims to cover. Constants would draw a cone that stops
            # short of its own sector or continues underground.
            mount_height = round(min(3.5, self.height_m - 0.5), 2)
            reach = z_max - z_min
            half_width = (x_max - x_min) / 2
            self.camera_mounts[cid] = CameraMount3D(
                camera_id=cid,
                name=names[cid],
                x=round((x_min + x_max) / 2, 2),
                y=mount_height,
                z=round(z_min, 2),
                yaw_deg=0.0,  # +Z, into its own sector
                pitch_deg=round(-math.degrees(math.atan2(mount_height, reach)), 1),
                fov_deg=round(
                    max(40.0, min(110.0, 2 * math.degrees(math.atan2(half_width, reach)))), 1
                ),
                range_m=round(math.hypot(mount_height, reach), 2),
                color="#B08D57",
            )

        # Perimeter only. Interior partitions would be an invention: nothing here
        # knows where the walls of the building are.
        self.walls = [
            [0.0, 0.0, self.width_m, 0.0, self.height_m],
            [self.width_m, 0.0, self.width_m, self.depth_m, self.height_m],
            [self.width_m, self.depth_m, 0.0, self.depth_m, self.height_m],
            [0.0, self.depth_m, 0.0, 0.0, self.height_m],
        ]
        return self.camera_mounts[camera_id]

    def remove_camera(self, camera_id: str) -> None:
        """Detach a camera and re-lay-out the remaining sectors."""
        if camera_id not in self.camera_mounts:
            return
        remaining = [cid for cid in self.camera_mounts if cid != camera_id]
        names = {cid: self.camera_mounts[cid].name for cid in remaining}
        self.camera_mounts.clear()
        self.sectors.clear()
        self.rooms.clear()
        self.walls.clear()
        for cid in remaining:
            self.add_camera(cid, name=names[cid])

    def project_camera_to_3d(
        self,
        camera_id: str,
        norm_x: float,
        norm_y: float,
    ) -> tuple[float, float, float]:
        """Normalised camera ground point to facility metres.

        A camera with no sector maps onto the whole floor. That is the only
        answer available - it is not known which part of the building it watches
        - and it is better than dropping the entity, which would take a person
        who is plainly there off the map.
        """
        sector = self.sectors.get(camera_id)
        if sector is None:
            return (
                round(self.width_m * max(0.0, min(1.0, norm_x)), 2),
                0.0,
                round(self.depth_m * max(0.0, min(1.0, norm_y)), 2),
            )
        return sector.project(norm_x, norm_y)

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
