"""The facility from directly above: where each camera looks, and who is in it.

This used to keep its own layout - six named camera footprints on a 0-100%
floorplan, keyed to the camera ids of a demo recording, produced whatever
cameras were actually connected. So a two-camera deployment was shown a
four-sector building with names like ``NORTH_CORRIDOR_A`` in it, and the twin
beside it disagreed, because it kept its own fabricated layout too.

There is one layout now, and it lives in
:class:`~vantage.spatial.twin.FacilitySpatialTwin`. The radar reads it, so the
overhead view and the 3D view cannot disagree, and both describe the cameras
that are running. Coordinates are metres on the facility ground plane rather
than percentages, for the same reason: two views quoting different units for the
same point is how they drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vantage.perception.contracts import BoundingBox
from vantage.spatial.twin import FacilitySpatialTwin


@dataclass(frozen=True, slots=True)
class CameraFootprint:
    """The part of the floor one camera watches, in metres."""

    camera_id: str
    name: str
    #: ``(x_min, z_min, x_max, z_max)`` on the ground plane.
    rect: tuple[float, float, float, float]
    #: Where the camera itself is, for drawing a line to its own footprint.
    origin: tuple[float, float]


@dataclass
class RadarEntityDot:
    """One entity on the overhead view."""

    global_id: str
    camera_id: str
    label: str
    x: float
    y: float
    motion: str
    speed: float
    activity: str
    last_updated: float


TRAIL_SECONDS = 20.0
"""How much of a path to keep. Long enough to read a direction of travel off,
short enough that a busy floor does not become a scribble."""


class FacilityRadarMap:
    """Projects multi-camera detections onto one overhead floor plan."""

    def __init__(self, twin: FacilitySpatialTwin | None = None) -> None:
        # Its own empty twin when none is shared, so a radar built alone behaves
        # the same way: no cameras, no footprints, nothing invented.
        self.twin = twin if twin is not None else FacilitySpatialTwin()
        self._active_dots: dict[str, RadarEntityDot] = {}
        self._trail_history: dict[str, list[tuple[float, float, float]]] = {}

    @property
    def footprints(self) -> dict[str, CameraFootprint]:
        """One per camera the twin knows about, derived from its sectors."""
        result: dict[str, CameraFootprint] = {}
        for camera_id, mount in self.twin.camera_mounts.items():
            sector = self.twin.sectors.get(camera_id)
            if sector is None:
                continue
            result[camera_id] = CameraFootprint(
                camera_id=camera_id,
                name=mount.name,
                rect=(sector.x_min, sector.z_min, sector.x_max, sector.z_max),
                origin=(mount.x, mount.z),
            )
        return result

    def register_camera(self, camera_id: str, name: str | None = None) -> None:
        """Give a camera a footprint, by giving it a sector in the twin."""
        self.twin.add_camera(camera_id, name=name)

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
        """Place an entity's foot point on the floor plan, in metres.

        The foot point rather than the box centre: a person's centre moves when
        they crouch or raise their arms, and their feet do not.
        """
        now = wall_time if wall_time is not None else time.time()

        foot_x, foot_y = box.bottom_center
        norm_u = max(0.0, min(1.0, foot_x / max(1, frame_width)))
        norm_v = max(0.0, min(1.0, foot_y / max(1, frame_height)))

        floor_x, _, floor_z = self.twin.project_camera_to_3d(camera_id, norm_u, norm_v)

        self._active_dots[global_id] = RadarEntityDot(
            global_id=global_id,
            camera_id=camera_id,
            label=label,
            x=floor_x,
            y=floor_z,
            motion=motion,
            speed=speed,
            activity=activity,
            last_updated=now,
        )

        trails = self._trail_history.setdefault(global_id, [])
        trails.append((floor_x, floor_z, now))
        self._trail_history[global_id] = [p for p in trails if now - p[2] <= TRAIL_SECONDS]

        return floor_x, floor_z

    def get_radar_state(
        self, active_window_s: float = 5.0, now: float | None = None
    ) -> dict[str, Any]:
        """The current overhead snapshot.

        Entities that have not been seen inside ``active_window_s`` are dropped
        rather than left in place: a dot that stops updating but keeps being
        drawn is a person the view claims is still standing there.

        ``now`` is injectable so that staleness can be tested against the same
        timeline the entities were recorded on, rather than against the wall
        clock the test is running under.
        """
        now = now if now is not None else time.time()
        live_dots = []
        for global_id, dot in list(self._active_dots.items()):
            if now - dot.last_updated > active_window_s:
                self._active_dots.pop(global_id, None)
                continue
            trail = [(p[0], p[1]) for p in self._trail_history.get(global_id, [])]
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
                    "trail": trail[-15:],
                }
            )

        return {
            "timestamp": now,
            "zones": [
                {
                    "camera_id": footprint.camera_id,
                    "name": footprint.name,
                    "rect": list(footprint.rect),
                    "origin": list(footprint.origin),
                }
                for footprint in self.footprints.values()
            ],
            "entities": live_dots,
            "active_count": len(live_dots),
        }
