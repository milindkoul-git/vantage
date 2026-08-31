"""Facility Journey Timeline & Cross-Camera Movement Tracker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vantage.perception.contracts import BoundingBox


@dataclass
class JourneyLeg:
    """A continuous sighting of an entity within one camera's field of view."""

    camera_id: str
    start_time: float
    end_time: float
    primary_activity: str
    primary_posture: str
    first_box: BoundingBox
    last_box: BoundingBox
    frame_count: int = 1

    @property
    def duration_s(self) -> float:
        return max(0.0, round(self.end_time - self.start_time, 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "primary_activity": self.primary_activity,
            "primary_posture": self.primary_posture,
            "frame_count": self.frame_count,
        }


@dataclass
class FacilityJourney:
    """The complete cross-camera journey of a global entity across the facility."""

    global_id: str
    label: str
    first_seen: float
    last_seen: float
    legs: list[JourneyLeg] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return max(0.0, round(self.last_seen - self.first_seen, 2))

    @property
    def cameras_traversed(self) -> list[str]:
        return [leg.camera_id for leg in self.legs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "label": self.label,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_duration_s": self.total_duration_s,
            "cameras_traversed": self.cameras_traversed,
            "legs": [leg.to_dict() for leg in self.legs],
        }


class FacilityJourneyTracker:
    """Maintains and records continuous cross-camera facility journeys."""

    def __init__(self) -> None:
        self._journeys: dict[str, FacilityJourney] = {}

    def record_sighting(
        self,
        global_id: str,
        label: str,
        camera_id: str,
        wall_time: float,
        box: BoundingBox,
        activity: str = "walking",
        posture: str = "standing",
    ) -> None:
        """Record a single frame observation of an entity."""
        if global_id not in self._journeys:
            self._journeys[global_id] = FacilityJourney(
                global_id=global_id,
                label=label,
                first_seen=wall_time,
                last_seen=wall_time,
                legs=[],
            )

        journey = self._journeys[global_id]
        journey.last_seen = wall_time

        if not journey.legs or journey.legs[-1].camera_id != camera_id:
            # New leg on a different camera!
            journey.legs.append(
                JourneyLeg(
                    camera_id=camera_id,
                    start_time=wall_time,
                    end_time=wall_time,
                    primary_activity=activity,
                    primary_posture=posture,
                    first_box=box,
                    last_box=box,
                    frame_count=1,
                )
            )
        else:
            # Update current leg
            leg = journey.legs[-1]
            leg.end_time = wall_time
            leg.last_box = box
            leg.frame_count += 1
            if activity != "idle":
                leg.primary_activity = activity
            if posture != "unknown":
                leg.primary_posture = posture

    def get_journey(self, global_id: str) -> FacilityJourney | None:
        return self._journeys.get(global_id)

    def get_active_journeys(self, active_window_s: float = 60.0) -> list[FacilityJourney]:
        now = time.time()
        return [j for j in self._journeys.values() if now - j.last_seen <= active_window_s]

    def get_all_journeys_dict(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in self._journeys.values()]
