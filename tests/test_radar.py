"""The overhead floor plan.

These used to assert that a projection landed inside a hard-coded zone rect for
a camera called ``cam_01_retail`` - a footprint the radar shipped with, and
produced whatever cameras were connected. What is checked now is that the radar
and the twin describe the same floor, that a camera's entities land in that
camera's own footprint, and that a dot stops being drawn when it stops updating.
"""

from __future__ import annotations

from vantage.multicam.radar import FacilityRadarMap
from vantage.perception.contracts import BoundingBox
from vantage.spatial.twin import FacilitySpatialTwin


def dot_for(
    radar: FacilityRadarMap, camera_id: str, global_id: str, box: BoundingBox, **kwargs
):
    return radar.project_entity(
        camera_id=camera_id,
        global_id=global_id,
        label="person",
        box=box,
        frame_width=640,
        frame_height=480,
        **kwargs,
    )


class TestFootprints:
    def test_an_unconfigured_radar_has_none(self) -> None:
        """It must not draw a building before it has been told about a camera."""
        assert FacilityRadarMap().get_radar_state()["zones"] == []

    def test_one_per_camera_the_twin_knows(self) -> None:
        radar = FacilityRadarMap(FacilitySpatialTwin(["entrance", "yard"]))
        zones = radar.get_radar_state()["zones"]
        assert {zone["camera_id"] for zone in zones} == {"entrance", "yard"}

    def test_the_radar_and_the_twin_agree_on_the_floor(self) -> None:
        """One layout, read by both, so the two views cannot drift apart."""
        twin = FacilitySpatialTwin(["a", "b", "c"])
        radar = FacilityRadarMap(twin)
        for zone in radar.get_radar_state()["zones"]:
            room = next(r for r in twin.rooms if r["id"] == f"sector_{zone['camera_id']}")
            assert list(zone["rect"]) == room["bounds"]

    def test_registering_a_camera_gives_it_a_footprint(self) -> None:
        radar = FacilityRadarMap()
        radar.register_camera("gate", name="Loading Gate")
        zones = radar.get_radar_state()["zones"]
        assert [zone["name"] for zone in zones] == ["Loading Gate"]


class TestProjection:
    def test_an_entity_lands_in_its_own_camera_footprint(self) -> None:
        radar = FacilityRadarMap(FacilitySpatialTwin(["entrance", "yard"]))
        x, y = dot_for(radar, "yard", "person_1", BoundingBox(100.0, 200.0, 300.0, 400.0))
        rect = next(
            zone["rect"]
            for zone in radar.get_radar_state()["zones"]
            if zone["camera_id"] == "yard"
        )
        assert rect[0] <= x <= rect[2]
        assert rect[1] <= y <= rect[3]

    def test_the_foot_point_is_used_not_the_centre(self) -> None:
        """A person's centre moves when they crouch; their feet do not."""
        radar = FacilityRadarMap(FacilitySpatialTwin(["a"]))
        standing = dot_for(radar, "a", "p", BoundingBox(100.0, 100.0, 200.0, 400.0))
        crouched = dot_for(radar, "a", "p", BoundingBox(100.0, 250.0, 200.0, 400.0))
        assert standing == crouched

    def test_a_trail_accumulates_and_is_reported(self) -> None:
        radar = FacilityRadarMap(FacilitySpatialTwin(["a"]))
        for step in range(3):
            dot_for(
                radar,
                "a",
                "p",
                BoundingBox(100.0 + step * 20, 200.0, 300.0 + step * 20, 400.0),
                wall_time=100.0 + step,
            )
        entity = radar.get_radar_state(now=102.0)["entities"][0]
        assert entity["id"] == "p"
        assert len(entity["trail"]) == 3


class TestStaleness:
    def test_a_dot_that_stops_updating_stops_being_drawn(self) -> None:
        """Otherwise the view claims someone is still standing there."""
        radar = FacilityRadarMap(FacilitySpatialTwin(["a"]))
        dot_for(radar, "a", "p", BoundingBox(0.0, 0.0, 10.0, 10.0), wall_time=1.0)
        state = radar.get_radar_state(active_window_s=0.5, now=5.0)
        assert state["entities"] == []
        assert state["active_count"] == 0
