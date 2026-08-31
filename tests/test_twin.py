"""The 3D facility twin: sector layout, projection, entity trails and zones.

These used to assert the twin's built-in demo building - four sectors called
"Retail Showroom Floor", "Plaza & Crosswalk" and so on, keyed to the camera ids
of a demo recording - which meant they passed on a deployment that had none of
those cameras and was being shown a facility that did not exist. What is checked
now is that the layout follows the cameras it was given, and that an unconfigured
twin is empty rather than furnished.
"""

from __future__ import annotations

import time

import pytest

from vantage.events.geofence import PolygonZone, ZoneType
from vantage.events.zone_registry import ZoneRegistry
from vantage.spatial.geometry.coordinates import Point2D
from vantage.spatial.geometry.polygon import Polygon
from vantage.spatial.projection import ConfiguredSectorProjection
from vantage.spatial.twin import FacilitySpatialTwin


class TestTheLayoutFollowsTheCameras:
    def test_a_twin_with_no_cameras_is_empty(self) -> None:
        """An unconfigured twin must not furnish a building nobody has.

        This is the assertion the previous version could not make: it shipped
        four rooms, eight interior walls and four camera mounts before it was
        told anything at all.
        """
        twin = FacilitySpatialTwin()
        assert twin.rooms == []
        assert twin.walls == []
        assert twin.camera_mounts == {}
        assert twin.get_digital_twin_state()["facility"]["rooms"] == []

    def test_each_camera_gets_a_sector_and_a_mount(self) -> None:
        twin = FacilitySpatialTwin(["entrance", "yard", "lobby"])
        assert set(twin.camera_mounts) == {"entrance", "yard", "lobby"}
        assert [room["id"] for room in twin.rooms] == [
            "sector_entrance",
            "sector_yard",
            "sector_lobby",
        ]
        # Perimeter only. Interior partitions would be invented.
        assert len(twin.walls) == 4

    def test_sectors_do_not_overlap(self) -> None:
        """Two cameras sharing floor would put one person in two places."""
        twin = FacilitySpatialTwin([f"cam{i}" for i in range(6)])
        boxes = [room["bounds"] for room in twin.rooms]
        for i, (ax0, az0, ax1, az1) in enumerate(boxes):
            for bx0, bz0, bx1, bz1 in boxes[i + 1 :]:
                separated = ax1 <= bx0 or bx1 <= ax0 or az1 <= bz0 or bz1 <= az0
                assert separated, "sectors overlap"

    def test_every_sector_is_inside_the_facility(self) -> None:
        twin = FacilitySpatialTwin(["a", "b", "c", "d", "e"])
        for x0, z0, x1, z1 in (room["bounds"] for room in twin.rooms):
            assert 0.0 <= x0 < x1 <= twin.width_m
            assert 0.0 <= z0 < z1 <= twin.depth_m

    def test_a_camera_can_be_added_and_removed(self) -> None:
        twin = FacilitySpatialTwin(["a", "b"])
        twin.add_camera("c", name="Loading Bay")
        assert twin.camera_mounts["c"].name == "Loading Bay"
        assert len(twin.rooms) == 3

        twin.remove_camera("b")
        assert set(twin.camera_mounts) == {"a", "c"}
        assert [room["id"] for room in twin.rooms] == ["sector_a", "sector_c"]

    def test_adding_a_camera_twice_is_not_an_error(self) -> None:
        twin = FacilitySpatialTwin(["a"])
        first = twin.add_camera("a")
        assert twin.add_camera("a") is first
        assert len(twin.rooms) == 1

    def test_mounts_look_down_from_inside_the_building(self) -> None:
        twin = FacilitySpatialTwin(["a", "b"])
        for mount in twin.camera_mounts.values():
            assert 0 < mount.y < twin.height_m
            assert mount.pitch_deg < 0
            assert mount.range_m > 0

    def test_facility_dimensions_are_configurable(self) -> None:
        twin = FacilitySpatialTwin(["a"], width_m=12.0, depth_m=8.0, height_m=3.0)
        assert twin.rooms[0]["bounds"][2] <= 12.0
        assert twin.rooms[0]["bounds"][3] <= 8.0

    def test_a_facility_with_no_extent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FacilitySpatialTwin(["a"], width_m=0.0)


class TestProjection:
    def test_a_point_lands_in_its_own_camera_sector(self) -> None:
        twin = FacilitySpatialTwin(["entrance", "yard"])
        x0, z0, x1, z1 = next(
            room["bounds"] for room in twin.rooms if room["id"] == "sector_yard"
        )
        wx, wy, wz = twin.project_camera_to_3d("yard", 0.5, 0.5)
        assert x0 <= wx <= x1
        assert z0 <= wz <= z1
        assert wy == 0.0

    def test_two_cameras_map_the_same_frame_point_to_different_places(self) -> None:
        """The whole purpose of the sector assignment."""
        twin = FacilitySpatialTwin(["entrance", "yard"])
        assert twin.project_camera_to_3d("entrance", 0.5, 0.5) != twin.project_camera_to_3d(
            "yard", 0.5, 0.5
        )

    def test_an_unknown_camera_maps_onto_the_whole_floor(self) -> None:
        """Not dropped: an entity that is plainly there stays on the map."""
        twin = FacilitySpatialTwin(["entrance"])
        wx, _, wz = twin.project_camera_to_3d("unconfigured", 1.0, 1.0)
        assert (wx, wz) == (twin.width_m, twin.depth_m)

    def test_a_point_outside_the_frame_is_clamped_not_extrapolated(self) -> None:
        twin = FacilitySpatialTwin(["a"])
        inside = twin.project_camera_to_3d("a", 1.0, 1.0)
        assert twin.project_camera_to_3d("a", 1.4, 1.4) == inside

    def test_inverted_sector_bounds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="inverted or empty"):
            ConfiguredSectorProjection(x_min=5.0, x_max=1.0, z_min=0.0, z_max=1.0)


class TestEntities:
    def test_position_velocity_and_trail(self) -> None:
        twin = FacilitySpatialTwin(["cam_a"])
        now = time.time()

        twin.update_entity_3d(
            global_id="person_01",
            camera_id="cam_a",
            label="person",
            foot_point=Point2D(0.2, 0.2),
            speed=0.5,
            bearing_deg=90.0,
            motion="walking",
            posture="standing",
            wall_time=now,
        )

        state = twin.get_digital_twin_state()
        assert len(state["entities"]) == 1
        entity = state["entities"][0]
        assert entity["entity_id"] == "person_01"
        assert entity["velocity"][0] > 0  # bearing 90 is +X
        assert len(state["trails"]["person_01"]) == 1

        twin.update_entity_3d(
            global_id="person_01",
            camera_id="cam_a",
            label="person",
            foot_point=Point2D(0.3, 0.2),
            speed=0.5,
            bearing_deg=90.0,
            motion="walking",
            posture="standing",
            wall_time=now + 0.1,
        )
        assert len(twin.get_digital_twin_state()["trails"]["person_01"]) == 2

    def test_a_stationary_entity_has_no_velocity(self) -> None:
        """A bearing with no speed behind it is not a direction of travel."""
        twin = FacilitySpatialTwin(["cam_a"])
        twin.update_entity_3d(
            global_id="person_02",
            camera_id="cam_a",
            label="person",
            foot_point=Point2D(0.5, 0.5),
            speed=0.0,
            bearing_deg=None,
            motion="stationary",
            posture="standing",
            wall_time=time.time(),
        )
        entity = twin.get_digital_twin_state()["entities"][0]
        assert entity["velocity"] == [0.0, 0.0, 0.0]


class TestZones:
    def test_a_zone_is_extruded_into_its_camera_sector(self) -> None:
        polygon = Polygon([(0.1, 0.1), (0.8, 0.1), (0.8, 0.8), (0.1, 0.8)])
        zone = PolygonZone(
            zone_id="zone_sec_vault",
            name="Security Vault",
            camera_id="cam_a",
            polygon=polygon,
            zone_type=ZoneType.EXCLUSION,
            severity="alert",
            color="#B33A2E",
        )
        registry = ZoneRegistry()
        registry.save_zone(zone)

        twin = FacilitySpatialTwin(["cam_a", "cam_b"], zone_registry=registry)
        state = twin.get_digital_twin_state(live_occupancies={"zone_sec_vault": 3})

        assert len(state["zones"]) == 1
        extruded = state["zones"][0]
        assert extruded["zone_id"] == "zone_sec_vault"
        assert extruded["occupancy"] == 3
        assert len(extruded["polygon_3d"]) == 4

        x0, z0, x1, z1 = next(
            room["bounds"] for room in twin.rooms if room["id"] == "sector_cam_a"
        )
        for x, z in extruded["polygon_3d"]:
            assert x0 <= x <= x1, "the zone left its camera's sector"
            assert z0 <= z <= z1
