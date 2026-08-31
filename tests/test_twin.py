"""Unit and Integration Tests for 3D Spatial Digital Twin Engine."""

from __future__ import annotations

import time

from vantage.events.geofence import PolygonZone, ZoneType
from vantage.events.zone_registry import ZoneRegistry
from vantage.spatial.geometry.coordinates import Point2D
from vantage.spatial.geometry.polygon import Polygon
from vantage.spatial.twin import FacilitySpatialTwin


def test_spatial_twin_facility_bounds() -> None:
    twin = FacilitySpatialTwin()
    assert twin.width_m == 40.0
    assert twin.depth_m == 24.0
    assert twin.height_m == 4.5
    assert len(twin.camera_mounts) == 4
    assert len(twin.rooms) == 4
    assert len(twin.walls) >= 6


def test_camera_frustums_3d_orientation() -> None:
    twin = FacilitySpatialTwin()
    cam1 = twin.camera_mounts["cam_01_retail"]
    assert (
        cam1.position == [2.0, 3.4, 2.0]
        if hasattr(cam1, "position")
        else (cam1.x == 2.0 and cam1.y == 3.4 and cam1.z == 2.0)
    )
    assert cam1.fov_deg == 72.0
    assert cam1.pitch_deg < 0  # Downward tilt
    assert cam1.range_m > 10.0


def test_camera_to_3d_projection() -> None:
    twin = FacilitySpatialTwin()

    # Cam 1 Retail Showroom (top-left sector)
    wx, wy, wz = twin.project_camera_to_3d("cam_01_retail", 0.0, 0.0)
    assert 0.0 <= wx <= 20.0
    assert wy == 0.0
    assert 0.0 <= wz <= 12.0

    # Cam 4 Pedestrian Entry Doorway (bottom-right sector)
    wx4, wy4, wz4 = twin.project_camera_to_3d("cam_04_doorway", 0.5, 0.5)
    assert 20.0 <= wx4 <= 40.0
    assert wy4 == 0.0
    assert 12.0 <= wz4 <= 24.0


def test_entity_3d_tracking_and_breadcrumbs() -> None:
    twin = FacilitySpatialTwin()
    now = time.time()

    # Frame 1: Entity moving East (bearing 90°)
    twin.update_entity_3d(
        global_id="person_01",
        camera_id="cam_01_retail",
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
    ent = state["entities"][0]
    assert ent["entity_id"] == "person_01"
    assert ent["velocity"][0] > 0  # Velocity along +X (East)
    assert "person_01" in state["trails"]
    assert len(state["trails"]["person_01"]) == 1

    # Frame 2: Entity updates position
    twin.update_entity_3d(
        global_id="person_01",
        camera_id="cam_01_retail",
        label="person",
        foot_point=Point2D(0.3, 0.2),
        speed=0.5,
        bearing_deg=90.0,
        motion="walking",
        posture="standing",
        wall_time=now + 0.1,
    )
    state2 = twin.get_digital_twin_state()
    assert len(state2["trails"]["person_01"]) == 2


def test_3d_geofence_extrusion_from_registry() -> None:
    poly = Polygon([(0.1, 0.1), (0.8, 0.1), (0.8, 0.8), (0.1, 0.8)])
    zone = PolygonZone(
        zone_id="zone_sec_vault",
        name="Security Vault",
        camera_id="cam_01_retail",
        polygon=poly,
        zone_type=ZoneType.EXCLUSION,
        severity="alert",
        color="#ff3b30",
    )
    registry = ZoneRegistry()
    registry.save_zone(zone)

    twin = FacilitySpatialTwin(zone_registry=registry)
    state = twin.get_digital_twin_state(live_occupancies={"zone_sec_vault": 3})

    assert len(state["zones"]) == 1
    z3d = state["zones"][0]
    assert z3d["zone_id"] == "zone_sec_vault"
    assert z3d["height_m"] == 2.8
    assert z3d["occupancy"] == 3
    assert len(z3d["polygon_3d"]) == 4
    # Ensure vertices are in metric 3D space
    for pt in z3d["polygon_3d"]:
        assert len(pt) == 2
        assert 0.0 <= pt[0] <= 40.0
        assert 0.0 <= pt[1] <= 24.0
