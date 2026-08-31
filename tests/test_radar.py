"""Tests for 2D Facility Floorplan Radar & Homography Projection."""

from __future__ import annotations

from vantage.multicam.radar import FacilityRadarMap
from vantage.perception.contracts import BoundingBox


def test_radar_projection_within_zone() -> None:
    radar = FacilityRadarMap()
    box = BoundingBox(100.0, 200.0, 300.0, 400.0)

    # Project on Retail zone (10-48% X, 10-48% Y)
    fx, fy = radar.project_entity(
        camera_id="cam_01_retail",
        global_id="global_person_1",
        label="person",
        box=box,
        frame_width=640,
        frame_height=480,
        motion="moving",
        speed=0.45,
        activity="walking",
        wall_time=100.0,
    )

    # Projected coordinates must be within the zone bounds
    assert 10.0 <= fx <= 48.0
    assert 10.0 <= fy <= 48.0


def test_radar_snapshot_state() -> None:
    radar = FacilityRadarMap()
    box = BoundingBox(50.0, 100.0, 150.0, 300.0)

    radar.project_entity(
        camera_id="cam_03_corridor",
        global_id="global_person_2",
        label="person",
        box=box,
        frame_width=640,
        frame_height=480,
    )

    state = radar.get_radar_state(active_window_s=10.0)
    assert len(state["zones"]) >= 4
    assert state["active_count"] == 1
    assert state["entities"][0]["id"] == "global_person_2"
    assert len(state["entities"][0]["trail"]) >= 1
