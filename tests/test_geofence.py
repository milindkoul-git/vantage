"""Comprehensive Unit & Concurrency Tests for Geofence Polygon Engine & ZoneRegistry."""

from __future__ import annotations

import pytest

from vantage.events.geofence import GeofenceEngine, PolygonZone, ZoneType
from vantage.events.zone_registry import ActiveZoneSnapshot, ZoneRegistry
from vantage.perception.contracts import BoundingBox
from vantage.spatial.geometry.coordinates import get_entity_foot_point
from vantage.spatial.geometry.polygon import Polygon
from vantage.state.contracts import EntityState, MotionState
from vantage.storage.sqlite_store import SqliteStore

# -----------------------------------------------------------------------------
# 1. GEOMETRY TESTS
# -----------------------------------------------------------------------------


def test_polygon_convex_inclusion() -> None:
    # Square [0.2, 0.2] to [0.8, 0.8]
    poly = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])

    assert poly.contains_point((0.5, 0.5)) is True
    assert poly.contains_point((0.1, 0.5)) is False
    assert poly.contains_point((0.9, 0.5)) is False
    assert poly.contains_point((0.5, 0.1)) is False
    assert poly.contains_point((0.5, 0.9)) is False


def test_polygon_concave_inclusion() -> None:
    # L-shaped polygon
    poly = Polygon(
        [
            (0.0, 0.0),
            (0.6, 0.0),
            (0.6, 0.3),
            (0.3, 0.3),
            (0.3, 0.6),
            (0.0, 0.6),
        ]
    )
    assert poly.contains_point((0.1, 0.1)) is True
    assert poly.contains_point((0.5, 0.1)) is True
    assert poly.contains_point((0.1, 0.5)) is True
    # The cutout corner
    assert poly.contains_point((0.5, 0.5)) is False


def test_polygon_boundary_deterministic_policy() -> None:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    # On vertex
    assert poly.contains_point((0.0, 0.0)) is True
    assert poly.contains_point((1.0, 1.0)) is True

    # On edge
    assert poly.contains_point((0.5, 0.0)) is True
    assert poly.contains_point((1.0, 0.5)) is True
    assert poly.contains_point((0.5, 1.0)) is True
    assert poly.contains_point((0.0, 0.5)) is True


def test_polygon_degenerate_and_validation() -> None:
    with pytest.raises(ValueError):
        Polygon([(0.1, 0.1), (0.2, 0.2)])  # Less than 3 points

    # Collinear points (0 area)
    poly_flat = Polygon([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    valid, _msg = poly_flat.is_valid()
    assert valid is False

    # Self-intersecting polygon (hourglass shape)
    poly_bowtie = Polygon([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
    valid, _msg = poly_bowtie.is_valid()
    assert valid is False


def test_entity_foot_point_calculation() -> None:
    box = BoundingBox(x1=100.0, y1=50.0, x2=140.0, y2=150.0)
    # Frame 1000x500
    # Foot point is (100 + 140) / 2 / 1000 = 0.12, 150 / 500 = 0.30
    foot = get_entity_foot_point(box, frame_width=1000, frame_height=500, normalize=True)
    assert pytest.approx(foot.x, abs=1e-4) == 0.12
    assert pytest.approx(foot.y, abs=1e-4) == 0.30


# -----------------------------------------------------------------------------
# 2. EXCLUSION RULE TESTS
# -----------------------------------------------------------------------------


def _make_state(
    entity_id: str,
    foot_x: float,
    foot_y: float,
    track_id: int = 1,
    speed: float = 0.3,
    bearing: float | None = 0.0,
) -> tuple[EntityState, BoundingBox]:
    # 100x100 normalized frame
    box = BoundingBox(
        x1=foot_x * 100 - 5,
        y1=foot_y * 100 - 20,
        x2=foot_x * 100 + 5,
        y2=foot_y * 100,
    )
    motion = MotionState.MOVING if speed > 0.1 else MotionState.STATIONARY
    st = EntityState(
        track_id=track_id,
        entity_id=entity_id,
        label="person",
        motion=motion,
        speed=speed,
        dwell_s=10.0,
        bearing_deg=bearing,
        distance=1.0,
        age_s=10.0,
        observed=True,
    )
    return st, box


def test_exclusion_zone_breach_and_cooldown() -> None:
    poly = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
    zone = PolygonZone(
        zone_id="vault",
        name="Server Vault",
        camera_id="cam_01",
        polygon=poly,
        zone_type=ZoneType.EXCLUSION,
        rule_config={"cooldown_s": 5.0},
        severity="alert",
    )
    snapshot = ActiveZoneSnapshot({"vault": zone})
    engine = GeofenceEngine()

    # Entity outside
    outside_st, b_out = _make_state("person_1", 0.1, 0.1, track_id=1)
    breaches, occ = engine.evaluate_snapshot(
        snapshot, "cam_01", [outside_st], 100, 100, wall_time=100.0, boxes={1: b_out}
    )
    assert len(breaches) == 0
    assert occ.get("vault") == 0

    # Entity inside -> Breach
    inside_st, b_in = _make_state("person_1", 0.5, 0.5, track_id=1)
    breaches, occ = engine.evaluate_snapshot(
        snapshot, "cam_01", [inside_st], 100, 100, wall_time=101.0, boxes={1: b_in}
    )
    assert len(breaches) == 1
    assert breaches[0].rule == "geofence_exclusion"
    assert breaches[0].entity_id == "person_1"
    assert breaches[0].zone_id == "vault"
    assert occ.get("vault") == 1

    # Immediate next frame -> Cooldown active, no spam
    breaches, occ = engine.evaluate_snapshot(
        snapshot, "cam_01", [inside_st], 100, 100, wall_time=102.0, boxes={1: b_in}
    )
    assert len(breaches) == 0
    assert occ.get("vault") == 1

    # After cooldown expires -> Breach fires again
    breaches, occ = engine.evaluate_snapshot(
        snapshot, "cam_01", [inside_st], 100, 100, wall_time=107.0, boxes={1: b_in}
    )
    assert len(breaches) == 1


# -----------------------------------------------------------------------------
# 3. OCCUPANCY CAPACITY RULE TESTS
# -----------------------------------------------------------------------------


def test_occupancy_limit_rule() -> None:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    zone = PolygonZone(
        zone_id="lobby_occ",
        name="Lobby Capacity",
        camera_id="cam_01",
        polygon=poly,
        zone_type=ZoneType.OCCUPANCY,
        rule_config={"max_occupancy": 2, "cooldown_s": 10.0},
        severity="notice",
    )
    snapshot = ActiveZoneSnapshot({"lobby_occ": zone})
    engine = GeofenceEngine()

    p1, b1 = _make_state("p1", 0.2, 0.2, track_id=1)
    p2, b2 = _make_state("p2", 0.4, 0.4, track_id=2)
    p3, b3 = _make_state("p3", 0.6, 0.6, track_id=3)
    boxes = {1: b1, 2: b2, 3: b3}

    # 2 people inside (limit is 2) -> OK
    breaches, occ = engine.evaluate_snapshot(
        snapshot, "cam_01", [p1, p2], 100, 100, wall_time=50.0, boxes=boxes
    )
    assert len(breaches) == 0
    assert occ.get("lobby_occ") == 2

    # 3 people inside -> Overcrowding Breach
    breaches, occ = engine.evaluate_snapshot(
        snapshot, "cam_01", [p1, p2, p3], 100, 100, wall_time=51.0, boxes=boxes
    )
    assert len(breaches) == 1
    assert breaches[0].rule == "geofence_occupancy_limit"
    assert breaches[0].evidence["current_occupancy"] == 3
    assert occ.get("lobby_occ") == 3


# -----------------------------------------------------------------------------
# 4. DWELL & LOITERING TESTS
# -----------------------------------------------------------------------------


def test_dwell_timer_and_tracker_loss() -> None:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    zone = PolygonZone(
        zone_id="till_dwell",
        name="Checkout Queue",
        camera_id="cam_01",
        polygon=poly,
        zone_type=ZoneType.DWELL,
        rule_config={"max_dwell_s": 5.0, "cooldown_s": 20.0},
    )
    snapshot = ActiveZoneSnapshot({"till_dwell": zone})
    engine = GeofenceEngine()

    p1, b1 = _make_state("p1", 0.3, 0.3, track_id=1)

    # Entry at t=10.0
    breaches, _ = engine.evaluate_snapshot(
        snapshot, "cam_01", [p1], 100, 100, wall_time=10.0, boxes={1: b1}
    )
    assert len(breaches) == 0

    # At t=12.0 (2s dwell, under 5s limit)
    breaches, _ = engine.evaluate_snapshot(
        snapshot, "cam_01", [p1], 100, 100, wall_time=12.0, boxes={1: b1}
    )
    assert len(breaches) == 0

    # Brief tracker loss at t=13.0 (empty state)
    breaches, _ = engine.evaluate_snapshot(
        snapshot, "cam_01", [], 100, 100, wall_time=13.0, boxes={}
    )

    # Re-observed at t=16.0 (total elapsed 6s >= 5s limit) -> Dwell breach fires!
    breaches, _ = engine.evaluate_snapshot(
        snapshot, "cam_01", [p1], 100, 100, wall_time=16.0, boxes={1: b1}
    )
    assert len(breaches) == 1
    assert breaches[0].rule == "geofence_dwell_violation"


# -----------------------------------------------------------------------------
# 5. DIRECTIONAL FLOW TESTS
# -----------------------------------------------------------------------------


def test_directional_flow_rule() -> None:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    zone = PolygonZone(
        zone_id="corridor_dir",
        name="One-Way Hall",
        camera_id="cam_01",
        polygon=poly,
        zone_type=ZoneType.DIRECTIONAL,
        rule_config={
            "allowed_bearing_deg": 0.0,
            "angular_tolerance_deg": 45.0,
            "min_speed": 0.2,
            "cooldown_s": 10.0,
        },
    )
    snapshot = ActiveZoneSnapshot({"corridor_dir": zone})
    engine = GeofenceEngine()

    # Person walking North (bearing 10°, within 0±45°) -> Allowed
    st_ok, b_ok = _make_state("p1", 0.5, 0.5, track_id=1, speed=0.5, bearing=10.0)
    breaches, _ = engine.evaluate_snapshot(
        snapshot, "cam_01", [st_ok], 100, 100, wall_time=10.0, boxes={1: b_ok}
    )
    assert len(breaches) == 0

    # Person walking South (bearing 180°, reverse) -> Violation!
    st_bad, b_bad = _make_state("p1", 0.5, 0.5, track_id=1, speed=0.5, bearing=180.0)
    breaches, _ = engine.evaluate_snapshot(
        snapshot, "cam_01", [st_bad], 100, 100, wall_time=11.0, boxes={1: b_bad}
    )
    assert len(breaches) == 1
    assert breaches[0].rule == "geofence_direction_violation"


# -----------------------------------------------------------------------------
# 6. PERSISTENCE & CONCURRENCY TESTS
# -----------------------------------------------------------------------------


def test_zone_persistence_and_atomic_snapshot(tmp_path) -> None:
    db_file = tmp_path / "test_zones.db"
    store = SqliteStore(db_file)

    registry = ZoneRegistry(store=store)
    assert registry.get_snapshot().count() == 0

    # Save zone
    poly = Polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])
    zone = PolygonZone(
        zone_id="z_test_1",
        name="Test Exclusion",
        camera_id="cam_01",
        polygon=poly,
        zone_type=ZoneType.EXCLUSION,
        rule_config={"cooldown_s": 15.0},
    )

    snap1 = registry.save_zone(zone)
    assert snap1.count() == 1
    assert snap1.version >= 2
    assert snap1.get_zone("z_test_1") is not None
    assert registry.last_update_latency_ms >= 0.0

    # Verify reload from storage
    new_registry = ZoneRegistry(store=store)
    snap_reloaded = new_registry.get_snapshot()
    assert snap_reloaded.count() == 1
    assert snap_reloaded.get_zone("z_test_1").name == "Test Exclusion"

    # Delete zone
    snap2 = registry.delete_zone("z_test_1")
    assert snap2.count() == 0
    assert snap2.version > snap1.version

    # Confirm deletion in store
    assert len(store.list_zones()) == 0
