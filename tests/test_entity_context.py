"""Unit and Integration Tests for Canonical Entity Intelligence Layer."""

from __future__ import annotations

import time

from vantage.entity.context import EntityContext
from vantage.entity.contracts import (
    EntitySnapshot,
    IdentityLevel,
)
from vantage.entity.manager import EntityContextManager
from vantage.perception.contracts import BoundingBox


def test_entity_context_lifecycle_and_identity_hierarchy() -> None:
    now = time.time()
    box = BoundingBox(10.0, 20.0, 50.0, 120.0)

    # 1. Initial Local Track creation
    ctx = EntityContext(
        global_id="global_person_1",
        label="person",
        initial_camera="cam_01",
        initial_track_id=17,
        initial_box=box,
        wall_time=now,
    )
    assert ctx.identity_level == IdentityLevel.LOCAL_TRACK
    assert ctx.current_camera == "cam_01"
    assert ctx.named_identity is None

    # 2. Camera Transition -> Global Associated
    box2 = BoundingBox(15.0, 25.0, 55.0, 125.0)
    ctx.update_spatial(
        camera_id="cam_02",
        box=box2,
        foot_point=(0.35, 0.90),
        wall_time=now + 5.0,
        world_position=(12.5, 0.0, 8.4),
    )
    assert ctx.identity_level == IdentityLevel.GLOBAL_ASSOCIATED
    assert ctx.current_camera == "cam_02"
    assert "cam_01" in ctx.recent_cameras
    assert "cam_02" in ctx.recent_cameras
    assert ctx.world_position == (12.5, 0.0, 8.4)

    # 3. Biometric / External Identity Attachment -> Named Confirmed
    ctx.attach_identity(
        name="Alice",
        similarity=0.92,
        margin=0.45,
        source="face_yunet_sface",
        wall_time=now + 6.0,
    )
    assert ctx.identity_level == IdentityLevel.NAMED_CONFIRMED
    assert ctx.named_identity == "Alice"
    assert ctx.identity_similarity == 0.92

    # 4. Kinematics and Activity Updates
    ctx.update_kinematics(
        speed_h_s=0.45, motion_state="walking", posture="standing", bearing_deg=90.0
    )
    ctx.update_activity(
        activities=["walking"],
        primary="walking",
        confidence=0.95,
        evidence="Speed 0.45 h/s",
        wall_time=now + 7.0,
    )
    ctx.update_zones(zones={"corridor_west", "retail_entrance"}, relations=["near_counter"])

    # 5. Event Attachment
    ev_data = {
        "id": "ev_101",
        "rule": "zone_entry",
        "severity": "info",
        "summary": "Entered retail_entrance",
        "timestamp": now + 7.0,
    }
    ctx.add_event(ev_data)

    # 6. Immutable Snapshot Generation
    snapshot = ctx.to_snapshot()
    assert isinstance(snapshot, EntitySnapshot)
    assert snapshot.global_id == "global_person_1"
    assert snapshot.identity.level == IdentityLevel.NAMED_CONFIRMED
    assert snapshot.identity.name == "Alice"
    assert snapshot.spatial.camera_id == "cam_02"
    assert snapshot.kinematics.motion_state == "walking"
    assert "corridor_west" in snapshot.zones.current_zones
    assert len(snapshot.events.active_events) == 1

    # Verify JSON Serialization
    snap_dict = snapshot.to_dict()
    assert snap_dict["global_id"] == "global_person_1"
    assert snap_dict["identity"]["name"] == "Alice"
    assert snap_dict["identity"]["is_named"] is True
    assert snap_dict["spatial"]["world_position"] == [12.5, 0.0, 8.4]


def test_entity_context_manager_multi_camera() -> None:
    manager = EntityContextManager(prune_timeout_s=10.0)
    now = time.time()
    box = BoundingBox(0.0, 0.0, 20.0, 60.0)

    # 1. Register entity on cam_01
    ctx1 = manager.get_or_create(
        global_id="global_person_1",
        label="person",
        camera_id="cam_01",
        track_id=1,
        box=box,
        wall_time=now,
    )
    assert ctx1.global_id == "global_person_1"

    # 2. Lookup by local track key
    found = manager.get_by_local("cam_01", 1)
    assert found is not None
    assert found.global_id == "global_person_1"

    # 3. Associate same global entity on cam_02 with local track 5
    ctx2 = manager.get_or_create(
        global_id="global_person_1",
        label="person",
        camera_id="cam_02",
        track_id=5,
        box=box,
        wall_time=now + 2.0,
    )
    assert ctx2 is ctx1  # Exact same entity context

    # 4. Lookup from second camera
    found2 = manager.get_by_local("cam_02", 5)
    assert found2 is ctx1

    # 5. Snapshots query
    snapshots = manager.get_active_snapshots(active_within_s=30.0)
    assert len(snapshots) == 1
    assert snapshots[0].global_id == "global_person_1"

    # 6. Prune stale entities
    pruned = manager.prune_stale(now=now + 15.0)
    assert pruned == 1
    assert len(manager.get_active_snapshots()) == 0
