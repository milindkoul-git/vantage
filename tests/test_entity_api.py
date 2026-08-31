"""Unit and Integration Tests for Canonical Entity Dashboard REST APIs."""

from __future__ import annotations

import time

from vantage.dashboard.api import DashboardApi
from vantage.multicam.pipeline import MultiCameraPipeline
from vantage.perception.contracts import BoundingBox


def test_entities_api_endpoints() -> None:
    pipeline = MultiCameraPipeline(
        camera_sources={"cam_01": "vid_01_retail_walkway.mp4"},
        model="yolox-nano",
        enable_pose=False,
    )
    api = DashboardApi(pipeline=pipeline)

    # 1. Initially no entities active
    res_empty = api.handle("entities", {})
    assert res_empty["available"] is True
    assert res_empty["count"] == 0

    # 2. Add an entity context to pipeline's entity manager
    now = time.time()
    ctx = pipeline.entity_manager.get_or_create(
        global_id="global_person_1",
        label="person",
        camera_id="cam_01",
        track_id=1,
        box=BoundingBox(10.0, 10.0, 50.0, 120.0),
        wall_time=now,
    )
    ctx.attach_identity(name="Bob", similarity=0.91)
    ctx.update_kinematics(speed_h_s=0.5, motion_state="walking")

    # 3. Query all active entities
    res_all = api.handle("entities", {})
    assert res_all["available"] is True
    assert res_all["count"] == 1
    ent_data = res_all["entities"][0]
    assert ent_data["global_id"] == "global_person_1"
    assert ent_data["identity"]["name"] == "Bob"
    assert ent_data["kinematics"]["motion_state"] == "walking"

    # 4. Query specific entity by ID
    res_single = api.handle("entities", {"id": "global_person_1"})
    assert res_single["available"] is True
    assert res_single["found"] is True
    assert res_single["entity"]["global_id"] == "global_person_1"

    # 5. Query non-existent entity
    res_missing = api.handle("entities", {"id": "non_existent"})
    assert res_missing["available"] is True
    assert res_missing["found"] is False

    pipeline.stop()
