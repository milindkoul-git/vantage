"""Unit Tests for /api/scene REST API Read Models."""

from __future__ import annotations

from vantage.dashboard.api import DashboardApi
from vantage.multicam.pipeline import MultiCameraPipeline


def test_scene_api_endpoints() -> None:
    pipeline = MultiCameraPipeline(
        camera_sources={"cam_01": "vid_01_retail_walkway.mp4"},
        model="yolox-nano",
        enable_pose=False,
    )
    api = DashboardApi(pipeline=pipeline)

    # 1. Query scene API
    res = api.handle("scene", {})
    assert res["available"] is True
    assert "cameras" in res

    # 2. Query scene API for specific camera
    res_cam = api.handle("scene", {"camera_id": "cam_01"})
    assert res_cam["available"] is True

    pipeline.stop()
