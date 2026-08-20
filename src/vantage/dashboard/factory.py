"""Assembling a dashboard from configuration."""

from __future__ import annotations

from vantage.core.logging import get_logger
from vantage.dashboard.api import DashboardApi
from vantage.dashboard.live import LiveFeed
from vantage.dashboard.server import DashboardServer

log = get_logger(__name__)


def build_dashboard(
    config,
    *,
    store=None,
    feed: LiveFeed | None = None,
    camera_id: str = "camera_01",
) -> DashboardServer:
    """Construct a server from a DashboardConfig.

    ``store`` and ``feed`` are both optional and independently so: a dashboard
    with only a store is a history browser, one with only a feed is a live
    viewer, and one with neither is an honest empty page that says why. Each
    endpoint reports its own availability rather than returning empty data a
    viewer cannot distinguish from a quiet scene.
    """
    return DashboardServer(
        DashboardApi(store=store, feed=feed, camera_id=camera_id),
        feed,
        host=config.host,
        port=config.port,
    )
