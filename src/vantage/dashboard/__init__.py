"""The dashboard: live view and history, on the standard library.

An HTTP server, MJPEG video and one self-contained HTML file. No FastAPI, no
React, no build step - the same trade made against SciPy, shapely and psutil
earlier, for the same reason: the standard library already does this, and the
alternative brings a dependency tree for a single-camera local UI.

Serves live camera footage with **no authentication**, so it binds to loopback
unless told otherwise, and binding wider logs a warning that says what it means.
Every endpoint is read-only: pruning and configuration stay on the CLI, where
the operator is by definition the person at the machine.
"""

from vantage.dashboard.live import LiveFeed, LiveSnapshot

__all__ = ["DashboardApi", "DashboardServer", "LiveFeed", "LiveSnapshot", "build_dashboard"]


def __getattr__(name: str):
    if name == "DashboardServer":
        from vantage.dashboard.server import DashboardServer

        return DashboardServer
    if name == "DashboardApi":
        from vantage.dashboard.api import DashboardApi

        return DashboardApi
    if name == "build_dashboard":
        from vantage.dashboard.factory import build_dashboard

        return build_dashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
