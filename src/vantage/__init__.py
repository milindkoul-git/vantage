"""Vantage - a modular intelligent video analytics platform.

Ten subsystems, each attached to the one before it without modifying it:
ingestion, detection, tracking, pose and object state, temporal activity,
spatial and interaction reasoning, an event engine, observation storage, a
local dashboard, and optional identity resolution.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from vantage.core.frame import Frame

try:
    __version__ = _installed_version("vantage")
except PackageNotFoundError:
    # Running from a source tree that was never installed - a git clone whose
    # dependencies were installed by hand, or a PyInstaller bundle where the
    # dist-info is not collected. Hard-coding the number here instead was worse
    # in exactly the way that matters: it silently disagreed with the wheel's
    # own metadata, so `vantage --version` reported 0.1.0 from a 0.10.0 build
    # and would have sent someone debugging the wrong release.
    __version__ = "0.0.0+unknown"

__all__ = ["Frame", "__version__"]
