"""Analytics over stored history: aggregation, baselines and anomaly detection.

Phase 11. Every stage before this one answers questions about the present frame
or the present entity. This one answers questions about accumulated history -
what usually happens here, and whether the last day looked like it.

Nothing here runs in the live pipeline. It reads the Phase 8 store, which means
it can be pointed at a database while the camera is running, or at one copied
off the machine entirely.
"""

from vantage.analytics.contracts import (
    AnalysisResult,
    Anomaly,
    Baseline,
    Bucket,
    Direction,
    Metric,
    Series,
    Slot,
)
from vantage.analytics.engine import AnalyticsEngine, AnalyticsParams

__all__ = [
    "AnalysisResult",
    "AnalyticsEngine",
    "AnalyticsParams",
    "Anomaly",
    "Baseline",
    "Bucket",
    "Direction",
    "Metric",
    "Series",
    "Slot",
]
