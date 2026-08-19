"""Entity state estimation: motion, dwell and path length from tracks.

The "object state" half of Phase 4. Applies to every tracked entity regardless
of class, needs no model and no weights, and costs microseconds - it reads the
motion estimate the tracker already maintains rather than computing a new one.
"""

from vantage.state.contracts import EntityState, MotionState, StateResult
from vantage.state.estimator import StateEstimator, StateParams

__all__ = [
    "EntityState",
    "MotionState",
    "StateEstimator",
    "StateParams",
    "StateResult",
]
