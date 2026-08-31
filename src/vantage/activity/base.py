"""The recogniser seam.

A Protocol rather than a base class, matching
:class:`~vantage.tracking.base.Tracker`: an implementation has to satisfy the
shape, not inherit an ancestor. That keeps a future learned recogniser - one
that buffers keypoint sequences and runs a graph over them - free of any
inheritance from the rule-based one, whose internals it would share nothing with.

The engine owns the timing, the pruning and the assembly of results, so a
replacement implements exactly three methods and nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vantage.activity.contracts import EntityActivity
from vantage.pose.contracts import Pose
from vantage.state.contracts import EntityState


@runtime_checkable
class Recognizer(Protocol):
    """Turns a stream of per-entity observations into activities."""

    def observe(self, state: EntityState, pose: Pose | None, now: float) -> EntityActivity:
        """Record one frame for one entity and report what it is doing.

        Args:
            state: The entity's motion state this frame.
            pose: Its skeleton, or ``None`` when pose is not running or this
                entity is not a person. An implementation must degrade rather
                than fail: the activities that need pose simply do not fire.
            now: Seconds since the run started, accumulated from the tracker's
                own elapsed times rather than read from a clock, so a recorded
                source replays identically.
        """
        ...

    def forget(self, track_ids: set[int]) -> None:
        """Drop every entity not in ``track_ids``.

        Called on every step. Anything keyed by track id leaks on a camera that
        runs for weeks unless something prunes it.
        """
        ...

    def reset(self) -> None:
        """Discard all state."""
        ...


ActivityRecognizer = Recognizer
