"""Time abstraction.

Timing is pervasive in a video pipeline (pacing, FPS, latency), and code that
calls :func:`time.perf_counter` directly is untestable without sleeping. Every
component that needs time takes a :class:`Clock`, so tests can drive time
deterministically with :class:`ManualClock`.

Two distinct time bases are exposed on purpose:

``monotonic``
    Never jumps; the only valid basis for measuring durations and latency.
``wall``
    Unix epoch seconds; the only valid basis for correlating with the outside
    world (log lines, stored observations, other cameras).

Frames carry both, because Phase 8 storage needs wall-clock timestamps while
Phase 12 performance work needs monotonic ones.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Source of time and the ability to wait."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin, guaranteed non-decreasing."""

    def wall(self) -> float:
        """Seconds since the Unix epoch."""

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``; a non-positive value returns immediately."""


class SystemClock:
    """Real time, backed by the highest-resolution counters available."""

    __slots__ = ()

    def monotonic(self) -> float:
        return time.perf_counter()

    def wall(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock:
    """Deterministic clock for tests.

    ``sleep`` advances the clock instead of blocking, so pacing logic can be
    exercised at full speed while still being verified exactly.
    """

    __slots__ = ("_mono", "_wall", "slept")

    def __init__(self, start_monotonic: float = 0.0, start_wall: float = 1_700_000_000.0) -> None:
        self._mono = float(start_monotonic)
        self._wall = float(start_wall)
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self._mono

    def wall(self) -> float:
        return self._wall

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.slept.append(seconds)
            self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Move both time bases forward by ``seconds``."""
        if seconds < 0:
            raise ValueError("cannot move a monotonic clock backwards")
        self._mono += seconds
        self._wall += seconds


SYSTEM_CLOCK = SystemClock()
"""Shared default instance; :class:`SystemClock` is stateless."""
