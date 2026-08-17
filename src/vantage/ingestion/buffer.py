"""The bounded hand-off between capture and consumption.

This is the single most consequential component of Phase 1, because of a
hardware fact established during environment discovery: this machine has no
CUDA GPU and 12 CPU cores shared with everything else. Phase 2 inference *will*
be slower than 30 fps capture. What happens at that moment is an architectural
choice, not an accident, and it is made here.

Three policies, each correct for a different source:

``LATEST``
    Evict the oldest queued frame to admit the newest. A live camera analysed
    two seconds late is worse than useless - it reports the past as the present.
    Latency stays bounded no matter how slow the consumer becomes.

``BLOCK``
    Stall the producer. A file has no real-time obligation, every frame matters,
    and reproducible results demand that the same input always produce the same
    output.

``DROP_NEW``
    Reject the arriving frame. Rarely right, but correct when the oldest frames
    are the reference (a triggered burst, a pre-event ring buffer).

Every discarded frame is counted. Silent frame loss would make downstream FPS
numbers a lie, so the drop counter is surfaced in stats and on the HUD.
"""

from __future__ import annotations

import threading
from collections import deque

from vantage.config.schema import Backpressure
from vantage.core.frame import Frame


class FrameBuffer:
    """Thread-safe bounded frame queue with a configurable overflow policy.

    Single-producer / single-consumer is the supported pattern; the lock makes
    it safe rather than fast, which is the right trade at video frame rates
    (tens of operations per second, not millions).
    """

    __slots__ = (
        "_capacity",
        "_closed",
        "_dropped",
        "_items",
        "_not_empty",
        "_not_full",
        "_lock",
        "_policy",
        "_high_water",
    )

    def __init__(self, capacity: int, policy: Backpressure) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if policy is Backpressure.AUTO:
            raise ValueError(
                "Backpressure.AUTO must be resolved to a concrete policy before "
                "constructing a FrameBuffer (see resolve_backpressure)"
            )
        self._capacity = capacity
        self._policy = policy
        self._items: deque[Frame] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._dropped = 0
        self._high_water = 0
        self._closed = False

    # -- introspection --------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def policy(self) -> Backpressure:
        return self._policy

    @property
    def dropped(self) -> int:
        """Frames discarded by the overflow policy since construction."""
        return self._dropped

    @property
    def high_water(self) -> int:
        """Deepest the queue has ever been - shows how close to saturation a run ran."""
        return self._high_water

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    # -- producer side --------------------------------------------------

    def put(self, frame: Frame, timeout: float | None = None) -> bool:
        """Offer a frame to the consumer.

        Returns:
            ``True`` if the frame was queued, ``False`` if it was dropped by the
            policy or the buffer is closed. Under :attr:`Backpressure.BLOCK` this
            waits for room (up to ``timeout``) rather than dropping.
        """
        with self._not_full:
            if self._closed:
                return False

            if len(self._items) >= self._capacity:
                if self._policy is Backpressure.DROP_NEW:
                    self._dropped += 1
                    return False
                if self._policy is Backpressure.LATEST:
                    self._items.popleft()
                    self._dropped += 1
                else:  # BLOCK
                    deadline_passed = not self._not_full.wait_for(
                        lambda: self._closed or len(self._items) < self._capacity,
                        timeout=timeout,
                    )
                    if deadline_passed or self._closed:
                        return False

            self._items.append(frame)
            if len(self._items) > self._high_water:
                self._high_water = len(self._items)
            self._not_empty.notify()
            return True

    # -- consumer side --------------------------------------------------

    def get(self, timeout: float | None = None) -> Frame | None:
        """Take the oldest queued frame.

        Returns ``None`` when the buffer is closed and drained, or when
        ``timeout`` elapses with nothing available. A ``None`` from a closed and
        empty buffer is the consumer's termination signal.
        """
        with self._not_empty:
            if not self._items:
                got = self._not_empty.wait_for(
                    lambda: bool(self._items) or self._closed, timeout=timeout
                )
                if not got or not self._items:
                    return None
            frame = self._items.popleft()
            self._not_full.notify()
            return frame

    def close(self) -> None:
        """Stop accepting frames and wake every waiter. Idempotent."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def clear(self) -> int:
        """Discard queued frames; returns how many. Used on shutdown to free RAM."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
            self._not_full.notify_all()
            return count


def resolve_backpressure(policy: Backpressure, is_live: bool) -> Backpressure:
    """Turn :attr:`Backpressure.AUTO` into the right concrete policy.

    Live sources cannot be slowed down - the world keeps moving whether or not
    we read it - so their queue drops. Files can be slowed down and should be,
    so nothing is lost.
    """
    if policy is not Backpressure.AUTO:
        return policy
    return Backpressure.LATEST if is_live else Backpressure.BLOCK
