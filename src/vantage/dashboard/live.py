"""The live feed: the seam between a running pipeline and any number of viewers.

Holds **only the latest frame**, never a queue. That is the same decision Phase 1
made for live capture, for the same reason: a viewer that falls behind should see
what is happening now, not work through a backlog of what already happened. A
queue here would also give a slow browser the power to grow memory in the
analysis process, which is exactly the wrong direction for that influence to run.

Threading
---------
The pipeline writes from the analysis thread; the HTTP server reads from one
thread per connected viewer. A lock around a single slot is enough - the
critical section is a reference assignment - and the JPEG encode happens on the
*writer's* side once per frame rather than once per viewer, so ten browsers cost
what one does.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    """What the dashboard shows about the present moment."""

    frame_index: int
    captured_at: float
    entities: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "captured_at": self.captured_at,
            "age_s": round(max(0.0, time.time() - self.captured_at), 2),
            "entities": list(self.entities),
            "events": list(self.events),
            "stats": self.stats,
            "health": self.health,
        }


class LiveFeed:
    """One slot for the newest annotated frame, plus the newest snapshot."""

    def __init__(self, jpeg_quality: int = 70, max_width: int = 960) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in 1..100")
        self._quality = int(jpeg_quality)
        self._max_width = int(max_width)
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._snapshot = LiveSnapshot(frame_index=-1, captured_at=0.0)
        # Lets a viewer block until something new exists rather than polling,
        # which is the difference between an idle dashboard costing nothing and
        # costing a core.
        self._updated = threading.Condition(self._lock)
        self._sequence = 0
        self._viewers = 0

    @property
    def viewers(self) -> int:
        return self._viewers

    @property
    def has_frame(self) -> bool:
        return self._jpeg is not None

    def publish(self, image: np.ndarray, snapshot: LiveSnapshot) -> None:
        """Called from the analysis thread, once per displayed frame.

        Encoding happens here rather than per viewer: JPEG is the expensive part
        and it does not depend on who is watching.
        """
        encoded = self._encode(image)
        with self._lock:
            if encoded is not None:
                self._jpeg = encoded
            self._snapshot = snapshot
            self._sequence += 1
            self._updated.notify_all()

    def publish_snapshot(self, snapshot: LiveSnapshot) -> None:
        """Update the state without a new frame - for headless runs."""
        with self._lock:
            self._snapshot = snapshot
            self._sequence += 1
            self._updated.notify_all()

    def _encode(self, image: np.ndarray) -> bytes | None:
        if image is None or image.size == 0:
            return None
        if self._max_width and image.shape[1] > self._max_width:
            scale = self._max_width / image.shape[1]
            image = cv2.resize(
                image,
                (self._max_width, max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        return buffer.tobytes() if ok else None

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return self._snapshot

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def wait_for_frame(
        self, last_sequence: int, timeout: float = 5.0
    ) -> tuple[int, bytes | None]:
        """Block until a frame newer than ``last_sequence`` exists.

        Returns the current sequence and JPEG. On timeout it returns what it
        has, so a viewer connected to a stalled pipeline gets a stale frame
        rather than a dropped connection - a frozen picture is a clearer signal
        that something is wrong than a broken image icon.
        """
        with self._lock:
            # Waits when there is nothing newer *or* nothing at all. The second
            # half matters: a viewer that connects before the first frame has
            # sequence 0 against a last_sequence of -1, so "nothing newer" is
            # false and the call returned (0, None) immediately. The server
            # loops on a None frame, so that viewer spun the CPU at full tilt
            # until the pipeline produced its first image.
            if self._sequence <= last_sequence or self._jpeg is None:
                self._updated.wait(timeout)
            return self._sequence, self._jpeg

    def viewer_opened(self) -> None:
        with self._lock:
            self._viewers += 1

    def viewer_closed(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)
