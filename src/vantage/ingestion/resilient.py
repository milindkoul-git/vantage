"""Automatic recovery for live sources.

A USB camera that is bumped, a laptop that suspends, an RTSP stream that drops:
all of these end a naive capture loop permanently. Since the point of this
platform is unattended operation, recovery belongs in the ingestion layer.

Implemented as a decorator around another :class:`FrameSource` rather than as
logic inside :class:`~vantage.ingestion.opencv_source.OpenCVSource`, so that any
future source implementation inherits the behaviour for free and neither class
grows the other's concerns.

Note that frame indices continue across a reconnect - the wrapper's own counter
is the sequence the consumer sees, so a reconnect appears as exactly what it is:
a time gap, not a restart.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from vantage.config.schema import ReconnectConfig
from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import SourceError, SourceExhausted, SourceOpenError, SourceReadError
from vantage.core.logging import get_logger
from vantage.ingestion.base import FrameSource, SourceInfo

log = get_logger(__name__)


class ReconnectingSource(FrameSource):
    """Wraps a live source and rebuilds it on failure, with capped backoff."""

    def __init__(
        self,
        factory: Callable[[], FrameSource],
        *,
        source_id: str,
        uri: str,
        policy: ReconnectConfig,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__(source_id=source_id, uri=uri, clock=clock)
        self._factory = factory
        self._policy = policy
        self._inner: FrameSource | None = None
        self._reconnects = 0

    @property
    def reconnects(self) -> int:
        """How many times the underlying source has been rebuilt."""
        return self._reconnects

    @property
    def inner(self) -> FrameSource | None:
        """The wrapped source, for diagnostics. May be ``None`` between attempts."""
        return self._inner

    def _open_impl(self) -> SourceInfo:
        # The initial open is not retried: if the camera is absent at startup
        # that is a configuration problem, and failing immediately with a clear
        # message beats retrying for a minute first.
        self._inner = self._factory()
        return self._inner.open()

    def _read_impl(self) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        try:
            frame = self._require_inner().read()
            return frame.image, frame.media_pts, dict(frame.metadata)
        except SourceExhausted:
            raise
        except SourceError as exc:
            log.warning(
                "live source failed, attempting recovery",
                extra={
                    "vantage_fields": {
                        "source_id": self.source_id,
                        "error": str(exc),
                        "frames_before_failure": self.frames_produced,
                    }
                },
            )
            return self._recover(exc)

    def _recover(
        self, original: SourceError
    ) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        delay = self._policy.initial_delay_s
        for attempt in range(1, self._policy.max_attempts + 1):
            self._discard_inner()
            self._clock.sleep(delay)
            try:
                candidate = self._factory()
                candidate.open()
                # An open that yields no frame is not a recovery; keep retrying.
                frame = candidate.read()
            except SourceError as exc:
                log.warning(
                    "reconnect attempt failed",
                    extra={
                        "vantage_fields": {
                            "source_id": self.source_id,
                            "attempt": attempt,
                            "of": self._policy.max_attempts,
                            "next_delay_s": round(
                                min(delay * self._policy.backoff, self._policy.max_delay_s), 2
                            ),
                            "error": str(exc),
                        }
                    },
                )
                delay = min(delay * self._policy.backoff, self._policy.max_delay_s)
                continue

            self._inner = candidate
            self._reconnects += 1
            self._info = frame_info = candidate.info
            log.info(
                "source reconnected",
                extra={
                    "vantage_fields": {
                        "source_id": self.source_id,
                        "attempts": attempt,
                        "reconnects": self._reconnects,
                        "resolution": f"{frame_info.width}x{frame_info.height}",
                    }
                },
            )
            metadata = dict(frame.metadata)
            metadata["reconnected"] = self._reconnects
            return frame.image, frame.media_pts, metadata

        raise SourceReadError(
            f"source {self.uri!r} did not recover after {self._policy.max_attempts} "
            f"reconnection attempts; last error: {original}"
        ) from original

    def _close_impl(self) -> None:
        self._discard_inner()

    def _discard_inner(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    def _require_inner(self) -> FrameSource:
        if self._inner is None:  # pragma: no cover - state machine guards this
            raise SourceOpenError(f"source {self.source_id!r} is not connected")
        return self._inner
