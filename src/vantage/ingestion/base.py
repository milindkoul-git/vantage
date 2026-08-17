"""The source abstraction.

:class:`FrameSource` is the seam that makes the platform replaceable at its
most hardware-coupled point. Everything above it - pacing, buffering, the
viewer, and every future perception stage - is written against this interface
and has no way to discover whether frames came from a webcam, a file, an RTSP
stream or a procedural generator.

Subclasses implement three small hooks (:meth:`_open_impl`, :meth:`_read_impl`,
:meth:`_close_impl`) and inherit the parts that must not be reimplemented
inconsistently: lifecycle state, frame indexing, dual timestamping, and the
read-only pixel contract. That is why the template-method shape is used here
rather than a bare protocol - correct frame indexing is a platform invariant,
not a per-source choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import SourceExhausted, SourceStateError
from vantage.core.frame import Frame
from vantage.core.logging import get_logger

log = get_logger(__name__)


class SourceKind(str, Enum):
    """Broad category of a source, used to pick sensible defaults."""

    CAMERA = "camera"
    FILE = "file"
    STREAM = "stream"
    SYNTHETIC = "synthetic"


class SourceState(str, Enum):
    """Lifecycle position. Enforced so misuse fails loudly and immediately."""

    CREATED = "created"
    OPEN = "open"
    EXHAUSTED = "exhausted"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What a source turned out to be, once opened.

    Requested settings are negotiated with the driver; this records what was
    actually granted. Consumers must trust these values over anything they asked
    for - a camera asked for 1920x1080 will happily deliver 640x480 instead.
    """

    source_id: str
    kind: SourceKind
    uri: str
    width: int
    height: int
    declared_fps: float | None = None
    """Frame rate reported by the driver or container. ``None`` when unknown -
    many webcam backends do not report one, so no consumer may depend on it."""

    frame_count: int | None = None
    """Total frames for finite sources; ``None`` when unbounded or unknown."""

    backend: str = "unknown"
    is_live: bool = False
    """Live sources cannot be rewound and have no meaningful media timeline.
    This drives the default backpressure policy."""

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def duration_s(self) -> float | None:
        if self.frame_count and self.declared_fps:
            return self.frame_count / self.declared_fps
        return None

    def describe(self) -> str:
        fps = f"{self.declared_fps:.2f}fps" if self.declared_fps else "fps unknown"
        frames = f"{self.frame_count} frames" if self.frame_count else "unbounded"
        return (
            f"{self.source_id} [{self.kind.value}] {self.width}x{self.height} {fps} "
            f"({frames}, backend={self.backend}, {'live' if self.is_live else 'recorded'})"
        )


class FrameSource(ABC):
    """Abstract producer of :class:`~vantage.core.frame.Frame` objects.

    Usable as a context manager::

        with create_source(config) as source:
            frame = source.read()

    Threading: a source instance is *not* thread-safe. The pipeline confines
    each source to a single capture thread, which is the only supported use.
    """

    def __init__(self, source_id: str, uri: str, clock: Clock = SYSTEM_CLOCK) -> None:
        self._source_id = source_id
        self._uri = uri
        self._clock = clock
        self._state = SourceState.CREATED
        self._index = 0
        self._info: SourceInfo | None = None

    # -- identity -------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def state(self) -> SourceState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state is SourceState.OPEN

    @property
    def frames_produced(self) -> int:
        """Frames successfully read since opening. Also the next frame's index."""
        return self._index

    @property
    def info(self) -> SourceInfo:
        """Negotiated source properties. Only valid after :meth:`open`."""
        if self._info is None:
            raise SourceStateError(
                f"source {self._source_id!r} has not been opened; call open() first"
            )
        return self._info

    # -- lifecycle ------------------------------------------------------

    def open(self) -> SourceInfo:
        """Acquire the underlying resource and validate it.

        Raises:
            SourceOpenError: the source could not be opened or produced no
                usable geometry. The message names the URI and the backend tried.
        """
        if self._state is SourceState.OPEN:
            return self.info
        if self._state is not SourceState.CREATED:
            raise SourceStateError(
                f"source {self._source_id!r} cannot be reopened from state {self._state.value}"
            )
        try:
            info = self._open_impl()
        except Exception:
            self._state = SourceState.FAILED
            raise
        self._info = info
        self._state = SourceState.OPEN
        log.info(
            "source opened",
            extra={
                "vantage_fields": {
                    "source_id": info.source_id,
                    "kind": info.kind.value,
                    "resolution": f"{info.width}x{info.height}",
                    "fps": info.declared_fps,
                    "backend": info.backend,
                    "live": info.is_live,
                }
            },
        )
        return info

    def read(self) -> Frame:
        """Acquire the next frame.

        Raises:
            SourceExhausted: the source ended normally (file EOF, synthetic
                frame budget reached). Callers treat this as loop termination.
            SourceReadError: acquisition failed in a way that is not recoverable
                by retrying.
            SourceStateError: the source is not open.
        """
        if self._state is not SourceState.OPEN:
            raise SourceStateError(
                f"source {self._source_id!r} is {self._state.value}, not open; "
                "call open() before read()"
            )
        try:
            image, media_pts, metadata = self._read_impl()
        except SourceExhausted:
            # Normal termination, not a fault: the source must not be left FAILED.
            self._state = SourceState.EXHAUSTED
            raise
        except Exception:
            self._state = SourceState.FAILED
            raise

        frame = Frame(
            image=image,
            index=self._index,
            source_id=self._source_id,
            capture_monotonic=self._clock.monotonic(),
            capture_wall=self._clock.wall(),
            media_pts=media_pts,
            metadata=metadata or {},
        )
        self._index += 1
        return frame

    def close(self) -> None:
        """Release the underlying resource. Idempotent and never raises."""
        if self._state is SourceState.CLOSED:
            return
        try:
            self._close_impl()
        except Exception as exc:  # pragma: no cover - driver dependent
            # Reported, not swallowed: a failed release is diagnostic
            # information, but it must not mask the caller's own exception
            # during unwinding.
            log.warning(
                "error while closing source",
                extra={"vantage_fields": {"source_id": self._source_id, "error": str(exc)}},
            )
        finally:
            self._state = SourceState.CLOSED
            log.debug(
                "source closed",
                extra={
                    "vantage_fields": {
                        "source_id": self._source_id,
                        "frames_produced": self._index,
                    }
                },
            )

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} id={self._source_id!r} uri={self._uri!r} "
            f"state={self._state.value} frames={self._index}>"
        )

    # -- subclass hooks -------------------------------------------------

    @abstractmethod
    def _open_impl(self) -> SourceInfo:
        """Acquire the resource and return its negotiated properties."""

    @abstractmethod
    def _read_impl(self) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        """Return ``(bgr_image, media_pts_seconds, metadata)`` for the next frame."""

    @abstractmethod
    def _close_impl(self) -> None:
        """Release the resource. Called at most once."""
