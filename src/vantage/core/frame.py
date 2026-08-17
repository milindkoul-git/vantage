"""The frame contract - the single data structure crossing every stage boundary.

This is the most important type in the platform. Detection, tracking, pose and
event stages will all consume :class:`Frame` and none of them will know, or be
able to discover, how the pixels were acquired.

Design decisions worth understanding:

Indexing
    :attr:`Frame.index` counts frames *produced by the source*, not frames
    *delivered to the consumer*. When the pipeline drops frames under
    backpressure the consumer sees a gap (``41 -> 44``) and therefore knows
    precisely what it missed. A tracker in Phase 3 must reason about elapsed
    time between observations rather than assume uniform spacing, so this
    distinction is load-bearing rather than cosmetic.

Immutability
    The pixel buffer is handed out read-only. Frames are shared by reference
    across stages for performance; a stage that quietly annotated a buffer
    another stage still held would produce bugs that are extremely hard to
    find. Stages that need to draw call :meth:`editable_copy`.

Colour space
    ``image`` is always HxWx3 ``uint8`` in **BGR** order, matching OpenCV. The
    convention is stated once, here, so that no downstream stage has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """A single acquired video frame plus its provenance."""

    image: np.ndarray
    """HxWx3 ``uint8`` BGR pixel buffer, flagged read-only."""

    index: int
    """0-based ordinal of this frame within the source's output sequence."""

    source_id: str
    """Identifier of the producing source; unique per camera in multi-camera setups."""

    capture_monotonic: float
    """Monotonic timestamp taken immediately after acquisition. For latency math."""

    capture_wall: float
    """Unix epoch seconds at acquisition. For storage and cross-system correlation."""

    media_pts: float | None = None
    """Presentation timestamp within the media, in seconds.

    Present for files and synthetic sources (where a frame has an intrinsic
    position on a timeline); ``None`` for live cameras, where "now" is the only
    meaningful timestamp.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific extras. Not a dumping ground for stage results - later
    phases attach their output to observation records, not to the frame."""

    def __post_init__(self) -> None:
        image = self.image
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Frame.image must be a numpy array, got {type(image).__name__}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Frame.image must have shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Frame.image must be uint8, got {image.dtype}")
        if self.index < 0:
            raise ValueError(f"Frame.index must be non-negative, got {self.index}")
        # Enforce the shared-immutable contract at construction so violations
        # surface at the offending write rather than as corrupted output later.
        if image.flags.writeable:
            image.flags.writeable = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.width, self.height

    @property
    def nbytes(self) -> int:
        return int(self.image.nbytes)

    def age_ms(self, now_monotonic: float) -> float:
        """Milliseconds elapsed since acquisition, measured on the monotonic base."""
        return (now_monotonic - self.capture_monotonic) * 1000.0

    def editable_copy(self) -> np.ndarray:
        """Return a writeable copy of the pixels, for annotation or transformation."""
        return self.image.copy()

    def describe(self) -> str:
        """Compact single-line description, for logs and diagnostics."""
        pts = "live" if self.media_pts is None else f"{self.media_pts:.3f}s"
        return f"{self.source_id}#{self.index} {self.width}x{self.height} pts={pts}"
