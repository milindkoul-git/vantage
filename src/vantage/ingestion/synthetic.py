"""Deterministic procedurally-generated video source.

This exists for three concrete reasons, all of which pay for its ~150 lines:

1. **The pipeline is testable without hardware.** Buffering, pacing, drop
   accounting, threading and shutdown are all exercised in CI on a machine with
   no camera, at full speed.
2. **Bugs are visible.** Every frame carries its own index and timestamp burned
   into the pixels, plus a sweep bar that advances one step per frame. A stale
   or duplicated frame on screen is immediately obvious rather than subtly wrong.
3. **It is free ground truth for later phases.** Object positions come from a
   closed-form function of the frame index, so :meth:`SyntheticSource.object_states`
   returns exact boxes for any frame. Phase 3 built on exactly this: the
   tracker's accuracy is measured against known ground truth rather than
   eyeballed (see :mod:`vantage.tracking.scenarios`).

Determinism is total: same seed and index, same pixels, on any machine.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import ConfigError, SourceExhausted
from vantage.ingestion.base import FrameSource, SourceInfo, SourceKind


@dataclass(frozen=True, slots=True)
class SyntheticObject:
    """Exact state of one generated object at one frame - ground truth."""

    object_id: int
    cx: float
    cy: float
    radius: float
    label: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Axis-aligned ``(x1, y1, x2, y2)`` in pixels."""
        return (
            int(round(self.cx - self.radius)),
            int(round(self.cy - self.radius)),
            int(round(self.cx + self.radius)),
            int(round(self.cy + self.radius)),
        )


def _reflect(value: float, low: float, high: float) -> float:
    """Fold ``value`` into ``[low, high]`` as if bouncing off both walls.

    Closed-form so a position depends only on the frame index, never on
    accumulated simulation state.
    """
    span = high - low
    if span <= 0:
        return low
    offset = (value - low) % (2.0 * span)
    if offset > span:
        offset = 2.0 * span - offset
    return low + offset


class SyntheticSource(FrameSource):
    """Generates animated frames from a seeded, closed-form motion model."""

    def __init__(
        self,
        source_id: str = "synthetic",
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        frames: int | None = None,
        seed: int = 7,
        objects: int = 4,
        uri: str = "synthetic://",
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__(source_id=source_id, uri=uri, clock=clock)
        if width < 16 or height < 16:
            raise ConfigError(f"synthetic source needs at least 16x16, got {width}x{height}")
        if fps <= 0:
            raise ConfigError(f"synthetic source fps must be positive, got {fps}")
        if frames is not None and frames < 1:
            raise ConfigError(f"synthetic source frames must be >= 1 or null, got {frames}")
        if objects < 0:
            raise ConfigError(f"synthetic source objects must be >= 0, got {objects}")

        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._frames = frames
        self._seed = int(seed)
        self._object_count = int(objects)
        self._background: np.ndarray | None = None
        self._motion: list[tuple[float, float, float, float, float, tuple[int, int, int]]] = []

    # -- ground truth ---------------------------------------------------

    def object_states(self, index: int) -> list[SyntheticObject]:
        """Exact object positions at frame ``index``, without rendering it."""
        if not self._motion:
            self._init_motion()
        states: list[SyntheticObject] = []
        t = index / self._fps
        for i, (x0, y0, vx, vy, radius, _color) in enumerate(self._motion):
            cx = _reflect(x0 + vx * t, radius, self._width - radius)
            cy = _reflect(y0 + vy * t, radius, self._height - radius)
            states.append(
                SyntheticObject(object_id=i, cx=cx, cy=cy, radius=radius, label=f"obj{i}")
            )
        return states

    # -- FrameSource hooks ----------------------------------------------

    def _open_impl(self) -> SourceInfo:
        self._init_motion()
        self._background = self._make_background()
        return SourceInfo(
            source_id=self.source_id,
            kind=SourceKind.SYNTHETIC,
            uri=self.uri,
            width=self._width,
            height=self._height,
            declared_fps=self._fps,
            frame_count=self._frames,
            backend="synthetic",
            is_live=False,
            extra={"seed": self._seed, "objects": self._object_count},
        )

    def _read_impl(self) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        index = self.frames_produced
        if self._frames is not None and index >= self._frames:
            raise SourceExhausted(
                f"synthetic source produced all {self._frames} configured frames"
            )
        pts = index / self._fps
        image = self._render(index, pts)
        return image, pts, {"synthetic": True}

    def _close_impl(self) -> None:
        self._background = None

    # -- rendering ------------------------------------------------------

    def _init_motion(self) -> None:
        if self._motion:
            return
        rng = np.random.default_rng(self._seed)
        motion = []
        for i in range(self._object_count):
            radius = float(rng.integers(18, max(19, min(self._width, self._height) // 8)))
            x0 = float(rng.uniform(radius, self._width - radius))
            y0 = float(rng.uniform(radius, self._height - radius))
            # Pixels per second; sign randomised, magnitude kept in a range that
            # is visible frame-to-frame without aliasing at 30 fps.
            vx = float(rng.uniform(60, 260)) * (1 if rng.random() > 0.5 else -1)
            vy = float(rng.uniform(40, 200)) * (1 if rng.random() > 0.5 else -1)
            hue = (i / max(1, self._object_count) + 0.11) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
            color = (int(b * 255), int(g * 255), int(r * 255))  # BGR
            motion.append((x0, y0, vx, vy, radius, color))
        self._motion = motion

    def _make_background(self) -> np.ndarray:
        """Static gradient plus grid, built once and reused per frame."""
        ys = np.linspace(0, 1, self._height, dtype=np.float32)[:, None]
        xs = np.linspace(0, 1, self._width, dtype=np.float32)[None, :]
        base = np.empty((self._height, self._width, 3), dtype=np.uint8)
        base[..., 0] = np.clip(40 + 60 * ys + 20 * xs, 0, 255).astype(np.uint8)  # B
        base[..., 1] = np.clip(28 + 35 * ys, 0, 255).astype(np.uint8)  # G
        base[..., 2] = np.clip(22 + 18 * xs, 0, 255).astype(np.uint8)  # R

        step = max(40, min(self._width, self._height) // 12)
        grid = (60, 48, 42)
        for x in range(0, self._width, step):
            cv2.line(base, (x, 0), (x, self._height), grid, 1, cv2.LINE_4)
        for y in range(0, self._height, step):
            cv2.line(base, (0, y), (self._width, y), grid, 1, cv2.LINE_4)
        return base

    def _render(self, index: int, pts: float) -> np.ndarray:
        assert self._background is not None  # guaranteed by _open_impl
        image = self._background.copy()

        for state, (_x0, _y0, _vx, _vy, _r, color) in zip(
            self.object_states(index), self._motion, strict=False
        ):
            center = (int(round(state.cx)), int(round(state.cy)))
            radius = int(round(state.radius))
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, center, radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                image,
                state.label,
                (center[0] - radius, center[1] - radius - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Sweep bar advancing exactly one step per frame: a duplicated or stale
        # frame is visible at a glance, which a smooth animation would hide.
        sweep_x = int((index * 7) % max(1, self._width))
        cv2.line(image, (sweep_x, 0), (sweep_x, self._height), (0, 230, 255), 2, cv2.LINE_4)

        cv2.putText(
            image,
            f"SYNTHETIC  frame {index:06d}  t={pts:8.3f}s",
            (12, self._height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image
