"""Frame sinks - where annotated frames end up.

:class:`FrameSink` is a two-method protocol so that the run loop is written once
and works identically with a window, with nothing at all (headless), or with
whatever a later phase adds (an MJPEG endpoint for the Phase 9 dashboard, for
instance). Keeping the display behind an interface is what lets the exact same
code path be exercised by tests on a machine with no GUI.

Key handling deliberately stops at "which key was pressed". Deciding what a key
*means* belongs to the application, not to the thing that owns a window.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from vantage.core.errors import VantageError
from vantage.core.logging import get_logger

log = get_logger(__name__)

KEY_NONE = -1


@runtime_checkable
class FrameSink(Protocol):
    """Destination for rendered frames."""

    def show(self, image: np.ndarray) -> int:
        """Present ``image``; return a pressed key code or :data:`KEY_NONE`."""

    def is_closed(self) -> bool:
        """Whether the destination has gone away (e.g. the user closed the window)."""

    def close(self) -> None:
        """Release any resources held."""


class NullSink:
    """Discards frames. The headless path, and the one used in tests."""

    __slots__ = ("frames_shown",)

    def __init__(self) -> None:
        self.frames_shown = 0

    def show(self, image: np.ndarray) -> int:
        self.frames_shown += 1
        return KEY_NONE

    def is_closed(self) -> bool:
        return False

    def close(self) -> None:
        pass


class WindowSink:
    """An OpenCV highgui window.

    Creating the window is deferred to the first frame so that a headless
    environment fails with an actionable message at the point of display rather
    than during startup - and so that constructing the object in a test is free.
    """

    __slots__ = ("_closed", "_created", "_scale", "_wait_ms", "_window_name")

    def __init__(self, window_name: str = "Vantage", scale: float = 1.0, wait_ms: int = 1) -> None:
        self._window_name = window_name
        self._scale = scale
        self._wait_ms = max(1, wait_ms)
        self._created = False
        self._closed = False

    @property
    def window_name(self) -> str:
        return self._window_name

    def show(self, image: np.ndarray) -> int:
        if self._closed:
            return KEY_NONE
        if not self._created:
            self._create()

        display = self._resize(image)
        try:
            cv2.imshow(self._window_name, display)
            key = cv2.waitKey(self._wait_ms) & 0xFF
        except cv2.error as exc:  # pragma: no cover - GUI backend dependent
            self._closed = True
            raise VantageError(
                f"the display window failed ({exc}). Run with --no-display to "
                "operate headless."
            ) from exc

        if self._window_was_closed():
            self._closed = True
        return int(key) if key != 255 else KEY_NONE

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._created:
            try:
                cv2.destroyWindow(self._window_name)
                # highgui needs an event loop turn to actually tear the window down.
                cv2.waitKey(1)
            except cv2.error as exc:  # pragma: no cover - GUI backend dependent
                log.debug("window teardown reported: %s", exc)
        self._created = False
        self._closed = True

    # -- internals ------------------------------------------------------

    def _create(self) -> None:
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            self._closed = True
            raise VantageError(
                "this OpenCV build has no GUI support, so no window can be opened "
                f"({exc}). Run with --no-display, or install the non-headless "
                "'opencv-python' package."
            ) from exc
        self._created = True

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if self._scale == 1.0:
            return image
        height, width = image.shape[:2]
        size = (max(1, int(width * self._scale)), max(1, int(height * self._scale)))
        interpolation = cv2.INTER_AREA if self._scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, size, interpolation=interpolation)

    def _window_was_closed(self) -> bool:
        """Detect the window's X button, which produces no key event."""
        try:
            return cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:  # pragma: no cover - backend dependent
            return True
