"""OpenCV-backed source for cameras, media files and network streams.

Chosen as the Phase 1 workhorse because it is the only option that covers all
three input classes through one API, ships its own FFmpeg build (verified: no
system FFmpeg is installed on this machine, and file decoding works anyway),
and exposes the platform-native capture backends. PyAV would give better
timestamp fidelity for RTSP and is a strong candidate to add *alongside* this
in a later phase - which is exactly why it sits behind
:class:`~vantage.ingestion.base.FrameSource` rather than being called directly.

Two behaviours here are deliberate and worth knowing about:

*Open means "frames actually arrive".* ``VideoCapture.isOpened()`` returns True
for devices that will never deliver a single frame. This source therefore reads
a probe frame during :meth:`open`, uses its shape as the authoritative
resolution (drivers routinely misreport ``CAP_PROP_FRAME_WIDTH``), and hands
that same frame back as frame 0 rather than discarding it.

*A failed read is interpreted by source kind.* For a file it means EOF, which
is normal termination. For a camera it means a transient glitch worth retrying
before declaring the device dead.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import SourceExhausted, SourceOpenError, SourceReadError
from vantage.core.logging import get_logger
from vantage.ingestion.base import FrameSource, SourceInfo, SourceKind

log = get_logger(__name__)

BACKENDS: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "ffmpeg": cv2.CAP_FFMPEG,
    "gstreamer": cv2.CAP_GSTREAMER,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
}


def resolve_backend(name: str, kind: SourceKind) -> tuple[int, str]:
    """Map a configured backend name to an OpenCV API id.

    ``auto`` picks per platform and source kind. On Windows that means Media
    Foundation for cameras: measured on this machine at ~30 fps versus ~15 fps
    for DirectShow on the same device, and unlike DirectShow it reports a usable
    ``CAP_PROP_FPS``.
    """
    key = (name or "auto").strip().lower()
    if key == "auto":
        if kind is SourceKind.CAMERA:
            if sys.platform == "win32":
                return cv2.CAP_MSMF, "msmf"
            if sys.platform == "darwin":
                return BACKENDS["avfoundation"], "avfoundation"
            return cv2.CAP_V4L2, "v4l2"
        return cv2.CAP_FFMPEG, "ffmpeg"
    if key not in BACKENDS:
        raise SourceOpenError(
            f"unknown capture backend {name!r}; valid options are "
            f"{sorted(BACKENDS)} or 'auto'"
        )
    return BACKENDS[key], key


class OpenCVSource(FrameSource):
    """Reads frames through ``cv2.VideoCapture``."""

    def __init__(
        self,
        target: int | str,
        *,
        source_id: str,
        kind: SourceKind,
        uri: str,
        backend: str = "auto",
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = None,
        loop: bool = False,
        read_retries: int = 3,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__(source_id=source_id, uri=uri, clock=clock)
        self._target = target
        self._kind = kind
        self._backend_name = backend
        self._requested = (width, height, fps)
        self._fourcc = fourcc
        self._loop = loop
        self._read_retries = max(0, read_retries)

        self._cap: cv2.VideoCapture | None = None
        self._pending: np.ndarray | None = None
        self._pending_pts: float | None = None
        self._loops = 0

    @property
    def kind(self) -> SourceKind:
        return self._kind

    # -- open -----------------------------------------------------------

    def _open_impl(self) -> SourceInfo:
        api, backend_name = resolve_backend(self._backend_name, self._kind)

        if self._kind is SourceKind.FILE:
            path = Path(str(self._target))
            if not path.exists():
                raise SourceOpenError(
                    f"video file not found: {path}. Check the path in source.uri, "
                    "or generate a test clip with 'vantage make-sample'."
                )
            if path.is_dir():
                raise SourceOpenError(f"source.uri points at a directory, not a file: {path}")

        started = time.perf_counter()
        cap = cv2.VideoCapture(self._target, api)
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(self._open_failure_message(backend_name))

        self._cap = cap
        self._apply_requested_properties(cap)

        # Validate by acquisition, not by flag. The probe frame is retained and
        # delivered as frame 0 so validation costs nothing.
        pts = self._read_position_seconds(cap)
        ok, image = cap.read()
        if not ok or image is None:
            cap.release()
            self._cap = None
            raise SourceOpenError(
                f"{self.uri!r} opened via {backend_name} but delivered no frames. "
                "The device may be in use by another application, blocked by the OS "
                "camera privacy setting, or the file may use an unsupported codec."
            )
        self._pending = image
        self._pending_pts = pts
        open_ms = (time.perf_counter() - started) * 1000.0

        height, width = image.shape[:2]
        self._warn_if_blank(image)
        declared_fps = self._sanitise_fps(cap.get(cv2.CAP_PROP_FPS))
        frame_count = self._sanitise_count(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._warn_on_negotiation_mismatch(width, height, declared_fps)

        return SourceInfo(
            source_id=self.source_id,
            kind=self._kind,
            uri=self.uri,
            width=int(width),
            height=int(height),
            declared_fps=declared_fps,
            frame_count=frame_count,
            backend=backend_name,
            is_live=self._kind in (SourceKind.CAMERA, SourceKind.STREAM),
            extra={
                "open_ms": round(open_ms, 1),
                "fourcc": _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                "loop": self._loop,
            },
        )

    def _apply_requested_properties(self, cap: cv2.VideoCapture) -> None:
        width, height, fps = self._requested
        # FOURCC first: several UVC drivers renegotiate the available
        # resolution/rate table when the pixel format changes.
        if self._fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self._fourcc))
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        if self._kind in (SourceKind.CAMERA, SourceKind.STREAM):
            # Shallowest possible driver queue: for live analysis a stale frame
            # is worth less than no frame. Not all backends honour this.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            # Respect rotation metadata so phone-recorded clips are not sideways.
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    def _warn_if_blank(self, image: np.ndarray) -> None:
        """Flag a source that opens and streams but carries no picture.

        A closed laptop privacy shutter is indistinguishable from a healthy
        camera by every property OpenCV reports: correct resolution, correct
        frame rate, successful reads, all-zero pixels. This is a warning rather
        than an error because a legitimately dark scene looks the same, and
        refusing to start would be worse than saying so.
        """
        if not image.any():
            log.warning(
                "source is delivering blank frames",
                extra={
                    "vantage_fields": {
                        "source_id": self.source_id,
                        "uri": self.uri,
                        "hint": "check for a closed privacy shutter, a lens cap, "
                        "or a completely dark scene",
                    }
                },
            )

    def _warn_on_negotiation_mismatch(
        self, width: int, height: int, fps: float | None
    ) -> None:
        want_w, want_h, want_fps = self._requested
        if want_w and want_h and (want_w != width or want_h != height):
            log.warning(
                "driver refused requested resolution",
                extra={
                    "vantage_fields": {
                        "source_id": self.source_id,
                        "requested": f"{want_w}x{want_h}",
                        "granted": f"{width}x{height}",
                    }
                },
            )
        if want_fps and fps and abs(want_fps - fps) > 0.5:
            log.warning(
                "driver refused requested frame rate",
                extra={
                    "vantage_fields": {
                        "source_id": self.source_id,
                        "requested": want_fps,
                        "granted": fps,
                    }
                },
            )

    def _open_failure_message(self, backend_name: str) -> str:
        if self._kind is SourceKind.CAMERA:
            return (
                f"could not open camera {self._target} via {backend_name}. "
                "Check that no other application holds the device, that Windows "
                "Settings > Privacy > Camera allows desktop apps, and that the index "
                "exists - run 'vantage probe' to list working devices."
            )
        if self._kind is SourceKind.STREAM:
            return (
                f"could not open stream {self.uri!r} via {backend_name}. "
                "Check the URL, network reachability and any required credentials."
            )
        return (
            f"could not open video file {self.uri!r} via {backend_name}. "
            "The container or codec may be unsupported by the bundled FFmpeg build."
        )

    # -- read -----------------------------------------------------------

    def _read_impl(self) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        if self._pending is not None:
            image, pts, self._pending, self._pending_pts = (
                self._pending,
                self._pending_pts,
                None,
                None,
            )
            return image, pts, self._frame_metadata()

        cap = self._cap
        if cap is None:  # pragma: no cover - guarded by base-class state machine
            raise SourceReadError(f"source {self.source_id!r} has no open capture handle")

        attempts = self._read_retries + 1
        for attempt in range(attempts):
            pts = self._read_position_seconds(cap)
            ok, image = cap.read()
            if ok and image is not None:
                return image, pts, self._frame_metadata()

            if self._kind is SourceKind.FILE:
                return self._handle_file_end(cap)

            # Live source: a single empty read is common under USB contention.
            if attempt < attempts - 1:
                log.debug(
                    "empty read, retrying",
                    extra={
                        "vantage_fields": {
                            "source_id": self.source_id,
                            "attempt": attempt + 1,
                            "of": attempts,
                        }
                    },
                )
                time.sleep(0.01 * (attempt + 1))

        raise SourceReadError(
            f"camera {self.uri!r} returned no frame after {attempts} attempts; "
            "the device was likely disconnected or reset"
        )

    def _handle_file_end(
        self, cap: cv2.VideoCapture
    ) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        if not self._loop:
            raise SourceExhausted(
                f"reached end of {self.uri!r} after {self.frames_produced} frames"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pts = self._read_position_seconds(cap)
        ok, image = cap.read()
        if not ok or image is None:
            raise SourceReadError(
                f"failed to rewind {self.uri!r} for looping playback; "
                "the container may not support seeking"
            )
        self._loops += 1
        log.debug(
            "looped file source",
            extra={"vantage_fields": {"source_id": self.source_id, "loop": self._loops}},
        )
        return image, pts, self._frame_metadata()

    def _frame_metadata(self) -> dict[str, Any]:
        return {"loop": self._loops} if self._loop else {}

    def _read_position_seconds(self, cap: cv2.VideoCapture) -> float | None:
        """Media timestamp of the frame ``read()`` is about to return.

        Live sources have no media timeline, so ``None`` is the honest answer -
        their only meaningful timestamp is the capture time the base class stamps.
        """
        if self._kind in (SourceKind.CAMERA, SourceKind.STREAM):
            return None
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec is None or msec < 0 or not np.isfinite(msec):
            return None
        return float(msec) / 1000.0

    # -- close ----------------------------------------------------------

    def _close_impl(self) -> None:
        self._pending = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _sanitise_fps(value: float | None) -> float | None:
        """Reject the placeholder values drivers report when they don't know."""
        if value is None or not np.isfinite(value) or value <= 0 or value > 1000:
            return None
        return round(float(value), 3)

    @staticmethod
    def _sanitise_count(value: float | None) -> int | None:
        if value is None or not np.isfinite(value) or value <= 0:
            return None
        return int(value)


def _decode_fourcc(value: float | None) -> str:
    """Render the numeric FOURCC property as its four characters."""
    if not value or not np.isfinite(value):
        return "unknown"
    code = int(value)
    if code <= 0:
        return "unknown"
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if chars.isprintable() else f"0x{code:08x}"


def probe_cameras(max_index: int = 5, backend: str = "auto") -> list[dict[str, Any]]:
    """Try to open camera indices ``0..max_index-1`` and report what works.

    Used by ``vantage probe``. Opening a camera is slow (~0.5-1.5 s each on
    Windows/MSMF), so the default range is deliberately small.
    """
    api, backend_name = resolve_backend(backend, SourceKind.CAMERA)
    results: list[dict[str, Any]] = []
    for index in range(max_index):
        started = time.perf_counter()
        cap = cv2.VideoCapture(index, api)
        entry: dict[str, Any] = {"index": index, "backend": backend_name, "available": False}
        if cap.isOpened():
            ok, image = cap.read()
            if ok and image is not None:
                height, width = image.shape[:2]
                entry.update(
                    available=True,
                    width=int(width),
                    height=int(height),
                    fps=OpenCVSource._sanitise_fps(cap.get(cv2.CAP_PROP_FPS)),
                    fourcc=_decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                    open_ms=round((time.perf_counter() - started) * 1000.0, 1),
                )
        cap.release()
        results.append(entry)
    return results
