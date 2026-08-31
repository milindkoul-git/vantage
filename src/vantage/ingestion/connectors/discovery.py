"""Physical IP Camera (RTSP/ONVIF) & USB Webcam Discovery and Probing Engine."""

from __future__ import annotations

import base64
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import cv2

from vantage.core.logging import get_logger

log = get_logger(__name__)

# Common RTSP IP Camera path patterns by vendor
VENDOR_RTSP_PRESETS: dict[str, str] = {
    "generic": "/stream1",
    "onvif": "/onvif1",
    "hikvision_main": "/Streaming/Channels/101",
    "hikvision_sub": "/Streaming/Channels/102",
    "dahua_main": "/cam/realmonitor?channel=1&subtype=0",
    "dahua_sub": "/cam/realmonitor?channel=1&subtype=1",
    "axis": "/axis-media/media.amp",
    "reolink": "/h264Preview_01_main",
    "tapo": "/live/ch0",
    "amcrest": "/cam/realmonitor?channel=1&subtype=0",
    "uniview": "/media/video1",
}


@dataclass(frozen=True, slots=True)
class DiscoveredCamera:
    """Discovered physical surveillance or capture hardware."""

    device_type: str  # 'usb_webcam' or 'ip_rtsp'
    device_id: str
    name: str
    uri: str
    resolutions: list[str] = field(default_factory=list)
    declared_fps: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_id": self.device_id,
            "name": self.name,
            "uri": self.uri,
            "resolutions": self.resolutions,
            "declared_fps": self.declared_fps,
            "extra": self.extra,
        }


class CameraDiscoveryService:
    """Probes USB webcam devices and local IP camera network endpoints."""

    def __init__(self) -> None:
        # Configure OpenCV FFmpeg capture options for low-latency RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
        )

    def discover_local_webcams(self, max_indices: int = 4) -> list[DiscoveredCamera]:
        """Enumerate and probe available USB / DirectShow / V4L2 webcams."""
        cameras: list[DiscoveredCamera] = []

        # On Windows, try MSMF and DSHOW backends
        backends = (
            [cv2.CAP_MSMF, cv2.CAP_DSHOW]
            if sys.platform == "win32"
            else [cv2.CAP_V4L2, cv2.CAP_ANY]
        )

        for index in range(max_indices):
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        backend_name = (
                            "msmf"
                            if backend == cv2.CAP_MSMF
                            else ("dshow" if backend == cv2.CAP_DSHOW else "v4l2")
                        )
                        cameras.append(
                            DiscoveredCamera(
                                device_type="usb_webcam",
                                device_id=f"webcam_{index}",
                                name=f"USB / Integrated Camera {index} ({w}x{h})",
                                uri=f"webcam:{index}",
                                resolutions=[f"{w}x{h}"],
                                declared_fps=round(float(fps), 1),
                                extra={"backend": backend_name, "device_index": index},
                            )
                        )
                        cap.release()
                        break  # Found working backend for this index
                    cap.release()

        return cameras

    def test_camera_connection(
        self,
        uri: str,
        timeout_s: float = 4.0,
    ) -> dict[str, Any]:
        """Test opening a camera URI (RTSP, webcam, or video file) and return metadata + thumbnail."""
        text = uri.strip()
        if not text:
            return {"status": "error", "error": "Camera URI cannot be empty"}

        # Low-latency RTSP TCP configuration
        if text.startswith("rtsp://") or text.startswith("rtsps://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
            )

        started = time.perf_counter()
        target = (
            int(text.split(":")[-1])
            if text.startswith("webcam:") and text.split(":")[-1].isdigit()
            else text
        )
        api = (
            cv2.CAP_MSMF if sys.platform == "win32" and isinstance(target, int) else cv2.CAP_ANY
        )

        cap = cv2.VideoCapture(target, api)
        if not cap.isOpened():
            cap.release()
            return {
                "status": "error",
                "error": f"Failed to connect to '{uri}'. Check network connectivity, credentials, and RTSP path.",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return {
                "status": "error",
                "error": f"Connected to '{uri}' but delivered no visual frames.",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        h, w = frame.shape[:2]
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Generate base64 thumbnail for UI preview
        thumb_h, thumb_w = 180, int(180 * (w / max(1, h)))
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        b64_thumb = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        return {
            "status": "connected",
            "uri": uri,
            "width": int(w),
            "height": int(h),
            "latency_ms": round(latency_ms, 1),
            "thumbnail": b64_thumb,
        }

    def get_presets(self) -> dict[str, str]:
        """Return common vendor RTSP preset paths."""
        return VENDOR_RTSP_PRESETS
