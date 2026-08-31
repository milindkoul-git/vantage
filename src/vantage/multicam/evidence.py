"""Automated Video Evidence Ring Buffering & Incident Clip Generation."""

from __future__ import annotations

import collections
import threading
from pathlib import Path

import cv2
import numpy as np

from vantage.core.logging import get_logger

log = get_logger(__name__)


class VideoEvidenceRecorder:
    """Maintains a rolling in-memory frame buffer and writes short MP4 evidence clips upon alerts."""

    def __init__(
        self,
        output_dir: str | Path = "data/evidence",
        buffer_seconds: float = 10.0,
        fps: float = 15.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = int(buffer_seconds * fps)
        self._buffers: dict[str, collections.deque[tuple[float, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._generated_clips: dict[str, str] = {}  # event_id -> file_path

    def push_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None:
        """Push a raw/annotated frame to the camera's rolling ring buffer."""
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = collections.deque(maxlen=self.max_frames)
            # Store scaled-down copy to save RAM (~360p)
            h, w = frame.shape[:2]
            scale = min(1.0, 480 / max(1, w))
            small = (
                cv2.resize(frame, (int(w * scale), int(h * scale)))
                if scale < 1.0
                else frame.copy()
            )
            self._buffers[camera_id].append((timestamp, small))

    def save_clip(self, event_id: str, camera_id: str) -> str | None:
        """Dump the recent rolling buffer for an event into an MP4 clip."""
        with self._lock:
            buf = self._buffers.get(camera_id)
            if not buf or len(buf) < 5:
                return None
            frames_to_write = list(buf)

        out_path = self.output_dir / f"{event_id}.mp4"
        try:
            h, w = frames_to_write[0][1].shape[:2]
            # Use hardware/software H264 encoder for browser playback compatibility
            writer = None
            try:
                fourcc = cv2.VideoWriter.fourcc(*"H264")
                writer = cv2.VideoWriter(str(out_path), cv2.CAP_MSMF, fourcc, self.fps, (w, h))
            except Exception:
                pass

            if writer is None or not writer.isOpened():
                try:
                    fourcc = cv2.VideoWriter.fourcc(*"avc1")
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.CAP_FFMPEG, fourcc, self.fps, (w, h)
                    )
                except Exception:
                    pass

            if writer is None or not writer.isOpened():
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))

            for _, frame_img in frames_to_write:
                # Add watermark timestamp
                cv2.rectangle(frame_img, (0, h - 24), (w, h), (10, 10, 10), -1)
                cv2.putText(
                    frame_img,
                    f"VANTAGE EVIDENCE | EVENT: {event_id} | CAM: {camera_id.upper()}",
                    (8, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 255, 200),
                    1,
                )
                writer.write(frame_img)
            writer.release()

            rel_url = f"/api/evidence/{event_id}.mp4"
            self._generated_clips[event_id] = str(out_path.resolve())
            log.debug(f"Generated video evidence clip for event {event_id} at {out_path}")
            return rel_url
        except Exception as exc:
            log.error(f"Failed to generate evidence clip for {event_id}: {exc}")
            return None

    def get_clip_path(self, event_id: str) -> Path | None:
        """Return the physical path to an event clip."""
        clean_id = event_id.replace(".mp4", "")
        p = self.output_dir / f"{clean_id}.mp4"
        return p if p.is_file() else None
