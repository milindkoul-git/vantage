"""Heads-up display for pipeline telemetry.

Pure rendering: :meth:`HudRenderer.render` takes a frame plus a
:class:`~vantage.ingestion.pipeline.PipelineStats` and returns a new image. It
touches no global state and opens no window, so it is fully testable headless -
which is how the tests exercise it.

The choice of what to show is deliberate. Alongside the requested stream, FPS,
frame number, resolution and source, it shows *capture* FPS next to *delivery*
FPS, plus queue depth and drop count. A gap between those two numbers is the
signature of a consumer that cannot keep up, which is precisely the condition
Phase 2 will introduce. Making it visible from day one means the failure mode
is diagnosed rather than guessed at.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from vantage.ingestion.pipeline import PipelineStats

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (245, 245, 245)
_DIM = (170, 170, 170)
_GOOD = (120, 230, 130)
_WARN = (90, 190, 250)
_BAD = (90, 90, 250)
_ACCENT = (255, 190, 80)


class HudRenderer:
    """Draws a telemetry panel over a copy of the frame."""

    def __init__(self, history: int = 120, scale: float = 1.0) -> None:
        self._fps_history: deque[float] = deque(maxlen=history)
        self._scale = scale

    def render(
        self,
        image: np.ndarray,
        stats: PipelineStats,
        frame_index: int,
        extra: list[str] | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of ``image``.

        ``image`` is never modified: frames are shared read-only across stages,
        and the viewer is not privileged.
        """
        canvas = image.copy()
        self._fps_history.append(stats.delivery_fps)

        lines = self._compose(stats, frame_index)
        if extra:
            lines.extend(("", value, _DIM) for value in extra)

        scale = _text_scale(canvas.shape[1])
        self._draw_panel(canvas, lines, scale)
        self._draw_sparkline(canvas, stats, scale)
        self._draw_footer(canvas, scale)
        return canvas

    # -- content --------------------------------------------------------

    def _compose(
        self, stats: PipelineStats, frame_index: int
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        fps_colour = _GOOD
        if stats.declared_fps and stats.delivery_fps < stats.declared_fps * 0.6:
            fps_colour = _WARN
        if stats.delivery_fps and stats.delivery_fps < 5:
            fps_colour = _BAD

        rows: list[tuple[str, str, tuple[int, int, int]]] = [
            ("source", f"{stats.source_id}  ({stats.kind}/{stats.backend})", _ACCENT),
            ("uri", _shorten(stats.uri, 46), _DIM),
            ("resolution", f"{stats.resolution}"
             + (f" @ {stats.declared_fps:g} fps" if stats.declared_fps else " @ fps unknown"), _WHITE),
            ("frame", f"#{frame_index}   delivered {stats.frames_delivered}", _WHITE),
            ("fps", f"{stats.delivery_fps:5.1f} out / {stats.capture_fps:5.1f} in"
                    f"   (mean {stats.mean_delivery_fps:.1f})", fps_colour),
            ("latency", f"p50 {stats.latency_ms_p50:.1f} ms   p95 {stats.latency_ms_p95:.1f} ms", _WHITE),
            ("acquire", f"p50 {stats.acquire_ms_p50:.1f} ms", _DIM),
        ]

        if stats.queue_capacity:
            depth_colour = _GOOD if stats.queue_depth < stats.queue_capacity else _WARN
            rows.append(
                (
                    "queue",
                    f"{stats.queue_depth}/{stats.queue_capacity} "
                    f"(peak {stats.queue_high_water}, {stats.backpressure})",
                    depth_colour,
                )
            )
        else:
            rows.append(("queue", "inline (no buffering)", _DIM))

        loss_colour = _GOOD if stats.frames_dropped == 0 else _WARN
        rows.append(
            (
                "loss",
                f"dropped {stats.frames_dropped} ({stats.drop_rate * 100:.1f}%)"
                f"   skipped {stats.frames_skipped}",
                loss_colour,
            )
        )
        if stats.reconnects:
            rows.append(("reconnects", str(stats.reconnects), _WARN))
        rows.append(("elapsed", f"{stats.elapsed_s:.1f} s", _DIM))
        return rows

    # -- drawing --------------------------------------------------------

    def _draw_panel(
        self,
        canvas: np.ndarray,
        rows: list[tuple[str, str, tuple[int, int, int]]],
        scale: float,
    ) -> None:
        pad = int(12 * scale)
        line_h = int(22 * scale)
        label_w = int(96 * scale)
        width = int(430 * scale)
        height = pad * 2 + line_h * len(rows)

        overlay_region = canvas[0 : min(height, canvas.shape[0]), 0 : min(width, canvas.shape[1])]
        darkened = (overlay_region.astype(np.float32) * 0.25).astype(np.uint8)
        overlay_region[:] = darkened
        cv2.rectangle(
            canvas,
            (0, 0),
            (min(width, canvas.shape[1]) - 1, min(height, canvas.shape[0]) - 1),
            (70, 70, 70),
            1,
        )

        y = pad + int(14 * scale)
        for label, value, colour in rows:
            cv2.putText(canvas, label, (pad, y), _FONT, 0.42 * scale, _DIM, 1, cv2.LINE_AA)
            cv2.putText(
                canvas, value, (pad + label_w, y), _FONT, 0.46 * scale, colour, 1, cv2.LINE_AA
            )
            y += line_h

    def _draw_sparkline(self, canvas: np.ndarray, stats: PipelineStats, scale: float) -> None:
        """Recent delivery-rate history - a stall shows as a visible dip."""
        if len(self._fps_history) < 2:
            return
        height, width = canvas.shape[:2]
        box_w, box_h = int(220 * scale), int(46 * scale)
        x0, y0 = width - box_w - int(12 * scale), int(12 * scale)
        if x0 < 0 or y0 + box_h > height:
            return

        region = canvas[y0 : y0 + box_h, x0 : x0 + box_w]
        region[:] = (region.astype(np.float32) * 0.25).astype(np.uint8)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (70, 70, 70), 1)

        peak = max(max(self._fps_history), stats.declared_fps or 1.0, 1.0)
        points = []
        count = len(self._fps_history)
        for i, value in enumerate(self._fps_history):
            px = x0 + int(i / max(1, count - 1) * (box_w - 2)) + 1
            py = y0 + box_h - 1 - int(min(value / peak, 1.0) * (box_h - 2))
            points.append((px, py))
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, _ACCENT, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"fps 0-{peak:.0f}",
            (x0 + int(6 * scale), y0 + box_h - int(6 * scale)),
            _FONT,
            0.36 * scale,
            _DIM,
            1,
            cv2.LINE_AA,
        )

    def _draw_footer(self, canvas: np.ndarray, scale: float) -> None:
        text = "q / ESC quit    s save frame    h toggle HUD"
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.42 * scale, 1)
        x = canvas.shape[1] - tw - int(12 * scale)
        y = canvas.shape[0] - int(12 * scale)
        if x < 0 or y - th < 0:
            return
        cv2.putText(canvas, text, (x, y), _FONT, 0.42 * scale, _DIM, 1, cv2.LINE_AA)


def _text_scale(width: int) -> float:
    """Keep the panel legible from 320px to 4K without hard-coding a size."""
    return max(0.75, min(1.6, width / 1280.0))


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."
