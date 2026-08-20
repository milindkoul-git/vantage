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
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vantage.ingestion.pipeline import PipelineStats

if TYPE_CHECKING:  # imported for typing only - viz must not require a detector
    from vantage.perception.contracts import DetectionResult
    from vantage.perception.engine import EngineInfo
    from vantage.tracking.contracts import TrackingResult
    from vantage.pose.contracts import PoseResult
    from vantage.state.contracts import StateResult
    from vantage.activity.contracts import ActivityResult
    from vantage.spatial.contracts import SpatialResult

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
        detection: "DetectionResult | None" = None,
        engine: "EngineInfo | None" = None,
        tracking: "TrackingResult | None" = None,
        entity_total: int = 0,
        pose: "PoseResult | None" = None,
        state: "StateResult | None" = None,
        activity: "ActivityResult | None" = None,
        spatial: "SpatialResult | None" = None,
    ) -> np.ndarray:
        """Return an annotated copy of ``image``.

        ``image`` is never modified: frames are shared read-only across stages,
        and the viewer is not privileged.
        """
        canvas = image.copy()
        self._fps_history.append(stats.delivery_fps)

        lines = self._compose(stats, frame_index)
        if engine is not None:
            lines.extend(self._compose_detection(detection, engine))
        if tracking is not None:
            lines.extend(self._compose_tracking(tracking, entity_total))
        if state is not None:
            lines.extend(self._compose_state(state))
        if pose is not None:
            lines.extend(self._compose_pose(pose))
        if activity is not None:
            lines.extend(self._compose_activity(activity))
        if spatial is not None:
            lines.extend(self._compose_spatial(spatial))
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

    def _compose_detection(
        self, detection: "DetectionResult | None", engine: "EngineInfo | None"
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        """Detection telemetry, shown only when a detector is attached."""
        if engine is None:
            return []

        rows: list[tuple[str, str, tuple[int, int, int]]] = [
            ("model", f"{engine.model} on {engine.backend}/{engine.device} ({engine.precision})", _ACCENT),
        ]
        if detection is None:
            rows.append(("objects", "waiting for first pass", _DIM))
            return rows

        summary = ", ".join(
            f"{count}x {label}" for label, count in sorted(detection.counts().items())
        )
        rows.append(("objects", f"{len(detection)}  {summary}" if summary else "0  none", _WHITE))

        # Inference time governs the maximum sustainable detection rate, so it
        # is coloured against that budget rather than an arbitrary threshold.
        total = detection.total_ms
        colour = _GOOD if total < 40 else (_WARN if total < 100 else _BAD)
        rows.append(
            (
                "detect",
                f"{total:5.1f} ms  (pre {detection.preprocess_ms:.1f} / "
                f"inf {detection.inference_ms:.1f} / post {detection.postprocess_ms:.1f})",
                colour,
            )
        )
        rows.append(("det rate", f"max {1000.0 / total:.1f} fps" if total > 0 else "n/a", _DIM))
        return rows

    def _compose_spatial(
        self, spatial: "SpatialResult"
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        """Zone occupancy, boundary crossings and the notable relations."""
        rows: list[tuple[str, str, tuple[int, int, int]]] = []

        occupancy = spatial.occupancy()
        if spatial.zones_defined:
            summary = (
                ", ".join(f"{n} in {name}" for name, n in sorted(occupancy.items()))
                or "all zones empty"
            )
            rows.append(("zones", summary, _WHITE if occupancy else _DIM))

        for entity, zone in spatial.crossings():
            rows.append((zone.event.value, f"{entity.entity_id} {zone.zone}", _WARN))

        counts = {k: v for k, v in spatial.counts().items() if k != "near"}
        if counts:
            rows.append(
                (
                    "relations",
                    ", ".join(f"{n} {name}" for name, n in sorted(counts.items())),
                    _WHITE,
                )
            )
        # Without motion state, interaction is only claimed on a confirmed
        # reach. That materially changes what the line above can mean, so it is
        # said rather than left for someone to infer from missing rows.
        if spatial.zones_defined or counts:
            if not spatial.state_available:
                rows.append(("note", "no motion state: reach-confirmed only", _DIM))
        return rows

    def _compose_activity(
        self, activity: "ActivityResult"
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        """What entities are doing, with transient events called out."""
        if not activity.entities:
            return []

        counts = {
            name: n for name, n in activity.counts().items() if name != "idle"
        }
        summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        rows = [("activity", summary or "nothing notable", _WHITE if counts else _DIM)]

        # A transient event is the one thing on this panel that will be gone in
        # a second, so it gets its own line in the warning colour rather than
        # being averaged into a tally.
        for entity in activity.entities:
            primary = entity.primary
            if primary is not None and primary.activity.is_transient:
                rows.append(
                    (
                        primary.activity.value,
                        f"{entity.entity_id} ({primary.confidence:.2f})",
                        _WARN,
                    )
                )
        return rows

    def _compose_state(
        self, state: "StateResult"
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        """Motion state, and the longest-standing entity."""
        if not state.states:
            return []
        counts = state.counts()
        summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        rows = [("motion", summary, _WHITE)]

        # The longest dwell is the number an operator scans for: a thing that
        # has not moved in a long time is what "anything unusual here?" means
        # before the event engine exists to say so.
        longest = max(state.states, key=lambda s: s.dwell_s)
        if longest.dwell_s >= 1.0:
            rows.append(
                (
                    "longest",
                    f"{longest.entity_id} {longest.motion.value} {longest.dwell_s:.0f}s",
                    _DIM,
                )
            )
        return rows

    def _compose_pose(
        self, pose: "PoseResult"
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        """Pose telemetry, shown only when an estimator is attached."""
        if not pose.people_seen:
            return [("pose", "no people", _DIM)]

        counts = pose.counts()
        summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        rows = [("pose", f"{len(pose)} of {pose.people_seen}: {summary}", _WHITE)]

        # Over-budget people are called out rather than left as a silent
        # difference between two numbers - it is the one pose failure that is
        # invisible in the picture, because the skipped person still has a box.
        if pose.skipped:
            rows.append(("skipped", f"{pose.skipped} over max_persons", _WARN))
        if len(pose):
            rows.append(("pose ms", f"{pose.total_ms / max(1, len(pose)):.1f} per person", _DIM))

        # Show why, when nothing could be classified. "unknown" on its own reads
        # as a fault; on a desk webcam it is the correct answer and the reason
        # is the only thing that says so.
        reasons = pose.unknown_reasons()
        if reasons:
            commonest = max(reasons.items(), key=lambda kv: kv[1])[0]
            rows.append(("why", _shorten(commonest, 46), _DIM))
        return rows

    def _compose_tracking(
        self, tracking: "TrackingResult", entity_total: int
    ) -> list[tuple[str, str, tuple[int, int, int]]]:
        """Tracking telemetry, shown only when a tracker is attached."""
        coasting = sum(1 for track in tracking.tracks if track.is_coasting)
        observed = len(tracking) - coasting

        rows: list[tuple[str, str, tuple[int, int, int]]] = [
            (
                "tracks",
                f"{len(tracking)} shown ({observed} seen, {coasting} predicted)",
                _WHITE if not coasting else _WARN,
            ),
        ]

        # Distinct entities over the whole run is the number an operator cares
        # about; the instantaneous count is already above.
        if entity_total:
            rows.append(("entities", f"{entity_total} total this run", _ACCENT))

        # A large gap between maintained and published tracks means the detector
        # is generating objects that never corroborate - a health signal that is
        # invisible from the published list alone.
        pending = max(0, tracking.active_count - len(tracking))
        if pending:
            rows.append(("pending", f"{pending} unconfirmed", _DIM))

        rows.append(
            (
                "track",
                f"{tracking.tracking_ms:5.2f} ms  (step {tracking.elapsed_s * 1000:.0f} ms)",
                _GOOD if tracking.tracking_ms < 5 else _WARN,
            )
        )
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


def _shorten(text: str, limit: int) -> str:
    """Trim to fit the HUD panel without wrapping it.

    ASCII only: the Hershey fonts OpenCV ships cannot render a real ellipsis and
    would draw a placeholder box instead.
    """
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
