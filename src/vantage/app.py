"""Composition root.

The only module that knows about all of configuration, ingestion, perception,
tracking and display at once. Everything below it is independently
constructible and independently testable; this is where the wiring lives so
that no subsystem has to import a sibling to do its job.

The Phase 2 and Phase 3 additions both bore that out. Adding detection, and
then adding tracking on top of it, changed this module and nothing in
:mod:`vantage.ingestion` - the frame contract absorbed both.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from vantage.config.schema import VantageConfig
from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.frame import Frame
from vantage.core.lifecycle import ShutdownController
from vantage.core.logging import get_logger
from vantage.core.metrics import LatencyTracker
from vantage.ingestion.pipeline import IngestionPipeline, PipelineStats
from vantage.ingestion.registry import create_source
from vantage.tracking.factory import build_tracker
from vantage.viz.hud import HudRenderer
from vantage.viz.overlay import draw_detections, draw_tracks
from vantage.viz.window import KEY_NONE, FrameSink, NullSink, WindowSink

if TYPE_CHECKING:  # detection is optional; importing it eagerly would make
    # onnxruntime/openvino a hard requirement for plain ingestion.
    from vantage.perception.engine import DetectionEngine
    from vantage.tracking.base import Tracker

log = get_logger(__name__)

KEY_QUIT = {ord("q"), 27}  # 'q' and ESC
KEY_SNAPSHOT = ord("s")
KEY_TOGGLE_HUD = ord("h")


@dataclass(slots=True)
class RunResult:
    """Outcome of one ingestion run, for the CLI and for tests."""

    frames: int
    elapsed_s: float
    mean_fps: float
    dropped: int
    skipped: int
    reason: str
    stats: dict[str, Any] = field(default_factory=dict)
    snapshots: list[str] = field(default_factory=list)
    detections_run: int = 0
    detection_summary: dict[str, Any] = field(default_factory=dict)
    tracking_steps: int = 0
    tracking_summary: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        base = (
            f"{self.frames} frames in {self.elapsed_s:.2f}s "
            f"({self.mean_fps:.2f} fps mean, {self.dropped} dropped, "
            f"{self.skipped} skipped) - stopped: {self.reason}"
        )
        if self.detections_run:
            detail = self.detection_summary
            base += (
                f"\ndetection: {self.detections_run} passes on "
                f"{detail.get('model', '?')} via {detail.get('backend', '?')}"
                f"/{detail.get('device', '?')}, "
                f"{detail.get('mean_total_ms', 0.0):.1f} ms mean "
                f"({detail.get('max_fps', 0.0):.1f} fps ceiling)"
            )
        if self.tracking_steps:
            detail = self.tracking_summary
            base += (
                f"\ntracking: {self.tracking_steps} steps, "
                f"{detail.get('entities_total', 0)} distinct entities, "
                f"{detail.get('active', 0)} active at end, "
                f"{detail.get('mean_ms', 0.0):.2f} ms mean"
            )
        return base


def run_ingestion(
    config: VantageConfig,
    *,
    shutdown: ShutdownController | None = None,
    sink: FrameSink | None = None,
    engine: "DetectionEngine | None" = None,
    clock: Clock = SYSTEM_CLOCK,
) -> RunResult:
    """Ingest from the configured source until it ends or the user stops it.

    Runs object detection over the delivered frames when ``detection.enabled``
    is set; otherwise this is exactly the Phase 1 ingestion loop.

    Args:
        config: Fully resolved configuration.
        shutdown: Shared shutdown controller. One is created if omitted, but the
            CLI passes its own so that Ctrl+C is handled at process level.
        sink: Destination for rendered frames. Defaults to a window when
            ``display.enabled`` is set, otherwise to :class:`NullSink`.
        engine: A pre-built detector. Supplied by tests and by the benchmark so
            they can inject a fake or reuse a loaded model; otherwise one is
            built from ``config.detection`` and owned by this call.
        clock: Time source; overridden in tests.
    """
    controller = shutdown or ShutdownController()
    owns_sink = sink is None
    sink = sink or _build_sink(config)
    hud = HudRenderer()
    hud_enabled = config.display.hud

    source = create_source(config.source, clock=clock)
    pipeline = IngestionPipeline(
        source,
        config.ingest,
        clock=clock,
        shutdown=_linked_event(controller),
    )

    # Built before the source opens: a missing model or absent runtime should
    # fail before a camera is acquired, not after.
    owns_engine = engine is None
    detector = engine if engine is not None else _build_engine(config)

    reason = "source ended"
    snapshots: list[str] = []
    last_stats_log = clock.monotonic()
    stats = pipeline.stats()
    latest_detection = None
    detection_stale = False
    detections_run = 0
    delivered_count = 0
    detect_latency = LatencyTracker(window=240)

    tracker = build_tracker(config.tracking)
    latest_tracking = None
    tracking_steps = 0
    track_latency = LatencyTracker(window=240)

    try:
        info = pipeline.start()
        log.info("ingestion started", extra={"vantage_fields": {"source": info.describe()}})

        for frame in pipeline.frames():
            stats = pipeline.stats()

            if detector is not None:
                # Counted on *delivered* frames, not on frame.index. Under
                # backpressure the source index has gaps, and a modulo on it
                # could line up so that detection never runs at all.
                should_detect = delivered_count % config.detection.interval == 0
                delivered_count += 1
                if should_detect:
                    latest_detection = detector.detect(frame)
                    detect_latency.observe(latest_detection.total_ms)
                    detection_stale = False
                    detections_run += 1
                    if latest_detection.detections:
                        log.debug(
                            "detections",
                            extra={"vantage_fields": {"summary": latest_detection.describe()}},
                        )
                    # Tracking steps only when detection does. Advancing it on
                    # skipped frames would be asking the motion model to
                    # extrapolate with no evidence, which costs accuracy and
                    # buys nothing: the displayed boxes are already carried
                    # forward between passes.
                    if tracker is not None:
                        latest_tracking = tracker.update(latest_detection, frame=frame)
                        track_latency.observe(latest_tracking.tracking_ms)
                        tracking_steps += 1
                        if latest_tracking.tracks:
                            log.debug(
                                "tracks",
                                extra={
                                    "vantage_fields": {"summary": latest_tracking.describe()}
                                },
                            )
                else:
                    # Carry the last pass forward so the display stays populated
                    # between inferences, but mark it so it is drawn as stale.
                    detection_stale = latest_detection is not None

            if config.display.enabled:
                image = (
                    hud.render(
                        frame.image,
                        stats,
                        frame.index,
                        detection=latest_detection,
                        engine=detector.info if detector else None,
                        tracking=latest_tracking,
                        entity_total=_entity_total(tracker),
                    )
                    if hud_enabled
                    else frame.editable_copy()
                )
                # Tracks supersede raw detections on screen. Drawing both would
                # double every box, and once identity exists it is the more
                # informative of the two.
                if latest_tracking is not None:
                    image = draw_tracks(image, latest_tracking, stale=detection_stale)
                elif latest_detection is not None:
                    image = draw_detections(image, latest_detection, stale=detection_stale)
                key = sink.show(image)
                if key != KEY_NONE:
                    action = _handle_key(key, frame, stats, config, snapshots)
                    if action == "quit":
                        reason = "user quit"
                        break
                    if action == "toggle_hud":
                        hud_enabled = not hud_enabled
                if sink.is_closed():
                    reason = "window closed"
                    break

            if controller.is_set():
                reason = "shutdown signal"
                break

            interval = config.app.stats_interval_s
            if interval and clock.monotonic() - last_stats_log >= interval:
                last_stats_log = clock.monotonic()
                _log_stats(stats)
        else:
            reason = (
                "frame limit reached"
                if config.ingest.max_frames is not None
                and stats.frames_delivered >= config.ingest.max_frames
                else "source ended"
            )
    finally:
        pipeline.close()
        if owns_sink:
            sink.close()
        if owns_engine and detector is not None:
            detector.close()

    final = pipeline.stats()
    result = RunResult(
        frames=final.frames_delivered,
        elapsed_s=final.elapsed_s,
        mean_fps=final.mean_delivery_fps,
        dropped=final.frames_dropped,
        skipped=final.frames_skipped,
        reason=reason,
        stats=final.to_dict(),
        snapshots=snapshots,
        detections_run=detections_run,
        detection_summary=_detection_summary(detector, detect_latency, detections_run),
        tracking_steps=tracking_steps,
        tracking_summary=_tracking_summary(tracker, track_latency, tracking_steps),
    )
    log.info("run complete", extra={"vantage_fields": {"summary": result.summary()}})
    return result


def _build_engine(config: VantageConfig) -> "DetectionEngine | None":
    """Construct the detector, or ``None`` when detection is disabled.

    Deliberately eager: resolving the model and compiling the graph before the
    camera opens means a misconfiguration fails in under a second instead of
    after a device has been acquired and frames are already flowing.
    """
    if not config.detection.enabled:
        return None

    from vantage.perception.engine import build_engine

    settings = config.detection
    confidence = _effective_confidence(config)
    engine = build_engine(
        settings.model,
        backend=settings.backend,
        device=settings.device,
        confidence=confidence,
        iou_threshold=settings.nms_iou,
        max_detections=settings.max_detections,
        keep_classes=settings.classes,
        model_dir=settings.model_dir,
        threads=settings.threads,
        allow_download=settings.allow_download,
    )
    if settings.warmup:
        engine.warmup(settings.warmup)
    return engine


def _effective_confidence(config: VantageConfig) -> float:
    """The detector threshold to actually use, given whether tracking is on.

    ByteTrack's second association pass only works if it is handed the
    low-scoring boxes an occluded object produces, so enabling tracking lowers
    the detector's floor to ``tracking.detection_floor``. Doing this silently
    would be indefensible - the user set ``detection.confidence`` and would see
    a different number honoured - so it is logged whenever it takes effect.

    Only ever lowers. A ``detection_floor`` above the configured confidence
    would discard boxes the user asked for, so the stricter of the two wins.
    """
    configured = config.detection.confidence
    if not config.tracking.enabled:
        return configured

    floor = min(config.tracking.detection_floor, configured)
    if floor < configured:
        log.info(
            "detector threshold lowered for tracking",
            extra={
                "vantage_fields": {
                    "detection_confidence": configured,
                    "effective": floor,
                    "reason": (
                        "ByteTrack matches low-confidence boxes to existing tracks; "
                        "filtering them at the detector would disable that"
                    ),
                }
            },
        )
    return floor


def _detection_summary(
    detector: "DetectionEngine | None", latency: LatencyTracker, passes: int
) -> dict[str, Any]:
    if detector is None or not passes:
        return {}
    mean_ms = latency.mean
    return {
        "model": detector.info.model,
        "backend": detector.info.backend,
        "device": detector.info.device,
        "precision": detector.info.precision,
        "license": detector.info.license,
        "passes": passes,
        "mean_total_ms": round(mean_ms, 2),
        "p50_ms": round(latency.percentile(50), 2),
        "p95_ms": round(latency.percentile(95), 2),
        "max_fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def _entity_total(tracker: "Tracker | None") -> int:
    """Distinct entities published so far, or 0 for a tracker that cannot say.

    The count comes from the tracker rather than from a set accumulated here,
    so a run lasting weeks does not carry every identifier it ever issued.
    """
    stats = getattr(tracker, "stats", None)
    if not callable(stats):
        return 0
    return int(stats().get("entities_published", 0))


def _tracking_summary(
    tracker: "Tracker | None",
    latency: LatencyTracker,
    steps: int,
) -> dict[str, Any]:
    if tracker is None or not steps:
        return {}
    summary: dict[str, Any] = {
        "steps": steps,
        "entities_total": _entity_total(tracker),
        "mean_ms": round(latency.mean, 3),
        "p95_ms": round(latency.percentile(95), 3),
    }
    # ByteTracker exposes health counters; the Protocol does not require them,
    # so a different tracker implementation simply contributes fewer fields
    # rather than breaking the run summary.
    stats = getattr(tracker, "stats", None)
    if callable(stats):
        summary.update(stats())
    return summary


def _build_sink(config: VantageConfig) -> FrameSink:
    if not config.display.enabled:
        return NullSink()
    return WindowSink(window_name=config.display.window_name, scale=config.display.scale)


def _linked_event(controller: ShutdownController) -> threading.Event:
    """Give the pipeline the controller's own flag, so one Ctrl+C stops everything."""
    return controller.event


def _handle_key(
    key: int,
    frame: Frame,
    stats: PipelineStats,
    config: VantageConfig,
    snapshots: list[str],
) -> str:
    if key in KEY_QUIT:
        return "quit"
    if key == KEY_TOGGLE_HUD:
        return "toggle_hud"
    if key == KEY_SNAPSHOT:
        path = _save_snapshot(frame, stats, config)
        if path:
            snapshots.append(str(path))
    return "continue"


def _save_snapshot(frame: Frame, stats: PipelineStats, config: VantageConfig) -> Path | None:
    directory = Path(config.display.snapshot_dir).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stats.source_id}_{frame.index:06d}.png"
        # imwrite needs a contiguous writable buffer; frames are read-only.
        if not cv2.imwrite(str(path), frame.editable_copy()):
            raise OSError("OpenCV declined to encode the image")
    except OSError as exc:
        log.warning(
            "could not save snapshot",
            extra={"vantage_fields": {"directory": str(directory), "error": str(exc)}},
        )
        return None
    log.info("snapshot saved", extra={"vantage_fields": {"path": str(path)}})
    return path


def _log_stats(stats: PipelineStats) -> None:
    log.info(
        "pipeline stats",
        extra={
            "vantage_fields": {
                "source_id": stats.source_id,
                "delivered": stats.frames_delivered,
                "fps_out": round(stats.delivery_fps, 2),
                "fps_in": round(stats.capture_fps, 2),
                "latency_p95_ms": round(stats.latency_ms_p95, 2),
                "queue": f"{stats.queue_depth}/{stats.queue_capacity}",
                "dropped": stats.frames_dropped,
            }
        },
    )
