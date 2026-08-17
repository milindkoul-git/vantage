"""Composition root.

The only module that knows about all of configuration, ingestion and display at
once. Everything below it is independently constructible and independently
testable; this is where the wiring lives so that no subsystem has to import a
sibling to do its job.

When Phase 2 arrives, the change is local: a detector is constructed here and
invoked inside :func:`run_ingestion`'s frame loop. Nothing in
:mod:`vantage.ingestion` changes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from vantage.config.schema import VantageConfig
from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.frame import Frame
from vantage.core.lifecycle import ShutdownController
from vantage.core.logging import get_logger
from vantage.ingestion.pipeline import IngestionPipeline, PipelineStats
from vantage.ingestion.registry import create_source
from vantage.viz.hud import HudRenderer
from vantage.viz.window import KEY_NONE, FrameSink, NullSink, WindowSink

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

    def summary(self) -> str:
        return (
            f"{self.frames} frames in {self.elapsed_s:.2f}s "
            f"({self.mean_fps:.2f} fps mean, {self.dropped} dropped, "
            f"{self.skipped} skipped) - stopped: {self.reason}"
        )


def run_ingestion(
    config: VantageConfig,
    *,
    shutdown: ShutdownController | None = None,
    sink: FrameSink | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> RunResult:
    """Ingest from the configured source until it ends or the user stops it.

    Args:
        config: Fully resolved configuration.
        shutdown: Shared shutdown controller. One is created if omitted, but the
            CLI passes its own so that Ctrl+C is handled at process level.
        sink: Destination for rendered frames. Defaults to a window when
            ``display.enabled`` is set, otherwise to :class:`NullSink`.
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

    reason = "source ended"
    snapshots: list[str] = []
    last_stats_log = clock.monotonic()
    stats = pipeline.stats()

    try:
        info = pipeline.start()
        log.info("ingestion started", extra={"vantage_fields": {"source": info.describe()}})

        for frame in pipeline.frames():
            stats = pipeline.stats()

            if config.display.enabled:
                image = (
                    hud.render(frame.image, stats, frame.index)
                    if hud_enabled
                    else frame.editable_copy()
                )
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
    )
    log.info("run complete", extra={"vantage_fields": {"summary": result.summary()}})
    return result


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
