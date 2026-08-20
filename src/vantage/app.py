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
from vantage.core.governor import GovernorParams, LoadGovernor
from vantage.core.lifecycle import ShutdownController
from vantage.core.logging import get_logger
from vantage.core.metrics import LatencyTracker
from vantage.core.resilience import StageRegistry
from vantage.core.resources import ResourceSampler
from vantage.ingestion.pipeline import IngestionPipeline, PipelineStats
from vantage.ingestion.registry import create_source
from vantage.tracking.factory import build_tracker
from vantage.viz.hud import HudRenderer
from vantage.viz.overlay import (
    draw_activities,
    draw_detections,
    draw_poses,
    draw_relations,
    draw_tracks,
    draw_zones,
)
from vantage.viz.window import KEY_NONE, FrameSink, NullSink, WindowSink

if TYPE_CHECKING:  # detection is optional; importing it eagerly would make
    # onnxruntime/openvino a hard requirement for plain ingestion.
    from vantage.activity.contracts import ActivityResult
    from vantage.activity.engine import ActivityEngine
    from vantage.events.engine import EventEngine
    from vantage.perception.engine import DetectionEngine
    from vantage.pose.contracts import PoseResult
    from vantage.pose.engine import PoseEngine
    from vantage.spatial.contracts import SpatialResult
    from vantage.spatial.engine import SpatialEngine
    from vantage.state.contracts import StateResult
    from vantage.state.estimator import StateEstimator
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
    pose_steps: int = 0
    pose_summary: dict[str, Any] = field(default_factory=dict)
    state_summary: dict[str, Any] = field(default_factory=dict)
    activity_summary: dict[str, Any] = field(default_factory=dict)
    spatial_summary: dict[str, Any] = field(default_factory=dict)
    stage_health: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    adaptive: dict[str, Any] = field(default_factory=dict)
    events_raised: int = 0
    events_summary: dict[str, Any] = field(default_factory=dict)
    storage_summary: dict[str, Any] = field(default_factory=dict)

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
        if self.pose_steps:
            detail = self.pose_summary
            base += (
                f"\npose: {self.pose_steps} passes on {detail.get('model', '?')}, "
                f"{detail.get('people', 0)} people estimated, "
                f"{detail.get('mean_ms_per_person', 0.0):.1f} ms mean per person"
            )
            if detail.get("skipped"):
                base += f", {detail['skipped']} skipped over budget"
        if self.state_summary:
            detail = self.state_summary
            base += (
                f"\nstate: {detail.get('entities', 0)} entities, "
                f"{detail.get('moving', 0)} moving at end"
            )
        if self.activity_summary:
            counts = self.activity_summary.get("counts") or {}
            summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
            base += f"\nactivity: {summary or 'nothing recognised'} at end"
        if self.spatial_summary:
            detail = self.spatial_summary
            occupancy = detail.get("occupancy") or {}
            relations = detail.get("relations") or {}
            parts = [f"{detail.get('zones', 0)} zones"]
            if occupancy:
                parts.append(
                    "occupied: "
                    + ", ".join(f"{n} in {name}" for name, n in sorted(occupancy.items()))
                )
            if relations:
                parts.append(", ".join(f"{n} {name}" for name, n in sorted(relations.items())))
            base += f"\nspatial: {'; '.join(parts)} at end"
        if self.resources:
            detail = self.resources
            memory = (
                f"{detail.get('rss_mb')} MB RSS"
                if detail.get("rss_mb") is not None
                else "memory unavailable"
            )
            growth = detail.get("growth_mb")
            if growth is not None and abs(growth) >= 1.0:
                memory += f" ({growth:+.1f} MB since start)"
            base += f"\nresources: {detail.get('cpu_cores', 0.0):.2f} cores, {memory}"
        if self.adaptive and self.adaptive.get("peak_interval", 1) > self.adaptive.get(
            "base_interval", 1
        ):
            detail = self.adaptive
            base += (
                f"\nadaptive: analysis interval {detail['base_interval']} -> "
                f"{detail['interval']} (peak {detail['peak_interval']}), "
                f"{detail['degraded_s']:.0f}s degraded"
            )
            if detail.get("at_ceiling_s", 0) > 0:
                base += f", {detail['at_ceiling_s']:.0f}s at the ceiling"
        if self.events_summary:
            detail = self.events_summary
            by_rule = detail.get("by_rule") or {}
            summary = ", ".join(f"{n} {name}" for name, n in sorted(by_rule.items()))
            base += (
                f"\nevents: {self.events_raised} raised"
                + (f" ({summary})" if summary else "")
                # Suppressions are reported, not hidden. A rule suppressing
                # thousands is either correctly debouncing a continuous state or
                # badly configured, and only the count tells the two apart.
                + f", {detail.get('suppressed', 0)} suppressed by cooldown"
            )
        if self.storage_summary:
            detail = self.storage_summary
            base += (
                f"\nstorage: {detail.get('events_written', 0)} events, "
                f"{detail.get('observations_written', 0)} observations written "
                f"in {detail.get('batches', 0)} batches"
            )
            # Dropped events are never folded into a total. Each is the output
            # of a rule that already decided it mattered.
            if detail.get("events_dropped"):
                base += f"; {detail['events_dropped']} EVENTS DROPPED"
            if detail.get("observations_dropped"):
                base += f"; {detail['observations_dropped']} observations dropped"
            if detail.get("write_errors"):
                base += f"; {detail['write_errors']} write errors"
        degraded = [stat for stat in self.stage_health.values() if stat.get("failures")]
        if degraded:
            # Never folded into a healthy-looking summary. A stage that failed
            # is the most important thing on the line it appears on.
            base += "\nDEGRADED: " + "; ".join(
                f"{s['name']} {s['failures']}/{s['calls']} failed"
                + (" (DISABLED)" if s.get("disabled") else "")
                for s in degraded
            )
        return base


def run_ingestion(
    config: VantageConfig,
    *,
    shutdown: ShutdownController | None = None,
    sink: FrameSink | None = None,
    engine: DetectionEngine | None = None,
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
    last_resource_log = last_stats_log
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

    estimator = _build_pose_engine(config)
    latest_pose = None
    pose_steps = 0
    pose_latency = LatencyTracker(window=240)
    state_estimator = _build_state_estimator(config)
    latest_state = None
    activity_engine = _build_activity_engine(config)
    latest_activity = None
    spatial_engine = _build_spatial_engine(config)
    latest_spatial = None
    event_engine = _build_event_engine(config)
    latest_events = None
    events_raised = 0
    store, store_writer, recorder = _build_storage(config)
    live_feed, dashboard = _build_dashboard(config, store)

    # One guard per stage. A stage that throws loses its frame, not the run;
    # a stage that throws repeatedly is disabled and said so, loudly.
    stages = StageRegistry(max_consecutive=config.app.stage_failure_budget)
    resources = ResourceSampler()
    latest_resources = None
    final_resources = None
    governor = _build_governor(config)
    analysis_cost = LatencyTracker(window=120)
    last_frame_monotonic = None

    try:
        info = pipeline.start()
        log.info("ingestion started", extra={"vantage_fields": {"source": info.describe()}})

        if governor is not None and not info.is_live:
            # A recorded source has no deadline: analysing it slowly produces
            # exactly the same answer as analysing it quickly, so shedding load
            # would trade information for nothing. Said out loud, because a
            # governor that silently does not run is worse than one that does.
            log.info(
                "adaptive load shedding not engaged",
                extra={
                    "vantage_fields": {
                        "reason": "source is recorded, so there is no frame deadline to miss"
                    }
                },
            )
            governor = None

        for frame in pipeline.frames():
            stats = pipeline.stats()

            # Measured between deliveries rather than taken from declared_fps:
            # a camera that claims 30 fps and delivers 22 has a real budget of
            # 45 ms, and shedding load against the number on the box would
            # leave the pipeline permanently behind.
            now_monotonic = clock.monotonic()
            frame_gap_ms = (
                (now_monotonic - last_frame_monotonic) * 1000.0
                if last_frame_monotonic is not None
                else 0.0
            )
            last_frame_monotonic = now_monotonic
            analysis_interval = config.detection.interval
            if governor is not None:
                analysis_interval = governor.observe(
                    analysis_cost.mean, frame_gap_ms, frame_gap_ms / 1000.0
                )

            if detector is not None:
                # Counted on *delivered* frames, not on frame.index. Under
                # backpressure the source index has gaps, and a modulo on it
                # could line up so that detection never runs at all.
                should_detect = delivered_count % analysis_interval == 0
                delivered_count += 1
                if should_detect:
                    detected = stages.guard("detection").run(detector.detect, frame)
                    if detected is not None:
                        latest_detection = detected
                        detect_latency.observe(latest_detection.total_ms)
                        detection_stale = False
                        detections_run += 1
                        if latest_detection.detections:
                            log.debug(
                                "detections",
                                extra={
                                    "vantage_fields": {"summary": latest_detection.describe()}
                                },
                            )
                    # Tracking steps only when detection does. Advancing it on
                    # skipped frames would be asking the motion model to
                    # extrapolate with no evidence, which costs accuracy and
                    # buys nothing: the displayed boxes are already carried
                    # forward between passes.
                    #
                    # Each stage below is separately guarded, and each depends
                    # on the previous one having produced something. A failed
                    # detection therefore skips the whole chain for this frame
                    # rather than feeding the tracker last frame's boxes as if
                    # they were new - which would corrupt identity rather than
                    # just lose a frame.
                    if tracker is not None and detected is not None:
                        tracked = stages.guard("tracking").run(
                            tracker.update, latest_detection, frame=frame
                        )
                    else:
                        tracked = None
                    if tracked is not None:
                        latest_tracking = tracked
                        track_latency.observe(latest_tracking.tracking_ms)
                        tracking_steps += 1
                        if latest_tracking.tracks:
                            log.debug(
                                "tracks",
                                extra={
                                    "vantage_fields": {"summary": latest_tracking.describe()}
                                },
                            )
                        if state_estimator is not None:
                            latest_state = stages.guard("state").run(
                                state_estimator.update,
                                latest_tracking,
                                default=latest_state,
                            )
                        # Pose counts tracking steps rather than delivered
                        # frames: its interval is relative to the frames the
                        # tracker actually advanced on, so the two intervals
                        # compose instead of interfering.
                        if estimator is not None and (
                            (tracking_steps - 1) % config.pose.interval == 0
                        ):
                            posed = stages.guard("pose").run(
                                estimator.estimate, frame, latest_tracking
                            )
                            if posed is not None:
                                latest_pose = posed
                                pose_steps += 1
                                if latest_pose.poses:
                                    pose_latency.observe(
                                        latest_pose.total_ms / len(latest_pose)
                                    )
                                    log.debug(
                                        "poses",
                                        extra={
                                            "vantage_fields": {
                                                "summary": latest_pose.describe()
                                            }
                                        },
                                    )
                        # Spatial runs before activity so both see the same
                        # frame's poses, and neither depends on the other.
                        if spatial_engine is not None:
                            latest_spatial = stages.guard("spatial").run(
                                spatial_engine.update,
                                latest_tracking,
                                latest_pose,
                                latest_state,
                                default=latest_spatial,
                            )
                            if latest_spatial is not None and (
                                latest_spatial.relations or latest_spatial.crossings()
                            ):
                                log.debug(
                                    "spatial",
                                    extra={
                                        "vantage_fields": {"summary": latest_spatial.describe()}
                                    },
                                )
                        # Events last: they are a reduction over everything
                        # above, so they need this frame's outputs, not the
                        # previous frame's.
                        if activity_engine is not None and latest_state is not None:
                            latest_activity = stages.guard("activity").run(
                                activity_engine.update,
                                latest_state,
                                latest_pose,
                                default=latest_activity,
                            )
                            notable = latest_activity.notable() if latest_activity else ()
                            if notable:
                                log.debug(
                                    "activity",
                                    extra={
                                        "vantage_fields": {
                                            "summary": "; ".join(e.describe() for e in notable)
                                        }
                                    },
                                )
                        if event_engine is not None:
                            latest_events = stages.guard("events").run(
                                event_engine.update,
                                latest_tracking,
                                latest_state,
                                latest_activity,
                                latest_spatial,
                                default=latest_events,
                            )
                            if latest_events is not None and latest_events.events:
                                events_raised += len(latest_events)
                        # Persistence last, and guarded: a full disk must lose
                        # rows, never frames.
                        if recorder is not None:
                            stages.guard("storage").run(
                                recorder.record,
                                state=latest_state,
                                pose=latest_pose,
                                activity=latest_activity,
                                spatial=latest_spatial,
                                events=latest_events,
                            )
                    # What one analysed frame actually cost, end to end. The
                    # governor divides this by the interval to get the cost per
                    # *delivered* frame, which is what has to fit the budget.
                    analysis_cost.observe((clock.monotonic() - now_monotonic) * 1000.0)
                else:
                    # Carry the last pass forward so the display stays populated
                    # between inferences, but mark it so it is drawn as stale.
                    detection_stale = latest_detection is not None

            # Overlays are rendered when *either* consumer wants them. A
            # headless run with a dashboard still needs the boxes drawn -
            # otherwise the browser shows raw video and the analysis is
            # invisible, which is the opposite of what was asked for.
            if config.display.enabled or live_feed is not None:
                image = (
                    hud.render(
                        frame.image,
                        stats,
                        frame.index,
                        detection=latest_detection,
                        engine=detector.info if detector else None,
                        tracking=latest_tracking,
                        entity_total=_entity_total(tracker),
                        pose=latest_pose,
                        state=latest_state,
                        activity=latest_activity,
                        spatial=latest_spatial,
                        events=latest_events,
                        stages=stages,
                        resources=latest_resources,
                    )
                    if hud_enabled
                    else frame.editable_copy()
                )
                # Tracks supersede raw detections on screen. Drawing both would
                # double every box, and once identity exists it is the more
                # informative of the two.
                if spatial_engine is not None and spatial_engine.zones:
                    # Scenery goes down first: a zone polygon drawn over a
                    # person's box would obscure the thing being analysed.
                    image = draw_zones(image, spatial_engine.zones, latest_spatial)
                if latest_tracking is not None:
                    image = draw_tracks(image, latest_tracking, stale=detection_stale)
                elif latest_detection is not None:
                    image = draw_detections(image, latest_detection, stale=detection_stale)
                # Skeletons go on top of boxes, not instead of them: the box is
                # what carries the entity id and the dashed/solid distinction.
                if latest_pose is not None:
                    image = draw_poses(
                        image,
                        latest_pose,
                        min_confidence=config.pose.min_keypoint_confidence,
                    )
                if latest_spatial is not None and latest_tracking is not None:
                    image = draw_relations(image, latest_spatial, latest_tracking)
                if latest_activity is not None and latest_tracking is not None:
                    image = draw_activities(image, latest_activity, latest_tracking)
                if live_feed is not None:
                    live_feed.publish(
                        image,
                        _live_snapshot(
                            frame,
                            stats,
                            latest_state,
                            latest_pose,
                            latest_activity,
                            latest_spatial,
                            latest_events,
                            stages,
                        ),
                    )
                # Guarded rather than skipped with `continue`: the shutdown
                # check, the stats log and the resource sample all come after
                # this block, and a headless run with a dashboard needs every
                # one of them. Written as a continue first, which silently
                # disabled all three.
                if config.display.enabled:
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

            stats_interval = config.app.stats_interval_s
            if stats_interval and clock.monotonic() - last_stats_log >= stats_interval:
                last_stats_log = clock.monotonic()
                _log_stats(stats)

            resource_interval = config.app.resource_interval_s
            if resource_interval and clock.monotonic() - last_resource_log >= resource_interval:
                last_resource_log = clock.monotonic()
                latest_resources = resources.sample()
                log.info("resources", extra={"vantage_fields": latest_resources.to_dict()})
        else:
            reason = (
                "frame limit reached"
                if config.ingest.max_frames is not None
                and stats.frames_delivered >= config.ingest.max_frames
                else "source ended"
            )
        # Sampled before teardown, deliberately. Taken after it, the reading
        # includes the models being released and reports a *negative* growth -
        # measured at -87 MB on a short run, which is true of the process and
        # says nothing about whether the run leaked.
        if config.app.resource_interval_s:
            final_resources = resources.total()
    finally:
        pipeline.close()
        if owns_sink:
            sink.close()
        if owns_engine and detector is not None:
            detector.close()
        if estimator is not None:
            estimator.close()
        if store_writer is not None:
            # Flushed before the store closes, or the last batch is lost on
            # every clean shutdown - which would be the most reproducible data
            # loss the system could have.
            store_writer.close()
        if dashboard is not None:
            dashboard.stop()
        if store is not None:
            _prune_store(store, config)
            store.close()

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
        pose_steps=pose_steps,
        pose_summary=_pose_summary(estimator, pose_latency, latest_pose, pose_steps),
        state_summary=_state_summary(latest_state),
        activity_summary=_activity_summary(latest_activity),
        spatial_summary=_spatial_summary(spatial_engine, latest_spatial),
        stage_health=stages.to_dict(),
        adaptive=(governor.stats.to_dict() if governor is not None else {}),
        events_raised=events_raised,
        events_summary=(dict(event_engine.stats()) if event_engine is not None else {}),
        storage_summary=(store_writer.stats.to_dict() if store_writer is not None else {}),
        resources=(final_resources.to_dict() if final_resources is not None else {}),
    )
    log.info("run complete", extra={"vantage_fields": {"summary": result.summary()}})
    return result


def _build_engine(config: VantageConfig) -> DetectionEngine | None:
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


def _build_pose_engine(config: VantageConfig) -> PoseEngine | None:
    """Construct the pose estimator, or ``None`` when pose is disabled.

    Built before the source opens, for the same reason the detector is: a
    missing model should fail before a camera is acquired.
    """
    if not config.pose.enabled:
        return None
    from vantage.pose.factory import pose_engine_from_config

    estimator = pose_engine_from_config(config.pose)
    estimator.warmup(config.detection.warmup)
    return estimator


def _build_state_estimator(config: VantageConfig) -> StateEstimator | None:
    """Construct the entity-state estimator, or ``None`` when it is off."""
    if not (config.state.enabled and config.tracking.enabled):
        return None
    from vantage.state import StateEstimator, StateParams

    return StateEstimator(
        StateParams(
            moving_above=config.state.moving_above,
            stationary_below=config.state.stationary_below,
            min_state_s=config.state.min_state_s,
            min_age_s=config.state.min_age_s,
        )
    )


def _build_activity_engine(config: VantageConfig) -> ActivityEngine | None:
    """Construct the activity recogniser, or ``None`` when it is off."""
    if not (config.activity.enabled and config.tracking.enabled and config.state.enabled):
        return None
    from vantage.activity.engine import build_activity_engine

    return build_activity_engine(config.activity)


def _build_governor(config: VantageConfig) -> LoadGovernor | None:
    """Construct the load governor, or ``None`` when it should not run.

    Live sources only. A recorded file has no deadline - analysing it slowly
    gives the same answer as analysing it quickly - so shedding load there would
    discard information in exchange for nothing. Whether the source is live is
    known only after it opens, so this returns the governor and the run loop
    stops consulting it if the source turns out to be recorded.
    """
    if not (config.app.adaptive.enabled and config.detection.enabled):
        return None
    adaptive = config.app.adaptive
    return LoadGovernor(
        base_interval=config.detection.interval,
        params=GovernorParams(
            headroom=adaptive.headroom,
            max_interval=adaptive.max_interval,
            raise_after_s=adaptive.raise_after_s,
            lower_after_s=adaptive.lower_after_s,
        ),
    )


def _build_dashboard(config: VantageConfig, store: Any) -> tuple[Any, Any]:
    """Start the dashboard, or return ``(None, None)`` when it is disabled.

    The live feed is created here rather than inside the server because the run
    loop publishes into it and the server reads from it; neither owns the other.
    """
    if not config.dashboard.enabled:
        return None, None
    from vantage.dashboard.factory import build_dashboard
    from vantage.dashboard.live import LiveFeed

    feed = LiveFeed(
        jpeg_quality=config.dashboard.jpeg_quality, max_width=config.dashboard.max_width
    )
    server = build_dashboard(
        config.dashboard,
        store=store,
        feed=feed,
        camera_id=config.source.id or "camera_01",
    )
    url = server.start()
    log.info("dashboard started", extra={"vantage_fields": {"url": url}})
    return feed, server


def _live_snapshot(
    frame: Frame,
    stats: PipelineStats,
    state: Any,
    pose: Any,
    activity: Any,
    spatial: Any,
    events: Any,
    stage_registry: Any,
) -> Any:
    """Assemble what the dashboard shows about right now.

    Built per displayed frame, so it stays small: names and short strings, never
    keypoint arrays. The browser polls this once a second; the picture comes
    down the MJPEG stream instead.
    """
    from vantage.dashboard.live import LiveSnapshot

    postures = {p.track_id: p.posture.value for p in pose} if pose is not None else {}
    activities = (
        {e.track_id: [o.activity.value for o in e] for e in activity}
        if activity is not None
        else {}
    )
    zones = {e.track_id: list(e.zone_names) for e in spatial} if spatial is not None else {}

    entities = []
    if state is not None:
        for entity in state:
            entities.append(
                {
                    "entity_id": entity.entity_id,
                    "label": entity.label,
                    "motion": entity.motion.value,
                    "speed": round(entity.speed, 3),
                    "posture": postures.get(entity.track_id),
                    "activities": activities.get(entity.track_id, []),
                    "zones": zones.get(entity.track_id, []),
                }
            )

    return LiveSnapshot(
        frame_index=frame.index,
        captured_at=frame.capture_wall,
        entities=tuple(entities),
        events=tuple(
            {
                "timestamp": event.capture_wall,
                "severity": event.severity.value,
                "summary": event.summary,
                "rule": event.rule,
                "zone": event.zone,
            }
            for event in (events or ())
        ),
        stats={
            "fps": round(stats.delivery_fps, 1),
            "dropped": stats.frames_dropped,
            "source": stats.source_id,
        },
        health=(stage_registry.to_dict() if stage_registry is not None else {}),
    )


def _build_storage(
    config: VantageConfig,
) -> tuple[Any, Any, Any]:
    """Open the store and its writer, or a triple of ``None`` when disabled."""
    if not config.storage.enabled:
        return None, None, None
    from vantage.storage.factory import build_storage

    return build_storage(config.storage, camera_id=config.source.id or "camera_01")


def _prune_store(store: Any, config: VantageConfig) -> None:
    """Apply retention at shutdown.

    At shutdown rather than continuously: a DELETE over a large table takes a
    lock, and taking one mid-run to reclaim space is how a storage subsystem
    starts costing frames. A run that never ends is a real case, so the CLI
    exposes 'vantage history prune' for it.
    """
    import time

    storage = config.storage
    now = time.time()
    try:
        removed: dict[str, int] = {}
        if storage.retention_days:
            removed.update(store.prune(now - storage.retention_days * 86400.0))
        if storage.event_retention_days:
            # Events are kept longer, so they are pruned against their own
            # horizon rather than the observation one.
            cutoff = now - storage.event_retention_days * 86400.0
            connection = store._require()
            cursor = connection.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            removed["events"] = removed.get("events", 0) + cursor.rowcount
        if any(removed.values()):
            log.info("retention applied", extra={"vantage_fields": removed})
    except Exception as exc:
        log.warning(
            "retention failed",
            extra={"vantage_fields": {"error": f"{type(exc).__name__}: {exc}"}},
        )


def _build_event_engine(config: VantageConfig) -> EventEngine | None:
    """Construct the event engine, or ``None`` when it is off."""
    if not (config.events.enabled and config.tracking.enabled):
        return None
    from vantage.events.engine import build_event_engine

    return build_event_engine(config.events)


def _build_spatial_engine(config: VantageConfig) -> SpatialEngine | None:
    """Construct the spatial analyser, or ``None`` when it is off."""
    if not (config.spatial.enabled and config.tracking.enabled):
        return None
    from vantage.spatial.engine import build_spatial_engine

    return build_spatial_engine(config.spatial)


def _spatial_summary(
    engine: SpatialEngine | None, latest: SpatialResult | None
) -> dict[str, Any]:
    if engine is None or latest is None:
        return {}
    return {
        "zones": len(engine.zones),
        "occupancy": latest.occupancy(),
        "relations": latest.counts(),
        "state_available": latest.state_available,
    }


def _activity_summary(latest: ActivityResult | None) -> dict[str, Any]:
    if latest is None:
        return {}
    return {
        "entities": len(latest),
        "pose_available": latest.pose_available,
        "counts": latest.counts(),
    }


def _pose_summary(
    estimator: PoseEngine | None,
    latency: LatencyTracker,
    latest: PoseResult | None,
    passes: int,
) -> dict[str, Any]:
    if estimator is None or not passes:
        return {}
    return {
        "model": estimator.info.model,
        "backend": estimator.info.backend,
        "device": estimator.info.device,
        "license": estimator.info.license,
        "keypoints": estimator.info.num_keypoints,
        "passes": passes,
        "people": len(latest) if latest else 0,
        "skipped": latest.skipped if latest else 0,
        "mean_ms_per_person": round(latency.mean, 2),
    }


def _state_summary(latest: StateResult | None) -> dict[str, Any]:
    if latest is None:
        return {}
    return {
        "entities": len(latest),
        "moving": len(latest.moving()),
        "counts": latest.counts(),
    }


def _detection_summary(
    detector: DetectionEngine | None, latency: LatencyTracker, passes: int
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


def _entity_total(tracker: Tracker | None) -> int:
    """Distinct entities published so far, or 0 for a tracker that cannot say.

    The count comes from the tracker rather than from a set accumulated here,
    so a run lasting weeks does not carry every identifier it ever issued.
    """
    stats = getattr(tracker, "stats", None)
    if not callable(stats):
        return 0
    return int(stats().get("entities_published", 0))


def _tracking_summary(
    tracker: Tracker | None,
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
