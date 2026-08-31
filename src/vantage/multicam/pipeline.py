"""Multi-Camera Concurrent Pipeline with Re-ID, Radar Map, Video Evidence, Threats & Events."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from vantage.config.schema import SourceConfig
from vantage.core.logging import get_logger
from vantage.dashboard.live import LiveFeed, LiveSnapshot
from vantage.entity.manager import EntityContextManager
from vantage.events.contracts import EventCandidate, Severity
from vantage.events.engine import EventEngine
from vantage.events.geofence import GeofenceEngine
from vantage.events.threats import ThreatDetectionEngine
from vantage.events.zone_registry import ZoneRegistry
from vantage.ingestion.base import FrameSource
from vantage.ingestion.registry import create_source
from vantage.multicam.evidence import VideoEvidenceRecorder
from vantage.multicam.journey import FacilityJourneyTracker
from vantage.multicam.radar import FacilityRadarMap
from vantage.multicam.reid import CrossCameraReIDTracker
from vantage.perception.contracts import BoundingBox, DetectionResult
from vantage.perception.engine import DetectionEngine, build_engine
from vantage.pose.engine import PoseEngine
from vantage.pose.factory import build_pose_engine
from vantage.spatial.geometry.coordinates import get_entity_foot_point
from vantage.spatial.twin import FacilitySpatialTwin
from vantage.state.contracts import StateResult
from vantage.state.estimator import StateEstimator
from vantage.storage.sqlite_store import SqliteStore
from vantage.tracking.bytetrack import ByteTracker, TrackerParams
from vantage.viz.overlay import draw_detections, draw_poses, draw_tracks

log = get_logger(__name__)

# Valid domain classes (suppresses out-of-domain noise like kite, toilet, tennis racket)
VALID_DOMAIN_CLASSES = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "backpack",
        "cell_phone",
        "bottle",
        "chair",
    }
)


def _letterbox_tile(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale frame into target tile while preserving aspect ratio with clean dark padding."""
    tile = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    fh, fw = frame.shape[:2]
    if fw <= 0 or fh <= 0:
        return tile

    scale = min(target_w / fw, target_h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    tile[y_off : y_off + nh, x_off : x_off + nw] = resized
    return tile


@dataclass
class CameraWorker:
    """Manages ingestion and inference for one camera stream."""

    camera_id: str
    source_path: str
    source: FrameSource
    tracker: ByteTracker
    state_estimator: StateEstimator
    live_feed: LiveFeed
    latest_annotated_frame: np.ndarray | None = None
    latest_state_result: StateResult | None = None
    latest_local_to_global: dict[int, str] = field(default_factory=dict)
    latest_fps: float = 20.0
    frame_count: int = 0
    is_running: bool = True
    thread: threading.Thread | None = None


class MultiCameraPipeline:
    """Orchestrates concurrent multi-camera feeds with Re-ID, Radar Digital Twin, Evidence & Threats."""

    def __init__(
        self,
        camera_sources: dict[str, str],  # camera_id -> video_path / rtsp_url / webcam:0
        *,
        store: SqliteStore | None = None,
        zone_registry: ZoneRegistry | None = None,
        model: str = "yolox-nano",
        conf_threshold: float = 0.40,
        enable_pose: bool = True,
        grid_width: int = 1280,
        grid_height: int = 720,
    ) -> None:
        self.camera_sources = camera_sources
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.store = store
        self.zone_registry = zone_registry or ZoneRegistry(store=self.store)
        self.geofence_engine = GeofenceEngine()
        self._zone_occupancies: dict[str, int] = {}
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._running = False

        # Shared detector and pose engine
        self._detector: DetectionEngine = build_engine(
            model=model,
            backend="openvino",
            device="cpu",
            confidence=conf_threshold,
        )
        self._pose_engine: PoseEngine | None = (
            build_pose_engine(
                model="rtmpose-s", backend="openvino", device="cpu", max_persons=6
            )
            if enable_pose
            else None
        )

        # Cross-Camera Re-ID, Radar Map, 3D Spatial Twin, Journeys, Threats, and Video Evidence
        self.reid_tracker = CrossCameraReIDTracker()
        self.radar_map = FacilityRadarMap()
        self.spatial_twin = FacilitySpatialTwin(zone_registry=self.zone_registry)
        self.journey_tracker = FacilityJourneyTracker()
        self.threat_engine = ThreatDetectionEngine()
        self.evidence_recorder = VideoEvidenceRecorder(output_dir="data/evidence")

        # Canonical Entity Intelligence & Unified Event Engine
        self.entity_manager = EntityContextManager()
        self.event_engine = EventEngine()

        # Transient Scene Graphs (Phase 17)
        from vantage.scene.graph import TransientSceneGraph

        self.scene_graphs: dict[str, TransientSceneGraph] = {}

        # Persistent Entity Relationship & Long-Horizon Intelligence (Phase 18)
        from vantage.relationship.service import RelationshipService

        self.relationship_service = RelationshipService(store=self.store)
        self.relationship_tracker = self.relationship_service.tracker

        # Incident Intelligence & Situational Reasoning (Phase 19)
        from vantage.incident.service import IncidentService

        self.incident_service = IncidentService(
            store=self.store,
            relationship_tracker=self.relationship_tracker,
        )

        # Active event tracking and history
        self._recent_events: list[dict[str, Any]] = []
        self._entity_last_camera: dict[
            str, tuple[str, float]
        ] = {}  # global_id -> (last_cam, last_wall_time)

        # Grid live feed for dashboard
        self.grid_feed = LiveFeed(jpeg_quality=75, max_width=grid_width)

        # Initialize workers per camera
        self.workers: dict[str, CameraWorker] = {}
        for cam_id, src_path in camera_sources.items():
            source = create_source(SourceConfig(uri=src_path, id=cam_id, loop=True))
            self.scene_graphs[cam_id] = TransientSceneGraph(camera_id=cam_id)
            worker = CameraWorker(
                camera_id=cam_id,
                source_path=src_path,
                source=source,
                tracker=ByteTracker(params=TrackerParams(min_hits=2, max_lost_s=1.5)),
                state_estimator=StateEstimator(),
                live_feed=LiveFeed(jpeg_quality=70, max_width=640),
            )
            self.workers[cam_id] = worker

    def start(self) -> None:
        """Start concurrent ingestion threads for all cameras."""
        self._running = True
        for cam_id, worker in self.workers.items():
            worker.source.open()
            t = threading.Thread(
                target=self._camera_worker_loop,
                args=(worker,),
                daemon=True,
                name=f"Worker-{cam_id}",
            )
            t.start()

        # Start grid compositor thread
        grid_thread = threading.Thread(
            target=self._grid_compositor_loop,
            daemon=True,
            name="Grid-Compositor",
        )
        grid_thread.start()
        log.info(f"Multi-Camera Pipeline started with {len(self.workers)} camera feeds.")

    def stop(self) -> None:
        """Stop all camera feeds."""
        self._running = False
        for worker in self.workers.values():
            worker.is_running = False
            worker.source.close()

    def _emit_event(
        self,
        rule: str,
        severity: str,
        summary: str,
        entity_id: str,
        camera_id: str,
        wall_time: float,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        """Process event candidate through the unified EventEngine policy."""
        candidate = EventCandidate(
            rule=rule,
            severity=severity,
            summary=summary,
            entity_id=entity_id,
            camera_id=camera_id,
            wall_time=wall_time,
            evidence=evidence or {},
            zone=camera_id.upper(),
        )

        event = self.event_engine.evaluate_candidate(candidate)
        if event is None:
            return

        ev_id = f"ev_{int(wall_time * 1000)}_{len(self._recent_events)}"
        worker = self.workers.get(camera_id)
        frame_index = worker.frame_count if worker is not None else 0
        fps = worker.latest_fps if worker is not None and worker.latest_fps > 0 else 0.0
        ev_dict: dict[str, Any] = {
            "id": ev_id,
            "timestamp": wall_time,
            "camera_id": camera_id,
            "rule": event.rule,
            "severity": event.severity.value,
            "summary": event.summary,
            "entity_id": event.entity_id,
            "related_id": event.related_id,
            "zone": event.zone or camera_id.upper(),
            "frame_index": frame_index,
            "elapsed_s": round(frame_index / fps, 2) if fps else 0.0,
            "evidence": dict(event.evidence),
        }

        # Auto-generate video evidence clip for ALERT or NOTICE events
        if event.severity in (Severity.ALERT, Severity.NOTICE):
            clip_url = self.evidence_recorder.save_clip(ev_id, camera_id)
            if clip_url:
                ev_dict["evidence"]["clip_url"] = clip_url

        # Record event on entity context
        if entity_id:
            ctx = self.entity_manager.get_by_global(entity_id)
            if ctx:
                ctx.add_event(ev_dict)

        with self._lock:
            self._recent_events.insert(0, ev_dict)
            if len(self._recent_events) > 100:
                self._recent_events.pop()

        if self.store:
            try:
                self.store.write_events([ev_dict])
            except Exception as exc:
                log.debug(f"Store write event error: {exc}")

        # Ingest into Situational Incident Intelligence (Phase 19)
        if hasattr(self, "incident_service") and self.incident_service:
            try:
                _target_inc, _decision, inc_cands = self.incident_service.ingest_event(
                    ev_dict, now=wall_time
                )
                for icand in inc_cands:
                    # Emit incident-level event candidate without recursion loop
                    self.event_engine.evaluate_candidate(icand)
            except Exception as exc:
                log.debug(f"Incident ingestion error: {exc}")

    def _camera_worker_loop(self, worker: CameraWorker) -> None:
        """Ingestion and analysis loop for one camera stream with real-time pacing."""
        frame_idx = 0
        latest_det = None
        target_fps = 20.0
        frame_interval_s = 1.0 / target_fps
        t_next = time.perf_counter()
        t_fps_calc = time.perf_counter()
        fps_frame_count = 0

        while self._running and worker.is_running:
            now = time.perf_counter()
            if now < t_next:
                time.sleep(max(0.001, min(0.01, t_next - now)))
                continue
            t_next = now + frame_interval_s

            raw_frame = worker.source.read()
            if raw_frame is None:
                # Loop recorded file if finished
                worker.source.close()
                worker.source.open()
                continue

            frame_idx += 1
            fps_frame_count += 1
            if now - t_fps_calc >= 1.0:
                worker.latest_fps = round(fps_frame_count / (now - t_fps_calc), 1)
                fps_frame_count = 0
                t_fps_calc = now

            # Push raw frame to video evidence rolling buffer
            self.evidence_recorder.push_frame(
                worker.camera_id, raw_frame.image, raw_frame.capture_wall
            )

            # 1. Detection (Interleaved every 2nd frame; filter domain classes)
            if frame_idx % 2 == 0 or latest_det is None:
                with self._infer_lock:
                    raw_det_res = self._detector.detect(raw_frame)
                    filtered_dets = [
                        d
                        for d in raw_det_res.detections
                        if d.label.lower().replace(" ", "_") in VALID_DOMAIN_CLASSES
                    ]
                    latest_det = DetectionResult(
                        detections=tuple(filtered_dets),
                        source_id=raw_det_res.source_id,
                        frame_index=raw_det_res.frame_index,
                        capture_wall=raw_det_res.capture_wall,
                        frame_size=raw_det_res.frame_size,
                        model=raw_det_res.model,
                        backend=raw_det_res.backend,
                        inference_ms=raw_det_res.inference_ms,
                    )
            det_res = latest_det

            # 2. Tracking
            track_res = worker.tracker.update(det_res, frame=raw_frame)
            confirmed_tracks = [t for t in track_res.tracks if getattr(t, "is_confirmed", True)]
            dets = (
                list(det_res.detections) if det_res and hasattr(det_res, "detections") else []
            )

            # 3. Cross-Camera Re-ID Association
            local_to_global = self.reid_tracker.update_camera(
                worker.camera_id, raw_frame, track_res.tracks
            )

            # 4. Pose
            pose_res = None
            if self._pose_engine and track_res.tracks and frame_idx % 3 == 0:
                with self._infer_lock:
                    pose_res = self._pose_engine.estimate(raw_frame, track_res)

            # 5. State, Journeys, Radar Map, Scene Graphs & Events
            state_res = worker.state_estimator.update(track_res)
            track_by_id = {t.track_id: t for t in track_res.tracks}
            obs_batch = []
            h_f, w_f = raw_frame.image.shape[:2]

            # 5a. Transient Scene Graph (Phase 17)
            scene_graph = self.scene_graphs.get(worker.camera_id)
            if scene_graph:
                _, scene_candidates = scene_graph.update(
                    tracks=confirmed_tracks,
                    raw_detections=dets,
                    now=raw_frame.capture_wall,
                    frame_width=w_f,
                    frame_height=h_f,
                )
                for cand in scene_candidates:
                    self._emit_event(
                        rule=cand.rule,
                        severity=cand.severity,
                        summary=cand.summary,
                        entity_id=cand.entity_id or "",
                        camera_id=cand.camera_id,
                        wall_time=cand.wall_time,
                        evidence=cand.evidence,
                    )

            # 5b. Persistent Entity Relationships & Long-Horizon Intelligence (Phase 18)
            if hasattr(self, "relationship_tracker") and len(confirmed_tracks) >= 2:
                active_entity_list = []
                for ct in confirmed_tracks:
                    c_gid = local_to_global.get(ct.track_id, ct.entity_id)
                    cx = (ct.box.x1 + ct.box.x2) / (2.0 * max(1, w_f))
                    cy = ct.box.y2 / max(1, h_f)
                    active_entity_list.append((c_gid, cx, cy, 0.5, None))
                rel_candidates = self.relationship_tracker.process_frame(
                    camera_id=worker.camera_id,
                    active_entities=active_entity_list,
                    scene_graph=scene_graph.last_snapshot if scene_graph else None,
                    entity_trajectories=None,
                    now=raw_frame.capture_wall,
                )
                for rcand in rel_candidates:
                    self._emit_event(
                        rule=rcand.rule,
                        severity=rcand.severity,
                        summary=rcand.summary,
                        entity_id=rcand.entity_id or "",
                        camera_id=rcand.camera_id,
                        wall_time=rcand.wall_time,
                        evidence=rcand.evidence,
                    )

            for st in state_res.states:
                gid = local_to_global.get(st.track_id, f"local_{st.track_id}")
                t_obj = track_by_id.get(st.track_id)
                b_box = t_obj.box if t_obj else BoundingBox(0.0, 0.0, 10.0, 10.0)

                # Project to 2D Facility Floorplan Radar
                self.radar_map.project_entity(
                    camera_id=worker.camera_id,
                    global_id=gid,
                    label=st.label,
                    box=b_box,
                    frame_width=w_f,
                    frame_height=h_f,
                    motion="walking" if st.speed > 0.15 else "stationary",
                    speed=round(st.speed, 2),
                    activity="walking" if st.speed > 0.15 else "idle",
                    wall_time=raw_frame.capture_wall,
                )

                # Project to 3D Spatial Digital Twin
                foot_pt = get_entity_foot_point(b_box, w_f, h_f, normalize=True)
                self.spatial_twin.update_entity_3d(
                    global_id=gid,
                    camera_id=worker.camera_id,
                    label=st.label,
                    foot_point=foot_pt,
                    speed=st.speed,
                    bearing_deg=st.bearing_deg,
                    motion="walking" if st.speed > 0.15 else "stationary",
                    posture=getattr(st, "posture", "standing"),
                    wall_time=raw_frame.capture_wall,
                )

                # Update Canonical Entity Intelligence
                ctx = self.entity_manager.get_or_create(
                    global_id=gid,
                    label=st.label,
                    camera_id=worker.camera_id,
                    track_id=st.track_id,
                    box=b_box,
                    wall_time=raw_frame.capture_wall,
                )
                ctx.update_spatial(
                    camera_id=worker.camera_id,
                    box=b_box,
                    foot_point=foot_pt.to_tuple(),
                    wall_time=raw_frame.capture_wall,
                )
                ctx.update_kinematics(
                    speed_h_s=st.speed,
                    motion_state="walking" if st.speed > 0.15 else "stationary",
                    posture=getattr(st, "posture", "standing"),
                    bearing_deg=st.bearing_deg,
                )
                ctx.update_activity(
                    activities=["walking"] if st.speed > 0.15 else ["idle"],
                    primary="walking" if st.speed > 0.15 else "idle",
                    confidence=1.0,
                    evidence=f"Speed {round(st.speed, 2)} h/s",
                    wall_time=raw_frame.capture_wall,
                )

                # Update Persistent Relationships on Canonical Entity
                if hasattr(self, "relationship_tracker"):
                    top_rels = self.relationship_tracker.get_relationships_for_entity(
                        gid, now=raw_frame.capture_wall
                    )
                    if top_rels:
                        ctx.update_persistent_relationships(
                            related_entities=[r.other_entity(gid) for r in top_rels[:5]],
                            relationship_types=[
                                r.primary_derived_pattern.value
                                if r.primary_derived_pattern
                                else "co_occurrence"
                                for r in top_rels[:5]
                            ],
                            active_relationships=[r.to_dict() for r in top_rels[:5]],
                        )

                # Journey Timeline
                self.journey_tracker.record_sighting(
                    global_id=gid,
                    label=st.label,
                    camera_id=worker.camera_id,
                    wall_time=raw_frame.capture_wall,
                    box=b_box,
                    activity="walking" if st.speed > 0.15 else "stationary",
                )

                # Verified Cross-Camera Transition Event (requires genuine departure & transit)
                prev_record = self._entity_last_camera.get(gid)
                if prev_record:
                    prev_cam, prev_t = prev_record
                    if (
                        prev_cam != worker.camera_id
                        and (raw_frame.capture_wall - prev_t) >= 2.0
                    ):
                        self._emit_event(
                            rule="cross_camera_handover",
                            severity="notice",
                            summary=f"{gid.upper()} transitioned from {prev_cam.upper()} to {worker.camera_id.upper()}",
                            entity_id=gid,
                            camera_id=worker.camera_id,
                            wall_time=raw_frame.capture_wall,
                            evidence={
                                "previous_camera": prev_cam,
                                "transit_time_s": round(raw_frame.capture_wall - prev_t, 1),
                            },
                        )
                self._entity_last_camera[gid] = (worker.camera_id, raw_frame.capture_wall)

                # Verified Loitering Event (minimum 30 seconds stationary dwell)
                if st.dwell_s > 30.0 and st.speed <= 0.05:
                    self._emit_event(
                        rule="loitering",
                        severity="notice",
                        summary=f"{gid.upper()} loitering in {worker.camera_id.upper()} (dwell: {st.dwell_s:.0f}s)",
                        entity_id=gid,
                        camera_id=worker.camera_id,
                        wall_time=raw_frame.capture_wall,
                        evidence={"dwell_s": round(st.dwell_s, 1)},
                    )

                # Collect observation for SQLite store
                if frame_idx % 10 == 0:
                    obs_batch.append(
                        {
                            "timestamp": raw_frame.capture_wall,
                            "camera_id": worker.camera_id,
                            "entity_id": gid,
                            "identity": gid.upper(),
                            "entity_type": st.label,
                            "motion": "moving" if st.speed > 0.15 else "stationary",
                            "speed": round(st.speed, 2),
                            "posture": getattr(st, "posture", "standing"),
                            "zones": [worker.camera_id.upper()],
                            "activities": ["walking" if st.speed > 0.15 else "idle"],
                            "frame_index": frame_idx,
                            "elapsed_s": round(raw_frame.capture_wall % 1000, 2),
                        }
                    )

            # Check for Security Threats (Tailgating / Wrong-Way)
            threats = self.threat_engine.evaluate_threats(
                camera_id=worker.camera_id,
                states=state_res.states,
                wall_time=raw_frame.capture_wall,
            )
            for thr in threats:
                self._emit_event(
                    rule=thr["rule"],
                    severity=thr["severity"],
                    summary=thr["summary"],
                    entity_id=thr["entity_id"],
                    camera_id=worker.camera_id,
                    wall_time=raw_frame.capture_wall,
                    evidence=thr.get("evidence"),
                )

            # Check Dynamic Polygonal Geofences (Immutable In-Memory Snapshot)
            zone_snapshot = self.zone_registry.get_snapshot()
            h_img, w_img = raw_frame.image.shape[:2]
            track_boxes = {t.track_id: t.box for t in track_res.tracks}
            breaches, occ_counts = self.geofence_engine.evaluate_snapshot(
                snapshot=zone_snapshot,
                camera_id=worker.camera_id,
                states=state_res.states,
                frame_width=w_img,
                frame_height=h_img,
                wall_time=raw_frame.capture_wall,
                boxes=track_boxes,
            )
            for br in breaches:
                self._emit_event(
                    rule=br.rule,
                    severity=br.severity,
                    summary=br.summary,
                    entity_id=br.entity_id,
                    camera_id=worker.camera_id,
                    wall_time=raw_frame.capture_wall,
                    evidence=br.evidence,
                )
            with self._lock:
                self._zone_occupancies.update(occ_counts)

            if self.store and obs_batch:
                try:
                    self.store.write_observations(obs_batch)
                except Exception as exc:
                    log.debug(f"Store write observations error: {exc}")

            # 6. Annotate Frame
            canvas = raw_frame.image.copy()

            # Draw Active Geofence Polygons & Occupancy Badges
            active_cam_zones = zone_snapshot.get_zones_for_camera(worker.camera_id)
            if active_cam_zones:
                overlay = canvas.copy()
                for z in active_cam_zones:
                    pts = np.array(
                        [[int(p.x * w_img), int(p.y * h_img)] for p in z.polygon.vertices],
                        dtype=np.int32,
                    )
                    b_col = (
                        (0, 0, 255)
                        if z.severity == "alert"
                        else (0, 200, 255)
                        if z.severity == "notice"
                        else (255, 200, 0)
                    )
                    cv2.fillPoly(overlay, [pts], b_col)
                    cv2.polylines(canvas, [pts], True, b_col, 2)

                    # Zone Badge
                    occ = self._zone_occupancies.get(z.zone_id, 0)
                    tag = f"{z.name.upper()} [{occ}]"
                    lx = max(4, min(w_img - 160, int(pts[0][0])))
                    ly = max(48, min(h_img - 20, int(pts[0][1])))
                    cv2.rectangle(
                        canvas, (lx, ly - 16), (lx + len(tag) * 8 + 8, ly + 2), (15, 15, 15), -1
                    )
                    cv2.putText(
                        canvas,
                        tag,
                        (lx + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 255, 255),
                        1,
                    )
                cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

            draw_detections(canvas, det_res)
            draw_tracks(canvas, track_res)
            if pose_res:
                draw_poses(canvas, pose_res)

            # Draw Global ID banner & Camera Watermark
            for t in track_res.tracks:
                gid = local_to_global.get(t.track_id, f"ID_{t.track_id}")
                x1, y1 = int(t.box.x1), int(t.box.y1)
                badge_text = f"{gid.upper()}"

                # Minimalistic graphite badge
                cv2.rectangle(
                    canvas,
                    (x1, max(0, y1 - 18)),
                    (x1 + len(badge_text) * 8 + 8, max(18, y1)),
                    (20, 25, 30),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (x1, max(0, y1 - 18)),
                    (x1 + len(badge_text) * 8 + 8, max(18, y1)),
                    (50, 60, 70),
                    1,
                )
                cv2.putText(
                    canvas,
                    badge_text,
                    (x1 + 4, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 210, 220),
                    1,
                    cv2.LINE_AA,
                )

            # Minimal Camera Header Watermark (Bottom left instead of huge top bar)
            cam_text = f"{worker.camera_id.upper()} | {worker.latest_fps:.1f} FPS"
            cv2.putText(
                canvas,
                cam_text,
                (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 160, 170),
                1,
                cv2.LINE_AA,
            )

            with self._lock:
                worker.latest_annotated_frame = canvas
                worker.latest_state_result = state_res
                worker.latest_local_to_global = local_to_global
                worker.frame_count = frame_idx

    def _grid_compositor_loop(self) -> None:
        """Dynamically arranges 1 to N cameras into a clean letterboxed matrix grid."""
        grid_frame_idx = 0
        while self._running:
            grid_frame_idx += 1
            n = len(self.workers)
            if n <= 0:
                time.sleep(0.05)
                continue

            # Compute dynamic grid geometry for 1 to N cameras
            if n == 1:
                cols, rows = 1, 1
            elif n == 2:
                cols, rows = 2, 1
            elif n in (3, 4):
                cols, rows = 2, 2
            elif n in (5, 6):
                cols, rows = 3, 2
            elif n in (7, 8, 9):
                cols, rows = 3, 3
            else:
                cols = int(math.ceil(math.sqrt(n)))
                rows = int(math.ceil(n / cols))

            tile_w = self.grid_width // max(cols, 1)
            tile_h = self.grid_height // max(rows, 1)

            grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
            with self._lock:
                worker_snapshots = [
                    (
                        w.camera_id,
                        w.latest_annotated_frame,
                        w.latest_state_result,
                        w.latest_local_to_global,
                        w.latest_fps,
                    )
                    for w in self.workers.values()
                ]
                events_copy = list(self._recent_events)

            all_entities = []
            for idx, (cam_id, frame, state_res, local_to_global, _fps) in enumerate(
                worker_snapshots
            ):
                r = idx // cols
                c = idx % cols
                x1 = c * tile_w
                y1 = r * tile_h

                if frame is not None and frame.size > 0:
                    tile = _letterbox_tile(frame, tile_w, tile_h)
                    grid[y1 : y1 + tile_h, x1 : x1 + tile_w] = tile
                else:
                    cv2.rectangle(grid, (x1, y1), (x1 + tile_w, y1 + tile_h), (10, 12, 15), -1)
                    cv2.putText(
                        grid,
                        f"CONNECTING {cam_id.upper()}...",
                        (x1 + 20, y1 + tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (80, 90, 100),
                        1,
                        cv2.LINE_AA,
                    )

                if state_res:
                    for st in state_res.states:
                        gid = local_to_global.get(st.track_id, f"local_{st.track_id}")
                        all_entities.append(
                            {
                                "entity_id": gid,
                                "identity": gid.upper(),
                                "label": st.label,
                                "camera": cam_id,
                                "motion": "moving" if st.speed > 0.15 else "stationary",
                                "speed": round(st.speed, 2),
                                "posture": getattr(st, "posture", "standing"),
                                "activities": ["walking" if st.speed > 0.15 else "idle"],
                                "zones": [cam_id.upper()],
                            }
                        )

            # Publish Full Telemetry Snapshot to Dashboard with LIVE EVENTS and RADAR state
            snapshot = LiveSnapshot(
                frame_index=grid_frame_idx,
                captured_at=time.time(),
                entities=tuple(all_entities),
                events=tuple(events_copy[:15]),
                stats={
                    "fps": 24.0,
                    "source": f"{n}-CAM FACILITY GRID",
                    "active_cameras": n,
                    "total_entities": len(all_entities),
                },
                health={
                    "multi_ingestion": {"calls": grid_frame_idx * n, "failures": 0},
                    "perception": {"calls": grid_frame_idx * n, "failures": 0},
                    "tracking": {"calls": grid_frame_idx * n, "failures": 0},
                    "cross_camera_reid": {"calls": grid_frame_idx, "failures": 0},
                    "radar_digital_twin": {"calls": grid_frame_idx, "failures": 0},
                },
            )
            self.grid_feed.publish(grid, snapshot)
            time.sleep(0.04)  # 25 FPS compositing
