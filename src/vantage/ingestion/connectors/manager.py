"""Dynamic Camera Hotplug and Connector Manager for Vantage Pipeline."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from vantage.config.schema import SourceConfig
from vantage.core.logging import get_logger
from vantage.dashboard.live import LiveFeed
from vantage.ingestion.registry import create_source
from vantage.state.estimator import StateEstimator
from vantage.tracking.bytetrack import ByteTracker, TrackerParams

if TYPE_CHECKING:
    from vantage.multicam.pipeline import MultiCameraPipeline

log = get_logger(__name__)


class CameraConnectorManager:
    """Orchestrates dynamic physical camera connectors, streaming workers, and 3D twin mounts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def attach_camera(
        self,
        pipeline: MultiCameraPipeline,
        camera_id: str,
        name: str,
        uri: str,
        mount_x: float = 20.0,
        mount_y: float = 3.5,
        mount_z: float = 12.0,
        yaw_deg: float = 0.0,
        pitch_deg: float = -25.0,
        fov_deg: float = 70.0,
        range_m: float = 16.0,
        color: str = "#00e5ff",
    ) -> dict[str, Any]:
        """Dynamically instantiate and connect a new RTSP stream or USB webcam without restarting."""
        with self._lock:
            if camera_id in pipeline.workers:
                return {
                    "status": "error",
                    "error": f"Camera '{camera_id}' is already connected.",
                }

            try:
                from vantage.multicam.pipeline import CameraWorker

                source = create_source(SourceConfig(uri=uri, id=camera_id, loop=True))
                worker = CameraWorker(
                    camera_id=camera_id,
                    source_path=uri,
                    source=source,
                    tracker=ByteTracker(params=TrackerParams(min_hits=2, max_lost_s=1.5)),
                    state_estimator=StateEstimator(),
                    live_feed=LiveFeed(jpeg_quality=70, max_width=640),
                )

                # Register in pipeline workers map
                pipeline.workers[camera_id] = worker

                # Start worker thread if pipeline is running
                if getattr(pipeline, "_running", False):
                    worker.source.open()
                    t = threading.Thread(
                        target=pipeline._camera_worker_loop,
                        args=(worker,),
                        daemon=True,
                        name=f"Worker-{camera_id}",
                    )
                    worker.thread = t
                    t.start()

                # Give it a sector in the twin. Through add_camera rather than
                # straight into the mounts dict: a mount with no sector behind it
                # projects every entity onto the whole floor, which puts them
                # somewhere plausible and wrong.
                if getattr(pipeline, "spatial_twin", None) is not None:
                    pipeline.spatial_twin.add_camera(camera_id, name=name)

                log.info(f"Dynamically attached camera '{camera_id}' ({name}) at {uri}")
                return {
                    "status": "ok",
                    "action": "attached",
                    "camera_id": camera_id,
                    "name": name,
                    "uri": uri,
                    "total_cameras": len(pipeline.workers),
                }
            except Exception as exc:
                log.error(f"Failed to attach camera '{camera_id}': {exc}", exc_info=True)
                return {"status": "error", "error": str(exc)}

    def detach_camera(
        self,
        pipeline: MultiCameraPipeline,
        camera_id: str,
    ) -> dict[str, Any]:
        """Safely disconnect and detach a camera from pipeline and 3D digital twin."""
        with self._lock:
            worker = pipeline.workers.get(camera_id)
            if not worker:
                return {"status": "error", "error": f"Camera '{camera_id}' not found."}

            try:
                worker.is_running = False
                worker.source.close()
                if worker.thread is not None:
                    worker.thread.join(timeout=1.5)

                del pipeline.workers[camera_id]

                if getattr(pipeline, "spatial_twin", None) is not None:
                    pipeline.spatial_twin.remove_camera(camera_id)

                log.info(f"Dynamically detached camera '{camera_id}'")
                return {
                    "status": "ok",
                    "action": "detached",
                    "camera_id": camera_id,
                    "total_cameras": len(pipeline.workers),
                }
            except Exception as exc:
                log.error(f"Failed to detach camera '{camera_id}': {exc}", exc_info=True)
                return {"status": "error", "error": str(exc)}
