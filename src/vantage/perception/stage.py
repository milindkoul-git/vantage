"""Decoupled asynchronous inference stage.

Moves object detection to an independent worker thread behind a bounded queue,
reusing the backpressure discipline from :mod:`vantage.ingestion.buffer`.

Why this exists
----------------
In single-threaded mode, running inference inside the frame consumer loop forces
the pipeline frame rate to match detection throughput. If inference costs 40ms,
the display and tracking loop drop to 25 FPS even when the camera is delivering
at 60 FPS.

With :class:`AsyncInferenceStage`:
1. The consumer thread enqueues the latest frame without blocking (under LATEST policy).
2. The detector worker runs inference at its own pace on GPU / CPU.
3. The consumer thread retrieves the most recent available :class:`DetectionResult`
   to drive tracking and rendering at full capture rate.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vantage.config.schema import Backpressure
from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.logging import get_logger
from vantage.ingestion.buffer import FrameBuffer
from vantage.perception.contracts import DetectionResult

if TYPE_CHECKING:
    from vantage.core.frame import Frame
    from vantage.perception.engine import DetectionEngine, EngineInfo

log = get_logger(__name__)


@dataclass(slots=True)
class AsyncInferenceStats:
    """Telemetry for the async inference stage."""

    submitted: int = 0
    inferences_run: int = 0
    dropped: int = 0
    queue_high_water: int = 0
    mean_inference_ms: float = 0.0
    latest_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "inferences_run": self.inferences_run,
            "dropped": self.dropped,
            "queue_high_water": self.queue_high_water,
            "mean_inference_ms": self.mean_inference_ms,
            "latest_latency_ms": self.latest_latency_ms,
        }


class AsyncInferenceStage:
    """Runs a :class:`DetectionEngine` on a dedicated background thread."""

    def __init__(
        self,
        engine: DetectionEngine,
        *,
        queue_size: int = 2,
        policy: Backpressure = Backpressure.LATEST,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._engine = engine
        self._buffer = FrameBuffer(capacity=queue_size, policy=policy)
        self._clock = clock
        self._lock = threading.Lock()

        self._latest_result: DetectionResult | None = None
        self._result_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._running = False
        self._stats = AsyncInferenceStats()
        self._total_inference_time = 0.0

    @property
    def engine(self) -> DetectionEngine:
        return self._engine

    @property
    def info(self) -> EngineInfo:
        return self._engine.info

    @property
    def stats(self) -> AsyncInferenceStats:
        with self._lock:
            self._stats.dropped = self._buffer.dropped
            self._stats.queue_high_water = self._buffer.high_water
            return self._stats

    @property
    def queue_depth(self) -> int:
        return len(self._buffer)

    def start(self) -> None:
        """Start the background inference worker."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="vantage-inference-worker",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "async inference worker started",
            extra={
                "vantage_fields": {
                    "model": self.info.model,
                    "queue_size": self._buffer.capacity,
                }
            },
        )

    def submit(self, frame: Frame) -> bool:
        """Submit a frame for detection.

        Returns True if admitted, False if dropped or stage stopped.
        """
        if not self._running:
            return False
        with self._lock:
            self._stats.submitted += 1
        return self._buffer.put(frame)

    def get_latest_result(self) -> DetectionResult | None:
        """Return the most recent detection result without waiting."""
        with self._lock:
            return self._latest_result

    def wait_for_result(self, timeout_s: float | None = None) -> DetectionResult | None:
        """Wait for at least one detection result."""
        self._result_event.wait(timeout=timeout_s)
        with self._lock:
            return self._latest_result

    def _worker_loop(self) -> None:
        """Continuously pulls frames from buffer and executes detection."""
        while self._running:
            frame = self._buffer.get(timeout=0.1)
            if frame is None:
                continue

            try:
                start = self._clock.monotonic()
                result = self._engine.detect(frame)
                duration_ms = (self._clock.monotonic() - start) * 1000.0

                with self._lock:
                    self._latest_result = result
                    self._stats.inferences_run += 1
                    self._total_inference_time += duration_ms
                    self._stats.mean_inference_ms = (
                        self._total_inference_time / self._stats.inferences_run
                    )
                    self._stats.latest_latency_ms = duration_ms
                    self._result_event.set()
            except Exception as e:
                log.error(
                    "inference worker error",
                    extra={"vantage_fields": {"error": str(e), "model": self.info.model}},
                )

    def stop(self) -> None:
        """Stop worker thread and release resources."""
        if not self._running:
            return
        self._running = False
        self._buffer.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        log.info("async inference worker stopped")

    def close(self) -> None:
        self.stop()
        self._engine.close()
