"""The ingestion pipeline: acquisition, pacing, buffering, measurement.

The contract every later phase is written against::

    with IngestionPipeline(source, config) as pipeline:
        for frame in pipeline.frames():
            ...            # detection, tracking, pose, events - all downstream

Consumers see an ordinary iterator of :class:`~vantage.core.frame.Frame`. They
never learn whether acquisition happened on this thread or another, whether
frames were dropped to keep latency bounded, or what kind of device produced
them. That is the whole point: the seam where Phase 2 attaches is a ``for``
loop, and attaching to it changes nothing here.

Two execution modes:

``THREADED`` (default)
    Acquisition runs on its own thread behind :class:`FrameBuffer`. Necessary
    the moment a consumer becomes slower than the source - which, on the
    CPU-only hardware this targets, is the moment a detector is added.

``INLINE``
    Acquisition on the calling thread, no queue, no thread. Fully
    deterministic; used by tests and by batch file processing where there is no
    real-time deadline to miss.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from vantage.config.schema import IngestConfig, IngestMode
from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import SourceExhausted
from vantage.core.frame import Frame
from vantage.core.logging import get_logger
from vantage.core.metrics import MetricsRegistry
from vantage.ingestion.base import FrameSource, SourceInfo
from vantage.ingestion.buffer import FrameBuffer, resolve_backpressure
from vantage.ingestion.pacing import MediaClockPacer, RatePacer, StrideFilter

log = get_logger(__name__)

_JOIN_TIMEOUT_S = 5.0
_POLL_TIMEOUT_S = 0.25


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Point-in-time snapshot of pipeline health.

    Feeds the HUD now and the Phase 12 metrics endpoint later; it is a plain
    dataclass of primitives so it serialises without special handling.
    """

    source_id: str
    kind: str
    backend: str
    uri: str
    width: int
    height: int
    declared_fps: float | None
    is_live: bool
    state: str

    frames_produced: int = 0
    frames_delivered: int = 0
    frames_dropped: int = 0
    frames_skipped: int = 0
    reconnects: int = 0

    capture_fps: float = 0.0
    delivery_fps: float = 0.0
    mean_delivery_fps: float = 0.0

    acquire_ms_p50: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_last: float = 0.0

    queue_depth: int = 0
    queue_capacity: int = 0
    queue_high_water: int = 0
    backpressure: str = "n/a"
    elapsed_s: float = 0.0

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def drop_rate(self) -> float:
        """Fraction of produced frames that never reached the consumer."""
        return self.frames_dropped / self.frames_produced if self.frames_produced else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestionPipeline:
    """Delivers frames from one :class:`FrameSource` to one consumer."""

    def __init__(
        self,
        source: FrameSource,
        config: IngestConfig | None = None,
        *,
        clock: Clock = SYSTEM_CLOCK,
        shutdown: threading.Event | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._source = source
        self._config = config or IngestConfig()
        self._clock = clock
        self._shutdown = shutdown or threading.Event()

        self._metrics = metrics or MetricsRegistry(name=f"ingest.{source.source_id}")
        self._capture_rate = self._metrics.rate("capture_fps")
        self._delivery_rate = self._metrics.rate("delivery_fps")
        self._acquire_latency = self._metrics.latency("acquire_ms")
        self._delivery_latency = self._metrics.latency("delivery_ms")
        self._error_counter = self._metrics.counter("errors")

        self._stride = StrideFilter(self._config.stride)
        self._pacer = RatePacer(self._config.target_fps, clock=clock)
        self._media_pacer: MediaClockPacer | None = None

        self._buffer: FrameBuffer | None = None
        self._thread: threading.Thread | None = None
        self._capture_done = threading.Event()
        self._error: BaseException | None = None
        self._delivered = 0
        self._started = False
        self._closed = False
        self._start_monotonic = 0.0

    # -- properties -----------------------------------------------------

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def info(self) -> SourceInfo:
        return self._source.info

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def shutdown_event(self) -> threading.Event:
        """The flag that stops delivery; share it with the application's controller."""
        return self._shutdown

    # -- lifecycle ------------------------------------------------------

    def start(self) -> SourceInfo:
        """Open the source and, in threaded mode, begin acquiring.

        Returns the negotiated :class:`SourceInfo`. Safe to call once; the
        second call returns the same info rather than restarting.
        """
        if self._started:
            return self._source.info
        info = self._source.open()
        self._started = True
        self._start_monotonic = self._clock.monotonic()

        if self._config.realtime and not info.is_live:
            self._media_pacer = MediaClockPacer(
                fallback_fps=info.declared_fps, clock=self._clock
            )

        if self._config.mode is IngestMode.THREADED:
            policy = resolve_backpressure(self._config.backpressure, info.is_live)
            self._buffer = FrameBuffer(capacity=self._config.queue_size, policy=policy)
            self._thread = threading.Thread(
                target=self._capture_loop,
                name=f"vantage-capture-{info.source_id}",
                daemon=True,
            )
            self._thread.start()
            log.debug(
                "capture thread started",
                extra={
                    "vantage_fields": {
                        "source_id": info.source_id,
                        "queue_size": self._config.queue_size,
                        "backpressure": policy.value,
                    }
                },
            )
        return info

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source ends, the limit is hit, or shutdown."""
        if not self._started:
            self.start()

        iterator = self._iter_threaded() if self._buffer is not None else self._iter_inline()
        for frame in iterator:
            self._delivered += 1
            now = self._clock.monotonic()
            self._delivery_rate.tick(now)
            self._delivery_latency.observe(frame.age_ms(now))
            yield frame
            if self._config.max_frames is not None and self._delivered >= self._config.max_frames:
                log.debug(
                    "frame limit reached",
                    extra={"vantage_fields": {"max_frames": self._config.max_frames}},
                )
                break

        # A failure on the capture thread surfaces here, on the consumer's
        # thread, where it can actually be handled.
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        """Stop acquisition and release the source. Idempotent; never raises."""
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()

        if self._buffer is not None:
            # Close before joining: a producer blocked on a full queue under
            # BLOCK backpressure is waiting on this condition.
            self._buffer.close()

        capture_thread_stuck = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            capture_thread_stuck = self._thread.is_alive()

        if self._buffer is not None:
            self._buffer.clear()

        if capture_thread_stuck:
            # Only reachable if a driver blocks inside read() indefinitely.
            # Releasing the handle now would mean calling release() while another
            # thread sits inside read() on it, which can take the process down at
            # the C level. Leaving it to the daemon thread and to process exit is
            # the lesser evil, and saying so beats a silent crash.
            log.warning(
                "capture thread is still blocked inside the driver; leaving the "
                "device handle to be released at process exit",
                extra={
                    "vantage_fields": {
                        "source_id": self._source.source_id,
                        "timeout_s": _JOIN_TIMEOUT_S,
                    }
                },
            )
        else:
            self._source.close()

        stats = self.stats()
        log.info(
            "ingestion stopped",
            extra={
                "vantage_fields": {
                    "source_id": stats.source_id,
                    "delivered": stats.frames_delivered,
                    "produced": stats.frames_produced,
                    "dropped": stats.frames_dropped,
                    "skipped": stats.frames_skipped,
                    "mean_fps": round(stats.mean_delivery_fps, 2),
                    "elapsed_s": round(stats.elapsed_s, 2),
                }
            },
        )

    def __enter__(self) -> "IngestionPipeline":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- delivery -------------------------------------------------------

    def _iter_threaded(self) -> Iterator[Frame]:
        assert self._buffer is not None
        buffer = self._buffer
        while not self._shutdown.is_set():
            frame = buffer.get(timeout=_POLL_TIMEOUT_S)
            if frame is None:
                if self._capture_done.is_set() and len(buffer) == 0:
                    return
                continue
            yield frame

    def _iter_inline(self) -> Iterator[Frame]:
        while not self._shutdown.is_set():
            frame = self._acquire_next()
            if frame is None:
                return
            yield frame

    def _capture_loop(self) -> None:
        """Producer thread: acquire, pace, enqueue."""
        assert self._buffer is not None
        try:
            while not self._shutdown.is_set():
                frame = self._acquire_next()
                if frame is None:
                    return
                # A False return means the policy dropped it; the buffer counts it.
                self._buffer.put(frame, timeout=_POLL_TIMEOUT_S)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer thread
            self._error = exc
            self._error_counter.inc()
            log.error(
                "capture loop failed",
                extra={
                    "vantage_fields": {
                        "source_id": self._source.source_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                },
            )
        finally:
            self._capture_done.set()
            self._buffer.close()

    def _acquire_next(self) -> Frame | None:
        """Read the next frame the consumer should see, honouring stride and pacing.

        Returns ``None`` when the source has ended.
        """
        while not self._shutdown.is_set():
            self._pacer.wait()

            started = self._clock.monotonic()
            try:
                frame = self._source.read()
            except SourceExhausted as exc:
                log.info(
                    "source exhausted",
                    extra={
                        "vantage_fields": {
                            "source_id": self._source.source_id,
                            "reason": str(exc),
                            "frames": self._source.frames_produced,
                        }
                    },
                )
                return None
            self._acquire_latency.observe((self._clock.monotonic() - started) * 1000.0)
            self._capture_rate.tick(self._clock.monotonic())

            if not self._stride.keep(frame.index):
                continue
            if self._media_pacer is not None:
                self._media_pacer.wait_for(frame.media_pts)
            return frame
        return None

    # -- observability --------------------------------------------------

    def stats(self) -> PipelineStats:
        """Snapshot of throughput, latency and loss. Cheap enough to call per frame."""
        info = self._source_info_or_placeholder()
        buffer = self._buffer
        reconnects = getattr(self._source, "reconnects", 0)

        return PipelineStats(
            source_id=info.source_id,
            kind=info.kind.value,
            backend=info.backend,
            uri=info.uri,
            width=info.width,
            height=info.height,
            declared_fps=info.declared_fps,
            is_live=info.is_live,
            state=self._source.state.value,
            frames_produced=self._source.frames_produced,
            frames_delivered=self._delivered,
            # `is not None`, never a truth test: FrameBuffer defines __len__, so
            # an empty queue is falsy and would silently report as inline.
            frames_dropped=buffer.dropped if buffer is not None else 0,
            frames_skipped=self._stride.skipped,
            reconnects=int(reconnects),
            capture_fps=self._capture_rate.rate,
            delivery_fps=self._delivery_rate.rate,
            mean_delivery_fps=self._delivery_rate.mean_rate,
            acquire_ms_p50=self._acquire_latency.percentile(50),
            latency_ms_p50=self._delivery_latency.percentile(50),
            latency_ms_p95=self._delivery_latency.percentile(95),
            latency_ms_last=self._delivery_latency.last,
            queue_depth=len(buffer) if buffer is not None else 0,
            queue_capacity=buffer.capacity if buffer is not None else 0,
            queue_high_water=buffer.high_water if buffer is not None else 0,
            backpressure=buffer.policy.value if buffer is not None else "inline",
            elapsed_s=max(0.0, self._clock.monotonic() - self._start_monotonic)
            if self._started
            else 0.0,
        )

    def _source_info_or_placeholder(self) -> SourceInfo:
        """Stats must work before open() and after close(), e.g. in error paths."""
        try:
            return self._source.info
        except Exception:
            from vantage.ingestion.base import SourceKind

            return SourceInfo(
                source_id=self._source.source_id,
                kind=SourceKind.SYNTHETIC,
                uri=self._source.uri,
                width=0,
                height=0,
                backend="unopened",
            )
