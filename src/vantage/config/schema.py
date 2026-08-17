"""Typed configuration schema.

Frozen dataclasses rather than pydantic: the validation this platform needs is
a few dozen range checks, and Phase 1 spends its dependency budget on video, not
on a validation framework. Each section validates itself in ``__post_init__``
and raises :class:`~vantage.core.errors.ConfigError` with a message stating what
was wrong *and* what a valid value looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vantage.core.errors import ConfigError


class Backpressure(str, Enum):
    """What to do when the consumer is slower than the source.

    This is the central tuning decision of the ingestion subsystem, and there is
    no single correct answer - only a correct answer per source type.
    """

    AUTO = "auto"
    """Choose per source: :attr:`LATEST` for live sources, :attr:`BLOCK` for files."""

    LATEST = "latest"
    """Drop the oldest queued frame to make room. Correct for live cameras:
    analysing a stale frame is worse than skipping it."""

    BLOCK = "block"
    """Stall the source until the consumer catches up. Correct for files:
    every frame is processed, so results are reproducible."""

    DROP_NEW = "drop_new"
    """Discard the incoming frame when full. Rarely what you want; provided for
    sources where the oldest frame is the reference (e.g. burst triggers)."""


class IngestMode(str, Enum):
    """Whether acquisition runs on its own thread."""

    THREADED = "threaded"
    """Capture on a dedicated thread behind a queue. Default: decouples a slow
    consumer from acquisition, which matters as soon as inference is added."""

    INLINE = "inline"
    """Capture on the consumer's thread. Deterministic, no queue - used by tests
    and by batch file processing where throughput beats latency."""


@dataclass(frozen=True, slots=True)
class ReconnectConfig:
    """Automatic recovery when a live source disappears mid-run."""

    enabled: bool = True
    max_attempts: int = 5
    initial_delay_s: float = 0.5
    max_delay_s: float = 10.0
    backoff: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ConfigError("source.reconnect.max_attempts must be >= 0 (0 disables retrying)")
        if self.initial_delay_s < 0 or self.max_delay_s < 0:
            raise ConfigError("source.reconnect delays must be >= 0")
        if self.max_delay_s < self.initial_delay_s:
            raise ConfigError(
                "source.reconnect.max_delay_s must be >= initial_delay_s "
                f"(got {self.max_delay_s} < {self.initial_delay_s})"
            )
        if self.backoff < 1.0:
            raise ConfigError("source.reconnect.backoff must be >= 1.0")


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Which video to ingest, and how to ask the driver for it."""

    uri: str = "synthetic://"
    """Source URI. Accepted forms:

    ``webcam:0``          local capture device by index
    ``file:path.mp4``     a media file (a bare existing path also works)
    ``synthetic://?...``  deterministic generated video, no hardware needed
    ``rtsp://...``        network stream, decoded through FFmpeg
    """

    id: str | None = None
    """Stable identifier stamped onto every :class:`~vantage.core.frame.Frame`.
    Derived from the URI when omitted. Set it explicitly for multi-camera work -
    Phase 8 storage will key observations on it."""

    backend: str = "auto"
    """Capture backend: ``auto``, ``msmf``, ``dshow``, ``ffmpeg``, ``gstreamer``,
    ``v4l2`` or ``any``. ``auto`` picks the best backend for the platform and
    source kind (MSMF for Windows cameras - measured ~2x faster than DirectShow
    on this machine - FFmpeg for files and network streams)."""

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    """Requested capture geometry and rate. Drivers negotiate: the values
    actually granted are reported in :class:`~vantage.ingestion.base.SourceInfo`
    and a mismatch is logged as a warning rather than silently accepted."""

    fourcc: str | None = None
    """Four-character capture codec, e.g. ``MJPG``. On USB webcams this is a
    genuine performance lever - many cap at low FPS in raw YUY2 but reach full
    rate in MJPG at the same resolution."""

    loop: bool = False
    """Restart file sources at EOF instead of ending the run."""

    read_retries: int = 3
    """Consecutive failed reads tolerated before the source is declared dead.
    USB cameras occasionally return a single empty frame under load."""

    reconnect: ReconnectConfig = field(default_factory=ReconnectConfig)

    def __post_init__(self) -> None:
        if not self.uri or not self.uri.strip():
            raise ConfigError("source.uri must not be empty (e.g. 'webcam:0' or 'synthetic://')")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"source.{name} must be a positive integer or null, got {value}")
        if self.fps is not None and self.fps <= 0:
            raise ConfigError(f"source.fps must be positive or null, got {self.fps}")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ConfigError(f"source.fourcc must be exactly 4 characters, got {self.fourcc!r}")
        if self.read_retries < 0:
            raise ConfigError("source.read_retries must be >= 0")


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """How frames move from the source to whatever consumes them."""

    mode: IngestMode = IngestMode.THREADED
    queue_size: int = 8
    """Frames buffered between capture and consumer. Small on purpose: a deep
    queue converts a throughput problem into an invisible latency problem, and
    at 1080p each frame costs ~6 MB of RAM."""

    backpressure: Backpressure = Backpressure.AUTO
    target_fps: float | None = None
    """Throttle acquisition to at most this rate. The cheapest way to fit a
    heavier Phase 2 model on limited CPU - process 10 good frames per second
    rather than fall progressively behind at 30."""

    stride: int = 1
    """Deliver every Nth frame. Unlike ``target_fps`` this is deterministic and
    resolution-independent, which is what file-based evaluation wants."""

    realtime: bool = False
    """Pace recorded sources (files, synthetic) to their native frame rate. Off
    by default: batch analysis should run as fast as the machine allows. Turn it
    on to preview recorded input the way a live camera would arrive. Ignored for
    live sources, which already run at their own pace."""

    max_frames: int | None = None
    """Stop after this many delivered frames. Used by smoke tests and by the
    ``--frames`` CLI flag; ``null`` runs until the source ends or you quit."""

    def __post_init__(self) -> None:
        if self.queue_size < 1:
            raise ConfigError("ingest.queue_size must be >= 1")
        if self.stride < 1:
            raise ConfigError("ingest.stride must be >= 1 (1 = keep every frame)")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ConfigError("ingest.target_fps must be positive or null")
        if self.max_frames is not None and self.max_frames < 1:
            raise ConfigError("ingest.max_frames must be >= 1 or null")


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Phase 2 object detection.

    Off by default: ingestion must keep working on machines with no model
    downloaded and no inference runtime installed.
    """

    enabled: bool = False
    model: str = "yolox-nano"
    """Catalog key. ``vantage models list`` shows sizes, accuracy and licences."""

    backend: str = "auto"
    """``auto`` | ``onnxruntime`` | ``openvino``. ``auto`` prefers OpenVINO,
    the only one of the two that can reach an Intel iGPU."""

    device: str = "auto"
    """``auto`` | ``cpu`` | ``gpu``. OpenVINO only; ``auto`` uses the GPU when
    one is genuinely present and says which it chose."""

    confidence: float = 0.35
    nms_iou: float = 0.45
    max_detections: int = 100

    classes: list[str] | None = None
    """Keep only these labels, e.g. ``[person]``. ``null`` keeps everything.
    Narrowing to what an application actually needs is both faster and better
    privacy practice than detecting everything and discarding most of it."""

    interval: int = 1
    """Run the detector on every Nth *delivered* frame.

    The single most effective CPU-fit lever: display stays smooth at full frame
    rate while inference runs at a sustainable fraction of it. Detections from
    the most recent pass are carried forward on skipped frames."""

    warmup: int = 2
    """Inference passes on a blank frame at startup, to absorb graph
    compilation and clock ramp-up before the first real frame arrives."""

    threads: int = 0
    """CPU threads for inference; ``0`` lets the runtime decide. Pin it when
    several cameras share a machine and must not fight over cores."""

    model_dir: str = "models"
    allow_download: bool = True
    """Fetch missing weights automatically. Turn off for air-gapped
    deployments, where a missing model should fail loudly instead."""

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ConfigError(
                f"detection.confidence must be between 0 and 1 (exclusive), got {self.confidence}"
            )
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ConfigError(f"detection.nms_iou must be between 0 and 1, got {self.nms_iou}")
        if self.max_detections < 1:
            raise ConfigError("detection.max_detections must be >= 1")
        if self.interval < 1:
            raise ConfigError("detection.interval must be >= 1 (1 = detect on every frame)")
        if self.warmup < 0:
            raise ConfigError("detection.warmup must be >= 0")
        if self.threads < 0:
            raise ConfigError("detection.threads must be >= 0 (0 = runtime decides)")
        if self.backend not in {"auto", "onnxruntime", "openvino"}:
            raise ConfigError(
                f"detection.backend must be 'auto', 'onnxruntime' or 'openvino', "
                f"got {self.backend!r}"
            )
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ConfigError(
                f"detection.device must be 'auto', 'cpu' or 'gpu', got {self.device!r}"
            )
        if self.classes is not None and not self.classes:
            raise ConfigError(
                "detection.classes is an empty list, which would discard every "
                "detection. Use null to keep all classes."
            )


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    """The Phase 1 diagnostic viewer.

    Strictly a *diagnostic* surface, not the Phase 9 dashboard. It exists to
    prove the pipeline works and to make timing problems visible; it contains no
    analysis logic and the pipeline runs identically without it.
    """

    enabled: bool = True
    window_name: str = "Vantage - Ingestion"
    hud: bool = True
    scale: float = 1.0
    """Display-only scaling. Never affects the frames handed to consumers."""

    snapshot_dir: str = "snapshots"
    """Where the ``s`` key writes PNGs. Relative paths resolve against the
    working directory - no path is hard-coded anywhere in the codebase."""

    def __post_init__(self) -> None:
        if not 0.05 <= self.scale <= 4.0:
            raise ConfigError(f"display.scale must be between 0.05 and 4.0, got {self.scale}")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Process-level behaviour."""

    log_level: str = "INFO"
    log_format: str = "console"
    stats_interval_s: float = 5.0
    """How often a metrics summary is logged. ``0`` disables periodic summaries."""

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigError(
                f"app.log_level must be one of {sorted(valid_levels)}, got {self.log_level!r}"
            )
        if self.log_format not in {"console", "json"}:
            raise ConfigError(f"app.log_format must be 'console' or 'json', got {self.log_format!r}")
        if self.stats_interval_s < 0:
            raise ConfigError("app.stats_interval_s must be >= 0 (0 disables)")


@dataclass(frozen=True, slots=True)
class VantageConfig:
    """Root configuration object."""

    app: AppConfig = field(default_factory=AppConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
