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
            raise ConfigError(
                "source.reconnect.max_attempts must be >= 0 (0 disables retrying)"
            )
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
            raise ConfigError(
                "source.uri must not be empty (e.g. 'webcam:0' or 'synthetic://')"
            )
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(
                    f"source.{name} must be a positive integer or null, got {value}"
                )
        if self.fps is not None and self.fps <= 0:
            raise ConfigError(f"source.fps must be positive or null, got {self.fps}")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ConfigError(
                f"source.fourcc must be exactly 4 characters, got {self.fourcc!r}"
            )
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
class TrackingConfig:
    """Phase 3 multi-object tracking.

    Off by default, like detection: enabling it changes what the pipeline
    produces, and that should be an explicit choice rather than something that
    happens to whoever runs the default config.

    The numeric defaults mirror
    :class:`~vantage.tracking.bytetrack.TrackerParams`, which are the measured
    output of ``vantage track tune``. They are exposed here so a deployment with
    a different camera or a different detector can re-tune without editing code,
    but the shipped values are the ones to prefer absent a reason.
    """

    enabled: bool = False

    detection_floor: float = 0.1
    """Confidence the detector is lowered to when tracking is enabled.

    This is the one setting whose purpose is not obvious, and it is load
    bearing. ByteTrack's second association pass exists to match *low*-scoring
    boxes to existing tracks, which is how identity survives a partial
    occlusion. If the detector keeps filtering at ``detection.confidence``
    (0.35 by default) those boxes never reach the tracker and the algorithm
    silently degrades to ordinary IoU tracking - working, but without the one
    property it was chosen for.

    So enabling tracking lowers the detector's floor to this value and lets the
    tracker do the filtering instead. The trade is real and worth stating: the
    detector now emits considerably more junk, and ``min_hits`` is what stops
    that junk becoming published tracks. Set this equal to
    ``detection.confidence`` to opt out and accept the weaker behaviour.
    """

    high_threshold: float = 0.3
    low_threshold: float = 0.1
    init_threshold: float = 0.5
    """Confidence bands. Above ``high`` a box can start a track; between ``low``
    and ``high`` it can only sustain one; below ``low`` it is ignored."""

    iou_high: float = 0.2
    iou_low: float = 0.4
    iou_tentative: float = 0.4
    """Minimum overlap for a match, per association pass."""

    min_hits: int = 3
    """Frames a track must be corroborated on before it is published."""

    max_lost_s: float = 1.5
    """Seconds a track survives unmatched before being dropped. In seconds
    rather than frames because the frame interval genuinely varies here, so a
    frame count would mean different things at different settings."""

    max_step_s: float = 2.0
    history: int = 30
    class_aware: bool = True

    measurement_noise: float = 0.05
    acceleration_noise: float = 2.0
    initial_velocity_noise: float = 1.0
    size_drift_noise: float = 0.2
    """Motion-model noise, as fractions of object height. See
    :class:`~vantage.tracking.kalman.MotionNoise`."""

    def __post_init__(self) -> None:
        for name in ("high_threshold", "low_threshold", "init_threshold", "detection_floor"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(
                    f"tracking.{name} must be between 0 and 1 (exclusive), got {value}"
                )
        if self.low_threshold >= self.high_threshold:
            raise ConfigError(
                "tracking.low_threshold must be below tracking.high_threshold, or the "
                "second association pass never sees a box "
                f"(got {self.low_threshold} >= {self.high_threshold})"
            )
        if self.init_threshold < self.high_threshold:
            raise ConfigError(
                "tracking.init_threshold must be >= tracking.high_threshold: a box too "
                "weak for the first association pass must not be able to create a track "
                f"(got {self.init_threshold} < {self.high_threshold})"
            )
        for name in ("iou_high", "iou_low", "iou_tentative"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"tracking.{name} must be between 0 and 1, got {value}")
        if self.min_hits < 1:
            raise ConfigError("tracking.min_hits must be >= 1 (1 publishes immediately)")
        if self.max_lost_s < 0:
            raise ConfigError("tracking.max_lost_s must be >= 0")
        if self.max_step_s <= 0:
            raise ConfigError("tracking.max_step_s must be positive")
        if self.history < 1:
            raise ConfigError("tracking.history must be >= 1")
        for name in (
            "measurement_noise",
            "acceleration_noise",
            "initial_velocity_noise",
            "size_drift_noise",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"tracking.{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class PoseConfig:
    """Human pose estimation. Off by default; requires tracking.

    Top-down estimation needs a person box, and taking that box from the
    tracker rather than the detector is what binds each skeleton to a stable
    anonymous entity instead of to an anonymous rectangle.
    """

    enabled: bool = False
    model: str = "rtmpose-s"
    backend: str = "auto"
    device: str = "auto"

    interval: int = 1
    """Estimate on every Nth frame that ran detection. Multiplies with
    ``detection.interval`` rather than replacing it: pose can never run on a
    frame the tracker did not update, because it would be cropping to a box no
    detector confirmed."""

    max_persons: int = 6
    """Hard cap on people estimated per frame, largest boxes first. Cost is
    linear in people, so without a cap a crowd silently costs the frame rate."""

    min_keypoint_confidence: float = 0.3
    """Below this a landmark counts as not observed. Measured on real frames:
    joints that are genuinely visible score 0.7 and above, invented ones fall
    under 0.25, so the boundary sits in an empty band rather than through a
    cluster."""

    classes: list[str] = field(default_factory=lambda: ["person"])

    include_face_keypoints: bool = True
    """Keep the five head landmarks - nose, eyes, ears. They are coordinates,
    not a face descriptor (see :mod:`vantage.pose.contracts`), but a deployment
    that would rather not carry them can drop them here and they are never
    constructed."""

    threads: int = 0
    model_dir: str = "models"
    allow_download: bool = True

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ConfigError(f"pose.interval must be >= 1, got {self.interval}")
        if self.max_persons < 1:
            raise ConfigError(f"pose.max_persons must be >= 1, got {self.max_persons}")
        if not 0.0 <= self.min_keypoint_confidence < 1.0:
            raise ConfigError(
                f"pose.min_keypoint_confidence must be in [0, 1), got "
                f"{self.min_keypoint_confidence}"
            )
        if not self.classes:
            raise ConfigError("pose.classes must name at least one label to estimate")


@dataclass(frozen=True, slots=True)
class StateConfig:
    """Motion state, dwell timing and path length for every tracked entity.

    Needs no model and no weights - it reads the velocity the tracker's Kalman
    filter already maintains - so it is on by default whenever tracking is.
    """

    enabled: bool = True

    moving_above: float = 0.15
    """Entity heights per second at which motion is declared. Height-relative
    rather than pixels, so one threshold holds across the frame regardless of
    how far away the entity is."""

    stationary_below: float = 0.08
    """The lower edge of the dead band. Between the two, state persists."""

    min_state_s: float = 0.5
    min_age_s: float = 0.3

    def __post_init__(self) -> None:
        for name in ("moving_above", "stationary_below", "min_state_s", "min_age_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"state.{name} must be >= 0")
        if self.stationary_below > self.moving_above:
            raise ConfigError(
                f"state.stationary_below ({self.stationary_below}) must not exceed "
                f"state.moving_above ({self.moving_above}); inverted, there is no dead "
                "band and the hysteresis that keeps dwell timings meaningful does nothing"
            )


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    """Temporal activity recognition. Needs no model; on whenever tracking is.

    Reads the signals state and pose already produce, so the cost is a bounded
    buffer per entity and some arithmetic. Without pose the posture-derived
    activities simply never fire, which is correct rather than degraded.
    """

    enabled: bool = True

    walking_speed: float = 0.15
    """Entity heights per second above which a moving entity is walking. Kept
    equal to ``state.moving_above``; see the cross-check in
    :meth:`VantageConfig.__post_init__`."""

    running_speed: float = 1.30
    """And above which it is running. A walk measures 0.6-0.9 h/s and a run
    1.5-2.5, so the boundary sits in the gap rather than through either."""

    sustain_s: float = 0.4
    """How long a continuous rule must hold before it is reported."""

    loiter_s: float = 20.0
    """Stationary for this long becomes loitering. A duration, not a judgement:
    what it means is a policy question for the event rules of a later phase."""

    transition_window_s: float = 2.5
    fall_window_s: float = 1.2
    """Upright to lying faster than this is a fall; slower is a deliberate
    lie-down and is reported as nothing at all. Read the limitations in
    :mod:`vantage.activity` before relying on this for anything."""

    transient_hold_s: float = 1.5
    posture_window_s: float = 0.6
    min_posture_confidence: float = 0.25
    min_keypoint_confidence: float = 0.3
    history: int = 240

    def __post_init__(self) -> None:
        if self.walking_speed <= 0:
            raise ConfigError("activity.walking_speed must be positive")
        if self.running_speed <= self.walking_speed:
            raise ConfigError(
                f"activity.running_speed ({self.running_speed}) must exceed "
                f"activity.walking_speed ({self.walking_speed}), or running can "
                "never be distinguished from walking"
            )
        if self.fall_window_s > self.transition_window_s:
            raise ConfigError(
                f"activity.fall_window_s ({self.fall_window_s}) must not exceed "
                f"activity.transition_window_s ({self.transition_window_s}): a fall "
                "is a fast posture transition, so it cannot be detectable over a "
                "longer span than transitions are paired over"
            )
        for name in (
            "sustain_s",
            "loiter_s",
            "transition_window_s",
            "fall_window_s",
            "transient_hold_s",
            "posture_window_s",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"activity.{name} must be >= 0")
        if self.history < 2:
            raise ConfigError("activity.history must be >= 2")


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    """One named region, as a polygon in normalised coordinates.

    Normalised to ``[0, 1]`` so a zone drawn against a 1080p stream still means
    the same part of the scene at 720p, or when a file of a different size is
    replayed. Pixel coordinates would quietly point somewhere else.
    """

    name: str
    points: list[list[float]] = field(default_factory=list)
    kind: str = "area"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("every spatial.zones entry needs a name")
        if len(self.points) < 3:
            raise ConfigError(
                f"zone {self.name!r} needs at least 3 points, got {len(self.points)}"
            )
        for index, point in enumerate(self.points):
            if len(point) != 2:
                raise ConfigError(
                    f"zone {self.name!r} point {index} must be [x, y], got {point!r}"
                )
            if not all(0.0 <= value <= 1.0 for value in point):
                raise ConfigError(
                    f"zone {self.name!r} point {index} is outside [0, 1]: {point!r}. "
                    "Zone coordinates are normalised so they survive a change of "
                    "resolution."
                )


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    """Zones and pairwise relations. Needs no model; on whenever tracking is.

    Distances are in entity heights under a common-ground assumption, never in
    metres. Read the limitations in :mod:`vantage.spatial` before treating any
    threshold here as a physical distance.
    """

    enabled: bool = True
    zones: list[ZoneConfig] = field(default_factory=list)

    near_distance: float = 1.5
    near_hysteresis: float = 0.3
    approach_rate: float = 0.25
    approach_window_s: float = 0.8

    interact_distance: float = 0.6
    interact_s: float = 1.0
    reach_confidence: float = 0.35
    """Minimum wrist landmark score for a reach to count as confirmed. A reach
    is the difference between claiming contact at 0.85 confidence and at 0.4."""

    zone_event_hold_s: float = 1.5
    max_entities: int = 24
    """Relations are pairwise, so cost is quadratic; above this only the largest
    boxes are paired."""

    history: int = 120

    def __post_init__(self) -> None:
        for name in (
            "near_distance",
            "near_hysteresis",
            "approach_rate",
            "approach_window_s",
            "interact_distance",
            "interact_s",
            "zone_event_hold_s",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"spatial.{name} must be >= 0")
        if self.interact_distance > self.near_distance:
            raise ConfigError(
                f"spatial.interact_distance ({self.interact_distance}) must not exceed "
                f"spatial.near_distance ({self.near_distance}): interaction is the "
                "closer relation, so a pair could otherwise be interacting without "
                "being near"
            )
        if self.max_entities < 2:
            raise ConfigError("spatial.max_entities must be >= 2 for a pair to exist")
        names = [zone.name for zone in self.zones]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ConfigError(
                f"duplicate zone names: {sorted(duplicates)}. Zone names identify a "
                "place in every observation record, so they have to be unique."
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

    stage_failure_budget: int = 5
    """Consecutive failures before an analysis stage is disabled for the run.

    A stage that throws on one frame has met a bad frame; a stage that throws on
    five in a row is broken, and continuing to call it costs latency on every
    frame and floods the log. Set higher for a flaky source you would rather
    limp along with, lower to fail fast."""

    resource_interval_s: float = 10.0
    """How often to sample process CPU and memory. Zero disables it.

    Cheap - two syscalls - but there is no reason to do it per frame, and the
    number it exists to reveal is a leak measured over hours."""
    """How often a metrics summary is logged. ``0`` disables periodic summaries."""

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigError(
                f"app.log_level must be one of {sorted(valid_levels)}, got {self.log_level!r}"
            )
        if self.log_format not in {"console", "json"}:
            raise ConfigError(
                f"app.log_format must be 'console' or 'json', got {self.log_format!r}"
            )
        if self.stats_interval_s < 0:
            raise ConfigError("app.stats_interval_s must be >= 0 (0 disables)")
        if self.stage_failure_budget < 1:
            raise ConfigError(
                "app.stage_failure_budget must be >= 1: a budget of 0 would disable "
                "every stage on its first bad frame"
            )
        if self.resource_interval_s < 0:
            raise ConfigError("app.resource_interval_s must be >= 0 (0 disables)")


@dataclass(frozen=True, slots=True)
class VantageConfig:
    """Root configuration object."""

    app: AppConfig = field(default_factory=AppConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    state: StateConfig = field(default_factory=StateConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def __post_init__(self) -> None:
        if self.tracking.enabled and not self.detection.enabled:
            raise ConfigError(
                "tracking.enabled requires detection.enabled: the tracker consumes "
                "detections and has no other source of objects. Enable detection, or "
                "pass --detect along with --track."
            )
        if self.pose.enabled and not self.tracking.enabled:
            raise ConfigError(
                "pose.enabled requires tracking.enabled: pose estimation is top-down "
                "and takes its person boxes from tracks, which is also what binds each "
                "skeleton to a stable entity id. Pass --track along with --pose."
            )
        if self.activity.enabled and self.tracking.enabled and not self.state.enabled:
            raise ConfigError(
                "activity.enabled requires state.enabled: every activity rule reads "
                "the motion state, so with state off the recogniser would report "
                "nothing but 'idle' for every entity. Disable activity too, or "
                "re-enable state."
            )
        if self.activity.enabled and self.activity.walking_speed > self.state.moving_above:
            raise ConfigError(
                f"activity.walking_speed ({self.activity.walking_speed}) must not exceed "
                f"state.moving_above ({self.state.moving_above}). Between the two, the "
                "state machine calls an entity moving while no locomotion rule fires, so "
                "the same entity is reported as moving and idle in the same frame."
            )
