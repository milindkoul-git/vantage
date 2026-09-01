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

    labels: tuple[str, ...] = ("person",)
    """Which detected classes have activities at all.

    "Walking", "running", "sitting down" and "falling" are things people do. Left
    open to every class, this engine reported that 73% of what it saw on five
    street clips was walking or running - most of it about cars, potted plants,
    traffic lights and handbags. `potted plant_2 is running` reached the event
    log.

    Widen it only for classes the rules genuinely describe. A dog walks and runs;
    a traffic light does not.
    """

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
        if not self.labels or not any(label.strip() for label in self.labels):
            raise ConfigError(
                "activity.labels cannot be empty: it decides which detected "
                "classes have activities at all, and an empty list would silence "
                "the whole subsystem without disabling it"
            )


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
class EventRuleConfig:
    """One configured rule. Validated when the config is read.

    A typo in an activity name would otherwise be a rule that can never fire,
    and silence is indistinguishable from calm.
    """

    type: str
    name: str = ""
    severity: str = "info"
    cooldown_s: float = 5.0
    zones: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    activity: str = ""
    relation: str = ""
    min_confidence: float = 0.0
    min_seconds: float = 0.0
    min_count: int = 2

    def __post_init__(self) -> None:
        if self.severity not in ("info", "notice", "alert"):
            raise ConfigError(
                f"event rule severity must be info, notice or alert, got {self.severity!r}"
            )
        # The rest is validated by RuleSpec, which owns the rule semantics.
        # Importing it here keeps one definition of what a valid rule is rather
        # than two that can drift apart.
        from vantage.events.contracts import Severity
        from vantage.events.rules import RuleSpec

        RuleSpec(
            type=self.type,
            name=self.name,
            severity=Severity(self.severity),
            cooldown_s=self.cooldown_s,
            zones=tuple(self.zones),
            labels=tuple(self.labels),
            activity=self.activity,
            relation=self.relation,
            min_confidence=self.min_confidence,
            min_seconds=self.min_seconds,
            min_count=self.min_count,
        )


@dataclass(frozen=True, slots=True)
class EventsConfig:
    """Discrete events raised from the continuous observations.

    Needs no model. On whenever tracking is, because the default rule set does
    nothing on a quiet scene: the only ALERT is a fall, and the zone rules are
    inert until zones are drawn.
    """

    enabled: bool = True
    rules: list[EventRuleConfig] = field(default_factory=list)
    """Empty means the built-in defaults, not "no rules". Turning the subsystem
    off is what ``enabled: false`` is for, and conflating the two would let a
    stray edit silence every alert with nothing saying so."""

    def __post_init__(self) -> None:
        names = [rule.name for rule in self.rules if rule.name]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ConfigError(
                f"duplicate event rule names: {sorted(duplicates)}. A rule name is "
                "the key a consumer filters on and the key a cooldown uses, so two "
                "rules sharing one would silence each other."
            )


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Persisting observations and events to a SQLite file.

    Off by default. A tool that silently created a growing database in whatever
    directory it was launched from would be a surprise, and an unwanted one on a
    machine where disk is the constraint. Turn it on with ``--store``.
    """

    enabled: bool = False
    path: str = "vantage.db"

    store_observations: bool = True
    """Whether to record continuous per-entity state as well as events.

    Events alone are tiny - a camera might produce a dozen a day. Observations
    are what make the history searchable, and also what fills the disk."""

    observation_interval: int = 15
    """Record observations on one analysed frame in N.

    At 30 fps with four entities, every frame is 120 rows a second - ten million
    a day, nearly all identical to their predecessor, because entity state
    changes on the scale of seconds. Sampling deliberately is reproducible;
    letting the queue overflow is not, because what is lost then depends on when
    the disk happened to be busy."""

    batch_size: int = 200
    flush_interval_s: float = 2.0
    """Rows are committed when either is reached. One transaction per batch is
    the difference between hundreds of rows a second and tens of thousands."""

    observation_queue: int = 5000
    event_queue: int = 1000
    """Separate queues on purpose: a flood of observations must never crowd out
    an event, and observations arrive a hundred times more often."""

    retention_days: float = 30.0
    event_retention_days: float = 365.0
    """Events are kept far longer than observations because they are rare and
    they are the record of what actually happened. Zero disables pruning, which
    on a long-running camera means the disk fills eventually."""

    heartbeat_interval_s: float = 60.0
    """How often to record that this camera is alive, in seconds.

    Lives here rather than under ``analytics`` because the recorder is what
    emits it, even though analytics is what needs it. One row a minute, written
    whether or not anything was seen - which is the entire point: an empty scene
    produces no observation rows, so without this the store cannot distinguish
    an empty room from a dead recorder, and every overnight hour becomes
    unjudgeable.
    """

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ConfigError("storage.path must not be empty")
        if self.heartbeat_interval_s <= 0:
            raise ConfigError("storage.heartbeat_interval_s must be positive")
        if self.observation_interval < 1:
            raise ConfigError("storage.observation_interval must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("storage.batch_size must be >= 1")
        if self.flush_interval_s <= 0:
            raise ConfigError("storage.flush_interval_s must be positive")
        for name in ("observation_queue", "event_queue"):
            if getattr(self, name) < 1:
                raise ConfigError(f"storage.{name} must be >= 1")
        for name in ("retention_days", "event_retention_days"):
            if getattr(self, name) < 0:
                raise ConfigError(f"storage.{name} must be >= 0 (0 disables pruning)")
        if (
            self.retention_days
            and self.event_retention_days
            and self.event_retention_days < self.retention_days
        ):
            raise ConfigError(
                f"storage.event_retention_days ({self.event_retention_days}) is below "
                f"storage.retention_days ({self.retention_days}): events are the rare, "
                "already-filtered record of what happened, so discarding them sooner "
                "than the observations around them is almost certainly a mistake"
            )


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """The local web dashboard. Off by default; ``--dashboard`` turns it on."""

    enabled: bool = False
    host: str = "127.0.0.1"
    """Loopback by default, and deliberately.

    This serves live camera footage and stored observations with **no
    authentication of any kind**. Binding to 0.0.0.0 puts that on the network
    for anything that can reach the port; it is allowed, because a deployment
    behind a reverse proxy is legitimate, but it has to be asked for and it
    logs a warning saying what it means."""

    port: int = 8080
    """0 asks the operating system for a free port.

    Useful when embedding or testing, where a fixed port would collide; the
    chosen one is logged and returned by ``start()``. Not useful for a dashboard
    a person needs to visit, since they would have to read the log to find it."""

    jpeg_quality: int = 70
    max_width: int = 960
    """Frames are downscaled for the browser. A 1080p MJPEG stream is several
    megabytes a second per viewer for detail nobody is reading on a dashboard."""

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ConfigError(
                f"dashboard.port must be in 0..65535 (0 = pick one), got {self.port}"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError("dashboard.jpeg_quality must be in 1..100")
        if self.max_width < 64:
            raise ConfigError("dashboard.max_width must be at least 64")
        if not self.host.strip():
            raise ConfigError("dashboard.host must not be empty")


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    """Analytics over stored history. Reads the store; never runs in the pipeline.

    Nothing here affects a live run. These values are the defaults the
    ``vantage analytics`` command starts from, so that a deployment which has
    settled on a bucket width or a sensitivity does not have to repeat it on
    every invocation.
    """

    interval_s: float = 3600.0
    period_hours: int = 168
    """168: normal varies by day of week as well as hour. 24: only by hour."""

    sensitivity: float = 3.5
    """Robust z-score at which a bucket is called anomalous. Measured behaviour
    at this value: roughly one false alarm every seven weeks, catching half of
    all +60% deviations and almost all +80% ones."""

    training_span_s: float = 2419200.0
    """Four weeks. Gives a weekly baseline four samples per slot."""

    infer_zeros: bool = True
    zero_reach: int = 2
    judge_empty: bool = False

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ConfigError("analytics.interval_s must be positive")
        if self.period_hours not in (24, 168):
            raise ConfigError("analytics.period_hours must be 24 or 168")
        if self.sensitivity <= 0:
            raise ConfigError("analytics.sensitivity must be positive")
        if self.training_span_s < 0:
            raise ConfigError("analytics.training_span_s must be >= 0")
        if self.zero_reach < 1:
            raise ConfigError("analytics.zero_reach must be >= 1")


@dataclass(frozen=True, slots=True)
class IncidentsConfig:
    """Grouping raised events into situational incidents.

    Purely derived: it reads the events the event engine already raises and
    needs no model, no extra inference and no second pass over the frame. On
    whenever events are, because a stream of individual alerts with nothing
    joining them is what an operator has to do in their head otherwise.

    The weights and thresholds live in
    :class:`~vantage.incident.config.IncidentCorrelatorConfig`; what is exposed
    here is the decision to run it at all, plus the two timeouts an operator
    actually tunes per site.
    """

    enabled: bool = True

    attach_threshold: float = 0.65
    """Correlation score above which an event joins an existing incident
    outright. Below :attr:`candidate_threshold` it starts a new one; between the
    two it starts a new one *and* records the link on both, because a guess
    recorded as a certainty is worse than two incidents an operator can merge."""

    candidate_threshold: float = 0.35
    quiescent_timeout_s: float = 60.0
    resolution_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not 0.0 < self.candidate_threshold < self.attach_threshold <= 1.0:
            raise ConfigError(
                f"incidents thresholds must satisfy 0 < candidate ({self.candidate_threshold}) "
                f"< attach ({self.attach_threshold}) <= 1"
            )
        if self.quiescent_timeout_s <= 0:
            raise ConfigError("incidents.quiescent_timeout_s must be positive")
        if self.resolution_timeout_s <= self.quiescent_timeout_s:
            raise ConfigError(
                f"incidents.resolution_timeout_s ({self.resolution_timeout_s}) must exceed "
                f"quiescent_timeout_s ({self.quiescent_timeout_s}): an incident cannot resolve "
                "before it has gone quiet"
            )


@dataclass(frozen=True, slots=True)
class RelationshipsConfig:
    """Which anonymous entities keep appearing together, and how strongly.

    Built from tracked positions only - the ids are the tracker's anonymous
    ``person_17``, never a name - and it is off by default because unlike
    incidents it accumulates state about pairs across a whole session, which is
    a thing a deployment should opt into rather than inherit.
    """

    enabled: bool = False

    labels: tuple[str, ...] = ("person",)
    """Which detected classes can be in an association at all.

    The vocabulary here - co-occurrence, recurrent proximity, following - is
    about people who keep turning up together. A car parked beside a bench is
    not an association, and left ungated the graph filled with them: one clip of
    an empty underpass produced 28 pairs, between a person, a television and a
    skateboard."""

    min_strength: float = 0.0
    """Floor for an edge to be reported at all."""

    proximity_gate: float = 0.15
    """How close two entities must be, as a fraction of the frame, to be
    considered a pair worth scoring at all.

    The gate exists to keep the work near-linear in a crowd rather than
    quadratic, and it replaced a rule that read "pair everything when there are
    five entities or fewer". That cliff had the behaviour exactly backwards:
    a clip with one or two people produced 34 associations, all of them between
    fragments of the same person left by an id switch, while a street with 24
    people in frame at once produced none at all, permanently.

    MEASURED across three clips at 0.06, 0.10, 0.15 and 0.25: the number of
    pairs *tracked* scales with the gate - a dense pedestrian street goes from 87
    to 519 - while the number that actually score above 0.20 stays at exactly one
    throughout. The gate bounds the work; the scorer decides what is real. 0.15
    is therefore chosen as the value that keeps a crowded frame to a few hundred
    tracked pairs rather than the thousand-entry cap, without excluding a pair
    the scorer would have kept.

    Widen it for a camera looking down a long corridor, where two people a
    seventh of the frame apart are far from each other in metres."""

    persist_interval_s: float = 30.0
    """How often the graph is flushed to the store, when there is one."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_strength <= 1.0:
            raise ConfigError("relationships.min_strength must be between 0 and 1")
        if not 0.0 < self.proximity_gate <= 1.5:
            raise ConfigError(
                "relationships.proximity_gate is a fraction of the frame and must be "
                f"in (0, 1.5]; got {self.proximity_gate}"
            )
        if self.persist_interval_s <= 0:
            raise ConfigError("relationships.persist_interval_s must be positive")


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    """Optional identity resolution. Off by default, and deliberately so.

    This is the only subsystem that answers *who*. It never runs unless it is
    turned on, it never enrols anyone from the live pipeline, and with nobody
    enrolled it reports every face as unknown - which is the truth.
    """

    enabled: bool = False
    path: str = "identities.db"
    """Its own database, separate from the observation store, so biometric
    templates can be backed up, permissioned and deleted on their own terms."""

    detector_model: str = "yunet-face"
    embedder_model: str = "sface"
    model_dir: str = "models"
    allow_download: bool = True
    face_score: float = 0.7

    threshold: float = 0.363
    """Cosine similarity for a match. The figure OpenCV Zoo publishes for these
    weights; inherited rather than measured, because verifying it properly needs
    a labelled face set this project has no business collecting."""

    margin: float = 0.05
    """How far ahead of the runner-up the winner must be. Without it, two people
    who score nearly the same are separated by whichever scored 0.001 higher,
    which is how a system confidently uses the wrong name."""

    interval: int = 10
    min_votes: int = 3
    """Agreeing observations before a name is committed. One face crop is one
    angle at one moment; committing on it means a badly timed frame names
    someone for the rest of their time on camera."""

    max_attempts: int = 25
    reverify_interval: int = 150
    """Steps between re-checking a resolved track. The tracker can swap two
    people who cross, and nothing else downstream would ever notice."""

    min_face_fraction: float = 0.04

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ConfigError("identity.path must not be empty")
        if not -1.0 <= self.threshold <= 1.0:
            raise ConfigError("identity.threshold must be in [-1, 1]")
        if self.margin < 0:
            raise ConfigError("identity.margin must be >= 0")
        for name in ("interval", "min_votes", "max_attempts"):
            if getattr(self, name) < 1:
                raise ConfigError(f"identity.{name} must be >= 1")
        if self.reverify_interval < 0:
            raise ConfigError("identity.reverify_interval must be >= 0 (0 disables)")
        if not 0.0 <= self.face_score <= 1.0:
            raise ConfigError("identity.face_score must be in [0, 1]")
        if not 0.0 <= self.min_face_fraction <= 1.0:
            raise ConfigError("identity.min_face_fraction must be in [0, 1]")


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
class AdaptiveConfig:
    """Adaptive load shedding, so analysis degrades instead of falling behind.

    Applies to **live sources only**. A recorded file has no deadline - it can
    be analysed as slowly as it likes and the result is identical - so shedding
    load there would discard information for nothing.
    """

    enabled: bool = True
    headroom: float = 0.7
    """Fraction of the frame budget analysis may occupy. Not 1.0: decode,
    overlay and display need the rest, and a target that consumed the whole
    budget would sit permanently on the edge of dropping frames."""

    max_interval: int = 8
    """Ceiling on the analysis interval. Past this the tracker's association
    gaps exceed what its motion model can bridge, so a slow system is the
    honest outcome rather than a fast one producing nonsense."""

    raise_after_s: float = 1.0
    lower_after_s: float = 6.0
    """Much longer than ``raise_after_s`` on purpose: raising early costs a
    little temporal resolution, lowering early puts the pipeline straight back
    into the overload it just escaped."""

    def __post_init__(self) -> None:
        if not 0.0 < self.headroom <= 1.0:
            raise ConfigError(f"app.adaptive.headroom must be in (0, 1], got {self.headroom}")
        if self.max_interval < 1:
            raise ConfigError("app.adaptive.max_interval must be >= 1")
        for name in ("raise_after_s", "lower_after_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"app.adaptive.{name} must be >= 0")


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

    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)

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
    events: EventsConfig = field(default_factory=EventsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    incidents: IncidentsConfig = field(default_factory=IncidentsConfig)
    relationships: RelationshipsConfig = field(default_factory=RelationshipsConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def __post_init__(self) -> None:
        if self.tracking.enabled and not self.detection.enabled:
            raise ConfigError(
                "tracking.enabled requires detection.enabled: the tracker consumes "
                "detections and has no other source of objects. Enable detection, or "
                "pass --detect along with --track."
            )
        if (
            self.app.adaptive.enabled
            and self.app.adaptive.max_interval < self.detection.interval
        ):
            raise ConfigError(
                f"app.adaptive.max_interval ({self.app.adaptive.max_interval}) is below "
                f"detection.interval ({self.detection.interval}): the governor would be "
                "asked to shed load below the floor it was told to start at"
            )
        if self.identity.enabled and not self.tracking.enabled:
            raise ConfigError(
                "identity.enabled requires tracking.enabled: identity resolves an "
                "existing anonymous entity into a name, and without tracks there is "
                "no entity to resolve. This is the seam the spec asked for - identity "
                "attaches to tracking, and tracking never depends on identity."
            )
        if self.incidents.enabled and not self.events.enabled:
            raise ConfigError(
                "incidents.enabled requires events.enabled: an incident is a group of "
                "raised events and has no other input. Enable events, or set "
                "incidents.enabled: false."
            )
        if self.relationships.enabled and not (self.tracking.enabled and self.state.enabled):
            raise ConfigError(
                "relationships.enabled requires tracking.enabled and state.enabled: the "
                "graph is built from where tracked entities are and how they are moving. "
                "Pass --track, or set relationships.enabled: false."
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
