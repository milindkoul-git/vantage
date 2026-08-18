"""Ground-truth scenarios and a simulated detector, for measuring tracking.

Why simulate the detector instead of running the real one
---------------------------------------------------------
The question this module exists to answer is "how good is the *tracker*", and
that question cannot be answered by pointing the whole pipeline at a video. Run
YOLOX over real footage and every measurement blends two error sources: boxes
the detector got wrong, and identities the tracker got wrong. Tuning against
that mixture optimises the tracker to compensate for one specific detector's
quirks, and the resulting parameters silently stop being right the moment the
model changes.

So the detector is replaced by a model of one. Ground truth is exact and known;
detections are generated from it with a controlled amount of the four things a
real detector actually does wrong:

* **localisation error** - boxes are close, not exact
* **misses** - especially when an object is partially hidden
* **false positives** - boxes where nothing is
* **confidence that tracks visibility** - the important one, because a
  detector's score falls when an object is occluded, and that fall is exactly
  the signal ByteTrack's second pass is built to exploit. A simulator that
  emitted uniform confidence would make the algorithm's central mechanism
  untestable and tune it to the wrong parameters.

Everything is seeded, so a scenario is byte-identical on every machine and a
tuning result is reproducible rather than an anecdote.

The scenarios themselves target the specific ways trackers fail: objects that
cross (which invites an identity swap), objects that hide behind each other
(which tests whether identity survives), crowds (which stress the assignment),
and erratic motion (which tests the motion model rather than the association).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vantage.perception.contracts import BoundingBox, Detection, DetectionResult

DEFAULT_FPS = 30.0
PERSON_CLASS = 0
PERSON_LABEL = "person"


@dataclass(frozen=True, slots=True)
class GroundTruthObject:
    """Where one object truly is at one instant."""

    object_id: int
    box: BoundingBox
    visibility: float = 1.0
    """Fraction of the object not hidden by another object, in ``[0, 1]``.

    Drives both the miss rate and the confidence of the simulated detection,
    which is what makes occlusion a meaningful test rather than a relabelling.
    """

    label: str = PERSON_LABEL
    class_id: int = PERSON_CLASS


@dataclass(frozen=True, slots=True)
class GroundTruthFrame:
    index: int
    timestamp: float
    objects: tuple[GroundTruthObject, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named, deterministic sequence of ground-truth frames."""

    name: str
    frames: tuple[GroundTruthFrame, ...]
    frame_size: tuple[int, int]
    fps: float = DEFAULT_FPS
    description: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def object_count(self) -> int:
        return len({obj.object_id for frame in self.frames for obj in frame.objects})

    @property
    def instance_count(self) -> int:
        """Total ground-truth boxes across all frames - the MOTA denominator."""
        return sum(len(frame.objects) for frame in self.frames)


@dataclass(frozen=True, slots=True)
class DetectorProfile:
    """How badly the simulated detector behaves.

    The defaults approximate YOLOX-nano as measured in Phase 2: good but not
    excellent localisation, a low miss rate on clearly visible objects, and
    occasional spurious boxes.
    """

    localisation_noise: float = 0.03
    """Box corner error as a fraction of object size, Gaussian."""

    miss_rate: float = 0.05
    """Probability of missing a fully visible object."""

    occluded_miss_scale: float = 3.0
    """How much more likely a miss becomes as visibility falls to zero. A
    detector does not fail gracefully on occlusion; it fails faster than
    linearly, which this reproduces."""

    false_positive_rate: float = 0.05
    """Expected spurious boxes per frame (Poisson)."""

    confidence_visible: float = 0.88
    confidence_floor: float = 0.15
    """Confidence at full visibility and at none. The simulated detector
    interpolates between them, so a half-occluded object lands in exactly the
    low-confidence band the tracker's second pass reads."""

    confidence_noise: float = 0.05
    false_positive_confidence: float = 0.45
    """Spurious boxes are not uniformly weak - a detector that only ever
    produced weak false positives would make the confidence gate look far more
    effective than it is."""

    seed: int = 12345

    def __post_init__(self) -> None:
        if not 0.0 <= self.miss_rate <= 1.0:
            raise ValueError(f"miss_rate must be in [0, 1], got {self.miss_rate}")
        if self.false_positive_rate < 0:
            raise ValueError("false_positive_rate must be >= 0")
        if not 0.0 < self.confidence_visible <= 1.0:
            raise ValueError("confidence_visible must be in (0, 1]")


def simulate_detections(
    scenario: Scenario, profile: DetectorProfile | None = None
) -> list[DetectionResult]:
    """Turn ground truth into the detections a plausible detector would emit."""
    settings = profile or DetectorProfile()
    rng = np.random.default_rng(settings.seed)
    width, height = scenario.frame_size

    results: list[DetectionResult] = []
    for frame in scenario.frames:
        detections: list[Detection] = []

        for obj in frame.objects:
            visibility = float(np.clip(obj.visibility, 0.0, 1.0))
            miss_probability = min(
                1.0,
                settings.miss_rate
                * (1.0 + settings.occluded_miss_scale * (1.0 - visibility) ** 2),
            )
            if rng.random() < miss_probability:
                continue

            box = _jitter(obj.box, settings.localisation_noise, rng, width, height)
            confidence = (
                settings.confidence_floor
                + (settings.confidence_visible - settings.confidence_floor) * visibility
                + rng.normal(0.0, settings.confidence_noise)
            )
            detections.append(
                Detection(
                    box=box,
                    class_id=obj.class_id,
                    label=obj.label,
                    confidence=float(np.clip(confidence, 0.01, 1.0)),
                )
            )

        for _ in range(int(rng.poisson(settings.false_positive_rate))):
            detections.append(_spurious(rng, width, height, settings))

        results.append(
            DetectionResult(
                detections=tuple(detections),
                source_id=scenario.name,
                frame_index=frame.index,
                capture_wall=frame.timestamp,
                frame_size=scenario.frame_size,
                model="simulated",
                backend="simulated",
            )
        )
    return results


def _jitter(
    box: BoundingBox, noise: float, rng: np.random.Generator, width: int, height: int
) -> BoundingBox:
    """Perturb corners independently, so size wobbles as well as position."""
    scale = max(box.width, box.height)
    offsets = rng.normal(0.0, noise * scale, size=4)
    x1 = box.x1 + offsets[0]
    y1 = box.y1 + offsets[1]
    x2 = box.x2 + offsets[2]
    y2 = box.y2 + offsets[3]
    # Independent corner noise can invert a small box; order the corners rather
    # than clamping, which would bias every jittered box in one direction.
    return BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)).clipped(width, height)


def _spurious(
    rng: np.random.Generator, width: int, height: int, settings: DetectorProfile
) -> Detection:
    box_w = float(rng.uniform(0.04, 0.14) * width)
    box_h = float(rng.uniform(0.10, 0.35) * height)
    x1 = float(rng.uniform(0, max(1.0, width - box_w)))
    y1 = float(rng.uniform(0, max(1.0, height - box_h)))
    confidence = float(
        np.clip(rng.normal(settings.false_positive_confidence, 0.15), 0.05, 0.95)
    )
    return Detection(
        box=BoundingBox(x1, y1, x1 + box_w, y1 + box_h).clipped(width, height),
        class_id=PERSON_CLASS,
        label=PERSON_LABEL,
        confidence=confidence,
    )


# -- scenario construction -------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Mover:
    """One object's trajectory as a closed-form function of time."""

    object_id: int
    x0: float
    y0: float
    vx: float
    vy: float
    width: float
    height: float
    depth: int = 0
    """Draw order. Higher is nearer the camera and occludes lower."""

    wobble: float = 0.0
    """Amplitude of a sinusoidal cross-track deviation, in pixels. Non-zero
    makes the motion genuinely non-constant-velocity, which is what tests the
    motion model rather than the association."""

    wobble_hz: float = 0.38
    """Frequency of that deviation. Amplitude alone is not enough to make motion
    hard - a wide, slow arc is still locally almost straight, and a
    constant-velocity filter handles it comfortably. Turning *sharply* is what
    breaks the motion assumption, so the frequency is separately controllable
    and the hard scenarios raise it rather than just the amplitude."""

    def box_at(self, t: float) -> BoundingBox:
        cx = self.x0 + self.vx * t
        cy = self.y0 + self.vy * t
        if self.wobble:
            phase = 2.0 * np.pi * self.wobble_hz * t + self.object_id
            cy += self.wobble * np.sin(phase)
            cx += 0.45 * self.wobble * np.sin(0.7 * phase + 1.3)
        return BoundingBox(
            cx - self.width / 2.0,
            cy - self.height / 2.0,
            cx + self.width / 2.0,
            cy + self.height / 2.0,
        )


def _build(
    name: str,
    movers: list[_Mover],
    *,
    frames: int,
    frame_size: tuple[int, int],
    fps: float = DEFAULT_FPS,
    description: str = "",
) -> Scenario:
    """Render movers into frames, computing visibility from mutual overlap."""
    width, height = frame_size
    built: list[GroundTruthFrame] = []

    for index in range(frames):
        t = index / fps
        boxes = {m.object_id: m.box_at(t) for m in movers}
        objects: list[GroundTruthObject] = []

        for mover in movers:
            box = boxes[mover.object_id]
            # Off-screen objects do not exist for evaluation purposes: a tracker
            # cannot be marked down for missing something no detector could see.
            clipped = box.clipped(width, height)
            if clipped.width < 2 or clipped.height < 2:
                continue

            occluders = [
                boxes[other.object_id] for other in movers if other.depth > mover.depth
            ]
            objects.append(
                GroundTruthObject(
                    object_id=mover.object_id,
                    box=clipped,
                    visibility=_visibility(clipped, occluders),
                )
            )

        built.append(
            GroundTruthFrame(index=index, timestamp=t, objects=tuple(objects))
        )

    return Scenario(
        name=name,
        frames=tuple(built),
        frame_size=frame_size,
        fps=fps,
        description=description,
    )


def _visibility(box: BoundingBox, occluders: list[BoundingBox]) -> float:
    """Fraction of ``box`` not covered by any occluder.

    Overlapping areas are summed rather than unioned, which over-counts when two
    occluders cover the same region. That errs toward reporting *less*
    visibility than is real, which is the safe direction for a benchmark: it
    makes the scenario harder than reality, never easier.
    """
    if box.area <= 0:
        return 0.0
    covered = 0.0
    for other in occluders:
        ix1 = max(box.x1, other.x1)
        iy1 = max(box.y1, other.y1)
        ix2 = min(box.x2, other.x2)
        iy2 = min(box.y2, other.y2)
        covered += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(np.clip(1.0 - covered / box.area, 0.0, 1.0))


def crossing_scenario(frames: int = 120) -> Scenario:
    """Two objects walking through each other. The canonical identity-swap test.

    When the boxes overlap almost perfectly, IoU alone cannot tell the two
    apart, and a tracker that leans only on overlap will swap them. What
    separates them is that they arrived with different velocities, so this
    measures whether the motion model is actually contributing.
    """
    return _build(
        "crossing",
        [
            _Mover(0, 120.0, 300.0, 220.0, 0.0, 70.0, 180.0, depth=0),
            _Mover(1, 1160.0, 300.0, -220.0, 0.0, 70.0, 180.0, depth=1),
        ],
        frames=frames,
        frame_size=(1280, 720),
        description="two objects cross paths at equal speed",
    )


def occlusion_scenario(frames: int = 150) -> Scenario:
    """A small object passes behind a large stationary one.

    Tests the thing the whole phase is for: does the identity survive a period
    with no evidence at all? The occluder is wide enough to hide the subject
    completely for roughly half a second.
    """
    return _build(
        "occlusion",
        [
            _Mover(0, 100.0, 380.0, 200.0, 0.0, 60.0, 150.0, depth=0),
            _Mover(1, 640.0, 400.0, 0.0, 0.0, 260.0, 320.0, depth=1),
        ],
        frames=frames,
        frame_size=(1280, 720),
        description="subject passes fully behind a static occluder",
    )


def crowd_scenario(frames: int = 150, count: int = 8) -> Scenario:
    """Several objects on overlapping paths, at mixed scales.

    Stresses the assignment step rather than the motion model: with eight
    objects there are enough plausible pairings that a greedy matcher makes
    visibly worse choices than an optimal one.
    """
    rng = np.random.default_rng(4)
    movers = []
    for i in range(count):
        movers.append(
            _Mover(
                object_id=i,
                x0=float(rng.uniform(80, 1200)),
                y0=float(rng.uniform(180, 560)),
                vx=float(rng.uniform(-160, 160)),
                vy=float(rng.uniform(-70, 70)),
                width=float(rng.uniform(45, 95)),
                height=float(rng.uniform(120, 230)),
                depth=i,
                wobble=float(rng.uniform(0, 14)),
            )
        )
    return _build(
        "crowd",
        movers,
        frames=frames,
        frame_size=(1280, 720),
        description=f"{count} objects with overlapping trajectories",
    )


def erratic_scenario(frames: int = 180) -> Scenario:
    """Objects that change direction sharply and constantly.

    A constant-velocity model is wrong here by construction, which is the point:
    it measures how much the filter's process noise is helping or hurting when
    the motion assumption does not hold.

    The amplitudes and frequencies here are deliberately aggressive. An earlier,
    gentler version of this scenario was too easy, and the parameter search
    exploited that: it chose a very stiff filter (large measurement noise, small
    process noise) that looked excellent on smooth arcs and then lost 8 points
    of MOTA the moment the motion actually turned. A benchmark that cannot
    distinguish a good motion model from a lucky one is worse than no benchmark,
    because it produces confident wrong defaults.
    """
    return _build(
        "erratic",
        [
            _Mover(0, 300.0, 250.0, 90.0, 40.0, 65.0, 165.0, depth=0, wobble=150.0, wobble_hz=0.9),
            _Mover(1, 900.0, 450.0, -70.0, -30.0, 80.0, 200.0, depth=1, wobble=180.0, wobble_hz=0.7),
            _Mover(2, 640.0, 150.0, 140.0, 25.0, 55.0, 140.0, depth=2, wobble=120.0, wobble_hz=1.3),
        ],
        frames=frames,
        frame_size=(1280, 720),
        description="sharp non-linear motion that violates the constant-velocity model",
    )


def sparse_scenario(frames: int = 120) -> Scenario:
    """Well-separated objects on clean trajectories - the easy case.

    Included deliberately. A parameter set that wins on the hard scenarios by
    becoming reckless will give itself away here, by inventing tracks or
    swapping identities in a situation where nothing should ever go wrong.
    """
    return _build(
        "sparse",
        [
            _Mover(0, 200.0, 200.0, 120.0, 30.0, 70.0, 170.0, depth=0),
            _Mover(1, 900.0, 500.0, -90.0, -25.0, 75.0, 185.0, depth=1),
        ],
        frames=frames,
        frame_size=(1280, 720),
        description="two well-separated objects, no interaction",
    )


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "sparse": sparse_scenario,
    "crossing": crossing_scenario,
    "occlusion": occlusion_scenario,
    "crowd": crowd_scenario,
    "erratic": erratic_scenario,
}


def build_suite(names: list[str] | None = None) -> list[Scenario]:
    """The scenarios to evaluate against, in increasing order of difficulty."""
    chosen = names or list(SCENARIOS)
    unknown = [name for name in chosen if name not in SCENARIOS]
    if unknown:
        raise ValueError(
            f"unknown scenario(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(SCENARIOS))}"
        )
    return [SCENARIOS[name]() for name in chosen]
