"""Scripted activity scenarios with ground truth.

Phase 3 established that the evaluation harness was worth more than the tracker
it measured - it found an off-by-one, an off-frame coasting bug worth five
points of MOTA, and two separate cases of a parameter search overfitting. The
same applies here with more force, because rules that look obviously correct on
paper are exactly the kind that fail on a boundary nobody thought to check.

What is synthetic and what is real
----------------------------------
A scenario scripts **motion and posture over time**, and everything downstream
of that is the real code: synthetic tracks are fed through the real
:class:`~vantage.state.estimator.StateEstimator`, whose output feeds the real
recogniser. So the harness measures the state machine and the rules *together*,
including the hysteresis and dwell timing they depend on. What it does not
measure is pose estimation - a scenario states the posture rather than deriving
it from pixels, because the accuracy of RTMPose is not what these rules are
being tested for.

The negatives matter as much as the positives
---------------------------------------------
Half of these scenarios exist to check that something does **not** fire. A
recogniser that reports ``falling`` whenever anyone lies down would score
perfectly on the positive cases and be useless in a building.
"""

from __future__ import annotations

from dataclasses import dataclass

from vantage.activity.contracts import Activity
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    KEYPOINT_NAMES,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Keypoint,
    Pose,
    Posture,
)
from vantage.tracking.contracts import Track, TrackState

BOX_HEIGHT = 160.0
"""Person box height in pixels. Speeds are expressed in heights per second, so
the absolute value only has to be large enough for the posture rules'
foreshortening guard."""


@dataclass(frozen=True, slots=True)
class Beat:
    """A stretch of time during which the entity does one thing."""

    seconds: float
    speed: float = 0.0
    """Heights per second of horizontal movement."""

    posture: Posture | None = Posture.STANDING
    """``None`` means pose is unavailable for this stretch - the entity is
    tracked but not landmarked."""

    arm_raised: bool = False
    expect: frozenset[Activity] = frozenset()
    """Continuous activities that should be reported during this beat, once any
    sustain window has elapsed."""

    grace_s: float = 1.0
    """How long into the beat to ignore before checking :attr:`expect`. Rules
    need their sustain windows, and the state machine needs its minimum hold;
    scoring the transition itself would measure the delay, not the decision."""


@dataclass(frozen=True, slots=True)
class ActivityScenario:
    """A scripted sequence with what should and should not be recognised."""

    name: str
    description: str
    beats: tuple[Beat, ...]
    events: tuple[Activity, ...] = ()
    """Transient activities that must each fire exactly once."""

    forbidden: frozenset[Activity] = frozenset()
    """Activities that must never appear. The point of the negative cases."""

    fps: float = 30.0

    @property
    def duration_s(self) -> float:
        return sum(beat.seconds for beat in self.beats)


def skeleton(
    posture: Posture, x: float, arm_raised: bool = False, confidence: float = 0.9
) -> dict[int, tuple[float, float]]:
    """Joint positions for a posture, centred on ``x``.

    Geometry mirrors what :mod:`vantage.pose.posture` reads: a 0.4-height torso,
    then legs placed to produce the hip-knee and knee-ankle drops each posture
    is defined by. Built from the same measurements the rules use, so a change
    to one without the other shows up as a failing scenario rather than as
    quietly wrong output.
    """
    top = 40.0
    torso = BOX_HEIGHT * 0.4
    shoulder_y = top
    hip_y = top + torso

    if posture is Posture.STANDING:
        knee_y, ankle_y = hip_y + torso * 0.95, hip_y + torso * 1.9
    elif posture is Posture.SITTING:
        knee_y, ankle_y = hip_y + torso * 0.10, hip_y + torso * 0.95
    elif posture is Posture.CROUCHING:
        knee_y, ankle_y = hip_y + torso * 0.15, hip_y + torso * 0.35
    else:  # lying: the torso runs horizontally, so hips sit beside the shoulders
        hip_y = shoulder_y + torso * 0.08
        knee_y, ankle_y = hip_y + torso * 0.05, hip_y + torso * 0.10

    sideways = torso if posture is Posture.LYING else 0.0
    points = {
        LEFT_SHOULDER: (x - 18.0, shoulder_y),
        RIGHT_SHOULDER: (x + 18.0, shoulder_y),
        LEFT_HIP: (x - 14.0 + sideways, hip_y),
        RIGHT_HIP: (x + 14.0 + sideways, hip_y),
        LEFT_KNEE: (x - 14.0 + sideways * 1.6, knee_y),
        RIGHT_KNEE: (x + 14.0 + sideways * 1.6, knee_y),
        LEFT_ANKLE: (x - 14.0 + sideways * 2.2, ankle_y),
        RIGHT_ANKLE: (x + 14.0 + sideways * 2.2, ankle_y),
    }
    wrist_y = shoulder_y - torso * 0.5 if arm_raised else hip_y
    points[LEFT_WRIST] = (x - 26.0, wrist_y)
    points[RIGHT_WRIST] = (x + 26.0, hip_y)
    return points


def make_pose(
    posture: Posture,
    x: float,
    box: BoundingBox,
    arm_raised: bool = False,
    confidence: float = 0.9,
) -> Pose:
    """A full 17-point pose for a posture, with unplaced joints scored zero."""
    from vantage.pose.posture import classify

    points = skeleton(posture, x, arm_raised)
    keypoints = tuple(
        Keypoint(*points[i], confidence) if i in points else Keypoint(0.0, 0.0, 0.0)
        for i in range(len(KEYPOINT_NAMES))
    )
    draft = Pose(keypoints=keypoints, track_id=1, entity_id="person_1", box=box)
    estimate = classify(draft)
    return Pose(
        keypoints=keypoints,
        track_id=1,
        entity_id="person_1",
        box=box,
        posture=estimate.posture,
        posture_confidence=estimate.confidence,
        posture_reason=estimate.reason,
        model="scenario",
    )


def make_track(
    x: float,
    velocity: tuple[float, float],
    frame: int,
    posture: Posture | None = None,
) -> Track:
    """One tracked person, with a box that matches the posture it is holding.

    The box used to be a fixed portrait rectangle whatever the skeleton was
    doing, so a scenario person lying on the floor still had the silhouette of
    one standing up. That is not a shape a detector produces, and posture now
    cross-checks the two: a horizontal torso inside an upright box is read as a
    contradiction rather than as a fall. A harness that feeds an impossible
    combination is testing something the world cannot present.
    """
    if posture is Posture.LYING:
        # Lengthwise on the ground: as wide as the person is tall, and about a
        # shoulder-width deep.
        half_length = BOX_HEIGHT / 2.0
        top = 40.0 + BOX_HEIGHT - 60.0
        box = BoundingBox(x - half_length, top, x + half_length, top + 60.0)
    else:
        box = BoundingBox(x - 30.0, 40.0, x + 30.0, 40.0 + BOX_HEIGHT)
    return Track(
        track_id=1,
        entity_id="person_1",
        box=box,
        label="person",
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=frame + 1,
        hits=frame + 1,
        time_since_update=0,
        start_frame=0,
        last_frame=frame,
        velocity=velocity,
    )


WALK = frozenset({Activity.WALKING})
RUN = frozenset({Activity.RUNNING})
LOITER = frozenset({Activity.LOITERING})

SCENARIOS: dict[str, ActivityScenario] = {
    "walk": ActivityScenario(
        name="walk",
        description="A person walks steadily across the frame.",
        beats=(
            Beat(seconds=1.0, speed=0.0, expect=frozenset()),
            Beat(seconds=5.0, speed=0.7, expect=WALK),
        ),
        forbidden=frozenset({Activity.RUNNING, Activity.FALLING, Activity.LOITERING}),
    ),
    "run": ActivityScenario(
        name="run",
        description="A person runs. Must not be reported as walking.",
        beats=(
            Beat(seconds=1.0, speed=0.0),
            Beat(seconds=4.0, speed=2.0, expect=RUN),
        ),
        forbidden=frozenset({Activity.FALLING, Activity.LOITERING}),
    ),
    "loiter": ActivityScenario(
        name="loiter",
        description="A person arrives, then stays put well past the dwell threshold.",
        beats=(
            Beat(seconds=3.0, speed=0.8, expect=WALK),
            Beat(seconds=30.0, speed=0.0, expect=LOITER, grace_s=22.0),
        ),
        forbidden=frozenset({Activity.RUNNING, Activity.FALLING}),
    ),
    "sit_down": ActivityScenario(
        name="sit_down",
        description="A person stands, sits, and stays seated.",
        beats=(
            Beat(seconds=2.5, speed=0.0, posture=Posture.STANDING),
            Beat(seconds=3.0, speed=0.0, posture=Posture.SITTING),
        ),
        events=(Activity.SITTING_DOWN,),
        forbidden=frozenset({Activity.FALLING, Activity.RUNNING, Activity.STANDING_UP}),
    ),
    "stand_up": ActivityScenario(
        name="stand_up",
        description="A seated person rises.",
        beats=(
            Beat(seconds=2.5, speed=0.0, posture=Posture.SITTING),
            Beat(seconds=3.0, speed=0.0, posture=Posture.STANDING),
        ),
        events=(Activity.STANDING_UP,),
        forbidden=frozenset({Activity.FALLING, Activity.SITTING_DOWN}),
    ),
    "fall": ActivityScenario(
        name="fall",
        description="A person goes from standing to lying with nothing in between.",
        beats=(
            Beat(seconds=3.0, speed=0.0, posture=Posture.STANDING),
            Beat(seconds=3.0, speed=0.0, posture=Posture.LYING),
        ),
        events=(Activity.FALLING,),
        forbidden=frozenset({Activity.SITTING_DOWN, Activity.RUNNING}),
    ),
    "fall_after_standing_a_while": ActivityScenario(
        name="fall_after_standing_a_while",
        description=(
            "The same fall, but after a long stand. Regression: timing the "
            "transition from when the previous posture became stable made this "
            "undetectable while a fall moments after standing up was caught."
        ),
        beats=(
            Beat(seconds=12.0, speed=0.0, posture=Posture.STANDING),
            Beat(seconds=3.0, speed=0.0, posture=Posture.LYING),
        ),
        events=(Activity.FALLING,),
        forbidden=frozenset({Activity.SITTING_DOWN}),
    ),
    "lie_down_slowly": ActivityScenario(
        name="lie_down_slowly",
        description=(
            "A deliberate lie-down, via a crouch. Must NOT be a fall - the "
            "single most important negative case here."
        ),
        beats=(
            Beat(seconds=3.0, speed=0.0, posture=Posture.STANDING),
            Beat(seconds=2.5, speed=0.0, posture=Posture.CROUCHING),
            Beat(seconds=2.5, speed=0.0, posture=Posture.SITTING),
            Beat(seconds=3.0, speed=0.0, posture=Posture.LYING),
        ),
        forbidden=frozenset({Activity.FALLING}),
    ),
    "arm_raised": ActivityScenario(
        name="arm_raised",
        description="A stationary person raises an arm and holds it.",
        beats=(
            Beat(seconds=2.0, speed=0.0, posture=Posture.STANDING),
            Beat(
                seconds=3.0,
                speed=0.0,
                posture=Posture.STANDING,
                arm_raised=True,
                expect=frozenset({Activity.ARM_RAISED}),
            ),
        ),
        forbidden=frozenset({Activity.FALLING, Activity.RUNNING}),
    ),
    "no_pose": ActivityScenario(
        name="no_pose",
        description=(
            "Walking with pose disabled. Locomotion must still work, and no "
            "posture-derived activity may be invented from nothing."
        ),
        beats=(
            Beat(seconds=1.0, speed=0.0, posture=None),
            Beat(seconds=4.0, speed=0.7, posture=None, expect=WALK),
        ),
        forbidden=frozenset(
            {
                Activity.FALLING,
                Activity.SITTING_DOWN,
                Activity.STANDING_UP,
                Activity.ARM_RAISED,
            }
        ),
    ),
    "jitter": ActivityScenario(
        name="jitter",
        description=(
            "A standing person whose detector box wobbles. Nothing should be "
            "recognised but idle and, eventually, loitering."
        ),
        beats=(
            Beat(
                seconds=25.0, speed=0.03, posture=Posture.STANDING, grace_s=22.0, expect=LOITER
            ),
        ),
        forbidden=frozenset(
            {Activity.WALKING, Activity.RUNNING, Activity.FALLING, Activity.SITTING_DOWN}
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class ScenarioFrame:
    """One generated frame: what to feed in, and what should come out."""

    index: int
    time_s: float
    track: Track
    pose: Pose | None
    expected: frozenset[Activity]
    scored: bool
    """Whether this frame is past its beat's grace period and counts."""

    elapsed_s: float


def generate(scenario: ActivityScenario) -> list[ScenarioFrame]:
    """Expand a scenario into per-frame inputs and expectations."""
    frames: list[ScenarioFrame] = []
    dt = 1.0 / scenario.fps
    x = 200.0
    index = 0
    time_s = 0.0

    for beat in scenario.beats:
        count = max(1, int(round(beat.seconds * scenario.fps)))
        beat_start = time_s
        for _ in range(count):
            velocity = (beat.speed * BOX_HEIGHT, 0.0)
            track = make_track(x, velocity, index, beat.posture)
            pose = (
                make_pose(beat.posture, x, track.box, beat.arm_raised)
                if beat.posture is not None
                else None
            )
            frames.append(
                ScenarioFrame(
                    index=index,
                    time_s=time_s,
                    track=track,
                    pose=pose,
                    expected=beat.expect,
                    scored=(time_s - beat_start) >= beat.grace_s,
                    elapsed_s=dt,
                )
            )
            x += beat.speed * BOX_HEIGHT * dt
            time_s += dt
            index += 1
    return frames


def build_suite(names: list[str] | None = None) -> list[ActivityScenario]:
    """The named scenarios, or all of them in declaration order."""
    if not names:
        return list(SCENARIOS.values())
    missing = [name for name in names if name not in SCENARIOS]
    if missing:
        raise KeyError(f"unknown scenarios {missing}; available: {sorted(SCENARIOS)}")
    return [SCENARIOS[name] for name in names]
