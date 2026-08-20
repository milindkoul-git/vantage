"""Scripted spatial scenarios with ground truth.

Actors follow interpolated paths, so relations arise from geometry rather than
from being asserted. Only the paths are synthetic: the zone tests, the distance
maths and the temporal gating are all the real code.

The negatives are the point
---------------------------
Interaction from 2-D geometry is the weakest claim in this phase and the easiest
to get plausibly wrong, so more than half of these scenarios exist to check that
something does **not** fire:

* ``walk_past_object`` - a person passing an object must not be reported as
  interacting with it, however close the boxes get in passing.
* ``two_people_meet`` - two people standing together are near each other, and
  are *not* "interacting", because geometry alone cannot support that claim
  between two people.
* ``far_apart`` - entities on opposite sides of the frame relate in no way at
  all, which is the case a proximity bug breaks first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import KEYPOINT_NAMES, LEFT_WRIST, Keypoint, Pose, Posture
from vantage.spatial.contracts import Relation, Zone, ZoneEvent
from vantage.tracking.contracts import Track, TrackState

FRAME = (640, 480)
PERSON_HEIGHT = 160.0
OBJECT_HEIGHT = 40.0


@dataclass(frozen=True, slots=True)
class Actor:
    """One entity and the ground path it follows."""

    track_id: int
    label: str
    path: tuple[tuple[float, float], ...]
    """Ground points in pixels. One point means stationary; more are
    interpolated evenly across the scenario."""

    height: float = PERSON_HEIGHT
    width: float = 60.0
    reaches: tuple[float, float] | None = None
    """Wrist position in pixels, when this actor should have a pose with a
    landmark there. ``None`` means no pose at all for this entity."""

    def position(self, fraction: float) -> tuple[float, float]:
        if len(self.path) == 1:
            return self.path[0]
        span = len(self.path) - 1
        scaled = min(max(fraction, 0.0), 1.0) * span
        index = min(int(scaled), span - 1)
        local = scaled - index
        (x1, y1), (x2, y2) = self.path[index], self.path[index + 1]
        return (x1 + (x2 - x1) * local, y1 + (y2 - y1) * local)

    def track(
        self, fraction: float, frame: int, velocity: tuple[float, float] = (0.0, 0.0)
    ) -> Track:
        """A track at this point on the path.

        ``velocity`` has to be supplied rather than defaulted, and the reason is
        a bug this harness had: leaving it at zero made the state estimator call
        every scripted actor STATIONARY however fast their path moved, which
        silently disabled the motion gate that interaction depends on. A
        scenario that cannot express motion cannot test a rule that reads it.
        """
        x, y = self.position(fraction)
        return Track(
            track_id=self.track_id,
            entity_id=f"{self.label}_{self.track_id}",
            box=BoundingBox(x - self.width / 2, y - self.height, x + self.width / 2, y),
            label=self.label,
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

    def pose(self, fraction: float) -> Pose | None:
        if self.reaches is None:
            return None
        track = self.track(fraction, 0)
        keypoints = [Keypoint(0.0, 0.0, 0.0) for _ in KEYPOINT_NAMES]
        keypoints[LEFT_WRIST] = Keypoint(self.reaches[0], self.reaches[1], 0.9)
        return Pose(
            keypoints=tuple(keypoints),
            track_id=self.track_id,
            entity_id=track.entity_id,
            box=track.box,
            posture=Posture.STANDING,
            posture_confidence=0.8,
            model="scenario",
        )


@dataclass(frozen=True, slots=True)
class SpatialScenario:
    """A scripted scene with what should and should not be recognised."""

    name: str
    description: str
    actors: tuple[Actor, ...]
    seconds: float = 6.0
    fps: float = 30.0
    zones: tuple[Zone, ...] = ()
    expect: tuple[tuple[Relation, int, int], ...] = ()
    """Relations that must hold at some point, as ``(relation, track, track)``."""

    forbidden: tuple[tuple[Relation, int, int], ...] = ()
    expect_zone_events: tuple[tuple[int, str, ZoneEvent], ...] = ()
    grace_s: float = 0.0
    """Ignore the first stretch when checking forbidden relations, for
    scenarios that legitimately start in the state being excluded later."""

    metadata: dict[str, object] = field(default_factory=dict)


LEFT_ZONE = Zone(name="left_half", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
DOORWAY = Zone(
    name="doorway",
    kind="entrance",
    points=((0.35, 0.5), (0.65, 0.5), (0.65, 1.0), (0.35, 1.0)),
)

SCENARIOS: dict[str, SpatialScenario] = {
    "zone_crossing": SpatialScenario(
        name="zone_crossing",
        description="One person walks through a doorway zone and out the far side.",
        actors=(Actor(1, "person", ((60.0, 400.0), (580.0, 400.0))),),
        zones=(DOORWAY,),
        expect_zone_events=(
            (1, "doorway", ZoneEvent.ENTERED),
            (1, "doorway", ZoneEvent.EXITED),
        ),
    ),
    "zone_overlap": SpatialScenario(
        name="zone_overlap",
        description=(
            "Overlapping zones. A person in the doorway is also in the left "
            "half, and must be reported in both rather than in whichever "
            "happens to be checked first."
        ),
        actors=(Actor(1, "person", ((230.0, 400.0),)),),
        zones=(LEFT_ZONE, DOORWAY),
        expect_zone_events=(
            (1, "left_half", ZoneEvent.ENTERED),
            (1, "doorway", ZoneEvent.ENTERED),
        ),
    ),
    "two_people_meet": SpatialScenario(
        name="two_people_meet",
        description=(
            "Two people walk together and stop. Near and approaching; never "
            "'interacting', which geometry cannot support between two people."
        ),
        actors=(
            Actor(1, "person", ((80.0, 400.0), (280.0, 400.0), (280.0, 400.0))),
            Actor(2, "person", ((560.0, 400.0), (360.0, 400.0), (360.0, 400.0))),
        ),
        expect=((Relation.APPROACHING, 1, 2), (Relation.NEAR, 1, 2)),
        forbidden=((Relation.INTERACTING, 1, 2),),
    ),
    "two_people_part": SpatialScenario(
        name="two_people_part",
        description="Two people start together and walk apart.",
        actors=(
            Actor(1, "person", ((300.0, 400.0), (60.0, 400.0))),
            Actor(2, "person", ((340.0, 400.0), (580.0, 400.0))),
        ),
        expect=((Relation.RECEDING, 1, 2),),
        forbidden=((Relation.APPROACHING, 1, 2),),
    ),
    "far_apart": SpatialScenario(
        name="far_apart",
        description="Entities at opposite edges relate in no way at all.",
        actors=(
            Actor(1, "person", ((60.0, 400.0),)),
            Actor(2, "person", ((600.0, 400.0),)),
        ),
        forbidden=(
            (Relation.NEAR, 1, 2),
            (Relation.APPROACHING, 1, 2),
            (Relation.INTERACTING, 1, 2),
        ),
    ),
    "walk_past_object": SpatialScenario(
        name="walk_past_object",
        description=(
            "A person walks straight past a static object, passing very close. "
            "The single most important negative: 2-D proximity in passing is "
            "not interaction."
        ),
        actors=(
            Actor(1, "person", ((60.0, 400.0), (600.0, 400.0))),
            Actor(2, "laptop", ((330.0, 400.0),), height=OBJECT_HEIGHT, width=60.0),
        ),
        seconds=4.0,
        forbidden=((Relation.INTERACTING, 1, 2),),
    ),
    "amble_past_object": SpatialScenario(
        name="amble_past_object",
        description=(
            "The same walk-past, slowly. Regression, and the reason interaction "
            "needs motion state: at a brisk 180 px/s duration alone excluded it, "
            "but at an ambling 45 px/s the sustain threshold was satisfied and "
            "49 frames of false interaction were reported. Raising interact_s "
            "only moves the speed at which that breaks."
        ),
        actors=(
            Actor(1, "person", ((60.0, 400.0), (600.0, 400.0))),
            Actor(2, "laptop", ((330.0, 400.0),), height=OBJECT_HEIGHT, width=60.0),
        ),
        seconds=12.0,
        forbidden=((Relation.INTERACTING, 1, 2),),
    ),
    "reach_while_walking": SpatialScenario(
        name="reach_while_walking",
        description=(
            "A person takes something in passing. A confirmed reach counts on "
            "its own, because a wrist landmark inside the box is direct "
            "evidence rather than an inference from two rectangles."
        ),
        actors=(
            Actor(1, "person", ((300.0, 400.0), (360.0, 400.0)), reaches=(330.0, 380.0)),
            Actor(2, "laptop", ((330.0, 400.0),), height=OBJECT_HEIGHT, width=60.0),
        ),
        seconds=5.0,
        expect=((Relation.INTERACTING, 1, 2),),
        metadata={"expected_confidence": 0.85},
    ),
    "linger_by_object": SpatialScenario(
        name="linger_by_object",
        description=(
            "A person stops beside an object and stays. Interaction is claimed, "
            "but only weakly - no reach was observed, and a flat image cannot "
            "rule out someone standing well behind it."
        ),
        actors=(
            Actor(1, "person", ((300.0, 400.0),)),
            Actor(2, "laptop", ((330.0, 400.0),), height=OBJECT_HEIGHT, width=60.0),
        ),
        expect=((Relation.INTERACTING, 1, 2),),
        metadata={"expected_confidence": 0.4},
    ),
    "reach_for_object": SpatialScenario(
        name="reach_for_object",
        description=(
            "The same geometry, but a wrist landmark falls inside the object's "
            "box. Reach-confirmed interaction, and materially more confident."
        ),
        actors=(
            Actor(1, "person", ((300.0, 400.0),), reaches=(330.0, 380.0)),
            Actor(2, "laptop", ((330.0, 400.0),), height=OBJECT_HEIGHT, width=60.0),
        ),
        expect=((Relation.INTERACTING, 1, 2),),
        metadata={"expected_confidence": 0.85},
    ),
}


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """One generated frame."""

    index: int
    time_s: float
    tracks: tuple[Track, ...]
    poses: tuple[Pose, ...]
    elapsed_s: float


def generate(scenario: SpatialScenario) -> list[SpatialFrame]:
    """Expand a scenario into per-frame tracks and poses."""
    frames: list[SpatialFrame] = []
    total = max(1, int(round(scenario.seconds * scenario.fps)))
    dt = 1.0 / scenario.fps

    for index in range(total):
        fraction = index / max(1, total - 1)
        previous = max(0, index - 1) / max(1, total - 1)
        tracks = tuple(
            actor.track(fraction, index, _velocity(actor, previous, fraction, dt))
            for actor in scenario.actors
        )
        poses = tuple(
            pose
            for pose in (actor.pose(fraction) for actor in scenario.actors)
            if pose is not None
        )
        frames.append(
            SpatialFrame(
                index=index,
                time_s=index * dt,
                tracks=tracks,
                poses=poses,
                elapsed_s=dt,
            )
        )
    return frames


def _velocity(actor: Actor, previous: float, current: float, dt: float) -> tuple[float, float]:
    """Pixels per second along the actor's path, by finite difference."""
    if previous == current or dt <= 0:
        return (0.0, 0.0)
    x0, y0 = actor.position(previous)
    x1, y1 = actor.position(current)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


def build_suite(names: list[str] | None = None) -> list[SpatialScenario]:
    if not names:
        return list(SCENARIOS.values())
    missing = [name for name in names if name not in SCENARIOS]
    if missing:
        raise KeyError(f"unknown scenarios {missing}; available: {sorted(SCENARIOS)}")
    return [SCENARIOS[name] for name in names]
