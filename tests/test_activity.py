"""Activity recognition: the rules, the engine, and the ground-truth suite.

No model, no weights, no runtime - activity is arithmetic over signals the
platform already produces, so everything here is checkable against hand-built
input.

The scenario suite at the bottom is the real gate. The unit tests below it pin
individual rules at their boundaries; the suite runs whole scripted sequences
through the *real* state estimator and checks both what fires and what must
never fire.
"""

from __future__ import annotations

import pytest

from vantage.activity.contracts import (
    Activity,
    ActivityObservation,
    ActivityResult,
    EntityActivity,
    to_observation_record,
)
from vantage.activity.engine import ActivityEngine
from vantage.activity.recognizer import ActivityParams, RuleRecognizer
from vantage.core.errors import ConfigError
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    KEYPOINT_NAMES,
    LEFT_SHOULDER,
    LEFT_WRIST,
    Keypoint,
    Pose,
    Posture,
)
from vantage.state.contracts import EntityState, MotionState

BOX = BoundingBox(100.0, 40.0, 160.0, 200.0)
DT = 1.0 / 30.0


def make_state(
    *,
    speed: float = 0.0,
    motion: MotionState = MotionState.STATIONARY,
    dwell_s: float = 0.0,
    track_id: int = 1,
    label: str = "person",
) -> EntityState:
    return EntityState(
        track_id=track_id,
        entity_id=f"{label}_{track_id}",
        label=label,
        motion=motion,
        speed=speed,
        dwell_s=dwell_s,
        bearing_deg=90.0 if motion is MotionState.MOVING else None,
        distance=0.0,
        age_s=10.0,
        observed=True,
    )


def make_pose(
    posture: Posture,
    *,
    confidence: float = 0.9,
    arm_raised: bool = False,
    wrist_confidence: float = 0.9,
    shoulder_confidence: float = 0.9,
) -> Pose:
    keypoints = [Keypoint(0.0, 0.0, 0.0) for _ in KEYPOINT_NAMES]
    keypoints[LEFT_SHOULDER] = Keypoint(120.0, 80.0, shoulder_confidence)
    keypoints[LEFT_WRIST] = Keypoint(120.0, 40.0 if arm_raised else 140.0, wrist_confidence)
    return Pose(
        keypoints=tuple(keypoints),
        track_id=1,
        entity_id="person_1",
        box=BOX,
        posture=posture,
        posture_confidence=confidence,
    )


def drive(
    recognizer: RuleRecognizer,
    *,
    seconds: float,
    state_factory,
    pose_factory=None,
    start: float = 0.0,
) -> tuple[EntityActivity, float]:
    """Feed a constant condition for a while; return the last report and time."""
    now = start
    activity = None
    for _ in range(max(1, int(round(seconds / DT)))):
        now += DT
        activity = recognizer.observe(
            state_factory(now), pose_factory(now) if pose_factory else None, now
        )
    return activity, now


class TestContracts:
    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError, match=r"confidence"):
            ActivityObservation(Activity.WALKING, 1.4, 0.0, "x")

    def test_transient_events_outrank_continuous_ones(self) -> None:
        """A fall matters more than the fact the person had been walking."""
        entity = EntityActivity(
            track_id=1,
            entity_id="person_1",
            label="person",
            observations=(
                ActivityObservation(Activity.WALKING, 0.95, 3.0, "fast"),
                ActivityObservation(Activity.FALLING, 0.40, 0.1, "quick drop"),
            ),
        )
        assert entity.primary.activity is Activity.FALLING

    def test_idle_never_outranks_a_real_activity(self) -> None:
        entity = EntityActivity(
            track_id=1,
            entity_id="person_1",
            label="person",
            observations=(
                ActivityObservation(Activity.IDLE, 1.0, 5.0, "nothing"),
                ActivityObservation(Activity.WALKING, 0.5, 1.0, "moving"),
            ),
        )
        assert entity.primary.activity is Activity.WALKING

    def test_counts_may_exceed_the_entity_count(self) -> None:
        """Activities overlap on purpose; a total that matched would be wrong."""
        entity = EntityActivity(
            track_id=1,
            entity_id="person_1",
            label="person",
            observations=(
                ActivityObservation(Activity.WALKING, 0.9, 1.0, "a"),
                ActivityObservation(Activity.ARM_RAISED, 0.8, 1.0, "b"),
            ),
        )
        result = ActivityResult(
            entities=(entity,), source_id="t", frame_index=1, capture_wall=1.0
        )
        assert sum(result.counts().values()) == 2
        assert len(result) == 1

    def test_pose_derived_activities_are_flagged(self) -> None:
        assert Activity.FALLING.needs_pose
        assert not Activity.WALKING.needs_pose

    def test_observation_record_is_json_serialisable(self) -> None:
        import json

        entity = EntityActivity(
            track_id=1,
            entity_id="person_1",
            label="person",
            observations=(ActivityObservation(Activity.LOITERING, 0.7, 45.0, "45s"),),
        )
        record = to_observation_record(entity, "camera_01", 1_700_000_000.0)
        json.dumps(record)
        assert record["identity"] is None
        assert record["observations"][0]["type"] == "loitering"
        assert record["timestamp"].startswith("2023-")


class TestParams:
    def test_running_must_exceed_walking(self) -> None:
        with pytest.raises(ConfigError, match=r"must exceed"):
            ActivityParams(walking_speed=1.0, running_speed=0.5)

    def test_fall_window_cannot_exceed_the_transition_window(self) -> None:
        with pytest.raises(ConfigError, match=r"fall_window_s"):
            ActivityParams(fall_window_s=5.0, transition_window_s=2.0)

    def test_negative_durations_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            ActivityParams(loiter_s=-1.0)


class TestLocomotion:
    def test_sustained_walking_is_reported(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer,
            seconds=2.0,
            state_factory=lambda t: make_state(speed=0.7, motion=MotionState.MOVING),
        )
        assert activity.has(Activity.WALKING)
        assert not activity.has(Activity.RUNNING)

    def test_one_fast_frame_is_not_running(self) -> None:
        """A detector box that jumped is not a person sprinting."""
        recognizer = RuleRecognizer()
        drive(
            recognizer,
            seconds=2.0,
            state_factory=lambda t: make_state(speed=0.7, motion=MotionState.MOVING),
        )
        spike = recognizer.observe(make_state(speed=6.0, motion=MotionState.MOVING), None, 99.0)
        assert not spike.has(Activity.RUNNING)

    def test_sustained_running_is_reported(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer,
            seconds=2.0,
            state_factory=lambda t: make_state(speed=2.0, motion=MotionState.MOVING),
        )
        assert activity.has(Activity.RUNNING)
        assert not activity.has(Activity.WALKING)

    def test_a_new_entity_reports_nothing_until_sustained(self) -> None:
        """Regression: the sustain check once compared a window against its own
        width, which floating point loses, and every continuous rule scored zero."""
        recognizer = RuleRecognizer()
        first = recognizer.observe(make_state(speed=0.7, motion=MotionState.MOVING), None, DT)
        assert first.has(Activity.IDLE)

        activity, _ = drive(
            recognizer,
            seconds=1.0,
            state_factory=lambda t: make_state(speed=0.7, motion=MotionState.MOVING),
            start=DT,
        )
        assert activity.has(Activity.WALKING)

    def test_stationary_entity_does_not_walk(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer, seconds=2.0, state_factory=lambda t: make_state(speed=0.9)
        )
        assert not activity.has(Activity.WALKING)


class TestLoitering:
    def test_fires_past_the_threshold(self) -> None:
        recognizer = RuleRecognizer(ActivityParams(loiter_s=5.0))
        activity, _ = drive(
            recognizer,
            seconds=1.0,
            state_factory=lambda t: make_state(dwell_s=9.0),
        )
        assert activity.has(Activity.LOITERING)

    def test_silent_below_the_threshold(self) -> None:
        recognizer = RuleRecognizer(ActivityParams(loiter_s=20.0))
        activity, _ = drive(
            recognizer, seconds=1.0, state_factory=lambda t: make_state(dwell_s=3.0)
        )
        assert not activity.has(Activity.LOITERING)
        assert activity.has(Activity.IDLE)

    def test_confidence_grows_with_dwell(self) -> None:
        recognizer = RuleRecognizer(ActivityParams(loiter_s=10.0))
        short, _ = drive(
            recognizer, seconds=0.5, state_factory=lambda t: make_state(dwell_s=11.0)
        )
        recognizer.reset()
        long, _ = drive(
            recognizer, seconds=0.5, state_factory=lambda t: make_state(dwell_s=40.0)
        )
        assert (
            long.get(Activity.LOITERING).confidence > short.get(Activity.LOITERING).confidence
        )


class TestPostureTransitions:
    def settle(self, recognizer: RuleRecognizer, posture: Posture, start: float, seconds=1.5):
        return drive(
            recognizer,
            seconds=seconds,
            state_factory=lambda t: make_state(),
            pose_factory=lambda t: make_pose(posture),
            start=start,
        )

    def test_sitting_down_is_detected(self) -> None:
        recognizer = RuleRecognizer()
        _, now = self.settle(recognizer, Posture.STANDING, 0.0)
        activity, _ = self.settle(recognizer, Posture.SITTING, now)
        assert activity.has(Activity.SITTING_DOWN)

    def test_standing_up_is_detected(self) -> None:
        recognizer = RuleRecognizer()
        _, now = self.settle(recognizer, Posture.SITTING, 0.0)
        activity, _ = self.settle(recognizer, Posture.STANDING, now)
        assert activity.has(Activity.STANDING_UP)

    def test_fall_after_a_long_stand_is_still_detected(self) -> None:
        """Regression, and the reason the harness exists.

        Timing the transition from when the previous posture became *stable*
        means a long stand reads as a long transition, so the fall is discarded
        for exceeding the window. A fall moments after standing up was caught
        and a fall after a minute of standing was not - exactly backwards.
        """
        recognizer = RuleRecognizer()
        _, now = self.settle(recognizer, Posture.STANDING, 0.0, seconds=30.0)
        activity, _ = self.settle(recognizer, Posture.LYING, now)
        assert activity.has(Activity.FALLING)

    def test_a_slow_lie_down_is_not_a_fall(self) -> None:
        """The most important negative case in the whole phase."""
        recognizer = RuleRecognizer()
        _, now = self.settle(recognizer, Posture.STANDING, 0.0)
        _, now = self.settle(recognizer, Posture.CROUCHING, now, seconds=2.0)
        _, now = self.settle(recognizer, Posture.SITTING, now, seconds=2.0)
        activity, _ = self.settle(recognizer, Posture.LYING, now, seconds=2.0)
        assert not activity.has(Activity.FALLING)

    def test_flickering_posture_does_not_fire_transitions(self) -> None:
        """Raw posture flickers; only stable posture may drive a transition."""
        recognizer = RuleRecognizer()
        now = 0.0
        fired = 0
        for i in range(120):
            now += DT
            posture = Posture.STANDING if i % 2 else Posture.SITTING
            activity = recognizer.observe(make_state(), make_pose(posture), now)
            if activity.has(Activity.SITTING_DOWN) or activity.has(Activity.STANDING_UP):
                fired += 1
        assert fired == 0

    def test_low_confidence_postures_are_ignored(self) -> None:
        recognizer = RuleRecognizer(ActivityParams(min_posture_confidence=0.5))
        _, now = drive(
            recognizer,
            seconds=1.5,
            state_factory=lambda t: make_state(),
            pose_factory=lambda t: make_pose(Posture.STANDING, confidence=0.1),
        )
        activity, _ = drive(
            recognizer,
            seconds=1.5,
            state_factory=lambda t: make_state(),
            pose_factory=lambda t: make_pose(Posture.LYING, confidence=0.1),
            start=now,
        )
        assert not activity.has(Activity.FALLING)

    def test_transient_events_expire(self) -> None:
        recognizer = RuleRecognizer(ActivityParams(transient_hold_s=0.5))
        _, now = self.settle(recognizer, Posture.STANDING, 0.0)
        # Long enough for SITTING to become the stable posture and fire, but
        # well inside the 0.5s hold this is checking the expiry of.
        activity, now = self.settle(recognizer, Posture.SITTING, now, seconds=0.5)
        assert activity.has(Activity.SITTING_DOWN)
        later, _ = self.settle(recognizer, Posture.SITTING, now, seconds=2.0)
        assert not later.has(Activity.SITTING_DOWN)


class TestArmRaised:
    def test_sustained_raise_is_reported(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer,
            seconds=1.5,
            state_factory=lambda t: make_state(),
            pose_factory=lambda t: make_pose(Posture.STANDING, arm_raised=True),
        )
        assert activity.has(Activity.ARM_RAISED)

    def test_lowered_arm_is_not_reported(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer,
            seconds=1.5,
            state_factory=lambda t: make_state(),
            pose_factory=lambda t: make_pose(Posture.STANDING, arm_raised=False),
        )
        assert not activity.has(Activity.ARM_RAISED)

    def test_an_unseen_wrist_is_not_a_raised_arm(self) -> None:
        """The trap: an unobserved joint sits at the frame origin.

        (0, 0) is above every shoulder, so a wrist that was never located reads
        as permanently raised unless confidence is checked first.
        """
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer,
            seconds=1.5,
            state_factory=lambda t: make_state(),
            pose_factory=lambda t: make_pose(
                Posture.STANDING, arm_raised=True, wrist_confidence=0.0
            ),
        )
        assert not activity.has(Activity.ARM_RAISED)

    def test_a_flickering_arm_is_not_held_up(self) -> None:
        recognizer = RuleRecognizer()
        now = 0.0
        activity = None
        for i in range(90):
            now += DT
            activity = recognizer.observe(
                make_state(), make_pose(Posture.STANDING, arm_raised=bool(i % 2)), now
            )
        assert not activity.has(Activity.ARM_RAISED)


class TestWithoutPose:
    def test_locomotion_still_works(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(
            recognizer,
            seconds=2.0,
            state_factory=lambda t: make_state(speed=0.7, motion=MotionState.MOVING),
        )
        assert activity.has(Activity.WALKING)

    def test_no_posture_activity_is_invented(self) -> None:
        recognizer = RuleRecognizer()
        activity, _ = drive(recognizer, seconds=5.0, state_factory=lambda t: make_state())
        for posture_derived in (
            Activity.FALLING,
            Activity.SITTING_DOWN,
            Activity.STANDING_UP,
            Activity.ARM_RAISED,
        ):
            assert not activity.has(posture_derived)


class TestLifecycle:
    def test_idle_is_reported_rather_than_nothing(self) -> None:
        recognizer = RuleRecognizer()
        activity = recognizer.observe(make_state(), None, DT)
        assert activity.has(Activity.IDLE)

    def test_retired_entities_are_pruned(self) -> None:
        recognizer = RuleRecognizer()
        for track_id in range(4):
            recognizer.observe(make_state(track_id=track_id), None, DT)
        assert recognizer.tracked == 4
        recognizer.forget({0})
        assert recognizer.tracked == 1

    def test_reset_clears_everything(self) -> None:
        recognizer = RuleRecognizer()
        recognizer.observe(make_state(), None, DT)
        recognizer.reset()
        assert recognizer.tracked == 0

    def test_recognizer_satisfies_the_protocol(self) -> None:
        from vantage.activity.base import Recognizer

        assert isinstance(RuleRecognizer(), Recognizer)


class TestEngine:
    def state_result(self, states, elapsed=DT, index=0):
        from vantage.state.contracts import StateResult

        return StateResult(
            states=tuple(states),
            source_id="t",
            frame_index=index,
            capture_wall=index * elapsed,
            elapsed_s=elapsed,
        )

    def test_time_comes_from_elapsed_not_a_clock(self) -> None:
        """So a recorded source replays identically, whatever the machine does."""
        engine = ActivityEngine()
        for i in range(10):
            engine.update(self.state_result([make_state()], elapsed=0.5, index=i))
        assert engine.elapsed_s == pytest.approx(5.0)

    def test_poses_are_paired_by_track(self) -> None:
        from vantage.pose.contracts import PoseResult

        engine = ActivityEngine()
        pose = Pose(
            keypoints=make_pose(Posture.STANDING).keypoints,
            track_id=2,
            entity_id="person_2",
            box=BOX,
            posture=Posture.STANDING,
            posture_confidence=0.9,
        )
        result = engine.update(
            self.state_result([make_state(track_id=1), make_state(track_id=2)]),
            PoseResult(
                poses=(pose,),
                source_id="t",
                frame_index=0,
                capture_wall=0.0,
                frame_size=(640, 480),
                people_seen=2,
            ),
        )
        assert len(result) == 2
        assert result.pose_available

    def test_pose_absent_is_recorded(self) -> None:
        """ "Did not happen" and "could not be seen" are different answers."""
        engine = ActivityEngine()
        result = engine.update(self.state_result([make_state()]))
        assert not result.pose_available

    def test_entities_are_pruned_through_the_engine(self) -> None:
        engine = ActivityEngine()
        engine.update(self.state_result([make_state(track_id=i) for i in range(3)]))
        engine.update(self.state_result([make_state(track_id=0)], index=1))
        assert engine.recognizer.tracked == 1

    def test_notable_excludes_idle(self) -> None:
        engine = ActivityEngine()
        result = engine.update(self.state_result([make_state()]))
        assert len(result) == 1
        assert result.notable() == ()


class TestScenarioSuite:
    """The gate: whole scripted sequences through the real state estimator."""

    def all_results(self):
        from vantage.activity.evaluation import evaluate
        from vantage.activity.scenarios import build_suite

        return [evaluate(scenario) for scenario in build_suite()]

    def test_every_scenario_passes(self) -> None:
        failures = [m.scenario for m in self.all_results() if not m.passed]
        assert not failures, f"scenarios failed: {failures}"

    def test_nothing_forbidden_ever_fires(self) -> None:
        for metrics in self.all_results():
            assert metrics.forbidden_firings == 0, f"{metrics.scenario}: {metrics.unexpected}"

    def test_every_expected_event_fires_exactly_once(self) -> None:
        """Twice is two alerts for one fall."""
        for metrics in self.all_results():
            assert metrics.events_found == len(metrics.events_expected), metrics.scenario
            assert metrics.event_duplicates == 0, metrics.scenario

    def test_events_are_detected_promptly(self) -> None:
        """Latency is a product decision, so it is asserted rather than observed."""
        for metrics in self.all_results():
            for name, latency in metrics.event_latency_s.items():
                assert latency <= 1.0, f"{metrics.scenario}/{name} took {latency:.2f}s"

    def test_the_suite_covers_both_outcomes(self) -> None:
        """A suite of only positive cases would prove very little."""
        from vantage.activity.scenarios import SCENARIOS

        assert any(s.events for s in SCENARIOS.values())
        assert any(s.forbidden for s in SCENARIOS.values())
        assert sum(len(s.forbidden) for s in SCENARIOS.values()) >= 10


class TestConfigWiring:
    def test_activity_requires_state(self) -> None:
        from vantage.config.schema import (
            DetectionConfig,
            StateConfig,
            TrackingConfig,
            VantageConfig,
        )

        with pytest.raises(ConfigError, match=r"requires state.enabled"):
            VantageConfig(
                detection=DetectionConfig(enabled=True),
                tracking=TrackingConfig(enabled=True),
                state=StateConfig(enabled=False),
            )

    def test_no_state_flag_also_disables_activity(self) -> None:
        """Otherwise the recogniser runs on nothing and reports only idle."""
        from vantage.cli import _flag_overrides, build_parser

        overrides = _flag_overrides(build_parser().parse_args(["run", "--track", "--no-state"]))
        assert "state.enabled=false" in overrides
        assert "activity.enabled=false" in overrides

    def test_no_activity_flag_leaves_state_alone(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        overrides = _flag_overrides(
            build_parser().parse_args(["run", "--track", "--no-activity"])
        )
        assert "activity.enabled=false" in overrides
        assert "state.enabled=false" not in overrides

    def test_walking_threshold_may_not_exceed_the_state_threshold(self) -> None:
        """Otherwise an entity is reported as moving and idle in the same frame.

        Found on a real clip: a person crossing at 0.175 h/s cleared the state
        machine's 0.15 but not the recogniser's 0.20, so the two stages
        disagreed about the same entity and the honest-looking "idle" was wrong.
        """
        from vantage.config.schema import (
            ActivityConfig,
            DetectionConfig,
            StateConfig,
            TrackingConfig,
            VantageConfig,
        )

        with pytest.raises(ConfigError, match=r"moving and idle in the same frame"):
            VantageConfig(
                detection=DetectionConfig(enabled=True),
                tracking=TrackingConfig(enabled=True),
                state=StateConfig(moving_above=0.15),
                activity=ActivityConfig(walking_speed=0.30),
            )

    def test_defaults_agree_across_the_two_stages(self) -> None:
        from vantage.config.schema import ActivityConfig, StateConfig

        assert ActivityConfig().walking_speed == StateConfig().moving_above
