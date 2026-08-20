"""Spatial: zones, proximity, approach and interaction.

No model, no weights, no runtime - this phase is geometry over the tracks and
poses the earlier ones produce, so every property here is checkable against
hand-built input.

The scenario suite at the bottom is the gate. It drives the **real** state
estimator, because interaction depends on motion state and a harness that
stubbed it would score the wrong thing - which it did, until the scripted tracks
were given real velocities.
"""

from __future__ import annotations

import pytest

from vantage.core.errors import ConfigError
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import KEYPOINT_NAMES, LEFT_WRIST, Keypoint, Pose, Posture
from vantage.spatial.analyzer import SpatialAnalyzer, SpatialParams, ground_distance
from vantage.spatial.contracts import (
    Relation,
    RelationObservation,
    Zone,
    ZoneEvent,
    to_scene_record,
)
from vantage.spatial.engine import SpatialEngine
from vantage.state.contracts import EntityState, MotionState, StateResult
from vantage.tracking.contracts import Track, TrackingResult, TrackState

FRAME = (640, 480)
DT = 1.0 / 30.0


def make_track(
    track_id: int = 1,
    *,
    x: float = 100.0,
    y: float = 400.0,
    label: str = "person",
    height: float = 160.0,
    width: float = 60.0,
) -> Track:
    """A track whose ground point is ``(x, y)``."""
    return Track(
        track_id=track_id,
        entity_id=f"{label}_{track_id}",
        box=BoundingBox(x - width / 2, y - height, x + width / 2, y),
        label=label,
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=10,
        hits=10,
        time_since_update=0,
        start_frame=0,
        last_frame=10,
    )


def make_state(track: Track, motion: MotionState) -> EntityState:
    return EntityState(
        track_id=track.track_id,
        entity_id=track.entity_id,
        label=track.label,
        motion=motion,
        speed=0.0 if motion is MotionState.STATIONARY else 0.8,
        dwell_s=5.0,
        bearing_deg=None,
        distance=0.0,
        age_s=10.0,
        observed=True,
    )


def make_pose(track: Track, wrist: tuple[float, float] | None) -> Pose:
    keypoints = [Keypoint(0.0, 0.0, 0.0) for _ in KEYPOINT_NAMES]
    if wrist is not None:
        keypoints[LEFT_WRIST] = Keypoint(wrist[0], wrist[1], 0.9)
    return Pose(
        keypoints=tuple(keypoints),
        track_id=track.track_id,
        entity_id=track.entity_id,
        box=track.box,
        posture=Posture.STANDING,
        posture_confidence=0.8,
    )


def tracking_of(*tracks, index: int = 0, elapsed: float = DT) -> TrackingResult:
    return TrackingResult(
        tracks=tuple(tracks),
        source_id="test",
        frame_index=index,
        capture_wall=index * elapsed,
        frame_size=FRAME,
        elapsed_s=elapsed,
    )


def state_of(*states, index: int = 0) -> StateResult:
    return StateResult(
        states=tuple(states),
        source_id="test",
        frame_index=index,
        capture_wall=0.0,
        elapsed_s=DT,
    )


def drive(engine: SpatialEngine, tracks, *, seconds: float, motion=None, poses=None):
    """Hold a configuration for a while and return the last result."""
    from vantage.pose.contracts import PoseResult

    result = None
    for index in range(max(1, int(seconds / DT))):
        state = (
            state_of(
                *[make_state(t, motion[t.track_id]) for t in tracks if t.track_id in motion]
            )
            if motion
            else None
        )
        pose_result = (
            PoseResult(
                poses=tuple(poses),
                source_id="test",
                frame_index=index,
                capture_wall=0.0,
                frame_size=FRAME,
                people_seen=len(poses),
            )
            if poses
            else None
        )
        result = engine.update(tracking_of(*tracks, index=index), pose_result, state)
    return result


class TestZoneGeometry:
    def test_point_inside_and_outside(self) -> None:
        zone = Zone(name="left", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
        assert zone.contains((100.0, 200.0), FRAME)
        assert not zone.contains((500.0, 200.0), FRAME)

    def test_concave_polygon(self) -> None:
        """Ray casting handles a notch; a bounding-box test would not."""
        zone = Zone(
            name="u",
            points=(
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.7, 1.0),
                (0.7, 0.3),
                (0.3, 0.3),
                (0.3, 1.0),
                (0.0, 1.0),
            ),
        )
        assert zone.contains((320.0, 48.0), FRAME)  # in the solid top bar
        assert not zone.contains((320.0, 400.0), FRAME)  # inside the notch

    def test_coordinates_must_be_normalised(self) -> None:
        """Pixel coordinates would silently point elsewhere at another resolution."""
        with pytest.raises(ValueError, match=r"outside"):
            Zone(name="bad", points=((0.0, 0.0), (640.0, 0.0), (640.0, 480.0)))

    def test_a_polygon_needs_three_points(self) -> None:
        with pytest.raises(ValueError, match=r"at least 3"):
            Zone(name="line", points=((0.0, 0.0), (1.0, 1.0)))

    def test_zero_sized_frame_is_not_a_crash(self) -> None:
        zone = Zone(name="z", points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
        assert not zone.contains((1.0, 1.0), (0, 0))


class TestZoneMembership:
    def zone_engine(self) -> SpatialEngine:
        return SpatialEngine(
            SpatialAnalyzer(
                (Zone(name="left", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))),)
            )
        )

    def test_entity_is_placed_by_its_feet(self) -> None:
        """A box centre drifts with posture; the ground point does not."""
        engine = self.zone_engine()
        result = engine.update(tracking_of(make_track(x=100.0)))
        assert result.entities[0].in_zone("left")

    def test_entering_raises_an_event(self) -> None:
        engine = self.zone_engine()
        result = engine.update(tracking_of(make_track(x=100.0)))
        assert result.crossings()
        assert result.crossings()[0][1].event is ZoneEvent.ENTERED

    def test_leaving_raises_an_exit_and_stops_reporting_presence(self) -> None:
        engine = self.zone_engine()
        engine.update(tracking_of(make_track(x=100.0)))
        result = engine.update(tracking_of(make_track(x=500.0), index=1))

        entity = result.entities[0]
        assert not entity.in_zone("left")
        assert "left" not in entity.zone_names
        assert any(z.event is ZoneEvent.EXITED for z in entity.zones)

    def test_exit_is_eventually_forgotten(self) -> None:
        engine = SpatialEngine(
            SpatialAnalyzer(
                (Zone(name="left", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))),),
                SpatialParams(zone_event_hold_s=0.2),
            )
        )
        engine.update(tracking_of(make_track(x=100.0)))
        result = drive(engine, [make_track(x=500.0)], seconds=2.0)
        assert result.entities[0].zones == ()

    def test_dwell_accumulates_in_footage_time(self) -> None:
        engine = self.zone_engine()
        result = drive(engine, [make_track(x=100.0)], seconds=3.0)
        assert result.entities[0].zones[0].dwell_s == pytest.approx(3.0, abs=0.1)

    def test_overlapping_zones_both_report(self) -> None:
        """A till inside a shop floor is in both, not in whichever is checked first."""
        engine = SpatialEngine(
            SpatialAnalyzer(
                (
                    Zone(name="floor", points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
                    Zone(name="till", points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))),
                )
            )
        )
        result = engine.update(tracking_of(make_track(x=100.0)))
        assert set(result.entities[0].zone_names) == {"floor", "till"}

    def test_retired_entities_release_their_zone_state(self) -> None:
        engine = self.zone_engine()
        engine.update(tracking_of(*[make_track(i, x=100.0) for i in range(4)]))
        engine.update(tracking_of(make_track(0, x=100.0), index=1))
        assert len(engine.analyzer._zone_state) == 1


class TestDistance:
    def test_is_symmetric(self) -> None:
        """A person is as far from a mug as the mug is from them."""
        a, b = make_track(1, x=100.0), make_track(2, x=200.0, height=40.0)
        assert ground_distance(a, b) == pytest.approx(ground_distance(b, a))

    def test_is_scale_free(self) -> None:
        """Two people a pace apart read the same near and far from the lens."""
        near = ground_distance(
            make_track(1, x=100.0, height=200.0), make_track(2, x=300.0, height=200.0)
        )
        far = ground_distance(
            make_track(1, x=100.0, height=100.0), make_track(2, x=200.0, height=100.0)
        )
        assert near == pytest.approx(far, abs=0.01)


class TestProximity:
    def test_near_entities_are_reported(self) -> None:
        engine = SpatialEngine()
        result = engine.update(tracking_of(make_track(1, x=300.0), make_track(2, x=340.0)))
        assert result.of(Relation.NEAR)

    def test_distant_entities_are_not(self) -> None:
        engine = SpatialEngine()
        result = engine.update(tracking_of(make_track(1, x=60.0), make_track(2, x=600.0)))
        assert not result.of(Relation.NEAR)

    def test_symmetric_relations_are_reported_once(self) -> None:
        """near(a,b) and near(b,a) are one fact; two edges would double every count."""
        engine = SpatialEngine()
        result = engine.update(tracking_of(make_track(1, x=300.0), make_track(2, x=340.0)))
        assert len(result.of(Relation.NEAR)) == 1

    def test_hysteresis_stops_boundary_flicker(self) -> None:
        params = SpatialParams(near_distance=1.0, near_hysteresis=0.5)
        engine = SpatialEngine(SpatialAnalyzer((), params))
        # Establish "near" well inside the threshold.
        engine.update(tracking_of(make_track(1, x=300.0), make_track(2, x=320.0)))
        # Now just outside the base threshold but inside the dead band.
        result = engine.update(
            tracking_of(make_track(1, x=300.0), make_track(2, x=480.0), index=1)
        )
        assert result.of(Relation.NEAR)

    def test_approach_and_recede(self) -> None:
        engine = SpatialEngine()
        for index in range(40):
            result = engine.update(
                tracking_of(
                    make_track(1, x=100.0), make_track(2, x=600.0 - index * 8.0), index=index
                )
            )
        assert result.of(Relation.APPROACHING)

        engine.reset()
        for index in range(40):
            result = engine.update(
                tracking_of(
                    make_track(1, x=100.0), make_track(2, x=200.0 + index * 8.0), index=index
                )
            )
        assert result.of(Relation.RECEDING)

    def test_static_pair_is_neither(self) -> None:
        engine = SpatialEngine()
        result = drive(engine, [make_track(1, x=100.0), make_track(2, x=200.0)], seconds=2.0)
        assert not result.of(Relation.APPROACHING)
        assert not result.of(Relation.RECEDING)

    def test_pairs_are_pruned_when_an_entity_goes(self) -> None:
        engine = SpatialEngine()
        engine.update(tracking_of(*[make_track(i, x=100.0 + i * 30) for i in range(4)]))
        assert engine.analyzer.tracked_pairs == 6
        engine.update(tracking_of(make_track(0, x=100.0), index=1))
        assert engine.analyzer.tracked_pairs == 0

    def test_pairing_is_capped(self) -> None:
        """Relations are quadratic, so the budget is explicit like pose's."""
        engine = SpatialEngine(SpatialAnalyzer((), SpatialParams(max_entities=3)))
        result = engine.update(
            tracking_of(*[make_track(i, x=100.0 + i * 20) for i in range(8)])
        )
        assert result.metadata["entities_paired"] == 3
        assert result.metadata["entities_total"] == 8


class TestInteraction:
    def person_and_object(self):
        return make_track(1, x=300.0), make_track(2, x=320.0, label="laptop", height=40.0)

    def test_stationary_person_beside_an_object_interacts_weakly(self) -> None:
        person, thing = self.person_and_object()
        engine = SpatialEngine()
        result = drive(
            engine,
            [person, thing],
            seconds=3.0,
            motion={1: MotionState.STATIONARY, 2: MotionState.STATIONARY},
        )
        interactions = result.of(Relation.INTERACTING)
        assert interactions
        assert interactions[0].confidence == pytest.approx(0.4)
        assert "no reach observed" in interactions[0].evidence

    def test_a_confirmed_reach_is_much_stronger(self) -> None:
        person, thing = self.person_and_object()
        engine = SpatialEngine()
        result = drive(
            engine,
            [person, thing],
            seconds=3.0,
            motion={1: MotionState.STATIONARY, 2: MotionState.STATIONARY},
            poses=[make_pose(person, wrist=(320.0, 380.0))],
        )
        assert result.of(Relation.INTERACTING)[0].confidence == pytest.approx(0.85)

    def test_a_moving_person_needs_a_reach(self) -> None:
        """Regression: duration alone cannot separate lingering from passing.

        Measured before this gate existed - a brisk walk-past produced nothing,
        but the same path at an amble produced 49 frames of false interaction,
        because a slow enough pass satisfies any sustain threshold.
        """
        person, thing = self.person_and_object()
        engine = SpatialEngine()
        result = drive(
            engine,
            [person, thing],
            seconds=3.0,
            motion={1: MotionState.MOVING, 2: MotionState.STATIONARY},
        )
        assert not result.of(Relation.INTERACTING)

    def test_a_moving_person_reaching_does_interact(self) -> None:
        """Taking something in passing is real, and a landmark is direct evidence."""
        person, thing = self.person_and_object()
        engine = SpatialEngine()
        result = drive(
            engine,
            [person, thing],
            seconds=3.0,
            motion={1: MotionState.MOVING, 2: MotionState.STATIONARY},
            poses=[make_pose(person, wrist=(320.0, 380.0))],
        )
        assert result.of(Relation.INTERACTING)

    def test_without_motion_state_only_a_reach_counts(self) -> None:
        person, thing = self.person_and_object()
        engine = SpatialEngine()
        assert not drive(engine, [person, thing], seconds=3.0).of(Relation.INTERACTING)

        engine.reset()
        result = drive(
            engine,
            [person, thing],
            seconds=3.0,
            poses=[make_pose(person, wrist=(320.0, 380.0))],
        )
        assert result.of(Relation.INTERACTING)

    def test_brief_proximity_is_not_interaction(self) -> None:
        person, thing = self.person_and_object()
        engine = SpatialEngine(SpatialAnalyzer((), SpatialParams(interact_s=2.0)))
        result = drive(
            engine,
            [person, thing],
            seconds=0.5,
            motion={1: MotionState.STATIONARY, 2: MotionState.STATIONARY},
        )
        assert not result.of(Relation.INTERACTING)

    def test_two_people_never_interact(self) -> None:
        """Geometry alone cannot support that claim between two people."""
        engine = SpatialEngine()
        result = drive(
            engine,
            [make_track(1, x=300.0), make_track(2, x=320.0)],
            seconds=3.0,
            motion={1: MotionState.STATIONARY, 2: MotionState.STATIONARY},
        )
        assert not result.of(Relation.INTERACTING)

    def test_a_low_confidence_wrist_is_not_a_reach(self) -> None:
        person, thing = self.person_and_object()
        pose = Pose(
            keypoints=tuple(
                Keypoint(320.0, 380.0, 0.05) if i == LEFT_WRIST else Keypoint(0.0, 0.0, 0.0)
                for i in range(len(KEYPOINT_NAMES))
            ),
            track_id=1,
            entity_id="person_1",
            box=person.box,
        )
        engine = SpatialEngine()
        result = drive(
            engine,
            [person, thing],
            seconds=3.0,
            motion={1: MotionState.MOVING, 2: MotionState.STATIONARY},
            poses=[pose],
        )
        assert not result.of(Relation.INTERACTING)

    def test_interaction_is_directed_person_first(self) -> None:
        person, thing = self.person_and_object()
        engine = SpatialEngine()
        result = drive(
            engine,
            [thing, person],  # object listed first on purpose
            seconds=3.0,
            motion={1: MotionState.STATIONARY, 2: MotionState.STATIONARY},
        )
        relation = result.of(Relation.INTERACTING)[0]
        assert relation.subject_id == "person_1"
        assert relation.object_id == "laptop_2"


class TestEngine:
    def test_time_is_footage_time(self) -> None:
        engine = SpatialEngine()
        for index in range(10):
            engine.update(tracking_of(make_track(), index=index, elapsed=0.5))
        assert engine.elapsed_s == pytest.approx(5.0)

    def test_state_availability_is_recorded(self) -> None:
        """Without it, interaction means something different, so it is reported."""
        engine = SpatialEngine()
        assert not engine.update(tracking_of(make_track())).state_available
        assert engine.update(
            tracking_of(make_track(), index=1),
            None,
            state_of(make_state(make_track(), MotionState.STATIONARY)),
        ).state_available

    def test_reset_clears_everything(self) -> None:
        engine = SpatialEngine()
        engine.update(tracking_of(make_track(1), make_track(2, x=200.0)))
        engine.reset()
        assert engine.elapsed_s == 0.0
        assert engine.analyzer.tracked_pairs == 0


class TestRecords:
    def test_scene_record_is_json_serialisable(self) -> None:
        import json

        engine = SpatialEngine(
            SpatialAnalyzer(
                (
                    Zone(
                        name="left",
                        kind="entrance",
                        points=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)),
                    ),
                )
            )
        )
        result = engine.update(tracking_of(make_track(1, x=100.0), make_track(2, x=140.0)))
        record = to_scene_record(result, "camera_01", 1_700_000_000.0)

        json.dumps(record)
        assert record["camera_id"] == "camera_01"
        assert record["timestamp"].startswith("2023-")
        assert all(node["identity"] is None for node in record["nodes"])
        assert record["nodes"][0]["zones"][0]["zone"] == "left"
        assert record["edges"][0]["relation"] == "near"

    def test_relation_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError, match=r"confidence"):
            RelationObservation(Relation.NEAR, "a", "b", 1, 2, 0.5, 1.5, 0.0, "x")

    def test_symmetric_key_orders_the_pair(self) -> None:
        forward = RelationObservation(Relation.NEAR, "a", "b", 1, 2, 0.5, 0.9, 0.0, "x")
        reverse = RelationObservation(Relation.NEAR, "b", "a", 2, 1, 0.5, 0.9, 0.0, "x")
        assert forward.key == reverse.key

    def test_directed_key_does_not(self) -> None:
        forward = RelationObservation(Relation.INTERACTING, "a", "b", 1, 2, 0.5, 0.9, 0.0, "x")
        reverse = RelationObservation(Relation.INTERACTING, "b", "a", 2, 1, 0.5, 0.9, 0.0, "x")
        assert forward.key != reverse.key


class TestParams:
    def test_interaction_cannot_be_looser_than_proximity(self) -> None:
        with pytest.raises(ConfigError, match=r"interact_distance"):
            SpatialParams(near_distance=0.5, interact_distance=1.0)

    def test_negative_thresholds_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            SpatialParams(near_distance=-1.0)

    def test_a_pair_needs_two_entities(self) -> None:
        with pytest.raises(ConfigError, match=r"max_entities"):
            SpatialParams(max_entities=1)


class TestConfigWiring:
    def test_zones_load_from_config(self) -> None:
        from vantage.config.schema import SpatialConfig, ZoneConfig
        from vantage.spatial.engine import build_spatial_engine

        engine = build_spatial_engine(
            SpatialConfig(
                zones=[ZoneConfig(name="door", points=[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]])]
            )
        )
        assert [z.name for z in engine.zones] == ["door"]

    def test_duplicate_zone_names_are_refused(self) -> None:
        """A zone name identifies a place in every observation record."""
        from vantage.config.schema import SpatialConfig, ZoneConfig

        points = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]
        with pytest.raises(ConfigError, match=r"duplicate zone names"):
            SpatialConfig(
                zones=[ZoneConfig(name="a", points=points), ZoneConfig(name="a", points=points)]
            )

    def test_no_spatial_flag(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        overrides = _flag_overrides(
            build_parser().parse_args(["run", "--track", "--no-spatial"])
        )
        assert "spatial.enabled=false" in overrides


class TestScenarioSuite:
    """The gate: scripted scenes through the real state estimator."""

    def all_results(self):
        from vantage.spatial.evaluation import evaluate
        from vantage.spatial.scenarios import build_suite

        return [evaluate(scenario) for scenario in build_suite()]

    def test_every_scenario_passes(self) -> None:
        failures = [m.scenario for m in self.all_results() if not m.passed]
        assert not failures, f"scenarios failed: {failures}"

    def test_nothing_forbidden_ever_fires(self) -> None:
        for metrics in self.all_results():
            assert metrics.forbidden_firings == 0, f"{metrics.scenario}: {metrics.unexpected}"

    def test_the_two_confidence_tiers_are_distinct(self) -> None:
        """The tiers are a claim about evidence; equal numbers would be a lie."""
        from vantage.spatial.evaluation import evaluate
        from vantage.spatial.scenarios import SCENARIOS

        def interaction(metrics) -> float:
            # Only the interaction key. Taking the max over every relation
            # picks up `near`, which both scenarios also satisfy at 0.8, and
            # compares the wrong pair of numbers entirely.
            return max(
                value
                for key, value in metrics.peak_confidence.items()
                if key.startswith("interacting")
            )

        weak = interaction(evaluate(SCENARIOS["linger_by_object"]))
        strong = interaction(evaluate(SCENARIOS["reach_for_object"]))
        assert weak == pytest.approx(0.4)
        assert strong > weak + 0.3

    def test_the_suite_covers_both_outcomes(self) -> None:
        from vantage.spatial.scenarios import SCENARIOS

        assert sum(len(s.forbidden) for s in SCENARIOS.values()) >= 6
        assert sum(len(s.expect) for s in SCENARIOS.values()) >= 5

    def test_scripted_actors_actually_move(self) -> None:
        """Regression: zero-velocity tracks made the state estimator call every
        actor stationary, which silently disabled the interaction motion gate."""
        from vantage.spatial.scenarios import SCENARIOS, generate

        frames = generate(SCENARIOS["amble_past_object"])
        walker = [f.tracks[0] for f in frames]
        assert any(abs(t.velocity[0]) > 1.0 for t in walker)
