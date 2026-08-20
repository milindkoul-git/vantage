"""The event engine: rules, debouncing, and the reduction that defines the phase.

The single property everything else serves: a condition true for many
consecutive frames produces **one** event. The scenario tests at the bottom
check that end to end by replaying the Phase 5 ground truth - which is the point
of having built it.
"""

from __future__ import annotations

import json

import pytest

from vantage.activity.contracts import (
    Activity,
    ActivityObservation,
    ActivityResult,
    EntityActivity,
)
from vantage.core.errors import ConfigError
from vantage.events.contracts import Event, EventResult, Severity
from vantage.events.engine import EventEngine
from vantage.events.rules import DEFAULT_RULES, RuleSpec, SceneContext, evaluate
from vantage.perception.contracts import BoundingBox
from vantage.spatial.contracts import (
    EntitySpatial,
    Relation,
    RelationObservation,
    SpatialResult,
    ZoneEvent,
    ZoneOccupancy,
)
from vantage.tracking.contracts import Track, TrackingResult, TrackState

DT = 1.0 / 30.0


def make_activity(
    activity: Activity,
    *,
    entity_id: str = "person_1",
    track_id: int = 1,
    confidence: float = 0.9,
    duration_s: float = 2.0,
    label: str = "person",
    index: int = 0,
) -> ActivityResult:
    return ActivityResult(
        entities=(
            EntityActivity(
                track_id=track_id,
                entity_id=entity_id,
                label=label,
                observations=(
                    ActivityObservation(activity, confidence, duration_s, "because"),
                ),
            ),
        ),
        source_id="test",
        frame_index=index,
        capture_wall=index * DT,
        elapsed_s=DT,
    )


def make_track(track_id: int = 1, label: str = "person") -> Track:
    return Track(
        track_id=track_id,
        entity_id=f"{label}_{track_id}",
        box=BoundingBox(0.0, 0.0, 50.0, 150.0),
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


def make_tracking(*tracks: Track, index: int = 0) -> TrackingResult:
    return TrackingResult(
        tracks=tracks or (make_track(),),
        source_id="test",
        frame_index=index,
        capture_wall=index * DT,
        frame_size=(640, 480),
        elapsed_s=DT,
    )


def make_spatial(
    *,
    zones: tuple[ZoneOccupancy, ...] = (),
    relations: tuple[RelationObservation, ...] = (),
    entity_id: str = "person_1",
    track_id: int = 1,
    label: str = "person",
    index: int = 0,
) -> SpatialResult:
    return SpatialResult(
        entities=(
            EntitySpatial(
                track_id=track_id,
                entity_id=entity_id,
                label=label,
                zones=zones,
                ground_point=(25.0, 150.0),
            ),
        ),
        relations=relations,
        source_id="test",
        frame_index=index,
        capture_wall=index * DT,
        elapsed_s=DT,
    )


class TestContracts:
    def test_an_event_needs_a_summary(self) -> None:
        with pytest.raises(ValueError, match="summary"):
            Event("rule", Severity.INFO, "  ", "person_1", 1, 0, 0.0, 0.0)

    def test_severity_ranks(self) -> None:
        assert Severity.ALERT.rank > Severity.NOTICE.rank > Severity.INFO.rank

    def test_the_cooldown_key_includes_the_entity(self) -> None:
        """Two people falling at once must be two events, not one."""
        first = Event("fall", Severity.ALERT, "a fell", "person_1", 1, 0, 0.0, 0.0)
        second = Event("fall", Severity.ALERT, "b fell", "person_2", 2, 0, 0.0, 0.0)
        assert first.key != second.key

    def test_record_is_serialisable_and_leaves_the_identity_seam(self) -> None:
        event = Event("fall", Severity.ALERT, "person_1 fell", "person_1", 1, 5, 1.0, 1.0)
        record = event.to_record("camera_01")
        json.dumps(record)
        assert record["identity"] is None
        assert record["severity"] == "alert"

    def test_result_reports_the_highest_severity(self) -> None:
        result = EventResult(
            events=(
                Event("a", Severity.INFO, "x", None, None, 0, 0.0, 0.0),
                Event("b", Severity.ALERT, "y", None, None, 0, 0.0, 0.0),
            ),
            source_id="t",
            frame_index=0,
            capture_wall=0.0,
        )
        assert result.highest is Severity.ALERT


class TestRuleValidation:
    def test_unknown_rule_type_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="unknown event rule type"):
            RuleSpec(type="teleportation")

    def test_a_typo_in_an_activity_is_caught_at_load(self) -> None:
        """Otherwise it is a rule that can never fire, and silence looks like calm."""
        with pytest.raises(ConfigError, match="unknown activity"):
            RuleSpec(type="activity", activity="flyng")

    def test_a_typo_in_a_relation_is_caught_at_load(self) -> None:
        with pytest.raises(ConfigError, match="unknown relation"):
            RuleSpec(type="relation", relation="besides")

    def test_negative_cooldown_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="cooldown"):
            RuleSpec(type="zone_entry", cooldown_s=-1.0)

    def test_label_is_derived_when_unnamed(self) -> None:
        assert RuleSpec(type="activity", activity="falling").label == "activity:falling"

    def test_an_explicit_name_wins(self) -> None:
        assert RuleSpec(type="activity", activity="falling", name="fall").label == "fall"


class TestActivityRule:
    def context(self, activity: ActivityResult | None, spatial=None) -> SceneContext:
        return SceneContext(
            tracking=make_tracking(),
            state=None,
            activity=activity,
            spatial=spatial,
            elapsed_s=1.0,
            frame_index=1,
            capture_wall=1.0,
            source_id="test",
        )

    def test_fires_on_the_named_activity(self) -> None:
        spec = RuleSpec(type="activity", activity="falling", severity=Severity.ALERT)
        events = evaluate(spec, self.context(make_activity(Activity.FALLING)))
        assert len(events) == 1
        assert events[0].severity is Severity.ALERT
        assert "falling" in events[0].summary

    def test_ignores_other_activities(self) -> None:
        spec = RuleSpec(type="activity", activity="falling")
        assert evaluate(spec, self.context(make_activity(Activity.WALKING))) == []

    def test_confidence_floor_is_applied(self) -> None:
        spec = RuleSpec(type="activity", activity="falling", min_confidence=0.8)
        weak = make_activity(Activity.FALLING, confidence=0.3)
        assert evaluate(spec, self.context(weak)) == []

    def test_duration_floor_is_applied(self) -> None:
        spec = RuleSpec(type="activity", activity="running", min_seconds=5.0)
        brief = make_activity(Activity.RUNNING, duration_s=1.0)
        assert evaluate(spec, self.context(brief)) == []

    def test_label_filter_is_applied(self) -> None:
        spec = RuleSpec(type="activity", activity="running", labels=("person",))
        vehicle = make_activity(Activity.RUNNING, label="car", entity_id="car_2")
        assert evaluate(spec, self.context(vehicle)) == []

    def test_evidence_carries_the_measurement(self) -> None:
        """Every event can be argued with."""
        spec = RuleSpec(type="activity", activity="falling")
        event = evaluate(spec, self.context(make_activity(Activity.FALLING)))[0]
        assert event.evidence["activity"] == "falling"
        assert event.evidence["why"] == "because"

    def test_no_activity_stage_means_no_events(self) -> None:
        """A rule must degrade rather than fail when a stage is not running."""
        spec = RuleSpec(type="activity", activity="falling")
        assert evaluate(spec, self.context(None)) == []

    def test_zone_filter_needs_the_entity_in_that_zone(self) -> None:
        spec = RuleSpec(type="activity", activity="loitering", zones=("till",))
        elsewhere = make_spatial(zones=(ZoneOccupancy("lobby", "area", 5.0),))
        assert evaluate(spec, self.context(make_activity(Activity.LOITERING), elsewhere)) == []

        inside = make_spatial(zones=(ZoneOccupancy("till", "area", 5.0),))
        events = evaluate(spec, self.context(make_activity(Activity.LOITERING), inside))
        assert len(events) == 1 and events[0].zone == "till"


class TestZoneRules:
    def context(self, spatial) -> SceneContext:
        return SceneContext(
            tracking=make_tracking(),
            state=None,
            activity=None,
            spatial=spatial,
            elapsed_s=1.0,
            frame_index=1,
            capture_wall=1.0,
            source_id="test",
        )

    def test_entry_fires_on_entry_only(self) -> None:
        entered = make_spatial(
            zones=(ZoneOccupancy("door", "entrance", 0.0, ZoneEvent.ENTERED),)
        )
        exited = make_spatial(zones=(ZoneOccupancy("door", "entrance", 3.0, ZoneEvent.EXITED),))
        spec = RuleSpec(type="zone_entry")
        assert len(evaluate(spec, self.context(entered))) == 1
        assert evaluate(spec, self.context(exited)) == []

    def test_exit_fires_on_exit_only(self) -> None:
        exited = make_spatial(zones=(ZoneOccupancy("door", "entrance", 3.0, ZoneEvent.EXITED),))
        assert len(evaluate(RuleSpec(type="zone_exit"), self.context(exited))) == 1

    def test_zone_filter_is_applied(self) -> None:
        entered = make_spatial(
            zones=(ZoneOccupancy("door", "entrance", 0.0, ZoneEvent.ENTERED),)
        )
        spec = RuleSpec(type="zone_entry", zones=("vault",))
        assert evaluate(spec, self.context(entered)) == []

    def test_dwell_fires_past_the_threshold(self) -> None:
        spec = RuleSpec(type="zone_dwell", min_seconds=30.0)
        brief = make_spatial(zones=(ZoneOccupancy("till", "area", 5.0),))
        assert evaluate(spec, self.context(brief)) == []
        long = make_spatial(zones=(ZoneOccupancy("till", "area", 45.0),))
        assert len(evaluate(spec, self.context(long))) == 1

    def test_dwell_ignores_a_zone_already_left(self) -> None:
        """A place the entity is no longer in cannot be dwelt in."""
        spec = RuleSpec(type="zone_dwell", min_seconds=1.0)
        left = make_spatial(zones=(ZoneOccupancy("till", "area", 45.0, ZoneEvent.EXITED),))
        assert evaluate(spec, self.context(left)) == []

    def test_occupancy_is_scene_level(self) -> None:
        """Attributing a crowd to one occupant would make the cooldown depend on
        who happened to be listed first."""
        spatial = SpatialResult(
            entities=tuple(
                EntitySpatial(i, f"person_{i}", "person", (ZoneOccupancy("hall", "area", 1.0),))
                for i in range(4)
            ),
            relations=(),
            source_id="test",
            frame_index=1,
            capture_wall=1.0,
        )
        events = evaluate(RuleSpec(type="zone_occupancy", min_count=3), self.context(spatial))
        assert len(events) == 1
        assert events[0].entity_id is None
        assert events[0].evidence["count"] == 4

    def test_occupancy_below_the_threshold_is_silent(self) -> None:
        spatial = make_spatial(zones=(ZoneOccupancy("hall", "area", 1.0),))
        assert (
            evaluate(RuleSpec(type="zone_occupancy", min_count=3), self.context(spatial)) == []
        )


class TestRelationRule:
    def test_fires_on_the_named_relation(self) -> None:
        relation = RelationObservation(
            Relation.INTERACTING, "person_1", "laptop_2", 1, 2, 0.4, 0.85, 2.0, "reached"
        )
        context = SceneContext(
            tracking=make_tracking(),
            state=None,
            activity=None,
            spatial=make_spatial(relations=(relation,)),
            elapsed_s=1.0,
            frame_index=1,
            capture_wall=1.0,
            source_id="test",
        )
        events = evaluate(RuleSpec(type="relation", relation="interacting_with"), context)
        assert len(events) == 1
        assert events[0].related_id == "laptop_2"

    def test_confidence_floor_separates_the_evidence_tiers(self) -> None:
        """Phase 6 reports proximity-only interaction at 0.4 and a confirmed
        reach at 0.85; a rule can demand the stronger one."""
        weak = RelationObservation(
            Relation.INTERACTING, "person_1", "laptop_2", 1, 2, 0.5, 0.4, 2.0, "proximity"
        )
        context = SceneContext(
            tracking=make_tracking(),
            state=None,
            activity=None,
            spatial=make_spatial(relations=(weak,)),
            elapsed_s=1.0,
            frame_index=1,
            capture_wall=1.0,
            source_id="test",
        )
        spec = RuleSpec(type="relation", relation="interacting_with", min_confidence=0.8)
        assert evaluate(spec, context) == []


class TestDebouncing:
    """The property the whole phase exists for."""

    def fire(self, engine: EventEngine, frames: int, activity: Activity) -> int:
        raised = 0
        for index in range(frames):
            raised += len(
                engine.update(
                    make_tracking(index=index),
                    None,
                    make_activity(activity, index=index),
                    None,
                )
            )
        return raised

    def test_a_condition_true_for_many_frames_fires_once(self) -> None:
        engine = EventEngine((RuleSpec(type="activity", activity="falling", cooldown_s=10.0),))
        assert self.fire(engine, 45, Activity.FALLING) == 1
        assert engine.suppressed == 44

    def test_the_cooldown_expires(self) -> None:
        engine = EventEngine((RuleSpec(type="activity", activity="falling", cooldown_s=0.5),))
        raised = self.fire(engine, 60, Activity.FALLING)
        # 60 frames is 2 seconds of footage; a 0.5s cooldown allows about four.
        assert 3 <= raised <= 5

    def test_zero_cooldown_fires_every_frame(self) -> None:
        """The escape hatch, and a check that suppression is the cooldown's doing."""
        engine = EventEngine((RuleSpec(type="activity", activity="falling", cooldown_s=0.0),))
        assert self.fire(engine, 10, Activity.FALLING) == 10

    def test_two_entities_are_two_events(self) -> None:
        """The one case where a missed alert matters most."""
        engine = EventEngine((RuleSpec(type="activity", activity="falling", cooldown_s=10.0),))
        both = ActivityResult(
            entities=tuple(
                EntityActivity(
                    i,
                    f"person_{i}",
                    "person",
                    (ActivityObservation(Activity.FALLING, 0.9, 1.0, "x"),),
                )
                for i in (1, 2)
            ),
            source_id="test",
            frame_index=0,
            capture_wall=0.0,
            elapsed_s=DT,
        )
        assert len(engine.update(make_tracking(), None, both, None)) == 2

    def test_one_rule_does_not_silence_another(self) -> None:
        engine = EventEngine(
            (
                RuleSpec(type="activity", activity="falling", name="fall", cooldown_s=10.0),
                RuleSpec(type="activity", activity="falling", name="fall2", cooldown_s=10.0),
            )
        )
        assert (
            len(engine.update(make_tracking(), None, make_activity(Activity.FALLING), None))
            == 2
        )

    def test_suppressions_are_counted_not_hidden(self) -> None:
        engine = EventEngine((RuleSpec(type="activity", activity="falling", cooldown_s=10.0),))
        self.fire(engine, 30, Activity.FALLING)
        assert engine.stats()["suppressed"] == 29


class TestLifecycle:
    def test_cooldowns_for_retired_entities_are_pruned(self) -> None:
        """Keyed by entity id, so on a long run this leaks unless pruned."""
        engine = EventEngine((RuleSpec(type="activity", activity="falling", cooldown_s=99.0),))
        many = ActivityResult(
            entities=tuple(
                EntityActivity(
                    i,
                    f"person_{i}",
                    "person",
                    (ActivityObservation(Activity.FALLING, 0.9, 1.0, "x"),),
                )
                for i in range(5)
            ),
            source_id="test",
            frame_index=0,
            capture_wall=0.0,
            elapsed_s=DT,
        )
        engine.update(make_tracking(*[make_track(i) for i in range(5)]), None, many, None)
        assert engine.tracked_keys == 5
        engine.update(make_tracking(make_track(0), index=1), None, None, None)
        assert engine.tracked_keys == 1

    def test_scene_level_cooldowns_are_kept(self) -> None:
        """A place does not go away when its occupants do."""
        engine = EventEngine((RuleSpec(type="zone_occupancy", min_count=1, cooldown_s=99.0),))
        spatial = make_spatial(zones=(ZoneOccupancy("hall", "area", 1.0),))
        engine.update(make_tracking(), None, None, spatial)
        assert engine.tracked_keys == 1
        engine.update(make_tracking(index=1), None, None, None)
        assert engine.tracked_keys == 1

    def test_reset_clears_everything(self) -> None:
        engine = EventEngine()
        engine.update(make_tracking(), None, make_activity(Activity.FALLING), None)
        engine.reset()
        assert engine.raised == 0 and engine.tracked_keys == 0

    def test_time_is_footage_time(self) -> None:
        engine = EventEngine()
        for index in range(10):
            engine.update(
                TrackingResult(
                    tracks=(make_track(),),
                    source_id="t",
                    frame_index=index,
                    capture_wall=0.0,
                    frame_size=(640, 480),
                    elapsed_s=0.5,
                )
            )
        assert engine.elapsed_s == pytest.approx(5.0)


class TestDefaults:
    def test_the_default_rules_are_conservative(self) -> None:
        """Nothing fires on a quiet scene, and the only ALERT is a fall."""
        alerts = [r for r in DEFAULT_RULES if r.severity is Severity.ALERT]
        assert [r.activity for r in alerts] == ["falling"]

    def test_a_quiet_scene_raises_nothing(self) -> None:
        engine = EventEngine()
        assert len(engine.update(make_tracking(), None, None, None)) == 0

    def test_an_empty_rule_list_means_the_defaults(self) -> None:
        """Not "no rules": silencing every alert must take `enabled: false`."""
        from vantage.config.schema import EventsConfig
        from vantage.events.engine import build_event_engine

        assert build_event_engine(EventsConfig(rules=[])).rules == DEFAULT_RULES

    def test_config_rules_replace_the_defaults(self) -> None:
        from vantage.config.schema import EventRuleConfig, EventsConfig
        from vantage.events.engine import build_event_engine

        engine = build_event_engine(
            EventsConfig(rules=[EventRuleConfig(type="zone_entry", name="door")])
        )
        assert [r.label for r in engine.rules] == ["door"]


class TestScenarioReduction:
    """End to end over the Phase 5 ground truth - the point of having built it."""

    def replay(self, name: str) -> tuple[list[Event], EventEngine]:
        from vantage.activity.engine import ActivityEngine
        from vantage.activity.scenarios import SCENARIOS, generate
        from vantage.pose.contracts import PoseResult
        from vantage.state.estimator import StateEstimator

        scenario = SCENARIOS[name]
        state_estimator, activity, events = StateEstimator(), ActivityEngine(), EventEngine()
        raised: list[Event] = []
        for frame in generate(scenario):
            tracking = TrackingResult(
                tracks=(frame.track,),
                source_id="s",
                frame_index=frame.index,
                capture_wall=frame.time_s,
                frame_size=(640, 480),
                elapsed_s=frame.elapsed_s,
            )
            state = state_estimator.update(tracking)
            pose = (
                PoseResult(
                    poses=(frame.pose,),
                    source_id="s",
                    frame_index=frame.index,
                    capture_wall=frame.time_s,
                    frame_size=(640, 480),
                    people_seen=1,
                )
                if frame.pose is not None
                else None
            )
            raised.extend(events.update(tracking, state, activity.update(state, pose), None))
        return raised, events

    def test_one_fall_produces_exactly_one_alert(self) -> None:
        raised, engine = self.replay("fall")
        alerts = [e for e in raised if e.severity is Severity.ALERT]
        assert len(alerts) == 1
        # The 45 frames the activity engine holds the transient for.
        assert engine.suppressed >= 40

    def test_a_deliberate_lie_down_produces_no_alert(self) -> None:
        """The negative carried all the way through from Phase 5."""
        raised, _ = self.replay("lie_down_slowly")
        assert [e for e in raised if e.severity is Severity.ALERT] == []

    def test_running_produces_one_notice(self) -> None:
        raised, _ = self.replay("run")
        assert len([e for e in raised if e.rule == "running"]) == 1

    def test_loitering_produces_one_notice(self) -> None:
        raised, _ = self.replay("loiter")
        assert len([e for e in raised if e.rule == "loitering"]) == 1

    def test_an_ordinary_walk_produces_nothing(self) -> None:
        raised, _ = self.replay("walk")
        assert raised == []


class TestPipelineIntegration:
    def test_events_appear_in_the_run_summary(self) -> None:
        from tests.fakes import make_engine
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            AppConfig,
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=60"),
            ingest=IngestConfig(max_frames=40),
            app=AppConfig(resource_interval_s=0),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config, engine=engine)
        assert "events:" in result.summary()
        assert result.events_summary["rules"] == len(DEFAULT_RULES)
