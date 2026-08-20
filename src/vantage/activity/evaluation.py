"""Scoring the recogniser against scripted ground truth.

Three numbers, because activities fail in three different ways
--------------------------------------------------------------
**Continuous accuracy** - for things like walking and loitering, the fraction of
scored frames on which the right activity was reported, and how often a wrong
one was.

**Event detection and latency** - for things like falling, whether it fired at
all, exactly once, and how long after the movement it took. A fall detected
eight seconds late is a different product from one detected in half a second,
and a single precision figure hides that completely.

**Forbidden firings** - how often something fired that must never fire. Weighted
hardest, because this is where a plausible-looking rule does real damage: a fall
alert that goes off whenever someone sits on the floor gets switched off by
whoever has to read it, taking the real alerts with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vantage.activity.contracts import Activity
from vantage.activity.engine import ActivityEngine
from vantage.activity.recognizer import ActivityParams, RuleRecognizer
from vantage.activity.scenarios import ActivityScenario, generate
from vantage.state.contracts import StateResult
from vantage.state.estimator import StateEstimator, StateParams
from vantage.tracking.contracts import TrackingResult


@dataclass(slots=True)
class ActivityMetrics:
    """How one scenario went."""

    scenario: str
    scored_frames: int = 0
    expected_hits: int = 0
    expected_total: int = 0
    forbidden_firings: int = 0
    events_expected: tuple[Activity, ...] = ()
    events_detected: dict[str, int] = field(default_factory=dict)
    event_latency_s: dict[str, float] = field(default_factory=dict)
    unexpected: dict[str, int] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        """Fraction of frames where an expected activity was actually reported."""
        return self.expected_hits / self.expected_total if self.expected_total else 1.0

    @property
    def events_found(self) -> int:
        return sum(1 for a in self.events_expected if self.events_detected.get(a.value))

    @property
    def event_duplicates(self) -> int:
        """Events that fired more than once. A fall reported twice is two alerts."""
        return sum(
            max(0, self.events_detected.get(a.value, 0) - 1) for a in self.events_expected
        )

    @property
    def passed(self) -> bool:
        return (
            self.recall >= 0.9
            and self.forbidden_firings == 0
            and self.events_found == len(self.events_expected)
            and self.event_duplicates == 0
        )

    def describe(self) -> str:
        parts = [f"recall {self.recall:6.1%}"]
        if self.events_expected:
            parts.append(f"events {self.events_found}/{len(self.events_expected)}")
            if self.event_latency_s:
                worst = max(self.event_latency_s.values())
                parts.append(f"latency {worst:.2f}s")
        parts.append(f"forbidden {self.forbidden_firings}")
        return f"{self.scenario:32s} " + "  ".join(parts)


def evaluate(
    scenario: ActivityScenario, params: ActivityParams | None = None
) -> ActivityMetrics:
    """Run one scenario end to end and score it.

    The synthetic tracks go through the **real** state estimator, so the
    hysteresis, dwell timing and minimum holds are all exercised rather than
    stubbed. Only pose is scripted.
    """
    metrics = ActivityMetrics(scenario=scenario.name, events_expected=scenario.events)
    state_estimator = StateEstimator(StateParams())
    engine = ActivityEngine(RuleRecognizer(params))

    # When each transient first fired, so latency is measured against the beat
    # boundary that caused it rather than against the start of the run.
    boundary = _event_boundary(scenario)
    seen_events: dict[str, float] = {}
    previously_active: set[str] = set()

    for frame in generate(scenario):
        tracking = TrackingResult(
            tracks=(frame.track,),
            source_id="scenario",
            frame_index=frame.index,
            capture_wall=frame.time_s,
            frame_size=(640, 480),
            elapsed_s=frame.elapsed_s,
        )
        state: StateResult = state_estimator.update(tracking)
        pose_result = _pose_result(frame, tracking)
        result = engine.update(state, pose_result)

        entity = result.entities[0]
        active = {o.activity.value for o in entity}

        for activity in scenario.events:
            if activity.value in active and activity.value not in previously_active:
                metrics.events_detected[activity.value] = (
                    metrics.events_detected.get(activity.value, 0) + 1
                )
                seen_events.setdefault(activity.value, frame.time_s)
        previously_active = active

        for forbidden in scenario.forbidden:
            if forbidden.value in active:
                metrics.forbidden_firings += 1
                metrics.unexpected[forbidden.value] = (
                    metrics.unexpected.get(forbidden.value, 0) + 1
                )

        if frame.scored:
            metrics.scored_frames += 1
            for expected in frame.expected:
                metrics.expected_total += 1
                if expected.value in active:
                    metrics.expected_hits += 1

    for name, when in seen_events.items():
        metrics.event_latency_s[name] = max(0.0, when - boundary)
    return metrics


def _pose_result(frame, tracking: TrackingResult):
    if frame.pose is None:
        return None
    from vantage.pose.contracts import Pose, PoseResult

    pose = Pose(
        keypoints=frame.pose.keypoints,
        track_id=frame.track.track_id,
        entity_id=frame.track.entity_id,
        box=frame.track.box,
        posture=frame.pose.posture,
        posture_confidence=frame.pose.posture_confidence,
        posture_reason=frame.pose.posture_reason,
        model="scenario",
    )
    return PoseResult(
        poses=(pose,),
        source_id="scenario",
        frame_index=frame.index,
        capture_wall=frame.time_s,
        frame_size=(640, 480),
        people_seen=1,
    )


def _event_boundary(scenario: ActivityScenario) -> float:
    """When the movement that should produce an event begins.

    The last beat boundary, which for every event scenario here is the moment
    the posture changes. Measuring latency from the start of the run instead
    would report how long the scenario's preamble was.
    """
    if len(scenario.beats) < 2:
        return 0.0
    return sum(beat.seconds for beat in scenario.beats[:-1])


def aggregate(results: list[ActivityMetrics]) -> ActivityMetrics:
    """Pool raw counts across scenarios.

    Counts rather than an average of percentages: a scenario with four scored
    frames must not weigh the same as one with nine hundred.
    """
    pooled = ActivityMetrics(scenario="POOLED")
    for metrics in results:
        pooled.scored_frames += metrics.scored_frames
        pooled.expected_hits += metrics.expected_hits
        pooled.expected_total += metrics.expected_total
        pooled.forbidden_firings += metrics.forbidden_firings
        pooled.events_expected = pooled.events_expected + metrics.events_expected
        for name, count in metrics.events_detected.items():
            pooled.events_detected[name] = pooled.events_detected.get(name, 0) + count
        for name, count in metrics.unexpected.items():
            pooled.unexpected[name] = pooled.unexpected.get(name, 0) + count
        for name, value in metrics.event_latency_s.items():
            pooled.event_latency_s[name] = max(pooled.event_latency_s.get(name, 0.0), value)
    return pooled


def format_table(results: list[ActivityMetrics]) -> str:
    """A readable report, with the pooled row last."""
    width = 78
    lines = [
        f"{'SCENARIO':32s} {'RECALL':>8s} {'EVENTS':>8s} {'LATENCY':>9s} {'FORBIDDEN':>10s}",
        "-" * width,
    ]
    for metrics in results:
        events = (
            f"{metrics.events_found}/{len(metrics.events_expected)}"
            if metrics.events_expected
            else "-"
        )
        latency = (
            f"{max(metrics.event_latency_s.values()):.2f}s" if metrics.event_latency_s else "-"
        )
        recall = f"{metrics.recall:.1%}" if metrics.expected_total else "-"
        flag = "" if metrics.passed else "  FAIL"
        lines.append(
            f"{metrics.scenario:32s} {recall:>8s} {events:>8s} {latency:>9s} "
            f"{metrics.forbidden_firings:>10d}{flag}"
        )

    pooled = aggregate(results)
    lines.append("-" * width)
    events = f"{pooled.events_found}/{len(pooled.events_expected)}"
    latency = f"{max(pooled.event_latency_s.values()):.2f}s" if pooled.event_latency_s else "-"
    lines.append(
        f"{'POOLED':32s} {pooled.recall:>8.1%} {events:>8s} {latency:>9s} "
        f"{pooled.forbidden_firings:>10d}"
    )
    if pooled.unexpected:
        lines.append("")
        lines.append("Forbidden activities that fired:")
        for name, count in sorted(pooled.unexpected.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:20s} {count} frames")
    return "\n".join(lines)
