"""Scoring zone assignment and relation detection against scripted ground truth.

Three outcomes, matching the three ways this phase can be wrong: a relation that
should have been found and was not, a relation that fired when it must not, and
a zone boundary crossing that was missed.

Forbidden firings are weighted hardest, as in Phase 5 and for the same reason.
An interaction rule that fires whenever a person walks past a table produces a
scene graph full of relationships that never happened, and everything the event
engine later builds on top of it inherits the fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vantage.pose.contracts import PoseResult
from vantage.spatial.analyzer import SpatialAnalyzer, SpatialParams
from vantage.spatial.contracts import Relation
from vantage.spatial.engine import SpatialEngine
from vantage.spatial.scenarios import SpatialScenario, generate
from vantage.state.estimator import StateEstimator, StateParams
from vantage.tracking.contracts import TrackingResult


@dataclass(slots=True)
class SpatialMetrics:
    """How one scenario went."""

    scenario: str
    frames: int = 0
    expected: tuple[tuple[Relation, int, int], ...] = ()
    found: dict[str, int] = field(default_factory=dict)
    forbidden_firings: int = 0
    unexpected: dict[str, int] = field(default_factory=dict)
    zone_events_expected: int = 0
    zone_events_found: int = 0
    peak_confidence: dict[str, float] = field(default_factory=dict)

    @property
    def expected_found(self) -> int:
        return sum(1 for item in self.expected if self.found.get(_label(item)))

    @property
    def passed(self) -> bool:
        return (
            self.expected_found == len(self.expected)
            and self.forbidden_firings == 0
            and self.zone_events_found == self.zone_events_expected
        )

    def describe(self) -> str:
        return (
            f"{self.scenario:22s} relations {self.expected_found}/{len(self.expected)}  "
            f"zones {self.zone_events_found}/{self.zone_events_expected}  "
            f"forbidden {self.forbidden_firings}"
        )


def _label(item: tuple[Relation, int, int]) -> str:
    relation, first, second = item
    if relation.is_symmetric:
        first, second = sorted((first, second))
    return f"{relation.value}:{first}:{second}"


def evaluate(scenario: SpatialScenario, params: SpatialParams | None = None) -> SpatialMetrics:
    """Run one scenario end to end and score it."""
    metrics = SpatialMetrics(
        scenario=scenario.name,
        expected=scenario.expect,
        zone_events_expected=len(scenario.expect_zone_events),
    )
    engine = SpatialEngine(SpatialAnalyzer(scenario.zones, params))
    # The REAL state estimator, not a stub. Interaction depends on motion
    # state, so a harness that faked it would be scoring the wrong thing.
    state_estimator = StateEstimator(StateParams())
    seen_zone_events: set[tuple[int, str, str]] = set()

    for frame in generate(scenario):
        tracking = TrackingResult(
            tracks=frame.tracks,
            source_id="scenario",
            frame_index=frame.index,
            capture_wall=frame.time_s,
            frame_size=(640, 480),
            elapsed_s=frame.elapsed_s,
        )
        pose_result = (
            PoseResult(
                poses=frame.poses,
                source_id="scenario",
                frame_index=frame.index,
                capture_wall=frame.time_s,
                frame_size=(640, 480),
                people_seen=len(frame.poses),
            )
            if frame.poses
            else None
        )
        result = engine.update(tracking, pose_result, state_estimator.update(tracking))
        metrics.frames += 1

        for relation in result.relations:
            key = _label((relation.relation, relation.subject_track, relation.object_track))
            metrics.found[key] = metrics.found.get(key, 0) + 1
            metrics.peak_confidence[key] = max(
                metrics.peak_confidence.get(key, 0.0), relation.confidence
            )
            if frame.time_s >= scenario.grace_s:
                for forbidden in scenario.forbidden:
                    if key == _label(forbidden):
                        metrics.forbidden_firings += 1
                        metrics.unexpected[key] = metrics.unexpected.get(key, 0) + 1

        for entity, occupancy in result.crossings():
            seen_zone_events.add((entity.track_id, occupancy.zone, occupancy.event.value))

    metrics.zone_events_found = sum(
        1
        for track_id, zone, event in scenario.expect_zone_events
        if (track_id, zone, event.value) in seen_zone_events
    )
    return metrics


def aggregate(results: list[SpatialMetrics]) -> SpatialMetrics:
    """Pool raw counts rather than averaging percentages."""
    pooled = SpatialMetrics(scenario="POOLED")
    for metrics in results:
        pooled.frames += metrics.frames
        pooled.expected = pooled.expected + metrics.expected
        pooled.forbidden_firings += metrics.forbidden_firings
        pooled.zone_events_expected += metrics.zone_events_expected
        pooled.zone_events_found += metrics.zone_events_found
        for key, count in metrics.found.items():
            pooled.found[key] = pooled.found.get(key, 0) + count
        for key, count in metrics.unexpected.items():
            pooled.unexpected[key] = pooled.unexpected.get(key, 0) + count
        for key, value in metrics.peak_confidence.items():
            # Scoped by scenario. Pooling on the relation key alone collapsed
            # linger_by_object and reach_for_object into one number, hiding the
            # two-tier confidence this table exists to show.
            pooled.peak_confidence[f"{metrics.scenario}/{key}"] = value
    return pooled


def format_table(results: list[SpatialMetrics]) -> str:
    width = 74
    lines = [
        f"{'SCENARIO':22s} {'RELATIONS':>10s} {'ZONES':>8s} {'FORBIDDEN':>10s} {'FRAMES':>8s}",
        "-" * width,
    ]
    for metrics in results:
        relations = (
            f"{metrics.expected_found}/{len(metrics.expected)}" if metrics.expected else "-"
        )
        zones = (
            f"{metrics.zone_events_found}/{metrics.zone_events_expected}"
            if metrics.zone_events_expected
            else "-"
        )
        flag = "" if metrics.passed else "  FAIL"
        lines.append(
            f"{metrics.scenario:22s} {relations:>10s} {zones:>8s} "
            f"{metrics.forbidden_firings:>10d} {metrics.frames:>8d}{flag}"
        )

    pooled = aggregate(results)
    lines.append("-" * width)
    lines.append(
        f"{'POOLED':22s} {f'{pooled.expected_found}/{len(pooled.expected)}':>10s} "
        f"{f'{pooled.zone_events_found}/{pooled.zone_events_expected}':>8s} "
        f"{pooled.forbidden_firings:>10d} {pooled.frames:>8d}"
    )

    # Interaction confidence is a claim about how much the evidence supports it,
    # so the two tiers are printed rather than left implicit in the code.
    reach = pooled.peak_confidence
    interactions = {k: v for k, v in reach.items() if "interacting" in k}
    if interactions:
        lines.append("")
        lines.append("Peak interaction confidence (evidence tier):")
        for key, value in sorted(interactions.items()):
            tier = "reach-confirmed" if value >= 0.8 else "proximity only"
            lines.append(f"  {key:36s} {value:.2f}  {tier}")
    if pooled.unexpected:
        lines.append("")
        lines.append("Forbidden relations that fired:")
        for key, count in sorted(pooled.unexpected.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {key:24s} {count} frames")
    return "\n".join(lines)
