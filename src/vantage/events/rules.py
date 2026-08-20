"""The rule set: a small table of parameterised types, not a language.

Six rule types cover every example the specification gives - someone entered a
zone, someone approached someone, someone interacted with an object, someone
loitered, someone fell, a zone got crowded. Each is configured from YAML with a
handful of parameters.

Why not a rule language
-----------------------
A general expression DSL over the scene graph was the tempting alternative. It
was rejected on the same grounds as the learned action model in Phase 5: it
would be a large amount of machinery whose failures are hard to explain, in
service of expressiveness nobody has asked for yet. A table of typed rules can
be validated at load time - an unknown activity name is caught when the config
is read, not when the situation finally occurs at three in the morning - and
every rule's behaviour can be stated in one sentence.

If the sixth rule type turns out not to be enough, adding a seventh is twenty
lines. Adding a parser is a subsystem.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vantage.activity.contracts import Activity, ActivityResult
from vantage.core.errors import ConfigError
from vantage.events.contracts import Event, Severity
from vantage.spatial.contracts import Relation, SpatialResult, ZoneEvent
from vantage.state.contracts import StateResult
from vantage.tracking.contracts import TrackingResult

DEFAULT_COOLDOWN_S = 5.0
"""How long a rule stays quiet about the same entity after firing.

Not zero, and the reason is arithmetic rather than taste. Phase 5 holds a
transient activity for 1.5 seconds so a slow consumer cannot miss it; at 30 fps
that is 45 frames on which ``falling`` is true. Without a cooldown one fall
produces 45 alerts, and a channel that cries 45 times teaches its reader to
ignore it - the same failure the fall rule avoids by refusing to hedge.
"""


@dataclass(frozen=True, slots=True)
class SceneContext:
    """Everything a rule may look at for one frame.

    Passed whole rather than as separate arguments so that adding a signal later
    does not change every rule's signature. Any of the results may be ``None``
    when that stage is not running, and every rule has to cope: a deployment
    without pose still wants zone events.
    """

    tracking: TrackingResult | None
    state: StateResult | None
    activity: ActivityResult | None
    spatial: SpatialResult | None
    elapsed_s: float
    frame_index: int
    capture_wall: float
    source_id: str


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """One configured rule."""

    type: str
    name: str = ""
    severity: Severity = Severity.INFO
    cooldown_s: float = DEFAULT_COOLDOWN_S

    zones: tuple[str, ...] = ()
    """Restrict to these zones. Empty means any zone."""

    labels: tuple[str, ...] = ()
    """Restrict to these entity classes. Empty means any."""

    activity: str = ""
    relation: str = ""
    min_confidence: float = 0.0
    min_seconds: float = 0.0
    min_count: int = 2

    def __post_init__(self) -> None:
        if self.type not in RULE_TYPES:
            raise ConfigError(
                f"unknown event rule type {self.type!r}. Available: {sorted(RULE_TYPES)}"
            )
        if self.cooldown_s < 0:
            raise ConfigError(f"event rule {self.label!r}: cooldown_s must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError(f"event rule {self.label!r}: min_confidence must be in [0, 1]")
        if self.type == "activity":
            if not self.activity:
                raise ConfigError("an 'activity' rule needs an activity name")
            valid = {a.value for a in Activity}
            if self.activity not in valid:
                # Caught when the config is read rather than when the situation
                # finally occurs. A typo here would otherwise be a rule that can
                # never fire, and silence is indistinguishable from calm.
                raise ConfigError(
                    f"event rule {self.label!r}: unknown activity {self.activity!r}. "
                    f"Available: {sorted(valid)}"
                )
        if self.type == "relation":
            if not self.relation:
                raise ConfigError("a 'relation' rule needs a relation name")
            valid = {r.value for r in Relation}
            if self.relation not in valid:
                raise ConfigError(
                    f"event rule {self.label!r}: unknown relation {self.relation!r}. "
                    f"Available: {sorted(valid)}"
                )
        if self.type == "zone_occupancy" and self.min_count < 1:
            raise ConfigError(f"event rule {self.label!r}: min_count must be >= 1")

    @property
    def label(self) -> str:
        """The rule's stable name - explicit, or derived from what it watches."""
        if self.name:
            return self.name
        parts = [self.type]
        if self.activity:
            parts.append(self.activity)
        if self.relation:
            parts.append(self.relation)
        if self.zones:
            parts.append("+".join(self.zones))
        return ":".join(parts)

    def wants_zone(self, zone: str | None) -> bool:
        return not self.zones or (zone is not None and zone in self.zones)

    def wants_label(self, label: str) -> bool:
        return not self.labels or label.lower() in {n.lower() for n in self.labels}


def _activity_rule(spec: RuleSpec, context: SceneContext) -> list[Event]:
    """Fires when an entity is doing a named thing."""
    if context.activity is None:
        return []
    wanted = Activity(spec.activity)
    zones_of = _zone_index(context)
    events: list[Event] = []

    for entity in context.activity:
        if not spec.wants_label(entity.label):
            continue
        observation = entity.get(wanted)
        if observation is None:
            continue
        if observation.confidence < spec.min_confidence:
            continue
        if observation.duration_s < spec.min_seconds:
            continue
        zone = _first_matching_zone(spec, zones_of.get(entity.track_id, ()))
        if spec.zones and zone is None:
            continue
        events.append(
            Event(
                rule=spec.label,
                severity=spec.severity,
                summary=(
                    f"{entity.entity_id} is {wanted.value.replace('_', ' ')}"
                    + (f" in {zone}" if zone else "")
                ),
                entity_id=entity.entity_id,
                track_id=entity.track_id,
                frame_index=context.frame_index,
                capture_wall=context.capture_wall,
                elapsed_s=context.elapsed_s,
                zone=zone,
                evidence={
                    "activity": wanted.value,
                    "confidence": round(observation.confidence, 3),
                    "duration_s": round(observation.duration_s, 2),
                    "why": observation.evidence,
                },
            )
        )
    return events


def _zone_crossing_rule(wanted: ZoneEvent) -> Callable[[RuleSpec, SceneContext], list[Event]]:
    """Build an entry or exit rule; the two differ only by which event they want."""

    def rule(spec: RuleSpec, context: SceneContext) -> list[Event]:
        if context.spatial is None:
            return []
        events: list[Event] = []
        for entity, occupancy in context.spatial.crossings():
            if occupancy.event is not wanted:
                continue
            if not spec.wants_zone(occupancy.zone) or not spec.wants_label(entity.label):
                continue
            verb = "entered" if wanted is ZoneEvent.ENTERED else "left"
            events.append(
                Event(
                    rule=spec.label,
                    severity=spec.severity,
                    summary=f"{entity.entity_id} {verb} {occupancy.zone}",
                    entity_id=entity.entity_id,
                    track_id=entity.track_id,
                    frame_index=context.frame_index,
                    capture_wall=context.capture_wall,
                    elapsed_s=context.elapsed_s,
                    zone=occupancy.zone,
                    evidence={
                        "zone_kind": occupancy.kind,
                        "dwell_s": round(occupancy.dwell_s, 2),
                    },
                )
            )
        return events

    return rule


def _zone_dwell_rule(spec: RuleSpec, context: SceneContext) -> list[Event]:
    """Fires when an entity has been in a zone longer than ``min_seconds``."""
    if context.spatial is None:
        return []
    events: list[Event] = []
    for entity in context.spatial:
        if not spec.wants_label(entity.label):
            continue
        for occupancy in entity.occupied:
            if not spec.wants_zone(occupancy.zone):
                continue
            if occupancy.dwell_s < spec.min_seconds:
                continue
            events.append(
                Event(
                    rule=spec.label,
                    severity=spec.severity,
                    summary=(
                        f"{entity.entity_id} has been in {occupancy.zone} for "
                        f"{occupancy.dwell_s:.0f}s"
                    ),
                    entity_id=entity.entity_id,
                    track_id=entity.track_id,
                    frame_index=context.frame_index,
                    capture_wall=context.capture_wall,
                    elapsed_s=context.elapsed_s,
                    zone=occupancy.zone,
                    evidence={
                        "dwell_s": round(occupancy.dwell_s, 2),
                        "threshold_s": spec.min_seconds,
                        "zone_kind": occupancy.kind,
                    },
                )
            )
    return events


def _zone_occupancy_rule(spec: RuleSpec, context: SceneContext) -> list[Event]:
    """Fires when a zone holds at least ``min_count`` entities. Scene-level."""
    if context.spatial is None:
        return []
    events: list[Event] = []
    for zone, count in sorted(context.spatial.occupancy().items()):
        if not spec.wants_zone(zone) or count < spec.min_count:
            continue
        events.append(
            Event(
                rule=spec.label,
                severity=spec.severity,
                summary=f"{count} entities in {zone}",
                # No entity: this is about the place, not any one occupant, and
                # attributing it to whichever happened to be listed first would
                # make the cooldown behave differently as people moved around.
                entity_id=None,
                track_id=None,
                frame_index=context.frame_index,
                capture_wall=context.capture_wall,
                elapsed_s=context.elapsed_s,
                zone=zone,
                evidence={"count": count, "threshold": spec.min_count},
            )
        )
    return events


def _relation_rule(spec: RuleSpec, context: SceneContext) -> list[Event]:
    """Fires on a spatial relation between two entities."""
    if context.spatial is None:
        return []
    wanted = Relation(spec.relation)
    events: list[Event] = []
    for relation in context.spatial.of(wanted):
        if relation.confidence < spec.min_confidence:
            continue
        if relation.duration_s < spec.min_seconds:
            continue
        events.append(
            Event(
                rule=spec.label,
                severity=spec.severity,
                summary=(
                    f"{relation.subject_id} {wanted.value.replace('_', ' ')} "
                    f"{relation.object_id}"
                ),
                entity_id=relation.subject_id,
                track_id=relation.subject_track,
                related_id=relation.object_id,
                frame_index=context.frame_index,
                capture_wall=context.capture_wall,
                elapsed_s=context.elapsed_s,
                evidence={
                    "relation": wanted.value,
                    "confidence": round(relation.confidence, 3),
                    "distance_heights": round(relation.distance, 3),
                    "duration_s": round(relation.duration_s, 2),
                    "why": relation.evidence,
                },
            )
        )
    return events


RULE_TYPES: dict[str, Callable[[RuleSpec, SceneContext], list[Event]]] = {
    "activity": _activity_rule,
    "zone_entry": _zone_crossing_rule(ZoneEvent.ENTERED),
    "zone_exit": _zone_crossing_rule(ZoneEvent.EXITED),
    "zone_dwell": _zone_dwell_rule,
    "zone_occupancy": _zone_occupancy_rule,
    "relation": _relation_rule,
}


def _zone_index(context: SceneContext) -> dict[int, tuple[str, ...]]:
    if context.spatial is None:
        return {}
    return {entity.track_id: entity.zone_names for entity in context.spatial}


def _first_matching_zone(spec: RuleSpec, zones: tuple[str, ...]) -> str | None:
    for zone in zones:
        if spec.wants_zone(zone):
            return zone
    return zones[0] if zones and not spec.zones else None


def evaluate(spec: RuleSpec, context: SceneContext) -> list[Event]:
    """Run one rule against one frame."""
    return RULE_TYPES[spec.type](spec, context)


DEFAULT_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        type="activity",
        activity="falling",
        severity=Severity.ALERT,
        cooldown_s=15.0,
        name="fall",
    ),
    RuleSpec(
        type="activity",
        activity="running",
        severity=Severity.NOTICE,
        min_seconds=1.0,
        name="running",
    ),
    RuleSpec(
        type="activity",
        activity="loitering",
        severity=Severity.NOTICE,
        cooldown_s=60.0,
        name="loitering",
    ),
    RuleSpec(type="zone_entry", severity=Severity.INFO, cooldown_s=2.0),
    RuleSpec(type="zone_exit", severity=Severity.INFO, cooldown_s=2.0),
)
"""A default set that does something sensible with no configuration at all.

Deliberately conservative: nothing here fires on a quiet scene, the only ALERT
is a fall, and the zone rules do nothing until zones are drawn. A default that
alerted on proximity or interaction would be noisy in exactly the deployments
least able to tune it.
"""
