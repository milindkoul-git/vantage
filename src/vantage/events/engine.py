"""The event engine: continuous observations in, discrete events out.

The rules are short. The engine is where the actual difficulty lives, and it is
one problem: **a condition that is true for many consecutive frames must produce
one event.**

Cooldown, keyed by rule and subject
-----------------------------------
Each firing is suppressed if the same rule has already fired about the same
entity within its cooldown. Both halves of that key matter:

* Without the **rule**, a loitering event would silence a fall.
* Without the **entity**, two people falling in the same second would produce
  one alert, and the second person is precisely who a missed alert fails.

Suppressions are counted, not discarded. A rule suppressing thousands of firings
is either correctly debouncing a continuous state or badly configured, and only
the count distinguishes those.

Footage time, not wall time
---------------------------
Cooldowns are measured in the same accumulated elapsed time the rest of the
platform uses, so a recorded source replays identically and a cooldown of ten
seconds means ten seconds of footage whatever the machine was doing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from vantage.activity.contracts import ActivityResult
from vantage.core.logging import get_logger
from vantage.events.contracts import Event, EventCandidate, EventResult, Severity
from vantage.events.rules import DEFAULT_RULES, RuleSpec, SceneContext, evaluate
from vantage.spatial.contracts import SpatialResult
from vantage.state.contracts import StateResult
from vantage.tracking.contracts import TrackingResult

log = get_logger(__name__)

DEFAULT_CANDIDATE_COOLDOWNS: dict[str, float] = {
    "tailgating": 25.0,
    "wrong_way_direction": 25.0,
    "loitering": 45.0,
    "cross_camera_handover": 30.0,
    "exclusion_breach": 20.0,
    "occupancy_limit": 20.0,
    "dwell_threshold": 30.0,
    "directional_flow": 25.0,
    "sudden_collapse": 20.0,
    "erratic_pacing": 30.0,
    "erratic_high_energy_motion": 25.0,
    "crouching_dwell": 30.0,
    "abrupt_direction_reversal": 20.0,
    "group_convergence": 40.0,
    "group_dispersion": 40.0,
    "unattended_object_dwell": 60.0,
    "following_pattern": 45.0,
    "recurring_proximity": 60.0,
    "recurrent_interaction": 60.0,
    "group_association": 60.0,
    "incident_escalation": 60.0,
    "incident_merge_candidate": 60.0,
}


class EventEngine:
    """Applies a rule set to each frame, debouncing what it produces."""

    def __init__(
        self,
        rules: tuple[RuleSpec, ...] | None = None,
        custom_cooldowns: dict[str, float] | None = None,
    ) -> None:
        self._rules = tuple(rules) if rules is not None else DEFAULT_RULES
        self._custom_cooldowns = dict(DEFAULT_CANDIDATE_COOLDOWNS)
        if custom_cooldowns:
            self._custom_cooldowns.update(custom_cooldowns)
        self._last_fired: dict[tuple[str, str | None], float] = {}
        self._elapsed = 0.0
        self._raised = 0
        self._suppressed = 0
        self._by_rule: dict[str, int] = {}

    @property
    def rules(self) -> tuple[RuleSpec, ...]:
        return self._rules

    @property
    def elapsed_s(self) -> float:
        return self._elapsed

    @property
    def raised(self) -> int:
        return self._raised

    @property
    def suppressed(self) -> int:
        return self._suppressed

    @property
    def tracked_keys(self) -> int:
        return len(self._last_fired)

    def stats(self) -> dict[str, object]:
        return {
            "rules": len(self._rules),
            "raised": self._raised,
            "suppressed": self._suppressed,
            "by_rule": dict(self._by_rule),
        }

    def update(
        self,
        tracking: TrackingResult | None = None,
        state: StateResult | None = None,
        activity: ActivityResult | None = None,
        spatial: SpatialResult | None = None,
    ) -> EventResult:
        """Evaluate every rule against this frame."""
        elapsed = _elapsed_of(tracking, state, activity, spatial)
        self._elapsed += elapsed
        frame_index, capture_wall, source_id = _frame_of(tracking, state, activity, spatial)

        context = SceneContext(
            tracking=tracking,
            state=state,
            activity=activity,
            spatial=spatial,
            elapsed_s=self._elapsed,
            frame_index=frame_index,
            capture_wall=capture_wall,
            source_id=source_id,
        )

        raised: list[Event] = []
        suppressed = 0
        for spec in self._rules:
            for event in evaluate(spec, context):
                if self._is_cooling(event, spec.cooldown_s):
                    suppressed += 1
                    continue
                self._last_fired[event.key] = self._elapsed
                self._by_rule[event.rule] = self._by_rule.get(event.rule, 0) + 1
                raised.append(event)

        self._raised += len(raised)
        self._suppressed += suppressed
        self._forget(tracking)

        for event in raised:
            # Every event is logged at a level matching its severity, so an
            # operator watching the log sees the same thing a consumer would.
            logger = log.warning if event.severity is Severity.ALERT else log.info
            logger(
                "event",
                extra={
                    "vantage_fields": {
                        "rule": event.rule,
                        "severity": event.severity.value,
                        "summary": event.summary,
                        **{f"evidence_{k}": v for k, v in event.evidence.items()},
                    }
                },
            )

        return EventResult(
            events=tuple(raised),
            source_id=source_id,
            frame_index=frame_index,
            capture_wall=capture_wall,
            elapsed_s=elapsed,
            suppressed=suppressed,
            metadata={"elapsed_total_s": round(self._elapsed, 2)},
        )

    def evaluate_candidate(self, candidate: EventCandidate) -> Event | None:
        """Evaluate a single EventCandidate against cooldown and suppression policy."""
        # Find cooldown
        cooldown_s = self._custom_cooldowns.get(candidate.rule, 15.0)
        for spec in self._rules:
            if spec.name == candidate.rule:
                cooldown_s = spec.cooldown_s
                break

        key = (candidate.rule, candidate.entity_id)
        current_time = candidate.wall_time if candidate.wall_time > 0 else self._elapsed
        last_t = self._last_fired.get(key, 0.0)

        if current_time - last_t < cooldown_s:
            self._suppressed += 1
            return None

        self._last_fired[key] = current_time
        self._raised += 1
        self._by_rule[candidate.rule] = self._by_rule.get(candidate.rule, 0) + 1

        # Normalize severity
        sev = candidate.severity
        if isinstance(sev, str):
            sev = (
                Severity(sev.lower())
                if sev.lower() in ("info", "notice", "alert")
                else Severity.INFO
            )

        event = Event(
            rule=candidate.rule,
            severity=sev,
            summary=candidate.summary,
            entity_id=candidate.entity_id,
            track_id=candidate.track_id,
            frame_index=candidate.frame_index,
            capture_wall=candidate.wall_time or time.time(),
            elapsed_s=candidate.elapsed_s or round(current_time % 1000, 2),
            evidence=dict(candidate.evidence),
            zone=candidate.zone,
            related_id=candidate.related_id,
        )

        logger = log.warning if event.severity is Severity.ALERT else log.info
        logger(
            "event",
            extra={
                "vantage_fields": {
                    "rule": event.rule,
                    "severity": event.severity.value,
                    "summary": event.summary,
                    "entity_id": event.entity_id,
                    "camera_id": candidate.camera_id,
                    **{f"evidence_{k}": v for k, v in event.evidence.items()},
                }
            },
        )
        return event

    def evaluate_candidates(self, candidates: Sequence[EventCandidate]) -> tuple[Event, ...]:
        """Evaluate a sequence of EventCandidates, returning allowed Events."""
        events: list[Event] = []
        for cand in candidates:
            ev = self.evaluate_candidate(cand)
            if ev is not None:
                events.append(ev)
        return tuple(events)

    def _is_cooling(self, event: Event, cooldown_s: float) -> bool:
        if cooldown_s <= 0:
            return False
        previous = self._last_fired.get(event.key)
        return previous is not None and (self._elapsed - previous) < cooldown_s

    def _forget(self, tracking: TrackingResult | None) -> None:
        """Drop cooldowns for entities the tracker has retired.

        Keyed by entity id, so on a camera running for weeks this grows without
        bound unless something prunes it - the same leak Phase 3 shipped by
        accident once. Scene-level keys (entity ``None``) are kept: they belong
        to a place, which does not go away.
        """
        if tracking is None:
            return
        live = {track.entity_id for track in tracking.tracks}
        for key in [k for k in self._last_fired if k[1] is not None and k[1] not in live]:
            del self._last_fired[key]

    def reset(self) -> None:
        self._last_fired.clear()
        self._elapsed = 0.0
        self._raised = self._suppressed = 0
        self._by_rule.clear()


def _elapsed_of(*results: object) -> float:
    for result in results:
        value = getattr(result, "elapsed_s", None)
        if value is not None:
            return max(0.0, float(value))
    return 0.0


def _frame_of(*results: object) -> tuple[int, float, str]:
    for result in results:
        if result is not None:
            return (
                int(getattr(result, "frame_index", 0)),
                float(getattr(result, "capture_wall", 0.0)),
                str(getattr(result, "source_id", "unknown")),
            )
    return (0, 0.0, "unknown")


def build_event_engine(config=None) -> EventEngine:
    """Construct from an :class:`~vantage.config.schema.EventsConfig`."""
    if config is None:
        return EventEngine()
    if not config.rules:
        # An empty list in the config means "the defaults", not "no rules at
        # all". Turning the subsystem off is what `enabled: false` is for, and
        # conflating the two would let a stray edit silence every alert without
        # anything saying so.
        return EventEngine()
    rules = tuple(
        RuleSpec(
            type=rule.type,
            name=rule.name,
            severity=Severity(rule.severity),
            cooldown_s=rule.cooldown_s,
            zones=tuple(rule.zones),
            labels=tuple(rule.labels),
            activity=rule.activity,
            relation=rule.relation,
            min_confidence=rule.min_confidence,
            min_seconds=rule.min_seconds,
            min_count=rule.min_count,
        )
        for rule in config.rules
    )
    log.info(
        "event rules loaded",
        extra={"vantage_fields": {"rules": ", ".join(r.label for r in rules)}},
    )
    return EventEngine(rules)
