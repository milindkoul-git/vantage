"""Event contracts: the difference between a state and something worth saying.

Everything before this phase produces **observations** - continuous statements
about what is true right now. ``person_3 is loitering``, ``person_3 is in the
doorway``, ``person_3 is near person_7``. They are true on every frame for as
long as they are true, which is exactly right for a state and exactly wrong for
an alert.

An **event** is discrete: it happened, at a time, once. The whole job of this
phase is that reduction, and the hard part of it is not deciding what is
interesting - the rules are short - but making sure a thing that is true for
forty-five consecutive frames produces *one* event rather than forty-five.

Why that matters more than it sounds
------------------------------------
Phase 5 holds a transient activity for 1.5 seconds so a slow consumer cannot
miss it. At 30 fps that is 45 frames of ``falling``. An event engine that
emitted per frame would produce 45 fall alerts for one fall, and whoever reads
them would learn to ignore the channel - which is the same failure the fall rule
itself was designed to avoid by refusing to hedge.

So every rule here carries a cooldown, and the engine keys it by rule *and*
entity so that two people falling at once are two events, not one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class Severity(str, Enum):
    """How much attention an event asks for.

    Three levels, not five. The distinction that survives contact with an
    operator is "tell me now", "tell me", and "write it down"; finer gradations
    get argued about and then ignored.
    """

    INFO = "info"
    """Worth recording. Someone entered a zone."""

    NOTICE = "notice"
    """Worth looking at when convenient. Someone has been loitering."""

    ALERT = "alert"
    """Worth interrupting for. Someone fell."""

    @property
    def rank(self) -> int:
        return {"info": 0, "notice": 1, "alert": 2}[self.value]


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened, once."""

    rule: str
    """Which rule fired. The stable key a consumer filters on."""

    severity: Severity
    summary: str
    """One human-readable sentence. Written by the rule, not templated here,
    because only the rule knows which of its inputs mattered."""

    entity_id: str | None
    """The entity the event is about, or ``None`` for scene-level events."""

    track_id: int | None
    frame_index: int
    capture_wall: float
    elapsed_s: float
    """Footage time when this fired, so a recorded source replays identically."""

    evidence: dict[str, object] = field(default_factory=dict)
    """What the rule actually measured. Every event can be argued with."""

    zone: str | None = None
    related_id: str | None = None
    """The other entity, for events about a pair."""

    def __post_init__(self) -> None:
        if not self.rule.strip():
            raise ValueError("an event needs a rule name")
        if not self.summary.strip():
            raise ValueError(f"event {self.rule!r} needs a summary")

    @property
    def key(self) -> tuple[str, str | None]:
        """Rule plus subject - what a cooldown is keyed on.

        Includes the entity so that two people falling at the same moment
        produce two events. Keying on the rule alone would silently drop the
        second, which is the one case where a missed alert matters most.
        """
        return (self.rule, self.entity_id)

    def describe(self) -> str:
        stamp = datetime.fromtimestamp(self.capture_wall, tz=UTC).strftime("%H:%M:%S")
        return f"[{stamp}] {self.severity.value.upper():6s} {self.summary}"

    def to_record(self, camera_id: str) -> dict[str, object]:
        """The storable form, for the phase that persists these.

        Plain primitives only, and ``identity`` present and always ``None`` -
        the same seam every earlier phase left, so one resolver can fill all of
        them at once rather than each record type needing its own migration.
        """
        return {
            "timestamp": datetime.fromtimestamp(self.capture_wall, tz=UTC).isoformat(),
            "camera_id": camera_id,
            "rule": self.rule,
            "severity": self.severity.value,
            "summary": self.summary,
            "entity_id": self.entity_id,
            "identity": None,
            "related_id": self.related_id,
            "zone": self.zone,
            "frame_index": self.frame_index,
            "elapsed_s": round(self.elapsed_s, 3),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class EventResult:
    """Events raised on one frame - usually none."""

    events: tuple[Event, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    elapsed_s: float = 0.0
    suppressed: int = 0
    """Rule firings held back by a cooldown on this frame.

    Reported rather than discarded silently: a rule suppressing thousands of
    firings is either correctly debouncing a continuous state or wrongly
    configured, and the count is what distinguishes the two.
    """

    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def of_severity(self, severity: Severity) -> tuple[Event, ...]:
        return tuple(event for event in self.events if event.severity is severity)

    @property
    def highest(self) -> Severity | None:
        return max((e.severity for e in self.events), key=lambda s: s.rank, default=None)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for event in self.events:
            tally[event.rule] = tally.get(event.rule, 0) + 1
        return tally

    def describe(self) -> str:
        if not self.events:
            return "no events"
        return "; ".join(event.describe() for event in self.events)
