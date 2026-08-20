"""Storage contracts: what gets written, and what can be asked of it.

Two record kinds, because they behave completely differently and conflating them
would force the wrong trade on one of them:

**Observations** are continuous. Every tracked entity produces one on every
analysed frame - at 30 fps with four entities that is 120 rows a second, ten
million a day. They are individually cheap to lose: the next frame says almost
the same thing.

**Events** are discrete and rare. A camera might produce a dozen a day. Each one
is the output of a rule that already decided it was worth someone's attention,
and losing one loses the thing the system exists to notice.

That difference drives everything downstream. Observations may be sampled on the
way in and pruned aggressively on the way out; events are neither. The writer
applies different queue policies to each, and says so when it drops anything.

The identity seam, still empty
------------------------------
Both tables carry an ``identity`` column that is always ``NULL``. Every phase
from 4 onward has emitted records with that field present and unset; putting the
column in now means the identity layer, if it is ever built, adds a resolver
rather than a schema migration over a table with ten million rows in it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RecordKind(str, Enum):
    OBSERVATION = "observation"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """One event as it exists on disk."""

    id: int
    timestamp: float
    """Wall-clock seconds. Stored as a REAL rather than a string because range
    queries over it are the single most common thing anyone asks."""

    camera_id: str
    rule: str
    severity: str
    summary: str
    entity_id: str | None
    identity: str | None
    related_id: str | None
    zone: str | None
    frame_index: int
    elapsed_s: float
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=UTC)

    def describe(self) -> str:
        stamp = self.when.strftime("%Y-%m-%d %H:%M:%S")
        where = f" [{self.zone}]" if self.zone else ""
        return f"{stamp}  {self.severity.upper():6s} {self.summary}{where}"


@dataclass(frozen=True, slots=True)
class StoredObservation:
    """One entity's state at one moment, as it exists on disk."""

    id: int
    timestamp: float
    camera_id: str
    entity_id: str
    identity: str | None
    entity_type: str
    motion: str | None
    speed: float | None
    posture: str | None
    zones: str | None
    """Comma-joined zone names. Denormalised deliberately - see the note in
    :mod:`vantage.storage.schema` on why this is not a join table."""

    activities: str | None
    frame_index: int
    elapsed_s: float

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=UTC)

    def describe(self) -> str:
        stamp = self.when.strftime("%H:%M:%S")
        parts = [p for p in (self.motion, self.posture, self.activities) if p]
        return f"{stamp}  {self.entity_id:12s} {', '.join(parts) or 'present'}"


@dataclass(frozen=True, slots=True)
class Query:
    """What to fetch. Every field optional; all supplied ones are ANDed.

    A dataclass rather than keyword arguments so a caller can build a query,
    pass it around, and log it. Ad-hoc SQL strings are deliberately not part of
    the interface: every filter here maps to an indexed column, and an interface
    that accepted arbitrary SQL would make that impossible to guarantee.
    """

    since: float | None = None
    until: float | None = None
    camera_id: str | None = None
    entity_id: str | None = None
    rule: str | None = None
    severity: str | None = None
    zone: str | None = None
    entity_type: str | None = None
    limit: int = 100
    newest_first: bool = True

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"query limit must be >= 1, got {self.limit}")
        if self.since is not None and self.until is not None and self.since > self.until:
            raise ValueError(
                f"query window is inverted: since={self.since} is after until={self.until}"
            )


@dataclass(slots=True)
class WriteStats:
    """What the writer has done, and what it has had to drop."""

    observations_queued: int = 0
    observations_written: int = 0
    observations_dropped: int = 0
    events_queued: int = 0
    events_written: int = 0
    events_dropped: int = 0
    batches: int = 0
    write_errors: int = 0
    last_error: str = ""

    @property
    def healthy(self) -> bool:
        return self.events_dropped == 0 and self.write_errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations_written": self.observations_written,
            "observations_dropped": self.observations_dropped,
            "events_written": self.events_written,
            "events_dropped": self.events_dropped,
            "batches": self.batches,
            "write_errors": self.write_errors,
            "last_error": self.last_error,
        }

    def describe(self) -> str:
        base = (
            f"{self.events_written} events, "
            f"{self.observations_written} observations in {self.batches} batches"
        )
        # Dropped events are called out separately and never folded into a
        # total. An event is the output of a rule that already decided it was
        # worth attention; losing one silently is the worst thing this
        # subsystem can do.
        if self.events_dropped:
            base += f"; {self.events_dropped} EVENTS DROPPED"
        if self.observations_dropped:
            base += f"; {self.observations_dropped} observations dropped"
        if self.write_errors:
            base += f"; {self.write_errors} write errors, last: {self.last_error}"
        return base


@runtime_checkable
class Store(Protocol):
    """What a backing store must do.

    A Protocol so a Postgres store can be added without inheriting anything
    from the SQLite one, whose internals it would share nothing with. The
    methods are deliberately few: this interface is what the rest of the
    platform is allowed to depend on, and every method added here is a method
    every future backend must implement.
    """

    def write_events(self, records: list[dict[str, Any]]) -> int: ...

    def write_observations(self, records: list[dict[str, Any]]) -> int: ...

    def write_heartbeats(self, records: list[dict[str, Any]]) -> int: ...

    """Record that a camera was alive at these moments.

    Part of the protocol rather than an extra on the SQLite implementation,
    because the writer calls it on whatever store it was given. Analytics needs
    it to tell an empty scene apart from a stopped recorder, and no arrangement
    of the observation rows can substitute for it."""

    def heartbeats(self, since: float, until: float) -> list[float]: ...

    def events(self, query: Query) -> list[StoredEvent]: ...

    def observations(self, query: Query) -> list[StoredObservation]: ...

    def counts(self) -> dict[str, int]: ...

    def prune(self, before: float) -> dict[str, int]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Timeline:
    """Events for one entity, in order. What "event timeline" means concretely."""

    entity_id: str
    events: tuple[StoredEvent, ...]

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[StoredEvent]:
        return iter(self.events)

    @property
    def span_s(self) -> float:
        if len(self.events) < 2:
            return 0.0
        stamps = [event.timestamp for event in self.events]
        return max(stamps) - min(stamps)

    def describe(self) -> str:
        if not self.events:
            return f"{self.entity_id}: nothing recorded"
        return f"{self.entity_id}: {len(self.events)} events over {self.span_s:.0f}s"
