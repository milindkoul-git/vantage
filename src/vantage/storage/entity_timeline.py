"""Entity timeline reconstruction (temporal scene memory).

Reconstructs the full lifecycle and chronology of an entity from stored
observations and events, collapsing continuous per-frame rows into discrete
state intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vantage.storage.contracts import Store


@dataclass(slots=True)
class TimelineEvent:
    """An event raised involving this entity."""

    timestamp: float
    rule: str
    severity: str
    summary: str
    zone: str | None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rule": self.rule,
            "severity": self.severity,
            "summary": self.summary,
            "zone": self.zone,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class TimelineSegment:
    """A contiguous period during which the entity maintained the same state."""

    start_time: float
    end_time: float
    motion: str | None
    mean_speed: float
    posture: str | None
    zones: list[str]
    activities: list[str]
    identity: str | None = None
    observation_count: int = 1

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": round(self.duration_s, 2),
            "motion": self.motion,
            "mean_speed": round(self.mean_speed, 2),
            "posture": self.posture,
            "zones": self.zones,
            "activities": self.activities,
            "identity": self.identity,
            "observation_count": self.observation_count,
        }


@dataclass(slots=True)
class EntityTimeline:
    """The complete chronological history of an entity."""

    entity_id: str
    camera_id: str
    first_seen: float
    last_seen: float
    identity: str | None
    segments: list[TimelineSegment] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def describe(self) -> str:
        zones_visited = {z for s in self.segments for z in s.zones if z}
        zone_str = f" in {', '.join(sorted(zones_visited))}" if zones_visited else ""
        return (
            f"Entity {self.entity_id} ({self.identity or 'anonymous'}): "
            f"{self.total_duration_s:.1f}s active{zone_str}, "
            f"{len(self.segments)} state intervals, {len(self.events)} events"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "camera_id": self.camera_id,
            "identity": self.identity,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_duration_s": round(self.total_duration_s, 2),
            "summary": self.describe(),
            "segments": [s.to_dict() for s in self.segments],
            "events": [e.to_dict() for e in self.events],
        }


def _parse_list(val: str | None) -> list[str]:
    if not val:
        return []
    # Values stored like ",till,lobby,"
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts


def build_entity_timeline(
    store: Store,
    entity_id: str,
    *,
    since: float | None = None,
    until: float | None = None,
) -> EntityTimeline | None:
    """Reconstruct an entity's story by projecting observations and events."""
    from vantage.storage.contracts import Query

    obs_query = Query(
        entity_id=entity_id,
        since=since,
        until=until,
        limit=10000,
        newest_first=False,
    )
    obs_rows = store.observations(obs_query)
    if not obs_rows:
        return None

    first_seen = obs_rows[0].timestamp
    last_seen = obs_rows[-1].timestamp
    camera_id = obs_rows[0].camera_id
    identity = next((r.identity for r in obs_rows if r.identity), None)

    segments: list[TimelineSegment] = []
    current_seg: TimelineSegment | None = None
    accum_speed = 0.0
    accum_count = 0

    for row in obs_rows:
        zones = _parse_list(row.zones)
        activities = _parse_list(row.activities)

        state_key = (row.motion, row.posture, tuple(sorted(zones)), tuple(sorted(activities)))

        if current_seg is None:
            current_seg = TimelineSegment(
                start_time=row.timestamp,
                end_time=row.timestamp,
                motion=row.motion,
                mean_speed=row.speed or 0.0,
                posture=row.posture,
                zones=zones,
                activities=activities,
                identity=row.identity,
                observation_count=1,
            )
            accum_speed = row.speed or 0.0
            accum_count = 1
        else:
            prev_key = (
                current_seg.motion,
                current_seg.posture,
                tuple(sorted(current_seg.zones)),
                tuple(sorted(current_seg.activities)),
            )
            if state_key == prev_key and (row.timestamp - current_seg.end_time) < 2.0:
                current_seg.end_time = row.timestamp
                accum_speed += row.speed or 0.0
                accum_count += 1
                current_seg.observation_count = accum_count
                current_seg.mean_speed = accum_speed / accum_count
                if row.identity and not current_seg.identity:
                    current_seg.identity = row.identity
            else:
                segments.append(current_seg)
                current_seg = TimelineSegment(
                    start_time=row.timestamp,
                    end_time=row.timestamp,
                    motion=row.motion,
                    mean_speed=row.speed or 0.0,
                    posture=row.posture,
                    zones=zones,
                    activities=activities,
                    identity=row.identity,
                    observation_count=1,
                )
                accum_speed = row.speed or 0.0
                accum_count = 1

    if current_seg is not None:
        segments.append(current_seg)

    # Query events for this entity
    ev_query = Query(
        entity_id=entity_id,
        since=since,
        until=until,
        limit=1000,
        newest_first=False,
    )
    event_rows = store.events(ev_query)
    timeline_events: list[TimelineEvent] = []
    for ev in event_rows:
        timeline_events.append(
            TimelineEvent(
                timestamp=ev.timestamp,
                rule=ev.rule,
                severity=ev.severity,
                summary=ev.summary,
                zone=ev.zone,
                evidence=ev.evidence or {},
            )
        )

    return EntityTimeline(
        entity_id=entity_id,
        camera_id=camera_id,
        first_seen=first_seen,
        last_seen=last_seen,
        identity=identity,
        segments=segments,
        events=timeline_events,
    )
