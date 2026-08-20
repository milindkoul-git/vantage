"""Assembling records: the one place that knows every stage's output shape.

Kept apart from the writer, which knows nothing about poses or zones, and from
the store, which knows nothing about frames. This module is the seam between the
analysis pipeline and persistence, and it is the only file that has to change
when a stage gains a field worth keeping.

Sampling, deliberately rather than by overflow
----------------------------------------------
One observation per entity per analysed frame is 120 rows a second with four
people at 30 fps - ten million a day. Most of those rows are identical to their
predecessor, because an entity's state changes on the scale of seconds, not
frames.

``observation_interval`` samples that stream: one analysed frame in N is
recorded. Chosen sampling is honest and reproducible; letting the queue overflow
is neither, because what you lose then depends on when the disk happened to be
busy. Events are never sampled - they are already the rare, deliberate output of
a rule.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vantage.activity.contracts import ActivityResult
from vantage.events.contracts import Event, EventResult
from vantage.pose.contracts import PoseResult
from vantage.spatial.contracts import SpatialResult
from vantage.state.contracts import StateResult
from vantage.storage.schema import wrap_list
from vantage.storage.writer import StoreWriter

if TYPE_CHECKING:
    from vantage.identity.contracts import IdentityResult


class Recorder:
    """Turns one frame's analysis into rows and hands them to the writer."""

    def __init__(
        self,
        writer: StoreWriter,
        *,
        camera_id: str = "camera_01",
        observation_interval: int = 15,
        store_observations: bool = True,
        heartbeat_interval_s: float = 60.0,
    ) -> None:
        if observation_interval < 1:
            raise ValueError("observation_interval must be >= 1")
        if heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        self._writer = writer
        self._camera_id = camera_id
        self._interval = observation_interval
        self._store_observations = store_observations
        self._heartbeat_interval_s = heartbeat_interval_s
        self._last_heartbeat = 0.0
        self._steps = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def steps(self) -> int:
        return self._steps

    def record(
        self,
        *,
        state: StateResult | None = None,
        pose: PoseResult | None = None,
        activity: ActivityResult | None = None,
        spatial: SpatialResult | None = None,
        events: EventResult | None = None,
        identity: IdentityResult | None = None,
    ) -> int:
        """Enqueue this frame's records. Returns how many were accepted."""
        accepted = 0

        # Events first, and never sampled. If a queue is under pressure the one
        # that must survive should already be in it.
        if events is not None:
            for event in events:
                accepted += self._writer.add_event(self._event_row(event))

        self._steps += 1
        self._beat()

        if not self._store_observations or state is None:
            return accepted
        if (self._steps - 1) % self._interval != 0:
            return accepted

        postures = (
            {p.track_id: p.posture.value for p in pose if p.posture.value != "unknown"}
            if pose is not None
            else {}
        )
        activities = (
            {
                entity.track_id: tuple(
                    o.activity.value for o in entity if o.activity.value != "idle"
                )
                for entity in activity
            }
            if activity is not None
            else {}
        )
        zones = (
            {entity.track_id: entity.zone_names for entity in spatial}
            if spatial is not None
            else {}
        )
        # The identity column has been present and NULL since Phase 8. This is
        # the line that fills it, and it is the only change storage needed to
        # accommodate an identity layer - which was the point of putting the
        # column there before anything could write to it.
        names = (
            {item.track_id: item.name for item in identity if item.known}
            if identity is not None
            else {}
        )

        for entity in state:
            accepted += self._writer.add_observation(
                self._observation(entity, postures, activities, zones, names, state)
            )
        return accepted

    def _beat(self) -> None:
        """Record that this camera is alive, about once a minute.

        Unconditional on there being anything to see, which is the entire point.
        An empty scene produces no observation rows, so without this the store
        cannot distinguish an empty room from a dead recorder - and analytics
        would either learn every outage as normal quiet or refuse to judge any
        overnight hour at all. One row a minute settles it.

        Emitted from ``record`` rather than from a timer thread so that it means
        what it says: a heartbeat is written because the pipeline completed a
        frame, so its presence is evidence the pipeline was actually working
        rather than evidence a thread was still scheduled.
        """
        now = time.time()
        if now - self._last_heartbeat < self._heartbeat_interval_s:
            return
        self._last_heartbeat = now
        self._writer.add_heartbeat({"camera_id": self._camera_id, "timestamp": now})

    def _event_row(self, event: Event) -> dict[str, Any]:
        """One event as database columns.

        Deliberately not ``Event.to_record()``. That method is the *export*
        shape - JSON for an API or a message queue - and it renders the
        timestamp as an ISO string, which is right for a reader and wrong for a
        column that range queries sort on.

        Using it here stored the string into a REAL column, which SQLite accepts
        without complaint because it is dynamically typed, and the first query
        that tried to format one failed with "'str' object cannot be interpreted
        as an integer". Export shape and storage shape are different concerns;
        the seam between them belongs here, in the one module that knows both.
        """
        return {
            "timestamp": event.capture_wall,
            "camera_id": self._camera_id,
            "rule": event.rule,
            "severity": event.severity.value,
            "summary": event.summary,
            "entity_id": event.entity_id,
            "identity": None,
            "related_id": event.related_id,
            "zone": event.zone,
            "frame_index": event.frame_index,
            "elapsed_s": event.elapsed_s,
            "evidence": event.evidence,
        }

    def _observation(
        self,
        entity: Any,
        postures: dict[int, str],
        activities: dict[int, tuple[str, ...]],
        zones: dict[int, tuple[str, ...]],
        names: dict[int, str],
        state: StateResult,
    ) -> dict[str, Any]:
        return {
            "timestamp": state.capture_wall,
            "camera_id": self._camera_id,
            "entity_id": entity.entity_id,
            # Filled when identity resolution is running and has committed a
            # name for this entity; None otherwise, which is every deployment
            # that never turns it on.
            "identity": names.get(entity.track_id),
            "entity_type": entity.label,
            "motion": entity.motion.value,
            "speed": round(entity.speed, 4),
            "posture": postures.get(entity.track_id),
            "zones": wrap_list(list(zones.get(entity.track_id, ()))),
            "activities": wrap_list(list(activities.get(entity.track_id, ()))),
            "frame_index": state.frame_index,
            "elapsed_s": round(entity.age_s, 3),
        }
