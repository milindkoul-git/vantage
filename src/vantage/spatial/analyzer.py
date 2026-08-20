"""Zone membership and pairwise relations, over time.

Three things are computed here, in increasing order of how much they can be
trusted.

**Zone membership** is the solid one. A polygon test on the entity's ground
point is exact; the only judgement in it is where the zone was drawn.

**Proximity and approach** rest on the common-ground assumption. Separation is
the distance between two ground points, divided by the mean of the two entity
heights so the answer is scale-free. That is a good approximation for entities
at similar depth and degrades as their depths diverge, because a camera cannot
see depth at all.

**Interaction** is the one that needs care, and it is the reason this module has
two confidence levels rather than one. Boxes overlapping in a 2-D image is
famously weak evidence of contact in a 3-D room - a person walking three metres
behind a table overlaps it perfectly. So interaction is only claimed when it is
*sustained*, and it is claimed twice as strongly when a wrist landmark is
actually inside the object's box, which is the difference between "a person was
in front of this" and "a person reached for this". Both cases say which test was
met in their evidence string.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from vantage.core.errors import ConfigError
from vantage.pose.contracts import LEFT_WRIST, Pose, RIGHT_WRIST
from vantage.state.contracts import MotionState
from vantage.spatial.contracts import (
    EntitySpatial,
    Relation,
    RelationObservation,
    Zone,
    ZoneEvent,
    ZoneOccupancy,
)
from vantage.tracking.contracts import Track


@dataclass(frozen=True, slots=True)
class SpatialParams:
    """Thresholds for zones and relations. Distances are in entity heights."""

    near_distance: float = 1.5
    """Ground separation at or below which two entities are near.

    Roughly an arm's length plus a margin, on the rule of thumb that a standing
    adult is about as tall as two paces are long. It is a threshold on an
    approximation, not a measurement in metres, and it should be tuned per
    camera rather than trusted as a physical distance."""

    near_hysteresis: float = 0.3
    """Extra distance required to *stop* being near, so a pair hovering at the
    boundary does not flicker. The same dead-band lesson as Phase 4's motion
    state, applied to a different signal."""

    approach_rate: float = 0.25
    """Heights per second of closing speed before approach is reported."""

    approach_window_s: float = 0.8
    """How far back separation is compared to compute that rate. Long enough to
    ignore box jitter, short enough that a direction change is not missed."""

    interact_distance: float = 0.6
    interact_s: float = 1.0
    """How long a person must stay close to an object before interaction is
    claimed. This is what excludes walking past a table."""

    reach_confidence: float = 0.35
    """Minimum wrist landmark score for a reach to count as confirmed."""

    zone_event_hold_s: float = 1.5
    max_entities: int = 24
    """Relations are pairwise, so cost is quadratic. Above this many entities
    only the largest boxes are paired, and the count considered is reported -
    the same explicit-budget approach pose takes, for the same reason."""

    history: int = 120

    def __post_init__(self) -> None:
        for name in (
            "near_distance",
            "near_hysteresis",
            "approach_rate",
            "approach_window_s",
            "interact_distance",
            "interact_s",
            "zone_event_hold_s",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"spatial.{name} must be >= 0")
        if self.interact_distance > self.near_distance:
            raise ConfigError(
                f"spatial.interact_distance ({self.interact_distance}) must not exceed "
                f"spatial.near_distance ({self.near_distance}): interaction is a closer "
                "relation than proximity, so a pair could otherwise be interacting "
                "without being near, which no consumer could make sense of"
            )
        if not 0.0 <= self.reach_confidence <= 1.0:
            raise ConfigError("spatial.reach_confidence must be in [0, 1]")
        if self.max_entities < 2:
            raise ConfigError("spatial.max_entities must be >= 2 for a pair to exist")


@dataclass(slots=True)
class _PairState:
    """What has been true of one pair of entities over time."""

    separations: deque[tuple[float, float]]
    near_since: float | None = None
    close_since: float | None = None
    interacting_since: float | None = None
    reach_seen: bool = False


class SpatialAnalyzer:
    """Assigns zones and derives relations, frame by frame."""

    def __init__(
        self, zones: tuple[Zone, ...] = (), params: SpatialParams | None = None
    ) -> None:
        self._zones = tuple(zones)
        self._params = params or SpatialParams()
        # (track_id, zone_name) -> (entered_at, last_inside_at)
        self._zone_state: dict[tuple[int, str], tuple[float, float]] = {}
        self._pairs: dict[tuple[int, int], _PairState] = {}

    @property
    def zones(self) -> tuple[Zone, ...]:
        return self._zones

    @property
    def params(self) -> SpatialParams:
        return self._params

    @property
    def tracked_pairs(self) -> int:
        return len(self._pairs)

    def reset(self) -> None:
        self._zone_state.clear()
        self._pairs.clear()

    # -- zones ------------------------------------------------------------

    def assign_zones(
        self, tracks: tuple[Track, ...], frame_size: tuple[int, int], now: float
    ) -> dict[int, tuple[ZoneOccupancy, ...]]:
        """Which zones each entity is in, plus any it has just left."""
        hold = self._params.zone_event_hold_s
        assigned: dict[int, list[ZoneOccupancy]] = {t.track_id: [] for t in tracks}
        live = {t.track_id for t in tracks}

        for track in tracks:
            point = track.box.bottom_center
            for zone in self._zones:
                key = (track.track_id, zone.name)
                inside = zone.contains(point, frame_size)
                record = self._zone_state.get(key)

                if inside:
                    if record is None:
                        self._zone_state[key] = (now, now)
                        event: ZoneEvent | None = ZoneEvent.ENTERED
                        dwell = 0.0
                    else:
                        entered_at, _ = record
                        self._zone_state[key] = (entered_at, now)
                        dwell = now - entered_at
                        event = ZoneEvent.ENTERED if dwell <= hold else None
                    assigned[track.track_id].append(
                        ZoneOccupancy(zone.name, zone.kind, dwell, event)
                    )
                elif record is not None:
                    entered_at, last_inside = record
                    if now - last_inside <= hold:
                        assigned[track.track_id].append(
                            ZoneOccupancy(
                                zone.name,
                                zone.kind,
                                last_inside - entered_at,
                                ZoneEvent.EXITED,
                            )
                        )
                    else:
                        del self._zone_state[key]

        # Entities the tracker retired take their zone records with them.
        for key in [k for k in self._zone_state if k[0] not in live]:
            del self._zone_state[key]

        return {track_id: tuple(zones) for track_id, zones in assigned.items()}

    # -- relations --------------------------------------------------------

    def relations(
        self,
        tracks: tuple[Track, ...],
        poses: dict[int, Pose],
        motion: dict[int, MotionState],
        now: float,
        elapsed: float,
    ) -> tuple[list[RelationObservation], int]:
        """Every relation holding this frame, and how many entities were paired."""
        considered = sorted(tracks, key=lambda t: t.box.area, reverse=True)[
            : self._params.max_entities
        ]
        live = {track.track_id for track in considered}
        found: list[RelationObservation] = []

        for index, first in enumerate(considered):
            for second in considered[index + 1 :]:
                key = tuple(sorted((first.track_id, second.track_id)))
                state = self._pairs.get(key)
                if state is None:
                    state = _PairState(separations=deque(maxlen=self._params.history))
                    self._pairs[key] = state
                found.extend(self._pair(first, second, state, poses, motion, now))

        # Pairs where either entity has gone are dropped whole.
        for key in [k for k in self._pairs if k[0] not in live or k[1] not in live]:
            del self._pairs[key]

        return found, len(considered)

    def _pair(
        self,
        first: Track,
        second: Track,
        state: _PairState,
        poses: dict[int, Pose],
        motion: dict[int, MotionState],
        now: float,
    ) -> list[RelationObservation]:
        separation = ground_distance(first, second)
        state.separations.append((now, separation))
        found: list[RelationObservation] = []
        params = self._params

        # Proximity, with a dead band so a pair at the boundary does not flicker.
        threshold = params.near_distance
        if state.near_since is not None:
            threshold += params.near_hysteresis
        if separation <= threshold:
            if state.near_since is None:
                state.near_since = now
            margin = 1.0 - min(1.0, separation / max(params.near_distance, 1e-6))
            found.append(
                _observation(
                    Relation.NEAR,
                    first,
                    second,
                    separation,
                    confidence=max(0.15, margin),
                    duration_s=now - state.near_since,
                    evidence=f"{separation:.2f} heights apart on the ground plane",
                )
            )
        else:
            state.near_since = None

        rate = self._closing_rate(state, now)
        if rate is not None and abs(rate) >= params.approach_rate:
            relation = Relation.APPROACHING if rate > 0 else Relation.RECEDING
            found.append(
                _observation(
                    relation,
                    first,
                    second,
                    separation,
                    confidence=min(1.0, abs(rate) / (params.approach_rate * 3.0)),
                    duration_s=params.approach_window_s,
                    evidence=f"separation changing {abs(rate):.2f} heights/s",
                )
            )

        interaction = self._interaction(
            first, second, state, poses, motion, now, separation
        )
        if interaction is not None:
            found.append(interaction)
        return found

    def _closing_rate(self, state: _PairState, now: float) -> float | None:
        """Heights per second the pair is closing; negative means separating.

        Measured across the window's endpoints rather than as a fitted slope:
        the signal is short, the endpoints are what a person would read off a
        graph of it, and a least-squares fit would imply a precision that
        detector box jitter does not support.
        """
        window = [(t, d) for t, d in state.separations if now - t <= self._params.approach_window_s]
        if len(window) < 2:
            return None
        span = window[-1][0] - window[0][0]
        if span < self._params.approach_window_s * 0.5:
            return None
        return (window[0][1] - window[-1][1]) / span

    def _interaction(
        self,
        first: Track,
        second: Track,
        state: _PairState,
        poses: dict[int, Pose],
        motion: dict[int, MotionState],
        now: float,
        separation: float,
    ) -> RelationObservation | None:
        """A person reaching for an object, or stopped beside one.

        Duration alone does not separate lingering from passing, and the
        harness measured exactly how badly. Walking past a static object at
        180 px/s produced nothing, but the same path at 45 px/s - an amble -
        produced **49 frames of false interaction**, because a slow enough
        walk-past satisfies any sustain threshold. Raising ``interact_s``
        only moves the speed at which it breaks.

        The discriminator that actually exists is motion state, which Phase 4
        already computes with hysteresis: someone lingering is STATIONARY,
        someone passing is MOVING. So proximity-only interaction now requires
        the person to have stopped. A reach - a wrist landmark inside the
        object's box - still counts on its own, because taking something while
        walking is real and the landmark is direct evidence rather than an
        inference from two rectangles.
        """
        person, thing = _person_and_object(first, second)
        if person is None or thing is None:
            state.close_since = None
            state.interacting_since = None
            state.reach_seen = False
            return None

        reaching = _wrist_inside(poses.get(person.track_id), thing, self._params.reach_confidence)
        stopped = motion.get(person.track_id) is MotionState.STATIONARY
        if not reaching and not stopped:
            # Moving, or motion unknown because state is not running. Without
            # either signal, proximity alone is not enough to claim contact.
            state.close_since = None
            state.interacting_since = None
            state.reach_seen = False
            return None
        if separation > self._params.interact_distance and not reaching:
            state.close_since = None
            state.interacting_since = None
            state.reach_seen = False
            return None

        if state.close_since is None:
            state.close_since = now
        state.reach_seen = state.reach_seen or reaching
        held = now - state.close_since
        if held < self._params.interact_s:
            # Still could be someone walking past. Nothing is claimed yet.
            return None

        if state.interacting_since is None:
            state.interacting_since = now

        if state.reach_seen:
            confidence, evidence = 0.85, (
                f"wrist inside the object box, sustained {held:.1f}s"
            )
        else:
            # Deliberately capped low. Two boxes close together in a flat image
            # is consistent with a person three metres behind the object, and
            # nothing in a single view rules that out.
            confidence, evidence = 0.4, (
                f"stationary {separation:.2f} heights away for {held:.1f}s, no reach "
                "observed (2-D proximity only)"
            )
        return RelationObservation(
            relation=Relation.INTERACTING,
            subject_id=person.entity_id,
            object_id=thing.entity_id,
            subject_track=person.track_id,
            object_track=thing.track_id,
            distance=separation,
            confidence=confidence,
            duration_s=now - state.interacting_since,
            evidence=evidence,
        )


def ground_distance(first: Track, second: Track) -> float:
    """Separation of two ground points, in mean entity heights.

    Normalising by the mean of the two heights rather than by either one keeps
    the measure symmetric: a person beside a mug should not be a different
    distance from the mug than the mug is from them.
    """
    ax, ay = first.box.bottom_center
    bx, by = second.box.bottom_center
    scale = max(1.0, (first.box.height + second.box.height) / 2.0)
    return math.dist((ax, ay), (bx, by)) / scale


def _person_and_object(first: Track, second: Track) -> tuple[Track | None, Track | None]:
    """Order a pair as (person, thing), or (None, None) if it is not one.

    Two people are not an interaction here - that is proximity, which is
    reported separately. Claiming "person A is interacting with person B" from
    geometry alone would be a much larger inference than the signal supports.
    """
    first_is_person = first.label.lower() == "person"
    second_is_person = second.label.lower() == "person"
    if first_is_person == second_is_person:
        return None, None
    return (first, second) if first_is_person else (second, first)


def _wrist_inside(pose: Pose | None, thing: Track, min_confidence: float) -> bool:
    """Whether either wrist landmark falls inside the object's box."""
    if pose is None:
        return False
    x1, y1, x2, y2 = thing.box.xyxy
    for index in (LEFT_WRIST, RIGHT_WRIST):
        wrist = pose.keypoint(index)
        if wrist is None or wrist.confidence < min_confidence:
            continue
        if x1 <= wrist.x <= x2 and y1 <= wrist.y <= y2:
            return True
    return False


def _observation(
    relation: Relation,
    first: Track,
    second: Track,
    distance: float,
    *,
    confidence: float,
    duration_s: float,
    evidence: str,
) -> RelationObservation:
    """Build a symmetric relation with a stable subject/object order."""
    subject, other = (first, second) if first.track_id <= second.track_id else (second, first)
    return RelationObservation(
        relation=relation,
        subject_id=subject.entity_id,
        object_id=other.entity_id,
        subject_track=subject.track_id,
        object_track=other.track_id,
        distance=distance,
        confidence=max(0.0, min(1.0, confidence)),
        duration_s=duration_s,
        evidence=evidence,
    )


def entity_spatial(track: Track, zones: tuple[ZoneOccupancy, ...]) -> EntitySpatial:
    return EntitySpatial(
        track_id=track.track_id,
        entity_id=track.entity_id,
        label=track.label,
        zones=zones,
        ground_point=track.box.bottom_center,
    )
