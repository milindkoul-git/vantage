"""Motion state with hysteresis, dwell timing and path length.

Two thresholds, not one
-----------------------
A single speed threshold makes any entity hovering near it flap between states
several times a second, and every flap resets the dwell timer - which destroys
the one measurement that makes state worth computing. So the transitions are
asymmetric: it takes :attr:`StateParams.moving_above` to start moving and a
lower :attr:`StateParams.stationary_below` to stop, with a dead band between
where whatever state currently holds simply continues.

A minimum hold on top
---------------------
Hysteresis alone still admits a fast flicker if the estimate jumps clean across
the dead band, which detector jitter on a large box does. So a change must also
survive :attr:`StateParams.min_state_s` before it is published. The cost is that
genuine transitions are reported late by up to that delay; the benefit is that
"stationary for 40 seconds" means it, and a later phase can build an alert on it.

Where the speed comes from
--------------------------
The tracker's Kalman filter, not frame-to-frame box differences. It already
estimates velocity from irregular timesteps - which is exactly what
``detection.interval`` and dropped frames produce - and it is already smoothed.
Differencing raw boxes here would rebuild a worse version of a filter the
platform already runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vantage.core.errors import ConfigError
from vantage.state.contracts import EntityState, MotionState, StateResult
from vantage.tracking.contracts import Track, TrackingResult


@dataclass(frozen=True, slots=True)
class StateParams:
    """Thresholds for the motion state machine, in heights per second."""

    moving_above: float = 0.15
    """Speed at which an entity is called moving.

    A person walking is roughly 0.8 heights per second, a slow shuffle 0.3. The
    threshold sits well below both so that slow genuine motion is not called
    stillness, and well above the residual jitter of a stationary box, which
    measures under 0.05 on this pipeline.
    """

    stationary_below: float = 0.08
    min_state_s: float = 0.5
    min_age_s: float = 0.3
    """How long an entity must have existed before any state is claimed. The
    filter's velocity estimate is near-meaningless on the first observations,
    when it is still dominated by its initial covariance."""

    def __post_init__(self) -> None:
        if self.stationary_below > self.moving_above:
            raise ConfigError(
                f"state.stationary_below ({self.stationary_below}) must not exceed "
                f"state.moving_above ({self.moving_above}); with the thresholds "
                "inverted there is no dead band and hysteresis does nothing"
            )
        for name in ("moving_above", "stationary_below", "min_state_s", "min_age_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"state.{name} must be >= 0")


class _EntityHistory:
    """Mutable per-entity accumulator. Never leaves this module."""

    __slots__ = (
        "age_s",
        "distance",
        "dwell_s",
        "last_center",
        "last_height",
        "motion",
        "pending",
        "pending_s",
        "smoothed_speed",
    )

    def __init__(self) -> None:
        self.motion = MotionState.UNKNOWN
        self.dwell_s = 0.0
        self.age_s = 0.0
        self.distance = 0.0
        self.last_center: tuple[float, float] | None = None
        self.last_height: float | None = None
        self.pending: MotionState | None = None
        self.pending_s = 0.0
        self.smoothed_speed = 0.0


class StateEstimator:
    """Computes motion state, speed, dwell, bearing and distance per entity."""

    def __init__(self, params: StateParams | None = None) -> None:
        self._params = params or StateParams()
        self._history: dict[int, _EntityHistory] = {}

    @property
    def params(self) -> StateParams:
        return self._params

    @property
    def tracked(self) -> int:
        return len(self._history)

    def update(self, tracking: TrackingResult) -> StateResult:
        """Advance every track's state by ``tracking.elapsed_s``."""
        elapsed = max(0.0, tracking.elapsed_s)
        states: list[EntityState] = []
        seen: set[int] = set()

        for track in tracking.tracks:
            seen.add(track.track_id)
            history = self._history.get(track.track_id)
            if history is None:
                history = _EntityHistory()
                self._history[track.track_id] = history
            states.append(self._advance(track, history, elapsed))

        # Entities the tracker has retired are dropped here rather than kept
        # "just in case". Phase 3 shipped an unbounded set of seen ids by
        # accident; a long-running camera makes any such structure a leak.
        for track_id in self._history.keys() - seen:
            del self._history[track_id]

        return StateResult(
            states=tuple(states),
            source_id=tracking.source_id,
            frame_index=tracking.frame_index,
            capture_wall=tracking.capture_wall,
            elapsed_s=elapsed,
            metadata={"entities": len(states)},
        )

    def _advance(self, track: Track, history: _EntityHistory, elapsed: float) -> EntityState:
        history.age_s += elapsed
        history.dwell_s += elapsed

        height = max(track.box.height, 1.0)
        vx, vy = track.velocity
        trans_speed = math.hypot(vx, vy) / height

        # Depth perspective scale expansion for walking towards/away from camera
        depth_speed = 0.0
        if history.last_height is not None and elapsed > 0:
            dh_dt = abs(height - history.last_height) / elapsed
            depth_speed = 1.2 * (dh_dt / height)
        history.last_height = height

        raw_speed = max(trans_speed, depth_speed)
        # Apply exponential moving average to filter bounding box regression jitter
        if history.age_s > elapsed and history.smoothed_speed > 0:
            history.smoothed_speed = 0.65 * history.smoothed_speed + 0.35 * raw_speed
        else:
            history.smoothed_speed = raw_speed
        speed = history.smoothed_speed

        centre = track.center
        step = (
            math.dist(centre, history.last_center) / height
            if history.last_center is not None
            else 0.0
        )
        history.last_center = centre

        target = self._target_state(speed, history.motion)
        if history.age_s < self._params.min_age_s:
            target = MotionState.UNKNOWN

        if target is not history.motion:
            if history.pending is target:
                history.pending_s += elapsed
            else:
                history.pending = target
                history.pending_s = 0.0
            if history.pending_s >= self._params.min_state_s:
                history.motion = target
                history.dwell_s = 0.0
                history.pending = None
                history.pending_s = 0.0
        else:
            history.pending = None
            history.pending_s = 0.0

        # Path length accumulates only while the entity is MOVING, and the
        # ordering matters: the state machine decides first, then distance
        # follows its verdict.
        #
        # The first attempt here discarded any single step below a fixed
        # fraction of a height, to stop a stationary box's sub-pixel wobble
        # accumulating into kilometres of imaginary travel over a long session.
        # Measured on a real clip, that silently broke the feature it was meant
        # to protect: a person crossing the frame at 2 px per frame against a
        # 343 px box produces steps of 0.006 heights, every one of them under
        # the floor, so 120 px of genuine travel recorded as **exactly zero**.
        # Gating on MOVING rejects the same jitter - a stationary entity
        # contributes nothing at all - without also rejecting slow motion,
        # because the hysteresis above has already made that judgement properly.
        if history.motion is MotionState.MOVING:
            history.distance += step

        bearing = None
        if history.motion is MotionState.MOVING and (vx or vy):
            # Clockwise from up, with y increasing downwards in image
            # coordinates: atan2(x, -y) rather than the usual atan2(y, x).
            bearing = math.degrees(math.atan2(vx, -vy)) % 360.0

        return EntityState(
            track_id=track.track_id,
            entity_id=track.entity_id,
            label=track.label,
            motion=history.motion,
            speed=speed,
            dwell_s=history.dwell_s,
            bearing_deg=bearing,
            distance=history.distance,
            age_s=history.age_s,
            observed=track.time_since_update == 0,
        )

    def _target_state(self, speed: float, current: MotionState) -> MotionState:
        """Which state this speed argues for, given where we already are."""
        if speed >= self._params.moving_above:
            return MotionState.MOVING
        if speed <= self._params.stationary_below:
            return MotionState.STATIONARY
        # Inside the dead band nothing changes; an entity that has never had a
        # state yet stays UNKNOWN rather than being assigned one arbitrarily.
        return current

    def reset(self) -> None:
        self._history.clear()
