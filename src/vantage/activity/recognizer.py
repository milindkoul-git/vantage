"""Rule-based temporal recognition over the signals Phases 3 and 4 produce.

Why rules, decided on evidence rather than taste
------------------------------------------------
A learned skeleton-action model was the obvious first choice and was surveyed
before this was written. Three things ruled it out, none of them a matter of
preference:

* **No permissively licensed export with real provenance exists.** OpenMMLab
  publishes PoseC3D and ST-GCN as PyTorch checkpoints but ships no ONNX SDK for
  them, and the hub's ``st-gcn`` results are unrelated models - traffic
  forecasting, weather, sign language.
* **The video classifiers that do exist are the wrong shape.** VideoMAE and
  friends label a *frame*, not an entity, which throws away the identity the
  whole platform is built around.
* **Their vocabularies are wrong.** Kinetics-400 offers ``abseiling``,
  ``zumba`` and ``shredding paper``; NTU is lab-recorded daily living. Neither
  contains ``loitering``.

So the recogniser is rules over measured signals, every one of which states its
grounds. :class:`~vantage.activity.base.Recognizer` is a Protocol precisely so
this can be swapped for a learned model when one is worth having - the engine,
the buffers and the contracts do not change.

The two things that make temporal rules work
--------------------------------------------
**Stable posture, not raw posture.** Per-frame posture flickers, and a rule
watching raw transitions would fire ``sitting_down`` several times a second.
Transitions are detected between *stable* postures - a posture that has held a
majority of a short window - and nothing else.

**Sustain windows on continuous rules.** ``running`` requires the speed to have
held, not to have touched a threshold once. A single fast frame from a detector
box that jumped is not a person running.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from vantage.core.errors import ConfigError
from vantage.activity.contracts import Activity, ActivityObservation, EntityActivity
from vantage.pose.contracts import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    Pose,
    Posture,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from vantage.state.contracts import EntityState, MotionState

_MAJORITY = 0.6
"""Fraction of a sustain window a continuous rule must hold on.

Not 1.0: a single frame where a detector box jumped would otherwise cancel a
rule that is plainly true, and the state machine upstream has already applied
hysteresis to the underlying signal."""

_POSTURE_SUPERMAJORITY = 0.7
"""Fraction of the posture window a posture needs to become the stable one.

A bare majority is not enough, and the failure is specific rather than
theoretical. Under a perfect frame-by-frame alternation the window holds an
even split, which a strict majority correctly refuses - but only while the
window has an even number of samples. Every frame where it holds an odd number,
one posture leads by one, the stable posture flips, and a transition fires. Run
against 120 alternating frames that produced **101 spurious transitions**. A
supermajority tops out around 53% under alternation and never promotes."""

_WINDOW_COVERAGE = 0.9
"""How much of the sustain period the window's own samples must span.

Guards the case where an entity is seen, disappears, and is seen again much
later: the buffer then holds plenty of history but the recent window holds a
single sample, which trivially "supports" any rule at 100%. One frame at
6 h/s after a gap was reported as running."""


def _support(window: list["_Sample"], threshold: float) -> float:
    """Fraction of a window at or above a speed threshold."""
    return sum(1 for sample in window if sample.speed >= threshold) / len(window)


@dataclass(frozen=True, slots=True)
class ActivityParams:
    """Thresholds for the rules. Speeds are in entity heights per second."""

    walking_speed: float = 0.15
    """Above this, a moving entity is walking.

    Deliberately equal to ``state.moving_above`` rather than a little higher.
    Set above it and a band opens where the state machine calls an entity
    MOVING while no locomotion rule fires, so the same entity is reported as
    moving *and* idle in the same frame - observed on a real clip at 0.175 h/s.
    The state machine has already applied hysteresis and a minimum hold to
    decide that motion is genuine; second-guessing it with a higher threshold
    here only produces a contradiction."""

    running_speed: float = 1.30
    """Above this, running rather than walking. A walk measures 0.6-0.9 heights
    per second and a run 1.5-2.5, so the boundary sits in the gap between them
    rather than through either."""

    sustain_s: float = 0.4
    """How long a continuous rule must hold before it is reported. Short enough
    to feel immediate, long enough that one bad box cannot trigger it."""

    loiter_s: float = 20.0
    """Stationary for this long becomes loitering rather than merely standing.
    A duration, not a judgement: what it means is a policy question for the
    event rules of a later phase."""

    transition_window_s: float = 2.5
    """Longest gap between two stable postures still counted as one transition.
    Wider than a real sit-down or stand-up takes, so that a posture briefly lost
    to occlusion mid-movement does not break the pair."""

    fall_window_s: float = 1.2
    """A change from upright to lying faster than this is a fall.

    A person choosing to lie down takes longer, and the separation is the entire
    basis of the distinction - there is nothing else in the signal that tells
    the two apart. See the limitations in the module documentation of
    :mod:`vantage.activity`.
    """

    transient_hold_s: float = 1.5
    """How long a moment-in-time activity keeps being reported after it happened,
    so a consumer sampling slowly cannot miss it entirely."""

    posture_window_s: float = 0.6
    min_posture_confidence: float = 0.25
    min_keypoint_confidence: float = 0.3
    history: int = 240
    """Samples retained per entity. At 30 fps this is eight seconds, which is
    more than the longest window any rule uses."""

    def __post_init__(self) -> None:
        if self.walking_speed <= 0:
            raise ConfigError("activity.walking_speed must be positive")
        if self.running_speed <= self.walking_speed:
            raise ConfigError(
                f"activity.running_speed ({self.running_speed}) must exceed "
                f"activity.walking_speed ({self.walking_speed}), or running can "
                "never be distinguished from walking"
            )
        for name in (
            "sustain_s",
            "loiter_s",
            "transition_window_s",
            "fall_window_s",
            "transient_hold_s",
            "posture_window_s",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"activity.{name} must be >= 0")
        if self.fall_window_s > self.transition_window_s:
            raise ConfigError(
                f"activity.fall_window_s ({self.fall_window_s}) must not exceed "
                f"activity.transition_window_s ({self.transition_window_s}); a fall "
                "is a fast case of a posture transition, so it cannot be detectable "
                "over a longer span than transitions are paired over"
            )
        if self.history < 2:
            raise ConfigError("activity.history must be >= 2")


@dataclass(slots=True)
class _Sample:
    """One frame of an entity, reduced to what the rules need.

    Features are extracted on the way in rather than keeping poses around: a
    buffer of raw skeletons for every entity would be the largest allocation in
    the platform, and none of the rules ever look at a joint twice.
    """

    t: float
    speed: float
    motion: MotionState
    dwell_s: float
    posture: Posture
    posture_confidence: float
    arm_raised: bool
    arm_confidence: float


class _EntityBuffer:
    """Per-entity history plus the transitions found in it."""

    __slots__ = ("samples", "stable_posture", "stable_since", "events", "first_seen")

    def __init__(self, now: float, history: int) -> None:
        self.samples: deque[_Sample] = deque(maxlen=history)
        self.stable_posture: Posture | None = None
        self.stable_since: float = now
        # (activity, when, confidence, evidence) for transient events.
        self.events: list[tuple[Activity, float, float, str]] = []
        self.first_seen = now


class RuleRecognizer:
    """Recognises activities from motion state, posture and keypoints."""

    def __init__(self, params: ActivityParams | None = None) -> None:
        self._params = params or ActivityParams()
        self._buffers: dict[int, _EntityBuffer] = {}

    @property
    def params(self) -> ActivityParams:
        return self._params

    @property
    def tracked(self) -> int:
        return len(self._buffers)

    def reset(self) -> None:
        self._buffers.clear()

    def forget(self, track_ids: set[int]) -> None:
        """Drop entities the tracker has retired.

        Called by the engine on every step. Phase 3 shipped an unbounded set of
        seen ids by accident once; on a camera that runs for weeks any structure
        keyed by track id is a leak unless something prunes it.
        """
        for track_id in self._buffers.keys() - track_ids:
            del self._buffers[track_id]

    def observe(
        self, state: EntityState, pose: Pose | None, now: float
    ) -> EntityActivity:
        """Record one frame for an entity and report what it is doing."""
        buffer = self._buffers.get(state.track_id)
        if buffer is None:
            buffer = _EntityBuffer(now, self._params.history)
            self._buffers[state.track_id] = buffer

        arm_raised, arm_confidence = _arm_raised(pose, self._params.min_keypoint_confidence)
        buffer.samples.append(
            _Sample(
                t=now,
                speed=state.speed,
                motion=state.motion,
                dwell_s=state.dwell_s,
                posture=pose.posture if pose else Posture.UNKNOWN,
                posture_confidence=pose.posture_confidence if pose else 0.0,
                arm_raised=arm_raised,
                arm_confidence=arm_confidence,
            )
        )
        self._update_stable_posture(buffer, now)

        observations = [
            observation
            for observation in (
                self._locomotion(buffer, state, now),
                self._loitering(state),
                self._arm(buffer, now),
            )
            if observation is not None
        ]
        observations.extend(self._transients(buffer, now))
        if not observations:
            observations.append(
                ActivityObservation(
                    activity=Activity.IDLE,
                    confidence=1.0,
                    duration_s=now - buffer.first_seen,
                    evidence="present, nothing else recognised",
                )
            )

        return EntityActivity(
            track_id=state.track_id,
            entity_id=state.entity_id,
            label=state.label,
            observations=tuple(observations),
        )

    # -- posture stability and transitions -------------------------------

    def _update_stable_posture(self, buffer: _EntityBuffer, now: float) -> None:
        """Promote the majority posture of a short window to the stable one.

        Requires a supermajority rather than a bare majority - see
        :data:`_POSTURE_SUPERMAJORITY` for the alternating-flicker case that
        makes the difference between 101 spurious transitions and none.
        """
        window = [
            sample
            for sample in buffer.samples
            if now - sample.t <= self._params.posture_window_s
            and sample.posture is not Posture.UNKNOWN
            and sample.posture_confidence >= self._params.min_posture_confidence
        ]
        if not window:
            return

        posture, count = Counter(sample.posture for sample in window).most_common(1)[0]
        if count < len(window) * _POSTURE_SUPERMAJORITY:
            return
        if posture is buffer.stable_posture:
            return

        previous = buffer.stable_posture
        buffer.stable_posture = posture
        buffer.stable_since = now
        if previous is None:
            return

        # How long the *transition* took, measured from the last frame that
        # still showed the old posture - not how long the old posture had held.
        #
        # The difference is the whole rule. Timing from when the previous
        # posture became stable means a person who stands for ten seconds and
        # then drops measures a "ten second transition", sails past
        # fall_window_s, and no fall is ever reported. Someone who had only
        # just stood up would be caught and someone who had been standing a
        # while would not, which is precisely backwards.
        last_previous = next(
            (sample.t for sample in reversed(buffer.samples) if sample.posture is previous),
            None,
        )
        if last_previous is None:
            return
        gap = now - last_previous
        if gap > self._params.transition_window_s:
            return

        confidence = min(
            1.0,
            sum(s.posture_confidence for s in window) / len(window),
        )
        event = _classify_transition(previous, posture, gap, self._params.fall_window_s)
        if event is not None:
            activity, evidence = event
            buffer.events.append((activity, now, confidence, evidence))

    def _transients(self, buffer: _EntityBuffer, now: float) -> list[ActivityObservation]:
        """Recent one-off events, held briefly so slow consumers see them."""
        buffer.events = [
            event
            for event in buffer.events
            if now - event[1] <= self._params.transient_hold_s
        ]
        return [
            ActivityObservation(
                activity=activity,
                confidence=confidence,
                duration_s=now - when,
                evidence=evidence,
            )
            for activity, when, confidence, evidence in buffer.events
        ]

    # -- continuous rules -------------------------------------------------

    def _locomotion(
        self, buffer: _EntityBuffer, state: EntityState, now: float
    ) -> ActivityObservation | None:
        """Walking or running, from sustained speed."""
        if state.motion is not MotionState.MOVING:
            return None
        if state.speed < self._params.walking_speed:
            return None

        window = self._sustained_window(buffer, now)
        if window is None:
            return None

        running = state.speed >= self._params.running_speed and _support(
            window, self._params.running_speed * 0.8
        ) >= _MAJORITY
        activity = Activity.RUNNING if running else Activity.WALKING
        threshold = self._params.running_speed if running else self._params.walking_speed
        support = _support(window, threshold)
        if support < _MAJORITY:
            return None

        margin = min(1.0, abs(state.speed - threshold) / max(threshold, 1e-6))
        # Sustained over a window is worth more than fast for an instant, so
        # both terms are in the score rather than speed alone.
        return ActivityObservation(
            activity=activity,
            confidence=max(0.0, min(1.0, 0.5 * margin + 0.5 * support)),
            duration_s=state.dwell_s,
            evidence=(
                f"{state.speed:.2f} h/s, held on {support:.0%} of the last "
                f"{self._params.sustain_s:.1f}s"
            ),
        )

    def _sustained_window(
        self, buffer: _EntityBuffer, now: float
    ) -> list[_Sample] | None:
        """The last ``sustain_s`` of samples, once that much history exists.

        The history check reads the *oldest sample in the buffer*, not the
        oldest sample in the window. Those are not the same thing and the
        difference is total: the window is selected by ``now - t <= sustain_s``,
        so its span can never exceed ``sustain_s``, and comparing that span
        against ``sustain_s`` is a test that floating point loses roughly always.
        Written that way first, every continuous rule scored **zero** on the
        harness while the event rules scored perfectly.
        """
        window = [s for s in buffer.samples if now - s.t <= self._params.sustain_s]
        if len(window) < 2:
            return None
        if now - window[0].t < self._params.sustain_s * _WINDOW_COVERAGE:
            return None
        return window

    def _loitering(self, state: EntityState) -> ActivityObservation | None:
        if state.motion is not MotionState.STATIONARY:
            return None
        if state.dwell_s < self._params.loiter_s:
            return None
        # Confidence saturates at twice the threshold: past that the answer is
        # certain and a number that keeps climbing would be meaningless.
        excess = (state.dwell_s - self._params.loiter_s) / max(self._params.loiter_s, 1e-6)
        return ActivityObservation(
            activity=Activity.LOITERING,
            confidence=max(0.0, min(1.0, 0.5 + 0.5 * excess)),
            duration_s=state.dwell_s,
            evidence=f"stationary for {state.dwell_s:.0f}s",
        )

    def _arm(self, buffer: _EntityBuffer, now: float) -> ActivityObservation | None:
        window = self._sustained_window(buffer, now)
        # Every frame, not a majority: an arm that flickers up and down is not
        # being held up, and this is the one rule where the underlying signal is
        # a clean geometric comparison rather than a noisy estimate.
        if window is None or not all(s.arm_raised for s in window):
            return None
        held = now - window[0].t
        confidence = sum(s.arm_confidence for s in window) / len(window)
        return ActivityObservation(
            activity=Activity.ARM_RAISED,
            confidence=max(0.0, min(1.0, confidence)),
            duration_s=held,
            evidence=f"wrist above shoulder for {held:.1f}s",
        )


def _classify_transition(
    previous: Posture, current: Posture, gap: float, fall_window_s: float
) -> tuple[Activity, str] | None:
    """Name a change between two stable postures, or return ``None``."""
    if current is Posture.LYING and previous in (Posture.STANDING, Posture.CROUCHING):
        if gap <= fall_window_s:
            return (
                Activity.FALLING,
                f"{previous.value} to lying in {gap:.1f}s",
            )
        # Slower than a fall: a deliberate lie-down. Reported as neither, rather
        # than as a low-confidence fall, because a hedged fall alert is worse
        # than none - it trains whoever reads it to ignore the real one.
        return None
    if previous is Posture.STANDING and current is Posture.SITTING:
        return Activity.SITTING_DOWN, f"standing to sitting in {gap:.1f}s"
    if previous is Posture.SITTING and current is Posture.STANDING:
        return Activity.STANDING_UP, f"sitting to standing in {gap:.1f}s"
    return None


def _arm_raised(pose: Pose | None, min_confidence: float) -> tuple[bool, float]:
    """Whether either wrist is above its shoulder, and how sure that is.

    Above means a smaller ``y``: image coordinates grow downwards. Both joints
    must be observed - an unseen wrist defaults to the frame origin, which sits
    above every shoulder and would read as a permanently raised arm.
    """
    if pose is None:
        return False, 0.0

    best = 0.0
    for wrist_index, shoulder_index in ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)):
        wrist = pose.keypoint(wrist_index)
        shoulder = pose.keypoint(shoulder_index)
        if wrist is None or shoulder is None:
            continue
        if wrist.confidence < min_confidence or shoulder.confidence < min_confidence:
            continue
        if wrist.y < shoulder.y:
            best = max(best, min(wrist.confidence, shoulder.confidence))
    return best > 0.0, best
