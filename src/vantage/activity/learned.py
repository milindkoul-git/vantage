"""Deterministic Spatio-Temporal Behavior Recognition & Model Seams.

Architecture:
- TemporalBehaviorRecognizer: High-level recognizer satisfying the Recognizer protocol.
- FeatureBasedTemporalRecognizer: Evaluates deterministic kinematic trajectories and skeletal dynamics.
- OptionalModelTemporalRecognizer: Model runtime seam for future ONNX graph/temporal neural models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vantage.activity.base import Recognizer
from vantage.activity.contracts import Activity, ActivityObservation, EntityActivity
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Pose,
    Posture,
)
from vantage.scene.window import EntityTemporalWindow
from vantage.state.contracts import EntityState, MotionState


@dataclass
class _TrackBehaviorState:
    """Internal state and observation window for one tracked entity."""

    window: EntityTemporalWindow = field(
        default_factory=lambda: EntityTemporalWindow(max_samples=60, max_span_s=5.0)
    )
    transient_active: dict[Activity, tuple[float, str]] = field(
        default_factory=dict
    )  # activity -> (expire_time, evidence)
    crouch_start_time: float | None = None
    last_bearing: float | None = None
    last_bearing_time: float | None = None


class FeatureBasedTemporalRecognizer:
    """Evaluates deterministic spatio-temporal features and skeletal dynamics over sliding windows."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.50,
        transient_hold_s: float = 2.0,
        pacing_window_s: float = 4.0,
        crouch_dwell_threshold_s: float = 8.0,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._transient_hold_s = transient_hold_s
        self._pacing_window_s = pacing_window_s
        self._crouch_dwell_threshold_s = crouch_dwell_threshold_s
        self._history: dict[int, _TrackBehaviorState] = {}

    def observe(self, state: EntityState, pose: Pose | None, now: float) -> EntityActivity:
        """Process one entity frame, extract temporal features, and recognize behaviors."""
        track_id = state.track_id
        bstate = self._history.setdefault(track_id, _TrackBehaviorState())

        # 1. Update temporal observation window
        posture = Posture.UNKNOWN
        if pose is not None and pose.posture is not Posture.UNKNOWN:
            posture = pose.posture
        elif hasattr(state, "posture") and state.posture:
            st_post = state.posture
            if isinstance(st_post, str):
                posture = (
                    Posture(st_post)
                    if st_post in Posture._value2member_map_
                    else Posture.UNKNOWN
                )
            else:
                posture = st_post

        box = (
            pose.box
            if pose is not None
            else getattr(state, "box", BoundingBox(0.0, 0.0, 100.0, 100.0))
        )

        bstate.window.add(
            timestamp=now,
            box=box,
            frame_width=1920,
            frame_height=1080,
            speed=state.speed,
            bearing_deg=getattr(state, "bearing_deg", None),
            posture=posture,
            pose=pose,
            zones=tuple(state.zones) if hasattr(state, "zones") else (),
        )

        # 2. Extract deterministic kinematics and skeletal dynamics
        kin = bstate.window.extract_kinematics()
        skel = bstate.window.extract_skeletal()

        observations: list[ActivityObservation] = []

        # 3. Evaluate Sudden Collapse (rapid downward hip acceleration + prone end-state)
        if skel.hip_drop_rate >= 0.25 and skel.is_prone:
            evidence = f"vertical hip descent {skel.hip_drop_rate:.2f} h/s into prone state"
            bstate.transient_active[Activity.SUDDEN_COLLAPSE] = (
                now + self._transient_hold_s,
                evidence,
            )
            bstate.transient_active[Activity.FALLING] = (now + self._transient_hold_s, evidence)

        # 4. Evaluate Abrupt Direction Reversal (bearing flip > 140 deg at speed > 0.3 h/s)
        curr_bearing = getattr(state, "bearing_deg", None)
        if (
            curr_bearing is not None
            and bstate.last_bearing is not None
            and bstate.last_bearing_time is not None
        ):
            dt = max(0.01, now - bstate.last_bearing_time)
            if dt < 1.0 and state.speed > 0.30:
                angle_diff = abs(curr_bearing - bstate.last_bearing)
                if angle_diff > 180.0:
                    angle_diff = 360.0 - angle_diff
                if angle_diff > 135.0:
                    evidence = f"bearing flip {angle_diff:.0f} deg in {dt:.2f}s at {state.speed:.2f} h/s"
                    bstate.transient_active[Activity.ABRUPT_DIRECTION_REVERSAL] = (
                        now + self._transient_hold_s,
                        evidence,
                    )

        if curr_bearing is not None:
            bstate.last_bearing = curr_bearing
            bstate.last_bearing_time = now

        # 5. Evaluate Erratic Pacing (high directional entropy in bounded area)
        if (
            kin.duration_s >= self._pacing_window_s
            and kin.directional_entropy >= 0.40
            and kin.pacing_ratio <= 0.60
        ):
            observations.append(
                ActivityObservation(
                    activity=Activity.ERRATIC_PACING,
                    confidence=min(1.0, 0.70 + kin.directional_entropy * 0.3),
                    duration_s=kin.duration_s,
                    evidence=f"directional entropy {kin.directional_entropy:.2f} >= 0.40, pacing ratio {kin.pacing_ratio:.2f}",
                )
            )

        # 6. Evaluate Crouching Dwell
        if posture is Posture.CROUCHING:
            if bstate.crouch_start_time is None:
                bstate.crouch_start_time = now
            crouch_duration = now - bstate.crouch_start_time
            if crouch_duration >= self._crouch_dwell_threshold_s:
                observations.append(
                    ActivityObservation(
                        activity=Activity.CROUCHING_DWELL,
                        confidence=0.90,
                        duration_s=crouch_duration,
                        evidence=f"crouching posture sustained {crouch_duration:.1f}s >= {self._crouch_dwell_threshold_s:.1f}s",
                    )
                )
        else:
            bstate.crouch_start_time = None

        # 7. Evaluate Erratic High-Energy Motion (acceleration spikes & speed variance)
        if kin.max_acceleration > 2.2 and kin.speed_variance > 0.30 and kin.mean_speed > 0.70:
            observations.append(
                ActivityObservation(
                    activity=Activity.ERRATIC_HIGH_ENERGY_MOTION,
                    confidence=min(1.0, 0.65 + kin.speed_variance * 0.35),
                    duration_s=kin.duration_s,
                    evidence=f"accel spike {kin.max_acceleration:.2f} h/s^2, speed variance {kin.speed_variance:.2f}",
                )
            )

        # 8. Evaluate Arm Raised
        if pose is not None:
            w_l = pose.keypoint(LEFT_WRIST)
            s_l = pose.keypoint(LEFT_SHOULDER)
            w_r = pose.keypoint(RIGHT_WRIST)
            s_r = pose.keypoint(RIGHT_SHOULDER)
            if (w_l and s_l and w_l.y < s_l.y and w_l.confidence > 0.2) or (
                w_r and s_r and w_r.y < s_r.y and w_r.confidence > 0.2
            ):
                observations.append(
                    ActivityObservation(
                        activity=Activity.ARM_RAISED,
                        confidence=0.85,
                        duration_s=state.dwell_s,
                        evidence="wrist landmark detected above shoulder level",
                    )
                )

        # 9. Append active transients
        for act, (exp_time, evidence) in list(bstate.transient_active.items()):
            if now < exp_time:
                observations.append(
                    ActivityObservation(
                        activity=act,
                        confidence=0.90,
                        duration_s=round(self._transient_hold_s - (exp_time - now), 2),
                        evidence=evidence,
                    )
                )
            else:
                del bstate.transient_active[act]

        # 9. Locomotion & Baseline Activities
        if state.motion is MotionState.MOVING:
            if kin.mean_speed > 1.30:
                observations.append(
                    ActivityObservation(
                        activity=Activity.RUNNING,
                        confidence=0.90,
                        duration_s=state.dwell_s,
                        evidence=f"speed {kin.mean_speed:.2f} h/s sustained over temporal window",
                    )
                )
            elif kin.mean_speed > 0.15:
                observations.append(
                    ActivityObservation(
                        activity=Activity.WALKING,
                        confidence=0.85,
                        duration_s=state.dwell_s,
                        evidence=f"speed {kin.mean_speed:.2f} h/s sustained over temporal window",
                    )
                )
        elif state.motion is MotionState.STATIONARY and state.dwell_s > 20.0:
            observations.append(
                ActivityObservation(
                    activity=Activity.LOITERING,
                    confidence=0.95,
                    duration_s=state.dwell_s,
                    evidence=f"stationary dwell {state.dwell_s:.1f}s > threshold 20.0s",
                )
            )

        if not observations:
            observations.append(
                ActivityObservation(
                    activity=Activity.IDLE,
                    confidence=1.0,
                    duration_s=state.dwell_s,
                    evidence="no active behavioral or locomotion pattern detected",
                )
            )

        return EntityActivity(
            track_id=state.track_id,
            entity_id=state.entity_id,
            label=state.label,
            observations=tuple(observations),
        )

    def forget(self, track_ids: set[int]) -> None:
        """Drop tracked entities that are no longer active."""
        for tid in list(self._history.keys()):
            if tid not in track_ids:
                del self._history[tid]

    def reset(self) -> None:
        """Discard all internal history."""
        self._history.clear()


class OptionalModelTemporalRecognizer:
    """Model runtime seam for future ONNX/neural spatio-temporal action classification models."""

    def __init__(self, model_path: str | None = None, *args: Any, **kwargs: Any) -> None:
        if model_path is None:
            raise NotImplementedError(
                "Optional ONNX neural model runtime requires a trained model checkpoint."
            )
        self.model_path = model_path

    def observe(self, state: EntityState, pose: Pose | None, now: float) -> EntityActivity:
        raise NotImplementedError(
            "Optional ONNX neural model runtime is reserved for future trained checkpoints."
        )

    def forget(self, track_ids: set[int]) -> None:
        pass

    def reset(self) -> None:
        pass


class TemporalBehaviorRecognizer:
    """Unified temporal behavior recognizer satisfying the :class:`Recognizer` protocol.

    Delegates to :class:`FeatureBasedTemporalRecognizer` by default and exposes
    a model runtime seam via :class:`OptionalModelTemporalRecognizer`.
    """

    def __init__(self, recognizer: Recognizer | None = None) -> None:
        self._delegate = recognizer or FeatureBasedTemporalRecognizer()

    def observe(self, state: EntityState, pose: Pose | None, now: float) -> EntityActivity:
        return self._delegate.observe(state, pose, now)

    def forget(self, track_ids: set[int]) -> None:
        self._delegate.forget(track_ids)

    def reset(self) -> None:
        self._delegate.reset()


# Backward compatibility aliases
LearnedActionClassifier = FeatureBasedTemporalRecognizer
LearnedTemporalRecognizer = OptionalModelTemporalRecognizer
