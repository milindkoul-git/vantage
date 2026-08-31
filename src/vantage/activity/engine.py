"""The activity engine: state and pose in, activities out.

Owns three things the recogniser should not have to: the clock, the pairing of
entities to their poses, and pruning.

Time is accumulated from the tracker's own elapsed values rather than read from
a wall or monotonic clock. That is not fussiness - it is what makes a recorded
source replay identically, and what keeps ``detection.interval`` and dropped
frames from silently stretching every duration the rules measure. A ten-second
loiter has to mean ten seconds of *footage*, whatever the machine was doing.
"""

from __future__ import annotations

from vantage.activity.base import Recognizer
from vantage.activity.contracts import (
    Activity,
    ActivityObservation,
    ActivityResult,
    EntityActivity,
)
from vantage.activity.hoi import HOIFusionEngine
from vantage.activity.recognizer import ActivityParams, RuleRecognizer
from vantage.core.logging import get_logger
from vantage.perception.contracts import Detection
from vantage.pose.contracts import PoseResult
from vantage.state.contracts import StateResult

log = get_logger(__name__)


class ActivityEngine:
    """Runs a :class:`~vantage.activity.base.Recognizer` over each frame."""

    def __init__(self, recognizer: Recognizer | None = None) -> None:
        self._recognizer: Recognizer = recognizer or RuleRecognizer()
        self._hoi = HOIFusionEngine()
        self._elapsed = 0.0

    @property
    def recognizer(self) -> Recognizer:
        return self._recognizer

    @property
    def elapsed_s(self) -> float:
        """Footage time seen so far."""
        return self._elapsed

    def update(
        self,
        state: StateResult,
        pose: PoseResult | None = None,
        detections: list[Detection] | tuple[Detection, ...] | None = None,
    ) -> ActivityResult:
        """Advance every entity by ``state.elapsed_s`` and report activities."""
        self._elapsed += max(0.0, state.elapsed_s)

        # Poses are keyed by track, and an entity without one is the normal
        # case rather than an error: a car has no posture, and a person beyond
        # pose.max_persons has no skeleton this frame.
        poses = pose.by_track() if pose is not None else {}

        entities: list[EntityActivity] = []
        for entity_state in state:
            p_pose = poses.get(entity_state.track_id)
            act = self._recognizer.observe(entity_state, p_pose, self._elapsed)
            # Check Human-Object Interactions if detections available
            if detections and entity_state.label == "person" and p_pose is not None:
                hoi_events = self._hoi.analyze(
                    person_box=p_pose.box,
                    pose=p_pose,
                    all_detections=detections,
                )
                if hoi_events:
                    extra_obs = list(act.observations)
                    for h in hoi_events:
                        if hasattr(Activity, h.verb.upper()):
                            act_enum = getattr(Activity, h.verb.upper())
                            extra_obs.append(
                                ActivityObservation(
                                    activity=act_enum,
                                    confidence=h.confidence,
                                    duration_s=0.0,
                                    evidence=h.evidence,
                                )
                            )
                    act = EntityActivity(
                        track_id=act.track_id,
                        entity_id=act.entity_id,
                        label=act.label,
                        observations=tuple(extra_obs),
                    )
            entities.append(act)

        self._recognizer.forget({entity_state.track_id for entity_state in state})

        return ActivityResult(
            entities=tuple(entities),
            source_id=state.source_id,
            frame_index=state.frame_index,
            capture_wall=state.capture_wall,
            elapsed_s=state.elapsed_s,
            pose_available=pose is not None,
            metadata={"elapsed_total_s": round(self._elapsed, 2)},
        )

    def reset(self) -> None:
        self._recognizer.reset()
        self._elapsed = 0.0


def build_activity_engine(config=None) -> ActivityEngine:
    """Construct from an :class:`~vantage.config.schema.ActivityConfig`."""
    if config is None:
        return ActivityEngine()
    if getattr(config, "mode", "rules") == "learned":
        from vantage.activity.learned import LearnedActionClassifier

        return ActivityEngine(
            LearnedActionClassifier(transient_hold_s=getattr(config, "transient_hold_s", 1.5))
        )
    return ActivityEngine(
        RuleRecognizer(
            ActivityParams(
                walking_speed=config.walking_speed,
                running_speed=config.running_speed,
                sustain_s=config.sustain_s,
                loiter_s=config.loiter_s,
                transition_window_s=config.transition_window_s,
                fall_window_s=config.fall_window_s,
                transient_hold_s=config.transient_hold_s,
                posture_window_s=config.posture_window_s,
                min_posture_confidence=config.min_posture_confidence,
                min_keypoint_confidence=config.min_keypoint_confidence,
                history=config.history,
            )
        )
    )
