"""The activity engine: state and pose in, activities out.

Owns four things the recogniser should not have to: the clock, the pairing of
entities to their poses, pruning, and which entities are even eligible.

Eligibility is not a detail. Measured on five street clips, this engine used to
report that 73% of everything it saw was walking or running - and the majority
of that was about cars, potted plants, traffic lights and handbags, because it
ran the recogniser over every tracked class. A further fifth came from coasting
boxes: predictions for objects the detector had stopped seeing, whose drift the
state estimator dutifully measured as motion. `potted plant_2 is running` was a
real event in a real store.

Two gates fix it, and both are about what the words mean. "Walking", "running",
"sitting down" and "falling" are things people do, so only entities whose label
says person are considered. And a coasting box is a guess about where something
went, not an observation of it, so nothing is asserted from one.

Time is accumulated from the tracker's own elapsed values rather than read from
a wall or monotonic clock. That is not fussiness - it is what makes a recorded
source replay identically, and what keeps ``detection.interval`` and dropped
frames from silently stretching every duration the rules measure. A ten-second
loiter has to mean ten seconds of *footage*, whatever the machine was doing.
"""

from __future__ import annotations

from collections.abc import Sequence

from vantage.activity.base import Recognizer
from vantage.activity.contracts import (
    Activity,
    ActivityObservation,
    ActivityResult,
    EntityActivity,
)
from vantage.activity.hoi import HOIFusionEngine
from vantage.activity.recognizer import ActivityParams, RuleRecognizer
from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger
from vantage.perception.contracts import Detection
from vantage.pose.contracts import PoseResult
from vantage.state.contracts import StateResult

log = get_logger(__name__)


class ActivityEngine:
    """Runs a :class:`~vantage.activity.base.Recognizer` over each frame."""

    def __init__(
        self,
        recognizer: Recognizer | None = None,
        labels: Sequence[str] = ("person",),
    ) -> None:
        self._recognizer: Recognizer = recognizer or RuleRecognizer()
        self._hoi = HOIFusionEngine()
        self._elapsed = 0.0
        self._labels = frozenset(label.strip().lower() for label in labels if label.strip())
        if not self._labels:
            raise ConfigError("activity.labels cannot be empty; it decides who has activities")

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
            if entity_state.label.lower() not in self._labels:
                continue
            if not entity_state.observed:
                # A coasting entity's box is a prediction. Its apparent motion is
                # the predictor's drift, and reporting an activity from it is
                # reporting on something nobody currently sees. The pose engine
                # already refuses these; so does this.
                continue
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

        # Forgetting is keyed on everything the tracker still holds, not on what
        # was eligible this frame: an entity skipped for one coasting frame must
        # keep the history that makes its dwell and its transitions meaningful.
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
            LearnedActionClassifier(transient_hold_s=getattr(config, "transient_hold_s", 1.5)),
            labels=config.labels,
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
        ),
        labels=config.labels,
    )
