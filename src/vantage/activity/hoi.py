"""Human-Object Interaction (HOI) & Contextual Scene Fusion.

Fuses human skeletal landmarks (wrists, shoulders, head) with detected objects
(phones, bottles, bags, vehicles, furniture) to derive high-level interaction verbs:
- <Person, talking_on_phone, CellPhone>
- <Person, holding, Bottle/Cup>
- <Person, carrying, Backpack/Handbag>
- <Person, riding, Bicycle/Motorcycle>
- <Person, seated_on, Chair/Bench>
- <Person, interacting_with, Object>
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from vantage.perception.contracts import BoundingBox, Detection
from vantage.pose.contracts import (
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    Pose,
)


def _box_iou(a: BoundingBox, b: BoundingBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = a.area + b.area - inter_area
    return inter_area / max(union_area, 1e-6)


@dataclass(frozen=True, slots=True)
class HOIInteraction:
    """A verified interaction between a person and a detected object."""

    verb: str
    target_class: str
    target_box: BoundingBox
    confidence: float
    evidence: str


class HOIFusionEngine:
    """Detects and validates human-object interactions in each video frame."""

    def __init__(
        self,
        *,
        hand_reach_threshold_ratio: float = 0.35,  # fraction of person box height
        min_interaction_confidence: float = 0.40,
    ) -> None:
        self._hand_reach_threshold_ratio = hand_reach_threshold_ratio
        self._min_confidence = min_interaction_confidence

    def analyze(
        self,
        person_box: BoundingBox,
        pose: Pose | None,
        all_detections: Sequence[Detection],
    ) -> list[HOIInteraction]:
        """Infer all active interactions for this person in the current frame."""
        interactions: list[HOIInteraction] = []
        p_box = person_box
        p_h = max(p_box.height, 1.0)
        reach_dist = self._hand_reach_threshold_ratio * p_h

        # Extract landmark coordinates
        l_wrist = pose.keypoint(LEFT_WRIST) if pose else None
        r_wrist = pose.keypoint(RIGHT_WRIST) if pose else None
        nose = pose.keypoint(NOSE) if pose else None

        wrists = [w for w in (l_wrist, r_wrist) if w and w.confidence > self._min_confidence]
        head_y = (
            nose.y
            if (nose and nose.confidence > self._min_confidence)
            else p_box.y1 + 0.15 * p_h
        )

        for det in all_detections:
            # Skip self (person)
            if det.label == "person" or det.confidence < self._min_confidence:
                continue

            obj_box = det.box
            obj_label = det.label.lower().replace(" ", "_")
            obj_center = obj_box.center

            # 1. Phone Interactions (Talking on Phone vs Holding Phone)
            if obj_label in ("cell_phone", "phone", "mobile", "mobile_phone"):
                # Distance to nearest wrist
                min_wrist_dist = min(
                    [math.dist((w.x, w.y), obj_center) for w in wrists],
                    default=math.dist(p_box.center, obj_center),
                )
                if min_wrist_dist <= reach_dist or p_box.contains(obj_center[0], obj_center[1]):
                    # Near head = talking on phone
                    if abs(obj_center[1] - head_y) <= 0.25 * p_h:
                        interactions.append(
                            HOIInteraction(
                                verb="talking_on_phone",
                                target_class=obj_label,
                                target_box=obj_box,
                                confidence=round(det.confidence * 0.95, 2),
                                evidence=f"phone at head height ({abs(obj_center[1] - head_y):.0f}px) held near ear",
                            )
                        )
                    else:
                        interactions.append(
                            HOIInteraction(
                                verb="holding_phone",
                                target_class=obj_label,
                                target_box=obj_box,
                                confidence=round(det.confidence * 0.90, 2),
                                evidence=f"phone within hand reach ({min_wrist_dist:.0f}px)",
                            )
                        )

            # 2. Drinking & Beverage Containers (Bottle / Cup)
            elif obj_label in ("bottle", "cup", "wine_glass", "can", "drink"):
                min_wrist_dist = min(
                    [math.dist((w.x, w.y), obj_center) for w in wrists],
                    default=math.dist(p_box.center, obj_center),
                )
                if min_wrist_dist <= reach_dist or p_box.contains(obj_center[0], obj_center[1]):
                    interactions.append(
                        HOIInteraction(
                            verb="holding_bottle",
                            target_class=obj_label,
                            target_box=obj_box,
                            confidence=round(det.confidence * 0.90, 2),
                            evidence=f"container held within hand reach ({min_wrist_dist:.0f}px)",
                        )
                    )

            # 3. Bags, Backpacks, Luggage (Carrying)
            elif obj_label in ("backpack", "handbag", "suitcase", "bag", "purse"):
                # Check torso overlap
                iou = _box_iou(p_box, obj_box)
                center_in_torso = (
                    p_box.x1 <= obj_center[0] <= p_box.x2
                    and p_box.y1 <= obj_center[1] <= p_box.y2
                )
                if iou > 0.15 or center_in_torso:
                    interactions.append(
                        HOIInteraction(
                            verb="carrying_baggage",
                            target_class=obj_label,
                            target_box=obj_box,
                            confidence=round(det.confidence * 0.92, 2),
                            evidence=f"bag overlapping torso (IoU: {iou:.2f})",
                        )
                    )

            # 4. Vehicles & Mobility (Riding Bicycle / Motorcycle)
            elif obj_label in ("bicycle", "motorcycle", "bike", "scooter"):
                iou = _box_iou(p_box, obj_box)
                # Significant overlap or feet/hips anchored to vehicle
                if iou > 0.25 or (
                    p_box.x1 < obj_center[0] < p_box.x2 and p_box.y2 >= obj_box.y1
                ):
                    interactions.append(
                        HOIInteraction(
                            verb="riding_vehicle",
                            target_class=obj_label,
                            target_box=obj_box,
                            confidence=round(det.confidence * 0.95, 2),
                            evidence=f"rider overlapping {obj_label} (IoU: {iou:.2f})",
                        )
                    )

            # 5. Seating Furniture (Seated on Chair / Bench / Couch)
            elif obj_label in ("chair", "couch", "bench", "sofa"):
                iou = _box_iou(p_box, obj_box)
                if iou > 0.20 or (
                    p_box.x1 <= obj_center[0] <= p_box.x2 and p_box.y2 > obj_box.y1
                ):
                    interactions.append(
                        HOIInteraction(
                            verb="seated_on_furniture",
                            target_class=obj_label,
                            target_box=obj_box,
                            confidence=round(det.confidence * 0.88, 2),
                            evidence=f"person positioned over {obj_label} (IoU: {iou:.2f})",
                        )
                    )

        return interactions
