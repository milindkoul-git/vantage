"""Non-maximum suppression.

Shared by every adapter, because suppression is a property of overlapping boxes
rather than of any model family.

Implemented in NumPy rather than delegating to ``cv2.dnn.NMSBoxes`` so that the
class-aware semantics are explicit and testable: suppression happens *within*
a class, never across classes. A person standing in front of a car must not
suppress the car, which is exactly what a class-agnostic NMS would do.
"""

from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over a single class.

    Args:
        boxes: ``(N, 4)`` array of ``x1, y1, x2, y2``.
        scores: ``(N,)`` confidences.
        iou_threshold: boxes overlapping a kept box by more than this are dropped.

    Returns:
        Indices of kept boxes, ordered by descending score.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]

        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

        union = areas[best] + areas[rest] - inter
        # Degenerate zero-area boxes would divide by zero; treat them as non-overlapping.
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Class-aware NMS: suppression happens only between boxes of the same class.

    Uses the standard coordinate-offset trick - each class is shifted into its
    own region of an imaginary plane, so a single NMS pass can never compare
    boxes across classes. One pass instead of one per class present.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    # Offset must exceed the largest coordinate so classes cannot possibly overlap.
    span = float(boxes.max()) + 1.0 if boxes.size else 1.0
    offsets = class_ids.astype(np.float64)[:, None] * span
    shifted = boxes.astype(np.float64) + offsets

    return nms(shifted, scores, iou_threshold)
