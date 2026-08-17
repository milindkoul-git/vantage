"""YOLOX adapter.

YOLOX (Megvii, **Apache-2.0**) is the Phase 2 detector. The licence is the
reason: the obvious alternative, Ultralytics YOLOv8/v11, is AGPL-3.0, which
would oblige anyone shipping a product built on this platform to open-source it
or buy a commercial licence. For a platform meant to be production-viable that
is a constraint worth avoiding at the very first model choice.

Two conventions of the official export are load-bearing, and both were verified
empirically against the reference image before this file was written:

*Input is BGR in 0-255, not RGB in 0-1.* YOLOX folded normalisation into its
weights, so the tensor is raw pixel values. Our frames are already BGR, so
there is no colour conversion on the hot path at all.

*Output is undecoded.* The released ONNX graphs stop before grid decoding, so
``(1, N, 85)`` holds raw offsets that must have grid positions added and stride
multipliers applied. N is the sum of feature-map cells over strides 8, 16 and
32 - 3549 for a 416x416 input.

Letterboxing pads to the **top-left**, not centred, which makes undoing it a
single division by the scale factor.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from vantage.perception.adapters.base import ModelAdapter, PreparedInput
from vantage.perception.contracts import BoundingBox, Detection
from vantage.perception.nms import batched_nms

PAD_VALUE = 114
"""Grey used by YOLOX for letterbox padding, matching its training augmentation."""

STRIDES: tuple[int, ...] = (8, 16, 32)


class YoloxAdapter(ModelAdapter):
    """Preprocess/decode for the YOLOX family (nano, tiny, s, m, l, x)."""

    def __init__(
        self,
        input_size: tuple[int, int],
        labels: tuple[str, ...],
        strides: tuple[int, ...] = STRIDES,
    ) -> None:
        super().__init__(input_size=input_size, labels=labels)
        self._strides = strides

    # -- input ----------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        height, width = self._input_size
        source_h, source_w = image.shape[:2]

        scale = min(height / source_h, width / source_w)
        new_w, new_h = int(source_w * scale), int(source_h * scale)

        canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
        if new_w > 0 and new_h > 0:
            # INTER_LINEAR matches the reference implementation; INTER_AREA would
            # be sharper when downscaling but would shift the operating point
            # away from what the weights were validated against.
            canvas[:new_h, :new_w] = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        return PreparedInput(
            tensor=tensor,
            scale=scale,
            pad=(0.0, 0.0),  # top-left anchored
            original_size=(source_w, source_h),
        )

    # -- output ---------------------------------------------------------

    def postprocess(
        self,
        outputs: list[np.ndarray],
        prepared: PreparedInput,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        if not outputs:
            raise ValueError("YOLOX adapter received no output tensors")

        predictions = np.asarray(outputs[0], dtype=np.float32)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            raise ValueError(
                "unexpected YOLOX output shape "
                f"{np.asarray(outputs[0]).shape}; expected (1, N, 5 + num_classes)"
            )

        boxes_cxcywh, scores, class_ids = self._decode(predictions, confidence)
        if boxes_cxcywh.size == 0:
            return []

        # cxcywh -> xyxy in model-input space, then undo the letterbox. Padding is
        # top-left, so the scale factor alone maps back to original coordinates.
        half = boxes_cxcywh[:, 2:4] / 2.0
        boxes = np.concatenate(
            [boxes_cxcywh[:, :2] - half, boxes_cxcywh[:, :2] + half], axis=1
        )
        boxes /= max(prepared.scale, 1e-9)

        keep = batched_nms(boxes, scores, class_ids, iou_threshold)
        if keep.size == 0:
            return []
        keep = keep[:max_detections]

        width, height = prepared.original_size
        detections: list[Detection] = []
        for index in keep:
            x1, y1, x2, y2 = (float(v) for v in boxes[index])
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            if width and height:
                box = box.clipped(width, height)
            if box.width <= 0 or box.height <= 0:
                # Entirely outside the frame after clipping - a real artefact of
                # edge predictions, and nothing downstream can use a zero-area box.
                continue
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    box=box,
                    class_id=class_id,
                    label=self.label_for(class_id),
                    confidence=min(1.0, float(scores[index])),
                )
            )
        return detections

    def _decode(
        self, predictions: np.ndarray, confidence: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply grid offsets and stride scaling, then threshold."""
        grids, strides = _grid_cache(self._input_size, self._strides)
        if grids.shape[0] != predictions.shape[0]:
            raise ValueError(
                f"YOLOX output has {predictions.shape[0]} predictions but the "
                f"{self._input_size[0]}x{self._input_size[1]} grid expects "
                f"{grids.shape[0]}; the model's input size and the configured "
                "size disagree"
            )

        centres = (predictions[:, :2] + grids) * strides
        sizes = np.exp(np.clip(predictions[:, 2:4], -20.0, 20.0)) * strides
        boxes = np.concatenate([centres, sizes], axis=1)

        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        # Final confidence is objectness times class probability, as YOLOX's own
        # postprocessing defines it. Using either alone over-reports badly.
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]

        mask = scores >= confidence
        return boxes[mask], scores[mask], class_ids[mask]


@lru_cache(maxsize=8)
def _grid_cache(
    input_size: tuple[int, int], strides: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Grid offsets and per-cell strides for one input geometry.

    Cached because it is fixed for the life of a model and rebuilding it per
    frame would add pointless allocation to every inference.
    """
    height, width = input_size
    grid_parts: list[np.ndarray] = []
    stride_parts: list[np.ndarray] = []

    for stride in strides:
        rows, cols = height // stride, width // stride
        xv, yv = np.meshgrid(np.arange(cols), np.arange(rows))
        grid_parts.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        stride_parts.append(np.full((rows * cols, 1), stride, dtype=np.float32))

    grids = np.concatenate(grid_parts, axis=0).astype(np.float32)
    stride_map = np.concatenate(stride_parts, axis=0)
    return grids, stride_map
