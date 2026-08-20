"""D-FINE adapter - a DETR-family detector, decoded very differently to YOLOX.

Why a second adapter family at all
-----------------------------------
YOLOX is trained on COCO, and COCO has 80 classes. That ceiling is not a
threshold or a confidence setting, it is the shape of the output tensor: there
is no neuron for an object the model was never trained on, so asking YOLOX for
a pen is asking a question it has no way to answer. D-FINE trained on
Objects365 has 365 classes, including the desk and office objects COCO omits
entirely - ``Pen/Pencil``, ``Marker``, ``Stapler``, ``Folder``, ``Calculator``,
``Notepaper``, ``Tape``.

The adapter seam existing already is what makes this additive: the engine, both
inference backends, the tracker, the overlay and the whole pipeline are
untouched. Only this file and a catalog entry are new.

Three decoding differences from YOLOX, all load-bearing
-------------------------------------------------------
1. **No grid, no anchors.** A DETR head emits a fixed set of *object queries*
   (300 here), each already a complete box prediction. There is nothing to
   decode against a stride grid.

2. **NMS is still needed, despite the theory.** Set prediction with Hungarian
   matching is supposed to teach the queries not to duplicate each other, and
   the DETR literature says suppression is unnecessary. That was written here
   as fact and then measured to be false: on a live frame at a 0.30 threshold
   this export produced six ``Person`` boxes for one person, two pairs of which
   overlapped at IoU 0.90 and 0.84. The duplicates are real, and downstream they
   are worse than cosmetic - Phase 3 would confirm each one as a separate track
   and invent people. So class-aware NMS runs, at a deliberately lenient default
   threshold that removes near-copies while leaving two genuinely overlapping
   objects alone.

3. **No letterbox.** The exported preprocessor resizes straight to 640x640
   without preserving aspect ratio (``do_pad: false``). Boxes come back
   normalised to ``[0, 1]``, so undoing it is a multiply by the original width
   and height independently - simpler than YOLOX's pad-and-scale, and a
   different code path rather than a special case of it.

Scores are **sigmoid, not softmax**: the head is trained with focal loss, so
each class is an independent probability and they do not sum to one.
"""

from __future__ import annotations

import cv2
import numpy as np

from vantage.perception.adapters.base import ModelAdapter, PreparedInput
from vantage.perception.contracts import BoundingBox, Detection
from vantage.perception.nms import batched_nms

BACKGROUND_CLASS = 0
"""The ``None`` slot every DETR head carries. Never a real detection."""


class DFineAdapter(ModelAdapter):
    """Preprocessing and decoding for D-FINE / DETR-style ONNX exports."""

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Resize to the model's input, scale to ``[0, 1]``, convert to CHW RGB.

        Note what is *absent*: no letterbox padding, and no ImageNet mean/std
        normalisation. Both were verified against the exported
        ``preprocessor_config.json`` (``do_pad: false``, ``do_normalize: false``,
        ``do_rescale: true``) rather than assumed from what similar models
        usually do - D-FINE folds the normalisation into the graph, and applying
        it here as well would quietly halve the detection quality.
        """
        height, width = self._input_size
        original_h, original_w = image.shape[:2]

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

        return PreparedInput(
            tensor=tensor,
            # Aspect ratio is not preserved, so a single scalar scale cannot
            # describe this transform. Boxes are normalised anyway, so the
            # original size is what postprocess actually needs.
            scale=1.0,
            pad=(0.0, 0.0),
            original_size=(original_w, original_h),
        )

    def postprocess(
        self,
        outputs: list[np.ndarray],
        prepared: PreparedInput,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        """Decode ``(logits, pred_boxes)`` into original-frame detections.

        ``iou_threshold`` gates the class-aware suppression described in the
        module docstring. It is applied *after* boxes are back in original-frame
        pixels, so the threshold means the same thing here as it does for YOLOX.
        """
        if len(outputs) < 2:
            raise ValueError(
                f"D-FINE expects two outputs (logits, pred_boxes), got {len(outputs)}"
            )

        logits, boxes = outputs[0], outputs[1]
        if logits.ndim != 3 or boxes.ndim != 3:
            raise ValueError(
                f"unexpected D-FINE output shapes: logits {logits.shape}, boxes {boxes.shape}"
            )

        scores = _sigmoid(logits[0])  # (queries, classes)
        candidates = boxes[0]  # (queries, 4) cxcywh, normalised

        # One class per query, rather than a top-k over the flattened
        # query x class grid. The flattened form is what the reference
        # postprocessor does and it lets a single box be reported as three
        # different things at once (SUV, Van and Car over identical corners,
        # observed on the test image). One object should be one Detection.
        class_ids = scores.argmax(axis=1)
        best = scores[np.arange(scores.shape[0]), class_ids]

        keep = (best >= confidence) & (class_ids != BACKGROUND_CLASS)
        if not keep.any():
            return []

        class_ids = class_ids[keep]
        best = best[keep]
        candidates = candidates[keep]

        order = np.argsort(-best)
        class_ids, best, candidates = class_ids[order], best[order], candidates[order]

        width, height = prepared.original_size
        cx, cy, bw, bh = candidates.T
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height

        # Suppress near-duplicate queries before truncating to max_detections,
        # so the cap counts distinct objects rather than copies of one.
        corners = np.stack([x1, y1, x2, y2], axis=1)
        kept = batched_nms(corners, best, class_ids, iou_threshold)[:max_detections]
        class_ids, best = class_ids[kept], best[kept]
        x1, y1, x2, y2 = x1[kept], y1[kept], x2[kept], y2[kept]

        detections: list[Detection] = []
        for index in range(len(kept)):
            box = BoundingBox(
                x1=float(x1[index]),
                y1=float(y1[index]),
                x2=float(x2[index]),
                y2=float(y2[index]),
            ).clipped(width, height)
            if box.width < 1.0 or box.height < 1.0:
                # A query that collapsed to nothing carries no information and
                # would fail downstream geometry; drop it here where the cause
                # is obvious.
                continue
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    box=box,
                    class_id=class_id,
                    label=self.label_for(class_id),
                    confidence=float(min(best[index], 1.0)),
                )
            )
        return detections


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic.

    The naive ``1 / (1 + exp(-x))`` overflows on the large negative logits a
    detection head produces in bulk - most of 300x366 scores are strongly
    negative - which floods the log with warnings and yields NaNs.
    """
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result
