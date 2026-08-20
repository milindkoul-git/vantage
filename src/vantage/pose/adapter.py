"""RTMPose adapter: person box in, 17 landmarks out.

Top-down, which is the whole reason this is cheap. A bottom-up estimator finds
every person in the full frame and then has to group limbs to bodies; a top-down
one is handed a box and only has to locate joints inside it. Phase 3 already
produces boxes with stable identity, so the expensive half of the problem is
solved before this file runs. The cost is linear in people rather than fixed,
which is the right trade for a scene with a handful of them and the reason
:class:`~vantage.pose.engine.PoseEngine` carries an explicit budget.

Every constant here was read out of the exported ``pipeline.json`` that ships in
the official OpenMMLab archive, not inferred from what similar models usually
do. That file specifies ``padding=1.25``, ``image_size=[192, 256]``, ImageNet
normalisation with ``to_rgb=true``, and ``simcc_split_ratio=2.0``. Getting any
of them wrong is silent: the model still runs and still returns 17 plausible
points, they are just in the wrong places.

SimCC decoding
--------------
RTMPose does not regress coordinates and does not emit heatmaps. It classifies
each axis independently into bins at ``split_ratio`` times the input resolution
- 384 bins across 192 pixels of width, 512 across 256 of height - so a joint is
the argmax of its x histogram paired with the argmax of its y histogram. Two
1-D distributions instead of one 2-D map is what makes the head so small, and it
means decoding is an argmax rather than a soft-argmax over a spatial grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vantage.core.errors import ConfigError
from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import FACE_KEYPOINTS, KEYPOINT_NAMES, Keypoint

BBOX_PADDING = 1.25
"""Person boxes are expanded 25% before cropping.

Detector boxes clip to the visible silhouette, so a joint on the boundary - a
wrist against the hip, an ankle at the bottom edge - lands on the very edge of
the crop where the model has least context. The padding is part of how the
network was trained, not a tunable.
"""

SIMCC_SPLIT_RATIO = 2.0
"""Bins per pixel along each axis."""

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
"""In 0-255 units, applied to RGB. This export normalises outside the graph."""


@dataclass(frozen=True, slots=True)
class PreparedCrop:
    """One person, warped to the model input, plus what undoes the warp."""

    tensor: np.ndarray
    center: np.ndarray
    """Box centre in original-frame pixels."""

    scale: np.ndarray
    """Width and height of the source region, after padding and aspect
    correction. With :attr:`center` this is the complete inverse transform,
    which is why no matrix is carried around."""

    box: BoundingBox


class RTMPoseAdapter:
    """Preprocessing and SimCC decoding for RTMPose ONNX exports.

    Not a :class:`~vantage.perception.adapters.base.ModelAdapter`. That
    interface takes a whole frame and returns detections; this takes a frame
    *and a box* and returns landmarks. Forcing one interface to cover both would
    mean an ``extra`` dict carrying the box and a return type that is sometimes
    one thing and sometimes another - a shared name hiding two contracts.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        labels: tuple[str, ...] = KEYPOINT_NAMES,
        include_face_keypoints: bool = True,
    ) -> None:
        if len(labels) != len(KEYPOINT_NAMES):
            raise ConfigError(
                f"RTMPose adapter expects {len(KEYPOINT_NAMES)} keypoint names, "
                f"got {len(labels)}"
            )
        self._input_size = input_size
        self._labels = tuple(labels)
        self._include_face = include_face_keypoints

    @property
    def input_size(self) -> tuple[int, int]:
        """``(height, width)``."""
        return self._input_size

    @property
    def labels(self) -> tuple[str, ...]:
        """Keypoint names actually emitted, face landmarks included or not."""
        if self._include_face:
            return self._labels
        return tuple(n for i, n in enumerate(self._labels) if i not in FACE_KEYPOINTS)

    @property
    def include_face_keypoints(self) -> bool:
        return self._include_face

    def static_input_shapes(self) -> dict[str, list[int]]:
        """Pinned to batch 1.

        The export declares a dynamic batch, and Phase 3.5 established what
        OpenVINO does with an unpinned dimension: it compiles a dynamic graph
        that runs roughly twice as slow. Batching several people into one pass
        would save a little per-person overhead, but it needs a fixed batch size
        chosen ahead of time and padding whenever fewer people are present, and
        at 3.5 ms per person the overhead being amortised is not the cost that
        matters.
        """
        height, width = self._input_size
        return {"input": [1, 3, height, width]}

    # -- preprocessing --------------------------------------------------

    def preprocess(self, image: np.ndarray, box: BoundingBox) -> PreparedCrop:
        """Warp the padded, aspect-corrected person region to the model input."""
        height, width = self._input_size
        center, scale = _center_scale(box, width / height)

        # For an upright crop the affine is a uniform scale plus a translation,
        # so it is written directly. MMPose builds the same matrix from three
        # corresponding points to support rotation, which this pipeline never
        # requests; the three-point construction reduces to exactly this when
        # the angle is zero.
        factor = width / scale[0]
        matrix = np.array(
            [
                [factor, 0.0, width / 2.0 - factor * center[0]],
                [0.0, factor, height / 2.0 - factor * center[1]],
            ],
            dtype=np.float32,
        )
        crop = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return PreparedCrop(
            tensor=np.ascontiguousarray(normalised.transpose(2, 0, 1)[None]),
            center=center,
            scale=scale,
            box=box,
        )

    # -- decoding -------------------------------------------------------

    def postprocess(self, outputs: list[np.ndarray], prepared: PreparedCrop) -> list[Keypoint]:
        """Decode ``(simcc_x, simcc_y)`` into original-frame landmarks."""
        if len(outputs) < 2:
            raise ValueError(
                f"RTMPose expects two outputs (simcc_x, simcc_y), got {len(outputs)}"
            )
        simcc_x, simcc_y = outputs[0], outputs[1]
        if simcc_x.ndim != 3 or simcc_y.ndim != 3:
            raise ValueError(
                f"unexpected RTMPose output shapes: simcc_x {simcc_x.shape}, "
                f"simcc_y {simcc_y.shape}"
            )
        x_bins, y_bins = simcc_x[0], simcc_y[0]
        if x_bins.shape[0] != len(KEYPOINT_NAMES):
            raise ValueError(
                f"model predicts {x_bins.shape[0]} keypoints, but this adapter decodes "
                f"the {len(KEYPOINT_NAMES)}-point COCO layout"
            )

        locations = np.stack([x_bins.argmax(axis=1), y_bins.argmax(axis=1)], axis=-1).astype(
            np.float32
        )

        # The weaker of the two axes, because a joint is only located as well as
        # its worst axis: a confident column paired with a flat row is a
        # horizontal position with no vertical one.
        scores = np.minimum(x_bins.max(axis=1), y_bins.max(axis=1))

        height, width = self._input_size
        points = locations / SIMCC_SPLIT_RATIO
        points = (
            points / np.array([width, height], dtype=np.float32) * prepared.scale
            + prepared.center
            - prepared.scale / 2.0
        )

        keypoints: list[Keypoint] = []
        for index in range(len(KEYPOINT_NAMES)):
            if not self._include_face and index in FACE_KEYPOINTS:
                continue
            keypoints.append(
                Keypoint(
                    x=float(points[index, 0]),
                    y=float(points[index, 1]),
                    # The SimCC head is trained with a KL objective and is not
                    # normalised at inference, so its peaks sit near [0, 1] but
                    # are not bounded by it - measured on real frames, they run
                    # from -0.72 to 1.06. Clipping is therefore a genuine
                    # clamp of a near-probability rather than a rescale that
                    # would misrepresent the spacing between scores.
                    confidence=float(np.clip(scores[index], 0.0, 1.0)),
                )
            )
        return keypoints


def _center_scale(box: BoundingBox, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Padded box centre and extent, widened or heightened to match ``aspect``.

    Aspect correction matters more than it looks: warping a wide box into a tall
    input without it would squash the person horizontally, and a model trained
    on undistorted crops reads a squashed one as a differently shaped body.
    """
    x1, y1, x2, y2 = box.xyxy
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    width = max(x2 - x1, 1.0) * BBOX_PADDING
    height = max(y2 - y1, 1.0) * BBOX_PADDING
    if width > height * aspect:
        height = width / aspect
    else:
        width = height * aspect
    return center, np.array([width, height], dtype=np.float32)
