"""Face detection and embedding: YuNet plus SFace, through OpenCV's own wrappers.

Model choice, on the usual grounds
----------------------------------
ArcFace via InsightFace is the obvious answer and is **not usable here**. Its
model zoo states, in writing, that "ALL models are available for non-commercial
research purposes only". That fails the same licence gate that ruled out
YOLO-World (GPL-3.0) in Phase 3.5 and Ultralytics pose (AGPL-3.0) in Phase 4.

The OpenCV Zoo pair is permissive and comes from the project that maintains it:

* **YuNet** - face detection with five landmarks. MIT.
* **SFace** - 128-dimensional embedding. Apache-2.0.

Why OpenCV's wrappers rather than this project's own backends
--------------------------------------------------------------
Every other model here runs through :mod:`vantage.perception.backends`, which
would give GPU execution and one consistent path. These two do not, deliberately.

SFace expects a 112x112 crop aligned by a similarity transform from five facial
landmarks onto a canonical template. Getting that transform subtly wrong does not
raise anything - it produces embeddings that are merely *worse*, so every
similarity drifts and the threshold that was measured no longer means what it
did. ``cv2.FaceRecognizerSF.alignCrop`` is the reference implementation for these
exact weights, written by the people who trained them.

The cost is that both run on OpenCV's DNN backend rather than OpenVINO, so they
are CPU-only here: measured at 12 ms to detect and 29 ms to embed. That is
affordable precisely because identification runs per *track* rather than per
frame - see :mod:`vantage.identity.engine`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger
from vantage.identity.contracts import EMBEDDING_DIM

log = get_logger(__name__)

DEFAULT_DETECT_SIZE = (320, 320)
MIN_FACE_PIXELS = 40
"""Faces smaller than this are not embedded.

A 20-pixel face upscaled to 112x112 produces a confident-looking vector from
almost no information, and the resulting similarity is noise wearing a number.
Refusing is better than averaging that into a template.
"""


class FaceRecognizer:
    """Finds a face in an image region and turns it into a template."""

    def __init__(
        self,
        detector_path: str | Path,
        embedder_path: str | Path,
        *,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
    ) -> None:
        detector_path, embedder_path = Path(detector_path), Path(embedder_path)
        for path in (detector_path, embedder_path):
            if not path.is_file():
                raise ConfigError(
                    f"face model missing at {path}. Fetch it with: "
                    "vantage models pull yunet-face sface"
                )
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(detector_path),
                "",
                DEFAULT_DETECT_SIZE,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
            )
            self._embedder = cv2.FaceRecognizerSF.create(str(embedder_path), "")
        except cv2.error as exc:
            raise ConfigError(f"could not load the face models: {exc}") from exc
        self._size = DEFAULT_DETECT_SIZE

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Faces in ``image`` as YuNet rows, or ``None``.

        Each row is 15 numbers: box, five landmarks, score. The landmarks are
        the reason this detector is here rather than reusing the person
        detector - alignment needs them.
        """
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        if width < MIN_FACE_PIXELS or height < MIN_FACE_PIXELS:
            return None
        if (width, height) != self._size:
            # The detector caches an input size and silently mis-scales boxes if
            # it is fed something else.
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        try:
            _, faces = self._detector.detect(image)
        except cv2.error:
            return None
        return faces if faces is not None and len(faces) else None

    def embed(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        """Align one detected face and return its template."""
        _, _, width, height = face_row[:4]
        if width < MIN_FACE_PIXELS or height < MIN_FACE_PIXELS:
            return None
        try:
            aligned = self._embedder.alignCrop(image, face_row)
            template = self._embedder.feature(aligned)
        except cv2.error:
            return None
        vector = np.asarray(template, dtype=np.float32).reshape(-1)
        if vector.size != EMBEDDING_DIM:
            log.warning(
                "unexpected embedding size",
                extra={"vantage_fields": {"got": int(vector.size), "expected": EMBEDDING_DIM}},
            )
            return None
        return vector

    def largest_face(self, image: np.ndarray) -> np.ndarray | None:
        """The biggest face in the image, which for a person crop is theirs.

        A person box can contain a bystander's face at its edge. The largest is
        the one the box was drawn around, and picking by score instead would
        sometimes prefer a sharper face in the background.
        """
        faces = self.detect(image)
        if faces is None:
            return None
        return max(faces, key=lambda row: float(row[2]) * float(row[3]))

    def template_for(self, image: np.ndarray) -> np.ndarray | None:
        """Detect the main face and embed it, or ``None`` if there is not one."""
        face = self.largest_face(image)
        if face is None:
            return None
        return self.embed(image, face)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two templates.

    Computed here rather than with ``FaceRecognizerSF.match`` so that a stored
    template - a list of floats out of the database - can be compared without
    reconstructing a recogniser. The arithmetic is identical.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denominator)
