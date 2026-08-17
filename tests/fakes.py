"""Scriptable test doubles.

:class:`FakeSource` implements the three :class:`~vantage.ingestion.base.FrameSource`
hooks and nothing else, which is exactly what makes it useful: if the pipeline
works against it, the pipeline genuinely depends only on the interface.

:class:`FakeBackend` does the same job for perception: the whole detection
engine - preprocessing, decoding, NMS, class filtering, timing - is exercised
with no model file and no inference runtime installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vantage.core.errors import SourceExhausted
from vantage.core.frame import Frame
from vantage.ingestion.base import FrameSource, SourceInfo, SourceKind
from vantage.perception.backends.base import BackendInfo, InferenceBackend


class FakeSource(FrameSource):
    """Returns frames, or raises, according to a scripted list.

    Each script entry is either an ``int`` (emit a frame filled with that value)
    or an ``Exception`` (raise it). An empty script means the source is exhausted.
    """

    def __init__(
        self,
        script: list[Any] | None = None,
        source_id: str = "fake",
        *,
        is_live: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(source_id=source_id, uri="fake://", **kwargs)
        self.script = list(script or [])
        self.opens = 0
        self.closes = 0
        self.open_error: Exception | None = None
        self._is_live = is_live

    def _open_impl(self) -> SourceInfo:
        self.opens += 1
        if self.open_error is not None:
            raise self.open_error
        return SourceInfo(
            source_id=self.source_id,
            kind=SourceKind.CAMERA if self._is_live else SourceKind.FILE,
            uri=self.uri,
            width=8,
            height=6,
            backend="fake",
            is_live=self._is_live,
        )

    def _read_impl(self) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        if not self.script:
            raise SourceExhausted("script finished")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return np.full((6, 8, 3), item % 256, dtype=np.uint8), None, {"scripted": item}

    def _close_impl(self) -> None:
        self.closes += 1


# -- perception ---------------------------------------------------------

YOLOX_GRID_416 = 3549
"""Predictions a 416x416 YOLOX head emits: 52^2 + 26^2 + 13^2."""


def yolox_prediction(
    class_id: int = 0,
    objectness: float = 0.9,
    class_score: float = 0.9,
    width: float = 16.0,
    height: float = 16.0,
    num_classes: int = 80,
    populate_all: bool = False,
    row: int = 0,
) -> np.ndarray:
    """Build a raw YOLOX output tensor with known contents.

    Values are in the *undecoded* form the exported graph produces: box centres
    are offsets within a grid cell and sizes are logarithmic, so the adapter's
    grid decoding is genuinely exercised rather than bypassed.
    """
    raw = np.zeros((1, YOLOX_GRID_416, 5 + num_classes), dtype=np.float32)
    stride = 8.0  # the first grid section

    rows = range(YOLOX_GRID_416) if populate_all else [row]
    for index in rows:
        raw[0, index, 0] = 0.5  # centre offset within the cell
        raw[0, index, 1] = 0.5
        raw[0, index, 2] = np.log(width / stride)
        raw[0, index, 3] = np.log(height / stride)
        raw[0, index, 4] = objectness
        raw[0, index, 5 + class_id] = class_score
    return raw


class FakeBackend(InferenceBackend):
    """Returns a canned output tensor, counting calls."""

    def __init__(self, output: np.ndarray | None = None) -> None:
        self._output = output if output is not None else yolox_prediction()
        self.calls = 0
        self.closed = False
        self.last_input: np.ndarray | None = None

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="fake", device="none", version="0", input_name="images", precision="fp32"
        )

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        self.calls += 1
        self.last_input = tensor
        return [self._output]

    def close(self) -> None:
        self.closed = True


def make_engine(
    *,
    keep_classes: list[str] | None = None,
    confidence: float = 0.3,
    output: np.ndarray | None = None,
):
    """A detection engine over a fake backend, plus a frame to feed it.

    The canned output contains one person and one car, so class filtering and
    per-class NMS both have something real to act on.
    """
    from vantage.perception.adapters.yolox import YoloxAdapter
    from vantage.perception.engine import DetectionEngine
    from vantage.perception.labels import COCO_80

    if output is None:
        output = yolox_prediction(class_id=0, objectness=0.9, class_score=0.9)
        # A car in a different grid cell, far enough away to survive NMS.
        car = yolox_prediction(class_id=2, objectness=0.9, class_score=0.8)
        output[0, 2000] = car[0, 0]

    adapter = YoloxAdapter(input_size=(416, 416), labels=COCO_80)
    backend = FakeBackend(output)
    engine = DetectionEngine(
        adapter=adapter,
        backend=backend,
        model_name="fake-model",
        license="test",
        confidence=confidence,
        keep_classes=keep_classes,
    )
    frame = Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        index=7,
        source_id="cam0",
        capture_monotonic=1.0,
        capture_wall=1_700_000_000.0,
    )
    return engine, frame
