"""The adapter interface.

Splitting preprocessing/decoding from execution is what makes the ONNX Runtime
and OpenVINO backends interchangeable: both receive an identical input tensor
and return raw output tensors, and neither can influence what the detections
mean.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vantage.perception.contracts import Detection


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """A frame reshaped into what the model expects, plus how to undo it."""

    tensor: np.ndarray
    """Batched input tensor, typically ``(1, 3, H, W)`` float32."""

    scale: float
    """Resize factor applied to the original frame."""

    pad: tuple[float, float] = (0.0, 0.0)
    """``(left, top)`` padding added after resizing, in model-input pixels."""

    original_size: tuple[int, int] = (0, 0)
    """``(width, height)`` of the frame before preprocessing."""

    extra: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Converts frames to model input and model output to detections."""

    def __init__(self, input_size: tuple[int, int], labels: tuple[str, ...]) -> None:
        self._input_size = input_size
        self._labels = labels

    @property
    def input_size(self) -> tuple[int, int]:
        """``(height, width)`` the model expects."""
        return self._input_size

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Shape a BGR uint8 HxWx3 frame into the model's input tensor."""

    @abstractmethod
    def postprocess(
        self,
        outputs: list[np.ndarray],
        prepared: PreparedInput,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        """Decode raw outputs into detections in **original frame** coordinates."""

    def static_input_shapes(self) -> dict[str, list[int]] | None:
        """Fully static shapes for every graph input, or ``None`` if not needed.

        Single-input models do not need this: the backend can pin input 0 from
        :attr:`input_size` alone. Multi-input graphs do, because a text-
        conditioned model has several dynamic inputs and OpenVINO's GPU plugin
        refuses to compile until *all* of them are static - it fails with
        "to_shape was called on a dynamic shape", which names no input and is
        therefore unhelpful about which one is at fault.
        """
        return None

    def label_for(self, class_id: int) -> str:
        """Name for a class index, degrading gracefully rather than raising.

        A model whose head has more classes than the configured label set is a
        configuration error, but it should surface as a visibly odd label
        rather than as a crash mid-stream.
        """
        if 0 <= class_id < len(self._labels):
            return self._labels[class_id]
        return f"class_{class_id}"
