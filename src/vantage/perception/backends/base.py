"""The backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """What a backend resolved to once loaded.

    Records what actually happened rather than what was requested: asking for
    the GPU and silently getting the CPU would invalidate every benchmark
    number, so the resolved device is reported and shown on the HUD.
    """

    name: str
    device: str
    version: str = "unknown"
    input_name: str = ""
    input_shape: tuple[int, ...] = ()
    precision: str = "fp32"
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.name} {self.version} on {self.device} ({self.precision})"


class InferenceBackend(ABC):
    """Executes a model graph. Knows nothing about detection."""

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """Resolved backend properties. Valid after construction."""

    @abstractmethod
    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Execute one forward pass and return the raw output tensors."""

    @abstractmethod
    def close(self) -> None:
        """Release the compiled model and any device resources."""

    def __enter__(self) -> "InferenceBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
