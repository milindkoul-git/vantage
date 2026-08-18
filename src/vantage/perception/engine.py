"""The detection engine.

Composes one :class:`ModelAdapter` with one :class:`InferenceBackend` and
produces :class:`DetectionResult`. This is the only perception type the rest of
the platform touches - the app calls :meth:`DetectionEngine.detect` and gets
structured records back.

The three stages are timed separately because they scale differently and are
fixed by different means: preprocessing is bound by resize cost (fix with a
smaller input), inference by the model and device (fix with a smaller model or
better backend), postprocessing by the number of candidate boxes (fix with a
higher confidence floor). A single "detection took 40 ms" number would hide
which lever to pull.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import ConfigError
from vantage.core.frame import Frame
from vantage.core.logging import get_logger
from vantage.perception.adapters import get_adapter
from vantage.perception.adapters.base import ModelAdapter
from vantage.perception.backends import create_backend
from vantage.perception.backends.base import InferenceBackend
from vantage.perception.catalog import ModelSpec, get_model_spec
from vantage.perception.contracts import Detection, DetectionResult
from vantage.perception.store import ModelStore

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """What the engine resolved to, for the HUD, logs and benchmark tables."""

    model: str
    backend: str
    device: str
    input_size: tuple[int, int]
    precision: str
    license: str
    num_classes: int

    def describe(self) -> str:
        height, width = self.input_size
        return (
            f"{self.model} ({width}x{height}, {self.num_classes} classes, {self.license}) "
            f"on {self.backend}/{self.device}"
        )


class DetectionEngine:
    """Runs a detector over frames."""

    def __init__(
        self,
        adapter: ModelAdapter,
        backend: InferenceBackend,
        *,
        model_name: str = "unknown",
        license: str = "unknown",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        keep_classes: Sequence[str] | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ConfigError(f"detection.confidence must be in (0, 1), got {confidence}")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ConfigError(f"detection.nms_iou must be in [0, 1], got {iou_threshold}")
        if max_detections < 1:
            raise ConfigError(f"detection.max_detections must be >= 1, got {max_detections}")

        self._adapter = adapter
        self._backend = backend
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._max_detections = max_detections
        self._clock = clock
        self._closed = False

        self._keep: frozenset[str] | None = None
        if keep_classes:
            self._keep = frozenset(name.strip().lower() for name in keep_classes if name.strip())
            unknown = self._keep - {label.lower() for label in adapter.labels}
            if unknown:
                raise ConfigError(
                    f"detection.classes contains labels this model cannot produce: "
                    f"{sorted(unknown)}. Valid labels include: "
                    f"{sorted(adapter.labels)[:8]}..."
                )

        self._info = EngineInfo(
            model=model_name,
            backend=backend.info.name,
            device=backend.info.device,
            input_size=adapter.input_size,
            precision=backend.info.precision,
            license=license,
            num_classes=len(adapter.labels),
        )

    # -- properties -----------------------------------------------------

    @property
    def info(self) -> EngineInfo:
        return self._info

    @property
    def labels(self) -> tuple[str, ...]:
        return self._adapter.labels

    @property
    def confidence(self) -> float:
        return self._confidence

    # -- inference ------------------------------------------------------

    def detect(self, frame: Frame) -> DetectionResult:
        """Detect objects in ``frame``, in original-frame pixel coordinates."""
        detections, timings = self._run(frame.image)
        return DetectionResult(
            detections=tuple(detections),
            source_id=frame.source_id,
            frame_index=frame.index,
            capture_wall=frame.capture_wall,
            frame_size=frame.resolution,
            model=self._info.model,
            backend=f"{self._info.backend}/{self._info.device}",
            preprocess_ms=timings[0],
            inference_ms=timings[1],
            postprocess_ms=timings[2],
        )

    def detect_image(self, image: np.ndarray) -> list[Detection]:
        """Detect in a bare BGR array. For benchmarking and offline evaluation."""
        detections, _ = self._run(image)
        return detections

    def _run(self, image: np.ndarray) -> tuple[list[Detection], tuple[float, float, float]]:
        if self._closed:
            raise RuntimeError("detection engine has been closed")

        start = self._clock.monotonic()
        prepared = self._adapter.preprocess(image)
        after_pre = self._clock.monotonic()

        outputs = self._backend.run(prepared.tensor, prepared.extra or None)
        after_infer = self._clock.monotonic()

        detections = self._adapter.postprocess(
            outputs,
            prepared,
            confidence=self._confidence,
            iou_threshold=self._iou_threshold,
            max_detections=self._max_detections,
        )
        if self._keep is not None:
            # Filtering after NMS, not before: suppression must still see the
            # discarded classes, or a filtered-out object stops suppressing the
            # duplicate boxes it overlaps.
            detections = [d for d in detections if d.label.lower() in self._keep]
        after_post = self._clock.monotonic()

        return detections, (
            (after_pre - start) * 1000.0,
            (after_infer - after_pre) * 1000.0,
            (after_post - after_infer) * 1000.0,
        )

    def warmup(self, iterations: int = 2) -> float:
        """Run inference on a blank frame to force lazy initialisation.

        First inference is dramatically slower than steady state - graph
        compilation, kernel selection, GPU clocks ramping. Without this the
        first real frame stalls the pipeline and the first benchmark sample is
        pure noise.

        Returns the total warmup time in milliseconds.
        """
        height, width = self._adapter.input_size
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.perf_counter()
        for _ in range(max(0, iterations)):
            self._run(blank)
        elapsed = (time.perf_counter() - started) * 1000.0
        log.debug(
            "engine warmed up",
            extra={
                "vantage_fields": {
                    "iterations": iterations,
                    "total_ms": round(elapsed, 1),
                    "backend": self._info.backend,
                }
            },
        )
        return elapsed

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> "DetectionEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_engine(
    model: str,
    *,
    backend: str = "auto",
    device: str = "auto",
    confidence: float = 0.35,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    keep_classes: Sequence[str] | None = None,
    model_dir: str | Path = "models",
    threads: int = 0,
    allow_download: bool = True,
    clock: Clock = SYSTEM_CLOCK,
) -> DetectionEngine:
    """Resolve a catalog key into a ready engine, fetching weights if needed."""
    spec: ModelSpec = get_model_spec(model)
    store = ModelStore(model_dir)
    path = store.ensure(spec, allow_download=allow_download)

    adapter_cls = get_adapter(spec.adapter)
    adapter = adapter_cls(input_size=spec.input_size, labels=spec.labels)
    inference_backend = create_backend(
        backend,
        path,
        device=device,
        threads=threads,
        input_shape=spec.input_size,
        input_shapes=adapter.static_input_shapes(),
    )

    engine = DetectionEngine(
        adapter=adapter,
        backend=inference_backend,
        model_name=spec.key,
        license=spec.license,
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        keep_classes=keep_classes,
        clock=clock,
    )
    log.info(
        "detection engine ready",
        extra={"vantage_fields": {"engine": engine.info.describe()}},
    )
    return engine
