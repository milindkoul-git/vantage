"""ONNX Runtime backend.

The portable baseline. It runs the same ONNX file everywhere, which makes it
the reference every other backend is measured against - if OpenVINO produced
different detections from the same weights, this is what would reveal it.

Note on execution providers: the stock ``onnxruntime`` wheel exposes only CPU
(plus an Azure stub). The OpenVINO execution provider lives in a separate
``onnxruntime-openvino`` distribution that *replaces* this one, since both
install a module named ``onnxruntime``. Rather than force that either/or on
anyone cloning this repo, Intel acceleration is reached through the native
OpenVINO backend instead, and this backend stays the clean portable CPU path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vantage.core.errors import VantageError
from vantage.core.logging import get_logger
from vantage.perception.backends.base import BackendInfo, InferenceBackend

log = get_logger(__name__)


class OnnxRuntimeBackend(InferenceBackend):
    """Runs an ONNX graph through ONNX Runtime."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str = "auto",
        threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise VantageError(
                f"model file not found: {path}. Fetch it with 'vantage models pull'."
            )

        requested = (device or "auto").strip().lower()
        if requested not in {"auto", "cpu"}:
            log.warning(
                "onnxruntime backend only offers CPU here; ignoring requested device",
                extra={"vantage_fields": {"requested": requested}},
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            # Left at 0 (= let ORT decide) by default. Pinning matters when
            # several cameras share a machine and must not fight for cores.
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._output_names = [output.name for output in self._session.get_outputs()]

        self._info = BackendInfo(
            name="onnxruntime",
            device="cpu",
            version=ort.__version__,
            input_name=self._input_name,
            input_shape=tuple(dim if isinstance(dim, int) else -1 for dim in model_input.shape),
            precision="fp32",
            extra={
                "providers": self._session.get_providers(),
                "threads": threads or "auto",
            },
        )
        log.info(
            "inference backend ready",
            extra={
                "vantage_fields": {
                    "backend": "onnxruntime",
                    "version": ort.__version__,
                    "device": "cpu",
                    "model": path.name,
                }
            },
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def run(
        self, tensor: np.ndarray, extra: dict[str, np.ndarray] | None = None
    ) -> list[np.ndarray]:
        feed = {self._input_name: tensor}
        if extra:
            feed.update(extra)
        outputs = self._session.run(self._output_names, feed)
        return [np.asarray(output) for output in outputs]

    def close(self) -> None:
        # ORT releases native resources when the session is collected; dropping
        # the reference is the supported way to do it.
        self._session = None  # type: ignore[assignment]
