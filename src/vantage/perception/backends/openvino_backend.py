"""OpenVINO backend.

The reason this exists: environment discovery found no CUDA GPU but an Intel
Iris Xe iGPU, and OpenVINO is the only runtime in reach that can actually use
it. It reads the same ONNX file directly - no conversion step, no second copy
of the weights, no lock-in.

Device selection is honest rather than optimistic. ``auto`` here means "GPU if
this machine really has a usable one, else CPU", resolved by asking the runtime
what it can see and reporting what was chosen. A silent fallback from GPU to
CPU would make every benchmark number meaningless.

Telemetry: OpenVINO ships an ``openvino-telemetry`` dependency. Given this
platform's privacy stance, opting out is set here before the import rather than
left to whatever the ambient default happens to be.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vantage.core.errors import VantageError
from vantage.core.logging import get_logger
from vantage.perception.backends.base import BackendInfo, InferenceBackend

log = get_logger(__name__)

# Must be set before openvino is imported to take effect.
os.environ.setdefault("OPENVINO_TELEMETRY_OPT_OUT", "1")


class OpenVinoBackend(InferenceBackend):
    """Runs an ONNX graph through the OpenVINO runtime, on CPU or Intel GPU."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str = "auto",
        threads: int = 0,
        input_shape: tuple[int, int] | None = None,
        input_shapes: dict[str, list[int]] | None = None,
    ) -> None:
        import openvino as ov

        path = Path(model_path)
        if not path.is_file():
            raise VantageError(
                f"model file not found: {path}. Fetch it with 'vantage models pull'."
            )

        self._core = ov.Core()
        available = list(self._core.available_devices)
        resolved = self._resolve_device(device, available)

        # DO NOT enable OpenVINO's CACHE_DIR here. It looks like free startup
        # time and costs a crash: on this Iris Xe / OpenVINO 2026.3 combination,
        # *loading* a cached GPU blob segfaults the process during interpreter
        # shutdown - after inference has completed and produced correct results,
        # so it presents as a mysterious exit code 139 rather than as a failure.
        # Bisected to CACHE_DIR specifically: writing the cache is clean, reading
        # it back is not.
        #
        # The optimisation it buys is small anyway. Measured on this machine:
        #   GPU compile, no cache    1058 ms
        #   GPU compile, warm cache   182 ms   <- the crashing path
        # Under a second of one-off startup, against a process that then runs
        # for hours. Not a trade worth making.
        config: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
        if threads > 0 and resolved.startswith("CPU"):
            config["INFERENCE_NUM_THREADS"] = str(threads)

        model = self._core.read_model(path)
        if input_shapes:
            # A multi-input graph: every dynamic input must be pinned, not just
            # the image, or the GPU plugin refuses to compile at all.
            model.reshape(dict(input_shapes.items()))
            log.debug(
                "pinned all graph inputs to static shapes",
                extra={"vantage_fields": {"inputs": sorted(input_shapes)}},
            )
        static = None if input_shapes else _static_shape_for(model, input_shape)
        if static is not None:
            # A graph exported with a dynamic input is legal but expensive: the
            # GPU plugin cannot specialise its kernels and falls back to a
            # general path. Measured on D-FINE here, 184 ms/frame dynamic
            # against 69 ms pinned - a 2.7x difference for a shape the adapter
            # already knows. YOLOX exports are static, which is why this never
            # surfaced before a DETR-family model was added.
            model.reshape({model.input(0).any_name: static})
            log.debug(
                "pinned dynamic input to a static shape",
                extra={"vantage_fields": {"shape": static}},
            )
        try:
            compiled = self._core.compile_model(model, resolved, config)
        except RuntimeError as exc:
            raise VantageError(
                f"OpenVINO could not compile the model for device {resolved!r}: {exc}. "
                f"Devices this machine reports: {available}. "
                "Use --device cpu to fall back."
            ) from exc

        self._compiled = compiled
        self._request = compiled.create_infer_request()
        self._input = compiled.input(0)
        self._output_count = len(compiled.outputs)

        try:
            device_name = self._core.get_property(resolved, "FULL_DEVICE_NAME")
        except Exception:  # pragma: no cover - property support varies by device
            device_name = resolved

        # Report the precision actually used, not the precision of the file on
        # disk. OpenVINO runs fp16 on Intel GPUs by default, which is why GPU
        # detections differ slightly from CPU ones - claiming fp32 here would
        # make that difference look like a bug.
        precision = "fp32"
        try:  # noqa: SIM105 - the comment below is the point
            precision = _normalise_precision(compiled.get_property("INFERENCE_PRECISION_HINT"))
        except Exception:  # pragma: no cover - property support varies by device
            pass

        self._info = BackendInfo(
            name="openvino",
            device=resolved.lower(),
            version=ov.__version__,
            input_name=self._input.get_any_name(),
            input_shape=tuple(self._input.get_partial_shape().get_max_shape()),
            precision=precision,
            extra={
                "compiled_model_cache": "disabled (see comment: crashes on GPU blob load)",
                "device_full_name": str(device_name),
                "available_devices": available,
                "performance_hint": "LATENCY",
                "threads": threads or "auto",
            },
        )
        log.info(
            "inference backend ready",
            extra={
                "vantage_fields": {
                    "backend": "openvino",
                    "version": ov.__version__,
                    "device": resolved,
                    "device_name": str(device_name),
                    "model": path.name,
                }
            },
        )

    @staticmethod
    def _resolve_device(requested: str, available: list[str]) -> str:
        """Map a config value onto a device this machine actually reports."""
        wanted = (requested or "auto").strip().upper()

        if wanted in {"AUTO", ""}:
            # Prefer the iGPU, but only if it is really present. OpenVINO also
            # offers a meta-device literally named "AUTO"; it is avoided here
            # because it hides which device ran, and that is the one thing a
            # benchmark must not hide.
            for candidate in ("GPU", "CPU"):
                if any(device.startswith(candidate) for device in available):
                    return next(d for d in available if d.startswith(candidate))
            raise VantageError(f"OpenVINO reports no usable devices (saw {available})")

        matches = [device for device in available if device.startswith(wanted)]
        if not matches:
            raise VantageError(
                f"OpenVINO device {requested!r} is not available on this machine. "
                f"Reported devices: {available}."
            )
        return matches[0]

    @property
    def info(self) -> BackendInfo:
        return self._info

    def run(
        self, tensor: np.ndarray, extra: dict[str, np.ndarray] | None = None
    ) -> list[np.ndarray]:
        if extra:
            feed: dict = {self._input.any_name: tensor}
            feed.update(extra)
        else:
            feed = {self._input: tensor}
        self._request.infer(feed)
        # copy=True is mandatory, not defensive. Tensor.data is a view onto the
        # infer request's internal buffer, which the *next* inference overwrites
        # in place - a retained output silently changes under its owner, and
        # the buffer outliving the request is a dangling pointer. Verified:
        # without the copy, a held result fails an equality check after the
        # following infer() call.
        return [
            np.array(self._request.get_output_tensor(index).data, copy=True)
            for index in range(self._output_count)
        ]

    def close(self) -> None:
        self._request = None
        self._compiled = None
        self._core = None


def _static_shape_for(model, input_shape: tuple[int, int] | None) -> list[int] | None:
    """The fully static input shape to compile with, or ``None`` to leave it alone."""
    partial = model.input(0).get_partial_shape()
    if not partial.is_dynamic:
        return None
    if input_shape is None:
        return None
    height, width = input_shape
    return [1, 3, int(height), int(width)]


def _normalise_precision(value: object) -> str:
    """Render OpenVINO's precision object as a short, comparable token.

    The runtime returns an ``ov.Type`` whose ``str`` is ``<Type: 'float32'>``,
    which is unreadable in a benchmark table and impossible to compare against
    the other backend's plain ``fp32``.
    """
    text = str(value).lower()
    for needle, name in (
        ("float16", "fp16"),
        ("float32", "fp32"),
        ("bfloat16", "bf16"),
        ("int8", "int8"),
    ):
        if needle in text:
            return name
    return text.strip("<>") or "unknown"
