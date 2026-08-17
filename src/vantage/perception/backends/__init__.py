"""Inference backends: runtimes that execute a graph.

A backend receives an input tensor and returns output tensors. It has no
opinion about what those numbers mean, which is what allows ONNX Runtime and
OpenVINO to be benchmarked against each other and produce byte-comparable
detections.

Both backends read the **same ONNX file**. Keeping the model format runtime-
neutral is deliberate: it preserves the option to move to a third runtime, or
to different hardware, without re-exporting weights.
"""

from vantage.perception.backends.base import BackendInfo, InferenceBackend

__all__ = ["BackendInfo", "InferenceBackend", "available_backends", "create_backend"]


def available_backends() -> dict[str, bool]:
    """Which backends can actually be constructed in this environment."""
    import importlib.util

    return {
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "openvino": importlib.util.find_spec("openvino") is not None,
    }


def create_backend(name: str, model_path, device: str = "auto", threads: int = 0):
    """Construct a backend by name, with an actionable error if it is missing."""
    from vantage.core.errors import ConfigError

    normalised = (name or "auto").strip().lower()
    availability = available_backends()

    if normalised == "auto":
        # OpenVINO first: it is the only one of the two that can reach the Intel
        # iGPU on this class of machine, and it falls back to a strong CPU path.
        for candidate in ("openvino", "onnxruntime"):
            if availability.get(candidate):
                normalised = candidate
                break
        else:
            raise ConfigError(
                "no inference backend is installed. Install one with "
                "'pip install onnxruntime' or 'pip install openvino'."
            )

    if normalised == "onnxruntime":
        if not availability["onnxruntime"]:
            raise ConfigError(
                "detection.backend is 'onnxruntime' but the package is not "
                "installed. Run: pip install onnxruntime"
            )
        from vantage.perception.backends.onnxruntime_backend import OnnxRuntimeBackend

        return OnnxRuntimeBackend(model_path, device=device, threads=threads)

    if normalised == "openvino":
        if not availability["openvino"]:
            raise ConfigError(
                "detection.backend is 'openvino' but the package is not "
                "installed. Run: pip install openvino"
            )
        from vantage.perception.backends.openvino_backend import OpenVinoBackend

        return OpenVinoBackend(model_path, device=device, threads=threads)

    raise ConfigError(
        f"unknown detection.backend {name!r}; valid values are "
        "'auto', 'onnxruntime' or 'openvino'"
    )
