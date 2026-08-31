"""Model optimization, layer fusion, and OpenVINO IR weight compression."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov

from vantage.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Statistics comparing original ONNX to optimized OpenVINO IR."""

    original_path: str
    optimized_path: str
    original_size_mb: float
    optimized_size_mb: float
    size_reduction_pct: float
    speedup_ratio: float


def optimize_model(
    onnx_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    compress_to_fp16: bool = True,
) -> OptimizationResult:
    """Optimize an ONNX graph and export to compressed OpenVINO IR (.xml + .bin)."""
    p_in = Path(onnx_path)
    if not p_in.is_file():
        raise FileNotFoundError(f"Source model file not found: {p_in}")

    out_dir = Path(output_dir) if output_dir else p_in.parent / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xml = out_dir / f"{p_in.stem}.xml"
    out_bin = out_dir / f"{p_in.stem}.bin"

    core = ov.Core()
    model = core.read_model(p_in)

    # Save to optimized IR with weight compression and fused operators
    ov.save_model(model, out_xml, compress_to_fp16=compress_to_fp16)

    orig_size = p_in.stat().st_size / (1024 * 1024)
    xml_size = out_xml.stat().st_size if out_xml.exists() else 0
    bin_size = out_bin.stat().st_size if out_bin.exists() else 0
    opt_size = (xml_size + bin_size) / (1024 * 1024)

    reduction = ((orig_size - opt_size) / orig_size) * 100.0 if orig_size > 0 else 0.0

    # Quick benchmark comparison
    t_onnx = _bench(core, p_in)
    t_ir = _bench(core, out_xml)
    speedup = t_onnx / max(t_ir, 0.001)

    log.info(
        f"Optimized {p_in.name}: {orig_size:.1f}MB -> {opt_size:.1f}MB ({reduction:.1f}% reduction, {speedup:.2f}x speedup)"
    )

    return OptimizationResult(
        original_path=str(p_in),
        optimized_path=str(out_xml),
        original_size_mb=round(orig_size, 2),
        optimized_size_mb=round(opt_size, 2),
        size_reduction_pct=round(reduction, 1),
        speedup_ratio=round(speedup, 2),
    )


def _bench(core: ov.Core, model_path: Path, runs: int = 15) -> float:
    try:
        model = core.read_model(model_path)
        compiled = core.compile_model(
            model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "EXECUTION_MODE_HINT": "PERFORMANCE"}
        )
        input_shape = model.input(0).shape
        static_shape = [
            s if isinstance(s, int) and s > 0 else (1 if i == 0 else 416)
            for i, s in enumerate(input_shape)
        ]
        dummy = np.zeros(static_shape, dtype=np.float32)

        # Warmup
        compiled([dummy])

        t0 = time.perf_counter()
        for _ in range(runs):
            compiled([dummy])
        return (time.perf_counter() - t0) / runs
    except Exception as exc:
        log.warning(f"Benchmark error for {model_path}: {exc}")
        return 1.0
