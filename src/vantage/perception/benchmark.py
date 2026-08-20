"""Backend benchmarking.

Phase 2's backend choice is supposed to rest on measurements from *this*
machine rather than on reputation, so the measurement is a first-class feature
rather than a script that got thrown away.

Methodology, and why each part matters:

*Warmup is excluded.* The first passes pay for graph compilation, kernel
selection and, on an iGPU, clock ramp-up. Including them would understate every
backend, and would understate the GPU most.

*Percentiles, not just the mean.* A backend with a good mean and a terrible p95
produces visible stutter. For a realtime pipeline the tail is what the viewer
actually perceives.

*Detection counts are reported alongside timings.* A backend that is fast
because it silently produced fewer detections is not faster, and running at
reduced precision is exactly how that happens - Intel GPUs default to fp16.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vantage.core.logging import get_logger
from vantage.perception.catalog import get_model_spec

log = get_logger(__name__)

ALL_TARGETS: tuple[tuple[str, str], ...] = (
    ("onnxruntime", "cpu"),
    ("openvino", "cpu"),
    ("openvino", "gpu"),
)


def resolve_targets(requested: str, availability: dict[str, bool]) -> list[tuple[str, str]]:
    """Turn a ``--backends`` string into concrete (backend, device) pairs."""
    text = (requested or "all").strip().lower()

    if text == "all":
        candidates = list(ALL_TARGETS)
    else:
        candidates = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            backend, _, device = chunk.partition(":")
            candidates.append((backend.strip(), (device or "auto").strip()))

    return [(backend, device) for backend, device in candidates if availability.get(backend)]


def benchmark(
    model: str,
    targets: list[tuple[str, str]],
    image: np.ndarray,
    iterations: int = 50,
    warmup: int = 5,
    model_dir: str = "models",
) -> list[dict[str, Any]]:
    """Time each target on the same image and return one record per target."""
    from vantage.perception.engine import build_engine

    spec = get_model_spec(model)
    results: list[dict[str, Any]] = []

    for backend, device in targets:
        record: dict[str, Any] = {"backend": backend, "device": device}
        try:
            engine = build_engine(spec.key, backend=backend, device=device, model_dir=model_dir)
        except Exception as exc:
            # A device that is unavailable is a legitimate outcome to report,
            # not a reason to abandon the whole benchmark.
            record.update(available=False, error=f"{type(exc).__name__}: {exc}")
            results.append(record)
            log.warning(
                "backend unavailable for benchmark",
                extra={
                    "vantage_fields": {"backend": backend, "device": device, "error": str(exc)}
                },
            )
            continue

        try:
            load_started = time.perf_counter()
            engine.warmup(warmup)
            warmup_ms = (time.perf_counter() - load_started) * 1000.0

            samples: list[float] = []
            inference_samples: list[float] = []
            detection_counts: list[int] = []

            for _ in range(max(1, iterations)):
                started = time.perf_counter()
                detections = engine.detect_image(image)
                samples.append((time.perf_counter() - started) * 1000.0)
                detection_counts.append(len(detections))

            ordered = sorted(samples)
            mean_ms = float(np.mean(samples))
            record.update(
                available=True,
                precision=engine.info.precision,
                iterations=len(samples),
                warmup_ms=round(warmup_ms, 1),
                mean_ms=round(mean_ms, 2),
                p50_ms=round(_percentile(ordered, 50), 2),
                p95_ms=round(_percentile(ordered, 95), 2),
                min_ms=round(ordered[0], 2),
                max_ms=round(ordered[-1], 2),
                fps=round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
                detections=int(round(float(np.mean(detection_counts)))),
            )
            del inference_samples
        finally:
            engine.close()

        results.append(record)

    return results


def _percentile(ordered: list[float], percent: float) -> float:
    if not ordered:
        return 0.0
    rank = max(1, int(round(percent / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def format_table(model: str, image_label: str, results: list[dict[str, Any]]) -> str:
    """Render benchmark records as a readable table."""
    spec = get_model_spec(model)
    lines = [
        f"Detection benchmark - {spec.key} ({spec.input_size[1]}x{spec.input_size[0]}, "
        f"{spec.license})",
        f"Input: {image_label}",
        "",
        f"  {'BACKEND':13s} {'DEVICE':7s} {'PREC':6s} {'MEAN':>8s} {'p50':>8s} "
        f"{'p95':>8s} {'FPS':>7s} {'DETS':>5s} {'LOAD':>9s}",
    ]

    usable = [r for r in results if r.get("available")]
    for record in results:
        if not record.get("available"):
            lines.append(
                f"  {record['backend']:13s} {record['device']:7s} "
                f"unavailable - {record.get('error', 'unknown reason')[:60]}"
            )
            continue
        lines.append(
            f"  {record['backend']:13s} {record['device']:7s} {record['precision']:6s} "
            f"{record['mean_ms']:7.2f}ms {record['p50_ms']:7.2f}ms {record['p95_ms']:7.2f}ms "
            f"{record['fps']:6.1f}  {record['detections']:5d} {record['warmup_ms']:8.0f}ms"
        )

    if usable:
        fastest = min(usable, key=lambda r: r["mean_ms"])
        lines += [
            "",
            f"  Fastest: {fastest['backend']}/{fastest['device']} at "
            f"{fastest['mean_ms']:.2f} ms ({fastest['fps']:.1f} fps ceiling).",
        ]
        counts = {r["detections"] for r in usable}
        if len(counts) > 1:
            lines.append(
                "  Note: backends disagree on detection count "
                f"({sorted(counts)}), which reduced precision can explain - "
                "compare the PREC column before treating one as faster."
            )
        lines.append(
            "  LOAD is one-off model compilation, paid at every start. "
            "CPU figures are sensitive to whatever else the machine is doing; "
            "GPU figures are not, which is visible in the p95 column."
        )
    return "\n".join(lines)
