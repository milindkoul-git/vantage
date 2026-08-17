"""The model catalog.

Every model the platform can use is declared here with its download URL, a
pinned SHA-256, its input geometry, its adapter, its label set, **and its
licence**. Weights are never committed to the repository; they are fetched on
demand and verified against the pin.

The licence field is not decoration. A detector's licence propagates to
whatever is built on it, and AGPL weights would quietly make this whole
platform AGPL. Recording it next to the URL means the constraint is visible at
the point of choice rather than discovered during a legal review.

SHA-256 pins mean a changed or substituted remote file fails loudly instead of
silently altering detection behaviour - the model is part of the system's
observable behaviour, so it deserves the same integrity treatment as code.
"""

from __future__ import annotations

from dataclasses import dataclass

from vantage.core.errors import ConfigError
from vantage.perception.labels import get_label_set

_YOLOX_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to fetch, verify, load and interpret one model."""

    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    adapter: str
    input_size: tuple[int, int]
    """``(height, width)`` the exported graph expects."""

    label_set: str
    license: str
    source: str
    description: str
    map_50_95: float | None = None
    """Reported COCO val AP, for choosing a size on evidence rather than vibes."""

    @property
    def labels(self) -> tuple[str, ...]:
        return get_label_set(self.label_set)

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def describe(self) -> str:
        accuracy = f"{self.map_50_95:.1f} mAP" if self.map_50_95 else "mAP n/a"
        return (
            f"{self.key:12s} {self.input_size[1]}x{self.input_size[0]}  "
            f"{self.size_bytes / 1e6:5.1f} MB  {accuracy:9s}  {self.license:10s} {self.description}"
        )


CATALOG: dict[str, ModelSpec] = {
    "yolox-nano": ModelSpec(
        key="yolox-nano",
        filename="yolox_nano.onnx",
        url=f"{_YOLOX_RELEASE}/yolox_nano.onnx",
        sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        size_bytes=3_659_407,
        adapter="yolox",
        input_size=(416, 416),
        label_set="coco80",
        license="Apache-2.0",
        source="https://github.com/Megvii-BaseDetection/YOLOX",
        description="Smallest YOLOX. The CPU-friendly default.",
        map_50_95=25.8,
    ),
    "yolox-tiny": ModelSpec(
        key="yolox-tiny",
        filename="yolox_tiny.onnx",
        url=f"{_YOLOX_RELEASE}/yolox_tiny.onnx",
        sha256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        size_bytes=20_219_662,
        adapter="yolox",
        input_size=(416, 416),
        label_set="coco80",
        license="Apache-2.0",
        source="https://github.com/Megvii-BaseDetection/YOLOX",
        description="Noticeably better than nano at similar input size.",
        map_50_95=32.8,
    ),
    "yolox-s": ModelSpec(
        key="yolox-s",
        filename="yolox_s.onnx",
        url=f"{_YOLOX_RELEASE}/yolox_s.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        size_bytes=35_858_002,
        adapter="yolox",
        input_size=(640, 640),
        label_set="coco80",
        license="Apache-2.0",
        source="https://github.com/Megvii-BaseDetection/YOLOX",
        description="640px input; the accuracy option, too slow for realtime CPU.",
        map_50_95=40.5,
    ),
}

DEFAULT_MODEL = "yolox-nano"


def get_model_spec(key: str) -> ModelSpec:
    """Look up a catalog entry, suggesting near misses on a typo."""
    normalised = (key or "").strip().lower()
    if normalised in CATALOG:
        return CATALOG[normalised]

    import difflib

    close = difflib.get_close_matches(normalised, list(CATALOG), n=1, cutoff=0.5)
    hint = f" (did you mean '{close[0]}'?)" if close else ""
    raise ConfigError(
        f"unknown detection.model {key!r}{hint}. Available models: {sorted(CATALOG)}. "
        "Run 'vantage models list' for details."
    )


def register_model(spec: ModelSpec) -> None:
    """Add a model to the catalog, e.g. a locally fine-tuned export."""
    CATALOG[spec.key] = spec
