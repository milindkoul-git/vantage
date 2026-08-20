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
_ONNX_COMMUNITY = "https://huggingface.co/onnx-community"
_MMPOSE_SDK = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"
_OPENCV_HF = "https://huggingface.co/opencv"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to fetch, verify, load and interpret one model."""

    key: str
    filename: str
    url: str
    sha256: str
    """SHA-256 of the file that is finally cached and loaded - the ``.onnx``
    itself, including when it arrives inside an archive."""

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

    task: str = "detect"
    """What the model produces: ``detect``, ``pose``. The catalog is shared
    because fetching, verifying, licensing and input geometry are identical
    concerns whatever the head predicts; only the decoding differs, and that is
    the adapter's job."""

    archive_member: str | None = None
    """Path inside the archive at :attr:`url`, when the upstream project ships a
    zip rather than a bare ``.onnx``.

    RTMPose is distributed this way by OpenMMLab. Re-uploads of the loose file
    exist on model hubs, but they are anonymous, carry no statement of which
    checkpoint or export config produced them, and a SHA pin cannot make a file
    trustworthy - it can only make it *unchanging*. Reading the member out of
    the authoritative archive costs ~40 lines in the store and keeps provenance
    intact."""

    archive_sha256: str | None = None
    """SHA-256 of the archive itself. Both this and :attr:`sha256` are checked:
    the archive on download, the extracted member before it is installed."""

    archive_size_bytes: int | None = None

    @property
    def is_archived(self) -> bool:
        return self.archive_member is not None

    @property
    def download_size_bytes(self) -> int:
        """Bytes actually transferred, which is the archive when there is one."""
        return self.archive_size_bytes or self.size_bytes

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
    "dfine-s-obj365": ModelSpec(
        key="dfine-s-obj365",
        filename="dfine_s_obj365.onnx",
        url=f"{_ONNX_COMMUNITY}/dfine_s_obj365-ONNX/resolve/main/onnx/model.onnx",
        sha256="372feaa33ac6ba67d7df8589628f6abc395b3c6981c4edc70dfcfe2949751120",
        size_bytes=42_123_225,
        adapter="dfine",
        input_size=(640, 640),
        label_set="objects365",
        license="Apache-2.0",
        source="https://github.com/Peterande/D-FINE",
        description="365 classes incl. Pen/Pencil, Marker, Stapler. 4.5x COCO's vocabulary.",
        map_50_95=None,
    ),
    "dfine-m-obj365": ModelSpec(
        key="dfine-m-obj365",
        filename="dfine_m_obj365.onnx",
        url=f"{_ONNX_COMMUNITY}/dfine_m_obj365-ONNX/resolve/main/onnx/model.onnx",
        sha256="2fb7d73e2df5be2b1032381d191d6d26d1bca44b23d3eb79af1cbd9e3b19356c",
        size_bytes=79_212_285,
        adapter="dfine",
        input_size=(640, 640),
        label_set="objects365",
        license="Apache-2.0",
        source="https://github.com/Peterande/D-FINE",
        description="Larger D-FINE. More accurate, roughly twice the cost of the small one.",
        map_50_95=None,
    ),
    "grounding-dino-tiny": ModelSpec(
        key="grounding-dino-tiny",
        filename="grounding_dino_tiny_fp16.onnx",
        url=f"{_ONNX_COMMUNITY}/grounding-dino-tiny-ONNX/resolve/main/onnx/model_fp16.onnx",
        sha256="04c18d2db35569f11c47732f2e05ed3a71559a8903823fc581e90b0e3168c9ff",
        size_bytes=360_393_267,
        adapter="grounding-dino",
        input_size=(800, 800),
        label_set="open-vocabulary",
        license="Apache-2.0",
        source="https://github.com/IDEA-Research/GroundingDINO",
        description="Open vocabulary: finds whatever you name. ~2.2 s/frame - discovery only.",
        map_50_95=None,
    ),
    "rtmpose-t": ModelSpec(
        key="rtmpose-t",
        filename="rtmpose_t_body7_256x192.onnx",
        url=f"{_MMPOSE_SDK}/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
        sha256="a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f",
        size_bytes=13_350_364,
        archive_member=(
            "20230831/rtmpose_onnx/"
            "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504/end2end.onnx"
        ),
        archive_sha256="937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb",
        archive_size_bytes=12_547_710,
        adapter="rtmpose",
        input_size=(256, 192),
        label_set="coco-keypoints",
        license="Apache-2.0",
        source="https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        description="Smallest RTMPose. 17 body keypoints, one person per pass.",
        task="pose",
    ),
    "rtmpose-s": ModelSpec(
        key="rtmpose-s",
        filename="rtmpose_s_body7_256x192.onnx",
        url=f"{_MMPOSE_SDK}/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        sha256="9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d",
        size_bytes=21_890_172,
        archive_member=(
            "20230831/rtmpose_onnx/"
            "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504/end2end.onnx"
        ),
        archive_sha256="7673922e531014906ca4f0f239b7e233b740146a10b632deaa2a28d45470d802",
        archive_size_bytes=20_496_303,
        adapter="rtmpose",
        input_size=(256, 192),
        label_set="coco-keypoints",
        license="Apache-2.0",
        source="https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        description="More accurate RTMPose. On the iGPU it costs 0.5 ms more than -t.",
        task="pose",
    ),
    "yunet-face": ModelSpec(
        key="yunet-face",
        filename="face_detection_yunet_2023mar.onnx",
        url=f"{_OPENCV_HF}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size_bytes=232_589,
        adapter="yunet",
        input_size=(320, 320),
        label_set="face",
        license="MIT",
        source="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        description="Face detector with 5 landmarks. Needed to align a crop for SFace.",
        task="face-detect",
    ),
    "sface": ModelSpec(
        key="sface",
        filename="face_recognition_sface_2021dec.onnx",
        url=f"{_OPENCV_HF}/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size_bytes=38_696_353,
        adapter="sface",
        input_size=(112, 112),
        label_set="face",
        license="Apache-2.0",
        source="https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface",
        description="Face embedding, 128-d. Apache-2.0, unlike ArcFace weights.",
        task="face-embed",
    ),
}

DEFAULT_MODEL = "yolox-nano"
DEFAULT_POSE_MODEL = "rtmpose-s"


def models_for_task(task: str) -> dict[str, ModelSpec]:
    """Catalog entries producing a given output type."""
    return {key: spec for key, spec in CATALOG.items() if spec.task == task}


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
