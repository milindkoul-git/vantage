"""Dataset fetcher, catalog and annotation loader for surveillance and HOI data."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

from vantage.core.logging import get_logger

log = get_logger(__name__)

# Standard domain target classes
DOMAIN_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "cell_phone",
    7: "bottle",
    8: "backpack",
    9: "chair",
}

CLASS_TO_ID = {v: k for k, v in DOMAIN_CLASSES.items()}


@dataclass(frozen=True, slots=True)
class AnnotatedSample:
    """One training image/frame with ground truth boxes and interactions."""

    image_path: str
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float, int], ...]  # (x1, y1, x2, y2, class_id)
    interactions: tuple[tuple[int, int, str], ...] = ()  # (person_idx, object_idx, verb)


@dataclass
class DatasetCatalog:
    """Manages local and remote training datasets."""

    data_dir: Path

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> dict[str, dict[str, str]]:
        return {
            "crowdhuman_mini": {
                "name": "CrowdHuman Surveillance Mini",
                "description": "Dense pedestrian crowds with high occlusion rates",
                "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
                "type": "video_surveillance",
            },
            "hoi_interactions_mini": {
                "name": "Human-Object Interactions Mini",
                "description": "Pedestrians holding phones, bottles, bags, and riding bikes",
                "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
                "type": "video_hoi",
            },
            "visdrone_security": {
                "name": "VisDrone Security Benchmark",
                "description": "Overhead/top-down perspectives of pedestrians and vehicles",
                "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
                "type": "surveillance_overhead",
            },
        }

    def fetch(self, dataset_key: str) -> Path:
        """Download and cache dataset samples."""
        info = self.list_datasets().get(dataset_key)
        if not info:
            raise ValueError(
                f"Unknown dataset key: {dataset_key}. Available: {list(self.list_datasets().keys())}"
            )

        dest = self.data_dir / f"{dataset_key}.mp4"
        if not dest.is_file():
            log.info(f"Downloading dataset {dataset_key} from {info['url']}...")
            urllib.request.urlretrieve(info["url"], dest)
            log.info(f"Cached {dataset_key} ({dest.stat().st_size / (1024 * 1024):.1f}MB)")
        return dest
