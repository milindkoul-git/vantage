"""Appearance-based re-identification matcher (external post-tracker).

Maintains a gallery of lost tracks and proposes re-linking newly spawned tracks
to expired identities across long occlusions (>1.5s).

Design Stance
-------------
ByteTrack remains strictly motion-only and non-biometric. This module operates
strictly *after* ByteTrack has finalized its geometric assignment. If a track is
lost past ByteTrack's max_lost_s, its appearance descriptor is retained in an
ephemeral gallery. When a new track appears, its crop is compared against lost
tracks to propose identity recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vantage.core.logging import get_logger

if TYPE_CHECKING:
    from vantage.perception.contracts import BoundingBox

log = get_logger(__name__)


def extract_appearance_descriptor(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Extract a normalized multi-channel color histogram from a bounding box crop."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.clipped(w, h).to_int()
    if x2 <= x1 or y2 <= y1:
        return np.zeros(64, dtype=np.float32)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(64, dtype=np.float32)

    # Convert to HSV for illumination robustness
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 8 Hue bins, 4 Saturation bins, 2 Value bins = 64 bins
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 2], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass(slots=True)
class LostTrackRecord:
    """Record of a lost track's appearance and metadata."""

    entity_id: str
    track_id: int
    label: str
    lost_timestamp: float
    descriptor: np.ndarray


@dataclass(slots=True)
class ReIdMatch:
    """A proposed re-link from a new track to an old entity."""

    new_track_id: int
    matched_entity_id: str
    similarity: float
    margin: float


class AppearanceMatcher:
    """Decoupled appearance matcher for long-occlusion track recovery."""

    def __init__(
        self,
        *,
        threshold: float = 0.75,
        margin: float = 0.10,
        max_lost_time_s: float = 8.0,
    ) -> None:
        self._threshold = threshold
        self._margin = margin
        self._max_lost_time_s = max_lost_time_s
        self._lost_gallery: dict[str, LostTrackRecord] = {}
        self._relink_map: dict[str, str] = {}  # temporary_id -> resolved_entity_id

    def record_lost(
        self,
        entity_id: str,
        track_id: int,
        label: str,
        timestamp: float,
        descriptor: np.ndarray,
    ) -> None:
        """Store a lost track in the appearance gallery."""
        self._lost_gallery[entity_id] = LostTrackRecord(
            entity_id=entity_id,
            track_id=track_id,
            label=label,
            lost_timestamp=timestamp,
            descriptor=descriptor,
        )

    def match_candidate(
        self,
        new_track_id: int,
        label: str,
        current_time: float,
        candidate_descriptor: np.ndarray,
    ) -> ReIdMatch | None:
        """Attempt to match a new track's descriptor against the lost gallery."""
        # Prune expired lost tracks
        self._prune(current_time)

        candidates: list[tuple[str, float]] = []
        for entity_id, record in self._lost_gallery.items():
            if record.label != label:
                continue
            sim = cosine_similarity(candidate_descriptor, record.descriptor)
            if sim >= self._threshold:
                candidates.append((entity_id, sim))

        if not candidates:
            return None

        # Sort by similarity descending
        candidates.sort(key=lambda c: c[1], reverse=True)
        best_entity, best_sim = candidates[0]
        runner_up_sim = candidates[1][1] if len(candidates) > 1 else 0.0
        margin = best_sim - runner_up_sim

        if margin < self._margin and len(candidates) > 1:
            log.debug(
                "reid rejected match below margin",
                extra={"vantage_fields": {"best": best_sim, "runner_up": runner_up_sim}},
            )
            return None

        # Matched successfully -> remove from gallery
        del self._lost_gallery[best_entity]
        return ReIdMatch(
            new_track_id=new_track_id,
            matched_entity_id=best_entity,
            similarity=best_sim,
            margin=margin,
        )

    def _prune(self, current_time: float) -> None:
        expired = [
            e_id
            for e_id, rec in self._lost_gallery.items()
            if (current_time - rec.lost_timestamp) > self._max_lost_time_s
        ]
        for e_id in expired:
            del self._lost_gallery[e_id]
