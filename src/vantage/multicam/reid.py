"""Cross-Camera Re-Identification (Re-ID) & Global Identity Fusion.

Associates entities across multi-angle overlapping and disjoint camera views using
appearance descriptors, temporal windows, and visual similarity thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from vantage.core.frame import Frame
from vantage.perception.contracts import BoundingBox
from vantage.tracking.contracts import Track


@dataclass(frozen=True, slots=True)
class VisualEmbedding:
    """Multi-zone color and texture descriptor for an entity."""

    vector: tuple[float, ...]
    quality: float = 1.0

    def cosine_similarity(self, other: VisualEmbedding) -> float:
        """Compute cosine similarity in [-1, 1]."""
        u = np.array(self.vector, dtype=np.float32)
        v = np.array(other.vector, dtype=np.float32)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))


class AppearanceDescriptorProvider(Protocol):
    """Protocol for appearance feature extractors (HSV, CNN, or Transformer ReID)."""

    def extract(self, image: np.ndarray, box: BoundingBox) -> VisualEmbedding: ...


class HSVAppearanceProvider:
    """Extracts multi-zone spatial color embeddings from entity bounding box crops."""

    def extract(self, image: np.ndarray, box: BoundingBox) -> VisualEmbedding:
        """Extract multi-tier color descriptor from person crop."""
        h, w = image.shape[:2]
        x1 = max(0, min(int(box.x1), w - 1))
        y1 = max(0, min(int(box.y1), h - 1))
        x2 = max(0, min(int(box.x2), w))
        y2 = max(0, min(int(box.y2), h))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < 6 or crop_h < 12:
            return VisualEmbedding(tuple([0.0] * 128), quality=0.0)

        # Quality estimate: resolution score + aspect ratio reasonableness
        quality = min(1.0, (crop_w * crop_h) / (40.0 * 100.0))
        aspect = crop_h / max(1.0, crop_w)
        if aspect < 1.2 or aspect > 4.5:
            quality *= 0.7

        crop = image[y1:y2, x1:x2]

        # 4 vertical zones: Head/Shoulders (0-25%), Upper Torso (25-50%), Lower Torso/Hips (50-75%), Legs (75-100%)
        zones = [
            crop[: int(crop_h * 0.25), :],
            crop[int(crop_h * 0.25) : int(crop_h * 0.50), :],
            crop[int(crop_h * 0.50) : int(crop_h * 0.75), :],
            crop[int(crop_h * 0.75) :, :],
        ]

        descriptor_parts = []
        for zone in zones:
            if zone.size == 0:
                descriptor_parts.extend([0.0] * 32)
                continue

            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            # 16-bin Hue, 8-bin Saturation, 8-bin Value (32 bins per zone)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

            hist = np.concatenate([h_hist, s_hist, v_hist])
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist = hist / norm
            descriptor_parts.extend(hist.tolist())

        return VisualEmbedding(tuple(descriptor_parts), quality=float(quality))


# Backward compatibility alias
VisualDescriptorExtractor = HSVAppearanceProvider


@dataclass
class EntityAppearanceMemory:
    """Multi-prototype appearance memory for a cross-camera global identity.

    Stores recent appearance embeddings, quality scores, and representative prototypes
    across camera angles to prevent single-observation drift.
    """

    recent_descriptors: deque[VisualEmbedding] = field(default_factory=lambda: deque(maxlen=10))
    representative_prototypes: list[VisualEmbedding] = field(default_factory=list)
    camera_specific_descriptors: dict[str, VisualEmbedding] = field(default_factory=dict)
    temporal_history: list[dict[str, Any]] = field(default_factory=list)

    def add_observation(
        self,
        camera_id: str,
        embedding: VisualEmbedding,
        wall_time: float,
    ) -> None:
        """Add a new observation and update representative prototypes."""
        self.recent_descriptors.append(embedding)
        self.camera_specific_descriptors[camera_id] = embedding

        # Prototype management: keep up to 4 high-quality prototypes across distinct viewpoints
        if embedding.quality >= 0.4:
            if not self.representative_prototypes:
                self.representative_prototypes.append(embedding)
            else:
                # Check if this embedding offers distinct appearance or higher quality
                max_sim = max(
                    embedding.cosine_similarity(p) for p in self.representative_prototypes
                )
                if max_sim < 0.92 and len(self.representative_prototypes) < 4:
                    self.representative_prototypes.append(embedding)
                elif embedding.quality > 0.8:
                    # Update lowest quality prototype
                    min_idx = min(
                        range(len(self.representative_prototypes)),
                        key=lambda i: self.representative_prototypes[i].quality,
                    )
                    if embedding.quality > self.representative_prototypes[min_idx].quality:
                        self.representative_prototypes[min_idx] = embedding

        self.temporal_history.append(
            {
                "camera_id": camera_id,
                "timestamp": wall_time,
                "quality": embedding.quality,
            }
        )
        if len(self.temporal_history) > 30:
            self.temporal_history.pop(0)

    def compute_similarity(self, candidate: VisualEmbedding) -> float:
        """Compute matching similarity against prototypes and recent observations."""
        candidates_to_test = list(self.representative_prototypes) or list(
            self.recent_descriptors
        )
        if not candidates_to_test:
            return 0.0

        sims = [candidate.cosine_similarity(p) for p in candidates_to_test]
        # Return weighted max similarity
        return max(sims)


@dataclass
class GlobalTrackEntry:
    """Internal state of a cross-camera global identity."""

    global_id: str
    label: str
    appearance: EntityAppearanceMemory
    last_camera: str
    last_seen_wall: float
    last_box: BoundingBox
    total_sightings: int = 1
    sighting_cameras: set[str] = field(default_factory=set)

    @property
    def embedding(self) -> VisualEmbedding:
        """Backward compatibility property returning the latest or first prototype embedding."""
        if self.appearance.representative_prototypes:
            return self.appearance.representative_prototypes[0]
        if self.appearance.recent_descriptors:
            return self.appearance.recent_descriptors[-1]
        return VisualEmbedding(tuple([0.0] * 128))


class CrossCameraReIDTracker:
    """Fuses local camera tracks into globally consistent identities across cameras."""

    def __init__(
        self,
        *,
        min_reid_similarity: float = 0.80,
        allow_overlapping: bool = True,
        min_transit_time_s: float = 0.0,
        max_transition_time_s: float = 45.0,
        extractor: AppearanceDescriptorProvider | None = None,
    ) -> None:
        self._extractor: AppearanceDescriptorProvider = extractor or HSVAppearanceProvider()
        self._min_similarity = min_reid_similarity
        self._allow_overlapping = allow_overlapping
        self._min_transit_time_s = min_transit_time_s
        self._max_transition_time_s = max_transition_time_s
        self._global_entities: dict[str, GlobalTrackEntry] = {}
        self._camera_local_to_global: dict[
            tuple[str, int], str
        ] = {}  # (camera_id, local_track_id) -> global_id
        self._next_global_idx = 1

    def update_camera(
        self,
        camera_id: str,
        frame: Frame,
        tracks: list[Track] | tuple[Track, ...],
    ) -> dict[int, str]:
        """Process tracks for one camera, returning mapping {local_track_id: global_id}."""
        now = frame.capture_wall
        mapping: dict[int, str] = {}

        # 1. Prune ancient global entries that have been inactive for too long
        expired = [
            gid
            for gid, entry in self._global_entities.items()
            if now - entry.last_seen_wall > self._max_transition_time_s
        ]
        for gid in expired:
            del self._global_entities[gid]

        # 2. Process each track
        for track in tracks:
            key = (camera_id, track.track_id)
            if key in self._camera_local_to_global:
                # Already associated with this specific camera's local track
                global_id = self._camera_local_to_global[key]
                if global_id in self._global_entities:
                    entry = self._global_entities[global_id]
                    entry.last_seen_wall = now
                    entry.last_camera = camera_id
                    entry.last_box = track.box
                    entry.total_sightings += 1
                    entry.sighting_cameras.add(camera_id)

                    # Extract descriptor and update appearance memory
                    emb = self._extractor.extract(frame.image, track.box)
                    entry.appearance.add_observation(camera_id, emb, now)

                mapping[track.track_id] = global_id
                continue

            # Extract visual appearance descriptor
            emb = self._extractor.extract(frame.image, track.box)

            # Search existing global entities for candidate matching across cameras
            best_match_id = None
            best_sim = self._min_similarity

            for gid, entry in self._global_entities.items():
                if entry.label != track.label:
                    continue

                dt = now - entry.last_seen_wall
                if (
                    not self._allow_overlapping
                    and entry.last_camera != camera_id
                    and dt < self._min_transit_time_s
                ):
                    continue
                if dt < 0 or dt > self._max_transition_time_s:
                    continue

                # Compute appearance similarity against prototype memory
                sim = entry.appearance.compute_similarity(emb)

                if sim > best_sim:
                    best_sim = sim
                    best_match_id = gid

            if best_match_id is not None:
                # Same person recognized across cameras!
                global_id = best_match_id
                entry = self._global_entities[global_id]
                entry.last_seen_wall = now
                entry.last_camera = camera_id
                entry.last_box = track.box
                entry.total_sightings += 1
                entry.sighting_cameras.add(camera_id)
                entry.appearance.add_observation(camera_id, emb, now)
            else:
                # Distinct unique person - assign brand new global ID
                global_id = f"global_{track.label}_{self._next_global_idx}"
                self._next_global_idx += 1
                app_mem = EntityAppearanceMemory()
                app_mem.add_observation(camera_id, emb, now)

                self._global_entities[global_id] = GlobalTrackEntry(
                    global_id=global_id,
                    label=track.label,
                    appearance=app_mem,
                    last_camera=camera_id,
                    last_seen_wall=now,
                    last_box=track.box,
                    sighting_cameras={camera_id},
                )

            self._camera_local_to_global[key] = global_id
            mapping[track.track_id] = global_id

        return mapping
