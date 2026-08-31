"""Unit and Integration Tests for Re-ID Appearance Memory and Multi-Prototype Matching."""

from __future__ import annotations

import time

import numpy as np

from vantage.core.frame import Frame
from vantage.multicam.reid import (
    CrossCameraReIDTracker,
    EntityAppearanceMemory,
    HSVAppearanceProvider,
    VisualEmbedding,
)
from vantage.perception.contracts import BoundingBox
from vantage.tracking.contracts import Track, TrackState


def test_hsv_appearance_provider() -> None:
    provider = HSVAppearanceProvider()

    # Synthetic person image: 100x40 BGR (blue shirt, dark pants)
    img = np.zeros((100, 40, 3), dtype=np.uint8)
    img[:50, :] = (255, 0, 0)  # Blue upper torso
    img[50:, :] = (30, 30, 30)  # Dark lower legs

    box = BoundingBox(0.0, 0.0, 40.0, 100.0)
    emb = provider.extract(img, box)

    assert isinstance(emb, VisualEmbedding)
    assert len(emb.vector) == 128
    assert emb.quality > 0.5
    assert emb.cosine_similarity(emb) > 0.999


def test_entity_appearance_memory_prototypes() -> None:
    mem = EntityAppearanceMemory()
    now = time.time()

    # 1. Add first observation
    v1 = tuple([0.5] * 128)
    emb1 = VisualEmbedding(vector=v1, quality=0.8)
    mem.add_observation("cam_01", emb1, now)

    assert len(mem.representative_prototypes) == 1
    assert len(mem.recent_descriptors) == 1

    # 2. Add very similar observation (should not create duplicate prototype)
    v2 = tuple([0.51] * 128)
    emb2 = VisualEmbedding(vector=v2, quality=0.82)
    mem.add_observation("cam_01", emb2, now + 0.1)

    assert len(mem.representative_prototypes) == 1
    assert len(mem.recent_descriptors) == 2

    # 3. Add distinct viewpoint observation (should store new prototype)
    v3 = tuple([0.1 if i % 2 == 0 else 0.8 for i in range(128)])
    emb3 = VisualEmbedding(vector=v3, quality=0.85)
    mem.add_observation("cam_02", emb3, now + 1.0)

    assert len(mem.representative_prototypes) == 2

    # 4. Compute similarity against candidate
    sim = mem.compute_similarity(emb1)
    assert sim > 0.99


def test_cross_camera_reid_tracker_association() -> None:
    tracker = CrossCameraReIDTracker(min_reid_similarity=0.75)

    img1 = np.full((120, 60, 3), 200, dtype=np.uint8)
    frame1 = Frame(
        image=img1, index=1, source_id="cam_01", capture_monotonic=100.0, capture_wall=100.0
    )

    track1 = Track(
        track_id=1,
        entity_id="person_1",
        box=BoundingBox(10.0, 10.0, 50.0, 110.0),
        label="person",
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=1,
        hits=1,
        time_since_update=0,
        start_frame=1,
        last_frame=1,
    )

    # 1. Register on cam_01
    map1 = tracker.update_camera("cam_01", frame1, [track1])
    assert 1 in map1
    gid = map1[1]
    assert gid.startswith("global_person_")

    # 2. Seen on cam_02 with similar appearance
    img2 = np.full((120, 60, 3), 195, dtype=np.uint8)
    frame2 = Frame(
        image=img2, index=50, source_id="cam_02", capture_monotonic=103.0, capture_wall=103.0
    )

    track2 = Track(
        track_id=5,
        entity_id="person_5",
        box=BoundingBox(12.0, 10.0, 52.0, 110.0),
        label="person",
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=1,
        hits=1,
        time_since_update=0,
        start_frame=50,
        last_frame=50,
    )

    map2 = tracker.update_camera("cam_02", frame2, [track2])
    assert 5 in map2
    assert map2[5] == gid  # Successfully fused to same global entity!
