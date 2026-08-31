"""Tests for the decoupled AsyncInferenceStage."""

from __future__ import annotations

import time

import numpy as np

from tests.fakes import make_engine
from vantage.config.schema import Backpressure
from vantage.core.frame import Frame
from vantage.perception.stage import AsyncInferenceStage


def _make_frame(index: int = 0) -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        index=index,
        capture_monotonic=1.0 + index * 0.033,
        capture_wall=100.0 + index * 0.033,
        source_id="cam_0",
    )


def test_async_stage_lifecycle() -> None:
    engine, _ = make_engine()
    stage = AsyncInferenceStage(engine, queue_size=2)
    assert stage.get_latest_result() is None
    assert stage.queue_depth == 0

    stage.start()
    frame = _make_frame(0)
    admitted = stage.submit(frame)
    assert admitted is True

    result = stage.wait_for_result(timeout_s=1.0)
    assert result is not None
    assert result.frame_index == 0

    stats = stage.stats
    assert stats.submitted == 1
    assert stats.inferences_run >= 1
    assert stats.dropped == 0

    stage.stop()
    assert stage.submit(_make_frame(1)) is False


def test_async_stage_backpressure_drop() -> None:
    engine, _ = make_engine()
    stage = AsyncInferenceStage(
        engine,
        queue_size=1,
        policy=Backpressure.LATEST,
    )
    stage.start()

    for i in range(10):
        stage.submit(_make_frame(i))

    time.sleep(0.1)
    res = stage.get_latest_result()
    assert res is not None
    stage.stop()
