"""Tests for the ingestion pipeline - delivery, threading, loss accounting, shutdown."""

from __future__ import annotations

import threading
import time

import pytest

from vantage.config.schema import Backpressure, IngestConfig, IngestMode
from vantage.core.clock import ManualClock
from vantage.core.errors import SourceReadError
from vantage.ingestion.pipeline import IngestionPipeline
from vantage.ingestion.synthetic import SyntheticSource
from tests.fakes import FakeSource


def synthetic(frames: int | None = 12, **kwargs) -> SyntheticSource:
    params = dict(width=64, height=48, fps=30.0, frames=frames, objects=1)
    params.update(kwargs)
    return SyntheticSource(source_id="t", **params)


class TestInlineMode:
    def test_delivers_every_frame_in_order(self) -> None:
        config = IngestConfig(mode=IngestMode.INLINE)
        with IngestionPipeline(synthetic(8), config) as pipeline:
            indices = [frame.index for frame in pipeline.frames()]
        assert indices == list(range(8))

    def test_uses_no_thread_and_no_queue(self) -> None:
        before = threading.active_count()
        config = IngestConfig(mode=IngestMode.INLINE)
        with IngestionPipeline(synthetic(4), config) as pipeline:
            list(pipeline.frames())
            assert pipeline.stats().backpressure == "inline"
        assert threading.active_count() == before

    def test_stride_samples_deterministically(self) -> None:
        config = IngestConfig(mode=IngestMode.INLINE, stride=3)
        with IngestionPipeline(synthetic(10), config) as pipeline:
            indices = [frame.index for frame in pipeline.frames()]
        assert indices == [0, 3, 6, 9]
        assert pipeline.stats().frames_skipped == 6

    def test_max_frames_stops_early(self) -> None:
        config = IngestConfig(mode=IngestMode.INLINE, max_frames=5)
        with IngestionPipeline(synthetic(None), config) as pipeline:
            assert len(list(pipeline.frames())) == 5

    def test_target_fps_paces_delivery(self) -> None:
        clock = ManualClock()
        config = IngestConfig(mode=IngestMode.INLINE, target_fps=10.0)
        with IngestionPipeline(synthetic(4), config, clock=clock) as pipeline:
            list(pipeline.frames())
        # Four waits: one before each of the four frames. The very first call
        # only arms the deadline, and a fifth wait precedes the read that
        # discovers exhaustion.
        assert clock.slept == pytest.approx([0.1] * 4)

    def test_realtime_paces_recorded_sources_to_their_own_timeline(self) -> None:
        clock = ManualClock()
        config = IngestConfig(mode=IngestMode.INLINE, realtime=True)
        with IngestionPipeline(synthetic(4, fps=20.0), config, clock=clock) as pipeline:
            list(pipeline.frames())
        # First frame anchors; each subsequent frame waits one 20 fps interval.
        assert clock.slept == pytest.approx([0.05, 0.05, 0.05])


class TestThreadedMode:
    def test_delivers_every_frame_of_a_recorded_source(self) -> None:
        """BLOCK backpressure is chosen automatically, so nothing is lost."""
        config = IngestConfig(mode=IngestMode.THREADED, queue_size=2)
        with IngestionPipeline(synthetic(30), config) as pipeline:
            indices = [frame.index for frame in pipeline.frames()]
        assert indices == list(range(30))
        assert pipeline.stats().frames_dropped == 0
        assert pipeline.stats().backpressure == Backpressure.BLOCK.value

    def test_slow_consumer_on_a_live_source_drops_instead_of_lagging(self) -> None:
        """The core Phase 2 scenario: inference slower than capture."""
        source = FakeSource([i % 255 for i in range(400)], source_id="live")
        config = IngestConfig(
            mode=IngestMode.THREADED, queue_size=2, backpressure=Backpressure.LATEST
        )
        delivered = []
        with IngestionPipeline(source, config) as pipeline:
            for frame in pipeline.frames():
                delivered.append(frame.index)
                time.sleep(0.004)  # a "slow detector"
                if len(delivered) >= 15:
                    break

        stats = pipeline.stats()
        assert stats.frames_dropped > 0, "a slow consumer must shed frames, not queue them"
        assert stats.queue_depth <= config.queue_size
        # Gaps in the delivered indices are the signal downstream stages rely on.
        assert delivered == sorted(delivered)
        assert max(delivered) >= len(delivered)

    def test_stops_cleanly_when_the_consumer_breaks_out(self) -> None:
        pipeline = IngestionPipeline(synthetic(None), IngestConfig(queue_size=4))
        with pipeline:
            for frame in pipeline.frames():
                if frame.index >= 3:
                    break
        assert pipeline.source.state.value == "closed"

    def test_capture_thread_error_surfaces_on_the_consumer_thread(self) -> None:
        source = FakeSource([1, 2, SourceReadError("device fell over")], source_id="live")
        with IngestionPipeline(source, IngestConfig(queue_size=4)) as pipeline:
            with pytest.raises(SourceReadError, match="device fell over"):
                list(pipeline.frames())

    def test_shutdown_event_stops_delivery(self) -> None:
        shutdown = threading.Event()
        pipeline = IngestionPipeline(synthetic(None), IngestConfig(queue_size=4), shutdown=shutdown)
        received = 0
        with pipeline:
            for _ in pipeline.frames():
                received += 1
                if received == 3:
                    shutdown.set()
        assert received >= 3

    def test_no_capture_thread_survives_close(self) -> None:
        before = threading.active_count()
        with IngestionPipeline(synthetic(20), IngestConfig(queue_size=2)) as pipeline:
            list(pipeline.frames())
        deadline = time.monotonic() + 5.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.01)
        assert threading.active_count() == before


class TestStats:
    def test_reports_source_identity_and_geometry(self) -> None:
        with IngestionPipeline(synthetic(5, width=128, height=96)) as pipeline:
            list(pipeline.frames())
            stats = pipeline.stats()
        assert stats.resolution == "128x96"
        assert stats.source_id == "t"
        assert stats.kind == "synthetic"
        assert stats.backend == "synthetic"
        assert stats.declared_fps == 30.0

    def test_counts_produced_delivered_and_skipped(self) -> None:
        config = IngestConfig(mode=IngestMode.INLINE, stride=2)
        with IngestionPipeline(synthetic(10), config) as pipeline:
            list(pipeline.frames())
            stats = pipeline.stats()
        assert stats.frames_produced == 10
        assert stats.frames_delivered == 5
        assert stats.frames_skipped == 5

    def test_measures_throughput_and_latency(self) -> None:
        config = IngestConfig(mode=IngestMode.INLINE)
        with IngestionPipeline(synthetic(20), config) as pipeline:
            list(pipeline.frames())
            stats = pipeline.stats()
        assert stats.mean_delivery_fps > 0
        assert stats.latency_ms_p50 >= 0
        assert stats.elapsed_s > 0

    def test_drop_rate_is_derived_safely_before_any_frames(self) -> None:
        pipeline = IngestionPipeline(synthetic(5))
        assert pipeline.stats().drop_rate == 0.0

    def test_stats_work_before_open_and_after_close(self) -> None:
        pipeline = IngestionPipeline(synthetic(5))
        assert pipeline.stats().state == "created"
        with pipeline:
            list(pipeline.frames())
        assert pipeline.stats().frames_delivered == 5

    def test_snapshot_is_serialisable(self) -> None:
        import json

        with IngestionPipeline(synthetic(3)) as pipeline:
            list(pipeline.frames())
            json.dumps(pipeline.stats().to_dict())


class TestLifecycle:
    def test_close_is_idempotent(self) -> None:
        pipeline = IngestionPipeline(synthetic(3))
        pipeline.start()
        pipeline.close()
        pipeline.close()

    def test_start_is_idempotent(self) -> None:
        pipeline = IngestionPipeline(synthetic(3))
        first = pipeline.start()
        assert pipeline.start() is first
        pipeline.close()

    def test_frames_starts_the_pipeline_implicitly(self) -> None:
        pipeline = IngestionPipeline(synthetic(3), IngestConfig(mode=IngestMode.INLINE))
        try:
            assert len(list(pipeline.frames())) == 3
        finally:
            pipeline.close()
