"""Tests for backpressure policy and rate control.

These cover the behaviour that decides what happens when Phase 2's detector is
slower than the camera, so they are the most consequential tests in the suite.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vantage.config.schema import Backpressure
from vantage.core.clock import ManualClock
from vantage.core.frame import Frame
from vantage.ingestion.buffer import FrameBuffer, resolve_backpressure
from vantage.ingestion.pacing import MediaClockPacer, RatePacer, StrideFilter


def frame(index: int) -> Frame:
    return Frame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        index=index,
        source_id="s",
        capture_monotonic=float(index),
        capture_wall=float(index),
    )


class TestBackpressureResolution:
    def test_live_sources_drop_to_stay_current(self) -> None:
        assert resolve_backpressure(Backpressure.AUTO, is_live=True) is Backpressure.LATEST

    def test_recorded_sources_block_to_stay_complete(self) -> None:
        assert resolve_backpressure(Backpressure.AUTO, is_live=False) is Backpressure.BLOCK

    def test_explicit_policy_is_respected(self) -> None:
        assert resolve_backpressure(Backpressure.DROP_NEW, is_live=True) is Backpressure.DROP_NEW

    def test_auto_must_be_resolved_before_use(self) -> None:
        with pytest.raises(ValueError):
            FrameBuffer(capacity=2, policy=Backpressure.AUTO)


class TestLatestPolicy:
    def test_evicts_oldest_and_counts_the_loss(self) -> None:
        buffer = FrameBuffer(capacity=2, policy=Backpressure.LATEST)
        for i in range(5):
            assert buffer.put(frame(i)) is True

        assert buffer.dropped == 3
        # The consumer sees the newest frames, and the index gap tells it what
        # it missed - the property downstream trackers depend on.
        assert [buffer.get().index for _ in range(2)] == [3, 4]

    def test_high_water_records_peak_depth(self) -> None:
        buffer = FrameBuffer(capacity=4, policy=Backpressure.LATEST)
        for i in range(3):
            buffer.put(frame(i))
        buffer.get()
        assert buffer.high_water == 3


class TestDropNewPolicy:
    def test_rejects_arrivals_and_keeps_the_oldest(self) -> None:
        buffer = FrameBuffer(capacity=2, policy=Backpressure.DROP_NEW)
        assert buffer.put(frame(0)) is True
        assert buffer.put(frame(1)) is True
        assert buffer.put(frame(2)) is False
        assert buffer.dropped == 1
        assert buffer.get().index == 0


class TestBlockPolicy:
    def test_producer_waits_for_room_and_loses_nothing(self) -> None:
        buffer = FrameBuffer(capacity=1, policy=Backpressure.BLOCK)
        produced: list[bool] = []

        def produce() -> None:
            for i in range(4):
                produced.append(buffer.put(frame(i), timeout=5.0))

        thread = threading.Thread(target=produce)
        thread.start()
        received = [buffer.get(timeout=5.0).index for _ in range(4)]
        thread.join(timeout=5.0)

        assert received == [0, 1, 2, 3]
        assert all(produced)
        assert buffer.dropped == 0

    def test_put_gives_up_on_timeout_rather_than_hanging(self) -> None:
        buffer = FrameBuffer(capacity=1, policy=Backpressure.BLOCK)
        buffer.put(frame(0))
        assert buffer.put(frame(1), timeout=0.05) is False

    def test_close_releases_a_blocked_producer(self) -> None:
        buffer = FrameBuffer(capacity=1, policy=Backpressure.BLOCK)
        buffer.put(frame(0))
        results: list[bool] = []

        thread = threading.Thread(target=lambda: results.append(buffer.put(frame(1), timeout=5.0)))
        thread.start()
        buffer.close()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), "close() must wake a producer blocked on a full queue"
        assert results == [False]


class TestBufferLifecycle:
    def test_get_returns_none_when_closed_and_drained(self) -> None:
        buffer = FrameBuffer(capacity=2, policy=Backpressure.LATEST)
        buffer.put(frame(0))
        buffer.close()
        assert buffer.get().index == 0
        assert buffer.get(timeout=0.1) is None

    def test_closed_buffer_rejects_new_frames(self) -> None:
        buffer = FrameBuffer(capacity=2, policy=Backpressure.LATEST)
        buffer.close()
        assert buffer.put(frame(0)) is False

    def test_clear_reports_how_much_was_discarded(self) -> None:
        buffer = FrameBuffer(capacity=4, policy=Backpressure.LATEST)
        for i in range(3):
            buffer.put(frame(i))
        assert buffer.clear() == 3
        assert len(buffer) == 0

    def test_capacity_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            FrameBuffer(capacity=0, policy=Backpressure.LATEST)


class TestStrideFilter:
    def test_keeps_every_frame_by_default(self) -> None:
        stride = StrideFilter(1)
        assert all(stride.keep(i) for i in range(10))
        assert stride.skipped == 0

    def test_selects_every_nth_frame_by_source_index(self) -> None:
        stride = StrideFilter(3)
        kept = [i for i in range(10) if stride.keep(i)]
        assert kept == [0, 3, 6, 9]
        assert stride.skipped == 6

    def test_selection_is_independent_of_where_the_run_started(self) -> None:
        """Keying on the source index makes the sampled set reproducible."""
        assert StrideFilter(4).keep(100) is True
        assert StrideFilter(4).keep(101) is False

    def test_rejects_invalid_stride(self) -> None:
        with pytest.raises(ValueError):
            StrideFilter(0)


class TestRatePacer:
    def test_no_target_means_no_waiting(self) -> None:
        clock = ManualClock()
        pacer = RatePacer(None, clock=clock)
        assert pacer.wait() == 0.0
        assert clock.slept == []

    def test_holds_the_configured_interval(self) -> None:
        clock = ManualClock()
        pacer = RatePacer(10.0, clock=clock)  # 100 ms apart
        pacer.wait()  # first call only arms the deadline
        assert pacer.wait() == pytest.approx(0.1)
        assert pacer.wait() == pytest.approx(0.1)
        assert pacer.target_fps == pytest.approx(10.0)

    def test_does_not_burst_to_catch_up_after_a_stall(self) -> None:
        clock = ManualClock()
        pacer = RatePacer(10.0, clock=clock)
        pacer.wait()
        clock.advance(1.0)  # a long stall - 10 slots missed

        assert pacer.wait() == 0.0  # resynchronise, do not sprint
        assert pacer.wait() == pytest.approx(0.1)  # and pace normally again

    def test_target_is_adjustable_at_runtime(self) -> None:
        """The control point Phase 12 adaptive sampling will drive."""
        clock = ManualClock()
        pacer = RatePacer(10.0, clock=clock)
        pacer.wait()
        pacer.set_target(5.0)
        pacer.wait()
        assert pacer.wait() == pytest.approx(0.2)

    def test_rejects_invalid_target(self) -> None:
        with pytest.raises(ValueError):
            RatePacer(0.0)


class TestMediaClockPacer:
    def test_anchors_on_the_first_frame(self) -> None:
        clock = ManualClock()
        pacer = MediaClockPacer(clock=clock)
        assert pacer.wait_for(5.0) == 0.0  # first pts, whatever it is, is "now"
        assert pacer.wait_for(5.5) == pytest.approx(0.5)

    def test_does_not_wait_when_already_behind(self) -> None:
        clock = ManualClock()
        pacer = MediaClockPacer(clock=clock)
        pacer.wait_for(0.0)
        clock.advance(10.0)
        assert pacer.wait_for(1.0) == 0.0

    def test_falls_back_to_a_fixed_interval_without_timestamps(self) -> None:
        clock = ManualClock()
        pacer = MediaClockPacer(fallback_fps=25.0, clock=clock)
        assert pacer.wait_for(None) == pytest.approx(0.04)

    def test_without_timestamps_or_fallback_it_does_not_wait(self) -> None:
        pacer = MediaClockPacer(fallback_fps=None, clock=ManualClock())
        assert pacer.wait_for(None) == 0.0
