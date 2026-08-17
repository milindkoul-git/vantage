"""Tests for the frame contract, clock, and metrics primitives."""

from __future__ import annotations

import numpy as np
import pytest

from vantage.core.clock import ManualClock, SystemClock
from vantage.core.frame import Frame
from vantage.core.metrics import Counter, LatencyTracker, MetricsRegistry, RateMeter


def make_frame(**kwargs) -> Frame:
    defaults = dict(
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        index=0,
        source_id="s",
        capture_monotonic=10.0,
        capture_wall=1_700_000_000.0,
    )
    defaults.update(kwargs)
    return Frame(**defaults)


class TestFrame:
    def test_exposes_geometry(self) -> None:
        frame = make_frame()
        assert frame.resolution == (6, 4)
        assert frame.width == 6 and frame.height == 4
        assert frame.nbytes == 4 * 6 * 3

    def test_pixels_are_read_only(self) -> None:
        """Frames are shared by reference across stages; nobody may mutate one."""
        frame = make_frame()
        with pytest.raises(ValueError):
            frame.image[0, 0, 0] = 255

    def test_editable_copy_is_writable_and_independent(self) -> None:
        frame = make_frame()
        copy = frame.editable_copy()
        copy[0, 0, 0] = 200
        assert frame.image[0, 0, 0] == 0

    def test_is_immutable(self) -> None:
        frame = make_frame()
        with pytest.raises(Exception):
            frame.index = 5  # type: ignore[misc]

    def test_age_uses_monotonic_base(self) -> None:
        frame = make_frame(capture_monotonic=10.0)
        assert frame.age_ms(10.025) == pytest.approx(25.0)

    @pytest.mark.parametrize(
        "bad,error",
        [
            ({"image": np.zeros((4, 6), dtype=np.uint8)}, ValueError),
            ({"image": np.zeros((4, 6, 3), dtype=np.float32)}, ValueError),
            ({"image": [[0, 0, 0]]}, TypeError),
            ({"index": -1}, ValueError),
        ],
    )
    def test_rejects_malformed_input(self, bad: dict, error: type[Exception]) -> None:
        with pytest.raises(error):
            make_frame(**bad)

    def test_describe_distinguishes_live_from_recorded(self) -> None:
        assert "live" in make_frame(media_pts=None).describe()
        assert "1.500s" in make_frame(media_pts=1.5).describe()


class TestClock:
    def test_manual_clock_sleep_advances_instead_of_blocking(self) -> None:
        clock = ManualClock(start_monotonic=0.0)
        clock.sleep(2.5)
        assert clock.monotonic() == 2.5
        assert clock.wall() == pytest.approx(1_700_000_002.5)
        assert clock.slept == [2.5]

    def test_manual_clock_ignores_non_positive_sleep(self) -> None:
        clock = ManualClock()
        clock.sleep(0)
        clock.sleep(-1)
        assert clock.slept == []

    def test_manual_clock_refuses_to_go_backwards(self) -> None:
        with pytest.raises(ValueError):
            ManualClock().advance(-1)

    def test_system_clock_is_monotonic(self) -> None:
        clock = SystemClock()
        assert clock.monotonic() <= clock.monotonic()
        assert clock.wall() > 1_600_000_000


class TestMetrics:
    def test_counter(self) -> None:
        counter = Counter()
        counter.inc()
        counter.inc(4)
        assert counter.value == 5

    def test_rate_meter_mean_rate_over_known_interval(self) -> None:
        meter = RateMeter()
        for i in range(11):  # 11 events spanning exactly 1.0s
            meter.tick(i * 0.1)
        assert meter.count == 11
        assert meter.mean_rate == pytest.approx(10.0)
        assert meter.rate == pytest.approx(10.0, abs=0.5)

    def test_rate_meter_is_zero_before_two_events(self) -> None:
        meter = RateMeter()
        assert meter.rate == 0.0
        meter.tick(1.0)
        assert meter.rate == 0.0
        assert meter.mean_rate == 0.0

    def test_rate_meter_ignores_non_advancing_time(self) -> None:
        meter = RateMeter()
        meter.tick(1.0)
        meter.tick(1.0)  # same instant - would divide by zero
        assert meter.rate == 0.0

    def test_rate_meter_is_not_skewed_by_bursty_arrival(self) -> None:
        """Averaging intervals, not rates.

        A source that delivers nine frames 10 ms apart then stalls for 210 ms
        emits ten frames per 300 ms - about 33 fps. Smoothing instantaneous
        rates would report roughly triple that; smoothing intervals does not.
        """
        meter = RateMeter(alpha=0.5)
        now = 0.0
        for _ in range(12):  # repeat the burst-then-stall pattern
            for _ in range(9):
                now += 0.010
                meter.tick(now)
            now += 0.210
            meter.tick(now)

        assert meter.mean_rate == pytest.approx(33.3, rel=0.05)
        assert meter.rate < 66.0, "smoothed rate must not double the true arrival rate"

    def test_latency_percentiles(self) -> None:
        tracker = LatencyTracker(window=100)
        for value in range(1, 101):
            tracker.observe(float(value))
        assert tracker.percentile(50) == pytest.approx(50.0)
        assert tracker.percentile(95) == pytest.approx(95.0)
        assert tracker.max == 100.0
        assert tracker.mean == pytest.approx(50.5)
        assert tracker.last == 100.0

    def test_latency_window_is_bounded_but_max_is_lifetime(self) -> None:
        tracker = LatencyTracker(window=3)
        for value in [100.0, 1.0, 2.0, 3.0]:
            tracker.observe(value)
        assert tracker.percentile(50) == pytest.approx(2.0)  # window dropped the 100
        assert tracker.max == 100.0  # but the peak is never forgotten

    def test_empty_tracker_reports_zero(self) -> None:
        tracker = LatencyTracker()
        assert tracker.percentile(50) == 0.0
        assert tracker.snapshot()["samples"] == 0

    def test_registry_snapshot_is_json_safe(self) -> None:
        import json

        registry = MetricsRegistry(name="root")
        registry.counter("dropped").inc(2)
        registry.rate("fps").tick(0.0)
        registry.rate("fps").tick(1.0)
        registry.latency("delay").observe(5.0)
        registry.child("camera").counter("errors").inc()

        snapshot = registry.snapshot()
        json.dumps(snapshot)  # must not raise
        assert snapshot["counters"]["dropped"] == 2
        assert snapshot["children"]["camera"]["counters"]["errors"] == 1
