"""Tests for the source abstraction, the synthetic generator, and reconnection."""

from __future__ import annotations

import pytest

from tests.fakes import FakeSource
from vantage.config.schema import ReconnectConfig
from vantage.core.clock import ManualClock
from vantage.core.errors import (
    ConfigError,
    SourceExhausted,
    SourceOpenError,
    SourceReadError,
    SourceStateError,
)
from vantage.ingestion.base import SourceKind, SourceState
from vantage.ingestion.resilient import ReconnectingSource
from vantage.ingestion.synthetic import SyntheticSource


class TestSourceLifecycle:
    def test_read_before_open_is_rejected(self) -> None:
        with pytest.raises(SourceStateError):
            FakeSource([1]).read()

    def test_info_before_open_is_rejected(self) -> None:
        with pytest.raises(SourceStateError):
            _ = FakeSource().info

    def test_indices_increase_and_timestamps_are_stamped(self) -> None:
        clock = ManualClock(start_monotonic=100.0)
        source = FakeSource([1, 2, 3], clock=clock)
        with source:
            frames = [source.read() for _ in range(3)]

        assert [f.index for f in frames] == [0, 1, 2]
        assert all(f.capture_monotonic == 100.0 for f in frames)
        assert all(f.source_id == "fake" for f in frames)
        assert source.frames_produced == 3

    def test_exhaustion_is_normal_termination_not_failure(self) -> None:
        source = FakeSource([])
        source.open()
        with pytest.raises(SourceExhausted):
            source.read()
        assert source.state is SourceState.EXHAUSTED

    def test_read_failure_marks_the_source_failed(self) -> None:
        source = FakeSource([SourceReadError("boom")])
        source.open()
        with pytest.raises(SourceReadError):
            source.read()
        assert source.state is SourceState.FAILED

    def test_open_failure_marks_the_source_failed(self) -> None:
        source = FakeSource()
        source.open_error = SourceOpenError("no device")
        with pytest.raises(SourceOpenError):
            source.open()
        assert source.state is SourceState.FAILED

    def test_close_is_idempotent(self) -> None:
        source = FakeSource([1])
        source.open()
        source.close()
        source.close()
        assert source.closes == 1
        assert source.state is SourceState.CLOSED

    def test_context_manager_closes_on_exception(self) -> None:
        source = FakeSource([1])
        with pytest.raises(RuntimeError):
            with source:
                raise RuntimeError("consumer blew up")
        assert source.state is SourceState.CLOSED

    def test_cannot_be_reopened_after_close(self) -> None:
        source = FakeSource([1])
        source.open()
        source.close()
        with pytest.raises(SourceStateError):
            source.open()


class TestSyntheticSource:
    def test_reports_its_negotiated_properties(self) -> None:
        source = SyntheticSource(width=320, height=240, fps=15.0, frames=5)
        with source as opened:
            info = opened.info
        assert info.resolution == (320, 240)
        assert info.declared_fps == 15.0
        assert info.frame_count == 5
        assert info.kind is SourceKind.SYNTHETIC
        assert info.is_live is False
        assert info.duration_s == pytest.approx(5 / 15.0)

    def test_produces_exactly_the_configured_frame_count(self) -> None:
        source = SyntheticSource(width=64, height=48, frames=4, objects=1)
        with source:
            for _ in range(4):
                source.read()
            with pytest.raises(SourceExhausted):
                source.read()

    def test_unbounded_when_no_frame_count_is_given(self) -> None:
        source = SyntheticSource(width=32, height=32, frames=None, objects=0)
        with source:
            assert source.info.frame_count is None
            assert source.read().index == 0

    def test_is_deterministic_for_a_given_seed(self) -> None:
        """Byte-identical output is what makes it usable as a test oracle."""

        def render(seed: int) -> bytes:
            source = SyntheticSource(width=96, height=72, frames=3, seed=seed, objects=3)
            with source:
                return b"".join(source.read().image.tobytes() for _ in range(3))

        assert render(11) == render(11)
        assert render(11) != render(12)

    def test_frames_carry_a_media_timeline(self) -> None:
        source = SyntheticSource(width=32, height=32, fps=10.0, frames=3, objects=0)
        with source:
            assert [source.read().media_pts for _ in range(3)] == [0.0, 0.1, 0.2]

    def test_object_states_are_ground_truth_within_the_frame(self) -> None:
        """Exact known positions - the basis for evaluating trackers in Phase 3."""
        source = SyntheticSource(width=200, height=150, frames=50, seed=5, objects=3)
        with source:
            for index in (0, 10, 49):
                states = source.object_states(index)
                assert len(states) == 3
                for state in states:
                    x1, y1, x2, y2 = state.bbox
                    assert 0 <= x1 < x2 <= 200
                    assert 0 <= y1 < y2 <= 150

    def test_object_positions_are_a_pure_function_of_the_index(self) -> None:
        source = SyntheticSource(width=200, height=150, frames=10, seed=5, objects=2)
        first = source.object_states(7)
        second = source.object_states(7)
        assert [(s.cx, s.cy) for s in first] == [(s.cx, s.cy) for s in second]

    @pytest.mark.parametrize(
        "kwargs",
        [{"width": 8}, {"height": 4}, {"fps": 0}, {"frames": 0}, {"objects": -1}],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict) -> None:
        with pytest.raises(ConfigError):
            SyntheticSource(**kwargs)


class TestReconnectingSource:
    def policy(self, **kwargs) -> ReconnectConfig:
        defaults = dict(
            enabled=True, max_attempts=3, initial_delay_s=0.01, max_delay_s=0.05, backoff=2.0
        )
        defaults.update(kwargs)
        return ReconnectConfig(**defaults)

    def test_passes_frames_through_when_healthy(self) -> None:
        inner = FakeSource([1, 2])
        source = ReconnectingSource(
            factory=lambda: inner, source_id="cam", uri="fake://", policy=self.policy()
        )
        with source:
            assert [source.read().index for _ in range(2)] == [0, 1]
        assert source.reconnects == 0

    def test_rebuilds_the_source_after_a_failure(self) -> None:
        sources = [FakeSource([1, SourceReadError("unplugged")]), FakeSource([2, 3])]
        source = ReconnectingSource(
            factory=lambda: sources.pop(0),
            source_id="cam",
            uri="fake://",
            policy=self.policy(),
            clock=ManualClock(),
        )
        with source:
            first = source.read()
            recovered = source.read()

        assert first.index == 0
        # Indices continue across the reconnect: a gap in time, not a restart.
        assert recovered.index == 1
        assert recovered.metadata["reconnected"] == 1
        assert source.reconnects == 1

    def test_gives_up_after_the_configured_attempts(self) -> None:
        def factory() -> FakeSource:
            failing = FakeSource([SourceReadError("still gone")])
            return failing

        source = ReconnectingSource(
            factory=factory,
            source_id="cam",
            uri="fake://",
            policy=self.policy(max_attempts=2),
            clock=ManualClock(),
        )
        source.open()
        with pytest.raises(SourceReadError, match="did not recover"):
            source.read()

    def test_backoff_is_capped(self) -> None:
        clock = ManualClock()

        def factory() -> FakeSource:
            return FakeSource([SourceReadError("gone")])

        source = ReconnectingSource(
            factory=factory,
            source_id="cam",
            uri="fake://",
            policy=self.policy(max_attempts=5, initial_delay_s=1.0, max_delay_s=4.0, backoff=2.0),
            clock=clock,
        )
        source.open()
        with pytest.raises(SourceReadError):
            source.read()
        assert clock.slept == [1.0, 2.0, 4.0, 4.0, 4.0]

    def test_exhaustion_is_not_treated_as_a_failure(self) -> None:
        """A finished source must not trigger a reconnect storm."""
        source = ReconnectingSource(
            factory=lambda: FakeSource([]),
            source_id="cam",
            uri="fake://",
            policy=self.policy(),
            clock=ManualClock(),
        )
        source.open()
        with pytest.raises(SourceExhausted):
            source.read()
        assert source.reconnects == 0

    def test_initial_open_failure_is_reported_immediately(self) -> None:
        def factory() -> FakeSource:
            failing = FakeSource()
            failing.open_error = SourceOpenError("camera absent")
            return failing

        source = ReconnectingSource(
            factory=factory, source_id="cam", uri="fake://", policy=self.policy()
        )
        with pytest.raises(SourceOpenError, match="camera absent"):
            source.open()
