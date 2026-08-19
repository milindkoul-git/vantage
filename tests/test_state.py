"""Entity state: hysteresis, dwell timing, path length and pruning.

No model, no weights, no runtime. State is derived arithmetic over the
tracker's output, so every property here is checkable against hand-built
tracks - which means these tests assert real behaviour rather than that the
code ran.
"""

from __future__ import annotations

import pytest

from vantage.core.errors import ConfigError
from vantage.perception.contracts import BoundingBox
from vantage.state import MotionState, StateEstimator, StateParams
from vantage.tracking.contracts import Track, TrackState, TrackingResult

HEIGHT = 100.0


def make_track(
    track_id: int = 1,
    *,
    center: tuple[float, float] = (100.0, 100.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    label: str = "person",
    time_since_update: int = 0,
) -> Track:
    """A track with a 50x100 box centred where asked."""
    cx, cy = center
    return Track(
        track_id=track_id,
        entity_id=f"{label}_{track_id}",
        box=BoundingBox(cx - 25.0, cy - HEIGHT / 2, cx + 25.0, cy + HEIGHT / 2),
        label=label,
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=10,
        hits=10,
        time_since_update=time_since_update,
        start_frame=0,
        last_frame=10,
        velocity=velocity,
    )


def step(estimator: StateEstimator, tracks, elapsed: float = 0.1, index: int = 0):
    return estimator.update(
        TrackingResult(
            tracks=tuple(tracks),
            source_id="test",
            frame_index=index,
            capture_wall=index * elapsed,
            frame_size=(640, 480),
            elapsed_s=elapsed,
        )
    )


def drive(estimator: StateEstimator, *, velocity, steps: int, elapsed: float = 0.1, start=0):
    """Advance one track for ``steps``, moving it consistently with ``velocity``."""
    vx, vy = velocity
    x, y = 100.0, 100.0
    last = None
    for i in range(steps):
        last = step(
            estimator,
            [make_track(center=(x, y), velocity=velocity)],
            elapsed=elapsed,
            index=start + i,
        )
        x += vx * elapsed
        y += vy * elapsed
    return last


class TestMotionState:
    def test_new_entity_is_unknown_not_stationary(self) -> None:
        """Absence of a velocity estimate is not evidence of stillness."""
        result = step(StateEstimator(), [make_track()])
        assert result.states[0].motion is MotionState.UNKNOWN

    def test_walking_entity_becomes_moving(self) -> None:
        estimator = StateEstimator()
        result = drive(estimator, velocity=(80.0, 0.0), steps=20)
        assert result.states[0].motion is MotionState.MOVING
        assert result.states[0].speed == pytest.approx(0.8, abs=0.01)

    def test_still_entity_becomes_stationary(self) -> None:
        estimator = StateEstimator()
        result = drive(estimator, velocity=(0.0, 0.0), steps=20)
        assert result.states[0].motion is MotionState.STATIONARY

    def test_speed_is_scale_invariant(self) -> None:
        """The same motion at two distances must read as the same speed.

        This is the whole reason speed is in heights per second: a box twice as
        tall is a person twice as close, and their pixel velocity doubles
        without them walking any faster.
        """
        near = StateEstimator()
        far = StateEstimator()
        for i in range(20):
            step(near, [make_track(center=(100.0, 100.0), velocity=(80.0, 0.0))], index=i)
        for i in range(20):
            small = Track(
                track_id=1,
                entity_id="person_1",
                box=BoundingBox(90.0, 75.0, 110.0, 125.0),  # half the height
                label="person",
                class_id=0,
                confidence=0.9,
                state=TrackState.CONFIRMED,
                age=10,
                hits=10,
                time_since_update=0,
                start_frame=0,
                last_frame=10,
                velocity=(40.0, 0.0),  # half the pixel speed
            )
            far_result = step(far, [small], index=i)
        near_result = step(near, [make_track(velocity=(80.0, 0.0))], index=21)
        assert near_result.states[0].speed == pytest.approx(far_result.states[0].speed, abs=0.01)


class TestHysteresis:
    def test_dead_band_does_not_flap(self) -> None:
        """A speed between the thresholds leaves the state alone.

        Without the dead band this alternates every frame, and each flip resets
        the dwell timer - which destroys the one measurement state exists for.
        """
        estimator = StateEstimator()
        drive(estimator, velocity=(80.0, 0.0), steps=20)  # settle into MOVING

        transitions = 0
        previous = MotionState.MOVING
        for i in range(40):
            # 0.11 h/s: inside the 0.08-0.15 dead band.
            result = step(estimator, [make_track(velocity=(11.0 * (-1) ** i, 0.0))], index=i)
            if result.states[0].motion is not previous:
                transitions += 1
                previous = result.states[0].motion
        assert transitions == 0
        assert previous is MotionState.MOVING

    def test_change_must_survive_min_state_s(self) -> None:
        """One fast frame does not flip a settled state."""
        estimator = StateEstimator(StateParams(min_state_s=0.5))
        drive(estimator, velocity=(0.0, 0.0), steps=20)
        assert estimator.update(
            TrackingResult(
                tracks=(make_track(velocity=(80.0, 0.0)),),
                source_id="t",
                frame_index=99,
                capture_wall=9.9,
                frame_size=(640, 480),
                elapsed_s=0.1,
            )
        ).states[0].motion is MotionState.STATIONARY

    def test_sustained_change_is_eventually_published(self) -> None:
        estimator = StateEstimator(StateParams(min_state_s=0.5))
        drive(estimator, velocity=(0.0, 0.0), steps=20)
        result = drive(estimator, velocity=(80.0, 0.0), steps=10, start=20)
        assert result.states[0].motion is MotionState.MOVING

    def test_dwell_resets_only_on_a_real_transition(self) -> None:
        estimator = StateEstimator()
        settled = drive(estimator, velocity=(0.0, 0.0), steps=30)
        before = settled.states[0].dwell_s
        after = drive(estimator, velocity=(0.0, 0.0), steps=10, start=30)
        assert after.states[0].dwell_s > before

    def test_min_age_holds_state_back(self) -> None:
        """Velocity on a brand-new track is dominated by its initial covariance."""
        estimator = StateEstimator(StateParams(min_age_s=1.0))
        result = drive(estimator, velocity=(80.0, 0.0), steps=5)  # 0.5s of life
        assert result.states[0].motion is MotionState.UNKNOWN


class TestDistance:
    def test_slow_motion_accumulates(self) -> None:
        """Regression: slow travel must not be discarded as jitter.

        The first implementation dropped any single step below a fixed fraction
        of a height, to keep a stationary box's wobble from accumulating. On a
        real clip a person crossing at 2px per frame against a 343px box
        produced steps of 0.006 heights - every one under the floor - so 120px
        of genuine travel recorded as exactly zero.
        """
        estimator = StateEstimator()
        result = drive(estimator, velocity=(20.0, 0.0), steps=60, elapsed=0.1)
        assert result.states[0].motion is MotionState.MOVING
        assert result.states[0].distance > 0.5

    def test_stationary_entity_accumulates_nothing(self) -> None:
        """The concern the discarded floor was meant to address, still handled."""
        estimator = StateEstimator()
        drive(estimator, velocity=(0.0, 0.0), steps=20)
        settled = estimator.update(
            TrackingResult(
                tracks=(make_track(center=(100.0, 100.0)),),
                source_id="t",
                frame_index=50,
                capture_wall=5.0,
                frame_size=(640, 480),
                elapsed_s=0.1,
            )
        )
        assert settled.states[0].motion is MotionState.STATIONARY
        distance = settled.states[0].distance
        for i in range(30):
            # Sub-pixel wobble around the same point.
            wobble = 0.4 if i % 2 else -0.4
            result = step(
                estimator, [make_track(center=(100.0 + wobble, 100.0))], index=51 + i
            )
        assert result.states[0].distance == pytest.approx(distance)

    def test_path_length_not_displacement(self) -> None:
        """Pacing back and forth accumulates rather than cancelling."""
        estimator = StateEstimator()
        out = drive(estimator, velocity=(60.0, 0.0), steps=25)
        back = drive(estimator, velocity=(-60.0, 0.0), steps=25, start=25)
        assert back.states[0].distance > out.states[0].distance


class TestBearing:
    def test_bearing_is_clockwise_from_up(self) -> None:
        estimator = StateEstimator()
        result = drive(estimator, velocity=(80.0, 0.0), steps=20)
        assert result.states[0].bearing_deg == pytest.approx(90.0, abs=1.0)

    def test_downward_motion_reads_as_180(self) -> None:
        """Image y grows downwards, so 'down the frame' is due south."""
        estimator = StateEstimator()
        result = drive(estimator, velocity=(0.0, 80.0), steps=20)
        assert result.states[0].bearing_deg == pytest.approx(180.0, abs=1.0)

    def test_stationary_entity_reports_no_bearing(self) -> None:
        """A direction of travel for something not travelling would be noise."""
        estimator = StateEstimator()
        result = drive(estimator, velocity=(0.0, 0.0), steps=20)
        assert result.states[0].bearing_deg is None


class TestLifecycle:
    def test_retired_tracks_are_pruned(self) -> None:
        """Phase 3 shipped an unbounded set of seen ids by accident once."""
        estimator = StateEstimator()
        step(estimator, [make_track(i) for i in range(5)])
        assert estimator.tracked == 5
        step(estimator, [make_track(0)], index=1)
        assert estimator.tracked == 1

    def test_state_survives_a_coasting_step(self) -> None:
        """A predicted box keeps its state rather than being re-measured."""
        estimator = StateEstimator()
        drive(estimator, velocity=(80.0, 0.0), steps=20)
        result = step(
            estimator,
            [make_track(velocity=(80.0, 0.0), time_since_update=3)],
            index=21,
        )
        assert result.states[0].motion is MotionState.MOVING
        assert result.states[0].observed is False

    def test_reset_clears_everything(self) -> None:
        estimator = StateEstimator()
        step(estimator, [make_track()])
        estimator.reset()
        assert estimator.tracked == 0


class TestObservationRecord:
    def test_shape_matches_the_specified_schema(self) -> None:
        estimator = StateEstimator()
        result = drive(estimator, velocity=(80.0, 0.0), steps=20)
        record = result.states[0].to_observation("camera_01", 1_700_000_000.0)

        assert record["camera_id"] == "camera_01"
        assert record["entity_id"] == "person_1"
        assert record["entity_type"] == "person"
        # Present and null: the seam an identity layer would later fill.
        assert "identity" in record and record["identity"] is None
        assert record["timestamp"].startswith("2023-")
        assert record["motion"]["state"] == "moving"
        assert record["observations"][0]["type"] == "moving"

    def test_record_is_json_serialisable(self) -> None:
        """It has to survive a socket or a database without a custom encoder."""
        import json

        estimator = StateEstimator()
        result = drive(estimator, velocity=(80.0, 0.0), steps=20)
        json.dumps(result.states[0].to_observation("cam", 1_700_000_000.0))


class TestParams:
    def test_inverted_thresholds_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="dead band"):
            StateParams(moving_above=0.1, stationary_below=0.5)

    def test_negative_values_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            StateParams(min_state_s=-1.0)

    def test_equal_thresholds_are_allowed(self) -> None:
        """A zero-width dead band is degenerate but not incoherent."""
        StateParams(moving_above=0.1, stationary_below=0.1)


class TestResultAggregates:
    def test_counts_and_moving(self) -> None:
        estimator = StateEstimator()
        for i in range(20):
            step(
                estimator,
                [
                    make_track(1, center=(100.0, 100.0), velocity=(80.0, 0.0)),
                    make_track(2, center=(300.0, 100.0), velocity=(0.0, 0.0)),
                ],
                index=i,
            )
        result = step(
            estimator,
            [
                make_track(1, velocity=(80.0, 0.0)),
                make_track(2, center=(300.0, 100.0), velocity=(0.0, 0.0)),
            ],
            index=21,
        )
        assert result.counts() == {"moving": 1, "stationary": 1}
        assert len(result.moving()) == 1
        assert set(result.by_track()) == {1, 2}
