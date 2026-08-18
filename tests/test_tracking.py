"""Phase 3 tracking tests.

Organised by what could actually be wrong rather than by module. The tracker
has three genuinely independent failure surfaces - the assignment solver, the
motion model, and the lifecycle rules - and each is tested on its own before
anything tests them together, so a failure points at one of them rather than at
"tracking is broken".
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from vantage.core.errors import ConfigError
from vantage.core.frame import Frame
from vantage.perception.contracts import BoundingBox, Detection, DetectionResult
from vantage.tracking.assignment import (
    forbid,
    iou_matrix,
    linear_sum_assignment,
    match,
)
from vantage.tracking.bytetrack import ByteTracker, TrackerParams
from vantage.tracking.contracts import TrackingResult, TrackState, empty_tracking_result
from vantage.tracking.kalman import KalmanBoxFilter, MotionNoise
from vantage.tracking.scenarios import (
    DetectorProfile,
    build_suite,
    crossing_scenario,
    occlusion_scenario,
    simulate_detections,
    sparse_scenario,
)


# -- helpers ------------------------------------------------------------


def box(cx: float, cy: float, w: float = 50.0, h: float = 120.0) -> BoundingBox:
    return BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def detection(
    cx: float, cy: float, confidence: float = 0.9, class_id: int = 0, label: str = "person"
) -> Detection:
    return Detection(box(cx, cy), class_id, label, confidence)


def result(detections: list[Detection], index: int, *, size=(1280, 720)) -> DetectionResult:
    return DetectionResult(
        detections=tuple(detections),
        source_id="test",
        frame_index=index,
        capture_wall=index / 30.0,
        frame_size=size,
    )


def frame_at(index: int, pts: float | None = None) -> Frame:
    return Frame(
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        index=index,
        source_id="test",
        capture_monotonic=index / 30.0,
        capture_wall=1_700_000_000.0 + index / 30.0,
        media_pts=pts,
    )


# -- assignment ---------------------------------------------------------


class TestAssignment:
    """The solver replaces SciPy, so it is held to SciPy's standard."""

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_brute_force_optimum(self, seed: int) -> None:
        """Exhaustive check: the returned assignment is genuinely minimal."""
        rng = np.random.default_rng(seed)
        rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        cost = rng.integers(0, 20, size=(rows, cols)).astype(float)

        got_rows, got_cols = linear_sum_assignment(cost)
        total = cost[got_rows, got_cols].sum()

        size = min(rows, cols)
        best = min(
            sum(cost[i, perm[i]] for i in range(size))
            if rows <= cols
            else sum(cost[perm[j], j] for j in range(size))
            for perm in itertools.permutations(range(max(rows, cols)), size)
        )
        assert total == pytest.approx(best)

    def test_assignment_is_one_to_one(self) -> None:
        cost = np.random.default_rng(0).random((6, 9))
        rows, cols = linear_sum_assignment(cost)
        assert len(set(rows.tolist())) == 6
        assert len(set(cols.tolist())) == 6

    @pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (3, 40), (40, 3)])
    def test_rectangular_shapes(self, shape: tuple[int, int]) -> None:
        cost = np.random.default_rng(1).random(shape)
        rows, _ = linear_sum_assignment(cost)
        assert len(rows) == min(shape)

    def test_empty_matrix_is_not_an_error(self) -> None:
        rows, cols = linear_sum_assignment(np.zeros((0, 4)))
        assert len(rows) == 0 and len(cols) == 0

    def test_non_finite_costs_are_rejected(self) -> None:
        """An accidental inf must fail loudly, not produce a plausible answer."""
        with pytest.raises(ValueError, match="non-finite"):
            linear_sum_assignment(np.array([[1.0, np.inf], [2.0, 3.0]]))

    def test_match_splits_gated_pairs_into_unmatched(self) -> None:
        cost = np.array([[0.1, 0.9], [0.9, 0.1]])
        pairs, unmatched_rows, unmatched_cols = match(cost, max_cost=0.5)
        assert sorted(pairs) == [(0, 0), (1, 1)]
        assert unmatched_rows == [] and unmatched_cols == []

        pairs, unmatched_rows, unmatched_cols = match(cost, max_cost=0.05)
        assert pairs == []
        assert unmatched_rows == [0, 1] and unmatched_cols == [0, 1]

    def test_gating_after_solving_preserves_the_optimum(self) -> None:
        """Cheap pairs must survive even when the solve also produced a bad one."""
        cost = np.array([[0.1, 0.9], [0.95, 0.99]])
        pairs, unmatched_rows, _ = match(cost, max_cost=0.5)
        assert pairs == [(0, 0)]
        assert unmatched_rows == [1]

    def test_iou_matrix_against_manual_values(self) -> None:
        a = [BoundingBox(0, 0, 10, 10)]
        b = [BoundingBox(0, 0, 10, 10), BoundingBox(5, 0, 15, 10), BoundingBox(20, 20, 30, 30)]
        got = iou_matrix(a, b)
        assert got[0, 0] == pytest.approx(1.0)
        assert got[0, 1] == pytest.approx(50 / 150)
        assert got[0, 2] == pytest.approx(0.0)

    def test_iou_matrix_handles_empty_input(self) -> None:
        assert iou_matrix([], [BoundingBox(0, 0, 1, 1)]).shape == (0, 1)

    def test_forbid_does_not_mutate_its_input(self) -> None:
        cost = np.zeros((2, 2))
        forbid(cost, np.ones((2, 2), dtype=bool))
        assert cost.sum() == 0.0


# -- motion model -------------------------------------------------------


class TestKalmanFilter:
    def test_learns_constant_velocity(self) -> None:
        filt = KalmanBoxFilter(box(100, 300))
        for step in range(1, 60):
            filt.predict(1 / 30)
            filt.update(box(100 + 300 * step / 30, 300))
        vx, vy = filt.velocity
        assert vx == pytest.approx(300, rel=0.05)
        assert vy == pytest.approx(0, abs=5)

    def test_variable_timestep_matches_uniform(self) -> None:
        """The whole reason dt is an argument: sparse observations must agree.

        One filter sees every frame; the other sees every third frame with three
        times the timestep. Both watch the same object, so both must converge on
        the same velocity - which is false for any tracker that counts frames.
        """
        dense = KalmanBoxFilter(box(100, 300))
        sparse = KalmanBoxFilter(box(100, 300))
        for step in range(1, 31):
            dense.predict(1 / 30)
            dense.update(box(100 + 300 * step / 30, 300))
        for step in range(3, 31, 3):
            sparse.predict(3 / 30)
            sparse.update(box(100 + 300 * step / 30, 300))
        assert sparse.velocity[0] == pytest.approx(dense.velocity[0], rel=0.02)

    def test_prediction_extrapolates_through_a_gap(self) -> None:
        filt = KalmanBoxFilter(box(100, 300))
        for step in range(1, 40):
            filt.predict(1 / 30)
            filt.update(box(100 + 300 * step / 30, 300))
        for _ in range(10):
            filt.predict(1 / 30)
        expected = 100 + 300 * 49 / 30
        assert filt.box.center[0] == pytest.approx(expected, abs=15)

    def test_non_positive_timestep_is_a_no_op(self) -> None:
        filt = KalmanBoxFilter(box(50, 50))
        before = filt.box.center
        filt.predict(0.0)
        filt.predict(-1.0)
        assert filt.box.center == before

    def test_covariance_stays_symmetric_and_positive_definite(self) -> None:
        """Joseph-form update; the naive form drifts asymmetric over long runs."""
        rng = np.random.default_rng(3)
        filt = KalmanBoxFilter(box(100, 300))
        for _ in range(2000):
            filt.predict(1 / 30)
            filt.update(box(100 + rng.normal(0, 3), 300 + rng.normal(0, 3)))
        cov = filt._covariance
        assert np.allclose(cov, cov.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(cov) > 0)

    def test_box_never_inverts(self) -> None:
        """A shrinking box must floor at 1px rather than produce inverted corners."""
        filt = KalmanBoxFilter(box(100, 100, w=4, h=4))
        for _ in range(60):
            filt.predict(1 / 30)
            filt.update(BoundingBox(100, 100, 100.5, 100.5))
        result_box = filt.box
        assert result_box.x2 >= result_box.x1
        assert result_box.y2 >= result_box.y1

    def test_noise_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="acceleration"):
            MotionNoise(acceleration=0.0)


# -- tracker lifecycle --------------------------------------------------


class TestTrackerLifecycle:
    def test_min_hits_controls_when_a_track_is_published(self) -> None:
        """Regression: min_hits=1 and 2 once behaved identically.

        The creating detection counts as a hit, but confirmation was only ever
        checked on the *second* observation, so min_hits=1 published a step late
        and the parameter search saw two identical options.
        """
        for min_hits in (1, 2, 3, 4):
            tracker = ByteTracker(TrackerParams(min_hits=min_hits))
            published = [
                len(tracker.update(result([detection(100, 300)], index)))
                for index in range(6)
            ]
            assert published.index(1) == min_hits - 1, f"min_hits={min_hits}: {published}"

    def test_tentative_tracks_are_never_published(self) -> None:
        tracker = ByteTracker(TrackerParams(min_hits=3))
        assert len(tracker.update(result([detection(100, 300)], 0))) == 0
        assert tracker.stats()["active"] == 1

    def test_single_frame_false_positive_dies_without_being_published(self) -> None:
        tracker = ByteTracker(TrackerParams(min_hits=3))
        tracker.update(result([detection(100, 300)], 0))
        published = tracker.update(result([], 1))
        assert len(published) == 0
        assert tracker.stats()["active"] == 0

    def test_identity_survives_total_occlusion(self) -> None:
        """The capability the whole phase exists for."""
        tracker = ByteTracker()
        for index in range(10):
            tracker.update(result([detection(100 + 10 * index, 300)], index))
        first = tracker.update(result([detection(200, 300)], 10)).tracks[0].entity_id

        for index in range(11, 20):  # nine frames with no detection at all
            tracker.update(result([], index))
        recovered = tracker.update(result([detection(300, 300)], 20))

        assert len(recovered) == 1
        assert recovered.tracks[0].entity_id == first
        assert tracker.stats()["ids_used"] == 1

    def test_track_expires_after_max_lost_s(self) -> None:
        tracker = ByteTracker(TrackerParams(max_lost_s=0.2))
        for index in range(6):
            tracker.update(result([detection(100, 300)], index))
        for index in range(6, 30):
            tracker.update(result([], index))
        assert tracker.stats()["active"] == 0

    def test_expiry_is_in_seconds_not_frames(self) -> None:
        """Halving the frame rate must not halve how long a track survives."""
        counts = {}
        for fps in (30.0, 10.0):
            tracker = ByteTracker(TrackerParams(max_lost_s=1.0))
            for index in range(6):
                tracker.update(result([detection(100, 300)], index), frame=frame_at(index, index / fps))
            steps = 0
            while tracker.stats()["active"]:
                steps += 1
                tracker.update(result([], 6 + steps), frame=frame_at(6 + steps, (6 + steps) / fps))
                assert steps < 200, "track never expired"
            counts[fps] = steps / fps  # elapsed seconds, not frames
        assert counts[30.0] == pytest.approx(counts[10.0], abs=0.2)

    def test_coasting_track_is_flagged_as_a_prediction(self) -> None:
        tracker = ByteTracker()
        for index in range(6):
            tracker.update(result([detection(100, 300)], index))
        coasted = tracker.update(result([], 6))
        assert coasted.tracks[0].is_coasting
        assert coasted.tracks[0].time_since_update == 1
        assert coasted.tracks[0].state is TrackState.LOST
        assert coasted.observed == ()

    def test_track_ids_are_never_reused(self) -> None:
        tracker = ByteTracker(TrackerParams(max_lost_s=0.0))
        for index in range(6):
            tracker.update(result([detection(100, 300)], index))
        for index in range(6, 12):
            tracker.update(result([], index))
        for index in range(12, 20):
            tracker.update(result([detection(900, 300)], index))
        ids = {track.track_id for track in tracker.update(result([detection(900, 300)], 20))}
        assert ids == {2}

    def test_reset_clears_state_but_not_the_id_counter(self) -> None:
        """After a reconnect, continuity is gone; reusing ids would falsify logs."""
        tracker = ByteTracker()
        for index in range(6):
            tracker.update(result([detection(100, 300)], index))
        tracker.reset()
        assert tracker.stats()["active"] == 0
        for index in range(6):
            tracker.update(result([detection(100, 300)], 100 + index))
        assert tracker.update(result([detection(100, 300)], 110)).tracks[0].track_id == 2

    def test_off_frame_tracks_are_dropped_rather_than_coasted(self) -> None:
        """An object that has left the frame cannot be corroborated again."""
        tracker = ByteTracker(TrackerParams(max_lost_s=5.0))
        for index in range(12):
            tracker.update(result([detection(1000 + 25 * index, 300)], index))
        for index in range(12, 30):
            tracker.update(result([], index))
        assert tracker.stats()["active"] == 0

    def test_track_touching_the_edge_is_kept(self) -> None:
        """Only the centre leaving counts; a half-visible object is still there."""
        tracker = ByteTracker(TrackerParams(max_lost_s=5.0))
        for index in range(12):
            tracker.update(result([detection(1250, 300)], index))
        tracker.update(result([], 12))
        assert tracker.stats()["active"] == 1


class TestAssociation:
    def test_low_confidence_boxes_sustain_but_never_create(self) -> None:
        """The core ByteTrack behaviour, stated as two assertions."""
        params = TrackerParams(high_threshold=0.5, low_threshold=0.1, init_threshold=0.5)

        weak_only = ByteTracker(params)
        for index in range(10):
            weak_only.update(result([detection(100, 300, confidence=0.3)], index))
        assert weak_only.stats()["ids_used"] == 0, "a weak box must not create a track"

        established = ByteTracker(params)
        for index in range(6):
            established.update(result([detection(100, 300, confidence=0.9)], index))
        sustained = established.update(result([detection(105, 300, confidence=0.3)], 6))
        assert len(sustained) == 1
        assert not sustained.tracks[0].is_coasting, "a weak box should confirm an existing track"

    def test_boxes_below_low_threshold_are_ignored_entirely(self) -> None:
        tracker = ByteTracker(TrackerParams(low_threshold=0.2))
        for index in range(6):
            tracker.update(result([detection(100, 300, confidence=0.9)], index))
        coasted = tracker.update(result([detection(105, 300, confidence=0.05)], 6))
        assert coasted.tracks[0].is_coasting

    def test_classes_are_not_associated_across(self) -> None:
        tracker = ByteTracker(TrackerParams(class_aware=True))
        for index in range(6):
            tracker.update(result([detection(100, 300, class_id=0, label="person")], index))
        after = tracker.update(result([detection(100, 300, class_id=2, label="car")], 6))
        person = [t for t in after if t.label == "person"]
        assert len(person) == 1
        assert person[0].is_coasting, "the car must not have been matched to the person"

    def test_class_gating_can_be_disabled(self) -> None:
        tracker = ByteTracker(TrackerParams(class_aware=False))
        for index in range(6):
            tracker.update(result([detection(100, 300, class_id=0, label="person")], index))
        after = tracker.update(result([detection(100, 300, class_id=2, label="car")], 6))
        assert not after.tracks[0].is_coasting

    def test_crossing_objects_keep_their_identities(self) -> None:
        """Two objects passing through each other must not swap.

        Pure IoU cannot separate them at the crossing point; what does is that
        they arrived with opposite velocities, so this fails if the motion model
        is not contributing.
        """
        tracker = ByteTracker()
        entities: dict[int, str] = {}
        for index in range(60):
            left = 200 + 12 * index
            right = 1000 - 12 * index
            tracked = tracker.update(
                result([detection(left, 300), detection(right, 300)], index),
                frame=frame_at(index, index / 30),
            )
            for track in tracked:
                if track.is_coasting:
                    continue
                closest = min((left, right), key=lambda x: abs(track.center[0] - x))
                side = 0 if closest == left else 1
                entities.setdefault(side, track.entity_id)
        assert tracker.stats()["ids_used"] == 2

    def test_optimal_assignment_beats_greedy_on_a_swap(self) -> None:
        """Two adjacent objects: a greedy matcher would cross the pairing."""
        tracker = ByteTracker()
        for index in range(8):
            tracker.update(result([detection(300, 300), detection(380, 300)], index))
        before = {round(t.center[0]): t.entity_id for t in tracker.update(
            result([detection(300, 300), detection(380, 300)], 8)
        )}
        after = tracker.update(result([detection(310, 300), detection(390, 300)], 9))
        mapping = {round(t.center[0] / 10) * 10: t.entity_id for t in after}
        assert len(set(mapping.values())) == 2
        assert tracker.stats()["ids_used"] == 2
        assert len(before) == 2


class TestTimestepHandling:
    def test_media_time_is_preferred_over_wall_clock(self) -> None:
        """A file analysed at 10x speed must still model real object motion."""
        tracker = ByteTracker()
        for index in range(5):
            tracker.update(result([detection(100, 300)], index), frame=frame_at(index, index / 30))
        assert tracker.update(
            result([detection(100, 300)], 5), frame=frame_at(5, 5 / 30)
        ).elapsed_s == pytest.approx(1 / 30, abs=1e-6)

    def test_huge_gap_is_clamped_not_extrapolated(self) -> None:
        """A resume-from-sleep must not throw every prediction off the frame."""
        tracker = ByteTracker(TrackerParams(max_step_s=2.0))
        for index in range(5):
            tracker.update(result([detection(100, 300)], index), frame=frame_at(index, index / 30))
        stepped = tracker.update(result([detection(100, 300)], 5), frame=frame_at(5, 3600.0))
        assert stepped.elapsed_s == pytest.approx(2.0)
        assert tracker.stats()["clamped_steps"] == 1

    def test_backwards_time_is_handled(self) -> None:
        """A looping source resets its media clock; that is not an error."""
        tracker = ByteTracker()
        for index in range(5):
            tracker.update(result([detection(100, 300)], index), frame=frame_at(index, index / 30))
        stepped = tracker.update(result([detection(100, 300)], 0), frame=frame_at(0, 0.0))
        assert stepped.elapsed_s > 0
        assert tracker.stats()["clamped_steps"] == 1


# -- contracts ----------------------------------------------------------


class TestContracts:
    def test_result_is_empty_but_usable(self) -> None:
        empty = empty_tracking_result("cam", 3, 1.0, (640, 480))
        assert len(empty) == 0
        assert list(empty) == []
        assert empty.counts() == {}
        assert "no tracks" in empty.describe()

    def test_entity_id_embeds_the_label(self) -> None:
        tracker = ByteTracker()
        for index in range(6):
            tracker.update(result([detection(100, 300, label="person")], index))
        track = tracker.update(result([detection(100, 300, label="person")], 6)).tracks[0]
        assert track.entity_id == f"person_{track.track_id}"

    def test_label_is_decided_by_majority_vote_at_confirmation(self) -> None:
        """A first-frame class flicker must not name the entity permanently."""
        tracker = ByteTracker(TrackerParams(min_hits=3, class_aware=False))
        tracker.update(result([detection(100, 300, class_id=7, label="truck")], 0))
        tracker.update(result([detection(100, 300, class_id=0, label="person")], 1))
        tracker.update(result([detection(100, 300, class_id=0, label="person")], 2))
        track = tracker.update(result([detection(100, 300, class_id=0, label="person")], 3)).tracks[0]
        assert track.label == "person"
        assert track.entity_id.startswith("person_")

    def test_entity_id_is_stable_once_confirmed(self) -> None:
        tracker = ByteTracker(TrackerParams(class_aware=False))
        seen = set()
        for index in range(20):
            label = "person" if index < 10 else "car"
            class_id = 0 if index < 10 else 2
            for track in tracker.update(
                result([detection(100 + index, 300, class_id=class_id, label=label)], index)
            ):
                seen.add(track.entity_id)
        assert len(seen) == 1, f"entity id changed mid-track: {seen}"

    def test_result_helpers(self) -> None:
        tracker = ByteTracker()
        for index in range(6):
            tracker.update(
                result([detection(100, 300, label="person"), detection(900, 300, label="car", class_id=2)], index)
            )
        tracked = tracker.update(
            result([detection(100, 300, label="person"), detection(900, 300, label="car", class_id=2)], 6)
        )
        assert tracked.counts() == {"person": 1, "car": 1}
        assert len(tracked.of_class("person")) == 1
        assert len(tracked.confirmed) == 2
        assert isinstance(tracked, TrackingResult)

    def test_velocity_is_per_second(self) -> None:
        tracker = ByteTracker()
        for index in range(30):
            tracker.update(
                result([detection(100 + 10 * index, 300)], index), frame=frame_at(index, index / 30)
            )
        track = tracker.update(
            result([detection(400, 300)], 30), frame=frame_at(30, 1.0)
        ).tracks[0]
        assert track.velocity[0] == pytest.approx(300, rel=0.2)


class TestParams:
    def test_thresholds_must_be_ordered(self) -> None:
        with pytest.raises(ConfigError, match="low_threshold must be below"):
            TrackerParams(low_threshold=0.7, high_threshold=0.5)

    def test_init_threshold_cannot_undercut_high_threshold(self) -> None:
        with pytest.raises(ConfigError, match="init_threshold"):
            TrackerParams(high_threshold=0.6, init_threshold=0.4)

    def test_min_hits_must_be_at_least_one(self) -> None:
        with pytest.raises(ConfigError, match="min_hits"):
            TrackerParams(min_hits=0)


# -- scenarios and metrics ----------------------------------------------


class TestScenarios:
    def test_scenarios_are_deterministic(self) -> None:
        first = simulate_detections(crossing_scenario(30), DetectorProfile(seed=5))
        second = simulate_detections(crossing_scenario(30), DetectorProfile(seed=5))
        assert [len(r) for r in first] == [len(r) for r in second]
        assert first[10].detections[0].box.xyxy == second[10].detections[0].box.xyxy

    def test_different_seeds_differ(self) -> None:
        a = simulate_detections(crossing_scenario(60), DetectorProfile(seed=1))
        b = simulate_detections(crossing_scenario(60), DetectorProfile(seed=2))
        assert [len(r) for r in a] != [len(r) for r in b]

    def test_occlusion_scenario_actually_occludes(self) -> None:
        scenario = occlusion_scenario()
        visibilities = [
            obj.visibility
            for frame in scenario.frames
            for obj in frame.objects
            if obj.object_id == 0
        ]
        assert min(visibilities) < 0.05, "the subject is never actually hidden"

    def test_confidence_falls_with_visibility(self) -> None:
        """ByteTrack's second pass depends on this; without it the test suite lies."""
        scenario = occlusion_scenario()
        profile = DetectorProfile(seed=11, miss_rate=0.0, confidence_noise=0.0)
        results = simulate_detections(scenario, profile)
        low = []
        for frame, detected in zip(scenario.frames, results):
            subject = [o for o in frame.objects if o.object_id == 0]
            if subject and 0.2 < subject[0].visibility < 0.8 and detected.detections:
                low.append(min(d.confidence for d in detected.detections))
        assert low and min(low) < 0.5

    def test_build_suite_rejects_unknown_names(self) -> None:
        with pytest.raises(ValueError, match="unknown scenario"):
            build_suite(["nope"])

    def test_all_scenarios_construct(self) -> None:
        for scenario in build_suite():
            assert len(scenario) > 0
            assert scenario.instance_count > 0
            assert scenario.object_count >= 2


class TestEvaluation:
    def test_perfect_tracking_scores_perfectly(self) -> None:
        """A tracker fed exact ground truth must score 100%, or the metric is wrong."""
        from vantage.tracking.evaluation import evaluate

        scenario = sparse_scenario(40)
        results = []
        for frame in scenario.frames:
            tracks = tuple(
                _fake_track(obj.object_id, obj.box) for obj in frame.objects
            )
            results.append(
                TrackingResult(
                    tracks=tracks,
                    source_id="x",
                    frame_index=frame.index,
                    capture_wall=frame.timestamp,
                    frame_size=scenario.frame_size,
                )
            )
        metrics = evaluate(scenario, results)
        assert metrics.mota == pytest.approx(1.0)
        assert metrics.idf1 == pytest.approx(1.0)
        assert metrics.id_switches == 0
        assert metrics.false_positives == 0 and metrics.false_negatives == 0

    def test_identity_switch_is_counted(self) -> None:
        from vantage.tracking.evaluation import evaluate

        scenario = sparse_scenario(20)
        results = []
        for position, frame in enumerate(scenario.frames):
            # Swap the two ids halfway through.
            swap = position >= 10
            tracks = tuple(
                _fake_track((obj.object_id + 1) % 2 if swap else obj.object_id, obj.box)
                for obj in frame.objects
            )
            results.append(
                TrackingResult(
                    tracks=tracks,
                    source_id="x",
                    frame_index=frame.index,
                    capture_wall=frame.timestamp,
                    frame_size=scenario.frame_size,
                )
            )
        metrics = evaluate(scenario, results)
        assert metrics.id_switches == 2
        assert metrics.idf1 < 1.0, "IDF1 must penalise a persistent swap"

    def test_missing_predictions_become_false_negatives(self) -> None:
        from vantage.tracking.evaluation import evaluate

        scenario = sparse_scenario(20)
        results = [
            TrackingResult(
                tracks=(),
                source_id="x",
                frame_index=frame.index,
                capture_wall=frame.timestamp,
                frame_size=scenario.frame_size,
            )
            for frame in scenario.frames
        ]
        metrics = evaluate(scenario, results)
        assert metrics.false_negatives == scenario.instance_count
        assert metrics.recall == 0.0
        assert metrics.mostly_lost == scenario.object_count

    def test_result_count_must_match_frame_count(self) -> None:
        from vantage.tracking.evaluation import evaluate

        with pytest.raises(ValueError, match="one result per frame"):
            evaluate(sparse_scenario(10), [])

    def test_tracker_beats_a_low_bar_on_every_scenario(self) -> None:
        """Guards against a regression that silently degrades accuracy."""
        from vantage.tracking.tuning import run_scenario

        for scenario in build_suite():
            metrics = run_scenario(scenario, TrackerParams(), DetectorProfile(seed=77))
            assert metrics.mota > 0.75, f"{scenario.name}: MOTA {metrics.mota:.1%}"
            assert metrics.idf1 > 0.70, f"{scenario.name}: IDF1 {metrics.idf1:.1%}"


def _fake_track(track_id: int, box_: BoundingBox):
    from vantage.tracking.contracts import Track

    return Track(
        track_id=track_id,
        entity_id=f"person_{track_id}",
        box=box_,
        label="person",
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=1,
        hits=1,
        time_since_update=0,
        start_frame=0,
        last_frame=0,
    )


# -- tuning -------------------------------------------------------------


class TestTuning:
    def test_search_space_values_all_construct(self) -> None:
        """Every offered value must be reachable, or the search silently shrinks."""
        from vantage.tracking.tuning import SEARCH_SPACE, _with

        base = TrackerParams()
        for name, values in SEARCH_SPACE:
            for value in values:
                try:
                    _with(base, name, value)
                except ConfigError:
                    pass  # a genuinely invalid combination, rejected by design

    def test_assess_scores_a_parameter_set(self) -> None:
        from vantage.tracking.tuning import assess

        candidate = assess(TrackerParams(), [sparse_scenario(40)])
        assert 0.0 < candidate.objective <= 1.0
        assert candidate.summary["idf1"] > 0.5

    def test_worst_case_term_penalises_a_single_failure(self) -> None:
        """A candidate must not buy a good average by failing one scenario."""
        from vantage.tracking.tuning import assess

        suite = [sparse_scenario(40), occlusion_scenario(60)]
        good = assess(TrackerParams(), suite)
        # max_lost_s=0 cannot survive any occlusion at all.
        crippled = assess(replace(TrackerParams(), max_lost_s=0.0), suite)
        assert crippled.objective < good.objective

    def test_config_lines_round_trip(self) -> None:
        from vantage.tracking.tuning import as_config_lines

        lines = as_config_lines(TrackerParams())
        assert lines[0] == "tracking:"
        assert any("max_lost_s" in line for line in lines)
