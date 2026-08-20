"""Pose: contracts, the RTMPose transform, posture rules and the engine budget.

Runs with no weights and no inference runtime. The decode is checked by
round-trip - a peak placed at a known bin must come back at the pixel it
encodes - which pins preprocessing and postprocessing against each other
without needing a model to agree with.

The posture tests build skeletons by hand. That is worth being honest about:
they prove the geometry rules do what the docstring says, not that a real
person in a real room is classified correctly. Only footage can show the
second, and the limits of the rules are written down in
:mod:`vantage.pose.posture` rather than implied by a passing test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import FakePoseBackend
from vantage.core.errors import ConfigError
from vantage.core.frame import Frame
from vantage.perception.contracts import BoundingBox
from vantage.pose.adapter import BBOX_PADDING, RTMPoseAdapter, _center_scale
from vantage.pose.contracts import (
    FACE_KEYPOINTS,
    KEYPOINT_NAMES,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    Keypoint,
    Pose,
    PoseResult,
    Posture,
)
from vantage.pose.engine import PoseEngine
from vantage.pose.posture import classify
from vantage.tracking.contracts import Track, TrackingResult, TrackState

INPUT_SIZE = (256, 192)  # (height, width)
BOX = BoundingBox(100.0, 50.0, 300.0, 450.0)


def make_pose(points: dict[int, tuple[float, float]], *, score: float = 0.9, box=None) -> Pose:
    """A skeleton with the named joints placed and everything else invisible."""
    keypoints = tuple(
        Keypoint(*points[i], score) if i in points else Keypoint(0.0, 0.0, 0.0)
        for i in range(len(KEYPOINT_NAMES))
    )
    return Pose(keypoints=keypoints, track_id=1, entity_id="person_1", box=box or BOX)


def upright(hip_y: float, knee_y: float, ankle_y: float) -> dict[int, tuple[float, float]]:
    """Shoulders at y=100 with a 100px torso, and legs placed as asked."""
    return {
        LEFT_SHOULDER: (140.0, 100.0),
        RIGHT_SHOULDER: (180.0, 100.0),
        LEFT_HIP: (145.0, hip_y),
        RIGHT_HIP: (175.0, hip_y),
        LEFT_KNEE: (145.0, knee_y),
        RIGHT_KNEE: (175.0, knee_y),
        LEFT_ANKLE: (145.0, ankle_y),
        RIGHT_ANKLE: (175.0, ankle_y),
    }


def make_track(track_id: int = 1, *, box=None, label="person", time_since_update=0) -> Track:
    return Track(
        track_id=track_id,
        entity_id=f"{label}_{track_id}",
        box=box or BOX,
        label=label,
        class_id=0,
        confidence=0.9,
        state=TrackState.CONFIRMED,
        age=10,
        hits=10,
        time_since_update=time_since_update,
        start_frame=0,
        last_frame=10,
    )


def make_frame(width: int = 640, height: int = 480) -> Frame:
    return Frame(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        index=1,
        source_id="test",
        capture_monotonic=1.0,
        capture_wall=1.0,
    )


def tracking_of(*tracks) -> TrackingResult:
    return TrackingResult(
        tracks=tuple(tracks),
        source_id="test",
        frame_index=1,
        capture_wall=1.0,
        frame_size=(640, 480),
    )


class TestContracts:
    def test_keypoint_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError, match=r"confidence"):
            Keypoint(1.0, 2.0, 1.5)

    def test_pose_rejects_a_wrong_keypoint_count(self) -> None:
        with pytest.raises(ValueError, match=r"keypoints"):
            Pose(
                keypoints=(Keypoint(0.0, 0.0, 0.5),) * 5,
                track_id=1,
                entity_id="person_1",
                box=BOX,
            )

    def test_visible_reports_full_coco_indices(self) -> None:
        pose = make_pose({LEFT_KNEE: (10.0, 10.0)})
        assert pose.visible(0.3) == (LEFT_KNEE,)

    def test_dropping_face_keypoints_renumbers_transparently(self) -> None:
        """A caller asking for a knee gets a knee, not an off-by-five elbow."""
        keypoints = tuple(
            Keypoint(float(i), float(i), 0.9)
            for i in range(len(KEYPOINT_NAMES))
            if i not in FACE_KEYPOINTS
        )
        pose = Pose(keypoints=keypoints, track_id=1, entity_id="p_1", box=BOX)

        assert not pose.has_face_keypoints
        assert pose.keypoint(LEFT_KNEE).x == float(LEFT_KNEE)
        assert pose.visible(0.3) == tuple(
            i for i in range(len(KEYPOINT_NAMES)) if i not in FACE_KEYPOINTS
        )

    def test_a_dropped_face_keypoint_is_none_not_a_zero(self) -> None:
        """ "Not collected" must stay distinguishable from "not visible"."""
        keypoints = tuple(
            Keypoint(1.0, 1.0, 0.9)
            for i in range(len(KEYPOINT_NAMES))
            if i not in FACE_KEYPOINTS
        )
        pose = Pose(keypoints=keypoints, track_id=1, entity_id="p_1", box=BOX)
        assert pose.keypoint(0) is None

    def test_result_aggregates(self) -> None:
        poses = (
            make_pose(upright(200.0, 300.0, 400.0)),
            make_pose(upright(200.0, 210.0, 290.0)),
        )
        classified = tuple(
            Pose(
                keypoints=p.keypoints,
                track_id=i,
                entity_id=f"person_{i}",
                box=BOX,
                posture=classify(p).posture,
            )
            for i, p in enumerate(poses)
        )
        result = PoseResult(
            poses=classified,
            source_id="t",
            frame_index=1,
            capture_wall=1.0,
            frame_size=(640, 480),
            people_seen=5,
        )
        assert result.counts() == {"standing": 1, "sitting": 1}
        assert result.skipped == 3
        assert set(result.by_track()) == {0, 1}


class TestPreprocessing:
    def test_aspect_is_corrected_to_the_model_input(self) -> None:
        """A wide box must be heightened, not squashed into a tall input."""
        wide = BoundingBox(0.0, 0.0, 400.0, 100.0)
        _, scale = _center_scale(wide, 192 / 256)
        assert scale[0] / scale[1] == pytest.approx(192 / 256, abs=1e-5)
        # Correction only ever grows the region, never crops it.
        assert scale[0] >= 400.0 * BBOX_PADDING - 1e-3

    def test_padding_expands_the_region(self) -> None:
        _, scale = _center_scale(BoundingBox(0.0, 0.0, 300.0, 400.0), 192 / 256)
        assert scale[1] == pytest.approx(400.0 * BBOX_PADDING)

    def test_tensor_shape_and_layout(self) -> None:
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        assert prepared.tensor.shape == (1, 3, 256, 192)
        assert prepared.tensor.dtype == np.float32

    def test_normalisation_is_applied(self) -> None:
        """A mid-grey image must not come back as raw 0-255 values."""
        adapter = RTMPoseAdapter(INPUT_SIZE)
        grey = np.full((480, 640, 3), 128, np.uint8)
        prepared = adapter.preprocess(grey, BOX)
        assert abs(float(prepared.tensor.mean())) < 1.0

    def test_static_shapes_pin_batch_to_one(self) -> None:
        """Phase 3.5 established what an unpinned dimension costs on OpenVINO."""
        assert RTMPoseAdapter(INPUT_SIZE).static_input_shapes() == {"input": [1, 3, 256, 192]}


class TestDecoding:
    def test_centre_bins_decode_to_the_box_centre(self) -> None:
        """The round trip that pins preprocessing and decoding to each other.

        A peak at the exact middle bin of both axes encodes the middle of the
        cropped region, which - because padding and aspect correction are both
        centred on the box - is the box centre in the original frame.
        """
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        backend = FakePoseBackend()
        keypoints = adapter.postprocess(backend.run(prepared.tensor), prepared)

        cx, cy = BOX.center
        assert keypoints[0].x == pytest.approx(cx, abs=0.5)
        assert keypoints[0].y == pytest.approx(cy, abs=0.5)

    def test_zero_bins_decode_to_the_padded_corner(self) -> None:
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        backend = FakePoseBackend(peaks=[(0, 0)] * 17)
        keypoints = adapter.postprocess(backend.run(prepared.tensor), prepared)

        expected_x = prepared.center[0] - prepared.scale[0] / 2
        expected_y = prepared.center[1] - prepared.scale[1] / 2
        assert keypoints[0].x == pytest.approx(float(expected_x), abs=0.5)
        assert keypoints[0].y == pytest.approx(float(expected_y), abs=0.5)

    def test_score_is_the_weaker_axis(self) -> None:
        """A confident column paired with a flat row is not a located joint."""
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        simcc_x = np.zeros((1, 17, 384), np.float32)
        simcc_y = np.zeros((1, 17, 512), np.float32)
        simcc_x[0, :, 100] = 0.9
        simcc_y[0, :, 100] = 0.2
        keypoints = adapter.postprocess([simcc_x, simcc_y], prepared)
        assert keypoints[0].confidence == pytest.approx(0.2)

    def test_out_of_range_scores_are_clipped(self) -> None:
        """The SimCC head is not normalised; measured peaks reach 1.06."""
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        backend = FakePoseBackend(score=1.4)
        keypoints = adapter.postprocess(backend.run(prepared.tensor), prepared)
        assert keypoints[0].confidence == 1.0

    def test_face_keypoints_are_never_constructed_when_disabled(self) -> None:
        adapter = RTMPoseAdapter(INPUT_SIZE, include_face_keypoints=False)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        keypoints = adapter.postprocess(FakePoseBackend().run(prepared.tensor), prepared)
        assert len(keypoints) == len(KEYPOINT_NAMES) - len(FACE_KEYPOINTS)
        assert len(adapter.labels) == len(keypoints)
        assert "nose" not in adapter.labels

    def test_missing_second_output_is_an_error(self) -> None:
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        with pytest.raises(ValueError, match=r"two outputs"):
            adapter.postprocess([np.zeros((1, 17, 384), np.float32)], prepared)

    def test_wrong_keypoint_count_is_an_error(self) -> None:
        """A hand or face model would decode into a nonsense body skeleton."""
        adapter = RTMPoseAdapter(INPUT_SIZE)
        prepared = adapter.preprocess(np.zeros((480, 640, 3), np.uint8), BOX)
        with pytest.raises(ValueError, match=r"21 keypoints"):
            adapter.postprocess(
                [np.zeros((1, 21, 384), np.float32), np.zeros((1, 21, 512), np.float32)],
                prepared,
            )


class TestPosture:
    def test_standing(self) -> None:
        estimate = classify(make_pose(upright(200.0, 300.0, 400.0)))
        assert estimate.posture is Posture.STANDING
        assert estimate.confidence > 0.5

    def test_sitting(self) -> None:
        estimate = classify(make_pose(upright(200.0, 210.0, 290.0)))
        assert estimate.posture is Posture.SITTING

    def test_crouching(self) -> None:
        estimate = classify(make_pose(upright(200.0, 215.0, 230.0)))
        assert estimate.posture is Posture.CROUCHING

    def test_lying(self) -> None:
        estimate = classify(
            make_pose(
                {
                    LEFT_SHOULDER: (100.0, 200.0),
                    RIGHT_SHOULDER: (100.0, 240.0),
                    LEFT_HIP: (250.0, 205.0),
                    RIGHT_HIP: (250.0, 245.0),
                }
            )
        )
        assert estimate.posture is Posture.LYING

    def test_leaning_is_not_lying(self) -> None:
        """A person bent over a desk must not raise what will become a fall alert."""
        estimate = classify(
            make_pose(
                {
                    LEFT_SHOULDER: (140.0, 100.0),
                    RIGHT_SHOULDER: (180.0, 100.0),
                    LEFT_HIP: (200.0, 200.0),  # 35 degrees off vertical
                    RIGHT_HIP: (230.0, 200.0),
                }
            )
        )
        assert estimate.posture is not Posture.LYING

    def test_no_legs_is_unknown_with_a_reason(self) -> None:
        """The ordinary desk-webcam case, and the honest answer."""
        estimate = classify(
            make_pose(
                {
                    LEFT_SHOULDER: (140.0, 100.0),
                    RIGHT_SHOULDER: (180.0, 100.0),
                    LEFT_HIP: (145.0, 200.0),
                    RIGHT_HIP: (175.0, 200.0),
                }
            )
        )
        assert estimate.posture is Posture.UNKNOWN
        assert "knees" in estimate.reason
        assert estimate.confidence == 0.0

    def test_no_torso_is_unknown(self) -> None:
        estimate = classify(make_pose({LEFT_KNEE: (10.0, 10.0)}))
        assert estimate.posture is Posture.UNKNOWN
        assert "hip" in estimate.reason

    def test_folded_thigh_without_ankles_is_unknown(self) -> None:
        """Not standing is known; sitting versus crouching is not."""
        points = upright(200.0, 210.0, 290.0)
        del points[LEFT_ANKLE]
        del points[RIGHT_ANKLE]
        estimate = classify(make_pose(points))
        assert estimate.posture is Posture.UNKNOWN
        assert "ankles" in estimate.reason

    def test_foreshortened_torso_is_refused(self) -> None:
        estimate = classify(
            make_pose(
                {
                    LEFT_SHOULDER: (140.0, 100.0),
                    RIGHT_SHOULDER: (180.0, 100.0),
                    LEFT_HIP: (145.0, 105.0),
                    RIGHT_HIP: (175.0, 105.0),
                    LEFT_KNEE: (145.0, 300.0),
                    RIGHT_KNEE: (175.0, 300.0),
                }
            )
        )
        assert estimate.posture is Posture.UNKNOWN
        assert "foreshortened" in estimate.reason

    def test_profile_view_uses_one_side(self) -> None:
        """Requiring both sides would return UNKNOWN for anyone not square-on."""
        estimate = classify(
            make_pose(
                {
                    LEFT_SHOULDER: (140.0, 100.0),
                    LEFT_HIP: (145.0, 200.0),
                    LEFT_KNEE: (145.0, 300.0),
                    LEFT_ANKLE: (145.0, 400.0),
                }
            )
        )
        assert estimate.posture is Posture.STANDING

    def test_low_confidence_joints_are_not_used(self) -> None:
        """Landmarks are always emitted; without a floor the rules run on guesses."""
        pose = make_pose(upright(200.0, 300.0, 400.0), score=0.15)
        assert classify(pose, min_keypoint_confidence=0.3).posture is Posture.UNKNOWN
        assert classify(pose, min_keypoint_confidence=0.1).posture is Posture.STANDING

    def test_confidence_tracks_the_weakest_joint(self) -> None:
        strong = classify(make_pose(upright(200.0, 300.0, 400.0), score=0.95))
        weak = classify(make_pose(upright(200.0, 300.0, 400.0), score=0.4))
        assert strong.posture is weak.posture
        assert strong.confidence > weak.confidence

    def test_every_posture_carries_a_reason(self) -> None:
        for points in (
            upright(200.0, 300.0, 400.0),
            upright(200.0, 210.0, 290.0),
            upright(200.0, 215.0, 230.0),
        ):
            assert classify(make_pose(points)).reason


class TestEngine:
    def build(self, **kwargs) -> tuple[PoseEngine, FakePoseBackend]:
        backend = FakePoseBackend()
        engine = PoseEngine(RTMPoseAdapter(INPUT_SIZE), backend, model_name="fake", **kwargs)
        return engine, backend

    def test_estimates_one_pose_per_person(self) -> None:
        engine, backend = self.build()
        result = engine.estimate(make_frame(), tracking_of(make_track(1), make_track(2)))
        assert len(result) == 2
        assert backend.calls == 2
        assert result.people_seen == 2

    def test_non_person_tracks_are_ignored(self) -> None:
        engine, _backend = self.build()
        result = engine.estimate(
            make_frame(), tracking_of(make_track(1, label="car"), make_track(2))
        )
        assert len(result) == 1
        assert result.poses[0].entity_id == "person_2"

    def test_coasting_tracks_are_skipped(self) -> None:
        """A predicted box is a guess about where something is, not a crop of it."""
        engine, backend = self.build()
        result = engine.estimate(make_frame(), tracking_of(make_track(1, time_since_update=4)))
        assert len(result) == 0
        assert backend.calls == 0

    def test_budget_caps_work_and_records_the_shortfall(self) -> None:
        engine, backend = self.build(max_persons=2)
        tracks = [make_track(i) for i in range(5)]
        result = engine.estimate(make_frame(), tracking_of(*tracks))
        assert len(result) == 2
        assert result.people_seen == 5
        assert result.skipped == 3
        assert backend.calls == 2

    def test_budget_keeps_the_largest_boxes(self) -> None:
        """Box area is a direct proxy for pixels per joint."""
        engine, _ = self.build(max_persons=1)
        small = make_track(1, box=BoundingBox(0.0, 0.0, 20.0, 40.0))
        large = make_track(2, box=BoundingBox(0.0, 0.0, 200.0, 400.0))
        result = engine.estimate(make_frame(), tracking_of(small, large))
        assert result.poses[0].track_id == 2

    def test_pose_carries_the_tracker_identity(self) -> None:
        engine, _ = self.build()
        result = engine.estimate(make_frame(), tracking_of(make_track(7)))
        assert result.poses[0].entity_id == "person_7"
        assert result.poses[0].track_id == 7

    def test_timings_are_recorded_separately(self) -> None:
        engine, _ = self.build()
        result = engine.estimate(make_frame(), tracking_of(make_track(1)))
        assert result.total_ms >= 0.0
        assert result.inference_ms >= 0.0

    def test_closing_twice_is_safe(self) -> None:
        engine, backend = self.build()
        engine.close()
        engine.close()
        assert backend.closed

    def test_use_after_close_is_an_error(self) -> None:
        engine, _ = self.build()
        engine.close()
        with pytest.raises(RuntimeError, match=r"closed"):
            engine.estimate(make_frame(), tracking_of(make_track(1)))

    def test_invalid_budget_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"max_persons"):
            self.build(max_persons=0)

    def test_invalid_keypoint_threshold_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"min_keypoint_confidence"):
            self.build(min_keypoint_confidence=1.5)


class TestCatalog:
    def test_pose_models_are_registered_and_permissively_licensed(self) -> None:
        from vantage.perception.catalog import models_for_task

        models = models_for_task("pose")
        assert models
        for spec in models.values():
            assert spec.license == "Apache-2.0"
            assert len(spec.labels) == len(KEYPOINT_NAMES)

    def test_building_a_pose_engine_from_a_detector_is_refused(self) -> None:
        from vantage.pose.factory import build_pose_engine

        with pytest.raises(ConfigError, match=r"not a pose estimator"):
            build_pose_engine(model="yolox-nano", allow_download=False)


class TestOverlayAndHud:
    """Drawing must agree with what the classifier actually saw."""

    def build_result(self, score: float = 0.9) -> PoseResult:
        pose = make_pose(upright(200.0, 300.0, 400.0), score=score)
        estimate = classify(pose)
        return PoseResult(
            poses=(
                Pose(
                    keypoints=pose.keypoints,
                    track_id=1,
                    entity_id="person_1",
                    box=BOX,
                    posture=estimate.posture,
                    posture_confidence=estimate.confidence,
                    posture_reason=estimate.reason,
                ),
            ),
            source_id="t",
            frame_index=1,
            capture_wall=1.0,
            frame_size=(640, 480),
            people_seen=1,
        )

    def test_draw_poses_marks_the_canvas(self) -> None:
        from vantage.viz.overlay import draw_poses

        canvas = np.zeros((480, 640, 3), np.uint8)
        out = draw_poses(canvas, self.build_result())
        assert out.any()

    def test_invisible_joints_are_not_drawn(self) -> None:
        """The picture must not show joints the classifier treated as absent."""
        from vantage.viz.overlay import draw_poses

        blank = draw_poses(np.zeros((480, 640, 3), np.uint8), self.build_result(score=0.05))
        assert not blank.any()

    def test_read_only_input_is_copied(self) -> None:
        from vantage.viz.overlay import draw_poses

        canvas = np.zeros((480, 640, 3), np.uint8)
        canvas.flags.writeable = False
        out = draw_poses(canvas, self.build_result())
        assert out is not canvas

    def test_hud_reports_the_budget_shortfall(self) -> None:
        from vantage.viz.hud import HudRenderer

        result = self.build_result()
        over_budget = PoseResult(
            poses=result.poses,
            source_id="t",
            frame_index=1,
            capture_wall=1.0,
            frame_size=(640, 480),
            people_seen=9,
        )
        rows = HudRenderer()._compose_pose(over_budget)
        assert any("over max_persons" in str(row) for row in rows)


class TestConfigWiring:
    def test_pose_without_tracking_is_refused(self) -> None:
        from vantage.config.schema import PoseConfig, VantageConfig

        with pytest.raises(ConfigError, match=r"requires tracking.enabled"):
            VantageConfig(pose=PoseConfig(enabled=True))

    def test_pose_flag_implies_detection_and_tracking(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        args = build_parser().parse_args(["run", "--pose"])
        overrides = _flag_overrides(args)
        assert "detection.enabled=true" in overrides
        assert "tracking.enabled=true" in overrides
        assert "pose.enabled=true" in overrides

    def test_face_keypoints_can_be_disabled_from_the_cli(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        args = build_parser().parse_args(["run", "--pose", "--no-face-keypoints"])
        assert "pose.include_face_keypoints=false" in _flag_overrides(args)


class TestUnknownIsExplained:
    """An unexplained "unknown" is indistinguishable from a broken classifier."""

    def unknown_result(self, count: int = 1) -> PoseResult:
        pose = make_pose({LEFT_SHOULDER: (140.0, 100.0), RIGHT_SHOULDER: (180.0, 100.0)})
        estimate = classify(pose)
        assert estimate.posture is Posture.UNKNOWN
        return PoseResult(
            poses=tuple(
                Pose(
                    keypoints=pose.keypoints,
                    track_id=i,
                    entity_id=f"person_{i}",
                    box=BOX,
                    posture=estimate.posture,
                    posture_reason=estimate.reason,
                )
                for i in range(count)
            ),
            source_id="t",
            frame_index=1,
            capture_wall=1.0,
            frame_size=(640, 480),
            people_seen=count,
        )

    def test_describe_carries_the_reason(self) -> None:
        assert "hip not visible" in self.unknown_result().describe()

    def test_repeated_reasons_are_counted_not_repeated(self) -> None:
        described = self.unknown_result(count=3).describe()
        assert described.count("hip not visible") == 1
        assert "x3" in described

    def test_a_classified_posture_adds_no_reason_noise(self) -> None:
        pose = make_pose(upright(200.0, 300.0, 400.0))
        estimate = classify(pose)
        result = PoseResult(
            poses=(
                Pose(
                    keypoints=pose.keypoints,
                    track_id=1,
                    entity_id="person_1",
                    box=BOX,
                    posture=estimate.posture,
                    posture_reason=estimate.reason,
                ),
            ),
            source_id="t",
            frame_index=1,
            capture_wall=1.0,
            frame_size=(640, 480),
            people_seen=1,
        )
        assert result.unknown_reasons() == {}
        assert " - " not in result.describe()

    def test_hud_shows_why(self) -> None:
        from vantage.viz.hud import HudRenderer

        rows = HudRenderer()._compose_pose(self.unknown_result())
        assert any("hip not visible" in str(row) for row in rows)

    def test_hud_text_stays_ascii(self) -> None:
        """OpenCV's Hershey fonts draw a box for anything they cannot render."""
        from vantage.viz.hud import _shorten

        assert _shorten("x" * 80, 20).isascii()
        assert len(_shorten("x" * 80, 20)) == 20
