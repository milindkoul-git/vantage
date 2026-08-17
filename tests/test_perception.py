"""Tests for the detection contracts, NMS, adapter and engine.

Everything here runs without a model file and without an inference runtime: the
YOLOX adapter is driven with hand-built tensors, and the engine is composed with
a fake backend. Tests that need real weights live in ``test_detection_model.py``
and skip themselves when the model is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.core.errors import ConfigError
from vantage.perception.adapters.yolox import YoloxAdapter
from vantage.perception.contracts import BoundingBox, Detection, DetectionResult
from vantage.perception.labels import COCO_80, get_label_set, register_label_set
from vantage.perception.nms import batched_nms, nms

from tests.fakes import FakeBackend, make_engine, yolox_prediction


class TestBoundingBox:
    def test_geometry(self) -> None:
        box = BoundingBox(10, 20, 30, 60)
        assert box.width == 20 and box.height == 40
        assert box.area == 800
        assert box.center == (20, 40)
        assert box.xywh == (10, 20, 20, 40)

    def test_bottom_center_is_the_ground_contact_point(self) -> None:
        """Zones and trajectories key on where an object meets the floor."""
        assert BoundingBox(10, 20, 30, 60).bottom_center == (20, 60)

    def test_inverted_corners_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="inverted"):
            BoundingBox(30, 10, 10, 20)

    def test_clipping_to_frame(self) -> None:
        clipped = BoundingBox(-15, -5, 700, 500).clipped(640, 480)
        assert clipped.xyxy == (0.0, 0.0, 640.0, 480.0)

    def test_iou_of_identical_boxes_is_one(self) -> None:
        box = BoundingBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self) -> None:
        assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(50, 50, 60, 60)) == 0.0

    def test_iou_half_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 15, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_zero_area_box_does_not_divide_by_zero(self) -> None:
        assert BoundingBox(5, 5, 5, 5).iou(BoundingBox(0, 0, 10, 10)) == 0.0


class TestDetectionContracts:
    def test_confidence_must_be_a_probability(self) -> None:
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="confidence"):
                Detection(BoundingBox(0, 0, 1, 1), 0, "person", bad)

    def test_result_is_iterable_and_sized(self) -> None:
        result = _result(
            [
                Detection(BoundingBox(0, 0, 1, 1), 0, "person", 0.9),
                Detection(BoundingBox(2, 2, 3, 3), 2, "car", 0.5),
            ]
        )
        assert len(result) == 2
        assert [d.label for d in result] == ["person", "car"]

    def test_counts_and_filters(self) -> None:
        result = _result(
            [
                Detection(BoundingBox(0, 0, 1, 1), 0, "person", 0.9),
                Detection(BoundingBox(2, 2, 3, 3), 0, "person", 0.4),
                Detection(BoundingBox(4, 4, 5, 5), 2, "car", 0.8),
            ]
        )
        assert result.counts() == {"person": 2, "car": 1}
        assert len(result.of_class("person")) == 2
        assert len(result.above(0.5)) == 2

    def test_total_ms_sums_the_stages(self) -> None:
        result = DetectionResult(
            detections=(),
            source_id="s",
            frame_index=0,
            capture_wall=0.0,
            frame_size=(10, 10),
            preprocess_ms=1.0,
            inference_ms=8.0,
            postprocess_ms=1.5,
        )
        assert result.total_ms == pytest.approx(10.5)

    def test_result_holds_no_pixels(self) -> None:
        """Detections outlive frames; a result pinning an image would leak."""
        fields = DetectionResult.__dataclass_fields__
        assert "image" not in fields and "frame" not in fields

    def test_describe_handles_the_empty_case(self) -> None:
        assert "nothing detected" in _result([]).describe()


class TestNms:
    def test_empty_input(self) -> None:
        assert nms(np.empty((0, 4)), np.empty((0,)), 0.5).size == 0

    def test_suppresses_overlapping_and_keeps_the_best(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, 0.5)
        assert sorted(keep.tolist()) == [0, 2]

    def test_keeps_everything_when_nothing_overlaps(self) -> None:
        boxes = np.array([[0, 0, 5, 5], [20, 20, 25, 25]], dtype=np.float32)
        keep = nms(boxes, np.array([0.5, 0.6], dtype=np.float32), 0.5)
        assert len(keep) == 2

    def test_returns_indices_in_descending_score_order(self) -> None:
        boxes = np.array([[0, 0, 5, 5], [20, 20, 25, 25], [40, 40, 45, 45]], dtype=np.float32)
        keep = nms(boxes, np.array([0.1, 0.9, 0.5], dtype=np.float32), 0.5)
        assert keep.tolist() == [1, 2, 0]

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            nms(np.array([[0, 0, 1, 1]]), np.array([0.5]), 1.5)

    def test_class_aware_nms_does_not_suppress_across_classes(self) -> None:
        """A person in front of a car must not delete the car."""
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        classes = np.array([0, 2])
        assert len(batched_nms(boxes, scores, classes, 0.5)) == 2

    def test_class_aware_nms_still_suppresses_within_a_class(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        classes = np.array([0, 0])
        assert len(batched_nms(boxes, scores, classes, 0.5)) == 1


class TestLabels:
    def test_coco_has_eighty_classes_starting_with_person(self) -> None:
        assert len(COCO_80) == 80
        assert COCO_80[0] == "person"

    def test_lookup_and_registration(self) -> None:
        assert get_label_set("coco80") is COCO_80
        register_label_set("tiny", ("a", "b"))
        assert get_label_set("tiny") == ("a", "b")

    def test_unknown_set_is_reported(self) -> None:
        with pytest.raises(KeyError):
            get_label_set("imagenet")


class TestYoloxAdapter:
    def adapter(self, size=(416, 416)) -> YoloxAdapter:
        return YoloxAdapter(input_size=size, labels=COCO_80)

    def test_preprocess_shape_and_dtype(self) -> None:
        prepared = self.adapter().preprocess(np.zeros((480, 640, 3), dtype=np.uint8))
        assert prepared.tensor.shape == (1, 3, 416, 416)
        assert prepared.tensor.dtype == np.float32

    def test_preprocess_preserves_aspect_ratio(self) -> None:
        prepared = self.adapter().preprocess(np.zeros((480, 640, 3), dtype=np.uint8))
        assert prepared.scale == pytest.approx(416 / 640)
        assert prepared.original_size == (640, 480)

    def test_preprocess_pads_with_the_training_grey(self) -> None:
        """Padding value is part of the model contract, not an arbitrary choice."""
        prepared = self.adapter().preprocess(np.zeros((100, 400, 3), dtype=np.uint8))
        # Bottom rows are padding for a wide image letterboxed into a square.
        assert prepared.tensor[0, 0, -1, -1] == pytest.approx(114.0)

    def test_preprocess_keeps_values_in_0_255(self) -> None:
        """YOLOX folded normalisation into its weights; rescaling would break it."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        prepared = self.adapter().preprocess(image)
        assert prepared.tensor.max() == pytest.approx(200.0)

    def test_postprocess_decodes_a_planted_box(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        # Cell 0 of the stride-8 grid, centred in its cell, 16x16 px, class 0.
        raw = yolox_prediction(class_id=0, objectness=0.9, class_score=0.9)
        detections = adapter.postprocess([raw], prepared, 0.3, 0.45, 100)

        assert len(detections) == 1
        assert detections[0].label == "person"
        assert detections[0].confidence == pytest.approx(0.81, abs=1e-4)

    def test_confidence_is_objectness_times_class_score(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        raw = yolox_prediction(class_id=0, objectness=0.5, class_score=0.5)
        # 0.25 falls below a 0.3 floor even though each factor alone exceeds it.
        assert adapter.postprocess([raw], prepared, 0.3, 0.45, 100) == []
        assert len(adapter.postprocess([raw], prepared, 0.2, 0.45, 100)) == 1

    def test_boxes_come_back_in_original_frame_coordinates(self) -> None:
        """The consumer must never see letterboxed coordinates."""
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((832, 832, 3), dtype=np.uint8))  # scale 0.5
        # Grid cell 1000, well away from the frame edge, so the box maps back
        # whole rather than being clipped and confusing the scale check.
        raw = yolox_prediction(class_id=0, objectness=0.9, class_score=0.9, row=1000)
        detection = adapter.postprocess([raw], prepared, 0.3, 0.45, 100)[0]

        # Model-space box is 16 px wide at scale 0.5, so 32 px in the original.
        assert detection.box.width == pytest.approx(32.0, abs=1.0)
        assert detection.box.x1 > 0, "box should not be touching the frame edge"

    def test_edge_boxes_are_clipped_not_dropped(self) -> None:
        """A partially visible object stays, trimmed to the frame."""
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((832, 832, 3), dtype=np.uint8))
        raw = yolox_prediction(class_id=0, objectness=0.9, class_score=0.9, row=0)
        detection = adapter.postprocess([raw], prepared, 0.3, 0.45, 100)[0]
        assert detection.box.x1 == 0.0
        assert detection.box.width > 0

    def test_detections_are_clipped_to_the_frame(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        raw = yolox_prediction(class_id=0, objectness=0.9, class_score=0.9, width=1e3, height=1e3)
        detection = adapter.postprocess([raw], prepared, 0.3, 0.45, 100)[0]
        assert detection.box.x2 <= 416 and detection.box.y2 <= 416

    def test_max_detections_is_honoured(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        raw = yolox_prediction(class_id=0, objectness=0.9, class_score=0.9, populate_all=True)
        assert len(adapter.postprocess([raw], prepared, 0.3, 0.99, 5)) == 5

    def test_mismatched_grid_size_is_reported_clearly(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        wrong = np.zeros((1, 100, 85), dtype=np.float32)
        with pytest.raises(ValueError, match="grid expects"):
            adapter.postprocess([wrong], prepared, 0.3, 0.45, 100)

    def test_malformed_output_is_reported(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="unexpected YOLOX output shape"):
            adapter.postprocess([np.zeros((1, 10, 3), dtype=np.float32)], prepared, 0.3, 0.45, 10)

    def test_no_outputs_is_reported(self) -> None:
        adapter = self.adapter()
        prepared = adapter.preprocess(np.zeros((416, 416, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="no output tensors"):
            adapter.postprocess([], prepared, 0.3, 0.45, 10)

    def test_unknown_class_id_degrades_instead_of_crashing(self) -> None:
        adapter = YoloxAdapter(input_size=(416, 416), labels=("person",))
        assert adapter.label_for(77) == "class_77"


class TestDetectionEngine:
    def test_produces_a_result_tied_to_its_frame(self) -> None:
        engine, frame = make_engine()
        result = engine.detect(frame)
        assert result.source_id == frame.source_id
        assert result.frame_index == frame.index
        assert result.frame_size == frame.resolution
        assert result.model == "fake-model"

    def test_times_each_stage_separately(self) -> None:
        engine, frame = make_engine()
        result = engine.detect(frame)
        assert result.preprocess_ms >= 0
        assert result.inference_ms >= 0
        assert result.postprocess_ms >= 0
        assert result.total_ms == pytest.approx(
            result.preprocess_ms + result.inference_ms + result.postprocess_ms
        )

    def test_class_filter_keeps_only_requested_labels(self) -> None:
        engine, frame = make_engine(keep_classes=["car"])
        assert engine.detect(frame).counts() == {"car": 1}

    def test_class_filter_rejects_labels_the_model_cannot_emit(self) -> None:
        with pytest.raises(ConfigError, match="cannot produce"):
            make_engine(keep_classes=["unicorn"])

    def test_confidence_threshold_is_validated(self) -> None:
        for bad in (0.0, 1.0, 1.4):
            with pytest.raises(ConfigError, match="confidence"):
                make_engine(confidence=bad)

    def test_warmup_runs_inference_without_a_frame(self) -> None:
        engine, _ = make_engine()
        backend: FakeBackend = engine._backend  # type: ignore[attr-defined]
        before = backend.calls
        engine.warmup(3)
        assert backend.calls == before + 3

    def test_close_is_idempotent_and_releases_the_backend(self) -> None:
        engine, _ = make_engine()
        backend: FakeBackend = engine._backend  # type: ignore[attr-defined]
        engine.close()
        engine.close()
        assert backend.closed is True

    def test_detect_after_close_is_rejected(self) -> None:
        engine, frame = make_engine()
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            engine.detect(frame)

    def test_info_describes_the_resolved_stack(self) -> None:
        engine, _ = make_engine()
        described = engine.info.describe()
        assert "fake-model" in described and "fake" in described


def _result(detections: list[Detection]) -> DetectionResult:
    return DetectionResult(
        detections=tuple(detections),
        source_id="cam0",
        frame_index=3,
        capture_wall=1.0,
        frame_size=(640, 480),
    )
