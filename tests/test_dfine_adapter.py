"""D-FINE adapter: preprocessing, DETR-style decoding, and catalog integration.

Runs with no model file and no inference runtime - the adapter is pure array
manipulation, so its decoding can be checked against hand-built tensors whose
correct answer is known exactly. That is a stronger test than running the real
model and eyeballing the boxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.perception.adapters import get_adapter
from vantage.perception.adapters.dfine import BACKGROUND_CLASS, DFineAdapter, _sigmoid
from vantage.perception.catalog import CATALOG, get_model_spec
from vantage.perception.labels import get_label_set

INPUT = (640, 640)
LABELS = get_label_set("objects365")
QUERIES = 300


def adapter() -> DFineAdapter:
    return DFineAdapter(input_size=INPUT, labels=LABELS)


def logits_for(entries: list[tuple[int, int, float]], queries: int = QUERIES) -> np.ndarray:
    """Build a logits tensor where ``(query, class)`` has a chosen probability."""
    raw = np.full((1, queries, len(LABELS)), -20.0, dtype=np.float32)
    for query, class_id, probability in entries:
        raw[0, query, class_id] = float(np.log(probability / (1.0 - probability)))
    return raw


def boxes_for(entries: dict[int, tuple[float, float, float, float]]) -> np.ndarray:
    """Normalised cxcywh boxes; unspecified queries collapse to a degenerate point."""
    raw = np.zeros((1, QUERIES, 4), dtype=np.float32)
    for query, box in entries.items():
        raw[0, query] = box
    return raw


class TestPreprocess:
    def test_produces_the_expected_tensor_shape_and_range(self) -> None:
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        prepared = adapter().preprocess(image)
        assert prepared.tensor.shape == (1, 3, 640, 640)
        assert prepared.tensor.dtype == np.float32
        assert prepared.tensor.max() <= 1.0 and prepared.tensor.min() >= 0.0

    def test_scales_to_zero_one_without_imagenet_normalisation(self) -> None:
        """The export sets do_normalize=false; applying it here would halve quality."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        prepared = adapter().preprocess(image)
        assert prepared.tensor.max() == pytest.approx(1.0)

        black = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        assert black.tensor.min() == pytest.approx(0.0)

    def test_converts_bgr_to_rgb(self) -> None:
        """Frames are BGR by contract; this model wants RGB."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # pure blue in BGR
        prepared = adapter().preprocess(image)
        # After BGR->RGB the blue channel is last.
        assert prepared.tensor[0, 2].mean() == pytest.approx(1.0)
        assert prepared.tensor[0, 0].mean() == pytest.approx(0.0)

    def test_records_the_original_size_rather_than_a_scale(self) -> None:
        """Aspect ratio is not preserved, so one scalar scale cannot describe it."""
        prepared = adapter().preprocess(np.zeros((300, 900, 3), dtype=np.uint8))
        assert prepared.original_size == (900, 300)

    def test_non_square_input_is_stretched_not_padded(self) -> None:
        prepared = adapter().preprocess(np.zeros((200, 800, 3), dtype=np.uint8))
        assert prepared.pad == (0.0, 0.0)
        assert prepared.tensor.shape[2:] == (640, 640)


class TestPostprocess:
    def test_decodes_a_box_into_original_frame_pixels(self) -> None:
        prepared = adapter().preprocess(np.zeros((400, 800, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9)]),
            boxes_for({0: (0.5, 0.5, 0.25, 0.5)}),  # centred, quarter wide, half tall
        ]
        detections = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)

        assert len(detections) == 1
        box = detections[0].box
        assert box.center == pytest.approx((400.0, 200.0))
        assert box.width == pytest.approx(200.0)
        assert box.height == pytest.approx(200.0)

    def test_confidence_comes_from_sigmoid_not_softmax(self) -> None:
        """Focal-loss heads emit independent per-class probabilities."""
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.8), (0, 9, 0.7)]),  # both high; softmax could not do this
            boxes_for({0: (0.5, 0.5, 0.2, 0.2)}),
        ]
        detections = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.8, abs=1e-3)

    def test_one_detection_per_query_not_one_per_class(self) -> None:
        """Regression: flattened top-k reported one box as SUV, Van and Car at once."""
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9), (0, 6, 0.85), (0, 7, 0.8)]),
            boxes_for({0: (0.5, 0.5, 0.2, 0.2)}),
        ]
        detections = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)
        assert len(detections) == 1, "a single query must yield a single detection"
        assert detections[0].class_id == 5

    def test_background_class_is_never_emitted(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, BACKGROUND_CLASS, 0.99)]),
            boxes_for({0: (0.5, 0.5, 0.4, 0.4)}),
        ]
        assert adapter().postprocess(outputs, prepared, 0.3, 0.45, 100) == []

    def test_confidence_threshold_is_applied(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9), (1, 6, 0.4)]),
            boxes_for({0: (0.3, 0.3, 0.2, 0.2), 1: (0.7, 0.7, 0.2, 0.2)}),
        ]
        assert len(adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)) == 2
        assert len(adapter().postprocess(outputs, prepared, 0.5, 0.45, 100)) == 1

    def test_results_are_ordered_by_confidence(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.5), (1, 6, 0.95), (2, 7, 0.7)]),
            boxes_for({
                0: (0.2, 0.2, 0.1, 0.1),
                1: (0.5, 0.5, 0.1, 0.1),
                2: (0.8, 0.8, 0.1, 0.1),
            }),
        ]
        detections = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)
        confidences = [d.confidence for d in detections]
        assert confidences == sorted(confidences, reverse=True)

    def test_max_detections_is_honoured(self) -> None:
        """Boxes must be well separated, or suppression removes them first."""
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        entries = [(q, 5, 0.9) for q in range(20)]
        boxes = {
            q: (0.025 + 0.05 * q, 0.025 + 0.05 * q, 0.03, 0.03) for q in range(20)
        }
        outputs = [logits_for(entries), boxes_for(boxes)]
        assert len(adapter().postprocess(outputs, prepared, 0.3, 0.45, 7)) == 7

    def test_duplicate_queries_are_suppressed(self) -> None:
        """Measured on a live frame: one person produced six overlapping boxes.

        The DETR literature says set prediction makes suppression unnecessary.
        This export disagrees, and unsuppressed duplicates become phantom tracks
        one phase downstream.
        """
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9), (1, 5, 0.85), (2, 5, 0.8)]),
            boxes_for({
                0: (0.50, 0.50, 0.40, 0.40),
                1: (0.51, 0.51, 0.40, 0.40),  # near-identical
                2: (0.49, 0.49, 0.41, 0.41),  # near-identical
            }),
        ]
        detections = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9, abs=1e-3)

    def test_genuinely_separate_objects_survive_suppression(self) -> None:
        """Suppression must not merge two real, distinct objects."""
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9), (1, 5, 0.88)]),
            boxes_for({0: (0.2, 0.2, 0.2, 0.2), 1: (0.8, 0.8, 0.2, 0.2)}),
        ]
        assert len(adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)) == 2

    def test_suppression_is_class_aware(self) -> None:
        """A person standing exactly in front of a chair is two objects."""
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9), (1, 9, 0.88)]),
            boxes_for({0: (0.5, 0.5, 0.4, 0.4), 1: (0.5, 0.5, 0.4, 0.4)}),
        ]
        detections = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)
        assert len(detections) == 2
        assert {d.class_id for d in detections} == {5, 9}

    def test_boxes_are_clipped_to_the_frame(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9)]),
            boxes_for({0: (0.05, 0.05, 0.4, 0.4)}),  # runs off the top-left corner
        ]
        box = adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)[0].box
        assert box.x1 >= 0.0 and box.y1 >= 0.0
        assert box.x2 <= 100.0 and box.y2 <= 100.0

    def test_degenerate_boxes_are_dropped(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [
            logits_for([(0, 5, 0.9)]),
            boxes_for({0: (0.5, 0.5, 0.0, 0.0)}),
        ]
        assert adapter().postprocess(outputs, prepared, 0.3, 0.45, 100) == []

    def test_empty_result_when_nothing_passes(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = [logits_for([]), boxes_for({})]
        assert adapter().postprocess(outputs, prepared, 0.3, 0.45, 100) == []

    def test_labels_come_from_the_objects365_set(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        pen = LABELS.index("Pen/Pencil")
        outputs = [logits_for([(0, pen, 0.9)]), boxes_for({0: (0.5, 0.5, 0.2, 0.2)})]
        assert adapter().postprocess(outputs, prepared, 0.3, 0.45, 100)[0].label == "Pen/Pencil"

    def test_wrong_output_count_fails_loudly(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="two outputs"):
            adapter().postprocess([logits_for([])], prepared, 0.3, 0.45, 100)

    def test_wrong_output_rank_fails_loudly(self) -> None:
        prepared = adapter().preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="unexpected"):
            adapter().postprocess(
                [np.zeros((3, 3)), np.zeros((3, 3))], prepared, 0.3, 0.45, 100
            )


class TestSigmoid:
    def test_matches_the_naive_form_in_the_safe_range(self) -> None:
        values = np.linspace(-20, 20, 200)
        assert np.allclose(_sigmoid(values), 1.0 / (1.0 + np.exp(-values)))

    def test_does_not_overflow_on_extreme_logits(self) -> None:
        """Most of a 300x366 logits grid is strongly negative."""
        with np.errstate(over="raise", invalid="raise"):
            out = _sigmoid(np.array([-800.0, -100.0, 0.0, 100.0, 800.0]))
        assert np.isfinite(out).all()
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)


class TestCatalogIntegration:
    @pytest.mark.parametrize("key", ["dfine-s-obj365", "dfine-m-obj365"])
    def test_entries_are_well_formed(self, key: str) -> None:
        spec = get_model_spec(key)
        assert spec.adapter == "dfine"
        assert spec.label_set == "objects365"
        assert spec.license == "Apache-2.0"
        assert spec.input_size == (640, 640)
        assert len(spec.sha256) == 64
        assert spec.size_bytes > 0
        assert spec.num_classes == 366

    def test_adapter_name_resolves(self) -> None:
        assert get_adapter("dfine") is DFineAdapter

    def test_every_catalog_entry_has_a_registered_adapter_and_label_set(self) -> None:
        """Guards against a catalog entry naming something that does not exist."""
        for spec in CATALOG.values():
            assert get_adapter(spec.adapter) is not None
            assert len(spec.labels) > 0

    def test_objects365_contains_what_coco_lacks(self) -> None:
        """The reason this model family was added at all."""
        from vantage.perception.labels import COCO_80

        for wanted in ("Pen/Pencil", "Marker", "Stapler", "Folder", "Calculator"):
            assert wanted in LABELS
            assert wanted.lower() not in {c.lower() for c in COCO_80}

    def test_background_label_is_index_zero(self) -> None:
        """Stripping it would shift every label by one."""
        assert LABELS[0] == "None"
        assert LABELS[BACKGROUND_CLASS] == "None"
