"""Open-vocabulary discovery: prompt handling, decoding, and the tier boundary.

Runs without the 360 MB model. The adapter is array manipulation plus
tokenisation, both of which can be checked against hand-built tensors - and the
architectural claims (discovery is a separate tier, fixed-vocabulary models are
rejected) are the part most worth pinning down, because they are what stops
someone wiring a two-second model into the live loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.core.errors import ConfigError
from vantage.perception.adapters import get_adapter
from vantage.perception.adapters.grounding_dino import (
    CLS_TOKEN,
    MAX_TOKENS,
    PERIOD_TOKEN,
    SEP_TOKEN,
    GroundingDinoAdapter,
)
from vantage.perception.catalog import get_model_spec
from vantage.perception.discovery import DiscoveryResult, build_discovery_engine

INPUT = (800, 800)


class FakeTokenizer:
    """Maps each word to a deterministic pair of ids, so spans are predictable."""

    class _Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    def encode(self, text: str, add_special_tokens: bool = True):
        return self._Encoding([1000 + len(w) for w in text.split()])


def adapter(prompts: list[str] | None = None) -> GroundingDinoAdapter:
    return GroundingDinoAdapter(
        input_size=INPUT, labels=("prompt",), prompts=prompts, tokenizer=FakeTokenizer()
    )


class TestPrompts:
    def test_prompts_are_lowercased_and_stripped(self) -> None:
        a = adapter(["  Pen ", "Coffee Mug"])
        assert a.prompts == ["pen", "coffee mug"]

    def test_empty_prompts_are_rejected_with_an_example(self) -> None:
        with pytest.raises(ConfigError, match="at least one prompt"):
            adapter([])
        with pytest.raises(ConfigError, match="at least one prompt"):
            adapter(["", "   "])

    def test_missing_tokenizer_names_the_install_command(self) -> None:
        a = GroundingDinoAdapter(input_size=INPUT, labels=("prompt",))
        with pytest.raises(ConfigError, match="pip install tokenizers"):
            a.set_prompts(["pen"])

    def test_tokens_are_padded_to_a_static_length(self) -> None:
        """The GPU plugin cannot compile a dynamic sequence length."""
        prepared = adapter(["pen"]).preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        assert prepared.extra["input_ids"].shape == (1, MAX_TOKENS)
        assert prepared.extra["attention_mask"].shape == (1, MAX_TOKENS)

    def test_special_tokens_and_separators_are_present(self) -> None:
        """The '.' between phrases is semantic, not decorative."""
        prepared = adapter(["pen", "mug"]).preprocess(np.zeros((10, 10, 3), dtype=np.uint8))
        ids = prepared.extra["input_ids"][0].tolist()
        assert ids[0] == CLS_TOKEN
        assert PERIOD_TOKEN in ids
        assert SEP_TOKEN in ids

    def test_attention_mask_covers_only_real_tokens(self) -> None:
        prepared = adapter(["pen"]).preprocess(np.zeros((10, 10, 3), dtype=np.uint8))
        mask = prepared.extra["attention_mask"][0]
        ids = prepared.extra["input_ids"][0]
        assert mask.sum() < MAX_TOKENS
        assert mask[: int(mask.sum())].all()
        assert not mask[int(mask.sum()) :].any()
        assert ids[int(mask.sum()) :].sum() == 0

    def test_too_many_prompts_fail_loudly(self) -> None:
        with pytest.raises(ConfigError, match="over the"):
            adapter([f"word{i}" for i in range(MAX_TOKENS)])

    def test_prompts_can_be_changed_without_rebuilding(self) -> None:
        a = adapter(["pen"])
        a.set_prompts(["stapler", "mug"])
        assert a.prompts == ["stapler", "mug"]


class TestPreprocess:
    def test_applies_imagenet_normalisation(self) -> None:
        """Unlike D-FINE, this export does not fold normalisation into the graph."""
        prepared = adapter(["pen"]).preprocess(np.zeros((50, 50, 3), dtype=np.uint8))
        # A black frame maps to -mean/std, which is distinctly negative.
        assert prepared.tensor.min() < -1.0

    def test_shape_and_original_size(self) -> None:
        prepared = adapter(["pen"]).preprocess(np.zeros((300, 500, 3), dtype=np.uint8))
        assert prepared.tensor.shape == (1, 3, 800, 800)
        assert prepared.original_size == (500, 300)

    def test_preprocess_without_prompts_is_an_error(self) -> None:
        a = GroundingDinoAdapter(input_size=INPUT, labels=("prompt",), tokenizer=FakeTokenizer())
        with pytest.raises(ConfigError, match="set_prompts"):
            a.preprocess(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_extra_inputs_are_supplied_for_the_fused_graph(self) -> None:
        prepared = adapter(["pen"]).preprocess(np.zeros((10, 10, 3), dtype=np.uint8))
        assert set(prepared.extra) == {
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "pixel_mask",
        }


class TestPostprocess:
    def _outputs(self, scores: dict[int, dict[int, float]], boxes: dict[int, tuple]):
        """``scores[query][token] = probability``."""
        logits = np.full((1, 20, MAX_TOKENS), -20.0, dtype=np.float32)
        for q, tokens in scores.items():
            for t, p in tokens.items():
                logits[0, q, t] = float(np.log(p / (1 - p)))
        raw_boxes = np.zeros((1, 20, 4), dtype=np.float32)
        for q, b in boxes.items():
            raw_boxes[0, q] = b
        return [logits, raw_boxes]

    def test_scores_a_phrase_by_its_best_token(self) -> None:
        """Multi-word prompts often align on the head noun only."""
        a = adapter(["coffee mug"])
        prepared = a.preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        # "coffee mug" -> two tokens starting at index 1
        outputs = self._outputs({0: {1: 0.2, 2: 0.9}}, {0: (0.5, 0.5, 0.2, 0.2)})
        detections = a.postprocess(outputs, prepared, 0.3, 0.5, 10)
        assert len(detections) == 1
        assert detections[0].label == "coffee mug"
        assert detections[0].confidence == pytest.approx(0.9, abs=1e-3)

    def test_label_is_the_prompt_the_user_typed(self) -> None:
        a = adapter(["pen", "stapler"])
        prepared = a.preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        # second phrase starts after "pen" + separator
        outputs = self._outputs({0: {3: 0.8}}, {0: (0.5, 0.5, 0.2, 0.2)})
        detections = a.postprocess(outputs, prepared, 0.3, 0.5, 10)
        assert detections and detections[0].label == "stapler"

    def test_confidence_threshold_applies(self) -> None:
        a = adapter(["pen"])
        prepared = a.preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = self._outputs({0: {1: 0.4}}, {0: (0.5, 0.5, 0.2, 0.2)})
        assert len(a.postprocess(outputs, prepared, 0.3, 0.5, 10)) == 1
        assert a.postprocess(outputs, prepared, 0.6, 0.5, 10) == []

    def test_boxes_land_in_original_frame_pixels(self) -> None:
        a = adapter(["pen"])
        prepared = a.preprocess(np.zeros((200, 400, 3), dtype=np.uint8))
        outputs = self._outputs({0: {1: 0.9}}, {0: (0.5, 0.5, 0.5, 0.5)})
        box = a.postprocess(outputs, prepared, 0.3, 0.5, 10)[0].box
        assert box.center == pytest.approx((200.0, 100.0))
        assert box.width == pytest.approx(200.0)
        assert box.height == pytest.approx(100.0)

    def test_duplicates_are_suppressed(self) -> None:
        a = adapter(["pen"])
        prepared = a.preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        outputs = self._outputs(
            {0: {1: 0.9}, 1: {1: 0.85}},
            {0: (0.5, 0.5, 0.4, 0.4), 1: (0.51, 0.51, 0.4, 0.4)},
        )
        assert len(a.postprocess(outputs, prepared, 0.3, 0.5, 10)) == 1

    def test_no_matches_returns_empty(self) -> None:
        a = adapter(["pen"])
        prepared = a.preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        assert a.postprocess(self._outputs({}, {}), prepared, 0.3, 0.5, 10) == []

    def test_wrong_output_count_fails_loudly(self) -> None:
        a = adapter(["pen"])
        prepared = a.preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="two outputs"):
            a.postprocess([np.zeros((1, 2, 3))], prepared, 0.3, 0.5, 10)


class TestTierBoundary:
    """Discovery and live detection must not be confusable."""

    def test_fixed_vocabulary_models_are_rejected_with_a_pointer(self) -> None:
        with pytest.raises(ConfigError, match="fixed-vocabulary"):
            build_discovery_engine(["pen"], model="yolox-nano")

    def test_the_error_names_the_right_command(self) -> None:
        with pytest.raises(ConfigError, match="vantage run --model"):
            build_discovery_engine(["pen"], model="dfine-s-obj365")

    def test_catalog_entry_is_marked_open_vocabulary(self) -> None:
        spec = get_model_spec("grounding-dino-tiny")
        assert spec.label_set == "open-vocabulary"
        assert spec.adapter == "grounding-dino"
        assert spec.license == "Apache-2.0"

    def test_adapter_is_registered(self) -> None:
        assert get_adapter("grounding-dino") is GroundingDinoAdapter


class TestResult:
    def test_describes_an_empty_result_usefully(self) -> None:
        result = DiscoveryResult(
            detections=(), prompts=("pen",), model="m", elapsed_ms=2100.0,
            frame_size=(640, 480),
        )
        assert "nothing matched" in result.describe()
        assert len(result) == 0

    def test_counts_by_label(self) -> None:
        from vantage.perception.contracts import BoundingBox, Detection

        d = Detection(BoundingBox(0, 0, 10, 10), 0, "pen", 0.9)
        result = DiscoveryResult(
            detections=(d, d), prompts=("pen",), model="m", elapsed_ms=2100.0,
            frame_size=(640, 480),
        )
        assert result.counts() == {"pen": 2}
        assert "2x pen" in result.describe()


class _RecordingBackend:
    """Counts passes and returns a canned hit on the first token of each prompt."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_ids: list[list[int]] = []

    @property
    def info(self):
        from vantage.perception.backends.base import BackendInfo

        return BackendInfo(name="fake", device="none")

    def run(self, tensor, extra=None):
        self.calls += 1
        self.seen_ids.append(extra["input_ids"][0].tolist() if extra else [])
        logits = np.full((1, 4, MAX_TOKENS), -20.0, dtype=np.float32)
        logits[0, 0, 1] = 3.0  # a confident hit on the first prompt token
        boxes = np.zeros((1, 4, 4), dtype=np.float32)
        boxes[0, 0] = (0.5, 0.5, 0.2, 0.2)
        return [logits, boxes]

    def close(self) -> None:
        pass


class TestOnePassPerPrompt:
    """Measured: batching prompts makes the model suppress all but the strongest.

    On the test image, ``dog, bicycle, car`` in one pass scored 0.90/0.09/0.08;
    queried separately they scored 0.92/0.89/0.59. The batched form is wrong,
    not merely weaker, so the engine trades time for correctness.
    """

    def _engine(self, prompts: list[str]):
        from vantage.perception.discovery import DiscoveryEngine

        backend = _RecordingBackend()
        a = adapter(prompts)
        return DiscoveryEngine(a, backend, "fake-model"), backend

    def test_runs_one_inference_per_prompt(self) -> None:
        engine, backend = self._engine(["pen", "mug", "chair"])
        engine.discover(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.3)
        assert backend.calls == 3

    def test_each_pass_carries_exactly_one_prompt(self) -> None:
        engine, backend = self._engine(["pen", "mug"])
        engine.discover(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.3)
        for ids in backend.seen_ids:
            # CLS, one phrase, PERIOD, SEP -> exactly one separator
            assert ids.count(PERIOD_TOKEN) == 1

    def test_prompt_list_is_restored_afterwards(self) -> None:
        """The engine must stay reusable and still report what was asked for."""
        engine, _ = self._engine(["pen", "mug", "chair"])
        result = engine.discover(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.3)
        assert engine.prompts == ["pen", "mug", "chair"]
        assert result.prompts == ("pen", "mug", "chair")

    def test_results_are_merged_and_sorted(self) -> None:
        engine, _ = self._engine(["pen", "mug"])
        result = engine.discover(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.3)
        assert len(result) == 2
        confidences = [d.confidence for d in result.detections]
        assert confidences == sorted(confidences, reverse=True)
        assert {d.label for d in result.detections} == {"pen", "mug"}

    def test_progress_callback_reports_each_prompt(self) -> None:
        engine, _ = self._engine(["pen", "mug"])
        seen = []
        engine.discover(
            np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.3,
            progress=lambda i, n, p: seen.append((i, n, p)),
        )
        assert seen == [(0, 2, "pen"), (1, 2, "mug")]

    def test_pass_count_is_reported(self) -> None:
        engine, _ = self._engine(["pen", "mug", "chair"])
        result = engine.discover(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.3)
        assert result.metadata["passes"] == 3

    def test_closed_engine_refuses_to_run(self) -> None:
        engine, _ = self._engine(["pen"])
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            engine.discover(np.zeros((10, 10, 3), dtype=np.uint8))
