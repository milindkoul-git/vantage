"""Grounding DINO adapter - open-vocabulary detection from text prompts.

What makes this different from every other adapter here
--------------------------------------------------------
Every other detector has a fixed vocabulary baked into its output tensor. This
one takes the vocabulary as *input*: you give it the words, and it looks for
those things. ``Pen``, ``stapler``, ``coffee mug`` - categories nobody trained
it on as classes.

That capability is not free, and the price is measured rather than guessed. On
the development machine (Intel Iris Xe, no CUDA):

===================================  ==========  =========
Model                                 Per frame   Throughput
===================================  ==========  =========
yolox-tiny (COCO, 80 classes)            18 ms     57 fps
dfine-s-obj365 (365 classes)             84 ms     12 fps
grounding-dino-tiny (open vocabulary)  2176 ms    0.46 fps
===================================  ==========  =========

**120x slower than the fixed-vocabulary detector.** That is not a tuning gap, it
is a different class of model, and it is why this runs as an on-demand discovery
pass rather than in the live loop. Two seconds is unusable for tracking - an
object moves metres in that time and no motion model bridges it - but it is
perfectly reasonable for "tell me what is on this desk".

Two structural facts found by measurement, both of which shape this adapter:

* **Text and image share one fused graph.** OWL-ViT lets you embed the prompts
  once and reuse them; this export does not. Every pass pays the full text plus
  vision cost, so there is no caching trick available.
* **The GPU plugin rejects the graph outright** unless the token sequence is
  pinned to a static length. Hence :data:`MAX_TOKENS` and the padding below.
"""

from __future__ import annotations

import cv2
import numpy as np

from vantage.core.errors import ConfigError
from vantage.perception.adapters.base import ModelAdapter, PreparedInput
from vantage.perception.contracts import BoundingBox, Detection
from vantage.perception.nms import batched_nms

MAX_TOKENS = 32
"""Fixed token budget: prompts are padded to this, because the GPU plugin cannot
compile a dynamic sequence length.

The value is small on purpose. Cross-attention cost grows super-linearly with
sequence length, and it was measured on this hardware:

======  ==========
Tokens   Per frame
======  ==========
32          3.1 s
64          4.8 s
128        14.1 s
256      GPU kernel failure (CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
======  ==========

32 tokens holds roughly eight to ten short prompts, which is what this feature
is actually for. A generous-looking 256 would quadruple the cost of every pass
and crash the iGPU outright."""

CLS_TOKEN = 101
SEP_TOKEN = 102
PAD_TOKEN = 0
PERIOD_TOKEN = 1012
"""BERT ids. Grounding DINO separates prompts with ``.`` and the separator is
semantically load-bearing, not cosmetic - it is how the model knows where one
phrase ends and the next begins."""


class GroundingDinoAdapter(ModelAdapter):
    """Open-vocabulary detection against a caller-supplied prompt list.

    Unlike the other adapters this one is stateful: it holds the prompts and the
    tokenised form of them, because the model needs both on every pass and
    re-tokenising per frame would be wasted work.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        labels: tuple[str, ...],
        prompts: list[str] | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__(input_size=input_size, labels=labels)
        self._tokenizer = tokenizer
        self._prompts: list[str] = []
        self._token_ids: np.ndarray | None = None
        self._attention: np.ndarray | None = None
        self._spans: list[tuple[int, int]] = []
        # `None` means "prompts will be set later"; an empty list means the
        # caller tried to supply them and had nothing, which is a mistake worth
        # reporting at construction rather than at the first frame. Collapsing
        # the two with a truthiness check hides the second case.
        if prompts is not None:
            self.set_prompts(prompts)

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        """Fix the vocabulary for subsequent passes.

        Tokenised once here rather than per frame. The token span belonging to
        each prompt is recorded too, because the model scores every *token*
        against every query and turning that back into "which phrase matched"
        requires knowing which tokens belonged to which phrase.
        """
        cleaned = [p.strip().lower() for p in prompts if p and p.strip()]
        if not cleaned:
            raise ConfigError(
                "open-vocabulary detection needs at least one prompt, e.g. "
                "--prompts 'pen, stapler, coffee mug'"
            )
        if self._tokenizer is None:
            raise ConfigError(
                "Grounding DINO needs a tokenizer. Install it with: pip install tokenizers"
            )

        ids: list[int] = [CLS_TOKEN]
        spans: list[tuple[int, int]] = []
        for phrase in cleaned:
            encoded = self._tokenizer.encode(phrase, add_special_tokens=False).ids
            if not encoded:
                continue
            start = len(ids)
            ids.extend(encoded)
            spans.append((start, len(ids)))
            ids.append(PERIOD_TOKEN)
        ids.append(SEP_TOKEN)

        if len(ids) > MAX_TOKENS:
            raise ConfigError(
                f"prompts tokenise to {len(ids)} tokens, over the {MAX_TOKENS} limit. "
                "Use fewer or shorter prompts."
            )

        attention = np.zeros((1, MAX_TOKENS), dtype=np.int64)
        attention[0, : len(ids)] = 1
        padded = np.full((1, MAX_TOKENS), PAD_TOKEN, dtype=np.int64)
        padded[0, : len(ids)] = ids

        self._prompts = cleaned
        self._token_ids = padded
        self._attention = attention
        self._spans = spans

    def static_input_shapes(self) -> dict[str, list[int]]:
        """All five inputs, pinned. See the note in the base class."""
        height, width = self._input_size
        return {
            "pixel_values": [1, 3, height, width],
            "input_ids": [1, MAX_TOKENS],
            "token_type_ids": [1, MAX_TOKENS],
            "attention_mask": [1, MAX_TOKENS],
            "pixel_mask": [1, height, width],
        }

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Resize to the model input and normalise with ImageNet statistics.

        Unlike D-FINE, this export does *not* fold normalisation into the graph,
        so it has to happen here. Getting this wrong is silent - the model still
        runs and still returns boxes, they are just wrong - which is why the
        constants are written out rather than imported from somewhere vague.
        """
        if self._token_ids is None:
            raise ConfigError("call set_prompts() before running detection")

        height, width = self._input_size
        original_h, original_w = image.shape[:2]

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalised = (rgb - mean) / std

        return PreparedInput(
            tensor=np.ascontiguousarray(normalised.transpose(2, 0, 1)[None]),
            scale=1.0,
            pad=(0.0, 0.0),
            original_size=(original_w, original_h),
            extra={
                "input_ids": self._token_ids,
                "token_type_ids": np.zeros((1, MAX_TOKENS), dtype=np.int64),
                "attention_mask": self._attention,
                "pixel_mask": np.ones((1, height, width), dtype=np.int64),
            },
        )

    def postprocess(
        self,
        outputs: list[np.ndarray],
        prepared: PreparedInput,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        """Decode ``(logits, pred_boxes)`` where logits score queries against tokens.

        The output is ``(queries, tokens)`` rather than ``(queries, classes)``:
        the model reports how well each candidate box matches each *word* of the
        prompt. A phrase's score is the maximum over the tokens that make it up,
        which is the standard reduction and handles multi-word prompts where
        only the head noun aligns strongly.
        """
        if len(outputs) < 2:
            raise ValueError(
                f"Grounding DINO expects two outputs, got {len(outputs)}"
            )
        logits, boxes = _sigmoid(outputs[0][0]), outputs[1][0]

        if not self._spans:
            return []

        # Reduce token scores to one score per prompt.
        phrase_scores = np.stack(
            [logits[:, start:end].max(axis=1) for start, end in self._spans], axis=1
        )
        class_ids = phrase_scores.argmax(axis=1)
        best = phrase_scores[np.arange(phrase_scores.shape[0]), class_ids]

        keep = best >= confidence
        if not keep.any():
            return []
        class_ids, best, candidates = class_ids[keep], best[keep], boxes[keep]

        order = np.argsort(-best)
        class_ids, best, candidates = class_ids[order], best[order], candidates[order]

        width, height = prepared.original_size
        cx, cy, bw, bh = candidates.T
        corners = np.stack(
            [
                (cx - bw / 2.0) * width,
                (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width,
                (cy + bh / 2.0) * height,
            ],
            axis=1,
        )
        kept = batched_nms(corners, best, class_ids, iou_threshold)[:max_detections]

        detections: list[Detection] = []
        for index in kept:
            box = BoundingBox(*(float(v) for v in corners[index])).clipped(width, height)
            if box.width < 1.0 or box.height < 1.0:
                continue
            phrase_index = int(class_ids[index])
            detections.append(
                Detection(
                    box=box,
                    class_id=phrase_index,
                    label=self._prompts[phrase_index],
                    confidence=float(min(best[index], 1.0)),
                )
            )
        return detections


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic."""
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result
