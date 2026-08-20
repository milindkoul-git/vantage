"""Open-vocabulary discovery: ask what is in a scene, in your own words.

A separate capability rather than a mode of the live pipeline, because the
measured cost makes them genuinely different jobs. Detection runs at 12-90 fps
and feeds a tracker; discovery takes about two seconds per frame and answers a
question. Pretending they are the same thing behind one switch would mean either
a live pipeline that stutters to a halt or a discovery feature nobody can find.

The split is honest about what each is for:

* **Live tracking** uses a fixed vocabulary - COCO's 80 classes or Objects365's
  365 - and maintains identity across frames at full frame rate.
* **Discovery** takes arbitrary text, runs once on a single frame, and returns
  what it found. Nothing is tracked, because two seconds between observations is
  far beyond what any motion model can associate across.

Requires the ``tokenizers`` package, which is deliberately optional: it exists
only for this feature, and the rest of the platform must keep working without
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import ConfigError, VantageError
from vantage.core.logging import get_logger
from vantage.perception.contracts import Detection

log = get_logger(__name__)

TOKENIZER_REPO = "onnx-community/grounding-dino-tiny-ONNX"
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_URL = f"https://huggingface.co/{TOKENIZER_REPO}/resolve/main/{TOKENIZER_FILE}"


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """What one open-vocabulary pass found."""

    detections: tuple[Detection, ...]
    prompts: tuple[str, ...]
    model: str
    elapsed_ms: float
    frame_size: tuple[int, int]
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.detections)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for detection in self.detections:
            tally[detection.label] = tally.get(detection.label, 0) + 1
        return tally

    def describe(self) -> str:
        if not self.detections:
            return f"nothing matched {', '.join(self.prompts)} ({self.elapsed_ms / 1000:.1f} s)"
        summary = ", ".join(
            f"{count}x {label}" for label, count in sorted(self.counts().items())
        )
        return f"{summary} ({self.elapsed_ms / 1000:.1f} s)"


def load_tokenizer(model_dir: str | Path = "models", allow_download: bool = True):
    """Fetch and load the BERT tokenizer the prompts are encoded with.

    Cached beside the weights, and downloaded on demand for the same reason the
    weights are: a 700 kB file that only this feature needs should not be
    committed to the repository or installed for everyone.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ConfigError(
            "open-vocabulary discovery needs the 'tokenizers' package. "
            'Install it with: pip install -e ".[discover]"'
        ) from exc

    path = Path(model_dir).expanduser() / TOKENIZER_FILE
    if not path.is_file():
        if not allow_download:
            raise VantageError(
                f"tokenizer not found at {path} and downloads are disabled. "
                f"Fetch {TOKENIZER_URL} manually."
            )
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("fetching tokenizer", extra={"vantage_fields": {"url": TOKENIZER_URL}})
        # Written to a temporary name and renamed, so an interrupted download
        # never leaves a truncated file that looks cached.
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(TOKENIZER_URL, partial)
        partial.replace(path)

    return Tokenizer.from_file(str(path))


class DiscoveryEngine:
    """Runs open-vocabulary detection on individual frames."""

    def __init__(self, adapter, backend, model_name: str, clock: Clock = SYSTEM_CLOCK) -> None:
        self._adapter = adapter
        self._backend = backend
        self._model = model_name
        self._clock = clock
        self._closed = False

    @property
    def prompts(self) -> list[str]:
        return self._adapter.prompts

    def set_prompts(self, prompts: list[str]) -> None:
        self._adapter.set_prompts(prompts)

    def discover(
        self,
        image: np.ndarray,
        *,
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 50,
        progress=None,
    ) -> DiscoveryResult:
        """Find the current prompts in one frame, one prompt per pass.

        Why one at a time, when the model accepts several phrases at once
        -----------------------------------------------------------------
        Because this export gets it wrong, measured. Grounding DINO separates
        phrases with ``.`` and is supposed to apply a block-diagonal mask so
        that each sub-sentence attends only to itself. This ONNX graph takes a
        one-dimensional attention mask and never builds that block structure, so
        every token attends to every other and the strongest phrase suppresses
        the rest.

        On the test image, prompts ``dog, bicycle, car`` in a single pass::

            dog 0.90    bicycle 0.09    car 0.08

        The same three, queried one at a time::

            dog 0.92    bicycle 0.89    car 0.59

        The batched form does not merely score lower, it is *wrong* - two of the
        three objects are plainly there and would be discarded at any sensible
        threshold. So the cost is linear in prompt count and the results are
        right, which is the correct side of that trade for a feature whose whole
        purpose is answering "is this thing here?".
        """
        if self._closed:
            raise RuntimeError("discovery engine has been closed")

        requested = list(self._adapter.prompts)
        started = self._clock.monotonic()
        found: list[Detection] = []

        for index, prompt in enumerate(requested):
            if progress is not None:
                progress(index, len(requested), prompt)
            self._adapter.set_prompts([prompt])
            prepared = self._adapter.preprocess(image)
            outputs = self._backend.run(prepared.tensor, prepared.extra or None)
            found.extend(
                self._adapter.postprocess(
                    outputs,
                    prepared,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )
            )

        elapsed_ms = (self._clock.monotonic() - started) * 1000.0
        # Restore what the caller asked for, so the engine is reusable and
        # `prompts` still reports the full list rather than the last one queried.
        self._adapter.set_prompts(requested)
        found.sort(key=lambda d: -d.confidence)

        return DiscoveryResult(
            detections=tuple(found[:max_detections]),
            prompts=tuple(requested),
            model=self._model,
            elapsed_ms=elapsed_ms,
            frame_size=(image.shape[1], image.shape[0]),
            metadata={
                "backend": self._backend.info.describe(),
                "passes": len(requested),
            },
        )

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> DiscoveryEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_discovery_engine(
    prompts: list[str],
    *,
    model: str = "grounding-dino-tiny",
    backend: str = "auto",
    device: str = "auto",
    model_dir: str | Path = "models",
    threads: int = 0,
    allow_download: bool = True,
    clock: Clock = SYSTEM_CLOCK,
) -> DiscoveryEngine:
    """Resolve a catalog key into a ready discovery engine."""
    from vantage.perception.adapters import get_adapter
    from vantage.perception.adapters.grounding_dino import GroundingDinoAdapter
    from vantage.perception.backends import create_backend
    from vantage.perception.catalog import get_model_spec
    from vantage.perception.store import ModelStore

    spec = get_model_spec(model)
    if spec.label_set != "open-vocabulary":
        raise ConfigError(
            f"{spec.key!r} is a fixed-vocabulary detector; discovery needs an "
            "open-vocabulary model such as 'grounding-dino-tiny'. "
            f"Use 'vantage run --model {spec.key}' for live detection instead."
        )

    store = ModelStore(model_dir)
    path = store.ensure(spec, allow_download=allow_download)
    tokenizer = load_tokenizer(model_dir, allow_download=allow_download)

    # Narrowed explicitly rather than trusting the generic registry. get_adapter
    # returns type[ModelAdapter], whose constructor takes neither prompts nor a
    # tokenizer - so calling it with them is only correct because the label_set
    # check above already established which adapter this is. A type checker
    # flagged that as three errors, and it was right to: the registry's type
    # does not express "this particular adapter needs more arguments".
    adapter_cls = get_adapter(spec.adapter)
    if not issubclass(adapter_cls, GroundingDinoAdapter):
        raise ConfigError(
            f"model {spec.key!r} declares label_set 'open-vocabulary' but its "
            f"adapter {spec.adapter!r} is not an open-vocabulary adapter. The "
            "catalog entry is inconsistent."
        )
    adapter = adapter_cls(
        input_size=spec.input_size,
        labels=spec.labels,
        prompts=prompts,
        tokenizer=tokenizer,
    )
    inference_backend = create_backend(
        backend,
        path,
        device=device,
        threads=threads,
        input_shape=spec.input_size,
        input_shapes=adapter.static_input_shapes(),
    )
    log.info(
        "discovery engine ready",
        extra={
            "vantage_fields": {
                "model": spec.key,
                "backend": inference_backend.info.describe(),
                "prompts": ", ".join(adapter.prompts),
            }
        },
    )
    return DiscoveryEngine(adapter, inference_backend, spec.key, clock=clock)
