"""Tests that need real weights and a real inference runtime.

These skip themselves when the model or runtime is absent, so the default
suite stays hardware- and download-free. Run them after::

    vantage models pull yolox-nano

They are the only place the decode is checked against ground truth: the fake
backend in the other suites proves the plumbing, but only real weights on a
real image prove that preprocessing, grid decoding and coordinate mapping are
mutually consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.perception.backends import available_backends
from vantage.perception.catalog import get_model_spec
from vantage.perception.store import ModelStore

pytestmark = pytest.mark.model

MODEL = "yolox-nano"


def _skip_unless_ready() -> str:
    spec = get_model_spec(MODEL)
    store = ModelStore("models")
    if not store.is_cached(spec):
        pytest.skip(f"{MODEL} not downloaded; run 'vantage models pull {MODEL}'")
    backends = available_backends()
    if not any(backends.values()):
        pytest.skip("no inference runtime installed")
    return "onnxruntime" if backends["onnxruntime"] else "openvino"


@pytest.fixture(scope="module")
def engine():
    backend = _skip_unless_ready()
    from vantage.perception.engine import build_engine

    built = build_engine(
        MODEL, backend=backend, device="cpu", model_dir="models", confidence=0.30
    )
    yield built
    built.close()


@pytest.fixture(scope="module")
def street_scene() -> np.ndarray:
    """A real photograph, because a synthetic scene cannot serve here.

    The generated frames used elsewhere contain coloured circles, which a COCO
    detector correctly refuses to label - useless as an accuracy oracle. The
    reference image is fetched once and cached beside the models rather than
    committed, keeping binaries out of git history; the test skips if it cannot
    be obtained.
    """
    import urllib.error
    import urllib.request
    from pathlib import Path

    import cv2

    path = Path("models/_test_dog.jpg")
    if not path.is_file():
        url = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            pytest.skip(f"reference image unavailable: {exc}")

    image = cv2.imread(str(path))
    if image is None:
        pytest.skip("reference image could not be decoded")
    return image


class TestRealDetection:
    def test_finds_the_expected_objects(self, engine, street_scene) -> None:
        """The end-to-end check: preprocessing, decode and mapping agree."""
        detections = engine.detect_image(street_scene)
        labels = {d.label for d in detections}
        assert "dog" in labels
        assert "bicycle" in labels

    def test_boxes_are_inside_the_frame_and_plausible(self, engine, street_scene) -> None:
        height, width = street_scene.shape[:2]
        detections = engine.detect_image(street_scene)
        assert detections, "expected at least one detection"

        for detection in detections:
            box = detection.box
            assert 0 <= box.x1 < box.x2 <= width
            assert 0 <= box.y1 < box.y2 <= height
            # A box covering essentially the whole frame usually means the
            # letterbox scale was applied in the wrong direction.
            assert box.area < 0.95 * width * height

    def test_the_dog_lands_where_the_dog_is(self, engine, street_scene) -> None:
        """Guards against a decode that finds the right classes in wrong places."""
        dogs = [d for d in engine.detect_image(street_scene) if d.label == "dog"]
        assert dogs, "no dog detected"
        best = max(dogs, key=lambda d: d.confidence)
        centre_x, centre_y = best.box.center
        # The dog occupies the lower-left quadrant of this 768x576 image.
        assert 100 < centre_x < 400
        assert 250 < centre_y < 500

    def test_confidence_threshold_reduces_detections(self, engine, street_scene) -> None:
        low = len(engine.detect_image(street_scene))
        engine._confidence = 0.85  # type: ignore[attr-defined]
        try:
            high = len(engine.detect_image(street_scene))
        finally:
            engine._confidence = 0.30  # type: ignore[attr-defined]
        assert high <= low

    def test_blank_input_detects_nothing(self, engine) -> None:
        assert engine.detect_image(np.zeros((480, 640, 3), dtype=np.uint8)) == []


class TestBackendAgreement:
    def test_cpu_backends_produce_the_same_detections(self, street_scene) -> None:
        """Swapping the runtime must not change what the platform reports."""
        backends = available_backends()
        if not (backends["onnxruntime"] and backends["openvino"]):
            pytest.skip("both runtimes required for a cross-backend comparison")

        from vantage.perception.engine import build_engine

        results = {}
        for name in ("onnxruntime", "openvino"):
            built = build_engine(
                MODEL, backend=name, device="cpu", model_dir="models", confidence=0.30
            )
            try:
                results[name] = sorted(
                    (d.label, round(d.confidence, 2)) for d in built.detect_image(street_scene)
                )
            finally:
                built.close()

        assert results["onnxruntime"] == results["openvino"]


class TestModelStoreIntegrity:
    def test_cached_model_matches_its_pinned_checksum(self) -> None:
        _skip_unless_ready()
        spec = get_model_spec(MODEL)
        store = ModelStore("models")
        assert store.ensure(spec, allow_download=False).is_file()

    def test_corrupt_cache_is_rejected_rather_than_used(self, tmp_path) -> None:
        spec = get_model_spec(MODEL)
        store = ModelStore(tmp_path)
        corrupt = store.path_for(spec)
        corrupt.write_bytes(b"not a model")

        from vantage.core.errors import VantageError

        with pytest.raises(VantageError, match="integrity verification"):
            store.ensure(spec, allow_download=False)
