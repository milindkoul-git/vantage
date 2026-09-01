"""Tests that need the real RTMPose weights and an inference runtime.

Skipped when the model or runtime is absent, so the default suite stays
download-free. Run them after::

    vantage models pull rtmpose-s

What only real weights can show
-------------------------------
The fake backend proves the plumbing and the decode arithmetic, but it cannot
prove that the *exported graph* matches what the adapter believes about it.
Every assumption checked here was read out of the shipped ``pipeline.json`` or
measured, and each one is silent if wrong: a mismatched bin count, a swapped
output order or a different keypoint layout all still produce 17 plausible
points in plausible places.

No committed photograph
-----------------------
There is deliberately no image of a person in this repository, so these tests
cannot assert that a real body is landmarked correctly - that was verified by
hand against live footage and is recorded in the README rather than pretended
at here. What they do assert is everything checkable without one: graph
signature, geometric containment, score range, and agreement between the two
runtimes.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.perception.backends import available_backends, create_backend
from vantage.perception.catalog import get_model_spec
from vantage.perception.contracts import BoundingBox
from vantage.perception.store import ModelStore
from vantage.pose.adapter import SIMCC_SPLIT_RATIO, RTMPoseAdapter
from vantage.pose.contracts import KEYPOINT_NAMES

pytestmark = pytest.mark.model

MODEL = "rtmpose-s"


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
def spec():
    _skip_unless_ready()
    return get_model_spec(MODEL)


@pytest.fixture(scope="module")
def backend(spec):
    name = _skip_unless_ready()
    adapter = RTMPoseAdapter(spec.input_size)
    path = ModelStore("models").ensure(spec, allow_download=False)
    created = create_backend(
        name,
        path,
        device="cpu",
        input_shape=spec.input_size,
        input_shapes=adapter.static_input_shapes(),
    )
    yield created
    created.close()


@pytest.fixture
def scene():
    """A textured frame and a box in it. Content is irrelevant to these checks."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    return image, BoundingBox(150.0, 60.0, 380.0, 440.0)


class TestGraphSignature:
    def test_catalog_input_size_matches_the_graph(self, spec, backend, scene) -> None:
        """A catalog typo here would silently distort every crop."""
        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        outputs = backend.run(prepared.tensor)
        assert prepared.tensor.shape == (1, 3, 256, 192)
        assert len(outputs) == 2

    def test_bin_counts_match_the_declared_split_ratio(self, spec, backend, scene) -> None:
        """``simcc_split_ratio`` is read from pipeline.json, so verify it holds."""
        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        simcc_x, simcc_y = backend.run(prepared.tensor)[:2]
        height, width = spec.input_size

        assert simcc_x.shape[-1] == int(width * SIMCC_SPLIT_RATIO)
        assert simcc_y.shape[-1] == int(height * SIMCC_SPLIT_RATIO)

    def test_output_order_is_x_then_y(self, spec, backend, scene) -> None:
        """Swapped, decoding still runs and every joint lands wrong."""
        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        simcc_x, simcc_y = backend.run(prepared.tensor)[:2]
        # The input is taller than it is wide, so the two axes have different
        # bin counts and the order is checkable rather than a coin flip.
        assert simcc_x.shape[-1] < simcc_y.shape[-1]

    def test_keypoint_count_matches_the_label_set(self, spec, backend, scene) -> None:
        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        simcc_x = backend.run(prepared.tensor)[0]
        assert simcc_x.shape[1] == len(KEYPOINT_NAMES) == len(spec.labels)


class TestDecodedOutput:
    def test_every_keypoint_lands_inside_the_padded_crop(self, spec, backend, scene) -> None:
        """The inverse transform cannot place a joint outside the region read.

        A bin index is bounded, so a decoded point outside the padded box means
        the mapping back to the frame is wrong - the failure mode that silently
        offsets an entire skeleton.
        """
        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        keypoints = adapter.postprocess(backend.run(prepared.tensor), prepared)

        left = prepared.center[0] - prepared.scale[0] / 2
        right = prepared.center[0] + prepared.scale[0] / 2
        top = prepared.center[1] - prepared.scale[1] / 2
        bottom = prepared.center[1] + prepared.scale[1] / 2

        assert len(keypoints) == len(KEYPOINT_NAMES)
        for name, keypoint in zip(KEYPOINT_NAMES, keypoints, strict=False):
            assert left - 1 <= keypoint.x <= right + 1, name
            assert top - 1 <= keypoint.y <= bottom + 1, name

    def test_scores_are_within_the_contract(self, spec, backend, scene) -> None:
        """The head is unnormalised; measured peaks reach 1.06 before clipping."""
        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        for keypoint in adapter.postprocess(backend.run(prepared.tensor), prepared):
            assert 0.0 <= keypoint.confidence <= 1.0

    def test_dropping_face_keypoints_leaves_the_rest_in_place(
        self, spec, backend, scene
    ) -> None:
        """The privacy switch must remove points, not shift the others."""
        image, box = scene
        full = RTMPoseAdapter(spec.input_size)
        trimmed = RTMPoseAdapter(spec.input_size, include_face_keypoints=False)

        prepared = full.preprocess(image, box)
        outputs = backend.run(prepared.tensor)
        all_points = full.postprocess(outputs, prepared)
        kept = trimmed.postprocess(outputs, trimmed.preprocess(image, box))

        assert len(kept) == len(all_points) - 5
        for index, keypoint in enumerate(kept):
            assert keypoint.xy == pytest.approx(all_points[index + 5].xy)


class TestBackendAgreement:
    def test_both_runtimes_decode_the_same_skeleton(self, spec, scene) -> None:
        """The same file through two runtimes must land in the same places.

        The check Phase 2 applied to detectors, applied to pose. It is the only
        thing that separates "the model works" from "this runtime's quirks and
        my decoding happen to cancel out".
        """
        backends = available_backends()
        if not all(backends.values()):
            pytest.skip("needs both onnxruntime and openvino installed")

        image, box = scene
        adapter = RTMPoseAdapter(spec.input_size)
        prepared = adapter.preprocess(image, box)
        path = ModelStore("models").ensure(spec, allow_download=False)

        decoded = []
        for name in ("onnxruntime", "openvino"):
            runtime = create_backend(
                name,
                path,
                device="cpu",
                input_shape=spec.input_size,
                input_shapes=adapter.static_input_shapes(),
            )
            try:
                decoded.append(adapter.postprocess(runtime.run(prepared.tensor), prepared))
            finally:
                runtime.close()

        first, second = decoded
        for name, a, b in zip(KEYPOINT_NAMES, first, second, strict=False):
            assert a.x == pytest.approx(b.x, abs=1.0), name
            assert a.y == pytest.approx(b.y, abs=1.0), name


class TestArchiveProvenance:
    def test_the_cached_member_still_matches_its_pin(self, spec) -> None:
        """Re-verifies the extracted graph, not the archive it arrived in."""
        from vantage.perception.store import sha256_of

        path = ModelStore("models").ensure(spec, allow_download=False)
        assert sha256_of(path) == spec.sha256
        assert spec.is_archived


class TestPipelineIntegration:
    def test_a_full_run_reports_pose_and_state(self) -> None:
        """The app wiring, end to end, with real weights.

        Synthetic video contains no people, so the assertion is about the
        pipeline running and reporting - not about finding anyone. That the
        estimator makes zero passes over a person-free scene is itself the
        correct behaviour, and a pose count above zero here would mean the
        person filter had failed.
        """
        _skip_unless_ready()
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            PoseConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=320&height=240&fps=30&frames=12"),
            ingest=IngestConfig(max_frames=8),
            detection=DetectionConfig(enabled=True, model="yolox-nano", device="cpu"),
            tracking=TrackingConfig(enabled=True),
            pose=PoseConfig(enabled=True, model=MODEL, device="cpu"),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config)

        assert result.frames == 8
        assert result.pose_steps > 0
        assert result.pose_summary["model"] == MODEL
        assert result.pose_summary["license"] == "Apache-2.0"
        # Synthetic circles are not people, so nothing is estimated - but the
        # field is a run total now, not the last frame's count, and the two are
        # reported separately. A five-clip evaluation had this summary announce
        # "0 people estimated" for a stage that had produced 305 skeletons,
        # because the final frame happened to be empty.
        assert result.pose_summary["people"] == 0
        assert result.pose_summary["people_at_end"] == 0
        assert "state" in result.summary()

    def test_state_runs_without_pose(self) -> None:
        """Object state needs no model, so it must not depend on one."""
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=320&height=240&fps=30&frames=12"),
            ingest=IngestConfig(max_frames=8),
            detection=DetectionConfig(enabled=True, model="yolox-nano", device="cpu"),
            tracking=TrackingConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config)

        assert result.pose_steps == 0
        assert result.state_summary["entities"] >= 0
