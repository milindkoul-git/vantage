"""Detection wired into config, the run loop, the overlay and the CLI.

Uses a fake backend throughout, so these run with no model file and no
inference runtime installed - the same reason the Phase 1 suite needs no camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.app import run_ingestion
from vantage.cli import main
from vantage.config.loader import load_config
from vantage.config.schema import (
    DetectionConfig,
    DisplayConfig,
    IngestConfig,
    IngestMode,
    SourceConfig,
    VantageConfig,
)
from vantage.core.errors import ConfigError
from vantage.perception.contracts import BoundingBox, Detection, DetectionResult
from vantage.viz.overlay import class_color, draw_detections

from tests.fakes import make_engine


def config_for(**detection_kwargs) -> VantageConfig:
    return VantageConfig(
        source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=12"),
        ingest=IngestConfig(mode=IngestMode.INLINE),
        detection=DetectionConfig(enabled=True, **detection_kwargs),
        display=DisplayConfig(enabled=False),
    )


class TestDetectionConfig:
    def test_disabled_by_default(self) -> None:
        assert load_config(None, []).detection.enabled is False

    def test_enabling_via_override(self) -> None:
        config = load_config(None, ["detection.enabled=true", "detection.model=yolox-tiny"])
        assert config.detection.enabled is True
        assert config.detection.model == "yolox-tiny"

    def test_classes_accepts_a_yaml_list(self) -> None:
        config = load_config(None, ["detection.classes=[person, car]"])
        assert config.detection.classes == ["person", "car"]

    def test_classes_rejects_a_bare_string(self) -> None:
        """Otherwise a string would be iterated character by character."""
        with pytest.raises(ConfigError, match="expected a list"):
            load_config(None, ["detection.classes=person"])

    def test_empty_class_list_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="empty list"):
            load_config(None, ["detection.classes=[]"])

    @pytest.mark.parametrize(
        "override",
        [
            "detection.confidence=0",
            "detection.confidence=1",
            "detection.nms_iou=2",
            "detection.interval=0",
            "detection.max_detections=0",
            "detection.backend=tensorrt",
            "detection.device=tpu",
            "detection.threads=-1",
        ],
    )
    def test_invalid_values_are_rejected(self, override: str) -> None:
        with pytest.raises(ConfigError):
            load_config(None, [override])

    def test_unknown_model_suggests_a_near_miss(self) -> None:
        from vantage.perception.catalog import get_model_spec

        with pytest.raises(ConfigError, match="did you mean 'yolox-nano'"):
            get_model_spec("yolox-nanoo")


class TestDetectionInRunLoop:
    def test_runs_the_detector_on_every_frame_by_default(self) -> None:
        engine, _ = make_engine()
        result = run_ingestion(config_for(), engine=engine)
        assert result.frames == 12
        assert result.detections_run == 12

    def test_interval_reduces_inference_without_dropping_frames(self) -> None:
        """The CPU-fit lever: display stays smooth, inference runs 1-in-N."""
        engine, _ = make_engine()
        result = run_ingestion(config_for(interval=4), engine=engine)
        assert result.frames == 12
        assert result.detections_run == 3

    def test_summary_reports_the_detection_stack(self) -> None:
        engine, _ = make_engine()
        result = run_ingestion(config_for(), engine=engine)
        summary = result.detection_summary
        assert summary["model"] == "fake-model"
        assert summary["passes"] == 12
        assert summary["max_fps"] > 0
        assert "detection:" in result.summary()

    def test_injected_engine_is_not_closed_by_the_run(self) -> None:
        """The caller owns what the caller supplied."""
        engine, _ = make_engine()
        run_ingestion(config_for(), engine=engine)
        assert engine._backend.closed is False  # type: ignore[attr-defined]

    def test_detection_disabled_leaves_the_phase_1_path_untouched(self) -> None:
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&frames=5"),
            ingest=IngestConfig(mode=IngestMode.INLINE),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config)
        assert result.frames == 5
        assert result.detections_run == 0
        assert result.detection_summary == {}

    def test_detections_are_drawn_when_display_is_on(self) -> None:
        from tests.test_app_and_viz import RecordingSink

        engine, _ = make_engine()
        sink = RecordingSink()
        config = config_for()
        config = VantageConfig(
            source=config.source,
            ingest=config.ingest,
            detection=config.detection,
            display=DisplayConfig(enabled=True),
        )
        result = run_ingestion(config, engine=engine, sink=sink)
        assert len(sink.images) == result.frames
        # Something was actually painted onto the generated frames.
        assert sink.images[0].max() > 0


class TestOverlay:
    def result_with(self, *detections: Detection) -> DetectionResult:
        return DetectionResult(
            detections=detections,
            source_id="cam0",
            frame_index=0,
            capture_wall=0.0,
            frame_size=(320, 240),
        )

    def test_draws_without_modifying_a_read_only_frame(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image.flags.writeable = False
        result = self.result_with(Detection(BoundingBox(10, 10, 100, 100), 0, "person", 0.9))

        canvas = draw_detections(image, result)
        assert image.max() == 0
        assert canvas.max() > 0

    def test_class_colours_are_stable_and_distinct(self) -> None:
        assert class_color(0) == class_color(0)
        assert class_color(0) != class_color(1)

    def test_label_stays_visible_for_a_box_at_the_top_edge(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        result = self.result_with(Detection(BoundingBox(5, 0, 90, 40), 0, "person", 0.9))
        canvas = draw_detections(image, result)
        # The label is drawn inside the box, so the top rows carry ink.
        assert canvas[0:40, 5:90].max() > 0

    def test_stale_detections_render_differently(self) -> None:
        result = self.result_with(Detection(BoundingBox(10, 10, 100, 100), 0, "person", 0.9))
        # Separate buffers: draw_detections paints writeable input in place.
        fresh = draw_detections(np.zeros((240, 320, 3), dtype=np.uint8), result, stale=False)
        stale = draw_detections(np.zeros((240, 320, 3), dtype=np.uint8), result, stale=True)
        assert not np.array_equal(fresh, stale)

    def test_writeable_input_is_drawn_in_place(self) -> None:
        """The documented ownership contract, relied on to avoid a copy per frame."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        result = self.result_with(Detection(BoundingBox(10, 10, 100, 100), 0, "person", 0.9))
        returned = draw_detections(image, result)
        assert returned is image
        assert image.max() > 0

    def test_empty_result_leaves_the_frame_alone(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        assert draw_detections(image, self.result_with()).max() == 0

    def test_box_larger_than_the_frame_does_not_raise(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        result = self.result_with(Detection(BoundingBox(0, 0, 320, 240), 5, "bus", 0.7))
        assert draw_detections(image, result).shape == image.shape


class TestModelsCli:
    def test_models_list_reports_licences(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["models", "list", "--json"]) == 0
        import json

        payload = json.loads(capsys.readouterr().out)
        keys = {entry["key"] for entry in payload["models"]}
        assert "yolox-nano" in keys
        assert all(entry["license"] == "Apache-2.0" for entry in payload["models"])

    def test_models_action_requires_a_name(self) -> None:
        assert main(["models", "pull"]) == 1

    def test_unknown_model_exits_non_zero(self) -> None:
        assert main(["models", "pull", "not-a-model"]) == 1
