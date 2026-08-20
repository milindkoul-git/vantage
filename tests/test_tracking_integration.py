"""Tracking wired into configuration, the run loop and the display.

Separate from :mod:`tests.test_tracking`, which tests the algorithm in
isolation. These tests are about the seams: that configuration reaches the
tracker unchanged, that the run loop produces tracks, and that the overlay
renders them without needing a detector installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.app import _effective_confidence, run_ingestion
from vantage.config.loader import load_config
from vantage.config.schema import (
    DetectionConfig,
    DisplayConfig,
    IngestConfig,
    SourceConfig,
    TrackingConfig,
    VantageConfig,
)
from vantage.core.errors import ConfigError
from vantage.tracking.bytetrack import ByteTracker, TrackerParams
from vantage.tracking.factory import build_tracker, params_from_config
from vantage.viz.hud import HudRenderer
from vantage.viz.overlay import draw_tracks, track_color


def _tracking_config(**kwargs) -> VantageConfig:
    return VantageConfig(
        source=SourceConfig(uri="synthetic://?width=320&height=240&fps=30&objects=2"),
        ingest=IngestConfig(max_frames=kwargs.pop("frames", 12)),
        detection=DetectionConfig(enabled=True),
        tracking=TrackingConfig(enabled=True, **kwargs),
        display=DisplayConfig(enabled=False),
    )


class TestConfiguration:
    def test_tracking_requires_detection(self) -> None:
        """The tracker consumes detections; enabling it alone is meaningless."""
        with pytest.raises(ConfigError, match=r"requires detection.enabled"):
            VantageConfig(
                detection=DetectionConfig(enabled=False),
                tracking=TrackingConfig(enabled=True),
            )

    def test_every_field_reaches_the_tracker(self) -> None:
        """Guards against a field being added to config and never plumbed through."""
        settings = TrackingConfig(
            enabled=True,
            high_threshold=0.55,
            low_threshold=0.15,
            init_threshold=0.65,
            iou_high=0.25,
            iou_low=0.45,
            iou_tentative=0.55,
            min_hits=4,
            max_lost_s=2.5,
            max_step_s=3.0,
            history=17,
            class_aware=False,
            measurement_noise=0.07,
            acceleration_noise=5.0,
            initial_velocity_noise=2.0,
            size_drift_noise=0.4,
        )
        params = params_from_config(settings)
        assert params.high_threshold == 0.55
        assert params.low_threshold == 0.15
        assert params.init_threshold == 0.65
        assert params.iou_high == 0.25
        assert params.iou_low == 0.45
        assert params.iou_tentative == 0.55
        assert params.min_hits == 4
        assert params.max_lost_s == 2.5
        assert params.max_step_s == 3.0
        assert params.history == 17
        assert params.class_aware is False
        assert params.noise.measurement == 0.07
        assert params.noise.acceleration == 5.0
        assert params.noise.initial_velocity == 2.0
        assert params.noise.size_drift == 0.4

    def test_shipped_defaults_match_the_tracker_defaults(self) -> None:
        """Config and code must not drift apart on the tuned values."""
        assert params_from_config(TrackingConfig()) == TrackerParams()

    def test_disabled_tracking_builds_nothing(self) -> None:
        assert build_tracker(TrackingConfig(enabled=False)) is None

    def test_enabled_tracking_builds_a_tracker(self) -> None:
        assert isinstance(build_tracker(TrackingConfig(enabled=True)), ByteTracker)

    def test_default_config_file_parses(self) -> None:
        config = load_config("configs/default.yaml")
        assert config.tracking.enabled is False
        assert params_from_config(config.tracking) == TrackerParams()

    @pytest.mark.parametrize(
        "override,message",
        [
            ("tracking.min_hits=0", "min_hits"),
            ("tracking.max_lost_s=-1", "max_lost_s"),
            ("tracking.high_threshold=1.5", "high_threshold"),
            ("tracking.acceleration_noise=0", "acceleration_noise"),
            ("tracking.low_threshold=0.9", "low_threshold must be below"),
        ],
    )
    def test_invalid_values_are_rejected_with_a_useful_message(
        self, override: str, message: str
    ) -> None:
        with pytest.raises(ConfigError, match=message):
            load_config("configs/default.yaml", [override])

    def test_unknown_tracking_key_suggests_the_right_one(self) -> None:
        with pytest.raises(ConfigError, match=r"did you mean 'min_hits'"):
            load_config("configs/default.yaml", ["tracking.min_hitz=2"])


class TestDetectionFloor:
    """The one place tracking silently changes another subsystem's setting."""

    def test_threshold_is_lowered_when_tracking_is_enabled(self) -> None:
        config = VantageConfig(
            detection=DetectionConfig(enabled=True, confidence=0.35),
            tracking=TrackingConfig(enabled=True, detection_floor=0.1),
        )
        assert _effective_confidence(config) == 0.1

    def test_threshold_is_untouched_without_tracking(self) -> None:
        config = VantageConfig(detection=DetectionConfig(enabled=True, confidence=0.35))
        assert _effective_confidence(config) == 0.35

    def test_floor_never_raises_the_threshold(self) -> None:
        """A floor above the configured confidence would discard wanted boxes."""
        config = VantageConfig(
            detection=DetectionConfig(enabled=True, confidence=0.2),
            tracking=TrackingConfig(enabled=True, detection_floor=0.5),
        )
        assert _effective_confidence(config) == 0.2

    def test_lowering_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Honouring a different number than the user set must be visible."""
        import logging

        config = VantageConfig(
            detection=DetectionConfig(enabled=True, confidence=0.35),
            tracking=TrackingConfig(enabled=True, detection_floor=0.1),
        )
        with caplog.at_level(logging.INFO, logger="vantage.app"):
            _effective_confidence(config)
        assert any("lowered for tracking" in record.message for record in caplog.records)


class TestRunLoop:
    def test_run_produces_tracking_telemetry(self) -> None:
        from tests.fakes import make_engine

        engine, _ = make_engine()
        result = run_ingestion(_tracking_config(), engine=engine)

        assert result.frames == 12
        assert result.tracking_steps == 12
        assert result.tracking_summary["steps"] == 12
        assert result.tracking_summary["entities_total"] >= 1
        assert result.tracking_summary["mean_ms"] >= 0.0
        assert "tracking:" in result.summary()

    def test_tracking_off_leaves_no_tracking_telemetry(self) -> None:
        from tests.fakes import make_engine

        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=320&height=240&fps=30&objects=2"),
            ingest=IngestConfig(max_frames=6),
            detection=DetectionConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config, engine=engine)
        assert result.tracking_steps == 0
        assert result.tracking_summary == {}
        assert "tracking:" not in result.summary()

    def test_tracking_steps_follow_the_detection_interval(self) -> None:
        """The tracker must not be advanced on frames the detector skipped."""
        from tests.fakes import make_engine

        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=320&height=240&fps=30&objects=2"),
            ingest=IngestConfig(max_frames=12),
            detection=DetectionConfig(enabled=True, interval=3),
            tracking=TrackingConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config, engine=engine)
        assert result.detections_run == 4
        assert result.tracking_steps == 4

    def test_entities_are_counted_across_the_whole_run(self) -> None:
        from tests.fakes import make_engine

        engine, _ = make_engine()
        result = run_ingestion(_tracking_config(frames=20), engine=engine)
        assert result.tracking_summary["entities_total"] >= 1
        assert result.tracking_summary["ids_used"] >= 1


class TestOverlay:
    def _tracks(self):
        tracker = ByteTracker()
        for index in range(8):
            from vantage.perception.contracts import (
                BoundingBox,
                Detection,
                DetectionResult,
            )

            det = Detection(BoundingBox(100 + index, 100, 160 + index, 280), 0, "person", 0.9)
            out = tracker.update(DetectionResult((det,), "t", index, index / 30.0, (640, 480)))
        return out

    def test_draws_in_place_on_a_writeable_buffer(self) -> None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        returned = draw_tracks(canvas, self._tracks())
        assert returned is canvas
        assert canvas.any(), "nothing was drawn"

    def test_read_only_input_is_copied(self) -> None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas.flags.writeable = False
        returned = draw_tracks(canvas, self._tracks())
        assert returned is not canvas
        assert not canvas.any()

    def test_track_colours_are_stable_and_distinct(self) -> None:
        assert track_color(3) == track_color(3)
        assert len({track_color(i) for i in range(12)}) == 12

    def test_empty_result_draws_nothing(self) -> None:
        from vantage.tracking.contracts import empty_tracking_result

        canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        draw_tracks(canvas, empty_tracking_result("t", 0, 0.0, (64, 64)))
        assert not canvas.any()

    def test_boxes_at_the_frame_edge_do_not_raise(self) -> None:
        from vantage.perception.contracts import (
            BoundingBox,
            Detection,
            DetectionResult,
        )

        tracker = ByteTracker(TrackerParams(min_hits=1))
        det = Detection(BoundingBox(-20, -30, 40, 60), 0, "person", 0.9)
        tracked = tracker.update(DetectionResult((det,), "t", 0, 0.0, (640, 480)))
        draw_tracks(np.zeros((480, 640, 3), dtype=np.uint8), tracked)


class TestHud:
    def test_tracking_panel_renders(self) -> None:
        from vantage.ingestion.pipeline import PipelineStats

        stats = PipelineStats(
            source_id="t",
            kind="synthetic",
            backend="synthetic",
            uri="synthetic://",
            width=640,
            height=480,
            declared_fps=30.0,
            is_live=False,
            state="running",
            frames_delivered=10,
            frames_produced=10,
            elapsed_s=1.0,
            capture_fps=30.0,
            delivery_fps=30.0,
            mean_delivery_fps=30.0,
            queue_capacity=8,
            backpressure="latest",
        )
        tracker = ByteTracker()
        from vantage.perception.contracts import (
            BoundingBox,
            Detection,
            DetectionResult,
        )

        for index in range(6):
            tracked = tracker.update(
                DetectionResult(
                    (Detection(BoundingBox(10, 10, 60, 130), 0, "person", 0.9),),
                    "t",
                    index,
                    index / 30.0,
                    (640, 480),
                )
            )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = HudRenderer().render(image, stats, 10, tracking=tracked, entity_total=3)
        assert rendered.shape == image.shape
        assert rendered is not image
        assert rendered.any()
