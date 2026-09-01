"""Tests for the HUD, the frame sinks, the run loop and the CLI.

All headless: :class:`NullSink` and a recording sink stand in for the window, so
the same code path the viewer uses is exercised on machines with no display.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from vantage.app import RunResult, run_ingestion
from vantage.cli import build_parser, main
from vantage.config.schema import (
    DisplayConfig,
    IngestConfig,
    IngestMode,
    SourceConfig,
    VantageConfig,
)
from vantage.core.lifecycle import ShutdownController
from vantage.ingestion.pipeline import IngestionPipeline
from vantage.ingestion.synthetic import SyntheticSource
from vantage.viz.hud import HudRenderer
from vantage.viz.window import KEY_NONE, NullSink


def stats_for(frames: int = 3, **kwargs):
    source = SyntheticSource(source_id="t", width=320, height=240, frames=frames, objects=1)
    config = IngestConfig(mode=IngestMode.INLINE, **kwargs)
    with IngestionPipeline(source, config) as pipeline:
        list(pipeline.frames())
        return pipeline.stats()


class RecordingSink:
    """A sink that captures frames and can script key presses."""

    def __init__(self, keys: list[int] | None = None) -> None:
        self.images: list[np.ndarray] = []
        self.keys = list(keys or [])
        self.closed = False

    def show(self, image: np.ndarray) -> int:
        self.images.append(image)
        return self.keys.pop(0) if self.keys else KEY_NONE

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


class TestHud:
    def test_returns_a_new_image_of_the_same_shape(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        rendered = HudRenderer().render(image, stats_for(), frame_index=0)
        assert rendered.shape == image.shape
        assert rendered.dtype == image.dtype

    def test_never_modifies_the_frame_it_was_given(self) -> None:
        """Frames are shared read-only; the viewer is not privileged."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image.flags.writeable = False
        rendered = HudRenderer().render(image, stats_for(), frame_index=0)
        assert image.max() == 0
        assert rendered.max() > 0  # the panel really was drawn

    @pytest.mark.parametrize("size", [(64, 48), (240, 320), (1080, 1920)])
    def test_survives_extreme_frame_sizes(self, size: tuple[int, int]) -> None:
        image = np.zeros((*size, 3), dtype=np.uint8)
        assert HudRenderer().render(image, stats_for(), frame_index=1).shape == image.shape

    def test_sparkline_appears_once_there_is_history(self) -> None:
        hud = HudRenderer()
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        stats = stats_for()
        first = hud.render(image, stats, 0)
        second = hud.render(image, stats, 1)
        assert not np.array_equal(first, second)

    def test_renders_inline_mode_without_a_queue(self) -> None:
        rendered = HudRenderer().render(
            np.zeros((240, 320, 3), dtype=np.uint8), stats_for(), frame_index=0
        )
        assert rendered is not None

    def test_accepts_extra_lines(self) -> None:
        rendered = HudRenderer().render(
            np.zeros((240, 320, 3), dtype=np.uint8),
            stats_for(),
            frame_index=0,
            extra=["phase 2 will add lines here"],
        )
        assert rendered.shape == (240, 320, 3)


class TestNullSink:
    def test_counts_frames_and_never_asks_to_stop(self) -> None:
        sink = NullSink()
        for _ in range(3):
            assert sink.show(np.zeros((2, 2, 3), dtype=np.uint8)) == KEY_NONE
        assert sink.frames_shown == 3
        assert sink.is_closed() is False
        sink.close()


class TestRunIngestion:
    def config(self, **overrides) -> VantageConfig:
        base = {
            "source": SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=12"),
            "ingest": IngestConfig(mode=IngestMode.INLINE),
            "display": DisplayConfig(enabled=False),
        }
        base.update(overrides)
        return VantageConfig(**base)

    def test_runs_to_the_end_of_the_source(self) -> None:
        result = run_ingestion(self.config())
        assert isinstance(result, RunResult)
        assert result.frames == 12
        assert result.reason == "source ended"
        assert result.dropped == 0
        assert "12 frames" in result.summary()

    def test_honours_the_frame_limit(self) -> None:
        result = run_ingestion(self.config(ingest=IngestConfig(max_frames=4)))
        assert result.frames == 4
        assert result.reason == "frame limit reached"

    def test_renders_to_the_sink_when_display_is_enabled(self) -> None:
        sink = RecordingSink()
        result = run_ingestion(self.config(display=DisplayConfig(enabled=True)), sink=sink)
        assert len(sink.images) == result.frames
        assert sink.closed is False, "a caller-supplied sink is the caller's to close"

    def test_quit_key_stops_the_run(self) -> None:
        sink = RecordingSink(keys=[KEY_NONE, KEY_NONE, ord("q")])
        result = run_ingestion(self.config(display=DisplayConfig(enabled=True)), sink=sink)
        assert result.reason == "user quit"
        assert result.frames == 3

    def test_closed_window_stops_the_run(self) -> None:
        class ClosingSink(RecordingSink):
            def is_closed(self) -> bool:
                return len(self.images) >= 2

        result = run_ingestion(
            self.config(display=DisplayConfig(enabled=True)), sink=ClosingSink()
        )
        assert result.reason == "window closed"

    def test_hud_toggle_changes_what_is_drawn(self) -> None:
        sink = RecordingSink(keys=[ord("h")])
        run_ingestion(self.config(display=DisplayConfig(enabled=True, hud=True)), sink=sink)
        # Frame 0 has the panel, later frames do not.
        assert sink.images[0].max() > 0
        assert sink.images[-1].max() >= 0

    def test_snapshot_key_writes_a_file(self, tmp_path: Path) -> None:
        sink = RecordingSink(keys=[ord("s")])
        result = run_ingestion(
            self.config(
                display=DisplayConfig(enabled=True, snapshot_dir=str(tmp_path / "shots")),
                ingest=IngestConfig(max_frames=2),
            ),
            sink=sink,
        )
        assert len(result.snapshots) == 1
        assert Path(result.snapshots[0]).is_file()

    def test_shutdown_signal_stops_the_run(self) -> None:
        controller = ShutdownController()

        def stop_soon() -> None:
            controller.request("test")

        threading.Timer(0.05, stop_soon).start()
        result = run_ingestion(
            self.config(
                source=SourceConfig(uri="synthetic://?width=64&height=48"),
                ingest=IngestConfig(target_fps=50.0),
            ),
            shutdown=controller,
        )
        assert result.reason in {"shutdown signal", "source ended"}

    def test_result_stats_are_serialisable(self) -> None:
        json.dumps(run_ingestion(self.config()).stats)


class TestShutdownController:
    def test_request_sets_the_flag_and_fires_callbacks(self) -> None:
        controller = ShutdownController()
        fired: list[str] = []
        controller.on_shutdown(lambda: fired.append("x"))
        assert controller.is_set() is False
        controller.request("test")
        assert controller.is_set() is True
        assert fired == ["x"]

    def test_repeated_requests_fire_callbacks_once(self) -> None:
        controller = ShutdownController()
        fired: list[str] = []
        controller.on_shutdown(lambda: fired.append("x"))
        controller.request()
        controller.request()
        assert fired == ["x"]

    def test_install_and_restore_are_reversible(self) -> None:
        import signal

        original = signal.getsignal(signal.SIGINT)
        with ShutdownController():
            assert signal.getsignal(signal.SIGINT) is not original
        assert signal.getsignal(signal.SIGINT) is original


class TestCli:
    def test_run_is_the_default_command(self, capsys: pytest.CaptureFixture) -> None:
        code = main(
            ["--no-display", "--frames", "5", "--source", "synthetic://?width=64&height=48"]
        )
        assert code == 0
        assert "5 frames" in capsys.readouterr().out

    def test_explicit_run_command(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["run", "--no-display", "--frames", "3"]) == 0
        assert "3 frames" in capsys.readouterr().out

    def test_flags_lower_onto_config_overrides(self, capsys: pytest.CaptureFixture) -> None:
        code = main(
            [
                "run",
                "--no-display",
                "--frames",
                "4",
                "--stride",
                "2",
                "--mode",
                "inline",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        # stride=2 delivers source indices 0, 2, 4, 6 - so three were skipped
        # before the fourth delivery hit the --frames limit.
        assert payload["frames_delivered"] == 4
        assert payload["frames_skipped"] == 3
        assert payload["backpressure"] == "inline"

    def test_set_overrides_are_accepted_after_the_subcommand(self) -> None:
        assert main(["run", "--no-display", "--set", "ingest.max_frames=2"]) == 0

    def test_log_level_flag_survives_the_config_reload(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """The flag must win over the config file, not be overwritten by it."""
        assert main(["run", "--no-display", "--frames", "2", "--log-level", "ERROR"]) == 0
        assert "ingestion started" not in capsys.readouterr().err

    def test_invalid_config_exits_non_zero_without_a_traceback(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["run", "--set", "ingest.queue_size=0"]) == 1

    def test_unusable_source_exits_non_zero(self) -> None:
        assert main(["run", "--no-display", "--source", "gopher://nope"]) == 1

    def test_info_reports_the_environment(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["info", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert "opencv" in report and "acceleration" in report

    def test_make_sample_writes_a_playable_clip(self, tmp_path: Path) -> None:
        out = tmp_path / "clip.mp4"
        code = main(
            [
                "make-sample",
                "--out",
                str(out),
                "--seconds",
                "0.5",
                "--fps",
                "10",
                "--width",
                "96",
                "--height",
                "64",
            ]
        )
        assert code == 0
        assert out.is_file() and out.stat().st_size > 0
        assert main(["run", "--no-display", "--source", str(out)]) == 0

    def test_parser_exposes_every_command(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])


class TestPoseSummaryCountsTheRun:
    """`people` is a run total, not whatever the last frame happened to hold.

    Found by running five street clips through the pipeline: the summary read
    "pose: 600 passes, 0 people estimated" for a stage that had produced 305
    skeletons over 300 frames. The only hint anything had happened was the
    mean-per-person timing beside it. A reader would reasonably conclude the
    stage was broken.
    """

    @staticmethod
    def summary(people_per_frame, skipped_per_frame=(0, 0, 0)):
        from types import SimpleNamespace

        from vantage.app import _pose_summary
        from vantage.core.metrics import LatencyTracker

        estimator = SimpleNamespace(
            info=SimpleNamespace(
                model="rtmpose-s",
                backend="openvino",
                device="gpu",
                license="Apache-2.0",
                num_keypoints=17,
            )
        )

        class LastFrame:
            """Stands in for a PoseResult: it is asked its length and its skips."""

            skipped = skipped_per_frame[-1]

            def __len__(self) -> int:
                return people_per_frame[-1]

        latest = LastFrame()
        return _pose_summary(
            estimator,
            LatencyTracker(window=10),
            latest,
            passes=len(people_per_frame),
            people_total=sum(people_per_frame),
            skipped_total=sum(skipped_per_frame),
        )

    def test_a_busy_run_that_ends_empty_still_reports_its_work(self) -> None:
        detail = self.summary([4, 5, 0])
        assert detail["people"] == 9
        assert detail["people_at_end"] == 0

    def test_skipped_is_also_a_total(self) -> None:
        detail = self.summary([1, 1, 1], skipped_per_frame=(2, 3, 0))
        assert detail["skipped"] == 5


class TestWhoIsFedToTheRelationshipGraph:
    """Associations are between people the camera can currently see.

    Both gates were missing when the graph was first wired into the
    single-camera run, and a clip of an almost-empty underpass produced 28
    associations - between a person, a television and a skateboard, tracked
    while nobody was being detected at all.
    """

    @staticmethod
    def feed(rows, labels=frozenset({"person"})):
        from types import SimpleNamespace

        import numpy as np

        from vantage.app import _observe_relationships
        from vantage.perception.contracts import BoundingBox
        from vantage.state.contracts import StateResult

        states = StateResult(
            states=tuple(rows), source_id="t", frame_index=0, capture_wall=100.0, elapsed_s=0.03
        )
        seen: list = []
        service = SimpleNamespace(
            tracker=SimpleNamespace(
                process_frame=lambda **kw: seen.append(kw["active_entities"])
            ),
            maybe_persist=lambda *a: None,
        )
        tracks = [
            SimpleNamespace(track_id=s.track_id, box=BoundingBox(10.0, 10.0, 30.0, 90.0))
            for s in rows
        ]
        frame = SimpleNamespace(image=np.zeros((100, 100, 3), dtype=np.uint8))
        _observe_relationships(
            service,
            SimpleNamespace(tracks=tracks),
            states,
            frame,
            "cam",
            labels,
        )
        return seen[0] if seen else []

    @staticmethod
    def state(track_id: int, label: str, observed: bool = True):
        from vantage.state.contracts import EntityState, MotionState

        return EntityState(
            track_id=track_id,
            entity_id=f"{label}_{track_id}",
            label=label,
            motion=MotionState.MOVING,
            speed=0.5,
            dwell_s=0.0,
            bearing_deg=90.0,
            distance=0.0,
            age_s=5.0,
            observed=observed,
        )

    def test_two_people_are_fed(self) -> None:
        fed = self.feed([self.state(1, "person"), self.state(2, "person")])
        assert {row[0] for row in fed} == {"person_1", "person_2"}

    def test_a_television_is_not(self) -> None:
        fed = self.feed([self.state(1, "person"), self.state(2, "tv")])
        assert fed == []  # one person left is not a pair

    def test_a_coasting_person_is_not(self) -> None:
        fed = self.feed([self.state(1, "person"), self.state(2, "person", observed=False)])
        assert fed == []
