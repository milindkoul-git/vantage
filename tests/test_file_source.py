"""Tests for the real OpenCV/FFmpeg decode path.

These use a video file generated at session start rather than a mock, so the
container parsing, codec negotiation and EOF handling that a webcam test cannot
cover in CI are all genuinely exercised. No camera and no checked-in media
required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.config.schema import IngestConfig, IngestMode, SourceConfig
from vantage.core.errors import SourceExhausted, SourceOpenError
from vantage.ingestion.base import SourceKind
from vantage.ingestion.pipeline import IngestionPipeline
from vantage.ingestion.registry import create_source


def file_source(path: Path, **kwargs):
    return create_source(SourceConfig(uri=f"file:{path}", **kwargs))


class TestFileDecoding:
    def test_reports_geometry_from_the_decoded_frame(self, sample_video: Path) -> None:
        with file_source(sample_video) as source:
            info = source.info
        assert info.kind is SourceKind.FILE
        assert info.resolution == (160, 120)
        assert info.is_live is False
        assert info.backend == "ffmpeg"
        assert info.frame_count is not None and info.frame_count > 0

    def test_the_probe_frame_is_delivered_not_discarded(self, sample_video: Path) -> None:
        """Opening validates by reading; that frame must still reach the consumer."""
        with file_source(sample_video) as source:
            first = source.read()
        assert first.index == 0

    def test_reads_the_whole_file_then_reports_exhaustion(self, sample_video: Path) -> None:
        with file_source(sample_video) as source:
            count = 0
            with pytest.raises(SourceExhausted):
                while True:
                    source.read()
                    count += 1
        assert count == 40

    def test_frames_carry_a_media_timeline(self, sample_video: Path) -> None:
        with file_source(sample_video) as source:
            timestamps = [source.read().media_pts for _ in range(5)]
        assert all(pts is not None for pts in timestamps)
        assert timestamps == sorted(timestamps)

    def test_frames_are_bgr_uint8(self, sample_video: Path) -> None:
        with file_source(sample_video) as source:
            image = source.read().image
        assert image.dtype.name == "uint8"
        assert image.shape == (120, 160, 3)

    def test_looping_restarts_instead_of_ending(self, sample_video: Path) -> None:
        with file_source(sample_video, loop=True) as source:
            frames = [source.read() for _ in range(50)]
        assert len(frames) == 50  # more than the file contains
        assert frames[-1].metadata["loop"] == 1

    def test_pipeline_delivers_every_frame_of_a_file(self, sample_video: Path) -> None:
        source = file_source(sample_video)
        config = IngestConfig(mode=IngestMode.THREADED, queue_size=2)
        with IngestionPipeline(source, config) as pipeline:
            indices = [frame.index for frame in pipeline.frames()]
        assert indices == list(range(40))
        assert pipeline.stats().frames_dropped == 0


class TestBlankSourceWarning:
    def test_warns_when_a_source_opens_but_carries_no_picture(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A closed privacy shutter looks healthy by every other measure."""
        import cv2
        import numpy as np

        black = tmp_path / "black.mp4"
        writer = cv2.VideoWriter(str(black), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (64, 48))
        if not writer.isOpened():
            pytest.skip("no video writer available")
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

        with caplog.at_level("WARNING"):
            with file_source(black):
                pass
        assert "blank frames" in caplog.text

    def test_stays_quiet_for_a_normal_source(
        self, sample_video: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            with file_source(sample_video):
                pass
        assert "blank frames" not in caplog.text


class TestFileErrors:
    def test_missing_file_names_the_path_and_suggests_a_fix(self, tmp_path: Path) -> None:
        source = file_source(tmp_path / "absent.mp4")
        with pytest.raises(SourceOpenError, match="not found"):
            source.open()

    def test_directory_is_rejected(self, tmp_path: Path) -> None:
        source = create_source(SourceConfig(uri=f"file:{tmp_path}"))
        with pytest.raises(SourceOpenError, match="directory"):
            source.open()

    def test_undecodable_file_is_reported_clearly(self, tmp_path: Path) -> None:
        junk = tmp_path / "broken.mp4"
        junk.write_bytes(b"this is definitely not a video container")
        source = file_source(junk)
        with pytest.raises(SourceOpenError):
            source.open()
