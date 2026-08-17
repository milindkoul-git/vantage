"""Shared fixtures.

The whole suite runs without a camera. Where a real decoder path must be
exercised, a video file is generated on the fly from the synthetic source -
so the file-source tests test the real OpenCV/FFmpeg path rather than a mock of
it, while still needing no checked-in binary assets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import pytest

from vantage.config.schema import IngestConfig, SourceConfig, VantageConfig
from vantage.ingestion.synthetic import SyntheticSource


@pytest.fixture(autouse=True)
def isolate_logging():
    """Give each test a pristine ``vantage`` logger.

    ``configure_logging`` deliberately stops the platform's records from
    propagating into a host application's root logger, and the CLI tests call
    it with assorted levels. Without this reset, one test's logging setup would
    silence ``caplog`` in the next.
    """
    logger = logging.getLogger("vantage")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
    handlers, level, propagate = saved
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def synthetic_source() -> SyntheticSource:
    return SyntheticSource(
        source_id="test", width=160, height=120, fps=30.0, frames=20, seed=3, objects=2
    )


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real 40-frame MP4 written by OpenCV's bundled FFmpeg."""
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    width, height, fps, total = 160, 120, 20.0, 40

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("no video writer available in this OpenCV build")

    source = SyntheticSource(
        source_id="gen", width=width, height=height, fps=fps, frames=total, seed=1, objects=2
    )
    with source:
        for _ in range(total):
            writer.write(source.read().image)
    writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("video file could not be encoded in this environment")
    return path


@pytest.fixture
def headless_config() -> VantageConfig:
    """A short, display-free run over the synthetic source."""
    return VantageConfig(
        source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=25"),
        ingest=IngestConfig(max_frames=10),
        display=_no_display(),
    )


def _no_display():
    from vantage.config.schema import DisplayConfig

    return DisplayConfig(enabled=False)
