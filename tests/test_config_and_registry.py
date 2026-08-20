"""Tests for configuration loading and source-URI interpretation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.config.loader import load_config
from vantage.config.schema import (
    Backpressure,
    IngestMode,
    SourceConfig,
    VantageConfig,
)
from vantage.core.errors import ConfigError
from vantage.ingestion.base import SourceKind
from vantage.ingestion.opencv_source import OpenCVSource, resolve_backend
from vantage.ingestion.registry import create_source, describe_schemes, parse_uri
from vantage.ingestion.resilient import ReconnectingSource
from vantage.ingestion.synthetic import SyntheticSource


class TestDefaults:
    def test_defaults_are_valid_without_any_file(self) -> None:
        config = load_config(path=None, overrides=[])
        assert isinstance(config, VantageConfig)
        assert config.ingest.stride == 1

    def test_bundled_default_file_parses(self) -> None:
        """The shipped configs/default.yaml must always load."""
        from vantage.config.loader import default_config_path

        path = default_config_path()
        if not path.is_file():
            pytest.skip("running from an installed package without the configs directory")
        config = load_config(path, [])
        assert config.source.uri.startswith("synthetic://")


class TestFileLoading:
    def test_reads_and_merges_a_yaml_file(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(
            "app:\n  log_level: DEBUG\ningest:\n  queue_size: 32\n  mode: inline\n",
            encoding="utf-8",
        )
        config = load_config(path, [])
        assert config.app.log_level == "DEBUG"
        assert config.ingest.queue_size == 32
        assert config.ingest.mode is IngestMode.INLINE
        assert config.ingest.stride == 1  # untouched keys keep their defaults

    def test_empty_file_is_equivalent_to_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(path, []).ingest.queue_size == 8

    def test_missing_explicit_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"not found"):
            load_config(tmp_path / "nope.yaml", [])

    def test_malformed_yaml_is_reported_with_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("app: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"invalid YAML"):
            load_config(path, [])

    def test_non_mapping_top_level_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"must be a mapping"):
            load_config(path, [])


class TestOverrides:
    def test_override_values_are_typed_not_strings(self) -> None:
        config = load_config(
            None,
            ["ingest.queue_size=16", "display.enabled=false", "ingest.target_fps=7.5"],
        )
        assert config.ingest.queue_size == 16
        assert config.display.enabled is False
        assert config.ingest.target_fps == pytest.approx(7.5)

    def test_null_override_clears_a_value(self) -> None:
        config = load_config(None, ["ingest.target_fps=null"])
        assert config.ingest.target_fps is None

    def test_override_beats_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("ingest:\n  queue_size: 4\n", encoding="utf-8")
        assert load_config(path, ["ingest.queue_size=99"]).ingest.queue_size == 99

    def test_string_values_survive(self) -> None:
        config = load_config(None, ["source.uri=webcam:0"])
        assert config.source.uri == "webcam:0"

    def test_malformed_override_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"dotted.key=value"):
            load_config(None, ["nonsense"])


class TestStrictness:
    def test_unknown_key_is_an_error_not_a_silent_no_op(self) -> None:
        with pytest.raises(ConfigError, match=r"unknown configuration key"):
            load_config(None, ["ingest.queue_sizes=4"])

    def test_a_near_miss_gets_a_suggestion(self) -> None:
        with pytest.raises(ConfigError, match=r"did you mean 'target_fps'"):
            load_config(None, ["ingest.targt_fps=10"])

    def test_wrong_type_is_reported_with_the_path(self) -> None:
        with pytest.raises(ConfigError, match=r"ingest.queue_size"):
            load_config(None, ["ingest.queue_size=lots"])

    def test_enum_values_are_validated(self) -> None:
        with pytest.raises(ConfigError, match=r"expected one of"):
            load_config(None, ["ingest.backpressure=whenever"])

    @pytest.mark.parametrize(
        "override",
        [
            "ingest.queue_size=0",
            "ingest.stride=0",
            "ingest.target_fps=-1",
            "display.scale=99",
            "app.log_level=CHATTY",
            "app.log_format=xml",
            "source.uri=",
            "source.fourcc=MJP",
            "source.reconnect.backoff=0.5",
            "source.reconnect.max_delay_s=0.1",
        ],
    )
    def test_semantic_validation_rejects_impossible_values(self, override: str) -> None:
        with pytest.raises(ConfigError):
            load_config(None, [override])

    def test_enum_accepts_its_own_string_form(self) -> None:
        assert load_config(None, ["ingest.backpressure=latest"]).ingest.backpressure is (
            Backpressure.LATEST
        )


class TestUriParsing:
    @pytest.mark.parametrize("uri", ["webcam:0", "camera:0", "device:0", "cam:0", "0"])
    def test_camera_forms(self, uri: str) -> None:
        parsed = parse_uri(uri)
        assert parsed.kind is SourceKind.CAMERA
        assert parsed.target == 0
        assert parsed.default_id() == "cam0"

    def test_camera_index_must_be_numeric(self) -> None:
        with pytest.raises(ConfigError, match=r"numeric device index"):
            parse_uri("webcam:front")

    def test_synthetic_query_parameters(self) -> None:
        parsed = parse_uri(
            "synthetic://?width=640&height=480&fps=15&frames=10&seed=3&objects=2"
        )
        assert parsed.kind is SourceKind.SYNTHETIC
        assert parsed.int_param("width", None) == 640
        assert parsed.float_param("fps", None) == pytest.approx(15.0)
        assert parsed.int_param("frames", None) == 10

    def test_synthetic_without_parameters(self) -> None:
        parsed = parse_uri("synthetic://")
        assert parsed.kind is SourceKind.SYNTHETIC
        assert parsed.int_param("width", 1280) == 1280

    def test_bad_synthetic_parameter_is_reported(self) -> None:
        parsed = parse_uri("synthetic://?width=wide")
        with pytest.raises(ConfigError, match=r"must be an integer"):
            parsed.int_param("width", None)

    def test_explicit_file_scheme(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"x")
        parsed = parse_uri(f"file:{target}")
        assert parsed.kind is SourceKind.FILE
        assert Path(parsed.target) == target
        assert parsed.default_id() == "clip"

    def test_bare_existing_path(self, tmp_path: Path) -> None:
        target = tmp_path / "lobby.mkv"
        target.write_bytes(b"x")
        assert parse_uri(str(target)).kind is SourceKind.FILE

    def test_windows_drive_letter_is_not_a_scheme(self) -> None:
        """'C:\\clips\\a.mp4' must not be read as scheme 'c'."""
        parsed = parse_uri(r"C:\clips\a.mp4")
        assert parsed.kind is SourceKind.FILE

    def test_media_filename_that_does_not_exist_yet_is_still_a_file(self) -> None:
        assert parse_uri("recordings/does-not-exist.mp4").kind is SourceKind.FILE

    @pytest.mark.parametrize(
        "uri", ["rtsp://host/stream", "http://host/feed.mjpg", "udp://1.2.3.4:5"]
    )
    def test_stream_schemes(self, uri: str) -> None:
        assert parse_uri(uri).kind is SourceKind.STREAM

    def test_stream_id_is_derived_from_the_host(self) -> None:
        assert parse_uri("rtsp://cam-7.local/stream1").default_id() == "cam-7.local"

    @pytest.mark.parametrize("uri", ["", "   ", "gopher://host/x", "just-a-word"])
    def test_unusable_uris_are_rejected_with_guidance(self, uri: str) -> None:
        with pytest.raises(ConfigError):
            parse_uri(uri)

    def test_schemes_are_documented_for_help_output(self) -> None:
        assert "webcam:N" in describe_schemes()


class TestSourceConstruction:
    def test_synthetic_uri_builds_a_synthetic_source(self) -> None:
        source = create_source(SourceConfig(uri="synthetic://?width=64&height=48&frames=2"))
        assert isinstance(source, SyntheticSource)
        with source:
            assert source.info.resolution == (64, 48)

    def test_explicit_id_overrides_the_derived_one(self) -> None:
        source = create_source(SourceConfig(uri="synthetic://", id="entrance"))
        assert source.source_id == "entrance"

    def test_live_sources_are_wrapped_for_reconnection(self) -> None:
        source = create_source(SourceConfig(uri="webcam:0"))
        assert isinstance(source, ReconnectingSource)

    def test_reconnection_can_be_disabled(self) -> None:
        from vantage.config.schema import ReconnectConfig

        source = create_source(
            SourceConfig(uri="webcam:0", reconnect=ReconnectConfig(enabled=False))
        )
        assert isinstance(source, OpenCVSource)

    def test_recorded_sources_are_not_wrapped(self) -> None:
        source = create_source(SourceConfig(uri="clips/x.mp4"))
        assert isinstance(source, OpenCVSource)


class TestBackendResolution:
    def test_auto_prefers_media_foundation_for_windows_cameras(self) -> None:
        import sys

        _, name = resolve_backend("auto", SourceKind.CAMERA)
        expected = {"win32": "msmf", "darwin": "avfoundation"}.get(sys.platform, "v4l2")
        assert name == expected

    def test_auto_uses_ffmpeg_for_files_and_streams(self) -> None:
        assert resolve_backend("auto", SourceKind.FILE)[1] == "ffmpeg"
        assert resolve_backend("auto", SourceKind.STREAM)[1] == "ffmpeg"

    def test_explicit_backend_is_honoured(self) -> None:
        assert resolve_backend("dshow", SourceKind.CAMERA)[1] == "dshow"

    def test_unknown_backend_lists_the_valid_options(self) -> None:
        from vantage.core.errors import SourceOpenError

        with pytest.raises(SourceOpenError, match=r"valid options"):
            resolve_backend("directdraw", SourceKind.CAMERA)
