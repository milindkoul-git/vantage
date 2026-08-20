"""URI parsing and source construction.

A single string identifies any input the platform can ingest::

    webcam:0                      local capture device, index 0
    file:clips/lobby.mp4          media file (a bare existing path works too)
    synthetic://?fps=30&frames=90 deterministic generated video
    rtsp://user:pw@host/stream1   network stream via FFmpeg

One string means a camera can be reconfigured from a config file, a CLI flag or
a future REST call without any of them knowing which class implements it. New
input types register a scheme here and become usable everywhere at once - the
extension point that keeps the ``SourceKind`` list from being closed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from vantage.config.schema import SourceConfig
from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import ConfigError
from vantage.ingestion.base import FrameSource, SourceKind

SourceFactory = Callable[[SourceConfig, "ParsedURI", Clock], FrameSource]

_STREAM_SCHEMES = frozenset({"rtsp", "rtmp", "http", "https", "udp", "tcp", "srt", "rtp"})
_CAMERA_SCHEMES = frozenset({"webcam", "camera", "device", "cam"})
_MEDIA_SUFFIXES = frozenset(
    {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}
)
_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class ParsedURI:
    """Result of interpreting a source URI."""

    __slots__ = ("kind", "params", "scheme", "target", "uri")

    def __init__(
        self,
        kind: SourceKind,
        target: int | str,
        uri: str,
        scheme: str,
        params: dict[str, str],
    ) -> None:
        self.kind = kind
        self.target = target
        self.uri = uri
        self.scheme = scheme
        self.params = params

    def __repr__(self) -> str:
        return f"<ParsedURI {self.kind.value} target={self.target!r} params={self.params}>"

    def default_id(self) -> str:
        """Stable identifier derived from the URI, used when none is configured."""
        if self.kind is SourceKind.CAMERA:
            return f"cam{self.target}"
        if self.kind is SourceKind.SYNTHETIC:
            return "synthetic"
        if self.kind is SourceKind.STREAM:
            host = urlparse(str(self.target)).hostname or "stream"
            return _ID_SAFE.sub("_", host)
        return _ID_SAFE.sub("_", Path(str(self.target)).stem) or "file"

    def int_param(self, name: str, default: int | None) -> int | None:
        raw = self.params.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"synthetic parameter {name}={raw!r} must be an integer") from exc

    def float_param(self, name: str, default: float | None) -> float | None:
        raw = self.params.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"synthetic parameter {name}={raw!r} must be a number") from exc


def parse_uri(uri: str) -> ParsedURI:
    """Interpret a source URI. Raises :class:`ConfigError` on anything ambiguous."""
    text = (uri or "").strip()
    if not text:
        raise ConfigError("source URI is empty; try 'webcam:0', a file path, or 'synthetic://'")

    scheme, separator, remainder = text.partition(":")
    scheme = scheme.lower()

    # A bare Windows path ("C:\\clips\\a.mp4") parses as scheme 'c'. Single-letter
    # schemes are always drive letters, never protocols.
    if not separator or len(scheme) == 1:
        return _parse_bare(text)

    if scheme in _CAMERA_SCHEMES:
        index_text = remainder.strip().lstrip("/")
        if not index_text.isdigit():
            raise ConfigError(
                f"camera URI {uri!r} needs a numeric device index, e.g. 'webcam:0'"
            )
        return ParsedURI(SourceKind.CAMERA, int(index_text), text, scheme, {})

    if scheme == "synthetic":
        parsed = urlparse(text)
        return ParsedURI(
            SourceKind.SYNTHETIC, "synthetic", text, scheme, dict(parse_qsl(parsed.query))
        )

    if scheme == "file":
        path = remainder
        if path.startswith("//"):
            path = urlparse(text).path.lstrip("/")
        path = path.strip()
        if not path:
            raise ConfigError(f"file URI {uri!r} has no path, e.g. 'file:clips/lobby.mp4'")
        return ParsedURI(SourceKind.FILE, str(Path(path).expanduser()), text, scheme, {})

    if scheme in _STREAM_SCHEMES:
        return ParsedURI(SourceKind.STREAM, text, text, scheme, {})

    raise ConfigError(
        f"unrecognised source URI {uri!r}. Supported forms: 'webcam:N', a file path or "
        "'file:PATH', 'synthetic://?...', or a stream URL "
        f"({', '.join(sorted(_STREAM_SCHEMES))})."
    )


def _parse_bare(text: str) -> ParsedURI:
    """Interpret a URI with no scheme: a device index or a filesystem path."""
    if text.isdigit():
        return ParsedURI(SourceKind.CAMERA, int(text), f"webcam:{text}", "webcam", {})

    path = Path(text).expanduser()
    if path.exists() or path.suffix.lower() in _MEDIA_SUFFIXES:
        return ParsedURI(SourceKind.FILE, str(path), f"file:{path}", "file", {})

    raise ConfigError(
        f"source {text!r} is neither an existing file, a media filename, nor a device index. "
        "Use 'webcam:0' for a camera or 'synthetic://' to run without hardware."
    )


# -- factories ----------------------------------------------------------


def _make_opencv(config: SourceConfig, parsed: ParsedURI, clock: Clock) -> FrameSource:
    from vantage.ingestion.opencv_source import OpenCVSource

    return OpenCVSource(
        target=parsed.target,
        source_id=config.id or parsed.default_id(),
        kind=parsed.kind,
        uri=parsed.uri,
        backend=config.backend,
        width=config.width,
        height=config.height,
        fps=config.fps,
        fourcc=config.fourcc,
        loop=config.loop,
        read_retries=config.read_retries,
        clock=clock,
    )


def _make_synthetic(config: SourceConfig, parsed: ParsedURI, clock: Clock) -> FrameSource:
    from vantage.ingestion.synthetic import SyntheticSource

    return SyntheticSource(
        source_id=config.id or parsed.default_id(),
        width=parsed.int_param("width", config.width or 1280) or 1280,
        height=parsed.int_param("height", config.height or 720) or 720,
        fps=parsed.float_param("fps", config.fps or 30.0) or 30.0,
        frames=parsed.int_param("frames", None),
        seed=parsed.int_param("seed", 7) or 7,
        objects=parsed.int_param("objects", 4) or 0,
        uri=parsed.uri,
        clock=clock,
    )


_FACTORIES: dict[SourceKind, SourceFactory] = {
    SourceKind.CAMERA: _make_opencv,
    SourceKind.FILE: _make_opencv,
    SourceKind.STREAM: _make_opencv,
    SourceKind.SYNTHETIC: _make_synthetic,
}


def register_factory(kind: SourceKind, factory: SourceFactory) -> None:
    """Override or add the implementation used for a source kind.

    The seam for swapping in a PyAV or GStreamer source later without touching
    any calling code.
    """
    _FACTORIES[kind] = factory


def create_source(config: SourceConfig, clock: Clock = SYSTEM_CLOCK) -> FrameSource:
    """Build (but do not open) the source described by ``config``.

    Live sources are additionally wrapped in
    :class:`~vantage.ingestion.resilient.ReconnectingSource` when reconnection
    is enabled, so a USB camera that is unplugged and replugged recovers by
    itself instead of ending the run.
    """
    parsed = parse_uri(config.uri)
    factory = _FACTORIES[parsed.kind]

    is_live = parsed.kind in (SourceKind.CAMERA, SourceKind.STREAM)
    if is_live and config.reconnect.enabled and config.reconnect.max_attempts > 0:
        from vantage.ingestion.resilient import ReconnectingSource

        return ReconnectingSource(
            factory=lambda: factory(config, parsed, clock),
            source_id=config.id or parsed.default_id(),
            uri=parsed.uri,
            policy=config.reconnect,
            clock=clock,
        )
    return factory(config, parsed, clock)


def describe_schemes() -> dict[str, str]:
    """Human-readable catalogue of accepted URI forms, for CLI help."""
    return {
        "webcam:N": "Local capture device by index (also 'camera:N', or a bare number).",
        "file:PATH": "Media file decoded via FFmpeg (a bare existing path also works).",
        "synthetic://?k=v": "Generated video. Params: width, height, fps, frames, seed, objects.",
        "rtsp://HOST/PATH": f"Network stream. Schemes: {', '.join(sorted(_STREAM_SCHEMES))}.",
    }
