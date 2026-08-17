"""Command-line entry point.

Every command resolves configuration the same way - defaults, then a YAML file,
then ``--set`` overrides, then the convenience flags - so a flag and a config
key can never diverge in behaviour. The typed flags are sugar that lowers onto
the same override mechanism.

Commands::

    vantage run           ingest from the configured source, with a live viewer
    vantage probe         report which cameras and URI forms are usable here
    vantage info          report the environment as the platform sees it
    vantage make-sample   write a deterministic test clip to disk
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

from vantage import __version__
from vantage.core.errors import VantageError
from vantage.core.lifecycle import ShutdownController
from vantage.core.logging import configure_logging, get_logger

log = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


COMMANDS = ("run", "probe", "info", "make-sample")


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they are accepted *after* the
    # subcommand, which is where people naturally type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="path to a YAML config file")
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set ingest.queue_size=16 (repeatable)",
    )
    common.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    common.add_argument("--log-format", default=None, choices=["console", "json"])

    parser = argparse.ArgumentParser(
        prog="vantage",
        description="Vantage - intelligent video analytics platform (Phase 1: ingestion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  vantage run --source synthetic://          run without any hardware\n"
            "  vantage run --source webcam:0              live camera with the HUD\n"
            "  vantage run --source clip.mp4 --realtime   play a file at natural speed\n"
            "  vantage run --no-display --frames 300      headless throughput check\n"
            "  vantage probe                              list usable cameras\n"
            "\n'run' is the default command, so 'vantage --source webcam:0' also works.\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"vantage {__version__}")

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser(
        "run", parents=[common], help="ingest frames from the configured source"
    )
    run.add_argument("--source", default=None, help="source URI (webcam:0, file path, synthetic://)")
    run.add_argument("--id", dest="source_id", default=None, help="identifier stamped on frames")
    run.add_argument("--backend", default=None, help="capture backend: auto, msmf, dshow, ffmpeg...")
    run.add_argument("--width", type=int, default=None, help="requested capture width")
    run.add_argument("--height", type=int, default=None, help="requested capture height")
    run.add_argument("--fps", type=float, default=None, help="requested capture frame rate")
    run.add_argument("--fourcc", default=None, help="capture codec, e.g. MJPG")
    run.add_argument("--loop", action="store_true", help="restart file sources at EOF")
    run.add_argument("--frames", type=int, default=None, help="stop after N delivered frames")
    run.add_argument("--target-fps", type=float, default=None, help="throttle delivery to N fps")
    run.add_argument("--stride", type=int, default=None, help="deliver every Nth frame")
    run.add_argument("--queue-size", type=int, default=None, help="frames buffered before the consumer")
    run.add_argument("--mode", choices=["threaded", "inline"], default=None)
    run.add_argument(
        "--backpressure",
        choices=["auto", "latest", "block", "drop_new"],
        default=None,
        help="what to discard when the consumer falls behind",
    )
    run.add_argument("--realtime", action="store_true", help="pace recorded input to its own timeline")
    run.add_argument("--no-display", action="store_true", help="run headless")
    run.add_argument("--no-hud", action="store_true", help="show video without the telemetry panel")
    run.add_argument("--scale", type=float, default=None, help="window scale factor")
    run.add_argument("--json", action="store_true", help="print the run summary as JSON")

    probe = sub.add_parser(
        "probe", parents=[common], help="report usable cameras and supported URI forms"
    )
    probe.add_argument("--max-index", type=int, default=4, help="highest camera index to try")
    probe.add_argument("--backend", default="auto", help="backend to probe with")
    probe.add_argument("--json", action="store_true")

    info = sub.add_parser(
        "info", parents=[common], help="report the environment as the platform sees it"
    )
    info.add_argument("--json", action="store_true")

    sample = sub.add_parser(
        "make-sample", parents=[common], help="write a deterministic synthetic clip"
    )
    sample.add_argument("--out", type=Path, default=Path("samples/sample.mp4"))
    sample.add_argument("--seconds", type=float, default=10.0)
    sample.add_argument("--fps", type=float, default=30.0)
    sample.add_argument("--width", type=int, default=1280)
    sample.add_argument("--height", type=int, default=720)
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument("--objects", type=int, default=4)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_with_default_command(argv))
    command = args.command or "run"

    configure_logging(level=args.log_level or "INFO", fmt=args.log_format or "console")

    try:
        if command == "run":
            return _cmd_run(args)
        if command == "probe":
            return _cmd_probe(args)
        if command == "info":
            return _cmd_info(args)
        if command == "make-sample":
            return _cmd_make_sample(args)
        parser.error(f"unknown command {command!r}")
        return EXIT_ERROR
    except VantageError as exc:
        # Expected, actionable failures: report the message, not a traceback.
        log.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - signal handler normally wins
        log.warning("interrupted")
        return EXIT_INTERRUPTED


def _with_default_command(argv: Sequence[str] | None) -> list[str]:
    """Treat ``run`` as the default subcommand.

    ``vantage --source webcam:0`` is what people type; requiring ``run`` for the
    overwhelmingly common case is friction with no benefit.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        return ["run"]
    first = tokens[0]
    if first in COMMANDS or first in ("-h", "--help", "--version"):
        return tokens
    return ["run", *tokens]


# -- commands -----------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from vantage.app import run_ingestion
    from vantage.config.loader import load_config

    overrides = list(args.overrides)
    overrides += _flag_overrides(args)
    config = load_config(args.config, overrides)
    configure_logging(level=config.app.log_level, fmt=config.app.log_format)

    with ShutdownController() as controller:
        result = run_ingestion(config, shutdown=controller)

    if getattr(args, "json", False):
        print(json.dumps({"summary": result.summary(), **result.stats}, indent=2))
    else:
        print(result.summary())
    return EXIT_OK


def _cmd_probe(args: argparse.Namespace) -> int:
    from vantage.ingestion.opencv_source import probe_cameras
    from vantage.ingestion.registry import describe_schemes

    cameras = probe_cameras(max_index=args.max_index, backend=args.backend)
    if args.json:
        print(json.dumps({"cameras": cameras, "schemes": describe_schemes()}, indent=2))
        return EXIT_OK

    print("Cameras")
    working = [entry for entry in cameras if entry["available"]]
    if not working:
        print("  none responded. Check the OS camera privacy setting and that no other")
        print("  application holds the device. 'synthetic://' works with no hardware.")
    for entry in working:
        print(
            f"  webcam:{entry['index']}  {entry['width']}x{entry['height']}"
            f"  fps={entry['fps'] if entry['fps'] else 'unknown'}"
            f"  fourcc={entry['fourcc']}  backend={entry['backend']}"
            f"  open={entry['open_ms']}ms"
        )
    unavailable = [entry["index"] for entry in cameras if not entry["available"]]
    if unavailable:
        print(f"  (no response from indices: {unavailable})")

    print("\nSupported source URIs")
    for form, description in describe_schemes().items():
        print(f"  {form:22s} {description}")
    return EXIT_OK


def _cmd_info(args: argparse.Namespace) -> int:
    report = environment_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return EXIT_OK
    for section, values in report.items():
        print(section)
        for key, value in values.items():
            print(f"  {key:22s} {value}")
    return EXIT_OK


def _cmd_make_sample(args: argparse.Namespace) -> int:
    import cv2

    from vantage.ingestion.synthetic import SyntheticSource

    total = max(1, int(round(args.seconds * args.fps)))
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    source = SyntheticSource(
        source_id="sample",
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=total,
        seed=args.seed,
        objects=args.objects,
    )
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise VantageError(
            f"could not open {out} for writing. Try an .mp4 or .avi extension, "
            "and check that the directory is writable."
        )

    written = 0
    try:
        with source:
            for _ in range(total):
                writer.write(source.read().image)
                written += 1
    finally:
        writer.release()

    size_mb = out.stat().st_size / (1024 * 1024) if out.exists() else 0.0
    print(
        f"wrote {written} frames to {out} "
        f"({args.width}x{args.height} @ {args.fps:g} fps, {size_mb:.1f} MB)"
    )
    return EXIT_OK


# -- helpers ------------------------------------------------------------


def _flag_overrides(args: argparse.Namespace) -> list[str]:
    """Lower the convenience flags onto config overrides.

    One resolution path means a flag can never drift from its config key.
    """
    mapping: list[tuple[str, object]] = [
        ("source.uri", args.source),
        ("source.id", args.source_id),
        ("source.backend", args.backend),
        ("source.width", args.width),
        ("source.height", args.height),
        ("source.fps", args.fps),
        ("source.fourcc", args.fourcc),
        ("ingest.max_frames", args.frames),
        ("ingest.target_fps", args.target_fps),
        ("ingest.stride", args.stride),
        ("ingest.queue_size", args.queue_size),
        ("ingest.mode", args.mode),
        ("ingest.backpressure", args.backpressure),
        ("display.scale", args.scale),
        # The logging flags must go through the config too, or the reload inside
        # _cmd_run would quietly discard them.
        ("app.log_level", args.log_level),
        ("app.log_format", args.log_format),
    ]
    overrides = [f"{key}={value}" for key, value in mapping if value is not None]

    # store_true flags: only meaningful when set, so absence leaves the file value.
    if args.loop:
        overrides.append("source.loop=true")
    if args.realtime:
        overrides.append("ingest.realtime=true")
    if args.no_display:
        overrides.append("display.enabled=false")
    if args.no_hud:
        overrides.append("display.hud=false")
    return overrides


def environment_report() -> dict[str, dict[str, object]]:
    """What the platform can see about this machine. Used by ``vantage info``."""
    import os

    import cv2
    import numpy as np

    build_flags = {
        "opencl": bool(cv2.ocl.haveOpenCL()),
        "threads": cv2.getNumThreads(),
        "cpu_baseline_optimised": cv2.useOptimized(),
    }
    camera_backends = [
        cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getCameraBackends()
    ]
    stream_backends = [
        cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getStreamBackends()
    ]

    return {
        "platform": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "vantage": {"version": __version__},
        "opencv": {
            "version": cv2.__version__,
            "numpy": np.__version__,
            **{k: str(v) for k, v in build_flags.items()},
            "camera_backends": ", ".join(camera_backends),
            "stream_backends": ", ".join(stream_backends),
        },
        "acceleration": _acceleration_report(),
    }


def _acceleration_report() -> dict[str, object]:
    """Report inference acceleration that is actually present.

    Phase 1 runs no models, but the answer here determines the Phase 2 runtime
    choice, so it is worth reporting honestly and early rather than discovering
    it during a benchmark.
    """
    import cv2

    report: dict[str, object] = {}
    try:
        report["opencl_device"] = (
            cv2.ocl.Device.getDefault().name() if cv2.ocl.haveOpenCL() else "none"
        )
    except cv2.error:  # pragma: no cover - driver dependent
        report["opencl_device"] = "unavailable"

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.cuda.is_available()
    except ImportError:
        report["torch"] = "not installed (not required for Phase 1)"

    try:
        import onnxruntime

        report["onnxruntime"] = onnxruntime.__version__
        report["onnx_providers"] = ", ".join(onnxruntime.get_available_providers())
    except ImportError:
        report["onnxruntime"] = "not installed (candidate runtime for Phase 2)"

    return report


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
