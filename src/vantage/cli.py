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


_DEFAULT_COMMAND = "run"


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
        description="Vantage - intelligent video analytics platform "
        "(ingestion, detection, tracking)",
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
    run.add_argument("--detect", action="store_true", help="enable object detection")
    run.add_argument("--model", default=None, help="detector to use (see 'vantage models list')")
    run.add_argument(
        "--detect-backend", choices=["auto", "onnxruntime", "openvino"], default=None
    )
    run.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
    run.add_argument("--conf", type=float, default=None, help="detection confidence threshold")
    run.add_argument(
        "--detect-interval",
        type=int,
        default=None,
        help="run the detector on every Nth delivered frame (1 = every frame)",
    )
    run.add_argument(
        "--classes",
        default=None,
        help="comma-separated labels to keep, e.g. --classes person,car",
    )
    run.add_argument(
        "--track",
        action="store_true",
        help="enable multi-object tracking (implies --detect)",
    )
    run.add_argument(
        "--track-min-hits",
        type=int,
        default=None,
        help="frames a track must be corroborated on before it is published",
    )
    run.add_argument(
        "--track-max-lost",
        type=float,
        default=None,
        help="seconds a track survives unmatched before it is dropped",
    )
    run.add_argument(
        "--pose",
        action="store_true",
        help="estimate human pose for tracked people (implies --track)",
    )
    run.add_argument("--pose-model", default=None, help="pose model (see 'vantage models list')")
    run.add_argument(
        "--pose-interval",
        type=int,
        default=None,
        help="estimate pose on every Nth tracking step (1 = every step)",
    )
    run.add_argument(
        "--pose-max-persons",
        type=int,
        default=None,
        help="most people to estimate per frame, largest boxes first",
    )
    run.add_argument(
        "--no-face-keypoints",
        action="store_true",
        help="drop the five head landmarks (nose, eyes, ears) before they are constructed",
    )
    run.add_argument(
        "--no-spatial",
        action="store_true",
        help="disable zones and relations, which are otherwise on with tracking",
    )
    run.add_argument(
        "--no-activity",
        action="store_true",
        help="disable activity recognition, which is otherwise on with tracking",
    )
    run.add_argument(
        "--no-state",
        action="store_true",
        help="disable motion/dwell state estimation, which is otherwise on with tracking",
    )
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

    models = sub.add_parser(
        "models", parents=[common], help="list, fetch and verify detection models"
    )
    models.add_argument(
        "action", choices=["list", "pull", "remove", "verify"], nargs="?", default="list"
    )
    models.add_argument("name", nargs="?", default=None, help="catalog key, e.g. yolox-nano")
    models.add_argument("--model-dir", default=None, help="where models are cached")
    models.add_argument("--json", action="store_true")

    discover = sub.add_parser(
        "discover",
        parents=[common],
        help="open-vocabulary detection: find whatever you can name, on one frame",
    )
    discover.add_argument(
        "--prompts",
        required=True,
        help="comma-separated things to look for, e.g. --prompts 'pen, stapler, mug'",
    )
    discover.add_argument(
        "--source", default=None, help="source URI to grab a frame from (default: webcam:0)"
    )
    discover.add_argument("--image", default=None, help="a still image instead of a source")
    discover.add_argument("--model", default=None, help="open-vocabulary model to use")
    discover.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
    discover.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    discover.add_argument("--model-dir", default=None)
    discover.add_argument(
        "--save", default=None, help="write an annotated PNG to this path"
    )
    discover.add_argument("--json", action="store_true")

    track = sub.add_parser(
        "track",
        parents=[common],
        help="evaluate and tune the tracker against ground-truth scenarios",
    )
    track.add_argument(
        "action",
        choices=["eval", "tune", "scenarios"],
        nargs="?",
        default="eval",
        help="eval: score the current parameters; tune: search for better ones; "
        "scenarios: list the benchmark scenarios",
    )
    track.add_argument(
        "--scenarios",
        default=None,
        help="comma-separated scenario names (default: all)",
    )
    track.add_argument(
        "--rounds", type=int, default=3, help="maximum tuning sweeps over the parameter space"
    )
    track.add_argument(
        "--validate",
        action="store_true",
        help="also score against the held-out detector profiles",
    )
    track.add_argument("--json", action="store_true")

    activity = sub.add_parser(
        "activity",
        parents=[common],
        help="evaluate activity recognition against scripted ground truth",
    )
    activity.add_argument(
        "action",
        choices=["eval", "scenarios"],
        nargs="?",
        default="eval",
        help="eval: score the rules; scenarios: list what each one checks",
    )
    activity.add_argument(
        "--scenarios",
        default=None,
        help="comma-separated scenario names (default: all)",
    )
    activity.add_argument("--json", action="store_true")

    spatial = sub.add_parser(
        "spatial",
        parents=[common],
        help="evaluate zones and relations against scripted ground truth",
    )
    spatial.add_argument(
        "action",
        choices=["eval", "scenarios"],
        nargs="?",
        default="eval",
        help="eval: score the geometry; scenarios: list what each one checks",
    )
    spatial.add_argument(
        "--scenarios", default=None, help="comma-separated scenario names (default: all)"
    )
    spatial.add_argument("--json", action="store_true")

    bench = sub.add_parser(
        "bench", parents=[common], help="benchmark detection backends on this machine"
    )
    bench.add_argument("--model", default=None, help="model to benchmark (default: yolox-nano)")
    bench.add_argument(
        "--backends",
        default="all",
        help="comma-separated backend/device pairs, e.g. 'onnxruntime:cpu,openvino:gpu', "
        "or 'all' to try every combination this machine supports",
    )
    bench.add_argument("--frames", type=int, default=50, help="timed iterations per backend")
    bench.add_argument("--warmup", type=int, default=5, help="untimed iterations per backend")
    bench.add_argument(
        "--image", default=None, help="image to benchmark on (default: a synthetic frame)"
    )
    bench.add_argument("--model-dir", default=None)
    bench.add_argument("--json", action="store_true")

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

    # Recorded on the parser so _with_default_command can never fall out of
    # sync with the registered subcommands.
    parser.set_defaults(_commands=tuple(sub.choices))
    return parser


def _subcommands(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    """Every registered subcommand name, straight from the parser."""
    return tuple(parser.get_default("_commands") or ())


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_with_default_command(argv, _subcommands(parser)))
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
        if command == "models":
            return _cmd_models(args)
        if command == "bench":
            return _cmd_bench(args)
        if command == "track":
            return _cmd_track(args)
        if command == "activity":
            return _cmd_activity(args)
        if command == "spatial":
            return _cmd_spatial(args)
        if command == "discover":
            return _cmd_discover(args)
        parser.error(f"unknown command {command!r}")
        return EXIT_ERROR
    except VantageError as exc:
        # Expected, actionable failures: report the message, not a traceback.
        log.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - signal handler normally wins
        log.warning("interrupted")
        return EXIT_INTERRUPTED


def _with_default_command(argv: Sequence[str] | None, commands: tuple[str, ...]) -> list[str]:
    """Treat ``run`` as the default subcommand.

    ``vantage --source webcam:0`` is what people type; requiring ``run`` for the
    overwhelmingly common case is friction with no benefit.

    ``commands`` is read off the parser rather than kept in a constant here.
    The hand-maintained version of this list silently omitted ``track`` when
    that command was added, which turned ``vantage track eval`` into
    ``vantage run track eval`` and produced an argument error naming the wrong
    thing. Deriving it removes the whole class of mistake.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        return [_DEFAULT_COMMAND]
    first = tokens[0]
    if first in commands or first in ("-h", "--help", "--version"):
        return tokens
    return [_DEFAULT_COMMAND, *tokens]


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


def _cmd_models(args: argparse.Namespace) -> int:
    from vantage.perception.catalog import CATALOG, get_model_spec
    from vantage.perception.store import ModelStore

    store = ModelStore(args.model_dir or "models")

    if args.action == "list":
        entries = []
        for spec in CATALOG.values():
            cached = store.is_cached(spec)
            entries.append(
                {
                    "key": spec.key,
                    "task": spec.task,
                    "input": f"{spec.input_size[1]}x{spec.input_size[0]}",
                    "size_mb": round(spec.size_bytes / 1e6, 1),
                    "map": spec.map_50_95,
                    # An open-vocabulary model has no fixed class list; reporting
                    # "1" for its placeholder label would be actively misleading.
                    # For a pose model this column counts keypoints, not
                    # classes, which is why the header says OUTPUTS and the TASK
                    # column exists to say which it is. Sharing one number under
                    # a "CLASSES" heading would read as a 17-class detector.
                    "outputs": (
                        "open" if spec.label_set == "open-vocabulary"
                        else str(spec.num_classes)
                    ),
                    "license": spec.license,
                    "cached": cached,
                    "path": str(store.path_for(spec)) if cached else None,
                    "source": spec.source,
                }
            )
        if args.json:
            print(json.dumps({"model_dir": str(store.directory), "models": entries}, indent=2))
            return EXIT_OK

        print(f"Models (cache: {store.directory})\n")
        # Width driven by the longest key so a new catalog entry cannot silently
        # break the alignment, which is what happened when the D-FINE keys landed.
        key_width = max(14, *(len(e["key"]) for e in entries)) if entries else 14
        print(
            f"  {'KEY':{key_width}s} {'TASK':6s} {'INPUT':9s} {'SIZE':>7s} {'mAP':>6s}  "
            f"{'OUTPUTS':>7s}  {'LICENSE':11s} STATUS"
        )
        for entry in entries:
            status = "cached" if entry["cached"] else "not downloaded"
            accuracy = f"{entry['map']:.1f}" if entry["map"] else "n/a"
            print(
                f"  {entry['key']:{key_width}s} {entry['task']:6s} {entry['input']:9s} "
                f"{entry['size_mb']:5.1f}MB "
                f"{accuracy:>6s}  {entry['outputs']:>7s}  {entry['license']:11s} {status}"
            )
        print("\n  OUTPUTS is classes for a detector and keypoints for a pose model.")
        print("\n  Fetch one with: vantage models pull <KEY>")
        print("  Weights are downloaded on demand and verified against a pinned SHA-256.")
        return EXIT_OK

    if not args.name:
        raise VantageError(f"'vantage models {args.action}' needs a model name, e.g. yolox-nano")
    spec = get_model_spec(args.name)

    if args.action == "pull":
        path = store.ensure(spec, progress=_download_progress)
        print(f"\r{spec.key}: ready at {path} ({spec.size_bytes / 1e6:.1f} MB, {spec.license})")
        return EXIT_OK

    if args.action == "verify":
        if not store.is_cached(spec):
            print(f"{spec.key}: not downloaded")
            return EXIT_ERROR
        store.ensure(spec, allow_download=False)  # raises on mismatch
        print(f"{spec.key}: checksum verified ({spec.sha256[:16]}...)")
        return EXIT_OK

    if args.action == "remove":
        removed = store.remove(spec)
        print(f"{spec.key}: {'removed' if removed else 'was not cached'}")
        return EXIT_OK

    return EXIT_ERROR


def _download_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    percent = 100.0 * done / total
    print(f"\rdownloading... {percent:5.1f}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", end="")


def _cmd_discover(args: argparse.Namespace) -> int:
    """Run one open-vocabulary pass and report what was found.

    Deliberately single-frame. The model takes roughly two seconds per pass on
    this class of hardware, so anything resembling a live loop would be a lie -
    see :mod:`vantage.perception.discovery` for the measurements behind that.
    """
    import json

    import cv2

    from vantage.perception.discovery import build_discovery_engine

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise VantageError("--prompts was empty; try --prompts 'pen, stapler, mug'")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise VantageError(f"could not read image: {args.image}")
        origin = args.image
    else:
        from vantage.config.schema import SourceConfig
        from vantage.ingestion.registry import create_source

        uri = args.source or "webcam:0"
        source = create_source(SourceConfig(uri=uri))
        source.open()
        try:
            # A few frames of headroom: webcams need several to settle exposure,
            # and discovering objects in an under-exposed first frame is a poor
            # test of a model that costs two seconds to run.
            for _ in range(8):
                frame = source.read()
        finally:
            source.close()
        image = frame.editable_copy()
        origin = uri

    from vantage.config.loader import load_config

    # Only --config and --set apply here; the run-loop flags are not part of
    # this command, so there is nothing to lower onto the config.
    config = load_config(args.config, list(args.overrides or []))

    engine = build_discovery_engine(
        prompts,
        model=args.model or "grounding-dino-tiny",
        device=args.device or config.detection.device,
        model_dir=args.model_dir or config.detection.model_dir,
        allow_download=config.detection.allow_download,
    )
    try:
        def _progress(index: int, total: int, prompt: str) -> None:
            print(f"  [{index + 1}/{total}] looking for {prompt!r}...", flush=True)

        print(
            f"open-vocabulary search over {len(prompts)} prompt(s). "
            "Each is a separate pass and takes several seconds."
        )
        result = engine.discover(image, confidence=args.conf, progress=_progress)
    finally:
        engine.close()

    if args.json:
        print(
            json.dumps(
                {
                    "source": origin,
                    "model": result.model,
                    "prompts": list(result.prompts),
                    "elapsed_ms": round(result.elapsed_ms, 1),
                    "detections": [
                        {
                            "label": d.label,
                            "confidence": round(d.confidence, 4),
                            "box": [round(v, 1) for v in d.box.xyxy],
                        }
                        for d in result.detections
                    ],
                },
                indent=2,
            )
        )
    else:
        print(f"\n{origin}: {result.describe()}")
        for detection in result.detections:
            x1, y1, x2, y2 = detection.box.to_int()
            print(
                f"   {detection.label:<22} {detection.confidence:.2f}  "
                f"[{x1},{y1},{x2},{y2}]"
            )
        if not result.detections:
            print("   (try a lower --conf, or different words)")

    if args.save:
        from vantage.perception.contracts import DetectionResult
        from vantage.viz.overlay import draw_detections

        annotated = draw_detections(
            image,
            DetectionResult(
                detections=result.detections,
                source_id="discover",
                frame_index=0,
                capture_wall=0.0,
                frame_size=result.frame_size,
            ),
        )
        if not cv2.imwrite(args.save, annotated):
            raise VantageError(f"could not write {args.save}")
        print(f"\nannotated frame -> {args.save}")

    return EXIT_OK


def _cmd_spatial(args: argparse.Namespace) -> int:
    """Score zone assignment and relation detection against ground truth."""
    from vantage.spatial.evaluation import aggregate, evaluate, format_table
    from vantage.spatial.scenarios import SCENARIOS, build_suite

    names = [n.strip() for n in args.scenarios.split(",")] if args.scenarios else None

    if args.action == "scenarios":
        if args.json:
            print(
                json.dumps(
                    {
                        name: {
                            "description": scenario.description,
                            "seconds": scenario.seconds,
                            "actors": len(scenario.actors),
                            "zones": [z.name for z in scenario.zones],
                            "expect": [
                                f"{r.value}:{a}:{b}" for r, a, b in scenario.expect
                            ],
                            "forbidden": [
                                f"{r.value}:{a}:{b}" for r, a, b in scenario.forbidden
                            ],
                        }
                        for name, scenario in SCENARIOS.items()
                    },
                    indent=2,
                )
            )
            return EXIT_OK
        print("Spatial scenarios\n")
        for name, scenario in SCENARIOS.items():
            print(f"  {name:22s} {scenario.seconds:5.1f}s  {scenario.description}")
            if scenario.forbidden:
                print(
                    f"  {'':22s}         must never fire: "
                    + ", ".join(f"{r.value}" for r, _, _ in scenario.forbidden)
                )
        return EXIT_OK

    results = [evaluate(scenario) for scenario in build_suite(names)]
    pooled = aggregate(results)

    if args.json:
        print(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "name": m.scenario,
                            "relations_found": m.expected_found,
                            "relations_expected": len(m.expected),
                            "zone_events_found": m.zone_events_found,
                            "zone_events_expected": m.zone_events_expected,
                            "forbidden_firings": m.forbidden_firings,
                            "peak_confidence": {
                                k: round(v, 3) for k, v in m.peak_confidence.items()
                            },
                            "passed": m.passed,
                        }
                        for m in results
                    ],
                    "pooled": {
                        "relations_found": pooled.expected_found,
                        "relations_expected": len(pooled.expected),
                        "zone_events_found": pooled.zone_events_found,
                        "zone_events_expected": pooled.zone_events_expected,
                        "forbidden_firings": pooled.forbidden_firings,
                    },
                },
                indent=2,
            )
        )
    else:
        print(format_table(results))

    failed = [m for m in results if not m.passed]
    if failed:
        print(f"\n{len(failed)} scenario(s) failed: {', '.join(m.scenario for m in failed)}")
        return EXIT_ERROR
    return EXIT_OK


def _cmd_activity(args: argparse.Namespace) -> int:
    """Score the activity rules against scripted ground truth."""
    from vantage.activity.evaluation import aggregate, evaluate, format_table
    from vantage.activity.scenarios import SCENARIOS, build_suite

    names = [n.strip() for n in args.scenarios.split(",")] if args.scenarios else None

    if args.action == "scenarios":
        if args.json:
            print(
                json.dumps(
                    {
                        name: {
                            "description": scenario.description,
                            "duration_s": round(scenario.duration_s, 1),
                            "events": [a.value for a in scenario.events],
                            "forbidden": sorted(a.value for a in scenario.forbidden),
                        }
                        for name, scenario in SCENARIOS.items()
                    },
                    indent=2,
                )
            )
            return EXIT_OK
        print("Activity scenarios\n")
        for name, scenario in SCENARIOS.items():
            print(f"  {name:32s} {scenario.duration_s:5.1f}s  {scenario.description}")
            if scenario.events:
                print(f"  {'':32s}        expects: {', '.join(a.value for a in scenario.events)}")
            if scenario.forbidden:
                print(
                    f"  {'':32s}        must never fire: "
                    f"{', '.join(sorted(a.value for a in scenario.forbidden))}"
                )
        return EXIT_OK

    results = [evaluate(scenario) for scenario in build_suite(names)]
    pooled = aggregate(results)

    if args.json:
        print(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "name": m.scenario,
                            "recall": round(m.recall, 4),
                            "scored_frames": m.scored_frames,
                            "events_found": m.events_found,
                            "events_expected": len(m.events_expected),
                            "event_latency_s": {k: round(v, 3) for k, v in m.event_latency_s.items()},
                            "forbidden_firings": m.forbidden_firings,
                            "passed": m.passed,
                        }
                        for m in results
                    ],
                    "pooled": {
                        "recall": round(pooled.recall, 4),
                        "events_found": pooled.events_found,
                        "events_expected": len(pooled.events_expected),
                        "forbidden_firings": pooled.forbidden_firings,
                    },
                },
                indent=2,
            )
        )
    else:
        print(format_table(results))

    # A forbidden firing or a missed event is a failure, not a low score. The
    # exit code says so, so this can gate a build.
    failed = [m for m in results if not m.passed]
    if failed:
        print(f"\n{len(failed)} scenario(s) failed: {', '.join(m.scenario for m in failed)}")
        return EXIT_ERROR
    return EXIT_OK


def _cmd_track(args: argparse.Namespace) -> int:
    """Evaluate or tune the tracker against the ground-truth scenarios.

    Deliberately a first-class command rather than a script in a corner. The
    claim "tracking works" is only meaningful if anyone can reproduce the number
    behind it, and that has to be one command away.
    """
    import json

    from vantage.tracking.bytetrack import TrackerParams
    from vantage.tracking.evaluation import format_table
    from vantage.tracking.scenarios import SCENARIOS, build_suite
    from vantage.tracking.tuning import (
        as_config_lines,
        assess,
        default_params_source,
        search,
        validate,
    )

    if args.action == "scenarios":
        rows = []
        for name in sorted(SCENARIOS):
            scenario = SCENARIOS[name]()
            rows.append(
                {
                    "name": name,
                    "frames": len(scenario),
                    "objects": scenario.object_count,
                    "instances": scenario.instance_count,
                    "description": scenario.description,
                }
            )
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{'scenario':<12} {'frames':>7} {'objects':>8}  description")
            print("-" * 76)
            for row in rows:
                print(
                    f"{row['name']:<12} {row['frames']:>7} {row['objects']:>8}  "
                    f"{row['description']}"
                )
        return EXIT_OK

    names = (
        [part.strip() for part in args.scenarios.split(",") if part.strip()]
        if args.scenarios
        else None
    )
    try:
        suite = build_suite(names)
    except ValueError as exc:
        raise VantageError(str(exc)) from exc

    if args.action == "tune":
        best, evaluations = search(suite, rounds=args.rounds)
        if args.json:
            print(
                json.dumps(
                    {
                        "evaluations": evaluations,
                        "objective": best.objective,
                        "summary": best.summary,
                        "params": default_params_source(best.params),
                    },
                    indent=2,
                )
            )
            return EXIT_OK
        print(format_table(list(best.metrics)))
        print(f"\n{evaluations} parameter sets evaluated")
        print(f"shipped defaults : {default_params_source(TrackerParams())}")
        print(f"search result    : {default_params_source(best.params)}")
        print("\nAs configuration:")
        for line in as_config_lines(best.params):
            print(f"  {line}")
        # Reporting the held-out result unprompted, because a tuning number
        # without one is the easiest kind of self-deception to publish.
        comparison = validate(best.params, scenarios=suite)
        print("\nHeld-out profiles (not used by the search):")
        for label, summary in comparison.items():
            print(
                f"  {label:<9} objective {summary['objective']:.4f}  "
                f"IDF1 {summary['idf1']:.1%}  MOTA {summary['mota']:.1%}  "
                f"IDs {int(summary['id_switches'])}"
            )
        return EXIT_OK

    candidate = assess(TrackerParams(), suite)
    if args.json:
        payload = {
            "objective": candidate.objective,
            "summary": candidate.summary,
            "scenarios": [metric.to_dict() for metric in candidate.metrics],
        }
        if args.validate:
            payload["held_out"] = validate(TrackerParams(), scenarios=suite)
        print(json.dumps(payload, indent=2, default=float))
        return EXIT_OK

    print(format_table(list(candidate.metrics)))
    if args.validate:
        print("\nHeld-out profiles:")
        for label, summary in validate(TrackerParams(), scenarios=suite).items():
            print(
                f"  {label:<9} objective {summary['objective']:.4f}  "
                f"IDF1 {summary['idf1']:.1%}  MOTA {summary['mota']:.1%}  "
                f"IDs {int(summary['id_switches'])}"
            )
    return EXIT_OK


def _cmd_bench(args: argparse.Namespace) -> int:
    """Measure detection latency per backend so the choice rests on numbers."""
    import numpy as np

    from vantage.perception.backends import available_backends
    from vantage.perception.benchmark import benchmark, format_table, resolve_targets
    from vantage.perception.catalog import DEFAULT_MODEL

    model = args.model or DEFAULT_MODEL
    targets = resolve_targets(args.backends, available_backends())
    if not targets:
        raise VantageError(
            "no inference backend is installed. Try: pip install onnxruntime openvino"
        )

    if args.image:
        import cv2

        image = cv2.imread(args.image)
        if image is None:
            raise VantageError(f"could not read benchmark image: {args.image}")
        image_label = args.image
    else:
        # A synthetic frame keeps the benchmark runnable anywhere. It measures
        # throughput honestly; it says nothing about accuracy, since generated
        # shapes are not COCO objects.
        from vantage.ingestion.synthetic import SyntheticSource

        with SyntheticSource(width=1280, height=720, frames=1, objects=5) as source:
            image = np.array(source.read().image)
        image_label = "synthetic 1280x720"

    results = benchmark(
        model=model,
        targets=targets,
        image=image,
        iterations=args.frames,
        warmup=args.warmup,
        model_dir=args.model_dir or "models",
    )

    if args.json:
        print(json.dumps({"model": model, "image": image_label, "results": results}, indent=2))
    else:
        print(format_table(model, image_label, results))
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
        ("detection.model", args.model),
        ("detection.backend", args.detect_backend),
        ("detection.device", args.device),
        ("detection.confidence", args.conf),
        ("detection.interval", args.detect_interval),
        ("tracking.min_hits", args.track_min_hits),
        ("tracking.max_lost_s", args.track_max_lost),
        ("pose.model", args.pose_model),
        ("pose.interval", args.pose_interval),
        ("pose.max_persons", args.pose_max_persons),
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
    if args.detect or args.track or args.pose:
        overrides.append("detection.enabled=true")
    if args.pose:
        # Pose is top-down and takes its boxes from tracks, so --pose implies
        # --track for the same reason --track implies --detect.
        overrides.append("pose.enabled=true")
    if args.no_face_keypoints:
        overrides.append("pose.include_face_keypoints=false")
    if args.no_state:
        overrides.append("state.enabled=false")
        # Every activity rule reads motion state, so without it the recogniser
        # could only ever report "idle". Turning it off too is the honest
        # consequence rather than leaving a subsystem running on nothing.
        overrides.append("activity.enabled=false")
    if args.no_activity:
        overrides.append("activity.enabled=false")
    if args.no_spatial:
        overrides.append("spatial.enabled=false")
    if args.track or args.pose:
        # --track implies --detect rather than erroring, because a tracker with
        # no detector cannot do anything at all; requiring both flags would be
        # pedantry with no failure mode to protect against.
        overrides.append("tracking.enabled=true")
    if args.classes:
        # Rendered as a YAML flow sequence so the loader parses it as a real
        # list; a bare string would be iterated character by character.
        names = [name.strip() for name in args.classes.split(",") if name.strip()]
        overrides.append("detection.classes=[" + ", ".join(names) + "]")
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
