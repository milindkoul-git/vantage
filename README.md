# Vantage

A modular platform for understanding what happens in video over time — not just what
appears in a single frame.

**Status: Phase 1 (video ingestion) complete.** No detection, tracking, pose, events,
storage, dashboard, or identity functionality exists yet. Those arrive in later phases,
deliberately and one at a time. Nothing in this repository is mocked or stubbed: what is
here works, and what is not here is absent rather than faked.

---

## 1. Purpose

The long-term goal is a system that reasons about scenes across time — who and what is
present, how they relate, what changed, and what is unusual — from live cameras and
recorded video, on ordinary hardware.

The architecture is organised so that each capability is a replaceable component behind a
stable contract:

```
CameraSource ─▶ Frame ─▶ DetectionEngine ─▶ TrackingEngine ─▶ PoseEngine
                                                  │
                                                  ▼
                        ContextEngine ◀── ActivityEngine ──▶ EventEngine ─▶ Storage ─▶ Analytics
                                                  ▲
                                       IdentityResolver (optional, much later)
```

Phase 1 builds the leftmost two boxes and the contract between them.

### Privacy stance

Entities are anonymous and stay anonymous. Identification is a separate, optional
subsystem that does not exist yet and that nothing else is allowed to depend on — tracking
must work fully without it. See [Identity, later](#8-identity-later).

---

## 2. What Phase 1 delivers

A video ingestion subsystem that is genuinely production-shaped rather than a capture loop:

- **Multiple source types** behind one interface — webcams, media files, RTSP/HTTP
  streams, and a deterministic synthetic generator that needs no hardware at all.
- **One URI string** selects any of them, from config, CLI, or (later) an API call.
- **Decoupled capture** on its own thread behind a bounded queue, with an explicit
  backpressure policy chosen per source type.
- **Rate control** — frame stride, target FPS, and native-timeline playback for files.
- **Automatic reconnection** for live sources that drop out.
- **Measurement** — capture and delivery FPS, acquisition and end-to-end latency
  percentiles, queue depth, dropped and skipped frame counts.
- **A diagnostic viewer** with a telemetry HUD.
- **Graceful shutdown** on Ctrl+C, verified to release the device and join its thread.
- **199 tests**, none of which need a camera.

---

## 3. Quick start

Requires Python 3.11+ (developed on 3.13.1).

```bash
git clone <this repo> && cd vantage

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e ".[dev]"
```

Then, in order of "does it work at all" to "does it work with my hardware":

```bash
vantage info                    # what the platform sees: OS, OpenCV, backends, acceleration
vantage run                     # synthetic video - works on any machine, no camera needed
vantage probe                   # which camera indices actually respond
vantage run --source webcam:0   # your camera, with the telemetry HUD
```

Press `q` or `Esc` to quit, `s` to save a PNG snapshot, `h` to toggle the HUD.

### More things to try

```bash
# Headless throughput check - no window, prints a JSON summary
vantage run --no-display --frames 300 --json

# Generate a test clip, then play it back at its natural speed
vantage make-sample --out samples/clip.mp4 --seconds 10
vantage run --source samples/clip.mp4 --realtime

# Simulate a heavy Phase 2 model: throttle to 10 fps and sample every 3rd frame
vantage run --source webcam:0 --target-fps 10 --stride 3

# Request a specific capture mode (MJPG often unlocks higher webcam frame rates)
vantage run --source webcam:0 --width 1280 --height 720 --fourcc MJPG

# Any config key can be overridden directly
vantage run --set ingest.queue_size=16 --set ingest.backpressure=block
```

Run the tests:

```bash
pytest
```

---

## 4. Architecture

### The contract

Everything crosses stage boundaries as a `Frame`:

```python
@dataclass(frozen=True, slots=True)
class Frame:
    image: np.ndarray          # HxWx3 uint8 BGR, read-only
    index: int                 # ordinal in the SOURCE's output sequence
    source_id: str             # stable per camera
    capture_monotonic: float   # for latency; never jumps
    capture_wall: float        # for storage and cross-camera correlation
    media_pts: float | None    # position on the media timeline; None for live
    metadata: dict
```

Three decisions here are load-bearing:

**`index` counts frames produced, not delivered.** When the pipeline drops frames to stay
current, the consumer sees a gap (`41 → 44`) and knows exactly what it missed. A tracker
must reason about elapsed time between observations rather than assume uniform spacing, so
this is not cosmetic.

**Pixels are read-only.** Frames are shared by reference across stages for performance. A
stage that quietly annotated a buffer another stage still held would produce bugs that are
very hard to find. Stages that need to draw call `editable_copy()`.

**Both clocks are carried.** Monotonic for measuring, wall for recording. Phase 8 storage
needs the second; Phase 12 optimisation needs the first.

### Layers

```
vantage/
├─ core/          primitives; depends on nothing else in the platform
│  ├─ frame.py       the Frame contract
│  ├─ clock.py       Clock protocol + ManualClock (makes timing testable)
│  ├─ metrics.py     counters, rate meters, latency percentiles
│  ├─ logging.py     structured logging, console or JSON
│  ├─ lifecycle.py   signal handling and cooperative shutdown
│  └─ errors.py      the exception hierarchy
│
├─ config/        typed schema (dataclasses) + strict YAML/CLI loader
│
├─ ingestion/     everything about getting frames
│  ├─ base.py         FrameSource ABC, SourceInfo, lifecycle state machine
│  ├─ registry.py     URI parsing and source construction
│  ├─ opencv_source.py cameras, files, streams via cv2.VideoCapture
│  ├─ synthetic.py    deterministic generated video
│  ├─ resilient.py    reconnection wrapper for live sources
│  ├─ buffer.py       bounded queue and backpressure policies
│  ├─ pacing.py       stride filter and rate pacers
│  └─ pipeline.py     ties it together; yields frames, measures everything
│
├─ viz/           diagnostic display only; contains no analysis logic
├─ app.py         composition root - the only module that knows about all layers
└─ cli.py         command-line entry point
```

Dependencies point inward. `core` imports nothing from the platform; `ingestion` imports
`core`; `viz` renders what `ingestion` measured; `app` wires them together.

### The consumer contract

This is the whole interface a Phase 2 detector needs:

```python
with IngestionPipeline(source, config.ingest) as pipeline:
    for frame in pipeline.frames():
        detections = detector.run(frame.image)     # Phase 2 goes here
        stats = pipeline.stats()                   # and telemetry is already there
```

A consumer cannot discover whether acquisition ran on this thread or another, whether
frames were dropped, or what kind of device produced them. That is the point.

---

## 5. Key design decisions

### Backpressure is a per-source-type choice, not a global one

This machine has no CUDA GPU. Phase 2 inference *will* be slower than 30 fps capture, so
what happens at that moment is an architectural decision made now rather than an accident
discovered later.

| Policy | Behaviour | Correct for |
|---|---|---|
| `latest` | Evict the oldest queued frame | **Live cameras.** Analysing a two-second-old frame reports the past as the present. Latency stays bounded however slow the consumer gets. |
| `block` | Stall the producer until there is room | **Files.** No real-time obligation, every frame matters, and reproducible results demand it. |
| `drop_new` | Reject the arriving frame | Rare — pre-event ring buffers where the oldest frames are the reference. |

`auto` (the default) resolves to `latest` for live sources and `block` for recorded ones.
Every dropped frame is counted and surfaced; silent loss would make the FPS numbers a lie.

### The synthetic source is infrastructure, not a toy

Seeded, procedurally animated video with the frame number and timestamp burned into the
pixels and a sweep bar that advances exactly one step per frame — so a stale or duplicated
frame is visible at a glance rather than subtly wrong. It exists because:

1. The entire pipeline is testable in CI on a machine with no camera, at full speed.
2. Object motion is a closed-form function of the frame index, so `object_states(index)`
   returns exact boxes for any frame — **free ground truth** for evaluating trackers in
   Phase 3 and beyond.

### Threading, not multiprocessing

Capture blocks inside C code that releases the GIL, so one thread is sufficient and avoids
pickling ~900 KB arrays per frame. When inference eventually saturates the CPU, the queue
boundary in `buffer.py` is exactly where a process boundary would be substituted — no
consumer code would change.

### Rate meters smooth intervals, not rates

Averaging instantaneous rates over-weights short intervals. Bursty USB delivery (a run of
10 ms gaps, then a 60 ms stall) was reported as ~100 fps against a true 27 fps until this
was corrected. `RateMeter` now smooths the inter-arrival interval and inverts it.

### Strict configuration

An unknown key is an error with a suggested correction, never a silently ignored setting.
A typo'd `targt_fps` that quietly does nothing costs an afternoon.

---

## 6. Configuration

Resolution order — later layers win: **built-in defaults → `configs/default.yaml` (or
`$VANTAGE_CONFIG`, or `--config`) → `--set key=value` → typed CLI flags.**

The convenience flags lower onto the same override mechanism, so a flag and a config key
can never diverge in behaviour.

```yaml
app:
  log_level: INFO          # DEBUG | INFO | WARNING | ERROR | CRITICAL
  log_format: console      # console (human) | json (aggregation)
  stats_interval_s: 5.0    # periodic telemetry summary; 0 disables

source:
  uri: "synthetic://?width=1280&height=720&fps=30&objects=5"
  id: null                 # defaults to a stable id derived from the URI
  backend: auto            # auto | msmf | dshow | ffmpeg | gstreamer | v4l2 | any
  width: null              # requested geometry; drivers may refuse (a warning is logged)
  height: null
  fps: null
  fourcc: null             # e.g. MJPG - often unlocks higher USB webcam frame rates
  loop: false              # restart file sources at EOF
  read_retries: 3          # tolerated consecutive empty reads before failing
  reconnect:               # live sources only
    enabled: true
    max_attempts: 5
    initial_delay_s: 0.5
    max_delay_s: 10.0
    backoff: 2.0

ingest:
  mode: threaded           # threaded (decoupled) | inline (deterministic, no thread)
  queue_size: 8            # small on purpose: a deep queue hides latency problems
  backpressure: auto       # auto | latest | block | drop_new
  target_fps: null         # throttle delivery - the cheapest CPU-fit lever for Phase 2
  stride: 1                # deliver every Nth frame (deterministic sampling)
  realtime: false          # pace recorded sources to their own timeline
  max_frames: null         # stop after N delivered frames

display:
  enabled: true
  window_name: "Vantage - Ingestion"
  hud: true
  scale: 1.0               # display only; never changes frames given to consumers
  snapshot_dir: snapshots
```

### Source URIs

| Form | Meaning |
|---|---|
| `webcam:0` | Capture device by index (also `camera:0`, or a bare `0`) |
| `file:clips/lobby.mp4` | Media file via FFmpeg (a bare existing path works too) |
| `synthetic://?width=&height=&fps=&frames=&seed=&objects=` | Generated video, no hardware |
| `rtsp://host/stream1` | Network stream (also `http`, `rtmp`, `udp`, `tcp`, `srt`, `rtp`) |

---

## 7. Verified behaviour

Measured on the development machine (Windows 11, i5-13500H, 16 GB RAM, Intel Iris Xe, no
CUDA, Python 3.13.1, OpenCV 5.0.0):

| Check | Result |
|---|---|
| Test suite | 199 passed in ~1.5 s, no camera required |
| Synthetic source, headless | ~2 400 fps at 640×480 |
| Webcam `webcam:0`, headless | 30.3 fps mean over 60 frames, 0 dropped, queue peak 1/8, acquisition p50 6.1 ms, delivery latency p95 0.3 ms |
| File playback (960×540, 100 frames) | 100/100 delivered, 0 dropped, `block` policy auto-selected, ~846 fps decode |
| Backpressure auto-selection | `latest` for the camera, `block` for the file — as designed |
| Graceful shutdown | SIGINT at 2.0 s → process exited at 2.06 s, source closed, zero leaked threads |
| Display | HUD renders correctly over both synthetic and camera frames; highgui window opens and tears down cleanly |

Backend measurement that set the Windows default: on this webcam, **MSMF sustained ~30 fps
and reported usable FPS metadata; DirectShow managed ~15 fps and reported none.**

---

## 8. Known limitations

**Phase 1 scope.** No detection, tracking, pose, activity recognition, events, alerts,
storage, dashboard, multi-camera orchestration, or identity. By design.

**One source per pipeline.** `IngestionPipeline` handles a single source. Multi-camera
support needs a manager that owns several pipelines and merges their telemetry; `Frame`
already carries `source_id` so nothing has to change to allow it.

**RTSP is implemented but untested here.** The code path exists and is the standard
FFmpeg one, but there was no stream available on this machine to verify against. Treat it
as unproven until you run it. PyAV would give better timestamp fidelity for network
streams and is the natural second `FrameSource` implementation.

**Camera shutdown can lag by up to 5 seconds.** If a driver blocks inside `read()`,
`close()` waits up to 5 s before giving up on the capture thread. The thread is a daemon
so the process still exits; the device handle is released either way.

**Blank frames are a warning, not an error.** A closed privacy shutter is indistinguishable
from a healthy camera by every property OpenCV reports — correct resolution, correct frame
rate, successful reads, all-zero pixels. Vantage logs a warning when the probe frame is
entirely blank, because a legitimately dark scene looks identical and refusing to start
would be worse. *This was hit for real during Phase 1 validation on this machine.*

**Driver negotiation is advisory.** Requested width/height/FPS/FOURCC are requests. What
was actually granted is reported in `SourceInfo` and a mismatch is logged. Always trust
`SourceInfo` over what you asked for.

**OpenCV 5.0.0 is very new.** It works well here, but parts of the CV ecosystem still pin
`opencv-python<5`. If a Phase 2 dependency conflicts, pin to the 4.12 line — the code uses
no APIs that differ between them.

**`configs/default.yaml` is found relative to the source checkout.** Installed as a wheel
outside a checkout, the bundled default is not located and built-in defaults apply. Pass
`--config` or set `$VANTAGE_CONFIG` in that situation.

---

## 9. Roadmap

Phase 1 is done. The order below reflects where the evidence points, not a fixed plan.

**Phase 2 — Object detection (next).** Given no CUDA and an Intel Iris Xe iGPU, the
expected choice is **ONNX Runtime with the OpenVINO execution provider**, benchmarked
against plain CPU ONNX Runtime, with a small modern detector (RT-DETR / YOLO-family, or
`yolov8n`-class) exported to ONNX and INT8-quantised. PyTorch is deliberately *not*
assumed: the CPU-only build installed on this machine would very likely lose to both.
The deliverable is a `DetectionEngine` producing a `Detection` record — boxes, class,
confidence, and the `Frame` it came from — and a benchmark table so the choice is made on
numbers rather than reputation.

**Phase 3 — Multi-object tracking.** ByteTrack first (no appearance model, no GPU cost,
strong on the accuracy/compute trade-off), with the synthetic source providing exact ground
truth for identity-switch measurement. Produces the stable anonymous `entity_id`
(`person_17`) the rest of the platform builds on.

**Phase 4 onward — Pose and object state → temporal activity → spatial/interaction
analysis → event engine → observation storage → dashboard → identity (optional) →
advanced analytics → multi-camera scaling.**

### Identity, later

The identity layer is optional, separate, and not yet designed. Two constraints hold now
and will keep holding:

1. **Tracking never depends on identity.** Anonymous tracking must work fully on its own.
2. **The seam already exists.** Identity resolution consumes a stable anonymous
   `entity_id` and returns a resolved identity or "unknown" — attachable without touching
   the tracking subsystem.

No facial recognition, biometric enrolment, or identity storage exists in this repository,
and none will be added except deliberately, in its own phase, with the access control and
audit logging such a subsystem requires.

### Ideas worth pursuing, collected while building Phase 1

- **Adaptive frame sampling.** `RatePacer.set_target()` already exists as the control
  point; drive it from measured inference latency so the system degrades by analysing
  fewer frames well rather than by falling progressively further behind.
- **Ground-truth-driven evaluation.** The synthetic source can generate deliberate
  occlusions and crossings, giving a repeatable tracker benchmark that needs no annotated
  dataset.
- **Frame gaps as a first-class signal.** Because `index` records what was dropped,
  downstream stages can weight their confidence by observation density instead of assuming
  a steady frame rate.
- **Metrics as an event source.** `PipelineStats` is already a flat, serialisable record;
  it can feed the same storage the observations do, making "the camera degraded at 14:03"
  a queryable event rather than a log line.

---

## 10. Development

```bash
pytest                          # 199 tests, ~1.5 s, no hardware needed
pytest -m hardware              # (reserved) tests that need a physical camera
vantage info --json             # environment report for a bug reference
vantage run --log-format json   # structured logs for aggregation
```

Conventions: type hints throughout, dependencies pointing inward, no silent exception
handling, no hard-coded paths, and no component that cannot be replaced without touching
its neighbours. Comments explain *why*, never *what*.

## License

MIT.
