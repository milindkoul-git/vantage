# Vantage

A modular platform for understanding what happens in video over time — not just what
appears in a single frame.

**Status: Phases 1-2 complete — video ingestion and object detection.** No tracking, pose,
events, storage, dashboard, or identity functionality exists yet. Those arrive in later
phases, deliberately and one at a time. Nothing in this repository is mocked or stubbed:
what is here works, and what is not here is absent rather than faked.

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

Phases 1-2 build the leftmost three boxes and the contracts between them.

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

## 2b. What Phase 2 delivers

Object detection, behind an interface that makes both the model and the runtime
replaceable:

- **Two interchangeable inference backends** — ONNX Runtime (portable CPU baseline) and
  OpenVINO (Intel CPU **and** iGPU) — reading the same ONNX file and verified to produce
  identical detections on CPU.
- **YOLOX detectors** (Apache-2.0) in three sizes, fetched on demand and **verified
  against pinned SHA-256 checksums**; weights are never committed.
- **Detections in original-frame pixel coordinates**, so nothing downstream ever learns
  that the detector letterboxed anything.
- **Class-aware NMS**, so a person standing in front of a car cannot suppress the car.
- **Class filtering** (`--classes person,car`) and **frame-interval inference**
  (`--detect-interval N`) — the two levers that make a detector fit CPU-only hardware.
- **A real benchmark command** (`vantage bench`) that measures this machine rather than
  quoting someone else's numbers.
- **Box overlay + detection telemetry on the HUD**, with carried-forward detections drawn
  dashed so a stale box is never mistaken for a fresh one.

**284 tests** — 276 needing neither a camera, a model file, nor an inference runtime, plus
8 that exercise real weights and skip cleanly without them.

---

## 2c. What Phase 3 delivers

Multi-object tracking: the point at which per-frame detections become persistent objects
with identity, which is the prerequisite for every later phase.

- **ByteTrack**, two-pass association on geometry alone. No appearance model, so it adds
  **well under 1 ms per frame** on top of detection (0.7 ms with 8 simultaneous objects)
  and, by construction, computes nothing biometric.
- **Anonymous stable identity** (`person_17`) assigned by the tracker and fixed for a
  track's lifetime, with the class settled by majority vote so a first-frame flicker
  cannot permanently mislabel an entity.
- **A real motion model** — a Kalman filter over `(cx, cy, w, h)` driven by **actual
  elapsed time** rather than a frame count, so `detection.interval` and dropped frames do
  not silently degrade it. Position is extrapolated; **size deliberately is not**, which
  fixes a failure mode the reference design has (see below).
- **An optimal assignment solver** (Jonker–Volgenant, ~100 lines, no SciPy) verified
  against brute-force enumeration, because greedy matching fails exactly when two objects
  pass each other — the case tracking exists for.
- **A ground-truth evaluation harness**: five seeded scenarios, a modelled detector, and
  full CLEAR MOT + IDF1 metrics, so tracking accuracy is a reproducible number rather than
  an impression. One command: `vantage track eval`.
- **Measured parameters, not inherited ones.** `vantage track tune` searches the parameter
  space and reports held-out results; the shipped defaults are its output.
- **Track overlay with motion trails and per-track colouring**, plus tracking telemetry on
  the HUD, with coasting (predicted) boxes drawn dashed so a guess is never shown as an
  observation.

**388 tests** (+104), 380 of which need no camera, no model file and no inference
runtime; the remaining 8 exercise real weights and deselect cleanly without them.

---

## 3. Quick start

Requires Python 3.11+ (developed on 3.13.1).

```bash
git clone <this repo> && cd vantage

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e ".[dev]"           # ingestion only
pip install -e ".[dev,detect]"    # + object detection (onnxruntime + openvino)
```

Then, in order of "does it work at all" to "does it work with my hardware":

```bash
vantage info                    # what the platform sees: OS, OpenCV, backends, acceleration
vantage run                     # synthetic video - works on any machine, no camera needed
vantage probe                   # which camera indices actually respond
vantage run --source webcam:0   # your camera, with the telemetry HUD
```

### Detection (Phase 2)

```bash
vantage models list             # available detectors: size, accuracy, licence, cached?
vantage models pull yolox-nano  # ~3.7 MB, checksum-verified
vantage bench                   # measure the backends on YOUR machine
vantage run --source webcam:0 --detect
```

Press `q` or `Esc` to quit, `s` to save a PNG snapshot, `h` to toggle the HUD.

### Tracking (Phase 3)

```bash
vantage run --source webcam:0 --track --device gpu   # --track implies --detect
```

Each object gets a stable anonymous label (`person_17`) and a motion trail. A **dashed**
box means the position is predicted rather than observed — the object is currently hidden.

Verify the tracker rather than taking its word for it:

```bash
vantage track scenarios     # the benchmark scenarios and what each one stresses
vantage track eval          # score the shipped parameters, per scenario
vantage track eval --validate --scenarios occlusion,crowd
vantage track tune          # search for better parameters; reports held-out results
```

`track eval` needs no camera, no model and no inference runtime — the detector is modelled,
so it measures the tracker rather than the pair of them.

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

# Detect only people, on the Intel iGPU, inferring on every 3rd frame
vantage run --source webcam:0 --detect --device gpu --classes person --detect-interval 3

# Compare backends on your own footage
vantage bench --image my_photo.jpg --frames 100

# Track only people, and publish a track faster (1 = as soon as it is seen)
vantage run --source webcam:0 --track --classes person --track-min-hits 2

# Hold identity through a longer occlusion (seconds, not frames)
vantage run --source webcam:0 --track --track-max-lost 3.0
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
├─ perception/    Phase 2: turning pixels into structured observations
│  ├─ contracts.py   BoundingBox, Detection, DetectionResult
│  ├─ engine.py      composes an adapter with a backend
│  ├─ adapters/      model families: input shaping + output decoding (YOLOX)
│  ├─ backends/      runtimes: ONNX Runtime, OpenVINO
│  ├─ catalog.py     model registry with URLs, checksums and licences
│  ├─ store.py       download, verify, cache
│  ├─ nms.py         class-aware non-maximum suppression
│  └─ benchmark.py   backend measurement
│
├─ tracking/      Phase 3: turning detections into persistent objects
│  ├─ contracts.py   Track, TrackingResult, TrackState
│  ├─ bytetrack.py   the tracker: two-pass association and lifecycle
│  ├─ kalman.py      constant-velocity filter with a real, variable timestep
│  ├─ assignment.py  optimal assignment (Jonker-Volgenant) + IoU matrices
│  ├─ base.py        the Tracker Protocol - the replaceability seam
│  ├─ scenarios.py   seeded ground truth + a modelled detector
│  ├─ evaluation.py  CLEAR MOT and IDF1
│  ├─ tuning.py      parameter search with held-out validation
│  └─ factory.py     config -> tracker (keeps config out of the algorithm)
│
├─ viz/           diagnostic display only; contains no analysis logic
├─ app.py         composition root - the only module that knows about all layers
└─ cli.py         command-line entry point
```

### Why detection splits into adapter + backend

Three things change for different reasons, so they are three types:

| Concern | Type | Changing it means |
|---|---|---|
| How a model family shapes input and decodes output | `ModelAdapter` | Adding RT-DETR touches one file |
| How a runtime executes a graph | `InferenceBackend` | Swapping ONNX Runtime for OpenVINO changes no detections |
| Turning that into records | `DetectionEngine` | The only type the rest of the platform sees |

This is verified rather than asserted: `test_cpu_backends_produce_the_same_detections`
runs both runtimes over the same image and requires identical output.

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

### The detector was chosen on licence first, accuracy second

The obvious default — Ultralytics YOLOv8/v11 — is **AGPL-3.0**. Building a product on it
obliges you to open-source that product or buy a commercial licence. That constraint
propagates from a single `pip install` into the whole platform, so it is decided here
rather than discovered in a legal review.

**YOLOX is Apache-2.0**, ships official pre-exported ONNX weights, and happens to expect
**BGR 0-255 input** — exactly what `Frame` already carries, so there is no colour
conversion on the hot path. Every catalog entry records its licence next to its URL.

### Tracking measures time in seconds, and its own elapsed time at that

Nearly every published tracker advances its motion model by one unit per call and calls
that a frame. That assumption is false here twice over: `detection.interval` runs the
detector on every Nth frame by configuration, and `latest` backpressure drops frames under
load. A tracker that assumed uniform spacing would under-predict motion after a gap by
exactly the factor it was wrong about — and being wrong about where an object went *is* an
identity switch.

So the Kalman filter takes `dt` as an argument, the process noise is the exact
discretisation of a constant-velocity model (`dt³/3`, `dt²/2`, `dt`) rather than a table
tuned at one frame rate, and track expiry is configured in **seconds** rather than frames.
`Frame` already carried both a monotonic and a media timestamp, so the information was
there; media time wins where it exists, because a 60-second clip analysed in 6 seconds
must still model objects as moving at their real speed.

Verified: a filter fed every third observation with three times the timestep converges on
the same velocity as one fed every frame.

### Size is filtered but never extrapolated

The reference SORT/ByteTrack filters give box size a velocity, and copying that produced a
real failure found during runtime validation. As an object slides behind an occluder the
detector still sees it, but only the visible part, so the reported box shrinks fast. The
filter reads that as genuine sustained shrinking. The object then vanishes completely, the
filter extrapolates, and the predicted box collapses — **measured at 1x1 pixels after 40
frames on real footage.**

That is not just an invisible box on screen. A degenerate box has an IoU of essentially
zero against anything, so when the object reappeared it could never be re-associated and a
new identity was issued. The bug was silently *causing* the identity losses it was hiding.

The fix is to drop size velocity entirely: the state is `(cx, cy, w, h, vx, vy)`, position
is advanced by its velocity, and size carries forward unchanged while being corrected by
measurements and allowed to drift as a random walk. The physical argument was always on
this side — an object's apparent size changes slowly, non-monotonically, and only because
its distance changed, whereas its position changes consistently and predictably.
Extrapolating size across a gap of no evidence asserts something nobody knows.

Worth 1 point of pooled MOTA, 5 identity switches and 9 points of occlusion IDF1 on the
benchmark — and it was invisible to every metric until someone looked at a rendered frame.

### The tracker cannot see pixels, and that is a guarantee

`Tracker.update()` takes detections and time. It has no access to the image. This is a
deliberate structural choice with two payoffs. It makes tracking testable and *tunable*
with no camera, no model and no runtime — which is what made the evaluation harness
possible at all. And it means appearance-based re-identification cannot be added by
accident: extracting visual features would require changing the interface, which is a
review-visible act rather than a quiet import. For a platform with this project's privacy
constraints, a tracker that structurally cannot compute an appearance signature is a
stronger guarantee than one that merely does not today.

### Model weights are treated as untrusted remote content

Pinned SHA-256, verified on every load rather than trusted because the file exists;
downloads written to a temp file and renamed only after the hash matches, so an
interrupted download can never masquerade as a cached model; a mismatch is a loud error
naming both digests, never a silent re-download that papers over a substituted upstream.

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

tracking:
  enabled: false           # requires detection.enabled
  detection_floor: 0.1     # detector threshold while tracking - see the note below
  high_threshold: 0.5      # above this a box may start a track
  low_threshold: 0.1       # between low and high a box may only sustain a track
  init_threshold: 0.5      # confidence needed to create a track (>= high_threshold)
  iou_high: 0.2            # minimum overlap, first association pass
  iou_low: 0.4             # minimum overlap, low-confidence second pass
  iou_tentative: 0.4       # minimum overlap for an unconfirmed track
  min_hits: 3              # frames of corroboration before a track is published
  max_lost_s: 1.5          # seconds a track survives unmatched - SECONDS, not frames
  max_step_s: 2.0          # elapsed times beyond this are clamped, not extrapolated
  history: 30              # centre positions kept per track, for motion trails
  class_aware: true        # never associate a detection across a class boundary
  measurement_noise: 0.05      # detector box error, as a fraction of object height
  acceleration_noise: 2.0      # unmodelled acceleration, object heights per second^2
  initial_velocity_noise: 1.0  # velocity prior for a new track

display:
  enabled: true
  window_name: "Vantage - Ingestion"
  hud: true
  scale: 1.0               # display only; never changes frames given to consumers
  snapshot_dir: snapshots
```

**`tracking.detection_floor` is the one setting whose purpose is not obvious**, and it is
load bearing. ByteTrack keeps identity through occlusion by matching the *low*-confidence
boxes an occluded object produces. If the detector keeps filtering at
`detection.confidence` (0.35), those boxes never reach the tracker and the algorithm
silently degrades to ordinary IoU tracking — still working, but without the one property it
was chosen for. So enabling tracking lowers the detector's floor to this value and lets the
tracker do the filtering instead. The run logs the change when it happens, because
honouring a different number than the one you configured should never be silent. The trade
is real: the detector now emits considerably more junk, and `min_hits` is what stops that
junk becoming published tracks. Set it equal to `detection.confidence` to opt out.

Every numeric value in the `tracking:` block is the output of `vantage track tune`, not a
copy of the reference paper's. Re-run it if you change detector or camera.

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

### Detection benchmark (`vantage bench`, yolox-nano @ 416x416, 60 iterations)

| Backend | Device | Precision | Mean | p50 | p95 | FPS ceiling |
|---|---|---|---|---|---|---|
| onnxruntime | cpu | fp32 | 19.94 ms | 13.55 ms | 54.09 ms | 50.1 |
| openvino | cpu | fp32 | 47.24 ms | 50.84 ms | 75.55 ms | 21.2 |
| **openvino** | **gpu (Iris Xe)** | **fp16** | **13.68 ms** | **13.43 ms** | **17.07 ms** | **73.1** |

The headline is the **p95 column, not the mean**. The iGPU is ~1.5x faster on average but
**3x more consistent**, because CPU inference competes with everything else on the machine
while the iGPU sits idle otherwise. For a realtime pipeline the tail is what a viewer
actually perceives as stutter. The CPU numbers move noticeably between runs; the GPU
numbers do not.

Two caveats stated honestly: OpenVINO runs **fp16** on Intel GPUs, so GPU detections differ
very slightly from CPU ones (a marginal 0.31-confidence box appears on CPU and not GPU) —
this is reported in the `PREC` column rather than hidden. And ONNX Runtime beating OpenVINO
*on CPU* here is the opposite of the folklore, which is exactly why the benchmark exists.

| Check | Result |
|---|---|
| Test suite | 380 passed in ~3.0 s with no camera, model or runtime; +8 model-backed |
| Synthetic source, headless | ~2 400 fps at 640×480 |
| Webcam `webcam:0`, headless | 30.3 fps mean over 60 frames, 0 dropped, queue peak 1/8, acquisition p50 6.1 ms, delivery latency p95 0.3 ms |
| File playback (960×540, 100 frames) | 100/100 delivered, 0 dropped, `block` policy auto-selected, ~846 fps decode |
| Backpressure auto-selection | `latest` for the camera, `block` for the file — as designed |
| Graceful shutdown | SIGINT at 2.0 s → process exited at 2.06 s, source closed, zero leaked threads |
| Display | HUD renders correctly over both synthetic and camera frames; highgui window opens and tears down cleanly |

Backend measurement that set the Windows default: on this webcam, **MSMF sustained ~30 fps
and reported usable FPS metadata; DirectShow managed ~15 fps and reported none.**

End-to-end Phase 2: file source → pipeline → detection → overlay → HUD, 30 frames with
`interval=2`, produced 15 detection passes at **8.3 ms mean on the iGPU** (120 fps ceiling),
correctly labelling dog / bicycle / car and drawing carried-forward boxes dashed.

### Tracking accuracy (`vantage track eval`)

Measured against the seeded ground-truth scenarios, each scored across five modelled
detector profiles (clean, typical, degraded, cluttered, harsh). Pooled from raw counts, so
a long scenario is not outvoted by a short one:

| | Shipped defaults | Reference-paper values |
|---|---|---|
| MOTA | **87.1%** | 84.3% |
| IDF1 | **91.1%** | 84.8% |
| Identity switches | **20** | 30 |
| Mostly-tracked objects | **74 / 85** | 66 / 85 |

Per scenario, and on the **held-out** detector profiles — different seeds *and* different
error magnitudes, never used by the search — the tuned values win on every scenario:

| IDF1 | sparse | crossing | occlusion | crowd | erratic | pooled | ID switches |
|---|---|---|---|---|---|---|---|
| **Shipped** | **97%** | **83%** | **84%** | **80%** | **63%** | **78.6%** | **32** |
| Reference | 90% | 79% | 79% | 76% | 54% | 73.1% | 38 |

Both columns run on the same corrected motion model, so this is a comparison of
*parameters*, not of implementations. The `erratic` scenario is the weakest for both — under
the harshest held-out profile (4.5 spurious boxes per frame, 9% localisation error, a
detector unsure of everything) sharp non-linear motion is genuinely hard, and it is left
visible rather than dropped from the table.

Cost: **0.7 ms per step with 8 simultaneous objects**, against a 33 ms frame budget.

### Occlusion tolerance, measured

The capability the phase exists for, with exact ground truth (object crossing at 200 px/s):

| Occlusion type | Identity survives |
|---|---|
| **Total** (detector returns nothing at all) | up to **1.5 s** — kept at 1.50 s, lost at 1.67 s, exactly `max_lost_s` |
| **Partial** (confidence falls to 0.25) | **indefinitely** — verified to 8 s |

The second row is ByteTrack's whole point. A partially visible object still produces a
low-scoring box, the second association pass consumes it, and identity is never at risk.
Only total disappearance is time-limited, and that limit is a configured number rather than
an emergent one.

### Runtime validation

| Check | Result |
|---|---|
| Full pipeline with tracking, real model | file → detection → tracking → overlay → HUD, 120 frames, **8.1 ms detect + 0.15 ms track**, 0 dropped |
| Live webcam, detection | 200 frames at **29.97 fps mean, 0 dropped**, 9.3 ms/frame on the iGPU |
| Tracking cost | 0.15 ms/step on real footage; 0.7 ms/step with 8 simultaneous objects |
| Graceful shutdown with tracking active | SIGINT → exited in **0.08 s**, code 0, tracker state reported in the run summary |
| Startup / shutdown | detector and tracker built before the camera opens; both closed on every exit path |
| Overlay | verified by rendering: coasting boxes dashed, entity ids and motion trails drawn, HUD reports `1 shown (0 seen, 1 predicted)` |

---

## 8. Known limitations

**Scope.** No pose, activity recognition, events, alerts, storage, dashboard, multi-camera
orchestration, or identity. By design.

**Tracking is motion-only, and long total occlusions break identity.** With no appearance
model there is nothing to re-identify an object by, so recovery depends entirely on the
predicted box still overlapping the object when it reappears. Measured: identity survives
**1.5 s** of total disappearance and then a new entity is issued. Partial occlusion is
unaffected and survives indefinitely, which is the case that actually dominates in
practice. Raising `tracking.max_lost_s` extends the window but also raises the chance of
reviving a track onto the wrong object, and the extrapolated position degrades anyway —
one measured failure came from a detector box shrinking as an object entered an occluder,
which inflated the velocity estimate before the object vanished. Fixing that class of
failure properly needs appearance features, which this phase deliberately does not have.

**Velocity damping while coasting was tried and rejected.** The hypothesis was that a
prediction should decay toward stationary as evidence ages. It measured neutral-to-negative
on the benchmark and did not recover the real-footage case it was written for, so it was
removed rather than shipped as a knob that does nothing. The actual cause of that failure
turned out to be size extrapolation, described next.

**OpenVINO compiled-model caching is deliberately disabled.** Enabling `CACHE_DIR` looks
like free startup time and costs a crash: on this Iris Xe / OpenVINO 2026.3 combination,
*loading* a cached GPU blob segfaults the process during interpreter shutdown — after
inference has completed and returned correct results, so it presents as a mysterious exit
code 139 rather than an obvious failure. Bisected to `CACHE_DIR` specifically; writing the
cache is clean, reading it back is not. The optimisation was worth ~0.9 s of one-off
startup (1058 ms → 182 ms), which is not a trade worth a crash. **Do not re-enable it
without re-testing that exact path** — the comment in `openvino_backend.py` says so too.

**Detection runs on the consumer thread.** Inference is synchronous inside the frame loop,
so a slow model directly reduces delivered FPS on live sources (the queue then drops frames
to keep latency bounded, exactly as designed). Moving inference to its own stage with its
own queue is a Phase 12 concern; `detection.interval` is the lever until then.

**GPU results differ slightly from CPU.** Intel GPUs execute fp16 by default. Marginal
low-confidence detections can appear on one and not the other. Reported in the `PREC`
column rather than papered over.

**Only YOLOX is implemented.** The adapter seam exists for RT-DETR and D-FINE, but adding
them is future work, not something already half-built.

**`yolox-s` is catalogued but not realtime here.** 640x640 input on this hardware is an
accuracy option for offline analysis, not for live camera work.

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
rate, successful reads, near-black pixels. Vantage logs a warning when the probe frame
carries no picture, because a legitimately dark scene looks identical and refusing to start
would be worse. *This was hit for real during Phase 1 validation on this machine, and the
check was found to be wrong during Phase 3 validation:* it tested for **exactly** zero
pixels, but a real sensor behind a closed shutter returns two or three counts of thermal
noise. A shuttered camera streamed 200 frames at a maximum pixel value of 3, the detector
correctly found nothing, and no warning explained why. The threshold is now
`BLANK_LEVEL = 8`.

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

**Phase 2 — Object detection. Done.** One correction worth recording: Phase 1 predicted
"ONNX Runtime with the OpenVINO execution provider". That turned out to be the wrong
packaging — the stock `onnxruntime` wheel ships no OpenVINO EP, and `onnxruntime-openvino`
*replaces* it (same module name), which is a trap to hand a collaborator. Native OpenVINO
reads ONNX directly and reaches the iGPU, so the two runtimes now sit side by side as
independent backends instead. INT8 quantisation was also deferred: fp16 on the iGPU already
clears 73 fps on a 30 fps pipeline, so the accuracy cost buys nothing yet.

**Phase 3 — Multi-object tracking. Done.** ByteTrack, as planned. The open design question
flagged here in advance — irregular timesteps from `detection.interval` — turned out to be
the single most consequential detail: it drove a variable-`dt` Kalman filter, a derived
rather than tabulated process noise, and track expiry measured in seconds. Two things were
not anticipated. First, enabling tracking has to *lower the detector's confidence floor*
(0.35 → 0.1), because ByteTrack's second pass is worthless without the low-scoring boxes an
occluded object produces; that in turn makes false-positive rejection (`min_hits`) load
bearing rather than incidental. Second, building the evaluation harness was worth more than
the tracker: it caught an off-by-one in `min_hits`, an off-frame coasting bug worth 5 points
of MOTA, and two separate cases of the parameter search overfitting a benchmark that was too
easy. INT8 quantisation is still deferred and still not needed.

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
pytest                          # 284 tests, ~2 s
pytest -m "not model"           # 276 tests needing no weights and no runtime
pytest -m model                 # 8 tests against real weights (skip if absent)
pytest -m hardware              # (reserved) tests that need a physical camera
vantage info --json             # environment report for a bug reference
vantage run --log-format json   # structured logs for aggregation
vantage bench --json            # backend measurements as machine-readable output
```

Conventions: type hints throughout, dependencies pointing inward, no silent exception
handling, no hard-coded paths, and no component that cannot be replaced without touching
its neighbours. Comments explain *why*, never *what*.

## License

MIT.
