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

## 2d. What Phase 3.5 delivers

An unplanned insertion, and the number says so. This improves the detection layer rather
than adding a subsystem, so it is 3.5 and not 4 - taking the number 4 would have implied
that pose estimation exists. It was prompted by a real failure report: the platform could
not see a pen, and no confidence threshold was ever going to fix that. COCO has no `pen`
output neuron, so YOLOX cannot answer the question at all.

Two tiers, because the measurements say they are different jobs:

- **A bigger fixed vocabulary.** **D-FINE detectors** (Apache-2.0) trained on Objects365 -
  **365 classes instead of COCO's 80**, including the desk and office objects COCO omits
  entirely (`Pen/Pencil`, `Marker`, `Stapler`, `Folder`, `Calculator`, `Notepaper`,
  `Tape`). A second adapter family behind the existing seam: the engine, both backends,
  the tracker, the overlay and the pipeline are untouched.
- **Open vocabulary, on demand.** `vantage discover --prompts "..."` runs Grounding DINO
  (Apache-2.0) against arbitrary words. Deliberately a separate command rather than a
  switch on the live pipeline: it costs ~12 s per prompt against the live detector's
  18 ms, and a ~700x gap is not a tuning gap.

Three things measurement changed about the design:

- **Prompts must be queried one at a time.** Batched, this export suppresses all but the
  strongest phrase - `dog, bicycle, car` scored 0.90 / 0.09 / 0.08 together and
  0.92 / 0.89 / 0.59 separately. That is wrong rather than merely weaker, so cost is
  linear in prompt count and the answers are right.
- **CPU beats the iGPU by 7x for discovery** - the opposite of the live pipeline's choice.
  OpenVINO spends ~155 s compiling the graph to save ~9 s of inference on a one-shot pass.
- **"DETR needs no NMS" was written here as fact, then measured to be false.** One person
  produced six boxes, two pairs overlapping at IoU 0.90 and 0.84. Unsuppressed, Phase 3
  confirms each as a separate track and invents people.

Two latent bugs surfaced with it: OpenVINO was compiling **dynamic**-shape graphs unpinned
(184 ms -> **84 ms** once reshaped before compile), which had never shown up because every
YOLOX export is static; and the backend interface could not carry a multi-input graph at
all.

**449 tests** (+61), 441 of which need no camera, no model file and no inference runtime.

---

## 2e. What Phase 4 delivers

Human pose and object state - the two halves of the spec's Phase 4, and both
turn a per-frame observation into something with duration, which is what the
activity recognition and event rules of later phases need.

**Pose: RTMPose, top-down.**

- **17 body landmarks per person** (Apache-2.0, from the official OpenMMLab
  distribution), attached to the tracker's anonymous ``entity_id`` rather than
  to an anonymous rectangle - so a later phase can ask what *this* entity has
  been doing.
- **Top-down, consuming tracks.** A bottom-up estimator finds every person in
  the frame and then groups limbs to bodies; being handed a box means only the
  second half of the problem is left. Phase 3 already produces boxes with stable
  identity, so the expensive half was already solved.
- **An explicit budget.** Cost is linear in people, so ``pose.max_persons`` caps
  it and the result records how many people were *offered* against how many were
  estimated. A skipped person appears on the HUD instead of quietly costing
  frame rate.
- **Coasting tracks are refused.** A predicted box is a guess about where a
  hidden object is; cropping to it returns the occluder, and the model would
  dutifully landmark it.

**Posture: four classes from geometry, or an honest refusal.**

Standing, sitting, crouching and lying, separated by two ratios - how far the
knees fall below the hips, and the ankles below the knees - both normalised by
torso length so they are free of scale and distance. Rules rather than a
classifier, because these separate cleanly by hand and a learned model would
need per-viewpoint labelled data and would return a number nobody can audit.

When the joints a rule needs are not visible the answer is **unknown, with the
reason attached** - "upright torso, but no knees visible: standing and sitting
are indistinguishable without them". That is the ordinary case for a desk
webcam, and it is the correct answer rather than a failure.

**Object state: motion, dwell, bearing and path length, for every entity.**

- No model, no weights, microseconds per frame: it reads the velocity the
  tracker's Kalman filter already maintains rather than computing a new one.
- **Speed in entity heights per second**, never pixels, so one threshold works
  across the frame - a person at the far end of a corridor covers a handful of
  pixels per second and the same person near the lens covers hundreds.
- **Hysteresis plus a minimum hold.** Two thresholds with a dead band between
  them, and a change must persist before it is published. Without both, an
  entity near the boundary flaps several times a second and every flap resets
  the dwell timer - destroying the one measurement the feature exists for.
- **Structured observation records** (``EntityState.to_observation``) in the
  shape the spec sketched for Phase 8 storage, with ``identity`` present and
  always ``None``: the seam the identity layer would later fill.

**Skeleton overlay** coloured by track id to match the boxes, drawing only the
joints the classifier treated as observed.

**546 tests** (+97), 527 of which need no camera, no model file and no inference
runtime; the remaining 19 exercise real weights and deselect cleanly.

---

## 2f. What Phase 5 delivers

Temporal activity recognition - the first thing here that cannot be answered
from a single frame at all. ``sitting_down`` is not a posture, it is a *change*
of posture; ``loitering`` is indistinguishable from standing until you know how
long it has lasted.

**Eight activities, each derivable and each stating its grounds.**

| Activity | Derived from | Needs pose |
|---|---|---|
| `walking`, `running` | sustained speed in heights/second | no |
| `loitering` | stationary dwell past a threshold | no |
| `idle` | present, nothing else recognised | no |
| `sitting_down`, `standing_up` | stable-posture transitions | yes |
| `falling` | upright to lying, fast | yes |
| `arm_raised` | wrist above shoulder, sustained | yes |

Several can hold at once, deliberately: a person can be walking with an arm up,
and forcing a single winner would discard one of two true statements. Every
observation carries **evidence** in words - `0.62 h/s, held on 100% of the last
0.4s` - so a surprising result can be argued with rather than merely believed.

**A ground-truth harness, which is again where the value was.** Eleven scripted
scenarios run through the *real* state estimator, scoring continuous recall,
event detection with latency, and - the half that matters - activities that
**must never fire**. `vantage activity eval` reports all three and exits
non-zero on failure, so it can gate a build.

**No model, and that was an evidence-based decision rather than a shortcut.**
See section 5.

**600 tests** (+54), 581 of which need no camera, no model file and no inference
runtime; the remaining 19 exercise real weights and deselect cleanly.

---

## 2g. What Phase 6 delivers

Spatial and interaction understanding - the first phase about *pairs* and
*places* rather than entities one at a time. It produces the two things the spec
sketched for the event engine to consume:

```text
Person #17  --approached-->  Person #21
Person #17  --entered-->     Zone "doorway"
```

**Zones.** Named polygons in **normalised** coordinates, so a zone drawn against
a 1080p stream still means the same part of the scene at 720p. An entity is in a
zone when its *ground point* - the bottom centre of its box - is inside; a
person's box centre drifts as they change posture, their feet do not. Overlapping
zones are supported, entry and exit are raised as events, and dwell accumulates
in footage time.

**Relations.** `near`, `approaching`, `receding` and `interacting_with`, each
carrying its evidence in words. Distances are in **entity heights**, never
pixels, so one threshold works across the frame.

**Interaction, reported at two confidence levels** - which is the honest core of
this phase. A wrist landmark inside the object's box is direct evidence and
scores **0.85**. Sustained proximity alone is capped at **0.4** and says so in
its evidence string, because two boxes close together in a flat image is
consistent with a person standing three metres behind the object.

**A scene graph**, `to_scene_record()`, as nodes and edges rather than prose -
because the next phase needs to *query* it, and `identity` is present and always
`None` on every node, the same seam the earlier phases left.

**A ground-truth harness** of ten scripted scenes, six of which exist to check
that something does **not** fire. `vantage spatial eval` exits non-zero on
failure.

**681 tests**, 662 of which need no camera, no model file and no inference
runtime; the remaining 19 exercise real weights and deselect cleanly.

---

## 2h. What Phase 7 delivers

The event engine - the reduction from *what is true* to *what happened*.

Everything before this phase produces **observations**: continuous statements
true on every frame for as long as they hold. `person_3 is loitering`,
`person_3 is in the doorway`. That is right for a state and wrong for an alert.
An **event** is discrete: it happened, at a time, once.

**The hard part is not the rules, it is the reduction.** Phase 5 holds a
transient activity for 1.5 s so a slow consumer cannot miss it - 45 frames at
30 fps. An engine that emitted per frame would raise **45 fall alerts for one
fall**, and whoever read them would learn to ignore the channel, which is the
same failure the fall rule avoided by refusing to hedge. Measured on the Phase 5
ground truth:

| Scenario | Events raised | Firings suppressed |
|---|---|---|
| `fall` | **1** alert | 45 |
| `lie_down_slowly` | **0** | 0 |
| `run` | 1 notice | 72 |
| `loiter` | 1 notice | 283 |
| `walk` | 0 | 0 |

**Cooldown keyed by rule *and* entity.** Without the rule, loitering would
silence a fall. Without the entity, two people falling in the same second would
produce one alert - and the second person is exactly who a missed alert fails.

**Six rule types, configured from YAML, validated at load.** `activity`,
`zone_entry`, `zone_exit`, `zone_dwell`, `zone_occupancy`, `relation`. A typo in
an activity name is caught when the config is read, not when the situation
finally occurs at three in the morning - because a rule that can never fire is
indistinguishable from a calm scene.

**Every event carries its evidence** as a dict of what the rule measured, and
`to_record()` emits the storable form with `identity` present and always `None`
- the same seam every earlier phase left.

**Suppressions are counted and reported**, never hidden. A rule suppressing
thousands of firings is either correctly debouncing a continuous state or badly
configured, and only the count tells the two apart.

**759 tests**, 740 of which need no camera, no model file and no inference
runtime; the remaining 19 exercise real weights and deselect cleanly.

---

## 2i. What Phase 8 delivers

Observation and event storage - the history that makes everything above
answerable after the fact.

**SQLite**, chosen on the same grounds as every dependency decision here: it is
in the standard library, needs no server, and handles the load. `Store` is a
Protocol, so Postgres can be added for the multi-camera phase without touching
anything that writes.

**The run loop never writes to disk.** It enqueues; a background thread batches
and commits. A slow disk must cost rows, never frames - a frame lost at the
camera is lost to every stage.

**Two queues, two policies**, and the difference is the design:

| | Observations | Events |
|---|---|---|
| Volume | ~120 rows/second | a dozen a day |
| Sampled? | yes, `observation_interval` | **never** |
| On overflow | dropped, counted, warned | dropped, counted, **logged as ERROR** |
| Retention | 30 days | 365 days |

A single shared queue was the first design and is wrong for exactly that reason:
under load you lose whatever arrived when it was full, and observations arrive a
hundred times more often than events.

**Volume is controlled by sampling, not by overflow.** Chosen sampling is
reproducible; what a full queue discards depends on when the disk was busy.

**Searchable, with a schema version from day one.** Time, entity, rule,
severity and zone are all indexed; `vantage history` queries them. The
`schema_version` table looks like ceremony for a schema nobody has changed - and
it is the one thing impossible to add later, because by then there are
databases in the field with no version marker.

**804 tests**, 785 of which need no camera, no model file and no inference
runtime.

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

### Bigger vocabulary, and open-vocabulary discovery (Phase 3.5)

```bash
vantage models list                    # note the CLASSES column
vantage run --source webcam:0 --track --device gpu --model dfine-s-obj365 --detect-interval 2
```

`dfine-s-obj365` knows 365 classes instead of COCO's 80, including `Pen/Pencil`, `Marker`,
`Stapler`, `Folder` and `Calculator`. It costs ~84 ms/frame on the iGPU against
`yolox-tiny`'s ~18 ms, so pair it with `--detect-interval 2`; the tracker's variable
timestep absorbs the gap and the display stays at full rate.

When even 365 classes is not enough, **discovery** takes arbitrary words:

```bash
pip install -e ".[discover]"           # adds the tokenizer, ~3 MB
vantage discover --prompts "pen, stapler, coffee mug" --source webcam:0
vantage discover --prompts "cable, adapter" --image desk.jpg --save found.png
```

This is a **separate tier on purpose**, not a switch on the live pipeline: it costs about
12 seconds per prompt and runs on one frame. See section 8 for the measurements behind
that split.

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

### Pose and object state (Phase 4)

```bash
vantage models pull rtmpose-s
vantage run --source webcam:0 --track --pose --device gpu
```

Each tracked person gets a skeleton coloured to match their box, and a posture
label when it can be determined. Motion state - moving or stationary, with how
long that has held - runs for **every** entity, person or not, and needs no
model at all.

```bash
vantage run --source webcam:0 --track --pose --pose-max-persons 3   # cap the cost
vantage run --source webcam:0 --track --pose --pose-interval 2      # every other step
vantage run --source webcam:0 --track --pose --no-face-keypoints    # drop head landmarks
vantage run --source webcam:0 --track --no-state                    # boxes only
```

If posture reads `unknown` on a desk webcam, that is the correct answer rather
than a fault: a seated person and a standing one are identical from the hips up,
and the classifier says so instead of guessing. The reason is carried on every
pose and printed in the debug log.

### Activity (Phase 5)

Runs automatically with tracking - no model, no flag, no cost worth measuring:

```bash
vantage run --source webcam:0 --track --pose --device gpu
```

Each entity gets a label under its box for what it is doing, and the HUD shows
the tally. Transient events - a fall, someone sitting down - are drawn in amber
and called out on their own HUD line, because they will be gone in a second.

Check the rules against ground truth rather than taking their word for it:

```bash
vantage activity scenarios     # what each scenario checks, including the negatives
vantage activity eval          # recall, event latency, forbidden firings
vantage activity eval --scenarios fall,lie_down_slowly
vantage run --track --no-activity     # turn it off
```

`activity eval` needs no camera, no model and no inference runtime, and exits
non-zero if any scenario fails - a forbidden firing or a missed event is a
failure, not a low score.

### Zones and relations (Phase 6)

Runs automatically with tracking. Relations need nothing extra; zones need to be
drawn, in normalised coordinates so they survive a change of resolution:

```yaml
spatial:
  zones:
    - name: doorway
      kind: entrance
      points: [[0.35, 0.5], [0.65, 0.5], [0.65, 1.0], [0.35, 1.0]]
```

```bash
vantage run --source webcam:0 --track --pose --config my.yaml
vantage spatial scenarios     # what each scene checks, including the negatives
vantage spatial eval          # relations, zone crossings, forbidden firings
vantage run --track --no-spatial
```

Zone outlines are drawn under everything else with their occupancy count, and
relations worth checking by eye - interaction and approach - are drawn as a line
between the two entities. `near` is deliberately not drawn: in a group everyone
is near everyone, and the mesh hides the one relation that matters.

### Events (Phase 7)

Runs automatically with tracking. The defaults do nothing on a quiet scene:

```bash
vantage events rules          # what is actually active, with cooldowns
vantage events types          # the six rule types
vantage run --track --no-events
```

Rules are configured under `events.rules`; an **empty list means the defaults**,
not "no rules", so a stray edit cannot silence every alert quietly. Turning the
subsystem off takes `enabled: false`.

```yaml
events:
  rules:
    - type: activity
      activity: falling
      severity: alert      # info | notice | alert
      cooldown_s: 15
      name: fall
    - type: zone_dwell
      zones: [entrance]
      min_seconds: 30
      severity: notice
    - type: relation
      relation: interacting_with
      min_confidence: 0.8  # 0.85 is reach-confirmed; 0.4 is proximity only
```

Events appear on the HUD coloured by severity and in the run summary with the
suppression count.

### History (Phase 8)

Off by default - a tool that silently created a growing database would be a
surprise. Turn it on per run:

```bash
vantage run --source webcam:0 --track --pose --store
```

Then query it, while the run is still going if you like (WAL means a reader
never blocks the writer):

```bash
vantage history stats                          # rows, span, size on disk
vantage history events --since 1h              # what happened
vantage history events --severity alert        # only the ones that matter
vantage history timeline --entity person_3     # one entity, oldest first
vantage history observations --zone till --since 30m
vantage history prune --older-than 30d
```

`prune` refuses to run without an explicit horizon, because the wrong guess
deletes data.

### One-click launch (Windows)

`webcam.bat` in the repository root runs the live pipeline without typing the
command. Double-click it, or run `.\webcam.bat`. An optional first word picks
what to run:

```bat
webcam.bat            :: pose (default) - people, skeletons, motion state
webcam.bat activity   :: pose plus activity recognition, tuned to demonstrate
webcam.bat objects    :: 365-class detection + tracking, no pose
webcam.bat plain      :: detection only, no tracking and no pose
webcam.bat checks     :: no camera - score the tracker and the activity rules
```

The detection modes use **different detectors on purpose**, and the reason is
measured rather than assumed. Pose costs about 5 ms per person on the iGPU, but
only when the detector leaves the GPU room: paired with `yolox-tiny` (10.8 ms)
the whole pipeline holds 30 fps, while the 365-class `dfine-s-obj365` (84 ms)
needs the frame budget for itself.

`activity` mode lowers `loiter_s` from 20 seconds to 5, because 20 seconds is a
long time to stand still in front of your own webcam to find out whether a
feature works. It is a **demo value, not a recommendation**, and the banner says
so rather than letting you infer the shipped default is 5. It also prints what
to physically do to trigger each activity.

`checks` runs both ground-truth harnesses - no camera, no weights, no inference
runtime - and exits non-zero if either fails. It takes no arguments, because the
two harnesses have different scenario names and forwarding a flag to both would
score one and fail the other.

Anything else you type is appended and therefore overrides the defaults:

```bat
webcam.bat activity --set activity.loiter_s=20
webcam.bat pose --pose-max-persons 2
webcam.bat objects --classes person,laptop
webcam.bat --source webcam:1 --no-hud
```

`VANTAGE_MODEL`, `VANTAGE_DEVICE`, `VANTAGE_SOURCE` and `VANTAGE_INTERVAL`
change the defaults without editing the file. No path is baked in - it resolves
the virtual environment and the model cache relative to itself, so it survives a
move or a fresh clone, and it holds the console open on failure so Explorer
cannot swallow the error.

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
├─ pose/          Phase 4: what shape a tracked person is in
│  ├─ contracts.py   Keypoint, Pose, PoseResult, Posture (+ the privacy note)
│  ├─ adapter.py     RTMPose: top-down affine, ImageNet norm, SimCC decode
│  ├─ engine.py      tracks in, skeletons out, with an explicit budget
│  ├─ posture.py     standing / sitting / crouching / lying, or a reasoned refusal
│  └─ factory.py     config -> pose engine
│
├─ state/         Phase 4: what a tracked entity is doing, whatever it is
│  ├─ contracts.py   MotionState, EntityState, the observation record
│  └─ estimator.py   hysteresis, dwell timing, path length; no model, no weights
│
├─ storage/       Phase 8: the history, in SQLite
│  ├─ contracts.py   StoredEvent, StoredObservation, Query, the Store Protocol
│  ├─ schema.py      DDL, indexes, pragmas, schema versioning
│  ├─ sqlite_store.py  batched writes, indexed reads, retention
│  ├─ writer.py      background thread; separate queues for events and rows
│  ├─ recorder.py    the seam between pipeline output and database columns
│  └─ query_cli.py   'vantage history'
│
├─ events/        Phase 7: what happened, once - the discrete reduction
│  ├─ contracts.py   Event, Severity, the storable record
│  ├─ rules.py       six parameterised rule types, validated at load
│  └─ engine.py      cooldown keyed by rule and entity; suppression counting
│
├─ spatial/       Phase 6: where entities are, and how they relate
│  ├─ contracts.py   Zone, Relation, the scene-graph record
│  ├─ analyzer.py    ray-cast zone tests, ground distance, the relation rules
│  ├─ engine.py      footage-time accounting, pose and state pairing
│  ├─ scenarios.py   scripted scenes, six of ten of them negatives
│  └─ evaluation.py  relations found, zone crossings, forbidden firings
│
├─ activity/      Phase 5: what has been happening, over time
│  ├─ contracts.py   Activity, ActivityObservation, the observation record
│  ├─ base.py        the Recognizer Protocol - the seam for a learned model
│  ├─ recognizer.py  the rules: sustain windows, stable posture, transitions
│  ├─ engine.py      buffers, pose pairing, footage-time accounting
│  ├─ scenarios.py   scripted ground truth, positive and negative
│  └─ evaluation.py  recall, event latency, forbidden firings
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

### Vocabulary is a property of the weights, not a setting

The most common misdiagnosis this platform invites is "the detector is bad, tune it".
Often the object simply is not in the class list. COCO has 80 classes and none of them is
a pen, so YOLOX has no output channel that could ever fire for one - no confidence
threshold, model size or NMS setting changes that. `vantage models list` reports the class
count per model for exactly this reason.

Objects365 (365 classes) is the answer for ordinary objects, and it is a weights swap
rather than an architecture change. The genuinely open-vocabulary models, which take
arbitrary text prompts, are a different trade entirely - see the discovery tier below.

### DETR-family models still need NMS here, whatever the papers say

The DETR line is built on set prediction with Hungarian matching, and the literature is
clear that non-maximum suppression is therefore unnecessary. That was written into the
D-FINE adapter as fact and then measured to be false: on a live frame at a 0.30 threshold
this export produced **six `Person` boxes for one person**, two pairs overlapping at IoU
0.90 and 0.84. Unsuppressed, Phase 3 confirms each as a separate track and the system
invents people. So class-aware NMS runs, and the module docstring records that the theory
lost to the measurement.

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

### Interaction needs motion state, because duration alone cannot tell
lingering from passing

Interaction started as "a person close to an object for long enough" - the
sustain window being what excludes someone walking past. That is wrong, and the
harness quantified exactly how wrong. Walking past a static object at 180 px/s
produced nothing, which looked like success. The *same path* at 45 px/s - an
amble - produced **49 frames of false interaction**, because a slow enough
walk-past satisfies any sustain threshold. Raising `interact_s` only moves the
speed at which it breaks.

The discriminator that actually exists is motion state, which Phase 4 already
computes with hysteresis: someone lingering is stationary, someone passing is
moving. Proximity-only interaction now requires the person to have stopped. A
confirmed reach still counts on its own, because taking something while walking
is real and a wrist landmark inside the box is direct evidence rather than an
inference from two rectangles.

Without motion state at all - pose running, state disabled - only reach-confirmed
interaction is claimed, and the HUD says so rather than leaving it to be inferred
from missing rows.

### A harness that cannot express motion cannot test a rule that reads it

The fix above did not appear to work. The scripted scenarios still reported the
false interactions, because their `Track` objects were built without velocities
and defaulted to zero - so the real state estimator dutifully called every actor
**stationary**, however fast their path moved, and the new motion gate was
satisfied on every frame.

The bug was in the harness, not the code under test, and it is the more
interesting kind: a test double that is wrong in a way that makes the system look
correct. Scenario tracks now derive velocity from their own paths by finite
difference, and a test asserts that scripted actors actually move.

### Activity recognition ships no model, and that was measured

The obvious choice was a learned skeleton-action model. It was surveyed before
the rules were written, and three findings ruled it out - none of them a matter
of taste:

* **No permissively licensed export with real provenance exists.** OpenMMLab
  publishes PoseC3D and ST-GCN as PyTorch checkpoints and ships no ONNX SDK for
  them (`/mmaction/v1.0/skeleton/onnx_sdk/` is a 404). The hub's `st-gcn`
  results are unrelated models: traffic forecasting, weather, sign language.
* **The video classifiers that do exist are the wrong shape.** VideoMAE and
  friends label a *frame*, not an entity - throwing away the identity this whole
  platform is built around, and answering "someone is doing X somewhere" when
  the question is "what is person_17 doing".
* **Their vocabularies are wrong.** Kinetics-400 offers `abseiling`, `zumba`
  and `shredding paper`. NTU is lab-recorded daily living. Neither contains
  `loitering`, and a model that confidently reports the wrong *kind* of thing is
  worse than a short list of things that are actually true.

So the recogniser is rules over measured signals.
:class:`~vantage.activity.base.Recognizer` is a Protocol precisely so a learned
model can replace it when one is worth having; the engine, buffers and contracts
would not change.

### Two stages must not disagree about the same entity

`state.moving_above` decides whether an entity is moving; `activity.walking_speed`
decides whether that counts as walking. Setting the second higher than the first
opens a band where both are true at once: the state machine reports **moving**
while no locomotion rule fires, so the entity is simultaneously moving and
`idle`.

This was not hypothetical. With the thresholds at 0.15 and 0.20, a real clip of
a person crossing the frame at 0.175 h/s produced exactly that contradiction.
The defaults are now equal and the configuration is rejected if they are not.
The state machine has already applied hysteresis and a minimum hold to decide
that motion is genuine; second-guessing it with a higher threshold downstream
only produces disagreement.

### A bare majority is not enough to debounce a flicker

Stable posture is what transitions are detected between, and promoting the
majority posture of a short window looked obviously sufficient. It is not, and
the failure is specific rather than theoretical.

Under a perfect frame-by-frame alternation the window holds an even split, which
a strict majority correctly refuses - **but only while the window holds an even
number of samples**. On every frame where it holds an odd number, one posture
leads by one, the stable posture flips, and a transition fires. Measured against
120 alternating frames: **101 spurious transitions**.

A supermajority tops out near 53% under alternation and never promotes. The cost
is real and was measured too: event latency rose from 0.30 s to 0.43 s.

### Posture is rules, and says "unknown" rather than guessing

Four postures separate on two ratios that anyone can check by hand against a
drawn skeleton. A learned classifier would need labelled data from each
camera's viewpoint and would return a number nobody can audit, for a
distinction this coarse.

The consequence worth stating is the refusal. When the joints a rule needs are
not visible - which is *most of the time* on a desk webcam, because it never
sees anyone's legs - the classifier returns `unknown` and the reason why. An
"unknown" that looks like a bug is much better than a confident "standing" that
is a coin flip, and the reason string is what makes the difference visible.

The rules read the image, not the world. A camera angled steeply down
compresses vertical drops and will eventually read a standing person as
crouching; one mounted sideways would read everyone as lying. There is no
horizon estimate or calibration to correct for that, so a tilted installation
needs the thresholds reviewed rather than trusted. This is written down because
the failure is quiet: the numbers stay confident while being wrong.

### Speed is measured in body heights, not pixels

A person walking at the far end of a corridor covers a handful of pixels per
second; the same person a metre from the lens covers hundreds. Both are
walking. Dividing pixel velocity by box height makes one threshold work
everywhere in the frame without camera calibration.

It is imperfect and the imperfection is specific: someone walking directly at
the camera grows rather than translates, and reads as slower than they are.
Fixing that needs ground-plane geometry, which needs calibration, which is a
Phase 6 concern.

### Model provenance beat convenience for the pose weights

RTMPose is distributed by OpenMMLab as a zip containing the graph. Loose
re-uploads of the same file exist on model hubs and would have been a one-line
catalog entry, but they are anonymous, carry no statement of which checkpoint or
export config produced them, and **a SHA-256 pin cannot make a file trustworthy
- it can only make it unchanging**. Reading the member out of the authoritative
archive cost about forty lines in the model store and kept the provenance
intact.

Archived models are pinned twice, on the archive as transferred and on the
extracted member before it is installed. The second pin is what preserves the
store's first rule - a cached file is always re-hashed against the pin for the
file it actually is - because what ends up cached is the member, not the archive.

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

### Pose cost, measured (Intel Iris Xe, one person)

Isolated, in a tight loop with nothing else running:

| Device | Preprocess | Inference | Decode | Total |
|---|---|---|---|---|
| CPU | 1.12 ms | 8.84 ms | 0.23 ms | **10.19 ms** |
| iGPU | 0.94 ms | 3.74 ms | 0.20 ms | **4.88 ms** |

Inside the live pipeline the same model costs more, and **how much more is not
stable**:

| Configuration | Detection | Pose, per person |
|---|---|---|
| Tight loop, nothing else running | - | 4.9 ms |
| `yolox-tiny` on iGPU | 14.3 ms | 7.0 ms |
| `yolox-s` on iGPU | 34.5 ms | 9.0 ms |
| `yolox-s` on iGPU, separate run | 44.2 ms | 12.6 ms |
| `yolox-s` on **CPU**, pose on iGPU | 280 ms | 9.1 ms |

The obvious reading is contention for a single integrated GPU, and the first
three rows support it. The last row does not: moving the detector off the GPU
entirely should have been the cheapest case for pose and was not, and two runs
of the same configuration differed by 3.6 ms. **So the cause is not established.**
Contention, iGPU clock behaviour under bursty load, and cache pressure from the
rest of the loop are all plausible and none was isolated.

What is safe to plan around is the range rather than the mechanism: budget
**5-13 ms per person** in a live pipeline on this hardware, not the 4.9 ms a
benchmark loop suggests. With four people that is up to 50 ms of a 33 ms frame,
which is why `pose.max_persons` and `pose.interval` exist and why the tracker's
variable timestep matters.

### Motion state on real footage

A 120-frame clip of a real person, panning for 60 frames and then held still:

======  ======================================================================
Frame   Observed
======  ======================================================================
2       appears; state `unknown` - no velocity estimate worth the name yet
25      `moving` at 0.154 h/s, bearing 84 degrees (due right, as the clip pans)
59      still `moving`, 0.21 heights of path accumulated
79      `stationary` - 0.63 s after the motion stopped, which is the 0.5 s
        minimum hold plus filter lag
119     `stationary` for 1.33 s, path length unchanged at 0.22 heights
======  ======================================================================

Path length stops accumulating when the entity stops, and the 0.21 heights
recorded matches the 70 px of travel against a 343 px box.

### Activity recognition (`vantage activity eval`)

Eleven scripted scenarios through the real state estimator. Recall is over
scored frames, latency is from the movement that should cause an event:

```
SCENARIO                           RECALL   EVENTS   LATENCY  FORBIDDEN
walk                               100.0%        -         -          0
run                                100.0%        -         -          0
loiter                             100.0%        -         -          0
sit_down                                -      1/1     0.43s          0
stand_up                                -      1/1     0.43s          0
fall                                    -      1/1     0.43s          0
fall_after_standing_a_while             -      1/1     0.43s          0
lie_down_slowly                         -        -         -          0
arm_raised                         100.0%        -         -          0
no_pose                            100.0%        -         -          0
jitter                             100.0%        -         -          0
POOLED                             100.0%      4/4     0.43s          0
```

Five of the eleven exist to check that something does **not** fire.
`lie_down_slowly` is the one that matters most: a deliberate lie-down via a
crouch must produce no fall at all. `jitter` checks that a standing person whose
box wobbles is never reported as walking, and `no_pose` that locomotion still
works with pose disabled while no posture-derived activity is invented from
nothing.

### Spatial and interaction (`vantage spatial eval`)

Ten scripted scenes through the real state estimator:

```
SCENARIO                RELATIONS    ZONES  FORBIDDEN   FRAMES
zone_crossing                   -      2/2          0      180
zone_overlap                    -      2/2          0      180
two_people_meet               2/2        -          0      180
two_people_part               1/1        -          0      180
far_apart                       -        -          0      180
walk_past_object                -        -          0      120
amble_past_object               -        -          0      360
reach_while_walking           1/1        -          0      150
linger_by_object              1/1        -          0      180
reach_for_object              1/1        -          0      180
POOLED                        6/6      4/4          0     1890

Peak interaction confidence (evidence tier):
  linger_by_object     0.40  proximity only
  reach_for_object     0.85  reach-confirmed
  reach_while_walking  0.85  reach-confirmed
```

Six of the ten are negatives. `amble_past_object` is the regression that forced
the motion gate; `two_people_meet` checks that two people standing together are
*near* and never *interacting*, since geometry alone cannot support that claim
between two people.

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

## 7b. Running this in production

Everything above measures whether the analysis is *correct*. This section is
about whether the process survives a fortnight on a camera, which is a different
question and was addressed as its own pass rather than assumed.

### A failing stage no longer stops the run

Every analysis stage was called directly inside the run loop with a single
`try` around the whole thing. That is fine for a benchmark and wrong for a
deployment: one malformed frame, one transient driver fault, and a camera that
was meant to run for weeks stops. This platform has already met exactly such a
fault, when the iGPU returned `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`.

Each stage now runs behind a guard. Measured by injecting faults into a real
detector:

| Injected fault | Before | After |
|---|---|---|
| Fails every 3rd frame | run dies on frame 3 | **all 30 frames complete**, 20 detections, 10 failures logged |
| Fails every frame | run dies on frame 1 | **all 30 frames complete**, stage disabled after 5, `DEGRADED` in summary |

This is not silent exception handling, which this project forbids. Every failure
is logged - the first with a traceback - every failure is counted into the run
summary and the HUD, and a stage that fails `app.stage_failure_budget` times in
a row is **disabled with an ERROR naming it**, because a stage failing every
frame is broken rather than unlucky and retrying it forever produces the worst
available outcome: a system neither working nor obviously broken.

`MemoryError` and `KeyboardInterrupt` are deliberately never caught. Skipping a
frame does not give memory back, and the second is the operator asking to stop.

### CPU and memory, without a dependency

The spec asks for both under observability, and for a long-running process a
slow leak is the characteristic failure - invisible in a frame rate until the
machine swaps. psutil would have worked and was declined for one number: CPU
comes from `time.process_time()` and memory from one platform call each
(`GetProcessMemoryInfo`, `/proc/self/statm`, `getrusage`). Where a platform is
not covered, memory reports **`None`** rather than zero, because a silent zero
would make a leak look like perfect health.

CPU is reported in **cores** - 1.0 is one core saturated - rather than a
percentage that means different things on different machines.

### Memory over a long run, measured

Full analysis chain over a synthetic source:

| Frames | RSS at end | Growth | Per frame |
|---|---|---|---|
| 500 | 67.3 MB | +18.5 MB | 37,000 B |
| 2,000 | 66.7 MB | +9.4 MB | 4,700 B |
| 6,000 | 67.9 MB | +9.9 MB | 1,650 B |

Resident memory is **flat at ~67 MB** whether the run is 500 frames or 6,000,
and per-frame growth *falls* as the run lengthens. That is the signature of
one-off warmup allocation, not retention. `pytest -m slow` asserts it.

### Adaptive load shedding

A detector taking 84 ms cannot run on every frame of a 30 fps camera. Left
alone the pipeline does not slow down - it **drops** frames under backpressure,
so analysis silently sees an arbitrary subset of reality while the frame rate
looks fine, and every later stage inherits that.

The governor computes the interval rather than hunting for it: if analysis costs
`C` and frames arrive every `B`, then analysing one frame in `N` costs `C/N` per
delivered frame, so the smallest workable `N` is `ceil(C / (B x headroom))`. It
lands in one step; a controller that nudged up and down would take seconds and
oscillate. Hysteresis then stops it twitching - raising is prompt because being
over budget is a live problem, lowering needs six seconds of headroom because
lowering early puts the pipeline straight back into the overload it escaped.

Measured, with a detector forced to take 60 ms against a 33 ms budget:

| Adaptive | Delivered | Dropped | Detections |
|---|---|---|---|
| **on** | **30.9 fps** | 67,440 | 102 |
| off | 15.9 fps | 132,684 | 200 |

Delivery rate doubles and drops halve, in exchange for analysing fewer frames.
That is the trade, stated plainly: temporal resolution of the analysis is given
up to keep the pipeline current.

**Live sources only.** A recorded file has no deadline - analysing it slowly
gives the same answer - so shedding load there would discard information for
nothing. When a recorded source is opened the governor stands down and says so,
because one that silently does not run is worse than one that does.

Every interval change is logged with its reason, and the run summary reports
peak interval and time degraded. A system that quietly changes how much of
reality it looks at makes every later result inexplicable.

### Types

`mypy` runs clean over all 80 source modules and gates CI. It is deliberately
not `--strict`: the value is in the mistakes a checker is uniquely good at, and
requiring annotations everywhere would mostly add noise to numpy-heavy code
where the useful type is "an array of some shape".

It earned its place immediately, finding two real defects in code written the
same hour:

* **`_shorten` defined twice** in the HUD. The second silently shadowed the
  first, which was dead from the moment it was written.
* **`interval` bound twice in the run loop** with different meanings *and
  different types* - the analysis interval, then the stats-logging interval.
  Functionally safe only because the first is reassigned at the top of every
  iteration; a latent trap either way.

Two configuration findings are worth recording, because both were wrong first:

* **Pinning `python_version` to the project floor was wrong.** At 3.11 mypy
  parses numpy's bundled stubs under 3.11 rules, where their `type` statements
  are a syntax error. Following the running interpreter fixes it.
* **Skipping numpy's stubs was worse.** It turned one parse error into
  seventeen "need type annotation" errors, because without them no array
  expression has an inferable type. The fix made the codebase less checkable,
  not more.
* **`warn_unreachable` is off**, and not for convenience. mypy narrows
  `sys.platform` to the platform it runs on, so every correct
  `if sys.platform == "linux"` branch is "unreachable" when checked on Windows.
  With CI on both, the flag fails one job or the other for correct code, and
  per-branch silencing is worse: the ignore needed on Windows becomes an
  unused-ignore error on Linux. It found one genuine dead statement before
  being turned off, which is fixed.

### Storage found two bugs that unit tests structurally could not

Both surfaced on the first run that actually stored something, and neither was
the kind of mistake a unit test was ever going to catch.

**A SQLite connection crossed threads.** The store was created on the caller's
thread and used from the writer's, which SQLite refuses outright: *"SQLite
objects created in a thread can only be used in that same thread."* Unit tests
did not catch it because unit tests do not cross threads. The tempting one-word
fix, `check_same_thread=False`, is wrong - it silences the guard without making
anything safe. Connections are now per-thread, which is genuinely safe, and WAL
is what lets them coexist.

**An ISO timestamp went into a numeric column.** Phase 7's `to_record()` renders
the timestamp as an ISO string, which is right for JSON and wrong for a REAL
column that range queries sort on. SQLite is dynamically typed and accepted it
without complaint; the first query that formatted one failed with *"'str' object
cannot be interpreted as an integer"*.

The lesson is the same in both: export shape and storage shape are different
concerns. `to_record()` stayed as it was - it is the API-facing form - and the
recorder now builds database rows from typed fields. Both have regression tests
that say why they exist.

A third, smaller one: `counts()` reported **4096 bytes for five hundred rows** -
the size of an empty database - because under WAL the data lives in a sidecar
file until checkpoint. Anyone using that figure to decide whether to prune would
have concluded there was nothing there.

### CI gates on the ground-truth harnesses

`.github/workflows/ci.yml` runs on Linux and Windows, Python 3.11 and 3.13. As
well as lint, format and unit tests, it runs the three accuracy harnesses as
build gates:

```bash
vantage track eval      # tracker: MOTA, IDF1, identity switches
vantage activity eval   # activity: recall, event latency, forbidden firings
vantage spatial eval    # spatial: relations, zone crossings, forbidden firings
```

Each already exits non-zero when a scenario regresses, so a change that quietly
makes the tracker worse - or makes the fall rule fire when someone lies down
deliberately - fails the build instead of being noticed months later. No weights
are downloaded; the ~19 tests needing real ONNX files deselect themselves.

Both platforms are built because the platform-specific paths genuinely differ:
memory reporting uses `/proc` on one and psapi on the other.

### One bug this pass found in itself

Wrapping each stage introduced eleven closures of the form
`lambda: detect(frame)` over a loop variable - the exact shape of a late-binding
bug. They were safe, because the guard calls them immediately, but nothing in
the signature said so and a later change that deferred the call would have
turned eleven safe closures into eleven wrong ones at once. `StageGuard.run`
now takes the arguments instead of a closure, which makes the whole class of
mistake unavailable. Ruff's `B023` is what surfaced it.

Also fixed while wiring the memory probe: the Windows call returned failure
every time because the process handle was passed as an undeclared integer,
which ctypes truncates to 32 bits. Memory read as "unavailable on this
platform" on the platform it was written for.

## 8. Known limitations

**Scope.** No activity recognition, events, alerts, storage, dashboard, multi-camera
orchestration, or identity. By design.

**Posture needs legs, and a desk webcam has none.** A seated person and a
standing one are identical from the hips up, so on a typical head-and-shoulders
webcam framing the classifier reports `unknown` and says why. That is correct,
not broken - but it does mean posture is close to useless for desk-mounted
cameras, and useful mainly for a camera that sees whole bodies.

**Posture reads the image, not the world.** No horizon estimate and no
calibration, so a steeply angled camera compresses vertical drops and will
eventually read standing as crouching. A tilted installation needs the
thresholds reviewed. The failure is quiet: the numbers stay confident.

**Pose cost is linear in people, and its in-pipeline cost is not fully
understood.** It runs at 4.9 ms per person in a tight loop but 5-13 ms inside
the live pipeline, varying by 3.6 ms between two runs of the same
configuration. Contention for the single iGPU is the obvious explanation and
the measurements only partly support it (see section 7). Budget the range, not
the benchmark. `pose.max_persons` and `pose.interval` bound the total; there is
no adaptive control that lowers them automatically as a room fills.

**Motion state cannot see motion toward the camera.** Speed is box displacement
over box height, so someone walking directly at the lens grows rather than
translates and reads as slower than they are. Correcting it needs ground-plane
geometry, which needs calibration.

**Transitions are reported late** by up to `state.min_state_s` (0.5 s by
default) plus filter lag - measured at 0.63 s end to end. That is the price of
dwell timings that mean something, and it is a poor trade for anything needing
sub-second reaction.

**`falling` is not a certified fall detector** and must not be relied on where
one is required. It reports that a body went from upright to horizontal quickly,
which is a different claim. It **needs legs** - posture needs hips and knees, so
a camera seeing people only from the waist up can never report a fall at all -
and it inherits the posture rules' blind spot for steeply angled cameras. A
person lowering themselves deliberately is reported as **nothing**, not as a
low-confidence fall, because a hedged alert teaches whoever reads it to ignore
the real one. Detection latency is 0.43 s on scripted ground truth.

**Spatial claims rest on a common ground plane, and a camera has no depth.**
Two people on opposite sides of a room can have boxes that overlap perfectly.
Proximity is a good approximation for entities at similar depth and degrades as
their depths diverge. Distances are entity heights, not metres, and thresholds
should be tuned per camera rather than trusted as physical distances. An object
that is held or wall-mounted breaks the anchor entirely: its "ground point" is
not on the ground, so every distance to it is wrong by however far off the floor
it is.

**Interaction is the weakest claim in the platform**, which is why it carries two
confidence levels rather than one. At 0.4 it means "a stationary person was
close to this for a while in a flat image" - not that they touched it. Only the
0.85 reach-confirmed tier rests on direct evidence.

**Activity thresholds are chosen, not tuned.** Unlike the tracker's parameters,
which are the output of a search against held-out data, the activity thresholds
were set from published gait figures and checked against the scenarios. The
scenarios confirm the rules behave as specified; they cannot confirm that
`loiter_s: 20` is the right number for your building.

**Open-vocabulary detection cannot run live on this hardware, and the split is deliberate.**
Measured, not estimated:

| Model | Vocabulary | Per frame | Throughput |
|---|---|---|---|
| `yolox-tiny` | 80 fixed | 18 ms | 57 fps |
| `dfine-s-obj365` | 365 fixed | 84 ms | 12 fps |
| `grounding-dino-tiny` | **any text** | ~12 s per prompt | 0.08 fps |

That is a ~700x gap, not a tuning gap, which is why discovery is a separate command rather
than a mode of `run`. Three findings shaped it, all from measurement:

* **Text and image share one fused graph**, so prompt embeddings cannot be precomputed the
  way OWL-ViT allows. Every pass pays the full cost.
* **One prompt per pass is mandatory.** Batched, the model suppresses all but the strongest
  phrase - `dog, bicycle, car` together scored 0.90/0.09/0.08, and separately 0.92/0.89/0.59.
  The batched form is *wrong*, not merely weaker, so cost is linear in prompt count.
* **CPU beats the iGPU here by 7x** for one-shot use: OpenVINO spends ~155 s compiling the
  graph to save 9 s of inference. Discovery therefore defaults to whatever backend is
  cheapest to start, not the fastest to run - the opposite of the live pipeline's choice.

Token budget is capped at 32 (~8-10 short prompts): cost grows super-linearly with sequence
length (32 -> 3.1 s, 128 -> 14.1 s) and 256 tokens crashes the iGPU kernel outright.

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

**Phase 3.5 — Larger detection vocabulary. Done, and unplanned.** Inserted ahead of the
spec's Phase 4 because a real failure demanded it: the platform could not see a pen, and
that is a property of COCO's output tensor rather than a threshold to tune. Numbered 3.5
so that the spec's phase numbers keep meaning what they meant. Section 2d has the
measurements.

**The remainder, as originally specified.** The order reflects where the evidence points,
not a fixed plan:

| # | Phase | Status |
|-----|-------|--------|
| 1 | Video ingestion | Done |
| 2 | Object detection | Done |
| 3 | Multi-object tracking | Done |
| 3.5 | Larger detection vocabulary | Done, inserted |
| 4 | Human pose & object state | Done |
| 5 | Temporal activity recognition | Done |
| 6 | Spatial & interaction understanding | Done |
| 7 | Event engine | Done |
| 8 | Observation & event storage | Done |
| 9 | Visualization / dashboard | Next |
| 10 | Identity & enrolment | Optional - see below |
| 11 | Advanced analytics | |
| 12 | Optimization / multi-camera scaling | |

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
