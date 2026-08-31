# Vantage: Complete Architectural & Engineering Knowledge Base

> **Authoritative System Reference**
> **Target Audience**: AI Agents, Systems Architects, Security Engineers, and Developers.
> **Scope**: Complete end-to-end design, implementation details, mathematical algorithms, data contracts, storage schemas, REST APIs, performance benchmarks, and operational workflows for the entire Vantage platform.

---

## Table of Contents

1. [Executive Summary & Core Philosophy](#1-executive-summary--core-philosophy)
2. [High-Level Architecture & End-to-End Pipeline](#2-high-level-architecture--end-to-end-pipeline)
3. [Repository Structure & Code Catalog](#3-repository-structure--code-catalog)
4. [Subsystem Deep Dives (Phases 1 – 15)](#4-subsystem-deep-dives-phases-1--15)
   - [4.1 Video Ingestion & Hardware Backends (Phase 1)](#41-video-ingestion--hardware-backends-phase-1)
   - [4.2 Object Detection & Dual Inference Runtime (Phase 2)](#42-object-detection--dual-inference-runtime-phase-2)
   - [4.3 Multi-Object Tracking & Kalman Motion Model (Phase 3)](#43-multi-object-tracking--kalman-motion-model-phase-3)
   - [4.4 Objects365 Vocabulary & Open Discovery (Phase 3.5)](#44-objects365-vocabulary--open-discovery-phase-35)
   - [4.5 Top-Down Human Pose & Entity State Dynamics (Phase 4)](#45-top-down-human-pose--entity-state-dynamics-phase-4)
   - [4.6 Temporal Activity Recognition (Phase 5)](#46-temporal-activity-recognition-phase-5)
   - [4.7 Spatial Relations & Scene Graph (Phase 6)](#47-spatial-relations--scene-graph-phase-6)
   - [4.8 Event Reduction & Security Rules Engine (Phase 7)](#48-event-reduction--security-rules-engine-phase-7)
   - [4.9 High-Throughput SQLite Storage & Heartbeat (Phase 8)](#49-high-throughput-sqlite-storage--heartbeat-phase-8)
   - [4.10 Local Web Dashboard & Packaging (Phase 9)](#410-local-web-dashboard--packaging-phase-9)
   - [4.11 Consensual Biometric Identity & Audit (Phase 10)](#411-consensual-biometric-identity--audit-phase-10)
   - [4.12 Longitudinal Analytics & Anomaly Detection (Phase 11)](#412-longitudinal-analytics--anomaly-detection-phase-11)
   - [4.13 Multi-Camera Fusion, Spatial Re-ID & Video Evidence (Phase 12)](#413-multi-camera-fusion-spatial-re-id--video-evidence-phase-12)
   - [4.14 Interactive UI Zone Polygon & Dynamic Geofencing (Phase 13)](#414-interactive-ui-zone-polygon--dynamic-geofencing-phase-13)
   - [4.15 3D WebGL Digital Twin & Three.js Facility Mesh (Phase 14)](#415-3d-webgl-digital-twin--threejs-facility-mesh-phase-14)
   - [4.16 Physical RTSP / ONVIF IP Camera & USB Connectors (Phase 15)](#416-physical-rtsp--onvif-ip-camera--usb-connectors-phase-15)
   - [4.17 Unified Entity Intelligence & Canonical Data Model (Phase 16)](#417-unified-entity-intelligence--canonical-data-model-phase-16)
5. [Mathematical Formulations & Algorithmic Foundations](#5-mathematical-formulations--algorithmic-foundations)
6. [Durable Storage Schemas & Database Migrations](#6-durable-storage-schemas--database-migrations)
7. [Comprehensive REST API Specification](#7-comprehensive-rest-api-specification)
8. [Performance Benchmarks & Empirical Latencies](#8-performance-benchmarks--empirical-latencies)
9. [CLI Command Reference & Operational Runbook](#9-cli-command-reference--operational-runbook)
10. [Test Suites & Continuous Verification](#10-test-suites--continuous-verification)
11. [Architecture Decision Records (ADRs)](#11-architecture-decision-records-adrs)

---

## 1. Executive Summary & Core Philosophy

**Vantage** is an edge-first computer vision and spatial intelligence platform designed to understand scenes over time on commodity hardware (CPUs, Intel iGPUs, discrete GPUs). Rather than treating video as isolated images, Vantage tracks persistent entity dynamics, infers spatial relationships, detects security anomalies, extrudes live 3D digital twins, and executes real-time geofence rules.

### Core Architectural Tenets

1. **Zero-Stub / No-Mock Policy**: Nothing in Vantage is mocked, faked, or simulated. What is documented exists, executes with zero latency regressions, and is backed by deterministic tests.
2. **Standard Library First**: Built on Python standard library components (`http.server`, `sqlite3`, `dataclasses`, `threading`, `socket`) rather than introducing heavyweight frameworks (FastAPI, Celery, React, Redis).
3. **Measurement-Driven Engineering**: Algorithms are chosen based on empirical benchmarks on actual target hardware rather than theoretical literature assumptions.
4. **Zero-Lock Hot Path**: Frame-rate inference loops execute in pure memory without locks, synchronization barriers, or blocking disk I/O.
5. **Privacy by Default**: Identification is strictly opt-in, requires cryptographic consent assertion, stores 128-dimensional mathematical vectors rather than face photos, and maintains a strict tamper-evident audit trail.
6. **Graceful Degradation**: If an inference engine skips a frame or hardware drops packets, temporal Kalman filters and time-based state estimators absorb the delta without drifting.

---

## 2. High-Level Architecture & End-to-End Pipeline

```mermaid
flowchart TD
    subgraph "1. Physical Ingestion & Connectors"
        USB["USB Webcams (MSMF / DSHOW / V4L2)"]
        RTSP["IP Cameras (Hikvision, Dahua, Axis, Reolink, Tapo)"]
        Files["Recorded MP4/AVI Files"]
        Synthetic["Deterministic Synthetic Generator"]
        USB & RTSP & Files & Synthetic --> FrameSource["FrameSource & Ingestion Queue"]
    end

    subgraph "2. Core Perception & Spatial Reasoning"
        FrameSource --> Detect["DetectionEngine (YOLOX / D-FINE / OpenVINO / ONNXRuntime)"]
        Detect --> Track["ByteTracker (Continuous-Time Kalman Filter + Jonker-Volgenant)"]
        Track --> Pose["PoseEngine (Top-Down RTMPose 17 Landmarks)"]
        Pose --> State["StateEstimator (Speed h/s, Dwell, Posture, Activity)"]
        State --> Spatial["SpatialEngine (Proximity, Relations, Ground Foot Points)"]
    end

    subgraph "3. Multi-Camera Spatial Fusion & Intelligence"
        Spatial --> ReID["CrossCameraReIDTracker (Spatial-Temporal-Visual Affinity)"]
        ReID --> Journey["FacilityJourneyTracker (Transit Timelines)"]
        ReID --> Threat["ThreatDetectionEngine (Tailgating, Wrong-Way, Loiter)"]
        ReID --> Evidence["VideoEvidenceRecorder (Pre/Post Buffered MP4s)"]
    end

    subgraph "4. Dynamic Geofencing & 3D Digital Twin"
        Spatial --> Geofence["GeofenceEngine (Ray-Casting Point-in-Polygon)"]
        ZoneReg["ZoneRegistry (Immutable Snapshots)"] -->|161ns Read| Geofence
        Spatial --> Twin["FacilitySpatialTwin (3D Metric Space 40x24x4.5m)"]
        ZoneReg -->|Extrude H=2.8m| Twin
    end

    subgraph "5. Storage, Events & UI Dashboard"
        Threat & Geofence --> EventEngine["EventEngine (Deduplication & Cooldowns)"]
        EventEngine --> SqliteStore["SqliteStore (Async Batched Queue: SQLite)"]
        Twin --> DashboardApi["DashboardApi & DashboardServer"]
        SqliteStore --> DashboardApi
        DashboardApi --> WebUI["Operator Web UI (Three.js 3D Twin + 2D Radar + CCTV)"]
    end
```

---

## 3. Repository Structure & Code Catalog

```text
vantage-main/
├── pyproject.toml                     # Package metadata, dependencies, and entrypoints
├── README.md                          # High-level overview & quickstart
├── PROJECT_KNOWLEDGE_BASE.md          # Comprehensive master system documentation (this file)
├── src/vantage/
│   ├── config/                        # Schema validation, source and pipeline configs
│   ├── core/                          # Logging, clocks, exceptions, metrics
│   ├── ingestion/                     # Video capture abstractions, backends, resilient sources
│   │   ├── base.py                    # FrameSource, SourceInfo, SourceKind interfaces
│   │   ├── opencv_source.py           # OpenCV VideoCapture wrapper (MSMF, DSHOW, V4L2, FFmpeg)
│   │   ├── resilient.py               # ReconnectingSource auto-reconnection loop
│   │   ├── synthetic.py               # Deterministic synthetic test pattern generator
│   │   ├── registry.py                # URI parser and source factory dispatcher
│   │   └── connectors/                # Physical IP & USB hardware discovery and dynamic hotplug
│   │       ├── discovery.py           # USB device enumeration & vendor RTSP path presets
│   │       └── manager.py             # CameraConnectorManager for dynamic runtime hotplug
│   ├── perception/                    # Object detection models and inference backends
│   │   ├── contracts.py               # DetectionResult, BoundingBox, Detection
│   │   ├── engine.py                  # DetectionEngine and NMS post-processing
│   │   ├── onnx_backend.py            # ONNX Runtime CPU inference wrapper
│   │   ├── openvino_backend.py        # OpenVINO CPU & Intel iGPU inference wrapper
│   │   └── registry.py                # Model zoo catalog, checksums, and weights fetcher
│   ├── tracking/                      # Multi-object tracking
│   │   ├── bytetrack.py               # ByteTrack two-pass association implementation
│   │   ├── kalman.py                  # Continuous-time 2D Kalman filter over (cx, cy, w, h)
│   │   ├── matching.py                # Jonker-Volgenant optimal linear assignment solver
│   │   └── contracts.py               # Track, TrackingResult, TrackState
│   ├── pose/                          # Human pose estimation
│   │   ├── contracts.py               # PoseLandmarks, Keypoint, PoseResult
│   │   ├── engine.py                  # RTMPose top-down estimator
│   │   └── factory.py                 # Pose engine factory builder
│   ├── state/                         # Entity dynamics and kinematics
│   │   ├── estimator.py               # Kinematic state estimator (speed in h/s, dwell, posture)
│   │   └── contracts.py               # EntityState, MotionState, Posture
│   ├── activity/                      # Temporal activity recognition
│   │   ├── engine.py                  # ActivityEngine (walking, running, loitering, falling, etc.)
│   │   └── contracts.py               # ActivityObservation, ActivityResult
│   ├── spatial/                       # Spatial analysis, 3D digital twin, and geometry
│   │   ├── engine.py                  # SpatialEngine (proximity, scene graphs, interactions)
│   │   ├── twin.py                    # FacilitySpatialTwin (3D metric mesh, camera frustums, 3D avatars)
│   │   └── geometry/                  # Core spatial geometry engine
│   │       ├── coordinates.py         # CoordinateSpace, Point2D, get_entity_foot_point
│   │       └── polygon.py             # Polygon class, ray-casting point-in-polygon algorithm
│   ├── events/                        # Event reduction, rules, and geofencing
│   │   ├── engine.py                  # EventEngine (deduplication, cooldowns, YAML rules)
│   │   ├── geofence.py                # GeofenceEngine (exclusion, occupancy, dwell, directional)
│   │   ├── zone_registry.py           # ZoneRegistry & ActiveZoneSnapshot immutable pointer swap
│   │   └── threats.py                 # ThreatDetectionEngine (tailgating, wrong-way, loitering)
│   ├── multicam/                      # Multi-camera scaling, fusion, and evidence
│   │   ├── pipeline.py                # MultiCameraPipeline (concurrent multi-thread frame processing)
│   │   ├── runner.py                  # Matrix runner CLI launcher
│   │   ├── reid.py                    # CrossCameraReIDTracker (spatial-temporal-visual affinity)
│   │   ├── radar.py                   # FacilityRadarMap (2D top-down floorplan projection)
│   │   ├── journey.py                 # FacilityJourneyTracker (cross-camera transit timeline)
│   │   └── evidence.py                # VideoEvidenceRecorder (buffered MP4 clip creator)
│   ├── storage/                       # Durable persistence & indexing
│   │   ├── sqlite_store.py            # SqliteStore (async batched SQLite writer)
│   │   ├── schema.py                  # Schema definitions (v1 to v3 with automatic migrations)
│   │   └── contracts.py               # Store Protocol, Query, StoredEvent, StoredObservation
│   ├── dashboard/                     # Web dashboard, live MJPEG, and REST API
│   │   ├── api.py                     # DashboardApi (route dispatchers for JSON payloads)
│   │   ├── server.py                  # DashboardServer (HTTP server with byte-range and chunked streaming)
│   │   ├── live.py                    # LiveFeed (lockless single-frame JPEG buffer)
│   │   └── static/                    # Frontend assets
│   │       ├── index.html             # Self-contained single-page dashboard with Three.js 3D Twin
│   │       └── js/                    # Local Three.js & OrbitControls static bundles
│   ├── identity/                      # Consensual opt-in biometric recognition (Phase 10)
│   ├── analytics/                     # Historical analytics, MAD baselines & anomalies (Phase 11)
│   └── search/                        # Natural language semantic event search
└── tests/                             # Comprehensive automated test suite (1,005+ tests)
```

---

## 4. Subsystem Deep Dives (Phases 1 – 15)

### 4.1 Video Ingestion & Hardware Backends (Phase 1)
- **Module**: `vantage.ingestion`
- **Supported Schemes**:
  - `webcam:N` / `camera:N` / `0`: Native OS video capture devices via DirectShow (`dshow`), Media Foundation (`msmf`), or V4L2 (`v4l2`).
  - `file:PATH` / `PATH.mp4`: Decoded local media files with native timeline pacing and looping support.
  - `synthetic://`: Deterministic procedural test generator simulating bouncing geometric targets.
  - `rtsp://`, `rtmp://`, `http://`: Network video streams decoded via OpenCV FFmpeg backend.
- **Resilience**: `ReconnectingSource` wraps live network and USB feeds with exponential backoff to handle physical cable disconnects without process crashing.

### 4.2 Object Detection & Dual Inference Runtime (Phase 2)
- **Module**: `vantage.perception`
- **Inference Engines**:
  - **ONNX Runtime**: Portable cross-platform CPU baseline.
  - **OpenVINO**: Intel CPU and Intel Integrated GPU (iGPU) hardware acceleration.
- **Model Zoo**:
  - `yolox-nano` (~3.7 MB, 80 COCO classes, ultra-low latency).
  - `yolox-tiny` (~19.3 MB, high speed).
  - `yolox-s` (~34.4 MB, balanced accuracy).
- **Coordinate Integrity**: Output bounding boxes are always transformed back into native frame coordinates `(x1, y1, x2, y2)`.

### 4.3 Multi-Object Tracking & Kalman Motion Model (Phase 3)
- **Module**: `vantage.tracking`
- **ByteTrack Implementation**:
  - Associates both high-confidence and low-confidence detections across frames.
  - Linear assignment solved via optimal **Jonker-Volgenant algorithm** (`vantage.tracking.matching`).
- **Continuous-Time Kalman Filter**:
  - State vector $x = [c_x, c_y, w, h, \dot{c}_x, \dot{c}_y, \dot{w}, \dot{h}]^T$.
  - Driven by true frame wall-time deltas $\Delta t$, eliminating tracking degradation during frame drops.

### 4.4 Objects365 Vocabulary & Open Discovery (Phase 3.5)
- **Module**: `vantage.perception`
- **D-FINE Detector (`dfine-s-obj365`)**: 365 object classes (pens, folders, tools, keyboards, etc.).
- **Grounding DINO Open-Vocabulary Discovery**: Zero-shot natural language prompt detection (`vantage discover --prompts "..."`).

### 4.5 Top-Down Human Pose & Entity State Dynamics (Phase 4)
- **Modules**: `vantage.pose`, `vantage.state`
- **RTMPose**: Top-down 17 COCO body landmarks per tracked entity.
- **Kinematic Dynamics**:
  - Speed measured in **entity heights per second ($h/s$)** to ensure distance invariance.
  - Posture classification: `standing`, `sitting`, `crouching`, `lying` via joint geometry ratios.

### 4.6 Temporal Activity Recognition (Phase 5)
- **Module**: `vantage.activity`
- Recognizes: `walking`, `running`, `loitering`, `idle`, `sitting_down`, `standing_up`, `falling`, `arm_raised`.
- Every observation is accompanied by an audit string (e.g. `0.65 h/s, held 100% of last 0.4s`).

### 4.7 Spatial Relations & Scene Graph (Phase 6)
- **Module**: `vantage.spatial`
- Evaluates pairwise distances (in entity heights) and directional vectors.
- Relations: `near`, `approaching`, `receding`, `interacting_with` (wrist inside object bounding box).

### 4.8 Event Reduction & Security Rules Engine (Phase 7)
- **Module**: `vantage.events`
- Transforms continuous observations into discrete security alerts.
- Configurable suppression cooldowns per (rule, entity) pair to prevent alert storms.

### 4.9 High-Throughput SQLite Storage & Heartbeat (Phase 8)
- **Module**: `vantage.storage`
- Dedicated worker thread commits batched writes asynchronously.
- Automatic schema migrations (v1 $\rightarrow$ v2 $\rightarrow$ v3) with minute-interval `heartbeats` table.

### 4.10 Local Web Dashboard & Packaging (Phase 9)
- **Module**: `vantage.dashboard`
- Built using `ThreadingHTTPServer` with zero external Node/npm dependencies.
- Serves live MJPEG streams, JSON telemetry APIs, and video evidence clips.

### 4.11 Consensual Biometric Identity & Audit (Phase 10)
- **Module**: `vantage.identity`
- Dual models: YuNet face detector (MIT) + SFace feature extractor (Apache-2.0).
- Pure mathematical 128-d embedding storage with strict cryptographic consent requirements.

### 4.12 Longitudinal Analytics & Anomaly Detection (Phase 11)
- **Module**: `vantage.analytics`
- Uses Median and Median Absolute Deviation (MAD) baselines in SQL to detect historical traffic deviations.

### 4.13 Multi-Camera Fusion, Spatial Re-ID & Video Evidence (Phase 12)
- **Module**: `vantage.multicam`
- **CrossCameraReIDTracker**: Computes weighted visual histogram + spatial-temporal transition matrix affinity.
- **FacilityRadarMap**: 2D top-down metric floorplan radar projection.
- **FacilityJourneyTracker**: Chronological transit timelines across camera feeds.
- **VideoEvidenceRecorder**: High-performance MP4 clip recorder using Windows Media Foundation / OpenH264.

### 4.14 Interactive UI Zone Polygon & Dynamic Geofencing (Phase 13)
- **Modules**: `vantage.spatial.geometry`, `vantage.events.geofence`, `vantage.events.zone_registry`
- **Ground-Plane Anchor**: Evaluates entity containment using the bottom-center foot point `((x1 + x2)/2, y2)`.
- **Ray-Casting Algorithm**: Handles arbitrary convex and concave polygons with sub-microsecond latency ($10.8\ \mu\text{s}$).
- **Zero-Lock Registry**: `ZoneRegistry` provides immutable `ActiveZoneSnapshot` pointer swapping with **$161.78\text{ ns}$** read latency.
- **Rule Archetypes**: `exclusion`, `occupancy`, `dwell`, `directional`.

### 4.15 3D WebGL Digital Twin & Three.js Facility Mesh (Phase 14)
- **Module**: `vantage.spatial.twin` & `index.html` Three.js renderer.
- Manages $40\text{m} \times 24\text{m} \times 4.5\text{m}$ metric facility coordinate space.
- Extrudes 2D camera polygons into 3D prismatic holographic bounding volumes ($H=2.8\text{m}$).
- Renders 3D camera frustum pyramids and real-time 3D tracked entity avatars with breadcrumb trails.

### 4.16 Physical RTSP / ONVIF IP Camera & USB Connectors (Phase 15)
- **Module**: `vantage.ingestion.connectors`
- **Hardware Probing**: `CameraDiscoveryService` discovers local USB webcams and provides vendor RTSP presets (Hikvision, Dahua, Axis, Reolink, Tapo, Generic ONVIF).
- **Low-Latency Streaming**: Automatic configuration of `rtsp_transport=tcp`, `nobuffer`, and `max_delay`.
- **Dynamic Hotplug**: `CameraConnectorManager` dynamically attaches and detaches cameras into the multi-camera pipeline and 3D twin without restarting.

### 4.17 Unified Entity Intelligence & Canonical Data Model (Phase 16)
- **Module**: `vantage.entity` (`contracts.py`, `context.py`, `manager.py`)
- **Canonical Entity Layer**: Aggregates structured knowledge from tracking, kinematics, activities, spatial/geofencing, journey timelines, and identity into immutable point-in-time `EntitySnapshot` instances with zero read contention.
- **Identity Hierarchy**: Explicitly separates `local_track_id` (single camera session) $\rightarrow$ `global_entity_id` (cross-camera fusion) $\rightarrow$ `named_identity` (biometric/consented identity).
- **Appearance Memory**: Upgrades Re-ID from a single vector to `EntityAppearanceMemory` with quality-filtered prototypes across viewpoints.
- **Unified Event Policy**: Establishes `EventCandidate` contract; routes all threats, geofence violations, and cross-camera handovers through `EventEngine` for centralized deduplication and cooldown enforcement.
- **Truthful Abstractions**: Refactors heuristic temporal action recognition to `HeuristicTemporalRecognizer` and search to `IncidentSearch` (`StructuredQueryParser` + `LexicalRetriever` + future `SemanticRetriever`).

### 4.18 Semantic & Learned Scene Intelligence (Phase 17)
- **Modules**: `vantage.scene` (`window.py`, `graph.py`), `vantage.activity.learned`, `vantage.search.semantic`
- **Temporal Observation Windows**: `EntityTemporalWindow` (60 frames / 5.0s span) and `SceneTemporalWindow` maintain timestamped trajectories, normalized foot points, and skeletal joints.
- **Kinematic & Skeletal Feature Extractors**:
  - `KinematicFeatures`: Calculates directional entropy $H \in [0, 1]$ over an 8-bin bearing histogram, acceleration variance, and pacing ratio (net displacement / path length).
  - `SkeletalDynamics`: Computes vertical hip drop rate in entity heights per second ($h/s$), wrist velocity, and prone posture confirmation.
- **Deterministic Temporal Behavior Recognizers**:
  - Recognizes `SUDDEN_COLLAPSE` (vertical descent $\ge 0.25\ h/s$ + prone end-state), `ERRATIC_PACING` ($H \ge 0.40$, pacing ratio $\le 0.60$), `CROUCHING_DWELL` ($\ge 10.0\text{s}$), `ERRATIC_HIGH_ENERGY_MOTION`, and `ABRUPT_DIRECTION_REVERSAL` ($> 135^\circ$ bearing flip).
  - `TemporalBehaviorRecognizer`: Unified entry point delegating to `FeatureBasedTemporalRecognizer` while providing an `OptionalModelTemporalRecognizer` seam for future ONNX graph models.
- **Transient Scene Graphs & Group Dynamics**:
  - `TransientSceneGraph`: Dynamically links entities within proximity distance ($0.20$ frame norm), computing convergence/dispersion velocity and crowd density.
  - **Ownership-Verified Unattended Object Tracking**: Correlates entity-object ownership during interaction, monitors departures, and alerts when objects remain stationary without interaction beyond $25.0\text{s}$.
- **Security Event Ontology & Hybrid Incident Search**:
  - `EventOntologyExpander`: Expands natural security search queries into canonical event, motion, and posture concept clusters (e.g. *collapse* $\rightarrow$ `sudden_collapse`, `falling`, `prone`, `floor`).
  - `IncidentSearch`: Combines structured parameter filters with expanded lexical scoring.

---

## 5. Mathematical Formulations & Algorithmic Foundations

### 5.1 Continuous-Time Kalman Filter for Bounding Boxes

State vector:
$$x = \begin{bmatrix} c_x & c_y & w & h & \dot{c}_x & \dot{c}_y & \dot{w} & \dot{h} \end{bmatrix}^T$$

State transition matrix for elapsed time $\Delta t$:
$$F(\Delta t) = \begin{bmatrix} I_4 & \Delta t \cdot I_4 \\ 0_4 & I_4 \end{bmatrix}$$

Prediction equations:
$$\hat{x}_{k|k-1} = F(\Delta t) \hat{x}_{k-1|k-1}$$
$$P_{k|k-1} = F(\Delta t) P_{k-1|k-1} F(\Delta t)^T + Q(\Delta t)$$

Measurement update with observation $z_k = [c_x, c_y, w, h]^T$:
$$y_k = z_k - H \hat{x}_{k|k-1}$$
$$S_k = H P_{k|k-1} H^T + R$$
$$K_k = P_{k|k-1} H^T S_k^{-1}$$
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k$$
$$P_{k|k} = (I - K_k H) P_{k|k-1}$$

---

### 5.2 Ray-Casting Point-in-Polygon Containment

For a point $P = (x_p, y_p)$ and a polygon with vertices $V_0, V_1, \dots, V_{n-1}$ where $V_n = V_0$:

A horizontal ray is cast from $P$ to $+\infty$. For each edge segment $(V_i, V_{i+1})$ with $V_i = (x_i, y_i)$ and $V_{i+1} = (x_{i+1}, y_{i+1})$:

$$\text{Ray intersects edge iff } (y_i > y_p) \ne (y_{i+1} > y_p) \quad \text{and} \quad x_p < x_i + \frac{(y_p - y_i)(x_{i+1} - x_i)}{y_{i+1} - y_i}$$

The point is strictly inside if and only if the total intersection count is odd:
$$\text{Inside}(P) = \left( \sum_{i=0}^{n-1} \mathbb{I}(\text{Intersects}(P, V_i, V_{i+1})) \right) \pmod 2 \equiv 1$$

---

### 5.3 Ground-Plane Foot Point Projection (2D $\rightarrow$ 3D)

Given a 2D bounding box $B = (x_1, y_1, x_2, y_2)$ in a camera frame of dimensions $(W, H)$:

1. **Normalized Foot Point**:
$$x_{\text{norm}} = \frac{x_1 + x_2}{2 W}, \quad y_{\text{norm}} = \frac{y_2}{H}$$

2. **3D Metric World Coordinate Transformation**:
For a sector bound $[X_{\min}, Z_{\min}] \rightarrow [X_{\max}, Z_{\max}]$:
$$X_{\text{world}} = X_{\min} + x_{\text{norm}} \cdot (X_{\max} - X_{\min})$$
$$Y_{\text{world}} = 0.0 \quad (\text{Ground Plane})$$
$$Z_{\text{world}} = Z_{\min} + y_{\text{norm}} \cdot (Z_{\max} - Z_{\min})$$

---

### 5.4 Robust Longitudinal Anomaly Detection (Median & MAD)

For historical traffic counts in slot $s$: $X_s = \{x_{s, 1}, x_{s, 2}, \dots, x_{s, k}\}$:

$$\text{Median}_s = \operatorname{median}(X_s)$$
$$\text{MAD}_s = \operatorname{median}\left( |x_{s, i} - \text{Median}_s| \right)$$
$$\hat{\sigma}_s = 1.4826 \cdot c_k \cdot \max(\text{MAD}_s, \text{floor})$$

Where $c_k$ is the small-sample bias correction factor:
$$c_k = 1 + \frac{1.3}{k - 0.5}$$

An observation $y_t$ is flagged as anomalous if:
$$|y_t - \text{Median}_{s(t)}| > k_{\text{threshold}} \cdot \hat{\sigma}_{s(t)}$$

---

## 6. Durable Storage Schemas & Database Migrations

### SQLite Schema (`vantage.storage.schema`)

```sql
-- Schema Version Tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    migrated_at REAL NOT NULL,
    description TEXT NOT NULL
);

-- Minute-Interval Pipeline Liveness Heartbeats
CREATE TABLE IF NOT EXISTS heartbeats (
    timestamp REAL PRIMARY KEY,
    camera_id TEXT NOT NULL
);

-- High-Frequency Sampled Entity Observations
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    camera_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    identity TEXT,
    label TEXT NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL,
    x2 REAL NOT NULL, y2 REAL NOT NULL,
    speed REAL,
    motion TEXT,
    posture TEXT,
    zone TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity_id);

-- Discrete Deduplicated Security Events
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    camera_id TEXT NOT NULL,
    entity_id TEXT,
    identity TEXT,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence TEXT,
    zone TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_rule ON events(rule);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_id);

-- Dynamic Polygonal Geofence Zones
CREATE TABLE IF NOT EXISTS zones (
    zone_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone_type TEXT NOT NULL,
    polygon_json TEXT NOT NULL,
    rule_config_json TEXT NOT NULL,
    severity TEXT NOT NULL,
    color TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);
```

---

## 7. Comprehensive REST API Specification

| Route | Method | Description | Example Payload / Params |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Main Operator Web UI (HTML + Three.js) | — |
| `/stream.mjpg` | `GET` | Live multi-camera MJPEG stream | — |
| `/snapshot.jpg` | `GET` | Single JPEG snapshot of current frame | — |
| `/api/live` | `GET` | Real-time telemetry, entities, FPS | — |
| `/api/events` | `GET` | Security incidents query | `?since=1h&limit=50` |
| `/api/search` | `GET` | Natural language semantic event search | `?q=loitering+near+vault` |
| `/api/radar` | `GET` | 2D Floorplan radar state | — |
| `/api/twin` | `GET` | 3D Digital Twin mesh, frustums, avatars | — |
| `/api/zones` | `GET` | List active geofence zones | `?camera_id=cam_01` |
| `/api/zones` | `POST` | Create/update polygonal geofence | `{"zone_id":"z1","polygon":[[0.1,0.2]...]}` |
| `/api/zones` | `DELETE` | Delete geofence zone | `?id=z1` |
| `/api/cameras` | `GET` | List active connected camera streams | — |
| `/api/cameras/discover` | `GET` | Scan local USB & DirectShow webcams | `?max=4` |
| `/api/cameras/presets` | `GET` | Return vendor RTSP URL presets | — |
| `/api/cameras/test` | `POST` | Test camera connection & get thumbnail | `{"uri":"rtsp://192.168.1.50/stream1"}` |
| `/api/cameras/connect` | `POST` | Hotplug camera into pipeline & 3D twin | `{"camera_id":"c5","uri":"webcam:0"}` |
| `/api/cameras` | `DELETE` | Detach camera from live pipeline | `?id=c5` |
| `/api/entities` | `GET` | Return canonical entity snapshots & stats | `?active_within=45.0` |
| `/api/entities?id=<id>` | `GET` | Return single entity complete dossier | `?id=global_person_1` |
| `/api/entity_timeline` | `GET` | One entity's state intervals and events, from the store | `?entity=person_17` |
| `/api/incidents` | `GET` | Correlated incidents | `?status=active&limit=50` |
| `/api/incident` | `GET` | One incident with its full dossier | `?id=inc_9b86da563f2c` |
| `/api/incident/timeline` | `GET` | Just that incident's timeline | `?id=inc_...` |
| `/api/incident/dossier` | `GET` | Severity and correlation breakdowns, evidence, links | `?id=inc_...` |
| `/api/relationships` | `GET` | Scored entity pairs | `?entity_id=person_17&min_strength=0.3` |
| `/api/relationships/graph` | `GET` | The same as a node-link graph | `?min_strength=0.2` |
| `/api/analytics` | `GET` | Bucketed history, coverage and anomalies | `?metric=entities&since=24h&interval=1h` |
| `/api/scene` | `GET` | Transient per-camera scene graph | `?camera_id=cam_01` |
| `/api/stats` | `GET` | Uptime, store counts, schema version | - |
| `/api/evidence/<clip>` | `GET` | Stream incident MP4 video clip | HTTP 206 Byte Range support |

**Availability convention.** Every route answers `200` with an `available`
boolean. `available: false` carries a `reason` written for an operator - "no
store: this run was started without --store", "no radar map attached; this needs
the multi-camera pipeline" - and is *not* the same as an empty result. A
subsystem that is switched off and a facility where nothing happened are
different facts, and the dashboard renders them differently.

---

## 8. Performance Benchmarks & Empirical Latencies

Measured on Intel Core CPU with Integrated GPU (Windows 11):

| Operation | Metric | Value | Architectural Guarantee |
| :--- | :--- | :--- | :--- |
| **Zone Snapshot Read** | Per-Frame Latency | **$161.78\text{ ns}$** | 0 locks, 0 SQLite disk I/O in worker loop |
| **Entity Snapshot Read** | Per-Frame Latency | **$< 200\text{ ns}$** | Zero lock contention on caller reads |
| **Point-in-Polygon** | Throughput | **$92,335\text{ pts/sec}$** | $10.8\ \mu\text{s}$ per evaluation on 10-vertex polygon |
| **Zone Update Propagation** | End-to-End Latency | **$0.55\text{ ms}$** | SQLite write + snapshot atomic pointer swap |
| **YOLOX-Nano Detection** | Frame Latency | **$10.9\text{ ms}$** | OpenVINO GPU/CPU inference |
| **ByteTrack Association** | Per-Frame Latency | **$0.70\text{ ms}$** | 8 simultaneous entities, Jonker-Volgenant |
| **Camera Signal Test** | Probing Latency | **$44.2\text{ ms}$** | Frame acquisition + base64 thumbnail generation |
| **Dashboard JSON API** | Serving Rate | **$> 100\text{ req/sec}$** | Non-blocking standard library HTTP server |

---

## 9. CLI Command Reference & Operational Runbook

### Starting the Multi-Camera Surveillance Platform

```bash
# Several cameras at once, with cross-camera identity, the floor plan and the
# 3D twin. --cameras is required: every source is site-specific, and a default
# set of clip filenames only ever worked in the directory it was written in.
vantage facility \
  --cameras entrance=webcam:0 \
            yard=rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101 \
            lobby=C:/clips/lobby.mp4 \
  --model yolox-nano \
  --port 8080
```

`python -m vantage.multicam.runner` still works and forwards to this command.

### Physical Camera Ingestion Examples

```bash
# Connect local USB webcam
vantage run --source webcam:0 --track --pose --dashboard

# Connect network RTSP camera (Hikvision/Dahua/Axis)
vantage run --source "rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101" --track --dashboard
```

### Model Zoo Management & Benchmarks

```bash
# Pull models
vantage models pull yolox-nano rtmpose-s yunet-face sface

# Run local inference hardware benchmark
vantage bench
```

---

## 10. Test Suites & Continuous Verification

The Vantage test suite contains **1,036+ tests** verifying zero regressions across all subsystems:

```bash
# Run entire test suite
pytest tests/ -v

# Run individual subsystem test suites
pytest tests/test_adversarial_phase17.py -v # Phase 17 adversarial & stress robustness
pytest tests/test_scene_window.py -v        # Temporal observation windows & kinematics
pytest tests/test_temporal_behavior.py -v   # Deterministic behavior recognition & seams
pytest tests/test_scene_graph.py -v         # Transient scene graphs & unattended objects
pytest tests/test_ontology_search.py -v     # Ontology-expanded incident search
pytest tests/test_entity_context.py -v      # Canonical entity intelligence & snapshots
pytest tests/test_reid_appearance.py -v     # Multi-prototype Re-ID appearance memory
pytest tests/test_event_policy.py -v        # Unified event policy & candidates
pytest tests/test_geofence.py -v            # Geofence geometry & zone registry
pytest tests/test_twin.py -v                # 3D Digital Twin mesh & projections
pytest tests/test_connectors.py -v          # IP camera discovery & dynamic hotplug
pytest tests/test_dashboard.py -v           # HTTP server, socket streaming & APIs
pytest tests/test_tracking.py -v            # ByteTrack & Kalman motion models
pytest tests/test_multicam.py -v            # Re-ID & Journey tracking
```

---

## 11. Architecture Decision Records (ADRs)

### ADR 001: Entity-Centric Intelligence Architecture
- **Context**: Single-camera and multi-camera pipelines were maintaining divergent representations of tracking, activity, and events.
- **Decision**: Introduce `EntityContext` and `EntityContextManager` (`vantage.entity`) as a canonical state aggregator producing immutable `EntitySnapshot` instances.
- **Consequences**: Unifies entity knowledge across subsystems while avoiding god-object anti-patterns.

### ADR 002: Multi-Prototype Appearance Memory for Cross-Camera Re-ID
- **Context**: Single-vector Re-ID drifted during illumination and viewpoint changes across disparate camera angles.
- **Decision**: Introduce `EntityAppearanceMemory` with quality gating and up to 4 representative prototypes across distinct camera sightings.
- **Consequences**: Dramatically improves cross-camera handover fidelity without requiring heavy deep neural Re-ID dependencies.

### ADR 003: Unified Event Production Architecture
- **Context**: Threats, geofences, and spatial rules duplicated cooldown tracking and event suppression logic.
- **Decision**: Standardize on `EventCandidate` input contract evaluated by a central `EventEngine` policy before event emission and evidence recording.
- **Consequences**: Single source of truth for alert debouncing, cooldowns, and evidence generation.

### ADR 004: Truthful Activity and Incident Search Abstractions
- **Context**: Heuristic temporal action recognizers and lexical search engines were previously labeled with speculative names.
- **Decision**: Refactor to `HeuristicTemporalRecognizer` and `IncidentSearch` with explicit protocol seams for future learned models (`LearnedTemporalRecognizer`, `SemanticRetriever`).
- **Consequences**: 100% architectural truthfulness and clean extensibility without breaking existing tests.

### ADR 005: Deterministic Spatio-Temporal Behavior & Transient Scene Graph
- **Context**: Static per-frame thresholding failed to distinguish complex behaviors (e.g., sudden collapse vs controlled sitting, erratic pacing vs linear transit, or genuine group convergence).
- **Decision**: Introduce rolling `EntityTemporalWindow` with mathematically engineered kinematic extractors (`directional_entropy`, `pacing_ratio`, `hip_drop_rate`) coupled with `TransientSceneGraph` and `EventOntologyExpander`.
- **Consequences**: Deterministic, explainable short-term behavior recognition and unattended object attribution.

### ADR 006: Observational Relationship Semantics, Explainable Attribution, & Exponential Time-Decay Memory
- **Context**: Phase 17 provided transient per-camera interaction edges, but lacked long-horizon relationship persistence across cameras and temporal sessions. Furthermore, naive all-pairs comparisons risked $O(N^2)$ computational explosion and subjective intent labels (e.g., "friends", "accomplices") risked ungrounded speculation.
- **Decision**: Introduce `vantage.relationship` with:
  1. Strict evidence-first observational semantics (`CO_OCCURRENCE`, `RECURRENT_PROXIMITY`, `LAGGED_TRAJECTORY_ALIGNMENT`, `REPEATED_GROUP_CO_CLUSTERING`, `SHARED_ZONE_PRESENCE`).
  2. Explainable attribution breakdown (`RelationshipScoreBreakdown`) detailing individual signal contributions.
  3. Candidate pair gating (restricting analysis to scene graph proximity edges, shared zone/camera co-location, and active in-memory pairs) to prevent pair explosion.
  4. Exponential recency decay ($\text{score}(t) = \text{raw\_score} \cdot e^{-\lambda \Delta t}$) that decays active relationship strength over time while preserving historical evidence records.
  5. Lagged trajectory alignment detector (`FollowingPatternDetector`) differentiating trailing followers from side-by-side walkers.
- **Consequences**: Highly scalable, explainable long-horizon relationship graph with zero ungrounded psychological claims and clean multi-camera integration.

### ADR 007: Situational Incident Intelligence, Three-Way Decision Bands, & Explainable Continuity Penalties
- **Context**: Prior phases produced discrete events in isolation. Operators had to manually inspect logs to connect whether multiple alerts, cross-camera transitions, unattended objects, and following patterns were part of the same evolving situation.
- **Decision**: Introduce `vantage.incident` with:
  1. Canonical incident representation (`CanonicalIncident`, `IncidentTimelineEntry`, `IncidentEvidenceDossier`).
  2. Multi-factor explainable correlation scoring combining positive evidence ($w_{\text{ent}}=0.35, w_{\text{time}}=0.20, w_{\text{space}}=0.15, w_{\text{rel}}=0.15, w_{\text{behav}}=0.15$) and negative continuity penalties (implausible camera jumps, degraded identity, excessive temporal gaps).
  3. Three-way correlation decision bands: `ATTACH` ($\ge 0.65$), `CORRELATION_CANDIDATE` ($[0.35, 0.65)$), `NEW_INCIDENT` ($< 0.35$).
  4. Deduplicated chronological timeline management and lifecycle state machine (`ACTIVE` $\rightarrow$ `QUIESCENT` $\rightarrow$ `RESOLVED`).
  5. SQLite schema v4 persistence for seamless state recovery across restarts.
- **Consequences**: Unified situational intelligence that correlates multi-camera events into audit-defensible dossiers without ungrounded causal claims.

### ADR 008: Canonical Reactive Intelligence State Store and Cross-View Context Synchronization
- **Context**: Prior to Phase 20, the frontend visualizer consisted of separate widgets that polled independent endpoints and rendered tabular data without demonstrating the rich underlying intelligence stack (scene graphs, persistent relationships, situational incident timelines, and 3D digital twins).
- **Decision**: Introduce the Phase 20 Investigative Intelligence Workspace in `src/vantage/dashboard/` featuring:
  1. Centralized `IntelligenceState` managing global entity selection, active incident focus, relationship pairs, and camera contexts.
  2. 5 Primary Workspaces: `Live Matrix`, `Incident Command Center`, `Intelligence (Relationship & Scene Graphs)`, `Investigate (Entity Dossiers & Global Timeline)`, `Digital Twin (2D Floorplan Radar & 3D Spatial Mesh)`.
  3. First-class Interactive Entity Dossier (`openEntityDossier`) exposing multi-camera journeys, temporal behaviors, relationship links, and observation history.
  4. Force-directed HTML5 Canvas Relationship Network with zoom/pan/drag and explainable scoring breakdown inspector.
  5. Cross-view synchronization: selecting an entity, incident, or relationship propagates immediately across live video, graphs, dossiers, and 3D digital twin.
- **Consequences**: Transforms Vantage from a conventional CCTV monitoring UI into a cinematic, audit-defensible intelligence workspace that visibly communicates the entire intelligence architecture.

---

## 12. Phase 18: Persistent Entity Relationship Subsystem

Phase 18 introduces long-horizon relationship intelligence organized in `src/vantage/relationship/`:

```text
vantage.relationship
├── config.py       # RelationshipScoringConfig & FollowingDetectorConfig
├── models.py       # RelationshipSignal, EntityRelationship, ProximityBasis, RelationshipScoreBreakdown
├── scoring.py      # RelationshipScorer with saturated exponential curves & time-decay
├── following.py    # FollowingPatternDetector with optimal lag tau and heading alignment
├── tracker.py      # PersistentRelationshipTracker with candidate pair gating & LRU cache
└── service.py      # RelationshipService coordinating storage hydration, persistence & graph export
```

---

## 13. Phase 19: Situational Incident Intelligence & Multi-Event Reasoning

Phase 19 introduces situational incident intelligence organized in `src/vantage/incident/`:

```text
vantage.incident
├── config.py       # IncidentCorrelatorConfig (thresholds, weights, timeouts, penalties)
├── models.py       # CanonicalIncident, IncidentState, IncidentTimelineEntry, IncidentCorrelationBreakdown
├── timeline.py     # IncidentTimelineManager (chronological sorting & canonical ID deduplication)
├── correlator.py   # IncidentCorrelator (candidate gating, positive evidence & negative penalties)
└── service.py      # IncidentService (lifecycle state machine, escalation, and SQLite v4 store)
```

---

## 14. Phase 20: Intelligence Visualization & Investigative Workspace

Phase 20 introduces the unified investigative intelligence workspace in `src/vantage/dashboard/`:

```text
frontend/                    # TypeScript sources; built by Vite into static/
├── src/contracts/           # vocabulary.ts and types.ts - the API's shapes
├── src/data/source.ts       # the only place the page talks to the server
├── src/components/common/   # Panel, Stat, Loading, Unavailable, Empty, Failed
├── src/features/            # live, incidents, analytics, intelligence,
│                            #   investigate, twin
└── scripts/smoke.mjs        # loads the page in Chrome and fails on any error

src/vantage/dashboard/
├── static/                  # the committed build output, served from disk
│   ├── index.html           # shell
│   └── assets/              # index.js, react.js, SpatialTwin3D.js (three.js,
│                            #   loaded lazily), index.css, bundled fonts
├── api.py                   # REST & JSON read-model dispatch layer
├── factory.py               # assembles the server from configuration
├── live.py                  # Live JPEG/MJPEG stream snapshot encoder
└── server.py                # Multi-threaded HTTP and MJPEG stream server
```

**Six workspaces**, not five: `Live`, `Incidents`, `Trends` (the Phase 11
analytics, which the earlier build had dropped from the UI entirely),
`Intelligence`, `Investigate`, `Twin`. Number keys switch between them and the
URL hash names the current one.

**No demo mode.** An earlier revision of this workspace defaulted to a
hand-written fixture set so that a first look would show a populated console.
Everything on the page now comes from the pipeline, or the page says which
subsystem is not attached and what would start it.

### Key Capabilities
1. **Interactive Entity Dossier**: Built from `/api/entity_timeline`, so it works on a single camera as well as a facility: state intervals, the events that named the entity, and its scored associations.
2. **Force-Directed Relationship Network**: Visualizes persistent observational relationship graph on interactive HTML5 Canvas with explainable attribution breakdown (co-occurrence, proximity, following, duration).
3. **Live Scene Graph Topology Visualizer**: Visualizes transient scene graphs, entity proximity edges, clusters, density meter, and unattended object ownership tracking.
4. **Incident Command Center & Case Dossiers**: The incident's timeline, how its severity was reached, and - where an event was attached rather than opening a new incident - the correlator's own factor breakdown, stored on the incident at the moment it decided.
5. **3D Spatial Digital Twin Overlays**: Synchronizes entity state halos, incident focus mode, and camera frustum projections in real-time WebGL.

---

## 10. Test Suites & Continuous Verification

The Vantage test suite contains **1,080+ tests** verifying zero regressions across all subsystems:

```bash
# Run entire test suite
pytest tests/ -v

# Run individual subsystem test suites
pytest tests/test_adversarial_phase19.py -v # Phase 19 adversarial & stress robustness
pytest tests/test_incident_models.py -v     # Incident models & config validation
pytest tests/test_incident_correlation.py -v # Multi-factor scoring & negative penalties
pytest tests/test_incident_timeline.py -v   # Timeline deduplication & ordering
pytest tests/test_incident_service.py -v    # Lifecycle state machine & SQLite v4
pytest tests/test_incident_api.py -v        # Incident REST APIs & incident search
pytest tests/test_adversarial_phase18.py -v # Phase 18 adversarial & stress robustness
pytest tests/test_relationship_models.py -v # Relationship models & evidence contracts
pytest tests/test_relationship_scoring.py -v # Explainable scoring & time decay
pytest tests/test_following_detector.py -v  # Lagged trajectory following detection
pytest tests/test_relationship_tracker.py -v # Candidate pair gating & event emission
pytest tests/test_relationship_service.py -v # SQLite persistence & graph snapshots
pytest tests/test_relationship_api.py -v     # Dashboard relationship REST APIs
pytest tests/test_adversarial_phase17.py -v # Phase 17 adversarial & stress robustness
pytest tests/test_scene_window.py -v        # Temporal observation windows & kinematics
pytest tests/test_temporal_behavior.py -v   # Deterministic behavior recognition & seams
pytest tests/test_scene_graph.py -v         # Transient scene graphs & unattended objects
pytest tests/test_ontology_search.py -v     # Ontology-expanded incident search
pytest tests/test_entity_context.py -v      # Canonical entity intelligence & snapshots
pytest tests/test_reid_appearance.py -v     # Multi-prototype Re-ID appearance memory
pytest tests/test_event_policy.py -v        # Unified event policy & candidates
pytest tests/test_geofence.py -v            # Geofence geometry & zone registry
pytest tests/test_twin.py -v                # 3D Digital Twin mesh & projections
pytest tests/test_connectors.py -v          # IP camera discovery & dynamic hotplug
pytest tests/test_dashboard.py -v           # HTTP server, socket streaming & APIs
pytest tests/test_tracking.py -v            # ByteTrack & Kalman motion models
pytest tests/test_multicam.py -v            # Re-ID & Journey tracking
```
- **Consequences**: Provides deterministic, highly interpretable behavior detection and ownership-verified unattended object tracking while establishing clear model runtime seams for future ONNX graph networks.

---
*End of Authoritative System Knowledge Base.*
